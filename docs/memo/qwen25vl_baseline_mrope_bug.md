# Qwen2.5-VL's sequential baseline was silently dropping M-RoPE

**Found:** 2026-08-25, GH200 session, while trying to get an interleaved g=4 RefCOCO/GQA
deliverable and hitting the interleaved correction contract's g=1 identity gate along the way.

**One-line summary:** `qwen25vl_executor.py`'s `full_inference` (and `head_inference`'s
`generate()` fallback, and `refcoco_gqa_batched_eval.py`'s independent baseline call) never passed
`mm_token_type_ids` to the stock model. Without it, `transformers`' own `compute_3d_position_ids`
cannot compute real M-RoPE and silently falls back to plain sequential 1D positions replicated
across all 3 mrope axes — every image token loses its real (temporal, height, width) grid position
and is treated as if it were text at its sequence offset. The AppCorr fork was never broken; it was
correctly computing M-RoPE the whole time via its own `get_rope_index` call, and disagreeing with a
baseline that wasn't.

## Why this is worth writing down, not just fixing

Every rule in `docs/memo/interleaved_correction_contract.md` was checked and passed. The bug lived
entirely in the *reference* the correction was being checked against, not in the correction. The
chain below is the shape of investigation that finds that kind of bug — narrowing a "the correction
is wrong" hypothesis all the way down until the evidence flips it into "the correction was right and
the reference was wrong." Skipping any one step here would have left the conclusion at the wrong
layer.

## The chain

1. **The g=1 identity gate, run for real.** `docs/memo/interleaved_correction_contract.md` requires
   `num_groups=1` (single round, full-depth correction, 100% token coverage) to reproduce the
   sequential baseline bit-for-bit — architecturally the same computation, so anything less than
   exact equality is a defect, not noise. Ran both configs through the real
   `SchedulerModule`/`WorkerModule`/`GroupTriggerPolicy` pipeline on the same 10 strided RealWorldQA
   samples, demanded **per-sample** prediction equality rather than trusting the aggregate.

2. **The trap: aggregate accuracy tied (70%==70%) while two samples silently disagreed.** Sample-
   level diff caught it; an aggregate-only gate would have called the broken version healthy —
   exactly the shape `docs/memo/interleaved_correction_contract.md` and this project's CLAUDE.md
   both warn about ("two conditions agreeing to every digit... means the mechanism never ran" has a
   mirror image: two conditions *disagreeing* on individual samples while *agreeing* in aggregate
   means the aggregate is hiding something, not confirming health).

3. **Transmission exonerated.** Precedent existed for exactly this failure mode: a 2026-08-17 bug
   where the Laplacian encoder predicted residuals from the native gaussian while the decoder
   predicted from the resampled base, leaving real error even at 100% delivery on non-dyadic
   resolutions. Reproduced the transmission round-trip in isolation (pure pixel bytes, no model) —
   bit-identical. Excluded by construction here: `realworldqa_offload_eval.py`'s
   `smart_resize(..., factor=112)` forces every dynamic-resolution target onto a 112-aligned grid,
   so the pyramid-level divisibility check that the 08-17 bug violated is satisfied automatically.

4. **Model-facing input tensors exonerated, properly this time.** First pass only diffed
   reconstructed image *bytes*; a peer review correctly called that insufficient — `smart_resize`,
   normalization, and patch embedding all sit between bytes and what the model actually consumes.
   Captured `pixel_values`/`image_grid_thw` as literally handed to `preprocess()` for both arms:
   bit-identical (`max_abs_diff=0.0`, every element).

5. **Vision tower output exonerated, with a coverage check first.** Reasoning: at `g=1`, LLM
   correction recomputes every position over full depth from identical inputs, so nothing it reads
   can depend on stale state *except* the vision tower's output, which the LLM correction does not
   itself recompute. Confirmed coverage first (`group_idx` selected all 1664/1664 merge-groups, no
   partial correction hiding), then compared the post-merger, pre-splice image embedding sequence:
   bit-identical.

6. **`inputs_embeds` (post-splice, pre-layer-0) exonerated.** The splice — where the vision output
   actually lands in the text sequence at the right image-token positions — sits between two
   already-exonerated checkpoints and wasn't itself covered by either. Bit-identical.

7. **Per-layer sweep found the divergence starts at layer 0, not accumulating from later layers.**
   Dumped the LLM residual stream after each of 64 layers, both arms. Relative error was already
   ~5.4% at layer 0's output — logic-bug scale, not the ~1e-3 numerical-reassociation floor a
   different SDPA kernel path would produce (both stock and fork use `attn_implementation="sdpa"` /
   `F.scaled_dot_product_attention` — same backend, confirmed from the loader code, not run to find
   out).

8. **Sub-block trace inside layer 0 pinned it to RoPE specifically.** `input_layernorm` output and
   the pre-RoPE V projection: bit-identical. Q and K the instant RoPE is applied: ~75%/~35%
   relative error, 87-88% of elements differing. Nothing before RoPE differs; RoPE is where it
   starts.

9. **Traced into the `transformers==5.13.0` source and confirmed by direct trace, not by reading.**
   `Qwen2_5_VLModel.compute_3d_position_ids`'s `can_compute_mrope` gate requires
   `mm_token_type_ids is not None`. `full_inference`'s stock call never supplied it. Monkeypatched
   `compute_3d_position_ids` to print its return value: `returned None`, confirmed directly. HF's
   fallback (`Qwen2_5_VLTextModel.forward`, `position_ids is None` branch):
   `torch.arange(seq_len).expand(3, B, -1)` — plain sequential 1D positions on all 3 mrope axes.

## The self-caught mistake, worth repeating for the next person

The first attempt to measure the prefill hidden state (checkpoint after step 6, before step 7) gave
`max_abs_diff=2488`, 99.97% of elements differing — an alarming number that would have looked like
confirmation of a real bug. It wasn't yet a real measurement: `outputs.hidden_states[-1]` from a
stock `output_hidden_states=True` call turned out to already be POST-final-norm in this
`transformers` build (`Qwen2_5_VLTextModel.forward` doesn't even accept `output_hidden_states` as a
parameter; `last_hidden_state` — already normed — is what flows into that output field), while the
fork's `context["llm_current_feature"]` is PRE-norm. Comparing them was comparing two different
quantities, not a divergence. **The tell was the scale, not the diff:** the stock side's per-token
RMS sat in [1.4, 3.9], matching the LLM-fork unit test's own recorded post-norm `stock_hidden` scale
almost exactly (documented in that test's own output); the fork side's RMS ran up to 56, matching
raw pre-norm residual-stream scale. Re-diffed correctly (applied the model's real `norm` to the
fork's output, compared against the stock's already-normed output) before drawing any conclusion.
A wrong comparison that produces an alarming number is worse than no comparison — the scale
statistics are what caught it, not the diff itself.

## The proof

Correctly-called stock (real `mm_token_type_ids`, not the fork's `position_ids` injected directly —
see below for why that distinction matters) vs. the fork's `g=1` output, same sample:

- First-token logits: **bit-identical**, `max_abs_diff=0.0`, 0/152064 elements differ.
- Full prefill hidden state, norm-aligned, all 1702 positions: **bit-identical**,
  `max_abs_diff=0.0`, 0/8,714,240 elements differ, including the final position that produces the
  answer.

Re-ran the original 10-sample g=1 gate with the fix applied to `full_inference`: **10/10 samples
match fork@g=1 exactly**, sample for sample, not just in aggregate. The aggregate stayed at 70%
(unchanged) because the fix moves individual samples in both directions relative to the old, wrong
baseline — the aggregate tying was never evidence of anything, which is the entire point of gate #2
in the contract memo.

## Why `mm_token_type_ids`, not `position_ids`

The first working proof injected `context["position_ids"]` — the fork's own `get_rope_index`
output — directly into the stock call. That proves the two CAN produce equal results, but as a
permanent fix it would make the baseline agree with the fork *by construction*: both would be
reading the same tensor from the same call site, so the baseline could never again independently
disagree with the fork about positions, including if the fork's derivation were itself wrong. The
actual fix passes `mm_token_type_ids` and lets `compute_3d_position_ids` derive positions the way
the model is designed to — keeping the baseline a genuine independent reference.

`Qwen25VLExecutor.full_inference` now asserts this independence on every call (not just once):
after the real forward pass, it compares what `compute_3d_position_ids` derived internally against
`context["position_ids"]` via `torch.equal`, unconditionally, and raises if they ever disagree. If
this assertion ever fires, the fork and the model's own derivation have diverged for a new reason
and nothing downstream should be trusted until that's understood.

## Fixed, where

- `offload/server/model/qwen25vl_executor.py`: `full_inference`'s prefill call, `full_inference`'s
  `generate()` continuation fallback, `head_inference`'s `generate()` continuation fallback — all
  three now pass `mm_token_type_ids`.
- `analysis/experiments/refcoco_gqa_batched_eval.py`: this driver does **not** go through
  `full_inference`/`head_inference` at all — it bypasses `SchedulerModule`/`WorkerModule` and
  reimplements the baseline/first-token/generate logic independently (a separate, third
  reimplementation of the same schedule, flagged earlier in this investigation as a risk in its own
  right: two drivers reimplementing the same thing is how a result table ends up holding two
  different algorithms in one column). Its `build_first_token_context`'s baseline branch and its
  `batched_generate_fallback` had the identical gap, independently, and needed the identical fix
  independently — confirmed by checking, not assumed to be covered by the executor-level fix.

## What this invalidates

Every "trusted" sequential/floor number in `analysis/experiments/QWEN25VL_APPCORR_LOG.md` was
produced by `full_inference` (the log's own §8 calls it the "mechanism-matched baseline," commit
`310c65a`) — including the RefCOCO 85.75% / GQA 60.84% baselines the current campaign was treating
as ground truth. Every gap, crossing point, and conclusion measured against those baselines in the
log's §7 (including the "~58% keep-rate, -1pp threshold" conclusion) is measured against a reference
that has now moved and needs re-establishing before being quoted again. `head_inference`'s
`generate()` fallback sharing the same gap means every multi-token Qwen answer generated anywhere in
this repo — RealWorldQA's longer freeform answers, RefCOCO's bbox-coordinate answers needing more
than one token, everything — was decoding continuation tokens with degraded position information,
for both arms equally (this doesn't explain the g=1 divergence, since the first token was already
fixed before either generate() fallback runs, but it does mean generation quality itself has been
understated across the board).

## A second, separate finding from verifying the fix on `refcoco_gqa_batched_eval.py`: a per-image memory leak, found and fixed

Running this driver across more than one image in a single process (independent of `--batch-size`,
reproduced at both `--batch-size 5` and `--batch-size 1`) OOM'd on the second image — 32B occupies
~65GB, but after one image's correction pass the process held ~93.8GB. Confirmed the *fix itself*
was already correct on the one image that completed pre-fix (`idx=0`, `--batch-size 1`, both arms:
`'485,0,644,137 bowl behind the others can only see '`, bit-for-bit identical text) — a
`--batch-size 5` run's slightly different bbox digits on the same image were batch-size-dependent
bf16 noise (the script's own docstring already anticipates this: "mod ordinary bf16
batch-size-invariance noise"), not an identity failure, and disappeared entirely once both arms ran
at the same batch size.

**Root cause, found by instrumenting rather than theorizing** (`torch.cuda.memory_allocated()`/
`memory_reserved()` before and after each image, gated behind `APPCORR_MEM_TRACE=1`): allocated
memory stayed pinned at 93.81GB from immediately after image 1 through the *start* of image 2's
build — never falling back toward the 66.91GB model-only baseline. A live reference, not
fragmentation (reserved was already saturated too, so `torch.cuda.empty_cache()` would not have
helped). The culprit is pure Python semantics: `first_token, context = build_first_token_context(...)`
only rebinds the name `context` *after* the right-hand side is fully evaluated, so the *previous*
iteration's `context` dict (its `vision_cache`/`kv_cache`/`llm_input_embeds` tensors, ~27GB/image on
the 32B model) stayed alive, referenced by that name, for the *entire* duration of building the next
image's context — guaranteeing two images' state resident simultaneously on every iteration, at any
batch size. Exactly the "if the release happens after the next image's context is already built,
both live simultaneously" shape predicted before instrumenting.

**Fixed** with an explicit `del context` right after the four tensors `batch_items` actually needs
are extracted from it — not `torch.cuda.empty_cache()`, which would have masked the leak and cost a
sync every image without addressing why the reference was live. Verified: a 5-sample RefCOCO run at
`--batch-size 1` that previously OOM'd on image 2 now completes all 5 (`allocated` returns to
66.94GB — the model-only baseline — before every image), and re-ran the g=1 identity gate on this
driver at those 5 samples: **sequential and interleaved@g=1 match exactly, sample for sample,
including `mean_iou` to four decimals (0.8124 both).** This was blocking any real-scale RefCOCO/GQA
run regardless of the M-RoPE fix being correct; it no longer does.

## The permanent gate

`analysis/experiments/qwen25vl_g1_identity_gate.py` — runs both arms through the real pipeline on
the same strided samples, asserts exact per-sample text equality (never the aggregate), exits
non-zero on any mismatch. This is the check that would have caught both the rule-3 increment-persist
bug (fixed earlier the same day, commit `e43e549`) and this one, and the next fork or config change
in this file should be run against it before any number from it is trusted.
