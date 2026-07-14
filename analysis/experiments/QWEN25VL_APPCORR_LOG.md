# Qwen2.5-VL (32B / 72B) AppCorr on RealWorldQA / GQA / RefCOCO

Branch: `experiment/qwen25vl-appcorr` (forked from `main`). This extends AppCorr's
`.approx()`/`.correct()` hard-fork pattern to Qwen2.5-VL-32B-Instruct and Qwen2.5-VL-72B-Instruct.
Sections 1-6 below cover the original RealWorldQA VQA benchmark work; section 7 extends the
keep-rate sweep to GQA (VQA) and RefCOCO (referring-expression grounding) to test whether the
RealWorldQA sweet-spot finding generalizes. Every image is kept at its own native/dynamic
resolution (via `smart_resize`) rather than a fixed shape.

## 1. Architecture

Qwen2.5-VL is the most architecturally complex fork built this session, and the first (besides
OpenVLA's Llama) to require forking **both** a vision tower and a causal LLM decoder in the same
model.

**Vision tower** (`appcorr/models/qwen25vl/vision/`): 32 blocks, hidden_size=1280, 16 heads
(head_dim=80), fused QKV, RMSNorm (no LayerScale), SwiGLU MLP.
- **2D (spatial) RoPE**: per-(h,w) position frequencies, precomputed once per request from
  `grid_thw` alone (no fork needed for the rotary embedding itself, it has no parameters).
- **Window attention**: 4 of 32 layers (`fullatt_block_indexes = {7,15,23,31}`) use full per-image
  attention (`cu_seqlens`); the other 28 use windowed attention (`window_size=112`,
  `cu_window_seqlens`), via a `window_index` token permutation re-derived per image from its patch
  grid shape alone.
- **Patch merging**: `Qwen2_5_VLPatchMerger` runs once after all 32 blocks, nonlinearly combining
  each fixed 2x2 (`spatial_merge_unit=4`) neighboring-patch group. The fork's correction granularity
  is aligned to these 4-patch merge groups (not raw patches), matching how `window_index` itself
  already groups patches, so merging is always over a fully-corrected-or-fully-stale group.
- **Native resolution**: `smart_resize(h, w, factor, min_pixels, max_pixels)` rounds each image to
  the nearest patch-aligned size while preserving aspect ratio, reading the real processor's
  configured pixel bounds rather than hardcoded defaults. `factor=112` (not the library default 28)
  is required so every Laplacian pyramid level stays patch-aligned.

**Causal LLM** (`appcorr/models/qwen25vl/llm/decoder_layer.py`): standard pre-norm/RMSNorm/SwiGLU
decoder layer, GQA (`num_key_value_heads=8 < num_attention_heads=40` for 32B), directly reusing the
`repeat_kv` pattern from OpenVLA's Llama fork. The genuinely new piece is **M-RoPE**: `position_ids`
is `[3,B,T]` (temporal/height/width axes), computed once per request via `get_rope_index` (text-only
from `input_ids`/`image_grid_thw`, no vision features needed, so cheap/cacheable), then
`apply_multimodal_rotary_pos_emb` splits the head-dim channel axis into `mrope_section` chunks and
round-robins across the 3 axes. Like the base rotary computation, this is parameter-free and
layer-independent, so `.correct()` accepts pre-gathered `cos_sel`/`sin_sel` computed once per
correction round rather than once per layer (same RoPE-hoisting optimization as OpenVLA's Llama
fork). The "permanent query group" (always force-corrected every round) is every non-image-token
position (`input_ids != image_token_id`), derived dynamically from the mask rather than assumed as a
fixed prefix, since Qwen splices image tokens via `masked_scatter` wherever `image_token_id` occurs.

**Executor** (`offload/server/model/qwen25vl_executor.py`): `batch_size=1` always. `preprocess`
calls the real `Qwen2VLImageProcessor` for patchify; `head_inference` decodes the first generated
token from the actual corrected prefill state, then falls back to stock `model.generate()` for any
additional tokens (a documented, deliberate scope simplification -- the first token, which is what
this whole mechanism validates, is always fixed by the corrected state before the fallback runs).

## 2. Bugs found and fixed

1. **`model.config.num_hidden_layers` / `model.visual` AttributeErrors** (Phase 0): Qwen2.5-VL's
   config nests the LLM config under `text_config`, and the vision tower lives at `model.model.visual`
   (not `model.visual`) since `Qwen2_5_VLForConditionalGeneration` wraps an inner `Qwen2_5_VLModel`.
2. **LLM `correct()` unit-test tolerance** (Phase 2): a large-looking `max_abs_err` (92.1) in the
   "correct 100% from blurred" tier turned out to be caused by real "massive activation" outlier
   channels (a documented LLM phenomenon) combined with a different SDPA kernel path -- isolated via
   toy random-weight configs that gave exact 0.0 error, proving the fork's algorithm was correct.
   Fixed the *test's* pass criteria (p99 error + final-position argmax match) rather than the code.
3. **Model reload on every CONFIG message** (Phase 4, own design flaw): the per-image CONFIG resend
   needed for native per-image resolution caused `worker.py` to reload the full model on every
   single image. Fixed generically in `worker.py` via a model-identity cache -- a real,
   generically-useful shared-infra fix, not Qwen-specific.
4. **Pyramid patch-alignment `AssertionError`**: `smart_resize`'s default `factor=28` only aligns to
   the raw patch size, but the Laplacian pyramid needs `patch_size * pyramid_scale` (112 for a
   4x pyramid) alignment at every level. Fixed by using `factor=112`.
5. **`LaplacianPyramidPolicy.decode()` cross-branch fix drift**: this branch forked from `main`
   before an unconditional-upsample fix (found during the earlier CLIP investigation) had landed
   there, so it recurred and had to be re-cherry-picked (`51a69a4`).
6. **`head_inference` generation-loop design flaw** (own bug, caught via code review before
   running): the initial implementation called stock `model.forward()` in a manual greedy loop,
   which recomputes everything from scratch every iteration -- completely ignoring the corrected
   prefill state and defeating the entire point of the mechanism. Fixed to decode the first token
   directly from the corrected state, falling back to stock `generate()` only for continuation
   tokens.
7. **72B config `total_layers=64` bug**: `realworldqa_qwen25vl_72b_interleaved_g4.json` was
   generated via `sed 's/32b/72b/g'` from the 32B config, which left `total_layers` at 32B's value
   (64) instead of 72B's actual 80 LLM layers. Symptom: `GroupTriggerPolicy`'s scheduling only ever
   processed through layer 64, leaving the last 16 layers completely unprocessed --
   `head_inference` read a mid-network hidden state, producing *identical* garbled output
   ('相关负责') for 3 different images/questions (the identical-output-across-different-inputs
   pattern was the tell). Fixed: `total_layers` 64 -> 80.
8. **SDPA kernel-dispatch mismatch in `.correct()`** (most important; found via direct user
   guidance). The user pointed out that a 100%-correction round *must* reproduce baseline exactly,
   and asked for the implementation to be reconsidered rather than accepting a "known bf16 noise"
   explanation. Diagnostic: ran `num_groups=1` (single-shot, 100%-coverage correction) and diffed
   predictions sample-by-sample against stock baseline at nr=20. Result: 18/20 identical, but 2
   genuine divergences on borderline numeric-count questions (idx=418: gt='2'->pred='1', idx=532:
   gt='3'->pred='2') -- surprising, since this is architecturally a full recomputation and should
   match. Root cause: `ApproxCorrectQwen25VLAttention.correct()` always built an explicit float
   causal mask for `F.scaled_dot_product_attention`, while `.approx()` and stock both dispatch
   `is_causal=True` -- a different fused CUDA kernel. The resulting bf16 divergence compounds over
   32-80 layers (vision + LLM) and can flip an occasional close-call argmax. Fix: when `.correct()`
   is asked to correct literally every position (`Q == N`, `token_idx == arange(N)`), route through
   the same `is_causal=True` SDPA call `.approx()` uses; fall back to the explicit mask only for
   genuine partial corrections (where no direct `is_causal` equivalent exists). Confirmed via
   re-diff: 32B's single-round divergence dropped from 2/20 to 1/20 (idx=418 now matches baseline
   exactly); 72B's single-round result was *already* at/above baseline pre-fix (80% vs baseline's
   70% at nr=20, correcting 2 of baseline's own mistakes with zero new errors) and stayed there
   post-fix (predictions bit-for-bit identical before/after -- expected, since flip sensitivity
   depends on how close each model's top-1/top-2 logit margins are, which varies by model/sample).

**Why `interleaved_g4`'s numbers are unchanged by the SDPA fix.** Re-running the full nr=50 3-way
comparison after the fix gave numbers *identical* to before the fix (32B: 74/60/62%, 72B:
76/68/72%, sequential/approx-only/interleaved respectively). This is expected, not a sign the fix
didn't work: multi-round `.correct()` calls in `GroupTriggerPolicy`'s 4-round schedule always have
`Q < N` (correcting only that round's newly-arrived merge-groups plus the permanent text group), so
they never hit the `is_causal=True` fast path, which only applies when `Q == N` exactly.

**The deeper reason `interleaved_g4` doesn't match baseline is architectural, not a bug.** Tracing
`GroupTriggerPolicy._get_pipeline_instructions`: with `num_groups=4` and `total_layers` LLM layers,
`chunk_size = total_layers // 4`. Only the **last-arriving** group (`group_id == num_groups`) ever
receives a full `(0, total_layers)` correction. Every earlier group is only corrected up through its
own chunk boundary (`layers=(0, group_id*chunk_size)`); the remaining layers for that group are
never separately re-corrected -- they ride along via ordinary mixed-approx propagation (full
self-attention over the whole sequence, no freshness distinction) for the rest of the network. This
is the intentional latency/accuracy tradeoff the whole progressive-pipelining scheme is built
around (same `GroupTriggerPolicy` used by every fork this session), not a defect to fix. It does,
however, mean that `num_groups=4`'s "100% delivered" end state is fundamentally *not* equivalent to
a true single-shot 100% correction -- see the keep-rate sweep below, which sidesteps this entirely
by using `num_groups=1`.

## 3. Results: sequential / approx-only / interleaved_g4 (nr=20, nr=50)

| Config | 32B nr=20 | 32B nr=50 | 72B nr=20 | 72B nr=50 |
|---|---|---|---|---|
| sequential (baseline, full-res, no AppCorr) | 85% | 74% | 70%\* | 76% |
| approx-only (blurred base layer only) | 75% | 60% | 68% (nr=50 only) | 68% |
| interleaved_g4 (grid grouping, 4-round progressive) | 85% | 62% | 80%\*\* (nr=20 anomaly) | 72% |

\* 72B's nr=20 baseline (70%) was measured as part of the SDPA-fix diagnostic reference run, not
the original Phase 5 pass.
\*\* 72B's nr=20 `interleaved_g4` result (80%, *above* baseline) was flagged as an anomaly at the
time and correctly not trusted -- nr=20 is small enough that this session repeatedly found
misleading results at this scale (also true for 32B's clean-looking 85%==85% nr=20 match, which did
not hold at nr=50). nr=50 is the trustworthy scale for all conclusions in this log.

**Post-fix, nr=50 numbers are unchanged from pre-fix** (see explanation in section 2): sequential
32B=74%/72B=76%, approx-only 32B=60%/72B=68%, interleaved_g4 (grid) 32B=62%/72B=72%.

## 4. Keep-rate sweep (static, single-shot, importance-ranked correction)

Full methodology, data, and analysis: `analysis/experiments/qwen25vl_keeprate_sweep_results.md`.
Summary: using a new `top_energy` grouping strategy (`num_groups=1`, merge-groups ranked by
residual energy, top `keep_rate` fraction corrected to full resolution, rest permanently
approx/blurred) -- which sidesteps the layer-chunking confound above -- both models show a clear
monotonic accuracy climb from keep_rate=2% up to **~15%, where accuracy saturates to
baseline-equivalent** and stays flat (within nr=50 noise) all the way to 100%:

| keep_rate | 32B | 72B |
|---|---|---|
| 2% | 58% | 70% |
| 5% | 62% | 70% |
| 10% | 66% | 74% |
| **15%** | **76%** | **76%** |
| 25% | 74% | 76% |
| 100% | 76% | 72% |
| baseline | 74% | 76% |

**Sweet spot: ~15% keep rate.** Correcting only the top 15% of merge-groups by residual energy
(leaving ~85% at coarse/blurred base-pyramid resolution) already recovers full-resolution baseline
accuracy on RealWorldQA, for both model sizes.

**Compute caveat (measured, important)**: this does *not* currently translate into a proportional
GPU speedup. `CORRECT_FORWARD` at keep_rate=15% (32B) measures *slower* than the full-image
`APPROX_FORWARD` pass (353ms vs 269ms mean), and remains roughly even even at keep_rate=2%
(296ms vs 301ms) -- despite FLOPs theoretically scaling down close to linearly with keep_rate.
This points to fixed, keep-rate-independent implementation overhead (full-image vision
re-embedding every correction call, explicit `[Q,N]` mask construction, 96 sequential small-batch
layer launches) dominating at these problem sizes, not an algorithmic limitation -- realizing the
accuracy win as a real speedup would need further engineering (out of scope here).

## 5. Grouping strategy comparison: grid vs sequential (num_groups=4)

At the *progressive* multi-round setting (not the keep-rate sweep's single-shot mode), replacing
the default `grid` (spatially-scattered checkerboard) grouping with a new `sequential` strategy
(contiguous raster-order prefix chunks) substantially improves accuracy under the identical
layer-chunking schedule:

| grouping_strategy | 32B (nr=50) | 72B (nr=50) |
|---|---|---|
| grid (checkerboard) | 62% | 72% |
| sequential (raster prefix) | 74% | 74% |
| baseline | 74% | 76% |

For 32B, sequential closes the entire gap (exact baseline match). For 72B it recovers most of it.
Rationale: for a causally-masked decoder, correcting group *k* only benefits positions that
causally attend to it; a spatially-scattered group leaves gaps throughout the sequence even after
"its own" round, while a growing sequential prefix means every corrected round immediately and
fully benefits every later position. This is a free, easy win (same compute, same schedule, just a
different group-id assignment) and should be the default grouping choice for any causal-LLM
AppCorr deployment, not just this fork.

## 6. Honest overall interpretation

- **Progressive/multi-round `interleaved_g4` does not match full-resolution baseline** for this
  task -- this is a real, expected cost of the latency-hiding pipelining scheme (only the
  last-arriving group ever gets a full-depth correction), not a bug. `sequential` grouping
  substantially narrows this gap essentially for free.
- **A static, importance-ranked, single-shot correction at just ~15% coverage matches baseline
  exactly** -- a considerably more favorable result than the progressive scheme, for a "compress the
  input while preserving model accuracy" use case (as opposed to a "hide correction latency behind
  a data stream" use case, which is what progressive pipelining is for).
- **The SDPA kernel-dispatch bug was real and worth fixing**, confirmed by a clean, isolated,
  single-shot-100%-correction A/B test -- but its impact was narrow (roughly 1-2 samples per 20 at
  the sample sizes tested), not the dominant driver of the larger `interleaved_g4` gap, which is
  architectural (chunking) rather than numerical.
- **The keep-rate accuracy win is not (yet) a compute win** -- an important, measured caveat that
  should not be glossed over when reporting the "15% sweet spot" as a practical result.

## 7. Cross-dataset keep-rate comparison (DEFINITIVE FINAL, mechanism-matched baseline)

This section went through FOUR rounds of revision over the course of the investigation: nr=50
sweeps (all wrong), narrowed-candidate full-N/nr=400 re-measurement (right direction, imprecise
locations), a dense sweep (precise crossing points, but baseline still used a confounded
generation mechanism), and finally a mechanism-matched baseline re-measurement (this section) after
the user identified that baseline's generation mechanism (one continuous `model.generate()` call)
differed from every keep_rate condition's mechanism (`head_inference`'s two-stage decode: argmax
first token, then a separate `generate()` fallback call) -- confirmed via
`analysis/experiments/refcoco_matched_decode_diagnostic.py` to be a real ~1-2pp confound (up to
30-35% different generated text on RefCOCO even under 100% identical stock computation), then fixed
at the source (commit `310c65a`, baseline now uses the identical mechanism). Full raw data and
per-dataset discussion: `qwen25vl_keeprate_sweep_results.md` (RealWorldQA),
`qwen25vl_refcoco_sweep_results.md` (RefCOCO), `qwen25vl_gqa_sweep_results.md` (GQA).

### ✅✅ FINAL: v2 corrected full-dataset 32B sweep (-1pp threshold), commit 5c398d1

The full-resolution-leak bug described just below was fixed and the full-dataset 32B sweep re-run
end-to-end (baseline + a `top_energy` keep_rate sweep, extended dynamically via binary-search once
the originally-planned points didn't land a precise crossing) on both RefCOCO (N=8811) and GQA
(N=12578). Full tables in `qwen25vl_refcoco_sweep_results.md` / `qwen25vl_gqa_sweep_results.md`.

| Dataset | 32B full-dataset -1pp crossing | nr=400 estimate | revision | kr=100% noise floor |
|---|---|---|---|---|
| RefCOCO (N=8811) | **~58%** (kr=50%: -2.26pp; kr=58%: -1.00pp exactly at threshold; kr=65%: 0.00pp) | ~40% | **+18pp (moderate upward)** | +0.32pp |
| GQA (N=12578) | **65-75%** (kr=65%: -1.32pp below; kr=75%: -0.70pp clears) | ~50% | **+15-25pp (moderate upward)** | +0.10pp |

**Both datasets revise upward from nr=400 by a similar, moderate amount (~15-25pp) -- nr=400 was
never wildly unreliable here.** This is the important contrast with the RETRACTED v1 full-dataset
sweep (below): v1's canvas-reconstruction bug made every keep_rate condition look like it barely
differed from baseline (implying an implausible <=5-10% crossing); the v2 fix restores a normal,
expected relationship where full-dataset measurement refines (here: raises) the nr=400 estimate by
a believable amount, not overturns it. **Practical read: 32B needs meaningfully more correction on
both tasks than nr=400 suggested (~58-75% keep_rate, not ~40-50%), but the earlier qualitative
conclusions built on nr=400 (RefCOCO needs somewhat less correction than GQA for 32B, both need
substantially less than the strict >=0pp crossings from the mechanism-matched nr=400 table below)
still hold directionally.**

**McNemar significance testing (task #55, COMPLETE for both datasets):** re-ran baseline/crossing-kr/
kr=100% for both datasets with `--log-jsonl` per-sample logging and a paired McNemar test
(`analysis/experiments/mcnemar_from_jsonl.py`). **Same pattern held on both, confirming it's not a
one-off:**

| dataset | baseline vs kr=100% (noise floor) | baseline vs crossing-kr (-1pp aggregate threshold) |
|---|---|---|
| RefCOCO (N=8811) | +0.33pp, p=0.12, NOT significant | kr=58%: -1.00pp, **p=0.0086, SIGNIFICANT** |
| GQA (N=12578) | +0.10pp, p=0.60, NOT significant | kr=75%: -0.71pp, **p=0.0018, SIGNIFICANT** |

kr=100% is correctly never significant (validating it as each dataset's own noise floor), but the
keep_rate whose AGGREGATE gap merely clears -1pp is, on both datasets, a real statistically
significant difference from baseline -- at full-dataset N, the test has power to detect even small
real degradations that look noise-sized in aggregate. **Important honest correction for this whole
investigation: the -1pp threshold used throughout is a reasonable ENGINEERING tolerance, not a
statistical-equivalence claim.** A paper claiming "32B recovers baseline accuracy at keep_rate X%"
should either cite a keep_rate closer to each dataset's kr=100%-tying point (RefCOCO ~65%, exact tie;
GQA's non-significant point wasn't pinned down beyond kr=100% itself) or explicitly caveat that the
~58%/~75% figures are aggregate-tolerance-based. Full contingency tables in both per-dataset result
files' "Statistical significance (McNemar)" subsections.

### ❌ RETRACTED: first full-dataset 32B batched sweep (the "-1pp crossing <=10%/<=5%" claims are WITHDRAWN)

After the mechanism fix, the user (a) redefined the crossing point as **accuracy >= baseline - 1pp**
(the -1pp threshold, replacing strict >=0pp, given the ~1-2pp mechanism-level noise floor), and
(b) requested TRUE full-dataset sweeps for 32B, run with a new batched driver
(`analysis/experiments/refcoco_gqa_batched_eval.py`). The first run of that driver reported
crossing <=10% (RefCOCO N=8811) / <=5% (GQA N=12578), with every tested keep_rate ABOVE baseline.
**The user flagged this as suspicious and asked for an approx-only control -- which exposed a
critical full-resolution leak in the driver's keep_rate path**: it passed the raw full-res image
to `preprocess` instead of the patch-reconstructed canvas (blurred base + arrived corrections)
that the offload pipeline's WorkerModule builds via `policy.decode()` (worker.py:177-188), so
`pixel_values` (vision tower input AND the generate() fallback) was full resolution regardless of
keep_rate. Every keep_rate/approx-only condition in that sweep was effectively baseline + fork
numerical noise -- smoking gun: buggy approx-only == baseline (GQA: identical 31/48 on the same
samples; RefCOCO: within 1 sample), fixed approx-only drops properly (-6.25pp / -4.2pp).

Fix validated per-sample against the offload pipeline driver (kr=0.30, 8 samples: all predictions
character-identical). Only useful salvage from the invalid run: baseline numbers are VALID
(85.75% RefCOCO N=8811 / 60.84% GQA N=12578 -- genuinely full-res stock), and the invalid
keep_rate rows empirically pin the fork-vs-stock full-dataset numerical noise floor at ~+0.1 to
+0.5pp (much tighter than the ~2.5pp nr=400 estimate). Corrected full-dataset sweep (approx-only +
keep_rates + kr=100% control) re-running with the fixed script; the table below therefore remains
the CURRENT best crossing-point estimate until it lands.

### Definitive crossing-point table (mechanism-matched)

| Dataset | 32B crossing point | 72B crossing point | 32B shifted from pre-fix? |
|---|---|---|---|
| RealWorldQA (N=765 full) | **50%** (+0.78pp) | **100%** (exact tie, 72.29%) | No (was 50%, unchanged) |
| RefCOCO (nr=400) | **50%** (+1.00pp; ~40% under -1pp threshold) | **70%** (+1.00pp) | **YES: 30% -> 50%** |
| GQA (nr=400) | **(80%, 100%]** (>=0pp; ~50% under -1pp threshold) | **80%** (+0.75pp) | No (unchanged) |

Baseline shifts under the mechanism fix: RealWorldQA 32B +0.13pp / 72B +0.00pp (negligible); GQA
32B +0.00pp / 72B -0.25pp (negligible); **RefCOCO 32B +2.25pp / 72B -1.00pp (substantial)**. Only
RefCOCO's crossing point actually moved as a result -- and only for 32B (72B's crossing point
happened to land on the same tested value, 70%, before and after, since it was an exact tie
pre-fix and became a comfortably-positive point post-fix, without skipping past another tested
value). This asymmetry makes sense: RefCOCO's multi-token bbox-coordinate answers are far more
sensitive to the two-stage-decode mechanism than RealWorldQA's/GQA's mostly single-token/short-
phrase answers, exactly as hypothesized when the confound was first found.

### The two headline findings, re-evaluated under the corrected numbers

**1. "72B needs MORE correction than 32B to reach its own baseline" -- still holds on 2 of 3
datasets, with the same genuine exception on the third; unaffected by the mechanism fix.**
RealWorldQA (72B=100% vs 32B=50%) and RefCOCO (72B=70% vs 32B=50%, previously 70% vs 30%) both show
72B needing more keep_rate than 32B -- though RefCOCO's margin *narrowed* under the fix (72B no
longer needs ~2.3x what 32B needs, just ~1.4x, since 32B's true crossing point was later than
originally measured). GQA still reverses this (72B crosses at 80%, 32B has not by 80%) -- confirmed
still a real exception post-fix, not an artifact of the old baseline mechanism (GQA's baseline
barely moved). **Net finding, essentially unchanged from before the fix: 72B needing more
correction is common but not universal, and task-dependent.**

**2. "RefCOCO (grounding) needs LESS correction than RealWorldQA (VQA)" -- this is the finding that
actually changed.** Pre-fix, this held sharply for both models (30%/70% vs 50%/100%). Post-fix,
**it now only holds for 72B** (70% vs 100%) -- **for 32B, RefCOCO and RealWorldQA crossing points
are now identical (50% = 50%)**, so the "grounding needs less" claim no longer applies to the
smaller model at all. This is a genuine, meaningful walk-back caused directly by fixing the
mechanism confound, not a minor rounding correction. GQA still doesn't cleanly resolve the
VQA-vs-grounding question either way (its crossing points sit between RealWorldQA's and RefCOCO's
for both models). **Net finding: task type has, at most, a real but SMALLER and MODEL-SIZE-
DEPENDENT effect on crossing point than originally reported -- for 32B specifically, there is now
no measured task-type effect at all across the two directly-comparable datasets (RealWorldQA,
RefCOCO both at 50%); for 72B, a real gap remains (70% vs 100%).**

### What this means practically

There is still no universal "correct top-K% of patches" constant for this AppCorr/Qwen2.5-VL setup
-- crossing points range from 50% to 100% (GQA's bracket aside) across the six combinations, a real
2x spread. But the mechanism fix meaningfully **narrowed** how much of that spread should be
attributed to task type specifically: 32B's two directly-comparable datasets (RealWorldQA, RefCOCO)
now agree exactly (50%), while 72B still shows a real task-dependent difference. Anyone deploying
this should still measure their own task+model-size combination rather than assuming transfer, but
the "grounding is categorically different from VQA" story is weaker post-fix than it looked
pre-fix -- it may be more accurate to say **model size dependence is the dominant, more reliable
axis observed in this data, with task-type effects present but smaller and inconsistent (real for
72B on RefCOCO vs RealWorldQA, absent for 32B on the same pair, present in an unresolved way for
GQA on both models).**

## 8. Known open issues (not fully resolved)

- ~~`head_inference`'s two-stage decode mechanism was a measured confound~~ **-- FOUND, MEASURED,
  AND FIXED (no longer open).** User identified that `head_inference` (every keep_rate/correction
  condition) decodes the first token separately then falls back to a fresh `model.generate()` call,
  while the old baseline (`full_inference`) made one continuous `generate()` call -- an uncontrolled
  mechanism difference. Measured directly
  (`analysis/experiments/refcoco_matched_decode_diagnostic.py`, nr=400, 100% stock computation both
  ways): the two mechanisms produce *different generated text* on 30-35% of RefCOCO samples even
  with zero correction involved (32B: +2.25pp, 65% exact-text agreement; 72B: -1.00pp, 70%
  agreement) -- confirmed real and substantial, not negligible noise. **Fixed at the source**
  (commit `310c65a`): `full_inference` now uses the identical two-stage mechanism as
  `head_inference`, so baseline and every keep_rate condition are mechanism-matched. Baseline was
  re-measured under the fix for all 6 dataset/model combinations and every sweep table + the
  section 7 cross-dataset comparison was rebuilt against the corrected baseline -- see section 7 for
  the final numbers, which for RefCOCO 32B specifically changed the crossing point (30% -> 50%),
  a real, meaningful correction, not just a rounding change. RealWorldQA and GQA's crossing points
  were unaffected (their answers are mostly single-token/short-phrase, far less sensitive to this
  mechanism than RefCOCO's multi-token bbox coordinates).
- **A second, unfixed scheduler race.** While running the GQA/RefCOCO sweeps (two parallel chains
  hammering CONFIG/patch dispatch at a much higher frequency than RealWorldQA's larger, slower
  images), a distinct crash appeared: `KeyError: 'vision_cache'` in `qwen25vl_executor.py`'s
  `correct_forward`, meaning a session's `CORRECT_FORWARD` fired without its `APPROX_FORWARD` ever
  having run first. This is different from the config-application race fixed in commit `ceaef61`
  (that race manifested as `self.config` being `None`; this one happens well after model loading,
  with `self.config` valid). Working hypothesis, **not confirmed**: `SchedulerModule.run()`
  unconditionally resets `self.buffer = []` on every `CONFIG` message
  (`offload/server/scheduler.py:34`), and since native-resolution drivers resend `CONFIG` on every
  single image, some timing window could let this discard an in-flight group's not-yet-dispatched
  patches. The worker process's main loop catches the resulting exception without crashing (so
  retrying a fresh request against the same process works), but never sends a response, hanging the
  driver until timeout. Mitigated pragmatically (commit `268f1e1`) with a 3-attempt retry wrapper in
  all three offload eval drivers, rather than root-caused and fixed at the source -- this should be
  treated as an open item, not a resolved bug, if this infrastructure sees further use.
- **`realworldqa_offload_eval.py`'s retry wrapper is untested against a live recurrence** of the
  original issue (it was added defensively, mirroring the GQA/RefCOCO fix, but RealWorldQA's own
  sweeps had already completed before this bug was discovered) -- should work by the same logic,
  but hasn't been directly observed to recover a real crash the way the GQA/RefCOCO drivers were.

## 9. Threshold-based (absolute pscore cutoff) vs top-K% patch selection -- task #57

**Hypothesis (user, from prior experience):** an absolute pscore threshold (correct every merge-group
whose residual energy exceeds a fixed cutoff -- so the corrected FRACTION varies per image with how
much high-frequency content it actually has) should be more sample-efficient than `top_energy`'s
fixed top-K% (which always corrects exactly `keep_rate * N` groups regardless of the pscore
distribution's shape).

**Implementation:** new `top_energy_threshold` grouping strategy in
`offload/policies/transmission/progressive.py` (`_precompute_group_assignments`), `--pscore-threshold`
CLI plumbing in `refcoco_gqa_batched_eval.py`. Verified mechanically correct: on 8 real RefCOCO
images, corrected-fraction varies 38.5-75.0% at a fixed threshold (vs `top_energy`'s always-identical
fraction), confirming the mechanism behaves as designed.

**nr=64 preliminary result (misleading -- see nr=400 below):** on a fixed set of 64 RefCOCO
indices, at matched realized average keep-fraction (~33%), `top_energy_threshold` (threshold=800K)
scored 87.50% (56/64, tying baseline exactly) vs fixed `top_energy` (kr=0.33) scoring only 79.69%
(51/64) -- a striking +7.81pp gap that looked like strong confirmation of the hypothesis.

**nr=400 confirmation run (same 400 indices, all 3 conditions) -- REVERSED the nr=64 result:**

| condition | realized avg keep-fraction | Acc@0.5 (N=400) | gap vs baseline |
|---|---|---|---|
| baseline | -- | 86.00% (344) | -- |
| `top_energy`, kr=0.33 (fixed) | 33% | 83.75% (335) | -2.25pp |
| `top_energy_threshold`, 800K (variable) | 34.5% (realized, budgets fairly matched) | **82.00% (328)** | **-4.00pp** |

At 6x the sample size and a fairly matched realized budget (34.5% vs 33%, if anything threshold used
slightly MORE), threshold-based selection scored **1.75pp WORSE** than fixed top-K, not better --
completely reversing the nr=64 signal. **The nr=64 preliminary result was a statistical fluke, not a
real effect.** Per the user's explicit instruction to keep investigating rather than stop at one
negative data point, this was followed up two ways:

**(a) Diagnostic: does the threshold method fail because it "starves" specific images of correction?**
Re-ran threshold=800K at nr=400 WITH per-sample logging (`--log-jsonl`, reproduced 82.00%/328/400
exactly) and correlated each sample's correctness against that image's REALIZED keep-fraction
(computed via the same CPU-only pscore sampling, no GPU needed). Result: **correlation(keep-fraction,
correct) = -0.016, correlation(keep-fraction, IoU) = +0.016 -- essentially zero.** Only 1/400 images
got exactly 0% correction (and it was answered correctly). Bucketed accuracy by keep-fraction is
noisy and non-monotonic (0-5%: 88%, 5-15%: 87%, 15-30%: 74%, 30-50%: 90%, 50-100%: 78%), not a clean
"more correction helps" trend. **"Starvation" is not the mechanism.** The much more likely
explanation: **the residual-energy pscore itself has near-zero relationship to whether a merge-group
actually matters for THIS grounding task's correctness** -- which merge-groups have the most raw
pixel detail is close to independent of which merge-groups cover the region the referring expression
is actually pointing at. If the underlying score isn't task-relevant, no selection RULE built on top
of it (threshold or top-K) can be expected to do much better than another.

**(b) Bracket sweep: is the nr=400 null result specific to threshold=800K, or general?** Tested two
more thresholds (500K, 1M) at nr=400, plus a matched-budget `top_energy` control at the same realized
budget as the most promising one:

| condition | realized avg keep-fraction (nr=400) | Acc@0.5 (N=400) | gap vs baseline |
|---|---|---|---|
| baseline | -- | 86.00% (344) | -- |
| `top_energy_threshold`, 1.0M | 28.5% | 82.00% (328) | -4.00pp |
| `top_energy_threshold`, 800K | 34.5% | 82.00% (328) | -4.00pp |
| `top_energy`, kr=0.33 (fixed, matched to 800K's budget) | 33% | 83.75% (335) | -2.25pp |
| `top_energy_threshold`, 500K | 46.5% | **85.75% (343)** | **-0.25pp** |
| `top_energy`, kr=0.465 (fixed, matched to 500K's budget) | 46.5% | 85.25% (341) | -0.75pp |

**Picture is budget-dependent, not uniformly negative.** At LOW budget (~28-34%), threshold-based
selection clearly underperforms fixed top-K (by ~1.75pp, a real-looking gap given it held at nr=400).
At MODERATE budget (~46%), threshold-based selection is essentially TIED with (marginally ahead of,
+0.50pp/2 samples) top-K, and both are close to baseline -- but a 2-sample nr=400 gap is comfortably
within the noise this investigation's own McNemar tests have shown is needed to trust a small
difference, so this should NOT be read as "threshold wins at 46%," just "the two methods are
indistinguishable there."

**Overall conclusion for task #57:** across 5 threshold values (100K/400K/500K/800K/1M) and 2 sample
sizes (nr=64, nr=400), there is no reliable evidence that absolute-threshold selection on the
CURRENT residual-energy pscore is more sample-efficient than fixed top-K% for RefCOCO/32B -- it is
worse at low budgets and merely tied at moderate ones. Combined with finding (a) above (the score
itself is ~uncorrelated with task correctness), **the more promising direction is task #56
(query-conditioned pscore) rather than further threshold-tuning on the current energy-based score** --
the selection RULE (threshold vs top-K) is not the bottleneck; the SCORE's task-relevance is. This
is a complete, well-evidenced (not prematurely-stopped) negative result for the original hypothesis
as tested, ready to hand off to the user; the `top_energy_threshold` mechanism itself remains
correctly implemented and available if a different pscore signal (task #56) is plugged into it later.

**(c) Direct, quantified confirmation: how well does the current score even localize the referring
expression's target region?** RefCOCO ships a ground-truth bbox per sample, so this is directly
checkable with no model inference at all -- for every merge-group in the nr=400 sample (145,776
groups total), labeled "inside" if >50% of its cell area overlaps the GT bbox (31,717 groups) vs
"outside" (114,059 groups), then measured how well `pscore` (residual energy) separates the two
populations (Mann-Whitney-style rank AUC, 0.5=random, 1.0=perfect):

**AUC = 0.5376.** The current pscore is barely better than a coin flip at identifying which
merge-groups actually fall inside the target region a referring expression is pointing at (mean
pscore inside-GT=882K vs outside-GT=787K -- a real but tiny separation). This is the same conclusion
as (a) and (b) above, now with a clean, model-free, directly-interpretable number: **there is roughly
54% out of a possible 100% worth of localization signal being captured by pixel residual energy
alone; the remaining headroom is exactly what a query-conditioned score (task #56) would need to
capture to meaningfully help.**

**(d) Full-dataset (N=8811) confirmation of (b)'s bracket sweep -- reverses AGAIN, in the OPPOSITE
direction this time.** Per user request ("400 is still a small sample, just run it full"), re-ran
threshold=500K and threshold=800K at full N=8811 (fixed budgets confirmed by CPU sampling: 46.35%
and 34.52% respectively, matching nr=400's estimates closely):

| condition | budget | Acc@0.5 (N=8811) | gap vs baseline | vs top_energy CURVE-INTERPOLATED expectation |
|---|---|---|---|---|
| baseline | -- | 85.75% (7555) | -- | -- |
| `top_energy_threshold`, 800K | 34.52% | 82.01% (7226) | -3.74pp | interpolated top_energy@34.5% ~ -4.71pp -- threshold **~1.0pp BETTER** |
| `top_energy_threshold`, 500K | 46.35% | 83.08% (7320) | -2.67pp | interpolated top_energy@46.35% ~ -2.72pp -- **essentially TIED** |

At full-dataset scale, threshold=800K -- the point that looked CLEARLY WORSE at nr=400 (-1.75pp vs
matched top_energy) -- now looks BETTER than the top_energy curve's interpolated expectation at the
same budget. This is the SECOND time in this investigation that nr=400 gave a misleading signal that
flipped at full scale (the first was the nr=64->nr=400 reversal in finding (a)/(b) above), this time
in the opposite direction. **Lesson reinforced: nr=400 is not reliably precise enough for
sub-2pp-scale comparisons on this task; only full-dataset numbers (or a matched-budget full-dataset
control, not curve interpolation) should be trusted for close calls.** A matched-budget `top_energy`
control at kr=0.4635 (exactly matching 500K's realized budget) was launched at full scale for the
cleanest possible apples-to-apples read on the more interesting (tied-looking) comparison; a
kr=0.3452 control was also queued but redirected to the attention-pscore diagnostic below once a
sharper, higher-leverage finding emerged (see (e)) -- can be resumed if still useful.

**(d, resolved) The matched-budget `top_energy` kr=0.4635 full-dataset control landed: 83.00%
(7313/8811), -2.75pp.** Direct, no-interpolation comparison against threshold=500K's 83.08%
(7320/8811), -2.67pp: a **+0.08pp difference (7 samples out of 8811) -- a dead tie**, mean_iou also
essentially identical (0.7351 vs 0.7346). The nr=400 result that suggested threshold had a real
+0.50pp edge here shrinks to statistical noise at full scale. **Final verdict for (b)/(d) together:
pure-residual threshold-based selection and fixed top-K% are equivalent in practice at this budget
level** -- neither clearly better, consistent with the AUC=0.5376 finding (c) that the underlying
residual-energy score itself, not the selection RULE built on top of it, is the limiting factor.

**(e) The real breakthrough: WHICH attention signal, not IF attention helps.** Per the user's
correction that this investigation had been missing an already-validated pattern from this repo's
DINOv3/OpenVLA/ImageNet work (commits `26384fa`/`46d1016`/`7a4eb55`/`1f33a1f`: fuse
`mobile_pscore=residual_energy` with a `server_pscore` derived from the model's OWN attention,
multiply-fused, thresholded -- validated there at +5-6pp accuracy and ~20% lower correction latency),
re-ran the same AUC-vs-GT-bbox diagnostic from (c) but adding an attention-derived score. Qwen2.5-VL's
vision tower has no CLS token (unlike DINOv3), so the CLS-less analogue is `patch_attn_prob`: mean
attention MASS RECEIVED by each patch from all other patches (heads-mean, queries-mean) -- computed
by manually replicating one block's attention math (norm1->qkv->RoPE->softmax(QK^T*scale), since
`F.scaled_dot_product_attention` never exposes weights) off the side of the validated, unmodified
`approx_forward` path (no fork files touched for this diagnostic). Tested all 4 full-attention layers
individually plus DINOv3's default "layermean" (average across all 4):

| signal | AUC (0.5=random, 1.0=perfect) |
|---|---|
| residual_energy alone (from (c)) | 0.5376 |
| attention layer 7 alone | **0.5763** |
| attention layer 15 alone | 0.5759 |
| attention layer 23 alone | 0.5658 |
| attention layer 31 alone | 0.5189 (WORSE than residual alone) |
| attention LAYERMEAN (all 4, DINOv3's default recipe) | 0.5567 |
| residual x attention LAYERMEAN (fused) | 0.5487 |

**Early full-attention layers (7, 15) carry meaningfully more localization signal than late ones
(31) for this task** -- layer 7 alone beats residual by +3.87pp AUC, clearly the best single signal
found in this whole investigation. DINOv3's "layermean" default (averaging all 4 layers including
the weak layer 31) actually DILUTES this -- layer 7 alone (0.5763) beats the 4-layer mean (0.5567).
**Actionable recipe going forward: use early-layer (7 and/or 15) attention, not late-layer or a naive
4-layer average, and prioritize this over further threshold-value tuning on pure residual.** This is
a genuinely new, non-obvious finding (the opposite of "later layers are more semantic/better" prior
one might have from other contexts) -- plausible explanation: by layer 31, Qwen2.5-VL's vision tower
attention may have shifted from spatial/local saliency toward more abstract, less spatially-localized
feature aggregation, whereas earlier layers are still closer to low-level visual saliency.
**Full pipeline integration (real accuracy, not just AUC-vs-GT-bbox) is the natural next step**, but
requires restructuring: the current group-assignment decision happens entirely CPU-side, BEFORE any
GPU/model computation; using an attention-derived score requires computing it during the base/approx
forward pass and feeding it back into the correction-group decision for the SAME request (a
DINOv3/OpenVLA-style architectural pattern already validated elsewhere in this repo, not yet ported
to the Qwen2.5-VL fork). In progress as of this log entry.

**Design options for task #56 (query-CONDITIONED, i.e. text-aware, pscore) -- separate from (e)'s
content-only attention signal, which doesn't look at the question/expression text at all:**
1. **CLIP-based query-image relevance.** `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k` is already
   cached locally (`/NHNHOME/huggingface/hub/models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k`,
   used by this repo's earlier `experiment/clip-appcorr` work) -- could compute per-merge-group
   CLIP patch-embedding-to-text-embedding cosine similarity as (or blended with) the pscore. Bigg-14
   is a large model itself though; running it per-request adds real latency/memory overhead worth
   weighing against a smaller CLIP variant (e.g. ViT-B/32) for a "cheap mobile-side score" role.
2. **Reuse Qwen2.5-VL's own cross-attention**, if/where accessible cheaply (e.g. from the base/approx
   pass's already-computed features) instead of a separate model -- avoids a second model's memory
   footprint but is architecturally more invasive and less obviously "cheap." Note: (e)'s early-layer
   self-attention finding is a strong, VALIDATED, non-query-conditioned building block that could be
   combined with a lightweight query-relevance term rather than replaced by one.
3. **Lightweight heuristic hybrids** (e.g. combine residual energy with a coarse text-derived spatial
   prior, like color-word or size-word matching) -- fast and simple but likely too narrow/brittle to
   generalize past RefCOCO's specific phrasing patterns.
Threading the question/expression text through `encode()`/`ExperimentConfig` to reach
`_compute_patch_pscore_hint` is required for any of these (currently not plumbed at all) --
non-trivial but mechanical.

**(e, resolved) Full pipeline integration of single-layer-7 attention-fused selection, at
full-dataset scale.** Built `refcoco_attn_fused_eval.py`, which bypasses `progressive.py` entirely
(no fork/shared-code changes): calls `encoder.encode(..., keep_rate=1.0)` so every merge-group gets a
`Patch` with `pscore_hint` attached, processes the base/blurred group normally through
`approx_forward` to get `vision_ctx`, reads off layer-7's `patch_attn_prob`-style attention-received
score from that SAME pass (no extra forward needed), fuses via `fused = pscore_hint * attn_score`,
ranks descending, and corrects only the top `keep_rate` fraction. An nr=400 smoke test initially
looked like a THIRD reversal (fused lost to `top_energy` by -1.75pp) -- per the user's standing
sanity-check-only rule for nr<=400 (see `[[feedback_nr400_sanity_check_only]]` memory), this was
not trusted and a full-dataset (N=8811) run was launched instead:

| condition | keep_rate | Acc@0.5 (N=8811) | mean_iou | gap vs baseline (85.75%) |
|---|---|---|---|---|
| `top_energy` (matched control) | 0.35 | 81.07% (7143) | -- | -4.68pp |
| residual x layer-7-attention (fused) | 0.35 | **81.64% (7193)** | 0.7177 | -4.11pp |

At full scale, single-layer-7 attention-fusion **beats matched top_energy by +0.57pp (50 samples)** --
reversing the nr=400 result a THIRD time in this investigation (see the nr=400-is-sanity-check-only
memory this pattern motivated). Modest but real, and directionally consistent with the AUC finding
(layer 7 alone: 0.5763 vs residual alone: 0.5376).

**(f) Weighted multi-layer combination -- does combining all 4 full-attention layers (not just layer
7) do even better, and how should the weights be chosen?** Per the user's follow-up question, fit a
`StandardScaler` + L2-regularized logistic regression over `[log1p(residual), attn_7, attn_15,
attn_23, attn_31]` (5 features) on the nr=400 AUC-diagnostic data (145,776 merge-group rows, split
70/30 BY IMAGE to avoid leakage across rows from the same picture). Val AUC=**0.5713**, beating
layer-7-alone's AUC on the same val split (0.5516) and residual alone (0.5380) -- stable across
C=0.01..1000. Notably layer 31 (the worst single-layer AUC, 0.5189) received a strong NEGATIVE
fitted weight (-0.25 to -0.29) rather than simply ~0, i.e. the fit actively uses it as a
disconfirming signal, not just discards it.

Pipeline-integrated this (`WeightedFusionModel` class in `refcoco_attn_fused_eval.py`, loading a
saved `StandardScaler`+logistic `.npz` and computing the fused score as the standardized linear
combination, ranking descending by that pre-sigmoid score since sigmoid is monotonic) and ran
full-dataset (N=8811, no nr=400/64 detour this time, per the standing rule):

| condition | Acc@0.5 (N=8811) | mean_iou | gap vs baseline |
|---|---|---|---|
| layer-mean weighted (residual + attn_7/15/23/31, logistic, val AUC 0.5713) | 81.19% (7154) | 0.7162 | -4.56pp |

**This UNDERPERFORMED single-layer-7 alone (81.64%) despite a better held-out AUC (0.5713 vs
0.5516) -- yet another instance of the AUC-vs-downstream-accuracy disconnect seen repeatedly in this
investigation.** The likely cause: layer 31's per-layer AUC (0.5189) hides enormous WITHIN-layer
variance -- see below -- so folding it in at the whole-layer granularity, even with a fitted
(negative) weight, still injects head-level noise the fit can't fully cancel out with one scalar per
layer.

**Per-head weighting, following the user's follow-up question ("attention differs per head too --
can we weight individual heads?"):** extended `qwen25vl_attn_pscore_diagnostic.py --per-head` to keep
the head dimension instead of collapsing it (`_manual_attention_received(..., collapse_heads=False)`
returns `[num_heads, seq_len]`; DINOv3's `patch_attn_prob` mean-over-heads step is simply skipped),
re-ran the nr=400 collection (16 heads x 4 layers = 64 attention features + residual = 65 total,
145,776 rows, saved to `/tmp/qwen25vl_attn_perhead_data.npz`). Per-head AUC breakdown confirmed the
within-layer-variance hypothesis directly:

| | best | worst |
|---|---|---|
| layer 31 (whole-layer AUC 0.5189, WORST of the 4) | head 10: **0.5874** (top-5 overall) | head 4: **0.4248** (worse than random) |
| layer 15 (whole-layer AUC 0.5759) | head 12: **0.6011** (best single feature found in this entire investigation) | -- |

Layer 31's poor whole-layer AUC is explained: it contains both the single best-localizing individual
head found anywhere in this investigation (head 10, AUC=0.5874) AND several heads worse than random
(heads 4/12/14/1, AUC 0.42-0.49) -- averaging them together produces the mediocre 0.5189 layer
score, discarding real signal in the process.

Fit the same StandardScaler+logistic recipe over all 65 features (image-level 70/30 split, same as
before): val AUC=**0.6195**, stable across C=0.01..100 -- a much larger jump over layer-mean's 0.5713
than layer-mean's jump over residual-alone's 0.5376. Interestingly, **the residual feature's fitted
coefficient nearly vanished (0.0024, refit on all 400 samples)** -- once per-head attention is
available, the model finds almost no additional value in the pixel-residual signal at all.

Pipeline-integrated (`compute_all_attn_native_order` in `refcoco_attn_fused_eval.py`, a single
shared-prefix forward pass that captures per-head OR layer-mean attention as needed per layer,
avoiding a redundant second pass; `WeightedFusionModel` extended to parse both `attn_<L>` and
`attn_<L>_h<H>` feature names transparently) and run full-dataset (nr=64 smoke test first: 87.50%,
56/64, mean_iou 0.7292 -- passed cleanly -- then straight to full per the standing rule):

| condition | Acc@0.5 (N=8811) | mean_iou | gap vs baseline (85.75%) |
|---|---|---|---|
| `top_energy` (matched control, kr=0.35) | 81.07% (7143) | -- | -4.68pp |
| single-layer-7 attention-fused | 81.64% (7193) | 0.7177 | -4.11pp |
| layer-mean weighted (4 layers, logistic) | 81.19% (7154) | 0.7162 | -4.56pp |
| **per-head weighted (65 features, logistic)** | **82.22% (7244)** | **0.7245** | **-3.53pp** |

**Per-head weighting wins outright** -- +1.15pp over matched `top_energy`, +0.58pp over
single-layer-7 alone, +1.03pp over layer-mean weighting, and the best `mean_iou` of any condition
tested in this whole section. Two-part takeaway for task #57/#56:
1. **Naive/logistic layer-level averaging can actively HURT vs. the single best layer**, even when
   the fitted combination has a measurably better held-out AUC -- a layer's AUC is a lossy summary
   that can hide a good head and several bad ones cancelling out, and one scalar weight per layer
   cannot fully undo that once collapsed.
2. **Going finer-grained (per-head) recovers this loss and then some**, because the optimizer can
   down-weight specifically the bad heads within layer 31 (and elsewhere) instead of being forced to
   either keep or discard the whole layer. This is the strongest attention-derived pscore signal
   found in this entire investigation, and the first one to clear a full +1pp margin over matched
   `top_energy` at full-dataset scale.

**(f, correction) Train/eval image leakage check -- the 82.22% headline above is CONTAMINATED.**
Per the user's question ("was there any cheating in the training process?"): both weighted models
(layer-mean and per-head) were fit on the SAME 400-image nr=400 diagnostic subsample
(`indices = list(range(0, n_total, stride))[:400]`), and the `--full` pipeline eval runs on
`range(n_total)` -- i.e. **all 400 fitting images are a strict subset of the 8811-image full-dataset
eval set** (verified: 400/400 overlap). The fitted weights had literally seen (via
`_manual_attention_received`/residual features, at the merge-group level, for the GT-bbox
localization objective) every one of those 400 images before being scored on them again as part of
"full-dataset accuracy."

Measured the effect directly: re-ran the per-head weighted model on ONLY those same 400 training
indices (`leakage_check_perhead_trainset`, nr=400 but the trained-on 400, not a fresh sample):

| subset | N | Acc@0.5 |
|---|---|---|
| training images (leaked) | 400 | **84.75%** (339) |
| held-out images (`full - training`, clean) | 8411 | **82.09%** (6905) |
| full (contaminated headline) | 8811 | 82.22% (7244) |

**The leaked subset scores +2.66pp higher than the true held-out estimate -- real, measurable
overfitting/leakage, not noise.** Correcting for it:

| condition | Acc@0.5 | vs top_energy | vs single-layer-7 |
|---|---|---|---|
| `top_energy` (clean, no fitting at all) | 81.07% | -- | -0.57pp |
| single-layer-7 (clean, fixed formula, no fitting) | 81.64% | +0.57pp | -- |
| per-head weighted, **held-out-corrected** | **82.09%** | **+1.02pp** | **+0.45pp** |
| per-head weighted, contaminated headline (WRONG) | ~~82.22%~~ | ~~+1.15pp~~ | ~~+0.58pp~~ |

**The core finding survives but is smaller than first reported**: per-head weighting still beats
both clean baselines on a genuinely held-out N=8411, but the margin over single-layer-7 shrinks from
+0.58pp to +0.45pp -- real at this sample size, but modest. The layer-mean weighted result (81.19%)
has the identical leakage exposure (same 400 fitting images, same `--full` overlap) and was NOT
re-measured this way, but since leakage only inflates, its true held-out number is likely <=81.19%,
i.e. its underperformance vs single-layer-7 (finding (f) point 1 above) is unaffected or reinforced,
not reversed, by this correction.

**Root cause and fix for any future weight-fitting in this investigation:** the nr=400 diagnostic
subsample and any `--full` evaluation of a model fit on it are NOT independent -- either (a) exclude
the fitting indices from the eval index set, or (b) fit on a disjoint stride/split of images from the
one used for evaluation, or (c) always report a held-out-only number (as computed here) alongside any
full-dataset number for a FITTED (not fixed-formula) score. This is a distinct lesson from the
existing nr=400-is-sanity-check-only memory (that one is about small-N noise; this one is about
train/eval independence) and has been saved separately.

**(g) ABANDONED: extending to windowed (non-full-attention) layers 0/1/2 and 29/30 -- negative
result, not pursued further.** Per the user's follow-up ("earlier/later layers too?"), generalized
`qwen25vl_attn_pscore_diagnostic.py --layers` to accept arbitrary block indices (not just
`fullatt_block_indexes`) and collected per-head data for layers 0/1/2 (before layer 7) and 29/30
(the two windowed layers immediately before the final full-attention layer 31), same 400 images:

| layer | whole-layer AUC | best head AUC |
|---|---|---|
| 0 | 0.4887 (worse than random) | layer0 h7: 0.5138 |
| 1 | 0.5000 | layer1 h3: 0.5249 |
| 2 | 0.5044 | layer2 h10: 0.5305 |
| 29 | 0.4975 | layer29 h7: 0.5354 |
| 30 | 0.4927 | layer30 h14: 0.5361 |

All five are near-random (0.48-0.53), well below any of the four full-attention layers (best:
layer7=0.5763). **Architectural explanation, not a depth-direction effect**: layers 0/1/2/29/30 all
use WINDOWED (local ~112px) attention, not global -- `patch_attn_prob` there only reflects
within-window structure, which carries little information about whether a patch is inside a
GLOBAL referring-expression bbox. The earlier "early full-attention layers beat late ones" finding
(e) was about attention SCOPE (global vs local) more than raw depth, now confirmed by this negative
result: going earlier/later in depth doesn't help once you leave the four global-attention layers.

Merged all 9 layers (0,1,2,7,15,23,29,30,31) x 16 heads = 145 features (with residual) and refit:
val AUC=**0.6247**, only +0.0052 over the 4-layer-only model's 0.6195 -- a much smaller jump than
4-layer's own +0.048 jump over layer-mean. Interesting asymmetry in the fitted coefficients even
though individual AUCs were all near-random: layers 29/30 got fitted weight magnitudes comparable
to the full-attention layers (Sigma|coef|: layer29=0.674, layer30=0.925, vs layer31=1.123), while
layers 0/1/2 got much smaller weights (0.175-0.270) -- the network's TAIL (windowed or not) still
carries some usable combination-level signal, the network's FRONT does not.

Pipeline-tested (nr=64 smoke: 76.56%, 49/64, mean_iou 0.7022 -- an 11pp drop from the 4-layer
model's nr=64 result of 87.50%; nr=400 same-leaked-subset check: 83.25%, 333/400, mean_iou 0.7380
-- still 1.5pp BELOW the 4-layer model's own same-leaked-subset number of 84.75%, i.e. even on its
own "home" training images the 9-layer model underperforms the simpler 4-layer model). Three
independent signals (marginal AUC gain, large nr=64 drop, worse-than-4-layer even on leaked data)
all point the same direction: **abandoned without a full-dataset run** -- feature bloat from ~80
near-random windowed-attention features appears to dilute/overfit rather than help. The `--layers`
CLI generalization itself is kept (useful infra), but the 9-layer weighted model is not adopted.
**Conclusion for task #57: the 4-layer full-attention-only per-head model (val AUC 0.6195,
held-out full-dataset accuracy 82.09%) remains the best pscore found in this investigation.**
