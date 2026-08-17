# DINOv3 correct-only low-precision inference status

**Status date:** 2026-08-03
**Implementation branch:** `develop/dinov3-approx-fp4`
**Implementation commit:** `0c4bf7c`

## Bottom line

Opposite direction of [dinov3_approx_low_precision_status.md](dinov3_approx_low_precision_status.md):
`.approx()` stays at whatever precision the model already runs (bf16), while `.correct()` — the
per-round recomputation of a small selected-token subset — runs its 5 eligible Linear layers
(qkv/proj/w1/w2/w3) in FP4 instead. Goal is reducing the correction GEMM's theoretical compute
further without touching approx's accuracy contribution. Only a theoretical-compute /
accuracy measurement right now — no latency claim (see Implementation notes for why).

**ADE20K m2f, FULL 2000-image validation set, `ade20k_m2f_interleaved_static.json` settings.**
All five arms measured on the same commit, same machine:

| Arm | mIoU | aAcc |
|---|---:|---:|
| floor: L2 approx-only, no correction | 56.013 | 84.856 |
| existing: static interleaved correction, bf16 | 61.012 | 86.930 |
| **new: static interleaved correction, correct_precision=fp4** | **61.191** | **87.034** |
| ceiling: full forward (all 40 layers), bf16 | 62.236 | 87.454 |
| reference: full forward (all 40 layers), **precision=fp4** | 62.167 | 87.404 |

floor→ceiling gap (bf16): **+6.223 mIoU / +2.599 aAcc**. Correction recovers **80.3%** of the mIoU
gap in bf16 and **83.2%** in fp4.

### Interleaved correction was discarding all but the last round (fixed 2026-08-16)

The interleaved rows above were measured with a defect, so **80.3% is not what this configuration is
worth**. `blocks_out_sum` is written only by the approx path; `correct_partial_token` read it and
never wrote back. Every round restarts at layer 0 and rebuilds any token outside the current group as
`x + blocks_out_sum`, so a corrected token's own value was thrown away at the next round — earlier
rounds survived only through the KV cache, where *other* tokens saw them when they attended.

It is worse than merely losing the correction: `prepare_tokens` re-embeds from the image as decoded
so far, so earlier groups became `refined x + degraded increment` — self-inconsistent, and on VGGT
measurably *worse* than the consistent approx floor. Full mechanism and the diagnostics that isolated
it: [[vggt_omega_status]].

The fix writes `ls1(attn_new) + mlp_out_new` back over the approximate increment. No extra compute,
no extra memory, and a no-op for one-shot correction, which reads the head out before any replay.

It is **unconditional** — there is no flag. Not persisting is the bug, not a setting, so leaving a
switch for it only invites a future measurement taken in the broken state. It shipped behind
`appcorr_kwargs.persist_correction_residual` in `96889a5`/`f365970` and the option was removed once
the effect was confirmed; `normalize_appcorr_kwargs` now *raises* on that key rather than ignoring
it, because a stale `--set ...=false` would otherwise yield an "off" arm that is really an on arm.
**The pre-fix arms below are therefore no longer reproducible**; they are kept here as the record.

Re-measured, same 2000 images, bf16 correction, pre-fix vs fixed:

| Arm | mIoU | aAcc | mIoU gap recovered |
|---|---:|---:|---:|
| floor: L2 approx-only | 56.013 | 84.856 | 0% |
| interleaved correction, **pre-fix** | 61.042 | 86.944 | 80.8% |
| **interleaved correction, fixed** | **61.597** | **87.237** | **89.7%** |
| ceiling: full forward | 62.236 | 87.454 | 100% |

**+0.555 mIoU / +0.293 aAcc, i.e. +8.9pp of the available gap, for free.** The pre-fix arm was
re-run rather than quoted, and it lands within 0.03 mIoU of the original 61.012 — which both confirms
the pair is like-for-like and confirms the published number was measured with the defect present.

The fp4 correction path keeps the gain, measured the same night:

| ADE20K arm | mIoU | aAcc | gap recovered |
|---|---:|---:|---:|
| bf16 correction, pre-fix | 61.042 | 86.944 | 80.8% |
| bf16 correction, fixed | 61.597 | 87.237 | 89.7% |
| fp4 correction, pre-fix | 61.010 | 86.971 | 80.3% |
| **fp4 correction, fixed** | **61.680** | **87.293** | **91.1%** |

### Re-measured again after the closed-loop transmission fix (2026-08-17)

The `[2, 0]` round trip was not lossless: the encoder built the Laplacian residual against the native
gaussian while the decoder predicted from the resampled base, so sending the *whole* residual still
left 1.85% relative L2 in the image the server saw. Both ADE20K correction rows above were therefore
measured on a slightly wrong input. Mechanism and the fix: [[vggt_omega_status]].

| ADE20K arm | mIoU | aAcc | gap recovered | vs ceiling |
|---|---:|---:|---:|---:|
| floor: L2 approx-only | 56.013 | 84.856 | 0% | 10.0% |
| **bf16 correction** | **61.846** | **87.309** | **93.7%** | **0.63%** |
| **fp4 correction** | **61.814** | **87.343** | **93.2%** | **0.68%** |
| ceiling: full forward | 62.236 | 87.454 | 100% | 0% |

**+4.0pp (bf16) and +2.1pp (fp4) over the open-loop numbers**, putting both within 0.7% of a full
forward. Floor and ceiling are unchanged and not re-run: `ade20k_m2f_approx_only_l2` sends levels
`[2]` with no residual and `ade20k_m2f_sequential` sends the image, so neither path can be affected --
on VGGT the corresponding rows reproduced bit-identically, which is the check that the fix is scoped
right.

fp4 and bf16 remain equivalent at matched selection (−0.032 mIoU), as they were before the fix.

**`correct_precision=fp4` is set on `ade20k_m2f_interleaved_static.json` here, not the
`..._correct_fp4_topk55.json` config.** That config selects with `token_keep_ratio: 0.55` while the
bf16 arms select with `token_keep_thres: 4e-5`, so it is not matched placement despite the "FP4
effect at matched placement" framing below. At genuinely matched selection **fp4 and bf16 are
equivalent** (−0.032 mIoU pre-fix, +0.083 fixed — both within run-to-run noise, neither direction
meaningful). The older 61.191 > 61.012 reading does not reproduce, and its provenance is unclear. The 61.191 row is left in the
table above rather than overwritten, but should not be quoted.

The fp4 run is a mix, not pure fp4: the log reports *160 FP4 + 40 FP8* correction Linears across 40
blocks, with `attn.proj` kept at fp8.

### Every family, pre-fix vs fixed

| family | metric | floor | pre-fix | fixed | ceiling | recovered, pre-fix → fixed |
|---|---|---:|---:|---:|---:|---|
| VGGT Co3D | rot_deg ↓ | 5.440 | 3.548 | **3.181** | 2.885 | 74.0% → **88.4%** |
| ADE20K m2f (bf16) | mIoU ↑ | 56.013 | 61.042 | **61.597** | 62.236 | 80.8% → **89.7%** |
| ADE20K m2f (fp4) | mIoU ↑ | 56.013 | 61.010 | **61.680** | 62.236 | 80.3% → **91.1%** |
| COCO detector | mAP ↑ | 0.5583 | 0.6011 | 0.6010 | 0.6314 | 58.6% → 58.4% |
| ImageNet cls | top1 ↑ | 84.498 | 87.896 | 87.890 | 88.108 | 94.1% → 94.0% |
| NYU depther | abs_rel ↓ | 0.05302 | 0.04940 | 0.04921 | 0.05013 | frame unusable |

NYU improves on all three of abs_rel / rmse / delta_1.25, but cannot be expressed as a recovered
fraction: both correction arms sit *past* the ceiling on abs_rel and rmse. L2 degradation costs NYU
depth almost nothing, so floor and ceiling are separated by about the metric noise.

**COCO is an unexplained null, and it is the interesting one.** Two hypotheses died on it:

- *Headroom* — "the fix recovers roughly half of whatever correction left on the table" fits VGGT
  (55% of 26.0pp) and ADE20K (46% of 19.2pp), and explains ImageNet trivially (5.9pp of headroom,
  i.e. 0.21 top1 points, is at the noise floor). COCO has **41.4pp of headroom, the most of any
  family, and gained nothing.**
- *Task density* — "per-token error reaches the metric more directly in dense tasks" does not
  separate COCO from ADE20K; both are dense.

COCO is the only family on `COCOWindowInterleaved` (9 window groups) with
`COCOWindowProgressiveLaplacian` transmission, so the round structure itself differs — but that is a
guess, not a finding. The wiring is not the explanation: `APPCORR_PERSIST_TRACE=1` shows the block
writing `blocks_out_sum` on this config (`tag=src0_layer0, rows=105`).

**What survives both nulls is a two-factor reading**, once the correction keep ratio each config
actually ran at is put next to the headroom:

| family | tokens corrected | headroom left by correction | fix gained |
|---|---:|---:|---:|
| ImageNet | 100.00% | 5.9pp | none |
| ADE20K | 41.13% | 19.2pp | +8.9pp |
| COCO | 20.60% | 41.4pp | none |

Persistence repairs staleness *among tokens that get corrected*, so it needs both a corrected mass
large enough for that staleness to matter and headroom for it to buy. ImageNet corrects everything
but has nothing left to win; COCO has the most to win but recomputes only a fifth of its tokens, so
its dominant error is the four fifths never recomputed at all, which persistence cannot touch.
ADE20K is the only one where both conditions hold.

That predicts something checkable: **raise COCO's keep ratio toward ADE20K's ~41% (its
`token_keep_thres` default is 0.002) and the fix should start showing.** Not yet run. Until it is,
this is a hypothesis fitted to three points, not a result.

Other memos whose interleaved accuracy numbers predate this fix:
[[ade20k_grid_vs_blockgrid_grouping]] (its "coherent correction timing" hypothesis is the same effect
the fix addresses, so the ranking may not survive), [[ade20k_cropcover_grouping_sweep]],
[[ade20k_sr_residual_pruning_sweep]], [[pyramid_degradation_native_vs_canvas]]. Latency and profiling
memos are unaffected — the fix reuses an already-materialised tensor.

**FP4 effect at matched placement (fp4 − bf16):**

| placement | Δ mIoU | Δ aAcc |
|---|---:|---:|
| correction only | **+0.179** | **+0.103** |
| whole forward pass | **−0.069** | **−0.051** |

**FP4 is near-lossless in *both* placements at full scale.** Note the correction arm comes out
*better* in FP4 than in bf16 — which cannot be a genuine benefit of quantization, so ~0.2pp is
simply the noise floor of this measurement. The whole-forward penalty (−0.069) sits well inside
that band. Quantizing the entire 40-layer forward pass to NVFP4 costs essentially nothing here.

> ⚠️ **This overturns the earlier N=100 conclusion.** At N=100 the whole-forward arm looked like
> −0.61pp vs correction's −0.05pp, and this doc previously concluded the near-losslessness was
> "correction-specific ... ~12x". **That did not survive the full dataset** — it was small-sample
> noise (mIoU averages over 150 classes, and 100 images leave many of them barely represented).
> A textbook instance of the repo's standing nr-is-a-sanity-check-only rule.

The full-scale result is also *milder* than the historical approx-side FP4 numbers
([dinov3_approx_low_precision_status.md](dinov3_approx_low_precision_status.md): ImageNet-1k
−0.164pp top-1, COCO −0.535 AP) — but those were measured in the faster/less-accurate FP4 mode
(`use_triton_kernel=True`, `use_dynamic_per_tensor_scale=False`) that cannot run on this install, so
the comparison is not apples-to-apples (see Environment).

**Reuse validation.** Two of these arms had prior full-2000 numbers measured on older commits, and
re-running them reproduced those numbers almost exactly — so the earlier figures were in fact
reusable, and the re-runs additionally supplied the aAcc column those docs never reported:

| arm | prior published | re-run here | Δ |
|---|---:|---:|---:|
| ceiling bf16 | 62.24 ([SR sweep memo](ade20k_sr_residual_pruning_sweep.md)) | 62.236 | −0.004 |
| correction bf16 | 61.03 ([crop_cover memo](ade20k_cropcover_grouping_sweep.md), thres 4e-5) | 61.012 | −0.018 |
| floor | 55.97 ([SR sweep memo](ade20k_sr_residual_pruning_sweep.md)) | 56.013 | +0.043 |

The floor was the one arm re-run out of genuine necessity: `a49aa7f` changed base-only decode in
`laplacian.py` (`if prev_lvl > 0 and 0 in levels:` → `if prev_lvl > 0:`) *after* 55.97 was
published, and the floor config (`pyramid_levels: [2]`, no level 0) is exactly the case that flips.
Measured impact turned out to be only +0.04 — because the executor re-resizes the decoded image to
the model canvas anyway (`_build_tta_inputs` → `_resize_short_side`), so the change amounts to one
interpolation instead of two.

Note the "full baseline in FP4" arm uses the **existing `precision` field, not a new one**:
`ADE20KSequentialPolicy` emits `APPROX_FORWARD{layers:(0,40)} + HEAD_INFERENCE` and never
`FULL_INFERENCE`, so its "full baseline" is literally an `.approx()` pass over all 40 layers of the
undegraded (Raw-transmitted) image — which `precision` already controls.

### Why quantization *must* show up, and how a wrong "no effect" reading was caught

An earlier version of this doc claimed FP4 on the full path was **bit-identical** to bf16 and
rationalized it as the integer per-pixel argmax metric being insensitive to float-level drift. That
was wrong — it was a plumbing bug (see the revert commit), not a real measurement. Two checks now
guard against repeating it:

1. **Weight-level round-trip error** (`bf16 → NVFP4 → dequantize`, real DINOv3 ViT-7B block-0
   weights, exactly the config the controller uses):

   | weight | shape | max abs err | mean abs err | rel L2 err | SQNR |
   |---|---|---:|---:|---:|---:|
   | attn.qkv | (12288, 4096) | 0.128906 | 0.00203914 | 9.35% | 20.59 dB |
   | attn.proj | (4096, 4096) | 0.091797 | 0.00213637 | 9.48% | 20.46 dB |
   | mlp.w1 | (8192, 4096) | 0.068359 | 0.00178176 | 9.38% | 20.56 dB |
   | mlp.w2 | (8192, 4096) | 0.087891 | 0.00162117 | 9.42% | 20.52 dB |
   | mlp.w3 | (4096, 8192) | 0.074219 | 0.00181475 | 9.35% | 20.58 dB |

   ~9.4% relative L2 error per tensor, across 5 Linears × 40 blocks. Any run reporting *zero*
   end-to-end change from this is not measuring what it thinks it is.

2. **Paired A/B on identical inputs.** Same config, same samples, only the precision field differs:
   correction path N=5 bf16 mIoU 47.369 vs fp4 50.187; full path N=100 bf16 52.923 vs fp4 52.311.
   Both differ, confirming the quantized modules are genuinely on the execution path. (The N=5
   correction pair's *direction* is meaningless at that sample size — it is used only as an
   executed-vs-not signal.)

## Implementation

`ExperimentConfig` accepts a new field, following the existing `precision` field's syntax:

```json
{
  "correct_precision": "bf16"
}
```

Valid values: `bf16` (original behavior), `fp8`, `fp4`. No `auto` mode (unlike `precision`) — not
requested, and `.correct()`'s row count varies every round so a fixed-threshold auto-select would
need different tuning than the approx side's.

`DINOv3CorrectPrecisionController` (new, in `dinov3_precision.py`) mirrors
`DINOv3ApproxPrecisionController`'s structure (clone the 40 BF16 blocks, convert the 5 eligible
Linears per block via TorchAO `quantize_`) but for `.correct()` instead of `.approx()`, with two
deliberate differences:

1. **No torch.compile.** `.correct()`'s selected-token count changes every round
   (`token_keep_ratio`/`token_keep_thres`-dependent), so a `dynamic=False` compile would recompile
   on every distinct shape. TorchAO's quantized `nn.Linear` (tensor-subclass dispatch) is correct
   in eager mode — compile there is a pure speed optimization on top — and this controller only
   targets accuracy right now, so eager-mode quantized `.correct()` is used directly. **This is why
   there's no latency number yet**: eager-mode dispatch overhead would make any measurement here
   meaningless as a compute-reduction estimate.
2. **`use_triton_kernel=False`, `use_dynamic_per_tensor_scale=True`** for the FP4 config (the
   approx controller's FP4 config uses the opposite: triton kernel + disabled per-tensor scale, for
   speed at the cost of accuracy). The installed TorchAO's Triton NVFP4 kernel requires the MSLK
   package (https://github.com/pytorch/MSLK), not installed here — eager dispatch avoids that
   dependency and is correctness-equivalent, and the accurate (dynamic per-tensor scale) mode is
   the right choice when latency isn't the target.

Wired into `dinov3_segmentor_m2f.py`'s real `partial_token` correction path
(`_correct_forward_partial_token_batched`) only — the only path exercised by
`ade20k_m2f_interleaved_static.json`. Not yet wired into the other 4 executors (classifier,
detector, depther, linear segmentor) or the `partial_channel` correction path — mechanical
follow-up if needed, same `ModelExecutor` hook pattern (`configure_dinov3_correct_precision`,
`begin_dinov3_correct_event`, `run_dinov3_correct_block`, `dinov3_correct_event_metadata`) as the
approx controller already uses across all 5.

**TorchAO API drift note:** this branch was originally written against TorchAO's `0.15.0+git...`
dev snapshot. The installed release (`torchao==0.17.0`) renamed
`NVFP4InferenceConfig`→`NVFP4DynamicActivationNVFP4WeightConfig` (same parameters) — fixed in both
the approx and correct controllers' FP4 init.

## Environment

- GPU: NVIDIA B200 x2, compute capability 10.0
- TorchAO: 0.17.0 (pip release, not the dev snapshot the approx-side doc used)
- Model: DINOv3 ViT-7B/16, ADE20K m2f segmentor
- New config: `ade20k_m2f_interleaved_static_correct_fp4.json`, pairs with the existing default
  `ade20k_m2f_interleaved_static.json` (only `correct_precision` differs)
- Full-baseline FP4 arm: no new config —
  `ade20k_m2f_sequential.json --set precision=fp4`

**TorchAO 0.17.0 forced an FP4 config change on the approx side too.** The original
`use_triton_kernel=True` / `use_dynamic_per_tensor_scale=False` (fastest, least accurate) cannot run
on this install: the Triton NVFP4 kernel asserts `per_tensor_scale is not None`, and enabling the
scale then requires the uninstalled MSLK package. Both the approx and correct controllers now use
`use_triton_kernel=False` / `use_dynamic_per_tensor_scale=True` (eager, accurate). **FP4 numbers
measured here are therefore not directly comparable to the historical ImageNet/COCO FP4 figures**,
and the fused-kernel latency advantage is gone.

## Next steps

- ~~Full-dataset (2000-sample) ADE20K confirmation.~~ **Done** — see the table above; it reversed
  the N=100 conclusion.
- **Actually implement NVFP4 acceleration** (accuracy is now shown to be a non-issue, so the
  remaining work is purely making it fast). Design notes agreed for that effort:
  - **Bucketize the correction query count and pad**, then discard the pad rows. No attention
    masking needed: in `.correct()` only the *query* count varies (K/V is always the full fixed
    cache) and queries attend independently, so padded queries just produce rows that get sliced
    away. The repo already does this for SDPA (`sdpa_query_bucket_size`,
    `appcorr/models/dinov3/layers/attention.py`); extending it to the FP4 **Linear** layers is the
    new work.
  - **Pad with zeros, not `torch.empty`.** The existing SDPA bucketing uses uninitialized memory
    (fine there, pads are discarded), but NVFP4 with `use_dynamic_per_tensor_scale=True` derives
    the per-tensor scale from the activation **amax** — garbage pads would inflate the scale and
    degrade the *real* rows' quantization. Zeros leave amax untouched.
  - **Guard the K/V scatter-back** so padded rows never enter the shared cache.
  - **Prefer bucket size 128 over 64**: TorchAO's Triton NVFP4 kernel requires
    `M % 128 == 0 and K % 64 == 0` and silently falls back otherwise, so bucketing is a
    *requirement* for the fused kernel, not just compile hygiene. Waste is ≤8% at ~1500 corrected
    tokens. (That kernel also needs the uninstalled MSLK package — see Environment.)
  - **Raise `torch._dynamo.cache_size_limit`** (default **8**, verified on torch 2.12.1;
    `accumulated_cache_size_limit` default 256). More than 8 bucket variants per code object makes
    Dynamo silently fall back to eager with only a warning — the easiest gotcha to miss here.
  - Remaining per-variant costs: one Inductor/Triton compile per bucket (40 blocks × B buckets,
    amortized by the `TORCHINDUCTOR_CACHE_DIR` the controller already sets up and the existing
    correct-bucket warmup), extra kernel memory, and guard-evaluation cost growing with variant
    count. Alternative is `dynamic=True` (single graph, no recompiles) at the cost of shape
    specialization — and it cannot satisfy FP4's static `M % 128` constraint.
- **Caveat to clean up before quoting any FP4-vs-bf16 delta as pure quantization effect:** the
  approx controller compiles the FP4 path but calls bf16 uncompiled (`_compiled_fp4_approx` vs
  `self.blocks[layer_idx].approx(...)` in `dinov3_precision.py`), so the whole-forward comparison
  conflates quantization with compilation. Isolating the former needs a compiled-bf16 arm.
- Extend to the other 4 executors / the `partial_channel` correction path if useful elsewhere.
