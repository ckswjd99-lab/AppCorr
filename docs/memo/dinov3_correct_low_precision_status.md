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

**ADE20K m2f, N=100 (first 100 samples, not full-dataset), `ade20k_m2f_interleaved_static.json`
settings:**

| Arm | mIoU | aAcc |
|---|---:|---:|
| floor: L2 approx-only, no correction | 46.72 | 81.36 |
| existing: static interleaved correction, bf16 | 52.08 | 83.51 |
| **new: static interleaved correction, correct_precision=fp4** | **52.03** | **83.53** |
| ceiling: full baseline (stock, no approximation) | 52.92 | 84.08 |

FP4 correction is statistically indistinguishable from bf16 correction on this sample (mIoU
**−0.05pp**, aAcc **+0.02pp** vs bf16 correction) — both recover ~86% of the floor→ceiling mIoU gap
(bf16: 86.4%, fp4: 85.7%) and ~79-80% of the aAcc gap. **No measurable accuracy cost from
quantizing the correction path to FP4 at N=100.** Full-dataset confirmation pending (per the
project's nr-sanity-first discipline — this is a first-pass check on 100/2000 samples).

## Does this generalize beyond correction, or is it correction-specific?

Ran the same question the other direction: quantize the **stock full-inference path** (`.forward()`,
no approximation/correction at all — `ade20k_m2f_sequential.json`, the ceiling arm above) to FP4
instead, via a new sibling controller (`DINOv3FullPrecisionController`, same commit range as this
doc), and compare against the same stock bf16 baseline, same N=100 samples:

| Arm | mIoU | aAcc |
|---|---:|---:|
| ceiling: full baseline, bf16 | 52.92288696479569 | 84.08007692187066 |
| **ceiling: full baseline, full_precision=fp4** | **52.92288696479569** | **84.08007692187066** |

**Bit-identical to 16 significant digits.** Verified this isn't a broken no-op: a direct block-level
comparison (block 0, real quantized weights vs. a separately-loaded bf16 reference, same random
input, same autocast context) shows the FP4 clone's weight tensor really is `NVFP4Tensor` and its
output really differs numerically from bf16 (max abs diff 174.85, mean abs diff 0.012 on a single
block's raw activations) — so quantization is genuinely in effect on the compute path, not silently
bypassed. The bit-identical final metrics are consistent with a real (if perhaps fortunate)
outcome: because the reported mIoU/aAcc are deterministic functions of *integer* per-pixel argmax
class labels (via `torch.bincount`), a measurable float-level activation difference at every layer
still produces identical final numbers as long as **no single pixel's argmax class flips**, across
the full 40-layer/(sliding-window crops) stack and all 100 images. Apparently none did, at this
scale.

**Answer to the original question, with the caveat that N=100 is small and the result could easily
be "no flips yet" rather than "will never flip": FP4 does NOT appear to be correction-specific here
— applying it broadly across the entire stock forward pass was, in this sample, at least as
lossless as confining it to the correction recompute (0.00pp vs correction's −0.05pp, both within
what's plausibly noise at N=100).** This is a more encouraging result than a naive "correction-only
recovery" story would predict, but full-dataset confirmation is needed before trusting it —
100 samples is not enough to rule out a low but nonzero flip rate that a larger sample would surface
(mean abs diff 0.012 at block 0, compounding over 40 layers, is not obviously negligible; it simply
didn't cross any argmax decision boundary in this particular sample of 100 images).

*(Earlier scratch note, now corrected: an initial single-sample smoke test wrongly compared the
full-fp4 config against the *correction*-path bf16 config — two different pipelines entirely
(different scheduler/transmission policy) — and reported a spurious −0.47pp "gap." The properly
paired comparison above, same `ade20k_m2f_sequential*.json` config on both sides, shows 0.00pp.)*

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

## Next steps

- Full-dataset (2000-sample) ADE20K confirmation.
- Real latency measurement, which requires solving the eager-mode-dispatch-overhead problem above
  (either a dynamic-shape-tolerant compile strategy, or bucketing `.correct()`'s token count the
  way `sdpa_query_bucket_size` already buckets attention).
- Extend to the other 4 executors / the `partial_channel` correction path if useful elsewhere.
