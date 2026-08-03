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
