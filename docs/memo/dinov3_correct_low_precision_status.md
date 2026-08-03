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
| ceiling: full baseline (all 40 layers), bf16 | 52.92 | 84.08 |
| reference: full baseline (all 40 layers), **precision=fp4** | 52.31 | 83.73 |

FP4 correction is statistically indistinguishable from bf16 correction on this sample (mIoU
**−0.05pp**, aAcc **+0.02pp** vs bf16 correction) — both recover ~86% of the floor→ceiling mIoU gap
(bf16: 86.4%, fp4: 85.7%) and ~79-80% of the aAcc gap. **No measurable accuracy cost from
quantizing the correction path to FP4 at N=100.** Full-dataset confirmation pending (per the
project's nr-sanity-first discipline — this is a first-pass check on 100/2000 samples).

**Is FP4 broadly harmless, or specifically harmless on the correction path?** The last row answers
this: applying FP4 to the *whole* forward pass (all 40 layers, no approximation involved) costs
**−0.61pp mIoU / −0.35pp aAcc**, ~12x the −0.05pp that confining FP4 to the correction recompute
costs. So the near-losslessness is **correction-specific, not a general property of FP4 on this
model** — consistent with the intuition that correction only recomputes a selected token subset
while the bulk of the representation still comes from the untouched bf16 approx pass. It is also
consistent in magnitude and sign with the historical approx-side FP4 numbers
([dinov3_approx_low_precision_status.md](dinov3_approx_low_precision_status.md): ImageNet-1k
−0.164pp top-1, COCO −0.535 AP).

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

- Full-dataset (2000-sample) ADE20K confirmation.
- Real latency measurement, which requires solving the eager-mode-dispatch-overhead problem above
  (either a dynamic-shape-tolerant compile strategy, or bucketing `.correct()`'s token count the
  way `sdpa_query_bucket_size` already buckets attention).
- Extend to the other 4 executors / the `partial_channel` correction path if useful elsewhere.
