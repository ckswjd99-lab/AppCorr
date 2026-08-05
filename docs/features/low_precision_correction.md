# Low-precision correction for DINOv3

Runs the correction pass's five Linears (`attn.qkv`, `attn.proj`, `mlp.w1/w2/w3`) in FP8 or NVFP4
while the approximate pass stays BF16. Correction computes a *delta* over a subset of tokens, so it
tolerates lower precision than approx does.

Measured on ADE20K (`ade20k_m2f_interleaved_static*`), 200 requests each, sequential runs on a
single **NVIDIA B200** (compute capability 10.0), torchao 0.15:

| config | CORRECT_FORWARD | vs bf16 | mIoU | aAcc |
|---|---:|---:|---:|---:|
| bf16 | 93.87 ms | — | 74.83 | 91.91 |
| `correct_precision: fp8` | 79.89 ms | 1.175x | 74.75 | 91.96 |
| `correct_precision: fp4`, threshold selection | 75.76 ms | 1.239x | 74.94 | 91.96 |
| **`fp4` + `token_keep_ratio: 0.55`** | **74.63 ms** | **1.258x** | **75.32** | **92.18** |

The last row is the recommended setting. Correction is ~10% of end-to-end request latency, so this
is ~2.5% off the request.

## Configuration

Top level of the config JSON:

```jsonc
{
  "correct_precision": "fp4",          // "bf16" (default) | "fp8" | "fp4"
  "correct_fp4_calib_events": 1,       // events spent calibrating before scales freeze
  "correct_fp4_proj_precision": "fp8", // fp4 only: attn.proj's precision. "fp8" (default) | "fp4"
  "correct_bucket_rows": 0,            // pad M to a multiple of this. 0 = off (recommended)
  "correct_compile": false             // torch.compile the correction block. Off (recommended)
}
```

Token selection, inside `appcorr_kwargs`:

```jsonc
{
  "token_keep_ratio": 0.55,   // top-k: correct exactly this fraction of candidates per image
  "token_keep_thres": null,   // must be null for top-k; a float selects the threshold path
  "token_keep_cap": 0         // threshold path only. 0 = off (recommended)
}
```

Reference configs, all under `offload/config/ade20k/`:

| file | setting |
|---|---|
| `..._correct_fp4_topk55.json` | **recommended** — fp4 + top-k 0.55 |
| `..._correct_fp4.json` | fp4 + threshold `4e-5` |
| `..._correct_fp8.json` | fp8 + threshold `4e-5` |

### Calibration

FP8 and NVFP4 each need a per-tensor activation scale. Correction inputs are selected-token
activations that do not exist at load time, so the scale is calibrated at runtime: for the first
`correct_fp4_calib_events` events every Linear runs an ordinary BF16 `F.linear` while recording
amax — numerically exact, just not accelerated — then freezes. Two lines appear once:

```
[FP4-correct] Prepared 160 FP4 + 40 FP8 correction Linear weights across 40 blocks ...
[FP4-correct] Froze 200 activation scales; the correction path is now on torch._scaled_mm.
```

Run with `-nw 5` or more so calibration completes inside warm-up. A Linear that never saw an
activation stays on the BF16 path rather than freezing a zero scale, so an unexercised layer cannot
take the model down.

### Choosing a selection mode

**Top-k** (`token_keep_ratio` set, `token_keep_thres: null`) gives every image the same number of
corrected tokens. **Threshold** (`token_keep_thres` set) varies per image with residual energy.

Top-k is recommended at ratio 0.55. Two properties matter:

* Threshold pads each batch to that batch's own maximum kept count. Per-sample kept spans 697–3136
  on ADE20K, so its padded width sits well above its mean and it computes considerably more than
  `M/sample` suggests. Top-k's width equals its mean, so nothing is padded.
* Top-k's query-plan builder is sync-free, which keeps the GPU saturated:

  | selection | wall | kernels | GPU idle |
  |---|---:|---:|---:|
  | threshold `4e-5` | 77.5 ms | 59.62 ms | 17.84 ms (23.0%) |
  | top-k 0.55 | 35.4 ms | 36.00 ms | 0% |

  On the threshold path a kernel-level speedup mostly turns into idle rather than latency. Tune this
  path on top-k.

Ratio 0.55 is the knee: 0.70 costs 8.8 ms for 0.15 less mIoU. Note that `M/sample` in the event log
is a pre-truncation counter — it reflects neither padding nor clipping, so it cannot be used to
compare compute across selection modes.

## Accuracy

Per-GEMM error against BF16, real DINOv3 block-0 weights, delta-scale input:

| | rel-L2 |
|---|---:|
| FP8 | 0.0379 |
| NVFP4, per-tensor scale kept | 0.1394 |
| NVFP4, per-tensor scale off | 0.2841 |

End-to-end mIoU does not track this — FP4 (0.1394) scores at or above FP8 (0.0379) and BF16.
Correction precision is not what limits segmentation quality at these settings. The per-tensor scale
is kept: dropping it doubles the error, and the speedup it would buy is spent on the epilogue that
becomes necessary without it.

`attn.proj` runs FP8 inside the FP4 path by default — its input is the attention-core output, whose
delta compresses least of the five (amax ratio 0.93 ImageNet / 0.73 COCO). Override with
`correct_fp4_proj_precision: "fp4"`.

## Latency breakdown

All figures below are one **NVIDIA B200**, the recommended setting (fp4 + top-k 0.55).

### Per request, across stages

Correction is ~10% of the request; the two BF16 stages around it are each ~2.5x larger.

| stage | ms/req | share |
|---|---:|---:|
| APPROX_FORWARD | 184.60 | 25.2% |
| HEAD_INFERENCE | 171.77 | 23.5% |
| MOBILE_ENCODE_G1 | 107.00 | 14.6% |
| **CORRECT_FORWARD** | **74.63** | **10.2%** |
| Decode | 67.22 | 9.2% |
| MOBILE_ENCODE_G2 | 47.40 | 6.5% |
| PREPARE_TOKENS | 34.74 | 4.7% |
| EXIT_ALL | 17.83 | 2.4% |

End-to-end latency 610.6 ms/req. Both APPROX and CORRECT run 1.9 times per request.

### Inside CORRECT_FORWARD

The 40-block loop is 97–98% of the stage; everything else (the K/V scatter-back into the cache,
batch assembly and splitting) is the remaining 1.23 ms and does not vary with precision.

| | ms | share |
|---|---:|---:|
| block loop | 39.33 | 97% |
| cache scatter | 1.23 | 3% |

### Inside the 40-block loop

Kernel time from `dinov3_correct_profile.py` at the real correction shape (B=2, N=3141), summing to
36.00 ms against a 35.4 ms wall — i.e. the GPU is saturated, so kernel savings here convert to
latency.

| bucket | ms | share |
|---|---:|---:|
| GEMM (`_scaled_mm`) | 9.68 | 26.9% |
| attention / SDPA | 6.63 | 18.4% |
| other (RoPE, epilogues) | 6.27 | 17.4% |
| elementwise / residual / norm | 4.61 | 12.8% |
| quantization (fp4/fp8) | 4.40 | 12.2% |
| gather / scatter / index | 4.40 | 12.2% |

Largest individual kernels:

| kernel | ms | calls | per block |
|---|---:|---:|---:|
| cuDNN flash SDPA | 6.63 | 40 | 1 |
| NVFP4 GEMM (`nvjet ... Avec16UE4M3`) | 3.64 | 80 | 2 |
| `_quantize_nvfp4_kernel` | 3.25 | 120 | 3 |
| NVFP4 GEMM (cutlass block-scaled) | 2.56 | 40 | 1 |
| `_rope_active_inplace_kernel` | 2.39 | 40 | 1 |
| `_active_token_update_kernel` | 2.37 | 40 | 1 |
| LayerNorm | 1.87 | 80 | 2 |
| FP8 GEMM (`attn.proj`) | 1.47 | 40 | 1 |
| `_fused_swiglu_epilogue_kernel` | 1.46 | 40 | 1 |

No bucket dominates: the largest is 27%, so further work here has a low ceiling. Making the GEMMs
free would leave 26.32 ms, a 1.37x bound on the loop and ~4% of the request.

## Attention precision

**SDPA runs in BF16** and there is no switch for it — `correct_precision` covers the five Linears
only. The Q/K/V it consumes are BF16: the K/V cache is stored BF16 and Q comes back from the NVFP4
`attn.qkv` GEMM already dequantized to BF16.

PyTorch's `scaled_dot_product_attention` does not accept FP8 tensors:

```
torch.float8_e4m3fn -> NotImplementedError: "mul_cuda" not implemented for 'Float8_e4m3fn'
```

FP8 attention on Blackwell exists at the cuDNN level but is not reachable through the PyTorch op, so
it would mean calling cuDNN directly or pulling in TransformerEngine. The payoff is bounded: SDPA is
18.4% of the loop and already runs at 587 TFLOPS, ~27% of B200's BF16 peak, which is normal for a
query-light shape (1464 queries against 3141 keys). Even a full 2x would be ~9% of the loop and ~1%
of the request, against an accuracy cost on the one part of the block that is a genuine
non-linearity.

## Implementation

| file | contents |
|---|---|
| `offload/server/model/dinov3_precision.py` | `DINOv3CorrectPrecisionController` — swaps in low-precision blocks, drives calibration |
| `offload/server/model/fp8_fast_linear.py` | `FastFP8Linear` — direct `torch._scaled_mm` with a compiled static-scale quantizer |
| `offload/server/model/fp4_fast_linear.py` | `FastFP4Linear`, `SharedFP4Activation` — NVFP4 with a fused epilogue; `attn.proj` routed to FP8 |
| `appcorr/.../triton_kernels/nvfp4_fused.py` | `quantize_nvfp4_swizzled` — NVFP4 quantizer, called directly |
| `appcorr/.../triton_kernels/token_update.py` | row/head gather-scatter, `scale_bias_inplace_triton`, `fused_swiglu_epilogue_triton`, `fused_layerscale_add` |
| `tests/test_nvfp4_fused_quantize.py` | byte-equality gate for the NVFP4 output format |
| `analysis/experiments/dinov3_correct_profile.py` | kernel buckets, launch gap, paired A/B, CUDA-graph probe |

Both paths call `torch._scaled_mm` and the quantizer directly rather than through TorchAO's
configs. The wrappers dominate the kernels they wrap: `Float8DynamicActivationFloat8WeightConfig`
costs 0.907 ms against the FP8 GEMM's 0.161 ms at M=1280, and `NVFP4Tensor.to_nvfp4` costs 29.6 us
against its own Triton kernel's 14.3 us.

NVFP4's output format is unforgiving — a wrong scale swizzle raises nothing and returns wrong
numbers — so `tests/test_nvfp4_fused_quantize.py` compares `qdata` and `scale` for exact byte
equality with TorchAO across the shapes the correction path uses. Run it after touching the
quantizer.

## Environment

**TorchAO 0.15 or 0.16.** 0.17 removed `convert_fp32_to_fp4_packed`, so the FP4 path falls back to
`NVFP4Tensor.to_nvfp4` — correct, but ~2x slower on quantization. It says so once at startup:

```
[FP4-correct] WARNING: convert_fp32_to_fp4_packed missing (torchao 0.17.0, /path/to/python);
using the ~2x slower NVFP4Tensor.to_nvfp4 path. Latency numbers are not comparable to torchao 0.15.
```

`run_local.sh` invokes bare `python`, so the interpreter — and therefore the torchao version —
follows PATH. Confirm which one a run picked up before comparing its latency against another run.

NVFP4 requires compute capability 10.0+; FP8 requires 8.9+. Below that the controller raises for an
explicit `fp4`/`fp8` setting. Triton kernels need `_configure_compile_environment()` before first
use or they fail with `Cannot find ptxas`; the controllers call it. Do not strip PATH to force an
interpreter — that drops `/usr/local/cuda/bin` and the linker fails with `cannot find -lcuda`.

## Settings to leave off

Present and functional, but measured to cost more than they return. Defaults are already off.

| setting | why |
|---|---|
| `token_keep_cap` | Fixed-width sync-free threshold builder. Truncates by index order, not by score: cap 1408 drops mIoU to 68.75; cap 2048 is slower than the dynamic width (81.66 ms). Top-k is the same fixed width with score-based selection. |
| `correct_bucket_rows` | Pads M to a bucket so shapes are finite. ~19% more rows at M=1027/bucket 256. Only pays if something consumes the fixed shapes; nothing currently does. |
| `correct_compile` | The correction block's shapes change per round, so a static graph recompiles and falls back past `torch._dynamo.cache_size_limit`. |

CUDA graph capture is also not useful here: it fails on the threshold path (capture forbids the host
syncs its builder performs) and on top-k it replays at 1.058x, because top-k already leaves no GPU
idle to reclaim. `dinov3_correct_profile.py --cuda-graph` re-tests this if the path changes.

## Benchmarking changes to this path

Latency claims should come from `offload/run_local.sh` end to end, n≥200.

* **Run conditions sequentially, not concurrently on separate GPUs.** They share CPU and PCIe, and
  the FP4 path issues more kernels, so it degrades disproportionately under contention.
* **Report a stage that cannot have changed** — `APPROX_FORWARD` or `HEAD_INFERENCE` — alongside
  `CORRECT_FORWARD`. Run-to-run drift reaches ~20%; if the unaffected stage moved, the comparison is
  void.
* **n=20 is not enough.** Per-request SD on `CORRECT_FORWARD` is ~50 ms.

`dinov3_correct_profile.py` explains where time goes and is not a latency measurement. Pass it
`--correct-precision` and the real selection settings or it profiles a path the workload never runs.
Its `launch gap` line — wall against summed kernel time — shows whether a kernel-level win can reach
latency at all.
