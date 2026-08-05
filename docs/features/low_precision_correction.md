# Low-precision correction for DINOv3

Runs the correction pass's five Linears (`attn.qkv`, `attn.proj`, `mlp.w1/w2/w3`) in FP8 or NVFP4
while the approximate pass stays BF16. Correction is where the precision can be cheap: it computes a
*delta* on a subset of tokens, so its numerical demands are lower than approx's, and it is ~10% of
end-to-end request latency, so the arithmetic is worth attacking without touching accuracy.

Measured on ADE20K (`ade20k_m2f_interleaved_static*`), 200 requests each, sequential runs on one GPU,
system python / torchao 0.15. `APPROX_FORWARD` and `HEAD_INFERENCE` agree within 1% across all rows,
which is what makes the comparison valid — neither can be affected by a correction-path change.

| config | CORRECT_FORWARD | vs bf16 | mIoU | aAcc |
|---|---:|---:|---:|---:|
| bf16 | 93.87 ms | — | 74.83 | 91.91 |
| `correct_precision: fp8` | 79.89 ms | 1.175x | 74.75 | 91.96 |
| `correct_precision: fp4` (threshold selection) | 75.76 ms | 1.239x | 74.94 | 91.96 |
| **`fp4` + `token_keep_ratio: 0.55`** | **74.63 ms** | **1.258x** | **75.32** | **92.18** |

The last row is the recommended setting: fastest, and more accurate than BF16 correction.

## Turning it on

Server-side settings, top level of the config JSON:

```jsonc
{
  "correct_precision": "fp4",          // "bf16" (default) | "fp8" | "fp4"
  "correct_fp4_calib_events": 1,       // fp4/fp8: events spent calibrating before freezing scales
  "correct_fp4_proj_precision": "fp8", // fp4 only: attn.proj's precision. "fp8" (default) | "fp4"
  "correct_bucket_rows": 0,            // pad M to a multiple of this. 0 = off; see "Not recommended"
  "correct_compile": false             // torch.compile the correction block. Opt-in, rarely pays
}
```

Token selection, inside `appcorr_kwargs`:

```jsonc
{
  "token_keep_ratio": 0.55,   // top-k: every image corrects exactly this fraction of candidates
  "token_keep_thres": null,   // must be null for top-k; a float selects the threshold path instead
  "token_keep_cap": 0         // threshold path only, and not recommended -- see below
}
```

`ade20k_m2f_interleaved_static_correct_fp4_topk55.json` is the reference config for the recommended
setting; `..._correct_fp4.json` and `..._correct_fp8.json` are the threshold-path variants.

### Calibration

FP8 and FP4 both need a per-tensor activation scale, and correction inputs are selected-token
activations that do not exist at load time. So the scale is calibrated at runtime: for the first
`correct_fp4_calib_events` events each Linear runs an ordinary BF16 `F.linear` while recording amax
— **numerically exact, just not accelerated** — and then freezes. You will see, once:

```
[FP4-correct] Prepared 160 FP4 + 40 FP8 correction Linear weights across 40 blocks ...
[FP4-correct] Froze 200 activation scales; the correction path is now on torch._scaled_mm.
```

Use `-nw 5` or more so calibration finishes inside warm-up. A Linear that saw no activation stays on
the BF16 path rather than freezing a zero scale, so an unexercised layer cannot take FP4 down.

## Why top-k rather than the threshold

Both were measured at the same average kept count (`M/sample` 1279) and top-k looked far worse —
65.33 ms but mIoU 67.84. That comparison was wrong. **The threshold pads every batch to its own
maximum**, and per-sample kept spans 697–3136, so its padded width sits well above its mean: it was
quietly computing much more than `M/sample` suggested. Matched on *compute* rather than on average
count, top-k at 0.55 is both faster and more accurate.

`M/sample` is a pre-truncation counter, so it does not show either the padding or any clipping —
do not use it to compare compute between selection modes.

Top-k has a second benefit: its query-plan builder (`_build_packed_query_state_fixed_k`) is
sync-free, and that closes the launch gap entirely.

| selection | wall | kernels | GPU idle |
|---|---:|---:|---:|
| threshold `4e-5` | 77.5 ms | 59.62 ms | 17.84 ms (23.0%) |
| **top-k 0.55** | **35.4 ms** | **36.00 ms** | **−0.64 ms (0%)** |

On the threshold path the GPU idles 23% of the pass, so kernel-time savings mostly evaporate; on
top-k they convert to latency. If you optimise anything further in this path, do it on top-k or the
measurement will mislead you.

## What is under the hood

| file | what |
|---|---|
| `offload/server/model/dinov3_precision.py` | `DINOv3CorrectPrecisionController` — swaps in the low-precision blocks, drives calibration |
| `offload/server/model/fp8_fast_linear.py` | `FastFP8Linear` — direct `torch._scaled_mm`, compiled static-scale quantizer |
| `offload/server/model/fp4_fast_linear.py` | `FastFP4Linear`, `SharedFP4Activation` — NVFP4 with a fused epilogue, `attn.proj` routed to FP8 |
| `appcorr/.../triton_kernels/nvfp4_fused.py` | `quantize_nvfp4_swizzled` — the NVFP4 quantizer, called directly |
| `appcorr/.../triton_kernels/token_update.py` | row/head gather-scatter, `scale_bias_inplace_triton`, `fused_swiglu_epilogue_triton`, `fused_layerscale_add` |
| `tests/test_nvfp4_fused_quantize.py` | byte-equality gate for the NVFP4 output format |
| `analysis/experiments/dinov3_correct_profile.py` | kernel buckets, launch gap, paired A/B, CUDA-graph probe |

The recurring theme is that the vendor wrappers cost more than the kernels they wrap. TorchAO's
`Float8DynamicActivationFloat8WeightConfig` is **5.6x** the FP8 GEMM it wraps (0.907 vs 0.161 ms at
M=1280); `NVFP4Tensor.to_nvfp4` is **2x** its own Triton kernel (29.6 vs 14.3 us). Both paths here
call `torch._scaled_mm` and the quantizer directly instead.

## Environment

**TorchAO 0.15 or 0.16 only.** 0.17 removed `convert_fp32_to_fp4_packed`, so the FP4 path falls back
to `NVFP4Tensor.to_nvfp4` — correct, but ~2x slower on quantization. It says so once at startup:

```
[FP4-correct] WARNING: convert_fp32_to_fp4_packed missing (torchao 0.17.0, /path/to/python);
using the ~2x slower NVFP4Tensor.to_nvfp4 path. Latency numbers are not comparable to torchao 0.15.
```

`run_local.sh` runs bare `python`, so the stack follows PATH — a shell with conda activated will use
a different interpreter and possibly a different torchao than the system one. **Check which
interpreter a run picked up before trusting its latency.** Five consecutive 200-request runs in the
development of this branch silently took the slow path for exactly this reason.

Triton kernels need `_configure_compile_environment()` before first use or they fail with
`Cannot find ptxas`; the controllers call it. Do not strip PATH to force an interpreter — that drops
`/usr/local/cuda/bin` and the linker fails with `cannot find -lcuda`.

## Accuracy

Per-GEMM error against BF16 on real DINOv3 block-0 weights, delta-scale input:

| | rel-L2 |
|---|---:|
| FP8 | 0.0379 |
| NVFP4, per-tensor scale kept | 0.1394 |
| NVFP4, per-tensor scale off | 0.2841 |

Despite FP8's 3.7x lower kernel error, end-to-end mIoU does not favour it (74.75 vs FP4's 74.94) —
correction precision is not what limits segmentation quality here. The per-tensor scale is kept
because dropping it doubles the error for a speedup that the epilogue work reclaims anyway.

`attn.proj` runs FP8 inside the FP4 path by default: its input is the attention-core output, whose
delta compresses least of the five (amax ratio 0.93 ImageNet / 0.73 COCO). Set
`correct_fp4_proj_precision: "fp4"` to override.

## Not recommended, with reasons

Each of these was implemented and measured. They are left in the tree (all default-off) so the
measurement is not repeated.

* **`token_keep_cap`** — a fixed-width sync-free threshold builder. It truncates by *index* order
  (a stable argsort of a boolean mask), not by score, so at cap 1408 mIoU collapses to 68.75; at
  2048 it is simply slower (81.66 ms) than the dynamic width. Top-k is the same fixed width with
  better selection. Keep at 0.
* **`correct_bucket_rows`** — pads M to a bucket so shapes are finite. Costs ~19% more rows at
  M=1027/bucket 256 and buys nothing on its own; it only matters if something consumes the fixed
  shapes, and CUDA graphs turned out not to. Keep at 0.
* **`correct_compile`** — the correction block's shapes change per round, so a static graph
  recompiles and falls back past `torch._dynamo.cache_size_limit`. Off by default.
* **CUDA graphs** — capture fails on the threshold path (`cudaErrorStreamCaptureInvalidated`:
  capture forbids syncs) and succeeds on top-k, where it replays at 1.058x, i.e. nothing, because
  top-k already has no idle to reclaim.
* **NVFP4 producer fusion (single kernel)** — folding LayerNorm into the quantizer forces one
  program per 128 rows, collapsing the grid from (K/64, M/128) to (M/128,): 10 programs against
  ~148 SMs. Measured 0.06–0.27x. A two-kernel split (mean/rstd, then normalise inline) would keep
  occupancy, but its ceiling is ~8 us/call, ~1 ms over the pass.
* **Splitting the SDPA batch** to avoid padding — 0.79–0.98x. Flash attention fills SMs from the
  batch and head axes, so halving the batch loses more to occupancy than the padding wastes. SDPA
  runs at 587 TFLOPS, ~27% of peak, which is normal for a query-light shape.
* **Switching to top-k *for the syncs alone*** — the syncs cost 0.2 ms of CPU, amortised by the
  query-plan cache to a handful per round. Top-k is worth it for compute and adaptivity reasons
  above, not for that.

## Measuring changes to this path

Latency claims come from `offload/run_local.sh`, end to end, n≥200. Two traps:

* **Separate runs drift ~20%.** `APPROX_FORWARD` measured 184 ms in one pair of runs and 205 ms in
  another, with identical code. Always report a stage that *cannot* have changed alongside the one
  that did; if it moved, the comparison is void.
* **Do not run two conditions concurrently on different GPUs.** They share CPU and PCIe, and the
  FP4 path — which issues more kernels — degrades disproportionately. One such pair showed FP4 25%
  *slower*, with `HEAD_INFERENCE` off by 28 ms as the tell.

`dinov3_correct_profile.py` explains *where* time goes and must not be quoted as latency. Give it
`--correct-precision` and the real selection settings, or it profiles a path the workload never
runs. Its `launch gap` line (wall vs summed kernel time) is the fastest way to tell whether a
kernel-level win can reach latency at all.

Finally: a change that measures *exactly* 1.00x, or bit-identical output where quantization is
involved, has almost certainly not executed. That happened four times during this work — blocks
swapped but never dispatched, a torchao observer that no-ops on a root module, a cache keyed on
`id()` of a fresh view, and a scatter guarded on `is_contiguous()` that always fell back. Assert on
the post-condition (module type, tensor dtype, kernel call counts in a profile) rather than trusting
a log line.
