# NVFP4 correction speedup — Phase 0 gate result: **FAIL**

**Status date:** 2026-08-04
**Branch:** `develop/dinov3-approx-fp4`
**Verdict:** do not implement the delta-decomposition acceleration. NVFP4 loses on the shapes this
workload actually produces, and the best achievable win is ~0.07% end-to-end.

## Why the gate existed

The accuracy side is settled and favourable — an exact `(a, d)` decomposition puts NVFP4 on the
correction delta and is 1.7×/4.1× more faithful than quantizing the whole forward
([dinov3_exact_decomposition_fp4_features.md](dinov3_exact_decomposition_fp4_features.md)). The open
question was whether it can be made **fast**, so implementation was gated on a benchmark rather than
assumed.

## What was measured

**MSLK is installable and helps a lot.** `pip install mslk==1.2.0 --index-url
https://download.pytorch.org/whl/cu130` works on this env (torch 2.12.1+cu130, CC 10.0);
`torchao.prototype.mx_formats.kernels._mslk_available` flips to `True` and `use_triton_kernel=True`
stops raising. It cuts the eager NVFP4 Linear from ~2.20 ms to ~0.59 ms per block (3.7×).

**torch.compile matters more than MSLK**, and needs the pipeline's own Triton setup
(`_configure_compile_environment()` in `offload/server/model/dinov3_precision.py`) — without it,
compilation of the NVFP4 path dies with a Triton subprocess error on the amax reduction.

Per-block totals (5 GEMMs, qkv/proj/w1/w2/w3, C=4096, H=8192), compiled + MSLK:

| M | bf16 | NVFP4 compiled | speedup |
|---:|---:|---:|---:|
| 1280 | 0.325 ms | 0.545 ms | 0.60× |
| 1536 | 0.362 | 0.543 | 0.67× |
| 2048 | 0.472 | 0.552 | 0.85× |
| **2560** | 0.597 | 0.548 | **1.09×** |
| 3072 | 0.691 | 0.552 | 1.25× |
| 4096 | 0.908 | 0.603 | 1.51× |
| 5120 | 1.125 | 0.706 | 1.59× |
| 8192 | 1.745 | 1.046 | 1.67× |

**Crossover ≈ M 2300.** NVFP4 carries a ~0.54 ms/block fixed cost (activation quantization + scale
swizzle + dispatch) that is flat in M, so it only pays once the GEMM work exceeds it.

Isolating the pieces at qkv (4096→12288) confirms the shape of the problem — the FP4 GEMM *is*
genuinely faster, it just cannot outrun its own setup cost at small M:

| M | bf16 GEMM | activation quant | FP4 GEMM alone | FP4 GEMM vs bf16 |
|---:|---:|---:|---:|---:|
| 1280 | 0.102 ms | 0.056 | 0.064 | **1.59×** |
| 5120 | 0.325 | 0.058 | 0.250 | **1.30×** |
| 20480 | 1.375 | 0.236 | 1.055 | **1.30×** |

## The decisive number: what M does this workload actually produce?

Instrumented `correct_partial_token` and ran `ade20k_m2f_interleaved_static.json` for 20 images
(6260 correction GEMM calls):

| statistic | M |
|---|---:|
| min | 5 |
| p25 | 448 |
| **median** | **1028** |
| mean | 1207 |
| p75 | 1856 |
| p90 | 2554 |
| max | 3141 |

Only **15.7%** of calls land at M ≥ 2300 (35.3% of the row-work). Applying the measured
speedup curve over this real distribution:

- **Enabling NVFP4 everywhere: 0.53× — i.e. 90% *slower*.** The long tail of small-M calls each pays
  the flat ~0.54 ms while bf16 costs a fraction of that.
- **Hybrid dispatch (NVFP4 only above the crossover): 1.04×** — saves **3.9%** of correction GEMM
  time, the ceiling for any amount of engineering here.

## Why that ceiling is not worth building for

Correction GEMMs are ~15% of `CORRECT_FORWARD` (167 ms/request measured on ADE20K full-2000), which
is itself ~12% of summed pipeline event time. So the best case is
`3.9% × 15% × 12% ≈ **0.07% end-to-end**` — against a Phase 1–3 build that requires ~10 GB/source of
new a-path cache, a delta rewrite of `correct_partial_token` + `SelfAttention.correct`, bucketing,
and a resolution of the `blocks_out_sum` semantics problem.

**Phases 1–3 of the plan are therefore not started.**

## What is worth doing instead

1. **Find where the other ~85% of `CORRECT_FORWARD` goes.** 167 ms/request against ~12–30 ms of
   GEMM means the bottleneck is elsewhere — cache management, the scatter/gather and
   `blocks_out_sum` reconstruction, `_build_packed_query_state`'s host sync (`.item()`), or per-layer
   dispatch across 40 blocks × sources. That is where a real speedup lives.
2. **Raise M if NVFP4 is ever revisited.** The crossover is a batching problem, not a kernel problem:
   batch more sliding-window crops into one correction GEMM and the picture changes. Worth
   re-checking on workloads whose natural M is larger (ImageNet B32 already sits at 8352 rows, where
   the historical approx-side FP4 numbers *did* win).
3. **Keep the accuracy result.** The exact `(a, d)` decomposition remains the right way to place FP4
   *if* FP4 is used at all; this gate only says the speed does not currently justify shipping it.

## Env note

MSLK 1.2.0 is now installed in the `appcorr` conda env. Two consequences:

- `use_triton_kernel=True` is now viable again. Both precision controllers in
  `offload/server/model/dinov3_precision.py` were switched to `use_triton_kernel=False` back when
  MSLK was missing; that choice can be revisited, but note the accuracy numbers in
  [dinov3_correct_low_precision_status.md](dinov3_correct_low_precision_status.md) and
  [dinov3_exact_decomposition_fp4_features.md](dinov3_exact_decomposition_fp4_features.md) were all
  measured with the eager/accurate path, so switching would invalidate the comparison unless re-run.
- Anything that compiles the NVFP4 path must call `_configure_compile_environment()` first.
