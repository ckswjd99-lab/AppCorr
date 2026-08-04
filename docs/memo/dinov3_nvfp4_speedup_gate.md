# NVFP4 correction speedup — Phase 0 gate result: **FAIL**

**Status date:** 2026-08-04
**Branch:** `develop/dinov3-approx-fp4`
**Verdict:** do not implement the delta-decomposition acceleration *for ADE20K*. NVFP4 loses on the
shapes that workload produces (~0.07% end-to-end ceiling). A follow-up found one regime where it
does pay -- large-batch constant-M ImageNet, 1.13x on CORRECT_FORWARD at bs=128 -- see the batching
section below.

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

## Follow-up: raising M by batching — it works, but only at large batch

The gate's own diagnosis was that NVFP4 needs M ≳ 2300 and ADE20K only reaches a median of 1028.
ImageNet is the natural counter-test: `imnet_interleaved_g4.json` uses grid grouping with
`token_keep_ratio=1.0`, so **M is exactly constant** — 69 tokens/image (64 patches + 5 pretokens)
times the batch — measured at 2208 @ bs=32 and 4416 @ bs=64 across all 496 correction GEMM calls.
Constant M also means `torch.compile(dynamic=False)` sees exactly one shape, so none of the
bucketing machinery from the original plan is needed here.

Measured `CORRECT_FORWARD` (10 warmup / 30 measured at bs=64; 8 / 20 at bs=128):

| batch | M | bf16 | NVFP4 + compile | speedup | top-1 bf16 → fp4 |
|---:|---:|---:|---:|---:|---|
| 64 | 4416 | 389.04 ms | 387.18 ms | 1.00× | 91.25 → 91.30 |
| **128** | **8832** | **761.21 ms** | **675.62 ms** | **1.13×** | 90.66 → 90.82 |

(min-of-run at bs=128: 585.06 → 438.16 ms, 1.34×.)

So **batching does deliver** — but note bs=64 is already well past the microbenchmark crossover
(where the isolated GEMMs are ~1.5× faster) and still shows nothing at the stage level. Only at
bs=128 does it surface. Back-solving the observed 1.13× against the measured 1.67× GEMM speedup puts
the GEMM share of `CORRECT_FORWARD` at **~28%** — consistent with the gate's estimate and confirming
that ~70% of the stage is non-GEMM work that no amount of quantization touches.

Accuracy is unaffected (top-1 within noise, top-5 identical), consistent with the correction path
being the benign place to put FP4.

**Practical reading:** NVFP4 on the correction path is worth enabling for large-batch,
constant-M workloads (ImageNet-style classification at bs≥128), and is not worth it for
ADE20K-style sliding-window segmentation where M is small and variable. The ~70% non-GEMM remainder
of `CORRECT_FORWARD` is still the bigger prize in both cases.

### What was wired to make this measurable

- `offload/server/model/dinov3_classifier.py` — the classifier executor now configures the correct-
  precision controller and routes its correction loop through `run_dinov3_correct_block`
  (previously only the m2f segmentor was wired, so `correct_precision` was a no-op on ImageNet).
- `offload/server/model/dinov3_precision.py` — `DINOv3CorrectPrecisionController` picks
  `use_triton_kernel` from whether MSLK is importable, and gained an opt-in `compile_enabled` path
  (with `_configure_compile_environment()` first, without which the NVFP4 compile dies on a Triton
  subprocess error).
- `offload/common/protocol.py` — new `correct_compile` flag. Opt-in on purpose: it only pays where M
  is constant and large, and would thrash recompiles on variable-M workloads like ADE20K.

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

---

# Gate re-opened — 2026-08-04: NVFP4 now beats BF16

The "NVFP4 is 6–9x slower" verdict above was measured against **stock torchao defaults**. Two
fixable overheads, neither intrinsic to FP4, accounted for essentially all of it. With both removed
the gate **passes**.

## The two overheads

### 1. Dynamic per-tensor scale forces a full amax scan every call

`use_dynamic_per_tensor_scale=True` runs `torch.max(torch.abs(input_tensor))` over the whole
activation on every forward (`nvfp4_tensor.py:581`). The earlier breakdown showed this at 42% of
quantization cost at M=1280, rising to **73% at M=20480** — it is memory-bound and scales with M.

TorchAO's observer flow replaces it with a calibrated static scale:

```python
quantize_(parent, NVFP4DynamicActivationNVFP4WeightConfig(use_triton_kernel=True, step="prepare"))
#   ... run calibration batches ...
quantize_(parent, NVFP4DynamicActivationNVFP4WeightConfig(use_triton_kernel=True, step="convert"))
```

**Trap — the flow silently no-ops on a root module.** `step="prepare"` *returns* a new
`NVFP4ObservedLinear`, and `quantize_` can only install that by assigning into a parent. Call it on a
bare `nn.Linear` and the replacement is discarded, leaving a plain `Linear`; `step="convert"` then
hits `if not isinstance(module, NVFP4ObservedLinear): return module` and returns it **unquantized**.
The result is a benchmark that appears to show a large speedup but is timing plain BF16 — it matched
BF16 to 1.00x at every M, which is what gave it away. The real serving path is safe (Linears are
children of a block), but always assert `type(mod.weight).__name__ == "NVFP4Tensor"` after convert.

After conversion, `weight.act_quant_kwargs.use_dynamic_per_tensor_scale` is `False` and
`weight.act_per_tensor_scale` holds the baked scalar.

### 2. The addmm epilogue costs two full [M, N] passes

Kernel profile of one `qkv`-shaped call at M=8192 (`torch.addmm(b, xq, wq.t())`):

| kernel | time |
|---|---:|
| `cutlass3x_sm100_..._block_scaled_ue4m3xf4_ue4m3xf4_f32` (the FP4 GEMM) | **95.9 us** |
| `elementwise_kernel` (per-tensor scale multiply) | 160.6 us |
| `elementwise_kernel` (bias add) | 146.4 us |
| *(bf16 reference: `nvjet_sm100_..._bias_TNT`, bias fused into the GEMM)* | *350.5 us* |

**The FP4 GEMM is 3.65x faster than BF16.** The 307 us of epilogue is what erased it.
`nvfp4_tensor.py:516-523` is explicit about why — `_scaled_mm` does not yet accept `scale_result`,
so the per-tensor scale becomes a separate kernel, and that in turn forces bias out of the GEMM:

```python
if scale_result is not None:
    result = result * scale_result.to(a.orig_dtype)   # full [M,N] pass
if should_add_bias_separately:
    result = result + bias.to(a.orig_dtype)           # another full [M,N] pass
```

Both are memory-bound over a 134 MB bf16 output. They fuse into one `torch.addcmul(bias, r, sr)`
(= `bias + r*scale`), halving the traffic. Deviation is `2.89e-3` rel-L2, pure bf16 rounding order.

## Measured, 5 correction GEMMs (qkv/proj/w1/w2/w3, C=4096, H=8192), ms/block

| M | bf16 | fp4 stock | + static scale | + static & fused | quant | gemm+epilogue | vs bf16 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1280 | 0.325 | 0.571 | 0.440 | **0.337** | 0.179 | 0.222 | 0.96x |
| 2560 | 0.596 | 0.836 | 0.670 | **0.484** | 0.180 | 0.394 | **1.23x** |
| 5120 | 1.128 | 1.619 | 1.400 | **0.922** | 0.176 | 0.816 | **1.22x** |
| 8192 | 1.762 | 2.733 | 2.258 | **1.389** | 0.193 | 1.273 | **1.27x** |

Net: **0.55x → 0.96x at M=1280, and a genuine 1.2–1.3x win from M≈2560 up.** Crossover ≈ **2400**.

## The one remaining fixed cost

The `quant` column is **flat at ~0.18 ms across a 6.4x range of M**. Without the amax scan the MSLK
kernel is launch-bound, not memory-bound. That flat 0.18 ms is the entire reason M=1280 still loses:
it is 53% of the FP4 total there but only 14% at M=8192.

So the crossover is now purely a batching question, and ADE20K's natural M≈1259 (bs=1) sits just
below it. Crop-level batching at bs=4 puts M≈5000 — comfortably in the 1.22x region. This is the
same conclusion the original gate reached ("raise M"), but the target has moved from an unreachable
~2300-at-6x-deficit to a crossover the existing batching work already clears.

## Reproduce

Both fixes are benchmark-local so far; **neither is in the serving path yet**. `use_triton_kernel`
is still `False` in `offload/server/model/dinov3_precision.py`, and the fused epilogue requires
either a torchao patch or a local dispatch. Wiring them up is the next step, and the accuracy tables
in [dinov3_correct_low_precision_status.md](dinov3_correct_low_precision_status.md) must be re-run
afterwards — they were all measured on the eager/accurate path.
