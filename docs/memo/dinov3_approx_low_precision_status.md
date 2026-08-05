# DINOv3 approx-only low-precision inference status

**Status date:** 2026-07-30  
**Implementation branch:** `develop/dinov3-approx-fp4`  
**Implementation commits:** `a49aa7f` (FP8/auto), `2d6e47b` (NVFP4), `0048f8f`
(local COCO loader fallback)

## Bottom line

- DINOv3 `APPROX_FORWARD` supports `bf16`, `fp8`, `auto`, and forced `fp4`.
- `CORRECT_FORWARD`, `FULL_INFERENCE`, heads, public outputs, and persistent cache tensors remain
  BF16.
- Low precision helps when one Linear invocation has enough rows:
  - ImageNet-1k L2 approx-only, B32, 8,352 rows: FP8 gives a 1.97x approx speedup and FP4 gives
    2.85x, with small accuracy changes.
  - COCO L2 approx-only, B1, 405/445/489 rows per detector source: FP8 is 18.3% slower and FP4 is
    54.6% slower than BF16.
- The default `auto` threshold of 3,072 rows therefore makes the right decision for both measured
  cases: FP8 for ImageNet B32 and BF16 for COCO B1.
- Static early-exit ImageNet does accelerate its approx stage, but the final mixed
  FP8-approx/BF16-correct path has not produced an end-to-end gain yet.

## Implementation

`ExperimentConfig` accepts:

```json
{
  "precision": "bf16",
  "fp8_auto_min_rows": 3072
}
```

Valid precision values are:

- `bf16`: original behavior.
- `fp8`: force FP8 for every eligible approx source.
- `auto`: choose FP8 when `input.numel() // input.shape[-1] >= fp8_auto_min_rows`; otherwise BF16.
- `fp4`: force the prototype NVFP4 path. There is no automatic FP4 selection.

The controller:

1. Keeps the original 40 BF16 ViT blocks for correct/full inference.
2. Clones the blocks for the requested low-precision approx path.
3. Converts `attn.qkv`, `attn.proj`, and `mlp.w1/w2/w3`: 5 Linear modules per block, 200 total.
4. Compiles stateful `block.approx` with `torch.compile(fullgraph=False, dynamic=False)`.
5. Specializes lazily by shape during warm-up.
6. Converts public outputs and stored KV/residual cache tensors back to BF16.

FP8 uses TorchAO `Float8DynamicActivationFloat8WeightConfig` with E4M3FN per-tensor dynamic
activation/weight quantization. FP4 uses the TorchAO prototype `NVFP4InferenceConfig` Triton path,
with dynamic per-tensor scaling disabled; per-16-value E4M3 block scales remain.

The shared controller is wired into the classifier, detector, depther, M2F segmentor, and linear
segmentor executors. Approx events record requested/effective precision, Linear row counts, and
FP8/FP4/BF16 source counts.

Unsupported forced FP8/FP4 configurations fail at model load. `auto` warns and falls back to BF16
when FP8 is unavailable. Existing configs remain BF16 by default.

## Environment

- GPU: NVIDIA B200, compute capability 10.0
- PyTorch: `2.10.0a0+b4e4ee81d3.nv25.12`
- CUDA reported by PyTorch: 13.1
- TorchAO: `0.15.0+git01374eb5`
- Model: DINOv3 ViT-7B/16
- Runtime measurements exclude configured warm-up requests.
- Approx/correct durations are CUDA-event measurements. End-to-end latency is the runtime's
  per-batch mobile-to-result measurement.

## ImageNet-1k static early exit

Full ImageNet-1k validation set, 50,000 samples, B16, 256x256, four grid correction groups,
`token_keep_ratio=0.4`, early-exit max-probability threshold 0.6.

| Path | Top-1 | Top-5 | Approx mean | Correct mean | End-to-end mean |
|---|---:|---:|---:|---:|---:|
| all BF16 | 87.260% | 98.230% | 110.12 ms | 128.89 ms | 365.42 ms |
| FP8 approx + BF16 cache/correct | 87.230% | 98.218% | 91.94 ms | 143.15 ms | 387.98 ms |

The current mixed path accelerates approx by 1.20x, while end-to-end latency is 6.17% worse.
Accuracy changes by -0.03 percentage point. The BF16 correct stage is not quantized, but correction
following the low-precision state is slower in this run; eliminating that overhead is required
before claiming an interleaved/static end-to-end benefit.

Relevant local result directories:

- `logs/offload/imnet_static_ee_bf16_full_20260729_173915`
- `logs/offload/imnet_static_ee_fp8approx_bf16cache_full_20260729_183046`

## ImageNet-1k L2 approx-only

Full ImageNet-1k validation set, 50,000 samples, B32, 256x256 output canvas, L2
downsample-then-upsample input, no correction. L2 lowers image information but does not reduce the
model input extent or token count.

Each sample has `16 * 16 + 1 CLS + 4 storage = 261` tokens. The Linear row count is therefore
`32 * 261 = 8,352`.

| Precision | Top-1 | Delta vs BF16 | Approx mean | Approx speedup | End-to-end mean | E2E speedup |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 84.498% | - | 202.62 ms | 1.00x | 240.68 ms | 1.00x |
| FP8 | 84.530% | +0.032 pp | 102.64 ms | 1.97x | 144.54 ms | 1.67x |
| FP4 | 84.334% | -0.164 pp | 71.13 ms | 2.85x | 110.61 ms | 2.18x |

This is the successful low-precision regime: the large B32 GEMMs amortize activation quantization,
scaling, and kernel-launch overhead.

Relevant local result directories:

- `logs/offload/approx_only_l2_bf16_256_full_20260729_192142`
- `logs/offload/approx_only_l2_fp8_256_full_20260729_193004`
- `logs/offload/approx_only_l2_fp4_256_full_20260729_201225`

## COCO L2 approx-only

Full COCO val2017, 5,000 images, B1, 1024x1024 output canvas, L2
downsample-then-upsample input, no correction.

| Precision | mAP | Delta vs BF16 | Approx mean | Approx change | End-to-end mean | E2E change |
|---|---:|---:|---:|---:|---:|---:|
| BF16 | 49.395 | - | 188.80 ms | - | 235.22 ms | - |
| FP8 | 49.431 | +0.036 AP | 223.27 ms | 18.3% slower | 271.31 ms | 15.3% slower |
| FP4 | 48.861 | -0.535 AP | 291.85 ms | 54.6% slower | 339.52 ms | 44.3% slower |

All three logs contain exactly 5,000 `APPROX_FORWARD` and 5,000 `HEAD_INFERENCE` events and no
`CORRECT_FORWARD` or `FULL_INFERENCE` event. Precision metadata confirms all ten detector sources
used the forced requested precision; there was no fallback.

The detector's ViT-7B backbone uses a 3x3 window wrapper plus one resized global source. Those ten
sources are executed as separate backbone calls rather than concatenated into one Linear input.
For B1, the observed per-source rows are:

```text
405 = 20 * 20 patches + 5 special tokens
445 = 20 * 22 patches + 5 special tokens
489 = 22 * 22 patches + 5 special tokens
```

The hardware therefore sees repeated GEMMs with roughly 400-500 rows, not one GEMM with the sum of
all ten sources. Dynamic activation quantization and scaling dominate at this size. FP4 additionally
uses a prototype Triton path and per-16-value scale processing. LayerNorm, RoPE, SDPA, softmax,
cache handling, and the detector head remain BF16.

If COCO could run at B32, the corresponding per-source rows would be 12,960/14,240/15,648 and the
low-precision tradeoff could reverse. In practice, B1 already records 4.34 GB of progressive cache,
so batch scaling must be tested with memory measurements; B4/B8 are safer next points than jumping
directly to B32.

Relevant local result directories:

- `logs/offload/coco_approx_only_l2_bf16_full_20260729_203342`
- `logs/offload/coco_approx_only_l2_fp8_full_20260729_205806`
- `logs/offload/coco_approx_only_l2_fp4_full_20260729_212632`

These logs are generated artifacts and are intentionally not committed.

## Validation and contracts

`analysis/experiments/dinov3_fp8_approx_validation.py` checks:

- config values and the 3,072-row auto boundary;
- 5 converted low-precision Linear weights per test block;
- FP8/NVFP4 scaled-matmul profiler paths;
- BF16 original weights and BF16 low-precision outputs/cache;
- finite partial-correction outputs;
- exact equality with the BF16 baseline after 100% BF16 correction.

`tests/test_laplacian_base_only_decode.py` covers L2 base-only reconstruction while preserving the
configured output canvas.

## Current limitations and next work

1. `auto` should remain enabled for mixed workload deployments; forced FP8/FP4 is counterproductive
   for small-row COCO B1 sources.
2. Measure COCO approx-only at B4 and B8 with peak GPU memory before attempting B32.
3. Profile the static early-exit correction path to explain why BF16 correction becomes slower
   after FP8 approx state.
4. COCO interleaved static BF16/FP8/FP4 has not been run yet.
5. Full low-precision runtime evaluations for NYU depth and ADE20K M2F/linear segmentation remain
   outstanding.
6. FP4 remains tied to a TorchAO prototype API and should not be treated as a stable deployment
   interface.

## Reproduction templates

Use the system environment containing the TorchAO build above and include the vendored DINOv3
package root:

```bash
export PYTHONPATH="$PWD/appcorr/models:$PWD"
```

ImageNet approx-only:

```bash
CUDA_VISIBLE_DEVICES=0 offload/run_local.sh \
  offload/config/imnet/imnet_approx_only_l2.json \
  -d <IMAGENET_ROOT> -nw 2 \
  --set precision=<bf16|fp8|fp4> \
  --set exp_id=<EXP_ID>
```

COCO approx-only:

```bash
CUDA_VISIBLE_DEVICES=0 offload/run_local.sh \
  offload/config/coco/coco_approx_only_l2.json \
  -d <COCO_ROOT> -nw 2 \
  --set precision=<bf16|fp8|fp4> \
  --set scheduler_kwargs.approx_only=true \
  --set dataset_kwargs.download_if_necessary=false \
  --set exp_id=<EXP_ID>
```

