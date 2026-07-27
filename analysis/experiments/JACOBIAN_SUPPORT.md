# Draft-Guided Support Experiment

This branch tests whether a low-resolution DINOv3 pass can identify a
GPU-friendly correction support. It deliberately keeps the existing
`partial_token` and `partial_channel` meanings unchanged.

## Frozen baseline

- Base commit: `6726d26694c1a13e4f659c4a94e552496ff23e6a`
  (`origin/main`, `Crop-aware grouping for ADE20k`)
- Development branch: `experiment/jacobian-support-oracle`
- Primary model: DINOv3 ViT-7B/16, BF16, 40 layers, hidden size 4096,
  32 heads, head size 128, SwiGLU hidden size 8192
- Primary device: NVIDIA B200 (SM100)
- Measured software: Python 3.12.3, PyTorch
  `2.10.0a0+b4e4ee81d3.nv25.12`, CUDA 13.1, Triton 3.5.1
- Historical failure reference: `develop/dinov3-csr`. Its unstructured
  PyTorch CSR correction took 42--108 seconds and used 16--22 GB of cache.
  No CSR implementation was copied into this branch.

The oracle output records the exact commit, dirty state, device, dtype, and
low-resolution implementation. OpenCV `pyrDown`/`pyrUp` is the canonical
low-resolution path. A PyTorch bicubic fallback exists only so import/smoke
failures remain diagnosable.

## Numerical contracts

`appcorr/models/dinov3/layers/jacobian_support.py` distinguishes:

- strict split JVP: `S dV + dS V`;
- linearized product delta:
  `(S + dS)(V + dV) - SV`;
- exact-probability product delta, where corrected probabilities use the
  corrected full-softmax denominator.

Sparse support never runs a new softmax over selected keys. The selected
probabilities are entries of the original full-softmax distribution. Full
`dS[B,H,N,N]` is not a persistent cache.

The product path has an explicit cached-product tier. The draft caches the
selected `SV`; correction performs one tensor-core corrected product and
subtracts that cache. This is a finite-difference product shortcut, not a
strict Jacobian evaluation, because it retains `dS dV`.

## Reproduction

Reference and kernel contracts:

```bash
PYTHONPATH="$PWD" python analysis/experiments/test_jacobian_support_math.py
PYTHONPATH="$PWD" python analysis/experiments/test_jacobian_attention_kernel.py
PYTHONPATH="$PWD" python analysis/experiments/test_jacobian_support_config.py
```

One real-image, four-step support audit:

```bash
PYTHONPATH="$PWD" python analysis/experiments/jacobian_support_oracle.py \
  --image /path/to/image.JPEG --num-groups 4 \
  --layers 0,10,20,30,39 --support 0.25,0.5 \
  --tail-epsilon 0.05,0.1 --output /tmp/oracle.json
```

Dense final-logit gate:

```bash
PYTHONPATH="$PWD" python analysis/experiments/jacobian_support_oracle.py \
  --image /path/to/image.JPEG --num-groups 1 --layers 0 \
  --support 0.5 --tail-epsilon 0.1 --dense-gate \
  --output /tmp/dense_gate.json
```

B200 workload-shape benchmark:

```bash
PYTHONPATH="$PWD" python analysis/experiments/benchmark_jacobian_attention.py \
  --presets imagenet,coco,ade20k,nyuv2 --queries 64 \
  --support 0.25,0.5,0.75 --warmup 20 --iterations 200 \
  --output /tmp/jacobian_b200.json
```

The benchmark reports p50, p95, and a bootstrap 95% confidence interval for
the median. `triton_product_pipeline_indirect` includes selected `dlogit`,
softmax JVP, and cached-product consumption. Its key descriptor is produced
by the draft and it performs no correction-time value gather.

## Preliminary B200 result

At 50% key support, the complete linearized product pipeline achieved:

| Workload shape | Dense pipeline | Sparse pipeline | Kernel speedup |
|---|---:|---:|---:|
| ImageNet, B=32, N=261 | 0.439 ms | 0.207 ms | 2.12x |
| COCO, B=1, N=4101 | 0.330 ms | 0.174 ms | 1.89x |
| ADE20K, B=1, N=3141 | 0.264 ms | 0.144 ms | 1.83x |
| NYUv2, B=8, N=2309 | 0.809 ms | 0.333 ms | 2.43x |

These numbers establish a compute crossover, not an end-to-end claim. They
exclude QKV projection, FFN correction, scheduler work, and transmission.
They must not be extrapolated to RTX 4090 or Jetson.

An earlier elementwise Triton prototype was 10x or more slower than cuBLAS.
The checked-in kernel uses `tl.dot` tensor-core tiles. Explicit
`index_select` packing still dominates at moderate support, so the retained
path consumes cached block descriptors and reads full V/dV indirectly.

## Accuracy gate and decision

The prescribed dense gate failed for strict Jacobian correction. On three
real ImageNet validation images, correcting a 1/4-resolution canvas to full
resolution through all 40 layers produced:

| Dense backend | Mean logit relative L2 | Mean cosine | Stock top-1 match |
|---|---:|---:|---:|
| Base only | 0.5267 | 0.8560 | 3/3 |
| Split JVP attention + JVP FFN | 1.7822 | -0.3814 | 0/3 |
| Linearized product attention + JVP FFN | 2.2128 | -0.3324 | 0/3 |
| Exact-probability product + JVP FFN | 1.6241 | -0.2657 | 0/3 |
| Exact-probability product + exact finite-difference FFN | 0.0066 | 1.0000 | 3/3 |

Four cumulative 25%-patch correction steps did not rescue the local Taylor
approximation: across layers 0/10/20/30/39 and four groups, median attention
relative L2 was 1.066 for split and 0.990 for linearized product; median FFN
JVP relative L2 was 0.954.

This activates the plan's stop condition: the branch does **not** connect
strict JVP correction to the stateful runtime or claim end-to-end speedup.
The support signal itself remains useful: on the sampled real image, 50%
base-attention block support preserved roughly 87--98% of exact product edge
energy in middle/late layers. The viable follow-up is therefore:

1. use exact corrected probability with the cached product shortcut;
2. use exact finite-difference SwiGLU products on selected channel blocks;
3. retain dense fallback for unstable layers;
4. repeat the final-logit gate before runtime integration.

That follow-up must be described as draft-guided finite-difference support,
not as sparse Jacobian evaluation.
