# Draft-Guided Support Experiment

This branch tests whether a low-resolution DINOv3 pass can identify a
GPU-friendly correction support. It deliberately keeps the existing
`partial_token` and `partial_channel` meanings unchanged.

The compact machine-readable result is
`analysis/experiments/results/jacobian_support_b200_summary.json`.

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

Independent exact-difference component sweep:

```bash
PYTHONPATH="$PWD" python analysis/experiments/jacobian_support_oracle.py \
  --image /path/to/image1.JPEG --image /path/to/image2.JPEG \
  --max-samples 2 --num-groups 1 --layers 0 \
  --support 0.5 --tail-epsilon 0.1 --exact-component-sweep \
  --sweep-ratios 0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1 \
  --output /tmp/exact_component_sweep.json
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

The prescribed dense gate failed for strict Jacobian correction. On 25 real
ImageNet validation images, correcting a 1/4-resolution canvas to full
resolution through all 40 layers produced:

| Dense backend | Mean logit relative L2 | Mean cosine | Stock top-1 match |
|---|---:|---:|---:|
| Base only | 0.4072 | 0.9110 | 25/25 |
| Split JVP attention + JVP FFN | 1.8550 | -0.4293 | 0/25 |
| Linearized product attention + JVP FFN | 2.0738 | -0.3811 | 0/25 |
| Exact-probability product + JVP FFN | 1.7152 | -0.3738 | 0/25 |
| Exact-probability product + exact finite-difference FFN | 0.0062 | 1.0000 | 25/25 |

The 25-sample gate used the first deterministic `ImageFolder` entries and is
therefore a parity gate, not an unbiased ImageNet accuracy estimate. Stock and
the exact finite-difference tier were correct on 24/25; all three JVP tiers
were correct on 0/25. This is far beyond the 1%p budget, so the planned
1,024-sample sparse-runtime sweep was stopped early.

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

## Exact finite-difference support sweep

As a follow-up sanity check, the same requested ratio was applied to input
token deltas, exact-probability attention edge blocks, and exact
finite-difference SwiGLU channel blocks. Three ImageNet images from distinct
classes were propagated through all 40 layers.

| Requested support | Token feature relative L2 | Token feature cosine |
|---:|---:|---:|
| 0% | 0.5583 | 0.8327 |
| 10% | 0.5583 | 0.8329 |
| 20% | 0.5550 | 0.8347 |
| 30% | 0.5425 | 0.8421 |
| 40% | 0.5215 | 0.8539 |
| 50% | 0.4868 | 0.8729 |
| 60% | 0.4411 | 0.8956 |
| 70% | 0.3906 | 0.9185 |
| 80% | 0.3090 | 0.9489 |
| 90% | 0.2133 | 0.9759 |
| 100% | 0.0224 | 0.9997 |

The mean curve moves from the base endpoint toward the full-resolution
endpoint. It is not mathematically monotonic per image: two of three samples
showed a small regression at 10--20% support, while one sample was monotonic
throughout. All samples improved consistently after the low-support region.
The 100% residual error is BF16/arithmetic-order drift rather than missing
support.

## Independent exact-difference component sweeps

The same three images were also evaluated by varying one component at a time
and holding the other two at 100%. Attention uses exact corrected softmax
probabilities and product deltas. FFN uses the exact finite difference of the
SwiGLU hidden product. Selection remains block structured.

| Requested support | Input token L2 / cosine | Attention edge L2 / cosine | FFN channel L2 / cosine |
|---:|---:|---:|---:|
| 0% | 0.5584 / 0.8326 | 0.5653 / 0.8246 | 0.5567 / 0.8340 |
| 10% | 0.5589 / 0.8372 | 0.5099 / 0.8582 | 0.5460 / 0.8407 |
| 20% | 0.5305 / 0.8506 | 0.3970 / 0.9160 | 0.5308 / 0.8500 |
| 30% | 0.4700 / 0.8798 | 0.3118 / 0.9487 | 0.5019 / 0.8665 |
| 40% | 0.4276 / 0.9006 | 0.2783 / 0.9593 | 0.4698 / 0.8834 |
| 50% | 0.3934 / 0.9162 | 0.2059 / 0.9778 | 0.4293 / 0.9035 |
| 60% | 0.3460 / 0.9337 | 0.1529 / 0.9875 | 0.3617 / 0.9321 |
| 70% | 0.3023 / 0.9499 | 0.1285 / 0.9911 | 0.3004 / 0.9534 |
| 80% | 0.2476 / 0.9662 | 0.0871 / 0.9960 | 0.2156 / 0.9763 |
| 90% | 0.1749 / 0.9833 | 0.0433 / 0.9990 | 0.1363 / 0.9905 |
| 100% | 0.0224 / 0.9997 | 0.0224 / 0.9997 | 0.0224 / 0.9997 |

Attention-edge support has the strongest early leverage: at a requested 50%
support its mean token-feature relative L2 is 0.2059, versus 0.3934 for input
tokens and 0.4293 for FFN channels. Its block rounding realizes 54.8% edge
support at that point; input realizes 50.2% and FFN exactly 50%.

The mean attention and FFN curves improve monotonically. The input-token mean
L2 has one small 0% to 10% regression, although cosine improves. At the
per-image level, token L2 is strictly monotonic for 2/3 input sweeps, 2/3
attention sweeps, and 3/3 FFN sweeps. The sole attention exception is a
90%-to-100% BF16/arithmetic-order endpoint drift on one image, not a missing
support effect.
