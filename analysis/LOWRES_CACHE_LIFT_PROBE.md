# Low-resolution cache lifting probe

## Question

Can DINOv3 run its approximate pass on a genuinely smaller token grid, then
spatially lift the cached state required by partial-token correction, instead of
upsampling the low-resolution image before the approximate pass?

The prototype preserves prefix tokens, interpolates patch tokens in 2D, caches
K before RoPE, and applies the target-grid RoPE after lifting. It reuses the
existing partial-token correction implementation.

## Implementation

- Driver: `analysis/experiments/dinov3_lowres_cache_lift_probe.py`
- Cache lifting: `analysis/shared/lowres_cache_lift.py`
- Unit tests: `analysis/shared/test_lowres_cache_lift.py`
- Experimental `cache_pre_rope_k` option:
  `appcorr/models/dinov3/layers/{attention,block}.py`

Normal partial-token execution leaves `cache_pre_rope_k` disabled, so the
experiment adds no cache allocation to existing configurations.

The lifted correction cache contains, per layer:

- pre-RoPE K lifted to the high-resolution grid, followed by high-resolution RoPE;
- V lifted to the high-resolution grid;
- `blocks_out_sum` lifted to the high-resolution grid;
- `server_pscore` lifted to the high-resolution grid.

CLS and four register tokens are copied without interpolation.

## Validation

Unit tests:

```bash
python -m pytest -q analysis/shared/test_lowres_cache_lift.py
```

Result: 3 passed.

Twenty-image ADE20K whole-image proxy:

```bash
python analysis/experiments/dinov3_lowres_cache_lift_probe.py \
  --max-samples 20 \
  --keep-ratios 0.25,0.5,0.75,1.0 \
  --include-image-upsample-baseline \
  --device cuda:0 \
  --output logs/analysis/dinov3_lowres_cache_lift_ade20_fixed.json
```

This is a fast concept probe, not the official ADE20K slide evaluation. Inputs
are resized to 512x512, the high-resolution SPM/deform path is shared by every
variant, and the reported time covers ViT approximate forward, cache lifting,
and ViT correction.

| Variant | Rel. L2 | Cosine | ViT time (ms) | mIoU (20 images) |
|---|---:|---:|---:|---:|
| Full | 0.0000 | 1.000000 | 34.25 | 50.80 |
| Image-upsample approx | 0.0587 | 0.997933 | 37.72 | 48.43 |
| Low-grid approx + lift | 0.3605 | 0.947118 | 33.08 | 39.12 |
| Low-grid + 25% correction | 0.3296 | 0.951383 | 70.71 | 41.13 |
| Low-grid + 50% correction | 0.2705 | 0.964322 | 73.32 | 43.57 |
| Low-grid + 75% correction | 0.1832 | 0.982896 | 77.25 | 48.18 |
| Low-grid + 100% correction | 0.00183 | 0.999998 | 82.26 | 50.80 |

The 100% image-upsample and low-grid correction paths produce the same final
feature error and mIoU. This validates the lifted-cache correction contract in
BF16. The remaining relative L2 error of about 0.0018 is the existing BF16
full-correction numerical tier.

At 512x512:

- low-grid approximate forward: 20.88 ms;
- cache lifting: 12.20 ms;
- combined approximate path: 33.08 ms;
- full forward: 34.25 ms;
- speedup without correction: 1.035x.

The low-resolution source cache is 407.9 MiB, but the materialized lifted cache
is 1286.4 MiB, equal to the existing high-grid correction cache.

## Native 896 crop timing

One-image backbone-only timing:

```bash
python analysis/experiments/dinov3_lowres_cache_lift_probe.py \
  --high-size 896 \
  --max-samples 1 \
  --keep-ratios 0.5,0.75,1.0 \
  --include-image-upsample-baseline \
  --skip-head \
  --device cuda:0 \
  --output logs/analysis/dinov3_lowres_cache_lift_896_smoke.json
```

| Variant | Rel. L2 | ViT time (ms) | Speedup vs full |
|---|---:|---:|---:|
| Full | 0.0000 | 82.01 | 1.00x |
| Image-upsample approx | 0.0342 | 87.87 | 0.93x |
| Low-grid approx + lift | 0.1904 | 62.18 | 1.32x |
| Low-grid + 50% correction | 0.1618 | 123.08 | 0.67x |
| Low-grid + 75% correction | 0.1085 | 138.34 | 0.59x |
| Low-grid + 100% correction | 0.00169 | 151.75 | 0.54x |

At 896x896, the low-grid approximate forward itself is 31.57 ms and cache
lifting costs 30.61 ms. The source cache is 1232.9 MiB and the lifted cache is
3926.7 MiB.

## Conclusion

The concept is mechanically valid and accelerates the approximate component at
the real crop size: 1.32x for the measured 896 case, including PyTorch cache
lifting. It is not yet a system-level compute win once useful correction is
included. The 20-image proxy needs roughly 75% correction to approach the
image-upsample baseline, and every measured correction setting is slower than a
direct full forward.

The immediate bottlenecks are:

1. eagerly materializing every layer's full high-resolution cache;
2. correction QKV/attention/FFN work, which is added after the approximate pass;
3. substantially worse low-grid features, requiring a high correction ratio.

A follow-up is justified only if cache lifting can be fused or made layer-lazy
and correction is overlapped with transmission. The current result does not
support claiming an end-to-end compute speedup.

## G=4 interleaved expand-once follow-up

The expand-once experiment does not run the 40-layer low-resolution pass and
then correct every group through all 40 layers. It follows the static G=4
frontier:

```text
A_low(0,10)
C_new_g1(0,10)
A_low(10,20) + A_fine_g1(10,20)
C_new_g2(0,20)
A_low(20,30) + A_fine_g1,g2(20,30)
C_new_g3(0,30)
A_low(30,40) + A_fine_g1,g2,g3(30,40)
C_new_g4(0,40)
```

Each patch becomes fine once. On arrival it is corrected from layer zero to the
current low-resolution frontier, then remains fine and advances through later
10-layer approximate chunks. Earlier-layer queries are never revisited when a
later group arrives.

Unarrived high-resolution positions resolve K/V from their low-resolution
parent. Arrived positions retain fine K/V overrides for each layer. The G=4
checkerboard grouping assigns one child from every 2x2 high-resolution cell to
each group.

Command:

```bash
python analysis/experiments/dinov3_expand_once_probe.py \
  --max-samples 100 \
  --device cuda:0 \
  --output logs/analysis/dinov3_expand_once_interleaved_ade100.json
```

This uses the same 512x512 whole-image ADE20K proxy as the earlier experiment.

| Variant | Relative L2 | Cosine | mIoU (100 images) |
|---|---:|---:|---:|
| Full | 0.0000 | 1.000000 | 52.04 |
| Interleaved expand-once | 0.1199 ± 0.0206 | 0.99259 ± 0.00258 | 50.63 |

The accuracy loss is 1.41 mIoU points on this proxy. Thus expand-once preserves
task accuracy substantially better than lifting the complete low-resolution
result and applying a single partial correction.

Steady-state CUDA-event work, excluding the first sample's kernel
initialization:

| Component | Mean time |
|---|---:|
| Low-grid approximate chunks | 21.24 ms |
| Newly arrived group corrections | 91.25 ms |
| Previously arrived fine-group advances | 32.98 ms |
| Total ViT work | 145.47 ms |
| Direct full ViT | 34.29 ms |

The total work is 4.24x direct full inference. This reference resolves a logical
high-resolution K/V view with PyTorch gather and RoPE separately for each active
range; it is not a fused production kernel. Nevertheless, even an ideal kernel
must still process every fine patch through all 40 layers once. The potential
system benefit is therefore transmission overlap and earlier frontier progress,
not lower aggregate FLOPs.

An earlier interrupted run used a non-interleaved 40-layer-approx-then-correct
schedule. Its partial output is invalid for this experiment and is not reported.

### Block-grid grouping

The same 100-image experiment was repeated with four contiguous spatial
quadrants (`block_grid`) instead of checkerboard groups:

```bash
python analysis/experiments/dinov3_expand_once_probe.py \
  --max-samples 100 \
  --group-strategy block_grid \
  --device cuda:0 \
  --output logs/analysis/dinov3_expand_once_interleaved_block_grid_ade100.json
```

All other settings and the 10-layer interleaved schedule were unchanged.

| Grouping | Relative L2 | Cosine | mIoU | Loss vs full |
|---|---:|---:|---:|---:|
| Full | 0.0000 | 1.000000 | 52.04 | 0.00 |
| Checkerboard | 0.1199 ± 0.0206 | 0.99259 ± 0.00258 | 50.63 | -1.41 |
| Block grid | 0.0667 ± 0.0138 | 0.99769 ± 0.00098 | 51.09 | -0.95 |

Block-grid grouping improved mIoU by 0.46 points over checkerboard and reduced
the mean final-feature relative L2 error by 44%. It therefore meets a 1.0-point
mIoU-loss criterion on this proxy, although the margin is only 0.05 points.

| Grouping | Low approx | New-group correction | Fine advance | Total ViT work |
|---|---:|---:|---:|---:|
| Checkerboard | 21.24 ms | 91.25 ms | 32.98 ms | 145.47 ms |
| Block grid | 21.20 ms | 92.70 ms | 33.24 ms | 147.14 ms |

These are steady-state means excluding the first sample. Block-grid work was
1.1% slower than checkerboard and 4.29x direct full inference. The better
accuracy is consistent with spatially coherent quadrants limiting early
fine/low-resolution boundary mixing; this explanation is an inference and
requires a grouping-order or boundary ablation to establish causally.

### FLOPs estimate

For the 512x512/256x256 probe, the high- and low-resolution sequences contain
1029 and 261 tokens respectively, including CLS and four register tokens. The
expand-once schedule sends every high-resolution patch through all 40 layers
once. Repeating the five prefix tokens in independently packed correction
ranges raises the high-resolution query work slightly, from 41,160 to 41,610
token-layers. It additionally runs 10,440 low-resolution token-layers.

Using the ViT-7B dimensions (`D=4096`, SwiGLU internal width 8192) and counting
the four attention projections, three SwiGLU projections, QK, and probability-V
products gives:

| Work | Relative to direct full ViT |
|---|---:|
| QKV/output/FFN projections | 126.46% |
| Attention matrix products | 107.53% |
| Combined transformer FLOPs | 125.55% |

Thus the algorithm performs approximately **26% more transformer FLOPs** than a
plain full-resolution ViT forward, before selector, cache gather, RoPE,
packing, and scatter overhead. The corresponding rough totals are 14.50 TFLOPs
for direct full and 18.21 TFLOPs for expand-once when a multiply-add is counted
as two FLOPs.

This is fundamentally different from the measured 4.29x runtime. The prototype
serializes many small active-token calls and repeatedly gathers and constructs
logical high-resolution K/V tensors in eager PyTorch. Those operations add
little to the model FLOPs estimate but dominate wall-clock execution. A fused
implementation could approach the roughly 1.26x compute floor, but cannot make
this schedule cheaper in aggregate FLOPs than direct full inference.
