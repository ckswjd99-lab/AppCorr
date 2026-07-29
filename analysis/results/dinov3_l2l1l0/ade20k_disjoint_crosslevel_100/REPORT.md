# ADE20K L2-L1-L0 disjoint cross-level correction

Date: 2026-07-28
Branch: `develop/dinov3-l2l1l0-tail-full`
Code base: `c81ad1f` plus the implementation committed with this report
Hardware: two NVIDIA B200 GPUs, one experiment process per GPU
Model/task: DINOv3 ViT-7B + Mask2Former, ADE20K, official 896-short-side
single-evaluation profile, first 100 validation images

## Method

Let `I2`, `I1`, and `I0` be the images reconstructed by the actual decoder
after receiving through L2, L1, and L0. The implementation measures residuals
after the decoder's resize, integer addition, and clipping:

```text
R1    = I1 - I2
Rfine = I0 - I1
Rall  = I0 - I2 = R1 + Rfine
```

For each L1 spatial cell, final-resolution patch energies are summed over the
aligned fine cells. Its mobile score is:

```text
max(energy(R1) - energy(Rfine), 0)
```

For an L0 token not selected at L1, its mobile score is `energy(Rall)`. The
server combines the normalized mobile score with layer-mean patch-received
attention using the existing geometric-mean fusion. This is monotonic in the
requested energy-times-attention product; thresholds are calibrated in the
geometric-mean score domain.

The actual L1 partial-token selection is captured from the shared GPU query
plan. Every later L0 `dindice` is filtered before kernel bucketing. CLS and four
register tokens remain present, while selected L1 patch tokens are absent from
all L0 candidate lists. The measured L1/L0 overlap was exactly zero.

Configuration:

```text
l1_token_keep_thres = 1e-5
l0_token_keep_thres = 4e-5
l1_pscore_mode = positive_residual_difference
l0_pscore_mode = conditional_cumulative_residual_energy
l1_l0_disjoint_support = true
```

## Calibration on 10 images

L0 threshold was fixed at `4e-5`; only the L1 threshold changed.

| L1 threshold | mIoU | selected patch-token-layers/request | L1 work | L0 work | L0 candidates excluded by L1 |
|---:|---:|---:|---:|---:|---:|
| `1e-5` | 47.9575 | 116,932.0 | 39,738.0 | 77,194.0 | 136,533.2 |
| `2e-5` | 47.6419 | 115,981.6 | 37,918.4 | 78,063.2 | 130,495.8 |
| `4e-5` | 47.7784 | 114,822.6 | 33,401.8 | 81,420.8 | 115,360.6 |
| `8e-5` | 47.1897 | 116,830.5 | 27,519.8 | 89,310.7 | 95,125.9 |

The `1e-5` point had the best 10-image mIoU and was selected before the
100-image run. The choice was made on a small calibration subset and is not
claimed to be globally optimal.

## Paired 100-image result

The baseline is the existing L2-L0 crop-cover interleaving with
`token_keep_thres=4e-5`. Both runs used the same ordered 100 images; all label
hashes matched.

| Metric | L2-L0 baseline | New L2-L1-L0 disjoint | Delta |
|---|---:|---:|---:|
| mIoU | 52.0812 | 50.6071 | **-1.4741 p** |
| aAcc | 83.5256 | 82.2443 | -1.2812 p |
| Selected patch-token-layers/request | 118,979.1 | 120,291.6 | +1.10% |
| Selected query-token-layers/request, including 5 prefix tokens | 119,531.7 | 121,037.4 | +1.26% |
| Full candidate patch-token-layers/request | 257,256.2 | 250,526.5 | -2.62% |
| L1 selected patch-token-layers/request | 0 | 35,071.9 | — |
| L0 selected patch-token-layers/request | 118,979.1 | 85,219.7 | -28.38% |
| L0 candidate patch-token-layers excluded by L1 | 0 | 120,368.3 | — |
| L1/L0 selected overlap | — | **0** | invariant passed |
| Bytes/image | 2,385,394 | 2,601,549 | +9.06% |

Per-image mIoU is noisy because each image contains a different class subset,
but it gives a paired direction check: the new method won on 38 images and the
baseline won on 62. The mean per-image delta was -0.993 points and the median
was -0.428 points. The reported dataset mIoU above is the authoritative global
confusion-matrix result.

## FLOPs estimate

For the 896x896 ViT source shape, `D=4096`, `M=8192`, and `N=3141` including
five prefix tokens. The dominant work per recomputed query-layer is estimated
as:

```text
8 D^2 + 4 N D + 6 D M = 387,006,464 FLOPs
```

The 100-image workload averaged 1.86 sliding-window sources per image, so a
full 40-layer approximate backbone pass is approximately 90.44 TFLOPs/image.

| Backbone work | L2-L0 baseline | New disjoint |
|---|---:|---:|
| Approximate pass | 90.44 TFLOPs | 90.44 TFLOPs |
| Correction | 46.26 TFLOPs | 46.84 TFLOPs |
| Correction / approximate | 51.15% | 51.79% |

This estimate covers the dominant ViT attention/projection/FFN arithmetic. It
does not include selection, packing, image decode, token preparation, or the
segmentation head. The new method does not reduce theoretical correction FLOPs
at the selected operating point.

## Timing

The two 100-image runs executed simultaneously on separate B200 GPUs and
shared host CPU/storage. GPU correction timings are useful for relative
diagnosis; mobile encode and end-to-end tail latency are affected by shared
host contention.

| Timing | L2-L0 baseline | New disjoint | Delta |
|---|---:|---:|---:|
| Correction average | 132.38 ms | 183.40 ms | +38.5% |
| Correction p50 | 113.31 ms | 153.34 ms | +35.3% |
| Correction p95 | 267.34 ms | 323.91 ms | +21.2% |
| End-to-end average | 1087.28 ms | 1155.14 ms | +6.2% |
| End-to-end p50 | 1089.72 ms | 1192.65 ms | +9.4% |
| End-to-end p95 | 1779.41 ms | 1610.23 ms | -9.5% |

The p95 end-to-end reversal is not treated as a speedup because shared CPU
contention produced large mobile-encode variance. The correction CUDA-event
measurements and p50 end-to-end both show a regression. The extra L1 round also
raises decode, token-preparation, transmission, and kernel-launch overhead.

## Conclusion

The implementation contract is successful: cross-level scores are computed
from exact decoded state deltas, L1 selections are captured from the real
kernel plan, prefix tokens remain active, and L1/L0 patch support is strictly
disjoint.

The performance hypothesis is rejected at this operating point. Never
revisiting a token after L1 is too strong: small pixel-space `Rfine` energy does
not guarantee that the token's final feature is already adequate after
nonlinear and cross-token propagation. The extra intermediate band also costs
9.1% more bytes and adds a correction round, while the selected correction
FLOPs are 1.3% higher rather than lower.

This mode should remain experimental and should not replace the L2-L0
baseline. A follow-up should permit conditional L0 re-entry for risky
L1-selected tokens, using the remaining `Rfine` signal and feature-space risk,
instead of enforcing a hard disjoint set.

## Reproduction

```bash
# New method
RECV_PORT=48790 SEND_PORT=48791 CUDA_VISIBLE_DEVICES=0 \
conda run --no-capture-output -n appcorr \
  offload/run_local.sh \
  offload/config/ade20k_m2f_interleaved_l2l1l0_static.json \
  -nr 100 -nw 1 --set device=cuda:0 \
  --set exp_id=ade20k_l2l1l0_disjoint_crosslevel_t1e5_nr100

# Paired L2-L0 baseline
RECV_PORT=49790 SEND_PORT=49791 CUDA_VISIBLE_DEVICES=1 \
conda run --no-capture-output -n appcorr \
  offload/run_local.sh \
  offload/config/ade20k_m2f_interleaved_static.json \
  -nr 100 -nw 1 --set device=cuda:0 \
  --set exp_id=ade20k_l2l0_baseline_t4e5_nr100
```

Raw logs remain ignored and are not part of the repository:

```text
logs/offload/ade20k_l2l1l0_disjoint_crosslevel_t1e5_nr100_20260728_232552/
logs/offload/ade20k_l2l0_baseline_t4e5_nr100_20260728_232542/
```
