# ADE20K L2-L1-L0 conditional re-entry

Date: 2026-07-29
Branch: `develop/dinov3-l2l1l0-tail-full`
Code base: `c81ad1f` plus the implementation committed with this report
Hardware: two NVIDIA B200 GPUs, one paired run per GPU
Model/task: DINOv3 ViT-7B + Mask2Former, ADE20K, official
896-short-side single-evaluation profile, first 100 validation images

## Question

The earlier strict L1/L0-disjoint experiment lost 1.47 mIoU points and did not
reduce correction work. This follow-up asks whether a conservative L1 score
and limited L0 re-entry can recover accuracy without giving up the hoped-for
compute reduction.

## Implemented policy

Let `I2`, `I1`, and `I0` be the images reconstructed by the real decoder after
receiving through levels L2, L1, and L0:

```text
R1    = I1 - I2
Rfine = I0 - I1
Rall  = I0 - I2
```

The L1 mobile score is:

```text
max(energy(R1) - energy(Rfine), 0)
```

A cell is additionally rejected at L1 when:

```text
energy(Rfine) / max(energy(R1), eps) > 0.5
```

This avoids correcting a cell early when substantial fine residual remains.
Negative differences are not used as ranking values: after multiplication or
geometric fusion with attention they do not provide a stable importance
ordering, and a negative threshold would make the selected workload difficult
to control.

At L0, the mobile score is selected after observing the actual L1 kernel
support:

```text
L1-selected token:       energy(I0 - I1)
token not selected L1:   energy(I0 - I2)
```

Both branches remain raw until this conditional selection, then use one common
normalization denominator. The server combines the chosen residual energy with
layer-mean received attention through the existing geometric-mean fusion.

For L1-selected tokens, L0 re-entry is a separate structured budget. Within
each real query plan, the implementation ranks them by the final fused score
and forcibly recomputes the highest-risk fraction. The remaining L1-selected
tokens are excluded, while tokens not selected at L1 follow the ordinary L0
threshold. This selection occurs before the existing padded/bucketed
partial-token kernel; it is not dense computation followed by masking.

The evaluated operating point was:

```text
l1_token_keep_thres = 1e-5
l0_token_keep_thres = 1.2e-4
l1_remaining_energy_ratio_max = 0.5
l1_l0_reentry_ratio = 0.5
```

## 20-image re-entry sweep

The safe gate was fixed at 0.5 and the initial L0 threshold at `4e-5`.
The re-entry ratio is the forced fraction of L1-selected tokens eligible at
L0. All rows use the same first 20 images.

| Forced L0 re-entry | mIoU | aAcc | Patch-token-layers/request | L1 work | L0 work | Observed overlap, total |
|---:|---:|---:|---:|---:|---:|---:|
| 0% | 47.3986 | 83.8029 | 140,463.3 | 9,236.4 | 131,226.9 | 0 |
| 10% | 47.9807 | 83.8141 | 143,665.9 | 9,236.4 | 134,429.5 | 64,091 |
| 25% | 48.5493 | 85.2626 | 148,376.3 | 9,236.4 | 139,139.9 | 158,219 |
| 50% | **50.0991** | 86.2245 | 156,249.7 | 9,236.4 | 147,013.3 | 315,768 |
| 100% | 49.6229 | **86.4147** | 172,036.1 | 9,236.4 | 162,799.7 | 631,536 |

Accuracy recovery is not monotonic, but 50% was the best mIoU point. It also
used 30.8% more correction work than the paired L2-L0 baseline
(119,450 patch-token-layers/request), so the L0 threshold was retuned.

## 20-image work-budget sweep

Re-entry was fixed at 50%.

| L0 threshold | mIoU | aAcc | Patch-token-layers/request | Correction time |
|---:|---:|---:|---:|---:|
| `6e-5` | 49.5791 | 85.2019 | 142,959.6 | 171.94 ms |
| `8e-5` | 49.9808 | 85.3173 | 132,284.8 | 198.60 ms |
| `1e-4` | 49.7701 | 83.9716 | 123,634.3 | 162.84 ms |
| `1.2e-4` | **50.3460** | 83.9869 | **115,815.2** | **155.51 ms** |

The paired 20-image L2-L0 baseline was 50.6021 mIoU, 85.9487 aAcc,
119,450 patch-token-layers, and 127.81 ms correction time. The `1.2e-4`
point was selected for the larger validation because it came within 0.26 mIoU
while reducing the measured correction work by 3.0%. The small-set aAcc and
latency already warned that this was not a clear win.

## Paired 100-image result

The baseline is the existing L2-L0 crop-cover interleaving with
`token_keep_thres=4e-5`. Both jobs processed the same ordered 100 images; all
label hashes matched.

| Metric | L2-L0 baseline | Conditional re-entry | Delta |
|---|---:|---:|---:|
| Global mIoU | **52.0812** | 50.5117 | **-1.5695 p** |
| Global aAcc | **83.5256** | 82.6657 | -0.8599 p |
| Patch-token-layers/request | **118,979.1** | 120,361.0 | +1.16% |
| Query-token-layers/request, including 5 prefix tokens | **119,531.7** | 121,106.8 | +1.32% |
| L1 patch-token-layers/request | 0 | 9,803.2 | — |
| L0 patch-token-layers/request | 118,979.1 | 110,557.8 | -7.08% |
| L1/L0 overlap token-layers, full run | 0 | 1,690,063 | intended |
| Bytes/image | **2,385,394** | 2,601,549 | +9.06% |
| Correction CUDA-event sum/request | **134.79 ms** | 179.58 ms | +33.23% |
| End-to-end/request | **1,165.83 ms** | 1,247.00 ms | +6.96% |

The dataset mIoU is the authoritative global confusion-matrix metric. As a
secondary paired diagnostic, mean per-image mIoU changed by -0.490 points
(bootstrap 95% CI `[-1.494, +0.567]`); the candidate won 38 of 100 images and
had a median delta of -0.444. The corresponding paired intervals were:

| Paired candidate-minus-baseline quantity | Mean | Bootstrap 95% CI |
|---|---:|---:|
| Bytes/image | +216,155 | `[+200,435, +233,287]` |
| Correction time | +44.79 ms | `[+26.99, +63.13]` |
| End-to-end time | +81.17 ms | `[+17.17, +144.65]` |

The previous strict-disjoint run reported 50.6071 mIoU and 120,291.6
patch-token-layers/request. Conditional re-entry produced 50.5117 mIoU and
120,361.0 patch-token-layers/request. Thus the more nuanced policy is
effectively level with strict disjointness on the 100-image result rather than
recovering the baseline gap.

The runs were simultaneous on separate B200 GPUs and shared the host, so
mobile encode and end-to-end variance include host contention. The correction
CUDA-event regression is nevertheless large, paired, and directionally
consistent with the extra correction round and selector/packing work.

## FLOPs estimate

At the 896x896 source shape, `D=4096`, `M=8192`, and `N=3141` including five
prefix tokens. The dominant work per recomputed query-layer is:

```text
8 D^2 + 4 N D + 6 D M = 387,006,464 FLOPs
```

The workload averages 1.86 sliding-window sources per image, making the full
40-layer approximate backbone pass approximately 90.44 TFLOPs/image.

| Backbone work | L2-L0 baseline | Conditional re-entry |
|---|---:|---:|
| Approximate pass | 90.44 TFLOPs | 90.44 TFLOPs |
| Correction | 46.26 TFLOPs | 46.87 TFLOPs |
| Correction / approximate | 51.15% | 51.82% |

This estimate covers dominant ViT attention/projection/FFN arithmetic. It does
not count selection, packing, image decode, token preparation, or the
segmentation head. Those omitted costs make the measured latency result worse,
not better.

## Conclusion

The implementation behaves as intended: it uses exact decoded residual
branches, applies the L1 safety gate, conditions L0 energy on actual L1 kernel
support, preserves common score normalization, and performs structured
highest-risk re-entry before the existing sparse-query kernel.

The system hypothesis is rejected at the evaluated point. Conditional
re-entry recovered accuracy within the 20-image tuning set, but did not
generalize to the paired 100-image set. It also failed to reduce correction
FLOPs, while the L1 band structurally added 9.1% transmission and another
decode/select/correction round.

This mode should remain an ablation and should not replace the L2-L0 baseline.
Further scalar-threshold tuning is unlikely to create a system win: even equal
selected FLOPs still pay the extra L1 bytes and launch overhead. A meaningful
follow-up would need either feature-space uncertainty that removes much more
L0 work, or a transmission representation in which L1 does not add bytes.

## Reproduction

```bash
# Conditional re-entry candidate
RECV_PORT=55400 SEND_PORT=55401 CUDA_VISIBLE_DEVICES=0 \
conda run --no-capture-output -n appcorr \
  offload/run_local.sh \
  offload/config/ade20k_m2f_interleaved_l2l1l0_conditional.json \
  -nr 100 -nw 1 --set device=cuda:0 \
  --set exp_id=ade20k_l2l1l0_forced_r50_l0t12e5_nr100

# Paired L2-L0 baseline
RECV_PORT=55500 SEND_PORT=55501 CUDA_VISIBLE_DEVICES=1 \
conda run --no-capture-output -n appcorr \
  offload/run_local.sh \
  offload/config/ade20k_m2f_interleaved_static.json \
  -nr 100 -nw 1 --set device=cuda:0 \
  --set exp_id=ade20k_l2l0_baseline_t4e5_nr100_recheck
```

Raw logs remain ignored and are not committed:

```text
logs/offload/ade20k_l2l1l0_forced_r50_l0t12e5_nr100_20260729_002524/
logs/offload/ade20k_l2l0_baseline_t4e5_nr100_recheck_20260729_002529/
```
