# ADE20K equal-threshold conditional L2-L1-L0

Date: 2026-07-29
Branch: `develop/dinov3-l2l1l0-tail-full`
Code base: `c81ad1f` plus the implementation committed with this report
Hardware: NVIDIA B200; the two progressive methods ran simultaneously on
separate GPUs, then full/sequential ran on one GPU
Model/task: DINOv3 ViT-7B + Mask2Former, ADE20K, official 896-short-side
single-evaluation profile, first 100 validation images

## Evaluated policy

This experiment removes both strict disjointness and a separate re-entry
budget. L1 and L0 use the same ordinary partial-token threshold:

```text
token_keep_thres = l1_token_keep_thres = l0_token_keep_thres = 4e-5
```

The L1 mobile score is ordinary decoded L1 residual energy. At L0, the server
uses the actual L1 threshold result to choose the residual branch:

```text
selected at L1:       energy(I0 - I1)
not selected at L1:   energy(I0 - I2)
```

Equivalently, when `R0 = I0-I2` and `R1 = I1-I2`, the selected branch is
`energy(R0-R1)`. Reversing the subtraction has the same value because the
score is squared L2 energy. Branch selection occurs before one common
normalization. The resulting mobile score is geometrically fused with
layer-mean received attention, exactly as in the L2-L0 configuration.

The L1 mask is not passed to the transformer block as a restriction. No token
is forcibly added or removed because it ran at L1. Every L0 candidate is
selected only by the same existing threshold and then executed by the
structured partial-token query kernel.

## Compared configurations

1. Full/sequential: raw full-resolution input and one ordinary full backbone
   forward.
2. L2-L0: existing interleaved partial-token config with threshold `4e-5`.
3. L2-L1-L0 conditional threshold: the policy above.

All three runs processed the same ordered 100 images; every original-label
hash matched.

## Accuracy and system result

| Metric | Full/sequential | L2-L0 | L2-L1-L0 equal threshold |
|---|---:|---:|---:|
| Global mIoU | **52.9229** | 52.0812 | 52.1872 |
| Gap to full | — | -0.8417 p | -0.7356 p |
| Global aAcc | **84.0801** | 83.5256 | 83.8675 |
| Bytes/image | 3,164,119 | **2,385,394** | 2,601,549 |
| Correction patch-token-layers/image | 0 | **118,979.1** | 218,370.6 |
| Correction TFLOPs/image | 0 | **46.26** | 84.80 |
| Total backbone TFLOPs/image | **90.44** | 136.70 | 175.24 |
| Correction CUDA-event sum | — | **126.30 ms** | 240.05 ms |
| Local end-to-end average | **404.69 ms** | 1,145.12 ms | 1,210.79 ms |

Relative to L2-L0, the new policy:

- gains only **+0.1060 mIoU points**;
- recovers 12.6% of the L2-L0-to-full mIoU gap;
- uses **83.5% more correction patch-token-layers**;
- uses 9.1% more transmitted bytes;
- raises measured correction time by **90.1%**;
- raises local end-to-end time by 5.7%.

The local end-to-end comparison includes each configuration's real encode,
decode, transmission, scheduling, and head path. Full/sequential uses one raw
transfer rather than progressive rounds, so its latency is not a
network-bandwidth sweep. Correction CUDA events and arithmetic counts are the
cleaner progressive-method comparison.

## Why work increased

The new method averaged:

```text
L1 selected patch-token-layers/image =  68,029.76
L0 selected patch-token-layers/image = 150,340.82
total                                 = 218,370.58
```

Of the L0 selected work, 133,932.11 token-layers/image also belonged to
L1-selected support. This is **89.1% of L0 work**. The conditional residual
correctly changes the score for these tokens, but at `4e-5` most still pass
the L0 threshold. Consequently, equal thresholding does not prevent
cross-level recomputation.

Using the same dominant per-query-layer estimate as the prior experiments:

```text
8 D^2 + 4 N D + 6 D M = 387,006,464 FLOPs
```

the equal-threshold correction alone is 93.76% of a full backbone forward.
Approximate plus correction backbone arithmetic is therefore 193.76% of full
sequential. L2-L0 is 151.15% by the same estimate.

## Paired diagnostics

Global confusion-matrix mIoU is authoritative. Mean per-image mIoU is noisy:
the equal-threshold method minus L2-L0 was -0.093 points with bootstrap 95% CI
`[-0.923, +0.706]`, despite the global mIoU difference being +0.106. It won
65/100 per-image mIoU comparisons, showing many small improvements offset by
larger losses on some images. There is no statistically persuasive accuracy
gain at this sample size.

## Conclusion

The requested semantics are implemented and validated, but the operating
point is inefficient. Treating every correction round identically is simple
and recovers a very small amount of accuracy, yet causes extensive L1/L0
support overlap. It nearly doubles correction work relative to L2-L0 and
still remains 0.736 mIoU below full.

This result explains why a cross-level budget was considered: without one,
the same threshold does not account for work already spent. However, the
earlier hard-disjoint and forced-reentry variants also failed to beat L2-L0.
For this representation, the extra L1 band has not demonstrated a useful
accuracy/compute operating point.

## Reproduction

```bash
# Equal-threshold conditional L2-L1-L0
RECV_PORT=55700 SEND_PORT=55701 CUDA_VISIBLE_DEVICES=0 \
conda run --no-capture-output -n appcorr \
  offload/run_local.sh \
  offload/config/ade20k_m2f_interleaved_l2l1l0_threshold_only.json \
  -nr 100 -nw 1 --set device=cuda:0 \
  --set exp_id=ade20k_l2l1l0_threshold_only_t4e5_nr100

# L2-L0
RECV_PORT=55800 SEND_PORT=55801 CUDA_VISIBLE_DEVICES=1 \
conda run --no-capture-output -n appcorr \
  offload/run_local.sh \
  offload/config/ade20k_m2f_interleaved_static.json \
  -nr 100 -nw 1 --set device=cuda:0 \
  --set exp_id=ade20k_l2l0_t4e5_nr100_threshold_compare

# Full/sequential
RECV_PORT=55900 SEND_PORT=55901 CUDA_VISIBLE_DEVICES=1 \
conda run --no-capture-output -n appcorr \
  offload/run_local.sh \
  offload/config/ade20k_m2f_sequential.json \
  -nr 100 -nw 1 --set device=cuda:0 \
  --set exp_id=ade20k_m2f_sequential_nr100_threshold_compare
```

Raw logs remain ignored:

```text
logs/offload/ade20k_l2l1l0_threshold_only_t4e5_nr100_20260729_004401/
logs/offload/ade20k_l2l0_t4e5_nr100_threshold_compare_20260729_004355/
logs/offload/ade20k_m2f_sequential_nr100_threshold_compare_20260729_004956/
```
