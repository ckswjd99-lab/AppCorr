# ADE20K L1-only and L2-to-L1 threshold correction

Date: 2026-07-29
Branch: `develop/dinov3-l2l1l0-tail-full`
Code base: `c81ad1f` plus the implementation committed with this report
Hardware: two NVIDIA B200 GPUs
Model/task: DINOv3 ViT-7B + Mask2Former, ADE20K, official 896-short-side
single-evaluation profile, first 100 validation images

## Question

This experiment establishes two intermediate-resolution anchors:

1. **L1-only:** transmit the level-1 Gaussian image and run the ordinary
   decomposed 40-layer backbone and head once, without correction.
2. **L2-to-L1:** transmit the level-2 base, run the 40-layer approximate
   backbone, then transmit the level-1 residual and run partial-token
   correction through all 40 layers with the existing threshold `4e-5`.

Both use the same ordered 100 images as the earlier full, L2-L0, and
L2-L1-L0 runs; all original-label hashes match.

## Controlled L1 endpoint

The generic progressive codec first computes a native-resolution Laplacian
band and then resizes it. Because resize and pyramid upsampling do not
commute, completing that residual was not exactly the same image as direct
L1-only transmission. The dedicated
`ADE20KL2L1ProgressiveLaplacian` policy instead sends

```text
R1 = projected(I1) - upsample(projected(I2)).
```

Decoding the complete residual therefore produces a pixel-identical L1 input.
A variable-aspect-ratio unit test checks this identity. This keeps codec error
out of the comparison.

The initial L1-only run used the legacy `FULL_INFERENCE` helper. Its accuracy
was valid (`52.7796` mIoU), but its execution path and timing were not directly
comparable to AppCorr. The primary L1-only result below is the rerun through
the same decomposed `prepare -> backbone -> head` path used by L2-to-L1.

## Result

| Metric | Full/sequential | L1-only | L2-to-L1, `t=4e-5` |
|---|---:|---:|---:|
| Global mIoU | **52.9229** | 52.6536 | 46.5090 |
| Gap to full | — | -0.2692 p | -6.4139 p |
| Global aAcc | **84.0801** | 83.8634 | 81.9009 |
| Bytes/image | 3,164,119 | **496,561** | 691,960 |
| Selected patch queries/sample | — | — | 701.54 / 3,136 |
| Selected patch-query-layer ratio | — | — | 22.37% |
| Covered fused-pscore mass | — | — | 99.53% |
| Correction patch-token-layers/image | 0 | 0 | 52,194.8 |
| Correction query-token-layers/image | 0 | 0 | 52,566.8 |
| Correction TFLOPs/image | 0 | 0 | 20.34 |
| Total backbone TFLOPs/image | 90.44 | 90.44 | 110.78 |
| Backbone approximate CUDA-event sum | 164.50 ms | 173.22 ms | 220.69 ms |
| Correction CUDA-event sum | — | — | 98.84 ms |
| Local end-to-end average | 404.69 ms | 578.56 ms | 953.52 ms |

L1-only is already close to full resolution: it loses only **0.269 mIoU**
while transmitting 84.3% fewer bytes. It does not reduce backbone FLOPs,
because the L1 image is expanded to the official model-input shape before
tokenization.

At the unchanged `4e-5` threshold, L2-to-L1 is **6.145 mIoU below its own
L1-only endpoint**. It selects only 22.37% of patch-query-layer work, even
though those queries contain 99.53% of the fused pscore mass. Thus the
threshold preserves the scalar score mass but not the feature consistency
needed to reproduce a coherent full L1 forward.

Using the same dominant per-query-layer estimate as the earlier experiments,

```text
8 D^2 + 4 N D + 6 D M = 387,006,464 FLOPs,
```

the correction costs 20.34 TFLOPs/image, or 22.49% of one full backbone
forward. Approximation plus correction is therefore 122.49% of ordinary
backbone arithmetic. L2-to-L1 also sends 39.4% more bytes than L1-only and
has two transmission/prepare phases.

Runtime timings were collected while the two B200 runs were active
concurrently and include different communication phase counts. They are
useful as observed system costs, not an isolated kernel benchmark.

## Accuracy diagnostics

Global confusion-matrix mIoU is authoritative. For paired per-image mIoU,
L2-to-L1 minus L1-only has mean `-2.844` points with bootstrap 95% CI
`[-4.773, -0.902]`; L2-to-L1 wins 35 of 100 images. The degradation is
therefore not just a global-aggregation artifact.

The exact input endpoint does not imply bitwise BF16 correction parity.
With 100% support on the first real image:

```text
L1-only mIoU:             69.25871
L2-to-L1 100% support:   69.25562
difference:              -0.00309 points
```

Prediction hashes differ, so this is near-parity rather than bitwise parity.
The remaining difference is consistent with arithmetic ordering/precision in
the approximate-then-correct path, not with the image codec.

## Comparison with the previous runs

The existing L2-L0 and equal-threshold L2-L1-L0 results were respectively
`52.0812` and `52.1872` mIoU. Those methods receive L0 information and are
not direct accuracy peers of the L1 endpoint, but they show that the failure
here is specific to stopping after an aggressively pruned L1 correction.

The same numeric threshold is not transferable across the two residual
levels:

```text
L2-to-L1 selected query-layer ratio: 22.37%
L2-to-L0 selected query-layer ratio: 46.25%
```

For L2-to-L1, the next meaningful calibration would lower the threshold or
target a substantially larger keep ratio and sweep against the L1-only
endpoint. The present `4e-5` point is not usable.

## Reproduction

```bash
# L1-only, decomposed forward
RECV_PORT=55600 SEND_PORT=55601 CUDA_VISIBLE_DEVICES=0 \
conda run --no-capture-output -n appcorr \
  offload/run_local.sh \
  offload/config/ade20k_m2f_approx_only_l1.json \
  -nr 100 -nw 1 --set device=cuda:0 \
  --set exp_id=ade20k_m2f_approx_only_l1_decomposed_nr100

# L2-to-L1, existing threshold
RECV_PORT=55500 SEND_PORT=55501 CUDA_VISIBLE_DEVICES=1 \
conda run --no-capture-output -n appcorr \
  offload/run_local.sh \
  offload/config/ade20k_m2f_l2l1_appcorr_t4e5.json \
  -nr 100 -nw 1 --set device=cuda:0 \
  --set exp_id=ade20k_m2f_l2l1_appcorr_t4e5_nr100

# One-image full-support near-parity check
RECV_PORT=55700 SEND_PORT=55701 CUDA_VISIBLE_DEVICES=1 \
conda run --no-capture-output -n appcorr \
  offload/run_local.sh \
  offload/config/ade20k_m2f_l2l1_appcorr_t4e5.json \
  -nr 1 -nw 0 --set device=cuda:0 \
  --set exp_id=ade20k_m2f_l2l1_fullsupport_parity_smoke \
  --set appcorr_kwargs.token_keep_thres=-1 \
  --set appcorr_kwargs.l1_token_keep_thres=-1
```

Raw logs remain ignored:

```text
logs/offload/ade20k_m2f_approx_only_l1_decomposed_nr100_20260729_011017/
logs/offload/ade20k_m2f_l2l1_appcorr_t4e5_nr100_20260729_010352/
logs/offload/ade20k_m2f_l2l1_fullsupport_parity_smoke_20260729_011126/
```
