# DINOv3 L1-only multi-task benchmark

Date: 2026-07-30

## Semantics

`L1-only` means that the client downsamples the image once, transmits only the
Laplacian/Gaussian L1 image, and reconstructs it at the model input size before
one ordinary full model forward. No L0 residual is transmitted and no
approx/correct, pscore, token selection, or correction kernel is invoked.

Because there are no later residual groups to interleave, these runs use the
one-shot `BatchCountBased` scheduler. The model, task head, target input shape,
and task-specific preprocessing are the same as the corresponding static
interleaved experiments.

## Evaluation

| Task | Validation size | L1 config | Result log |
|---|---:|---|---|
| ImageNet-1K | 50,000 | `offload/config/imnet_approx_only_l1.json`, batch 32 | `logs/offload/imnet_approx_only_l1_20260729_143223` |
| ADE20K | 2,000 | `offload/config/ade20k_m2f_approx_only_l1.json`, batch 1 | `logs/offload/ade20k_m2f_approx_only_l1_decomposed_full2000_20260730_005917` |
| ADE20K calibration | 100-image paired subset | same | `logs/offload/ade20k_m2f_approx_only_l1_decomposed_nr100_20260729_011017` |
| COCO val2017 | 5,000 | `offload/config/coco_approx_only_l1.json`, batch 1 | `logs/offload/coco_approx_only_l1_20260729_143225` |
| NYUv2 | 654 | `offload/config/nyu_approx_only_l1.json`, batch 1, TTA enabled | `logs/offload/nyu_approx_only_l1_20260729_144456` |

All runs used DINOv3 ViT-7B and the released task head. COCO ran in the
`appcorr` environment because its dataset loader depends on FiftyOne. The
ADE20K progressive ablations below remain paired 100-image calibration
results, while the L1-only endpoint now also has a complete 2,000-image
validation result.

## Results

### ImageNet-1K

The full-resolution comparator is the existing 50,000-image exact L0
evaluation committed on `experiment/jacobian-support-oracle` in
`analysis/experiments/results/jacobian_policy_token50_special_attnffn_imagenet1k_summary.json`.
It uses the same 256x256 input and checkpoint.

| Input | Top-1 | Top-5 | Delta top-1 | Delta top-5 |
|---|---:|---:|---:|---:|
| Full resolution | 88.104% | 98.438% | - | - |
| L1 only | 87.754% | 98.358% | -0.350 pp | -0.080 pp |

### ADE20K full validation

The complete 2,000-image L1-only run and the existing exact sequential
baseline use the same DINOv3 ViT-7B + Mask2Former checkpoint, official
896-short-side single-evaluation profile, and full validation order.

| Input | mIoU | aAcc | Delta mIoU | Delta aAcc | Bytes/image |
|---|---:|---:|---:|---:|---:|
| Full resolution | 62.2358 | 87.4542 | - | - | 3,180,075 |
| L1 only | 61.3669 | 87.1351 | -0.8688 pp | -0.3191 pp | 494,816 |

L1-only reduces transmitted bytes by 84.44% while losing 0.869 mIoU. This is
a larger accuracy gap than the first-100 result below, but remains below one
mIoU point. As before, the L1 image is reconstructed at the normal input size
and executes the complete backbone, so this result does not reduce backbone
FLOPs.

The L1-only run averaged 425.5 ms end-to-end, 162.8 ms for the backbone, and
173.6 ms for the head. The exact baseline log was collected on an earlier
date with different warm-up and host conditions, so its latency is not used
for a controlled speedup claim.

### ADE20K (100-image paired calibration)

The exact full-resolution and L1-only runs used the same DINOv3 ViT-7B +
Mask2Former model, 896-short-side preprocessing, evaluation order, and label
hashes. L1-only used the decomposed one-shot path and its dedicated
target-exact codec.

| Input | mIoU | aAcc | Delta mIoU | Delta aAcc | Bytes/image | Backbone TFLOPs/image |
|---|---:|---:|---:|---:|---:|---:|
| Full resolution | 52.9229 | 84.0801 | - | - | 3,164,119 | 90.44 |
| L1 only | 52.6536 | 83.8634 | -0.2692 pp | -0.2167 pp | 496,561 | 90.44 |

L1-only reduced transmitted bytes by 84.31%, but it did not reduce model
compute: the L1 image is reconstructed at the normal input size and still
executes one full backbone forward. Its end-to-end time was 578.6 ms/image,
versus 404.7 ms/image for the sequential full-resolution run, so this result
is an accuracy/traffic observation rather than a demonstrated latency
speedup.

The same 100-image subset was also used for the earlier progressive
correction variants:

| Method | mIoU | Gap to full | Bytes/image | Correction TFLOPs/image | Result |
|---|---:|---:|---:|---:|---|
| L2→L0 interleaved static, `t=4e-5` | 52.0812 | -0.8417 pp | 2,385,394 | 46.26 | Existing two-level baseline |
| L2→L1→L0, equal threshold | 52.1872 | -0.7356 pp | 2,601,549 | 84.80 | +0.1060 pp over L2→L0, but +83.5% correction work |
| L2→L1→L0, strict disjoint | 50.6071 | -2.3158 pp | 2,601,549 | 46.84 | Rejected |
| L2→L1→L0, conditional re-entry | 50.5117 | -2.4112 pp | 2,601,549 | 46.87 | Rejected |
| L2→L1 endpoint, `t=4e-5` | 46.5090 | -6.4139 pp | 691,960 | 20.34 | Stops at L1; threshold loses too much support |

The equal-threshold three-level scheme was the only L2→L1→L0 variant to
improve on L2→L0, but the 0.1060-point mIoU gain required 83.5% more correction
work, 90.1% more correction latency, and 9.1% more bytes. Strict disjoint and
conditional re-entry spent approximately the same correction FLOPs as L2→L0
but lost 1.47 and 1.57 mIoU relative to it. The L2→L1 endpoint's low score was
not a broken correction implementation: a first-image 100%-support check
matched L1 within 0.0031 mIoU.

Detailed source reports:

- `analysis/results/dinov3_l2l1l0/ade20k_l1_l2l1_100/REPORT.md`
- `analysis/results/dinov3_l2l1l0/ade20k_threshold_only_100/REPORT.md`
- `analysis/results/dinov3_l2l1l0/ade20k_disjoint_crosslevel_100/REPORT.md`
- `analysis/results/dinov3_l2l1l0/ade20k_conditional_reentry_100/REPORT.md`

### NYUv2

The full-resolution comparator is the existing complete evaluation
`logs/offload/nyu_sequential_20260702_112512`.

| Input | AbsRel | RMSE | RMSE log | delta1 |
|---|---:|---:|---:|---:|
| Full resolution | 0.050130 | 0.193051 | 0.072483 | 0.978550 |
| L1 only | 0.050245 | 0.193650 | 0.072579 | 0.978283 |
| Relative degradation | +0.228% | +0.310% | +0.132% | -0.027 pp |

### COCO val2017

No exact sequential 5,000-image result exists in the local log archive. The
closest local full-validation comparator is the previously completed
full-arrival static AppCorr run
`logs/offload/coco_interleaved_static_20260708_114242`. It receives the
full-resolution residual but uses thresholded partial correction, so it is
reported as a local reference and not mislabeled as exact full inference.

| Input/method | AP | AP50 | AP75 | Delta AP |
|---|---:|---:|---:|---:|
| Full-arrival static AppCorr reference | 62.831 | 80.356 | 69.588 | - |
| L1 only | 59.317 | 76.531 | 65.717 | -3.514 |

The released DINOv3 paper reports 66.1 AP for the corresponding ViT-7B COCO
detector. Its official preprocessing/evaluation protocol is not identical to
the fixed 1024x1024 AppCorr runtime, so the L1-only difference to that number
(-6.783 AP) is contextual rather than a controlled local delta.

## Interpretation

- ImageNet classification, ADE20K segmentation, and NYUv2 depth retain most
  full-resolution accuracy from an L1 image. The full ADE20K validation gap is
  0.869 mIoU, larger than the 0.269-point first-100 estimate but still modest.
  On the paired 100-image subset, L1-only was more accurate than every tested
  L2-based correction policy. A strong L1 baseline therefore leaves limited
  task-metric headroom unless its full-resolution compute can also be reduced.
- COCO detection is different: L1-only loses 3.514 AP even against the
  thresholded full-arrival local reference. Fine spatial detail matters for
  localization, so COCO retains a meaningful reason to transmit and correct
  the L0 residual.
- Any claimed benefit of L2/L1/L0 AppCorr should therefore be compared against
  L1-only, not only against an L2 draft. For classification and depth, the
  system contribution must be justified primarily by transmission/overlap or
  by reducing L1 compute; top-line accuracy recovery alone is not enough.

## Commands

```bash
CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n appcorr \
  offload/run_local.sh offload/config/imnet_approx_only_l1.json \
  -nw 1 --set batch_size=32

CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n appcorr \
  offload/run_local.sh offload/config/ade20k_m2f_approx_only_l1.json \
  -nr 2000 -nw 1 --set device=cuda:0 \
  --set exp_id=ade20k_m2f_approx_only_l1_decomposed_full2000

CUDA_VISIBLE_DEVICES=0 conda run --no-capture-output -n appcorr \
  offload/run_local.sh offload/config/coco_approx_only_l1.json -nw 1

CUDA_VISIBLE_DEVICES=1 conda run --no-capture-output -n appcorr \
  offload/run_local.sh offload/config/nyu_approx_only_l1.json \
  -nw 1 --set device=cuda:0
```
