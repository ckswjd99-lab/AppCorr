# SAM 3 detector path on COCO: approx-then-correct

Companion to `sam3_coco_interleaved_results.md`, which covers the tracker path. Same images, same
arms, same 55% of tokens recomputed — the only change is which head consumes the vision features.

`Sam3Model` prompted with the **category name as text**, one forward per category present in the
image. Full COCO val2017, 4952 images / 34,104 annotations, 1008x1008, bf16, single B200.

## What this path measures, and what the tracker path measures

They are not two views of one number.

| | tracker (`Sam3TrackerModel`) | detector (`Sam3Model`) |
|---|---|---|
| prompt | ground-truth **box** | category name as **text** |
| output | 3 masks per box, model picks one | 200 DETR queries, ranked by logit |
| predictions on full COCO | 34,104 — exactly one per annotation | 352,280 |

The tracker path emits one prediction per GT annotation with the GT's own category, so **recall is 1
by construction**: nothing is missed, nothing is invented, and AP moves only with mask IoU. It
isolates *outlining*.

The detector path folds three failures into one number — objects not found, objects invented, and
objects outlined badly — so a drop here does not say which got worse. Note also that prediction count
is protocol-determined (top-30 per prompt, capped at 100 per image) and identical across arms, so
approximation damage shows up as **correct predictions ranked lower**, never as fewer predictions.

**Simplification worth naming:** only categories present in the GT are prompted. Real open-vocabulary
evaluation also asks about absent concepts and scores the model's ability to answer "none" — that is
what SA-Co's cgF1 does. This setup is easier, is applied identically to every arm (so arm-to-arm
comparison holds), and its absolute value must not be placed next to SAM 3's published SA-Co numbers.

## Results

| arm | mask AP | AP50 | AP75 | small | medium | large | recovery | correction cost |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| floor | 0.4332 | 0.6822 | 0.4607 | 0.2019 | 0.4892 | 0.6709 | 0% | — |
| one-shot, 55% | 0.5010 | 0.7614 | 0.5429 | 0.3364 | 0.5412 | 0.6722 | 89.2% | 1.00x |
| interleaved g=4, `aligned` | 0.5010 | 0.7624 | 0.5436 | 0.3359 | 0.5402 | 0.6740 | 89.2% | 0.63x |
| **interleaved g=4, `pre_global`** | **0.5020** | 0.7632 | 0.5452 | 0.3377 | 0.5412 | 0.6745 | **90.5%** | **0.60x** |
| ceiling | 0.5092 | 0.7694 | 0.5544 | 0.3462 | 0.5505 | 0.6794 | 100% | — |

Gap is 0.0761 AP. **`pre_global` beats one-shot by 1.3pp of recovery at 0.60x the correction
compute** — the same direction the tracker path showed in feature space (0.4642 vs 0.4664 rel-L2),
here large enough to see in the task metric.

## Detection is harder to approximate than outlining

| | tracker | detector |
|---|---:|---:|
| floor→ceiling gap | 6.35pp (10.6% of ceiling) | **7.61pp (14.9%)** |
| small-object loss | −0.1207 (26.7%) | **−0.1443 (41.7%)** |
| AP50 loss | −0.0449 (5.1%) | **−0.0872 (11.3%)** |
| AP75 loss | −0.0833 (12.6%) | −0.0937 (16.9%) |

Both tasks lose almost everything on small objects and almost nothing on large ones — an L2 pyramid
discards high-frequency detail, and large masks barely notice (0.7–1.3%). But the detector loses
**1.4x more**, and the split between AP50 and AP75 says why. The tracker's damage is concentrated at
the strict threshold (AP75 loss is 2.5x its AP50 loss): objects are still roughly located, only the
precise boundary degrades. The detector loses 11.3% at AP50 too, where boundary precision hardly
matters — those are predictions that fell out of the ranking entirely.

That is the expected asymmetry: a blurred small object degrades a tracker mask gracefully, but drops
a detector's classification logit below its competitors and takes the prediction with it. Detection
has a sharper failure threshold because there is no partial credit for finding something.

**Correction repairs exactly what the approximation broke**: small-object AP recovers 93.2% of its
gap, and AP50 recovers 90.9% against AP75's 87.7% — the lost detections come back before the precise
boundaries do. 55% of tokens is enough for both tasks; the 3.2pp lower total recovery (89.2% vs
92.4%) does not justify a different keep ratio for detection.

## Readout protocol

`--det-per-cat 30 --det-max-dets 100`, chosen by sweeping the **ceiling** arm to convergence on 200
images (`logs/sam3_det_protocol`):

| per_cat | 1 | 5 | 10 | 30 | 100 |
|---|---:|---:|---:|---:|---:|
| mask AP | 0.4044 | 0.5418 | 0.5570 | **0.5631** | 0.5635 |
| predictions | 587 | 2,935 | 5,870 | 14,440 | 20,000 (saturated) |

AP is a ranking metric — COCOeval sorts by score and sweeps the threshold itself — so the previous
hand-set `--det-score-thresh 0.3` could only discard recall the metric would have used. It scores
0.5535 on the same slice. Tuning on the ceiling is deliberate: the exact forward gets its best score,
so no approximation is read against a handicapped reference. The setting is then frozen for all arms.

## Caveats

- 200-image slices are optimistic by 5.4pp on this path (0.5631 vs 0.5092 at full scale). Only the
  full-set numbers above are quotable.
- Wall clock here is offline-driver time and excludes transmission, which is what interleaving exists
  to overlap. It is also not comparable to the tracker path's: this one ran the vision tower once per
  category prompt until that was hoisted out of the loop (see the commit following these numbers) —
  3.5 redundant passes per image on COCO.
