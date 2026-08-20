# SAM 3 on LVIS: does a harder dataset need a higher keep ratio?

Third dataset after COCO tracker and COCO detector, and the one that tests whether the method holds
when the approximation hurts much more. Detector path (`Sam3Model`, category name as text prompt),
**full LVIS v1 val: 19,626 images, 244,639 annotations, 1,203 categories**, 1008x1008, bf16.

Scored with **LVISEval**, not COCOeval. LVIS annotations are federated — each image is exhaustively
labelled for only some categories — so COCOeval would count correct detections of un-annotated
objects as false positives. LVISEval also splits AP by frequency bucket (rare / common / frequent),
which is the axis this dataset exists for.

**No `--max-boxes` cap.** The 20-box limit carried over from COCO's tracker path discards 38% of the
rare annotations (1,200 → 745) by truncating each image's annotation list — precisely the long tail
being measured. Subsetting was rejected for the same reason: 5,000 images keeps only 38% of rare
categories and 21% of rare annotations, leaving APr too noisy to separate arms.

Produced by `scripts/sam3_lvis_full.sh`.

## Results

| metric | floor | one-shot | `pre_global` | ceiling | one-shot recovery | `pre_global` recovery |
|---|---:|---:|---:|---:|---:|---:|
| **AP** | 0.4121 | 0.5484 | 0.5477 | 0.5638 | **89.8%** | **89.4%** |
| AP50 | 0.6116 | 0.7487 | 0.7455 | 0.7571 | 94.3% | 92.1% |
| AP75 | 0.4366 | 0.5886 | 0.5889 | 0.6068 | 89.3% | 89.5% |
| APr (rare) | 0.4545 | 0.6072 | 0.6038 | 0.6199 | 92.3% | 90.3% |
| APc (common) | 0.4265 | 0.5598 | 0.5612 | 0.5777 | 88.2% | 89.1% |
| APf (frequent) | 0.3775 | 0.5097 | 0.5080 | 0.5237 | 90.5% | 89.3% |
| small | 0.2094 | 0.4003 | 0.3987 | 0.4144 | 93.1% | 92.3% |
| medium | 0.5636 | 0.6871 | 0.6870 | 0.7056 | 87.0% | 86.9% |
| large | 0.7386 | 0.7781 | 0.7789 | 0.7873 | 81.1% | 82.8% |

Correction cost: one-shot 1.00x, `pre_global` 0.60x. Gap is 0.1517 AP. **Preservation**
(`arm / ceiling`, the figure to lead with) is 73.1% at the floor, **97.3%** for one-shot and
**97.1%** for `pre_global` -- the widest gap of any benchmark here, and still 97% preserved. See
`sam3_summary.md`.

## The result: a 2.4x wider gap, and recovery holds

| | floor→ceiling gap | recovery at 55% |
|---|---:|---:|
| COCO tracker | 6.35pp (10.6% of ceiling) | 92.3% |
| COCO detector | 7.61pp (14.9%) | 90.5% |
| **LVIS detector** | **15.17pp (26.9%)** | **89.4%** |

The approximation costs 2.4x more here than on COCO tracker, and correction still returns ~90% of it
from the same 55% of tokens. **A harder dataset does not need a higher keep ratio** — a 3pp spread in
recovery across a 2.4x spread in damage is not a reason to change the knob.

**LVIS is not "harder" in a new way — it has more small objects.** Relative degradation by size is
49.5% / 20.1% / 6.2% (small / medium / large), against COCO detector's 41.7% / 11.1% / 1.3%. An L2
pyramid discards high-frequency detail, and for a small object that detail *is* the object. LVIS's
1,203 categories include `almond`, `bead`, `wristband`; COCO's 80 do not. The vulnerability is the
same one, exposed more often.

**Concept rarity is orthogonal to both damage and repair.** Degradation is 26.7% / 26.2% / 27.9%
across rare / common / frequent, and recovery is 90.3% / 89.1% / 89.3%. Whatever makes a concept rare
is unrelated to whether its pixels survive downsampling. The method is insensitive to vocabulary size
and to the long tail.

## Retraction: `pre_global` is not more accurate than one-shot

Across all three measurements the sign is inconsistent:

| | one-shot | `pre_global` | delta |
|---|---:|---:|---:|
| COCO tracker | 92.4% | 92.3% | −0.1pp |
| COCO detector | 89.2% | 90.5% | **+1.3pp** |
| LVIS detector | 89.8% | 89.4% | −0.4pp |

An earlier note called the COCO detector edge non-coincidental because it agreed in direction with
the tracker path's feature-space L2. LVIS reverses it. Within ±1pp these are a tie, and the claim to
make is **equal accuracy at 0.60x the correction compute** — which is the more useful claim anyway.

## An artifact of this harness, stated so it is not misread

APr > APc > APf (0.6199 > 0.5777 > 0.5237 at the ceiling) — the *opposite* of the usual long-tail
ordering. That is this setup, not the model: only categories present in the GT are prompted, so a
rare category means being handed "this concept is here" and finding its one or two instances, while a
frequent category (`person`, `car`) means finding all several dozen and losing recall for each miss.

So this APr does **not** measure open-vocabulary rare-concept detection, and none of these absolutes
belong next to SAM 3's published LVIS numbers, which prompt the full 1,203-word vocabulary. Every arm
shares the setup, so arm-to-arm comparison — the thing being measured — is unaffected.

## Operational note

The `lvis` package (0.5.3, 2020) calls `np.float`, removed in numpy 1.24, and killed a completed
19,626-image ceiling arm at the accumulate step — 56 minutes of GPU lost because predictions only
existed in memory. Patched in place (`site-packages/lvis/eval.py`, original kept as `.orig`), and the
driver now dumps predictions **before** scoring with `--score-only` to re-score. Third-party
evaluators fail for reasons unrelated to the run; that risk should not be paid in GPU hours.
