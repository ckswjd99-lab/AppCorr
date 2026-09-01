# SAM 3 approx-then-correct: the whole result in one table

Six measurements, three benchmarks, two task heads, every one on a full evaluation set. Per-dataset
detail is in `sam3_coco_interleaved_results.md` (tracker), `sam3_coco_detector_results.md`,
`sam3_lvis_results.md` and `sam3_saco_gold_results.md`. This page is the summary and the caveats.

Setup throughout: SAM 3 at 1008x1008 (72x72 = 5184 tokens, 32 layers, global attention at 7/15/23/31),
bf16, approximate input = pyramid level 2, patch score = residual energy x average attention,
**55% of tokens recomputed**, interleaved `g=4 --bounds pre_global`.

## Result

Preservation is `arm / ceiling` — the fraction of the exact forward's score that survives. Recovery,
`(arm - floor) / (ceiling - floor)`, is secondary: it divides by the gap, and how wide that gap is
belongs to the dataset, not to the method.

| benchmark | metric | floor | **ours** | ceiling | **preservation** | recovery | gap |
|---|---|---:|---:|---:|---:|---:|---:|
| COCO tracker | mask AP | 0.5374 | 0.5961 | 0.6010 | **99.2%** | 92.3% | 10.6% |
| COCO detector | mask AP | 0.4332 | 0.5020 | 0.5092 | **98.6%** | 90.6% | 14.9% |
| LVIS detector | mask AP | 0.4121 | 0.5477 | 0.5638 | **97.1%** | 89.4% | 26.9% |
| SA-Co `crowded` | cgF1 | 0.5315 | 0.5762 | 0.5895 | **97.8%** | 77.2% | 9.8% |
| SA-Co `sa1b` | cgF1 | 0.5278 | 0.5376 | 0.5394 | **99.7%** | 84.2% | 2.2% |
| SA-Co `attributes` | cgF1 | 0.5396 | 0.5436 | 0.5421 | 100.3% | n/a | 0.5% |

**55% of tokens preserves 97-100% of the exact forward, at 0.60x the correction compute, while the
damage it has to repair varies 54x across these rows.** That insensitivity is the claim.

Recovery on the same runs ranges 77-92% and looks far more erratic than the method is — on
`attributes` it is undefined (a 0.5% gap yields "160%"), which is exactly why preservation leads.

## Interleaving is cheaper, not better

Correcting group `r` over depth `bounds[r]` costs `(1/g)·Σ bounds[r]` token-layers:
`g=1` 1.00x, `g=4 aligned` 0.63x, `g=4 pre_global` 0.60x, `g=16` 0.53x — all at identical coverage
(the union of rounds equals the one-shot token set exactly, gated).

One-shot vs `pre_global` preservation, all six rows: 99.2/99.2, 98.4/98.6, 97.3/97.1, 98.3/97.8,
99.7/99.7, 100.1/100.3. **Within 0.5pp everywhere.** Two earlier claims in the other direction were
withdrawn — "8.3% more faithful" was a leak of not-yet-arrived data, and a 5.3pp deficit on `crowded`
was recovery-rate magnification of a 0.0031 cgF1 difference.

`pre_global` earns its 0.60x by deferring SAM 3's **global** layers: rounds ending at 7/15/23 instead
of 8/16/24 put 7 global-layer corrections in the schedule instead of 10, and a corrected query on a
global layer attends over all 5184 K/V rather than its window's 576. Predicted 0.70, measured 0.69 of
`aligned`'s wall clock.

## What predicts the damage

**Only object size — specifically, how many pixels an object still spans in the L2 image.** Not
relative area, not rarity, not vocabulary size, not task type.

| | object side at L2 | approximation cost |
|---|---:|---:|
| SA-Co `crowded` | 10.2 px | 9.8% |
| SA-Co `sa1b` | 16.3 px | 2.2% |
| SA-Co `attributes` | 38.6 px | 0.5% |

`crowded` and `sa1b` have nearly identical *relative* object area (0.0028 vs 0.0029) and differ 4.5x
in cost, because SA-1B images are 2089x1500 against MetaCLIP's 816x640. Native resolution and area
fraction multiply.

Everything else is orthogonal. LVIS degrades 26.7/26.2/27.9% and recovers 90.3/89.1/89.3% across
rare/common/frequent — whatever makes a concept rare has nothing to do with whether its pixels
survive downsampling. Within a dataset the split by size is stark: COCO detector loses 41.7% of
small-object AP and 1.3% of large.

## Why partial recompute works at all

SA-Co's cgF1 factors as `positive_micro_F1 x IL_MCC` — mask quality times whether the concept is
recognised as present — so the two can be read apart, which AP cannot do.

| subset | mask quality cost | recognition cost |
|---|---:|---:|
| `crowded` | 8.8% | **1.1%** |
| `attributes` | 0.5% | **−0.1%** |

**The approximation barely touches recognition.** Deciding that "black tire" occurs in an image
survives an L2 downsample; outlining it does not. So the approximate pass has already settled *what
is there*, and correction only has to refine *where the edges are* — which is why 55% of the tokens
is enough, and why the number holds across a 54x range of damage.

## Caveats

- **Wall clock here is offline-driver time and excludes transmission**, which is what interleaving
  exists to overlap. The 0.60x is a compute claim; a latency claim has to come from the offload
  pipeline. Correction is also launch-bound in this driver (720 SDPA calls against one-shot's 288,
  from a serial 9-window Python loop), so the compute saving does not show up in the clock.
- **The detector path prompts only categories present in the GT.** Real open-vocabulary evaluation
  also asks about absent concepts; that is what SA-Co does. These absolutes must not be placed beside
  SAM 3's published COCO/LVIS numbers.
- **LVIS APr > APc > APf here** (0.6199 > 0.5777 > 0.5237), the reverse of the usual ordering, for the
  same reason: a rare category means being told the concept is present and finding its one instance.
- **SA-Co is reported per subset, never pooled.** Its domains differ 18x in median object area, so a
  single "SA-Co number" would say more about subset choice than about the method.
- **Keep ratio was not swept.** 55% is one point; whether it is the knee is unmeasured.

## Not measured

- Keep ratios other than 55%; `g` other than 1 and 4 on full sets.
- Serving-path latency with transmission overlap.
- SA-Co subsets `metaclip`, `wiki_common`, `fg_food`, `fg_sports_equipment`; SA-Co/Silver; SA-Co/VEval.
- The tracker head on LVIS or SA-Co.
