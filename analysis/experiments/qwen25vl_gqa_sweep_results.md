# Qwen2.5-VL GQA (testdev_balanced) keep-rate sweep

Methodology identical to the RealWorldQA sweep (`qwen25vl_keeprate_sweep_results.md`):
`grouping_strategy=top_energy`, `num_groups=1` (static single-shot correction, merge-groups ranked
by residual energy, top `keep_rate` fraction corrected to real resolution, rest permanently
approx/blurred). All runs nr=50 (strided sample of GQA's 12,578-question testdev_balanced split,
short free-form single-word/phrase answers, exact-match-after-normalization scoring). Driver:
`analysis/experiments/gqa_offload_eval.py`.

## ⚠ REVISED at nr=400: baseline was UNDERestimated (opposite direction from RealWorldQA), and both models show a real, clean elbow

A larger re-measurement (nr=400, 8x the original nr=50, at narrowed candidates baseline/15/50/100%)
was run after RealWorldQA's nr=50 sweep turned out unreliable. Two findings here:

| keep_rate | 32B accuracy (nr=400) | gap to baseline | 72B accuracy (nr=400) | gap to baseline |
|---|---|---|---|---|
| baseline | 60.50% (242/400) | -- | 59.25% (237/400) | -- |
| 15% | 57.25% (229/400) | -3.25pp | 56.00% (224/400) | -3.25pp |
| 50% | 59.75% (239/400) | -0.75pp | 57.50% (230/400) | -1.75pp |
| 65% | 60.00% (240/400) | -0.50pp | 58.00% (232/400) | -1.25pp |
| **80%** | 60.00% (240/400) | -0.50pp | **59.75% (239/400)** | **+0.50pp** |
| 100% | 61.25% (245/400) | +0.75pp | 60.75% (243/400) | +1.50pp |

**Precise crossing points (dense follow-up sweep, added after the initial narrowed-candidate run):**
- **72B crosses at keep_rate=80%** (+0.50pp) -- first point at or above baseline.
- **32B does NOT cross by 80%** (still -0.50pp there, identical to its 65% value) -- its true crossing
  point lies somewhere in **(80%, 100%]**; this sweep did not test a point between 80% and 100% so
  the exact location isn't pinned down further. Reported honestly as a bracket, not a false-precision
  single number.

**This is a genuine, honest EXCEPTION to the pattern found on RealWorldQA and RefCOCO.** On both of
those datasets, 72B needed a *higher* keep_rate than 32B to reach its own baseline (RealWorldQA:
100% vs 50%; RefCOCO: 70% vs 30%). On GQA, the relationship flips: at keep_rate=80%, 72B has already
crossed (+0.50pp) while 32B has not (-0.50pp) -- so on GQA specifically, 72B needs *less or equal*
correction relative to 32B, not more. See `QWEN25VL_APPCORR_LOG.md` section 7 for the full
cross-dataset reconciliation of this exception against the other two datasets' pattern.

**1. nr=50 UNDERestimated GQA baseline accuracy, the opposite direction from RealWorldQA (which
OVERestimated it by ~4-5pp).** nr=50 put both models' baseline at exactly 50%; nr=400 puts them at
60.50% (32B) and 59.25% (72B) -- a **~9-10.5pp underestimate**. This is an important cross-dataset
lesson: nr=50's unreliability is not a single consistent bias (e.g. "always optimistic") -- it can
swing in either direction depending on which 50 examples happen to get sampled from a given
dataset's stride pattern.

**2. Both models now show a real, clean, monotonic elbow that nr=50's noise had obscured.** nr=50's
32B curve was erratic (44/46/42/38/44/46/48% across keep_rate, no visible trend, all points below
its 50% baseline) and was honestly reported as "no clean elbow identifiable." At nr=400, the same
model shows a sensible climb: -3.25pp (15%) -> -0.75pp (50%) -> +0.75pp (100%) -- the underlying
signal was real, nr=50 was just too small a sample to see it clearly. 72B shows the same shape
(-3.25pp -> -1.75pp -> +1.50pp), and -- like RefCOCO's 72B revision -- reveals a real gap at low
keep_rate (15%: -3.25pp) that nr=50 did not show (nr=50 had 72B essentially flat/at-baseline from
2% onward). So GQA's revision cuts against the earlier "72B is simply robust regardless of task"
framing in the same direction as RefCOCO's revision did: 72B's gaps are real but smaller than 32B's
gaps at the same keep_rate, not zero.

## Full curve (nr=50, ORIGINAL -- see revision above for the more reliable nr=400 numbers)

| keep_rate | 32B accuracy | 72B accuracy |
|-----------|-------------|-------------|
| 2%   | 44% (22/50) | 48% (24/50) |
| 5%   | 46% (23/50) | 48% (24/50) |
| 10%  | 42% (21/50) | 50% (25/50) |
| 15%  | 38% (19/50) | 50% (25/50) |
| 25%  | 44% (22/50) | 50% (25/50) |
| 50%  | 46% (23/50) | 50% (25/50) |
| 100% | 48% (24/50) | 52% (26/50) |
| baseline (full-res, no AppCorr) | 50% (25/50) | 50% (25/50) |

## Interpretation

**72B**: flat and stable at/above baseline across the *entire* curve, from keep_rate=2% onward
(48-52%, baseline 50%) -- essentially the same pattern seen on RealWorldQA and RefCOCO: this model
size tolerates aggressive blurring on GQA with no measurable cost.

**32B**: notably different from its own RealWorldQA behavior. Every single keep_rate point sits
*below* baseline (38-48% vs baseline 50%), including keep_rate=100% (48%, still 2pp under
baseline) -- unlike RealWorldQA, where 32B's curve reliably reached or exceeded baseline by
keep_rate=15-25%. There is no clean monotonic climb; the curve is noisy (44/46/42/38/44/46/48%)
rather than the clear saturating shape seen on RealWorldQA and RefCOCO's 32B curves.

**Is this a real effect or nr=50 noise?** Calibration check: on RealWorldQA, 32B's baseline
accuracy was measured multiple times across this session (74% nr=50, consistently) and its
100%-keep_rate/single-round-correction result also independently landed at 74-76% -- a tight,
reproducible cluster. Here, even 32B's own **100%-keep_rate point (48%) doesn't recover its own
baseline (50%)**, a 2pp gap that is well within one-sample nr=50 noise (2pp = exactly 1/50) --
that specific gap is not meaningful. But the fact that literally every point in the 2-100% sweep
sits at or below baseline, with a 10pp range (38-48%) across points that should be doing
increasingly *more* correction, is a different, less easily dismissed pattern than RealWorldQA's:
there, the *low* end of the curve (2-10%) was clearly worse and the *high* end clearly recovered.
Here, there's no such clean separation between low and high keep_rate -- 15% (worst, 38%) sits
between 10% (42%) and 25% (44%), not a smooth trend. This reads as a genuinely noisier signal for
32B on GQA specifically, plausibly because GQA's answers are much shorter (often single words) and
its scoring is exact-match rather than a numeric/letter extraction with more tolerance -- a single
token-level slip (e.g. "front" vs "in front", already handled by the substring-match fallback, but
still less forgiving than RealWorldQA's looser scoring) has more leverage over the aggregate score
at n=50. **Conclusion: no clean elbow is identifiable for 32B on GQA from this data** -- the
honest read is "32B's GQA accuracy hovers noisily a few points under baseline across the whole
tested range, this dataset+model combination needs a larger sample size (or its own dedicated noise
calibration) to say anything more precise than that."

**72B's result, by contrast, is clean**: baseline-matching from keep_rate=2% onward, consistent
with its behavior on RealWorldQA (baseline-matching from keep_rate≈2-15%) and RefCOCO (see
`qwen25vl_refcoco_sweep_results.md`). The larger model appears to be reliably robust to
keep-rate-based blurring across all three tested tasks; the smaller model's behavior is
task-dependent and, on GQA specifically, too noisy at nr=50 to characterize with confidence.
