# Qwen2.5-VL GQA (testdev_balanced) keep-rate sweep

## ✅ v2 CORRECTED full-dataset (N=12578), 32B, -1pp crossing threshold -- FINAL for 32B

Full GQA testdev_balanced split (all 12,578 questions), fixed batched driver (commit `5c398d1`).
Baseline + a `top_energy` keep_rate sweep, extended dynamically once the planned points didn't pin
a precise crossing on the first pass:

| keep_rate | accuracy (N=12578) | gap vs baseline |
|---|---|---|
| baseline | 60.84% (7653) | -- |
| 0% (approx-only) | 55.16% (6938) | -5.68pp |
| 20% | 56.77% (7141) | -4.07pp |
| 30% | 57.69% (7256) | -3.15pp |
| 40% | 58.40% (7345) | -2.44pp |
| 50% | 58.83% (7400) | -2.01pp |
| **65%** | **59.52% (7487)** | **-1.32pp (still below threshold)** |
| **75%** | **60.14% (7564)** | **-0.70pp (clears -1pp threshold)** |
| 100% (noise floor) | 60.94% (7665) | +0.10pp |

**32B full-dataset -1pp crossing point: between 65% and 75%** (threshold = 59.84%; kr=65% is
-1.32pp, still below; kr=75% is -0.70pp, clears). Narrowed via 2 rounds of dynamic binary-search
extension (0.50->0.75 probe, then 0.75->0.65 to bisect) to a final 10-point-wide bracket.

**vs nr=400 estimate (~50%, under the -1pp definition):** the full-dataset crossing (65-75%) is
**~15-25pp higher** than the nr=400 estimate -- a real, moderate upward revision (nr=400
underestimated how much correction 32B needs on GQA), consistent in direction and magnitude with
RefCOCO's revision (~18pp). Nowhere near the wildly-wrong "<=5%" the RETRACTED buggy v1 full-dataset
sweep produced -- nr=400 itself was never that unreliable, the v1 canvas-reconstruction bug was the
actual problem.

**kr=100% noise floor: +0.10pp** -- tighter than RefCOCO's (+0.32pp), consistent with GQA's mostly
single-word/short-phrase answers being less sensitive to the residual bf16/kernel-path noise that
affects RefCOCO's multi-token bbox-coordinate answers more. Gaps smaller than this floor at other
keep_rates should not be read as "correction beats baseline." See the statistical significance
subsection below for the formal paired test.

## Statistical significance (McNemar), full dataset

Baseline and the two edge conditions (kr=75%, kr=100%) were re-run with `--log-jsonl` over the same
12,578 samples so `analysis/experiments/mcnemar_from_jsonl.py` could run a paired McNemar test
(re-measured values match the table above exactly: baseline 60.84%, kr=75% 60.14%, kr=100% 60.94%).

| comparison | gap | discordant pairs (A-only / B-only) | McNemar p-value | significant? |
|---|---|---|---|---|
| baseline vs kr=75% (clears the -1pp aggregate threshold) | -0.71pp | 443 / 354 | **0.0018** | **YES** |
| baseline vs kr=100% (noise floor control) | +0.10pp | 209 / 221 | 0.5958 | No |

**Same pattern as RefCOCO, confirming it's not a one-off.** kr=100% (architecturally equivalent to
baseline) is correctly NOT significant (p=0.60) -- validates it as the noise floor. But kr=75%, the
point whose aggregate gap already clears the -1pp engineering threshold, is nonetheless
**statistically significant vs baseline (p=0.0018)** -- at N=12578 the test easily detects a real,
if small, accuracy cost. **The -1pp aggregate threshold is an engineering tolerance, not a
statistical-equivalence claim, on GQA just as on RefCOCO.** A "recovers baseline" claim in a paper
should either test a keep_rate well above 75% (kr=100% ties statistically; the exact keep_rate where
significance first disappears wasn't pinned down here) or explicitly caveat that ~75% is an
aggregate-tolerance threshold.

## ❌ RETRACTED: first full-dataset (N=12578) batched sweep -- ALL keep_rate rows INVALID (full-resolution leak)

Same critical bug as RefCOCO's retracted section (full diagnosis in
`qwen25vl_refcoco_sweep_results.md`): the first version of `refcoco_gqa_batched_eval.py` fed the
raw full-resolution image to `preprocess` instead of the patch-reconstructed canvas, so every
keep_rate/approx-only condition ran on full-res pixels. GQA's smoking gun was the starkest: buggy
approx-only scored IDENTICAL to baseline on the same 48 samples (64.58% == 64.58%, same 31/48
count); after the fix it correctly drops to 60.42% (-4.2pp). The fixed script was validated
per-sample against the offload pipeline driver (8/8 predictions character-identical).

Retracted numbers, kept only as a record (they measure fork-vs-stock numerical noise at full
resolution, not keep_rate behavior): baseline 60.84% (VALID, unaffected); "kr=5%" 61.09%;
"kr=10%" 60.92%; "kr=20%" 60.95%; "kr=30%" 61.01%. The "crossing <=5%" conclusion is WITHDRAWN.
A corrected full-dataset sweep (approx-only + keep_rates + kr=100% control) is being re-run with
the fixed script; results will be added here when complete.

Methodology identical to the RealWorldQA sweep (`qwen25vl_keeprate_sweep_results.md`):
`grouping_strategy=top_energy`, `num_groups=1` (static single-shot correction, merge-groups ranked
by residual energy, top `keep_rate` fraction corrected to real resolution, rest permanently
approx/blurred). All runs nr=50 (strided sample of GQA's 12,578-question testdev_balanced split,
short free-form single-word/phrase answers, exact-match-after-normalization scoring). Driver:
`analysis/experiments/gqa_offload_eval.py`.

## ✅ DEFINITIVE (mechanism-matched baseline, commit 310c65a)

Baseline (`full_inference`) was re-measured after fixing it to use the same two-stage decode
mechanism as every keep_rate condition (full diagnosis in `qwen25vl_refcoco_sweep_results.md`).
GQA's answers are mostly single words/short phrases, so -- like RealWorldQA, unlike RefCOCO -- the
fix barely moves anything:

| model | old baseline (confounded) | new baseline (matched) | shift |
|---|---|---|---|
| 32B | 60.50% (242/400) | **60.50% (242/400)** | +0.00pp (identical) |
| 72B | 59.25% (237/400) | **59.00% (236/400)** | -0.25pp (1 sample) |

**Crossing points are UNCHANGED**: 32B still does not cross by 80% (still -0.50pp there, baseline
identical) -- true crossing point remains bracketed to **(80%, 100%]**. 72B still crosses at
**80%**, now with +0.75pp margin instead of +0.50pp (baseline moved down slightly, making the
crossing marginally more comfortable, not less). Recomputed table:

| keep_rate | 32B accuracy | gap (matched) | gap (old, confounded) | 72B accuracy | gap (matched) | gap (old, confounded) |
|---|---|---|---|---|---|---|
| 15% | 57.25% | -3.25pp | -3.25pp | 56.00% | -3.00pp | -3.25pp |
| 50% | 59.75% | -0.75pp | -0.75pp | 57.50% | -1.50pp | -1.75pp |
| 65% | 60.00% | -0.50pp | -0.50pp | 58.00% | -1.00pp | -1.25pp |
| **80%** | 60.00% | -0.50pp | -0.50pp | **59.75%** | **+0.75pp** | +0.50pp |
| 100% | 61.25% | +0.75pp | +0.75pp | 60.75% | +1.75pp | +1.50pp |

**GQA's crossing points survive the fix unchanged, same as RealWorldQA and unlike RefCOCO** --
confirms the earlier hypothesis that this confound matters far more for RefCOCO's multi-token bbox
answers than for short-answer VQA tasks. The "72B needs LESS correction than 32B on GQA" exception
(discussed below) is confirmed to be real, not a mechanism artifact.

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
