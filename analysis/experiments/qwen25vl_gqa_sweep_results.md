# Qwen2.5-VL GQA (testdev_balanced) keep-rate sweep

Methodology identical to the RealWorldQA sweep (`qwen25vl_keeprate_sweep_results.md`):
`grouping_strategy=top_energy`, `num_groups=1` (static single-shot correction, merge-groups ranked
by residual energy, top `keep_rate` fraction corrected to real resolution, rest permanently
approx/blurred). All runs nr=50 (strided sample of GQA's 12,578-question testdev_balanced split,
short free-form single-word/phrase answers, exact-match-after-normalization scoring). Driver:
`analysis/experiments/gqa_offload_eval.py`.

## Full curve

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
