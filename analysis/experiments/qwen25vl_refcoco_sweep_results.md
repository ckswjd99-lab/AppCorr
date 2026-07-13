# Qwen2.5-VL RefCOCO (val) keep-rate sweep

## ✅ v2 CORRECTED full-dataset (N=8811), 32B, -1pp crossing threshold -- FINAL for 32B

Full RefCOCO val split (all 8811 samples), fixed batched driver (commit `5c398d1`, canvas-reconstruction
bug fixed and validated per-sample against the offload pipeline: 8/8 predictions character-identical
at kr=0.30). Baseline + a `top_energy` keep_rate sweep, extended dynamically (binary-search style)
once the planned points didn't land a precise crossing on the first pass:

| keep_rate | Acc@0.5 (N=8811) | gap vs baseline | mean IoU |
|---|---|---|---|
| baseline | 85.75% (7555) | -- | 0.7620 |
| 0% (approx-only) | 74.76% (6587) | -10.99pp | 0.6502 |
| 30% | 80.75% (7115) | -5.00pp | 0.7080 |
| 35% | 81.07% (7143) | -4.68pp | 0.7153 |
| 40% | 82.24% (7246) | -3.51pp | 0.7235 |
| 50% | 83.49% (7356) | -2.26pp | 0.7381 |
| **58%** | **84.75% (7467)** | **-1.00pp (exactly at threshold)** | 0.7495 |
| 65% | 85.75% (7555) | 0.00pp (exact tie w/ baseline) | 0.7577 |
| 100% (noise floor) | 86.07% (7584) | +0.32pp | 0.7654 |

**32B full-dataset -1pp crossing point: ~58%** (precisely bracketed: 50% is -2.26pp below threshold,
58% lands essentially exactly on the -1pp boundary (84.75% == 84.75%), 65% already exceeds baseline
outright). This is the most precise crossing measurement in this investigation -- pinned via 2 rounds
of dynamic binary-search extension (58% then would-have-been-further-narrowed, but 58% landed close
enough to the exact boundary that no third point was needed).

**vs nr=400 estimate (~40%, under the -1pp definition):** the full-dataset crossing (~58%) is
**~18pp higher** than the nr=400 estimate -- a real, moderate upward revision (nr=400 somewhat
underestimated how much correction 32B needs), but nowhere near the wildly-wrong "<=10%" the
RETRACTED buggy v1 full-dataset sweep produced (see below). This is reassuring: nr=400 itself was
never that unreliable on this dataset -- the v1 full-dataset bug was the actual problem, not
small-sample noise.

**kr=100% noise floor: +0.32pp.** This is the pipeline's own fork-vs-stock numerical noise floor at
full-dataset scale (kr=100% is architecturally equivalent to baseline, any nonzero gap is bf16/kernel
noise, not signal) -- gaps smaller than this at other keep_rates should not be read as "correction
beats baseline," only as "within measurement noise." See the statistical significance subsection
below for the formal paired test.

## Statistical significance (McNemar), full dataset

Aggregate-accuracy gaps alone can't distinguish "real small effect" from "noise" -- both baseline and
each keep_rate condition were re-run with `--log-jsonl` so `analysis/experiments/mcnemar_from_jsonl.py`
could run a paired McNemar test over the SAME 8811 samples (re-measured values match the table above
exactly: baseline 85.75%, kr=58% 84.75%, kr=100% 86.07%).

| comparison | gap | discordant pairs (A-only / B-only) | McNemar p-value | significant? |
|---|---|---|---|---|
| baseline vs kr=58% (the -1pp crossing point) | -1.00pp | 592 / 504 | **0.0086** | **YES** |
| baseline vs kr=100% (noise floor control) | +0.33pp | 146 / 175 | 0.1181 | No |

**This is an important, honest correction to the earlier framing.** The kr=100% control confirms
what was expected: architecturally-equivalent-to-baseline computation produces a gap that is NOT
statistically significant (p=0.12) -- this validates using it as "the noise floor." But **kr=58%,
the point whose AGGREGATE gap exactly equals the -1pp threshold, is actually a statistically
SIGNIFICANT difference from baseline (p=0.0086)**, not noise. At N=8811 the test has enough power to
detect a real (if small) accuracy cost even when the aggregate delta looks noise-sized. **Practical
implication: the -1pp aggregate threshold is a reasonable ENGINEERING tolerance, but it should not be
read as "statistically indistinguishable from baseline" -- that stronger claim would require a
keep_rate closer to 65% (which ties baseline exactly in this sweep, 7555/7555, and would very likely
be non-significant if re-measured with logging, though that specific pair wasn't re-run with
`--log-jsonl` here).** Anyone citing "32B recovers RefCOCO baseline accuracy at ~58% keep_rate" in a
paper should either use kr~65% instead, or explicitly caveat that ~58% is an engineering
(aggregate-tolerance) threshold, not a statistical-equivalence one.

## ❌ RETRACTED: first full-dataset (N=8811) batched sweep -- ALL keep_rate rows INVALID (full-resolution leak)

The user flagged the results below as counter-intuitive ("even 5-10% keep_rate beats baseline?
suspicious") and asked for an approx-only control. That control exposed a critical bug in the
FIRST version of `refcoco_gqa_batched_eval.py`: the keep_rate path passed the RAW FULL-RESOLUTION
image to `executor.preprocess` for every group, instead of the canvas reconstructed from the
transmitted patches (blurred base + arrived corrections) the way the offload pipeline's
WorkerModule does (`worker.py:177-188`, `policy.decode(patch_buffer, ...)`). Consequence:
`pixel_values` -- feeding both the vision tower AND the `generate()` fallback -- was full
resolution regardless of keep_rate, so every keep_rate/approx-only condition silently ran as
~baseline + fork numerical noise. Smoking gun (same 48 strided samples, buggy script): approx-only
(zero correction, fully blurred) scored within 1 sample of baseline on RefCOCO (70.83% vs 72.92%)
and IDENTICAL to baseline on GQA (64.58% == 64.58%) -- impossible if the blur were actually
reaching the model. After the fix, approx-only correctly drops to 66.67% / mean IoU 0.5763 (-6.25pp
/ -0.10 IoU vs same-sample baseline). The fixed script was then validated per-sample against the
offload pipeline driver (kr=0.30, 8 samples): all 8 predictions character-identical, same IoU/acc.

The retracted numbers, kept ONLY as a record of the bug's magnitude (they measure the fork's
numerical noise at full-resolution, NOT keep_rate behavior): baseline 85.75% (VALID -- baseline is
genuinely full-res stock computation and is unaffected); "kr=10%" 86.23%; "kr=20%" 86.14%;
"kr=30%" 86.10%; "kr=35%" 86.15%. The "crossing <=10%" conclusion previously drawn from these is
WITHDRAWN. (Incidentally these rows DO empirically confirm the ~+0.4pp full-dataset noise floor of
the fork-vs-stock computation path at N=8811 -- consistent with, and much tighter than, the ~2.5pp
nr=400 estimate.) A corrected full-dataset sweep (approx-only + keep_rates + kr=100% control) is
being re-run with the fixed script; results will be added here when complete.

## ✅ DEFINITIVE (mechanism-matched baseline, commit 310c65a) -- supersedes every section below

The two-stage-decode confound described in the next section was **fixed at the source**, not just
documented: baseline (`full_inference`) now uses the identical two-stage decode mechanism as every
keep_rate condition (`head_inference`). Baseline was re-measured under the fix (nr=400):

| model | old baseline (confounded) | new baseline (matched) | shift |
|---|---|---|---|
| 32B | 83.75% (335/400) | **86.00% (344/400)** | +2.25pp |
| 72B | 92.25% (369/400) | **91.25% (365/400)** | -1.00pp |

(These shifts exactly match the standalone `matched_decode` diagnostic's prediction -- a good
consistency check that the fix behaves as expected.)

**Recomputed gaps and crossing points** (keep_rate condition accuracies are unchanged from the
original sweep -- only baseline changed, so every gap below is recalculated against the new,
matched baseline):

| keep_rate | 32B Acc@0.5 | gap (new) | gap (old, confounded) | 72B Acc@0.5 | gap (new) | gap (old, confounded) |
|---|---|---|---|---|---|---|
| 25% | 79.75% | -6.25pp | -4.00pp | 90.25% | -1.00pp | -2.00pp |
| 30% | 84.50% | -1.50pp | +0.75pp | 89.75% | -1.50pp | -2.50pp |
| 35% | 84.75% | -1.25pp | +1.00pp | 89.25% | -2.00pp | -3.00pp |
| 40% | 85.25% | -0.75pp | +1.50pp | 88.75% | -2.50pp | -3.50pp |
| **50%** | **87.00%** | **+1.00pp** | +3.25pp | 88.75% | -2.50pp | -3.50pp |
| 60% | 87.50% | +1.50pp | +3.75pp | 90.50% | -0.75pp | -1.75pp |
| **70%** | 88.75% | +2.75pp | +5.00pp | **92.25%** | **+1.00pp** | +0.00pp |
| 80% | 89.25% | +3.25pp | +5.50pp | 92.25% | +1.00pp | +0.00pp |
| 90% | 89.50% | +3.50pp | +5.75pp | 92.00% | +0.75pp | -0.25pp |
| 100% | 88.50% | +2.50pp | +4.75pp | 91.50% | +0.25pp | -0.75pp |

**32B's crossing point MOVED: 30% → 50%.** This is a real, meaningful correction, not a rounding
change. Under the old (confounded) baseline, 30% already looked like it exceeded baseline
(+0.75pp) -- but that baseline was artificially *low* (83.75% instead of the true 86.00%), making
correction look like it had "already worked" one full sweep-step too early. Under the properly
matched baseline, 30/35/40% are all still *below* baseline (-1.50/-1.25/-0.75pp), and the crossing
doesn't happen until **50%** (+1.00pp).

**72B's crossing point did NOT move: still 70%.** The old baseline's confound (-1.00pp, i.e. the
old baseline was artificially *high*) happened to not change which tested point crosses first --
70% was an exact tie under the old baseline (+0.00pp) and is now comfortably positive (+1.00pp)
under the corrected, lower baseline. The *direction* of 72B's confound partially cancelled against
the correction pipeline's own tendency to score higher at high keep_rate, leaving the crossing
point coincidentally stable.

**Revised headline finding**: 32B's RefCOCO crossing point (50%) is now **identical** to its
RealWorldQA crossing point (also 50%, see `qwen25vl_keeprate_sweep_results.md`) -- the earlier
"RefCOCO needs less correction than RealWorldQA" claim for 32B **no longer holds**; they tie. For
72B, RefCOCO (70%) is still lower than RealWorldQA (100%), so the "less correction" finding
survives, but only for the larger model now, not both. See `QWEN25VL_APPCORR_LOG.md` section 7 for
the full, final cross-dataset table.

## ⚠ Methodological confound found and measured: the two-stage decode mechanism is NOT neutral
(historical record -- the fix and final numbers are in the section above)

The user asked directly: is "keep_rate=100% sometimes exceeds baseline" (see below, 32B: +4.75pp)
a genuine correction-pipeline effect, or an artifact of `head_inference` using a *different
generation mechanism* than the true baseline? `head_inference` decodes only the first token from
the (corrected) prefill hidden state, then falls back to a **separate** stock `model.generate()`
call on `[input_ids, first_token]`; the true baseline makes **one continuous** `model.generate()`
call for the whole answer. These are architecturally different call patterns even when the
underlying weights/computation are otherwise identical.

`analysis/experiments/refcoco_matched_decode_diagnostic.py` isolates this: it runs **both**
generation mechanisms using 100% identical, unforked stock computation (no correction fork
involved at all) on the same 400 RefCOCO samples/images/prompts, and compares them directly:

| model | true_baseline (1 continuous call) | matched_decode (2-stage, mirrors `head_inference`) | gap | exact-text agreement |
|---|---|---|---|---|
| 32B | 83.75% (335/400) | 86.00% (344/400) | **+2.25pp** | **64.8%** (259/400) |
| 72B | 92.25% (369/400) | 91.25% (365/400) | **-1.00pp** | **70.2%** (281/400) |

**Confirmed: the two-stage decode mechanism is a real, substantial confound, not a negligible
artifact.** Even under 100% identical stock computation (zero correction involved), the two
mechanisms produce *different generated text* on 30-35% of samples -- floating-point
non-associativity between "recompute the full prefix from scratch in a fresh `generate()` call"
(two-stage) and "one continuous incrementally-cached decode" (single call) is large enough to
flip a meaningful fraction of RefCOCO's multi-token bbox-coordinate answers.

**What this means for the "100% keep_rate exceeds baseline" numbers reported below:**
- **72B**: the mechanism confound (-1.00pp) is close to the *entire* originally-observed
  keep_rate=100% gap (-0.75pp, see table below) -- for 72B, this artifact plausibly explains
  essentially all of it. Little to no genuine correction-pipeline effect need be invoked.
- **32B**: the mechanism confound (+2.25pp) explains **roughly half** of the originally-observed
  keep_rate=100% gap (+4.75pp) -- the remaining ~+2.5pp is not accounted for by this mechanism and
  is more likely genuine numerical divergence in the correction pipeline itself (the bf16/SDPA-
  kernel-dispatch noise documented earlier in `QWEN25VL_APPCORR_LOG.md`), though this residual was
  not independently re-verified after isolating the mechanism effect.

**Practical implication**: all "keep_rate=X vs baseline" gaps in this file (and the RealWorldQA/GQA
sweep files, which use the same `head_inference` mechanism) carry an un-removed ±1-2pp mechanism-
level noise floor from this confound, on top of whatever genuine correction-quality signal exists.
Crossing points and large gaps (several pp or more) are still meaningful; gaps of ~1-2pp or less
should be read as within this noise floor rather than as precise measurements.

## Original sweep (methodology, kept for the record)

Methodology identical to the RealWorldQA sweep: `grouping_strategy=top_energy`, `num_groups=1`
(static single-shot correction). All runs nr=50 (strided sample of RefCOCO val's 8811 examples).
Task: referring-expression comprehension (text -> bounding box), scored Acc@0.5 (IoU >= 0.5) plus
mean IoU as a more continuous signal. Driver: `analysis/experiments/refcoco_offload_eval.py`. See
that file's docstring for the input-direction note (dataset's own template is captioning-direction;
this driver uses it in the standard grounding direction, `answer[0]` as the referring expression).

## ⚠ REVISED at nr=400 (SUPERSEDED -- see "DEFINITIVE" section at top; this section's crossing
## points, 32B=30%/72B=70%, used the mechanism-confounded baseline. 32B's is now known wrong;
## the correct value is 50%. Kept for the record, not as current findings.)

A larger re-measurement (nr=400, 8x the original nr=50, at narrowed candidates baseline/25/40/50%)
was run after RealWorldQA's own nr=50 sweep turned out to be unreliable (see
`qwen25vl_keeprate_sweep_results.md`'s revision section). Applying the same scrutiny here changes
the picture in **two different ways for the two models**:

| keep_rate | 32B Acc@0.5 (nr=400) | gap to baseline | 32B mean IoU | 72B Acc@0.5 (nr=400) | gap to baseline | 72B mean IoU |
|---|---|---|---|---|---|---|
| baseline | 83.75% (335/400) | -- | 0.746 | 92.25% (369/400) | -- | 0.814 |
| 25% | 79.75% (319/400) | -4.00pp | 0.705 | 90.25% (361/400) | -2.00pp | 0.778 |
| **30%** | **84.50% (338/400)** | **+0.75pp** | 0.734 | 89.75% (359/400) | -2.50pp | 0.774 |
| 35% | 84.75% (339/400) | +1.00pp | 0.731 | 89.25% (357/400) | -3.00pp | 0.780 |
| 40% | 85.25% (341/400) | +1.50pp | 0.736 | 88.75% (355/400) | -3.50pp | 0.785 |
| 50% | 87.00% (348/400) | +3.25pp | 0.758 | 88.75% (355/400) | -3.50pp | 0.773 |
| 60% | 87.50% (350/400) | +3.75pp | 0.768 | 90.50% (362/400) | -1.75pp | 0.791 |
| **70%** | 88.75% (355/400) | +5.00pp | 0.789 | **92.25% (369/400)** | **+0.00pp** | 0.801 |
| 80% | 89.25% (357/400) | +5.50pp | 0.793 | 92.25% (369/400) | +0.00pp | 0.809 |
| 90% | 89.50% (358/400) | +5.75pp | 0.794 | 92.00% (368/400) | -0.25pp | 0.813 |
| 100% | 88.50% (354/400) | +4.75pp | 0.789 | 91.50% (366/400) | -0.75pp | 0.805 |

**RefCOCO FULLY COMPLETE (both models, all 8 points x baseline).** 32B crosses at 30% and stays
above through 100%. 72B crosses at 70% and hovers essentially at baseline from 70% through 100%
(exact ties at 70%/80%, tiny -0.25/-0.75pp dips at 90%/100% that are within sample noise, not a
real re-widening).

**72B's precise RefCOCO crossing point: keep_rate=70%** -- another exact tie with baseline (92.25%,
369/400 both, identical down to the sample count -- same pattern as 72B's RealWorldQA crossing at
100%). Every point below 70% (25% through 60%) stayed below baseline.

**32B's precise crossing point: keep_rate=30%.** The dense sweep pins this down exactly: 25% is
still -4.00pp below baseline, but 30% is already +0.75pp above (84.50% vs 83.75%), and stays above
through 35/40/50%. This narrows the earlier "25-40%" bracket to a clean single crossing point --
meaningfully earlier than nr=50's original ~50% estimate. The qualitative conclusion (RefCOCO needs
more correction than RealWorldQA's VQA task) still holds, but note RealWorldQA's OWN elbow was
revised upward to 50% (32B) / 100% (72B) in the dense sweep -- so the "RefCOCO needs more" framing
now needs re-examining once RealWorldQA's revised numbers are factored in; see
`QWEN25VL_APPCORR_LOG.md` section 7 for the full cross-dataset reconciliation.

**72B: does NOT cleanly recover to baseline, contradicting the original "flat/robust from 2%"
claim.** At nr=50, 72B looked essentially saturated at every tested keep_rate (88-94%, close to its
90% baseline, no visible trend). At nr=400, there is a real, if modest, persistent gap: 25%=-2.00pp,
40%=-3.50pp, 50%=-3.50pp -- the gap does not close by 50%, and if anything widens slightly from 25%
to 40% before flattening. This is a genuine revision: 72B's RefCOCO robustness at low keep_rate was
an nr=50 artifact, not a real property. The gap (2-3.5pp) is still much smaller than 32B's low-end
gap (4pp at 25%, and 32B needed to go all the way to ~35-40% to close it), so the *relative* finding
("72B needs less correction than 32B for this task") still holds -- but "72B is fully robust
regardless of keep_rate" does not.

## Full curve (nr=50, ORIGINAL -- see revision above for the more reliable nr=400 numbers)

| keep_rate | 32B Acc@0.5 | 32B mean IoU | 72B Acc@0.5 | 72B mean IoU |
|-----------|------------|-------------|------------|-------------|
| 2%   | 72% (36/50) | 0.621 | 90% (45/50) | 0.800 |
| 5%   | 72% (36/50) | 0.631 | 88% (44/50) | 0.785 |
| 10%  | 78% (39/50) | 0.640 | 88% (44/50) | 0.769 |
| 15%  | 74% (37/50) | 0.654 | 92% (46/50) | 0.783 |
| 25%  | 76% (38/50) | 0.655 | 90% (45/50) | 0.781 |
| 50%  | 88% (44/50) | 0.762 | 94% (47/50) | 0.796 |
| 100% | 90% (45/50) | 0.791 | 90% (45/50) | 0.815 |
| baseline (full-res, no AppCorr) | 84% (42/50) | 0.743 | 90% (45/50) | 0.813 |

## Interpretation

**This is the clearest task-dependent signal across all three datasets tested this session.**
Unlike RealWorldQA and GQA, RefCOCO's grounding task shows a **real, visible cost at low keep_rate
for 32B**: both Acc@0.5 and mean IoU climb steadily from keep_rate=2% (72%, IoU 0.621) up through
50% (88%, IoU 0.762) before leveling off near keep_rate=100%/baseline (90%/84%, IoU
0.79/0.74) -- a genuine ~15-16pp accuracy gap and ~0.12-0.14 IoU gap between the low and high ends
of the curve, tracking together (accuracy and the more continuous IoU signal agree, which is a good
consistency check that this isn't a metric-thresholding artifact). **32B's elbow for RefCOCO sits
around keep_rate=50%, substantially higher than RealWorldQA's ~15%** -- i.e. the hypothesis that
precise spatial localization needs more of the image corrected than semantic VQA does is
*supported* by this data, at least for the smaller model.

**72B is flat and near-baseline across the entire curve** (88-94% Acc@0.5, IoU 0.77-0.82 throughout,
including keep_rate=2%) -- the same robustness pattern seen on RealWorldQA and GQA. For this model
size, even 2% keep rate is enough to localize the referred object at baseline accuracy on this
sample. Notably 72B's accuracy is already very high even at the coarsest setting -- referring
expressions in RefCOCO's val split are frequently the *only* prominent object of that description in
the scene, so even a blurred image may carry enough coarse spatial/color/category signal for a
72B-scale model to localize it approximately; a harder or more cluttered grounding benchmark might
show a lower floor.

**Cross-dataset conclusion**: the "how much correction is needed" question does **not** have a
single universal answer -- it is genuinely task-dependent (RefCOCO's spatial grounding needs
substantially more than RealWorldQA's/GQA's semantic VQA, at least for the 32B model) and
model-size-dependent (72B is robust across all three tasks tested at even the lowest keep_rate
tried, 2%, while 32B's required keep_rate varies from ~15% to ~50% depending on the task). See
`QWEN25VL_APPCORR_LOG.md`'s cross-dataset comparison section for the full picture.
