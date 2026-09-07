# Gemma 3 4B unified axis: five datasets, and the first place 55% is not enough

Full test sets, `keep=0.55`, L2 degradation with the per-axis pyramid rule, pscore = residual energy
x mean incoming attention. Axis = 27 SigLIP + 34 LLM decoder = 61 stages. Every arm corrects the
same 2256/4096 patches and 141/256 image tokens (plus text in the final round); `interleaved_g4`
differs from `corrected` only in *schedule*, at 0.63x the correction compute.

| dataset | n | ceiling | floor | gap | corrected | prsv | interleaved_g4 | prsv | schedule |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ChartQA | 2500 | 0.5876 | 0.1896 | **39.80pp** | 0.5988 | 101.9% | 0.5760 | 98.0% | **−2.28pp** |
| InfoVQA | 2801 | 0.3956 | 0.1901 | **20.55pp** | 0.3190 | **80.6%** | 0.3246 | **82.0%** | +0.56pp |
| TextVQA | 5000 | 0.6087 | 0.4696 | **13.91pp** | 0.6046 | 99.3% | 0.5927 | 97.4% | **−1.19pp** |
| POPE | 9000 | 0.8416 | 0.8296 | 1.20pp | 0.8464 | 100.6% | 0.8441 | 100.3% | −0.23pp |
| RealWorldQA | 765 | 0.4288 | 0.4209 | 0.78pp | 0.4458 | 104.0% | 0.4458 | 104.0% | 0.00pp |

Bold schedule effects have a 95% CI excluding zero.

## InfoVQA breaks the 97-99% band

Every benchmark measured so far -- SAM 3 on COCO/LVIS/SA-Co, Qwen2.5-VL across twelve tasks, Gemma 3
on ChartQA and TextVQA -- preserved 97-100% of the exact forward at 55% recompute. **InfoVQA
preserves 80.6%**, against a 20.55pp gap that is wide enough to measure properly, with a CI far from
zero (−7.67pp [−9.15, −6.17]). This is the first dataset where 55% recompute is demonstrably not
enough, and it is a result rather than a defect: the arm reproduces, the degradation obeys the
per-axis rule (verified: infographics are the worst case for the short-side bug, 1.80x median), and
the decode path matches stock generate.

The schedule effect flips sign with it: **+0.56pp [−0.74, +1.87]**, i.e. interleaving costs nothing
here, where it cost 2.28pp on ChartQA and 1.19pp on TextVQA with CIs excluding zero.

Infographics are tall (median aspect 2.14, up to 10.58) and densely typeset, so a plausible reading
is that the damage is spread across the whole page rather than concentrated in a selectable minority
-- leaving top-k selection with no target to aim at, and leaving nothing for a correction schedule
to get wrong either. **This is a hypothesis; it has not been measured.**

## Only three of the five can carry a preservation number

The floor-to-ceiling gap spans **50x** across these datasets on one model, one degradation level, one
verified degradation rule. Where the gap is 1pp, preservation has no resolution: POPE's 100.6% and
RealWorldQA's 104.0% are arithmetic, not evidence, and are not reported as results. RealWorldQA's
gap is not even significant (CI [−1.70, +3.27]).

So the interpretable rows are **ChartQA**, **TextVQA** and **InfoVQA**. ChartQA and TextVQA land in
the 97-99% band measured on SAM 3 COCO/LVIS/SA-Co, now extended to a bidirectional vision +
LLM-prefill axis rather than a vision encoder alone. InfoVQA does not, and that is the point of the
section above: a wide gap is necessary for the number to mean anything, not sufficient for it to be
high.

The same ordering holds on Qwen2.5-VL (`analysis/qwen_vl_prefill/README.md`): ChartQA 55.64pp >
TextVQA 19.06pp >> POPE 4.93pp > RealWorldQA 2.48pp. Two architectures, same ranking — the damage
tracks the task, not the model. Qwen's gaps are uniformly wider, which is consistent with Gemma 3
discarding information before degradation even applies (896x896 squash, then 4x4 pooling to 256
tokens), though that mechanism has not been measured.

## "Insensitive" was the wrong word

On RealWorldQA the degradation changes **250 of 765 predictions (33%)** and on POPE **950 of 9000
(10.6%)**, while the accuracy barely moves. The answers move; they just move in both directions
almost equally (RealWorldQA 45 vs 44, POPE 529 vs 421). At 42.9% accuracy on 4-way multiple choice,
much of RealWorldQA sits near the model's competence floor, where perturbation reshuffles borderline
answers rather than destroying information.

## The schedule penalty appears exactly where it can be seen

Interleaving costs a real ~1-2pp against one-shot on both wide-gap datasets (CIs exclude zero) and
nothing measurable on the two narrow ones. That is the expected pattern rather than an inconsistency:
with a 1.20pp gap there is no room for any correction schedule to distinguish itself.

On ChartQA the round count does **not** order the result (g2 −3.48, g4 −2.28, g8 −2.48 against
one-shot; g8 vs g4 p=0.772), which rules out the first explanation reached for — that finer
splitting hurts because early groups are corrected over shallow prefixes — since that predicts
monotone decay. A penalty insensitive to `g` points at the fact of splitting rather than its
granularity. **Untested**: the check is per-group feature-space L2 against the exact forward.

## corrected sits above the ceiling on three of five

+1.12pp (ChartQA), +1.70pp (RealWorldQA), +0.49pp (POPE), −0.41pp (TextVQA), −7.67pp (InfoVQA).
Except on InfoVQA every CI includes zero, so each alone reads as "indistinguishable from exact".
The consistent sign among the other four is not explained.

Two candidates were ruled out by measurement rather than argument:

- **Not the decode path.** `keep=1.0` through the axis reproduces stock `model.generate` on
  **765/765 RealWorldQA samples** — identical scores, 58 of 61 differing predictions being trailing
  whitespace. The preservation figures are not inflated by the fork's greedy loop.
- **Not run-to-run noise.** Each arm reproduces exactly across processes.

What remains is the mixed-resolution input: a corrected arm's image is neither the original nor the
degraded one, but sharp at the selected 55% and blurred elsewhere. Testing that needs a control arm
with a *random* 55% sharp, to separate "our selection helps" from "mixing helps". The three positive datasets are
binary-scored and both negative ones are continuous (TextVQA soft accuracy, InfoVQA ANLS), which is
the direction the scoring-variance idea predicts. It is weak evidence: InfoVQA's −7.67pp is real
damage rather than a scoring artefact, so it does not separate the two explanations. A random-55%
control arm would.

## Reproduce

```
analysis/experiments/run_gemma3_chartqa_full.sh    # 8 arms
analysis/experiments/run_gemma3_realworld.sh       # realworldqa, textvqa, pope x 4 arms
analysis/experiments/run_gemma3_infovqa.sh         # infovqa x 4 arms
analysis/experiments/gemma3_status.py              # dataset x arm table, reads results/ directly
```

## Arm naming

`corrected` is the default and the only one recommended: tokens lead, and the patch mask is derived
from them so `patch_mask_any_to_token(pm) == sel_tok` holds exactly -- the identity interleaved needs
to split patches into groups that map onto token groups. It was called `corrected_t` while three
variants were being compared.

`corrected_split` (independent vision/LLM budgets, formerly `corrected`) and `corrected_patchled`
(patches lead, formerly `corrected_j`) are kept because they were measured, not because they are
recommended. On full ChartQA all three tie (p >= 0.64) and `corrected_patchled` reaches that tie
while correcting 45% more LLM tokens. Neither composes with interleaving.

## GQA across three models: the gap is the model's, not the task's (2026-08-26)

Gemma 3's GQA floor sits 0.11pp ABOVE its ceiling, which looks alarming until the same dataset is
read across the models that ran it:

| model | floor | ceiling | gap |
|---|---:|---:|---:|
| Qwen2.5-VL 32B | 55.24 | 60.80 | **+5.56** |
| LLaVA-OV2 8B | 61.87 | 62.97 | +1.10 |
| Gemma 3 4B | 42.96 | 42.84 | **-0.11** |

Same task, same degradation level, same 12,578 samples. So "GQA does not need resolution" is wrong --
Qwen loses 5.56pp to the same degradation. The insensitivity is Gemma 3's.

This is what the hypothesis two sections up predicts (Gemma 3 discards information before degradation
applies: 896x896 squash, then 4x4 pooling to a FIXED 256 image tokens, where Qwen and OV2 scale token
count with resolution). Supporting measurement, still not a test of it: on GQA the L2 degradation
removes MORE relative signal than on TextVQA (0.277 vs 0.169 mean relative residual over 40 images),
yet Gemma 3's TextVQA gap is 13.1pp and its GQA gap is zero -- so the degradation is working and the
loss is absorbed elsewhere. GQA's images are also small (median 640x427), so Gemma 3 upscales them to
the canvas before pooling, and an upscale carries nothing to remove.

**Still not measured**, and the check that would settle it: constrain Qwen to 256 image tokens and
see whether its GQA gap collapses toward Gemma 3's. Until then this is a consistent story, not a
demonstrated mechanism.
