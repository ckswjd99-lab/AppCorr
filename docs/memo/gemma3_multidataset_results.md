# Gemma 3 4B unified axis: four datasets, and where the method stops mattering

Full test sets, `keep=0.55`, L2 degradation with the per-axis pyramid rule, pscore = residual energy
x mean incoming attention. Axis = 27 SigLIP + 34 LLM decoder = 61 stages. Every arm corrects the
same 2256/4096 patches and 141/256 image tokens (plus text in the final round); `interleaved_g4`
differs from `corrected_t` only in *schedule*, at 0.63x the correction compute.

| dataset | n | ceiling | floor | gap | corrected_t | prsv | interleaved_g4 | prsv | schedule |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ChartQA | 2500 | 0.5876 | 0.1896 | **39.80pp** | 0.5988 | 101.9% | 0.5760 | 98.0% | **−2.28pp** |
| TextVQA | 5000 | 0.6087 | 0.4696 | **13.91pp** | 0.6046 | 99.3% | 0.5927 | 97.4% | **−1.19pp** |
| POPE | 9000 | 0.8416 | 0.8296 | 1.20pp | 0.8464 | 100.6% | 0.8441 | 100.3% | −0.23pp |
| RealWorldQA | 765 | 0.4288 | 0.4209 | 0.78pp | 0.4458 | 104.0% | 0.4458 | 104.0% | 0.00pp |

Bold schedule effects have a 95% CI excluding zero.

## Only two of the four can carry a preservation number

The floor-to-ceiling gap spans **50x** across these datasets on one model, one degradation level, one
verified degradation rule. Where the gap is 1pp, preservation has no resolution: POPE's 100.6% and
RealWorldQA's 104.0% are arithmetic, not evidence, and are not reported as results. RealWorldQA's
gap is not even significant (CI [−1.70, +3.27]).

So the interpretable rows are **ChartQA (39.80pp, 101.9% / 98.0%)** and **TextVQA (13.91pp, 99.3% /
97.4%)**. Both land in the 97-99% band measured on SAM 3 COCO/LVIS/SA-Co, now extended to a
bidirectional vision + LLM-prefill axis rather than a vision encoder alone.

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

## corrected_t sits above the ceiling on three of four

+1.12pp (ChartQA), +1.70pp (RealWorldQA), +0.49pp (POPE), −0.41pp (TextVQA). Every CI includes zero,
so each alone reads as "indistinguishable from exact". The consistent sign is not explained.

Two candidates were ruled out by measurement rather than argument:

- **Not the decode path.** `keep=1.0` through the axis reproduces stock `model.generate` on
  **765/765 RealWorldQA samples** — identical scores, 58 of 61 differing predictions being trailing
  whitespace. The preservation figures are not inflated by the fork's greedy loop.
- **Not run-to-run noise.** Each arm reproduces exactly across processes.

What remains is the mixed-resolution input: a corrected arm's image is neither the original nor the
degraded one, but sharp at the selected 55% and blurred elsewhere. Testing that needs a control arm
with a *random* 55% sharp, to separate "our selection helps" from "mixing helps". The three positive
datasets are binary-scored and the one negative is continuous (VQA soft accuracy); InfoVQA (ANLS)
is queued as the second continuous point.

## Reproduce

```
analysis/experiments/run_gemma3_chartqa_full.sh    # 8 arms
analysis/experiments/run_gemma3_realworld.sh       # realworldqa, textvqa, pope x 4 arms
analysis/experiments/gemma3_status.py              # dataset x arm table, reads results/ directly
```
