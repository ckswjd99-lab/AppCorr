# Gemma 3 4B — AppCorr on the unified vision+prefill axis, full ChartQA

Full ChartQA test (n=2500), `keep=0.55`, L2 degradation, pscore = residual energy x mean incoming
attention. Axis = 27 SigLIP layers + 34 LLM decoder layers = 61 stages; correction cost is measured
against the **one-shot correction pass**, not against the approximate pass (both are paid by every
corrected arm, so the ratio isolates the correction work and the end-to-end saving is smaller).

Selection sizes: 2253-2256 of 4096 patches, 180 of 272 LLM tokens (141 image + 38.5 text) for
`corrected_t`/`corrected`/interleaved; 261 for `corrected_j`.

| arm | acc | preservation | correction cost | vs ceiling (95% CI) | p |
|---|---:|---:|---:|---|---:|
| ceiling | 0.5876 | 100.0% | — | — | — |
| `corrected_t` | 0.5988 | 101.9% | 1.00x | +1.12pp [−0.16, +2.44] | 0.104 |
| `corrected` | 0.5956 | 101.4% | 1.00x | +0.80pp [−0.72, +2.28] | 0.307 |
| `corrected_j` | 0.5984 | 101.8% | 1.00x | +1.08pp [−0.16, +2.36] | 0.108 |
| `interleaved_g2` | 0.5640 | 96.0% | 0.75x | −2.36pp [−3.84, −0.92] | 0.002 |
| `interleaved_g4` | 0.5760 | 98.0% | 0.63x | −1.16pp [−2.56, +0.24] | 0.123 |
| `interleaved_g8` | 0.5740 | 97.7% | 0.56x | −1.36pp [−2.80, +0.08] | 0.075 |
| floor | 0.1896 | 32.3% | 0 | −39.80pp [−41.96, −37.72] | <0.001 |

Floor-to-ceiling gap is **39.8pp**, wide enough for preservation to mean something.

## The three one-shot selection modes are interchangeable

`corrected` − `corrected_t` = −0.32pp (p=0.643); `corrected_j` − `corrected_t` = −0.04pp (p=1.000).
All three sit within noise of the exact forward. **`corrected_j` reaches the same accuracy while
correcting 45% more LLM tokens (261 vs 180), so it is strictly dominated.** Use `corrected_t`: a
single token budget controls the whole axis and the patch mask is derived from it.

This **retracts an earlier 120-sample reading** in which `corrected_j` (0.733) led the three arms and
suggested that giving the LLM half a larger share helps. That ranking does not survive the full set.

## Interleaving costs a fixed penalty that does not depend on the round count

Against one-shot `corrected_t`: g2 −3.48pp, g4 −2.28pp, g8 −2.48pp (all p≤0.001). The penalty is
real and roughly **constant at ~2.5pp** — round-count comparisons are all null (g4 vs g2 p=0.063,
g8 vs g4 p=0.772, g8 vs g2 p=0.158), and the ordering is not monotone in `g`.

**This contradicts the expectation that more rounds would help** (more forward passes follow each
correction). It also rules out the first explanation reached for — that finer splitting hurts because
early groups are corrected only over shallow stages — since that predicts monotone decay in `g`, and
there is none.

A penalty that is insensitive to `g` points at the *fact* of splitting rather than its granularity:
every token except the last group's is corrected at some prefix of the axis and never revisited, and
that is equally true at g=2 as at g=8. **This is a hypothesis, not a measurement** — the way to test
it is feature-space relative L2 per group against the exact forward, which has not been run.

## What the interleaved arms actually support

g4 and g8 are **statistically indistinguishable from the exact forward** (CIs include 0) at 0.63x and
0.56x correction cost. The claim is not "interleaving beats one-shot" — it does not — but "37-44%
less correction compute, still within noise of exact." g2 is the odd one out: it is the most
expensive interleaved arm *and* the only one significantly below the ceiling, so it is not on the
frontier and should not be reported as a design point.

98.0% preservation matches the 97-99% band measured on COCO tracker/detector, LVIS, and SA-Co, now
extended to a bidirectional vision+LLM-prefill axis rather than a vision encoder alone.

## Caveats

- Correction cost is a **model** (`tokens x width^2`), not wall clock. No latency measurement exists
  anywhere in this repo, and neither does a transmission-to-compute ratio.
- The preservation figures above 100% mean "indistinguishable from the ceiling," not "better than
  exact." Every one-shot CI includes zero.
- Single keep ratio (0.55). The 39.8pp gap makes a keep sweep worth running.
