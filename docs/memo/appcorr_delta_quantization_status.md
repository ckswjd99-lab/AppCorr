# Quantizing the correction delta — what survives, and why it is not FP4

**Status date:** 2026-08-18
**Branch:** `develop/fp4-no-per-tensor-scale`
**Harness:** `offload/server/model/delta_split_linear.py`, `analysis/experiments/dinov3_delta_sparsity.py`

## The question this answers

If the L0 baseline can simply be run in FP4, AppCorr is not needed. That threat is **real at FP4**:

| ADE20K full-2000 | mIoU | transfer/image |
|---|---:|---:|
| L0 baseline, all Linears FP4 | **62.2208** | 1800.83 KB |
| AppCorr, approx bf16 + correct FP4 (41.27% recompute) | 61.814 | 2405.78 KB |
| L0 baseline bf16 (ceiling) | 62.236 | — |

L0-FP4 wins on accuracy (+0.41) *and* on bytes (−25%; measured on a matched 50-image slice,
1852.95 vs 2493.60 KB). Server compute is the one axis AppCorr leads, 538.96 vs 700.85 ms/request,
and it gives 71 ms of that back on the client encode. **FP4 is orthogonal to AppCorr and cannot be
the argument.** Attributing the transfer overhead: 84% is inherent to the progressive scheme, 16% is
the closed-loop transmission fix (2393.13 → 2493.60), and the `crop_cover` group-count change moved
it by 0.00 KB.

## Where the asymmetry actually is

ADE20K, 50-image slice, floor (no correction) 45.349, ceiling (L0 bf16) 54.160, dense bf16
correction 54.591. Every row same slice, errors 0, delta rows verified fallback-free.

| correction | format | sparsity | mIoU | recovered |
|---|---|---|---:|---:|
| — (L0 baseline) | ternary | — | **0.525** | −508.7% |
| dense recompute | fp4 | — | 54.158 | 100.0% |
| dense recompute | **ternary** | — | **14.109** | **−354.5%** |
| delta propagation | fp4 | off | 53.692 | 94.7% |
| delta propagation | **ternary** | off | **48.519** | **36.0%** |
| delta propagation | none | unstructured 50% | 53.344 | 90.7% |
| delta propagation | none | **2:4** | 52.182 | 77.5% |
| delta propagation | **fp4** | **2:4** | **52.920** | **85.9%** |
| delta propagation | ternary | 2:4 | 48.419 | 34.9% |

Three readings:

- **Ternary destroys the L0 baseline (0.525) but not the delta (48.519).** That is the asymmetry FP4
  never produced, and it is what makes the format choice non-orthogonal to the method.
- **Dense recompute in a bad format is worse than not correcting** (14.109, below the 45.349 floor):
  it overwrites a good approximation with a requantized one. Delta propagation cannot do that — the
  base stays BF16, so a useless delta degenerates to "no correction". **That is the whole mechanism.**
- **FP4 and 2:4 compose for free.** 52.920 with both against 52.182 with sparsity alone; 4 bits and
  2x sparsity together cost nothing extra over sparsity alone (at n=50, within noise). The 2:4
  structure itself costs 1.16 mIoU against unstructured 50% (52.182 vs 53.344).

## Sparsity shows NO delta-specific advantage — the control reverses it

Added 2026-08-18 after measuring the control this table was missing. Applying 2:4 to the **raw L0
activation** (`approx_act_sparsity`, no correction anywhere in the pipeline) costs almost nothing:

| 2:4 applied to | dense | sparse | cost |
|---|---:|---:|---:|
| raw L0 activation (50-image slice, sequential config) | 54.160 | **54.119** | **-0.041** |
| correction delta (50-image slice, interleaved config) | 54.591 | 52.182 | **-2.409** |
| correction delta (full 2000, interleaved config) | 61.846 | 61.562 | -0.284 |

**Sparsifying the raw activation is ~59x cheaper than sparsifying the delta.** The asymmetry does not
merely vanish, it runs the other way, and the offline proxy pointed the wrong direction: per-layer
output error said the delta was 2.24x *better* (0.0462 vs 0.1036). The sign flipped between the
proxy and the task metric.

The mechanism is obvious in hindsight and consistent with the shape data above: an activation is
redundant (top 10% of entries hold 77% of its energy), so discarding the small half discards little.
The delta is the correction itself — discarding half of it discards half the correction.

**Consequences.** Do not claim 2:4 as delta-specific; one control run refutes it. The earlier reading
in this memo — that 2.24-2.32x made sparsification "worth building" as a delta technique — is
withdrawn. The -0.284 mIoU cost at full scale remains a true and useful engineering result, but its
cause is that *this model tolerates 2:4 broadly*, not that the delta form protects it.

The format axis is unaffected: ternary still destroys the L0 baseline (0.525) while the delta form
survives (48.519). **The asymmetry lives in bit width, not in sparsity** — argue the method there.

## The mechanism is magnitude, not distribution — say so carefully

Three separate probes agree that the delta is **not** an easier tensor to quantize:

| probe (COCO 1024, per-layer) | `a+d` | `d` |
|---|---:|---:|
| block16 error, relative to own output | 0.0865 | **0.1085** |
| outlier ratio (global amax / mean block amax), median | 25.7 | 24.3 |
| top-1% energy share | 0.498 | **0.433** |

The delta is slightly *harder* per unit norm, no spikier, and no more outlier-prone. What it is, is
**smaller**: `||Wd|| / ||W(a+d)||` averages 0.343 at 1024px and falls to 0.12–0.17 by block 39. Every
measured win is that factor. Sparsification looks 2.24–2.32x better on the delta, but pure magnitude
predicts 1/0.343 = 2.92x, so per unit norm sparsification is *also* slightly worse on the delta.

So the claim to make is **"AppCorr shrinks and protects what gets quantized"**, not "the delta
quantizes better". The second is falsifiable with one plot and a reviewer will ask for it.

Note the earlier `dinov3_exact_decomposition_fp4_features.md` result (FP4-on-delta 1.7x/4.1x better)
compares `exact_bf16_fp4` against `fp4_full` — the delta arm simply has fewer quantized GEMMs. It is
not evidence of a distributional advantage and should not be cited as such.

## Depth

The magnitude effect strengthens with depth, which the shipped policy does not exploit:

| block | `d` share of output | 2:4 sparsification error, `a+d` → `d` |
|---|---:|---|
| 10 | 0.60–0.73 | 0.135 → 0.103 (1.3x) |
| 20 | 0.36–0.61 | 0.157 → 0.085 (1.9x) |
| 30 | 0.28–0.42 | 0.150 → 0.057 (2.6x) |
| 39 | 0.12–0.17 | 0.155 → 0.030 (5.2x) |

A depth-dependent precision/sparsity schedule is the obvious follow-up.

## Harness notes

`correct_delta_split` = `off` | `exact` | `quant_full` | `quant_delta`; `correct_quant_format` =
`fp4` | `ternary` | `none`; `correct_delta_sparsity` = `off` | `2:4` | `unstructured50`.

**Fake-quant throughout, and slower than what it measures — never time it.** The emulator's offset
against the real `_scaled_mm` correction path measured 0.077 mIoU on the 50-image slice (54.081 vs
54.158).

`exact` mode is the wiring gate: it splits the Linear without quantizing, so it should reproduce the
dense path. It is **not** bit-identical — `Linear(a) + Linear(d)` reassociates the sum — so the gate
is "close, and zero fallbacks", with the fallback counter being the load-bearing half. Six wiring
faults were caught by it and none reached a measurement; the two worth remembering are that the
correction routes on `_effective_precision()` (so the harness has to claim its own route, or it is
installed and never called) and that the base cache must be generation-keyed, because
`begin_dinov3_correct_event` fires per correction rather than per request.

## Next

- ~~Full-2000 confirmation of the delta + 2:4 arm.~~ **Done: 61.562 (-0.284 against dense bf16
  61.846), 89.1% of the floor-ceiling gap.** But see the sparsity control above before using it.
- Round-to-round error feedback: carry `d - dequant(quant(d))` into the next interleaved round. This
  is structurally impossible without AppCorr's multi-round correction and is the most promising
  remaining candidate for closing the ternary gap.
- Stochastic rounding on the delta — unbiasedness matters more than per-element accuracy for an
  increment.
- The transmission overhead. None of the above touches the 34.6% byte deficit, and until that moves,
  the bandwidth axis stays lost regardless of what the compute axis does.
