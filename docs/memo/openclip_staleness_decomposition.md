# Why OpenCLIP's interleaved recovery is low: a 2x2 decomposition

COCO retrieval, full 5000-image split, CLIP-bigG, sequential 4-group schedule. All arms measured
2026-08-26 on the same code; the g=1 k=1.00 identity reproduced the ceiling bit-exactly first
(67.92 / 50.64), so none of this is a fork defect.

| arm | i2t R@1 | t2i R@1 | recovery (i2t) |
|---|---:|---:|---:|
| floor (approx-only) | 50.14 | 40.37 | 0% |
| interleaved g=4, keep=0.50 | 59.10 | 44.95 | 50.4% |
| one-shot g=1, keep=0.50 | 64.86 | 47.36 | **82.8%** |
| interleaved g=4, keep=1.00 | 66.26 | 49.22 | 90.7% |
| one-shot g=1, keep=1.00 | 67.92 | 50.64 | 100% (identity) |

Decomposition of the 17.78pp i2t gap at the keep=0.50 operating point:

    selection cost   (ceiling - g1k0.5)   3.06pp
    staleness cost   (g1k0.5  - g4k0.5)   5.76pp   <- dominant
    staleness @ k=1  (ceiling - g4k1.0)   1.66pp

**Staleness interacts with keep rate.** At keep=1.0 every token is eventually corrected, so all
K/V ends fresh and only ordering staleness remains (1.66pp). At keep=0.5 the unselected half keeps
degraded K/V forever AND the selected half is corrected piecemeal -- 5.76pp, 3.5x larger. Reading
the k=1.0 staleness as an upper bound for k=0.5 is exactly wrong, and was in fact mis-read that way
for one message before this table existed.

**The practical point: CLIP wants one-shot, not interleaved.** One-shot 50% recompute recovers
82.8%; splitting the same recompute into 4 rounds drops it to 50.4%. That is the opposite of SAM 3
(55% recompute interleaved recovers ~90%) and inverts, for this model only, the
interleaved-is-cheap story: here the 1/g critical-compute saving costs 5.76pp.

Hypothesis only, not measured: CLIP's output is a single final-layer CLS embedding -- a global pool
over every token -- so stale K/V anywhere degrades THE output; dense-prediction models (DINOv3,
SAM 3) have local outputs dominated by local tokens, which the schedule corrects together. A
per-position sensitivity probe would settle it; not run.

Table implication, undecided (user's call): an OpenCLIP one-shot row would show the real trade --
larger critical compute, +5.8pp i2t over the interleaved row at the same keep.

## Table handling of near-zero-gap rows (2026-08-26, user decision)

Rows where floor ~= ceiling (OV2 RefCOCO 0.00pp; Gemma3 GQA -0.11pp / RealWorldQA 0.78pp / POPE
1.20pp; OV2 GQA 1.10pp) produce meaningless preservation ratios (>100%, or ratios of noise). The
options were: blank the ratio cells, drop the rows, or leave as-is. **User's call: leave them
exactly as they are; they will handle presentation themselves later.** Do not "fix" these rows in
make_eval_table.py.
