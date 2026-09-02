# Quarter-wavefront L2L1L0 — full-2000 sweep results (2026-09-04, B200)

Branch `develop/l2l1l0-quarter-wavefront`. Schedule per the user scenario: 6 arrival
groups (L2 base / complete-L1 global / 4 L0 spatial quarters) x 5 balanced 8-layer
chunks (`ADE20KWindowInterleaved`, N=5). Selection: L1 = global top-rho1 (ratio mode,
thresholds null), L0 = per-quarter top-rho0 over quarter-minus-L1-selected, conditional
re-entry ratio r. Costs are recomputed patch token-layers per request from the worker
counters (mean over 2000; total candidate pool 268,213.9).

| arm | rho1 | r | mIoU | L1 t-l | L0 t-l | total t-l |
|---|---|---|---|---|---|---|
| p25_r00 | 0.25 | 0    | 60.556 | 11,616 | 83,450  | 95,066  |
| p50_r00 | 0.50 | 0    | 60.934 | 23,231 | 63,962  | 87,193  |
| p50_r25 | 0.50 | 0.25 | 61.233 | 23,231 | 87,553  | 110,785 |
| p50_r50 | 0.50 | 0.50 | 61.326 | 23,231 | 107,806 | 131,037 |

Flat crop-16 anchors (2026-09-02 sweep, same 2000): disjoint r=0 61.126 (~125k t-l,
extrapolated), r=0.10 61.363 @ 140,188, r=0.25 61.563 @ 162,875, r=0.50 61.719 @
197,049; L2-L0 61.826, eq-thr 61.902, ceiling 62.236, floor 56.013.

## Findings

1. **Front-loading dominates in-family**: p50_r00 beats p25_r00 on BOTH axes
   (+0.38 mIoU, -8% cost) — moving budget to the global 8-layer L1 round is strictly
   better than leaving it to deeper L0 rounds under disjoint support.
2. **The wavefront re-entry buy-back has a knee, unlike flat**: slope r00->r25 =
   0.0127 mIoU/1k t-l (1.4-2.8x flat's 0.0046-0.0088 in the same cost range), then
   r25->r50 = 0.0046/1k. Flat was linear with no knee; the quarter schedule is not.
3. **Pareto**: p50_r25 (61.233 @ 110.8k) dominates flat disjoint (61.126 @ ~125k) —
   more accurate AND cheaper. p50_r50 (61.326 @ 131k) sits at parity with flat r=0.10
   (61.363 @ 140k). The rho=0.5 family tops out ~61.33, below L2-L0 (61.826): whether
   the curve crosses L2-L0 at larger rho0 budgets is the open follow-up.
4. Cost signature exact at smoke scale (L1 = rho1*N*8, integer-exact; totals sum).
   Critical-path split (arrival-tagged FLOPs) was not measured in this sweep — the
   counters above are totals; a flops-mode pass is the next step if the critical axis
   is pursued.

Verdict so far: the depth-staggered quarter schedule buys the disjoint operating
region more cheaply than flat and restores a knee to the re-entry trade — but at
matched large budgets flat L2-L0 still holds the accuracy high ground. Follow-ups,
pending user's call: (a) rho0 sweep at r=0.25 to test crossing L2-L0, (b) arrival-split
critical-compute measurement, (c) bandwidth-matched ahead_layers variant.
