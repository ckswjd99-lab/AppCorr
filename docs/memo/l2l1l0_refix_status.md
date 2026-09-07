# L2-L1-L0 re-verification on the fixed codebase — the July rejection was the persist bug

**Status date:** 2026-09-02
**Branch:** `develop/l2l1l0-refixed` (= `develop/dinov3-l2l1l0-tail-full` + main @ b9abaa5)
**Scripts:** `scripts/l2l1l0_gates.sh`, `scripts/l2l1l0_paired100.sh`, `scripts/l2l1l0_full2000.sh`

## Why this re-run exists

Every accuracy number in `analysis/results/dinov3_l2l1l0/` and
`dinov3_l1_only_full/` (2026-07-28..30) predates two later fixes:

1. **Persist fix** (`96889a5`/`ac0238f`, 08-16): pre-fix, a corrected token's own
   value was discarded when a later round replayed the blocks — exactly the
   "correct at L1, move on to L0" step that multi-level correction is made of.
   The July variant ranking matches that bug's fingerprint: only the policy that
   re-corrected 89% of L1's support at L0 (equal threshold) survived; both
   policies that trusted the L1 correction to persist (disjoint, re-entry) lost
   ~1.5 mIoU.
2. **Closed-loop transmission fix** (`378e21d`, 08-17): the branch's
   `_compute_residual_band`/`_projected_residual_band` were still open-loop —
   confirmed in code during the merge (the decode-reference unit test failed
   until both were rebuilt on `_closed_loop_residual`). With L1 sent as one
   complete group, the L2→L1→L0 chain is now lossless by construction.

Merge notes and the two smoke-caught regressions: commits `5e1895d`, `f521450`.
Wiring gates: `[persist] blocks_out_sum write executed` fires on both the L2-L0
and L2-L1-L0 configs; 37 unit tests pass.

## Paired 100-image result (same first-100 protocol as July)

| arm | July (pre-fix) | 2026-09-02 (fixed) | Δ | correction TF/img |
|---|---:|---:|---:|---:|
| full / sequential | 52.9229 | **52.9214** | −0.0015 | 0 |
| L2-L0, t=4e-5 | 52.0812 | **52.7344** | +0.65 | 46.17 |
| L2-L1-L0 strict disjoint | 50.6071 | **52.5616** | **+1.95** | 46.36 |
| L2-L1-L0 equal threshold | 52.1872 | **53.4176** | +1.23 | 84.77 |

Protocol checks: sequential reproduces July to 0.002 (unaffected by both fixes);
selected patch-token-layers reproduce July's (119.3k vs 119.0k; disjoint 119.8k
vs 120.3k; eq-thr 219.0k vs 218.4k), so correction TF matches the July closed
form (46.26 / 46.84 / 84.80) within ~1% — the accuracy movement is the fixes,
not a selection shift.

## Verdicts

- **The July "no useful L1 operating point" conclusion is overturned.** It was
  measured through the persist bug.
- **Strict disjoint is now level with L2-L0** (−0.17, inside the ±0.9 paired
  CI July measured on this protocol) **at equal correction FLOPs** (46.4 vs
  46.2 TF). Pre-fix it lost 1.47 — ~88% of that loss was the bug. What
  disjointness buys: 31% of the correction work (36.6k of 119.8k
  token-layers) moves one round earlier, where it overlaps the L0
  transmission — i.e. same total, lower *critical* compute. Quantifying that
  needs the arrival-anchored accounting the VLM track uses; the offload
  pipeline only logs wall-clock stages today.
- **Equal threshold lands ABOVE full** (53.42 vs 52.92 on n=100) — the
  above-ceiling pattern the VLM streaming arms keep showing. n=100 noise is
  ±0.9 here; full-2000 will say whether it is real.
- Costs unchanged from July: the L1 band still adds ~9-12% bytes
  (2715 vs 2424 KB/img here) and one decode/select round.

## Full-2000 promotion (complete, 2026-09-02)

Floor 56.013 / ceiling 62.236 reused (fix-independent, reuse-validated).
GH200 note: the m2f interleaved working set is ~70GB steady and allocator
fragmentation OOM'd the 95GB card by image 260 — `PYTORCH_CUDA_ALLOC_CONF=
expandable_segments:True` is REQUIRED here (memory then flat for 2000 images);
July's B200 (192GB) never saw this.

| arm | mIoU | gap recovered | selected ptl/req | corr TF/img |
|---|---:|---:|---:|---:|
| floor (L2 approx-only) | 56.013 | 0% | — | 0 |
| L2-L0, t=4e-5 | **61.826** | 93.4% | 118,450 | 45.8 |
| L2-L1-L0 strict disjoint | **61.126** | 82.2% | 120,820 (L1 35,895 + L0 84,924) | 46.8 |
| L2-L1-L0 equal threshold | **61.902** | 94.6% | 217,449 (L1 68,035 + L0 149,414) | 84.2 |
| ceiling (full) | 62.236 | 100% | — | — |

(L2-L0 reproduces the canonical post-fix 61.846 to 0.02 across a different GPU
and the merged branch.)

## Final verdicts (supersede the paired-100 section's optimism)

- **Disjoint's July −1.47 splits into ~0.77 bug + ~0.70 genuine.** At full-2000
  the never-revisit cost is real: −0.70 vs L2-L0 at equal total correction
  FLOPs. The July mechanism reading ("pixel-space Rfine small ≠ final feature
  adequate") survives at half size, now on clean footing. What disjoint still
  buys: ~30% of correction executes one round earlier (overlappable) — whether
  that critical-compute gain is worth −0.70 needs arrival-anchored accounting.
- **Equal threshold: +0.08 over L2-L0 for +84% correction work** — same shape
  as July (+0.11 for +83.5%). Its n=100 above-full reading (53.42 vs 52.92) did
  NOT survive full-2000 (61.90 vs ceiling 62.24) — another instance of the
  standing n=100-is-a-sanity-check rule.
- **The July hard rejection softens but the operating points still don't win
  on accuracy-per-total-FLOP.** The two doors the fixes opened: (1) the
  critical-FLOPs axis for disjoint, (2) **conditional re-entry re-tuned
  post-fix** — July tuned it as compensation for the bug; a small re-entry
  ratio on top of disjoint may now buy back the 0.70 for a fraction of
  eq-thr's 38 extra TF. That sweep (reentry_ratio ∈ {0.1, 0.25, 0.5}, paired
  100) is the obvious next experiment, and cheap.

## Not yet re-tested

- Conditional re-entry (July 50.51): pointless to re-run as-is — its design
  compensates for a bug that no longer exists; revisit only if disjoint's
  full-2000 shows a gap worth closing.
- L2→L1 endpoint (July 46.51 vs L1-only 52.65): the "scalar mass ≠ feature
  consistency" reading needs re-checking post-fix before it is quoted again.
- Tail-full (`final_full_layers`) interaction with L2-L1-L0.
- The L1-only anchors (ADE20K −0.87 vs full, ImageNet −0.35, NYU ~0, COCO
  −3.5 AP) are approx-only paths — unaffected by both fixes, still standing:
  **any multi-level claim must still be argued against L1-only.**
