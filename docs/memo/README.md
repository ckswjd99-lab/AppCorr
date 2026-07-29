# docs/memo

Catch-all folder for experiment notes, results, and design memos — the place to dump "이것저것"
(findings, sweep tables, design decisions) so they don't get lost in scratch dirs or commit messages.

One markdown file per topic. Keep raw numbers + the conclusion; link to the branch/config that
produced them.

## Index

- [ade20k_sr_residual_pruning_sweep.md](ade20k_sr_residual_pruning_sweep.md) — ADE20K m2f (ViT-7B):
  full-val threshold sweep of appcorr / appcorr+SR / appcorr+SR+SR-residual-pruning. Finding:
  SR-residual pruning only beats plain appcorr in a narrow ~30% recompute band; SR base alone ≈ a wash.
- [ade20k_grid_vs_blockgrid_grouping.md](ade20k_grid_vs_blockgrid_grouping.md) — ADE20K m2f: interleaved
  `grid` (dispersed) vs `block_grid` (contiguous quadrants). block_grid ~+0.5 mIoU at matched recompute
  on full 2000 (nr=100 was misleadingly reversed by noise). Includes how block_grid maps onto the
  sliding-window crops (global-coord quadrants) and why coherent per-crop correction timing helps.
- [ade20k_cropcover_grouping_sweep.md](ade20k_cropcover_grouping_sweep.md) — ADE20K m2f: dedicated
  `crop_cover` policy (groups aligned to sliding crops, per-image variable group count). Full-2000
  5-point sweep (8–67% recompute): crop_cover dominates the frontier — +1.56 mIoU over grid at matched
  ~42%, matches block_grid's best (61.3) at less compute. Recompute plateaus ~67% (zero-residual
  patches never corrected); avg-attention is per-crop.
- [dinov3_approx_low_precision_status.md](dinov3_approx_low_precision_status.md) — DINOv3 ViT-7B
  approx-only FP8/auto/NVFP4 implementation and full ImageNet-1k/COCO measurements. Large-row
  ImageNet B32 gains 1.97x/2.85x approx speedups with FP8/FP4, while small-row COCO B1 is slower;
  includes the 3x3-window row-count explanation and remaining interleaved work.

## Related work elsewhere in the repo (not in this folder)

- **DINOv3 CSR** (sparse attention + FFN + token pruning) — branch `develop/dinov3-csr`. ImageNet-1k
  ViT-7B: attention-CSR lossless vs exact (93.55 vs 93.16 top-1), FFN-CSR trades accuracy for
  hidden-sparsity (25%→90.23, 50%→92.19); token pruning works. Configs
  `imnet_interleaved_static_csr*.json`.
- **Qwen2.5-VL progressive prefill** — `analysis/qwen_vl_prefill/` (committed). Cheap correction is
  near-lossless on VQA/captioning; only RefCOCO grounding pays ~−3pp.
