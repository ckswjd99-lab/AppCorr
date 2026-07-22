# docs/memo

Catch-all folder for experiment notes, results, and design memos — the place to dump "이것저것"
(findings, sweep tables, design decisions) so they don't get lost in scratch dirs or commit messages.

One markdown file per topic. Keep raw numbers + the conclusion; link to the branch/config that
produced them.

## Index

- [ade20k_sr_residual_pruning_sweep.md](ade20k_sr_residual_pruning_sweep.md) — ADE20K m2f (ViT-7B):
  full-val threshold sweep of appcorr / appcorr+SR / appcorr+SR+SR-residual-pruning. Finding:
  SR-residual pruning only beats plain appcorr in a narrow ~30% recompute band; SR base alone ≈ a wash.

## Related work elsewhere in the repo (not in this folder)

- **DINOv3 CSR** (sparse attention + FFN + token pruning) — branch `develop/dinov3-csr`. ImageNet-1k
  ViT-7B: attention-CSR lossless vs exact (93.55 vs 93.16 top-1), FFN-CSR trades accuracy for
  hidden-sparsity (25%→90.23, 50%→92.19); token pruning works. Configs
  `imnet_interleaved_static_csr*.json`.
- **Qwen2.5-VL progressive prefill** — `analysis/qwen_vl_prefill/` (committed). Cheap correction is
  near-lossless on VQA/captioning; only RefCOCO grounding pays ~−3pp.
