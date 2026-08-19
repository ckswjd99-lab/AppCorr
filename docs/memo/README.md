# docs/memo

Catch-all folder for experiment notes, results, and design memos — the place to dump "이것저것"
(findings, sweep tables, design decisions) so they don't get lost in scratch dirs or commit messages.

One markdown file per topic. Keep raw numbers + the conclusion; link to the branch/config that
produced them.

## Index

- [sam3_coco_detector_results.md](sam3_coco_detector_results.md) — the same five arms on SAM 3's
  DETECTOR path (text prompt) over full COCO. 55% recompute recovers 89.2%; `pre_global` reaches
  90.5% at 0.60x compute. Detection loses 1.4x more to approximation than the tracker path does, and
  the AP50/AP75 split shows why: lost *detections*, not blurred boundaries. Also records the
  readout-protocol sweep that replaced the hand-set score threshold.
- [sam3_coco_interleaved_results.md](sam3_coco_interleaved_results.md) — SAM 3 tracker on **full
  COCO val2017** (4952 images), 5 arms on one commit. 55% recompute recovers 92.4% of the
  floor-ceiling gap; interleaved `pre_global` (g=4) matches one-shot to four decimals at 0.60x the
  correction compute and 0.69x wall clock. Includes why deferring SAM 3's global layers (7/15/23/31)
  by one layer wins 30%, and why the wall clock understates the compute saving (launch-bound).
- [interleaved_correction_contract.md](interleaved_correction_contract.md) — **read before writing
  any interleaved correction path.** Four rules every fork has re-broken (correct this round's group
  only, never the accumulated set; the stream is cumulative but the corrected set is not; persist the
  corrected increment; coverage must equal the one-shot set) plus the four gates that catch them.
  Each rule is invisible in one-shot correction, which is why every new fork ships it broken.
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
- [dinov3_correct_low_precision_status.md](dinov3_correct_low_precision_status.md) — opposite
  direction: approx stays bf16, `.correct()` quantized to FP4 instead (new `correct_precision`
  config field). **Full 2000**, 5 arms on one commit: FP4 is near-lossless *both* on correction
  (+0.18 mIoU vs bf16) and on the whole 40-layer forward (−0.07) — ~0.2pp is just the noise floor.
  This **reversed the N=100 read** that FP4 was "correction-specific (−0.61 vs −0.05)". Includes
  the weight-level bf16→NVFP4 error table (~9.4% rel L2, ~20.5 dB SQNR), the paired-A/B protocol
  that caught a plumbing bug faking a "bit-identical" result, reuse-validation of three prior
  full-2000 figures, and the bucketing/padding/dynamo-limit design notes for the actual NVFP4
  speedup work. No latency claim yet.

- [dinov3_exact_decomposition_fp4_features.md](dinov3_exact_decomposition_fp4_features.md) —
  rebuilds approx/correct as an **exact** `(a, d)` decomposition (linear: `d' = W d`, bias cancels;
  nonlinear: `g(a+d) - g(a)`, no Taylor truncation) and measures feature fidelity with NVFP4 placed
  per path. FP4 on the correction delta cuts feature error 29% (ImageNet) / 73% (COCO) vs
  quantizing the whole forward, while FP4 on the base costs the same as quantizing everything — all
  the damage is in the base path. Includes the two telescoping controls that validate the
  implementation and the `‖d‖/‖x‖` measurement explaining COCO's larger gain.

- [dinov3_nvfp4_speedup_gate.md](dinov3_nvfp4_speedup_gate.md) — **negative result**: NVFP4 is not
  worth accelerating on this workload. With MSLK installed and torch.compile working, NVFP4 beats
  bf16 only above M≈2300, but the measured correction-GEMM distribution has median M=1028 and only
  15.7% of calls above the crossover. Enabling it everywhere is 0.53× (90% slower); the best hybrid
  saves 3.9% of correction GEMM time ≈ 0.07% end-to-end. Includes the MSLK install recipe, the
  quant-vs-GEMM cost split (the FP4 GEMM itself *is* 1.3–1.6× faster), and where to look instead.

- [dinov3_correct_forward_profile.md](dinov3_correct_forward_profile.md) — profiles what the
  non-GEMM ~70% of CORRECT_FORWARD is. GPU: index/gather/scatter is the biggest non-GEMM cost (24%
  of wall, more than attention + LayerNorm combined) from packing selected tokens and scattering K/V
  back. CPU: the stage is **launch-bound** — 200 `.item()` calls stall the host ~145 ms against a
  183.8 ms GPU wall. Root cause: the query-plan cache is silently disabled whenever the pscore is not
  a `*_layermean` variant, which is exactly ImageNet's config; enabling it removes 195 of 200 syncs
  and is worth ~10% of the stage — comparable to the whole NVFP4 win, for a config change.

## Related work elsewhere in the repo (not in this folder)

- **DINOv3 CSR** (sparse attention + FFN + token pruning) — branch `develop/dinov3-csr`. ImageNet-1k
  ViT-7B: attention-CSR lossless vs exact (93.55 vs 93.16 top-1), FFN-CSR trades accuracy for
  hidden-sparsity (25%→90.23, 50%→92.19); token pruning works. Configs
  `imnet_interleaved_static_csr*.json`.
- **Qwen2.5-VL progressive prefill** — `analysis/qwen_vl_prefill/` (committed). Cheap correction is
  near-lossless on VQA/captioning; only RefCOCO grounding pays ~−3pp.
