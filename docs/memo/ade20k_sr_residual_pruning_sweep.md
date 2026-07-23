# ADE20K m2f — SR base & SR-residual pruning threshold sweep

**Model:** DINOv3 ViT-7B + Mask2Former head (`dinov3_segmentor_m2f`), backbone
`dinov3_vit7b16_pretrain_lvd1689m`, head `dinov3_vit7b16_ade20k_m2f_head`.
**Dataset:** ADE20K (HF `merve/scene_parse_150`, `validation` split, **2000 images**, 150 classes),
sliding-window inference at 896 crop / 596 stride, `server_rescale_to=original`.
**Metric:** mIoU (full 2000-image val).
**Transmission:** ProgressiveLaplacian, `pyramid_levels=[2,0]`, grid grouping, `num_groups=4`,
interleaved-static schedule (`GroupTrigger`). Correction method = `partial_token`.

## What was tested

Five configurations, plus a threshold sweep over the correction budget ("recompute rate" = fraction
of patches actually re-computed during correction, controlled by `appcorr_kwargs.token_keep_thres`):

1. **baseline** — exact full-image inference (`ade20k_m2f_sequential.json`, Raw transmission). Upper bound.
2. **approx_only** — low-res approx, no correction (`ade20k_m2f_approx_only_l2.json`). Lower bound (0% recompute).
3. **appcorr** — interleaved-static partial-token correction (`ade20k_m2f_interleaved_static.json`).
4. **appcorr+SR** — same, but the group-0 base image is Real-ESRGAN-upscaled before the approx pass
   (`ade20k_m2f_interleaved_static_sr.json`, `scheduler_kwargs.lowres_sr=true`,
   model `realesr_general_x4v3`).
5. **appcorr+SR+residprune** — appcorr+SR, but the partial-token **pruning signal** (which tokens to
   correct) is driven by a **server-side |real-HR − SR-base| per-patch residual** instead of the
   mobile's SR-unaware transmitted-refinement energy (`sr_residual_pscore=true`,
   `ade20k_m2f_interleaved_static_sr_residprune.json`).

The pscore that drives pruning is, for all configs: `pscore = √(residual · avg_attn)` (geo-mean of a
patch-attention "server" score and a residual "mobile" score, each L1-normalized to sum 1). Only the
**residual term** differs between appcorr(+SR) and residprune (mobile transmitted-refinement energy
vs server SR-vs-HR residual).

## Anchors (full 2000)

| config | recompute | mIoU |
|---|---:|---:|
| baseline (exact) | 100% | **62.24** |
| approx_only (no correction) | 0% | **55.97** |

## Threshold sweep — mIoU @ measured recompute rate (full 2000)

Thresholds were calibrated per config (the same threshold gives different recompute rates because the
pscore distributions differ). x-axis is the **measured** average recompute rate.

| target | appcorr | appcorr+SR | appcorr+SR+**residprune** |
|---|---:|---:|---:|
| ~70% | 60.74 (72.9%) | 60.55 (72.9%) | 60.45 (70.4%) |
| ~57% | 60.34 (57.5%) | 60.24 (57.9%) | 60.13 (58.5%) |
| ~42% | 59.26 (40.8%) | 59.02 (41.8%) | 59.57 (45.3%) |
| ~28% | 58.06 (25.1%) | 58.31 (26.6%) | **58.83 (29.6%)** |
| ~11% | 57.49 (10.0%) | 56.76 (11.4%) | 56.78 (11.7%) |

### residprune vs appcorr at matched recompute (appcorr interpolated to residprune's recompute %)

| residprune point | recompute | residprune mIoU | appcorr (interp) | Δ |
|---|---:|---:|---:|---:|
| t70 | 70.4% | 60.45 | ~60.67 | −0.22 |
| t55 | 58.5% | 60.13 | ~60.37 | −0.24 |
| t40 | 45.3% | 59.57 | ~59.55 | +0.02 |
| t25 | 29.6% | 58.83 | ~58.40 | **+0.43** |
| t12 | 11.7% | 56.78 | ~57.55 | **−0.77** |

## Findings

1. **SR-residual pruning helps only in a narrow mid-low band.** residprune beats plain appcorr at
   matched recompute only around **~30% recompute (+0.43 mIoU)**, ties at ~45%, and is **slightly
   worse at high recompute (55–70%, ≈−0.2)** and **clearly worse at very-low recompute (~12%,
   −0.77)**. It is not a global win. The user's hypothesis (SR-base-vs-HR residual is a better
   "which tokens to fix" signal) holds *only* where the correction budget is moderate-to-tight but
   not extreme.
2. **SR base alone (appcorr vs appcorr+SR) is ≈ a wash** for segmentation mIoU — appcorr is a hair
   higher at most points; appcorr+SR is slightly better at ~25% and notably worse at ~12%. A better
   group-0 base does not, by itself, move final mIoU.
3. All three trade off monotonically between baseline (62.24) and approx_only (55.97) as recompute
   drops, as expected.

## Why the residual signal is less selective (mechanism)

- pscore maps are normalized to sum=1, so only the **shape/selectivity** of the residual matters.
- The mobile transmitted-refinement energy `|HR − low-res base|` is concentrated on edges/high-freq
  patches → very selective → prunes flat regions well.
- The server `|HR − SR base|` residual is spread more uniformly (SR errors are diffuse), so at a fixed
  budget it is a *less* peaked "importance" signal, except in the mid-low band where its content-aware
  errors line up with what actually needs correcting.

## Implementation notes (this feature, on top of the existing partial-token path)

- `offload/server/worker.py` — group-0 lowres-SR base: super-resolve `input_lr_native` (which carries
  the true image aspect) then interpolate to the **original image size** (`target_shapes[0]`) so the SR
  base's token grid matches the real image's sliding-window layout; scale [0,1]→[0,255].
- `offload/server/model/dinov3_segmentor_m2f.py` — `_build_mobile_pscore_hint_maps` gains
  `sr_residual_pscore`: when set, the per-patch mobile hint is replaced by the server-computed
  per-patch RMS of `(real-HR image − SR base)` (resized/pooled to the patch grid).
- `offload/mobile/dataset.py` — `_intersect_and_union` now resizes a size-mismatched prediction to the
  GT grid (nearest) instead of raising. SR-path predictions can drift by ≤1 patch (SR base vs real
  token-grid rounding); this only affects the SR configs (plain appcorr never mismatches).
- Env: `realesrgan`/`basicsr` need a `torchvision.transforms.functional_tensor` shim on
  torchvision≥0.17 (added to the `appcorr` conda env's site-packages, not the repo). SR weight
  `realesr-general-x4v3.pth` in `~/cjpark/weights/realesrgan/`.
- New configs: `ade20k_m2f_interleaved_static_sr.json`, `ade20k_m2f_approx_only_l2_sr.json`,
  `ade20k_m2f_interleaved_static_sr_residprune.json`.

## Reproduce

Per run (full val): `offload/run_local.sh <config> -nw 0 --set appcorr_kwargs.token_keep_thres=<T>`
in the `appcorr` conda env. Sweep driver + raw per-job logs were in the session scratchpad
(`sweep.sh`, `results_gpu0.txt`, `results_gpu1.txt`).
