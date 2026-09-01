# ADE20K m2f — `crop_cover` grouping + per-image variable group count (sweep)

> **Pre-fix numbers.** Every interleaved-correction figure here was measured before
> the persist fix (2026-08-16, `96889a5`), while interleaved correction discarded
> all but the last round. On ADE20K the same configuration moved 61.042 -> 61.597 mIoU once the
> corrected increment was persisted (80.8% -> 89.7% of the floor-ceiling gap). Comparisons *within*
> this memo shared the defect and are likely to survive it; the absolute values will not. See
> [[dinov3_correct_low_precision_status]].
>
> **Two fixes, not one.** These numbers also predate the closed-loop transmission fix
> (2026-08-17, `378e21d`): the `[2,0]` Laplacian round trip lost 1.85% relative L2 on ADE20K even
> with the whole residual sent, because the encoder predicted from the native gaussian while the
> decoder predicted from the resampled base. Correcting both fixes moved ADE20K interleaved from
> 80.8% of the floor-ceiling gap to **93.7%**. Anything measured on this page carries both.

**Setup:** same backbone/data as the grid/block_grid memo — DINOv3 ViT-7B + m2f head, ADE20K val
2000, sliding-window (896 crop / 596 stride), partial-token correction (`mobile_pscore=residual_energy`,
`server_pscore=patch_attn_prob_layermean`, `pscore_fusion=geo_mean`), base = unchanged 1/4-res
(`pyramid_levels=[2,0]`). What changes: a **dedicated ADE20K m2f crop-cover policy** (analogous to the
COCO window policy) instead of the fixed 4-group grid/block_grid.

- **transmission** `ADE20KWindowProgressiveLaplacian` (`grouping_strategy="crop_cover"`): each level-0
  residual patch is assigned to the **first (row-major) sliding crop that covers its center**. So group
  `i` = crop `i` minus crops `1..i-1`; receiving group `i` completes correction of crop `i`. The number
  of groups **N = the per-image sliding-crop count** (varies with aspect ratio), carried on every patch
  via the new `Patch.num_correction_groups` field.
- **scheduler** `ADE20KWindowInterleaved`: reads N per image, splits the 40 layers into N even chunks,
  corrects group `g` up to the current approx frontier then advances one chunk; final group (`g==N`)
  corrects the whole model + head. Structurally = GroupTrigger but N is dynamic per image.
- server `dinov3_segmentor_m2f._build_crop_cover_group_map` mirrors the same first-cover rule from
  **global** pixel coords, so an overlap patch gets the same group in every crop that contains it.

## Results — full-2000 recompute sweep

| thres | recompute | mIoU |
|---|---:|---:|
| 5e-4 | 8.0%  | 57.98 |
| 2e-4 | 28.5% | 60.21 |
| 1e-4 | 42.0% | 60.82 |
| 4e-5 | 54.3% | 61.03 |
| 5e-7 | 67.1% | 61.25 |

Clean monotone curve. `recompute` = corrected token-ops / full sliding-window token-ops, **both summed
over crops (overlap counted per crop)** — the honest FLOP fraction (`_aggregate_cache_features` sums the
per-source `_token_prune_{kept,full}_patch_total`).

## vs grid / block_grid (same partial-token config)

| method | recompute | mIoU |
|---|---:|---:|
| grid (sweep, matched) | 40.8% | 59.26 |
| **crop_cover** | **42.0%** | **60.82** (**+1.56** at matched) |
| grid | 72.9% | 60.74 |
| **crop_cover** | **54.3%** | **61.03** (beats grid@73% with ~0.75× the compute) |
| block_grid | 74.1% | 61.31 |
| **crop_cover** | **67.1%** | **61.25** (≈ block_grid with less compute) |

**crop_cover dominates the compute–accuracy frontier**: it matches the best fixed-grouping accuracy
(block_grid 61.31) at meaningfully lower recompute, and at matched ~42% it is +1.56 mIoU over grid.
Aligning the correction groups to the sliding crops (so each crop is corrected coherently *and* in one
transmission group) beats both the dispersed (grid) and the fixed-quadrant (block_grid) schemes.

## Recompute plateaus at ~65–67% (structural, not a bug)

Lowering `token_keep_thres` below ~4e-5 barely raises recompute (2e-5→60%, 6e-6→65%, 5e-7→67%; then
flat). Cause: `mobile_pscore=residual_energy` with `geo_mean` fusion → any patch whose HR residual over
the 1/4-res base is **zero** (flat regions the base already reconstructs exactly, ~1/3 of patches) has
pscore 0 and is **never** corrected, regardless of threshold. So ~65% is crop_cover's natural ceiling
here — i.e. it never spends compute on flat regions. 73% (block_grid's operating point) is unreachable
with this fusion, which is why the sweep tops out at 67%.

## Avg-attention (server_pscore) is per-crop

Each sliding crop is an independent ViT forward, so `patch_attn_prob_layermean` is cached per source
(`all_cache_features[src_idx]`). An overlap patch therefore has a **different** avg-attention in each
crop it belongs to, and the pruning decision is per-crop (it can be kept in one crop, pruned in
another). `mobile_pscore` (global residual energy) is projected onto each crop's local tokens; the fused
pscore differs per crop via the attention term.

## Configs / code

- crop_cover is now the `ade20k_m2f_interleaved_static.json` config (transmission
  `ADE20KWindowProgressiveLaplacian`, scheduler `ADE20KWindowInterleaved`, thres 4e-5 = the ~54% point).
  The per-config sweep files were consolidated away; reproduce the other sweep points with
  `--set appcorr_kwargs.token_keep_thres=<5e-4|2e-4|1e-4|4e-5|5e-7>` (→ ~8 / 28 / 42 / 54 / 67%).
  `ade20k_m2f_interleaved_dynamic.json` runs the same crop_cover transmission under the dynamic
  (approx-ahead) `ADE20KInterleavedDynamic` scheduler.
- `offload/policies/transmission/ade20k_window_progressive.py`,
  `offload/policies/scheduling/ade20k_window_trigger.py`,
  `_build_crop_cover_group_map` in `offload/server/model/dinov3_segmentor_m2f.py`,
  `Patch.num_correction_groups` in `offload/common/protocol.py`.
