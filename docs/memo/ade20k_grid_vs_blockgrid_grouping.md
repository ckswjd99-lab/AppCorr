# ADE20K m2f — interleaved `grid` vs `block_grid` grouping

**Setup:** same as the SR sweep — DINOv3 ViT-7B + m2f head, ADE20K val, sliding-window (896 crop /
596 stride), interleaved-static (`GroupTrigger`, ProgressiveLaplacian, `pyramid_levels=[2,0]`,
`num_groups=4`), partial-token correction, `token_keep_thres=0.0001`. Only `grouping_strategy`
changes.

- **grid** (default): 2×2 pattern `[[1,2],[3,4]]` *tiled* across the token grid → each of the 4 groups
  is a spatially **dispersed** 1/4 subsample (Bayer-like), so every region has 1/4 of its tokens in
  each group.
- **block_grid**: 4 **contiguous quadrants** (top-left / top-right / bottom-left / bottom-right).

> **Both numbers below predate the persist fix (2026-08-16, `96889a5`) and were measured
> while interleaved correction was discarding all but the last round.** `correct_partial_token` read
> `blocks_out_sum` and never wrote back, so a corrected token's own value was rebuilt from the stale
> approximate increment at the next round — see [[vggt_omega_status]] for the mechanism and
> [[dinov3_correct_low_precision_status]] for the re-measured ADE20K arms.
>
> This matters here specifically, because the hypothesis below — that block_grid wins by making
> correction timing *coherent within a crop* — is the same consistency effect the fix addresses. The
> ranking may not survive it. Re-running needs four full-2000 arms (grid and block_grid × flag off
> and on) via `--set transmission_kwargs.grouping_strategy=...`; the per-strategy configs are gone
> but both strategies are still in the transmission policy. Not yet done.
>
> **Two fixes, not one.** These numbers also predate the closed-loop transmission fix
> (2026-08-17, `378e21d`): the `[2,0]` Laplacian round trip lost 1.85% relative L2 on ADE20K even
> with the whole residual sent, because the encoder predicted from the native gaussian while the
> decoder predicted from the resampled base. Correcting both fixes moved ADE20K interleaved from
> 80.8% of the floor-ceiling gap to **93.7%**. Anything measured on this page carries both.

## Results

| grouping | mIoU | recompute | n |
|---|---:|---:|---:|
| grid (interleaved) | 60.74 | 72.88% | 2000 |
| **block_grid (quadrants)** | **61.31** | 74.11% | 2000 |

At matched recompute (grid slope ~0.026 mIoU/pp near 73% → grid@74.1% ≈ 60.77), **block_grid is
~+0.5 mIoU better**.

**Caution / lesson:** at **nr=100** the ranking was REVERSED (block_grid 52.26 @75.0% vs grid 53.40
@73.3%) — pure small-sample noise (the first 100 val images happened to favor grid). Only the full
2000-image numbers are trustworthy for a ~0.5pp effect. (See also
[[feedback_nr400_sanity_check_only]] — same lesson.)

## How block_grid interacts with sliding-window (the tricky part)

Grouping is assigned per sliding **crop** but from **global image coordinates**, in
`dinov3_segmentor_m2f._build_crop_block_grid_group_map`:

- Each crop token's global pixel center = `crop_offset(y1,x1) + local_idx*patch + patch/2`; horizontal
  flip TTA mirrors the column; then the token's global patch (row,col) is bucketed into the 2×2 block
  of the *full-image* grid.
- Consequences: (1) **overlap-consistent** — a physical patch in the crop overlap gets the SAME
  quadrant in every crop that contains it (would break if quadrants were per-crop local); (2) a crop
  can hold **1–4 groups** depending on where it sits (corner crop = 1 group, central crop = up to 4).

## Why block_grid wins (hypothesis)

The interleaved schedule corrects group `g` at layer-chunk `g`. So:
- **grid**: every crop contains all 4 groups → at each chunk only ~1/4 of a crop's tokens are fresh,
  3/4 stale — attention always runs on a *mixed fresh/stale* feature map, every chunk.
- **block_grid**: a corner crop is (almost) all one group → its tokens are corrected **together at one
  chunk**, so within a crop the correction timing is *coherent* and attention sees a more internally
  consistent feature map.

Coherent per-crop correction timing seems to help the m2f head / attention slightly (+0.5 mIoU),
despite the extreme cross-crop timing imbalance (top-left crops corrected early, bottom-right crops
only at the last chunk). This is the opposite of the initial (nr=100, wrong) intuition that the
timing imbalance would hurt.

## Config

Historical: the `grid` / `block_grid` interleaved-static configs (`ProgressiveLaplacian`,
`grouping_strategy: "grid"` / `"block_grid"`, `num_groups: 4`). These per-strategy configs were
removed when the ADE20K m2f configs were consolidated (crop_cover superseded block_grid on the
frontier — see [[ade20k_cropcover_grouping_sweep]]); reproduce by setting
`transmission_kwargs.grouping_strategy` on a `ProgressiveLaplacian` + `GroupTrigger` config.

## Next idea (not yet run)

"Promote low-volume (low-residual, cheaply-compressed) patches into group 0 (base burst) and finalize
them at input" — heavier group 0, same grid/block_grid for groups 1..N, but those groups become
sparser (low-residual patches already done) → less recompute per group. Would layer on top of
block_grid.
