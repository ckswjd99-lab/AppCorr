# Pyramid content-resolution audit

## Invariant

Spatial preprocessing and pyramid construction have distinct roles:

1. Build every Gaussian/Laplacian level from the original image.
2. Resize each already-constructed level to its transmission/model target.
3. A low-resolution model pass must consume the decoded native low level,
   rather than downsampling a reconstructed high-resolution canvas.

Upsampling a low level to the high-resolution canvas is still valid for
residual reconstruction or for a baseline whose inference contract explicitly
requires a high-resolution input shape. It must not be used as the source of a
later low-resolution model pass.

## DINOv3 offload paths

| Workload/path | Status | Reason |
|---|---|---|
| ADE20K M2F generic/native Laplacian encoding | Correct | `LaplacianPyramidPolicy` calls `cv2.pyrDown` on the original emitted image before `_project_band_to_target`. |
| ADE20K M2F `lowres_expand_once` | Corrected on this branch | Group 0 is decoded separately with `decode_lowres`; the low ViT crops now come directly from that original-derived L1 image. The reconstructed L0 canvas is no longer downsampled to create the low pass. |
| ADE20K M2F L1-only baseline | Correct | The encoder creates native L1 first. Decoding upsamples L1 only because this baseline deliberately runs the ordinary high-token M2F inference contract. |
| ADE20K linear-head configs | Incorrect under the invariant | The configs do not request `emit_original_image`, so `ADE20KLoader` resizes before the transmission policy sees the image. |
| COCO generic `Laplacian` / `ProgressiveLaplacian` configs | Correct | `COCO2017Loader` emits the native image and the common codec constructs the native pyramid before fixed-grid projection. |
| COCO window-specific progressive codec | Corrected on this branch | The detector base content is reduced from the native image first, then the base and L0 are projected independently onto their 352x352 and 1024x1024 model grids. |
| NYUv2 AppCorr Laplacian paths | Correct | `NYUDepthLoader` emits native RGB, and the inherited common encoder constructs the native pyramid before the NYU fixed-grid projection. |
| ImageNet Laplacian/AppCorr paths | Incorrect under the invariant | `ImageNetLoader` applies `Resize` and `CenterCrop` before the transmission policy, so the native image is unavailable when the pyramid is built. |

OpenVLA, pi0-FAST, and NORA are not included in this table because they use
model-specific image preprocessing/progressive-token paths rather than these
DINOv3 Laplacian transmission policies.

## Required follow-ups outside ADE20K M2F

1. ImageNet must expose the native image and apply the same classification
   spatial transform independently to already-created pyramid levels.
2. ADE20K linear-head configs must set `emit_original_image=true` and preserve
   the original target metadata through the existing native Laplacian policy.

These changes require new baselines because they alter the actual image content
seen by the model, even when tensor shapes and network payload structure remain
unchanged.
