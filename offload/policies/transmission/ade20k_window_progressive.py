import numpy as np
from typing import Generator, List

from offload.common.protocol import ExperimentConfig, Patch

from .progressive import ProgressiveLPyramidPolicy


class ADE20KWindowProgressiveLaplacianPolicy(ProgressiveLPyramidPolicy):
    """
    ADE20K m2f crop-cover progressive codec.

    Base (group 0) is the unchanged 1/4-res (level-2) full image. Groups 1..N carry full-res
    (level-0) residuals grouped by the m2f sliding-window crops (896 crop / 596 stride): each
    residual patch is assigned to the FIRST (row-major) crop that covers its center, so overlap
    regions go to the earlier crop and group i = crop_i minus crops 1..i-1. Receiving group i
    therefore completes correction of crop i (its overlap with earlier crops already arrived), plus
    the corresponding parts of later crops. N = the per-image sliding-crop count (varies with aspect)
    and is carried on every patch via `num_correction_groups` so the scheduler can chunk layers by N.

    Assumes batch_size == 1 (as ADE20K m2f always runs).
    """

    @staticmethod
    def _patch_hw(config: ExperimentConfig) -> tuple[int, int]:
        ps = config.patch_size
        if isinstance(ps, int):
            return int(ps), int(ps)
        return int(ps[0]), int(ps[1])

    def _crop_params(self, config: ExperimentConfig) -> tuple[int, int]:
        prof = config.get_input_profile_config()
        return int(prof.get("server_crop_size", 896)), int(prof.get("server_stride", 596))

    @staticmethod
    def _compute_crops(h_img: int, w_img: int, crop: int, stride: int) -> List[tuple[int, int, int, int]]:
        # Must mirror dinov3_segmentor_m2f's sliding-window crop layout exactly.
        h_crop = w_crop = crop
        if h_crop > h_img and w_crop > w_img:
            h_crop = w_crop = min(h_img, w_img)
        h_grids = max(h_img - h_crop + stride - 1, 0) // stride + 1
        w_grids = max(w_img - w_crop + stride - 1, 0) // stride + 1
        crops = []
        for hi in range(h_grids):
            for wi in range(w_grids):
                y1 = hi * stride
                x1 = wi * stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crops.append((y1, y2, x1, x2))
        return crops

    def _crops_for_image(self, config: ExperimentConfig, image_hw):
        ph, pw = self._patch_hw(config)
        h0, w0 = self._target_hw_for_level(config, 0, image_hw)  # patch-aligned model-input pixels
        gh, gw = h0 // ph, w0 // pw
        crop, stride = self._crop_params(config)
        return self._compute_crops(gh * ph, gw * pw, crop, stride)

    def _precompute_group_assignments(self, strategy, residual_structure, num_groups):
        if strategy != "crop_cover":
            return super()._precompute_group_assignments(strategy, residual_structure, num_groups)

        structure = list(residual_structure)
        if not structure:
            return np.zeros(0, dtype=int)

        ph, pw = self._patch_hw(self._active_config)
        crops = self._crops_for_image(self._active_config, self._active_image_hw)
        group_ids = np.empty(len(structure), dtype=int)
        for i, item in enumerate(structure):
            r, c = int(item["row"]), int(item["col"])
            cy = r * ph + ph // 2
            cx = c * pw + pw // 2
            g = 0
            for idx, (y1, y2, x1, x2) in enumerate(crops):
                if y1 <= cy < y2 and x1 <= cx < x2:
                    g = idx + 1
                    break
            group_ids[i] = g if g > 0 else len(crops)  # fallback: last crop
        return group_ids

    def encode(self, images: np.ndarray, config: ExperimentConfig) -> Generator[List[Patch], None, None]:
        self._active_config = config
        image_list = self._as_image_list(images)
        preserve = self._is_preserve_input_shape(config)
        # crop-cover assumes batch=1; the crop layout is taken from image 0.
        self._active_image_hw = tuple(int(v) for v in image_list[0].shape[:2]) if preserve else None
        n = len(self._crops_for_image(config, self._active_image_hw))

        for group in super().encode(images, config):
            for p in group:
                p.num_correction_groups = n
            yield group
