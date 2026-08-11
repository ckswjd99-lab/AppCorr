import math
import zlib
from typing import Generator, List

import cv2
import numpy as np

from offload.common.protocol import ExperimentConfig, Patch

from .progressive import ProgressiveLPyramidPolicy


class COCOWindowProgressiveLaplacianPolicy(ProgressiveLPyramidPolicy):
    """
    COCO detector specific progressive codec.

    Group 0 carries the detector-global input, resized to the same spatial size
    used by DINOv3's global branch. Groups 1..9 carry full-resolution residuals
    aligned to the detector's row-major 3x3 local windows.
    """

    _N_WINDOWS_H = 3
    _N_WINDOWS_W = 3
    _BASE_RES_LEVEL = -1
    _RESIDUAL_RES_LEVEL = 0

    @staticmethod
    def _patch_hw(config: ExperimentConfig) -> tuple[int, int]:
        if isinstance(config.patch_size, int):
            return config.patch_size, config.patch_size
        return int(config.patch_size[0]), int(config.patch_size[1])

    @classmethod
    def _window_slices(cls, config: ExperimentConfig) -> tuple[list[int], list[int], list[int], list[int]]:
        h, w = config.image_shape[:2]
        ph, pw = cls._patch_hw(config)

        if h % ph != 0 or w % pw != 0:
            raise ValueError(
                f"[COCOWindowProgressive] image_shape {(h, w)} must be divisible by patch_size {(ph, pw)}"
            )

        win_h = math.ceil((h // cls._N_WINDOWS_H) / ph) * ph
        win_w = math.ceil((w // cls._N_WINDOWS_W) / pw) * pw
        all_h = [win_h] * (cls._N_WINDOWS_H - 1) + [h - win_h * (cls._N_WINDOWS_H - 1)]
        all_w = [win_w] * (cls._N_WINDOWS_W - 1) + [w - win_w * (cls._N_WINDOWS_W - 1)]

        if any(size <= 0 for size in all_h + all_w):
            raise ValueError(
                f"[COCOWindowProgressive] invalid detector window sizes: rows={all_h}, cols={all_w}"
            )
        if any(size % ph != 0 for size in all_h) or any(size % pw != 0 for size in all_w):
            raise ValueError(
                f"[COCOWindowProgressive] detector window sizes must be patch-aligned: rows={all_h}, cols={all_w}"
            )

        h_cum = [0] + list(np.cumsum(all_h))
        w_cum = [0] + list(np.cumsum(all_w))
        return all_h, all_w, h_cum, w_cum

    @classmethod
    def _base_hw(cls, config: ExperimentConfig) -> tuple[int, int]:
        all_h, all_w, _, _ = cls._window_slices(config)
        return all_h[0], all_w[0]

    @classmethod
    def _project_to_model_grid(cls, image: np.ndarray, config: ExperimentConfig) -> np.ndarray:
        h, w = config.image_shape[:2]
        if image.shape[:2] == (h, w):
            projected = image
        else:
            interpolation = cv2.INTER_AREA if h <= image.shape[0] and w <= image.shape[1] else cv2.INTER_LINEAR
            projected = cv2.resize(image, (w, h), interpolation=interpolation)
        return np.ascontiguousarray(np.clip(projected, 0, 255).astype(np.uint8, copy=False))

    @classmethod
    def _downsample_base(cls, native_image: np.ndarray, config: ExperimentConfig) -> np.ndarray:
        """Group-0 global image: degrade in NATIVE coordinates, hand over at the configured size.

        Takes the **original** image, not the model-canvas projection. That distinction is the whole
        point: the canvas is an upscale of the original (COCO's median 480x640 becomes 1024x1024), so
        downscaling the canvas to the 352x352 the global branch consumes is only a 1.36x/1.82x
        reduction of the original -- an upsample/downsample round trip that leaves the real content
        almost intact. The approximate pass then loses nothing, the residual carries nothing, and the
        approx-only floor measures the same as the full-transmission ceiling (0.7033 vs 0.7034 mAP at
        n=100), which makes the comparison meaningless. See the pyramid contract in AGENTS.md.

        Degrading first at 1/`global_base_downscale` of the *original* makes L0 vs the global a real
        gap: 480x640 -> 160x213 keeps 11% of the pixels rather than 40%. The result is then resized
        to `_base_hw` because the network and the server are both defined by the model's
        configuration -- the transmitted size stays the DINOv3 global input size regardless of how
        hard the content was degraded.
        """
        factor = int(config.transmission_kwargs.get('global_base_downscale', cls._N_WINDOWS_H))
        base_h, base_w = cls._base_hw(config)
        if factor > 1:
            nh, nw = native_image.shape[:2]
            native_image = cls._resize_to_hw(
                native_image,
                (max(1, round(nh / factor)), max(1, round(nw / factor))),
                np.uint8,
            )
        return cls._resize_to_hw(native_image, (base_h, base_w), np.uint8)

    @classmethod
    def _upsample_base(cls, base: np.ndarray, config: ExperimentConfig) -> np.ndarray:
        h, w = config.image_shape[:2]
        return cls._resize_to_hw(base, (h, w), np.uint8)

    @classmethod
    def _window_group_id_for_patch(cls, patch_row: int, patch_col: int, config: ExperimentConfig) -> int:
        ph, pw = cls._patch_hw(config)
        all_h, all_w, h_cum, w_cum = cls._window_slices(config)
        y = patch_row * ph
        x = patch_col * pw

        ih = 0
        while ih + 1 < len(h_cum) and y >= h_cum[ih + 1]:
            ih += 1
        iw = 0
        while iw + 1 < len(w_cum) and x >= w_cum[iw + 1]:
            iw += 1
        if ih >= len(all_h) or iw >= len(all_w):
            raise ValueError(
                f"[COCOWindowProgressive] patch ({patch_row}, {patch_col}) fell outside detector windows"
            )
        return 1 + ih * cls._N_WINDOWS_W + iw

    def encode(self, images: np.ndarray, config: ExperimentConfig) -> Generator[List[Patch], None, None]:
        image_list = self._as_image_list(images)
        comp_lvl = config.transmission_kwargs.get('compression_level', 1)
        mobile_pscore = self._resolve_mobile_pscore(config)

        projected_images = [
            self._project_to_model_grid(image, config)
            for image in image_list
        ]

        base_patches: List[Patch] = []
        bases: List[np.ndarray] = []

        ph, pw = self._patch_hw(config)
        h, w, c = config.image_shape
        full_grid_w = w // pw
        all_h, all_w, h_cum, w_cum = self._window_slices(config)

        # The base is built from the ORIGINAL image, not `projected_images` -- degrading the canvas
        # projection would be degrading something that is already an upscale. The residual below
        # still works in canvas coordinates, because that is the space the server reconstructs in.
        for b_idx, image in enumerate(image_list):
            base = self._downsample_base(image, config)
            bases.append(base)

            base_h, base_w = base.shape[:2]
            base_grid_h, base_grid_w = base_h // ph, base_w // pw
            base_crops = (
                base.reshape(base_grid_h, ph, base_grid_w, pw, c)
                .transpose(0, 2, 1, 3, 4)
                .reshape(-1, ph, pw, c)
            )
            for spatial_idx, crop in enumerate(base_crops):
                base_patches.append(
                    Patch(
                        b_idx,
                        spatial_idx,
                        zlib.compress(crop.astype(np.uint8).tobytes(), level=comp_lvl),
                        self._BASE_RES_LEVEL,
                        0,
                    )
                )

        for patch in base_patches:
            patch.batch_group_total = len(base_patches)
        yield base_patches

        if config.transmission_kwargs.get('base_only', False):
            # Ablation mode: send only the global base, no windowed correction —
            # for measuring "how good is the fast global pass alone" against a
            # single-shot FULL_INFERENCE scheduler (BatchCountBased), the same
            # methodology as coco_approx_only_l2.json but using this policy's
            # actual production base generation instead of a plain pyramid level.
            return

        preds = [self._upsample_base(base, config) for base in bases]
        for group_id in range(1, self._N_WINDOWS_H * self._N_WINDOWS_W + 1):
            group_patches: List[Patch] = []
            window_idx = group_id - 1
            ih, iw = divmod(window_idx, self._N_WINDOWS_W)
            y0, y1 = int(h_cum[ih]), int(h_cum[ih + 1])
            x0, x1 = int(w_cum[iw]), int(w_cum[iw + 1])
            win_h, win_w = int(all_h[ih]), int(all_w[iw])
            win_grid_h, win_grid_w = win_h // ph, win_w // pw

            for b_idx, (image, pred) in enumerate(zip(projected_images, preds)):
                residual = image[y0:y1, x0:x1].astype(np.int16) - pred[y0:y1, x0:x1].astype(np.int16)
                residual_crops = (
                    residual.reshape(win_grid_h, ph, win_grid_w, pw, c)
                    .transpose(0, 2, 1, 3, 4)
                    .reshape(-1, ph, pw, c)
                )
                row_offset = y0 // ph
                col_offset = x0 // pw
                for local_idx, crop in enumerate(residual_crops):
                    local_row, local_col = divmod(local_idx, win_grid_w)
                    spatial_idx = (row_offset + local_row) * full_grid_w + (col_offset + local_col)
                    crop_i16 = np.ascontiguousarray(crop.astype(np.int16, copy=False))
                    group_patches.append(
                        Patch(
                            b_idx,
                            spatial_idx,
                            zlib.compress(crop_i16.tobytes(), level=comp_lvl),
                            self._RESIDUAL_RES_LEVEL,
                            group_id,
                            pscore_hint=self._compute_patch_pscore_hint(crop_i16, mobile_pscore),
                        )
                    )

            if not group_patches:
                continue
            for patch in group_patches:
                patch.batch_group_total = len(group_patches)
            yield group_patches

    def decode(self, patches: List[Patch], config: ExperimentConfig, canvas: np.ndarray = None) -> np.ndarray:
        batch_size = config.batch_size
        h, w, c = config.image_shape
        ph, pw = self._patch_hw(config)
        base_h, base_w = self._base_hw(config)

        final_images = np.zeros((batch_size, h, w, c), dtype=np.uint8) if canvas is None else canvas
        has_base_patch = any(
            0 <= patch.image_idx < batch_size
            and (patch.group_id == 0 or patch.res_level == self._BASE_RES_LEVEL)
            for patch in patches
        )
        if canvas is not None and not has_base_patch:
            self._apply_residual_patches_in_place(final_images, patches, config)
            return final_images

        patches_per_batch = {b_idx: [] for b_idx in range(batch_size)}
        for patch in patches:
            if 0 <= patch.image_idx < batch_size:
                patches_per_batch[patch.image_idx].append(patch)

        for b_idx, batch_patches in patches_per_batch.items():
            if not batch_patches:
                continue

            base = np.zeros((base_h, base_w, c), dtype=np.uint8)
            residual = np.zeros((h, w, c), dtype=np.int16)
            saw_base = False

            for patch in batch_patches:
                if patch.group_id == 0 or patch.res_level == self._BASE_RES_LEVEL:
                    self._place_patch(base, patch, config, np.uint8)
                    saw_base = True
                elif patch.res_level == self._RESIDUAL_RES_LEVEL:
                    self._place_patch(residual, patch, config, np.int16)

            if saw_base:
                pred = self._upsample_base(base, config)
            elif canvas is not None:
                pred = np.ascontiguousarray(canvas[b_idx])
            else:
                pred = np.zeros((h, w, c), dtype=np.uint8)

            final_images[b_idx] = np.clip(pred.astype(np.int16) + residual, 0, 255).astype(np.uint8)

        return final_images

    def _apply_residual_patches_in_place(
        self,
        canvas: np.ndarray,
        patches: List[Patch],
        config: ExperimentConfig,
    ) -> None:
        _, w, c = config.image_shape
        ph, pw = self._patch_hw(config)
        grid_w = (w + pw - 1) // pw

        for patch in patches:
            if patch.res_level != self._RESIDUAL_RES_LEVEL:
                continue
            if not (0 <= patch.image_idx < canvas.shape[0]):
                continue

            row, col = divmod(patch.spatial_idx, grid_w)
            y, x = row * ph, col * pw
            th = min(ph, canvas.shape[1] - y)
            tw = min(pw, canvas.shape[2] - x)
            if th <= 0 or tw <= 0:
                continue

            if hasattr(patch, '_decompressed_cache'):
                raw = patch._decompressed_cache
            else:
                raw = zlib.decompress(patch.data)
                patch._decompressed_cache = raw

            residual = np.frombuffer(raw, dtype=np.int16).reshape(ph, pw, c)
            target = canvas[patch.image_idx, y:y + th, x:x + tw]
            target[...] = np.clip(
                target.astype(np.int16) + residual[:th, :tw],
                0,
                255,
            ).astype(np.uint8)

    def decode_lowres(self, patches: List[Patch], config: ExperimentConfig) -> np.ndarray:
        batch_size = config.batch_size
        base_h, base_w = self._base_hw(config)
        c = config.image_shape[2]
        lowres_images = np.zeros((batch_size, base_h, base_w, c), dtype=np.uint8)
        for patch in patches:
            if 0 <= patch.image_idx < batch_size and (
                patch.group_id == 0 or patch.res_level == self._BASE_RES_LEVEL
            ):
                self._place_patch(lowres_images[patch.image_idx], patch, config, np.uint8)
        return lowres_images
