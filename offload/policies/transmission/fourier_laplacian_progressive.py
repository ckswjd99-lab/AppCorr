import math
import zlib
from typing import Dict, Generator, List

import cv2
import numpy as np

from offload.common.protocol import ExperimentConfig, Patch
from .progressive import ProgressiveLPyramidPolicy
from .fourier_laplacian_hybrid import _pack_base_payload, _unpack_base_payload


class FourierLaplacianProgressivePolicy(ProgressiveLPyramidPolicy):
    """
    General-purpose sibling of FourierLaplacianHybridPolicy (which is COCO
    detector / windowed-specific): group 0 is a whole-image 2D DCT low-pass
    reconstruction instead of a Gaussian-pyramid downsample (see
    fourier_laplacian_hybrid.py's docstring for why: no block-grid artifacts,
    smaller + faster than either the Gaussian pyramid or per-patch DCT). The
    residual (image - DCT prediction) is split into transmission groups using
    ProgressiveLPyramidPolicy's existing, *unmodified* grouping-strategy
    machinery (_precompute_group_assignments: grid/uniform/geometric), so
    num_groups=1 (ADE20K/NYU-style single correction group) and num_groups>1
    (ImageNet-style multi-group grid) both work without new grouping logic —
    grid with num_groups=1 degenerates to a single group automatically.

    Strictly assumes a 2-level pyramid_levels=[<base_level>, 0] (base +
    residual) — matches every existing *_appcorr.json / *_interleaved*.json
    config in this repo. Supports preserve_input_shape (per-image variable
    target resolution), same convention as FourierProgressiveTransmissionPolicy.
    """

    def _keep_hw(self, config: ExperimentConfig, image_hw) -> tuple[int, int, int, int]:
        H, W = self._target_hw_for_level(config, 0, image_hw)
        levels = sorted(config.transmission_kwargs.get('pyramid_levels', [2, 0]), reverse=True)
        base_level = levels[0]
        keep_h = min(max(math.ceil(H / (2 ** base_level)), 1), H)
        keep_w = min(max(math.ceil(W / (2 ** base_level)), 1), W)
        return H, W, keep_h, keep_w

    @staticmethod
    def _pyramid_base_level(config: ExperimentConfig) -> int:
        levels = sorted(config.transmission_kwargs.get('pyramid_levels', [2, 0]), reverse=True)
        return levels[0]

    @staticmethod
    def _dct_base_encode(image: np.ndarray, keep_h: int, keep_w: int) -> tuple[np.ndarray, np.ndarray]:
        H, W, C = image.shape
        coeff_stack = np.empty((C, keep_h, keep_w), dtype=np.float32)
        pred = np.empty((H, W, C), dtype=np.float32)
        for c in range(C):
            coeff = cv2.dct(np.ascontiguousarray(image[:, :, c], dtype=np.float32))
            coeff_stack[c] = coeff[:keep_h, :keep_w]
            full = np.zeros((H, W), dtype=np.float32)
            full[:keep_h, :keep_w] = coeff_stack[c]
            pred[:, :, c] = cv2.idct(full)
        pred_uint8 = np.clip(np.rint(pred), 0, 255).astype(np.uint8)
        return coeff_stack, pred_uint8

    @staticmethod
    def _dct_base_decode(coeff_stack: np.ndarray, H: int, W: int, keep_h: int, keep_w: int) -> np.ndarray:
        C = coeff_stack.shape[0]
        rec = np.empty((H, W, C), dtype=np.float32)
        for c in range(C):
            full = np.zeros((H, W), dtype=np.float32)
            full[:keep_h, :keep_w] = coeff_stack[c]
            rec[:, :, c] = cv2.idct(full)
        return np.clip(np.rint(rec), 0, 255).astype(np.uint8)

    def _get_base_reconstruction(self, patch: Patch, H: int, W: int) -> np.ndarray:
        """Memoized on the Patch object (see fourier_laplacian_hybrid.py)."""
        cached = getattr(patch, '_fdb_base_rec', None)
        if cached is not None and cached.shape[:2] == (H, W):
            return cached
        info = _unpack_base_payload(zlib.decompress(patch.data))
        rec = self._dct_base_decode(info['coeff_stack'], H, W, info['keep_h'], info['keep_w'])
        patch._fdb_base_rec = rec
        return rec

    # --- Encoding -------------------------------------------------------------

    def encode(self, images: np.ndarray, config: ExperimentConfig) -> Generator[List[Patch], None, None]:
        image_list = self._as_image_list(images)
        B = len(image_list)
        num_groups = max(int(config.transmission_kwargs.get('num_groups', 1)), 1)
        grouping_strategy = config.transmission_kwargs.get('grouping_strategy', 'grid')
        comp_lvl = config.transmission_kwargs.get('compression_level', 1)
        mobile_pscore = self._resolve_mobile_pscore(config)
        preserve = self._is_preserve_input_shape(config)
        ph, pw = config.patch_size
        base_level = self._pyramid_base_level(config)

        base_patches: List[Patch] = []
        residual_crops_per_image: List[np.ndarray] = []
        structures: List[List[dict]] = []
        grid_dims: List[tuple[int, int]] = []

        for b_idx, image in enumerate(image_list):
            image_hw = image.shape[:2] if preserve else None
            H, W, keep_h, keep_w = self._keep_hw(config, image_hw)
            resized = self._resize_to_hw(image, (H, W), np.uint8)
            C = resized.shape[-1]

            coeff_stack, pred = self._dct_base_encode(resized, keep_h, keep_w)
            payload = _pack_base_payload(C, H, W, keep_h, keep_w, coeff_stack)
            compressed = zlib.compress(payload, level=comp_lvl)
            base_patches.append(Patch(b_idx, 0, compressed, base_level, 0))

            if H % ph != 0 or W % pw != 0:
                raise ValueError(f"[FourierLaplacianProgressive] shape {(H, W)} not divisible by patch {(ph, pw)}")
            gh, gw = H // ph, W // pw
            residual = resized.astype(np.int16) - pred.astype(np.int16)
            crops = (
                residual.reshape(gh, ph, gw, pw, C)
                .transpose(0, 2, 1, 3, 4)
                .reshape(-1, ph, pw, C)
            )
            residual_crops_per_image.append(crops)
            grid_dims.append((gh, gw))
            structures.append([
                {'spatial_idx': i, 'res_level': 0, 'grid_hw': (gh, gw), 'row': i // gw, 'col': i % gw}
                for i in range(gh * gw)
            ])

        for p in base_patches:
            p.batch_group_total = len(base_patches)
        yield base_patches  # Yield group 0 (whole-image DCT base) immediately.

        if config.transmission_kwargs.get('base_only', False):
            # Ablation mode: base only, no residual correction — mirrors
            # *_approx_only_l2.json's methodology for a "fast global pass
            # alone" measurement, using this policy's own DCT base instead of
            # a Gaussian pyramid level.
            return

        group_assignments = [
            self._precompute_group_assignments(grouping_strategy, structures[b_idx], num_groups)
            for b_idx in range(B)
        ]
        for group_id in range(1, num_groups + 1):
            group_patches: List[Patch] = []
            for b_idx, crops in enumerate(residual_crops_per_image):
                assign = group_assignments[b_idx]
                for i in range(crops.shape[0]):
                    if int(assign[i]) != group_id:
                        continue
                    crop = np.ascontiguousarray(crops[i], dtype=np.int16)
                    data = zlib.compress(crop.tobytes(), level=comp_lvl)
                    group_patches.append(Patch(
                        b_idx, i, data, 0, group_id,
                        pscore_hint=self._compute_patch_pscore_hint(crop, mobile_pscore),
                    ))
            if not group_patches:
                continue
            for p in group_patches:
                p.batch_group_total = len(group_patches)
            yield group_patches

    # --- Decoding ---------------------------------------------------------------

    def decode_lowres(self, patches: List[Patch], config: ExperimentConfig):
        B = config.batch_size
        C = config.image_shape[2]
        preserve = self._is_preserve_input_shape(config)
        target_shapes = self._collect_target_shapes(patches, B) if preserve else [None] * B
        base_level = self._pyramid_base_level(config)

        base_by_batch: Dict[int, Patch] = {}
        for p in patches:
            if (p.group_id == 0 or p.res_level == base_level) and 0 <= p.image_idx < B:
                base_by_batch.setdefault(p.image_idx, p)

        results = []
        for b in range(B):
            H, W = self._target_hw_for_level(config, 0, target_shapes[b])
            base_patch = base_by_batch.get(b)
            if base_patch is None:
                results.append(np.zeros((H, W, C), dtype=np.uint8))
            else:
                results.append(self._get_base_reconstruction(base_patch, H, W))

        if preserve:
            return results
        return np.stack(results) if results else np.zeros((B, *config.image_shape), dtype=np.uint8)

    def decode(self, patches: List[Patch], config: ExperimentConfig, canvas: np.ndarray = None):
        B = config.batch_size
        C = config.image_shape[2]
        preserve = self._is_preserve_input_shape(config)
        target_shapes = self._collect_target_shapes(patches, B) if preserve else [None] * B
        base_level = self._pyramid_base_level(config)

        base_by_batch: Dict[int, Patch] = {}
        residual_by_batch: Dict[int, List[Patch]] = {b: [] for b in range(B)}
        for p in patches:
            if not (0 <= p.image_idx < B):
                continue
            if p.group_id == 0 or p.res_level == base_level:
                base_by_batch.setdefault(p.image_idx, p)
            else:
                residual_by_batch[p.image_idx].append(p)

        results = []
        for b in range(B):
            H, W = self._target_hw_for_level(config, 0, target_shapes[b])
            base_patch = base_by_batch.get(b)
            if base_patch is None:
                results.append(np.zeros((H, W, C), dtype=np.uint8))
                continue
            pred = self._get_base_reconstruction(base_patch, H, W)
            residuals = residual_by_batch[b]
            if not residuals:
                results.append(pred)
                continue
            residual = np.zeros((H, W, C), dtype=np.int16)
            for p in residuals:
                self._place_patch(residual, p, config, np.int16)
            results.append(np.clip(pred.astype(np.int16) + residual, 0, 255).astype(np.uint8))

        if preserve:
            return results
        return np.stack(results)

    @staticmethod
    def _collect_target_shapes(patches: List[Patch], B: int) -> List:
        target_shapes = [None] * B
        for p in patches:
            if p.target_shape and 0 <= p.image_idx < B and target_shapes[p.image_idx] is None:
                target_shapes[p.image_idx] = p.target_shape
        return target_shapes
