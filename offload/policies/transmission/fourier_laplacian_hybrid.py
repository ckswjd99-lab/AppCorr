import math
import struct
import zlib
from typing import Dict, Generator, List

import cv2
import numpy as np

from offload.common.protocol import ExperimentConfig, Patch
from .coco_window_progressive import COCOWindowProgressiveLaplacianPolicy

# --- Wire format for the group 0 (whole-image DCT base) payload -------------
# One patch per image, holding all channels' truncated DCT coefficients as a
# single [C, keep_h, keep_w] block. Unlike FourierProgressive's per-16x16-patch
# DCT (where |coefficient| stays under ~4100 and a flat int16 round-trip is
# safe), a *whole-image* DCT's coefficients scale with sqrt(H*W): for a
# 1024x1024 image the DC term alone can exceed 150000, ~4-5x past int16 range.
# A flat float32 payload fixed that but turned out not "negligible" at all —
# ~730KB per image, ~27% of total bytes in practice. Instead we store int16
# with a single adaptive per-payload scale factor (chosen so the largest
# coefficient just fits int16's range), i.e. a 16-bit float-like encoding: full
# use of the 16-bit mantissa, quantization error only in the low bits.
_MAGIC = b'FDB1'
_HEADER_FMT = '<4sBBHHHHBIf'  # magic, version, C, H, W, keep_h, keep_w, dtype_code, n_values, scale
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_DTYPE_INT16 = 1
_INT16_LIMIT = float(np.iinfo(np.int16).max) - 1.0


def _pack_base_payload(C: int, H: int, W: int, keep_h: int, keep_w: int, coeff_stack: np.ndarray) -> bytes:
    max_abs = float(np.max(np.abs(coeff_stack))) if coeff_stack.size else 0.0
    scale = max(1.0, max_abs / _INT16_LIMIT)
    arr = np.clip(np.rint(coeff_stack / scale), -_INT16_LIMIT, _INT16_LIMIT).astype(np.int16)
    arr = np.ascontiguousarray(arr)
    header = struct.pack(_HEADER_FMT, _MAGIC, 1, C, H, W, keep_h, keep_w, _DTYPE_INT16, arr.size, scale)
    return header + arr.tobytes()


def _unpack_base_payload(data: bytes) -> Dict:
    magic, version, C, H, W, keep_h, keep_w, dtype_code, n_values, scale = struct.unpack_from(_HEADER_FMT, data, 0)
    if magic != _MAGIC:
        raise ValueError("FourierLaplacianHybrid: corrupt base payload (bad magic)")
    if dtype_code != _DTYPE_INT16:
        raise ValueError(f"FourierLaplacianHybrid: unsupported base dtype code {dtype_code}")
    raw = np.frombuffer(data, dtype=np.int16, count=n_values, offset=_HEADER_SIZE)
    values = raw.astype(np.float32) * scale
    coeff_stack = values.reshape(C, keep_h, keep_w)
    return {'C': C, 'H': H, 'W': W, 'keep_h': keep_h, 'keep_w': keep_w, 'coeff_stack': coeff_stack}


class FourierLaplacianHybridPolicy(COCOWindowProgressiveLaplacianPolicy):
    """
    group 0: a single *whole-image* 2D DCT low-pass reconstruction (per
    channel, not per 16x16 patch). Measured against this project's original
    per-patch DCT truncation and against the parent class's Gaussian/window
    downsample: whole-image DCT truncation has no per-patch block boundaries
    (so no block-grid artifact), is smaller (concentrates energy globally
    instead of paying a per-patch DC/low-freq cost 4096 times), and is much
    faster to encode/decode (one big transform instead of thousands of small
    ones). See dct_comparison_examples/full_image_dct_test/ for the
    measurements this is based on.

    groups 1..9: byte-for-byte the same windowed pixel-domain residual
    correction as the parent COCOWindowProgressiveLaplacianPolicy (inherited
    unchanged) — already proven to reach baseline detection accuracy — just
    predicted from the DCT base instead of a resized/blurred base. A DCT
    coefficient isn't spatially localized (every coefficient contributes to
    every pixel), so it can't itself be split into per-window correction
    groups the way pixel residuals or per-patch DCT residuals can; the DCT
    trick is only used for the single global group 0 approximation.

    keep_h/keep_w (the retained low-frequency square) default to being derived
    from transmission_kwargs.pyramid_levels exactly like FourierProgressive's
    per-patch low_h/low_w, just applied to the full image_shape instead of a
    single patch: keep = ceil(image_dim / 2**base_level). transmission_kwargs.
    dct_keep overrides this with an exact (h, w) or scalar value; the special
    string dct_keep="window" instead matches the parent class's group-0 base
    size exactly (`_base_hw`: one 3x3-window cell, ~1/3 of each image
    dimension, patch-aligned) — useful when the DCT base is meant to
    arrive as/before the global window does and be immediately followed by
    per-window correction, so its resolution tracks that window's own scale
    rather than an independent pyramid_levels setting.
    """

    @classmethod
    def _keep_hw(cls, config: ExperimentConfig) -> tuple[int, int]:
        explicit = config.transmission_kwargs.get('dct_keep')
        if explicit == 'window':
            return cls._base_hw(config)
        if explicit:
            if isinstance(explicit, (list, tuple)):
                return int(explicit[0]), int(explicit[1])
            return int(explicit), int(explicit)
        H, W = config.image_shape[:2]
        levels = sorted(config.transmission_kwargs.get('pyramid_levels', [2, 0]), reverse=True)
        base_level = levels[0]
        keep_h = min(max(math.ceil(H / (2 ** base_level)), 1), H)
        keep_w = min(max(math.ceil(W / (2 ** base_level)), 1), W)
        return keep_h, keep_w

    @staticmethod
    def _dct_base_encode(image: np.ndarray, keep_h: int, keep_w: int) -> tuple[np.ndarray, np.ndarray]:
        """Returns (coeff_stack [C,keep_h,keep_w] float32, full-size uint8 prediction)."""
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
        """Memoized on the Patch object: group 0 never changes across decode()
        calls for a given request, so the (only moderately cheap, but still
        avoidable) IDCT only needs to run once."""
        cached = getattr(patch, '_fdb_base_rec', None)
        if cached is not None:
            return cached
        info = _unpack_base_payload(zlib.decompress(patch.data))
        rec = self._dct_base_decode(info['coeff_stack'], H, W, info['keep_h'], info['keep_w'])
        patch._fdb_base_rec = rec
        return rec

    # --- Encoding -------------------------------------------------------------

    def encode(self, images: np.ndarray, config: ExperimentConfig) -> Generator[List[Patch], None, None]:
        image_list = self._as_image_list(images)
        comp_lvl = config.transmission_kwargs.get('compression_level', 1)
        mobile_pscore = self._resolve_mobile_pscore(config)
        keep_h, keep_w = self._keep_hw(config)

        projected_images = [self._project_to_model_grid(image, config) for image in image_list]

        ph, pw = self._patch_hw(config)
        h, w, c = config.image_shape
        full_grid_w = w // pw
        all_h, all_w, h_cum, w_cum = self._window_slices(config)

        base_patches: List[Patch] = []
        preds: List[np.ndarray] = []
        for b_idx, image in enumerate(projected_images):
            coeff_stack, pred = self._dct_base_encode(image, keep_h, keep_w)
            preds.append(pred)
            payload = _pack_base_payload(c, h, w, keep_h, keep_w, coeff_stack)
            compressed = zlib.compress(payload, level=comp_lvl)
            base_patches.append(Patch(b_idx, 0, compressed, self._BASE_RES_LEVEL, 0))

        for patch in base_patches:
            patch.batch_group_total = len(base_patches)
        yield base_patches  # Yield group 0 (whole-image DCT base) immediately.

        if config.transmission_kwargs.get('base_only', False):
            # Ablation mode: send only the global DCT approximation, no windowed
            # correction — mirrors Laplacian/pyramid_levels=[<lvl>] with no 0
            # level (e.g. coco_approx_only_l2.json), for a "what does the fast
            # global pass alone get you" comparison against a single-shot
            # FULL_INFERENCE scheduler (BatchCountBased), independent of the
            # windowed appcorr correction pipeline entirely.
            return

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

    # --- Decoding ---------------------------------------------------------------

    def decode_lowres(self, patches: List[Patch], config: ExperimentConfig) -> np.ndarray:
        B = config.batch_size
        H, W, C = config.image_shape
        results = np.zeros((B, H, W, C), dtype=np.uint8)
        for p in patches:
            if (p.group_id == 0 or p.res_level == self._BASE_RES_LEVEL) and 0 <= p.image_idx < B:
                results[p.image_idx] = self._get_base_reconstruction(p, H, W)
        return results

    def decode(self, patches: List[Patch], config: ExperimentConfig, canvas: np.ndarray = None) -> np.ndarray:
        B = config.batch_size
        H, W, C = config.image_shape

        base_by_batch: Dict[int, Patch] = {}
        residual_by_batch: Dict[int, List[Patch]] = {b: [] for b in range(B)}
        for p in patches:
            if not (0 <= p.image_idx < B):
                continue
            if p.group_id == 0 or p.res_level == self._BASE_RES_LEVEL:
                base_by_batch.setdefault(p.image_idx, p)
            elif p.res_level == self._RESIDUAL_RES_LEVEL:
                residual_by_batch[p.image_idx].append(p)

        final_images = np.zeros((B, H, W, C), dtype=np.uint8)
        for b in range(B):
            base_patch = base_by_batch.get(b)
            if base_patch is None:
                continue
            pred = self._get_base_reconstruction(base_patch, H, W)
            residuals = residual_by_batch[b]
            if not residuals:
                final_images[b] = pred
                continue
            residual = np.zeros((H, W, C), dtype=np.int16)
            for p in residuals:
                self._place_patch(residual, p, config, np.int16)
            final_images[b] = np.clip(pred.astype(np.int16) + residual, 0, 255).astype(np.uint8)

        return final_images
