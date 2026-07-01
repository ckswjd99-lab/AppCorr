import math
import struct
import zlib
from typing import Dict, List, Generator, Optional, Tuple

import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

from offload.common.protocol import Patch, ExperimentConfig
from .laplacian import LaplacianPyramidPolicy
from .coco_window_progressive import COCOWindowProgressiveLaplacianPolicy

# --- Wire format ------------------------------------------------------------
# Each patch payload is: struct header + raw coefficients, zlib-compressed as a
# single blob. group 0 stores only the low-frequency [low_h, low_w, C] block.
# group >= 1 stores the full [ph, pw, C] array with the low-frequency block
# zeroed out (i.e. exactly the coefficients outside the low-frequency rectangle).
#
# Coefficients are stored as int16 by default (rounded to the nearest integer)
# rather than float32. For uint8 pixel input, cv2.dct's orthonormal scaling
# keeps |coefficient| well under 2**14 for any patch_size used in this project
# (e.g. <= 4080 for 16x16), so int16 rounding is lossless for all practical
# purposes: it only adds sub-integer roundoff (<=0.5 per coefficient) on top of
# the DCT/IDCT float roundoff that already exists, and it halves the payload
# size versus float32 while giving zlib far more redundancy to exploit (many
# high-frequency coefficients quantize to runs of zero/small integers instead
# of noisy mantissa bits). float32 remains available via
# transmission_kwargs.coeff_dtype = "float32" for exact float bit transport.
_MAGIC = b'FPC1'
_HEADER_FMT = '<4sBBHHBHHBI'  # magic, version, group_id, ph, pw, C, low_h, low_w, dtype_code, n_values
_HEADER_SIZE = struct.calcsize(_HEADER_FMT)
_DTYPE_FLOAT32 = 0
_DTYPE_INT16 = 1
_INT16_LIMIT = float(np.iinfo(np.int16).max) - 1.0


def _dct2(channel: np.ndarray) -> np.ndarray:
    """2D DCT of a single-channel patch (JPEG-style, low frequencies top-left)."""
    return cv2.dct(np.ascontiguousarray(channel, dtype=np.float32))


def _idct2(coeff: np.ndarray) -> np.ndarray:
    """Inverse 2D DCT of a single-channel coefficient block."""
    return cv2.idct(np.ascontiguousarray(coeff, dtype=np.float32))


def _split_lowfreq(coeff: np.ndarray, low_h: int, low_w: int) -> np.ndarray:
    """Extract the top-left [low_h, low_w, C] low-frequency block."""
    return np.ascontiguousarray(coeff[:low_h, :low_w, :])


def _merge_coefficients(
    low_block: np.ndarray,
    high_full: Optional[np.ndarray],
    ph: int,
    pw: int,
    C: int,
    low_h: int,
    low_w: int,
) -> np.ndarray:
    """Combine a group0 low-frequency block with a high-frequency array (which
    already has zeros in the low-frequency region) into full [ph, pw, C] DCT
    coefficients."""
    full = high_full.copy() if high_full is not None else np.zeros((ph, pw, C), dtype=np.float32)
    full[:low_h, :low_w, :] = low_block
    return full


def _dtype_code(name: str) -> int:
    if name == 'int16':
        return _DTYPE_INT16
    if name == 'float32':
        return _DTYPE_FLOAT32
    raise ValueError(f"FourierProgressive: unknown coeff_dtype '{name}' (expected 'int16' or 'float32')")


def _pack_payload(group_id: int, ph: int, pw: int, C: int, low_h: int, low_w: int, array: np.ndarray, dtype_code: int) -> bytes:
    if dtype_code == _DTYPE_INT16:
        # Coefficients from an 8-bit-pixel DCT comfortably fit int16; clip
        # defensively instead of raising, since a rare out-of-range value should
        # degrade gracefully (small clipping error) rather than crash encoding.
        arr = np.clip(np.rint(array), -_INT16_LIMIT, _INT16_LIMIT).astype(np.int16)
    else:
        arr = np.ascontiguousarray(array, dtype=np.float32)
    arr = np.ascontiguousarray(arr)
    header = struct.pack(
        _HEADER_FMT, _MAGIC, 1, group_id, ph, pw, C, low_h, low_w, dtype_code, arr.size
    )
    return header + arr.tobytes()


def _unpack_payload(data: bytes) -> Dict:
    magic, version, group_id, ph, pw, C, low_h, low_w, dtype_code, n_values = struct.unpack_from(
        _HEADER_FMT, data, 0
    )
    if magic != _MAGIC:
        raise ValueError("FourierProgressive: corrupt payload (bad magic)")

    if dtype_code == _DTYPE_INT16:
        raw = np.frombuffer(data, dtype=np.int16, count=n_values, offset=_HEADER_SIZE)
        values = raw.astype(np.float32)
    elif dtype_code == _DTYPE_FLOAT32:
        values = np.frombuffer(data, dtype=np.float32, count=n_values, offset=_HEADER_SIZE)
    else:
        raise ValueError(f"FourierProgressive: unsupported coefficient dtype code {dtype_code}")

    if group_id == 0:
        array = values.reshape(low_h, low_w, C)
    else:
        array = values.reshape(ph, pw, C)
    return {
        'version': version, 'group_id': group_id, 'ph': ph, 'pw': pw, 'C': C,
        'low_h': low_h, 'low_w': low_w, 'array': array,
    }


class FourierProgressiveTransmissionPolicy(LaplacianPyramidPolicy):
    """
    Progressive frequency-domain transmission.

    Each config.patch_size patch is transformed with a per-channel 2D DCT.
    group 0 carries only the low-frequency coefficients (top-left rectangle,
    sized from the base pyramid level) so the server can cheaply reconstruct
    an approximate full-resolution image and run APPROX_FORWARD. Later
    group(s) carry the remaining coefficients so the server can merge them
    with the already-received group 0 data and run CORRECT_FORWARD on the
    exact reconstruction. Coefficients are transported as rounded int16 by
    default (near-lossless; see _pack_payload), zlib-compressed on top.

    Two group layouts are supported, both compatible with the same wire
    format and decode() logic:
      - binary (default): group 0 (low-freq, all patches) then a single
        group 1 (remaining coefficients, all patches). Matches schedulers
        like ADE20KApproxCorrect that expect exactly two groups.
      - windowed (transmission_kwargs.windowed_groups=true): group 0 as
        above, then groups 1..9 each carrying the remaining coefficients for
        one raster 3x3 spatial window (reusing
        COCOWindowProgressiveLaplacianPolicy's window partition). Matches
        COCOWindowInterleaved/-Dynamic schedulers, which expect a sequence of
        window-sized correction groups rather than one big group.

    Unlike LaplacianPyramidPolicy, the patch grid resolution is constant
    across groups (there's no spatial downsampling) — only the amount of
    frequency content sent differs per group. Inherits shape/preserve-mode
    helpers from LaplacianPyramidPolicy for consistency with existing configs.
    """

    _N_WINDOWS_H = COCOWindowProgressiveLaplacianPolicy._N_WINDOWS_H
    _N_WINDOWS_W = COCOWindowProgressiveLaplacianPolicy._N_WINDOWS_W

    def _lowfreq_region(self, config: ExperimentConfig) -> Tuple[int, int, int]:
        ph, pw = config.patch_size
        levels = sorted(config.transmission_kwargs.get('pyramid_levels', [2, 0]), reverse=True)
        base_level = levels[0]
        low_h = min(max(math.ceil(ph / (2 ** base_level)), 1), ph)
        low_w = min(max(math.ceil(pw / (2 ** base_level)), 1), pw)
        return low_h, low_w, base_level

    @staticmethod
    def _coeff_dtype_code(config: ExperimentConfig) -> int:
        return _dtype_code(str(config.transmission_kwargs.get('coeff_dtype', 'int16')))

    @staticmethod
    def _is_windowed(config: ExperimentConfig) -> bool:
        return bool(config.transmission_kwargs.get('windowed_groups', False))

    @staticmethod
    def _split_patches(image: np.ndarray, ph: int, pw: int) -> Tuple[np.ndarray, int, int]:
        H, W, C = image.shape
        if H % ph != 0 or W % pw != 0:
            raise ValueError(f"[FourierProgressive] Image shape {(H, W)} not divisible by patch size {(ph, pw)}")
        gh, gw = H // ph, W // pw
        crops = image.reshape(gh, ph, gw, pw, C).transpose(0, 2, 1, 3, 4).reshape(-1, ph, pw, C)
        return crops, gh, gw

    @staticmethod
    def _idct2_patch(coeff: np.ndarray, C: int) -> np.ndarray:
        rec = np.empty(coeff.shape, dtype=np.float32)
        for c in range(C):
            rec[:, :, c] = _idct2(coeff[:, :, c])
        return np.clip(np.rint(rec), 0, 255).astype(np.uint8)

    @staticmethod
    def _place_reconstructed(canvas: np.ndarray, rec: np.ndarray, spatial_idx: int, gw: int, ph: int, pw: int) -> None:
        H, W = canvas.shape[:2]
        r, c = divmod(spatial_idx, gw)
        y, x = r * ph, c * pw
        th, tw = min(ph, H - y), min(pw, W - x)
        if th <= 0 or tw <= 0:
            return
        canvas[y:y + th, x:x + tw] = rec[:th, :tw]

    def _reconstruct_patch(self, low_patch: Patch, high_patch: Optional[Patch], ph: int, pw: int, C: int) -> np.ndarray:
        """Reconstruct one patch's pixels, memoized on the Patch objects
        themselves (mirrors LaplacianPyramidPolicy._place_patch's
        `_decompressed_cache` pattern). A merged (low+high) or low-freq-only
        reconstruction is deterministic once its inputs are known, so caching
        it directly on the relevant Patch avoids redoing zlib decompress +
        IDCT on every decode() call as the accumulated patch buffer grows
        across rounds — without this, decode() cost grows from O(patches) to
        O(patches * rounds), which is what made windowed decode() ~12x slower
        than the Laplacian baseline in practice."""
        if high_patch is not None:
            cached = getattr(high_patch, '_fpc_merged_rec', None)
            if cached is not None:
                return cached
            low_info = _unpack_payload(zlib.decompress(low_patch.data))
            high_info = _unpack_payload(zlib.decompress(high_patch.data))
            full_coeff = _merge_coefficients(
                low_info['array'], high_info['array'], ph, pw, C,
                low_info['low_h'], low_info['low_w'],
            )
            rec = self._idct2_patch(full_coeff, C)
            high_patch._fpc_merged_rec = rec
            return rec

        cached = getattr(low_patch, '_fpc_lowres_rec', None)
        if cached is not None:
            return cached
        low_info = _unpack_payload(zlib.decompress(low_patch.data))
        full_coeff = np.zeros((ph, pw, C), dtype=np.float32)
        full_coeff[:low_info['low_h'], :low_info['low_w'], :] = low_info['array']
        rec = self._idct2_patch(full_coeff, C)
        low_patch._fpc_lowres_rec = rec
        return rec

    @staticmethod
    def _pack_high_freq(coeff: np.ndarray, low_h: int, low_w: int, comp_lvl: int, dtype_code: int) -> Tuple[bytes, float]:
        """Returns (compressed payload, energy of the coefficients being sent).
        The energy doubles as this correction patch's pscore_hint: it tells the
        appcorr token-selection logic (mobile_pscore="residual_energy") how much
        this round's data actually changes the reconstruction, so it can decide
        which tokens are worth recomputing. Without this, correction patches all
        look equally (un)important and the scheduler may not recompute any
        tokens at all."""
        ph, pw, C = coeff.shape
        high_full = coeff.copy()
        high_full[:low_h, :low_w, :] = 0.0
        energy = float(np.sum(np.square(high_full, dtype=np.float64)))
        payload = _pack_payload(1, ph, pw, C, low_h, low_w, high_full, dtype_code)
        return zlib.compress(payload, level=comp_lvl), energy

    # --- Encoding -----------------------------------------------------------

    def encode(self, images: np.ndarray, config: ExperimentConfig) -> Generator[List[Patch], None, None]:
        if self._is_windowed(config):
            yield from self._encode_windowed(images, config)
        else:
            yield from self._encode_binary(images, config)

    def _encode_group0(self, images: np.ndarray, config: ExperimentConfig, allow_preserve: bool):
        image_list = self._as_image_list(images)
        ph, pw = config.patch_size
        comp_lvl = config.transmission_kwargs.get('compression_level', 1)
        dtype_code = self._coeff_dtype_code(config)
        low_h, low_w, base_level = self._lowfreq_region(config)
        preserve = allow_preserve and self._is_preserve_input_shape(config)
        image_hws = [img.shape[:2] for img in image_list] if preserve else [None] * len(image_list)

        # Compute DCT coefficients + group 0 (low-frequency) patches. Coefficient
        # arrays are cached so later phases don't recompute the DCT.
        low_patches: List[Patch] = []
        cached_coeffs: List[Tuple[np.ndarray, int, int]] = [None] * len(image_list)

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self._process_image_group0, b, image, config, low_h, low_w,
                    base_level, comp_lvl, dtype_code, image_hws[b],
                )
                for b, image in enumerate(image_list)
            ]
            for b, f in enumerate(futures):
                patches_b, coeff_b, gh, gw = f.result()
                low_patches.extend(patches_b)
                cached_coeffs[b] = (coeff_b, gh, gw)

        g0_total = len(low_patches)
        for p in low_patches:
            p.batch_group_total = g0_total

        return image_list, low_patches, cached_coeffs, low_h, low_w, comp_lvl, dtype_code

    def _encode_binary(self, images: np.ndarray, config: ExperimentConfig) -> Generator[List[Patch], None, None]:
        image_list, low_patches, cached_coeffs, low_h, low_w, comp_lvl, dtype_code = self._encode_group0(
            images, config, allow_preserve=True,
        )
        yield low_patches  # Yield group 0 (low-frequency) immediately.

        # Phase 2: build group 1 (remaining coefficients) from cached DCT arrays.
        high_patches: List[Patch] = []
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self._process_image_group1, b, cached_coeffs[b][0], low_h, low_w, comp_lvl, dtype_code,
                )
                for b in range(len(image_list))
            ]
            for f in futures:
                high_patches.extend(f.result())

        g1_total = len(high_patches)
        for p in high_patches:
            p.batch_group_total = g1_total
        yield high_patches

    def _encode_windowed(self, images: np.ndarray, config: ExperimentConfig) -> Generator[List[Patch], None, None]:
        if self._is_preserve_input_shape(config):
            raise NotImplementedError(
                "FourierProgressive: windowed_groups + preserve_input_shape is not supported "
                "(window boundaries assume a shared config.image_shape grid)."
            )

        image_list, low_patches, cached_coeffs, low_h, low_w, comp_lvl, dtype_code = self._encode_group0(
            images, config, allow_preserve=False,
        )
        yield low_patches  # Yield group 0 (low-frequency) immediately.

        # Precompute each grid position's raster window id once (identical grid
        # for every image since preserve_input_shape is disallowed here).
        _, gh, gw = cached_coeffs[0]
        window_ids = np.empty(gh * gw, dtype=np.int64)
        for i in range(gh * gw):
            row, col = divmod(i, gw)
            window_ids[i] = COCOWindowProgressiveLaplacianPolicy._window_group_id_for_patch(row, col, config)

        num_windows = self._N_WINDOWS_H * self._N_WINDOWS_W
        for window_group_id in range(1, num_windows + 1):
            idxs = np.flatnonzero(window_ids == window_group_id)
            if idxs.size == 0:
                continue
            group_patches: List[Patch] = []
            for b_idx in range(len(image_list)):
                coeff_arr, _, _ = cached_coeffs[b_idx]
                for i in idxs:
                    i = int(i)
                    compressed, energy = self._pack_high_freq(coeff_arr[i], low_h, low_w, comp_lvl, dtype_code)
                    group_patches.append(Patch(
                        image_idx=b_idx, spatial_idx=i, data=compressed,
                        res_level=0, group_id=window_group_id, pscore_hint=energy,
                    ))
            for p in group_patches:
                p.batch_group_total = len(group_patches)
            yield group_patches

    def _process_image_group0(
        self, b_idx, image, config, low_h, low_w, base_level, comp_lvl, dtype_code, image_hw,
    ):
        ph, pw = config.patch_size
        H, W = self._target_hw_for_level(config, 0, image_hw)
        resized = self._resize_to_hw(image, (H, W), np.uint8)
        crops, gh, gw = self._split_patches(resized, ph, pw)
        num_crops = crops.shape[0]
        C = crops.shape[-1]

        coeff_arr = np.empty((num_crops, ph, pw, C), dtype=np.float32)
        patches: List[Patch] = []

        for i in range(num_crops):
            crop = crops[i].astype(np.float32)
            coeff = np.empty((ph, pw, C), dtype=np.float32)
            for c in range(C):
                coeff[:, :, c] = _dct2(crop[:, :, c])
            coeff_arr[i] = coeff

            low_block = _split_lowfreq(coeff, low_h, low_w)
            high_full = coeff.copy()
            high_full[:low_h, :low_w, :] = 0.0
            residual_energy = float(np.sum(np.square(high_full, dtype=np.float64)))

            payload = _pack_payload(0, ph, pw, C, low_h, low_w, low_block, dtype_code)
            compressed = zlib.compress(payload, level=comp_lvl)
            patches.append(Patch(
                image_idx=b_idx, spatial_idx=i, data=compressed,
                res_level=base_level, group_id=0, pscore_hint=residual_energy,
            ))

        return patches, coeff_arr, gh, gw

    def _process_image_group1(self, b_idx, coeff_arr, low_h, low_w, comp_lvl, dtype_code):
        num_crops = coeff_arr.shape[0]
        patches: List[Patch] = []

        for i in range(num_crops):
            compressed, energy = self._pack_high_freq(coeff_arr[i], low_h, low_w, comp_lvl, dtype_code)
            patches.append(Patch(
                image_idx=b_idx, spatial_idx=i, data=compressed,
                res_level=0, group_id=1, pscore_hint=energy,
            ))

        return patches

    # --- Decoding -------------------------------------------------------------

    def decode_lowres(self, patches: List[Patch], config: ExperimentConfig) -> np.ndarray:
        """Zero-fill missing (high-frequency) coefficients and reconstruct a
        full-size, low-frequency-only approximation image from group 0 patches."""
        B = config.batch_size
        C = config.image_shape[2]
        ph, pw = config.patch_size
        preserve = self._is_preserve_input_shape(config)

        group0 = [p for p in patches if p.group_id == 0]
        target_shapes = self._collect_target_shapes(patches, B) if preserve else [None] * B

        group0_by_batch: Dict[int, List[Patch]] = {b: [] for b in range(B)}
        for p in group0:
            if 0 <= p.image_idx < B:
                group0_by_batch[p.image_idx].append(p)

        results = []
        for b in range(B):
            H, W = self._target_hw_for_level(config, 0, target_shapes[b])
            canvas = np.zeros((H, W, C), dtype=np.uint8)
            gw = W // pw
            for p in group0_by_batch[b]:
                rec = self._reconstruct_patch(p, None, ph, pw, C)
                self._place_reconstructed(canvas, rec, p.spatial_idx, gw, ph, pw)
            results.append(canvas)

        if preserve:
            return results
        return np.stack(results) if results else np.zeros((B, *config.image_shape), dtype=np.uint8)

    def decode(self, patches: List[Patch], config: ExperimentConfig, canvas: np.ndarray = None) -> np.ndarray:
        """Reconstruct the best available image from whatever groups have been
        accumulated so far. Every group0 (low-frequency) patch is placed as a
        baseline reconstruction; any patch that additionally has a matching
        high-frequency patch (group_id >= 1, from either the binary or the
        windowed layout) is overwritten with the exact merged reconstruction.
        This naturally supports partial windowed arrival (some spatial windows
        already corrected, others still low-frequency-only)."""
        B = config.batch_size
        C = config.image_shape[2]
        ph, pw = config.patch_size
        preserve = self._is_preserve_input_shape(config)

        group0 = [p for p in patches if p.group_id == 0]
        high = [p for p in patches if p.group_id >= 1]

        if not high:
            # Only low-frequency data available so far — same as decode_lowres.
            return self.decode_lowres(patches, config)

        group0_index = {(p.image_idx, p.spatial_idx): p for p in group0}
        high_index: Dict[Tuple[int, int], Patch] = {}
        for p in high:
            key = (p.image_idx, p.spatial_idx)
            if key not in group0_index:
                raise RuntimeError(
                    "FourierProgressive.decode: high-frequency patch "
                    f"(image_idx={p.image_idx}, spatial_idx={p.spatial_idx}, group_id={p.group_id}) "
                    "has no matching group 0 (low-frequency) patch among the patches passed to "
                    "decode(). FourierProgressive requires the server to retain group 0 patches for "
                    "the same request_id and pass them together with later groups for CORRECT_FORWARD "
                    "decoding — it cannot reconstruct high-frequency-only data."
                )
            high_index[key] = p  # windows are spatially disjoint, so no key collides across groups

        target_shapes = self._collect_target_shapes(patches, B) if preserve else [None] * B

        group0_by_batch: Dict[int, List[Patch]] = {b: [] for b in range(B)}
        for p in group0:
            if 0 <= p.image_idx < B:
                group0_by_batch[p.image_idx].append(p)

        results = []
        for b in range(B):
            H, W = self._target_hw_for_level(config, 0, target_shapes[b])
            canvas_img = np.zeros((H, W, C), dtype=np.uint8)
            gw = W // pw
            for p in group0_by_batch[b]:
                key = (p.image_idx, p.spatial_idx)
                hp = high_index.get(key)
                rec = self._reconstruct_patch(p, hp, ph, pw, C)
                self._place_reconstructed(canvas_img, rec, p.spatial_idx, gw, ph, pw)
            results.append(canvas_img)

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
