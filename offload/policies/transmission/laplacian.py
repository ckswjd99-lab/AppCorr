import numpy as np
import cv2
import zlib
from typing import List
from concurrent.futures import ThreadPoolExecutor
from offload.common.protocol import Patch, ExperimentConfig
from ..interface import ITransmissionPolicy

class LaplacianPyramidPolicy(ITransmissionPolicy):
    """
    Generalized Laplacian Pyramid with Zlib Compression.
    Enforces strict uint8 consistency for lossless residual coding.
    """

    @staticmethod
    def _as_image_list(images) -> List[np.ndarray]:
        if isinstance(images, np.ndarray):
            if images.ndim != 4:
                raise RuntimeError(f"Expected image batch [B,H,W,C], got {tuple(images.shape)}")
            return [np.ascontiguousarray(images[b]) for b in range(images.shape[0])]
        if isinstance(images, (list, tuple)):
            return [np.ascontiguousarray(image) for image in images]
        raise RuntimeError(f"Unsupported image batch type: {type(images)!r}")

    @staticmethod
    def _target_hw_for_level(config: ExperimentConfig, lvl: int) -> tuple[int, int]:
        H, W = config.image_shape[:2]
        scale = 2 ** lvl
        if H % scale != 0 or W % scale != 0:
            raise RuntimeError(f"Target image shape {(H, W)} is not divisible by pyramid scale {scale}")
        return H // scale, W // scale

    @staticmethod
    def _resize_to_hw(image: np.ndarray, target_hw: tuple[int, int], dtype) -> np.ndarray:
        target_h, target_w = target_hw
        if image.shape[:2] == (target_h, target_w):
            return np.ascontiguousarray(image.astype(dtype, copy=False))

        interpolation = cv2.INTER_AREA
        if target_h > image.shape[0] or target_w > image.shape[1]:
            interpolation = cv2.INTER_LINEAR

        if dtype == np.int16:
            resized = cv2.resize(image.astype(np.float32), (target_w, target_h), interpolation=interpolation)
            resized = np.rint(resized)
            resized = np.clip(resized, np.iinfo(np.int16).min, np.iinfo(np.int16).max).astype(np.int16)
            return np.ascontiguousarray(resized)

        resized = cv2.resize(image, (target_w, target_h), interpolation=interpolation)
        return np.ascontiguousarray(resized.astype(dtype, copy=False))

    def _project_band_to_target(self, image: np.ndarray, lvl: int, config: ExperimentConfig, dtype) -> np.ndarray:
        return self._resize_to_hw(image, self._target_hw_for_level(config, lvl), dtype)

    @staticmethod
    def _build_native_gaussians(image: np.ndarray, max_lvl: int) -> dict[int, np.ndarray]:
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        gaussians = {0: np.ascontiguousarray(image)}
        curr = gaussians[0]
        for i in range(1, max_lvl + 1):
            curr = cv2.pyrDown(curr)
            gaussians[i] = np.ascontiguousarray(curr.astype(np.uint8, copy=False))
        return gaussians

    @staticmethod
    def _iterative_upsample_native(img, start_lvl, end_lvl, gaussians):
        curr = img
        for next_lvl in range(start_lvl - 1, end_lvl - 1, -1):
            th, tw = gaussians[next_lvl].shape[:2]
            curr = cv2.pyrUp(curr, dstsize=(tw, th)).astype(np.uint8)
        return curr

    def encode(self, images: np.ndarray, config: ExperimentConfig) -> List[Patch]:
        image_list = self._as_image_list(images)
        # Yield patches layer by layer
        levels = sorted(config.transmission_kwargs.get('pyramid_levels', [2, 1, 0]), reverse=True)
        
        for tgt_lvl_idx, tgt_lvl in enumerate(levels):
            layer_patches = []
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(self._process_image_encode_single_layer, b, image, tgt_lvl_idx, levels, config)
                    for b, image in enumerate(image_list)
                ]
                for f in futures:
                    layer_patches.extend(f.result())
            yield layer_patches

    def decode(self, patches: List[Patch], config: ExperimentConfig, canvas: np.ndarray = None) -> np.ndarray:
        B = config.batch_size
        H, W, C = config.image_shape
        
        # Group patches by Batch
        layer_data_per_batch = {b: [] for b in range(B)}
        for p in patches:
            layer_data_per_batch[p.image_idx].append(p)
            
        if canvas is None:
            final_images = np.zeros((B, H, W, C), dtype=np.uint8)
        else:
            final_images = canvas

        # Per-image reconstruction
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(self._process_image_decode, b, layer_data_per_batch[b], config): b
                for b in range(B)
            }
            for f in futures:
                b, img = f.result()
                final_images[b] = img

        return final_images

    def decode_lowres(self, patches: List[Patch], config: ExperimentConfig) -> np.ndarray:
        B = config.batch_size
        H, W, C = config.image_shape
        levels = sorted(config.transmission_kwargs.get('pyramid_levels', [2, 1, 0]), reverse=True)
        base_lvl = levels[0]
        base_h = H // (2 ** base_lvl)
        base_w = W // (2 ** base_lvl)

        lowres_images = np.zeros((B, base_h, base_w, C), dtype=np.uint8)
        base_patches_per_batch = {b: [] for b in range(B)}
        for patch in patches:
            if patch.res_level == base_lvl:
                base_patches_per_batch[patch.image_idx].append(patch)

        for b_idx, batch_patches in base_patches_per_batch.items():
            for patch in batch_patches:
                self._place_patch(lowres_images[b_idx], patch, config, np.uint8)

        return lowres_images

    # --- Helpers for Parallel Execution ---

    def _iterative_upsample(self, img, start_lvl, end_lvl, H, W):
        """Upsamples image iteratively, enforcing uint8 cast at each step."""
        curr = img
        gap = start_lvl - end_lvl
        for k in range(gap):
            next_lvl = start_lvl - 1 - k
            th, tw = H // (2**next_lvl), W // (2**next_lvl)
            curr = cv2.pyrUp(curr, dstsize=(tw, th))
            curr = curr.astype(np.uint8) 
        return curr

    def _process_image_encode_single_layer(self, b_idx, image, tgt_lvl_idx, levels, config):
        max_lvl = max(levels)
        local_patches = []
        
        gaussians = self._build_native_gaussians(image, max_lvl)
            
        tgt_lvl = levels[tgt_lvl_idx]
        curr_g = gaussians[tgt_lvl]
        
        if tgt_lvl_idx == 0:
            # Base Layer
            projected = self._project_band_to_target(curr_g, tgt_lvl, config, np.uint8)
            self._create_patches_vectorized(local_patches, projected, b_idx, tgt_lvl, config, np.uint8)
        else:
            prev_lvl = levels[tgt_lvl_idx - 1]
            prev_g = gaussians[prev_lvl]
            pred = self._iterative_upsample_native(prev_g, prev_lvl, tgt_lvl, gaussians)
            residual = curr_g.astype(np.int16) - pred.astype(np.int16)
            projected = self._project_band_to_target(residual, tgt_lvl, config, np.int16)
            self._create_patches_vectorized(local_patches, projected, b_idx, tgt_lvl, config, np.int16)
            
        return local_patches

    def _process_image_decode(self, b_idx, patches, config):
        H, W, C = config.image_shape
        levels = sorted(config.transmission_kwargs.get('pyramid_levels', [2, 1, 0]), reverse=True)
        strict_base_lvl = levels[0]

        if not patches:
            return b_idx, np.zeros((H, W, C), dtype=np.uint8)

        # Organize by level
        layer_data = {}
        all_levels = set()
        for p in patches:
            all_levels.add(p.res_level)
            layer_data.setdefault(p.res_level, []).append(p)
        
        sorted_levels = sorted(list(all_levels), reverse=True)
        
        # Reconstruct Base
        bh = H // (2**strict_base_lvl)
        bw = W // (2**strict_base_lvl)
        curr_img = np.zeros((bh, bw, C), dtype=np.uint8)
        
        if strict_base_lvl in layer_data:
            for p in layer_data[strict_base_lvl]:
                self._place_patch(curr_img, p, config, np.uint8)
        
        prev_lvl = strict_base_lvl

        # Apply Residuals
        for lvl in sorted_levels:
            if lvl == strict_base_lvl: continue
            
            curr_img = self._iterative_upsample(curr_img, prev_lvl, lvl, H, W)
            
            if lvl in layer_data:
                rh, rw = curr_img.shape[:2]
                res_img = np.zeros((rh, rw, C), dtype=np.int16)
                for p in layer_data[lvl]:
                    self._place_patch(res_img, p, config, np.int16)
                
                curr_img = np.clip(curr_img.astype(np.int16) + res_img, 0, 255).astype(np.uint8)
            
            prev_lvl = lvl
        
        if prev_lvl > 0:
            curr_img = self._iterative_upsample(curr_img, prev_lvl, 0, H, W)
            
        return b_idx, curr_img
        
    def _create_patches_vectorized(self, patch_list, image, b_idx, lvl, config, dtype):
        ph, pw = config.patch_size
        H, W, C = image.shape
        comp_lvl = config.transmission_kwargs.get('compression_level', 1)

        # Extract crops using vectorized operations, assuming strict divisibility
        assert H % ph == 0 and W % pw == 0, f"Image shape {(H, W)} must be divisible by patch size {(ph, pw)}"
        gh, gw = H // ph, W // pw
        
        crops = image.reshape(gh, ph, gw, pw, C).transpose(0, 2, 1, 3, 4).reshape(-1, ph, pw, C)
        
        # Compress sequentially
        num_crops = crops.shape[0]

        for i in range(num_crops):
             # Loop sequentially to match row-major indexing
             data = crops[i].astype(dtype).tobytes()
             compressed = zlib.compress(data, level=comp_lvl)
             
             patch_list.append(Patch(b_idx, i, compressed, lvl))

    def _place_patch(self, canvas, patch, config, dtype):
        ph, pw = config.patch_size
        H, W, C = canvas.shape
        gw = (W + pw - 1) // pw 
        
        r, c = divmod(patch.spatial_idx, gw)
        y, x = r * ph, c * pw
        th, tw = min(ph, H - y), min(pw, W - x)
        
        if th <= 0 or tw <= 0: return

        try:
            # Use cached decompressed bytes if available
            if hasattr(patch, '_decompressed_cache'):
                raw = patch._decompressed_cache
            else:
                raw = zlib.decompress(patch.data)
                patch._decompressed_cache = raw
            
            chunk = np.frombuffer(raw, dtype=dtype).reshape(ph, pw, C)
            canvas[y:y+th, x:x+tw] = chunk

        except Exception as e:
            # Handle decompression error
            print(f"!!! [Laplacian] Decompress Error Lvl{patch.res_level}: {e}")
