import numpy as np
from typing import List
import zlib, cv2

from offload.common.protocol import Patch, ExperimentConfig
from .interface import ITransmissionPolicy

class RawTransmissionPolicy(ITransmissionPolicy):
    """Encodes/Decodes patches with Batch dimension support using vectorized ops."""

    def encode(self, images: np.ndarray, config: ExperimentConfig) -> List[Patch]:
        """
        Input: images (B, H, W, C)
        Output: Flattened list of patches.
        Optimization: Uses numpy reshape/transpose to avoid nested slicing loops.
        """
        patches = []
        B, H, W, C = images.shape
        ph, pw = config.patch_size
        
        # Grid dimensions
        gh, gw = H // ph, W // pw
        
        reshaped = images.reshape(B, gh, ph, gw, pw, C)
        transposed = reshaped.transpose(0, 1, 3, 2, 4, 5)
        patch_tensor = transposed.reshape(B, gh * gw, ph, pw, C)
        
        num_patches = gh * gw
        for b in range(B):
            for i in range(num_patches):
                data = patch_tensor[b, i].tobytes()
                patches.append(Patch(image_idx=b, spatial_idx=i, data=data))
                
        return patches

    def decode(self, patches: List[Patch], config: ExperimentConfig) -> np.ndarray:
        """
        Input: List of Patch objects (sparse).
        Output: Reconstructed Tensor (B, H, W, C).
        """
        B = config.batch_size
        H, W, C = config.image_shape
        ph, pw = config.patch_size
        gw = W // pw
        
        # Initialize canvas
        batch_tensor = np.zeros((B, H, W, C), dtype=np.uint8)
        
        for p in patches:
            # Calculate grid coordinates
            r, c = divmod(p.spatial_idx, gw)
            y, x = r * ph, c * pw
            
            # Restore patch array
            chunk = np.frombuffer(p.data, dtype=np.uint8).reshape(ph, pw, C)
            
            # Place on canvas
            batch_tensor[p.image_idx, y:y+ph, x:x+pw] = chunk
            
        return batch_tensor

class ZlibTransmissionPolicy(ITransmissionPolicy):
    """
    Lossless Coding Policy.
    Compresses each patch using zlib (DEFLATE = LZ77 + Huffman).
    """

    def encode(self, images: np.ndarray, config: ExperimentConfig) -> List[Patch]:
        """
        Input: images (B, H, W, C)
        Output: List of compressed patches.
        """
        patches = []
        B, H, W, C = images.shape
        ph, pw = config.patch_size
        gh, gw = H // ph, W // pw

        compression_level = config.transmission_kwargs.get('compression_level', 1)
        
        # 1. Vectorized Patchify (Same as Raw)
        reshaped = images.reshape(B, gh, ph, gw, pw, C)
        transposed = reshaped.transpose(0, 1, 3, 2, 4, 5)
        patch_tensor = transposed.reshape(B, gh * gw, ph, pw, C)
        
        num_patches = gh * gw
        
        # 2. Compress and Wrap
        for b in range(B):
            for i in range(num_patches):
                raw_bytes = patch_tensor[b, i].tobytes()
                compressed_data = zlib.compress(raw_bytes, level=compression_level)
                patches.append(Patch(image_idx=b, spatial_idx=i, data=compressed_data))
                
        return patches

    def decode(self, patches: List[Patch], config: ExperimentConfig) -> np.ndarray:
        """
        Input: List of compressed Patch objects.
        Output: Reconstructed Tensor (B, H, W, C).
        """
        B = config.batch_size
        H, W, C = config.image_shape
        ph, pw = config.patch_size
        gw = W // pw
        
        batch_tensor = np.zeros((B, H, W, C), dtype=np.uint8)
        
        for p in patches:
            # 1. Decompress
            try:
                raw_bytes = zlib.decompress(p.data)
                chunk = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(ph, pw, C)
                
                # 2. Place on canvas
                r, c = divmod(p.spatial_idx, gw)
                y, x = r * ph, c * pw
                
                batch_tensor[p.image_idx, y:y+ph, x:x+pw] = chunk
            except Exception as e:
                print(f"!!! [ZlibPolicy] Decompression failed for patch {p.spatial_idx}: {e}")
            
        return batch_tensor

class LaplacianPyramidPolicy(ITransmissionPolicy):
    """
    Generalized Laplacian Pyramid with Zlib Compression.
    Enforces strict uint8 consistency for lossless residual coding.
    """

    def encode(self, images: np.ndarray, config: ExperimentConfig) -> List[Patch]:
        patches = []
        B, H, W, C = images.shape
        
        # Sort levels descending (Base -> Detail)
        levels = sorted(config.transmission_kwargs.get('pyramid_levels', [2, 1, 0]), reverse=True)
        max_lvl = max(levels)

        for b in range(B):
            # 1. Build Ground Truth Gaussian Pyramid
            gaussians = {0: images[b]}
            for i in range(1, max_lvl + 1):
                gaussians[i] = cv2.pyrDown(gaussians[i-1])
            
            prev_img = None
            prev_lvl = -1
            
            for lvl in levels:
                curr_g = gaussians[lvl]
                
                if prev_img is None:
                    # Base Layer (uint8)
                    self._create_patches(patches, curr_g, b, lvl, config, np.uint8)
                else:
                    # Residual Layer (int16)
                    # Predict current from previous using strict upsampling
                    pred = self._iterative_upsample(prev_img, prev_lvl, lvl, H, W)
                    residual = curr_g.astype(np.int16) - pred.astype(np.int16)
                    self._create_patches(patches, residual, b, lvl, config, np.int16)
                
                # Update state: Decoder has exact current_g now
                prev_img = curr_g
                prev_lvl = lvl

        return patches

    def decode(self, patches: List[Patch], config: ExperimentConfig) -> np.ndarray:
        B = config.batch_size
        H, W, C = config.image_shape
        
        # 1. Group patches by Batch and Level
        layer_data = {b: {} for b in range(B)}
        all_levels = set()
        for p in patches:
            all_levels.add(p.res_level)
            layer_data[p.image_idx].setdefault(p.res_level, []).append(p)
            
        sorted_levels = sorted(list(all_levels), reverse=True)
        if not sorted_levels: return np.zeros((B, H, W, C), dtype=np.uint8)

        final_images = np.zeros((B, H, W, C), dtype=np.uint8)

        for b in range(B):
            if not layer_data[b]: continue
            
            # 2. Reconstruct Base
            base_lvl = sorted_levels[0]
            bh, bw = H // (2**base_lvl), W // (2**base_lvl)
            curr_img = np.zeros((bh, bw, C), dtype=np.uint8)
            
            if base_lvl in layer_data[b]:
                for p in layer_data[b][base_lvl]:
                    self._place_patch(curr_img, p, config, np.uint8)
            
            prev_lvl = base_lvl

            # 3. Apply Residuals
            for lvl in sorted_levels[1:]:
                # Upsample current base to next level size
                curr_img = self._iterative_upsample(curr_img, prev_lvl, lvl, H, W)
                
                # Add Residuals
                if lvl in layer_data[b]:
                    rh, rw = curr_img.shape[:2]
                    res_img = np.zeros((rh, rw, C), dtype=np.int16)
                    for p in layer_data[b][lvl]:
                        self._place_patch(res_img, p, config, np.int16)
                    
                    # Add and Clip (int16 math -> uint8)
                    curr_img = np.clip(curr_img.astype(np.int16) + res_img, 0, 255).astype(np.uint8)
                
                prev_lvl = lvl

            final_images[b] = curr_img

        return final_images

    # --- Helpers ---

    def _iterative_upsample(self, img, start_lvl, end_lvl, H, W):
        """Upsamples image iteratively, enforcing uint8 cast at each step."""
        curr = img
        gap = start_lvl - end_lvl
        for k in range(gap):
            next_lvl = start_lvl - 1 - k
            th, tw = H // (2**next_lvl), W // (2**next_lvl)
            curr = cv2.pyrUp(curr, dstsize=(tw, th))
            curr = curr.astype(np.uint8) # Critical for consistency
        return curr

    def _create_patches(self, patch_list, image, b_idx, lvl, config, dtype):
        ph, pw = config.patch_size
        H, W = image.shape[:2]
        gh, gw = (H + ph - 1) // ph, (W + pw - 1) // pw # Ceil division
        comp_lvl = config.transmission_kwargs.get('compression_level', 1)
        
        idx = 0
        for r in range(gh):
            for c in range(gw):
                y1, x1 = r * ph, c * pw
                y2, x2 = min(H, y1 + ph), min(W, x1 + pw)
                
                crop = np.ascontiguousarray(image[y1:y2, x1:x2])
                compressed = zlib.compress(crop.astype(dtype).tobytes(), level=comp_lvl)
                
                patch_list.append(Patch(b_idx, idx, compressed, lvl))
                idx += 1

    def _place_patch(self, canvas, patch, config, dtype):
        ph, pw = config.patch_size
        H, W, C = canvas.shape
        gw = (W + pw - 1) // pw # Ceil division
        
        r, c = divmod(patch.spatial_idx, gw)
        y, x = r * ph, c * pw
        th, tw = min(ph, H - y), min(pw, W - x)
        
        if th <= 0 or tw <= 0: return

        try:
            raw = zlib.decompress(patch.data)
            canvas[y:y+th, x:x+tw] = np.frombuffer(raw, dtype=dtype).reshape(th, tw, C)
        except Exception as e:
            print(f"!!! [Laplacian] Decompress Error Lvl{patch.res_level}: {e}")