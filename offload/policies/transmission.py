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
            curr = images[b]
            for i in range(1, max_lvl + 1):
                curr = cv2.pyrDown(curr)
                gaussians[i] = curr
            
            prev_img = None
            prev_lvl = -1
            
            for lvl in levels:
                curr_g = gaussians[lvl]
                
                if prev_img is None:
                    # Base Layer (uint8)
                    self._create_patches(patches, curr_g, b, lvl, config, np.uint8)
                    prev_img = curr_g
                    prev_lvl = lvl
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
        
        # Determine strict Base Level from Config
        levels = sorted(config.transmission_kwargs.get('pyramid_levels', [2, 1, 0]), reverse=True)
        strict_base_lvl = levels[0]  # The highest level is the Base

        # 1. Group patches by Batch and Level
        layer_data = {b: {} for b in range(B)}
        all_levels = set()
        for p in patches:
            all_levels.add(p.res_level)
            layer_data[p.image_idx].setdefault(p.res_level, []).append(p)
            
        # Levels to process (excluding Base which is handled separately)
        sorted_levels = sorted(list(all_levels), reverse=True)
        
        # Init Canvas (Start with black if Base is missing)
        final_images = np.zeros((B, H, W, C), dtype=np.uint8)

        for b in range(B):
            if not layer_data[b]: continue
            
            # 2. Reconstruct Base
            # Calculate Base dimensions
            bh = H // (2**strict_base_lvl)
            bw = W // (2**strict_base_lvl)
            curr_img = np.zeros((bh, bw, C), dtype=np.uint8)
            
            # Decode Base only if present
            if strict_base_lvl in layer_data[b]:
                for p in layer_data[b][strict_base_lvl]:
                    self._place_patch(curr_img, p, config, np.uint8)
            
            prev_lvl = strict_base_lvl

            # 3. Apply Residuals
            for lvl in sorted_levels:
                if lvl == strict_base_lvl:
                    continue 
                
                # Upsample current base to next level size
                curr_img = self._iterative_upsample(curr_img, prev_lvl, lvl, H, W)
                
                # Add Residuals (int16)
                if lvl in layer_data[b]:
                    rh, rw = curr_img.shape[:2]
                    res_img = np.zeros((rh, rw, C), dtype=np.int16)
                    for p in layer_data[b][lvl]:
                        self._place_patch(res_img, p, config, np.int16)
                    
                    # Add and Clip (int16 math -> uint8)
                    curr_img = np.clip(curr_img.astype(np.int16) + res_img, 0, 255).astype(np.uint8)
                
                prev_lvl = lvl
            
            # Final Upsample to original size if needed
            if prev_lvl > 0:
                curr_img = self._iterative_upsample(curr_img, prev_lvl, 0, H, W)

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
            curr = curr.astype(np.uint8) 
        return curr

    def _create_patches(self, patch_list, image, b_idx, lvl, config, dtype):
        ph, pw = config.patch_size
        H, W = image.shape[:2]
        gh, gw = (H + ph - 1) // ph, (W + pw - 1) // pw 
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
        gw = (W + pw - 1) // pw 
        
        r, c = divmod(patch.spatial_idx, gw)
        y, x = r * ph, c * pw
        th, tw = min(ph, H - y), min(pw, W - x)
        
        if th <= 0 or tw <= 0: return

        try:
            raw = zlib.decompress(patch.data)
            canvas[y:y+th, x:x+tw] = np.frombuffer(raw, dtype=dtype).reshape(th, tw, C)
        except Exception as e:
            # Silent fail or log if strictly needed
            print(f"!!! [Laplacian] Decompress Error Lvl{patch.res_level}: {e}")

class ProgressiveLPyramidPolicy(LaplacianPyramidPolicy):
    """
    Progressive Laplacian Pyramid with 'Uniform Diff' Grouping.
    Includes debug stats for group capacity.
    """

    def encode(self, images: np.ndarray, config: ExperimentConfig) -> List[Patch]:
        patches = []
        B, H, W, C = images.shape
        
        levels = sorted(config.transmission_kwargs.get('pyramid_levels', [2, 0]), reverse=True)
        max_lvl = max(levels)
        num_groups = config.transmission_kwargs.get('num_groups', 4)
        comp_lvl = config.transmission_kwargs.get('compression_level', 1)

        batch_candidates = [[] for _ in range(B)]

        # 1. Generate Candidates (Per Image)
        for b in range(B):
            gaussians = {0: images[b]}
            curr = images[b]
            for i in range(1, max_lvl + 1):
                curr = cv2.pyrDown(curr)
                gaussians[i] = curr
            
            prev_img = None
            prev_lvl = -1
            
            for lvl in levels:
                curr_g = gaussians[lvl]
                
                if prev_img is None:
                    # Group 0: Base Layer (Always Group 0)
                    self._create_patches_with_group(
                        patches, curr_g, b, lvl, config, np.uint8, 
                        group_id=0, compression=comp_lvl
                    )
                    prev_img = curr_g
                    prev_lvl = lvl
                else:
                    # Residual Layer: Collect
                    pred = self._iterative_upsample(prev_img, prev_lvl, lvl, H, W)
                    residual = curr_g.astype(np.int16) - pred.astype(np.int16)
                    
                    self._collect_residual_candidates(
                        batch_candidates[b], residual, b, lvl, config, 
                        dtype=np.int16, compression=comp_lvl
                    )
                    
                    prev_img = curr_g
                    prev_lvl = lvl

        # 2. Apply 'Uniform Diff' Grouping (Batch-Wise)
        if any(batch_candidates):
            self._apply_uniform_diff_grouping(patches, batch_candidates, num_groups)

        # 3. Inject Metadata & Calculate Stats
        group_counts = {}
        group_bytes = {}

        for p in patches:
            # Count patches per group
            group_counts[p.group_id] = group_counts.get(p.group_id, 0) + 1
            # Sum bytes per group
            p_size = len(p.data)
            group_bytes[p.group_id] = group_bytes.get(p.group_id, 0) + p_size
            
        for p in patches:
            p.batch_group_total = group_counts[p.group_id]

        # 4. Sort by Group ID
        patches.sort(key=lambda x: x.group_id)
        
        return patches

    def _apply_uniform_diff_grouping(self, final_patch_list, batch_candidates, num_groups):
        """
        Implements 'uniform_diff' strategy using NumPy.
        Assigns Group IDs based on the AVERAGE size distribution of the batch.
        """
        B = len(batch_candidates)
        if B == 0: return
        N = len(batch_candidates[0])
        if N == 0: return

        # Extract sizes -> [B, N]
        sizes_matrix = np.zeros((B, N), dtype=np.int32)
        for b in range(B):
            for i in range(N):
                sizes_matrix[b, i] = batch_candidates[b][i]['size']

        # 1. Sort norms (sizes) independently for each batch -> [B, N]
        sorted_indices = np.argsort(sizes_matrix, axis=1)
        sorted_sizes = np.take_along_axis(sizes_matrix, sorted_indices, axis=1)

        # 2. Determine Group Splits based on AVERAGE distribution
        avg_sorted_sizes = np.mean(sorted_sizes, axis=0)
        
        cumsum_sizes = np.cumsum(avg_sorted_sizes)
        total_size = cumsum_sizes[-1]
        
        if total_size == 0 or num_groups <= 0:
            for b in range(B):
                for c in batch_candidates[b]:
                    self._add_patch(final_patch_list, c, 1)
            return

        target_sum = total_size / num_groups
        boundaries = np.arange(1, num_groups) * target_sum
        
        # 3. Rank -> Group ID
        rank_to_group_id = np.searchsorted(boundaries, cumsum_sizes) + 1
        
        # 4. Map back and create patches
        for b in range(B):
            for rank in range(N):
                spatial_idx_at_rank = sorted_indices[b, rank]
                assigned_group = int(rank_to_group_id[rank])
                c = batch_candidates[b][spatial_idx_at_rank]
                self._add_patch(final_patch_list, c, assigned_group)

    def _add_patch(self, patch_list, c, group_id):
        patch_list.append(Patch(
            image_idx=c['image_idx'],
            spatial_idx=c['spatial_idx'],
            data=c['data'],
            res_level=c['res_level'],
            group_id=group_id
        ))

    # --- Standard Helpers ---

    def _create_patches_with_group(self, patch_list, image, b_idx, lvl, config, dtype, group_id, compression):
        ph, pw = config.patch_size
        H, W = image.shape[:2]
        gh, gw = (H + ph - 1) // ph, (W + pw - 1) // pw
        idx = 0
        for r in range(gh):
            for c in range(gw):
                y1, x1 = r * ph, c * pw
                y2, x2 = min(H, y1 + ph), min(W, x1 + pw)
                crop = np.ascontiguousarray(image[y1:y2, x1:x2])
                compressed = zlib.compress(crop.astype(dtype).tobytes(), level=compression)
                patch_list.append(Patch(b_idx, idx, compressed, lvl, group_id))
                idx += 1

    def _collect_residual_candidates(self, candidate_list, image, b_idx, lvl, config, dtype, compression):
        ph, pw = config.patch_size
        H, W = image.shape[:2]
        gh, gw = (H + ph - 1) // ph, (W + pw - 1) // pw
        idx = 0
        for r in range(gh):
            for c in range(gw):
                y1, x1 = r * ph, c * pw
                y2, x2 = min(H, y1 + ph), min(W, x1 + pw)
                crop = np.ascontiguousarray(image[y1:y2, x1:x2])
                compressed = zlib.compress(crop.astype(dtype).tobytes(), level=compression)
                candidate_list.append({
                    'image_idx': b_idx, 'spatial_idx': idx, 'res_level': lvl,
                    'data': compressed, 'size': len(compressed)
                })
                idx += 1
