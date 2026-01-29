import numpy as np
from typing import List
import zlib, cv2
from concurrent.futures import ThreadPoolExecutor

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

class FullImageCompressionPolicy(ITransmissionPolicy):
    """
    Compresses the entire image at once (JPEG/PNG).
    Sends data in the first patch (spatial_idx=0).
    Sends empty patches for the rest to satisfy Worker logic.
    Forces sequential processing but high compression efficiency.
    """

    def encode(self, images: np.ndarray, config: ExperimentConfig) -> List[Patch]:
        patches = []
        B, H, W, C = images.shape
        ph, pw = config.patch_size

        gh, gw = H // ph, W // pw
        num_patches = gh * gw
        
        fmt = config.transmission_kwargs.get('format', 'jpg').lower()
        quality = config.transmission_kwargs.get('quality', 95 if fmt == 'jpg' else 3)
        ext = '.jpg' if fmt == 'jpg' else '.png'
        
        encode_params = []
        if fmt == 'jpg':
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif fmt == 'png':
            encode_params = [cv2.IMWRITE_PNG_COMPRESSION, quality]
            
        for b in range(B):
            # 1. Encode Full Image
            success, encoded_img = cv2.imencode(ext, images[b], encode_params)
            if not success:
                raise RuntimeError(f"Failed to encode full image {b} with format {fmt}")
            
            full_data = encoded_img.tobytes()
            
            # 2. Create Patches
            # Patch 0 carries data
            patches.append(Patch(image_idx=b, spatial_idx=0, data=full_data))
            
            # Patches 1..N-1 carry empty data (metadata only)
            empty_data = b''
            for i in range(1, num_patches):
                patches.append(Patch(image_idx=b, spatial_idx=i, data=empty_data))
                
        return patches

    def decode(self, patches: List[Patch], config: ExperimentConfig) -> np.ndarray:
        B = config.batch_size
        H, W, C = config.image_shape
        
        batch_tensor = np.zeros((B, H, W, C), dtype=np.uint8)
        
        # We expect patches to come in any order, but we only need spatial_idx=0
        # Group by image_idx
        data_patches = {}
        
        for p in patches:
            if p.spatial_idx == 0 and len(p.data) > 0:
                 data_patches[p.image_idx] = p.data
        
        for b, data in data_patches.items():
            try:
                raw_bytes = np.frombuffer(data, dtype=np.uint8)
                img = cv2.imdecode(raw_bytes, cv2.IMREAD_COLOR)
                
                if img is None:
                     raise ValueError(f"imdecode returned None for image {b}")
                
                if img.shape != (H, W, C):
                     # Resize or warn? Best effort resize if dimension mismatch (e.g. padding issues)
                     # But strict policy assumes matching config. 
                     if img.shape[:2] != (H, W):
                         img = cv2.resize(img, (W, H))
                         
                batch_tensor[b] = img
                
            except Exception as e:
                print(f"!!! [FullImageCompressionPolicy] Decode failed for image {b}: {e}")
                
        return batch_tensor

class LaplacianPyramidPolicy(ITransmissionPolicy):
    """
    Generalized Laplacian Pyramid with Zlib Compression.
    Enforces strict uint8 consistency for lossless residual coding.
    """

    def encode(self, images: np.ndarray, config: ExperimentConfig) -> List[Patch]:
        B = images.shape[0]
        patches = []
        
        # Optimize: Parallelize per-image processing
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._process_image_encode, b, images[b], config)
                for b in range(B)
            ]
            for f in futures:
                patches.extend(f.result())

        return patches

    def decode(self, patches: List[Patch], config: ExperimentConfig) -> np.ndarray:
        B = config.batch_size
        H, W, C = config.image_shape
        
        # Group patches by Batch
        layer_data_per_batch = {b: [] for b in range(B)}
        for p in patches:
            layer_data_per_batch[p.image_idx].append(p)
            
        final_images = np.zeros((B, H, W, C), dtype=np.uint8)

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

    def _process_image_encode(self, b_idx, image, config):
        levels = sorted(config.transmission_kwargs.get('pyramid_levels', [2, 1, 0]), reverse=True)
        max_lvl = max(levels)
        H, W = image.shape[:2]
        
        local_patches = []
        
        # Build Pyramid
        gaussians = {0: image}
        curr = image
        for i in range(1, max_lvl + 1):
            curr = cv2.pyrDown(curr)
            gaussians[i] = curr
        
        prev_img = None
        prev_lvl = -1
        
        for lvl in levels:
            curr_g = gaussians[lvl]
            
            if prev_img is None:
                # Base Layer (uint8)
                self._create_patches_vectorized(local_patches, curr_g, b_idx, lvl, config, np.uint8)
                prev_img = curr_g
                prev_lvl = lvl
            else:
                # Residual Layer (int16)
                pred = self._iterative_upsample(prev_img, prev_lvl, lvl, H, W)
                residual = curr_g.astype(np.int16) - pred.astype(np.int16)
                
                self._create_patches_vectorized(local_patches, residual, b_idx, lvl, config, np.int16)
                
                prev_img = curr_g
                prev_lvl = lvl
                
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

        # Vectorized crop extraction: (gh, gw, ph, pw, C)
        # Assumes strict divisibility.
        assert H % ph == 0 and W % pw == 0, f"Image shape {(H, W)} must be divisible by patch size {(ph, pw)}"
        gh, gw = H // ph, W // pw
        
        crops = image.reshape(gh, ph, gw, pw, C).transpose(0, 2, 1, 3, 4).reshape(-1, ph, pw, C)
        
        # Sequential Compression (ThreadPool handles parallel images)
        num_crops = crops.shape[0]

        for i in range(num_crops):
             # To ensure index matches original logic (row-major), we loop sequentially
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
            # Optimization: Cache decompressed bytes
            if hasattr(patch, '_decompressed_cache'):
                raw = patch._decompressed_cache
            else:
                raw = zlib.decompress(patch.data)
                patch._decompressed_cache = raw
            
            chunk = np.frombuffer(raw, dtype=dtype).reshape(ph, pw, C)
            canvas[y:y+th, x:x+tw] = chunk

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
        B = images.shape[0]
        num_groups = config.transmission_kwargs.get('num_groups', 4)

        batch_candidates = [[] for _ in range(B)]

        # Generate Candidates (Parallel per Image)
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._process_image_encode_progressive, b, images[b], config)
                for b in range(B)
            ]
            for b, (local_patches, local_candidates) in enumerate([f.result() for f in futures]):
                 # local_patches are Base layers (always group 0)
                 patches.extend(local_patches)
                 # local_candidates are Residuals waiting for grouping
                 batch_candidates[b] = local_candidates

        # Apply Grouping (Batch-Wise) based on Strategy
        grouping_strategy = config.transmission_kwargs.get('grouping_strategy', 'uniform_diff')
        
        if any(batch_candidates):
            if grouping_strategy == 'uniform_diff':
                self._apply_uniform_diff_grouping(patches, batch_candidates, num_groups)
            elif grouping_strategy == 'random':
                self._apply_random_grouping(patches, batch_candidates, num_groups)
            elif grouping_strategy == 'grid':
                self._apply_grid_grouping(patches, batch_candidates, num_groups)
            elif grouping_strategy == 'geometric':
                self._apply_geometric_grouping(patches, batch_candidates, num_groups)
            else:
                print(f"!!! [Transmission] Unknown strategy '{grouping_strategy}'. Fallback to uniform_diff.")
                self._apply_uniform_diff_grouping(patches, batch_candidates, num_groups)

        # Inject Metadata & Calculate Stats
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

        # Sort by Group ID
        patches.sort(key=lambda x: x.group_id)
        
        return patches

    def _process_image_encode_progressive(self, b_idx, image, config):
        levels = sorted(config.transmission_kwargs.get('pyramid_levels', [2, 0]), reverse=True)
        max_lvl = max(levels)
        H, W = image.shape[:2]
        comp_lvl = config.transmission_kwargs.get('compression_level', 1)
        
        local_patches = []
        local_candidates = []
        
        # Build Pyramid
        gaussians = {0: image}
        curr = image
        for i in range(1, max_lvl + 1):
            curr = cv2.pyrDown(curr)
            gaussians[i] = curr
        
        prev_img = None
        prev_lvl = -1
        
        for lvl in levels:
            curr_g = gaussians[lvl]
            
            if prev_img is None:
                # Group 0: Base Layer (Always Group 0)
                # Use vectorized creation
                self._create_patches_with_group_vectorized(
                    local_patches, curr_g, b_idx, lvl, config, np.uint8, 
                    group_id=0, compression=comp_lvl
                )
                prev_img = curr_g
                prev_lvl = lvl
            else:
                # Residual Layer: Collect
                pred = self._iterative_upsample(prev_img, prev_lvl, lvl, H, W)
                residual = curr_g.astype(np.int16) - pred.astype(np.int16)
                
                # Use vectorized collection
                self._collect_residual_candidates_vectorized(
                    local_candidates, residual, b_idx, lvl, config, 
                    dtype=np.int16, compression=comp_lvl
                )
                
                prev_img = curr_g
                prev_lvl = lvl
        
        return local_patches, local_candidates

    # --- Vectorized Helpers for Progressive ---

    def _create_patches_with_group_vectorized(self, patch_list, image, b_idx, lvl, config, dtype, group_id, compression):
        ph, pw = config.patch_size
        H, W, C = image.shape

        # Strict divisibility check
        if H % ph != 0 or W % pw != 0:
            raise ValueError(f"[ProgressiveLPyramidPolicy] Image shape {(H, W)} not divisible by patch {(ph, pw)}")
            
        gh, gw = H // ph, W // pw
        
        # Vectorized crop
        
        # Vectorized crop
        crops = image.reshape(gh, ph, gw, pw, C).transpose(0, 2, 1, 3, 4).reshape(-1, ph, pw, C)
        num_crops = crops.shape[0]

        for i in range(num_crops):
            data = crops[i].astype(dtype).tobytes()
            compressed = zlib.compress(data, level=compression)
            patch_list.append(Patch(b_idx, i, compressed, lvl, group_id))

    def _collect_residual_candidates_vectorized(self, candidate_list, image, b_idx, lvl, config, dtype, compression):
        ph, pw = config.patch_size
        H, W, C = image.shape

        # Strict divisibility check
        if H % ph != 0 or W % pw != 0:
             # Just return or error? 
             # Residuals should match base resolution if pyramid ops are consistent.
             raise ValueError(f"[Residual] Image shape {(H, W)} not divisible by patch {(ph, pw)}")

        gh, gw = H // ph, W // pw
        
        crops = image.reshape(gh, ph, gw, pw, C).transpose(0, 2, 1, 3, 4).reshape(-1, ph, pw, C)
        num_crops = crops.shape[0]

        for i in range(num_crops):
            data = crops[i].astype(dtype).tobytes()
            compressed = zlib.compress(data, level=compression)
            candidate_list.append({
                'image_idx': b_idx, 'spatial_idx': i, 'res_level': lvl,
                'data': compressed, 'size': len(compressed)
            })

    def _apply_random_grouping(self, final_patch_list, batch_candidates, num_groups):
        """Random assignment of patches to groups."""
        for b_idx, candidates in enumerate(batch_candidates):
            if not candidates: continue
            
            num_tokens = len(candidates)
            # Random group IDs: 1..num_groups
            group_ids = np.random.randint(1, num_groups + 1, size=num_tokens)
            
            for i, cand in enumerate(candidates):
                self._add_patch(
                    final_patch_list, cand,
                    group_id=int(group_ids[i])
                )

    def _apply_grid_grouping(self, final_patch_list, batch_candidates, num_groups):
        """Grid-based deterministic assignment."""
        s = int(num_groups ** 0.5)
        
        for b_idx, candidates in enumerate(batch_candidates):
            if not candidates: continue
            
            num_tokens = len(candidates)
            side = int(num_tokens ** 0.5)
            
            # Pattern: 1..num_groups mapped to s x s grid
            pattern = np.arange(1, num_groups + 1).reshape(s, s)
            
            rep_h = (side + s - 1) // s
            rep_w = (side + s - 1) // s
            
            grid_2d = np.tile(pattern, (rep_h, rep_w))[:side, :side]
            group_ids = grid_2d.flatten()
            
            # Best effort for non-square or mismatches
            if len(group_ids) < num_tokens:
                 group_ids = np.resize(group_ids, num_tokens)
            elif len(group_ids) > num_tokens:
                 group_ids = group_ids[:num_tokens]

            for i, cand in enumerate(candidates):
                self._add_patch(
                    final_patch_list, cand,
                    group_id=int(group_ids[i])
                )

    def _apply_geometric_grouping(self, final_patch_list, batch_candidates, num_groups):
        """Geometric distribution based assignment."""
        for b_idx, candidates in enumerate(batch_candidates):
            if not candidates: continue
            
            num_tokens = len(candidates)
            probs = np.random.rand(num_tokens)
            
            # floor(-log2(1 - p)) + 1
            group_ids = np.floor(-np.log2(1 - probs)) + 1
            group_ids = np.clip(group_ids, 1, num_groups).astype(int)
            
            for i, cand in enumerate(candidates):
                self._add_patch(
                    final_patch_list, cand,
                    group_id=int(group_ids[i])
                )

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
