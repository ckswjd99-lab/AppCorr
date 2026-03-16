import numpy as np
from typing import List, Generator
import zlib, cv2
from concurrent.futures import ThreadPoolExecutor

from offload.common.protocol import Patch, ExperimentConfig
from .interface import ITransmissionPolicy

class RawTransmissionPolicy(ITransmissionPolicy):
    """Encodes/Decodes patches with Batch dimension support using vectorized ops."""

    def encode(self, images: np.ndarray, config: ExperimentConfig) -> Generator[List[Patch], None, None]:
        """Encode images to patches using vectorized operations."""
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
                
        yield patches

    def decode(self, patches: List[Patch], config: ExperimentConfig, canvas: np.ndarray = None) -> np.ndarray:
        """Decode patches into image canvas."""
        B = config.batch_size
        H, W, C = config.image_shape
        ph, pw = config.patch_size
        gw = W // pw
        
        # Initialize canvas if not provided
        if canvas is None:
            batch_tensor = np.zeros((B, H, W, C), dtype=np.uint8)
        else:
            batch_tensor = canvas
        
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

    def encode(self, images: np.ndarray, config: ExperimentConfig) -> Generator[List[Patch], None, None]:
        """Encode images into zlib compressed patches."""
        patches = []
        B, H, W, C = images.shape
        ph, pw = config.patch_size
        gh, gw = H // ph, W // pw

        compression_level = config.transmission_kwargs.get('compression_level', 1)
        
        # Vectorized Patchify (Same as Raw)
        reshaped = images.reshape(B, gh, ph, gw, pw, C)
        transposed = reshaped.transpose(0, 1, 3, 2, 4, 5)
        patch_tensor = transposed.reshape(B, gh * gw, ph, pw, C)
        
        num_patches = gh * gw
        
        all_patches = []
        # Compress and Wrap
        for b in range(B):
            for i in range(num_patches):
                raw_bytes = patch_tensor[b, i].tobytes()
                compressed_data = zlib.compress(raw_bytes, level=compression_level)
                all_patches.append(Patch(image_idx=b, spatial_idx=i, data=compressed_data))
                
        yield all_patches

    def decode(self, patches: List[Patch], config: ExperimentConfig, canvas: np.ndarray = None) -> np.ndarray:
        """Decode zlib compressed patches."""
        B = config.batch_size
        H, W, C = config.image_shape
        ph, pw = config.patch_size
        gw = W // pw
        
        if canvas is None:
            batch_tensor = np.zeros((B, H, W, C), dtype=np.uint8)
        else:
            batch_tensor = canvas
        
        def _decompress_patch(p):
            try:
                raw_bytes = zlib.decompress(p.data)
                chunk = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(ph, pw, C)
                r, c = divmod(p.spatial_idx, gw)
                y, x = r * ph, c * pw
                batch_tensor[p.image_idx, y:y+ph, x:x+pw] = chunk
            except Exception as e:
                print(f"!!! [ZlibPolicy] Decompression failed for patch {p.spatial_idx}: {e}")

        with ThreadPoolExecutor() as executor:
            executor.map(_decompress_patch, patches)
            
        return batch_tensor

class FullImageCompressionPolicy(ITransmissionPolicy):
    """
    Compresses the entire image at once (JPEG/PNG).
    Sends data in the first patch (spatial_idx=0).
    Sends empty patches for the rest to satisfy Worker logic.
    Forces sequential processing but high compression efficiency.
    """

    def encode(self, images: np.ndarray, config: ExperimentConfig) -> Generator[List[Patch], None, None]:
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
            
        all_patches = []
        for b in range(B):
            # Encode Full Image
            success, encoded_img = cv2.imencode(ext, images[b], encode_params)
            if not success:
                raise RuntimeError(f"Failed to encode full image {b} with format {fmt}")
            
            full_data = encoded_img.tobytes()
            
            # Create Patches
            # Patch 0 carries data
            all_patches.append(Patch(image_idx=b, spatial_idx=0, data=full_data))
            
            # Patches 1..N-1 carry empty data (metadata only)
            empty_data = b''
            for i in range(1, num_patches):
                all_patches.append(Patch(image_idx=b, spatial_idx=i, data=empty_data))
                
        yield all_patches

    def decode(self, patches: List[Patch], config: ExperimentConfig, canvas: np.ndarray = None) -> np.ndarray:
        B = config.batch_size
        H, W, C = config.image_shape
        
        if canvas is None:
            batch_tensor = np.zeros((B, H, W, C), dtype=np.uint8)
        else:
            batch_tensor = canvas
        
        # Extract full image data from spatial_idx=0
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
                     # Best effort resize if dimension mismatch
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
        # Yield patches layer by layer
        levels = sorted(config.transmission_kwargs.get('pyramid_levels', [2, 1, 0]), reverse=True)
        
        for tgt_lvl_idx, tgt_lvl in enumerate(levels):
            layer_patches = []
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(self._process_image_encode_single_layer, b, images[b], tgt_lvl_idx, levels, config)
                    for b in range(B)
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
        H, W = image.shape[:2]
        local_patches = []
        
        gaussians = {0: image}
        curr = image
        for i in range(1, max_lvl + 1):
            curr = cv2.pyrDown(curr)
            gaussians[i] = curr
            
        tgt_lvl = levels[tgt_lvl_idx]
        curr_g = gaussians[tgt_lvl]
        
        if tgt_lvl_idx == 0:
            # Base Layer
            self._create_patches_vectorized(local_patches, curr_g, b_idx, tgt_lvl, config, np.uint8)
        else:
            prev_lvl = levels[tgt_lvl_idx - 1]
            prev_g = gaussians[prev_lvl]
            pred = self._iterative_upsample(prev_g, prev_lvl, tgt_lvl, H, W)
            residual = curr_g.astype(np.int16) - pred.astype(np.int16)
            self._create_patches_vectorized(local_patches, residual, b_idx, tgt_lvl, config, np.int16)
            
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

class ProgressiveLPyramidPolicy(LaplacianPyramidPolicy):
    """
    Progressive Laplacian Pyramid with 'Uniform Diff' Grouping.
    Includes debug stats for group capacity.
    """

    def encode(self, images: np.ndarray, config: ExperimentConfig) -> Generator[List[Patch], None, None]:
        B = images.shape[0]
        num_groups = config.transmission_kwargs.get('num_groups', 4)

        base_patches = []
        gaussians_batch = [None] * B

        # Generate base layers
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._process_image_base_layer, b, images[b], config)
                for b in range(B)
            ]
            for b, f in enumerate(futures):
                local_patches, gaussians = f.result()
                base_patches.extend(local_patches)
                gaussians_batch[b] = gaussians
                
        # Add metadata to group 0
        g0_total = len(base_patches)
        for p in base_patches:
            p.batch_group_total = g0_total

        yield base_patches # Yield Group 0 (Base Layer) Immediately!

        # Compute remaining residuals
        batch_candidates = [[] for _ in range(B)]
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._process_image_residuals, b, gaussians_batch[b], config)
                for b in range(B)
            ]
            for b, f in enumerate(futures):
                batch_candidates[b] = f.result()
                 
        residual_patches = []
        # Group residuals
        grouping_strategy = config.transmission_kwargs.get('grouping_strategy', 'uniform_diff')
        
        if any(batch_candidates):
            if grouping_strategy == 'uniform_diff':
                self._apply_uniform_diff_grouping(residual_patches, batch_candidates, num_groups)
            elif grouping_strategy == 'random':
                self._apply_random_grouping(residual_patches, batch_candidates, num_groups)
            elif grouping_strategy == 'grid':
                self._apply_grid_grouping(residual_patches, batch_candidates, num_groups)
            elif grouping_strategy == 'geometric':
                self._apply_geometric_grouping(residual_patches, batch_candidates, num_groups)
            else:
                print(f"!!! [Transmission] Unknown strategy '{grouping_strategy}'. Fallback to uniform_diff.")
                self._apply_uniform_diff_grouping(residual_patches, batch_candidates, num_groups)

        # Add metadata to residuals
        group_counts = {}
        for p in residual_patches:
            group_counts[p.group_id] = group_counts.get(p.group_id, 0) + 1
            
        for p in residual_patches:
            p.batch_group_total = group_counts[p.group_id]

        # Group residuals into lists and yield
        grouped = {}
        for p in residual_patches:
             grouped.setdefault(p.group_id, []).append(p)
             
        # Yield groups sequentially by ID
        for g in sorted(grouped.keys()):
             if grouped[g]:
                  yield grouped[g]

    def _process_image_base_layer(self, b_idx, image, config):
        levels = sorted(config.transmission_kwargs.get('pyramid_levels', [2, 0]), reverse=True)
        max_lvl = max(levels)
        comp_lvl = config.transmission_kwargs.get('compression_level', 1)
        
        gaussians = {0: image}
        curr = image
        for i in range(1, max_lvl + 1):
            curr = cv2.pyrDown(curr)
            gaussians[i] = curr
            
        local_patches = []
        base_lvl = levels[0] # Highest level index is the base layer
        
        # Use vectorized creation
        self._create_patches_with_group_vectorized(
            local_patches, gaussians[base_lvl], b_idx, base_lvl, config, np.uint8, 
            group_id=0, compression=comp_lvl
        )
        return local_patches, gaussians

    def _process_image_residuals(self, b_idx, gaussians, config):
        levels = sorted(config.transmission_kwargs.get('pyramid_levels', [2, 0]), reverse=True)
        H, W = config.image_shape[:2]
        comp_lvl = config.transmission_kwargs.get('compression_level', 1)
        
        local_candidates = []
        
        # Start from base layer and upsample
        prev_lvl = levels[0]
        prev_img = gaussians[prev_lvl]
        
        for lvl in levels[1:]:
            curr_g = gaussians[lvl]
            
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
        
        return local_candidates

    # --- Vectorized Helpers for Progressive ---

    def _create_patches_with_group_vectorized(self, patch_list, image, b_idx, lvl, config, dtype, group_id, compression):
        ph, pw = config.patch_size
        H, W, C = image.shape

        # Verify exact divisibility
        if H % ph != 0 or W % pw != 0:
            raise ValueError(f"[ProgressiveLPyramidPolicy] Image shape {(H, W)} not divisible by patch {(ph, pw)}")
            
        gh, gw = H // ph, W // pw
        
        # Extract crops via reshaping
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
        """Assign patches to random groups."""
        for b_idx, candidates in enumerate(batch_candidates):
            if not candidates: continue
            
            num_tokens = len(candidates)
            # Generate random group IDs
            group_ids = np.random.randint(1, num_groups + 1, size=num_tokens)
            
            for i, cand in enumerate(candidates):
                self._add_patch(
                    final_patch_list, cand,
                    group_id=int(group_ids[i])
                )

    def _apply_grid_grouping(self, final_patch_list, batch_candidates, num_groups):
        """Assign patches based on a grid pattern."""
        s = int(num_groups ** 0.5)
        
        for b_idx, candidates in enumerate(batch_candidates):
            if not candidates: continue
            
            num_tokens = len(candidates)
            side = int(num_tokens ** 0.5)
            # Create grid assignment pattern
            pattern = np.arange(1, num_groups + 1).reshape(s, s)
            
            rep_h = (side + s - 1) // s
            rep_w = (side + s - 1) // s
            
            grid_2d = np.tile(pattern, (rep_h, rep_w))[:side, :side]
            group_ids = grid_2d.flatten()
            
            # Resize array for mismatches
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
        """Assign patches based on geometric distribution."""
        for b_idx, candidates in enumerate(batch_candidates):
            if not candidates: continue
            
            num_tokens = len(candidates)
            probs = np.random.rand(num_tokens)
            
            # Calculate group ID
            group_ids = np.floor(-np.log2(1 - probs)) + 1
            group_ids = np.clip(group_ids, 1, num_groups).astype(int)
            
            for i, cand in enumerate(candidates):
                self._add_patch(
                    final_patch_list, cand,
                    group_id=int(group_ids[i])
                )

    def _apply_uniform_diff_grouping(self, final_patch_list, batch_candidates, num_groups):
        """Assign group IDs based on average batch size distribution."""
        B = len(batch_candidates)
        if B == 0: return
        N = len(batch_candidates[0])
        if N == 0: return

        # Extract patch sizes
        sizes_matrix = np.zeros((B, N), dtype=np.int32)
        for b in range(B):
            for i in range(N):
                sizes_matrix[b, i] = batch_candidates[b][i]['size']

        # Sort patch sizes per batch
        sorted_indices = np.argsort(sizes_matrix, axis=1)
        sorted_sizes = np.take_along_axis(sizes_matrix, sorted_indices, axis=1)

        # Calculate group splits using average sizes
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
        
        # Map rank to group ID
        rank_to_group_id = np.searchsorted(boundaries, cumsum_sizes) + 1
        
        # Assign groups to patches
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
