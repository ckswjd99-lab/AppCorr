import numpy as np
import cv2
import zlib
from typing import List, Generator
from concurrent.futures import ThreadPoolExecutor
from offload.common.protocol import Patch, ExperimentConfig
from .laplacian import LaplacianPyramidPolicy

class ProgressiveLPyramidPolicy(LaplacianPyramidPolicy):
    """
    Progressive Laplacian Pyramid with 'Uniform Diff' Grouping.
    Includes debug stats for group capacity.
    """

    def encode(self, images: np.ndarray, config: ExperimentConfig) -> Generator[List[Patch], None, None]:
        B = images.shape[0]
        num_groups = config.transmission_kwargs.get('num_groups', 4)
        levels = sorted(config.transmission_kwargs.get('pyramid_levels', [2, 0]), reverse=True)

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

        # When there is only one logical correction group, stream each residual level
        # as soon as it is ready so the scheduler can run per-level correction passes.
        if num_groups == 1 and len(levels) > 1:
            for phase_id, lvl in enumerate(levels[1:], start=1):
                prev_lvl = levels[phase_id - 1]
                group_patches = []
                with ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(
                            self._process_image_single_residual_level,
                            b,
                            gaussians_batch[b],
                            prev_lvl,
                            lvl,
                            phase_id,
                            config,
                        )
                        for b in range(B)
                    ]
                    for f in futures:
                        group_patches.extend(f.result())

                if group_patches:
                    total_in_group = len(group_patches)
                    for p in group_patches:
                        p.batch_group_total = total_in_group
                    yield group_patches
            return

        grouping_strategy = config.transmission_kwargs.get('grouping_strategy', 'uniform_diff')

        if grouping_strategy == 'uniform_diff':
            # Collect all then group (Non-pipelined fallback)
            batch_candidates = [[] for _ in range(B)]
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(self._process_image_residuals, b, gaussians_batch[b], config)
                    for b in range(B)
                ]
                for b, f in enumerate(futures):
                    batch_candidates[b] = f.result()
                    
            residual_patches = []
            if any(batch_candidates):
                self._apply_uniform_diff_grouping(residual_patches, batch_candidates, num_groups)

            group_counts = {}
            for p in residual_patches:
                group_counts[p.group_id] = group_counts.get(p.group_id, 0) + 1
            for p in residual_patches:
                p.batch_group_total = group_counts[p.group_id]

            grouped = {}
            for p in residual_patches:
                grouped.setdefault(p.group_id, []).append(p)
            for g in sorted(grouped.keys()):
                if grouped[g]:
                    yield grouped[g]
        else:
            # Pipelined transmission for data-independent strategies
            # Pre-calculate group assignments
            residual_structure = self._collect_residual_metadata(gaussians_batch[0], config)
            N = len(residual_structure)
            group_assignments = self._precompute_group_assignments(grouping_strategy, N, num_groups)
            
            # Compress and yield group-by-group
            for g_id in range(1, num_groups + 1):
                group_patches = []
                with ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(self._process_image_group_residuals, b, gaussians_batch[b], residual_structure, group_assignments, g_id, config)
                        for b in range(B)
                    ]
                    for f in futures:
                        group_patches.extend(f.result())
                
                if group_patches:
                    total_in_group = len(group_patches)
                    for p in group_patches:
                        p.batch_group_total = total_in_group
                    yield group_patches

    def _process_image_single_residual_level(self, b_idx, gaussians, prev_lvl, lvl, group_id, config):
        H, W = config.image_shape[:2]
        comp_lvl = config.transmission_kwargs.get('compression_level', 1)

        prev_img = gaussians[prev_lvl]
        curr_g = gaussians[lvl]
        pred = self._iterative_upsample(prev_img, prev_lvl, lvl, H, W)
        residual = curr_g.astype(np.int16) - pred.astype(np.int16)

        local_patches = []
        self._create_patches_with_group_vectorized(
            local_patches,
            residual,
            b_idx,
            lvl,
            config,
            np.int16,
            group_id=group_id,
            compression=comp_lvl,
        )
        return local_patches

    def _collect_residual_metadata(self, gaussians, config):
        """Map pyramid structure to get spatial_idx and res_level."""
        levels = sorted(config.transmission_kwargs.get('pyramid_levels', [2, 0]), reverse=True)
        H, W = config.image_shape[:2]
        ph, pw = config.patch_size
        
        structure = []
        for lvl in levels[1:]:
            scale = 2 ** lvl
            lh, lw = H // scale, W // scale
            gh, gw = lh // ph, lw // pw
            num_crops = gh * gw
            for i in range(num_crops):
                structure.append({'spatial_idx': i, 'res_level': lvl})
        return structure

    def _precompute_group_assignments(self, strategy, N, num_groups):
        """Pre-calculate group ID for N items based on strategy."""
        if strategy == 'grid':
            # Simplified grid: side of total tokens
            s = int(num_groups ** 0.5)
            side = int(N ** 0.5)
            pattern = np.arange(1, num_groups + 1).reshape(s, s)
            rep_h = (side + s - 1) // s
            rep_w = (side + s - 1) // s
            grid_2d = np.tile(pattern, (rep_h, rep_w))[:side, :side]
            group_ids = grid_2d.flatten()
            if len(group_ids) < N:
                 group_ids = np.resize(group_ids, N)
            elif len(group_ids) > N:
                 group_ids = group_ids[:N]
            return group_ids
            
        elif strategy == 'random':
            return np.random.randint(1, num_groups + 1, size=N)
            
        elif strategy == 'geometric':
            probs = np.random.rand(N)
            group_ids = np.floor(-np.log2(1 - probs)) + 1
            return np.clip(group_ids, 1, num_groups).astype(int)
            
        else:
            # Fallback to group 1
            return np.ones(N, dtype=int)

    def _process_image_group_residuals(self, b_idx, gaussians, structure, group_assignments, target_group, config):
        """Compress only patches belonging to target_group for one image."""
        H, W = config.image_shape[:2]
        comp_lvl = config.transmission_kwargs.get('compression_level', 1)
        levels = sorted(config.transmission_kwargs.get('pyramid_levels', [2, 0]), reverse=True)
        local_patches = []
        
        # Up-sample sequentially and collect group members
        
        prev_lvl = levels[0]
        prev_img = gaussians[prev_lvl]
        
        struct_idx = 0
        for lvl in levels[1:]:
            curr_g = gaussians[lvl]
            pred = self._iterative_upsample(prev_img, prev_lvl, lvl, H, W)
            residual = curr_g.astype(np.int16) - pred.astype(np.int16)
            
            # Identify patches in this level
            ph, pw = config.patch_size
            rh, rw = residual.shape[:2]
            gh, gw = rh // ph, rw // pw
            num_crops = gh * gw
            
            # Check which patches in this level belong to target_group
            for i in range(num_crops):
                if group_assignments[struct_idx] == target_group:
                    # Compress
                    y = (i // gw) * ph
                    x = (i % gw) * pw
                    crop = residual[y:y+ph, x:x+pw]
                    data = crop.astype(np.int16).tobytes()
                    compressed = zlib.compress(data, level=comp_lvl)
                    local_patches.append(Patch(b_idx, i, compressed, lvl, target_group))
                struct_idx += 1
            
            prev_img = curr_g
            prev_lvl = lvl
            
        return local_patches


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
