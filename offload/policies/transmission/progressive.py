import numpy as np
import zlib
from typing import List, Generator
from concurrent.futures import ThreadPoolExecutor
from offload.common.protocol import Patch, ExperimentConfig, normalize_appcorr_kwargs
from .laplacian import LaplacianPyramidPolicy

class ProgressiveLPyramidPolicy(LaplacianPyramidPolicy):
    """
    Progressive Laplacian Pyramid with 'Uniform Diff' Grouping.
    Includes debug stats for group capacity.
    """

    @staticmethod
    def _compute_patch_residual_rms(crop: np.ndarray) -> float:
        crop_f32 = crop.astype(np.float32, copy=False)
        return float(np.sqrt(np.square(crop_f32, dtype=np.float32).mean(dtype=np.float32)))

    @staticmethod
    def _compute_patch_residual_energy(crop: np.ndarray) -> float:
        crop_f32 = crop.astype(np.float32, copy=False)
        return float(np.square(crop_f32, dtype=np.float32).sum(dtype=np.float32))

    def _resolve_mobile_pscore(self, config: ExperimentConfig) -> str:
        return str(
            normalize_appcorr_kwargs(
                getattr(config, "appcorr_kwargs", {}),
                getattr(config, "transmission_kwargs", {}),
            ).get("mobile_pscore", "none")
        )

    def _compute_patch_pscore_hint(self, crop: np.ndarray, mobile_pscore: str) -> float:
        if mobile_pscore == "residual_energy":
            return self._compute_patch_residual_energy(crop)
        return self._compute_patch_residual_rms(crop)

    def encode(self, images: np.ndarray, config: ExperimentConfig) -> Generator[List[Patch], None, None]:
        image_list = self._as_image_list(images)
        B = len(image_list)
        num_groups = config.transmission_kwargs.get('num_groups', 4)
        mobile_pscore = self._resolve_mobile_pscore(config)
        preserve = self._is_preserve_input_shape(config)

        base_patches = []
        gaussians_batch = [None] * B
        image_hws = [img.shape[:2] for img in image_list] if preserve else [None] * B

        # Generate base layers
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(self._process_image_base_layer, b, image, config, image_hws[b])
                for b, image in enumerate(image_list)
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

        grouping_strategy = config.transmission_kwargs.get('grouping_strategy', 'uniform_diff')

        if grouping_strategy == 'uniform_diff':
            # Collect all then group (Non-pipelined fallback)
            batch_candidates = [[] for _ in range(B)]
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(self._process_image_residuals, b, gaussians_batch[b], config, mobile_pscore, image_hws[b])
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
            # Pre-calculate group assignments per image (may differ with preserve_input_shape)
            per_image_assignments = []
            for b in range(B):
                if grouping_strategy in ('top_energy', 'top_energy_threshold'):
                    residual_structure = self._collect_residual_metadata_scored(gaussians_batch[b], config, image_hws[b])
                else:
                    residual_structure = self._collect_residual_metadata(gaussians_batch[b], config, image_hws[b])
                per_image_assignments.append(
                    self._precompute_group_assignments(grouping_strategy, residual_structure, num_groups, config)
                )

            # Compress and yield group-by-group
            for g_id in range(1, num_groups + 1):
                group_patches = []
                with ThreadPoolExecutor() as executor:
                    futures = [
                        executor.submit(
                            self._process_image_group_residuals,
                            b,
                            gaussians_batch[b],
                            per_image_assignments[b],
                            g_id,
                            config,
                            mobile_pscore,
                            image_hws[b],
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

    def _collect_residual_metadata(self, gaussians, config, image_hw=None):
        """Map pyramid structure to get spatial_idx and res_level."""
        levels = sorted(config.transmission_kwargs.get('pyramid_levels', [2, 0]), reverse=True)
        ph, pw = config.patch_size

        structure = []
        for lvl in levels[1:]:
            target_hw = self._target_hw_for_level(config, lvl, image_hw)
            lh, lw = target_hw
            gh, gw = lh // ph, lw // pw
            num_crops = gh * gw
            for i in range(num_crops):
                row, col = divmod(i, gw)
                structure.append({
                    'spatial_idx': i,
                    'res_level': lvl,
                    'grid_hw': (gh, gw),
                    'row': row,
                    'col': col,
                })
        return structure

    def _collect_residual_metadata_scored(self, gaussians, config, image_hw=None):
        """Same as `_collect_residual_metadata`, but also computes each crop's actual residual
        pixel data and attaches an importance score (`_compute_patch_pscore_hint`, same signal
        `_process_image_group_residuals` uses per-patch) -- needed by the 'top_energy' keep-rate
        grouping strategy to rank merge-groups by how much high-frequency detail their residual
        actually carries, before deciding which ones are worth ever transmitting/correcting."""
        levels = sorted(config.transmission_kwargs.get('pyramid_levels', [2, 0]), reverse=True)
        ph, pw = config.patch_size
        mobile_pscore = "residual_energy"  # keep-rate importance ranking always uses energy, regardless of mobile_pscore config

        structure = []
        prev_lvl = levels[0]
        prev_img = gaussians[prev_lvl]
        for lvl in levels[1:]:
            curr_g = gaussians[lvl]
            pred = self._iterative_upsample_native(prev_img, prev_lvl, lvl, gaussians)
            residual = curr_g.astype(np.int16) - pred.astype(np.int16)
            residual = self._project_band_to_target(residual, lvl, config, np.int16, image_hw)

            rh, rw = residual.shape[:2]
            gh, gw = rh // ph, rw // pw
            num_crops = gh * gw
            for i in range(num_crops):
                row, col = divmod(i, gw)
                y, x = row * ph, col * pw
                crop = residual[y : y + ph, x : x + pw]
                pscore = self._compute_patch_pscore_hint(crop, mobile_pscore)
                structure.append({
                    'spatial_idx': i,
                    'res_level': lvl,
                    'grid_hw': (gh, gw),
                    'row': row,
                    'col': col,
                    'pscore': pscore,
                })
            prev_img = curr_g
            prev_lvl = lvl
        return structure

    def _precompute_group_assignments(self, strategy, residual_structure, num_groups, config=None):
        """Pre-calculate group ID for N items based on strategy."""
        if isinstance(residual_structure, int):
            N = residual_structure
            structure = None
        else:
            structure = list(residual_structure)
            N = len(structure)

        if strategy == 'grid':
            s = int(num_groups ** 0.5)
            if s * s != num_groups:
                raise ValueError(f"grid grouping requires a square num_groups, got {num_groups}")
            if structure is not None and all('row' in item and 'col' in item for item in structure):
                pattern = np.arange(1, num_groups + 1).reshape(s, s)
                return np.asarray(
                    [pattern[int(item['row']) % s, int(item['col']) % s] for item in structure],
                    dtype=int,
                )

            # Legacy square fallback for callers that only provide N.
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

        elif strategy == 'block_grid':
            s = int(num_groups ** 0.5)
            if s * s != num_groups:
                raise ValueError(f"block_grid grouping requires a square num_groups, got {num_groups}")
            if structure is not None and all('row' in item and 'col' in item for item in structure):
                grid_hw_by_level = {}
                for item in structure:
                    level_key = int(item.get('res_level', 0))
                    if item.get('grid_hw') is not None:
                        grid_hw_by_level[level_key] = tuple(int(v) for v in item['grid_hw'])

                fallback_grid_h = max(int(item['row']) for item in structure) + 1
                fallback_grid_w = max(int(item['col']) for item in structure) + 1
                group_ids = []
                for item in structure:
                    grid_h, grid_w = grid_hw_by_level.get(
                        int(item.get('res_level', 0)),
                        (fallback_grid_h, fallback_grid_w),
                    )
                    group_row = min(int(item['row']) * s // max(int(grid_h), 1), s - 1)
                    group_col = min(int(item['col']) * s // max(int(grid_w), 1), s - 1)
                    group_ids.append(group_row * s + group_col + 1)
                return np.asarray(group_ids, dtype=int)

            # Legacy square fallback for callers that only provide N.
            side = int(N ** 0.5)
            rows = np.arange(side)[:, None]
            cols = np.arange(side)[None, :]
            grid_2d = (rows * s // max(side, 1)) * s + (cols * s // max(side, 1)) + 1
            group_ids = grid_2d.astype(int).flatten()
            if len(group_ids) < N:
                group_ids = np.resize(group_ids, N)
            elif len(group_ids) > N:
                group_ids = group_ids[:N]
            return group_ids
            
        elif strategy == 'sequential':
            # Contiguous prefix chunks in flattened (raster) sequence order -- for autoregressive
            # (causally-masked) decoders, correcting group k only benefits positions that causally
            # attend to it; a spatially-scattered group (e.g. 'grid's checkerboard tiling) leaves
            # gaps throughout the sequence, so many later positions still depend on uncorrected
            # earlier ones even after their own group arrives. Taking prefix chunks in sequence
            # order instead means every corrected group extends a strictly-growing corrected
            # *prefix*, so intermediate (pre-100%) rounds get maximal benefit from what has arrived.
            if structure is not None and all('spatial_idx' in item for item in structure):
                order = np.asarray([int(item['spatial_idx']) for item in structure], dtype=int)
            else:
                order = np.arange(N, dtype=int)
            return 1 + (order * num_groups) // max(N, 1)

        elif strategy == 'top_energy':
            # Importance-ranked keep-rate thresholding: rank every merge-group's residual by
            # `_compute_patch_pscore_hint` (residual energy) and only ever transmit/correct the top
            # `keep_rate` fraction (transmission_kwargs['keep_rate'], default 1.0 = keep everything).
            # The rest are assigned group_id=0 -- the same id the base/coarse pyramid layer uses, so
            # `correct_forward`'s `group_map[0] == group_id` lookup for group_id=1 (the only residual
            # group callers should configure with num_groups=1) never matches them: they simply never
            # get corrected, remaining approx-only (from the base layer) for the whole request. This
            # is a *static* one-shot selection (not a progressive multi-round schedule), so it should
            # always be paired with num_groups=1 in the caller's config.
            if structure is None or not all('pscore' in item for item in structure):
                raise ValueError(
                    "'top_energy' grouping requires per-item 'pscore' -- pass residual_structure "
                    "built via _collect_residual_metadata_scored(), not _collect_residual_metadata()."
                )
            keep_rate = 1.0
            if config is not None:
                keep_rate = float(config.transmission_kwargs.get('keep_rate', 1.0))
            keep_rate = min(max(keep_rate, 0.0), 1.0)
            scores = np.asarray([float(item['pscore']) for item in structure], dtype=np.float64)
            keep_n = int(round(keep_rate * N))
            group_ids = np.zeros(N, dtype=int)
            if keep_n > 0:
                # argsort descending, ties broken by original (raster) order for determinism
                order = np.argsort(-scores, kind='stable')
                keep_idx = order[:keep_n]
                group_ids[keep_idx] = 1
            return group_ids

        elif strategy == 'top_energy_threshold':
            # Absolute-threshold keep-rate selection: correct every merge-group whose residual
            # importance score (`pscore`) meets or exceeds an ABSOLUTE cutoff
            # (transmission_kwargs['pscore_threshold']), instead of a fixed top-K% fraction like
            # 'top_energy'. The number of corrected groups therefore varies per image with how
            # much residual energy it actually contains -- a texture-heavy image gets a larger
            # correction budget than a flat/smooth one at the same threshold, whereas
            # 'top_energy' always corrects exactly keep_rate*N groups regardless of the pscore
            # distribution's shape. Same group_id=0/1 semantics as 'top_energy' (uncorrected
            # groups get group_id=0, matching the base/coarse layer's id so they're never
            # selected by correct_forward's group_id=1 lookup) -- still a *static* one-shot
            # selection, pair with num_groups=1.
            if structure is None or not all('pscore' in item for item in structure):
                raise ValueError(
                    "'top_energy_threshold' grouping requires per-item 'pscore' -- pass "
                    "residual_structure built via _collect_residual_metadata_scored(), not "
                    "_collect_residual_metadata()."
                )
            threshold = 0.0
            if config is not None:
                threshold = float(config.transmission_kwargs.get('pscore_threshold', 0.0))
            scores = np.asarray([float(item['pscore']) for item in structure], dtype=np.float64)
            group_ids = np.where(scores >= threshold, 1, 0).astype(int)
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

    def _process_image_group_residuals(
        self,
        b_idx,
        gaussians,
        group_assignments,
        target_group,
        config,
        mobile_pscore,
        image_hw=None,
    ):
        """Compress only patches belonging to target_group for one image."""
        comp_lvl = config.transmission_kwargs.get('compression_level', 1)
        levels = sorted(config.transmission_kwargs.get('pyramid_levels', [2, 0]), reverse=True)
        local_patches = []
        
        # Up-sample sequentially and collect group members
        
        prev_lvl = levels[0]
        prev_img = gaussians[prev_lvl]
        
        struct_idx = 0
        for lvl in levels[1:]:
            curr_g = gaussians[lvl]
            pred = self._iterative_upsample_native(prev_img, prev_lvl, lvl, gaussians)
            residual = curr_g.astype(np.int16) - pred.astype(np.int16)
            residual = self._project_band_to_target(residual, lvl, config, np.int16, image_hw)

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
                    pscore_hint = self._compute_patch_pscore_hint(crop, mobile_pscore)
                    local_patches.append(
                        Patch(
                            b_idx,
                            i,
                            compressed,
                            lvl,
                            target_group,
                            pscore_hint=pscore_hint,
                        )
                    )
                struct_idx += 1
            
            prev_img = curr_g
            prev_lvl = lvl
            
        return local_patches


    def _process_image_base_layer(self, b_idx, image, config, image_hw=None):
        levels = sorted(config.transmission_kwargs.get('pyramid_levels', [2, 0]), reverse=True)
        max_lvl = max(levels)
        comp_lvl = config.transmission_kwargs.get('compression_level', 1)

        gaussians = self._build_native_gaussians(image, max_lvl)

        local_patches = []
        base_lvl = levels[0] # Highest level index is the base layer
        base_band = self._project_band_to_target(gaussians[base_lvl], base_lvl, config, np.uint8, image_hw)
        
        # Use vectorized creation
        self._create_patches_with_group_vectorized(
            local_patches, base_band, b_idx, base_lvl, config, np.uint8,
            group_id=0, compression=comp_lvl
        )
        return local_patches, gaussians

    def _process_image_residuals(self, b_idx, gaussians, config, mobile_pscore, image_hw=None):
        levels = sorted(config.transmission_kwargs.get('pyramid_levels', [2, 0]), reverse=True)
        comp_lvl = config.transmission_kwargs.get('compression_level', 1)

        local_candidates = []

        # Start from base layer and upsample
        prev_lvl = levels[0]
        prev_img = gaussians[prev_lvl]

        for lvl in levels[1:]:
            curr_g = gaussians[lvl]

            # Residual Layer: Collect
            pred = self._iterative_upsample_native(prev_img, prev_lvl, lvl, gaussians)
            residual = curr_g.astype(np.int16) - pred.astype(np.int16)
            residual = self._project_band_to_target(residual, lvl, config, np.int16, image_hw)
            
            # Use vectorized collection
            self._collect_residual_candidates_vectorized(
                local_candidates, residual, b_idx, lvl, config, 
                dtype=np.int16, compression=comp_lvl, mobile_pscore=mobile_pscore
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

    def _collect_residual_candidates_vectorized(
        self,
        candidate_list,
        image,
        b_idx,
        lvl,
        config,
        dtype,
        compression,
        mobile_pscore,
    ):
        ph, pw = config.patch_size
        H, W, C = image.shape

        # Strict divisibility check
        if H % ph != 0 or W % pw != 0:
             raise ValueError(f"[Residual] Image shape {(H, W)} not divisible by patch {(ph, pw)}")

        gh, gw = H // ph, W // pw
        
        crops = image.reshape(gh, ph, gw, pw, C).transpose(0, 2, 1, 3, 4).reshape(-1, ph, pw, C)
        num_crops = crops.shape[0]

        for i in range(num_crops):
            crop = crops[i].astype(dtype)
            data = crop.tobytes()
            compressed = zlib.compress(data, level=compression)
            pscore_hint = self._compute_patch_pscore_hint(crop, mobile_pscore)
            candidate_list.append({
                'image_idx': b_idx, 'spatial_idx': i, 'res_level': lvl,
                'data': compressed, 'size': len(compressed), 'pscore_hint': pscore_hint,
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
            group_id=group_id,
            pscore_hint=float(c.get('pscore_hint', 0.0)),
        ))
