import numpy as np
from typing import Generator, List

from offload.common.protocol import (
    ExperimentConfig,
    Patch,
    normalize_appcorr_kwargs,
)

from .progressive import ProgressiveLPyramidPolicy


class ADE20KWindowProgressiveLaplacianPolicy(ProgressiveLPyramidPolicy):
    """
    ADE20K m2f crop-cover progressive codec.

    Base (group 0) is the unchanged 1/4-res (level-2) full image. Groups 1..N carry full-res
    (level-0) residuals grouped by the m2f sliding-window crops (896 crop / 596 stride): each
    residual patch is assigned to the FIRST (row-major) crop that covers its center, so overlap
    regions go to the earlier crop and group i = crop_i minus crops 1..i-1. Receiving group i
    therefore completes correction of crop i (its overlap with earlier crops already arrived), plus
    the corresponding parts of later crops. N = the per-image sliding-crop count (varies with aspect)
    and is carried on every patch via `num_correction_groups` so the scheduler can chunk layers by N.

    Assumes batch_size == 1 (as ADE20K m2f always runs).
    """

    @staticmethod
    def _patch_hw(config: ExperimentConfig) -> tuple[int, int]:
        ps = config.patch_size
        if isinstance(ps, int):
            return int(ps), int(ps)
        return int(ps[0]), int(ps[1])

    def _crop_params(self, config: ExperimentConfig) -> tuple[int, int]:
        prof = config.get_input_profile_config()
        return int(prof.get("server_crop_size", 896)), int(prof.get("server_stride", 596))

    @staticmethod
    def _compute_crops(h_img: int, w_img: int, crop: int, stride: int) -> List[tuple[int, int, int, int]]:
        # Must mirror dinov3_segmentor_m2f's sliding-window crop layout exactly.
        h_crop = w_crop = crop
        if h_crop > h_img and w_crop > w_img:
            h_crop = w_crop = min(h_img, w_img)
        h_grids = max(h_img - h_crop + stride - 1, 0) // stride + 1
        w_grids = max(w_img - w_crop + stride - 1, 0) // stride + 1
        crops = []
        for hi in range(h_grids):
            for wi in range(w_grids):
                y1 = hi * stride
                x1 = wi * stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crops.append((y1, y2, x1, x2))
        return crops

    def _crops_for_image(self, config: ExperimentConfig, image_hw):
        ph, pw = self._patch_hw(config)
        h0, w0 = self._target_hw_for_level(config, 0, image_hw)  # patch-aligned model-input pixels
        gh, gw = h0 // ph, w0 // pw
        crop, stride = self._crop_params(config)
        return self._compute_crops(gh * ph, gw * pw, crop, stride)

    def _resolve_num_groups(self, config, image_list) -> int:
        """crop_cover's group count is a property of the image, not a setting.

        Each token goes to the first sliding crop covering its centre, so the number of groups is
        exactly the number of crops -- 1 to 5 in practice at 896/596, typically 2-3, and dependent on
        each image's aspect ratio. `encode` emits `range(1, n + 1)`, so `n` has to cover the largest
        image in the batch or that image's last crop is never transmitted.

        This used to be a hand-set `num_groups: 16` in every crop_cover config -- headroom large
        enough to be safe, meaningless as a number, and easily misread as "16 correction rounds"
        (the scheduler chunks layers by `num_correction_groups`, which is the crop count, so the two
        were never the same thing). Computing it removes the magic value, and removes the trap that
        a crop_cover config written *without* `num_groups` would silently fall back to the base
        class default of 4.
        """
        if str(config.transmission_kwargs.get('grouping_strategy', '')) != "crop_cover":
            return super()._resolve_num_groups(config, image_list)
        crop, stride = self._crop_params(config)
        n = 1
        for img in image_list:
            hw = img.shape[:2]
            h0, w0 = self._target_hw_for_level(config, 0, hw)
            ph, pw = self._patch_hw(config)
            gh, gw = h0 // ph, w0 // pw
            n = max(n, len(self._compute_crops(gh * ph, gw * pw, crop, stride)))
        return n

    def _precompute_group_assignments(self, strategy, residual_structure, num_groups, config=None):
        # Signature must track ProgressiveLPyramidPolicy's (config was added there for the
        # crop_cover group-count derivation); the parent's encode calls this positionally.
        if strategy != "crop_cover":
            return super()._precompute_group_assignments(
                strategy, residual_structure, num_groups, config
            )

        structure = list(residual_structure)
        if not structure:
            return np.zeros(0, dtype=int)

        ph, pw = self._patch_hw(self._active_config)
        crops = self._crops_for_image(self._active_config, self._active_image_hw)
        group_ids = np.empty(len(structure), dtype=int)
        for i, item in enumerate(structure):
            r, c = int(item["row"]), int(item["col"])
            cy = r * ph + ph // 2
            cx = c * pw + pw // 2
            g = 0
            for idx, (y1, y2, x1, x2) in enumerate(crops):
                if y1 <= cy < y2 and x1 <= cx < x2:
                    g = idx + 1
                    break
            group_ids[i] = g if g > 0 else len(crops)  # fallback: last crop
        return group_ids

    def encode(self, images: np.ndarray, config: ExperimentConfig) -> Generator[List[Patch], None, None]:
        self._active_config = config
        image_list = self._as_image_list(images)
        preserve = self._is_preserve_input_shape(config)
        # crop-cover assumes batch=1; the crop layout is taken from image 0.
        self._active_image_hw = tuple(int(v) for v in image_list[0].shape[:2]) if preserve else None
        n = len(self._crops_for_image(config, self._active_image_hw))

        for group in super().encode(images, config):
            for p in group:
                p.num_correction_groups = n
            yield group


class ADE20KL2L1ProgressiveLaplacianPolicy(ProgressiveLPyramidPolicy):
    """Emit an L2 base and an exact target-space L1 residual.

    The generic progressive codec resizes a native-resolution Laplacian band.
    Resize and pyramid upsampling do not commute, so its completed L1 image can
    differ slightly from a directly transmitted L1 image.  This two-level
    policy instead forms the residual after projecting both levels:

        residual_l1 = projected_l1 - upsample(projected_l2)

    Consequently, decoding all patches is pixel-identical to the L1-only
    Laplacian input.  This makes L1-only versus L2->L1 AppCorr a controlled
    comparison whose endpoint differs only by the correction computation.
    """

    @staticmethod
    def _validate_config(config: ExperimentConfig) -> None:
        levels = list(config.transmission_kwargs.get("pyramid_levels", []))
        if levels != [2, 1]:
            raise ValueError(
                "ADE20KL2L1ProgressiveLaplacian requires "
                f"pyramid_levels=[2, 1], got {levels}"
            )
        if not bool(
            config.transmission_kwargs.get("preserve_input_shape", False)
        ):
            raise ValueError(
                "ADE20KL2L1ProgressiveLaplacian requires "
                "preserve_input_shape=true"
            )
        appcorr_options = normalize_appcorr_kwargs(
            config.appcorr_kwargs,
            config.transmission_kwargs,
        )
        if appcorr_options["method"] != "partial_token":
            raise ValueError(
                "ADE20KL2L1ProgressiveLaplacian requires "
                "appcorr_kwargs.method='partial_token'"
            )

    @staticmethod
    def _mark_group(
        patches: List[Patch],
        *,
        target_shape: tuple[int, int],
    ) -> None:
        group_total = len(patches)
        for patch in patches:
            patch.batch_group_total = group_total
            patch.num_correction_groups = 1
            patch.target_shape = target_shape

    def encode(
        self,
        images: np.ndarray,
        config: ExperimentConfig,
    ) -> Generator[List[Patch], None, None]:
        self._validate_config(config)
        image_list = self._as_image_list(images)
        if len(image_list) != 1 or int(config.batch_size) != 1:
            raise ValueError(
                "ADE20KL2L1ProgressiveLaplacian supports batch_size=1"
            )

        image = image_list[0]
        image_hw = tuple(int(value) for value in image.shape[:2])
        base_patches, gaussians = self._process_image_base_layer(
            0,
            image,
            config,
            image_hw,
        )
        self._mark_group(base_patches, target_shape=image_hw)
        yield base_patches

        l2_target = self._project_band_to_target(
            gaussians[2],
            2,
            config,
            np.uint8,
            image_hw,
        )
        l1_target = self._project_band_to_target(
            gaussians[1],
            1,
            config,
            np.uint8,
            image_hw,
        )
        l1_prediction = self._iterative_upsample_to_hw(
            l2_target,
            2,
            1,
            l1_target.shape[:2],
        )
        l1_residual = (
            l1_target.astype(np.int16)
            - l1_prediction.astype(np.int16)
        )

        candidates: list[dict] = []
        self._collect_residual_candidates_vectorized(
            candidates,
            l1_residual,
            0,
            1,
            config,
            dtype=np.int16,
            compression=int(
                config.transmission_kwargs.get("compression_level", 1)
            ),
            mobile_pscore=self._resolve_mobile_pscore(config),
        )
        l1_patches = [
            Patch(
                image_idx=int(candidate["image_idx"]),
                spatial_idx=int(candidate["spatial_idx"]),
                data=candidate["data"],
                res_level=int(candidate["res_level"]),
                group_id=1,
                pscore_hint=float(candidate["pscore_hint"]),
            )
            for candidate in candidates
        ]
        self._mark_group(l1_patches, target_shape=image_hw)
        yield l1_patches


class ADE20KWindowL2L1L0ProgressiveLaplacianPolicy(
    ADE20KWindowProgressiveLaplacianPolicy
):
    """Emit an L2 base, one complete L1 residual, then crop-cover L0 groups.

    ADE20K uses a variable-size, patch-aligned image and a dynamic number of
    sliding-window crops.  The complete L1 band is correction group 1; the
    existing crop-cover L0 groups are shifted to 2..N+1.
    """

    _VALID_L1_PSCORE_MODES = frozenset({
        "residual_energy",
        "positive_residual_difference",
    })
    _VALID_L0_PSCORE_MODES = frozenset({
        "incremental_residual_energy",
        "conditional_cumulative_residual_energy",
    })

    @staticmethod
    def _validate_l2l1l0_config(config: ExperimentConfig) -> None:
        levels = list(config.transmission_kwargs.get("pyramid_levels", []))
        if levels != [2, 1, 0]:
            raise ValueError(
                "ADE20KWindowL2L1L0ProgressiveLaplacian requires "
                f"pyramid_levels=[2, 1, 0], got {levels}"
            )
        if not bool(config.transmission_kwargs.get("preserve_input_shape", False)):
            raise ValueError(
                "ADE20KWindowL2L1L0ProgressiveLaplacian requires "
                "preserve_input_shape=true"
            )
        grouping_strategy = str(
            config.transmission_kwargs.get("grouping_strategy", "crop_cover")
        )
        if grouping_strategy != "crop_cover":
            raise ValueError(
                "ADE20KWindowL2L1L0ProgressiveLaplacian requires "
                f"grouping_strategy='crop_cover', got {grouping_strategy!r}"
            )
        appcorr_options = normalize_appcorr_kwargs(
            config.appcorr_kwargs,
            config.transmission_kwargs,
        )
        if appcorr_options["method"] != "partial_token":
            raise ValueError(
                "ADE20KWindowL2L1L0ProgressiveLaplacian requires "
                "appcorr_kwargs.method='partial_token'"
            )
        l1_pscore_mode = appcorr_options["l1_pscore_mode"]
        if l1_pscore_mode not in (
            ADE20KWindowL2L1L0ProgressiveLaplacianPolicy
            ._VALID_L1_PSCORE_MODES
        ):
            raise ValueError(
                f"Unsupported L1 pscore mode {l1_pscore_mode!r}; expected one "
                f"of {sorted(ADE20KWindowL2L1L0ProgressiveLaplacianPolicy._VALID_L1_PSCORE_MODES)}"
            )
        l0_pscore_mode = appcorr_options["l0_pscore_mode"]
        if l0_pscore_mode not in (
            ADE20KWindowL2L1L0ProgressiveLaplacianPolicy
            ._VALID_L0_PSCORE_MODES
        ):
            raise ValueError(
                f"Unsupported L0 pscore mode {l0_pscore_mode!r}; expected one "
                f"of {sorted(ADE20KWindowL2L1L0ProgressiveLaplacianPolicy._VALID_L0_PSCORE_MODES)}"
            )
        uses_cross_level_score = (
            l1_pscore_mode == "positive_residual_difference"
            or l0_pscore_mode == "conditional_cumulative_residual_energy"
        )
        if uses_cross_level_score and appcorr_options["mobile_pscore"] != "residual_energy":
            raise ValueError(
                "Cross-level L1/L0 pscore modes require "
                "appcorr_kwargs.mobile_pscore='residual_energy'"
            )
        support_mode = appcorr_options["l1_l0_support_mode"]
        if (
            support_mode in {
                "conditional_threshold",
                "conditional_reentry",
            }
            and l0_pscore_mode
            != "conditional_cumulative_residual_energy"
        ):
            raise ValueError(
                f"{support_mode} requires "
                "l0_pscore_mode='conditional_cumulative_residual_energy'"
            )
        if (
            appcorr_options["l1_remaining_energy_ratio_max"] is not None
            and l1_pscore_mode != "positive_residual_difference"
        ):
            raise ValueError(
                "l1_remaining_energy_ratio_max requires "
                "l1_pscore_mode='positive_residual_difference'"
            )

    @staticmethod
    def _mark_group(
        patches: List[Patch],
        *,
        num_correction_groups: int,
        target_shape: tuple[int, int],
    ) -> None:
        group_total = len(patches)
        for patch in patches:
            patch.batch_group_total = group_total
            patch.num_correction_groups = num_correction_groups
            patch.target_shape = target_shape

    def _projected_residual_band(
        self,
        gaussians,
        *,
        previous_level: int,
        level: int,
        config: ExperimentConfig,
        image_hw: tuple[int, int],
    ) -> np.ndarray:
        # Closed loop -- see LaplacianPyramidPolicy._closed_loop_residual. The L1 band is one
        # complete group, so the decoder's state before L0 is exactly project(gaussians[1]).
        return self._closed_loop_residual(
            gaussians,
            previous_level,
            level,
            config,
            image_hw,
        )

    def _encode_residual_candidates(
        self,
        residual: np.ndarray,
        *,
        image_idx: int,
        level: int,
        config: ExperimentConfig,
        mobile_pscore: str,
    ) -> list[dict]:
        candidates: list[dict] = []
        self._collect_residual_candidates_vectorized(
            candidates,
            residual,
            image_idx,
            level,
            config,
            dtype=np.int16,
            compression=int(
                config.transmission_kwargs.get("compression_level", 1)
            ),
            mobile_pscore=mobile_pscore,
        )
        return candidates

    @staticmethod
    def _patch_energy_map(
        residual: np.ndarray,
        patch_hw: tuple[int, int],
    ) -> np.ndarray:
        """Return sum-of-squares energy for each spatial transmission patch."""
        ph, pw = patch_hw
        height, width, channels = residual.shape
        if height % ph != 0 or width % pw != 0:
            raise ValueError(
                f"Residual shape {(height, width)} is not divisible by "
                f"patch shape {(ph, pw)}"
            )
        grid_h, grid_w = height // ph, width // pw
        residual_f32 = residual.astype(np.float32, copy=False)
        patches = residual_f32.reshape(
            grid_h,
            ph,
            grid_w,
            pw,
            channels,
        ).transpose(0, 2, 1, 3, 4)
        return np.square(patches, dtype=np.float32).sum(
            axis=(2, 3, 4),
            dtype=np.float32,
        )

    @staticmethod
    def _aggregate_energy_to_grid(
        fine_energy: np.ndarray,
        target_grid_hw: tuple[int, int],
    ) -> np.ndarray:
        """Sum fine-patch energy over aligned coarse spatial cells."""
        fine_h, fine_w = fine_energy.shape
        target_h, target_w = target_grid_hw
        if target_h <= 0 or target_w <= 0:
            raise ValueError(f"Invalid target energy grid {target_grid_hw}")

        aggregated = np.zeros((target_h, target_w), dtype=np.float32)
        for row in range(target_h):
            row_start = row * fine_h // target_h
            row_end = max((row + 1) * fine_h // target_h, row_start + 1)
            for col in range(target_w):
                col_start = col * fine_w // target_w
                col_end = max(
                    (col + 1) * fine_w // target_w,
                    col_start + 1,
                )
                aggregated[row, col] = fine_energy[
                    row_start:min(row_end, fine_h),
                    col_start:min(col_end, fine_w),
                ].sum(dtype=np.float32)
        return aggregated

    @classmethod
    def _positive_residual_difference_score(
        cls,
        l1_effect: np.ndarray,
        l0_after_l1: np.ndarray,
        *,
        patch_hw: tuple[int, int],
        l1_grid_hw: tuple[int, int],
        remaining_ratio_max: float | None = None,
    ) -> np.ndarray:
        l1_effect_energy = cls._aggregate_energy_to_grid(
            cls._patch_energy_map(l1_effect, patch_hw),
            l1_grid_hw,
        )
        remaining_energy = cls._aggregate_energy_to_grid(
            cls._patch_energy_map(l0_after_l1, patch_hw),
            l1_grid_hw,
        )
        score = np.maximum(
            l1_effect_energy - remaining_energy,
            0.0,
        )
        if remaining_ratio_max is not None:
            denominator = np.maximum(
                l1_effect_energy,
                np.finfo(np.float32).eps,
            )
            safe_l1 = (
                remaining_energy / denominator
                <= float(remaining_ratio_max)
            )
            score = np.where(safe_l1, score, 0.0)
        return score

    @staticmethod
    def _set_candidate_score_map(
        candidates: list[dict],
        score_map: np.ndarray,
        *,
        field_name: str = "pscore_hint",
    ) -> None:
        flat_scores = np.asarray(score_map, dtype=np.float32).reshape(-1)
        if len(candidates) != flat_scores.shape[0]:
            raise RuntimeError(
                "Candidate/pscore shape mismatch: "
                f"{len(candidates)} vs {flat_scores.shape[0]}"
            )
        for candidate, score in zip(candidates, flat_scores):
            candidate[field_name] = float(score)

    def _reconstruction_residuals_at_l0(
        self,
        gaussians,
        l1_residual: np.ndarray,
        l0_residual: np.ndarray,
        *,
        config: ExperimentConfig,
        image_hw: tuple[int, int],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build exact decoded L2, L1, and L0 states before measuring deltas.

        The decoder clips after adding each band. Measuring state differences
        here therefore matches the actual images presented to the server,
        including resize and clipping effects.
        """
        l2_hw = self._target_hw_for_level(config, 2, image_hw)
        l1_hw = self._target_hw_for_level(config, 1, image_hw)
        l0_hw = self._target_hw_for_level(config, 0, image_hw)
        l2_state = self._resize_to_hw(
            gaussians[2],
            l2_hw,
            np.uint8,
        )

        l1_prediction = self._iterative_upsample_to_hw(
            l2_state,
            2,
            1,
            l1_hw,
        )
        l1_state = np.clip(
            l1_prediction.astype(np.int16) + l1_residual,
            0,
            255,
        ).astype(np.uint8)

        l2_at_l0 = self._iterative_upsample_to_hw(
            l2_state,
            2,
            0,
            l0_hw,
        )
        l1_at_l0 = self._iterative_upsample_to_hw(
            l1_state,
            1,
            0,
            l0_hw,
        )
        l0_state = np.clip(
            l1_at_l0.astype(np.int16) + l0_residual,
            0,
            255,
        ).astype(np.uint8)

        l1_effect = (
            l1_at_l0.astype(np.int16) - l2_at_l0.astype(np.int16)
        )
        l0_after_l1 = (
            l0_state.astype(np.int16) - l1_at_l0.astype(np.int16)
        )
        l0_from_l2 = (
            l0_state.astype(np.int16) - l2_at_l0.astype(np.int16)
        )
        return l1_effect, l0_after_l1, l0_from_l2

    @staticmethod
    def _patches_from_candidates(
        candidates: list[dict],
        *,
        group_id: int,
        assignments: np.ndarray | None = None,
        local_group_id: int | None = None,
    ) -> List[Patch]:
        patches = []
        for candidate_idx, candidate in enumerate(candidates):
            if (
                assignments is not None
                and int(assignments[candidate_idx]) != int(local_group_id)
            ):
                continue
            patches.append(
                Patch(
                    image_idx=int(candidate["image_idx"]),
                    spatial_idx=int(candidate["spatial_idx"]),
                    data=candidate["data"],
                    res_level=int(candidate["res_level"]),
                    group_id=group_id,
                    pscore_hint=float(candidate.get("pscore_hint", 0.0)),
                    pscore_hint_if_l1_corrected=float(
                        candidate.get(
                            "pscore_hint_if_l1_corrected",
                            0.0,
                        )
                    ),
                )
            )
        return patches

    def encode(
        self,
        images: np.ndarray,
        config: ExperimentConfig,
    ) -> Generator[List[Patch], None, None]:
        self._validate_l2l1l0_config(config)
        image_list = self._as_image_list(images)
        if len(image_list) != 1 or int(config.batch_size) != 1:
            raise ValueError(
                "ADE20KWindowL2L1L0ProgressiveLaplacian supports batch_size=1"
            )

        image = image_list[0]
        image_hw = tuple(int(value) for value in image.shape[:2])
        self._active_config = config
        self._active_image_hw = image_hw
        l0_num_groups = len(self._crops_for_image(config, image_hw))
        num_correction_groups = 1 + l0_num_groups
        mobile_pscore = self._resolve_mobile_pscore(config)

        base_patches, gaussians = self._process_image_base_layer(
            0,
            image,
            config,
            image_hw,
        )
        self._mark_group(
            base_patches,
            num_correction_groups=num_correction_groups,
            target_shape=image_hw,
        )
        yield base_patches

        l1_residual = self._projected_residual_band(
            gaussians,
            previous_level=2,
            level=1,
            config=config,
            image_hw=image_hw,
        )
        l0_residual = self._projected_residual_band(
            gaussians,
            previous_level=1,
            level=0,
            config=config,
            image_hw=image_hw,
        )
        (
            l1_effect,
            l0_after_l1,
            l0_from_l2,
        ) = self._reconstruction_residuals_at_l0(
            gaussians,
            l1_residual,
            l0_residual,
            config=config,
            image_hw=image_hw,
        )
        appcorr_options = normalize_appcorr_kwargs(
            config.appcorr_kwargs,
            config.transmission_kwargs,
        )
        l1_candidates = self._encode_residual_candidates(
            l1_residual,
            image_idx=0,
            level=1,
            config=config,
            mobile_pscore=mobile_pscore,
        )
        patch_hw = self._patch_hw(config)
        if appcorr_options["l1_pscore_mode"] == "positive_residual_difference":
            l1_grid_hw = (
                l1_residual.shape[0] // patch_hw[0],
                l1_residual.shape[1] // patch_hw[1],
            )
            l1_score = self._positive_residual_difference_score(
                l1_effect,
                l0_after_l1,
                patch_hw=patch_hw,
                l1_grid_hw=l1_grid_hw,
                remaining_ratio_max=appcorr_options[
                    "l1_remaining_energy_ratio_max"
                ],
            )
            self._set_candidate_score_map(l1_candidates, l1_score)
        l1_patches = self._patches_from_candidates(
            l1_candidates,
            group_id=1,
        )
        self._mark_group(
            l1_patches,
            num_correction_groups=num_correction_groups,
            target_shape=image_hw,
        )
        yield l1_patches

        l0_candidates = self._encode_residual_candidates(
            l0_residual,
            image_idx=0,
            level=0,
            config=config,
            mobile_pscore=mobile_pscore,
        )
        if (
            appcorr_options["l0_pscore_mode"]
            == "conditional_cumulative_residual_energy"
        ):
            # The server chooses between these exact decoded-state energies
            # after observing the actual L1 correction mask:
            #   not corrected at L1 -> energy(I0 - I2)
            #   corrected at L1     -> energy(I0 - I1)
            # The two raw maps are normalized with one common denominator
            # after the conditional choice, rather than independently.
            self._set_candidate_score_map(
                l0_candidates,
                self._patch_energy_map(l0_from_l2, patch_hw),
            )
            self._set_candidate_score_map(
                l0_candidates,
                self._patch_energy_map(l0_after_l1, patch_hw),
                field_name="pscore_hint_if_l1_corrected",
            )
        residual_structure = self._collect_residual_metadata(
            gaussians,
            config,
            image_hw,
        )
        l0_structure = [
            item
            for item in residual_structure
            if int(item["res_level"]) == 0
        ]
        assignments = self._precompute_group_assignments(
            "crop_cover",
            l0_structure,
            l0_num_groups,
        )
        if len(assignments) != len(l0_candidates):
            raise RuntimeError(
                "ADE20K L0 assignment/candidate mismatch: "
                f"{len(assignments)} vs {len(l0_candidates)}"
            )

        for local_group_id in range(1, l0_num_groups + 1):
            group_patches = self._patches_from_candidates(
                l0_candidates,
                group_id=local_group_id + 1,
                assignments=assignments,
                local_group_id=local_group_id,
            )
            if not group_patches:
                raise RuntimeError(
                    f"ADE20K L0 crop-cover group {local_group_id} is empty"
                )
            self._mark_group(
                group_patches,
                num_correction_groups=num_correction_groups,
                target_shape=image_hw,
            )
            yield group_patches
