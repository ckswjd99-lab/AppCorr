from concurrent.futures import ThreadPoolExecutor
from typing import Generator, List
import zlib

import numpy as np

from offload.common.protocol import (
    ExperimentConfig,
    Patch,
    normalize_appcorr_kwargs,
)

from .progressive import ProgressiveLPyramidPolicy


class L2L1L0ProgressiveLPyramidPolicy(ProgressiveLPyramidPolicy):
    """Transmit L2, the complete L1 band, then spatial chunks of the L0 band."""

    _SUPPORTED_GROUPING_STRATEGIES = {"grid", "block_grid"}

    @staticmethod
    def _validate_config(config: ExperimentConfig) -> tuple[int, str]:
        levels = list(config.transmission_kwargs.get("pyramid_levels", [2, 1, 0]))
        if levels != [2, 1, 0]:
            raise ValueError(
                "L2L1L0ProgressiveLaplacian requires "
                f"transmission_kwargs.pyramid_levels=[2, 1, 0], got {levels}"
            )
        if bool(config.transmission_kwargs.get("preserve_input_shape", False)):
            raise ValueError(
                "L2L1L0ProgressiveLaplacian currently supports fixed-size inputs only"
            )

        l0_num_groups = int(config.transmission_kwargs.get("l0_num_groups", 4))
        if l0_num_groups <= 0:
            raise ValueError(
                f"transmission_kwargs.l0_num_groups must be positive, got {l0_num_groups}"
            )

        grouping_strategy = str(
            config.transmission_kwargs.get("grouping_strategy", "grid")
        )
        if grouping_strategy not in (
            L2L1L0ProgressiveLPyramidPolicy._SUPPORTED_GROUPING_STRATEGIES
        ):
            raise ValueError(
                "L2L1L0ProgressiveLaplacian supports grouping_strategy values "
                f"{sorted(L2L1L0ProgressiveLPyramidPolicy._SUPPORTED_GROUPING_STRATEGIES)}, "
                f"got {grouping_strategy!r}"
            )

        appcorr_options = normalize_appcorr_kwargs(
            config.appcorr_kwargs,
            config.transmission_kwargs,
        )
        if appcorr_options["method"] != "partial_token":
            raise ValueError(
                "L2L1L0ProgressiveLaplacian currently requires "
                "appcorr_kwargs.method='partial_token'"
            )
        return l0_num_groups, grouping_strategy

    @staticmethod
    def _mark_group(
        patches: List[Patch],
        *,
        num_correction_groups: int,
    ) -> None:
        group_total = len(patches)
        for patch in patches:
            patch.batch_group_total = group_total
            patch.num_correction_groups = num_correction_groups

    def _compute_residual_band(
        self,
        gaussians,
        prev_level: int,
        level: int,
        config: ExperimentConfig,
        image_hw: tuple[int, int] | None = None,
    ) -> np.ndarray:
        # Closed loop (see LaplacianPyramidPolicy._closed_loop_residual): the decoder predicts
        # from the previous level as *transmitted*, not from the native gaussian, so the residual
        # must be built against that same predictor. The L1 band is sent as one complete group,
        # so at L0 time the decoder's state is exactly project(gaussians[1]) and the chain stays
        # lossless level to level. The open-loop form this replaces left ~2% relative L2 even
        # with every residual delivered.
        return self._closed_loop_residual(gaussians, prev_level, level, config, image_hw)

    def _encode_complete_residual_band(
        self,
        image_idx: int,
        residual: np.ndarray,
        *,
        level: int,
        group_id: int,
        config: ExperimentConfig,
        mobile_pscore: str,
    ) -> List[Patch]:
        compression_level = int(
            config.transmission_kwargs.get("compression_level", 1)
        )
        candidates = []
        self._collect_residual_candidates_vectorized(
            candidates,
            residual,
            image_idx,
            level,
            config,
            dtype=np.int16,
            compression=compression_level,
            mobile_pscore=mobile_pscore,
        )
        return [
            Patch(
                image_idx=candidate["image_idx"],
                spatial_idx=candidate["spatial_idx"],
                data=candidate["data"],
                res_level=candidate["res_level"],
                group_id=group_id,
                pscore_hint=float(candidate.get("pscore_hint", 0.0)),
            )
            for candidate in candidates
        ]

    def _compute_and_encode_complete_residual_band(
        self,
        image_idx: int,
        gaussians,
        *,
        previous_level: int,
        level: int,
        group_id: int,
        config: ExperimentConfig,
        mobile_pscore: str,
    ) -> List[Patch]:
        residual = self._compute_residual_band(
            gaussians,
            previous_level,
            level,
            config,
        )
        return self._encode_complete_residual_band(
            image_idx,
            residual,
            level=level,
            group_id=group_id,
            config=config,
            mobile_pscore=mobile_pscore,
        )

    def _encode_residual_group(
        self,
        image_idx: int,
        residual: np.ndarray,
        assignments: np.ndarray,
        *,
        level: int,
        local_group_id: int,
        output_group_id: int,
        config: ExperimentConfig,
        mobile_pscore: str,
    ) -> List[Patch]:
        patch_h, patch_w = config.patch_size
        height, width, channels = residual.shape
        grid_h, grid_w = height // patch_h, width // patch_w
        crops = (
            residual.reshape(
                grid_h,
                patch_h,
                grid_w,
                patch_w,
                channels,
            )
            .transpose(0, 2, 1, 3, 4)
            .reshape(-1, patch_h, patch_w, channels)
        )
        if assignments.shape[0] != crops.shape[0]:
            raise RuntimeError(
                f"Assignment/crop mismatch: {assignments.shape[0]} vs {crops.shape[0]}"
            )

        compression_level = int(
            config.transmission_kwargs.get("compression_level", 1)
        )
        patches = []
        for spatial_idx in np.flatnonzero(assignments == local_group_id):
            crop = crops[int(spatial_idx)].astype(np.int16, copy=False)
            compressed = zlib.compress(
                crop.tobytes(),
                level=compression_level,
            )
            patches.append(
                Patch(
                    image_idx=image_idx,
                    spatial_idx=int(spatial_idx),
                    data=compressed,
                    res_level=level,
                    group_id=output_group_id,
                    pscore_hint=self._compute_patch_pscore_hint(
                        crop,
                        mobile_pscore,
                    ),
                )
            )
        return patches

    def encode(
        self,
        images: np.ndarray,
        config: ExperimentConfig,
    ) -> Generator[List[Patch], None, None]:
        l0_num_groups, grouping_strategy = self._validate_config(config)
        num_correction_groups = 1 + l0_num_groups
        image_list = self._as_image_list(images)
        batch_size = len(image_list)
        mobile_pscore = self._resolve_mobile_pscore(config)

        base_patches = []
        gaussians_batch = [None] * batch_size
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self._process_image_base_layer,
                    image_idx,
                    image,
                    config,
                    None,
                )
                for image_idx, image in enumerate(image_list)
            ]
            for image_idx, future in enumerate(futures):
                local_patches, gaussians = future.result()
                base_patches.extend(local_patches)
                gaussians_batch[image_idx] = gaussians
        self._mark_group(
            base_patches,
            num_correction_groups=num_correction_groups,
        )
        yield base_patches

        l1_patches = []
        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self._compute_and_encode_complete_residual_band,
                    image_idx,
                    gaussians,
                    previous_level=2,
                    level=1,
                    group_id=1,
                    config=config,
                    mobile_pscore=mobile_pscore,
                )
                for image_idx, gaussians in enumerate(gaussians_batch)
            ]
            for future in futures:
                l1_patches.extend(future.result())
        self._mark_group(
            l1_patches,
            num_correction_groups=num_correction_groups,
        )
        yield l1_patches

        with ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self._compute_residual_band,
                    gaussians,
                    1,
                    0,
                    config,
                )
                for gaussians in gaussians_batch
            ]
            l0_residuals = [future.result() for future in futures]
        per_image_assignments = []
        for residual in l0_residuals:
            patch_h, patch_w = config.patch_size
            grid_h = residual.shape[0] // patch_h
            grid_w = residual.shape[1] // patch_w
            structure = [
                {
                    "spatial_idx": row * grid_w + col,
                    "res_level": 0,
                    "grid_hw": (grid_h, grid_w),
                    "row": row,
                    "col": col,
                }
                for row in range(grid_h)
                for col in range(grid_w)
            ]
            per_image_assignments.append(
                self._precompute_group_assignments(
                    grouping_strategy,
                    structure,
                    l0_num_groups,
                )
            )

        for local_group_id in range(1, l0_num_groups + 1):
            output_group_id = local_group_id + 1
            group_patches = []
            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        self._encode_residual_group,
                        image_idx,
                        l0_residuals[image_idx],
                        per_image_assignments[image_idx],
                        level=0,
                        local_group_id=local_group_id,
                        output_group_id=output_group_id,
                        config=config,
                        mobile_pscore=mobile_pscore,
                    )
                    for image_idx in range(batch_size)
                ]
                for future in futures:
                    group_patches.extend(future.result())

            if not group_patches:
                raise RuntimeError(
                    f"L0 correction group {local_group_id} is empty"
                )
            self._mark_group(
                group_patches,
                num_correction_groups=num_correction_groups,
            )
            yield group_patches
