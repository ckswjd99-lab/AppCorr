from typing import Any, Generator, List

import cv2
import numpy as np

from offload.common.protocol import ExperimentConfig, Patch

from .laplacian import LaplacianPyramidPolicy
from .progressive import ProgressiveLPyramidPolicy
from .raw import RawTransmissionPolicy


class _NYUAppCorrFixedGridMixin:
    """Use native image pyramids but decode onto the fixed depther ViT grid."""

    @staticmethod
    def _model_hw(config: ExperimentConfig) -> tuple[int, int]:
        height, width = config.image_shape[:2]
        return int(height), int(width)

    @staticmethod
    def _is_preserve_input_shape(config: ExperimentConfig) -> bool:
        return False

    @staticmethod
    def _target_hw_for_level(
        config: ExperimentConfig,
        lvl: int,
        image_hw: tuple[int, int] | None = None,
    ) -> tuple[int, int]:
        height, width = _NYUAppCorrFixedGridMixin._model_hw(config)
        scale = 2 ** lvl
        if height % scale != 0 or width % scale != 0:
            raise RuntimeError(f"Target image shape {(height, width)} is not divisible by pyramid scale {scale}")
        return height // scale, width // scale

    def _resize_decoded_to_model_grid(self, decoded, config: ExperimentConfig):
        target_h, target_w = self._model_hw(config)
        if isinstance(decoded, list):
            resized = []
            for item in decoded:
                if isinstance(item, dict):
                    image = item.get("image")
                    if image is None:
                        resized.append(item)
                        continue
                    next_item = dict(item)
                    next_item["image"] = self._resize_image_to_model_grid(image, target_h, target_w)
                    resized.append(next_item)
                else:
                    resized.append(self._resize_image_to_model_grid(item, target_h, target_w))
            return resized

        if not isinstance(decoded, np.ndarray) or decoded.ndim != 4:
            return decoded
        if decoded.shape[1:3] == (target_h, target_w):
            return decoded

        out = np.empty((decoded.shape[0], target_h, target_w, decoded.shape[3]), dtype=decoded.dtype)
        for batch_idx in range(decoded.shape[0]):
            out[batch_idx] = self._resize_image_to_model_grid(decoded[batch_idx], target_h, target_w)
        return out

    def _process_image_decode(self, b_idx, patches, config):
        b_idx, image = super()._process_image_decode(b_idx, patches, config)
        target_h, target_w = self._model_hw(config)
        return b_idx, self._resize_image_to_model_grid(image, target_h, target_w)

    @staticmethod
    def _resize_image_to_model_grid(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
        image = np.ascontiguousarray(image)
        if image.shape[:2] == (target_h, target_w):
            return image
        interpolation = cv2.INTER_AREA
        if target_h > image.shape[0] or target_w > image.shape[1]:
            interpolation = cv2.INTER_LINEAR
        return np.ascontiguousarray(cv2.resize(image, (target_w, target_h), interpolation=interpolation))


class NYUAppCorrRawTransmissionPolicy(RawTransmissionPolicy):
    """Raw NYU depther transmission on the fixed 768x768 model grid."""

    def encode(self, images: Any, config: ExperimentConfig) -> Generator[List[Patch], None, None]:
        image_list = [images[b] for b in range(images.shape[0])] if isinstance(images, np.ndarray) else list(images)
        target_h, target_w = config.image_shape[:2]
        patches = []

        for image_idx, item in enumerate(image_list):
            metadata = {}
            image = item
            if isinstance(item, dict):
                metadata = dict(item.get("metadata") or {})
                image = item.get("image")
            if image is None or image.ndim != 3:
                raise RuntimeError(f"Expected HWC image, got {None if image is None else image.shape}")

            image = np.ascontiguousarray(image)
            target_shape = metadata.get("target_shape") or image.shape[:2]
            resized = _NYUAppCorrFixedGridMixin._resize_image_to_model_grid(
                image,
                int(target_h),
                int(target_w),
            )
            header = np.array(
                [resized.shape[0], resized.shape[1], resized.shape[2], target_shape[0], target_shape[1]],
                dtype=np.int32,
            ).tobytes()
            patch = Patch(image_idx=image_idx, spatial_idx=0, data=header + resized.tobytes())
            patch.batch_group_total = len(image_list)
            patch.target_shape = tuple(int(v) for v in target_shape)
            patches.append(patch)

        yield patches


class NYUAppCorrLaplacianPolicy(_NYUAppCorrFixedGridMixin, LaplacianPyramidPolicy):
    """Single or multi-level Laplacian NYU transmission on the fixed model grid."""

    def decode(self, patches: List[Patch], config: ExperimentConfig, canvas: np.ndarray = None) -> np.ndarray:
        decoded = super().decode(patches, config, canvas=canvas)
        return self._resize_decoded_to_model_grid(decoded, config)


class NYUAppCorrProgressiveLaplacianPolicy(_NYUAppCorrFixedGridMixin, ProgressiveLPyramidPolicy):
    """Progressive Laplacian NYU AppCorr transmission on the fixed model grid."""

    def decode(self, patches: List[Patch], config: ExperimentConfig, canvas: np.ndarray = None) -> np.ndarray:
        decoded = super().decode(patches, config, canvas=canvas)
        return self._resize_decoded_to_model_grid(decoded, config)
