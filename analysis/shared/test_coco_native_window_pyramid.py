import json
from pathlib import Path

import numpy as np

from offload.common.protocol import ExperimentConfig
from offload.policies.transmission.coco_window_progressive import (
    COCOWindowProgressiveLaplacianPolicy,
)
from offload.policies.transmission.laplacian import LaplacianPyramidPolicy
from offload.policies.transmission.raw import RawTransmissionPolicy


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "offload" / "config" / "coco_interleaved_static.json"


def _load_config() -> ExperimentConfig:
    return ExperimentConfig(**json.loads(CONFIG_PATH.read_text(encoding="utf-8")))


def _test_image() -> np.ndarray:
    rows = np.arange(301, dtype=np.uint16)[:, None]
    cols = np.arange(503, dtype=np.uint16)[None, :]
    return np.stack(
        (
            np.broadcast_to(rows % 251, (301, 503)),
            np.broadcast_to(cols % 253, (301, 503)),
            (17 * rows + 29 * cols) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)


def test_coco_window_base_reduces_native_content_before_projection():
    config = _load_config()
    policy = COCOWindowProgressiveLaplacianPolicy()
    image = _test_image()
    base_h, base_w = policy._base_hw(config)
    model_h, model_w = config.image_shape[:2]
    content_hw = (
        round(image.shape[0] * base_h / model_h),
        round(image.shape[1] * base_w / model_w),
    )
    expected = policy._resize_to_hw(
        policy._resize_to_hw(image, content_hw, np.uint8),
        (base_h, base_w),
        np.uint8,
    )

    actual = policy._build_native_base(image, config)

    np.testing.assert_array_equal(actual, expected)


def test_coco_window_full_arrival_reconstructs_projected_l0_exactly():
    config = _load_config()
    policy = COCOWindowProgressiveLaplacianPolicy()
    image = _test_image()
    groups = list(policy.encode([image], config))
    reconstructed = None
    for group in groups:
        reconstructed = policy.decode(group, config, canvas=reconstructed)

    expected = policy._project_to_model_grid(image, config)[None]
    np.testing.assert_array_equal(reconstructed, expected)


def test_coco_l1_and_l2_only_decode_to_fixed_detector_grid():
    image = _test_image()
    for level in (1, 2):
        config = _load_config()
        config.transmission_policy_name = "Laplacian"
        config.transmission_kwargs = {
            "pyramid_levels": [level],
            "compression_level": 1,
        }
        policy = LaplacianPyramidPolicy()
        groups = list(policy.encode([image], config))

        decoded = policy.decode(groups[0], config)

        assert decoded.shape == (1, *config.image_shape)


def test_coco_raw_projects_native_image_to_fixed_detector_grid():
    config = _load_config()
    config.transmission_policy_name = "Raw"
    policy = RawTransmissionPolicy()
    image = _test_image()

    patches = next(policy.encode([image], config))
    decoded = policy.decode(patches, config)

    expected = COCOWindowProgressiveLaplacianPolicy._project_to_model_grid(
        image,
        config,
    )[None]
    assert decoded.shape == (1, *config.image_shape)
    np.testing.assert_array_equal(decoded, expected)
