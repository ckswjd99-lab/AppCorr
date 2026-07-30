import json
from pathlib import Path

import numpy as np
import torch

from offload.common.protocol import ExperimentConfig, OpType
from offload.policies.scheduling.ade20k_window_trigger import (
    ADE20KWindowInterleavedPolicy,
)
from offload.policies.transmission.ade20k_window_progressive import (
    ADE20KWindowProgressiveLaplacianPolicy,
)
from offload.server.model.dinov3_lowres_expand_once import (
    build_parent_token_index,
    lift_token_grid,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = (
    REPO_ROOT
    / "offload"
    / "config"
    / "ade20k_m2f_lowres_expand_once_block_grid.json"
)


def _load_config() -> ExperimentConfig:
    return ExperimentConfig(**json.loads(CONFIG_PATH.read_text(encoding="utf-8")))


def test_runtime_config_uses_fixed_g4_interleaving():
    config = _load_config()
    policy = ADE20KWindowInterleavedPolicy(config)
    policy.num_groups = 4

    expected = {
        0: [(OpType.APPROX_FORWARD, (0, 10))],
        1: [
            (OpType.CORRECT_FORWARD, (0, 10)),
            (OpType.APPROX_FORWARD, (10, 20)),
        ],
        2: [
            (OpType.CORRECT_FORWARD, (0, 20)),
            (OpType.APPROX_FORWARD, (20, 30)),
        ],
        3: [
            (OpType.CORRECT_FORWARD, (0, 30)),
            (OpType.APPROX_FORWARD, (30, 40)),
        ],
        4: [(OpType.CORRECT_FORWARD, (0, 40))],
    }
    for group_id, expected_compute in expected.items():
        instructions = policy._get_pipeline_instructions(group_id, config)
        actual_compute = [
            (instruction.op_type, instruction.params.get("layers"))
            for instruction in instructions
            if instruction.op_type in {OpType.APPROX_FORWARD, OpType.CORRECT_FORWARD}
        ]
        assert actual_compute == expected_compute


def test_runtime_parent_lift_repeats_each_low_patch():
    low = torch.arange(2 + 4, dtype=torch.float32).view(1, 6, 1)
    lifted = lift_token_grid(low, (2, 2), (4, 4), num_prefix=2)
    parent = build_parent_token_index(
        (2, 2),
        (4, 4),
        num_prefix=2,
        device=torch.device("cpu"),
    )
    torch.testing.assert_close(lifted, low.index_select(1, parent))


def test_ade_base_is_built_from_native_image_before_model_resize():
    config = _load_config()
    config.input_profile_kwargs["mobile_resize_short_side"] = 64
    policy = ADE20KWindowProgressiveLaplacianPolicy()
    rows = np.arange(37, dtype=np.uint8)[:, None]
    cols = np.arange(53, dtype=np.uint8)[None, :]
    image = np.stack(
        (
            np.broadcast_to(rows, (37, 53)),
            np.broadcast_to(cols, (37, 53)),
            (rows + cols) % 251,
        ),
        axis=-1,
    )

    base_group = next(policy.encode([image], config))
    for patch in base_group:
        patch.target_shape = image.shape[:2]
    decoded_native = policy.decode_lowres(base_group, config)[0]
    native_l1 = policy._build_native_gaussians(image, max_lvl=1)[1]
    expected = policy._project_band_to_target(
        native_l1,
        lvl=1,
        config=config,
        dtype=np.uint8,
        image_hw=image.shape[:2],
    )

    np.testing.assert_array_equal(decoded_native, expected)
