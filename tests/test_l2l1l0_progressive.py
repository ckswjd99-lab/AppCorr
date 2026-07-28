import unittest

import numpy as np
import torch

from offload.common.protocol import ExperimentConfig, Patch, Task
from offload.policies import get_transmission
from offload.policies.transmission.l2l1l0_progressive import (
    L2L1L0ProgressiveLPyramidPolicy,
)
from offload.server.model.dinov3_classifier import DINOv3ClassifierExecutor


def make_config(batch_size=1):
    return ExperimentConfig(
        model_name="dinov3_classifier",
        batch_size=batch_size,
        image_shape=(256, 256, 3),
        patch_size=(16, 16),
        scheduler_policy_name="GroupTrigger",
        transmission_policy_name="L2L1L0ProgressiveLaplacian",
        transmission_kwargs={
            "pyramid_levels": [2, 1, 0],
            "grouping_strategy": "grid",
            "l0_num_groups": 4,
            "total_layers": 40,
            "compression_level": 1,
        },
        appcorr_kwargs={
            "method": "partial_token",
            "mobile_pscore": "residual_energy",
            "mobile_pscore_weight": 1.0,
            "server_pscore": "cls_attn_prob_layermean",
            "server_pscore_weight": 1.0,
            "pscore_fusion": "multiply",
        },
    )


class L2L1L0TransmissionTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.image = rng.integers(
            0,
            256,
            size=(1, 256, 256, 3),
            dtype=np.uint8,
        )
        self.config = make_config()
        self.policy = L2L1L0ProgressiveLPyramidPolicy()

    def test_group_layout_and_metadata(self):
        groups = list(self.policy.encode(self.image, self.config))
        self.assertEqual([group[0].group_id for group in groups], list(range(6)))
        self.assertEqual([len(group) for group in groups], [16, 64, 64, 64, 64, 64])
        self.assertEqual(
            [{patch.res_level for patch in group} for group in groups],
            [{2}, {1}, {0}, {0}, {0}, {0}],
        )
        for group in groups:
            for patch in group:
                self.assertEqual(patch.batch_group_total, len(group))
                self.assertEqual(patch.num_correction_groups, 5)

    def test_final_decode_is_pixel_exact(self):
        patches = []
        for group in self.policy.encode(self.image, self.config):
            patches.extend(group)
        decoded = self.policy.decode(patches, self.config)
        np.testing.assert_array_equal(decoded, self.image)

    def test_registry(self):
        self.assertIsInstance(
            get_transmission("L2L1L0ProgressiveLaplacian"),
            L2L1L0ProgressiveLPyramidPolicy,
        )


class CoarseToFineSupportTest(unittest.TestCase):
    def setUp(self):
        self.config = make_config()

    def test_l1_patch_mapping(self):
        first = Patch(
            image_idx=0,
            spatial_idx=0,
            data=b"",
            res_level=1,
            group_id=1,
        )
        last = Patch(
            image_idx=0,
            spatial_idx=63,
            data=b"",
            res_level=1,
            group_id=1,
        )
        self.assertEqual(
            DINOv3ClassifierExecutor._model_spatial_indices_for_patch(
                first,
                self.config,
            ),
            (0, 1, 16, 17),
        )
        self.assertEqual(
            DINOv3ClassifierExecutor._model_spatial_indices_for_patch(
                last,
                self.config,
            ),
            (238, 239, 254, 255),
        )

    def test_complete_l1_support_covers_model_grid_once(self):
        support = []
        for spatial_idx in range(64):
            patch = Patch(
                image_idx=0,
                spatial_idx=spatial_idx,
                data=b"",
                res_level=1,
                group_id=1,
            )
            support.extend(
                DINOv3ClassifierExecutor._model_spatial_indices_for_patch(
                    patch,
                    self.config,
                )
            )
        self.assertEqual(sorted(support), list(range(256)))

    def test_l1_energy_hint_is_repeated_and_normalized(self):
        executor = DINOv3ClassifierExecutor.__new__(DINOv3ClassifierExecutor)
        executor.device = torch.device("cpu")
        patches = [
            Patch(
                image_idx=0,
                spatial_idx=0,
                data=b"",
                res_level=1,
                group_id=1,
                pscore_hint=3.0,
            ),
            Patch(
                image_idx=0,
                spatial_idx=1,
                data=b"",
                res_level=1,
                group_id=1,
                pscore_hint=1.0,
            ),
        ]
        score_map = executor._build_mobile_pscore_hint_map(
            Task(0, 0, patches, []),
            {},
            self.config,
            batch_size=1,
            num_patches=256,
        )

        torch.testing.assert_close(score_map.sum(), torch.tensor(1.0))
        torch.testing.assert_close(
            score_map[0, torch.tensor([0, 1, 16, 17])],
            torch.full((4,), 3.0 / 16.0),
        )
        torch.testing.assert_close(
            score_map[0, torch.tensor([2, 3, 18, 19])],
            torch.full((4,), 1.0 / 16.0),
        )


if __name__ == "__main__":
    unittest.main()
