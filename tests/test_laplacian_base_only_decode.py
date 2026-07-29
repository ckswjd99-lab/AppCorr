import unittest

import numpy as np

from offload.common.protocol import ExperimentConfig
from offload.policies.transmission.laplacian import (
    LaplacianPyramidPolicy,
)


class LaplacianBaseOnlyDecodeTest(unittest.TestCase):
    def setUp(self):
        self.policy = LaplacianPyramidPolicy()
        self.config = ExperimentConfig(
            batch_size=1,
            image_shape=(64, 64, 3),
            patch_size=(16, 16),
            transmission_policy_name="Laplacian",
            transmission_kwargs={
                "pyramid_levels": [2],
                "compression_level": 1,
            },
        )
        rng = np.random.default_rng(7)
        self.image = rng.integers(
            0,
            256,
            size=(1, 64, 64, 3),
            dtype=np.uint8,
        )

    def test_fixed_shape_base_only_is_expanded_to_model_input_size(self):
        patches = list(self.policy.encode(self.image, self.config))[0]
        decoded = self.policy.decode(patches, self.config)

        gaussian = self.policy._build_native_gaussians(
            self.image[0],
            max_lvl=2,
        )[2]
        expected = self.policy._iterative_upsample(
            gaussian,
            start_lvl=2,
            end_lvl=0,
            H=64,
            W=64,
        )
        self.assertEqual(decoded.shape, (1, 64, 64, 3))
        np.testing.assert_array_equal(decoded[0], expected)

    def test_preserved_shape_base_only_is_expanded_to_target_shape(self):
        self.config.transmission_kwargs["preserve_input_shape"] = True
        patches = list(self.policy.encode(self.image, self.config))[0]
        for patch in patches:
            patch.target_shape = (64, 64)

        decoded = self.policy.decode(patches, self.config)

        self.assertIsInstance(decoded, list)
        self.assertEqual(decoded[0].shape, (64, 64, 3))


if __name__ == "__main__":
    unittest.main()
