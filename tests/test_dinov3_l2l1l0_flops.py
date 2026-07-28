import unittest
from unittest import mock

from analysis.experiments.dinov3_l2l1l0_eval import (
    block_flops,
    bootstrap_accuracy_summary,
    estimate_request_flops,
    full_backbone_flops,
    wait_until_worker_ready,
)
from analysis.experiments.summarize_dinov3_l2l1l0_sweep import (
    paired_accuracy_delta,
)
from offload.common.protocol import ExperimentConfig


class FlopsEstimatorTest(unittest.TestCase):
    def test_bootstrap_accuracy_summary_is_deterministic(self):
        first = bootstrap_accuracy_summary(
            [True, True, False, True],
            num_resamples=100,
            seed=7,
        )
        second = bootstrap_accuracy_summary(
            [True, True, False, True],
            num_resamples=100,
            seed=7,
        )

        self.assertEqual(first, second)
        self.assertEqual(first["accuracy_percent"], 75.0)
        self.assertLessEqual(
            first["ci95_low_percent"],
            first["accuracy_percent"],
        )
        self.assertGreaterEqual(
            first["ci95_high_percent"],
            first["accuracy_percent"],
        )

    def test_paired_bootstrap_uses_per_sample_prediction_difference(self):
        delta = paired_accuracy_delta(
            [True, False, True, False],
            [True, True, False, False],
            num_resamples=100,
            seed=3,
        )
        self.assertEqual(delta["delta_top1_points"], 0.0)
        self.assertLess(delta["ci95_low_points"], 0.0)
        self.assertGreater(delta["ci95_high_points"], 0.0)

    def test_worker_readiness_handshake_orders_config_before_time_sync(self):
        control_queue = mock.Mock()
        result_queue = mock.Mock()
        result_queue.get.return_value = 123.5
        config = ExperimentConfig()

        ready_at = wait_until_worker_ready(
            control_queue,
            result_queue,
            config,
            timeout=7.0,
        )

        self.assertEqual(ready_at, 123.5)
        self.assertEqual(
            control_queue.put.call_args_list,
            [
                mock.call(("CONFIG", config)),
                mock.call(("TIME_SYNC", None)),
            ],
        )
        result_queue.get.assert_called_once_with(timeout=7.0)

    def test_full_inference_matches_reference(self):
        config = ExperimentConfig()
        estimate = estimate_request_flops(
            [{"type": "FULL_INFERENCE", "params": {}}],
            config,
        )
        self.assertEqual(
            estimate["total_dominant_flops"],
            full_backbone_flops(),
        )
        self.assertEqual(estimate["ratio_to_full_backbone"], 1.0)

    def test_tail_full_schedule_counts_each_layer_once_for_full_keep(self):
        config = ExperimentConfig(
            transmission_policy_name="L2L1L0ProgressiveLaplacian",
            transmission_kwargs={
                "pyramid_levels": [2, 1, 0],
                "l0_num_groups": 4,
            },
            appcorr_kwargs={
                "token_keep_ratio": 1.0,
                "token_keep_thres": None,
            },
        )
        events = [
            {"type": "APPROX_FORWARD", "params": {"layers": (0, 8)}},
            {
                "type": "CORRECT_FORWARD",
                "params": {"layers": (0, 8), "group_id": 1},
            },
            {"type": "APPROX_FORWARD", "params": {"layers": (8, 16)}},
            {
                "type": "CORRECT_FORWARD",
                "params": {"layers": (0, 16), "group_id": 2},
            },
            {"type": "APPROX_FORWARD", "params": {"layers": (16, 24)}},
            {
                "type": "CORRECT_FORWARD",
                "params": {"layers": (0, 24), "group_id": 3},
            },
            {"type": "APPROX_FORWARD", "params": {"layers": (24, 31)}},
            {
                "type": "CORRECT_FORWARD",
                "params": {"layers": (0, 31), "group_id": 4},
            },
            {"type": "APPROX_FORWARD", "params": {"layers": (31, 38)}},
            {
                "type": "CORRECT_FORWARD",
                "params": {"layers": (0, 38), "group_id": 5},
            },
            {
                "type": "APPROX_FORWARD",
                "params": {
                    "layers": (38, 40),
                    "phase": "final_full",
                },
            },
        ]
        estimate = estimate_request_flops(events, config)
        full_block = block_flops(261, 261)

        self.assertEqual(estimate["approx_flops"], 38 * full_block)
        self.assertEqual(estimate["final_full_flops"], 2 * full_block)
        self.assertGreater(estimate["correction_flops"], 0)


if __name__ == "__main__":
    unittest.main()
