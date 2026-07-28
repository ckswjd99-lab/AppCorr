import itertools
import unittest

import torch

from offload.common.protocol import ExperimentConfig, OpType, Patch
from offload.policies.scheduling.group_trigger import (
    GroupTriggerPolicy,
    build_balanced_layer_boundaries,
)
from offload.server.model.dinov3_classifier import DINOv3ClassifierExecutor


class BalancedLayerBoundariesTest(unittest.TestCase):
    def test_even_existing_schedule(self):
        self.assertEqual(
            build_balanced_layer_boundaries(40, 4, 0),
            (0, 10, 20, 30, 40),
        )

    def test_tail_full_schedule(self):
        self.assertEqual(
            build_balanced_layer_boundaries(40, 5, 2),
            (0, 8, 16, 24, 31, 38),
        )
        self.assertEqual(
            build_balanced_layer_boundaries(40, 5, 3),
            (0, 8, 16, 23, 30, 37),
        )

    def test_rejects_too_short_progressive_prefix(self):
        with self.assertRaises(ValueError):
            build_balanced_layer_boundaries(40, 5, 36)


class GroupTriggerTailFullTest(unittest.TestCase):
    @staticmethod
    def _config(final_full_layers):
        return ExperimentConfig(
            scheduler_policy_name="GroupTrigger",
            scheduler_kwargs={"final_full_layers": final_full_layers},
            transmission_policy_name="ProgressiveLaplacian",
            transmission_kwargs={"total_layers": 40, "num_groups": 5},
        )

    @staticmethod
    def _patch(group_id, num_correction_groups=5):
        return Patch(
            image_idx=0,
            spatial_idx=0,
            data=b"",
            group_id=group_id,
            batch_group_total=1,
            num_correction_groups=num_correction_groups,
        )

    def test_group_instruction_sequence_with_two_tail_layers(self):
        policy = GroupTriggerPolicy()
        config = self._config(final_full_layers=2)

        expected_ranges = {
            0: [(OpType.APPROX_FORWARD, (0, 8))],
            1: [
                (OpType.CORRECT_FORWARD, (0, 8)),
                (OpType.APPROX_FORWARD, (8, 16)),
            ],
            2: [
                (OpType.CORRECT_FORWARD, (0, 16)),
                (OpType.APPROX_FORWARD, (16, 24)),
            ],
            3: [
                (OpType.CORRECT_FORWARD, (0, 24)),
                (OpType.APPROX_FORWARD, (24, 31)),
            ],
            4: [
                (OpType.CORRECT_FORWARD, (0, 31)),
                (OpType.APPROX_FORWARD, (31, 38)),
            ],
            5: [
                (OpType.CORRECT_FORWARD, (0, 38)),
                (OpType.APPROX_FORWARD, (38, 40)),
            ],
        }

        for group_id, expected in expected_ranges.items():
            instructions = policy._get_pipeline_instructions(
                self._patch(group_id),
                config,
            )
            observed = [
                (instruction.op_type, instruction.params["layers"])
                for instruction in instructions
                if instruction.op_type
                in {OpType.APPROX_FORWARD, OpType.CORRECT_FORWARD}
            ]
            self.assertEqual(observed, expected)

        final_instructions = policy._get_pipeline_instructions(
            self._patch(5),
            config,
        )
        final_forward = next(
            instruction
            for instruction in final_instructions
            if instruction.op_type == OpType.APPROX_FORWARD
        )
        self.assertEqual(final_forward.params["cache_mode"], "none")
        self.assertEqual(final_forward.params["phase"], "final_full")

    def test_existing_schedule_is_unchanged_when_disabled(self):
        config = ExperimentConfig(
            scheduler_policy_name="GroupTrigger",
            scheduler_kwargs={},
            transmission_policy_name="ProgressiveLaplacian",
            transmission_kwargs={"total_layers": 40, "num_groups": 4},
        )
        policy = GroupTriggerPolicy()

        final_instructions = policy._get_pipeline_instructions(
            self._patch(4, num_correction_groups=0),
            config,
        )
        compute = [
            instruction
            for instruction in final_instructions
            if instruction.op_type
            in {OpType.APPROX_FORWARD, OpType.CORRECT_FORWARD}
        ]
        self.assertEqual(len(compute), 1)
        self.assertEqual(compute[0].op_type, OpType.CORRECT_FORWARD)
        self.assertEqual(compute[0].params["layers"], (0, 40))

    def test_decide_uses_patch_correction_group_count(self):
        config = self._config(final_full_layers=2)
        policy = GroupTriggerPolicy()
        patch = self._patch(0)
        task = policy.decide([patch], config, itertools.count())

        self.assertIsNotNone(task)
        approx = next(
            instruction
            for instruction in task.instructions
            if instruction.op_type == OpType.APPROX_FORWARD
        )
        self.assertEqual(approx.params["layers"], (0, 8))


class _DummyBlock(torch.nn.Module):
    def __init__(self, increment):
        super().__init__()
        self.increment = increment

    def forward(self, x, rope):
        del rope
        return x + self.increment


class _DummyBackbone:
    def __init__(self):
        self.blocks = [_DummyBlock(1.0), _DummyBlock(2.0)]


class _DummyModel:
    def __init__(self):
        self.backbone = _DummyBackbone()


class NoCacheForwardTest(unittest.TestCase):
    def test_suffix_uses_stock_forward_without_new_cache(self):
        executor = DINOv3ClassifierExecutor.__new__(DINOv3ClassifierExecutor)
        executor.device = torch.device("cpu")
        executor.model = _DummyModel()
        existing_cache = {"layer0_kv": torch.ones(1)}
        context = {
            "current_feature": torch.zeros(1, 3, 4),
            "input_tokens": torch.zeros(1, 3, 4),
            "rope_sincos": None,
            "cache_feature": existing_cache,
        }

        metadata = executor.approx_forward(
            {"layers": (0, 2), "cache_mode": "none"},
            context,
            ExperimentConfig(),
        )

        torch.testing.assert_close(
            context["current_feature"],
            torch.full((1, 3, 4), 3.0),
        )
        self.assertIs(context["cache_feature"], existing_cache)
        self.assertEqual(set(existing_cache), {"layer0_kv"})
        self.assertEqual(metadata["cache_mode"], "none")


if __name__ == "__main__":
    unittest.main()
