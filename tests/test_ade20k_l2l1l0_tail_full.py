import types
import unittest

import numpy as np
import torch

from offload.common.protocol import (
    ExperimentConfig,
    OpType,
    Patch,
    Task,
    normalize_appcorr_kwargs,
)
from appcorr.models.dinov3.layers.block import SelfAttentionBlock
from offload.policies.scheduling.ade20k_window_trigger import (
    ADE20KWindowInterleavedPolicy,
)
from offload.policies.transmission.ade20k_window_progressive import (
    ADE20KL2L1ProgressiveLaplacianPolicy,
    ADE20KWindowL2L1L0ProgressiveLaplacianPolicy,
)
from offload.policies.transmission.laplacian import LaplacianPyramidPolicy
from offload.server.model.dinov3_segmentor_m2f import (
    DINOv3SegmentorM2FExecutor,
)


def make_config(
    *,
    final_full_layers=3,
    original_hw=(48, 80),
):
    return ExperimentConfig(
        model_name="dinov3_segmentor_m2f",
        batch_size=1,
        image_shape=(64, 64, 3),
        patch_size=(16, 16),
        input_profile_name="dinov3_ade20k_m2f_official",
        input_profile_kwargs={
            "mobile_resize_short_side": 64,
            "server_crop_size": 64,
            "server_stride": 48,
            "server_eval_mode": "single",
        },
        scheduler_policy_name="ADE20KWindowInterleaved",
        scheduler_kwargs={
            "total_layers": 40,
            "final_full_layers": final_full_layers,
        },
        transmission_policy_name=(
            "ADE20KWindowL2L1L0ProgressiveLaplacian"
        ),
        transmission_kwargs={
            "pyramid_levels": [2, 1, 0],
            "grouping_strategy": "crop_cover",
            "num_groups": 16,
            "compression_level": 1,
            "preserve_input_shape": True,
        },
        appcorr_kwargs={
            "generated_from_client": True,
            "method": "partial_token",
            "token_keep_thres": 4e-5,
            "l1_token_keep_thres": 5e-5,
            "l0_token_keep_thres": 4e-5,
            "l1_pscore_mode": "positive_residual_difference",
            "l0_pscore_mode": (
                "conditional_cumulative_residual_energy"
            ),
            "l1_l0_disjoint_support": True,
            "mobile_pscore": "residual_energy",
            "mobile_pscore_weight": 1.0,
            "server_pscore": "patch_attn_prob_layermean",
            "server_pscore_weight": 1.0,
            "pscore_fusion": "geo_mean",
        },
    )


class ADE20KTailFullScheduleTest(unittest.TestCase):
    def test_dynamic_groups_hold_back_three_layers(self):
        config = make_config(final_full_layers=3)
        policy = ADE20KWindowInterleavedPolicy()
        policy.num_groups = 5

        expected = {
            0: [(OpType.APPROX_FORWARD, (0, 8))],
            1: [
                (OpType.CORRECT_FORWARD, (0, 8)),
                (OpType.APPROX_FORWARD, (8, 16)),
            ],
            2: [
                (OpType.CORRECT_FORWARD, (0, 16)),
                (OpType.APPROX_FORWARD, (16, 23)),
            ],
            3: [
                (OpType.CORRECT_FORWARD, (0, 23)),
                (OpType.APPROX_FORWARD, (23, 30)),
            ],
            4: [
                (OpType.CORRECT_FORWARD, (0, 30)),
                (OpType.APPROX_FORWARD, (30, 37)),
            ],
            5: [
                (OpType.CORRECT_FORWARD, (0, 37)),
                (OpType.APPROX_FORWARD, (37, 40)),
            ],
        }
        for group_id, expected_compute in expected.items():
            instructions = policy._get_pipeline_instructions(group_id, config)
            observed = [
                (instruction.op_type, instruction.params["layers"])
                for instruction in instructions
                if instruction.op_type
                in {OpType.APPROX_FORWARD, OpType.CORRECT_FORWARD}
            ]
            self.assertEqual(observed, expected_compute)

        final_forward = next(
            instruction
            for instruction in policy._get_pipeline_instructions(5, config)
            if instruction.op_type == OpType.APPROX_FORWARD
        )
        self.assertEqual(final_forward.params["cache_mode"], "none")
        self.assertEqual(final_forward.params["phase"], "final_full")

    def test_disabled_tail_preserves_existing_final_correction(self):
        config = make_config(final_full_layers=0)
        policy = ADE20KWindowInterleavedPolicy()
        policy.num_groups = 2
        instructions = policy._get_pipeline_instructions(2, config)
        compute = [
            instruction
            for instruction in instructions
            if instruction.op_type
            in {OpType.APPROX_FORWARD, OpType.CORRECT_FORWARD}
        ]
        self.assertEqual(len(compute), 1)
        self.assertEqual(compute[0].op_type, OpType.CORRECT_FORWARD)
        self.assertEqual(compute[0].params["layers"], (0, 40))


class ADE20KL2L1L0TransmissionTest(unittest.TestCase):
    def test_l2_l1_completed_decode_matches_l1_only(self):
        config = make_config(original_hw=(48, 80))
        config.transmission_policy_name = (
            "ADE20KL2L1ProgressiveLaplacian"
        )
        config.transmission_kwargs = {
            "pyramid_levels": [2, 1],
            "grouping_strategy": "single",
            "num_groups": 1,
            "compression_level": 1,
            "preserve_input_shape": True,
        }
        config.appcorr_kwargs = {
            "generated_from_client": True,
            "method": "partial_token",
            "token_keep_thres": 4e-5,
            "mobile_pscore": "residual_energy",
        }
        rng = np.random.default_rng(20260729)
        image = rng.integers(
            0,
            256,
            size=(48, 80, 3),
            dtype=np.uint8,
        )

        policy = ADE20KL2L1ProgressiveLaplacianPolicy()
        groups = list(policy.encode([image], config))
        self.assertEqual(
            [(group[0].group_id, group[0].res_level) for group in groups],
            [(0, 2), (1, 1)],
        )
        for group in groups:
            for patch in group:
                self.assertEqual(patch.batch_group_total, len(group))
                self.assertEqual(patch.num_correction_groups, 1)
                self.assertEqual(patch.target_shape, image.shape[:2])
        decoded = policy.decode(
            [patch for group in groups for patch in group],
            config,
        )[0]

        l1_config = ExperimentConfig(
            **{
                **config.__dict__,
                "transmission_policy_name": "Laplacian",
                "transmission_kwargs": {
                    "pyramid_levels": [1],
                    "compression_level": 1,
                    "preserve_input_shape": True,
                },
                "appcorr_kwargs": {"enabled": False},
            }
        )
        reference_policy = LaplacianPyramidPolicy()
        reference_patches = [
            patch
            for group in reference_policy.encode([image], l1_config)
            for patch in group
        ]
        for patch in reference_patches:
            patch.target_shape = image.shape[:2]
        reference = reference_policy.decode(
            reference_patches,
            l1_config,
        )[0]
        np.testing.assert_array_equal(decoded, reference)

    def test_positive_difference_keeps_only_l1_dominant_regions(self):
        policy = ADE20KWindowL2L1L0ProgressiveLaplacianPolicy
        l1_dominant = np.full((2, 2, 1), 3, dtype=np.int16)
        fine_weak = np.ones((2, 2, 1), dtype=np.int16)
        score = policy._positive_residual_difference_score(
            l1_dominant,
            fine_weak,
            patch_hw=(1, 1),
            l1_grid_hw=(1, 1),
        )
        self.assertEqual(float(score[0, 0]), 32.0)

        fine_dominant = policy._positive_residual_difference_score(
            fine_weak,
            l1_dominant,
            patch_hw=(1, 1),
            l1_grid_hw=(1, 1),
        )
        self.assertEqual(float(fine_dominant[0, 0]), 0.0)

    def test_remaining_energy_ratio_gate_rejects_unsafe_l1_cells(self):
        policy = ADE20KWindowL2L1L0ProgressiveLaplacianPolicy
        l1_effect = np.full((2, 2, 1), 4, dtype=np.int16)
        remaining = np.full((2, 2, 1), 3, dtype=np.int16)

        ungated = policy._positive_residual_difference_score(
            l1_effect,
            remaining,
            patch_hw=(1, 1),
            l1_grid_hw=(1, 1),
        )
        gated = policy._positive_residual_difference_score(
            l1_effect,
            remaining,
            patch_hw=(1, 1),
            l1_grid_hw=(1, 1),
            remaining_ratio_max=0.5,
        )
        self.assertGreater(float(ungated[0, 0]), 0.0)
        self.assertEqual(float(gated[0, 0]), 0.0)

    def test_variable_shape_group_layout_and_decode_reference(self):
        config = make_config()
        rng = np.random.default_rng(20260728)
        image = rng.integers(0, 256, size=(48, 80, 3), dtype=np.uint8)
        policy = ADE20KWindowL2L1L0ProgressiveLaplacianPolicy()

        groups = list(policy.encode([image], config))
        self.assertEqual(
            [group[0].group_id for group in groups],
            [0, 1, 2, 3],
        )
        self.assertEqual([len(group) for group in groups], [2, 8, 16, 12])
        self.assertEqual(
            [{patch.res_level for patch in group} for group in groups],
            [{2}, {1}, {0}, {0}],
        )
        for group in groups:
            for patch in group:
                self.assertEqual(patch.batch_group_total, len(group))
                self.assertEqual(patch.num_correction_groups, 3)
                self.assertEqual(patch.target_shape, image.shape[:2])

        decoded = policy.decode(
            [patch for group in groups for patch in group],
            config,
        )
        reference_policy = LaplacianPyramidPolicy()
        reference_patches = [
            patch
            for group in reference_policy.encode([image], config)
            for patch in group
        ]
        for patch in reference_patches:
            patch.target_shape = image.shape[:2]
        reference = reference_policy.decode(reference_patches, config)
        np.testing.assert_array_equal(decoded[0], reference[0])

    def test_encoded_hints_match_exact_cross_level_state_deltas(self):
        config = make_config(original_hw=(64, 64))
        rng = np.random.default_rng(7)
        image = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)
        policy = ADE20KWindowL2L1L0ProgressiveLaplacianPolicy()

        _, gaussians = policy._process_image_base_layer(
            0,
            image,
            config,
            image.shape[:2],
        )
        l1_residual = policy._projected_residual_band(
            gaussians,
            previous_level=2,
            level=1,
            config=config,
            image_hw=image.shape[:2],
        )
        l0_residual = policy._projected_residual_band(
            gaussians,
            previous_level=1,
            level=0,
            config=config,
            image_hw=image.shape[:2],
        )
        l1_effect, l0_after_l1, l0_from_l2 = (
            policy._reconstruction_residuals_at_l0(
                gaussians,
                l1_residual,
                l0_residual,
                config=config,
                image_hw=image.shape[:2],
            )
        )
        expected_l1 = policy._positive_residual_difference_score(
            l1_effect,
            l0_after_l1,
            patch_hw=(16, 16),
            l1_grid_hw=(2, 2),
        ).reshape(-1)
        expected_l0 = policy._patch_energy_map(
            l0_from_l2,
            (16, 16),
        ).reshape(-1)
        expected_l0_after_l1 = policy._patch_energy_map(
            l0_after_l1,
            (16, 16),
        ).reshape(-1)

        groups = list(policy.encode([image], config))
        observed_l1 = np.asarray(
            [patch.pscore_hint for patch in groups[1]],
            dtype=np.float32,
        )
        l0_patches = sorted(
            [
                patch
                for group in groups[2:]
                for patch in group
            ],
            key=lambda patch: patch.spatial_idx,
        )
        observed_l0 = np.asarray(
            [patch.pscore_hint for patch in l0_patches],
            dtype=np.float32,
        )
        observed_l0_after_l1 = np.asarray(
            [
                patch.pscore_hint_if_l1_corrected
                for patch in l0_patches
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(observed_l1, expected_l1)
        np.testing.assert_allclose(observed_l0, expected_l0)
        np.testing.assert_allclose(
            observed_l0_after_l1,
            expected_l0_after_l1,
        )

    def test_l1_energy_hint_expands_to_final_token_cells(self):
        config = make_config(original_hw=(64, 64))
        executor = DINOv3SegmentorM2FExecutor.__new__(
            DINOv3SegmentorM2FExecutor
        )
        executor.device = torch.device("cpu")
        patch = Patch(
            image_idx=0,
            spatial_idx=0,
            data=b"",
            res_level=1,
            group_id=1,
            pscore_hint=3.0,
            target_shape=(64, 64),
        )
        result = executor._build_mobile_pscore_hint_maps(
            Task(0, 0, [patch], []),
            [np.zeros((64, 64, 3), dtype=np.uint8)],
            config,
            [0],
        )
        self.assertIsNotNone(result)
        hint_map, hint_hw = result[0]
        self.assertEqual(hint_hw, (4, 4))
        expected_nonzero = torch.tensor([0, 1, 4, 5])
        torch.testing.assert_close(
            hint_map[0, expected_nonzero],
            torch.full((4,), 0.25),
        )
        self.assertEqual(int(torch.count_nonzero(hint_map)), 4)


class ADE20KM2FExecutionContractTest(unittest.TestCase):
    def test_support_mode_normalization_preserves_legacy_disjoint(self):
        legacy = normalize_appcorr_kwargs({
            "l1_l0_disjoint_support": True,
        })
        self.assertEqual(legacy["l1_l0_support_mode"], "disjoint")
        self.assertTrue(legacy["l1_l0_disjoint_support"])

        conditional = normalize_appcorr_kwargs({
            "l1_l0_support_mode": "conditional_reentry",
            "l1_l0_reentry_ratio": 0.25,
        })
        self.assertFalse(conditional["l1_l0_disjoint_support"])
        self.assertEqual(
            conditional["l1_l0_support_mode"],
            "conditional_reentry",
        )
        self.assertEqual(conditional["l1_l0_reentry_ratio"], 0.25)

        threshold_only = normalize_appcorr_kwargs({
            "token_keep_thres": 4e-5,
            "l1_token_keep_thres": 4e-5,
            "l0_token_keep_thres": 4e-5,
            "l1_l0_support_mode": "conditional_threshold",
        })
        self.assertEqual(
            threshold_only["l1_l0_support_mode"],
            "conditional_threshold",
        )
        self.assertFalse(threshold_only["l1_l0_disjoint_support"])
        self.assertEqual(
            DINOv3SegmentorM2FExecutor._token_keep_threshold_for_group(
                threshold_only,
                group_id=1,
                l2l1l0_mode=True,
            ),
            4e-5,
        )
        self.assertEqual(
            DINOv3SegmentorM2FExecutor._token_keep_threshold_for_group(
                threshold_only,
                group_id=2,
                l2l1l0_mode=True,
            ),
            4e-5,
        )

    def test_conditional_threshold_uses_pscore_without_forced_reentry(self):
        config = make_config(final_full_layers=0)
        config.appcorr_kwargs.pop("l1_l0_disjoint_support")
        config.appcorr_kwargs.update({
            "l1_l0_support_mode": "conditional_threshold",
            "l1_pscore_mode": "residual_energy",
            "l0_pscore_mode": "conditional_cumulative_residual_energy",
            "l1_token_keep_thres": 4e-5,
            "l0_token_keep_thres": 4e-5,
        })
        self.assertTrue(
            DINOv3SegmentorM2FExecutor
            ._uses_l1_l0_conditional_pscore(config)
        )
        self.assertFalse(
            DINOv3SegmentorM2FExecutor
            ._uses_l1_l0_conditional_reentry(config)
        )

    def test_conditional_hints_choose_branch_before_normalization(self):
        primary = [torch.tensor([[8.0, 2.0]])]
        after_l1 = [torch.tensor([[1.0, 2.0]])]
        l1_mask = [torch.tensor([[False, True, False]])]

        effective, reentry_masks = (
            DINOv3SegmentorM2FExecutor
            ._build_conditional_mobile_pscore_hints(
                primary,
                after_l1,
                l1_mask,
                group_id=2,
                num_pretokens=1,
            )
        )
        torch.testing.assert_close(
            effective[0],
            torch.tensor([[1.0 / 3.0, 2.0 / 3.0]]),
        )
        torch.testing.assert_close(
            reentry_masks[0],
            torch.tensor([[True, False]]),
        )

    def test_reentry_budget_uses_highest_combined_risk(self):
        scores = torch.tensor([[0.7, 0.9, 0.8, 0.1]])
        candidates = torch.tensor([[True, True, True, False]])
        eligible = SelfAttentionBlock._select_reentry_eligibility(
            scores,
            candidates,
            reentry_ratio=1.0 / 3.0,
        )
        torch.testing.assert_close(
            eligible,
            torch.tensor([[False, True, False, True]]),
        )
        forced = SelfAttentionBlock._apply_reentry_budget(
            torch.zeros_like(candidates),
            candidates,
            eligible,
        )
        torch.testing.assert_close(
            forced,
            torch.tensor([[False, True, False, False]]),
        )

    def test_partial_token_plan_counts_are_weighted_by_round_depth(self):
        caches = [
            {
                "_partial_token_kept_patch_total": torch.tensor(10.0),
                "_partial_token_full_patch_total": torch.tensor(20.0),
                "_partial_token_sample_total": torch.tensor(1.0),
            },
            {
                "_partial_token_kept_patch_total": torch.tensor(4.0),
                "_partial_token_full_patch_total": torch.tensor(8.0),
                "_partial_token_sample_total": torch.tensor(1.0),
            },
        ]

        before = DINOv3SegmentorM2FExecutor._partial_token_plan_totals(
            caches
        )
        caches[0]["_partial_token_kept_patch_total"] = torch.tensor(16.0)
        caches[0]["_partial_token_full_patch_total"] = torch.tensor(32.0)
        caches[0]["_partial_token_sample_total"] = torch.tensor(2.0)
        caches[1]["_partial_token_kept_patch_total"] = torch.tensor(7.0)
        caches[1]["_partial_token_full_patch_total"] = torch.tensor(14.0)
        caches[1]["_partial_token_sample_total"] = torch.tensor(2.0)

        DINOv3SegmentorM2FExecutor._accumulate_partial_token_layer_totals(
            caches,
            before,
            layer_count=14,
            phase="l1",
        )
        aggregate = DINOv3SegmentorM2FExecutor._aggregate_cache_features(
            caches
        )
        self.assertEqual(
            float(aggregate["_partial_token_kept_patch_layer_total"]),
            (6.0 + 3.0) * 14,
        )
        self.assertEqual(
            float(aggregate["_partial_token_full_patch_layer_total"]),
            (12.0 + 6.0) * 14,
        )
        self.assertEqual(
            float(aggregate["_partial_token_sample_layer_total"]),
            2.0 * 14,
        )
        self.assertEqual(
            float(aggregate["_partial_token_l1_kept_patch_layer_total"]),
            (6.0 + 3.0) * 14,
        )
        self.assertEqual(
            float(aggregate["_partial_token_l1_full_patch_layer_total"]),
            (12.0 + 6.0) * 14,
        )

    def test_l1_selection_is_removed_from_l0_candidates(self):
        query_state = types.SimpleNamespace(
            active_batch_idx=torch.tensor([0, 0, 0]),
            active_token_idx=torch.tensor([0, 2, 4]),
            query_valid_mask=torch.ones(1, 3, dtype=torch.bool),
        )
        plan = types.SimpleNamespace(fixed_query_state=query_state)
        cache = {"_partial_token_query_plan_cache": {"shared": plan}}
        context = {}
        items = [{
            "src_idx": 0,
            "input_tokens": torch.zeros(1, 5, 2),
        }]
        DINOv3SegmentorM2FExecutor._capture_l1_selected_token_masks(
            items,
            cache,
            context,
            num_pretokens=1,
        )

        dindice = torch.tensor([[0, 1, 2, 3, 4]])
        filtered, excluded = (
            DINOv3SegmentorM2FExecutor._exclude_l1_selected_dindice(
                dindice,
                context["m2f_l1_selected_token_masks"],
                src_idx=0,
                num_pretokens=1,
            )
        )
        torch.testing.assert_close(
            filtered,
            torch.tensor([[0, 1, 3]]),
        )
        self.assertEqual(float(excluded), 2.0)
        selected_mask = context["m2f_l1_selected_token_masks"][0]
        self.assertFalse(
            bool(selected_mask.gather(1, filtered[:, 1:]).any())
        )

    def test_l1_full_plan_and_shifted_l0_plans(self):
        config = make_config()
        executor = DINOv3SegmentorM2FExecutor.__new__(
            DINOv3SegmentorM2FExecutor
        )
        executor.device = torch.device("cpu")
        backbone = types.SimpleNamespace(n_storage_tokens=0)
        executor.model = types.SimpleNamespace(
            segmentation_model=[types.SimpleNamespace(backbone=backbone)]
        )
        input_tokens = torch.zeros(1, 17, 4)
        context = {
            "m2f_x_backbones": [input_tokens],
            "m2f_group_maps": [
                torch.tensor([[1] * 8 + [2] * 8], dtype=torch.long)
            ],
        }

        executor.prepare_group_maps_and_dindices(None, context, config)
        plans = context["m2f_group_plans"][0]
        self.assertEqual(set(plans), {1, 2, 3})
        self.assertEqual(plans[1].full_dindice.shape[1], 17)
        self.assertEqual(plans[2].full_dindice.shape[1], 9)
        self.assertEqual(plans[3].full_dindice.shape[1], 9)
        torch.testing.assert_close(
            plans[1].full_dindice[0],
            torch.arange(17),
        )

    def test_final_suffix_uses_stock_blocks_without_cache_growth(self):
        class DummyBlock(torch.nn.Module):
            def __init__(self, increment):
                super().__init__()
                self.increment = increment

            def forward(self, x, rope):
                del rope
                return x + self.increment

            def approx(self, *args, **kwargs):
                raise AssertionError("deferred suffix must not call approx()")

        config = make_config()
        executor = DINOv3SegmentorM2FExecutor.__new__(
            DINOv3SegmentorM2FExecutor
        )
        executor.device = torch.device("cpu")
        executor.autocast_dtype = torch.bfloat16
        backbone = types.SimpleNamespace(
            blocks=[DummyBlock(1.0), DummyBlock(2.0)]
        )
        adapter = types.SimpleNamespace(
            backbone=backbone,
            interaction_indexes=[1],
        )
        executor.model = types.SimpleNamespace(segmentation_model=[adapter])
        cache = {"existing": torch.ones(1)}
        context = {
            "m2f_x_backbones": [torch.zeros(1, 4, 3)],
            "m2f_rope_sincos": [None],
            "m2f_current_features": [torch.zeros(1, 4, 3)],
            "m2f_intermediate_raw": [[]],
            "m2f_current_layer": 0,
            "m2f_cache_features": [cache],
        }

        metadata = executor.approx_forward(
            {"layers": (0, 2), "cache_mode": "none"},
            context,
            config,
        )
        torch.testing.assert_close(
            context["m2f_current_features"][0],
            torch.full((1, 4, 3), 3.0),
        )
        self.assertEqual(len(context["m2f_intermediate_raw"][0]), 1)
        self.assertEqual(set(cache), {"existing"})
        self.assertEqual(metadata["cache_mode"], "none")


if __name__ == "__main__":
    unittest.main()
