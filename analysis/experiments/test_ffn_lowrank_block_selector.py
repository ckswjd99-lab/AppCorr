#!/usr/bin/env python3
"""Small CPU contracts for the FFN low-rank block selector."""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from appcorr.models.dinov3.layers.ffn_block_selector import (
    build_joint_swiglu_low_rank_factors,
    exact_swiglu_delta_selected_blocks,
    low_rank_swiglu_channel_score,
    mask_diagnostics,
    oracle_swiglu_channel_score,
    select_ffn_block_mask,
    select_ffn_2to4_mask,
    select_ffn_row_topk_mask,
)


class FFNLowRankBlockSelectorTest(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(71)
        self.base_x = torch.randn(2, 7, 16, generator=generator)
        self.corrected_x = self.base_x + 0.1 * torch.randn(
            self.base_x.shape,
            generator=generator,
        )
        self.gate_weight = torch.randn(24, 16, generator=generator) / 4
        self.up_weight = torch.randn(24, 16, generator=generator) / 4
        self.down_weight = torch.randn(16, 24, generator=generator) / 24**0.5
        self.gate_bias = torch.randn(24, generator=generator) / 8
        self.up_bias = torch.randn(24, generator=generator) / 8

    def test_full_support_matches_dense_exact_delta(self) -> None:
        score = oracle_swiglu_channel_score(
            self.base_x,
            self.corrected_x,
            self.gate_weight,
            self.up_weight,
            self.down_weight,
            gate_bias=self.gate_bias,
            up_bias=self.up_bias,
        )
        mask, _ = select_ffn_block_mask(
            score,
            keep_ratio=1.0,
            token_block_size=4,
            channel_block_size=8,
        )
        actual = exact_swiglu_delta_selected_blocks(
            self.base_x,
            self.corrected_x,
            self.gate_weight,
            self.up_weight,
            self.down_weight,
            mask,
            gate_bias=self.gate_bias,
            up_bias=self.up_bias,
            token_block_size=4,
        )
        base_hidden = F.silu(
            F.linear(self.base_x, self.gate_weight, self.gate_bias)
        ) * F.linear(self.base_x, self.up_weight, self.up_bias)
        corrected_hidden = F.silu(
            F.linear(self.corrected_x, self.gate_weight, self.gate_bias)
        ) * F.linear(self.corrected_x, self.up_weight, self.up_bias)
        expected = F.linear(corrected_hidden - base_hidden, self.down_weight)
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    def test_low_rank_score_and_mask_contract(self) -> None:
        factors = build_joint_swiglu_low_rank_factors(
            self.gate_weight,
            self.up_weight,
            rank=8,
            oversample=2,
            power_iterations=1,
            seed=17,
        )
        score, projected = low_rank_swiglu_channel_score(
            self.base_x,
            self.corrected_x,
            factors,
            self.down_weight,
            gate_bias=self.gate_bias,
            up_bias=self.up_bias,
        )
        self.assertEqual(score.shape, (2, 7, 24))
        self.assertEqual(projected.shape, (2, 7, 8))
        mask, block_score = select_ffn_block_mask(
            score,
            keep_ratio=0.5,
            token_block_size=4,
            channel_block_size=8,
        )
        self.assertEqual(mask.shape, score.shape)
        self.assertEqual(block_score.shape, (2, 2, 3))
        self.assertTrue(torch.equal(mask[:, :4], mask[:, :1].expand(-1, 4, -1)))
        self.assertTrue(torch.equal(mask[:, 4:], mask[:, 4:5].expand(-1, 3, -1)))
        blocked_channels = mask.reshape(2, 7, 3, 8)
        self.assertTrue(
            torch.equal(
                blocked_channels,
                blocked_channels[..., :1].expand_as(blocked_channels),
            )
        )

    def test_oracle_diagnostics_are_one_for_same_mask(self) -> None:
        score = oracle_swiglu_channel_score(
            self.base_x,
            self.corrected_x,
            self.gate_weight,
            self.up_weight,
            self.down_weight,
            gate_bias=self.gate_bias,
            up_bias=self.up_bias,
        )
        mask, _ = select_ffn_block_mask(
            score,
            keep_ratio=0.5,
            token_block_size=1,
            channel_block_size=8,
        )
        diagnostics = mask_diagnostics(mask, mask, score)
        self.assertAlmostEqual(diagnostics["oracle_block_recall"], 1.0)
        self.assertAlmostEqual(diagnostics["retained_energy_vs_oracle"], 1.0)


    def test_2to4_mask_keeps_two_per_group(self) -> None:
        score = torch.arange(2 * 3 * 12, dtype=torch.float32).reshape(2, 3, 12)
        mask = select_ffn_2to4_mask(score)
        self.assertEqual(mask.shape, score.shape)
        grouped = mask.reshape(2, 3, 3, 4)
        self.assertTrue(torch.equal(grouped.sum(dim=-1), torch.full((2, 3, 3), 2)))
        expected = torch.tensor([False, False, True, True])
        self.assertTrue(torch.equal(grouped[0, 0, 0], expected))

    def test_row_topk_mask_is_independent_per_token(self) -> None:
        score = torch.tensor(
            [[[9.0, 8.0, 1.0, 0.0], [0.0, 1.0, 8.0, 9.0]]]
        )
        mask = select_ffn_row_topk_mask(score, keep_ratio=0.5)
        expected = torch.tensor(
            [[[True, True, False, False], [False, False, True, True]]]
        )
        self.assertTrue(torch.equal(mask, expected))

if __name__ == "__main__":
    unittest.main()
