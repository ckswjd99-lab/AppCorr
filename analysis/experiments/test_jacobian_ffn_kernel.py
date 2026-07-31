#!/usr/bin/env python3
"""CUDA parity tests for block-sparse FFN down projection."""

from __future__ import annotations

import unittest

from analysis.shared.cuda_environment import configure_triton_cuda_environment

configure_triton_cuda_environment()

import torch
import torch.nn.functional as F

from appcorr.models.dinov3.layers.jacobian_support import (
    ffn_block_index_from_mask,
)
from appcorr.models.dinov3.layers.triton_kernels import (
    block_sparse_down_projection_triton,
    block_sparse_ffn_delta_triton,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class BlockSparseDownProjectionKernelTest(unittest.TestCase):
    def test_matches_dense_mask_with_ragged_support(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(47)
        batch, tokens, channels, outputs = 2, 13, 96, 80
        token_block_size = 4
        channel_block_size = 16
        hidden_delta = torch.randn(
            batch,
            tokens,
            channels,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        weight = torch.randn(
            outputs,
            channels,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        mask = torch.zeros(
            batch,
            tokens,
            channels,
            device="cuda",
            dtype=torch.bool,
        )
        support = (
            ((0, 2), (1, 4, 5), (0,), (2, 3)),
            ((1,), (0, 2), (3, 4, 5), (2,)),
        )
        for batch_index, token_blocks in enumerate(support):
            for token_block, channel_blocks in enumerate(token_blocks):
                token_start = token_block * token_block_size
                token_end = min(token_start + token_block_size, tokens)
                for channel_block in channel_blocks:
                    channel_start = channel_block * channel_block_size
                    channel_end = min(
                        channel_start + channel_block_size,
                        channels,
                    )
                    mask[
                        batch_index,
                        token_start:token_end,
                        channel_start:channel_end,
                    ] = True

        block_index = ffn_block_index_from_mask(
            mask,
            channel_block_size=channel_block_size,
            token_block_size=token_block_size,
        )
        actual = block_sparse_down_projection_triton(
            hidden_delta,
            weight,
            block_index,
            token_block_size=token_block_size,
            channel_block_size=channel_block_size,
        )
        expected = F.linear(hidden_delta.masked_fill(~mask, 0), weight)

        torch.testing.assert_close(
            actual.float(),
            expected.float(),
            rtol=1e-2,
            atol=1e-2,
        )

    def test_full_ffn_delta_matches_dense_mask_with_ragged_support(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(83)
        batch, tokens, hidden, channels, outputs = 2, 13, 64, 96, 80
        token_block_size = 4
        channel_block_size = 16
        base_x = torch.randn(
            batch,
            tokens,
            hidden,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        corrected_x = base_x + 0.05 * torch.randn(
            base_x.shape,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        gate_weight = torch.randn(
            channels,
            hidden,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        ) / hidden**0.5
        up_weight = torch.randn(
            channels,
            hidden,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        ) / hidden**0.5
        down_weight = torch.randn(
            outputs,
            channels,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        ) / channels**0.5
        gate_bias = torch.randn(
            channels,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        up_bias = torch.randn(
            channels,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        mask = torch.zeros(
            batch,
            tokens,
            channels,
            device="cuda",
            dtype=torch.bool,
        )
        support = (
            ((0, 2), (1, 4, 5), (0,), (2, 3)),
            ((1,), (0, 2), (3, 4, 5), (2,)),
        )
        for batch_index, token_blocks in enumerate(support):
            for token_block, channel_blocks in enumerate(token_blocks):
                token_start = token_block * token_block_size
                token_end = min(token_start + token_block_size, tokens)
                for channel_block in channel_blocks:
                    channel_start = channel_block * channel_block_size
                    channel_end = min(
                        channel_start + channel_block_size,
                        channels,
                    )
                    mask[
                        batch_index,
                        token_start:token_end,
                        channel_start:channel_end,
                    ] = True

        block_index = ffn_block_index_from_mask(
            mask,
            channel_block_size=channel_block_size,
            token_block_size=token_block_size,
        )
        corrected_gate = F.silu(
            F.linear(corrected_x, gate_weight, gate_bias)
        )
        actual = block_sparse_ffn_delta_triton(
            base_x,
            corrected_x,
            corrected_gate,
            gate_weight,
            up_weight,
            down_weight,
            block_index,
            gate_bias=gate_bias,
            up_bias=up_bias,
            token_block_size=token_block_size,
            channel_block_size=channel_block_size,
        )
        base_gate = F.silu(F.linear(base_x, gate_weight, gate_bias))
        base_up = F.linear(base_x, up_weight, up_bias)
        corrected_up = F.linear(corrected_x, up_weight, up_bias)
        expected = F.linear(
            (
                corrected_gate * corrected_up - base_gate * base_up
            ).masked_fill(~mask, 0),
            down_weight,
        )

        torch.testing.assert_close(
            actual.float(),
            expected.float(),
            rtol=1e-2,
            atol=2e-2,
        )


if __name__ == "__main__":
    unittest.main()
