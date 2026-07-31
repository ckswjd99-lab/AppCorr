#!/usr/bin/env python3
"""CUDA parity tests for the regular packed attention-delta kernel."""

from __future__ import annotations

import unittest

from analysis.shared.cuda_environment import configure_triton_cuda_environment

configure_triton_cuda_environment()

import torch

from appcorr.models.dinov3.layers.triton_kernels.jacobian_attention import (
    packed_attention_delta_triton,
)
from appcorr.models.dinov3.layers.triton_kernels.jacobian_attention_block import (
    block_product_delta_triton,
)
from appcorr.models.dinov3.layers.triton_kernels.jacobian_probability import (
    selected_softmax_jvp_triton,
)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class PackedAttentionDeltaKernelTest(unittest.TestCase):
    def _run_case(
        self,
        shape: tuple[int, int, int, int, int],
        dtype: torch.dtype,
    ) -> None:
        batch, heads, queries, keys, head_dim = shape
        generator = torch.Generator(device="cuda").manual_seed(17)
        probability = torch.rand(
            batch,
            heads,
            queries,
            keys,
            generator=generator,
            device="cuda",
            dtype=dtype,
        )
        probability /= probability.sum(dim=-1, keepdim=True)
        delta_probability = 0.01 * torch.randn(
            probability.shape,
            generator=generator,
            device="cuda",
            dtype=dtype,
        )
        value = torch.randn(
            batch,
            heads,
            keys,
            head_dim,
            generator=generator,
            device="cuda",
            dtype=dtype,
        )
        delta_value = 0.01 * torch.randn(
            value.shape,
            generator=generator,
            device="cuda",
            dtype=dtype,
        )

        for backend in ("split_jvp", "product_delta"):
            cached_base = (
                probability @ value if backend == "product_delta" else None
            )
            outputs = [
                packed_attention_delta_triton(
                    probability,
                    delta_probability,
                    value,
                    delta_value,
                    backend=backend,
                )
            ]
            if cached_base is not None:
                outputs.append(
                    packed_attention_delta_triton(
                        probability,
                        delta_probability,
                        value,
                        delta_value,
                        backend=backend,
                        base_support_output=cached_base,
                    )
                )
            if backend == "split_jvp":
                reference = (
                    probability.float() @ delta_value.float()
                    + delta_probability.float() @ value.float()
                )
            else:
                reference = (
                    (probability.float() + delta_probability.float())
                    @ (value.float() + delta_value.float())
                    - probability.float() @ value.float()
                )
            tolerance = 2e-5 if dtype == torch.float32 else 1e-2
            for actual in outputs:
                torch.testing.assert_close(
                    actual.float(),
                    reference,
                    rtol=tolerance,
                    atol=tolerance,
                )

    def test_fp32_irregular_shape(self) -> None:
        self._run_case((2, 4, 7, 48, 37), torch.float32)

    def test_bf16_vit7b_head_shape(self) -> None:
        self._run_case((1, 32, 8, 64, 128), torch.bfloat16)

    def test_empty_support(self) -> None:
        probability = torch.empty(1, 2, 3, 0, device="cuda")
        value = torch.empty(1, 2, 0, 8, device="cuda")
        output = packed_attention_delta_triton(
            probability,
            probability,
            value,
            value,
            backend="product_delta",
        )
        self.assertEqual(output.shape, (1, 2, 3, 8))
        self.assertEqual(output.count_nonzero().item(), 0)

    def test_indirect_key_descriptor_matches_gathered_values(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(29)
        probability = torch.rand(
            2, 4, 7, 32, generator=generator, device="cuda", dtype=torch.bfloat16
        )
        probability /= probability.sum(dim=-1, keepdim=True)
        delta_probability = 0.01 * torch.randn(
            probability.shape,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        value = torch.randn(
            2, 4, 73, 128, generator=generator, device="cuda", dtype=torch.bfloat16
        )
        delta_value = 0.01 * torch.randn(
            value.shape,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        key_index = torch.stack([
            torch.randperm(73, device="cuda")[:32],
            torch.randperm(73, device="cuda")[:32],
        ])
        gather_index = key_index[:, None, :, None].expand(-1, 4, -1, 128)
        selected_value = value.gather(2, gather_index)
        selected_delta_value = delta_value.gather(2, gather_index)
        cached_base = probability @ selected_value
        actual = packed_attention_delta_triton(
            probability,
            delta_probability,
            value,
            delta_value,
            backend="product_delta",
            base_support_output=cached_base,
            key_index=key_index,
        )
        reference = (
            (probability.float() + delta_probability.float())
            @ (selected_value.float() + selected_delta_value.float())
            - probability.float() @ selected_value.float()
        )
        torch.testing.assert_close(actual.float(), reference, rtol=1e-2, atol=1e-2)

    def test_selected_softmax_jvp_preserves_full_probability_denominator(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(37)
        batch, heads, queries, tokens, selected, head_dim = 2, 4, 19, 73, 48, 32
        query = torch.randn(
            batch, heads, queries, head_dim,
            generator=generator, device="cuda", dtype=torch.bfloat16,
        )
        delta_query = 0.01 * torch.randn(
            query.shape, generator=generator, device="cuda", dtype=torch.bfloat16
        )
        key = torch.randn(
            batch, heads, tokens, head_dim,
            generator=generator, device="cuda", dtype=torch.bfloat16,
        )
        delta_key = 0.01 * torch.randn(
            key.shape, generator=generator, device="cuda", dtype=torch.bfloat16
        )
        key_index = torch.stack([
            torch.randperm(tokens, device="cuda")[:selected]
            for _ in range(batch)
        ])
        gather_index = key_index[:, None, :, None].expand(
            -1, heads, -1, head_dim
        )
        selected_key = key.gather(2, gather_index)
        selected_delta_key = delta_key.gather(2, gather_index)
        scale = head_dim**-0.5
        # These are selected entries of a full-softmax probability.  Their row
        # mass is deliberately below one; the kernel must not renormalize it.
        full_logits = query.float() @ key.float().transpose(-2, -1) * scale
        full_probability = full_logits.softmax(dim=-1)
        gather_probability = key_index[:, None, None, :].expand(
            -1, heads, queries, -1
        )
        probability = full_probability.gather(3, gather_probability).to(torch.bfloat16)
        delta_logits = (
            delta_query.float() @ selected_key.float().transpose(-2, -1)
            + query.float() @ selected_delta_key.float().transpose(-2, -1)
        ) * scale
        expected = probability.float() * (
            delta_logits
            - (probability.float() * delta_logits).sum(dim=-1, keepdim=True)
        )
        actual = selected_softmax_jvp_triton(
            probability,
            query,
            delta_query,
            key,
            delta_key,
            key_index,
            scale=scale,
        )
        self.assertTrue(torch.all(probability.float().sum(dim=-1) < 1))
        torch.testing.assert_close(
            actual.float(), expected, rtol=1e-2, atol=1e-3
        )

    def test_query_head_block_product_descriptor(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(41)
        batch, heads, queries, keys, head_dim = 1, 4, 7, 19, 16
        base_probability = torch.rand(
            batch,
            heads,
            queries,
            keys,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        base_probability /= base_probability.sum(dim=-1, keepdim=True)
        corrected_probability = torch.rand(
            base_probability.shape,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        corrected_probability /= corrected_probability.sum(
            dim=-1,
            keepdim=True,
        )
        base_value = torch.randn(
            batch,
            heads,
            keys,
            head_dim,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        corrected_value = torch.randn(
            base_value.shape,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        block_index = torch.tensor(
            [[[[0, 3], [1, 4]], [[2, 4], [0, 2]]]],
            device="cuda",
            dtype=torch.int32,
        )
        actual = block_product_delta_triton(
            base_probability,
            corrected_probability,
            base_value,
            corrected_value,
            block_index,
            head_group_size=2,
            query_block_size=4,
            key_block_size=4,
        )
        reference = torch.zeros_like(actual, dtype=torch.float32)
        for head in range(heads):
            for query in range(queries):
                descriptor = block_index[0, head // 2, query // 4]
                selected = torch.cat([
                    torch.arange(
                        int(block) * 4,
                        min((int(block) + 1) * 4, keys),
                        device="cuda",
                    )
                    for block in descriptor
                ])
                reference[0, head, query] = (
                    corrected_probability[0, head, query, selected].float()
                    @ corrected_value[0, head, selected].float()
                    - base_probability[0, head, query, selected].float()
                    @ base_value[0, head, selected].float()
                )
        torch.testing.assert_close(
            actual.float(),
            reference,
            rtol=1e-2,
            atol=1e-2,
        )

    def test_query_head_block_product_ragged_descriptor(self) -> None:
        generator = torch.Generator(device="cuda").manual_seed(43)
        batch, heads, queries, keys, head_dim = 1, 4, 7, 19, 16
        base_probability = torch.rand(
            batch,
            heads,
            queries,
            keys,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        base_probability /= base_probability.sum(dim=-1, keepdim=True)
        corrected_probability = torch.rand(
            base_probability.shape,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        corrected_probability /= corrected_probability.sum(
            dim=-1,
            keepdim=True,
        )
        base_value = torch.randn(
            batch,
            heads,
            keys,
            head_dim,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        corrected_value = torch.randn(
            base_value.shape,
            generator=generator,
            device="cuda",
            dtype=torch.bfloat16,
        )
        block_index = torch.tensor(
            [[[[0, 3, -1], [1, 4, 2]], [[2, -1, -1], [0, 2, -1]]]],
            device="cuda",
            dtype=torch.int32,
        )
        actual = block_product_delta_triton(
            base_probability,
            corrected_probability,
            base_value,
            corrected_value,
            block_index,
            head_group_size=2,
            query_block_size=4,
            key_block_size=4,
        )
        reference = torch.zeros_like(actual, dtype=torch.float32)
        for head in range(heads):
            for query in range(queries):
                descriptor = block_index[0, head // 2, query // 4]
                selected = torch.cat([
                    torch.arange(
                        int(block) * 4,
                        min((int(block) + 1) * 4, keys),
                        device="cuda",
                    )
                    for block in descriptor
                    if int(block) >= 0
                ])
                reference[0, head, query] = (
                    corrected_probability[0, head, query, selected].float()
                    @ corrected_value[0, head, selected].float()
                    - base_probability[0, head, query, selected].float()
                    @ base_value[0, head, selected].float()
                )
        torch.testing.assert_close(
            actual.float(),
            reference,
            rtol=1e-2,
            atol=1e-2,
        )


if __name__ == "__main__":
    unittest.main()
