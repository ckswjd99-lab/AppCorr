#!/usr/bin/env python3
"""Numerical contract tests for draft-guided Jacobian support.

The tests deliberately use float64 on CPU so failures diagnose the reference
math rather than BF16 or kernel accumulation.  Run directly or through pytest:

    python analysis/experiments/test_jacobian_support_math.py
"""

from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from appcorr.models.dinov3.layers.jacobian_support import (
    attention_delta,
    attention_edge_energy,
    exact_attention_delta,
    exact_swiglu_delta,
    select_attention_block_support,
    select_ffn_block_support,
    softmax_jvp,
    swiglu_jvp,
)


class AttentionDeltaTest(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(7)
        self.q = torch.randn(2, 4, 7, 8, generator=generator, dtype=torch.float64)
        self.k = torch.randn(2, 4, 9, 8, generator=generator, dtype=torch.float64)
        self.v = torch.randn(2, 4, 9, 6, generator=generator, dtype=torch.float64)
        self.dq = 0.02 * torch.randn(
            self.q.shape, generator=generator, dtype=torch.float64
        )
        self.dk = 0.02 * torch.randn(
            self.k.shape, generator=generator, dtype=torch.float64
        )
        self.dv = 0.02 * torch.randn(
            self.v.shape, generator=generator, dtype=torch.float64
        )

    def test_softmax_jvp_matches_autograd(self) -> None:
        logits = torch.matmul(self.q, self.k.transpose(-2, -1)) / self.q.shape[-1] ** 0.5
        dlogits = (
            torch.matmul(self.dq, self.k.transpose(-2, -1))
            + torch.matmul(self.q, self.dk.transpose(-2, -1))
        ) / self.q.shape[-1] ** 0.5
        _, autograd_jvp = torch.func.jvp(
            lambda value: torch.softmax(value, dim=-1),
            (logits,),
            (dlogits,),
        )
        actual = softmax_jvp(torch.softmax(logits, dim=-1), dlogits)
        torch.testing.assert_close(actual, autograd_jvp, rtol=1e-12, atol=1e-12)

    def test_product_is_split_plus_cross_term(self) -> None:
        split = attention_delta(
            self.q, self.k, self.v, self.dq, self.dk, self.dv, backend="split_jvp"
        )
        product = attention_delta(
            self.q,
            self.k,
            self.v,
            self.dq,
            self.dk,
            self.dv,
            backend="product_delta",
        )
        torch.testing.assert_close(
            product.delta,
            split.delta + split.cross_term,
            rtol=1e-12,
            atol=1e-12,
        )

    def test_memory_efficient_edge_energy_matches_explicit_vectors(self) -> None:
        result = attention_delta(
            self.q, self.k, self.v, self.dq, self.dk, self.dv
        )
        for backend in ("split_jvp", "product_delta"):
            actual = attention_edge_energy(
                result.base_probability,
                result.delta_probability,
                self.v,
                self.dv,
                backend=backend,
            )
            explicit = (
                result.base_probability.unsqueeze(-1) * self.dv.unsqueeze(-3)
                + result.delta_probability.unsqueeze(-1) * self.v.unsqueeze(-3)
            )
            if backend == "product_delta":
                explicit += (
                    result.delta_probability.unsqueeze(-1)
                    * self.dv.unsqueeze(-3)
                )
            expected = explicit.square().sum(dim=-1).float()
            torch.testing.assert_close(actual, expected, rtol=5e-3, atol=1e-6)

    def test_exact_probability_product_is_exact_attention_delta(self) -> None:
        product = attention_delta(
            self.q,
            self.k,
            self.v,
            self.dq,
            self.dk,
            self.dv,
            backend="product_delta",
            probability_mode="exact",
        )
        exact = exact_attention_delta(
            self.q, self.k, self.v, self.dq, self.dk, self.dv
        )
        torch.testing.assert_close(product.delta, exact, rtol=1e-12, atol=1e-12)

    def test_sparse_support_does_not_renormalize(self) -> None:
        full = attention_delta(
            self.q, self.k, self.v, self.dq, self.dk, self.dv
        )
        support = torch.zeros_like(full.base_probability, dtype=torch.bool)
        support[..., :3] = True
        sparse = attention_delta(
            self.q,
            self.k,
            self.v,
            self.dq,
            self.dk,
            self.dv,
            backend="product_delta",
            support_mask=support,
        )
        selected_mass = (sparse.base_probability * support).sum(dim=-1)
        self.assertTrue(torch.all(selected_mass < 1))
        torch.testing.assert_close(
            sparse.base_probability,
            full.base_probability,
            rtol=0,
            atol=0,
        )

    def test_block_support_is_shared_and_forces_residual_keys(self) -> None:
        base = attention_delta(
            self.q, self.k, self.v, self.dq, self.dk, self.dv
        ).base_probability
        residual_keys = torch.zeros(2, 9, dtype=torch.bool)
        residual_keys[:, -1] = True
        support, stats = select_attention_block_support(
            base,
            keep_ratio=0.25,
            key_block_size=3,
            query_block_size=4,
            head_group_size=2,
            residual_key_mask=residual_keys,
        )
        self.assertTrue(support[..., -1].all())
        torch.testing.assert_close(support[:, 0], support[:, 1])
        torch.testing.assert_close(support[:, 2], support[:, 3])
        torch.testing.assert_close(support[:, :, 0], support[:, :, 3])
        self.assertGreater(stats.probability_mass, 0)
        self.assertLessEqual(stats.kept_fraction, 1)


class SwiGLUTest(unittest.TestCase):
    def setUp(self) -> None:
        generator = torch.Generator().manual_seed(11)
        self.x = torch.randn(2, 7, 5, generator=generator, dtype=torch.float64)
        self.dx = torch.randn(2, 7, 5, generator=generator, dtype=torch.float64)
        self.w_gate = torch.randn(11, 5, generator=generator, dtype=torch.float64)
        self.w_up = torch.randn(11, 5, generator=generator, dtype=torch.float64)
        self.w_down = torch.randn(5, 11, generator=generator, dtype=torch.float64)

    def _ffn(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.linear(x, self.w_gate)
        up = F.linear(x, self.w_up)
        return F.linear(F.silu(gate) * up, self.w_down)

    def test_swiglu_jvp_matches_autograd(self) -> None:
        _, expected = torch.func.jvp(self._ffn, (self.x,), (self.dx,))
        actual = swiglu_jvp(
            self.x, self.dx, self.w_gate, self.w_up, self.w_down
        )
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)

    def test_exact_swiglu_delta(self) -> None:
        actual = exact_swiglu_delta(
            self.x, self.dx, self.w_gate, self.w_up, self.w_down
        )
        torch.testing.assert_close(
            actual,
            self._ffn(self.x + self.dx) - self._ffn(self.x),
            rtol=1e-12,
            atol=1e-12,
        )

    def test_full_channel_support_matches_dense(self) -> None:
        mask = select_ffn_block_support(
            torch.rand(2, 7, 11, dtype=torch.float64),
            keep_ratio=1.0,
            channel_block_size=4,
            token_block_size=3,
        )
        self.assertTrue(mask.all())
        dense = swiglu_jvp(
            self.x, self.dx, self.w_gate, self.w_up, self.w_down
        )
        selected = swiglu_jvp(
            self.x,
            self.dx,
            self.w_gate,
            self.w_up,
            self.w_down,
            channel_mask=mask,
        )
        torch.testing.assert_close(selected, dense, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
