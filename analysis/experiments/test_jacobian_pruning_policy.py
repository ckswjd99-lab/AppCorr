#!/usr/bin/env python3
"""Tests for the lightweight Jacobian support policy allocator."""

from __future__ import annotations

import unittest

import numpy as np

from analysis.experiments.fit_jacobian_pruning_policy import (
    COMPONENTS,
    allocate,
    decreasing_isotonic,
)


class JacobianPruningPolicyTest(unittest.TestCase):
    def test_decreasing_isotonic_pools_violations(self) -> None:
        fitted = decreasing_isotonic(np.asarray([3.0, 1.0, 2.0, 0.0]))
        np.testing.assert_allclose(fitted, [3.0, 1.5, 1.5, 0.0])
        self.assertTrue(np.all(np.diff(fitted) <= 0))

    def test_allocator_prunes_least_sensitive_component_first(self) -> None:
        grid = np.asarray([0.0, 0.5, 1.0])
        component_curves = {
            "input_token": np.asarray([1.0, 0.5, 0.0]),
            "attention_edge": np.asarray([0.1, 0.05, 0.0]),
            "ffn_channel": np.asarray([0.5, 0.25, 0.0]),
        }
        curves = {
            (layer, component): component_curves[component]
            for layer in range(40)
            for component in COMPONENTS
        }
        policy = allocate(
            target_pruning=1 / 3,
            grid=grid,
            curves=curves,
            component_costs={component: 1.0 for component in COMPONENTS},
        )
        self.assertAlmostEqual(policy["achieved_pruning_rate"], 1 / 3)
        self.assertAlmostEqual(
            policy["component_summary"]["attention_edge"]["mean_pruning"],
            1.0,
        )
        self.assertAlmostEqual(
            policy["component_summary"]["input_token"]["mean_pruning"],
            0.0,
        )
        self.assertAlmostEqual(
            policy["component_summary"]["ffn_channel"]["mean_pruning"],
            0.0,
        )
        self.assertLess(
            policy["optimized_predicted_rms"],
            policy["uniform_predicted_rms"],
        )


if __name__ == "__main__":
    unittest.main()
