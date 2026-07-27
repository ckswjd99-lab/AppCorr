#!/usr/bin/env python3
"""Configuration contract tests for the jacobian_support runtime mode."""

from __future__ import annotations

import unittest

from offload.common.protocol import normalize_appcorr_kwargs


class JacobianSupportConfigTest(unittest.TestCase):
    def test_defaults_do_not_change_existing_method(self) -> None:
        options = normalize_appcorr_kwargs({"method": "partial_token"})
        self.assertEqual(options["method"], "partial_token")
        self.assertEqual(options["attn_delta_backend"], "auto")

    def test_jacobian_options_are_normalized(self) -> None:
        options = normalize_appcorr_kwargs({
            "method": "jacobian_support",
            "attn_delta_backend": "product_delta",
            "attn_probability_mode": "exact",
            "attn_support_mode": "tail_mass",
            "attn_tail_epsilon": 0.05,
            "attn_key_block_size": 32,
            "ffn_predictor": "derivative_bound",
            "ffn_support_keep_ratio": 0.25,
        })
        self.assertEqual(options["method"], "jacobian_support")
        self.assertEqual(options["attn_delta_backend"], "product_delta")
        self.assertEqual(options["attn_probability_mode"], "exact")
        self.assertEqual(options["attn_key_block_size"], 32)
        self.assertEqual(options["ffn_support_keep_ratio"], 0.25)

    def test_invalid_backend_fails_early(self) -> None:
        with self.assertRaisesRegex(ValueError, "attn_delta_backend"):
            normalize_appcorr_kwargs({
                "method": "jacobian_support",
                "attn_delta_backend": "csr",
            })


if __name__ == "__main__":
    unittest.main()
