import pytest
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from analysis.experiments.ade20k_qkv_low_rank_eval import (
    BASE_CANVAS_MODE,
    ffn_cache_bytes,
    run_base,
    run_full,
    run_low_rank_delta_correction,
    run_qkv_delta_correction,
    scaled_native_pyramid_level,
)
from analysis.experiments.ade20k_low_rank_approx_token_eval import (
    PROJECTIONS,
    projection_module,
    run_approx,
    run_token_correction,
    token_support,
)
from appcorr.models.dinov3.layers.block import SelfAttentionBlock
from appcorr.models.dinov3.layers.low_rank import (
    dense_linear_flop_ratio,
    factorize_linear_weight,
    factorize_linear_weight_activation_aware,
)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_full_rank_delta_matches_dense_linear(dtype: torch.dtype) -> None:
    torch.manual_seed(11)
    weight = torch.randn(9, 7, dtype=dtype)
    delta_x = torch.randn(3, 5, 7, dtype=dtype)

    factors = factorize_linear_weight(
        weight,
        max_rank=7,
        factor_dtype=dtype,
        exact=True,
    )

    expected = F.linear(delta_x, weight, bias=None)
    actual = factors.apply(delta_x)
    tolerance = 2e-5 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(
        actual,
        expected,
        rtol=tolerance,
        atol=tolerance,
    )


def test_nested_rank_uses_requested_prefix() -> None:
    torch.manual_seed(17)
    weight = torch.randn(12, 8)
    delta_x = torch.randn(4, 8)
    factors = factorize_linear_weight(weight, max_rank=6, exact=True)

    rank = 3
    expected = F.linear(
        F.linear(delta_x, factors.right[:rank]),
        factors.left[:, :rank],
    )
    torch.testing.assert_close(factors.apply(delta_x, rank), expected)
    assert factors.factor_bytes(rank) < factors.factor_bytes()
    assert factors.spectral_energy_fraction(rank) <= factors.spectral_energy_fraction()


def test_low_rank_recovers_known_rank_matrix() -> None:
    torch.manual_seed(23)
    left = torch.randn(18, 4)
    right = torch.randn(4, 13)
    weight = left @ right
    delta_x = torch.randn(2, 7, 13)

    factors = factorize_linear_weight(
        weight,
        max_rank=4,
        oversample=4,
        power_iterations=2,
    )

    torch.testing.assert_close(
        factors.apply(delta_x),
        F.linear(delta_x, weight),
        rtol=2e-4,
        atol=2e-4,
    )


def test_activation_aware_full_rank_recovers_dense_linear() -> None:
    torch.manual_seed(29)
    weight = torch.randn(11, 7, dtype=torch.float64)
    input_rms = torch.rand(7, dtype=torch.float64) + 0.1
    delta_x = torch.randn(3, 5, 7, dtype=torch.float64)

    factors = factorize_linear_weight_activation_aware(
        weight,
        input_rms,
        max_rank=7,
        factor_dtype=torch.float64,
        exact=True,
    )

    torch.testing.assert_close(
        factors.apply(delta_x),
        F.linear(delta_x, weight),
        rtol=1e-6,
        atol=1e-6,
    )


def test_activation_aware_svd_reduces_anisotropic_delta_error() -> None:
    torch.manual_seed(30)
    weight = torch.randn(16, 8)
    input_rms = torch.tensor([20.0, 12.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    delta_x = torch.randn(4096, 8) * input_rms
    expected = F.linear(delta_x, weight)

    weight_factors = factorize_linear_weight(
        weight,
        max_rank=2,
        exact=True,
    )
    aware_factors = factorize_linear_weight_activation_aware(
        weight,
        input_rms,
        max_rank=2,
        exact=True,
    )
    weight_error = F.mse_loss(weight_factors.apply(delta_x), expected)
    aware_error = F.mse_loss(aware_factors.apply(delta_x), expected)

    assert aware_error < weight_error * 0.2


def test_qkv_rank_512_flop_ratio() -> None:
    assert dense_linear_flop_ratio(4096, 12288, 512) == pytest.approx(1 / 6)


def test_base_canvas_builds_native_l2_before_model_scaling() -> None:
    rng = np.random.default_rng(31)
    original = rng.integers(0, 256, size=(65, 97, 3), dtype=np.uint8)
    target_hw = (128, 192)

    actual = scaled_native_pyramid_level(original, level=2, target_hw=target_hw)
    native_l2 = cv2.pyrDown(cv2.pyrDown(original))
    expected = np.asarray(
        Image.fromarray(native_l2).resize(
            (target_hw[1], target_hw[0]),
            Image.Resampling.BILINEAR,
        ),
        dtype=np.uint8,
    )

    np.testing.assert_array_equal(actual, expected)
    assert native_l2.shape[:2] == (17, 25)
    assert BASE_CANVAS_MODE == "native_pyramid_then_scale"


def test_dense_qkv_delta_recovers_direct_transformer() -> None:
    torch.manual_seed(37)
    blocks = torch.nn.ModuleList(
        [
            SelfAttentionBlock(
                dim=32,
                num_heads=4,
                ffn_ratio=2.0,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                init_values=0.2,
            ).eval()
            for _ in range(5)
        ]
    )
    for block in blocks:
        for module in block.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    torch.nn.init.normal_(module.bias, std=0.02)
        torch.nn.init.constant_(block.ls1.gamma, 0.2)
        torch.nn.init.constant_(block.ls2.gamma, 0.2)
    backbone = torch.nn.Module()
    backbone.blocks = blocks
    base = torch.randn(1, 21, 32)
    full = base + 0.1 * torch.randn_like(base)

    base_states, base_qkv, _, _ = run_base(backbone, base, rope=None)
    expected, _ = run_full(backbone, full, rope=None)
    actual, _ = run_qkv_delta_correction(
        backbone,
        full,
        base_states,
        base_qkv,
        rope=None,
        factors={},
        rank=None,
    )

    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)


def test_full_rank_ffn_delta_recovers_direct_transformer() -> None:
    torch.manual_seed(41)
    blocks = torch.nn.ModuleList(
        [
            SelfAttentionBlock(
                dim=32,
                num_heads=4,
                ffn_ratio=2.0,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                init_values=0.2,
            ).eval()
            for _ in range(5)
        ]
    )
    for block in blocks:
        for module in block.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    torch.nn.init.normal_(module.bias, std=0.02)
        torch.nn.init.constant_(block.ls1.gamma, 0.2)
        torch.nn.init.constant_(block.ls2.gamma, 0.2)
    backbone = torch.nn.Module()
    backbone.blocks = blocks
    base = torch.randn(1, 21, 32)
    full = base + 0.1 * torch.randn_like(base)
    base_states, base_qkv, base_ffn, _ = run_base(
        backbone,
        base,
        rope=None,
    )
    expected, _ = run_full(backbone, full, rope=None)
    middle_end = len(blocks) - 2
    gate_factors = {}
    up_factors = {}
    down_factors = {}
    for layer_index in range(2, middle_end):
        block = blocks[layer_index]
        gate_factors[layer_index] = factorize_linear_weight(
            block.mlp.w1.weight,
            max_rank=32,
            exact=True,
        )
        up_factors[layer_index] = factorize_linear_weight(
            block.mlp.w2.weight,
            max_rank=32,
            exact=True,
        )
        down_factors[layer_index] = factorize_linear_weight(
            block.mlp.w3.weight,
            max_rank=32,
            exact=True,
        )

    actual, _ = run_low_rank_delta_correction(
        backbone,
        full,
        base_states,
        base_qkv,
        base_ffn,
        rope=None,
        qkv_factors={},
        ffn_gate_factors=gate_factors,
        ffn_up_factors=up_factors,
        ffn_down_factors=down_factors,
        qkv_rank=None,
        ffn_rank=32,
    )

    torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)
    assert ffn_cache_bytes(base_ffn) > 0


def test_full_rank_approx_and_keep_all_correction_match_dense() -> None:
    torch.manual_seed(43)
    blocks = torch.nn.ModuleList(
        [
            SelfAttentionBlock(
                dim=32,
                num_heads=4,
                ffn_ratio=2.0,
                qkv_bias=True,
                proj_bias=True,
                ffn_bias=True,
                init_values=0.2,
            ).eval()
            for _ in range(5)
        ]
    )
    for block in blocks:
        for module in block.modules():
            if isinstance(module, torch.nn.Linear):
                torch.nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    torch.nn.init.normal_(module.bias, std=0.02)
        torch.nn.init.constant_(block.ls1.gamma, 0.2)
        torch.nn.init.constant_(block.ls2.gamma, 0.2)
    backbone = torch.nn.Module()
    backbone.blocks = blocks
    base = torch.randn(1, 21, 32)
    full = base + 0.1 * torch.randn_like(base)
    factors = {projection: {} for projection in PROJECTIONS}
    for layer_index, block in enumerate(blocks):
        for projection in PROJECTIONS:
            module = projection_module(block, projection)
            factors[projection][layer_index] = factorize_linear_weight(
                module.weight,
                max_rank=min(module.weight.shape),
                exact=True,
            )

    approx = run_approx(
        backbone,
        base,
        rope=None,
        factors=factors,
        targets=frozenset(PROJECTIONS),
        rank=32,
        collect_score=False,
    )
    dense_base, _ = run_full(backbone, base, rope=None)
    torch.testing.assert_close(
        approx.states[-1],
        dense_base,
        rtol=5e-5,
        atol=5e-5,
    )

    active = torch.arange(full.shape[1]).unsqueeze(0)
    corrected, _ = run_token_correction(
        backbone,
        full,
        approx,
        active,
        rope=None,
    )
    dense_full, _ = run_full(backbone, full, rope=None)
    torch.testing.assert_close(corrected, dense_full, rtol=5e-5, atol=5e-5)


def test_token_support_keeps_specials_and_half_patches() -> None:
    full = np.full((32, 32, 3), 255, dtype=np.uint8)
    base = np.zeros_like(full)
    attention = torch.arange(9, dtype=torch.float32).unsqueeze(0)

    selected = token_support(full, base, attention, token_rate=0.5)

    assert selected.shape == (1, 7)
    torch.testing.assert_close(selected[0, :5], torch.arange(5))
    assert torch.all(selected[0, 5:] >= 5)
