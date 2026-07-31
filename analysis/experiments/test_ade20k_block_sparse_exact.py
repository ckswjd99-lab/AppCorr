from argparse import Namespace

import cv2
import numpy as np
from PIL import Image
import pytest
import torch

from analysis.experiments.ade20k_block_sparse_exact_eval import (
    BASE_CANVAS_MODE,
    Policy,
    SPECIAL_TOKENS,
    SupportAccumulator,
    corrected_attention,
    corrected_ffn,
    replace_spm_features,
    run_base_block,
    scaled_native_pyramid_level,
)
from appcorr.models.dinov3.layers.block import SelfAttentionBlock
from appcorr.models.dinov3.layers.jacobian_support import (
    attention_block_index_from_mask,
    ffn_block_index_from_mask,
)


def test_all_support_recovers_dense_block() -> None:
    torch.manual_seed(7)
    block = SelfAttentionBlock(
        dim=32,
        num_heads=4,
        ffn_ratio=3.0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        init_values=0.2,
    ).eval()
    for module in block.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.normal_(module.bias, std=0.02)
    torch.nn.init.constant_(block.ls1.gamma, 0.2)
    torch.nn.init.constant_(block.ls2.gamma, 0.2)
    base = torch.randn(1, 21, 32)
    current = base + 0.1 * torch.randn_like(base)
    active = torch.arange(current.shape[1]).unsqueeze(0)
    base_attention, base_next = run_base_block(block, base, rope=None)
    args = Namespace(
        query_block=8,
        query_chunk=16,
        key_block=16,
        head_group=4,
        ffn_channel_block=16,
        ffn_token_block=8,
    )
    support = SupportAccumulator()
    corrected_attention_state = corrected_attention(
        block,
        base,
        base_attention,
        current,
        active,
        None,
        1.0,
        args,
        support,
    )
    weight_score = (
        block.mlp.w2.weight.float().norm(dim=1)
        * block.mlp.w3.weight.float().norm(dim=0)
    )
    corrected = corrected_ffn(
        block,
        base_attention,
        corrected_attention_state,
        base_next,
        active,
        1.0,
        args,
        support,
        weight_score,
    )
    expected = block(current)
    torch.testing.assert_close(corrected, expected, rtol=2e-5, atol=2e-5)


def test_special_token_count_matches_dinov3_contract() -> None:
    assert SPECIAL_TOKENS == 5
    policy = Policy("endpoint", token_rate=1.0, attention_rate=1.0, ffn_rate=1.0)
    assert policy.token_rate == policy.attention_rate == policy.ffn_rate == 1.0


def test_replace_spm_features_preserves_full_vit_context() -> None:
    full_source = {
        "x_backbone": "full tokens",
        "rope_sincos": "full rope",
        "spm_c_cat": "full c",
        "spm_c1_raw": "full c1",
        "spm_c2_len": 2,
        "spm_c3_len": 3,
    }
    base_source = {
        "x_backbone": "base tokens",
        "rope_sincos": "base rope",
        "spm_c_cat": "base c",
        "spm_c1_raw": "base c1",
        "spm_c2_len": 20,
        "spm_c3_len": 30,
    }

    result = replace_spm_features(full_source, base_source)

    assert result["x_backbone"] == "full tokens"
    assert result["rope_sincos"] == "full rope"
    assert result["spm_c_cat"] == "base c"
    assert result["spm_c1_raw"] == "base c1"
    assert result["spm_c2_len"] == 20
    assert result["spm_c3_len"] == 30
    assert full_source["spm_c_cat"] == "full c"


def test_base_canvas_builds_native_pyramid_before_scaling() -> None:
    rng = np.random.default_rng(123)
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
    assert actual.shape == (128, 192, 3)
    assert native_l2.shape[:2] == (17, 25)
    assert BASE_CANVAS_MODE == "native_pyramid_then_scale"


def test_level_zero_uses_the_same_model_scaling_rule() -> None:
    rng = np.random.default_rng(456)
    original = rng.integers(0, 256, size=(63, 101, 3), dtype=np.uint8)
    target_hw = (112, 180)

    actual = scaled_native_pyramid_level(original, level=0, target_hw=target_hw)
    expected = np.asarray(
        Image.fromarray(original).resize(
            (target_hw[1], target_hw[0]),
            Image.Resampling.BILINEAR,
        ),
        dtype=np.uint8,
    )

    np.testing.assert_array_equal(actual, expected)


def test_ragged_attention_descriptor_uses_negative_padding() -> None:
    mask = torch.zeros(1, 4, 4, 12, dtype=torch.bool)
    mask[:, 0:2, 0:2, 0:4] = True
    mask[:, 0:2, 2:4, 4:12] = True
    mask[:, 2:4, 0:2, 0:8] = True
    mask[:, 2:4, 2:4, 8:12] = True

    descriptor = attention_block_index_from_mask(
        mask,
        key_block_size=4,
        query_block_size=2,
        head_group_size=2,
    )

    assert descriptor.tolist() == [
        [[[0, -1], [1, 2]], [[0, 1], [2, -1]]]
    ]


def test_ffn_descriptor_round_trips_structured_mask() -> None:
    mask = torch.zeros(1, 5, 24, dtype=torch.bool)
    mask[:, 0:2, 0:8] = True
    mask[:, 2:4, 8:24] = True
    mask[:, 4:5, 16:24] = True

    descriptor = ffn_block_index_from_mask(
        mask,
        channel_block_size=8,
        token_block_size=2,
    )

    assert descriptor.tolist() == [[[0, -1], [1, 2], [2, -1]]]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@torch.inference_mode()
def test_triton_correction_matches_dense_mask_end_to_end() -> None:
    torch.manual_seed(59)
    block = SelfAttentionBlock(
        dim=64,
        num_heads=4,
        ffn_ratio=2.0,
        qkv_bias=True,
        proj_bias=True,
        ffn_bias=True,
        init_values=0.2,
    ).cuda().bfloat16().eval()
    for module in block.modules():
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                torch.nn.init.normal_(module.bias, std=0.02)
    torch.nn.init.constant_(block.ls1.gamma, 0.2)
    torch.nn.init.constant_(block.ls2.gamma, 0.2)
    base = torch.randn(
        1,
        21,
        64,
        device="cuda",
        dtype=torch.bfloat16,
    )
    current = base + 0.1 * torch.randn_like(base)
    active = torch.tensor(
        [[0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16]],
        device="cuda",
    )
    base_attention, base_next = run_base_block(block, base, rope=None)
    common = {
        "query_block": 4,
        "query_chunk": 8,
        "key_block": 4,
        "head_group": 2,
        "ffn_channel_block": 16,
        "ffn_token_block": 4,
    }
    dense_args = Namespace(**common, correction_backend="dense_mask")
    triton_args = Namespace(**common, correction_backend="triton")

    dense_attention = corrected_attention(
        block,
        base,
        base_attention,
        current,
        active,
        None,
        0.5,
        dense_args,
        SupportAccumulator(),
    )
    triton_attention = corrected_attention(
        block,
        base,
        base_attention,
        current,
        active,
        None,
        0.5,
        triton_args,
        SupportAccumulator(),
    )
    torch.testing.assert_close(
        triton_attention.float(),
        dense_attention.float(),
        rtol=5e-3,
        atol=5e-3,
    )

    weight_score = (
        block.mlp.w2.weight.float().norm(dim=1)
        * block.mlp.w3.weight.float().norm(dim=0)
    )
    dense_next = corrected_ffn(
        block,
        base_attention,
        dense_attention,
        base_next,
        active,
        0.5,
        dense_args,
        SupportAccumulator(),
        weight_score,
    )
    triton_next = corrected_ffn(
        block,
        base_attention,
        triton_attention,
        base_next,
        active,
        0.5,
        triton_args,
        SupportAccumulator(),
        weight_score,
    )
    torch.testing.assert_close(
        triton_next.float(),
        dense_next.float(),
        rtol=5e-3,
        atol=5e-3,
    )
