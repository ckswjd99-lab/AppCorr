import torch

from analysis.shared.lowres_cache_lift import (
    apply_rope_to_cached_key,
    lift_partial_token_cache,
    lift_token_grid,
)


def test_lift_token_grid_preserves_prefix_and_shape():
    prefix = torch.tensor([[[10.0], [20.0]]])
    patch = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1)
    lifted = lift_token_grid(
        torch.cat((prefix, patch), dim=1),
        low_hw=(2, 2),
        high_hw=(4, 4),
        num_prefix_tokens=2,
    )
    assert lifted.shape == (1, 18, 1)
    torch.testing.assert_close(lifted[:, :2], prefix)


def test_apply_rope_to_cached_key_keeps_prefix():
    key = torch.randn(1, 6, 2, 4)
    sin = torch.zeros(4, 4)
    cos = torch.ones(4, 4)
    result = apply_rope_to_cached_key(key, (sin, cos), num_prefix_tokens=2)
    torch.testing.assert_close(result, key)


def test_lift_partial_token_cache_builds_high_resolution_contract():
    tag = "layer0"
    low_hw = (2, 2)
    high_hw = (4, 4)
    prefix = 2
    n_low = prefix + 4
    cache = {
        f"{tag}_kv": torch.randn(1, n_low, 2, 2, 4),
        f"{tag}_pre_rope_k": torch.randn(1, n_low, 2, 4),
        f"{tag}_blocks_out_sum": torch.randn(1, n_low, 8),
        f"{tag}_server_pscore": torch.randn(1, n_low),
    }
    rope = (torch.zeros(16, 4), torch.ones(16, 4))
    lifted = lift_partial_token_cache(
        cache,
        [tag],
        low_hw,
        high_hw,
        prefix,
        rope,
    )
    n_high = prefix + 16
    assert lifted[f"{tag}_kv"].shape == (1, n_high, 2, 2, 4)
    assert lifted[f"{tag}_blocks_out_sum"].shape == (1, n_high, 8)
    assert lifted[f"{tag}_server_pscore"].shape == (1, n_high)
