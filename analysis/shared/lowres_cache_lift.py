from __future__ import annotations

from typing import Any, Dict, Iterable

import torch
import torch.nn.functional as F

from appcorr.models.dinov3.layers.attention import rope_apply


def lift_token_grid(
    tensor: torch.Tensor,
    low_hw: tuple[int, int],
    high_hw: tuple[int, int],
    num_prefix_tokens: int,
    *,
    mode: str = "bilinear",
) -> torch.Tensor:
    """Interpolate the patch-token axis while preserving non-spatial prefix tokens."""
    if tensor.ndim < 2:
        raise ValueError(f"Expected [B, N, ...] tensor, got {tuple(tensor.shape)}")
    low_h, low_w = (int(value) for value in low_hw)
    high_h, high_w = (int(value) for value in high_hw)
    expected_tokens = num_prefix_tokens + low_h * low_w
    if tensor.shape[1] != expected_tokens:
        raise ValueError(
            f"Token count {tensor.shape[1]} does not match prefix={num_prefix_tokens} "
            f"plus low grid {low_h}x{low_w}"
        )

    prefix = tensor[:, :num_prefix_tokens]
    patch = tensor[:, num_prefix_tokens:]
    trailing_shape = patch.shape[2:]
    patch = patch.reshape(tensor.shape[0], low_h, low_w, -1).permute(0, 3, 1, 2)
    original_dtype = patch.dtype
    interpolate_dtype = (
        torch.float32
        if patch.device.type == "cpu" and patch.dtype in {torch.float16, torch.bfloat16}
        else patch.dtype
    )
    interpolate_kwargs: Dict[str, Any] = {}
    if mode in {"linear", "bilinear", "bicubic", "trilinear"}:
        interpolate_kwargs["align_corners"] = False
    lifted = F.interpolate(
        patch.to(dtype=interpolate_dtype),
        size=(high_h, high_w),
        mode=mode,
        **interpolate_kwargs,
    ).to(dtype=original_dtype)
    lifted = lifted.permute(0, 2, 3, 1).reshape(
        tensor.shape[0],
        high_h * high_w,
        *trailing_shape,
    )
    return torch.cat((prefix, lifted), dim=1)


def apply_rope_to_cached_key(
    pre_rope_key: torch.Tensor,
    rope: tuple[torch.Tensor, torch.Tensor] | None,
    num_prefix_tokens: int,
) -> torch.Tensor:
    """Apply the target-grid RoPE to a cached [B, N, H, Dh] key."""
    if rope is None:
        return pre_rope_key
    sin, cos = rope
    patch_key = pre_rope_key[:, num_prefix_tokens:].transpose(1, 2)
    key_dtype = patch_key.dtype
    patch_key = rope_apply(
        patch_key.to(dtype=sin.dtype),
        sin,
        cos,
    ).to(dtype=key_dtype)
    return torch.cat(
        (
            pre_rope_key[:, :num_prefix_tokens],
            patch_key.transpose(1, 2),
        ),
        dim=1,
    )


def lift_partial_token_cache(
    low_cache: Dict[str, Any],
    layer_tags: Iterable[str],
    low_hw: tuple[int, int],
    high_hw: tuple[int, int],
    num_prefix_tokens: int,
    high_rope: tuple[torch.Tensor, torch.Tensor] | None,
) -> Dict[str, Any]:
    """Lift the numerical cache required by partial-token correction."""
    high_cache: Dict[str, Any] = {}
    for tag in layer_tags:
        kv_key = f"{tag}_kv"
        pre_rope_key = f"{tag}_pre_rope_k"
        residual_key = f"{tag}_blocks_out_sum"
        pscore_key = f"{tag}_server_pscore"
        missing = [
            key
            for key in (kv_key, pre_rope_key, residual_key, pscore_key)
            if key not in low_cache
        ]
        if missing:
            raise KeyError(f"Missing low-resolution cache entries for {tag}: {missing}")

        lifted_pre_k = lift_token_grid(
            low_cache[pre_rope_key],
            low_hw,
            high_hw,
            num_prefix_tokens,
        )
        lifted_k = apply_rope_to_cached_key(
            lifted_pre_k,
            high_rope,
            num_prefix_tokens,
        )
        lifted_v = lift_token_grid(
            low_cache[kv_key][:, :, 1],
            low_hw,
            high_hw,
            num_prefix_tokens,
        )
        high_cache[kv_key] = torch.stack((lifted_k, lifted_v), dim=2)
        high_cache[residual_key] = lift_token_grid(
            low_cache[residual_key],
            low_hw,
            high_hw,
            num_prefix_tokens,
        )
        high_cache[pscore_key] = lift_token_grid(
            low_cache[pscore_key].unsqueeze(-1),
            low_hw,
            high_hw,
            num_prefix_tokens,
        ).squeeze(-1)
    return high_cache


def tensor_cache_nbytes(cache: Dict[str, Any]) -> int:
    return sum(
        value.numel() * value.element_size()
        for value in cache.values()
        if torch.is_tensor(value)
    )
