from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn.functional as F

from appcorr.models.dinov3.layers.attention import PackedQueryState, rope_apply


def lift_token_grid(
    tensor: torch.Tensor,
    low_hw: tuple[int, int],
    high_hw: tuple[int, int],
    num_prefix: int,
) -> torch.Tensor:
    """Nearest-neighbor lift of patch tokens, preserving CLS/register tokens."""
    low_h, low_w = low_hw
    high_h, high_w = high_hw
    if tensor.shape[1] != num_prefix + low_h * low_w:
        raise ValueError(
            f"Low token count {tensor.shape[1]} does not match "
            f"prefix={num_prefix}, grid={low_hw}"
        )
    prefix = tensor[:, :num_prefix]
    patch = tensor[:, num_prefix:]
    trailing = patch.shape[2:]
    patch = patch.reshape(tensor.shape[0], low_h, low_w, -1).permute(0, 3, 1, 2)
    lifted = F.interpolate(patch, size=high_hw, mode="nearest")
    lifted = lifted.permute(0, 2, 3, 1).reshape(
        tensor.shape[0],
        high_h * high_w,
        *trailing,
    )
    return torch.cat((prefix, lifted), dim=1)


def build_parent_token_index(
    low_hw: tuple[int, int],
    high_hw: tuple[int, int],
    num_prefix: int,
    device: torch.device,
) -> torch.Tensor:
    low_h, low_w = low_hw
    high_h, high_w = high_hw
    if high_h % low_h != 0 or high_w % low_w != 0:
        raise ValueError(
            f"High grid {high_hw} must be an integer multiple of low grid {low_hw}"
        )
    scale_h = high_h // low_h
    scale_w = high_w // low_w
    rows = torch.arange(high_h, device=device, dtype=torch.long) // scale_h
    cols = torch.arange(high_w, device=device, dtype=torch.long) // scale_w
    patch_parent = (rows[:, None] * low_w + cols[None, :]).reshape(-1) + num_prefix
    return torch.cat(
        (
            torch.arange(num_prefix, device=device, dtype=torch.long),
            patch_parent,
        )
    )


def _apply_high_rope(
    pre_rope_key: torch.Tensor,
    high_rope,
    num_prefix: int,
) -> torch.Tensor:
    if high_rope is None:
        return pre_rope_key
    sin, cos = high_rope
    patch_key = pre_rope_key[:, num_prefix:].transpose(1, 2)
    key_dtype = patch_key.dtype
    patch_key = rope_apply(
        patch_key.to(dtype=sin.dtype),
        sin,
        cos,
    ).to(dtype=key_dtype)
    return torch.cat(
        (
            pre_rope_key[:, :num_prefix],
            patch_key.transpose(1, 2),
        ),
        dim=1,
    )


def _build_query_state(
    active_indices: torch.Tensor,
    num_prefix: int,
) -> PackedQueryState:
    num_active = active_indices.numel()
    device = active_indices.device
    active_pos = torch.arange(num_active, device=device, dtype=torch.long)
    valid_mask = torch.ones((1, num_active), device=device, dtype=torch.bool)
    return PackedQueryState(
        active_batch_idx=torch.zeros(num_active, device=device, dtype=torch.long),
        active_pos_idx=active_pos,
        active_token_idx=active_indices,
        query_valid_mask=valid_mask,
        active_query_pos_padded=active_pos.unsqueeze(0),
        active_query_mask=valid_mask,
        all_valid=True,
        active_patch_mask=active_indices >= num_prefix,
        active_rope_idx=(active_indices - num_prefix).clamp_min(0),
    )


def _update_layer_override(
    previous: Dict[str, torch.Tensor] | None,
    active_indices: torch.Tensor,
    active_kv: torch.Tensor,
    num_prefix: int,
) -> Dict[str, torch.Tensor]:
    if previous is None:
        return {
            "indices": active_indices.detach().clone(),
            "kv": active_kv.detach().clone(),
        }
    previous_indices = previous["indices"]
    keep = previous_indices >= num_prefix
    if active_indices.numel() > num_prefix:
        keep &= ~torch.isin(previous_indices, active_indices[num_prefix:])
    return {
        "indices": torch.cat((previous_indices[keep], active_indices)),
        "kv": torch.cat((previous["kv"][:, keep], active_kv), dim=1),
    }


def _resolve_logical_high_kv(
    state: Dict[str, Any],
    tag: str,
    high_rope,
    num_prefix: int,
) -> torch.Tensor:
    low_cache = state["low_cache"]
    parent_index = state["parent_token_index"]
    pre_rope_key = low_cache[f"{tag}_pre_rope_k"].index_select(1, parent_index)
    high_key = _apply_high_rope(pre_rope_key, high_rope, num_prefix)
    high_value = low_cache[f"{tag}_kv"][:, :, 1].index_select(1, parent_index)
    logical_kv = torch.stack((high_key, high_value), dim=2)
    override = state["overrides"].get(tag)
    if override is not None:
        logical_kv[:, override["indices"]] = override["kv"]
    return logical_kv


def _run_low_range(
    backbone,
    state: Dict[str, Any],
    low_rope,
    interaction_indexes: set[int],
    start_layer: int,
    end_layer: int,
) -> Dict[int, torch.Tensor]:
    captures: Dict[int, torch.Tensor] = {}
    low_x = state["low_x"]
    low_cache = state["low_cache"]
    for layer_idx in range(start_layer, end_layer):
        tag = state["layer_tags"][layer_idx]
        low_x, low_cache = backbone.blocks[layer_idx].approx(
            low_x,
            low_rope,
            low_cache,
            tag=tag,
            appcorr_method="partial_token",
            server_pscore="cls_attn_prob",
            server_pscore_weight=1.0,
            cache_pre_rope_k=True,
            debug=False,
        )
        if layer_idx in interaction_indexes:
            captures[layer_idx] = low_x
    state["low_x"] = low_x
    state["low_cache"] = low_cache
    return captures


def _run_active_range(
    backbone,
    state: Dict[str, Any],
    x_active: torch.Tensor,
    active_indices: torch.Tensor,
    high_rope,
    interaction_indexes: set[int],
    start_layer: int,
    end_layer: int,
) -> tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    num_prefix = state["num_prefix"]
    query_state = _build_query_state(active_indices, num_prefix)
    captures: Dict[int, torch.Tensor] = {}
    for layer_idx in range(start_layer, end_layer):
        block = backbone.blocks[layer_idx]
        tag = state["layer_tags"][layer_idx]
        logical_kv = _resolve_logical_high_kv(
            state,
            tag,
            high_rope,
            num_prefix,
        )
        attention_cache = {f"{tag}_kv": logical_kv}
        attention_out, attention_cache = block.attn.correct(
            block.norm1(x_active),
            dindice=active_indices.unsqueeze(0),
            rope=high_rope,
            cache_feature=attention_cache,
            tag=tag,
            appcorr_method="partial_token",
            fixed_query_state=query_state,
            sdpa_query_bucket_size=0,
        )
        x_attn = x_active + block.ls1(attention_out).to(dtype=x_active.dtype)
        x_active = x_attn + block.ls2(block.mlp(block.norm2(x_attn))).to(
            dtype=x_active.dtype
        )
        active_kv = attention_cache[f"{tag}_kv"][:, active_indices]
        state["overrides"][tag] = _update_layer_override(
            state["overrides"].get(tag),
            active_indices,
            active_kv,
            num_prefix,
        )
        if layer_idx in interaction_indexes:
            captures[layer_idx] = x_active
    return x_active, captures


def _install_low_captures(
    state: Dict[str, Any],
    captures: Dict[int, torch.Tensor],
) -> None:
    for layer_idx, value in captures.items():
        state["high_intermediates"][layer_idx] = lift_token_grid(
            value,
            state["low_hw"],
            state["high_hw"],
            state["num_prefix"],
        )


def _install_active_captures(
    state: Dict[str, Any],
    captures: Dict[int, torch.Tensor],
    active_indices: torch.Tensor,
) -> None:
    for layer_idx, value in captures.items():
        if layer_idx not in state["high_intermediates"]:
            raise RuntimeError(
                f"Missing low-resolution capture for interaction layer {layer_idx}"
            )
        output = state["high_intermediates"][layer_idx].clone()
        output[0, active_indices] = value.to(dtype=output.dtype)
        state["high_intermediates"][layer_idx] = output


def initialize_state(
    low_tokens: torch.Tensor,
    low_hw: tuple[int, int],
    high_hw: tuple[int, int],
    num_prefix: int,
    total_layers: int,
) -> Dict[str, Any]:
    if low_tokens.shape[0] != 1:
        raise ValueError("lowres_expand_once currently supports batch size 1 per source")
    return {
        "low_x": low_tokens.clone(),
        "low_cache": {},
        "low_hw": tuple(low_hw),
        "high_hw": tuple(high_hw),
        "num_prefix": int(num_prefix),
        "parent_token_index": build_parent_token_index(
            low_hw,
            high_hw,
            num_prefix,
            low_tokens.device,
        ),
        "layer_tags": [f"lowres_expand_layer{idx}" for idx in range(total_layers)],
        "overrides": {},
        "patch_states": {},
        "group_indices": {},
        "prefix_state": None,
        "high_intermediates": {},
        "arrived_group": 0,
        "frontier": 0,
    }


def correct_new_group(
    backbone,
    state: Dict[str, Any],
    high_tokens: torch.Tensor,
    high_rope,
    group_id: int,
    group_dindex: torch.Tensor | None,
    interaction_indexes: set[int],
    end_layer: int,
) -> None:
    num_prefix = state["num_prefix"]
    if group_dindex is None:
        active_indices = torch.arange(
            num_prefix,
            device=high_tokens.device,
            dtype=torch.long,
        )
    else:
        if group_dindex.ndim != 2 or group_dindex.shape[0] != 1:
            raise ValueError(
                f"Expected group dindex [1,N], got {tuple(group_dindex.shape)}"
            )
        active_indices = group_dindex[0].to(
            device=high_tokens.device,
            dtype=torch.long,
            non_blocking=True,
        )
    group_input = high_tokens[0, active_indices].contiguous()
    group_output, captures = _run_active_range(
        backbone,
        state,
        group_input,
        active_indices,
        high_rope,
        interaction_indexes,
        0,
        end_layer,
    )
    state["prefix_state"] = group_output[:num_prefix]
    state["patch_states"][group_id] = group_output[num_prefix:]
    state["group_indices"][group_id] = active_indices
    state["arrived_group"] = max(int(state["arrived_group"]), int(group_id))
    state["frontier"] = int(end_layer)
    _install_active_captures(state, captures, active_indices)


def advance_frontier(
    backbone,
    state: Dict[str, Any],
    low_rope,
    high_rope,
    interaction_indexes: set[int],
    start_layer: int,
    end_layer: int,
) -> None:
    low_captures = _run_low_range(
        backbone,
        state,
        low_rope,
        interaction_indexes,
        start_layer,
        end_layer,
    )
    _install_low_captures(state, low_captures)

    arrived_group = int(state["arrived_group"])
    if arrived_group <= 0:
        state["frontier"] = int(end_layer)
        return

    prefix_state = state["prefix_state"]
    if prefix_state is None:
        raise RuntimeError("Fine groups arrived without a prefix state")
    num_prefix = state["num_prefix"]
    combined_indices = [
        torch.arange(
            num_prefix,
            device=prefix_state.device,
            dtype=torch.long,
        )
    ]
    combined_values = [prefix_state]
    spans: Dict[int, tuple[int, int]] = {}
    cursor = num_prefix
    for group_id in range(1, arrived_group + 1):
        group_indices = state["group_indices"].get(group_id)
        group_values = state["patch_states"].get(group_id)
        if group_indices is None or group_values is None:
            continue
        patch_indices = group_indices[num_prefix:]
        combined_indices.append(patch_indices)
        combined_values.append(group_values)
        spans[group_id] = (cursor, cursor + group_values.shape[0])
        cursor += group_values.shape[0]

    active_indices = torch.cat(combined_indices, dim=0)
    active_values = torch.cat(combined_values, dim=0)
    output, captures = _run_active_range(
        backbone,
        state,
        active_values,
        active_indices,
        high_rope,
        interaction_indexes,
        start_layer,
        end_layer,
    )
    state["prefix_state"] = output[:num_prefix]
    for group_id, (start, end) in spans.items():
        state["patch_states"][group_id] = output[start:end]
    state["frontier"] = int(end_layer)
    _install_active_captures(state, captures, active_indices)


def finalize_high_output(state: Dict[str, Any]) -> torch.Tensor:
    output = lift_token_grid(
        state["low_x"],
        state["low_hw"],
        state["high_hw"],
        state["num_prefix"],
    )
    prefix_state = state["prefix_state"]
    if prefix_state is not None:
        output[0, : state["num_prefix"]] = prefix_state
    for group_id, values in state["patch_states"].items():
        indices = state["group_indices"][group_id][state["num_prefix"] :]
        output[0, indices] = values
    return output


def state_cache_nbytes(state: Dict[str, Any]) -> int:
    total = 0
    for value in state["low_cache"].values():
        if torch.is_tensor(value):
            total += value.numel() * value.element_size()
    for override in state["overrides"].values():
        total += override["indices"].numel() * override["indices"].element_size()
        total += override["kv"].numel() * override["kv"].element_size()
    return total
