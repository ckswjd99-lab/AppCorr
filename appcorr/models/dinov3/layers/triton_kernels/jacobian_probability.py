"""Selected-logit softmax JVP producer for block-supported correction."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _selected_delta_logits_kernel(
    probability_ptr,
    query_ptr,
    delta_query_ptr,
    key_ptr,
    delta_key_ptr,
    key_index_ptr,
    delta_logits_ptr,
    center_ptr,
    stride_p_b,
    stride_p_h,
    stride_p_q,
    stride_p_k,
    stride_q_b,
    stride_q_h,
    stride_q_q,
    stride_q_d,
    stride_dq_b,
    stride_dq_h,
    stride_dq_q,
    stride_dq_d,
    stride_k_b,
    stride_k_h,
    stride_k_k,
    stride_k_d,
    stride_dk_b,
    stride_dk_h,
    stride_dk_k,
    stride_dk_d,
    stride_index_b,
    stride_index_k,
    stride_dl_b,
    stride_dl_h,
    stride_dl_q,
    stride_dl_k,
    stride_center_b,
    stride_center_h,
    stride_center_q,
    num_queries: tl.constexpr,
    num_keys: tl.constexpr,
    head_dim: tl.constexpr,
    scale,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    tile = tl.program_id(2)
    query_block = tile // tl.cdiv(num_keys, BLOCK_K)
    key_block = tile % tl.cdiv(num_keys, BLOCK_K)
    query_offsets = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    packed_key_offsets = key_block * BLOCK_K + tl.arange(0, BLOCK_K)
    dim_offsets = tl.arange(0, BLOCK_D)
    query_mask = query_offsets < num_queries
    key_mask = packed_key_offsets < num_keys
    dim_mask = dim_offsets < head_dim
    value_key_offsets = tl.load(
        key_index_ptr
        + batch * stride_index_b
        + packed_key_offsets * stride_index_k,
        mask=key_mask,
        other=0,
    )

    query = tl.load(
        query_ptr
        + batch * stride_q_b
        + head * stride_q_h
        + query_offsets[:, None] * stride_q_q
        + dim_offsets[None, :] * stride_q_d,
        mask=query_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    delta_query = tl.load(
        delta_query_ptr
        + batch * stride_dq_b
        + head * stride_dq_h
        + query_offsets[:, None] * stride_dq_q
        + dim_offsets[None, :] * stride_dq_d,
        mask=query_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    key = tl.load(
        key_ptr
        + batch * stride_k_b
        + head * stride_k_h
        + value_key_offsets[:, None] * stride_k_k
        + dim_offsets[None, :] * stride_k_d,
        mask=key_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    delta_key = tl.load(
        delta_key_ptr
        + batch * stride_dk_b
        + head * stride_dk_h
        + value_key_offsets[:, None] * stride_dk_k
        + dim_offsets[None, :] * stride_dk_d,
        mask=key_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )
    delta_logits = (
        tl.dot(delta_query, tl.trans(key))
        + tl.dot(query, tl.trans(delta_key))
    ) * scale
    probability = tl.load(
        probability_ptr
        + batch * stride_p_b
        + head * stride_p_h
        + query_offsets[:, None] * stride_p_q
        + packed_key_offsets[None, :] * stride_p_k,
        mask=query_mask[:, None] & key_mask[None, :],
        other=0.0,
    )
    delta_logits = tl.where(
        query_mask[:, None] & key_mask[None, :],
        delta_logits,
        0.0,
    )
    tl.store(
        delta_logits_ptr
        + batch * stride_dl_b
        + head * stride_dl_h
        + query_offsets[:, None] * stride_dl_q
        + packed_key_offsets[None, :] * stride_dl_k,
        delta_logits,
        mask=query_mask[:, None] & key_mask[None, :],
    )
    partial_center = tl.sum(probability * delta_logits, axis=1)
    tl.atomic_add(
        center_ptr
        + batch * stride_center_b
        + head * stride_center_h
        + query_offsets * stride_center_q,
        partial_center,
        mask=query_mask,
        sem="relaxed",
    )


@triton.jit
def _finish_softmax_jvp_kernel(
    probability_ptr,
    delta_logits_ptr,
    center_ptr,
    output_ptr,
    num_elements,
    keys_per_row: tl.constexpr,
    BLOCK: tl.constexpr,
):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < num_elements
    row = offsets // keys_per_row
    probability = tl.load(probability_ptr + offsets, mask=mask, other=0.0)
    delta_logits = tl.load(delta_logits_ptr + offsets, mask=mask, other=0.0)
    center = tl.load(center_ptr + row, mask=mask, other=0.0)
    tl.store(
        output_ptr + offsets,
        probability * (delta_logits - center),
        mask=mask,
    )


def selected_softmax_jvp_triton(
    probability: torch.Tensor,
    query: torch.Tensor,
    delta_query: torch.Tensor,
    key: torch.Tensor,
    delta_key: torch.Tensor,
    key_index: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    """Produce selected ``dS`` using the draft's full-softmax probabilities."""

    if probability.ndim != 4:
        raise ValueError("probability must be [B,H,Q,K]")
    if query.shape != delta_query.shape or key.shape != delta_key.shape:
        raise ValueError("query/delta_query and key/delta_key must match")
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("query and key must be four-dimensional")
    batch, heads, queries, selected_keys = probability.shape
    if query.shape[:3] != (batch, heads, queries):
        raise ValueError("query shape does not match probability")
    if key.shape[:2] != (batch, heads) or key.shape[-1] != query.shape[-1]:
        raise ValueError("key shape does not match query")
    if key_index.shape != (batch, selected_keys):
        raise ValueError("key_index must be [B,K_selected]")
    tensors = (probability, query, delta_query, key, delta_key, key_index)
    if not all(tensor.is_cuda for tensor in tensors):
        raise RuntimeError("selected_softmax_jvp_triton requires CUDA tensors")
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("all tensors must share a device")
    if len({tensor.dtype for tensor in tensors[:-1]}) != 1:
        raise ValueError("all floating-point tensors must share a dtype")

    probability = probability.contiguous()
    query = query.contiguous()
    delta_query = delta_query.contiguous()
    key = key.contiguous()
    delta_key = delta_key.contiguous()
    key_index = key_index.contiguous()
    delta_logits = torch.empty_like(probability)
    center = torch.zeros(
        (batch, heads, queries),
        device=probability.device,
        dtype=torch.float32,
    )
    block_m, block_k = 16, 32
    block_d = max(16, triton.next_power_of_2(query.shape[-1]))
    key_tiles = triton.cdiv(selected_keys, block_k)
    grid = (batch, heads, triton.cdiv(queries, block_m) * key_tiles)
    with torch.cuda.device(probability.device):
        _selected_delta_logits_kernel[grid](
            probability,
            query,
            delta_query,
            key,
            delta_key,
            key_index,
            delta_logits,
            center,
            *probability.stride(),
            *query.stride(),
            *delta_query.stride(),
            *key.stride(),
            *delta_key.stride(),
            *key_index.stride(),
            *delta_logits.stride(),
            *center.stride(),
            num_queries=queries,
            num_keys=selected_keys,
            head_dim=query.shape[-1],
            scale=float(scale),
            BLOCK_M=block_m,
            BLOCK_K=block_k,
            BLOCK_D=block_d,
            num_warps=4,
            num_stages=2,
        )
        output = torch.empty_like(probability)
        num_elements = probability.numel()
        _finish_softmax_jvp_kernel[(triton.cdiv(num_elements, 256),)](
            probability,
            delta_logits,
            center,
            output,
            num_elements,
            keys_per_row=selected_keys,
            BLOCK=256,
        )
    return output
