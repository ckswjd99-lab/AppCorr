"""Query/head-block structured exact product-delta attention consumer."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _block_product_delta_kernel(
    base_probability_ptr,
    corrected_probability_ptr,
    base_value_ptr,
    corrected_value_ptr,
    block_index_ptr,
    output_ptr,
    stride_p0_b,
    stride_p0_h,
    stride_p0_q,
    stride_p0_k,
    stride_p1_b,
    stride_p1_h,
    stride_p1_q,
    stride_p1_k,
    stride_v0_b,
    stride_v0_h,
    stride_v0_k,
    stride_v0_d,
    stride_v1_b,
    stride_v1_h,
    stride_v1_k,
    stride_v1_d,
    stride_i_b,
    stride_i_hg,
    stride_i_qb,
    stride_i_s,
    stride_o_b,
    stride_o_h,
    stride_o_q,
    stride_o_d,
    num_queries: tl.constexpr,
    num_keys: tl.constexpr,
    head_dim: tl.constexpr,
    HEAD_GROUP_SIZE: tl.constexpr,
    QUERY_BLOCK_SIZE: tl.constexpr,
    KEY_BLOCK_SIZE: tl.constexpr,
    SELECTED_BLOCKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    query_block = tl.program_id(2)
    local_query_offsets = tl.arange(0, BLOCK_M)
    query_offsets = query_block * QUERY_BLOCK_SIZE + local_query_offsets
    dim_offsets = tl.arange(0, BLOCK_D)
    query_mask = (
        (local_query_offsets < QUERY_BLOCK_SIZE)
        & (query_offsets < num_queries)
    )
    dim_mask = dim_offsets < head_dim
    head_group = head // HEAD_GROUP_SIZE
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for selected_offset in range(SELECTED_BLOCKS):
        key_block = tl.load(
            block_index_ptr
            + batch * stride_i_b
            + head_group * stride_i_hg
            + query_block * stride_i_qb
            + selected_offset * stride_i_s
        )
        local_key_offsets = tl.arange(0, BLOCK_K)
        key_offsets = key_block * KEY_BLOCK_SIZE + local_key_offsets
        key_mask = (
            (local_key_offsets < KEY_BLOCK_SIZE)
            & (key_offsets < num_keys)
        )
        base_probability = tl.load(
            base_probability_ptr
            + batch * stride_p0_b
            + head * stride_p0_h
            + query_offsets[:, None] * stride_p0_q
            + key_offsets[None, :] * stride_p0_k,
            mask=query_mask[:, None] & key_mask[None, :],
            other=0.0,
        )
        corrected_probability = tl.load(
            corrected_probability_ptr
            + batch * stride_p1_b
            + head * stride_p1_h
            + query_offsets[:, None] * stride_p1_q
            + key_offsets[None, :] * stride_p1_k,
            mask=query_mask[:, None] & key_mask[None, :],
            other=0.0,
        )
        base_value = tl.load(
            base_value_ptr
            + batch * stride_v0_b
            + head * stride_v0_h
            + key_offsets[:, None] * stride_v0_k
            + dim_offsets[None, :] * stride_v0_d,
            mask=key_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        corrected_value = tl.load(
            corrected_value_ptr
            + batch * stride_v1_b
            + head * stride_v1_h
            + key_offsets[:, None] * stride_v1_k
            + dim_offsets[None, :] * stride_v1_d,
            mask=key_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        accumulator += tl.dot(corrected_probability, corrected_value)
        accumulator -= tl.dot(base_probability, base_value)

    tl.store(
        output_ptr
        + batch * stride_o_b
        + head * stride_o_h
        + query_offsets[:, None] * stride_o_q
        + dim_offsets[None, :] * stride_o_d,
        accumulator,
        mask=query_mask[:, None] & dim_mask[None, :],
    )


def block_product_delta_triton(
    base_probability: torch.Tensor,
    corrected_probability: torch.Tensor,
    base_value: torch.Tensor,
    corrected_value: torch.Tensor,
    block_index: torch.Tensor,
    *,
    head_group_size: int,
    query_block_size: int,
    key_block_size: int,
) -> torch.Tensor:
    """Compute selected ``P1V1-P0V0`` from block descriptors."""

    if base_probability.shape != corrected_probability.shape:
        raise ValueError("base/corrected probability shapes must match")
    if base_value.shape != corrected_value.shape:
        raise ValueError("base/corrected value shapes must match")
    if base_probability.ndim != 4 or base_value.ndim != 4:
        raise ValueError("probability and value tensors must be four-dimensional")
    batch, heads, queries, keys = base_probability.shape
    if base_value.shape[:3] != (batch, heads, keys):
        raise ValueError("value shape must be [B,H,K,D]")
    if min(head_group_size, query_block_size, key_block_size) <= 0:
        raise ValueError("block sizes must be positive")
    head_groups = triton.cdiv(heads, head_group_size)
    query_blocks = triton.cdiv(queries, query_block_size)
    if block_index.ndim != 4 or block_index.shape[:3] != (
        batch,
        head_groups,
        query_blocks,
    ):
        raise ValueError(
            "block_index must be [B,ceil(H/HG),ceil(Q/QB),selected_blocks]"
        )
    tensors = (
        base_probability,
        corrected_probability,
        base_value,
        corrected_value,
        block_index,
    )
    if not all(tensor.is_cuda for tensor in tensors):
        raise RuntimeError("block_product_delta_triton requires CUDA tensors")
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("all tensors must share a device")
    if len({tensor.dtype for tensor in tensors[:-1]}) != 1:
        raise ValueError("all floating-point tensors must share a dtype")
    if block_index.dtype not in {torch.int32, torch.int64}:
        raise ValueError("block_index must be int32 or int64")
    selected_blocks = block_index.shape[-1]
    if selected_blocks <= 0:
        raise ValueError("at least one key block must be selected")

    base_probability = base_probability.contiguous()
    corrected_probability = corrected_probability.contiguous()
    base_value = base_value.contiguous()
    corrected_value = corrected_value.contiguous()
    block_index = block_index.contiguous()
    head_dim = base_value.shape[-1]
    output = torch.empty(
        batch,
        heads,
        queries,
        head_dim,
        device=base_probability.device,
        dtype=base_probability.dtype,
    )
    block_m = max(16, triton.next_power_of_2(query_block_size))
    block_k = max(16, triton.next_power_of_2(key_block_size))
    block_d = max(16, triton.next_power_of_2(head_dim))
    grid = (batch, heads, query_blocks)
    with torch.cuda.device(base_probability.device):
        _block_product_delta_kernel[grid](
            base_probability,
            corrected_probability,
            base_value,
            corrected_value,
            block_index,
            output,
            *base_probability.stride(),
            *corrected_probability.stride(),
            *base_value.stride(),
            *corrected_value.stride(),
            *block_index.stride(),
            *output.stride(),
            num_queries=queries,
            num_keys=keys,
            head_dim=head_dim,
            HEAD_GROUP_SIZE=head_group_size,
            QUERY_BLOCK_SIZE=query_block_size,
            KEY_BLOCK_SIZE=key_block_size,
            SELECTED_BLOCKS=selected_blocks,
            BLOCK_M=block_m,
            BLOCK_K=block_k,
            BLOCK_D=block_d,
            num_warps=4,
            num_stages=3,
        )
    return output
