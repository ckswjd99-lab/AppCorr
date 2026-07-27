"""Tensor-core attention correction over regularly packed block support."""

from __future__ import annotations

from typing import Literal

import torch
import triton
import triton.language as tl


@triton.jit
def _packed_attention_delta_dot_kernel(
    probability_ptr,
    delta_probability_ptr,
    value_ptr,
    delta_value_ptr,
    key_index_ptr,
    base_output_ptr,
    output_ptr,
    stride_p_b,
    stride_p_h,
    stride_p_q,
    stride_p_k,
    stride_dp_b,
    stride_dp_h,
    stride_dp_q,
    stride_dp_k,
    stride_v_b,
    stride_v_h,
    stride_v_k,
    stride_v_d,
    stride_dv_b,
    stride_dv_h,
    stride_dv_k,
    stride_dv_d,
    stride_index_b,
    stride_index_k,
    stride_base_b,
    stride_base_h,
    stride_base_q,
    stride_base_d,
    stride_o_b,
    stride_o_h,
    stride_o_q,
    stride_o_d,
    num_queries: tl.constexpr,
    num_keys: tl.constexpr,
    head_dim: tl.constexpr,
    USE_INDIRECT: tl.constexpr,
    MODE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    batch = tl.program_id(0)
    head = tl.program_id(1)
    query_block = tl.program_id(2)
    query_offsets = query_block * BLOCK_M + tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, BLOCK_D)
    query_mask = query_offsets < num_queries
    dim_mask = dim_offsets < head_dim

    first_accumulator = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    second_accumulator = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    for key_start in range(0, num_keys, BLOCK_K):
        key_offsets = key_start + tl.arange(0, BLOCK_K)
        key_mask = key_offsets < num_keys
        if USE_INDIRECT:
            value_key_offsets = tl.load(
                key_index_ptr
                + batch * stride_index_b
                + key_offsets * stride_index_k,
                mask=key_mask,
                other=0,
            )
        else:
            value_key_offsets = key_offsets
        probability = tl.load(
            probability_ptr
            + batch * stride_p_b
            + head * stride_p_h
            + query_offsets[:, None] * stride_p_q
            + key_offsets[None, :] * stride_p_k,
            mask=query_mask[:, None] & key_mask[None, :],
            other=0.0,
        )
        delta_probability = tl.load(
            delta_probability_ptr
            + batch * stride_dp_b
            + head * stride_dp_h
            + query_offsets[:, None] * stride_dp_q
            + key_offsets[None, :] * stride_dp_k,
            mask=query_mask[:, None] & key_mask[None, :],
            other=0.0,
        )
        value = tl.load(
            value_ptr
            + batch * stride_v_b
            + head * stride_v_h
            + value_key_offsets[:, None] * stride_v_k
            + dim_offsets[None, :] * stride_v_d,
            mask=key_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        delta_value = tl.load(
            delta_value_ptr
            + batch * stride_dv_b
            + head * stride_dv_h
            + value_key_offsets[:, None] * stride_dv_k
            + dim_offsets[None, :] * stride_dv_d,
            mask=key_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )

        if MODE == 0:
            first_accumulator = tl.dot(
                probability,
                delta_value,
                first_accumulator,
                input_precision="ieee",
            )
            second_accumulator = tl.dot(
                delta_probability,
                value,
                second_accumulator,
                input_precision="ieee",
            )
        else:
            first_accumulator = tl.dot(
                probability + delta_probability,
                value + delta_value,
                first_accumulator,
                input_precision="ieee",
            )
            if MODE == 1:
                second_accumulator = tl.dot(
                    probability,
                    value,
                    second_accumulator,
                    input_precision="ieee",
                )

    if MODE == 0:
        output = first_accumulator + second_accumulator
    elif MODE == 1:
        output = first_accumulator - second_accumulator
    else:
        cached_base = tl.load(
            base_output_ptr
            + batch * stride_base_b
            + head * stride_base_h
            + query_offsets[:, None] * stride_base_q
            + dim_offsets[None, :] * stride_base_d,
            mask=query_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        output = first_accumulator - cached_base

    tl.store(
        output_ptr
        + batch * stride_o_b
        + head * stride_o_h
        + query_offsets[:, None] * stride_o_q
        + dim_offsets[None, :] * stride_o_d,
        output,
        mask=query_mask[:, None] & dim_mask[None, :],
    )


def packed_attention_delta_triton(
    probability: torch.Tensor,
    delta_probability: torch.Tensor,
    value: torch.Tensor,
    delta_value: torch.Tensor,
    *,
    backend: Literal["split_jvp", "product_delta"],
    base_support_output: torch.Tensor | None = None,
    key_index: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute correction over packed support with no sparse renormalization.

    For ``product_delta``, supplying draft-cached ``probability @ value`` uses
    one tensor-core product and one cached subtraction.  Omitting it is a
    diagnostic path that computes both products in the kernel.
    """

    if backend not in {"split_jvp", "product_delta"}:
        raise ValueError(f"Unknown backend: {backend}")
    tensors = (probability, delta_probability, value, delta_value)
    if not all(tensor.is_cuda for tensor in tensors):
        raise RuntimeError("packed_attention_delta_triton requires CUDA tensors")
    if probability.ndim != 4 or delta_probability.shape != probability.shape:
        raise ValueError("probability tensors must have shape [B,H,Q,K]")
    if value.ndim != 4 or delta_value.shape != value.shape:
        raise ValueError("value tensors must have shape [B,H,K,D]")
    if probability.shape[:2] != value.shape[:2]:
        raise ValueError("probability and value batch/head dimensions must match")
    if key_index is None and probability.shape[-1] != value.shape[-2]:
        raise ValueError(
            "probability key count must match value key count without key_index"
        )
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("all inputs must be on the same CUDA device")
    if len({tensor.dtype for tensor in tensors}) != 1:
        raise ValueError("all inputs must use the same dtype")
    if probability.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise ValueError(f"Unsupported dtype: {probability.dtype}")

    probability = probability.contiguous()
    delta_probability = delta_probability.contiguous()
    value = value.contiguous()
    delta_value = delta_value.contiguous()
    batch, heads, queries, keys = probability.shape
    head_dim = value.shape[-1]
    output_shape = (batch, heads, queries, head_dim)
    if keys == 0:
        return torch.zeros(
            output_shape,
            device=probability.device,
            dtype=probability.dtype,
        )

    use_indirect = key_index is not None
    if key_index is not None:
        if key_index.shape != (batch, keys):
            raise ValueError(
                f"key_index must have shape {(batch, keys)}, "
                f"got {tuple(key_index.shape)}"
            )
        if key_index.device != probability.device:
            raise ValueError("key_index must be on the input CUDA device")
        if key_index.dtype not in {torch.int32, torch.int64}:
            raise ValueError("key_index must be int32 or int64")
        key_index = key_index.contiguous()
    else:
        key_index = torch.empty(
            (1, 1),
            device=probability.device,
            dtype=torch.int32,
        )

    if base_support_output is not None:
        if backend != "product_delta":
            raise ValueError("base_support_output is only valid for product_delta")
        if base_support_output.shape != output_shape:
            raise ValueError(
                f"base_support_output must have shape {output_shape}, "
                f"got {tuple(base_support_output.shape)}"
            )
        if (
            base_support_output.device != probability.device
            or base_support_output.dtype != probability.dtype
        ):
            raise ValueError("base_support_output must match input device and dtype")
        base_support_output = base_support_output.contiguous()
        mode = 2
    else:
        base_support_output = torch.empty(
            (1, 1, 1, 1),
            device=probability.device,
            dtype=probability.dtype,
        )
        mode = 0 if backend == "split_jvp" else 1

    output = torch.empty(
        output_shape,
        device=probability.device,
        dtype=probability.dtype,
    )
    block_m = 16
    block_k = 32
    block_d = max(16, triton.next_power_of_2(head_dim))
    grid = (batch, heads, triton.cdiv(queries, block_m))
    with torch.cuda.device(probability.device):
        _packed_attention_delta_dot_kernel[grid](
            probability,
            delta_probability,
            value,
            delta_value,
            key_index,
            base_support_output,
            output,
            *probability.stride(),
            *delta_probability.stride(),
            *value.stride(),
            *delta_value.stride(),
            *key_index.stride(),
            *base_support_output.stride(),
            *output.stride(),
            num_queries=queries,
            num_keys=keys,
            head_dim=head_dim,
            USE_INDIRECT=use_indirect,
            MODE=mode,
            BLOCK_M=block_m,
            BLOCK_K=block_k,
            BLOCK_D=block_d,
            num_warps=4,
            num_stages=3,
        )
    return output
