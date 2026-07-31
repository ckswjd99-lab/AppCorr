"""Block-sparse FFN down projection over token/channel support descriptors."""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _packed_ffn_hidden_delta_kernel(
    base_x_ptr,
    corrected_x_ptr,
    corrected_gate_ptr,
    gate_weight_ptr,
    up_weight_ptr,
    gate_bias_ptr,
    up_bias_ptr,
    block_index_ptr,
    packed_hidden_ptr,
    stride_x_b,
    stride_x_t,
    stride_x_h,
    stride_g1_b,
    stride_g1_t,
    stride_g1_c,
    stride_wg_c,
    stride_wg_h,
    stride_wu_c,
    stride_wu_h,
    stride_i_b,
    stride_i_tb,
    stride_i_s,
    stride_p_b,
    stride_p_tb,
    stride_p_s,
    stride_p_t,
    stride_p_c,
    num_tokens: tl.constexpr,
    num_hidden: tl.constexpr,
    num_channels: tl.constexpr,
    TOKEN_BLOCK_SIZE: tl.constexpr,
    CHANNEL_BLOCK_SIZE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    batch = tl.program_id(0)
    token_block = tl.program_id(1)
    selected_offset = tl.program_id(2)
    channel_block = tl.load(
        block_index_ptr
        + batch * stride_i_b
        + token_block * stride_i_tb
        + selected_offset * stride_i_s
    )
    local_token_offsets = tl.arange(0, BLOCK_T)
    token_offsets = token_block * TOKEN_BLOCK_SIZE + local_token_offsets
    local_channel_offsets = tl.arange(0, BLOCK_C)
    channel_offsets = (
        channel_block * CHANNEL_BLOCK_SIZE + local_channel_offsets
    )
    token_mask = (
        (local_token_offsets < TOKEN_BLOCK_SIZE)
        & (token_offsets < num_tokens)
    )
    channel_mask = (
        (channel_block >= 0)
        & (local_channel_offsets < CHANNEL_BLOCK_SIZE)
        & (channel_offsets >= 0)
        & (channel_offsets < num_channels)
    )
    gate0_accumulator = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
    up0_accumulator = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
    up1_accumulator = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)

    for hidden_start in range(0, num_hidden, BLOCK_H):
        hidden_offsets = hidden_start + tl.arange(0, BLOCK_H)
        hidden_mask = hidden_offsets < num_hidden
        base_x = tl.load(
            base_x_ptr
            + batch * stride_x_b
            + token_offsets[:, None] * stride_x_t
            + hidden_offsets[None, :] * stride_x_h,
            mask=token_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        )
        corrected_x = tl.load(
            corrected_x_ptr
            + batch * stride_x_b
            + token_offsets[:, None] * stride_x_t
            + hidden_offsets[None, :] * stride_x_h,
            mask=token_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        )
        gate_weight = tl.load(
            gate_weight_ptr
            + hidden_offsets[:, None] * stride_wg_h
            + channel_offsets[None, :] * stride_wg_c,
            mask=hidden_mask[:, None] & channel_mask[None, :],
            other=0.0,
        )
        up_weight = tl.load(
            up_weight_ptr
            + hidden_offsets[:, None] * stride_wu_h
            + channel_offsets[None, :] * stride_wu_c,
            mask=hidden_mask[:, None] & channel_mask[None, :],
            other=0.0,
        )
        gate0_accumulator += tl.dot(base_x, gate_weight)
        up0_accumulator += tl.dot(base_x, up_weight)
        up1_accumulator += tl.dot(corrected_x, up_weight)

    gate_bias = tl.load(
        gate_bias_ptr + channel_offsets,
        mask=channel_mask,
        other=0.0,
    )
    up_bias = tl.load(
        up_bias_ptr + channel_offsets,
        mask=channel_mask,
        other=0.0,
    )
    gate0 = (gate0_accumulator + gate_bias[None, :]).to(tl.bfloat16)
    up0 = (up0_accumulator + up_bias[None, :]).to(tl.bfloat16)
    up1 = (up1_accumulator + up_bias[None, :]).to(tl.bfloat16)
    corrected_gate = tl.load(
        corrected_gate_ptr
        + batch * stride_g1_b
        + token_offsets[:, None] * stride_g1_t
        + channel_offsets[None, :] * stride_g1_c,
        mask=token_mask[:, None] & channel_mask[None, :],
        other=0.0,
    )
    gate0_float = gate0.to(tl.float32)
    base_gate = (
        gate0_float * tl.sigmoid(gate0_float)
    ).to(tl.bfloat16)
    hidden_delta = (
        corrected_gate * up1 - base_gate * up0
    ).to(tl.bfloat16)
    tl.store(
        packed_hidden_ptr
        + batch * stride_p_b
        + token_block * stride_p_tb
        + selected_offset * stride_p_s
        + local_token_offsets[:, None] * stride_p_t
        + local_channel_offsets[None, :] * stride_p_c,
        hidden_delta,
        mask=token_mask[:, None] & channel_mask[None, :],
    )


@triton.jit
def _packed_ffn_gate_kernel(
    base_x_ptr,
    gate_weight_ptr,
    gate_bias_ptr,
    block_index_ptr,
    packed_gate_ptr,
    stride_x_b,
    stride_x_t,
    stride_x_h,
    stride_w_c,
    stride_w_h,
    stride_i_b,
    stride_i_tb,
    stride_i_s,
    stride_p_b,
    stride_p_tb,
    stride_p_s,
    stride_p_t,
    stride_p_c,
    num_tokens: tl.constexpr,
    num_hidden: tl.constexpr,
    num_channels: tl.constexpr,
    TOKEN_BLOCK_SIZE: tl.constexpr,
    CHANNEL_BLOCK_SIZE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    batch = tl.program_id(0)
    token_block = tl.program_id(1)
    selected_offset = tl.program_id(2)
    channel_block = tl.load(
        block_index_ptr
        + batch * stride_i_b
        + token_block * stride_i_tb
        + selected_offset * stride_i_s
    )
    local_tokens = tl.arange(0, BLOCK_T)
    tokens = token_block * TOKEN_BLOCK_SIZE + local_tokens
    local_channels = tl.arange(0, BLOCK_C)
    channels = channel_block * CHANNEL_BLOCK_SIZE + local_channels
    token_mask = (
        (local_tokens < TOKEN_BLOCK_SIZE)
        & (tokens < num_tokens)
    )
    channel_mask = (
        (channel_block >= 0)
        & (local_channels < CHANNEL_BLOCK_SIZE)
        & (channels >= 0)
        & (channels < num_channels)
    )
    accumulator = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
    for hidden_start in range(0, num_hidden, BLOCK_H):
        hidden = hidden_start + tl.arange(0, BLOCK_H)
        hidden_mask = hidden < num_hidden
        x = tl.load(
            base_x_ptr
            + batch * stride_x_b
            + tokens[:, None] * stride_x_t
            + hidden[None, :] * stride_x_h,
            mask=token_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        )
        weight = tl.load(
            gate_weight_ptr
            + hidden[:, None] * stride_w_h
            + channels[None, :] * stride_w_c,
            mask=hidden_mask[:, None] & channel_mask[None, :],
            other=0.0,
        )
        accumulator += tl.dot(x, weight)
    bias = tl.load(
        gate_bias_ptr + channels,
        mask=channel_mask,
        other=0.0,
    )
    pre_activation = (accumulator + bias[None, :]).to(tl.bfloat16)
    pre_activation_float = pre_activation.to(tl.float32)
    gate = (
        pre_activation_float * tl.sigmoid(pre_activation_float)
    ).to(tl.bfloat16)
    tl.store(
        packed_gate_ptr
        + batch * stride_p_b
        + token_block * stride_p_tb
        + selected_offset * stride_p_s
        + local_tokens[:, None] * stride_p_t
        + local_channels[None, :] * stride_p_c,
        gate,
        mask=token_mask[:, None] & channel_mask[None, :],
    )


@triton.jit
def _packed_ffn_up_delta_kernel(
    base_x_ptr,
    corrected_x_ptr,
    corrected_gate_ptr,
    up_weight_ptr,
    up_bias_ptr,
    block_index_ptr,
    packed_gate_ptr,
    packed_hidden_ptr,
    stride_x_b,
    stride_x_t,
    stride_x_h,
    stride_g1_b,
    stride_g1_t,
    stride_g1_c,
    stride_w_c,
    stride_w_h,
    stride_i_b,
    stride_i_tb,
    stride_i_s,
    stride_p_b,
    stride_p_tb,
    stride_p_s,
    stride_p_t,
    stride_p_c,
    num_tokens: tl.constexpr,
    num_hidden: tl.constexpr,
    num_channels: tl.constexpr,
    TOKEN_BLOCK_SIZE: tl.constexpr,
    CHANNEL_BLOCK_SIZE: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    batch = tl.program_id(0)
    token_block = tl.program_id(1)
    selected_offset = tl.program_id(2)
    channel_block = tl.load(
        block_index_ptr
        + batch * stride_i_b
        + token_block * stride_i_tb
        + selected_offset * stride_i_s
    )
    local_tokens = tl.arange(0, BLOCK_T)
    tokens = token_block * TOKEN_BLOCK_SIZE + local_tokens
    local_channels = tl.arange(0, BLOCK_C)
    channels = channel_block * CHANNEL_BLOCK_SIZE + local_channels
    token_mask = (
        (local_tokens < TOKEN_BLOCK_SIZE)
        & (tokens < num_tokens)
    )
    channel_mask = (
        (channel_block >= 0)
        & (local_channels < CHANNEL_BLOCK_SIZE)
        & (channels >= 0)
        & (channels < num_channels)
    )
    base_accumulator = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
    corrected_accumulator = tl.zeros((BLOCK_T, BLOCK_C), dtype=tl.float32)
    for hidden_start in range(0, num_hidden, BLOCK_H):
        hidden = hidden_start + tl.arange(0, BLOCK_H)
        hidden_mask = hidden < num_hidden
        base_x = tl.load(
            base_x_ptr
            + batch * stride_x_b
            + tokens[:, None] * stride_x_t
            + hidden[None, :] * stride_x_h,
            mask=token_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        )
        corrected_x = tl.load(
            corrected_x_ptr
            + batch * stride_x_b
            + tokens[:, None] * stride_x_t
            + hidden[None, :] * stride_x_h,
            mask=token_mask[:, None] & hidden_mask[None, :],
            other=0.0,
        )
        weight = tl.load(
            up_weight_ptr
            + hidden[:, None] * stride_w_h
            + channels[None, :] * stride_w_c,
            mask=hidden_mask[:, None] & channel_mask[None, :],
            other=0.0,
        )
        base_accumulator += tl.dot(base_x, weight)
        corrected_accumulator += tl.dot(corrected_x, weight)
    bias = tl.load(
        up_bias_ptr + channels,
        mask=channel_mask,
        other=0.0,
    )
    base_up = (base_accumulator + bias[None, :]).to(tl.bfloat16)
    corrected_up = (
        corrected_accumulator + bias[None, :]
    ).to(tl.bfloat16)
    packed_offsets = (
        batch * stride_p_b
        + token_block * stride_p_tb
        + selected_offset * stride_p_s
        + local_tokens[:, None] * stride_p_t
        + local_channels[None, :] * stride_p_c
    )
    base_gate = tl.load(
        packed_gate_ptr + packed_offsets,
        mask=token_mask[:, None] & channel_mask[None, :],
        other=0.0,
    )
    corrected_gate = tl.load(
        corrected_gate_ptr
        + batch * stride_g1_b
        + tokens[:, None] * stride_g1_t
        + channels[None, :] * stride_g1_c,
        mask=token_mask[:, None] & channel_mask[None, :],
        other=0.0,
    )
    hidden_delta = (
        corrected_gate * corrected_up - base_gate * base_up
    ).to(tl.bfloat16)
    tl.store(
        packed_hidden_ptr + packed_offsets,
        hidden_delta,
        mask=token_mask[:, None] & channel_mask[None, :],
    )


@triton.jit
def _packed_block_sparse_down_projection_kernel(
    packed_hidden_ptr,
    weight_ptr,
    block_index_ptr,
    output_ptr,
    stride_p_b,
    stride_p_tb,
    stride_p_s,
    stride_p_t,
    stride_p_c,
    stride_w_o,
    stride_w_c,
    stride_i_b,
    stride_i_tb,
    stride_i_s,
    stride_y_b,
    stride_y_t,
    stride_y_o,
    num_tokens: tl.constexpr,
    num_channels: tl.constexpr,
    num_outputs: tl.constexpr,
    TOKEN_BLOCK_SIZE: tl.constexpr,
    CHANNEL_BLOCK_SIZE: tl.constexpr,
    SELECTED_BLOCKS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_O: tl.constexpr,
):
    batch = tl.program_id(0)
    token_block = tl.program_id(1)
    output_block = tl.program_id(2)
    local_token_offsets = tl.arange(0, BLOCK_T)
    token_offsets = token_block * TOKEN_BLOCK_SIZE + local_token_offsets
    output_offsets = output_block * BLOCK_O + tl.arange(0, BLOCK_O)
    token_mask = (
        (local_token_offsets < TOKEN_BLOCK_SIZE)
        & (token_offsets < num_tokens)
    )
    output_mask = output_offsets < num_outputs
    local_channel_offsets = tl.arange(0, BLOCK_C)
    accumulator = tl.zeros((BLOCK_T, BLOCK_O), dtype=tl.float32)

    for selected_offset in range(SELECTED_BLOCKS):
        channel_block = tl.load(
            block_index_ptr
            + batch * stride_i_b
            + token_block * stride_i_tb
            + selected_offset * stride_i_s
        )
        channel_offsets = (
            channel_block * CHANNEL_BLOCK_SIZE + local_channel_offsets
        )
        channel_mask = (
            (channel_block >= 0)
            & (local_channel_offsets < CHANNEL_BLOCK_SIZE)
            & (channel_offsets >= 0)
            & (channel_offsets < num_channels)
        )
        packed_hidden = tl.load(
            packed_hidden_ptr
            + batch * stride_p_b
            + token_block * stride_p_tb
            + selected_offset * stride_p_s
            + local_token_offsets[:, None] * stride_p_t
            + local_channel_offsets[None, :] * stride_p_c,
            mask=token_mask[:, None] & channel_mask[None, :],
            other=0.0,
        )
        weight = tl.load(
            weight_ptr
            + channel_offsets[:, None] * stride_w_c
            + output_offsets[None, :] * stride_w_o,
            mask=channel_mask[:, None] & output_mask[None, :],
            other=0.0,
        )
        accumulator += tl.dot(packed_hidden, weight)

    tl.store(
        output_ptr
        + batch * stride_y_b
        + token_offsets[:, None] * stride_y_t
        + output_offsets[None, :] * stride_y_o,
        accumulator,
        mask=token_mask[:, None] & output_mask[None, :],
    )


@triton.jit
def _block_sparse_down_projection_kernel(
    hidden_ptr,
    weight_ptr,
    block_index_ptr,
    output_ptr,
    stride_x_b,
    stride_x_t,
    stride_x_c,
    stride_w_o,
    stride_w_c,
    stride_i_b,
    stride_i_tb,
    stride_i_s,
    stride_y_b,
    stride_y_t,
    stride_y_o,
    num_tokens: tl.constexpr,
    num_channels: tl.constexpr,
    num_outputs: tl.constexpr,
    TOKEN_BLOCK_SIZE: tl.constexpr,
    CHANNEL_BLOCK_SIZE: tl.constexpr,
    SELECTED_BLOCKS: tl.constexpr,
    BLOCK_T: tl.constexpr,
    BLOCK_C: tl.constexpr,
    BLOCK_O: tl.constexpr,
):
    batch = tl.program_id(0)
    token_block = tl.program_id(1)
    output_block = tl.program_id(2)

    local_token_offsets = tl.arange(0, BLOCK_T)
    token_offsets = token_block * TOKEN_BLOCK_SIZE + local_token_offsets
    output_offsets = output_block * BLOCK_O + tl.arange(0, BLOCK_O)
    token_mask = (
        (local_token_offsets < TOKEN_BLOCK_SIZE)
        & (token_offsets < num_tokens)
    )
    output_mask = output_offsets < num_outputs
    accumulator = tl.zeros((BLOCK_T, BLOCK_O), dtype=tl.float32)

    for selected_offset in range(SELECTED_BLOCKS):
        channel_block = tl.load(
            block_index_ptr
            + batch * stride_i_b
            + token_block * stride_i_tb
            + selected_offset * stride_i_s
        )
        local_channel_offsets = tl.arange(0, BLOCK_C)
        channel_offsets = (
            channel_block * CHANNEL_BLOCK_SIZE + local_channel_offsets
        )
        channel_mask = (
            (channel_block >= 0)
            & (local_channel_offsets < CHANNEL_BLOCK_SIZE)
            & (channel_offsets >= 0)
            & (channel_offsets < num_channels)
        )
        hidden = tl.load(
            hidden_ptr
            + batch * stride_x_b
            + token_offsets[:, None] * stride_x_t
            + channel_offsets[None, :] * stride_x_c,
            mask=token_mask[:, None] & channel_mask[None, :],
            other=0.0,
        )
        weight = tl.load(
            weight_ptr
            + channel_offsets[:, None] * stride_w_c
            + output_offsets[None, :] * stride_w_o,
            mask=channel_mask[:, None] & output_mask[None, :],
            other=0.0,
        )
        accumulator += tl.dot(hidden, weight)

    tl.store(
        output_ptr
        + batch * stride_y_b
        + token_offsets[:, None] * stride_y_t
        + output_offsets[None, :] * stride_y_o,
        accumulator,
        mask=token_mask[:, None] & output_mask[None, :],
    )


def block_sparse_down_projection_triton(
    hidden_delta: torch.Tensor,
    weight: torch.Tensor,
    block_index: torch.Tensor,
    *,
    token_block_size: int,
    channel_block_size: int,
) -> torch.Tensor:
    """Compute ``hidden_delta @ weight.T`` using selected channel blocks only."""

    if hidden_delta.ndim != 3:
        raise ValueError("hidden_delta must be [B,T,C]")
    if weight.ndim != 2:
        raise ValueError("weight must be [O,C]")
    batch, tokens, channels = hidden_delta.shape
    outputs, weight_channels = weight.shape
    if weight_channels != channels:
        raise ValueError("hidden_delta and weight channel dimensions must match")
    if min(token_block_size, channel_block_size) <= 0:
        raise ValueError("block sizes must be positive")
    token_blocks = triton.cdiv(tokens, token_block_size)
    if block_index.ndim != 3 or block_index.shape[:2] != (
        batch,
        token_blocks,
    ):
        raise ValueError(
            "block_index must be [B,ceil(T/token_block_size),selected_blocks]"
        )
    if not hidden_delta.is_cuda or not weight.is_cuda or not block_index.is_cuda:
        raise RuntimeError(
            "block_sparse_down_projection_triton requires CUDA tensors"
        )
    if len({hidden_delta.device, weight.device, block_index.device}) != 1:
        raise ValueError("all tensors must share a device")
    if hidden_delta.dtype != weight.dtype:
        raise ValueError("hidden_delta and weight must share a dtype")
    if block_index.dtype not in {torch.int32, torch.int64}:
        raise ValueError("block_index must be int32 or int64")
    selected_blocks = block_index.shape[-1]
    if selected_blocks <= 0:
        raise ValueError("at least one channel block must be selected")

    hidden_delta = hidden_delta.contiguous()
    weight = weight.contiguous()
    block_index = block_index.contiguous()
    output = torch.empty(
        batch,
        tokens,
        outputs,
        device=hidden_delta.device,
        dtype=hidden_delta.dtype,
    )
    block_t = triton.next_power_of_2(token_block_size)
    block_c = max(16, triton.next_power_of_2(channel_block_size))
    block_o = 128
    grid = (batch, token_blocks, triton.cdiv(outputs, block_o))
    with torch.cuda.device(hidden_delta.device):
        _block_sparse_down_projection_kernel[grid](
            hidden_delta,
            weight,
            block_index,
            output,
            *hidden_delta.stride(),
            *weight.stride(),
            *block_index.stride(),
            *output.stride(),
            num_tokens=tokens,
            num_channels=channels,
            num_outputs=outputs,
            TOKEN_BLOCK_SIZE=token_block_size,
            CHANNEL_BLOCK_SIZE=channel_block_size,
            SELECTED_BLOCKS=selected_blocks,
            BLOCK_T=block_t,
            BLOCK_C=block_c,
            BLOCK_O=block_o,
            num_warps=8,
            num_stages=3,
        )
    return output


def block_sparse_ffn_delta_triton(
    base_x: torch.Tensor,
    corrected_x: torch.Tensor,
    corrected_gate: torch.Tensor,
    gate_weight: torch.Tensor,
    up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    block_index: torch.Tensor,
    *,
    gate_bias: torch.Tensor | None = None,
    up_bias: torch.Tensor | None = None,
    token_block_size: int,
    channel_block_size: int,
) -> torch.Tensor:
    """Compute selected exact SwiGLU finite differences and sparse down projection."""

    if base_x.shape != corrected_x.shape or base_x.ndim != 3:
        raise ValueError("base_x and corrected_x must be matching [B,T,H] tensors")
    batch, tokens, hidden = base_x.shape
    if corrected_gate.ndim != 3 or corrected_gate.shape[:2] != (batch, tokens):
        raise ValueError("corrected_gate must be [B,T,C]")
    channels = corrected_gate.shape[-1]
    if gate_weight.shape != (channels, hidden):
        raise ValueError("gate_weight must be [C,H]")
    if up_weight.shape != gate_weight.shape:
        raise ValueError("up_weight must match gate_weight")
    if down_weight.ndim != 2 or down_weight.shape[1] != channels:
        raise ValueError("down_weight must be [O,C]")
    token_blocks = triton.cdiv(tokens, token_block_size)
    if block_index.ndim != 3 or block_index.shape[:2] != (
        batch,
        token_blocks,
    ):
        raise ValueError("block_index shape does not match base_x")
    tensors = (
        base_x,
        corrected_x,
        corrected_gate,
        gate_weight,
        up_weight,
        down_weight,
        block_index,
    )
    if not all(tensor.is_cuda for tensor in tensors):
        raise RuntimeError("block_sparse_ffn_delta_triton requires CUDA tensors")
    if len({tensor.device for tensor in tensors}) != 1:
        raise ValueError("all tensors must share a device")
    if len({tensor.dtype for tensor in tensors[:-1]}) != 1:
        raise ValueError("all floating-point tensors must share a dtype")
    if base_x.dtype != torch.bfloat16:
        raise ValueError("the current FFN kernel specialization requires bfloat16")
    if block_index.dtype not in {torch.int32, torch.int64}:
        raise ValueError("block_index must be int32 or int64")
    selected_blocks = block_index.shape[-1]
    if selected_blocks <= 0:
        raise ValueError("at least one channel block must be selected")

    if gate_bias is None:
        gate_bias = torch.zeros(
            channels,
            device=base_x.device,
            dtype=base_x.dtype,
        )
    if up_bias is None:
        up_bias = torch.zeros_like(gate_bias)
    if gate_bias.shape != (channels,) or up_bias.shape != (channels,):
        raise ValueError("gate/up bias must have shape [C]")
    if gate_bias.device != base_x.device or up_bias.device != base_x.device:
        raise ValueError("bias tensors must share the input device")
    gate_bias = gate_bias.to(dtype=base_x.dtype).contiguous()
    up_bias = up_bias.to(dtype=base_x.dtype).contiguous()

    base_x = base_x.contiguous()
    corrected_x = corrected_x.contiguous()
    corrected_gate = corrected_gate.contiguous()
    gate_weight = gate_weight.contiguous()
    up_weight = up_weight.contiguous()
    down_weight = down_weight.contiguous()
    block_index = block_index.contiguous()
    block_t = triton.next_power_of_2(token_block_size)
    block_c = max(16, triton.next_power_of_2(channel_block_size))
    block_h = 64
    packed_shape = (
        batch,
        token_blocks,
        selected_blocks,
        token_block_size,
        channel_block_size,
    )
    packed_hidden = torch.empty(
        packed_shape,
        device=base_x.device,
        dtype=base_x.dtype,
    )
    with torch.cuda.device(base_x.device):
        if token_block_size < 32:
            # Small token blocks have enough occupancy for the fused kernel and
            # avoid a second packed intermediate.
            _packed_ffn_hidden_delta_kernel[
                (batch, token_blocks, selected_blocks)
            ](
                base_x,
                corrected_x,
                corrected_gate,
                gate_weight,
                up_weight,
                gate_bias,
                up_bias,
                block_index,
                packed_hidden,
                *base_x.stride(),
                *corrected_gate.stride(),
                *gate_weight.stride(),
                *up_weight.stride(),
                *block_index.stride(),
                *packed_hidden.stride(),
                num_tokens=tokens,
                num_hidden=hidden,
                num_channels=channels,
                TOKEN_BLOCK_SIZE=token_block_size,
                CHANNEL_BLOCK_SIZE=channel_block_size,
                BLOCK_T=block_t,
                BLOCK_C=block_c,
                BLOCK_H=block_h,
                num_warps=8,
                num_stages=3,
            )
        else:
            # Larger token tiles put too much register pressure on the three
            # simultaneous accumulators. Split gate and up projections there.
            packed_gate = torch.empty_like(packed_hidden)
            _packed_ffn_gate_kernel[
                (batch, token_blocks, selected_blocks)
            ](
                base_x,
                gate_weight,
                gate_bias,
                block_index,
                packed_gate,
                *base_x.stride(),
                *gate_weight.stride(),
                *block_index.stride(),
                *packed_gate.stride(),
                num_tokens=tokens,
                num_hidden=hidden,
                num_channels=channels,
                TOKEN_BLOCK_SIZE=token_block_size,
                CHANNEL_BLOCK_SIZE=channel_block_size,
                BLOCK_T=block_t,
                BLOCK_C=block_c,
                BLOCK_H=block_h,
                num_warps=8,
                num_stages=3,
            )
            _packed_ffn_up_delta_kernel[
                (batch, token_blocks, selected_blocks)
            ](
                base_x,
                corrected_x,
                corrected_gate,
                up_weight,
                up_bias,
                block_index,
                packed_gate,
                packed_hidden,
                *base_x.stride(),
                *corrected_gate.stride(),
                *up_weight.stride(),
                *block_index.stride(),
                *packed_hidden.stride(),
                num_tokens=tokens,
                num_hidden=hidden,
                num_channels=channels,
                TOKEN_BLOCK_SIZE=token_block_size,
                CHANNEL_BLOCK_SIZE=channel_block_size,
                BLOCK_T=block_t,
                BLOCK_C=block_c,
                BLOCK_H=block_h,
                num_warps=8,
                num_stages=3,
            )

    outputs = down_weight.shape[0]
    output = torch.empty(
        batch,
        tokens,
        outputs,
        device=base_x.device,
        dtype=base_x.dtype,
    )
    block_o = 128
    with torch.cuda.device(base_x.device):
        _packed_block_sparse_down_projection_kernel[
            (batch, token_blocks, triton.cdiv(outputs, block_o))
        ](
            packed_hidden,
            down_weight,
            block_index,
            output,
            *packed_hidden.stride(),
            *down_weight.stride(),
            *block_index.stride(),
            *output.stride(),
            num_tokens=tokens,
            num_channels=channels,
            num_outputs=outputs,
            TOKEN_BLOCK_SIZE=token_block_size,
            CHANNEL_BLOCK_SIZE=channel_block_size,
            SELECTED_BLOCKS=selected_blocks,
            BLOCK_T=block_t,
            BLOCK_C=block_c,
            BLOCK_O=block_o,
            num_warps=8,
            num_stages=3,
        )
    return output
