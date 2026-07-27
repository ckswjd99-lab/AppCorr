"""Reference math and structured support selection for draft-guided correction.

This module intentionally contains plain PyTorch implementations.  They serve as
the numerical oracle for optimized kernels and keep the distinction between the
strict first-order JVP and the product-delta shortcut explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor


AttentionDeltaBackend = Literal["split_jvp", "product_delta"]
AttentionProbabilityMode = Literal["linearized", "exact"]


@dataclass(frozen=True)
class AttentionDelta:
    delta: Tensor
    base_output: Tensor
    corrected_output: Tensor
    base_probability: Tensor
    delta_probability: Tensor
    delta_logits: Tensor
    cross_term: Tensor


@dataclass(frozen=True)
class SupportStatistics:
    kept_fraction: float
    probability_mass: float
    block_count: int
    total_block_count: int


def _check_attention_inputs(q: Tensor, k: Tensor, v: Tensor, dq: Tensor, dk: Tensor, dv: Tensor) -> None:
    if q.ndim != 4:
        raise ValueError(f"Expected attention tensors [B,H,Q,D], got q={tuple(q.shape)}")
    if q.shape != dq.shape:
        raise ValueError(f"q and dq must match, got {tuple(q.shape)} and {tuple(dq.shape)}")
    if k.shape != dk.shape or v.shape != dv.shape:
        raise ValueError("k/dk and v/dv must have matching shapes")
    if q.shape[:2] != k.shape[:2] or k.shape[:3] != v.shape[:3]:
        raise ValueError("q, k, and v must share batch/head and k/v sequence dimensions")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k head dimensions must match")


def _as_support_mask(mask: Tensor | None, reference: Tensor) -> Tensor | None:
    if mask is None:
        return None
    try:
        return torch.broadcast_to(mask.to(device=reference.device, dtype=torch.bool), reference.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"Support mask {tuple(mask.shape)} cannot broadcast to {tuple(reference.shape)}"
        ) from exc


def softmax_jvp(
    base_probability: Tensor,
    delta_logits: Tensor,
    support_mask: Tensor | None = None,
) -> Tensor:
    """Apply the softmax Jacobian using the original full-softmax probability.

    With a sparse support, dropped logit perturbations are modeled as zero.  The
    centering scalar still uses probabilities normalized over the full key set;
    the selected entries are never re-softmaxed over their sparse subset.
    """

    if base_probability.shape != delta_logits.shape:
        raise ValueError("base_probability and delta_logits must have identical shapes")
    mask = _as_support_mask(support_mask, base_probability)
    effective_delta_logits = delta_logits if mask is None else delta_logits.masked_fill(~mask, 0)
    center = (base_probability * effective_delta_logits).sum(dim=-1, keepdim=True)
    delta_probability = base_probability * (effective_delta_logits - center)
    return delta_probability if mask is None else delta_probability.masked_fill(~mask, 0)


def attention_delta(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    dq: Tensor,
    dk: Tensor,
    dv: Tensor,
    *,
    scale: float | None = None,
    backend: AttentionDeltaBackend = "split_jvp",
    probability_mode: AttentionProbabilityMode = "linearized",
    support_mask: Tensor | None = None,
) -> AttentionDelta:
    """Compute dense or support-masked attention correction.

    ``product_delta`` computes ``(S+dS)(V+dV)-SV`` and therefore retains the
    second-order ``dS*dV`` term.  ``probability_mode="exact"`` uses the full
    corrected softmax; it is an oracle path and is not support-renormalized.
    """

    _check_attention_inputs(q, k, v, dq, dk, dv)
    if backend not in {"split_jvp", "product_delta"}:
        raise ValueError(f"Unknown attention delta backend: {backend}")
    if probability_mode not in {"linearized", "exact"}:
        raise ValueError(f"Unknown probability mode: {probability_mode}")

    scale_value = float(scale if scale is not None else q.shape[-1] ** -0.5)
    logits = torch.matmul(q, k.transpose(-2, -1)) * scale_value
    base_probability = torch.softmax(logits, dim=-1)
    base_output = torch.matmul(base_probability, v)

    delta_logits = (
        torch.matmul(dq, k.transpose(-2, -1))
        + torch.matmul(q, dk.transpose(-2, -1))
    ) * scale_value
    mask = _as_support_mask(support_mask, base_probability)

    if probability_mode == "linearized":
        delta_probability = softmax_jvp(base_probability, delta_logits, mask)
        corrected_probability = base_probability + delta_probability
    else:
        corrected_logits = torch.matmul(q + dq, (k + dk).transpose(-2, -1)) * scale_value
        exact_probability = torch.softmax(corrected_logits, dim=-1)
        delta_probability = exact_probability - base_probability
        if mask is not None:
            delta_probability = delta_probability.masked_fill(~mask, 0)
        corrected_probability = base_probability + delta_probability

    if mask is None:
        probability_used = base_probability
        value_used = v
        delta_value_used = dv
        corrected_probability_used = corrected_probability
    else:
        mask_value = mask.to(dtype=base_probability.dtype)
        probability_used = base_probability * mask_value
        value_used = v
        delta_value_used = dv
        corrected_probability_used = corrected_probability * mask_value

    split_delta = (
        torch.matmul(probability_used, delta_value_used)
        + torch.matmul(delta_probability, value_used)
    )
    cross_term = torch.matmul(delta_probability, delta_value_used)

    if backend == "split_jvp":
        delta = split_delta
        corrected_output = base_output + delta
    else:
        supported_base_output = torch.matmul(probability_used, value_used)
        supported_corrected_output = torch.matmul(
            corrected_probability_used,
            value_used + delta_value_used,
        )
        delta = supported_corrected_output - supported_base_output
        corrected_output = base_output + delta

    return AttentionDelta(
        delta=delta,
        base_output=base_output,
        corrected_output=corrected_output,
        base_probability=base_probability,
        delta_probability=delta_probability,
        delta_logits=delta_logits,
        cross_term=cross_term,
    )


def exact_attention_delta(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    dq: Tensor,
    dk: Tensor,
    dv: Tensor,
    *,
    scale: float | None = None,
) -> Tensor:
    """Exact attention difference, including all logit and value cross terms."""

    _check_attention_inputs(q, k, v, dq, dk, dv)
    scale_value = float(scale if scale is not None else q.shape[-1] ** -0.5)
    base_probability = torch.softmax(torch.matmul(q, k.transpose(-2, -1)) * scale_value, dim=-1)
    corrected_probability = torch.softmax(
        torch.matmul(q + dq, (k + dk).transpose(-2, -1)) * scale_value,
        dim=-1,
    )
    return torch.matmul(corrected_probability, v + dv) - torch.matmul(base_probability, v)


def attention_edge_energy(
    base_probability: Tensor,
    delta_probability: Tensor,
    v: Tensor,
    dv: Tensor,
    *,
    backend: AttentionDeltaBackend,
) -> Tensor:
    """Squared output-vector energy attributable to every query/key edge."""

    if base_probability.shape != delta_probability.shape:
        raise ValueError("Probability tensors must match")
    if v.shape != dv.shape:
        raise ValueError("v and dv must match")
    if backend == "split_jvp":
        first_coefficient = base_probability
        first_value = dv
        second_coefficient = delta_probability
        second_value = v
    elif backend == "product_delta":
        first_coefficient = base_probability + delta_probability
        first_value = v + dv
        second_coefficient = -base_probability
        second_value = v
    else:
        raise ValueError(f"Unknown attention delta backend: {backend}")

    # ||a*x + b*y||^2 without constructing [B,H,Q,K,D], which is prohibitive
    # for the 2K–4K-token dense-prediction workloads.
    first_norm_sq = first_value.float().square().sum(dim=-1).unsqueeze(-2)
    second_norm_sq = second_value.float().square().sum(dim=-1).unsqueeze(-2)
    cross_dot = (first_value.float() * second_value.float()).sum(dim=-1).unsqueeze(-2)
    first_coefficient = first_coefficient.float()
    second_coefficient = second_coefficient.float()
    return (
        first_coefficient.square() * first_norm_sq
        + second_coefficient.square() * second_norm_sq
        + 2 * first_coefficient * second_coefficient * cross_dot
    ).clamp_min_(0)


def _expand_block_mask(block_mask: Tensor, block_size: int, target_size: int) -> Tensor:
    return block_mask.repeat_interleave(block_size, dim=-1)[..., :target_size]


def select_attention_block_support(
    base_probability: Tensor,
    *,
    keep_ratio: float | None = None,
    tail_epsilon: float | None = None,
    key_block_size: int = 16,
    query_block_size: int = 8,
    head_group_size: int = 1,
    residual_key_mask: Tensor | None = None,
) -> tuple[Tensor, SupportStatistics]:
    """Select a query-block/head-group shared set of contiguous key blocks."""

    if base_probability.ndim != 4:
        raise ValueError("base_probability must be [B,H,Q,K]")
    if (keep_ratio is None) == (tail_epsilon is None):
        raise ValueError("Specify exactly one of keep_ratio or tail_epsilon")
    if key_block_size <= 0 or query_block_size <= 0 or head_group_size <= 0:
        raise ValueError("Block sizes must be positive")
    if keep_ratio is not None and not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be in (0, 1]")
    if tail_epsilon is not None and not 0 <= tail_epsilon < 1:
        raise ValueError("tail_epsilon must be in [0, 1)")

    batch, heads, queries, keys = base_probability.shape
    num_head_groups = math.ceil(heads / head_group_size)
    num_query_blocks = math.ceil(queries / query_block_size)
    num_key_blocks = math.ceil(keys / key_block_size)

    pad_heads = num_head_groups * head_group_size - heads
    pad_queries = num_query_blocks * query_block_size - queries
    pad_keys = num_key_blocks * key_block_size - keys
    padded = F.pad(base_probability, (0, pad_keys, 0, pad_queries, 0, pad_heads))
    block_probability = padded.reshape(
        batch,
        num_head_groups,
        head_group_size,
        num_query_blocks,
        query_block_size,
        num_key_blocks,
        key_block_size,
    ).sum(dim=(-1, 2, 4))
    block_probability = block_probability / float(head_group_size * query_block_size)

    forced_blocks = None
    if residual_key_mask is not None:
        residual_key_mask = torch.broadcast_to(
            residual_key_mask.to(device=base_probability.device, dtype=torch.bool),
            (batch, keys),
        )
        forced_blocks = F.pad(residual_key_mask, (0, pad_keys)).reshape(
            batch, num_key_blocks, key_block_size
        ).any(dim=-1)
        forced_blocks = forced_blocks[:, None, None, :].expand(
            -1, num_head_groups, num_query_blocks, -1
        )

    sorted_score, sorted_idx = torch.sort(block_probability, dim=-1, descending=True)
    if keep_ratio is not None:
        keep_count = max(1, min(num_key_blocks, math.ceil(num_key_blocks * keep_ratio)))
        selected_sorted = torch.arange(
            num_key_blocks, device=base_probability.device
        ) < keep_count
        selected_sorted = selected_sorted.view(1, 1, 1, -1).expand_as(sorted_score)
    else:
        cumulative = sorted_score.cumsum(dim=-1)
        target = (1.0 - float(tail_epsilon)) * block_probability.sum(dim=-1, keepdim=True)
        selected_sorted = cumulative - sorted_score < target
        selected_sorted[..., 0] = True

    selected_blocks = torch.zeros_like(block_probability, dtype=torch.bool)
    selected_blocks.scatter_(-1, sorted_idx, selected_sorted)
    if forced_blocks is not None:
        selected_blocks |= forced_blocks

    selected_keys = _expand_block_mask(selected_blocks, key_block_size, keys)
    selected_keys = selected_keys.repeat_interleave(head_group_size, dim=1)[:, :heads]
    selected_keys = selected_keys.repeat_interleave(query_block_size, dim=2)[:, :, :queries]

    mass = (base_probability * selected_keys).sum(dtype=torch.float64)
    full_mass = base_probability.sum(dtype=torch.float64).clamp_min(torch.finfo(torch.float64).eps)
    kept_blocks = int(selected_blocks.sum().item())
    total_blocks = selected_blocks.numel()
    stats = SupportStatistics(
        kept_fraction=float(selected_keys.float().mean().item()),
        probability_mass=float((mass / full_mass).item()),
        block_count=kept_blocks,
        total_block_count=total_blocks,
    )
    return selected_keys, stats


def silu_derivative(x: Tensor) -> Tensor:
    sigmoid = torch.sigmoid(x)
    return sigmoid * (1 + x * (1 - sigmoid))


def swiglu_jvp(
    x: Tensor,
    dx: Tensor,
    w_gate: Tensor,
    w_up: Tensor,
    w_down: Tensor,
    *,
    gate_bias: Tensor | None = None,
    up_bias: Tensor | None = None,
    channel_mask: Tensor | None = None,
) -> Tensor:
    """Dense reference for the SwiGLU first-order correction."""

    if x.shape != dx.shape:
        raise ValueError("x and dx must have identical shapes")
    gate = F.linear(x, w_gate, gate_bias)
    up = F.linear(x, w_up, up_bias)
    delta_gate = F.linear(dx, w_gate)
    delta_up = F.linear(dx, w_up)
    delta_hidden = silu_derivative(gate) * up * delta_gate + F.silu(gate) * delta_up
    if channel_mask is not None:
        try:
            channel_mask = torch.broadcast_to(
                channel_mask.to(device=x.device, dtype=torch.bool),
                delta_hidden.shape,
            )
        except RuntimeError as exc:
            raise ValueError("channel_mask cannot broadcast to the hidden shape") from exc
        delta_hidden = delta_hidden.masked_fill(~channel_mask, 0)
    return F.linear(delta_hidden, w_down)


def exact_swiglu_delta(
    x: Tensor,
    dx: Tensor,
    w_gate: Tensor,
    w_up: Tensor,
    w_down: Tensor,
    *,
    gate_bias: Tensor | None = None,
    up_bias: Tensor | None = None,
) -> Tensor:
    base_gate = F.linear(x, w_gate, gate_bias)
    base_up = F.linear(x, w_up, up_bias)
    corrected_gate = F.linear(x + dx, w_gate, gate_bias)
    corrected_up = F.linear(x + dx, w_up, up_bias)
    hidden_delta = F.silu(corrected_gate) * corrected_up - F.silu(base_gate) * base_up
    return F.linear(hidden_delta, w_down)


def swiglu_channel_energy(
    x: Tensor,
    dx: Tensor,
    w_gate: Tensor,
    w_up: Tensor,
    w_down: Tensor,
    *,
    gate_bias: Tensor | None = None,
    up_bias: Tensor | None = None,
) -> Tensor:
    gate = F.linear(x, w_gate, gate_bias)
    up = F.linear(x, w_up, up_bias)
    delta_gate = F.linear(dx, w_gate)
    delta_up = F.linear(dx, w_up)
    delta_hidden = silu_derivative(gate) * up * delta_gate + F.silu(gate) * delta_up
    down_column_norm = w_down.float().square().sum(dim=0).sqrt()
    return delta_hidden.float().abs() * down_column_norm


def swiglu_training_free_score(
    x: Tensor,
    dx: Tensor,
    w_gate: Tensor,
    w_up: Tensor,
    w_down: Tensor,
    *,
    gate_bias: Tensor | None = None,
    up_bias: Tensor | None = None,
) -> Tensor:
    """Residual/derivative/weight-norm upper-bound score from the draft state."""

    gate = F.linear(x, w_gate, gate_bias)
    up = F.linear(x, w_up, up_bias)
    dx_norm = dx.float().square().sum(dim=-1, keepdim=True).sqrt()
    gate_row_norm = w_gate.float().square().sum(dim=-1).sqrt()
    up_row_norm = w_up.float().square().sum(dim=-1).sqrt()
    down_column_norm = w_down.float().square().sum(dim=0).sqrt()
    factor = (
        (silu_derivative(gate).float() * up.float()).abs() * gate_row_norm
        + F.silu(gate).float().abs() * up_row_norm
    )
    return dx_norm * factor * down_column_norm


def select_ffn_block_support(
    channel_score: Tensor,
    *,
    keep_ratio: float,
    channel_block_size: int = 128,
    token_block_size: int = 8,
    reduce: Literal["sum", "max"] = "sum",
) -> Tensor:
    """Select contiguous FFN channel blocks shared by a token block."""

    if channel_score.ndim < 2:
        raise ValueError("channel_score must end in [tokens, channels]")
    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be in (0, 1]")
    if channel_block_size <= 0 or token_block_size <= 0:
        raise ValueError("Block sizes must be positive")
    if reduce not in {"sum", "max"}:
        raise ValueError(f"Unknown token-block reduction: {reduce}")

    leading_shape = channel_score.shape[:-2]
    tokens, channels = channel_score.shape[-2:]
    token_blocks = math.ceil(tokens / token_block_size)
    channel_blocks = math.ceil(channels / channel_block_size)
    pad_tokens = token_blocks * token_block_size - tokens
    pad_channels = channel_blocks * channel_block_size - channels
    padded = F.pad(channel_score, (0, pad_channels, 0, pad_tokens))
    blocked = padded.reshape(
        *leading_shape,
        token_blocks,
        token_block_size,
        channel_blocks,
        channel_block_size,
    ).sum(dim=-1)
    if reduce == "sum":
        block_score = blocked.sum(dim=-2)
    else:
        block_score = blocked.amax(dim=-2)

    keep_blocks = max(1, min(channel_blocks, math.ceil(channel_blocks * keep_ratio)))
    selected_idx = torch.topk(block_score, k=keep_blocks, dim=-1).indices
    selected_blocks = torch.zeros_like(block_score, dtype=torch.bool)
    selected_blocks.scatter_(-1, selected_idx, True)
    mask = _expand_block_mask(selected_blocks, channel_block_size, channels)
    mask = mask.unsqueeze(-2).expand(*leading_shape, token_blocks, token_block_size, channels)
    return mask.reshape(*leading_shape, token_blocks * token_block_size, channels)[..., :tokens, :]


def relative_l2_error(actual: Tensor, reference: Tensor, eps: float = 1e-12) -> float:
    numerator = (actual.float() - reference.float()).norm()
    denominator = reference.float().norm().clamp_min(eps)
    return float((numerator / denominator).item())


def gap_recovery(delta: Tensor, exact_delta: Tensor, eps: float = 1e-12) -> float:
    error = (delta.float() - exact_delta.float()).norm()
    gap = exact_delta.float().norm().clamp_min(eps)
    return float((1 - error / gap).item())
