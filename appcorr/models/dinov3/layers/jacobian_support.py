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


@dataclass(frozen=True)
class LowRankSwiGLUFactors:
    """Joint right-subspace approximation for SwiGLU gate/up projections."""

    input_basis: Tensor
    gate_coefficient: Tensor
    up_coefficient: Tensor

    @property
    def rank(self) -> int:
        return int(self.input_basis.shape[-1])


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


def attention_block_index_from_mask(
    support_mask: Tensor,
    *,
    key_block_size: int,
    query_block_size: int,
    head_group_size: int,
) -> Tensor:
    """Compress an expanded structured attention mask into block indices.

    Descriptors with fewer selected blocks are padded with ``-1``.  This occurs
    when a fixed top-k support is unioned with forced key blocks that may
    already be present in some query/head descriptors.
    """

    if support_mask.ndim != 4:
        raise ValueError("support_mask must be [B,H,Q,K]")
    if min(key_block_size, query_block_size, head_group_size) <= 0:
        raise ValueError("block sizes must be positive")
    batch, heads, queries, keys = support_mask.shape
    key_blocks = math.ceil(keys / key_block_size)
    padded = F.pad(
        support_mask.to(dtype=torch.bool),
        (0, key_blocks * key_block_size - keys),
    )
    blocked = padded.reshape(
        batch,
        heads,
        queries,
        key_blocks,
        key_block_size,
    ).any(dim=-1)
    head_start = torch.arange(
        0,
        heads,
        head_group_size,
        device=support_mask.device,
    )
    query_start = torch.arange(
        0,
        queries,
        query_block_size,
        device=support_mask.device,
    )
    descriptor_mask = blocked.index_select(1, head_start).index_select(
        2,
        query_start,
    )
    counts = descriptor_mask.sum(dim=-1)
    selected_blocks = int(counts.max().item())
    if selected_blocks <= 0:
        raise ValueError("At least one key block must be selected")
    block_ids = torch.arange(
        key_blocks,
        device=support_mask.device,
        dtype=torch.int64,
    ).view(1, 1, 1, -1)
    padded_ids = torch.where(descriptor_mask, block_ids, key_blocks)
    selected_idx = padded_ids.sort(dim=-1).values[..., :selected_blocks]
    selected_idx = torch.where(selected_idx == key_blocks, -1, selected_idx)
    return selected_idx.to(torch.int32)


def ffn_block_index_from_mask(
    channel_mask: Tensor,
    *,
    channel_block_size: int,
    token_block_size: int,
) -> Tensor:
    """Compress a token/channel-block mask into ragged channel-block indices."""

    if channel_mask.ndim != 3:
        raise ValueError("channel_mask must be [B,T,C]")
    if min(channel_block_size, token_block_size) <= 0:
        raise ValueError("block sizes must be positive")
    batch, tokens, channels = channel_mask.shape
    token_blocks = math.ceil(tokens / token_block_size)
    channel_blocks = math.ceil(channels / channel_block_size)
    padded = F.pad(
        channel_mask.to(dtype=torch.bool),
        (
            0,
            channel_blocks * channel_block_size - channels,
            0,
            token_blocks * token_block_size - tokens,
        ),
    )
    blocked = padded.reshape(
        batch,
        token_blocks,
        token_block_size,
        channel_blocks,
        channel_block_size,
    )
    selected_blocks = blocked.any(dim=-1).any(dim=2)
    reconstructed = (
        selected_blocks[:, :, None, :, None]
        .expand_as(blocked)
        .reshape_as(padded)
    )
    if not torch.equal(
        reconstructed[:, :tokens, :channels],
        channel_mask.to(dtype=torch.bool),
    ):
        raise ValueError(
            "channel_mask must select complete channel blocks shared within token blocks"
        )

    counts = selected_blocks.sum(dim=-1)
    max_selected = int(counts.max().item())
    if max_selected <= 0:
        raise ValueError("At least one channel block must be selected")
    block_ids = torch.arange(
        channel_blocks,
        device=channel_mask.device,
        dtype=torch.int64,
    ).view(1, 1, -1)
    padded_ids = torch.where(selected_blocks, block_ids, channel_blocks)
    selected_idx = padded_ids.sort(dim=-1).values[..., :max_selected]
    selected_idx = torch.where(
        selected_idx == channel_blocks,
        -1,
        selected_idx,
    )
    return selected_idx.to(torch.int32)


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


@torch.no_grad()
def build_joint_swiglu_low_rank_factors(
    w_gate: Tensor,
    w_up: Tensor,
    *,
    rank: int,
    oversample: int = 8,
    power_iterations: int = 1,
    seed: int = 0,
) -> LowRankSwiGLUFactors:
    """Approximate gate/up weights in one shared low-rank input subspace.

    The basis approximates the leading eigenspace of
    ``W_gate.T W_gate + W_up.T W_up``.  It is model-static and can therefore
    be built offline or once at model load.  The approximate forward caches
    only ``x @ input_basis`` for each token.
    """

    if w_gate.ndim != 2 or w_up.ndim != 2 or w_gate.shape != w_up.shape:
        raise ValueError("w_gate and w_up must have the same [channels, hidden] shape")
    channels, hidden = w_gate.shape
    if rank <= 0 or rank > min(channels, hidden):
        raise ValueError("rank must be in [1, min(channels, hidden)]")
    if oversample < 0:
        raise ValueError("oversample must be non-negative")
    if power_iterations < 0:
        raise ValueError("power_iterations must be non-negative")

    sketch_rank = min(rank + oversample, channels, hidden)
    generator = torch.Generator(device=w_gate.device)
    generator.manual_seed(seed)
    basis = torch.randn(
        hidden,
        sketch_rank,
        generator=generator,
        device=w_gate.device,
        dtype=w_gate.dtype,
    )
    basis = torch.linalg.qr(basis.float(), mode="reduced").Q.to(w_gate.dtype)

    # Subspace iteration reads the two model weights but never materializes a
    # hidden-by-hidden covariance matrix.
    for _ in range(power_iterations + 1):
        gate_range = F.linear(basis.transpose(0, 1), w_gate).transpose(0, 1)
        up_range = F.linear(basis.transpose(0, 1), w_up).transpose(0, 1)
        next_basis = (
            w_gate.transpose(0, 1) @ gate_range
            + w_up.transpose(0, 1) @ up_range
        )
        basis = torch.linalg.qr(next_basis.float(), mode="reduced").Q.to(
            w_gate.dtype
        )

    gate_range = w_gate @ basis
    up_range = w_up @ basis
    gram = (
        gate_range.float().transpose(0, 1) @ gate_range.float()
        + up_range.float().transpose(0, 1) @ up_range.float()
    )
    # The caller commonly runs under BF16 autocast; CUDA eigh requires FP32.
    _, eigenvectors = torch.linalg.eigh(gram.float())
    rotation = eigenvectors[:, -rank:].flip(dims=(-1,))
    input_basis = (basis.float() @ rotation).to(w_gate.dtype).contiguous()
    return LowRankSwiGLUFactors(
        input_basis=input_basis,
        gate_coefficient=(w_gate @ input_basis).contiguous(),
        up_coefficient=(w_up @ input_basis).contiguous(),
    )


def project_swiglu_low_rank(
    x: Tensor,
    factors: LowRankSwiGLUFactors,
) -> Tensor:
    """Project token states into the model-static low-rank input space."""

    if x.shape[-1] != factors.input_basis.shape[0]:
        raise ValueError("x hidden size does not match the low-rank basis")
    return x @ factors.input_basis


def low_rank_swiglu_channel_score(
    base_projected: Tensor,
    corrected_x: Tensor,
    factors: LowRankSwiGLUFactors,
    w_down: Tensor,
    *,
    gate_bias: Tensor | None = None,
    up_bias: Tensor | None = None,
) -> Tensor:
    """Predict exact finite-difference channel energy from cached draft state."""

    corrected_projected = project_swiglu_low_rank(corrected_x, factors)
    base_gate = F.linear(
        base_projected,
        factors.gate_coefficient,
        gate_bias,
    )
    base_up = F.linear(
        base_projected,
        factors.up_coefficient,
        up_bias,
    )
    corrected_gate = F.linear(
        corrected_projected,
        factors.gate_coefficient,
        gate_bias,
    )
    corrected_up = F.linear(
        corrected_projected,
        factors.up_coefficient,
        up_bias,
    )
    hidden_delta = (
        F.silu(corrected_gate) * corrected_up
        - F.silu(base_gate) * base_up
    )
    down_column_norm = w_down.float().square().sum(dim=0).sqrt()
    return hidden_delta.float().abs() * down_column_norm


def exact_swiglu_delta_selected_blocks(
    base_gate: Tensor,
    base_up: Tensor,
    corrected_x: Tensor,
    w_gate: Tensor,
    w_up: Tensor,
    w_down: Tensor,
    channel_mask: Tensor,
    *,
    gate_bias: Tensor | None = None,
    up_bias: Tensor | None = None,
    token_block_size: int = 8,
) -> Tensor:
    """Evaluate exact finite differences only for selected channel blocks.

    ``base_gate`` and ``base_up`` are outputs already produced by the
    approximate FFN and cached for the active tokens.  The selector is shared
    within each token block, so each loop body maps directly to one structured
    gate/up/down GEMM workload in the optimized implementation.
    """

    if corrected_x.ndim != 3:
        raise ValueError("corrected_x must have shape [batch, tokens, hidden]")
    batch, tokens, _ = corrected_x.shape
    channels = w_gate.shape[0]
    if base_gate.shape != (batch, tokens, channels):
        raise ValueError("base_gate shape does not match corrected_x/w_gate")
    if base_up.shape != base_gate.shape:
        raise ValueError("base_up must match base_gate")
    if channel_mask.shape != base_gate.shape:
        raise ValueError("channel_mask must match base_gate")
    if w_up.shape != w_gate.shape or w_down.shape[1] != channels:
        raise ValueError("SwiGLU weight shapes are inconsistent")
    if token_block_size <= 0:
        raise ValueError("token_block_size must be positive")

    output = corrected_x.new_zeros(batch, tokens, w_down.shape[0])
    for batch_index in range(batch):
        for token_start in range(0, tokens, token_block_size):
            token_end = min(tokens, token_start + token_block_size)
            block_mask = channel_mask[
                batch_index,
                token_start:token_end,
            ]
            shared_mask = block_mask[0]
            if not torch.equal(
                block_mask,
                shared_mask.unsqueeze(0).expand_as(block_mask),
            ):
                raise ValueError("channel support must be shared within a token block")
            selected = shared_mask.nonzero(as_tuple=False).flatten()
            if selected.numel() == 0:
                continue
            corrected_block = corrected_x[
                batch_index,
                token_start:token_end,
            ]
            corrected_gate = F.linear(
                corrected_block,
                w_gate.index_select(0, selected),
                None if gate_bias is None else gate_bias.index_select(0, selected),
            )
            corrected_up = F.linear(
                corrected_block,
                w_up.index_select(0, selected),
                None if up_bias is None else up_bias.index_select(0, selected),
            )
            base_hidden = (
                F.silu(
                    base_gate[
                        batch_index,
                        token_start:token_end,
                    ].index_select(-1, selected)
                )
                * base_up[
                    batch_index,
                    token_start:token_end,
                ].index_select(-1, selected)
            )
            hidden_delta = F.silu(corrected_gate) * corrected_up - base_hidden
            output[batch_index, token_start:token_end] = F.linear(
                hidden_delta,
                w_down.index_select(1, selected),
            )
    return output


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
