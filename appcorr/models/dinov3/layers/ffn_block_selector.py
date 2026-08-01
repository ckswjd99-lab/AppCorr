"""Training-free block selectors for exact SwiGLU correction.

The low-rank approximation in this module is used only to choose structured
FFN channel support.  Values propagated through the model are always computed
from the original gate, up, and down projection weights.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class LowRankSwiGLUFactors:
    """One shared input basis and gate/up coefficients for a SwiGLU layer."""

    input_basis: Tensor
    gate_coefficient: Tensor
    up_coefficient: Tensor

    @property
    def rank(self) -> int:
        return int(self.input_basis.shape[1])

    def truncate(self, rank: int) -> "LowRankSwiGLUFactors":
        if rank <= 0 or rank > self.rank:
            raise ValueError(f"rank must be in [1, {self.rank}], got {rank}")
        return LowRankSwiGLUFactors(
            input_basis=self.input_basis[:, :rank].contiguous(),
            gate_coefficient=self.gate_coefficient[:, :rank].contiguous(),
            up_coefficient=self.up_coefficient[:, :rank].contiguous(),
        )


@torch.no_grad()
def build_joint_swiglu_low_rank_factors(
    gate_weight: Tensor,
    up_weight: Tensor,
    *,
    rank: int,
    oversample: int = 8,
    power_iterations: int = 1,
    seed: int = 0,
) -> LowRankSwiGLUFactors:
    """Find the leading input subspace shared by gate and up projections.

    Randomized subspace iteration approximates the leading eigenspace of
    ``W_gate.T W_gate + W_up.T W_up`` without materializing that large Gram
    matrix.  Factors are model-static and may be built once at model load.
    """

    if gate_weight.ndim != 2 or gate_weight.shape != up_weight.shape:
        raise ValueError("gate_weight and up_weight must share [channels, hidden]")
    channels, hidden = gate_weight.shape
    if rank <= 0 or rank > min(channels, hidden):
        raise ValueError("rank is outside the valid matrix dimensions")
    if oversample < 0 or power_iterations < 0:
        raise ValueError("oversample and power_iterations must be non-negative")

    sketch_rank = min(rank + oversample, channels, hidden)
    generator = torch.Generator(device=gate_weight.device).manual_seed(seed)
    basis = torch.randn(
        hidden,
        sketch_rank,
        generator=generator,
        device=gate_weight.device,
        dtype=gate_weight.dtype,
    )
    basis = torch.linalg.qr(basis.float(), mode="reduced").Q.to(gate_weight.dtype)

    for _ in range(power_iterations + 1):
        gate_range = gate_weight @ basis
        up_range = up_weight @ basis
        next_basis = (
            gate_weight.transpose(0, 1) @ gate_range
            + up_weight.transpose(0, 1) @ up_range
        )
        basis = torch.linalg.qr(next_basis.float(), mode="reduced").Q.to(
            gate_weight.dtype
        )

    gate_range = gate_weight @ basis
    up_range = up_weight @ basis
    small_gram = (
        gate_range.float().transpose(0, 1) @ gate_range.float()
        + up_range.float().transpose(0, 1) @ up_range.float()
    )
    _, eigenvectors = torch.linalg.eigh(small_gram)
    rotation = eigenvectors[:, -rank:].flip(dims=(-1,))
    input_basis = (basis.float() @ rotation).to(gate_weight.dtype).contiguous()
    return LowRankSwiGLUFactors(
        input_basis=input_basis,
        gate_coefficient=(gate_weight @ input_basis).contiguous(),
        up_coefficient=(up_weight @ input_basis).contiguous(),
    )


def project_swiglu_input(x: Tensor, factors: LowRankSwiGLUFactors) -> Tensor:
    if x.shape[-1] != factors.input_basis.shape[0]:
        raise ValueError("input hidden size does not match the selector basis")
    return x @ factors.input_basis


def _weighted_hidden_delta_score(hidden_delta: Tensor, down_weight: Tensor) -> Tensor:
    if hidden_delta.shape[-1] != down_weight.shape[1]:
        raise ValueError("hidden delta and down projection channel sizes differ")
    down_energy = down_weight.float().square().sum(dim=0)
    return hidden_delta.float().square() * down_energy


def low_rank_swiglu_channel_score(
    base_x: Tensor,
    corrected_x: Tensor,
    factors: LowRankSwiGLUFactors,
    down_weight: Tensor,
    *,
    gate_bias: Tensor | None = None,
    up_bias: Tensor | None = None,
    base_projected: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Predict output-weighted exact finite-difference channel energy.

    Returns the channel score and the projected base state that an approximate
    pass would cache.  The predicted values are never propagated as features.
    """

    if base_projected is None:
        base_projected = project_swiglu_input(base_x, factors)
    corrected_projected = project_swiglu_input(corrected_x, factors)
    base_gate = F.linear(base_projected, factors.gate_coefficient, gate_bias)
    corrected_gate = F.linear(
        corrected_projected,
        factors.gate_coefficient,
        gate_bias,
    )
    base_up = F.linear(base_projected, factors.up_coefficient, up_bias)
    corrected_up = F.linear(
        corrected_projected,
        factors.up_coefficient,
        up_bias,
    )
    hidden_delta = (
        F.silu(corrected_gate) * corrected_up
        - F.silu(base_gate) * base_up
    )
    return _weighted_hidden_delta_score(hidden_delta, down_weight), base_projected


def oracle_swiglu_channel_score(
    base_x: Tensor,
    corrected_x: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
    *,
    gate_bias: Tensor | None = None,
    up_bias: Tensor | None = None,
) -> Tensor:
    """Compute the exact hidden finite-difference score for diagnostics."""

    base_gate = F.linear(base_x, gate_weight, gate_bias)
    corrected_gate = F.linear(corrected_x, gate_weight, gate_bias)
    base_up = F.linear(base_x, up_weight, up_bias)
    corrected_up = F.linear(corrected_x, up_weight, up_bias)
    hidden_delta = (
        F.silu(corrected_gate) * corrected_up
        - F.silu(base_gate) * base_up
    )
    return _weighted_hidden_delta_score(hidden_delta, down_weight)


def static_swiglu_channel_score(
    reference: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
) -> Tensor:
    """Input-independent weight sensitivity broadcast over tokens."""

    gate_energy = gate_weight.float().square().sum(dim=1)
    up_energy = up_weight.float().square().sum(dim=1)
    down_energy = down_weight.float().square().sum(dim=0)
    score = (gate_energy + up_energy) * down_energy
    return score.view(*([1] * (reference.ndim - 1)), -1).expand(
        *reference.shape[:-1],
        score.numel(),
    )


def select_ffn_block_mask(
    channel_score: Tensor,
    *,
    keep_ratio: float,
    token_block_size: int = 8,
    channel_block_size: int = 128,
) -> tuple[Tensor, Tensor]:
    """Select a fixed number of contiguous channel blocks per token block.

    Returns an expanded ``[B,T,C]`` mask and the unexpanded block scores
    ``[B,ceil(T/TB),ceil(C/CB)]``.
    """

    if channel_score.ndim != 3:
        raise ValueError("channel_score must have shape [batch, tokens, channels]")
    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be in (0, 1]")
    if token_block_size <= 0 or channel_block_size <= 0:
        raise ValueError("block sizes must be positive")

    batch, tokens, channels = channel_score.shape
    token_blocks = math.ceil(tokens / token_block_size)
    channel_blocks = math.ceil(channels / channel_block_size)
    padded = F.pad(
        channel_score,
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
    block_score = blocked.sum(dim=(2, 4))
    keep_blocks = max(1, min(channel_blocks, math.ceil(channel_blocks * keep_ratio)))
    selected = torch.topk(block_score, k=keep_blocks, dim=-1).indices
    block_mask = torch.zeros_like(block_score, dtype=torch.bool)
    block_mask.scatter_(-1, selected, True)
    expanded = (
        block_mask[:, :, None, :, None]
        .expand(
            batch,
            token_blocks,
            token_block_size,
            channel_blocks,
            channel_block_size,
        )
        .reshape(batch, token_blocks * token_block_size, channel_blocks * channel_block_size)
    )
    return expanded[:, :tokens, :channels], block_score


def select_ffn_2to4_mask(channel_score: Tensor) -> Tensor:
    """Keep the two highest-score channels in every contiguous group of four."""

    if channel_score.ndim != 3:
        raise ValueError("channel_score must have shape [batch, tokens, channels]")
    channels = channel_score.shape[-1]
    groups = math.ceil(channels / 4)
    padded = F.pad(channel_score, (0, groups * 4 - channels))
    grouped = padded.reshape(*channel_score.shape[:-1], groups, 4)
    selected = torch.topk(grouped, k=2, dim=-1).indices
    mask = torch.zeros_like(grouped, dtype=torch.bool)
    mask.scatter_(-1, selected, True)
    return mask.reshape(*channel_score.shape[:-1], groups * 4)[..., :channels]


def select_ffn_row_topk_mask(
    channel_score: Tensor,
    *,
    keep_ratio: float = 0.5,
) -> Tensor:
    """Select an unconstrained top fraction independently for every token row."""

    if channel_score.ndim != 3:
        raise ValueError("channel_score must have shape [batch, tokens, channels]")
    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be in (0, 1]")
    channels = channel_score.shape[-1]
    keep = max(1, min(channels, math.ceil(channels * keep_ratio)))
    selected = torch.topk(channel_score, k=keep, dim=-1).indices
    mask = torch.zeros_like(channel_score, dtype=torch.bool)
    mask.scatter_(-1, selected, True)
    return mask


def exact_swiglu_delta_selected_blocks(
    base_x: Tensor,
    corrected_x: Tensor,
    gate_weight: Tensor,
    up_weight: Tensor,
    down_weight: Tensor,
    channel_mask: Tensor,
    *,
    gate_bias: Tensor | None = None,
    up_bias: Tensor | None = None,
    token_block_size: int = 8,
) -> Tensor:
    """Compute exact finite differences using only the selected weight rows.

    This reference implementation uses small dense GEMMs for each structured
    block.  It preserves the arithmetic of the intended sparse algorithm and
    deliberately avoids caching full base gate/up activations.
    """

    if base_x.shape != corrected_x.shape or base_x.ndim != 3:
        raise ValueError("base_x and corrected_x must share [batch,tokens,hidden]")
    if channel_mask.shape != (*base_x.shape[:-1], gate_weight.shape[0]):
        raise ValueError("channel_mask shape does not match the FFN channels")

    batch, tokens, _ = base_x.shape
    output = torch.zeros(
        batch,
        tokens,
        down_weight.shape[0],
        device=base_x.device,
        dtype=base_x.dtype,
    )
    for batch_index in range(batch):
        for token_start in range(0, tokens, token_block_size):
            token_end = min(token_start + token_block_size, tokens)
            block_mask = channel_mask[batch_index, token_start]
            if not torch.equal(
                channel_mask[batch_index, token_start:token_end],
                block_mask.expand(token_end - token_start, -1),
            ):
                raise ValueError("channel support must be shared within each token block")
            selected = block_mask.nonzero(as_tuple=False).flatten()
            if selected.numel() == 0:
                continue
            base_chunk = base_x[batch_index, token_start:token_end]
            corrected_chunk = corrected_x[batch_index, token_start:token_end]
            gate_selected = gate_weight.index_select(0, selected)
            up_selected = up_weight.index_select(0, selected)
            down_selected = down_weight.index_select(1, selected)
            gate_bias_selected = (
                None if gate_bias is None else gate_bias.index_select(0, selected)
            )
            up_bias_selected = (
                None if up_bias is None else up_bias.index_select(0, selected)
            )
            base_hidden = F.silu(
                F.linear(base_chunk, gate_selected, gate_bias_selected)
            ) * F.linear(base_chunk, up_selected, up_bias_selected)
            corrected_hidden = F.silu(
                F.linear(corrected_chunk, gate_selected, gate_bias_selected)
            ) * F.linear(corrected_chunk, up_selected, up_bias_selected)
            output[batch_index, token_start:token_end] = F.linear(
                corrected_hidden - base_hidden,
                down_selected,
                bias=None,
            )
    return output


def mask_diagnostics(
    predicted_mask: Tensor,
    oracle_mask: Tensor,
    oracle_channel_score: Tensor,
) -> dict[str, float]:
    """Compare a predicted mask with the equal-work exact-score oracle."""

    predicted = predicted_mask.bool()
    oracle = oracle_mask.bool()
    intersection = (predicted & oracle).sum().float()
    oracle_count = oracle.sum().clamp_min(1).float()
    total_energy = oracle_channel_score.double().sum().clamp_min(1e-30)
    predicted_energy = oracle_channel_score.masked_fill(~predicted, 0).double().sum()
    oracle_energy = oracle_channel_score.masked_fill(~oracle, 0).double().sum()
    return {
        "oracle_block_recall": float((intersection / oracle_count).item()),
        "total_energy_retained": float((predicted_energy / total_energy).item()),
        "oracle_energy_retained": float((oracle_energy / total_energy).item()),
        "retained_energy_vs_oracle": float(
            (predicted_energy / oracle_energy.clamp_min(1e-30)).item()
        ),
    }
