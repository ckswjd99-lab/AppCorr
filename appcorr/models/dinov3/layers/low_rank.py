"""Low-rank linear factors for approximate correction deltas.

The approximate pass continues to use the original dense weight.  These
factors are only used for the bias-free correction term

    delta_y = delta_x @ W.T ~= (delta_x @ B.T) @ A.T

where ``W ~= A @ B``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class LowRankLinearFactors:
    """Nested rank factors produced from one maximum-rank decomposition."""

    left: Tensor
    right: Tensor
    singular_values: Tensor
    weight_frobenius_norm: float

    def __post_init__(self) -> None:
        if self.left.ndim != 2 or self.right.ndim != 2:
            raise ValueError("low-rank factors must be matrices")
        if self.left.shape[1] != self.right.shape[0]:
            raise ValueError(
                "factor rank mismatch: "
                f"left={tuple(self.left.shape)} right={tuple(self.right.shape)}"
            )
        if self.singular_values.shape != (self.left.shape[1],):
            raise ValueError(
                "singular_values must have one entry per factor column, got "
                f"{tuple(self.singular_values.shape)} for rank {self.left.shape[1]}"
            )

    @property
    def max_rank(self) -> int:
        return int(self.left.shape[1])

    @property
    def in_features(self) -> int:
        return int(self.right.shape[1])

    @property
    def out_features(self) -> int:
        return int(self.left.shape[0])

    def apply(self, delta_x: Tensor, rank: int | None = None) -> Tensor:
        """Apply the rank-``rank`` correction using two bias-free linears."""

        selected_rank = self.max_rank if rank is None else int(rank)
        if selected_rank <= 0 or selected_rank > self.max_rank:
            raise ValueError(
                f"rank must be in [1, {self.max_rank}], got {selected_rank}"
            )
        if delta_x.shape[-1] != self.in_features:
            raise ValueError(
                f"expected input width {self.in_features}, got {delta_x.shape[-1]}"
            )
        reduced = F.linear(delta_x, self.right[:selected_rank])
        return F.linear(reduced, self.left[:, :selected_rank])

    def factor_bytes(self, rank: int | None = None) -> int:
        selected_rank = self.max_rank if rank is None else int(rank)
        if selected_rank <= 0 or selected_rank > self.max_rank:
            raise ValueError(
                f"rank must be in [1, {self.max_rank}], got {selected_rank}"
            )
        left_elements = self.out_features * selected_rank
        right_elements = selected_rank * self.in_features
        return int(
            (left_elements * self.left.element_size())
            + (right_elements * self.right.element_size())
        )

    def spectral_energy_fraction(self, rank: int | None = None) -> float:
        selected_rank = self.max_rank if rank is None else int(rank)
        if selected_rank <= 0 or selected_rank > self.max_rank:
            raise ValueError(
                f"rank must be in [1, {self.max_rank}], got {selected_rank}"
            )
        denominator = max(self.weight_frobenius_norm**2, torch.finfo(torch.float64).eps)
        numerator = self.singular_values[:selected_rank].double().square().sum().item()
        return float(numerator / denominator)


@torch.no_grad()
def factorize_linear_weight(
    weight: Tensor,
    max_rank: int,
    *,
    oversample: int = 16,
    power_iterations: int = 1,
    factor_dtype: torch.dtype | None = None,
    exact: bool = False,
) -> LowRankLinearFactors:
    """Factorize an ``[out_features, in_features]`` linear weight.

    ``exact=False`` uses randomized SVD and is intended for the 7B DINOv3
    matrices.  ``exact=True`` is useful for small numerical tests.
    Singular values are split symmetrically between the two factors to avoid
    putting the complete scale in either GEMM.
    """

    if weight.ndim != 2:
        raise ValueError(f"weight must be a matrix, got {tuple(weight.shape)}")
    limit = min(weight.shape)
    if max_rank <= 0 or max_rank > limit:
        raise ValueError(f"max_rank must be in [1, {limit}], got {max_rank}")
    if oversample < 0:
        raise ValueError(f"oversample must be non-negative, got {oversample}")
    if power_iterations < 0:
        raise ValueError(
            f"power_iterations must be non-negative, got {power_iterations}"
        )

    if exact and weight.dtype in (torch.float32, torch.float64):
        work_weight = weight.detach()
    else:
        work_weight = weight.detach().float()
    weight_norm = float(torch.linalg.vector_norm(work_weight).item())
    if exact:
        u, singular_values, vh = torch.linalg.svd(
            work_weight,
            full_matrices=False,
        )
        u = u[:, :max_rank]
        singular_values = singular_values[:max_rank]
        v = vh[:max_rank].transpose(0, 1)
    else:
        sketch_rank = min(limit, max_rank + oversample)
        u, singular_values, v = torch.svd_lowrank(
            work_weight,
            q=sketch_rank,
            niter=power_iterations,
        )
        order = singular_values.argsort(descending=True)[:max_rank]
        u = u[:, order]
        singular_values = singular_values[order]
        v = v[:, order]

    sqrt_s = singular_values.clamp_min(0).sqrt()
    storage_dtype = factor_dtype or weight.dtype
    left = (u * sqrt_s.unsqueeze(0)).to(dtype=storage_dtype).contiguous()
    right = (
        v * sqrt_s.unsqueeze(0)
    ).transpose(0, 1).to(dtype=storage_dtype).contiguous()
    return LowRankLinearFactors(
        left=left,
        right=right,
        singular_values=singular_values.detach().cpu(),
        weight_frobenius_norm=weight_norm,
    )


@torch.no_grad()
def factorize_linear_weight_activation_aware(
    weight: Tensor,
    input_rms: Tensor,
    max_rank: int,
    *,
    oversample: int = 16,
    power_iterations: int = 1,
    factor_dtype: torch.dtype | None = None,
    rms_floor_ratio: float = 1e-3,
    exact: bool = False,
) -> LowRankLinearFactors:
    """Factorize a weight under a diagonal correction-input covariance.

    With per-input-channel RMS ``s``, the calibration objective is

        ||(W - W_r) diag(s)||_F^2.

    We therefore factorize ``W diag(s)`` and fold ``diag(s)^-1`` into the
    right factor. Multiplying every RMS by one common constant leaves ``W_r``
    unchanged, so RMS values are normalized before SVD for numerical stability.
    """

    if input_rms.ndim != 1 or input_rms.shape[0] != weight.shape[1]:
        raise ValueError(
            "input_rms must have one value per input feature, got "
            f"{tuple(input_rms.shape)} for weight {tuple(weight.shape)}"
        )
    if rms_floor_ratio <= 0:
        raise ValueError(
            f"rms_floor_ratio must be positive, got {rms_floor_ratio}"
        )
    if not torch.isfinite(input_rms).all():
        raise ValueError("input_rms contains non-finite values")

    work_dtype = (
        weight.dtype
        if exact and weight.dtype in (torch.float32, torch.float64)
        else torch.float32
    )
    scale = input_rms.detach().to(device=weight.device, dtype=work_dtype)
    positive = scale[scale > 0]
    if positive.numel() == 0:
        raise ValueError("input_rms must contain at least one positive value")
    reference = positive.mean()
    floor = reference * rms_floor_ratio
    scale = scale.clamp_min(floor) / reference
    scaled_weight = weight.detach().to(dtype=work_dtype) * scale.unsqueeze(0)
    scaled_factors = factorize_linear_weight(
        scaled_weight,
        max_rank,
        oversample=oversample,
        power_iterations=power_iterations,
        factor_dtype=torch.float32,
        exact=exact,
    )
    storage_dtype = factor_dtype or weight.dtype
    return LowRankLinearFactors(
        left=scaled_factors.left.to(dtype=storage_dtype).contiguous(),
        right=(
            scaled_factors.right / scale.unsqueeze(0)
        ).to(dtype=storage_dtype).contiguous(),
        singular_values=scaled_factors.singular_values,
        weight_frobenius_norm=scaled_factors.weight_frobenius_norm,
    )


def dense_linear_flop_ratio(
    in_features: int,
    out_features: int,
    rank: int,
) -> float:
    """Return two-factor FLOPs divided by one dense linear's FLOPs."""

    if min(in_features, out_features, rank) <= 0:
        raise ValueError("dimensions and rank must be positive")
    return float(rank * (in_features + out_features) / (in_features * out_features))
