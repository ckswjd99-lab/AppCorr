"""FP4 with the block-scale granularity dialled down, for accuracy studies only.

NVFP4's 16-element block scale cannot be *removed*: `torch._scaled_mm`'s FP4 path takes the scale
tensor as a required operand and there is no "no scale" encoding. What can be done is make every
block share the same scale, which is numerically identical to having no per-block scaling at all.
This module sweeps that axis -- block16 (real NVFP4) through row to a single scale per tensor -- to
answer how much of FP4's survivability comes from the block scales rather than from the 4-bit format.

**The sweep moves the activation scale only; weights stay at block16.** Weight scales are computed
once offline and stored, so coarsening them saves nothing at inference and would only confound the
result. Activation scales are what gets recomputed on every call -- the cost worth attacking -- and
activations are where the outliers that a block scale defends against actually live.

**This is fake quantization**: values are quantized and immediately dequantized, then multiplied in
BF16. That is deliberate. The question is numerical, `_scaled_mm` cannot express the coarser
granularities anyway, and a real packed kernel would add a swizzle-layout bug surface for no gain on
a question about accuracy. Nothing here is a latency path and nothing here should become one.

Faithfulness to the real path, so the sweep can be trusted:

- The grid is E2M1 -- {0, .5, 1, 1.5, 2, 3, 4, 6} and negatives, 6 being the max magnitude.
- Scales are `amax / 6` **rounded to E4M3**, because that is the type NVFP4 stores a block scale in.
  Keeping that rounding at every granularity means the sweep isolates granularity, not scale dtype.
- Groups run along the contraction dimension (`in_features` for a weight, K for an activation),
  matching how NVFP4 lays its blocks out.
- Both operands are quantized, as in the real path.

One known deviation: rounding is round-half-away-from-zero, where NVFP4's hardware converter is
round-to-nearest-even. It shifts exact ties only. The `block16` arm exists to bound the total of all
such deviations -- it should land on the real kernel's number, and if it does not, nothing else in
the sweep is trustworthy either.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# E2M1: the representable magnitudes of an FP4 element.
_E2M1_LEVELS = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)
_E2M1_MAX = 6.0

GRANULARITIES = ("block16", "block32", "block64", "block256", "block1024", "row", "tensor")


def _group_size(granularity: str, k: int) -> int:
    """Elements per scale along the contraction dim; `k` means one scale per row."""
    if granularity == "row":
        return k
    if granularity == "tensor":
        return -1  # sentinel: one scale for the whole tensor
    if granularity.startswith("block"):
        gs = int(granularity[len("block") :])
        if gs <= 0 or k % gs:
            raise ValueError(f"granularity {granularity!r} does not divide K={k}")
        return gs
    raise ValueError(f"unknown granularity {granularity!r}, expected one of {GRANULARITIES}")


def _round_to_e2m1(u: torch.Tensor) -> torch.Tensor:
    """Round |u| onto the E2M1 magnitude grid, preserving sign. `u` is already scale-normalised."""
    boundaries = torch.tensor(
        [(_E2M1_LEVELS[i] + _E2M1_LEVELS[i + 1]) / 2 for i in range(len(_E2M1_LEVELS) - 1)],
        device=u.device,
        dtype=torch.float32,
    )
    levels = torch.tensor(_E2M1_LEVELS, device=u.device, dtype=torch.float32)
    mag = u.abs().to(torch.float32)
    idx = torch.bucketize(mag, boundaries)
    return torch.sign(u).to(torch.float32) * levels[idx]


def fake_quantize_fp4(x: torch.Tensor, granularity: str) -> torch.Tensor:
    """Quantize-dequantize `x` to FP4 with scales at the requested granularity.

    Returns a tensor of `x`'s dtype holding only values representable as
    `E2M1_level * E4M3_scale` within each group.
    """
    if granularity == "kernel":
        raise ValueError("'kernel' is the real path, not an emulated granularity")
    orig_dtype, orig_shape = x.dtype, x.shape
    k = orig_shape[-1]
    gs = _group_size(granularity, k)

    flat = x.reshape(-1, k).to(torch.float32)
    grouped = flat.reshape(1, -1) if gs < 0 else flat.reshape(-1, gs)

    amax = grouped.abs().amax(dim=-1, keepdim=True)
    # A group that is entirely zero has no scale to speak of; 1.0 leaves its zeros as zeros instead
    # of producing NaN through a division by zero.
    scale = torch.where(amax > 0, amax / _E2M1_MAX, torch.ones_like(amax))
    # NVFP4 keeps the block scale in E4M3. Round through it so the sweep varies granularity alone.
    scale = scale.to(torch.float8_e4m3fn).to(torch.float32)
    # E4M3's smallest normal is ~2**-9; anything that flushed to zero would divide by zero below.
    scale = torch.where(scale > 0, scale, torch.ones_like(scale))

    q = _round_to_e2m1(grouped / scale) * scale
    return q.reshape(-1, k).reshape(orig_shape).to(orig_dtype)


def fake_quantize_ternary(x: torch.Tensor, granularity: str) -> torch.Tensor:
    """Quantize-dequantize `x` to {-a, 0, +a} per group, TWN-style.

    Ternary is given its best shot deliberately: a strawman would prove nothing about whether the
    format is survivable. Naively reusing `amax/1` as the scale sends almost every entry to zero, so
    this uses Ternary Weight Networks' rule instead -- threshold `0.7 * mean|x|`, and the surviving
    magnitude `a` set to the mean of the entries above it, which is the least-squares optimum for a
    fixed threshold. Groups follow the same axis and sizes as the FP4 path so the two differ only in
    the value grid: 8 levels against 3, i.e. ~4 bits against ~1.58.
    """
    orig_dtype, orig_shape = x.dtype, x.shape
    k = orig_shape[-1]
    gs = _group_size(granularity, k)

    flat = x.reshape(-1, k).to(torch.float32)
    grouped = flat.reshape(1, -1) if gs < 0 else flat.reshape(-1, gs)

    mag = grouped.abs()
    delta = 0.7 * mag.mean(dim=-1, keepdim=True)
    keep = mag > delta
    # An all-zero group keeps `a` at zero and every entry at zero, which is what it already was.
    count = keep.sum(dim=-1, keepdim=True).clamp_min(1)
    alpha = (mag * keep).sum(dim=-1, keepdim=True) / count
    q = torch.sign(grouped) * keep * alpha
    return q.reshape(-1, k).reshape(orig_shape).to(orig_dtype)


FORMATS = ("fp4", "ternary")


def fake_quantize(x: torch.Tensor, fmt: str, granularity: str) -> torch.Tensor:
    if fmt == "fp4":
        return fake_quantize_fp4(x, granularity)
    if fmt == "ternary":
        return fake_quantize_ternary(x, granularity)
    raise ValueError(f"fmt must be one of {FORMATS}, got {fmt!r}")


class GranularityFP4Linear(nn.Module):
    """`nn.Linear` in FP4 with the **activation** scale granularity dialled down.

    The weight keeps its 16-element block scales throughout. Weight scales are computed once,
    offline, and stored -- they cost nothing at inference, so there is no reason to coarsen them and
    doing so would only confound the measurement. Activation scales are the ones recomputed on every
    call, which is both the runtime cost worth removing and the operand whose outliers the block
    scale is actually protecting.

    The activation is quantized per call, which is the *favourable* choice for the coarse
    granularities: a scale computed from this batch's own amax beats a static one baked in ahead of
    time. Whatever damage shows up here is a lower bound on what a static flat scale would do.
    """

    weight_granularity = "block16"

    def __init__(self, linear: nn.Linear, act_granularity: str, fmt: str = "fp4") -> None:
        super().__init__()
        self.act_granularity = act_granularity
        self.fmt = fmt
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        weight = linear.weight.detach()
        # Validate against K once, here, rather than on the first forward.
        _group_size(act_granularity, self.in_features)
        # The weight takes the same format as the activation. For the FP4 granularity sweep only
        # the activation scale moves and the weight stays on 16-element blocks; for ternary the point
        # is the value grid, so both operands change together.
        self.register_buffer(
            "weight", fake_quantize(weight, fmt, self.weight_granularity), persistent=False
        )
        self.register_buffer(
            "bias",
            None if linear.bias is None else linear.bias.detach().clone(),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(fake_quantize(x, self.fmt, self.act_granularity), self.weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"fmt={self.fmt}, act_granularity={self.act_granularity}, "
            f"weight_granularity={self.weight_granularity}"
        )


def convert_linears_to_granularity_fp4(
    block: nn.Module, names: set[str], *, act_granularity: str, fmt: str = "fp4"
) -> int:
    """Swap named Linears for `GranularityFP4Linear`. Returns how many were swapped.

    Unlike the real path there is no FP8 route for `attn.proj`: the point of this sweep is to hold
    *which* layers are low precision fixed while only the activation scale granularity moves, so all
    five run FP4 exactly as the `proj_precision="fp4"` arm of the real path does.
    """
    count = 0
    for name in sorted(names):
        parent_path, _, attr = name.rpartition(".")
        parent = block.get_submodule(parent_path) if parent_path else block
        child = getattr(parent, attr)
        if not isinstance(child, nn.Linear):
            raise RuntimeError(f"{name} is {type(child).__name__}, expected nn.Linear")
        setattr(parent, attr, GranularityFP4Linear(child, act_granularity, fmt))
        count += 1
    return count
