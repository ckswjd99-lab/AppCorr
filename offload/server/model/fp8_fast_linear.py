"""A hand-rolled FP8 Linear for the DINOv3 correction path.

TorchAO's `Float8DynamicActivationFloat8WeightConfig` leaves ~5.6x of overhead on top of the FP8
GEMM. Measured on the serving stack (system python, torch 2.10 / torchao 0.15), five correction
GEMMs per block, bias-free, M=1280:

    bf16                       0.324 ms
    torchao FP8                0.907 ms   (0.36x -- slower than bf16)
    raw torch._scaled_mm       0.161 ms   (2.01x -- exactly the theoretical 2x)

So the FP8 tensor cores were never the problem. Three things recover the kernel's speed:

1. **Call `torch._scaled_mm` directly.** It *is* the CUTLASS/cuBLAS FP8 GEMM; the wrapper around it
   is what costs. Bias rides along inside the GEMM (no per-tensor output scale here, so
   `_scaled_mm` accepts it), which keeps the epilogue free -- the trap that made NVFP4 slow.
2. **Fuse the activation quantization.** Eager `(x.float()/s).clamp().to(e4m3)` allocates an fp32
   [M,K] and costs *more than the GEMM* (0.379 ms vs 0.161 ms). Compiled, it is one pass.
3. **Use a static activation scale.** Dropping the per-call `amax` reduction takes quantization from
   0.138 -> 0.073 ms; combined with (1) the block lands at 1.38x bf16 at M=1280, rising to 1.67x.

Accuracy, measured against BF16 on real DINOv3 ViT-7B block-0 weights with a delta-scale input:
**rel-L2 0.0379**, versus 0.1394 for NVFP4 with a per-tensor scale and 0.2841 without one. Per-row
weight scaling was measured too and is not worth the complexity (0.0378) -- the error is dominated
by *activation* quantization, so finer weight granularity buys nothing.

The activation scale is calibrated at runtime, not at load: correction inputs are selected-token
activations that do not exist until requests arrive. While calibrating, `forward` runs an ordinary
BF16 `F.linear` and records amax, so calibration requests are numerically exact -- just not
accelerated.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


E4M3_MAX = 448.0
_FP8_DTYPE = torch.float8_e4m3fn


def pad_rows_to_bucket(x2d: torch.Tensor, bucket: int) -> tuple[torch.Tensor, int]:
    """Zero-pad `x2d` up to a multiple of `bucket` rows. Returns (padded, original_rows).

    Why: the correction path's selected-token count M changes every round, so every shape-specialised
    consumer downstream -- torch.compile graphs, and CUDA graph capture in particular -- sees an
    unbounded set of shapes. Rounding M up to a handful of buckets makes those shapes finite.

    Zeros, not `torch.empty`: an uninitialised pad would feed garbage into the activation amax and
    (for NVFP4) into a block scale, corrupting real rows. Zeros are exactly representable in both
    FP8 and FP4 and contribute nothing to either scale.
    """
    rows = x2d.shape[0]
    if bucket <= 0 or rows % bucket == 0:
        return x2d, rows
    pad = bucket - rows % bucket
    return torch.nn.functional.pad(x2d, (0, 0, 0, pad)), rows


def _quantize_static(x: torch.Tensor, inv_scale: torch.Tensor) -> torch.Tensor:
    """One fused pass: scale into FP8 range and cast. Compiled by FastFP8Linear."""
    return (x * inv_scale).clamp(-E4M3_MAX, E4M3_MAX).to(_FP8_DTYPE)


# dynamic=True on purpose. ADE20K's selected-token count M changes every correction round (median
# ~1028), so a static-shape graph would recompile per distinct M and silently fall back to eager
# once it passes torch._dynamo.cache_size_limit (default 8). The quantizer is elementwise, so a
# dynamic-shape graph fuses just as well.
_quantize_static_compiled = torch.compile(_quantize_static, dynamic=True)


class FastFP8Linear(nn.Module):
    """Drop-in for `nn.Linear` that runs `torch._scaled_mm` in FP8.

    Lifecycle: constructed in *calibrating* state (exact BF16 + amax observation), then
    `freeze_activation_scale()` bakes the static scale and switches to the FP8 path.
    """

    def __init__(self, linear: nn.Linear, bucket_rows: int = 0) -> None:
        super().__init__()
        self.bucket_rows = max(0, int(bucket_rows))
        if linear.in_features % 16 or linear.out_features % 16:
            raise ValueError(
                "FP8 _scaled_mm requires both dimensions divisible by 16, got "
                f"{linear.in_features}x{linear.out_features}"
            )
        weight = linear.weight.detach()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.orig_dtype = weight.dtype

        # Weight is static: quantize once, per-tensor, at construction.
        w_scale = (weight.abs().amax().float() / E4M3_MAX).clamp(min=1e-12)
        w_q = (weight / w_scale.to(weight.dtype)).clamp(-E4M3_MAX, E4M3_MAX).to(_FP8_DTYPE)
        # _scaled_mm wants mat2 column-major; .t() of a contiguous [out, in] gives exactly that.
        self.register_buffer("weight_q_t", w_q.t(), persistent=False)
        self.register_buffer("weight_scale", w_scale, persistent=False)
        self.register_buffer(
            "bias",
            None if linear.bias is None else linear.bias.detach().clone(),
            persistent=False,
        )
        # BF16 weight is kept only for the calibration phase, then dropped.
        self.register_buffer("weight_hp", weight.clone(), persistent=False)

        self.register_buffer("amax", torch.zeros((), device=weight.device), persistent=False)
        self.register_buffer("act_scale", torch.zeros((), device=weight.device), persistent=False)
        self.register_buffer(
            "act_inv_scale", torch.zeros((), device=weight.device), persistent=False
        )
        self.calibrating = True

    @torch.no_grad()
    def freeze_activation_scale(self) -> bool:
        """Bake the observed amax into a static scale and drop the BF16 weight.

        Returns True once frozen. Returns False if this Linear has not seen an activation yet --
        freezing on amax=0 would bake a garbage scale, and a layer that some routing path skips
        should simply stay in the exact BF16 state until it is exercised, not take FP8 down for
        the whole model.
        """
        if not self.calibrating:
            return True
        if float(self.amax) <= 0.0:
            return False
        scale = (self.amax.float() / E4M3_MAX).clamp(min=1e-12)
        self.act_scale = scale
        # Precompute the reciprocal in the activation dtype so the hot path is a single multiply.
        self.act_inv_scale = (1.0 / scale).to(self.orig_dtype)
        self.weight_hp = None
        self.calibrating = False
        return True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.calibrating:
            with torch.no_grad():
                self.amax = torch.maximum(self.amax, x.abs().amax())
            return F.linear(x, self.weight_hp, self.bias)

        orig_shape = x.shape
        x2d = x.reshape(-1, orig_shape[-1])
        x2d, rows = pad_rows_to_bucket(x2d, self.bucket_rows)
        x_q = _quantize_static_compiled(x2d, self.act_inv_scale)
        out = torch._scaled_mm(
            x_q,
            self.weight_q_t,
            self.act_scale,
            self.weight_scale,
            bias=self.bias,  # fused into the GEMM epilogue -- no extra [M,N] pass
            out_dtype=self.orig_dtype,
        )
        if out.shape[0] != rows:
            out = out[:rows]
        return out.reshape(*orig_shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        state = "calibrating" if self.calibrating else "fp8"
        return f"in={self.in_features}, out={self.out_features}, state={state}"


def convert_linears_to_fast_fp8(block: nn.Module, names: set[str], bucket_rows: int = 0) -> int:
    """Swap the named `nn.Linear` children of `block` for `FastFP8Linear`. Returns the count."""
    converted = 0
    for name in sorted(names):
        parent_path, _, attr = name.rpartition(".")
        parent = block.get_submodule(parent_path) if parent_path else block
        child = getattr(parent, attr)
        if not isinstance(child, nn.Linear):
            raise RuntimeError(f"{name} is {type(child).__name__}, expected nn.Linear")
        setattr(parent, attr, FastFP8Linear(child, bucket_rows=bucket_rows))
        converted += 1
    return converted
