"""Hand-rolled NVFP4 Linears for the DINOv3 correction path, with an FP8 route for `attn.proj`.

Companion to `fp8_fast_linear.FastFP8Linear`, for the same reason: TorchAO's dispatch adds avoidable
work on top of a GEMM that is already fast. Measured on the serving stack (system python, torch 2.10
/ torchao 0.15), five correction GEMMs per block:

    M=1280   bf16 0.325 ms | fp4 GEMM alone 0.102 (3.18x) | fp4 activation quantization 0.148
    M=8192   bf16 1.780 ms | fp4 GEMM alone 0.509 (3.49x) | fp4 activation quantization 0.193

Three facts drive the design.

**The per-tensor scale stays on.** Turning it off is faster (2.0-2.7x) but doubles the error --
rel-L2 0.2841 vs 0.1394 on real DINOv3 block-0 weights -- so it is not the trade we want. Keeping it
means the GEMM result must be multiplied by `act_scale * weight_scale`. TorchAO applies that as its
own `[M,N]` kernel and, because `_scaled_mm` rejects bias alongside an output scale, is then forced
to add bias as *another* `[M,N]` kernel. `torch.addcmul(bias, gemm_out, scale)` does both in one.

**Activation quantization is launch-bound, not bandwidth-bound.** 0.148 ms at M=1280 and only
0.193 ms at M=8192 -- 6.4x the data for 30% more time. The win is in removing *calls*, not bytes.
`mlp.w1` and `mlp.w2` are handed the identical `norm2(x)` tensor yet each quantizes it independently;
sharing one quantization takes the block from 5 calls to 4 at no numerical cost. That is what
`SharedFP4Activation` does. Going further -- to 1 call -- means fusing the quantization into its
producer (LayerNorm for qkv and w1/w2, SwiGLU for w3), which needs a custom Triton kernel emitting
NVFP4's block-scale + swizzled layout. **Not implemented here**; PyTorch's compiler cannot express
that layout, so it has to be written by hand. `attn.proj` could never be fused anyway: its input
comes out of SDPA.

**`attn.proj` runs in FP8 by default.** It is the worst FP4 candidate of the five on both axes. Its
input is the attention-core output, a convex combination of V rows, whose delta compresses least of
the five (amax ratio 0.93 ImageNet / 0.73 COCO); and it is the one input that no producer fusion can
ever reach. FP8 gives it ~2x instead of ~3.5x on the GEMM but at rel-L2 0.038 instead of 0.139.
Override with `proj_precision="fp4"`.

TorchAO version: 0.15 is correct here and the `<0.16` pin in requirements.txt stands. With the
per-tensor scale kept, 0.17's only relevant addition is its observer flow for static scaling -- and
this module implements its own observer, which works on 0.15. 0.15's Triton quantization kernel
needs no MSLK. So nothing about this path wants the upgrade.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .fp8_fast_linear import FastFP8Linear


_E4M3 = torch.float8_e4m3fn
_FP4X2 = torch.float4_e2m1fn_x2


def _nvfp4_helpers():
    from torchao.prototype.mx_formats.nvfp4_tensor import (
        NVFP4Tensor,
        per_tensor_amax_to_scale,
    )

    return NVFP4Tensor, per_tensor_amax_to_scale


class SharedFP4Activation:
    """Caches one quantized activation so sibling Linears fed the same tensor can reuse it.

    Identity is (data_ptr, shape, dtype), and the cache holds a **strong reference** to the input.
    That reference is what makes the key sound: while it is held the storage cannot be freed, so no
    later tensor can be handed the same address and served a stale quantization. Keying on `id()`
    instead would be both unsound (ids are recycled) and useless -- `x.reshape(...)` hands back a
    fresh view object every call, so an id-keyed cache never hits at all.
    """

    __slots__ = ("_x", "_value")

    def __init__(self) -> None:
        self._x = None
        self._value = None

    def get(self, x: torch.Tensor):
        c = self._x
        if (
            c is not None
            and c.data_ptr() == x.data_ptr()
            and c.shape == x.shape
            and c.dtype == x.dtype
        ):
            return self._value
        return None

    def put(self, x: torch.Tensor, value) -> None:
        self._x = x
        self._value = value


class FastFP4Linear(nn.Module):
    """`nn.Linear` in NVFP4: torch._scaled_mm plus a single fused scale-and-bias epilogue."""

    def __init__(self, linear: nn.Linear, share: SharedFP4Activation | None = None) -> None:
        super().__init__()
        if linear.in_features % 32 or linear.out_features % 16:
            raise ValueError(
                "Packed FP4 _scaled_mm needs in_features % 32 == 0 and out_features % 16 == 0, "
                f"got {linear.in_features}x{linear.out_features}"
            )
        NVFP4Tensor, per_tensor_amax_to_scale = _nvfp4_helpers()
        weight = linear.weight.detach()
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.orig_dtype = weight.dtype
        self._share = share

        # Weight is static: quantize once at construction, keeping its per-tensor scale.
        w_scale = per_tensor_amax_to_scale(weight.abs().amax())
        w_q = NVFP4Tensor.to_nvfp4(
            weight, block_size=16, per_tensor_scale=w_scale, is_swizzled_scales=True
        )
        self._weight_q_t = w_q.t()
        self.register_buffer("weight_scale", w_scale, persistent=False)
        self.register_buffer(
            "bias",
            None if linear.bias is None else linear.bias.detach().clone(),
            persistent=False,
        )
        # BF16 weight is needed only while calibrating, then dropped.
        self.register_buffer("weight_hp", weight.clone(), persistent=False)
        self.register_buffer("amax", torch.zeros((), device=weight.device), persistent=False)
        self.register_buffer("act_scale", torch.zeros((), device=weight.device), persistent=False)
        self.register_buffer("out_scale", torch.zeros((), device=weight.device), persistent=False)
        self.calibrating = True

    @torch.no_grad()
    def freeze_activation_scale(self) -> bool:
        """Bake the observed amax into a static activation scale.

        Returns False when nothing was observed -- freezing on amax=0 would bake a garbage scale, so
        an unexercised Linear stays on the exact BF16 path rather than taking the model down.
        """
        if not self.calibrating:
            return True
        if float(self.amax) <= 0.0:
            return False
        _, per_tensor_amax_to_scale = _nvfp4_helpers()
        self.act_scale = per_tensor_amax_to_scale(self.amax)
        # The product both operands were divided by. Folded into the epilogue, not applied as its
        # own [M,N] kernel the way torchao does it.
        self.out_scale = (self.act_scale * self.weight_scale).to(self.orig_dtype)
        self.weight_hp = None
        self.calibrating = False
        return True

    def _quantize(self, x2d: torch.Tensor):
        if self._share is not None:
            cached = self._share.get(x2d)
            if cached is not None:
                return cached
        NVFP4Tensor, _ = _nvfp4_helpers()
        x_q = NVFP4Tensor.to_nvfp4(
            x2d,
            block_size=16,
            per_tensor_scale=self.act_scale,
            is_swizzled_scales=True,
            use_triton_kernel=True,
        )
        if self._share is not None:
            self._share.put(x2d, x_q)
        return x_q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.calibrating:
            with torch.no_grad():
                self.amax = torch.maximum(self.amax, x.abs().amax())
            return F.linear(x, self.weight_hp, self.bias)

        orig_shape = x.shape
        x2d = x.reshape(-1, orig_shape[-1])
        x_q = self._quantize(x2d)
        w_t = self._weight_q_t
        out = torch._scaled_mm(
            x_q.qdata.view(_FP4X2),
            w_t.qdata.view(_FP4X2),
            x_q.scale.view(_E4M3),
            w_t.scale.t().view(_E4M3),
            bias=None,  # cannot ride along while a per-tensor output scale is still pending
            out_dtype=self.orig_dtype,
        )
        if self.bias is not None:
            out = torch.addcmul(self.bias, out, self.out_scale)  # bias + out*scale, one kernel
        else:
            out = out * self.out_scale
        return out.reshape(*orig_shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        state = "calibrating" if self.calibrating else "fp4"
        return f"in={self.in_features}, out={self.out_features}, state={state}"


def convert_linears_to_fast_fp4(
    block: nn.Module, names: set[str], *, proj_precision: str = "fp8"
) -> tuple[int, int]:
    """Swap named Linears for FastFP4Linear, routing `attn.proj` to FP8 by default.

    `mlp.w1` and `mlp.w2` share one quantized activation. Returns (fp4_count, fp8_count).
    """
    shared = SharedFP4Activation()
    fp4_count = fp8_count = 0
    for name in sorted(names):
        parent_path, _, attr = name.rpartition(".")
        parent = block.get_submodule(parent_path) if parent_path else block
        child = getattr(parent, attr)
        if not isinstance(child, nn.Linear):
            raise RuntimeError(f"{name} is {type(child).__name__}, expected nn.Linear")
        if name.endswith("attn.proj") and proj_precision == "fp8":
            setattr(parent, attr, FastFP8Linear(child))
            fp8_count += 1
        else:
            share = shared if name.endswith(("mlp.w1", "mlp.w2")) else None
            setattr(parent, attr, FastFP4Linear(child, share=share))
            fp4_count += 1
    return fp4_count, fp8_count
