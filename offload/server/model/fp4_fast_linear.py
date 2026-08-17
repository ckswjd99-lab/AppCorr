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

import os

import torch
import torch.nn.functional as F
from torch import nn

from .fp8_fast_linear import FastFP8Linear, pad_rows_to_bucket


def _torchao_version() -> str:
    try:
        import torchao

        return getattr(torchao, "__version__", "?")
    except Exception:
        return "?"


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

    _warned_slow_quantizer = False

    def __init__(
        self,
        linear: nn.Linear,
        share: SharedFP4Activation | None = None,
        bucket_rows: int = 0,
        per_tensor_scale: bool = False,
    ) -> None:
        super().__init__()
        self.bucket_rows = max(0, int(bucket_rows))
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
        self._use_per_tensor = bool(per_tensor_scale)
        w_scale = (
            per_tensor_amax_to_scale(weight.abs().amax())
            if self._use_per_tensor
            else torch.ones((), device=weight.device)
        )
        w_q = NVFP4Tensor.to_nvfp4(
            weight,
            block_size=16,
            per_tensor_scale=w_scale if self._use_per_tensor else None,
            is_swizzled_scales=True,
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
        self.out_scale_f = 0.0
        # Calibration exists only to observe an activation amax and bake it into a static per-tensor
        # scale. With no per-tensor scale there is nothing to observe: both operands are quantized
        # against their block scales alone and the output scale is 1. Skipping it means the module is
        # usable immediately, which is what lets the approximate path -- which has no calibration
        # driver of its own -- use this class at all.
        if self._use_per_tensor:
            self.calibrating = True
        else:
            self.act_scale = torch.ones((), device=weight.device)
            self.out_scale = torch.ones((), device=weight.device, dtype=self.orig_dtype)
            self.out_scale_f = 1.0
            self.weight_hp = None
            self.calibrating = False

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
        # With the per-tensor scale off, the activation quantizer must divide by 1, not by an
        # observed amax -- otherwise the operands are scaled but the epilogue no longer undoes it.
        self.act_scale = (
            per_tensor_amax_to_scale(self.amax)
            if self._use_per_tensor
            else torch.ones_like(self.amax)
        )
        # The product both operands were divided by. Folded into the epilogue, not applied as its
        # own [M,N] kernel the way torchao does it.
        self.out_scale = (self.act_scale * self.weight_scale).to(self.orig_dtype)
        # Materialise the scalar on the host ONCE. Reading it per call would be an .item() on a
        # CUDA tensor, i.e. a sync in the middle of every FP4 Linear.
        self.out_scale_f = float(self.out_scale)
        self.weight_hp = None
        self.calibrating = False
        return True

    def _quantize(self, x2d: torch.Tensor):
        """Returns (qdata, scale) ready for _scaled_mm.

        Calls the Triton quantizer directly rather than through `NVFP4Tensor.to_nvfp4`. Same kernel,
        but the wrapper's tensor-subclass construction costs more than the kernel does: 29.6 us
        against 14.3 us at M=1280, K=4096. Across a block's three FP4 quantizations (qkv, the shared
        w1/w2, and w3) that is ~46 us, which is what takes FastFP4Linear from 0.96x to ~1.11x of
        BF16 -- more than folding the quantization into its producer would save (~25 us).
        """
        if self._share is not None:
            cached = self._share.get(x2d)
            if cached is not None:
                return cached
        try:
            from appcorr.models.dinov3.layers.triton_kernels.nvfp4_fused import (
                quantize_nvfp4_swizzled,
            )
        except ImportError:
            # TorchAO 0.17 dropped convert_fp32_to_fp4_packed, so the direct kernel is unavailable
            # and this falls back to NVFP4Tensor.to_nvfp4 -- correct, but ~2x slower (29.6 vs
            # 14.3 us at M=1280), which is exactly the wrapper overhead the direct call was added to
            # remove. Say so loudly: five consecutive end-to-end runs silently took this path
            # because the shell had conda activated, and the resulting 18ms regression was chased
            # through the fusion, the epilogue and the instrumentation before the interpreter was
            # checked.
            if not FastFP4Linear._warned_slow_quantizer:
                FastFP4Linear._warned_slow_quantizer = True
                import sys as _sys

                print(
                    "[FP4-correct] WARNING: convert_fp32_to_fp4_packed missing (torchao "
                    f"{_torchao_version()}, {_sys.executable}); using the ~2x slower "
                    "NVFP4Tensor.to_nvfp4 path. Latency numbers are not comparable to torchao 0.15.",
                    flush=True,
                )
            NVFP4Tensor, _ = _nvfp4_helpers()
            t = NVFP4Tensor.to_nvfp4(
                x2d, block_size=16, per_tensor_scale=self.act_scale,
                is_swizzled_scales=True, use_triton_kernel=True,
            )
            x_q = (t.qdata, t.scale)
            if self._share is not None:
                self._share.put(x2d, x_q)
            return x_q

        x_q = quantize_nvfp4_swizzled(x2d, self.act_scale)
        if self._share is not None:
            self._share.put(x2d, x_q)
        return x_q

    def forward_raw(self, x: torch.Tensor):
        """GEMM output with the scale-and-bias epilogue left undone, plus what it needs.

        Returns (raw_2d, scale, bias, orig_shape), or None while calibrating -- there is no epilogue
        to defer then, since the module is running an ordinary BF16 F.linear. Lets a consumer that
        immediately combines two of these (SwiGLU) fold both epilogues into its own kernel.
        """
        if self.calibrating:
            return None
        orig_shape = x.shape
        x2d = x.reshape(-1, orig_shape[-1])
        x2d, rows = pad_rows_to_bucket(x2d, self.bucket_rows)
        x_qdata, x_scale = self._quantize(x2d)
        w_t = self._weight_q_t
        # Without a per-tensor output scale there is nothing to defer: `_scaled_mm` takes the bias
        # and the consumer is handed a scale of 1 with no bias left to add, so its fused epilogue
        # collapses to a copy. Leaving the bias out here instead would make the consumer apply it.
        #
        # Kept for consistency with `forward`, not for speed -- the A/B in `forward` shows the two
        # are indistinguishable in CORRECT_FORWARD time. What they are *not* is numerically
        # identical: folding gives 61.4828 mIoU on ADE20K full-2000 against 61.4208 deferred, so the
        # fold is the better default by 0.062, well outside the ~0.0001 run-to-run noise.
        # APPCORR_FP4_RAW_NO_FOLD restores the pre-bypass behaviour for an A/B: the consumer gets a
        # real scale and bias back and applies its own epilogue. Measurement hatch, not a mode.
        fold_bias = not self._use_per_tensor and not os.environ.get("APPCORR_FP4_RAW_NO_FOLD")
        out = torch._scaled_mm(
            x_qdata.view(_FP4X2),
            w_t.qdata.view(_FP4X2),
            x_scale.view(_E4M3),
            w_t.scale.t().view(_E4M3),
            bias=self.bias if fold_bias else None,
            out_dtype=self.orig_dtype,
        )
        if out.shape[0] != rows:
            out = out[:rows].contiguous()
        if fold_bias:
            return out, 1.0, None, orig_shape
        return out, self.out_scale_f, self.bias, orig_shape

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.calibrating:
            with torch.no_grad():
                self.amax = torch.maximum(self.amax, x.abs().amax())
            return F.linear(x, self.weight_hp, self.bias)

        orig_shape = x.shape
        x2d = x.reshape(-1, orig_shape[-1])
        x2d, rows = pad_rows_to_bucket(x2d, self.bucket_rows)
        x_qdata, x_scale = self._quantize(x2d)
        w_t = self._weight_q_t
        if not self._use_per_tensor:
            # No per-tensor output scale to apply, so `_scaled_mm` can take the bias itself and the
            # whole [M, N] epilogue disappears.
            #
            # Do not read a latency claim into this. Measured solo on ADE20K (200 requests, two runs
            # per arm, order reversed between pairs), CORRECT_FORWARD is 85.71 ms with the epilogue
            # bypassed and 86.07 ms without -- a 0.36 ms gap against a 1.6-3.1 ms spread between
            # repeats of the *same* arm. The epilogue is real work but it is a rounding error next to
            # the ~86 ms stage. The earlier 89.20 vs 103.91 ms pair was measured under three-way GPU
            # contention and does not reproduce. Dropping the per-tensor scale is justified by
            # numerics, not by time.
            out = torch._scaled_mm(
                x_qdata.view(_FP4X2),
                w_t.qdata.view(_FP4X2),
                x_scale.view(_E4M3),
                w_t.scale.t().view(_E4M3),
                bias=self.bias,
                out_dtype=self.orig_dtype,
            )
            if out.shape[0] != rows:
                out = out[:rows]
            return out.reshape(*orig_shape[:-1], self.out_features)
        out = torch._scaled_mm(
            x_qdata.view(_FP4X2),
            w_t.qdata.view(_FP4X2),
            x_scale.view(_E4M3),
            w_t.scale.t().view(_E4M3),
            bias=None,  # cannot ride along while a per-tensor output scale is still pending
            out_dtype=self.orig_dtype,
        )
        # In-place epilogue: torch.addcmul allocates a second [M, N] and dispatches to the
        # non-vectorised elementwise kernel. At the real correction shape this epilogue is what
        # returns half of FP4's GEMM win (elementwise 14.96 -> 33.39 ms against -36.07 ms of GEMM),
        # so it is worth a dedicated kernel that reads and writes the same buffer once.
        from appcorr.models.dinov3.layers.triton_kernels import scale_bias_inplace_triton

        if os.environ.get("APPCORR_ATEN_EPILOGUE") or not scale_bias_inplace_triton(
            out, self.out_scale_f, self.bias
        ):
            if self.bias is not None:
                out = torch.addcmul(self.bias, out, self.out_scale)
            else:
                out = out * self.out_scale
        if out.shape[0] != rows:
            out = out[:rows]
        return out.reshape(*orig_shape[:-1], self.out_features)

    def extra_repr(self) -> str:
        state = "calibrating" if self.calibrating else "fp4"
        return f"in={self.in_features}, out={self.out_features}, state={state}"


def convert_linears_to_fast_fp4(
    block: nn.Module,
    names: set[str],
    *,
    proj_precision: str = "fp8",
    bucket_rows: int = 0,
    per_tensor_scale: bool = False,
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
            setattr(parent, attr, FastFP8Linear(child, bucket_rows=bucket_rows))
            fp8_count += 1
        else:
            share = shared if name.endswith(("mlp.w1", "mlp.w2")) else None
            setattr(
                parent,
                attr,
                FastFP4Linear(
                    child, share=share, bucket_rows=bucket_rows,
                    per_tensor_scale=per_tensor_scale,
                ),
            )
            fp4_count += 1
    return fp4_count, fp8_count
