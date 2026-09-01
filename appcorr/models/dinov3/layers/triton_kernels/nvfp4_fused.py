"""NVFP4 activation quantization, standalone first so it can later be fused into its producer.

Why this exists: on the correction path the FP4 GEMM is 3.18x faster than BF16 (0.102 vs 0.325 ms
per block at M=1280), but quantizing the activation costs 0.148 ms -- more than the GEMM -- so
`FastFP4Linear` nets 0.96x and the format's advantage is entirely eaten.

That cost is almost all overhead, not work. At M=1280 one call moves ~13.1 MB (10.5 MB read + 2.6 MB
written), which is ~4.3 us at HBM bandwidth, against ~30 us measured. The same conclusion falls out
of the shape sweep: 6.4x the data for 30% more time (0.148 -> 0.193 ms from M=1280 to M=8192). So
folding the quantization into the kernel that already produces its input -- LayerNorm for
`attn.qkv` and `mlp.w1`/`w2`, SwiGLU for `mlp.w3` -- should reclaim most of it.

**Producer fusion was tried and abandoned** -- see docs/memo/dinov3_nvfp4_speedup_gate.md. Folding
the LayerNorm in forces one program per 128 rows (the reduction needs the whole row), which collapses
the grid from (K/64, M/128) to (M/128,): 10 programs at M=1280 against 148 SMs. It measured 0.06-0.27x
of the unfused pair. What actually paid was calling this kernel directly instead of through
`NVFP4Tensor.to_nvfp4`, whose wrapper costs more than the kernel itself.

This module is therefore the quantizer alone, and the byte-equality gate exists so that the output
format stays provably correct. The format is
unforgiving -- a wrong scale swizzle produces no error, just wrong numbers -- so
`tests/test_nvfp4_fused_quantize.py` gates on exact equality of both `qdata` and `scale`.

Layout notes, read off `torchao/prototype/mx_formats/kernels.py::quantize_nvfp4_triton_kernel`:

* Grid is (ceil(N/64), ceil(M/128)); each program owns a [128, 64] tile, i.e. 128 rows x 4 blocks
  of 16.
* Scales for a tile are [128, 4] e4m3, stored as `reshape(4, 32, 4).permute(1, 0, 2).reshape(32,16)`
  at a per-tile stride of 32*16, tiles ordered `pid_m * num_pid_n + pid_n`.
* Data is packed two e2m1 values per byte, so the qdata tile is [128, 32] uint8.
* With a per-tensor scale, the block scale stored is `(block_amax / 6) / tensor_scale` clamped to
  [E4M3_EPS, 448], and the data is divided by `tensor_scale * stored_block_scale`.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Only TorchAO 0.15 exposes this; 0.17 dropped it in favour of MSLK. Import failure is handled by
# callers, which fall back to NVFP4Tensor.to_nvfp4 -- correct everywhere, just ~2x slower.
from torchao.prototype.mx_formats.kernels import convert_fp32_to_fp4_packed

NVFP4_FUSED_AVAILABLE = True


F4_E2M1_MAX = 6.0
F8E4M3_MAX = 448.0
E4M3_EPS = 1.5258789e-05


@triton.jit
def _quantize_nvfp4_kernel(
    x_ptr,
    tensor_scale_ptr,
    q_ptr,
    s_ptr,
    stride_xm,
    stride_xn,
    M,
    N,
    MASK: tl.constexpr,
):
    # Triton only reads globals declared as tl.constexpr, so these live in the kernel body -- the
    # same reason torchao's version repeats them.
    F4_E2M1_MAX = 6.0
    F8E4M3_MAX = 448.0
    E4M3_EPS = 1.5258789e-05

    pid_n = tl.program_id(0)
    pid_m = tl.program_id(1)

    offs_m = pid_m * 128 + tl.arange(0, 128)[:, None]
    offs_n = pid_n * 64 + tl.arange(0, 64)[None, :]
    if MASK:
        mask = (offs_m < M) & (offs_n < N)
        other = 0.0
    else:
        mask = None
        other = None
    x = tl.load(x_ptr + offs_m * stride_xm + offs_n * stride_xn, mask=mask, other=other)

    x_blocks = x.to(tl.float32).reshape(128, 4, 16)
    block_amax = tl.max(x_blocks.abs(), axis=2)

    tensor_scale = tl.load(tensor_scale_ptr)
    block_scale_f32 = (block_amax / F4_E2M1_MAX).to(tl.float32)
    scaled = tl.clamp(block_scale_f32 / tensor_scale, E4M3_EPS, F8E4M3_MAX)
    scales = scaled.to(tl.float8e4nv)
    total_scale = tensor_scale * scales.to(tl.float32)[:, :, None]
    x_blocks = tl.div_rn(x_blocks, total_scale)

    if MASK:
        scale_offs_n = pid_n * 4 + tl.arange(0, 4)[None, :]
        scales = tl.where((offs_m < M) & (scale_offs_n < N // 16), scales, 0.0)

    packed_scales = scales.reshape(4, 32, 4).permute(1, 0, 2).reshape(32, 16)
    s_off_m = tl.arange(0, 32)[:, None]
    s_off_n = tl.arange(0, 16)[None, :]
    tl.store(
        s_ptr
        + (pid_m * tl.num_programs(0) + pid_n) * (32 * 16)
        + s_off_m * 16
        + s_off_n,
        packed_scales,
    )

    x_fp4x2 = convert_fp32_to_fp4_packed(x_blocks.reshape(128, 32, 2).split())
    q_off_m = pid_m * 128 + tl.arange(0, 128)[:, None]
    q_off_n = pid_n * 32 + tl.arange(0, 32)[None, :]
    if MASK:
        q_mask = (q_off_m < M) & (q_off_n < N // 2)
    else:
        q_mask = None
    tl.store(q_ptr + q_off_m * (N // 2) + q_off_n, x_fp4x2, mask=q_mask)


def quantize_nvfp4_swizzled(x: torch.Tensor, tensor_scale: torch.Tensor):
    """Quantize `x` [M, K] to NVFP4, returning (qdata uint8 [M, K/2], swizzled e4m3 scales).

    Byte-identical to `NVFP4Tensor.to_nvfp4(x, per_tensor_scale=tensor_scale,
    is_swizzled_scales=True, use_triton_kernel=True)` -- see tests/test_nvfp4_fused_quantize.py.
    """
    assert x.dim() == 2, f"expected [M, K], got {tuple(x.shape)}"
    M, K = x.shape
    assert K % 16 == 0, f"K must be a multiple of the 16-element block, got {K}"
    grid_n = triton.cdiv(K, 64)
    grid_m = triton.cdiv(M, 128)
    qdata = torch.empty((M, K // 2), device=x.device, dtype=torch.uint8)
    # Scales are stored tile by tile, each tile contributing a fixed 32x16 chunk, so the buffer is
    # sized by tile count rather than by M*K/16 -- padding rows of a partial tile still occupy space.
    scale = torch.empty((grid_m * grid_n * 32 * 16,), device=x.device, dtype=torch.float8_e4m3fn)
    _quantize_nvfp4_kernel[(grid_n, grid_m)](
        x,
        tensor_scale,
        qdata,
        scale,
        x.stride(0),
        x.stride(1),
        M,
        K,
        MASK=(M % 128 != 0) or (K % 64 != 0),
    )
    return qdata, scale
