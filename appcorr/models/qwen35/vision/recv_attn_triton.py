"""Triton two-pass column sum for `Qwen35VisionAttention._received_attention` (L4b).

The torch reference materialises the [H, C, L] bf16 softmax per 1024-query chunk (GEMM ->
softmax -> fp32 reduce), which at 2109 image tokens costs 4-5x the whole tower per pass
(docs/memo/qwen_correct_forward_profile.md). This module never materialises P: pass 1 computes
per-(head, query) row max `m` and denominator `l` over the segment, pass 2 recomputes the scores
per key block and accumulates the column sums of `exp(s - m) / l`.

Numerics reproduce the reference's roundings on purpose, because the result feeds patch
selection and a changed score changes which groups get corrected:
  s  = bf16(q . k)   (fp32 MMA accumulation, rounded to bf16 like the cuBLAS output)
  s  = bf16(s * scaling)   (fp32 opmath, bf16 result -- torch's bf16 x python-scalar mul)
  p  = bf16(expf(s - m) / l)   (libdevice expf + IEEE div_rn, the softmax epilogue)
  column sum in fp32 of the bf16 p, mean over heads in fp32.
The only admitted difference from the reference is fp32 summation ORDER: `l` is an online
running sum instead of the softmax kernel's block reduction, and the column sum is one
accumulation per key block instead of per 1024-query chunk. Whether that ever flips a
selection is what the snapshot gate on 192 RWQA + 448 VisDrone samples decides; the reference
path stays the default until it passes (`APPCORR_RECV_ATTN=triton` opts in).
"""
from __future__ import annotations

import os

import torch

try:  # triton is present in every GPU env we run; the torch path is the fallback anyway
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
    _TRITON_OK = True
except Exception:  # pragma: no cover - CPU-only envs
    _TRITON_OK = False

# Tile config (BLOCK_M, BLOCK_N, num_warps, num_stages); the head-dim tile is
# next_power_of_2(Dh) = 128 for Dh 72 (`tl.arange` needs a power of two, so the 56 zero columns
# are dead MMA work we cannot trim without splitting the dot). Overridable for the tuning sweep
# (`qwen_recv_attn_triton_test.py --sweep`); a tile change moves fp32 summation order only.
_CFG = tuple(int(v) for v in os.environ.get("APPCORR_RECV_ATTN_CFG", "64,64,4,2").split(","))
_BLOCK_M, _BLOCK_N, _NUM_WARPS, _NUM_STAGES = _CFG
_ENV_CONFIGURED = False


def set_config(block_m: int, block_n: int, num_warps: int, num_stages: int) -> None:
    global _BLOCK_M, _BLOCK_N, _NUM_WARPS, _NUM_STAGES
    _BLOCK_M, _BLOCK_N, _NUM_WARPS, _NUM_STAGES = block_m, block_n, num_warps, num_stages


def available() -> bool:
    return _TRITON_OK and torch.cuda.is_available()


def _ensure_compile_env() -> None:
    """Triton writes/executes under its cache dir; /tmp is noexec on this box, so route it to
    the same tree the DINOv3 kernels use (`_configure_compile_environment`)."""
    global _ENV_CONFIGURED
    if _ENV_CONFIGURED:
        return
    try:
        from offload.server.model.dinov3_precision import _configure_compile_environment
        _configure_compile_environment()
    except Exception:
        cache = os.path.join(os.path.expanduser("~"), ".cache", "appcorr", "torchinductor")
        os.makedirs(os.path.join(cache, "triton"), exist_ok=True)
        os.environ.setdefault("TRITON_CACHE_DIR", os.path.join(cache, "triton"))
    _ENV_CONFIGURED = True


if _TRITON_OK:

    @triton.jit
    def _scores(q, k, scaling):
        # q [BM, D] bf16, k [BN, D] bf16 -> fp32 [BM, BN] with the reference's two roundings.
        s = tl.dot(q, tl.trans(k))                      # fp32 accumulate
        s = s.to(tl.bfloat16).to(tl.float32)            # cuBLAS bf16 output
        s = (s * scaling).to(tl.bfloat16).to(tl.float32)  # torch mul(bf16, scalar)
        return s

    @triton.jit
    def _rowstats_kernel(
        q_ptr, k_ptr, m_ptr, l_ptr,
        stride_q_n, stride_q_h, stride_q_d,
        stride_k_n, stride_k_h, stride_k_d,
        stride_m_h, stride_m_n,
        L, scaling,
        HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_h = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, BLOCK_D)
        q = tl.load(q_ptr + pid_h * stride_q_h + offs_m[:, None] * stride_q_n + offs_d[None, :] * stride_q_d,
                    mask=(offs_m[:, None] < L) & (offs_d[None, :] < HEAD_DIM), other=0.0)
        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        for n0 in range(0, L, BLOCK_N):
            offs_n = n0 + tl.arange(0, BLOCK_N)
            k = tl.load(k_ptr + pid_h * stride_k_h + offs_n[:, None] * stride_k_n + offs_d[None, :] * stride_k_d,
                        mask=(offs_n[:, None] < L) & (offs_d[None, :] < HEAD_DIM), other=0.0)
            s = _scores(q, k, scaling)
            s = tl.where(offs_n[None, :] < L, s, float("-inf"))
            m_new = tl.maximum(m_i, tl.max(s, axis=1))
            l_i = l_i * libdevice.exp(m_i - m_new) + tl.sum(libdevice.exp(s - m_new[:, None]), axis=1)
            m_i = m_new
        tl.store(m_ptr + pid_h * stride_m_h + offs_m * stride_m_n, m_i, mask=offs_m < L)
        tl.store(l_ptr + pid_h * stride_m_h + offs_m * stride_m_n, l_i, mask=offs_m < L)

    @triton.jit
    def _colsum_kernel(
        q_ptr, k_ptr, m_ptr, l_ptr, out_ptr,
        stride_q_n, stride_q_h, stride_q_d,
        stride_k_n, stride_k_h, stride_k_d,
        stride_m_h, stride_m_n,
        stride_o_h, stride_o_n,
        L, scaling,
        HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_h = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)
        k = tl.load(k_ptr + pid_h * stride_k_h + offs_n[:, None] * stride_k_n + offs_d[None, :] * stride_k_d,
                    mask=(offs_n[:, None] < L) & (offs_d[None, :] < HEAD_DIM), other=0.0)
        acc = tl.zeros([BLOCK_N], tl.float32)
        for m0 in range(0, L, BLOCK_M):
            offs_m = m0 + tl.arange(0, BLOCK_M)
            q = tl.load(q_ptr + pid_h * stride_q_h + offs_m[:, None] * stride_q_n + offs_d[None, :] * stride_q_d,
                        mask=(offs_m[:, None] < L) & (offs_d[None, :] < HEAD_DIM), other=0.0)
            m_i = tl.load(m_ptr + pid_h * stride_m_h + offs_m * stride_m_n, mask=offs_m < L, other=0.0)
            l_i = tl.load(l_ptr + pid_h * stride_m_h + offs_m * stride_m_n, mask=offs_m < L, other=1.0)
            s = _scores(q, k, scaling)                  # same dot orientation as pass 1
            p = libdevice.div_rn(libdevice.exp(s - m_i[:, None]), l_i[:, None])
            p = p.to(tl.bfloat16).to(tl.float32)        # softmax's bf16 output
            p = tl.where(offs_m[:, None] < L, p, 0.0)
            acc += tl.sum(p, axis=0)
        tl.store(out_ptr + pid_h * stride_o_h + offs_n * stride_o_n, acc, mask=offs_n < L)


def received_attention(q: torch.Tensor, k: torch.Tensor, scaling: float, segment_ranges) -> torch.Tensor:
    """Drop-in for the torch `_received_attention`: q, k [T, H, Dh] bf16 (post-RoPE) -> [T] fp32,
    mean over heads and queries of the attention each token receives, divided by its segment's
    length."""
    assert available(), "triton received-attention requested without triton/CUDA"
    _ensure_compile_env()
    T, H, Dh = q.shape
    block_d = max(16, triton.next_power_of_2(Dh))
    col = torch.zeros(T, device=q.device, dtype=torch.float32)
    for start, length in segment_ranges:
        end = start + length
        q_seg, k_seg = q[start:end], k[start:end]
        m = torch.empty(H, length, device=q.device, dtype=torch.float32)
        l = torch.empty_like(m)
        out = torch.empty_like(m)
        grid_m = (triton.cdiv(length, _BLOCK_M), H)
        _rowstats_kernel[grid_m](
            q_seg, k_seg, m, l,
            q_seg.stride(0), q_seg.stride(1), q_seg.stride(2),
            k_seg.stride(0), k_seg.stride(1), k_seg.stride(2),
            m.stride(0), m.stride(1),
            length, float(scaling),
            HEAD_DIM=Dh, BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N, BLOCK_D=block_d,
            num_warps=_NUM_WARPS, num_stages=_NUM_STAGES,
        )
        grid_n = (triton.cdiv(length, _BLOCK_N), H)
        _colsum_kernel[grid_n](
            q_seg, k_seg, m, l, out,
            q_seg.stride(0), q_seg.stride(1), q_seg.stride(2),
            k_seg.stride(0), k_seg.stride(1), k_seg.stride(2),
            m.stride(0), m.stride(1),
            out.stride(0), out.stride(1),
            length, float(scaling),
            HEAD_DIM=Dh, BLOCK_M=_BLOCK_M, BLOCK_N=_BLOCK_N, BLOCK_D=block_d,
            num_warps=_NUM_WARPS, num_stages=_NUM_STAGES,
        )
        col[start:end] = out.mean(dim=0) / length
    return col
