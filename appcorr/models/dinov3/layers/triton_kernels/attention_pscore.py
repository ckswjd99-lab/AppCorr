from __future__ import annotations

from typing import Optional, Tuple

import torch
import triton
import triton.language as tl

from ._strict import note_fallback


_MAIN_BLOCK_M = 32
_MAIN_BLOCK_N = 128
_MAIN_BLOCK_D = 128
_MAIN_NUM_WARPS = 4
_MAIN_NUM_STAGES = 2
_LOG2_E = 1.4426950408889634


@triton.jit
def _pscore_from_cudnn_lse_keyblock_atomic_kernel(
    q_ptr,
    k_ptr,
    lse_ptr,
    pscore_acc_ptr,
    stride_q_b,
    stride_q_h,
    stride_q_n,
    stride_q_d,
    stride_k_b,
    stride_k_h,
    stride_k_n,
    stride_k_d,
    stride_lse_b,
    stride_lse_h,
    stride_lse_n,
    stride_pscore_b,
    stride_pscore_n,
    num_tokens: tl.constexpr,
    scale_log2,
    inv_mean_denom,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_kb = tl.program_id(2)

    k_offsets = pid_kb * BLOCK_N + tl.arange(0, BLOCK_N)
    d_offsets = tl.arange(0, BLOCK_D)

    k = tl.load(
        k_ptr
        + pid_b * stride_k_b
        + pid_h * stride_k_h
        + k_offsets[:, None] * stride_k_n
        + d_offsets[None, :] * stride_k_d,
        mask=k_offsets[:, None] < num_tokens,
        other=0.0,
    )

    col_acc = tl.zeros((BLOCK_N,), tl.float32)

    for q_start in range(0, num_tokens, BLOCK_M):
        q_offsets = q_start + tl.arange(0, BLOCK_M)
        q = tl.load(
            q_ptr
            + pid_b * stride_q_b
            + pid_h * stride_q_h
            + q_offsets[:, None] * stride_q_n
            + d_offsets[None, :] * stride_q_d,
            mask=q_offsets[:, None] < num_tokens,
            other=0.0,
        )
        scores = tl.dot(q, tl.trans(k), input_precision="tf32") * scale_log2
        lse = tl.load(
            lse_ptr
            + pid_b * stride_lse_b
            + pid_h * stride_lse_h
            + q_offsets * stride_lse_n,
            mask=q_offsets < num_tokens,
            other=float("inf"),
        ) * 1.4426950408889634
        probs = tl.exp2(scores - lse[:, None])
        probs = tl.where(
            (q_offsets[:, None] < num_tokens) & (k_offsets[None, :] < num_tokens),
            probs,
            0.0,
        )
        col_acc += tl.sum(probs, axis=0)

    tl.atomic_add(
        pscore_acc_ptr + pid_b * stride_pscore_b + k_offsets * stride_pscore_n,
        col_acc * inv_mean_denom,
        sem="relaxed",
        mask=k_offsets < num_tokens,
    )


def _unsupported_reason(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> str | None:
    """Why the fused pscore kernel cannot run, or None if it can.

    A reason string rather than a bool so the fallback can say which constraint failed -- several of
    these (the 32-head, 128-dim shape in particular) are specific to the ViT-7B this kernel was
    written for, and a different backbone falls back for a legitimate reason that still needs to be
    visible in a latency measurement.
    """
    if torch.is_grad_enabled():
        return "autograd is enabled"
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        return "q/k/v are not all on CUDA"
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        return f"q/k/v must be 4-D, got {q.ndim}/{k.ndim}/{v.ndim}"
    if q.shape != k.shape or q.shape != v.shape:
        return f"q/k/v shapes differ: {tuple(q.shape)}/{tuple(k.shape)}/{tuple(v.shape)}"
    if q.dtype not in {torch.float16, torch.bfloat16}:
        return f"dtype {q.dtype} is not fp16/bf16"
    if k.dtype != q.dtype or v.dtype != q.dtype:
        return "q/k/v dtypes differ"

    _batch, num_heads, num_tokens, head_dim = q.shape
    if num_heads != 32:
        return f"kernel is specialised for 32 heads, got {num_heads}"
    if head_dim != _MAIN_BLOCK_D:
        return f"kernel is specialised for head_dim {_MAIN_BLOCK_D}, got {head_dim}"
    if not (0 < num_tokens <= 4096):
        return f"num_tokens {num_tokens} outside (0, 4096]"
    return None


def _supports_main_attention_pscore(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> bool:
    return _unsupported_reason(q, k, v) is None


def _sdpa_lse_cudnn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    try:
        out = torch.ops.aten._scaled_dot_product_cudnn_attention(
            q,
            k,
            v,
            None,
            True,
            0.0,
            False,
            False,
            scale=float(scale),
        )
    except RuntimeError as exc:
        # Do not swallow this. A RuntimeError here is usually Triton failing to compile, which is
        # indistinguishable from "unsupported shape" to the caller and produces a silently slower
        # run rather than an error.
        note_fallback("sdpa_with_pscore_triton", "the fused attention call raised",
                      detail=f"{type(exc).__name__}: {exc}")
        return None

    attn_out, lse = out[0], out[1]
    if not torch.is_tensor(attn_out) or not torch.is_tensor(lse):
        return None
    if lse.ndim == 4 and lse.shape[-1] == 1:
        lse = lse.squeeze(-1)
    if lse.ndim != 3:
        return None
    return attn_out, lse


def _sdpa_with_pscore_from_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    attn_out: torch.Tensor,
    lse: torch.Tensor,
    scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch, num_heads, num_tokens, _head_dim = q.shape
    num_k_blocks = triton.cdiv(num_tokens, _MAIN_BLOCK_N)
    pscore_acc = torch.zeros((batch, num_tokens), device=q.device, dtype=torch.float32)

    with torch.cuda.device(q.device):
        _pscore_from_cudnn_lse_keyblock_atomic_kernel[(batch, num_heads, num_k_blocks)](
            q,
            k,
            lse,
            pscore_acc,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            q.stride(3),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            k.stride(3),
            lse.stride(0),
            lse.stride(1),
            lse.stride(2),
            pscore_acc.stride(0),
            pscore_acc.stride(1),
            num_tokens,
            float(scale) * _LOG2_E,
            1.0 / float(num_heads * num_tokens),
            BLOCK_M=_MAIN_BLOCK_M,
            BLOCK_N=_MAIN_BLOCK_N,
            BLOCK_D=_MAIN_BLOCK_D,
            num_warps=_MAIN_NUM_WARPS,
            num_stages=_MAIN_NUM_STAGES,
        )

    return attn_out, pscore_acc.to(dtype=torch.bfloat16)


def sdpa_with_pscore_triton(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    scale: float,
    *,
    debug_name: str | None = None,
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    """Return cuDNN SDPA output and exact key-side mean attention probability.

    This is intentionally specialized for the ADE20K M2F hot path:
    CUDA inference, 32 heads, bf16/fp16, head_dim=128, no mask, no dropout.
    Unsupported shapes return None so the caller can use the dense fallback.
    """
    _ = debug_name
    reason = _unsupported_reason(q, k, v)
    if reason is not None:
        note_fallback("sdpa_with_pscore_triton", reason)
        return None

    cudnn_out = _sdpa_lse_cudnn(q, k, v, scale)
    if cudnn_out is None:
        note_fallback("sdpa_with_pscore_triton", "cuDNN SDPA did not return a usable (out, lse)")
        return None

    attn_out, lse = cudnn_out
    return _sdpa_with_pscore_from_lse(q, k, attn_out, lse, scale)
