import triton
import triton.language as tl

import torch
from typing import Tuple


@triton.jit
def _rope_partial_batch_kernel(
    Q_ptr, K_ptr,         # [B, H, N, D]
    Cos_ptr, Sin_ptr,     # [Grid, D]
    Ind_ptr,              # [B, Num_Patches] -> 2D Tensor Input
    # Strides for 4D Q/K
    stride_q_b, stride_q_h, stride_q_n, stride_q_d,
    stride_k_b, stride_k_h, stride_k_n, stride_k_d,
    # Strides for Cos/Sin
    stride_cos_n, stride_cos_d,
    stride_sin_n, stride_sin_d,
    # Strides for Indices (New!)
    stride_ind_b, stride_ind_n, 
    prefix_len,
    HALF_BLOCK: tl.constexpr
):
    # Grid: (Num_Patches, Batch, Head)
    pid_patch = tl.program_id(0) # Column index for dindice
    pid_b = tl.program_id(1)     # Batch index
    pid_h = tl.program_id(2)     # Head index

    # Load Patch Index (Batch-Aware)
    ind_offset = (pid_b * stride_ind_b) + (pid_patch * stride_ind_n)
    global_idx = tl.load(Ind_ptr + ind_offset)
    
    # RoPE Table Index (Assume RoPE table starts from first patch)
    rope_idx = global_idx - prefix_len

    # Calculate Pointers (Using explicit 4D strides)
    # Q_ptr + b*SB + h*SH + n*SN
    q_offset = (pid_b * stride_q_b) + (pid_h * stride_q_h) + (global_idx * stride_q_n)
    k_offset = (pid_b * stride_k_b) + (pid_h * stride_k_h) + (global_idx * stride_k_n)
    
    # Cos/Sin Pointers
    cos_offset = (rope_idx * stride_cos_n)
    sin_offset = (rope_idx * stride_sin_n)

    # RoPE Rotation Logic (Same as before)
    r_l = tl.arange(0, HALF_BLOCK)
    r_r = r_l + HALF_BLOCK

    # Load Vectors
    q_l = tl.load(Q_ptr + q_offset + r_l * stride_q_d)
    q_r = tl.load(Q_ptr + q_offset + r_r * stride_q_d)
    k_l = tl.load(K_ptr + k_offset + r_l * stride_k_d)
    k_r = tl.load(K_ptr + k_offset + r_r * stride_k_d)

    c_l = tl.load(Cos_ptr + cos_offset + r_l * stride_cos_d)
    c_r = tl.load(Cos_ptr + cos_offset + r_r * stride_cos_d)
    s_l = tl.load(Sin_ptr + sin_offset + r_l * stride_sin_d)
    s_r = tl.load(Sin_ptr + sin_offset + r_r * stride_sin_d)

    # Apply Rotation: [-x2, x1]
    q_new_l = q_l * c_l - q_r * s_l
    q_new_r = q_r * c_r + q_l * s_r
    k_new_l = k_l * c_l - k_r * s_l
    k_new_r = k_r * c_r + k_l * s_r

    # Store (In-place)
    tl.store(Q_ptr + q_offset + r_l * stride_q_d, q_new_l)
    tl.store(Q_ptr + q_offset + r_r * stride_q_d, q_new_r)
    tl.store(K_ptr + k_offset + r_l * stride_k_d, k_new_l)
    tl.store(K_ptr + k_offset + r_r * stride_k_d, k_new_r)


def apply_rope_partial_triton(
    q: torch.Tensor, 
    k: torch.Tensor, 
    rope: tuple, 
    dindice: torch.Tensor
) -> tuple:
    # q, k: [B, H, N, D]
    # dindice: [B, Total_Updates] (Includes Pretokens + Patches)
    
    sin, cos = rope
    B, H, N, D = q.shape
    prefix_len = N - sin.shape[-2] # usually 5 (pretokens)
    
    # Early exit / Validate
    if dindice.shape[1] <= prefix_len:
        return q, k

    # Prepare Indices (Slicing 2D Tensor)
    dindice_patches = dindice[:, prefix_len:].contiguous()
    
    num_patches_per_batch = dindice_patches.shape[1]

    # Kernel Launch
    # Grid: (Patch Dim, Batch Dim, Head Dim)
    grid = (num_patches_per_batch, B, H)
    
    half_block_val = int(D // 2)

    _rope_partial_batch_kernel[grid](
        q, k,
        cos, sin,
        dindice_patches,
        # Strides for Q/K
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        # Strides for Cos/Sin
        cos.stride(-2), cos.stride(-1),
        sin.stride(-2), sin.stride(-1),
        # Strides for Indices (Batch Stride, Column Stride)
        dindice_patches.stride(0), dindice_patches.stride(1),
        prefix_len,
        HALF_BLOCK=half_block_val
    )

    return q, k


@triton.jit
def _rope_active_inplace_kernel(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    patch_mask_ptr,
    rope_idx_ptr,
    stride_q_t,
    stride_q_h,
    stride_q_d,
    stride_k_t,
    stride_k_h,
    stride_k_d,
    stride_cos_n,
    stride_cos_d,
    stride_sin_n,
    stride_sin_d,
    num_active,
    HALF_DIM: tl.constexpr,
    BLOCK_HALF: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    if pid_t >= num_active:
        return

    is_patch = tl.load(patch_mask_ptr + pid_t)
    rope_idx = tl.load(rope_idx_ptr + pid_t)

    offs_l = tl.arange(0, BLOCK_HALF)
    mask = offs_l < HALF_DIM
    offs_r = offs_l + HALF_DIM

    q_base = q_ptr + pid_t * stride_q_t + pid_h * stride_q_h
    k_base = k_ptr + pid_t * stride_k_t + pid_h * stride_k_h
    cos_base = cos_ptr + rope_idx * stride_cos_n
    sin_base = sin_ptr + rope_idx * stride_sin_n

    q_l = tl.load(q_base + offs_l * stride_q_d, mask=mask)
    q_r = tl.load(q_base + offs_r * stride_q_d, mask=mask)
    k_l = tl.load(k_base + offs_l * stride_k_d, mask=mask)
    k_r = tl.load(k_base + offs_r * stride_k_d, mask=mask)

    c_l = tl.load(cos_base + offs_l * stride_cos_d, mask=mask)
    c_r = tl.load(cos_base + offs_r * stride_cos_d, mask=mask)
    s_l = tl.load(sin_base + offs_l * stride_sin_d, mask=mask)
    s_r = tl.load(sin_base + offs_r * stride_sin_d, mask=mask)

    q_new_l = q_l * c_l - q_r * s_l
    q_new_r = q_r * c_r + q_l * s_r
    k_new_l = k_l * c_l - k_r * s_l
    k_new_r = k_r * c_r + k_l * s_r

    q_out_l = tl.where(is_patch, q_new_l, q_l)
    q_out_r = tl.where(is_patch, q_new_r, q_r)
    k_out_l = tl.where(is_patch, k_new_l, k_l)
    k_out_r = tl.where(is_patch, k_new_r, k_r)

    tl.store(q_base + offs_l * stride_q_d, q_out_l, mask=mask)
    tl.store(q_base + offs_r * stride_q_d, q_out_r, mask=mask)
    tl.store(k_base + offs_l * stride_k_d, k_out_l, mask=mask)
    tl.store(k_base + offs_r * stride_k_d, k_out_r, mask=mask)


def apply_rope_active_inplace_triton(
    q_active: torch.Tensor,
    k_active: torch.Tensor,
    rope: Tuple[torch.Tensor, torch.Tensor],
    patch_mask: torch.Tensor,
    rope_idx: torch.Tensor,
) -> None:
    if not (
        q_active.is_cuda
        and k_active.is_cuda
        and patch_mask.is_cuda
        and rope_idx.is_cuda
    ):
        raise RuntimeError("apply_rope_active_inplace_triton requires CUDA tensors")

    sin, cos = rope
    if not (sin.is_cuda and cos.is_cuda):
        raise RuntimeError("apply_rope_active_inplace_triton requires CUDA RoPE tensors")
    if q_active.device != k_active.device or q_active.device != sin.device or q_active.device != cos.device:
        raise RuntimeError("apply_rope_active_inplace_triton tensors must be on the same CUDA device")
    if q_active.ndim != 3 or k_active.ndim != 3:
        raise RuntimeError("q_active and k_active must have shape [T, H, D]")
    if q_active.shape != k_active.shape:
        raise RuntimeError(f"q/k shape mismatch: {tuple(q_active.shape)} vs {tuple(k_active.shape)}")
    if patch_mask.shape[0] != q_active.shape[0] or rope_idx.shape[0] != q_active.shape[0]:
        raise RuntimeError("patch_mask and rope_idx must have one entry per active token")
    if patch_mask.ndim != 1 or rope_idx.ndim != 1 or patch_mask.stride(0) != 1 or rope_idx.stride(0) != 1:
        raise RuntimeError("patch_mask and rope_idx must be contiguous 1D tensors")

    num_active, num_heads, head_dim = q_active.shape
    if num_active == 0 or head_dim == 0:
        return
    if head_dim % 2 != 0:
        raise RuntimeError(f"RoPE head_dim must be even, got {head_dim}")

    half_dim = head_dim // 2
    block_half = triton.next_power_of_2(half_dim)
    grid = (num_active, num_heads)

    with torch.cuda.device(q_active.device):
        _rope_active_inplace_kernel[grid](
            q_active,
            k_active,
            cos,
            sin,
            patch_mask,
            rope_idx,
            q_active.stride(0), q_active.stride(1), q_active.stride(2),
            k_active.stride(0), k_active.stride(1), k_active.stride(2),
            cos.stride(-2), cos.stride(-1),
            sin.stride(-2), sin.stride(-1),
            num_active,
            HALF_DIM=half_dim,
            BLOCK_HALF=block_half,
        )


