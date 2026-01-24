import triton
import triton.language as tl

import torch


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
def _fused_layerscale_add_kernel(
    x_ptr,           # Pointer to input x (Residual)
    attn_ptr,        # Pointer to x_attn (To be scaled)
    gamma_ptr,       # Pointer to gamma (LayerScale parameter)
    out_ptr,         # Pointer to output
    n_elements,      # Total number of elements (B * N * D)
    d_dim,           # Dimension D (for broadcasting gamma)
    BLOCK_SIZE: tl.constexpr,
):
    # Map program ID to data offset
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to prevent out-of-bounds access
    mask = offsets < n_elements

    # Load gamma index: Since gamma is (D,), we need modulo operator
    # This enables broadcasting: (B, N, D) vs (D,)
    gamma_offsets = offsets % d_dim

    # Load data
    x_val = tl.load(x_ptr + offsets, mask=mask)
    attn_val = tl.load(attn_ptr + offsets, mask=mask)
    gamma_val = tl.load(gamma_ptr + gamma_offsets, mask=mask)

    # Computation: x + attn * gamma
    output = x_val + attn_val * gamma_val

    # Store result
    tl.store(out_ptr + offsets, output, mask=mask)

def fused_layerscale_add(x, x_attn, gamma):
    # Flatten inputs to treat them as 1D vectors
    n_elements = x.numel()
    d_dim = x.shape[-1]
    
    # Allocate output buffer
    output = torch.empty_like(x)
    
    # Grid definition: How many blocks needed
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    
    # Launch kernel
    _fused_layerscale_add_kernel[grid](
        x, x_attn, gamma, output,
        n_elements, d_dim,
        BLOCK_SIZE=1024, # Optimized for most GPU architectures
    )
    return output
