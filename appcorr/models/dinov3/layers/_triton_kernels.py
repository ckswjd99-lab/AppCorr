import triton
import triton.language as tl

import torch


@triton.jit
def _rope_full_kernel(
    Q_ptr, K_ptr,         # [B, H, N, D]
    Cos_ptr, Sin_ptr,     # [N, D]
    # Strides for Q/K
    stride_q_b, stride_q_h, stride_q_n, stride_q_d,
    stride_k_b, stride_k_h, stride_k_n, stride_k_d,
    # Strides for Cos/Sin
    stride_cos_n, stride_cos_d,
    stride_sin_n, stride_sin_d,
    HALF_BLOCK: tl.constexpr
):
    # Grid: (N, Batch, Head) -> Directly maps to token position
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    # 1. Offsets Calculation
    # No indirection needed, pid_n is the sequence index
    q_offset = (pid_b * stride_q_b) + (pid_h * stride_q_h) + (pid_n * stride_q_n)
    k_offset = (pid_b * stride_k_b) + (pid_h * stride_k_h) + (pid_n * stride_k_n)
    
    # Cos/Sin usually aligns with sequence index
    cos_offset = (pid_n * stride_cos_n)
    sin_offset = (pid_n * stride_sin_n)

    # 2. RoPE Rotation Logic (Identical to partial)
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


def apply_rope_full_triton(
    q: torch.Tensor, 
    k: torch.Tensor, 
    rope: tuple
) -> tuple:
    # q, k: [B, H, N, D]
    sin, cos = rope
    B, H, N, D = q.shape

    # 1. Kernel Launch Config
    # Grid maps to (Sequence_Length, Batch, Head)
    grid = (N, B, H)
    
    half_block_val = int(D // 2)

    # 2. Launch
    _rope_full_kernel[grid](
        q, k,
        cos, sin,
        # Strides
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        cos.stride(-2), cos.stride(-1),
        sin.stride(-2), sin.stride(-1),
        # Const
        HALF_BLOCK=half_block_val
    )

    return q, k


@triton.jit
def _rope_partial_kernel(
    Q_ptr, K_ptr,         # [B, H, N, D]
    Cos_ptr, Sin_ptr,     # [Grid, D]
    Ind_ptr,              # [Num_Patches]
    # Strides for 4D Q/K
    stride_q_b, stride_q_h, stride_q_n, stride_q_d,
    stride_k_b, stride_k_h, stride_k_n, stride_k_d,
    # Strides for Cos/Sin
    stride_cos_n, stride_cos_d,
    stride_sin_n, stride_sin_d,
    prefix_len,
    HALF_BLOCK: tl.constexpr # <--- 여기에 주목. 아예 반 쪼개서 받음.
):
    # Grid: (Num_Patches, Batch, Head)
    pid_patch = tl.program_id(0)
    pid_b = tl.program_id(1)
    pid_h = tl.program_id(2)

    # 1. Load Patch Index
    global_idx = tl.load(Ind_ptr + pid_patch)
    rope_idx = global_idx - prefix_len

    # 2. Calculate Pointers (Using explicit 4D strides)
    # Q_ptr + b*SB + h*SH + n*SN
    q_offset = (pid_b * stride_q_b) + (pid_h * stride_q_h) + (global_idx * stride_q_n)
    k_offset = (pid_b * stride_k_b) + (pid_h * stride_k_h) + (global_idx * stride_k_n)
    
    # Cos/Sin Pointers
    cos_offset = (rope_idx * stride_cos_n)
    sin_offset = (rope_idx * stride_sin_n)

    # 3. RoPE Rotation Logic
    # arange에 상수를 바로 꽂았으니 에러가 날 수 없음
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
    sin, cos = rope
    B, H, N, D = q.shape
    prefix_len = N - sin.shape[-2]
    
    # 1. Early exit
    if dindice.shape[0] <= prefix_len:
        return q, k

    # 2. Prepare Inputs
    dindice_patches = dindice[prefix_len:].contiguous()
    num_patches = dindice_patches.shape[0]

    # 3. Kernel Launch
    grid = (num_patches, B, H)
    
    # D는 Head Dimension (보통 64, 128 등 짝수)
    # 커널 내부에서 연산 안 하고 Python에서 미리 나눠서 상수로 넘김
    half_block_val = int(D // 2)

    _rope_partial_kernel[grid](
        q, k,
        cos, sin,
        dindice_patches,
        # Strides
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        cos.stride(-2), cos.stride(-1),
        sin.stride(-2), sin.stride(-1),
        prefix_len,
        # arange 에러 방지용 상수 전달
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