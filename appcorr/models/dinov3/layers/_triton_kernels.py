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


@triton.jit
def _token_prune_select_compact_kernel(
    scores_ptr, dindice_ptr,
    out_dindice_sel_ptr, out_query_pos_idx_ptr, out_query_valid_mask_ptr, out_kept_patch_count_ptr,
    stride_scores_b, stride_scores_m,
    stride_dindice_b, stride_dindice_m,
    stride_out_dindice_b, stride_out_dindice_m,
    stride_out_qpos_b, stride_out_qpos_m,
    stride_out_valid_b, stride_out_valid_m,
    stride_out_count_b,
    num_tokens_sel,
    num_pretokens,
    token_prune_threshold,
    token_prune_min_keep,
    BLOCK_M: tl.constexpr,
    TOPK_MAX: tl.constexpr,
):
    pid_b = tl.program_id(0)
    offs = tl.arange(0, BLOCK_M)
    token_mask = offs < num_tokens_sel

    scores = tl.load(scores_ptr + pid_b * stride_scores_b + offs * stride_scores_m, mask=token_mask, other=float("-inf"))
    dindice = tl.load(dindice_ptr + pid_b * stride_dindice_b + offs * stride_dindice_m, mask=token_mask, other=0)

    patch_mask = (offs >= num_pretokens) & token_mask
    patch_count = tl.maximum(num_tokens_sel - num_pretokens, 0)
    threshold_keep = patch_mask & (scores >= token_prune_threshold)
    threshold_count = tl.sum(threshold_keep, axis=0)
    use_topk = (patch_count > 0) & (threshold_count < token_prune_min_keep)
    k = tl.minimum(token_prune_min_keep, patch_count)

    selected_topk = tl.zeros([BLOCK_M], dtype=tl.int1)
    neg_inf = tl.full([BLOCK_M], float("-inf"), tl.float32)
    large_idx = tl.full([BLOCK_M], BLOCK_M, tl.int32)
    for pick_iter in range(TOPK_MAX):
        masked_scores = tl.where(patch_mask & (~selected_topk), scores, neg_inf)
        best_score = tl.max(masked_scores, axis=0)
        best_mask = patch_mask & (~selected_topk) & (scores == best_score)
        best_idx_candidates = tl.where(best_mask, offs, large_idx)
        best_idx = tl.min(best_idx_candidates, axis=0)
        selected_topk = selected_topk | ((offs == best_idx) & (pick_iter < k))

    selected_patch = tl.where(use_topk, selected_topk, threshold_keep)
    prefix_keep = (offs < num_pretokens) & token_mask
    selected_all = prefix_keep | selected_patch
    out_pos = tl.cumsum(selected_all.to(tl.int32), axis=0) - 1

    store_mask = selected_all & token_mask
    tl.store(
        out_query_pos_idx_ptr + pid_b * stride_out_qpos_b + out_pos * stride_out_qpos_m,
        offs,
        mask=store_mask,
    )
    tl.store(
        out_dindice_sel_ptr + pid_b * stride_out_dindice_b + out_pos * stride_out_dindice_m,
        dindice,
        mask=store_mask,
    )
    tl.store(
        out_query_valid_mask_ptr + pid_b * stride_out_valid_b + out_pos * stride_out_valid_m,
        1,
        mask=store_mask,
    )

    kept_patch_count = tl.sum(selected_patch, axis=0)
    tl.store(out_kept_patch_count_ptr + pid_b * stride_out_count_b, kept_patch_count)


def _token_prune_select_compact_torch(
    scores: torch.Tensor,
    dindice: torch.Tensor,
    num_pretokens: int,
    token_prune_threshold: float,
    token_prune_min_keep: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    B, num_tokens_sel = scores.shape
    dindice_sel = torch.zeros_like(dindice)
    query_pos_idx = torch.zeros_like(dindice)
    query_valid_mask = torch.zeros((B, num_tokens_sel), device=scores.device, dtype=torch.bool)
    kept_patch_count = torch.zeros((B,), device=scores.device, dtype=torch.int32)

    if num_tokens_sel == 0:
        return dindice_sel, query_pos_idx, query_valid_mask, kept_patch_count

    prefix_pos = torch.arange(num_pretokens, device=scores.device, dtype=torch.long)
    if num_pretokens > 0:
        query_pos_idx[:, :num_pretokens] = prefix_pos
        dindice_sel[:, :num_pretokens] = dindice[:, :num_pretokens]
        query_valid_mask[:, :num_pretokens] = True

    patch_scores = scores[:, num_pretokens:]
    if patch_scores.shape[1] == 0:
        return dindice_sel, query_pos_idx, query_valid_mask, kept_patch_count

    for b in range(B):
        keep_patch = torch.where(patch_scores[b] >= token_prune_threshold)[0] + num_pretokens
        if keep_patch.numel() < token_prune_min_keep:
            k = min(token_prune_min_keep, patch_scores.shape[1])
            if k > 0:
                keep_patch = torch.topk(patch_scores[b], k=k, dim=0, largest=True).indices + num_pretokens
        keep_patch, _ = torch.sort(keep_patch)
        out_pos = num_pretokens + keep_patch.numel()
        if keep_patch.numel() > 0:
            query_pos_idx[b, num_pretokens:out_pos] = keep_patch
            dindice_sel[b, num_pretokens:out_pos] = dindice[b, keep_patch]
            query_valid_mask[b, num_pretokens:out_pos] = True
        kept_patch_count[b] = keep_patch.numel()

    return dindice_sel, query_pos_idx, query_valid_mask, kept_patch_count


def token_prune_select_compact_triton(
    scores: torch.Tensor,
    dindice: torch.Tensor,
    num_pretokens: int,
    token_prune_threshold: float,
    token_prune_min_keep: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    scores = scores.contiguous()
    dindice = dindice.to(device=scores.device, dtype=torch.long, non_blocking=True).contiguous()

    if (
        not scores.is_cuda
        or not dindice.is_cuda
        or scores.ndim != 2
        or dindice.ndim != 2
        or scores.shape != dindice.shape
    ):
        raise RuntimeError(
            "token_prune_select_compact_triton requires matching 2D CUDA tensors for "
            f"`scores` and `dindice`, but got scores(shape={tuple(scores.shape)}, cuda={scores.is_cuda}) "
            f"and dindice(shape={tuple(dindice.shape)}, cuda={dindice.is_cuda})."
        )

    B, num_tokens_sel = scores.shape
    if num_tokens_sel == 0:
        empty_mask = torch.zeros_like(dindice, dtype=torch.bool)
        empty_count = torch.zeros((B,), device=scores.device, dtype=torch.int32)
        return torch.zeros_like(dindice), torch.zeros_like(dindice), empty_mask, empty_count

    block_m = triton.next_power_of_2(num_tokens_sel)
    topk_max = 16
    if block_m > 128 or token_prune_min_keep > topk_max:
        raise RuntimeError(
            "token_prune_select_compact_triton shape is outside the current Triton specialization: "
            f"num_tokens_sel={num_tokens_sel}, BLOCK_M={block_m}, token_prune_min_keep={token_prune_min_keep}, "
            f"TOPK_MAX={topk_max}."
        )

    dindice_sel = torch.zeros_like(dindice)
    query_pos_idx = torch.zeros_like(dindice)
    query_valid_mask_i8 = torch.zeros((B, num_tokens_sel), device=scores.device, dtype=torch.int8)
    kept_patch_count = torch.zeros((B,), device=scores.device, dtype=torch.int32)

    with torch.cuda.device(scores.device):
        _token_prune_select_compact_kernel[(B,)](
            scores, dindice,
            dindice_sel, query_pos_idx, query_valid_mask_i8, kept_patch_count,
            scores.stride(0), scores.stride(1),
            dindice.stride(0), dindice.stride(1),
            dindice_sel.stride(0), dindice_sel.stride(1),
            query_pos_idx.stride(0), query_pos_idx.stride(1),
            query_valid_mask_i8.stride(0), query_valid_mask_i8.stride(1),
            kept_patch_count.stride(0),
            num_tokens_sel,
            num_pretokens,
            token_prune_threshold,
            token_prune_min_keep,
            BLOCK_M=block_m,
            TOPK_MAX=topk_max,
        )

    return dindice_sel, query_pos_idx, query_valid_mask_i8.to(dtype=torch.bool), kept_patch_count
