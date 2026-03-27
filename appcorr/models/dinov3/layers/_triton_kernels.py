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
def _masked_residual_add_kernel(
    out_ptr,
    x_sel_ptr,
    x_old_ptr,
    dx_ptr,
    valid_ptr,
    stride_out_b, stride_out_m, stride_out_c,
    stride_xsel_b, stride_xsel_m, stride_xsel_c,
    stride_xold_b, stride_xold_m, stride_xold_c,
    stride_dx_b, stride_dx_m, stride_dx_c,
    stride_valid_b, stride_valid_m,
    num_tokens_sel,
    dim_c,
    BLOCK_C: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_c = tl.program_id(2)

    if pid_m >= num_tokens_sel:
        return

    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = offs_c < dim_c

    x_sel = tl.load(
        x_sel_ptr + pid_b * stride_xsel_b + pid_m * stride_xsel_m + offs_c * stride_xsel_c,
        mask=c_mask,
    )
    x_old = tl.load(
        x_old_ptr + pid_b * stride_xold_b + pid_m * stride_xold_m + offs_c * stride_xold_c,
        mask=c_mask,
    )
    dx = tl.load(
        dx_ptr + pid_b * stride_dx_b + pid_m * stride_dx_m + offs_c * stride_dx_c,
        mask=c_mask,
    )
    is_valid = tl.load(valid_ptr + pid_b * stride_valid_b + pid_m * stride_valid_m)
    dx = tl.where(is_valid != 0, dx, 0.0)
    out = x_sel + x_old + dx
    tl.store(
        out_ptr + pid_b * stride_out_b + pid_m * stride_out_m + offs_c * stride_out_c,
        out,
        mask=c_mask,
    )


def masked_residual_add_triton(
    x_sel: torch.Tensor,
    x_old: torch.Tensor,
    dx: torch.Tensor,
    query_valid_mask: torch.Tensor,
) -> torch.Tensor:
    if (
        not x_sel.is_cuda
        or not x_old.is_cuda
        or not dx.is_cuda
        or not query_valid_mask.is_cuda
    ):
        valid = query_valid_mask.unsqueeze(-1).to(dtype=dx.dtype)
        return x_sel + x_old.to(dtype=x_sel.dtype) + dx.to(dtype=x_sel.dtype) * valid

    out = torch.empty_like(x_sel)
    x_old = x_old.to(dtype=x_sel.dtype).contiguous()
    dx = dx.to(dtype=x_sel.dtype).contiguous()
    query_valid_mask = query_valid_mask.contiguous()
    x_sel = x_sel.contiguous()

    B, num_tokens_sel, dim_c = x_sel.shape
    block_c = 128
    grid = (B, num_tokens_sel, triton.cdiv(dim_c, block_c))

    with torch.cuda.device(x_sel.device):
        _masked_residual_add_kernel[grid](
            out,
            x_sel,
            x_old,
            dx,
            query_valid_mask,
            out.stride(0), out.stride(1), out.stride(2),
            x_sel.stride(0), x_sel.stride(1), x_sel.stride(2),
            x_old.stride(0), x_old.stride(1), x_old.stride(2),
            dx.stride(0), dx.stride(1), dx.stride(2),
            query_valid_mask.stride(0), query_valid_mask.stride(1),
            num_tokens_sel,
            dim_c,
            BLOCK_C=block_c,
        )

    return out


@triton.jit
def _masked_token_update_kernel(
    x_out_ptr,
    x_attn_ptr,
    x_delta_ptr,
    dindice_ptr,
    valid_ptr,
    stride_xout_b, stride_xout_n, stride_xout_c,
    stride_xattn_b, stride_xattn_m, stride_xattn_c,
    stride_xdelta_b, stride_xdelta_m, stride_xdelta_c,
    stride_dindice_b, stride_dindice_m,
    stride_valid_b, stride_valid_m,
    num_tokens_sel,
    dim_c,
    BLOCK_C: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_c = tl.program_id(2)

    if pid_m >= num_tokens_sel:
        return

    is_valid = tl.load(valid_ptr + pid_b * stride_valid_b + pid_m * stride_valid_m)
    if is_valid == 0:
        return

    token_idx = tl.load(dindice_ptr + pid_b * stride_dindice_b + pid_m * stride_dindice_m)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = offs_c < dim_c

    x_attn = tl.load(
        x_attn_ptr + pid_b * stride_xattn_b + pid_m * stride_xattn_m + offs_c * stride_xattn_c,
        mask=c_mask,
    )
    x_delta = tl.load(
        x_delta_ptr + pid_b * stride_xdelta_b + pid_m * stride_xdelta_m + offs_c * stride_xdelta_c,
        mask=c_mask,
    )
    tl.store(
        x_out_ptr + pid_b * stride_xout_b + token_idx * stride_xout_n + offs_c * stride_xout_c,
        x_attn + x_delta,
        mask=c_mask,
    )


def masked_token_update_triton(
    x_base: torch.Tensor,
    dindice_sel: torch.Tensor,
    x_attn_sel: torch.Tensor,
    x_delta: torch.Tensor,
    query_valid_mask: torch.Tensor,
) -> torch.Tensor:
    if (
        not x_base.is_cuda
        or not dindice_sel.is_cuda
        or not x_attn_sel.is_cuda
        or not x_delta.is_cuda
        or not query_valid_mask.is_cuda
    ):
        x_out = x_base.clone()
        for b in range(x_base.shape[0]):
            valid = query_valid_mask[b]
            if not torch.any(valid):
                continue
            idx = dindice_sel[b, valid]
            x_out[b, idx] = (x_attn_sel[b, valid] + x_delta[b, valid]).to(dtype=x_out.dtype)
        return x_out

    x_out = x_base.clone()
    dindice_sel = dindice_sel.contiguous()
    x_attn_sel = x_attn_sel.to(dtype=x_out.dtype).contiguous()
    x_delta = x_delta.to(dtype=x_out.dtype).contiguous()
    query_valid_mask = query_valid_mask.contiguous()

    B, num_tokens_sel, dim_c = x_attn_sel.shape
    block_c = 128
    grid = (B, num_tokens_sel, triton.cdiv(dim_c, block_c))

    with torch.cuda.device(x_base.device):
        _masked_token_update_kernel[grid](
            x_out,
            x_attn_sel,
            x_delta,
            dindice_sel,
            query_valid_mask,
            x_out.stride(0), x_out.stride(1), x_out.stride(2),
            x_attn_sel.stride(0), x_attn_sel.stride(1), x_attn_sel.stride(2),
            x_delta.stride(0), x_delta.stride(1), x_delta.stride(2),
            dindice_sel.stride(0), dindice_sel.stride(1),
            query_valid_mask.stride(0), query_valid_mask.stride(1),
            num_tokens_sel,
            dim_c,
            BLOCK_C=block_c,
        )

    return x_out


@triton.jit
def _token_prune_select_compact_kernel(
    dx_ptr, dindice_ptr,
    out_dindice_sel_ptr, out_query_pos_idx_ptr, out_query_valid_mask_ptr, out_kept_patch_count_ptr,
    stride_dx_b, stride_dx_m, stride_dx_c,
    stride_dindice_b, stride_dindice_m,
    stride_out_dindice_b, stride_out_dindice_m,
    stride_out_qpos_b, stride_out_qpos_m,
    stride_out_valid_b, stride_out_valid_m,
    stride_out_count_b,
    num_tokens_sel,
    dim_c,
    num_pretokens,
    token_prune_threshold,
    token_prune_min_keep,
    BLOCK_M: tl.constexpr,
    BLOCK_C: tl.constexpr,
    TOPK_MAX: tl.constexpr,
):
    pid_b = tl.program_id(0)
    offs = tl.arange(0, BLOCK_M)
    token_mask = offs < num_tokens_sel

    score_acc = tl.zeros([BLOCK_M], dtype=tl.float32)
    for c_start in tl.range(0, dim_c, BLOCK_C):
        offs_c = c_start + tl.arange(0, BLOCK_C)
        c_mask = offs_c < dim_c
        dx = tl.load(
            dx_ptr
            + pid_b * stride_dx_b
            + offs[:, None] * stride_dx_m
            + offs_c[None, :] * stride_dx_c,
            mask=token_mask[:, None] & c_mask[None, :],
            other=0.0,
        )
        score_acc += tl.sum(tl.abs(dx), axis=1)
    scores = score_acc / dim_c
    scores = tl.where(token_mask, scores, float("-inf"))
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
    dx: torch.Tensor,
    dindice: torch.Tensor,
    num_pretokens: int,
    token_prune_threshold: float,
    token_prune_min_keep: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    scores = dx.abs().mean(dim=-1)
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
    dx: torch.Tensor,
    dindice: torch.Tensor,
    num_pretokens: int,
    token_prune_threshold: float,
    token_prune_min_keep: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dx = dx.contiguous()
    dindice = dindice.to(device=dx.device, dtype=torch.long, non_blocking=True).contiguous()

    if (
        not dx.is_cuda
        or not dindice.is_cuda
        or dx.ndim != 3
        or dindice.ndim != 2
        or dx.shape[:2] != dindice.shape
    ):
        raise RuntimeError(
            "token_prune_select_compact_triton requires CUDA tensors `dx[B, M, C]` and "
            f"`dindice[B, M]`, but got dx(shape={tuple(dx.shape)}, cuda={dx.is_cuda}) "
            f"and dindice(shape={tuple(dindice.shape)}, cuda={dindice.is_cuda})."
        )

    B, num_tokens_sel, dim_c = dx.shape
    if num_tokens_sel == 0:
        empty_mask = torch.zeros_like(dindice, dtype=torch.bool)
        empty_count = torch.zeros((B,), device=dx.device, dtype=torch.int32)
        return torch.zeros_like(dindice), torch.zeros_like(dindice), empty_mask, empty_count

    block_m = triton.next_power_of_2(num_tokens_sel)
    block_c = 128
    topk_max = 16
    if block_m > 128 or token_prune_min_keep > topk_max:
        raise RuntimeError(
            "token_prune_select_compact_triton shape is outside the current Triton specialization: "
            f"num_tokens_sel={num_tokens_sel}, BLOCK_M={block_m}, token_prune_min_keep={token_prune_min_keep}, "
            f"TOPK_MAX={topk_max}."
        )

    dindice_sel = torch.zeros_like(dindice)
    query_pos_idx = torch.zeros_like(dindice)
    query_valid_mask_i8 = torch.zeros((B, num_tokens_sel), device=dx.device, dtype=torch.int8)
    kept_patch_count = torch.zeros((B,), device=dx.device, dtype=torch.int32)

    with torch.cuda.device(dx.device):
        _token_prune_select_compact_kernel[(B,)](
            dx, dindice,
            dindice_sel, query_pos_idx, query_valid_mask_i8, kept_patch_count,
            dx.stride(0), dx.stride(1), dx.stride(2),
            dindice.stride(0), dindice.stride(1),
            dindice_sel.stride(0), dindice_sel.stride(1),
            query_pos_idx.stride(0), query_pos_idx.stride(1),
            query_valid_mask_i8.stride(0), query_valid_mask_i8.stride(1),
            kept_patch_count.stride(0),
            num_tokens_sel,
            dim_c,
            num_pretokens,
            token_prune_threshold,
            token_prune_min_keep,
            BLOCK_M=block_m,
            BLOCK_C=block_c,
            TOPK_MAX=topk_max,
        )

    return dindice_sel, query_pos_idx, query_valid_mask_i8.to(dtype=torch.bool), kept_patch_count
