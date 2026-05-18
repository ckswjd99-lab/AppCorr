import triton
import triton.language as tl

import torch
from typing import Tuple


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
