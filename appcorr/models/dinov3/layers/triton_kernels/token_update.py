import triton
import triton.language as tl

import torch
from typing import Tuple


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
    *,
    clone_base: bool = True,
) -> torch.Tensor:
    if (
        not x_base.is_cuda
        or not dindice_sel.is_cuda
        or not x_attn_sel.is_cuda
        or not x_delta.is_cuda
        or not query_valid_mask.is_cuda
    ):
        x_out = x_base.clone() if clone_base else x_base
        for b in range(x_base.shape[0]):
            valid = query_valid_mask[b]
            if not torch.any(valid):
                continue
            idx = dindice_sel[b, valid]
            x_out[b, idx] = (x_attn_sel[b, valid] + x_delta[b, valid]).to(dtype=x_out.dtype)
        return x_out

    x_out = x_base.clone() if clone_base else x_base
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
def _active_token_update_kernel(
    x_out_ptr,
    batch_idx_ptr,
    token_idx_ptr,
    x_attn_ptr,
    x_delta_ptr,
    stride_xout_b, stride_xout_n, stride_xout_c,
    stride_xattn_t, stride_xattn_c,
    stride_xdelta_t, stride_xdelta_c,
    num_active,
    dim_c,
    BLOCK_C: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_c = tl.program_id(1)

    if pid_t >= num_active:
        return

    batch_idx = tl.load(batch_idx_ptr + pid_t)
    token_idx = tl.load(token_idx_ptr + pid_t)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = offs_c < dim_c

    x_attn = tl.load(
        x_attn_ptr + pid_t * stride_xattn_t + offs_c * stride_xattn_c,
        mask=c_mask,
    )
    x_delta = tl.load(
        x_delta_ptr + pid_t * stride_xdelta_t + offs_c * stride_xdelta_c,
        mask=c_mask,
    )
    tl.store(
        x_out_ptr + batch_idx * stride_xout_b + token_idx * stride_xout_n + offs_c * stride_xout_c,
        x_attn + x_delta,
        mask=c_mask,
    )


def active_token_update_triton(
    x_base: torch.Tensor,
    active_batch_idx: torch.Tensor,
    active_token_idx: torch.Tensor,
    x_attn_active: torch.Tensor,
    x_delta_active: torch.Tensor,
    *,
    clone_base: bool = True,
) -> torch.Tensor:
    if active_batch_idx.numel() == 0:
        return x_base.clone() if clone_base else x_base

    if (
        not x_base.is_cuda
        or not active_batch_idx.is_cuda
        or not active_token_idx.is_cuda
        or not x_attn_active.is_cuda
        or not x_delta_active.is_cuda
    ):
        x_out = x_base.clone() if clone_base else x_base
        x_out[active_batch_idx, active_token_idx] = (
            x_attn_active + x_delta_active
        ).to(dtype=x_out.dtype)
        return x_out

    x_out = x_base.clone() if clone_base else x_base
    active_batch_idx = active_batch_idx.contiguous()
    active_token_idx = active_token_idx.contiguous()
    x_attn_active = x_attn_active.to(dtype=x_out.dtype).contiguous()
    x_delta_active = x_delta_active.to(dtype=x_out.dtype).contiguous()

    num_active, dim_c = x_attn_active.shape
    block_c = 128
    grid = (num_active, triton.cdiv(dim_c, block_c))

    with torch.cuda.device(x_base.device):
        _active_token_update_kernel[grid](
            x_out,
            active_batch_idx,
            active_token_idx,
            x_attn_active,
            x_delta_active,
            x_out.stride(0), x_out.stride(1), x_out.stride(2),
            x_attn_active.stride(0), x_attn_active.stride(1),
            x_delta_active.stride(0), x_delta_active.stride(1),
            num_active,
            dim_c,
            BLOCK_C=block_c,
        )

    return x_out




@triton.jit
def _scatter_rows_kernel(
    dst_ptr, batch_idx_ptr, token_idx_ptr, src_ptr,
    stride_db, stride_dn, stride_st,
    num_active, dim_c,
    BLOCK_C: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_c = tl.program_id(1)
    if pid_t >= num_active:
        return
    b = tl.load(batch_idx_ptr + pid_t)
    t = tl.load(token_idx_ptr + pid_t)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = offs_c < dim_c
    v = tl.load(src_ptr + pid_t * stride_st + offs_c, mask=c_mask)
    tl.store(dst_ptr + b * stride_db + t * stride_dn + offs_c, v, mask=c_mask)


@triton.jit
def _gather_rows_kernel(
    out_ptr, batch_idx_ptr, token_idx_ptr, src_ptr,
    stride_sb, stride_sn, stride_ot,
    num_active, dim_c,
    BLOCK_C: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_c = tl.program_id(1)
    if pid_t >= num_active:
        return
    b = tl.load(batch_idx_ptr + pid_t)
    t = tl.load(token_idx_ptr + pid_t)
    offs_c = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    c_mask = offs_c < dim_c
    v = tl.load(src_ptr + b * stride_sb + t * stride_sn + offs_c, mask=c_mask)
    tl.store(out_ptr + pid_t * stride_ot + offs_c, v, mask=c_mask)


def _trailing_is_packed(t: torch.Tensor, first_row_dim: int) -> bool:
    """True when dims from `first_row_dim` onward are laid out contiguously within a row.

    Weaker than `is_contiguous()` on purpose. `kv_new = qkv_new[:, 1:]` slices a [M, 3, H, Dh]
    tensor, so it is *not* contiguous -- its row stride is 3*H*Dh, not 2*H*Dh -- yet the 2*H*Dh
    elements of each row still sit back to back, which is all these kernels need (they take the row
    stride as a parameter). Requiring full contiguity here silently sent the K/V scatter back to the
    aten fallback, which is exactly the 6.5x that was meant to be removed.
    """
    expected = 1
    for d in range(t.dim() - 1, first_row_dim - 1, -1):
        if t.stride(d) != expected:
            return False
        expected *= t.shape[d]
    return True


def _rows_view(t: torch.Tensor):
    """Collapse a [B, N, ...] tensor to [B, N, C] without copying, if the row layout allows."""
    if t.dim() < 3 or not _trailing_is_packed(t, 2):
        return None
    C = 1
    for d in range(2, t.dim()):
        C *= t.shape[d]
    return t.as_strided((t.shape[0], t.shape[1], C), (t.stride(0), t.stride(1), 1))


def scatter_rows_triton(dst, batch_idx, token_idx, src, block_c: int = 1024) -> bool:
    """`dst[batch_idx, token_idx] = src`, but ~6.5x faster than aten advanced indexing.

    Measured at the ADE20K correction shapes (B=4, N=1029, M=2068, row = 2*32*128): aten
    `_index_put_impl_` 78.0 us vs 12.1 us here, bit-identical. `aten::index_put_` runs a generic
    kernel that recomputes an element-wise offset for every element; this one resolves the row
    address once per row and streams the row.

    **Assumes the (batch, token) pairs are unique**, which they are on the correction path --
    `active_token_idx` holds distinct selected tokens per image. With duplicates, both this and
    `index_put_` are last-writer-wins with no defined order, so results would differ between them.

    Returns False (writing nothing) when the layout is unsupported, so callers can fall back.
    """
    d = _rows_view(dst)
    if d is None or src.dim() < 2 or not _trailing_is_packed(src, 1):
        return False
    C = 1
    for _d in range(1, src.dim()):
        C *= src.shape[_d]
    if d.shape[-1] != C:
        return False
    s = src.as_strided((src.shape[0], C), (src.stride(0), 1))
    n, dim_c = batch_idx.numel(), d.shape[-1]
    if n == 0:
        return True
    _scatter_rows_kernel[(n, triton.cdiv(dim_c, block_c))](
        d, batch_idx, token_idx, s,
        d.stride(0), d.stride(1), s.stride(0),
        n, dim_c, BLOCK_C=block_c,
    )
    return True


def gather_rows_triton(src, batch_idx, token_idx, block_c: int = 1024):
    """`src[batch_idx, token_idx]` as a contiguous [M, C] tensor, ~3.6x faster than aten.

    41.0 us -> 11.4 us at the ADE20K correction shapes, bit-identical. Returns None when the
    layout is unsupported so callers can fall back.
    """
    s = _rows_view(src)
    if s is None:
        return None
    n, dim_c = batch_idx.numel(), s.shape[-1]
    out = torch.empty((n, dim_c), device=src.device, dtype=src.dtype)
    if n == 0:
        return out.reshape(n, *src.shape[2:])
    _gather_rows_kernel[(n, triton.cdiv(dim_c, block_c))](
        out, batch_idx, token_idx, s,
        s.stride(0), s.stride(1), out.stride(0),
        n, dim_c, BLOCK_C=block_c,
    )
    return out.reshape(n, *src.shape[2:])


@triton.jit
def _scatter_heads_kernel(
    dst_ptr, batch_idx_ptr, pos_idx_ptr, src_ptr,
    stride_db, stride_dh, stride_dt,
    stride_sm, stride_sh,
    num_active, head_dim,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    if pid_m >= num_active:
        return
    b = tl.load(batch_idx_ptr + pid_m)
    t = tl.load(pos_idx_ptr + pid_m)
    offs = tl.arange(0, BLOCK_D)
    d_mask = offs < head_dim
    v = tl.load(src_ptr + pid_m * stride_sm + pid_h * stride_sh + offs, mask=d_mask)
    tl.store(dst_ptr + b * stride_db + pid_h * stride_dh + t * stride_dt + offs, v, mask=d_mask)


@triton.jit
def _gather_heads_kernel(
    out_ptr, batch_idx_ptr, pos_idx_ptr, src_ptr,
    stride_sb, stride_sh, stride_st,
    stride_om, stride_oh,
    num_active, head_dim,
    BLOCK_D: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_h = tl.program_id(1)
    if pid_m >= num_active:
        return
    b = tl.load(batch_idx_ptr + pid_m)
    t = tl.load(pos_idx_ptr + pid_m)
    offs = tl.arange(0, BLOCK_D)
    d_mask = offs < head_dim
    v = tl.load(src_ptr + b * stride_sb + pid_h * stride_sh + t * stride_st + offs, mask=d_mask)
    tl.store(out_ptr + pid_m * stride_om + pid_h * stride_oh + offs, v, mask=d_mask)


def _heads_ok(t: torch.Tensor) -> bool:
    """The head-axis kernels index dims 0 and 2 and stream dim 3, so only dim 3 must be packed."""
    return t.dim() == 4 and t.stride(3) == 1


def scatter_heads_triton(dst, batch_idx, pos_idx, src) -> bool:
    """`dst[batch_idx, :, pos_idx] = src` for dst [B, H, T, Dh] and src [M, H, Dh].

    The head axis is a full slice, so each (batch, pos) pair touches H separate Dh-long runs rather
    than one contiguous row -- which is why `scatter_rows_triton` does not apply here. Regular
    enough to address directly: the grid is (M, H) and each program streams one head's Dh values.

    Same uniqueness assumption as `scatter_rows_triton`. Returns False when unsupported.
    """
    if not _heads_ok(dst) or src.dim() != 3 or src.stride(2) != 1:
        return False
    n, H, Dh = src.shape
    if n == 0:
        return True
    if dst.shape[1] != H or dst.shape[3] != Dh:
        return False
    _scatter_heads_kernel[(n, H)](
        dst, batch_idx, pos_idx, src,
        dst.stride(0), dst.stride(1), dst.stride(2),
        src.stride(0), src.stride(1),
        n, Dh, BLOCK_D=triton.next_power_of_2(Dh),
    )
    return True


def gather_heads_triton(src, batch_idx, pos_idx):
    """`src[batch_idx, :, pos_idx]` -> contiguous [M, H, Dh], for src [B, H, T, Dh].

    Mirror of `scatter_heads_triton`. Returns None when unsupported.
    """
    if not _heads_ok(src):
        return None
    n = batch_idx.numel()
    H, Dh = src.shape[1], src.shape[3]
    out = torch.empty((n, H, Dh), device=src.device, dtype=src.dtype)
    if n == 0:
        return out
    _gather_heads_kernel[(n, H)](
        out, batch_idx, pos_idx, src,
        src.stride(0), src.stride(1), src.stride(2),
        out.stride(0), out.stride(1),
        n, Dh, BLOCK_D=triton.next_power_of_2(Dh),
    )
    return out
