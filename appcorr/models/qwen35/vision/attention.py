"""
attention.py

Forked `transformers.models.qwen3_5_moe.modeling_qwen3_5_moe.Qwen3_5MoeVisionAttention` exposing
`.approx()`/`.correct()`, following the same partial_token pattern as every other AppCorr vision
fork. It is a direct descendant of `appcorr/models/qwen25vl/vision/attention.py` -- same fused QKV,
same 2D vision RoPE, same batch-free `[seq, dim]` layout, same `cu_seqlens` segment split -- with
one structural simplification and one addition.

**Gone: windowed attention.** Qwen2.5-VL alternated FULL per-image attention (4 of 32 layers, via
`fullatt_block_indexes`) with WINDOWED attention (the other 28, `window_size=112`), which forced the
tower to permute patches into merge-group-major window order and forced this file to carry two sets
of segment boundaries. Qwen3.5's vision config sets `window_size=None` and
`fullatt_block_indexes=None`: **every one of its 27 layers is full per-image attention**. So there
is one `cu_seqlens`, no window permutation, and `token_idx` indexes the natural raster sequence
rather than a permuted one. Anything in the 2.5 fork that mentions `window_index` has no analogue
here, and its absence is the point rather than an oversight.

**Added: received-attention collection.** `collect_attn_mean` accumulates, per token, the mean
attention it RECEIVES -- the column mean of the attention matrix, which is the term the standard
patch score multiplies residual energy by. On Qwen2.5-VL this was only defensible on the 4 full
attention layers (a windowed layer's columns are normalised over a 112-token window, so its column
means are not comparable across windows and mixing them into a layer average is meaningless). Here
that caveat disappears with the windowing: all 27 layers are full attention, so every layer's
columns are normalised over the same per-image domain and the layer mean is well defined.

The QK product is recomputed from `q`/`k` that this call has already materialised for SDPA -- no
extra projection, hence nothing for the FLOP hooks to double-count (contrast Gemma 3, which re-runs
`_qkv` and needs the `PSCORE` stage; see `appcorr/flops/counter.py`). It is a bare matmul rather
than `F.scaled_dot_product_attention` precisely because SDPA fuses the softmax away and never
materialises the weights.
"""

import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# L4a switch (see docs/memo/qwen_correct_forward_profile.md); False restores the pre-2026-09-08 form.
RECV_ATTN_FUSED_SUM = True
# "torch" = the chunked softmax reference (L4a fused reduce); "triton" = the two-pass column-sum
# kernel in recv_attn_triton.py (L4b); "auto" (default) = triton below RECV_ATTN_TRITON_MAX_T
# patch tokens, torch above. Selection identity of the kernel vs the reference gated PASS
# 192/192 RealWorldQA + 256/256 VisDrone Det (2026-09-08, default tile config); the tile sweep
# (2026-09-09, qwen_recv_attn_triton_test.py --sweep) found no config faster than the default and
# the kernel loses to torch past ~15k tokens (8436: 4.5 vs 7.4 ms; 19264: 21.1 vs 18.2), so the
# switch is by size. Read once at import so a gate run can pick the path from the environment.
RECV_ATTN_IMPL = os.environ.get("APPCORR_RECV_ATTN", "auto")
RECV_ATTN_TRITON_MAX_T = int(os.environ.get("APPCORR_RECV_ATTN_TRITON_MAX_T", "12000"))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_vision(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Matches stock `apply_rotary_pos_emb_vision` exactly (fp32 upcast during rotation)."""
    orig_q_dtype, orig_k_dtype = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


class ApproxCorrectQwen35VisionAttention(nn.Module):
    def __init__(self, qkv: nn.Linear, proj: nn.Linear, num_heads: int, head_dim: int, scaling: float):
        super().__init__()
        self.qkv = qkv
        self.proj = proj
        self.num_heads = num_heads
        self.head_dim = head_dim
        # Taken from the stock module rather than recomputed as `head_dim ** -0.5`: they agree
        # today, and reading it means they cannot silently stop agreeing.
        self.scaling = scaling

    @classmethod
    def from_stock(cls, attn: nn.Module) -> "ApproxCorrectQwen35VisionAttention":
        return cls(qkv=attn.qkv, proj=attn.proj, num_heads=attn.num_heads,
                   head_dim=attn.head_dim, scaling=attn.scaling)

    def _qkv_heads(self, x: torch.Tensor):
        """x: [T, dim] -> q,k,v each [T, num_heads, head_dim]."""
        seq_length = x.shape[0]
        qkv = self.qkv(x).reshape(seq_length, 3, self.num_heads, self.head_dim).permute(1, 0, 2, 3)
        return qkv.unbind(0)

    @staticmethod
    def segment_ranges_from_cu_seqlens(cu_seqlens: torch.Tensor):
        """`[(start, length), ...]` on the CPU side. Called once per request by the backbone, never
        per layer: `.tolist()` is a GPU->CPU sync and 27 of them per request is a real stall."""
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        starts = cu_seqlens[:-1].tolist()
        return list(zip(starts, lengths))

    def _sdpa_segment(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """q,k,v: [T_seg, H, Dh] -> [T_seg, H, Dh]."""
        q = q.transpose(0, 1).unsqueeze(0)  # [1, H, T, Dh]
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        out = F.scaled_dot_product_attention(q, k, v, scale=self.scaling)
        return out.squeeze(0).transpose(0, 1)  # [T, H, Dh]

    def _received_attention(self, q: torch.Tensor, k: torch.Tensor, segment_ranges) -> torch.Tensor:
        """Mean attention each token RECEIVES, over heads and queries. Returns [T] float32.

        Chunked over queries: a 4096-token image at 16 heads would materialise a
        [16, 4096, 4096] fp32 weight matrix -- 1 GB -- in one go. The column sum accumulates across
        chunks exactly, so chunking changes nothing but peak memory.

        Divided by the segment's own length, so each image's tokens are on the same scale no matter
        how many patches it has -- without that, a large image's tokens would score systematically
        lower purely because its attention mass is spread over more rows.
        """
        use_triton = RECV_ATTN_IMPL == "triton" or (
            RECV_ATTN_IMPL == "auto" and q.shape[0] <= RECV_ATTN_TRITON_MAX_T)
        if use_triton and q.is_cuda and q.dtype == torch.bfloat16:
            from .recv_attn_triton import available, received_attention
            if available():
                return received_attention(q, k, self.scaling, segment_ranges)
        T = q.shape[0]
        col = torch.zeros(T, device=q.device, dtype=torch.float32)
        chunk = 1024
        for start, length in segment_ranges:
            end = start + length
            k_seg = k[start:end].transpose(0, 1)              # [H, L, Dh]
            for s0 in range(start, end, chunk):
                e0 = min(s0 + chunk, end)
                q_c = q[s0:e0].transpose(0, 1)                # [H, C, Dh]
                w = torch.softmax((q_c @ k_seg.transpose(-1, -2)) * self.scaling, dim=-1)
                if RECV_ATTN_FUSED_SUM:
                    # Upcast inside the reduction instead of materialising an fp32 copy of the
                    # [H, C, L] weights first: the copy alone was 82 ms of the 288 ms this
                    # method costs per pass at 2109 image tokens (profile memo, L4a). Same
                    # bf16-rounded weights, fp32 accumulation; only the reduce kernel's
                    # summation order can differ from `.float().sum(dim=1)`.
                    col[start:end] += w.sum(dim=1, dtype=torch.float32).mean(dim=0)
                else:
                    col[start:end] += w.float().sum(dim=1).mean(dim=0)
            col[start:end] /= length
        return col

    def forward(self, x: torch.Tensor, segment_ranges, position_embeddings) -> torch.Tensor:
        """Identical to stock forward -- used for plain (non approx/correct) validation."""
        seq_length = x.shape[0]
        q, k, v = self._qkv_heads(x)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        outs = [self._sdpa_segment(q[s:s + l], k[s:s + l], v[s:s + l]) for s, l in segment_ranges]
        return self.proj(torch.cat(outs, dim=0).reshape(seq_length, -1))

    def approx(self, x: torch.Tensor, segment_ranges, position_embeddings,
               cache_feature: Dict[str, Any], tag: str, collect_attn_mean: bool = False):
        """Full attention over all T tokens, caching post-RoPE K/V as `{tag}_kv` [T, H, 2, Dh]."""
        seq_length = x.shape[0]
        q, k, v = self._qkv_heads(x)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        cache_feature[f"{tag}_kv"] = torch.stack([k, v], dim=2)  # [T, H, 2, Dh]
        if collect_attn_mean:
            cache_feature[f"{tag}_attn_mean"] = self._received_attention(q, k, segment_ranges)

        outs = [self._sdpa_segment(q[s:s + l], k[s:s + l], v[s:s + l]) for s, l in segment_ranges]
        out = self.proj(torch.cat(outs, dim=0).reshape(seq_length, -1))
        return out, cache_feature

    def correct(self, x_sel: torch.Tensor, token_idx: torch.Tensor, segment_ranges,
                position_embeddings_sel, cache_feature: Dict[str, Any], tag: str, plan=None,
                span=None):
        """
        Args:
            x_sel: [Q, dim] -- norm1(x) already sliced to the corrected rows.
            token_idx: [Q] long -- absolute row indices. Unlike the Qwen2.5-VL fork these index the
                NATURAL sequence: there is no window permutation to undo.
            position_embeddings_sel: (cos, sin) already gathered at `token_idx`.
            span: optional `(lo, hi)`: `token_idx == arange(lo, hi)` (a keep=1.0 streaming band),
                so the K/V scatter is a slice copy. Values identical to the index path.
            plan: optional, from `backbone.correct_plan()`: `"single"` when the request has one
                segment (every single-image request -- every query row attends to the whole
                cache, so there is nothing to split), else `(order, inv_order, segs)` in the
                Qwen2.5-VL fork's shape (query rows sorted by position and, per segment owning
                any, its slice [a, b) of the sorted rows). With a plan this call issues no
                GPU->CPU sync; without one every segment costs a `.any()` sync plus two boolean
                gathers -- 27 layers x 4 rounds = 108 syncs per request on the streaming axis,
                the largest single item of its launch/sync-bound profile (2026-09-08). Same SDPA
                on the same rows either way: outputs are bitwise identical.
        """
        kv = cache_feature[f"{tag}_kv"]  # [T, H, 2, Dh]
        q_new, k_new, v_new = self._qkv_heads(x_sel)
        cos, sin = position_embeddings_sel
        q_new, k_new = apply_rotary_pos_emb_vision(q_new, k_new, cos, sin)

        token_idx = token_idx.to(kv.device)
        if span is not None:
            # contiguous rows [lo, hi) == token_idx (caller's guarantee): strided copy, no index_put
            lo, hi = span
            kv[lo:hi, :, 0].copy_(k_new.to(dtype=kv.dtype))
            kv[lo:hi, :, 1].copy_(v_new.to(dtype=kv.dtype))
        else:
            kv[token_idx, :, 0] = k_new.to(dtype=kv.dtype)
            kv[token_idx, :, 1] = v_new.to(dtype=kv.dtype)
        cache_feature[f"{tag}_kv"] = kv
        k_full, v_full = kv.unbind(2)

        Q = token_idx.shape[0]
        if plan == "single":
            out = self._sdpa_segment(q_new, k_full, v_full)
        elif plan is not None:
            order, inv_order, segs = plan
            q_sorted = q_new[order]
            outs = [self._sdpa_segment(q_sorted[a:b], k_full[start:start + length],
                                       v_full[start:start + length])
                    for start, length, a, b in segs]
            out = (torch.cat(outs, dim=0) if len(outs) > 1 else outs[0])[inv_order]
        else:
            out = torch.zeros((Q, self.num_heads, self.head_dim), device=x_sel.device, dtype=q_new.dtype)
            for start, length in segment_ranges:
                seg_mask = (token_idx >= start) & (token_idx < start + length)
                if not bool(seg_mask.any()):
                    # An image contributing no corrected token this round is expected: groups are
                    # assigned per request, and a multi-image request corrects them at different
                    # rounds.
                    continue
                out[seg_mask] = self._sdpa_segment(
                    q_new[seg_mask], k_full[start:start + length], v_full[start:start + length]
                )

        return self.proj(out.reshape(Q, -1)), cache_feature
