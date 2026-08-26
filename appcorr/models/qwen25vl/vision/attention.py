"""
attention.py

Forked `transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.Qwen2_5_VLVisionAttention` exposing
`.approx()`/`.correct()`, following the same partial_token pattern as every other AppCorr vision
fork this session (DINOv3/OpenVLA/CLIP), adapted for Qwen2.5-VL's specifics:

    - **Fused QKV** (single `nn.Linear(dim, dim*3)`), not separate q/k/v projections like CLIP.
    - **2D (spatial) RoPE**: `position_embeddings = (cos, sin)` is precomputed ONCE per request
      (depends only on each image's `grid_thw`, not pixel content -- unlike DINOv2/SigLIP/CLIP which
      have no RoPE at all, and unlike DINOv3 which recomputes nothing extra either since its RoPE is
      also grid-shape-only) and passed in unchanged to every layer's approx()/correct() call. No
      "hoisting" needed here in the sense OpenVLA's LLM fork needed it -- this is already invariant
      for the whole request, not just one correction round.
    - **No batch dimension in the raw representation**: hidden_states is `[seq_length, dim]` (one
      image's full patch count, no padding -- batch_size=1 is enforced at the executor level so
      there's never more than one image's patches in a single call).
    - **Segment-split attention (cu_seqlens)**: unlike DINOv3/OpenVLA/CLIP's single global
      bidirectional attention domain, Qwen2.5-VL alternates between FULL per-image attention (4 of
      32 layers, `fullatt_block_indexes`) and WINDOWED attention (the other 28, `window_size=112`).
      Either way, attention is computed independently per contiguous segment demarcated by
      `cu_seqlens` (full-attention layers) or `cu_window_seqlens` (windowed layers) -- both are pure
      functions of `grid_thw` (+ config constants), reused directly from
      `transformers.vision_utils` (no fork needed, they have no learnable parameters and are already
      exact/cheap). `.correct()`'s query set can straddle multiple segments (if the corrected
      merge-groups landed in different attention windows after the window-index permutation), so it
      groups `token_idx` by which `cu_seqlens_now` segment each falls into and runs SDPA once per
      segment against that segment's full (cached) K/V, mirroring stock's own per-segment split.
"""

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class ApproxCorrectQwen25VLVisionAttention(nn.Module):
    def __init__(self, qkv: nn.Linear, proj: nn.Linear, num_heads: int, head_dim: int):
        super().__init__()
        self.qkv = qkv
        self.proj = proj
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scaling = head_dim ** -0.5

    @classmethod
    def from_stock(cls, attn: nn.Module) -> "ApproxCorrectQwen25VLVisionAttention":
        return cls(qkv=attn.qkv, proj=attn.proj, num_heads=attn.num_heads, head_dim=attn.head_dim)

    def _qkv_heads(self, x: torch.Tensor):
        """x: [T, dim] -> q,k,v each [T, num_heads, head_dim]."""
        seq_length = x.shape[0]
        qkv = self.qkv(x).reshape(seq_length, 3, self.num_heads, self.head_dim).permute(1, 0, 2, 3)
        return qkv.unbind(0)

    @staticmethod
    def _segment_ranges(cu_seqlens: torch.Tensor):
        """Kept as a standalone helper (used once by `backbone.py.prepare_full_tokens` to build the
        segment-range lists cached in `ctx`) -- no longer called per-layer/per-call here, since that
        required a GPU->CPU sync (`.tolist()`) on every one of a request's ~32-64 layer calls."""
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        starts = cu_seqlens[:-1].tolist()
        return list(zip(starts, lengths))

    def _sdpa_segment(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """q,k,v: [T_seg, num_heads, head_dim] -> [T_seg, num_heads, head_dim]."""
        q = q.transpose(0, 1).unsqueeze(0)  # [1, H, T, Dh]
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        out = F.scaled_dot_product_attention(q, k, v)
        return out.squeeze(0).transpose(0, 1)  # [T, H, Dh]

    def forward(self, x: torch.Tensor, segment_ranges, position_embeddings) -> torch.Tensor:
        """Identical to stock forward -- used for plain (non approx/correct) validation.
        `segment_ranges`: precomputed `[(start, length), ...]` list (see backbone.py's
        `prepare_full_tokens`), replacing what used to be a `cu_seqlens` tensor re-decoded via
        `.tolist()` (a GPU->CPU sync) on every call."""
        seq_length = x.shape[0]
        q, k, v = self._qkv_heads(x)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        outs = []
        for start, length in segment_ranges:
            sl = slice(start, start + length)
            outs.append(self._sdpa_segment(q[sl], k[sl], v[sl]))
        out = torch.cat(outs, dim=0).reshape(seq_length, -1)
        return self.proj(out)

    @torch.no_grad()
    def incoming_attention(self, x: torch.Tensor, segment_ranges, position_embeddings) -> torch.Tensor:
        """Column mass of this layer's attention -- how much the rest of ITS OWN SEGMENT reads FROM
        each token -- head- and query-averaged, matching Gemma3's `_incoming_attention` (no CLS
        token there either; this project's standing server_pscore is residual-energy x received-
        attention, never CLS-based). Computed PER SEGMENT (`segment_ranges`) since Qwen's attention
        domains are segment-local (per-image or per-window; a token never attends outside its own
        segment), so pooling has to respect that boundary rather than averaging across the whole
        permuted sequence at once. [T] -- one value per token in `x`, permuted-sequence order.

        Redundant QK product against the one `approx()` already computes -- same one-time cost
        Gemma3's docstring accepts ("one extra QK product and no extra weights"), not threaded out
        of the SDPA call since that call doesn't materialize `attn_prob` at all.
        """
        q, k, _ = self._qkv_heads(x)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        q = q.transpose(0, 1)  # [H, T, Dh]
        k = k.transpose(0, 1)
        col = torch.zeros(x.shape[0], device=x.device, dtype=torch.float32)
        for start, length in segment_ranges:
            sl = slice(start, start + length)
            q_seg, k_seg = q[:, sl], k[:, sl]  # [H, L, Dh]
            w = torch.softmax((q_seg.float() @ k_seg.float().transpose(-1, -2)) * self.scaling, dim=-1)  # [H, L, L]
            col[sl] = w.sum(dim=1).mean(dim=0) / length  # mean over queries (within segment), then heads
        return col

    def approx(self, x: torch.Tensor, segment_ranges, position_embeddings, cache_feature: Dict[str, Any], tag: str):
        """Full attention over all T tokens (per-segment, per stock's own dispatch), caching raw K/V
        (post-RoPE) as `cache_feature[f"{tag}_kv"]`, shape [T, num_heads, 2, head_dim]."""
        seq_length = x.shape[0]
        q, k, v = self._qkv_heads(x)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)

        cache_feature[f"{tag}_kv"] = torch.stack([k, v], dim=2)  # [T, H, 2, Dh]

        outs = []
        for start, length in segment_ranges:
            sl = slice(start, start + length)
            outs.append(self._sdpa_segment(q[sl], k[sl], v[sl]))
        out = torch.cat(outs, dim=0).reshape(seq_length, -1)
        out = self.proj(out)
        return out, cache_feature

    def correct(
        self,
        x_sel: torch.Tensor,
        token_idx: torch.Tensor,
        segment_ranges,
        position_embeddings_sel,
        cache_feature: Dict[str, Any],
        tag: str,
    ):
        """
        Args:
            x_sel: [Q, dim] -- norm1(x) already sliced to the query/corrected rows (Q rows, absolute
                positions given by `token_idx`, in the WINDOW-PERMUTED sequence's row order).
            token_idx: [Q] long -- absolute row indices into the permuted sequence.
            segment_ranges: precomputed `[(start, length), ...]` list (full-image or window,
                whichever this layer uses) for the CURRENT permuted sequence -- used to figure out
                which cached-K/V segment each query row's attention should be computed against.
            position_embeddings_sel: (cos_sel, sin_sel) already gathered at `token_idx`.
        Returns:
            attn_out: [Q, dim] -- attention output for the query positions only (post proj).
        """
        kv = cache_feature[f"{tag}_kv"]  # [T, H, 2, Dh]
        q_new, k_new, v_new = self._qkv_heads(x_sel)  # each [Q, H, Dh]
        cos, sin = position_embeddings_sel
        q_new, k_new = apply_rotary_pos_emb_vision(q_new, k_new, cos, sin)

        token_idx = token_idx.to(kv.device)
        kv[token_idx, :, 0] = k_new.to(dtype=kv.dtype)
        kv[token_idx, :, 1] = v_new.to(dtype=kv.dtype)
        cache_feature[f"{tag}_kv"] = kv
        k_full, v_full = kv.unbind(2)  # each [T, H, Dh]

        Q = token_idx.shape[0]
        out = torch.zeros((Q, self.num_heads, self.head_dim), device=x_sel.device, dtype=q_new.dtype)
        for start, length in segment_ranges:
            seg_mask = (token_idx >= start) & (token_idx < start + length)
            if not bool(seg_mask.any()):
                continue
            q_seg = q_new[seg_mask]
            k_seg = k_full[start : start + length]
            v_seg = v_full[start : start + length]
            out[seg_mask] = self._sdpa_segment(q_seg, k_seg, v_seg)

        out = out.reshape(Q, -1)
        out = self.proj(out)
        return out, cache_feature
