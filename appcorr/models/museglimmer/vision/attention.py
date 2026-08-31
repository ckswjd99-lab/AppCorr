"""
attention.py

Forked `transformers.models.muse_glimmer.modeling_muse_glimmer.MuseGlimmerVisionAttention`
exposing `.approx()`/`.correct()`, the same partial-token pattern as
`appcorr/models/qwen25vl/vision/attention.py` (which this file is adapted from -- Muse Glimmer's
vision attention is the same segment-split, 2D-RoPE, no-CLS shape). Deltas from the 2.5 fork:

  - **Separate q/k/v projections** (`q_proj`/`k_proj`/`v_proj`, bias=True) instead of one fused
    `qkv` linear; output projection is `proj` as in 2.5.
  - **Per-layer full-vs-window dispatch comes from `config.layer_types`** ("full_attention" /
    "window_attention"), not `fullatt_block_indexes` -- handled in backbone.py; this file only ever
    sees the segment ranges for its own layer.
  - **RoPE**: `apply_rotary_pos_emb_vision` is byte-identical to stock MG's (fp32 upcast, cos/sin
    `unsqueeze(-2)`), which is itself identical to Qwen2.5-VL's. cos/sin come precomputed from the
    backbone's `MuseGlimmerVisionRotaryEmbedding` call (a pure function of grid_thw).

Everything about K/V caching, per-segment SDPA, query-chunked incoming attention, and the
correction update (`kv[token_idx] = new`) carries over unchanged from the 2.5 fork.
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
    """Matches stock MG `apply_rotary_pos_emb_vision` exactly (fp32 upcast during rotation)."""
    orig_q_dtype, orig_k_dtype = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed.to(orig_q_dtype), k_embed.to(orig_k_dtype)


class ApproxCorrectMuseGlimmerVisionAttention(nn.Module):
    def __init__(self, q_proj: nn.Linear, k_proj: nn.Linear, v_proj: nn.Linear, proj: nn.Linear,
                 num_heads: int, head_dim: int):
        super().__init__()
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.proj = proj
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scaling = head_dim ** -0.5

    @classmethod
    def from_stock(cls, attn: nn.Module) -> "ApproxCorrectMuseGlimmerVisionAttention":
        return cls(q_proj=attn.q_proj, k_proj=attn.k_proj, v_proj=attn.v_proj, proj=attn.proj,
                   num_heads=attn.num_heads, head_dim=attn.head_dim)

    def _qkv_heads(self, x: torch.Tensor):
        """x: [T, dim] -> q,k,v each [T, num_heads, head_dim]."""
        T = x.shape[0]
        q = self.q_proj(x).reshape(T, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(T, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(T, self.num_heads, self.head_dim)
        return q, k, v

    @staticmethod
    def _segment_ranges(cu_seqlens: torch.Tensor):
        """One-time `[(start, length), ...]` decode (the only GPU->CPU sync), cached in ctx by
        backbone.prepare_full_tokens -- same rationale as the 2.5 fork."""
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        starts = cu_seqlens[:-1].tolist()
        return list(zip(starts, lengths))

    def _sdpa_segment(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """q,k,v: [T_seg, H, Dh] -> [T_seg, H, Dh]."""
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        out = F.scaled_dot_product_attention(q, k, v)
        return out.squeeze(0).transpose(0, 1)

    def forward(self, x: torch.Tensor, segment_ranges, position_embeddings) -> torch.Tensor:
        """Stock-equivalent forward for plain validation."""
        T = x.shape[0]
        q, k, v = self._qkv_heads(x)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        outs = []
        for start, length in segment_ranges:
            sl = slice(start, start + length)
            outs.append(self._sdpa_segment(q[sl], k[sl], v[sl]))
        return self.proj(torch.cat(outs, dim=0).reshape(T, -1))

    @torch.no_grad()
    def incoming_attention(self, x: torch.Tensor, segment_ranges, position_embeddings) -> torch.Tensor:
        """Received-attention column mass, head- and query-averaged, per segment. Query-chunked
        for the same OOM reason as the 2.5 fork (a full-attention segment on a large image would
        materialize [H, L, L] fp32). [T], permuted-sequence order."""
        q, k, _ = self._qkv_heads(x)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        q = q.transpose(0, 1)
        k = k.transpose(0, 1)
        col = torch.zeros(x.shape[0], device=x.device, dtype=torch.float32)
        chunk = 1024
        for start, length in segment_ranges:
            sl = slice(start, start + length)
            q_seg, k_seg = q[:, sl].float(), k[:, sl].float()
            acc = torch.zeros(length, device=x.device, dtype=torch.float32)
            for q0 in range(0, length, chunk):
                q1 = min(q0 + chunk, length)
                w = torch.softmax((q_seg[:, q0:q1] @ k_seg.transpose(-1, -2)) * self.scaling, dim=-1)
                acc += w.sum(dim=1).mean(dim=0)
            col[sl] = acc / length
        return col

    def approx(self, x: torch.Tensor, segment_ranges, position_embeddings, cache_feature: Dict[str, Any], tag: str):
        """Full per-segment attention over all T tokens, caching post-RoPE K/V under
        `{tag}_kv`, shape [T, H, 2, Dh]."""
        T = x.shape[0]
        q, k, v = self._qkv_heads(x)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        cache_feature[f"{tag}_kv"] = torch.stack([k, v], dim=2)
        outs = []
        for start, length in segment_ranges:
            sl = slice(start, start + length)
            outs.append(self._sdpa_segment(q[sl], k[sl], v[sl]))
        out = self.proj(torch.cat(outs, dim=0).reshape(T, -1))
        return out, cache_feature

    def correct(self, x_sel: torch.Tensor, token_idx: torch.Tensor, segment_ranges,
                position_embeddings_sel, cache_feature: Dict[str, Any], tag: str):
        """x_sel: [Q, dim] norm1-ed query rows; token_idx: [Q] absolute permuted-sequence rows.
        Updates cached K/V at the query rows, then attends each query against its own segment's
        full cached K/V -- identical contract to the 2.5 fork."""
        kv = cache_feature[f"{tag}_kv"]
        q_new, k_new, v_new = self._qkv_heads(x_sel)
        cos, sin = position_embeddings_sel
        q_new, k_new = apply_rotary_pos_emb_vision(q_new, k_new, cos, sin)

        token_idx = token_idx.to(kv.device)
        kv[token_idx, :, 0] = k_new.to(dtype=kv.dtype)
        kv[token_idx, :, 1] = v_new.to(dtype=kv.dtype)
        cache_feature[f"{tag}_kv"] = kv
        k_full, v_full = kv.unbind(2)

        Q = token_idx.shape[0]
        out = torch.zeros((Q, self.num_heads, self.head_dim), device=x_sel.device, dtype=q_new.dtype)
        for start, length in segment_ranges:
            seg_mask = (token_idx >= start) & (token_idx < start + length)
            if not bool(seg_mask.any()):
                continue
            out[seg_mask] = self._sdpa_segment(q_new[seg_mask],
                                               k_full[start:start + length],
                                               v_full[start:start + length])
        return self.proj(out.reshape(Q, -1)), cache_feature
