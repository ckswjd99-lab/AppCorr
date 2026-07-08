"""
attention.py

Forked `transformers.models.clip.modeling_clip.CLIPAttention` (CLIP-ViT-bigG/14 vision tower)
exposing `.approx()` / `.correct()` entry points, following the same partial_token pattern used
by `appcorr/models/openvla/vision/attention.py` (which itself mirrors
`appcorr/models/dinov3/layers/attention.py::SelfAttention.approx_partial_token/correct_partial_token`).

Like OpenVLA's DINOv2/SigLIP towers, CLIP's vision tower uses plain learned absolute position
embeddings baked into the input before block 0 (see `CLIPVisionEmbeddings.forward`), not RoPE, so a
corrected token's fresh K/V can be spliced into the cache verbatim -- no rotation bookkeeping.

Differences from the OpenVLA template this was forked from:
    - CLIP has separate `q_proj`/`k_proj`/`v_proj` (not a fused `qkv` linear) and no q_norm/k_norm.
    - CLIP has exactly 1 prefix token (CLS only, no register tokens).
    - `self.scale = head_dim**-0.5`, identical to SDPA's default scale, so no explicit scale is
      threaded through `F.scaled_dot_product_attention` (matches stock numerically).

This eval harness forces batch_size=1 per request (same convention as
`analysis/experiments/dinov3_classifier_offload_eval.py`), so `token_idx` below is a single index
set shared by the one batch item -- no per-batch-item variable-length query packing is needed
(unlike DINOv3's classifier executor, which supports batch_size>1 with per-image masks).
"""

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ApproxCorrectCLIPAttention(nn.Module):
    """Drop-in replacement for `transformers.models.clip.modeling_clip.CLIPAttention`.

    Constructed via `from_stock()`, reusing the stock module's submodules by reference (no weight
    copy), so a freshly-built instance is numerically identical to the stock module for `.forward()`.
    """

    def __init__(self, q_proj: nn.Linear, k_proj: nn.Linear, v_proj: nn.Linear, out_proj: nn.Linear,
                 num_heads: int, head_dim: int):
        super().__init__()
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.out_proj = out_proj
        self.num_heads = num_heads
        self.head_dim = head_dim

    @classmethod
    def from_stock(cls, attn: nn.Module) -> "ApproxCorrectCLIPAttention":
        return cls(
            q_proj=attn.q_proj,
            k_proj=attn.k_proj,
            v_proj=attn.v_proj,
            out_proj=attn.out_proj,
            num_heads=attn.num_heads,
            head_dim=attn.head_dim,
        )

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)  # [B, H, T, Dh]

    def _qkv_heads(self, x: torch.Tensor):
        q = self._heads(self.q_proj(x))
        k = self._heads(self.k_proj(x))
        v = self._heads(self.v_proj(x))
        return q, k, v  # each [B, H, T, Dh]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Identical to stock `CLIPAttention.forward` -- used for plain (non approx/correct) validation."""
        B, N, C = x.shape
        q, k, v = self._qkv_heads(x)
        x_out = F.scaled_dot_product_attention(q, k, v)
        x_out = x_out.transpose(1, 2).reshape(B, N, C)
        return self.out_proj(x_out)

    def approx(self, x: torch.Tensor, cache_feature: Dict[str, Any], tag: str,
               collect_cls_attn: bool = False):
        """Full self-attention over all N tokens; caches raw K/V [B, H, N, 2, Dh] for later
        `.correct()` calls to patch. Numerically identical output to `.forward()`.

        If `collect_cls_attn`, also caches the CLS-token attention distribution (head-averaged,
        `{tag}_cls_attn` [B, N]) -- the per-layer ingredient of `cls_attn_prob_layermean`."""
        B, N, C = x.shape
        q, k, v = self._qkv_heads(x)

        cache_feature[f"{tag}_kv"] = torch.stack([k, v], dim=3)  # [B, H, N, 2, Dh]

        if collect_cls_attn:
            attn_cls = torch.softmax(
                (q[:, :, 0:1] @ k.transpose(-2, -1)) * (self.head_dim ** -0.5), dim=-1
            )  # [B, H, 1, N]
            cache_feature[f"{tag}_cls_attn"] = attn_cls.mean(dim=1).squeeze(1).detach()  # [B, N]

        x_out = F.scaled_dot_product_attention(q, k, v)
        x_out = x_out.transpose(1, 2).reshape(B, N, C)
        x_out = self.out_proj(x_out)
        return x_out, cache_feature

    def correct(
        self,
        x_sel: torch.Tensor,
        token_idx: torch.Tensor,
        cache_feature: Dict[str, Any],
        tag: str,
    ):
        """
        Args:
            x_sel: [B, Q, C] -- layer_norm1(x) already sliced to the query/corrected positions.
            token_idx: [Q] -- absolute token positions (0-indexed into the full N-length sequence).
        Returns:
            attn_out: [B, Q, C] -- attention output for the query positions only (post out_proj).
        """
        kv = cache_feature[f"{tag}_kv"]  # [B, H, N, 2, Dh]

        q_new, k_new, v_new = self._qkv_heads(x_sel)  # each [B, H, Q, Dh]
        token_idx = token_idx.to(kv.device)
        kv[:, :, token_idx, 0] = k_new.to(dtype=kv.dtype)
        kv[:, :, token_idx, 1] = v_new.to(dtype=kv.dtype)
        cache_feature[f"{tag}_kv"] = kv

        k_full, v_full = kv.unbind(3)  # each [B, H, N, Dh]
        attn_out = F.scaled_dot_product_attention(q_new, k_full, v_full)  # [B, H, Q, Dh]
        attn_out = attn_out.transpose(1, 2).reshape(x_sel.shape[0], x_sel.shape[1], self.num_heads * self.head_dim)
        attn_out = self.out_proj(attn_out)
        return attn_out, cache_feature
