"""
attention.py

Forked timm `Attention` (see timm/models/vision_transformer.py) exposing `.approx()` / `.correct()`
entry points, following the same partial_token pattern as
`appcorr/models/dinov3/layers/attention.py::SelfAttention.approx_partial_token/correct_partial_token`.

Unlike DINOv3, OpenVLA's DINOv2/SigLIP towers use plain learned absolute position embeddings baked
into the input *before* block 0 runs (see timm's `VisionTransformer._pos_embed`) rather than RoPE, so
there is no rope argument to thread through attention -- a corrected token's fresh K/V can be spliced
into the cache verbatim, no rotation bookkeeping needed.

Shared by both towers (DINOv2 has CLS+register prefix tokens with real LayerScale; SigLIP has neither
prefix tokens nor LayerScale -- `ls1`/`ls2` are `nn.Identity()` for SigLIP) since the block-level control
flow (`x + ls(attn(norm(x)))`) is identical either way; only weights/dims/presence-of-LayerScale differ,
which is a construction-time detail, not a control-flow one.

Design (mirrors dinov3's correct_partial_token calling convention exactly): for the vision towers, the
"corrected" token set and the "query" (output-recompute) set are the *same* positions -- a patch either
just received higher-res data (gets corrected) or it didn't (stays approximate). This is unlike Phase 2's
causal LLM correction, where the query set is a strict superset of the corrected set (it also always
includes the permanent text-suffix group) -- that asymmetry is handled at the block level in
`llm/llama_prefill_layer.py`, not here.
"""

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ApproxCorrectAttention(nn.Module):
    """Drop-in replacement for `timm.models.vision_transformer.Attention`.

    Constructed via `from_stock()`, which reuses the stock module's submodules by reference (no
    weight copy needed), so a freshly-built instance is numerically identical to the stock module
    for `.forward()`.
    """

    def __init__(self, qkv: nn.Linear, q_norm: nn.Module, k_norm: nn.Module, proj: nn.Linear,
                 proj_drop: nn.Module, num_heads: int, head_dim: int):
        super().__init__()
        self.qkv = qkv
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.proj = proj
        self.proj_drop = proj_drop
        self.num_heads = num_heads
        self.head_dim = head_dim

    @classmethod
    def from_stock(cls, attn: nn.Module) -> "ApproxCorrectAttention":
        return cls(
            qkv=attn.qkv,
            q_norm=attn.q_norm,
            k_norm=attn.k_norm,
            proj=attn.proj,
            proj_drop=attn.proj_drop,
            num_heads=attn.num_heads,
            head_dim=attn.head_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Identical to stock `Attention.forward` -- used for plain (non approx/correct) validation."""
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        x = F.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def _qkv_heads(self, x: torch.Tensor):
        B, T, _ = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)  # each [B, H, T, Dh]
        q, k = self.q_norm(q), self.k_norm(k)
        return q, k, v

    def approx(self, x: torch.Tensor, cache_feature: Dict[str, Any], tag: str,
               collect_cls_attn: bool = False):
        """Full self-attention over all N tokens; caches raw K/V [B, H, N, 2, Dh] (post q_norm/k_norm)
        for later `.correct()` calls to patch. Numerically identical output to `.forward()`.

        If `collect_cls_attn`, also caches the CLS-token attention distribution (head-averaged,
        `{tag}_cls_attn` [B, N]) -- the per-layer ingredient of AppCorr's `cls_attn_prob_layermean`
        server pscore (see offload/common/protocol.py), used for importance-ranked patch selection.
        Costs one extra [1, N] score row + softmax per layer (negligible vs the full SDPA)."""
        B, N, C = x.shape
        q, k, v = self._qkv_heads(x)

        cache_feature[f"{tag}_kv"] = torch.stack([k, v], dim=3)  # [B, H, N, 2, Dh]

        if collect_cls_attn:
            # CLS is token 0 (timm _pos_embed concatenates [cls, reg..., patches]). Match SDPA's
            # default softmax scale of head_dim**-0.5.
            attn_cls = torch.softmax(
                (q[:, :, 0:1] @ k.transpose(-2, -1)) * (self.head_dim ** -0.5), dim=-1
            )  # [B, H, 1, N]
            cache_feature[f"{tag}_cls_attn"] = attn_cls.mean(dim=1).squeeze(1).detach()  # [B, N]

        x_out = F.scaled_dot_product_attention(q, k, v)
        x_out = x_out.transpose(1, 2).reshape(B, N, C)
        x_out = self.proj(x_out)
        x_out = self.proj_drop(x_out)
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
            x_sel: [B, Q, C] -- norm1(x) already sliced to the query/corrected positions (Q tokens,
                same absolute positions for every batch item -- see `token_idx`).
            token_idx: [Q] -- absolute token positions (0-indexed into the full N-length sequence)
                that `x_sel` corresponds to. Used to (a) splice fresh K/V into the cache at exactly
                these positions and (b) run SDPA for exactly this query set.
        Returns:
            attn_out: [B, Q, C] -- attention output for the query positions only (post proj/proj_drop).
        """
        kv = cache_feature[f"{tag}_kv"]  # [B, H, N, 2, Dh]
        B = kv.shape[0]

        q_new, k_new, v_new = self._qkv_heads(x_sel)  # each [B, H, Q, Dh]
        token_idx = token_idx.to(kv.device)
        kv[:, :, token_idx, 0] = k_new.to(dtype=kv.dtype)
        kv[:, :, token_idx, 1] = v_new.to(dtype=kv.dtype)
        cache_feature[f"{tag}_kv"] = kv

        k_full, v_full = kv.unbind(3)  # each [B, H, N, Dh]
        attn_out = F.scaled_dot_product_attention(q_new, k_full, v_full)  # [B, H, Q, Dh]
        attn_out = attn_out.transpose(1, 2).reshape(x_sel.shape[0], x_sel.shape[1], self.num_heads * self.head_dim)
        attn_out = self.proj(attn_out)
        attn_out = self.proj_drop(attn_out)
        return attn_out, cache_feature
