"""
block.py

Forked `transformers.models.clip.modeling_clip.CLIPEncoderLayer` exposing `.approx()`/`.correct()`,
mirroring `appcorr/models/openvla/vision/block.py::ApproxCorrectBlock`. CLIP has no LayerScale at
all (unlike DINOv2), so the block math is the plain pre-LN residual form:
    x = x + attn(layer_norm1(x))
    x = x + mlp(layer_norm2(x))

Same invariants as the OpenVLA vision block fork (audited there against
`analysis/experiments/audit_progressive_semantics.py`, directly applicable here since it's the same
bidirectional-ViT accepted-approximation structure):
    - `x` is the full [B, N, C] residual stream, restarted from layer-0 tokens each correction round.
    - Non-queried positions are reconstructed as `x + {tag}_blocks_out_sum` (dead values -- attention
      reads K/V from cache, never from the stream; norm1/mlp touch only queried rows).
    - Multi-round correction is not bit-exact for this bidirectional tower (a round-1 token's cached
      K/V at layer >= 1 is stale w.r.t. later-round corrections and never revisited) -- single-round
      100% correction matches stock to bf16 noise; that is the only exactness guarantee needed here
      since this eval harness always does one correction round per request (all corrected patches
      arrive together, per the DINOv3-classifier-style GroupTrigger schedule).
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from .attention import ApproxCorrectCLIPAttention


class ApproxCorrectCLIPEncoderLayer(nn.Module):
    def __init__(self, layer_norm1: nn.Module, attn: ApproxCorrectCLIPAttention, layer_norm2: nn.Module,
                 mlp: nn.Module):
        super().__init__()
        self.layer_norm1 = layer_norm1
        self.attn = attn
        self.layer_norm2 = layer_norm2
        self.mlp = mlp

    @classmethod
    def from_stock(cls, layer: nn.Module) -> "ApproxCorrectCLIPEncoderLayer":
        return cls(
            layer_norm1=layer.layer_norm1,
            attn=ApproxCorrectCLIPAttention.from_stock(layer.self_attn),
            layer_norm2=layer.layer_norm2,
            mlp=layer.mlp,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Identical to stock `CLIPEncoderLayer.forward`."""
        x = x + self.attn(self.layer_norm1(x))
        x = x + self.mlp(self.layer_norm2(x))
        return x

    def approx(self, x: torch.Tensor, cache_feature: Dict[str, Any], tag: str,
               collect_cls_attn: bool = False):
        """Full block forward over all N tokens; caches the total block delta
        (`{tag}_blocks_out_sum` = attn contribution + MLP contribution) so `.correct()` can
        reconstruct stale positions exactly via `x_in + blocks_out_sum`, and caches raw K/V (via
        `self.attn.approx`) so `.correct()` can splice in fresh K/V for corrected positions."""
        x_attn, cache_feature = self.attn.approx(self.layer_norm1(x), cache_feature, tag,
                                                 collect_cls_attn=collect_cls_attn)
        cache_feature[f"{tag}_blocks_out_sum"] = x_attn.detach().clone()

        x_mid = x + x_attn
        mlp_out = self.mlp(self.layer_norm2(x_mid))
        cache_feature[f"{tag}_blocks_out_sum"] = cache_feature[f"{tag}_blocks_out_sum"] + mlp_out.detach()

        x_out = x_mid + mlp_out
        return x_out, cache_feature

    def correct(self, x: torch.Tensor, token_idx: torch.Tensor, cache_feature: Dict[str, Any], tag: str):
        """
        Args:
            x: [B, N, C] -- current residual stream (see module docstring for the staleness invariant).
            token_idx: [Q] -- absolute positions being corrected this round.
        Returns:
            x_out: [B, N, C] -- `token_idx` positions hold freshly recomputed values; all other
                positions are reconstructed to exactly match a full approx-only forward.
        """
        token_idx = token_idx.to(x.device)
        x_active = x[:, token_idx]  # [B, Q, C]
        x_norm_sel = self.layer_norm1(x_active)

        x_attn_sel, cache_feature = self.attn.correct(x_norm_sel, token_idx, cache_feature, tag)
        x_attn_active = x_active + x_attn_sel
        mlp_out_new = self.mlp(self.layer_norm2(x_attn_active))

        blocks_out_sum = cache_feature[f"{tag}_blocks_out_sum"]
        x_out = x + blocks_out_sum.to(dtype=x.dtype)
        x_out = x_out.clone()
        x_out[:, token_idx] = (x_attn_active + mlp_out_new).to(dtype=x_out.dtype)

        return x_out, cache_feature
