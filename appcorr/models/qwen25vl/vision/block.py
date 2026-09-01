"""
block.py

Forked `Qwen2_5_VLVisionBlock` exposing `.approx()`/`.correct()`, mirroring
`appcorr/models/openclip/vision/block.py` (no LayerScale, plain pre-LN residual:
`x = x + attn(norm1(x)); x = x + mlp(norm2(x))`) with RMSNorm instead of LayerNorm and the
segment-split attention call signature (cu_seqlens/position_embeddings) threaded through.

Same staleness invariants as every prior AppCorr fork this session: `x` is the full [T, dim]
residual stream; non-queried positions are reconstructed as `x + {tag}_blocks_out_sum` (dead
values -- attention reads K/V from cache, never from the stream; norm/mlp only touch queried rows).
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from .attention import ApproxCorrectQwen25VLVisionAttention


class ApproxCorrectQwen25VLVisionBlock(nn.Module):
    def __init__(self, norm1: nn.Module, attn: ApproxCorrectQwen25VLVisionAttention, norm2: nn.Module, mlp: nn.Module):
        super().__init__()
        self.norm1 = norm1
        self.attn = attn
        self.norm2 = norm2
        self.mlp = mlp

    @classmethod
    def from_stock(cls, blk: nn.Module) -> "ApproxCorrectQwen25VLVisionBlock":
        return cls(
            norm1=blk.norm1,
            attn=ApproxCorrectQwen25VLVisionAttention.from_stock(blk.attn),
            norm2=blk.norm2,
            mlp=blk.mlp,
        )

    def forward(self, x: torch.Tensor, segment_ranges, position_embeddings) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), segment_ranges, position_embeddings)
        x = x + self.mlp(self.norm2(x))
        return x

    def approx(self, x: torch.Tensor, segment_ranges, position_embeddings, cache_feature: Dict[str, Any], tag: str):
        x_attn, cache_feature = self.attn.approx(self.norm1(x), segment_ranges, position_embeddings, cache_feature, tag)
        cache_feature[f"{tag}_blocks_out_sum"] = x_attn.detach().clone()

        x_mid = x + x_attn
        mlp_out = self.mlp(self.norm2(x_mid))
        cache_feature[f"{tag}_blocks_out_sum"] = cache_feature[f"{tag}_blocks_out_sum"] + mlp_out.detach()

        x_out = x_mid + mlp_out
        return x_out, cache_feature

    def correct(
        self,
        x: torch.Tensor,
        token_idx: torch.Tensor,
        segment_ranges,
        position_embeddings_sel,
        cache_feature: Dict[str, Any],
        tag: str,
    ):
        token_idx = token_idx.to(x.device)
        x_active = x[token_idx]  # [Q, dim]
        x_norm_sel = self.norm1(x_active)

        x_attn_sel, cache_feature = self.attn.correct(
            x_norm_sel, token_idx, segment_ranges, position_embeddings_sel, cache_feature, tag
        )
        x_attn_active = x_active + x_attn_sel
        mlp_out_new = self.mlp(self.norm2(x_attn_active))

        blocks_out_sum = cache_feature[f"{tag}_blocks_out_sum"]
        # `x + blocks_out_sum` (out-of-place add) already allocates a fresh tensor -- the extra
        # `.clone()` previously here was a redundant copy of that fresh allocation.
        x_out = x + blocks_out_sum.to(dtype=x.dtype)
        x_out[token_idx] = (x_attn_active + mlp_out_new).to(dtype=x_out.dtype)

        return x_out, cache_feature
