"""
block.py

Forked `Qwen3_5MoeVisionBlock` exposing `.approx()`/`.correct()`. Plain pre-LN residual, same as
the Qwen2.5-VL fork this descends from:

    x = x + attn(norm1(x))
    x = x + mlp(norm2(x))

with one difference worth naming because it is easy to carry over wrongly: Qwen3.5's vision block
uses **LayerNorm**, not the RMSNorm that Qwen2.5-VL's vision block used. Both are taken from the
stock block rather than reconstructed, so this is a fact about the fork's provenance rather than
something the code has to branch on -- but a reader porting further changes between the two files
should not assume the norms match.

Staleness invariants are the same as every prior AppCorr fork: `x` is the full [T, dim] residual
stream, restarted from layer-0 tokens each correction round; non-queried positions are
reconstructed as `x + {tag}_blocks_out_sum` (dead values -- attention reads K/V from the cache,
never from the stream; norm/mlp touch only queried rows).
"""

from typing import Any, Dict

import torch
import torch.nn as nn

from .attention import ApproxCorrectQwen35VisionAttention


class ApproxCorrectQwen35VisionBlock(nn.Module):
    def __init__(self, norm1: nn.Module, attn: ApproxCorrectQwen35VisionAttention,
                 norm2: nn.Module, mlp: nn.Module):
        super().__init__()
        self.norm1 = norm1
        self.attn = attn
        self.norm2 = norm2
        self.mlp = mlp

    @classmethod
    def from_stock(cls, blk: nn.Module) -> "ApproxCorrectQwen35VisionBlock":
        return cls(
            norm1=blk.norm1,
            attn=ApproxCorrectQwen35VisionAttention.from_stock(blk.attn),
            norm2=blk.norm2,
            mlp=blk.mlp,
        )

    def forward(self, x: torch.Tensor, segment_ranges, position_embeddings) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), segment_ranges, position_embeddings)
        x = x + self.mlp(self.norm2(x))
        return x

    def approx(self, x: torch.Tensor, segment_ranges, position_embeddings,
               cache_feature: Dict[str, Any], tag: str, collect_attn_mean: bool = False):
        x_attn, cache_feature = self.attn.approx(
            self.norm1(x), segment_ranges, position_embeddings, cache_feature, tag,
            collect_attn_mean=collect_attn_mean,
        )
        cache_feature[f"{tag}_blocks_out_sum"] = x_attn.detach().clone()

        x_mid = x + x_attn
        mlp_out = self.mlp(self.norm2(x_mid))
        cache_feature[f"{tag}_blocks_out_sum"] = cache_feature[f"{tag}_blocks_out_sum"] + mlp_out.detach()

        return x_mid + mlp_out, cache_feature

    def correct(self, x: torch.Tensor, token_idx: torch.Tensor, segment_ranges,
                position_embeddings_sel, cache_feature: Dict[str, Any], tag: str):
        token_idx = token_idx.to(x.device)
        x_active = x[token_idx]  # [Q, dim]

        x_attn_sel, cache_feature = self.attn.correct(
            self.norm1(x_active), token_idx, segment_ranges, position_embeddings_sel, cache_feature, tag
        )
        x_attn_active = x_active + x_attn_sel
        mlp_out_new = self.mlp(self.norm2(x_attn_active))

        blocks_out_sum = cache_feature[f"{tag}_blocks_out_sum"]
        x_out = x + blocks_out_sum.to(dtype=x.dtype)      # out-of-place add -> already a fresh tensor
        x_out[token_idx] = (x_attn_active + mlp_out_new).to(dtype=x_out.dtype)

        # Persist the CORRECTED increment over the approximate one -- rule 3 of
        # docs/memo/interleaved_correction_contract.md, unconditional. Without it a later round
        # rebuilds this token from the stale approximate increment and silently discards what this
        # round fixed, so interleaved keeps only its final round's work. One-shot correction cannot
        # expose the omission, which is exactly why it survived unnoticed in the CLIP and SAM 3
        # forks until an interleaved arm scored BELOW its own approx-only floor.
        # `[token_idx]`, not `[:, token_idx]`: this residual stream carries no batch dimension.
        new_sum = blocks_out_sum.clone()
        new_sum[token_idx] = ((x_attn_active - x_active) + mlp_out_new).to(new_sum.dtype)
        cache_feature[f"{tag}_blocks_out_sum"] = new_sum

        return x_out, cache_feature
