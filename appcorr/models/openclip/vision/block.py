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
               collect_cls_attn: bool = False, collect_attn_mean: bool = False):
        """Full block forward over all N tokens; caches the total block delta
        (`{tag}_blocks_out_sum` = attn contribution + MLP contribution) so `.correct()` can
        reconstruct stale positions exactly via `x_in + blocks_out_sum`, and caches raw K/V (via
        `self.attn.approx`) so `.correct()` can splice in fresh K/V for corrected positions."""
        x_attn, cache_feature = self.attn.approx(self.layer_norm1(x), cache_feature, tag,
                                                 collect_cls_attn=collect_cls_attn,
                                                 collect_attn_mean=collect_attn_mean)
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

        # Write this block's *corrected* increment back over the approximate one, so a later round
        # that replays this block for a different token group reproduces the corrected value here
        # instead of falling back to the stale approximate increment. This is rule 3 of
        # docs/memo/interleaved_correction_contract.md, and it is unconditional -- skipping it is
        # not a configuration, it is the bug.
        #
        # This fork carried the PRE-fix shape: it read `blocks_out_sum` and never wrote back. DINOv3
        # got the fix as `ac0238f`; CLIP and (inherited from CLIP) SAM 3 did not, and the contract
        # memo says so in as many words. One-shot correction cannot expose it -- there is no later
        # round to read the stale value -- which is why it survived here unnoticed.
        #
        # Symptom that found it: interleaved g=4 at keep=0.25 scored 78.44 top-1 against an
        # approx-only FLOOR of 79.22 (640-image subset), i.e. correcting made it worse than not
        # correcting. That is the documented signature: `prepare_tokens` re-embeds from the image as
        # decoded so far, so an earlier group ends up as `refined x + degraded increment` --
        # self-inconsistent, and measurably below the consistent floor.
        #
        # `x_attn_active - x_active` is the attention contribution; adding `mlp_out_new` gives
        # exactly the increment `approx()` above would have stored (see its two writes to the same
        # key), so the two paths stay interchangeable. Both terms are already materialised, so it
        # costs nothing, and it is a no-op for one-shot correction.
        blocks_out_sum = blocks_out_sum.clone()
        blocks_out_sum[:, token_idx] = (
            (x_attn_active - x_active) + mlp_out_new
        ).to(blocks_out_sum.dtype)
        cache_feature[f"{tag}_blocks_out_sum"] = blocks_out_sum

        return x_out, cache_feature
