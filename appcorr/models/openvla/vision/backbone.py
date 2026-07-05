"""
backbone.py

Wraps a stock (pretrained) timm `VisionTransformer` featurizer -- either OpenVLA's DINOv2
(`vit_large_patch14_reg4_dinov2.lvd142m`, 5 prefix tokens: 1 CLS + 4 register) or SigLIP
(`vit_so400m_patch14_siglip_224`, 0 prefix tokens) -- with approx/correct forward passes built
from `ApproxCorrectBlock`.

Confirmed by Phase 0's oracle dump (`analysis/experiments/openvla_oracle_dump.py`): OpenVLA's
monkey-patched `forward = get_intermediate_layers(n={len(blocks)-2})` runs *every* block (24/24 for
DINOv2, 27/27 for SigLIP) and simply discards the last block's output, using the second-to-last --
it does not stop early. `extract_block_idx` below reproduces that exactly.

patch_embed/_pos_embed/patch_drop/norm_pre are reused directly from the stock featurizer (unchanged,
not forked) -- they have no "approx vs. correct" distinction since tokenization is always exact, and
re-running them for a correction round is cheap (a single strided conv + elementwise add) relative to
the 24-27 transformer blocks that follow.
"""

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from .block import ApproxCorrectBlock


class ApproxCorrectViTBackbone(nn.Module):
    def __init__(self, featurizer: nn.Module):
        super().__init__()
        self.featurizer = featurizer
        self.blocks = nn.ModuleList([ApproxCorrectBlock.from_stock(b) for b in featurizer.blocks])
        self.num_prefix_tokens = featurizer.num_prefix_tokens
        # Matches the stock monkey-patch: `get_intermediate_layers(n={len(blocks) - 2})`.
        self.extract_block_idx = len(featurizer.blocks) - 2

    def prepare_full_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """patch_embed -> _pos_embed -> patch_drop -> norm_pre, exactly matching timm's
        `VisionTransformer._intermediate_layers` prelude. Always exact (no approximation) --
        tokenization has no notion of "corrected vs. stale"."""
        x = self.featurizer.patch_embed(pixel_values)
        x = self.featurizer._pos_embed(x)
        x = self.featurizer.patch_drop(x)
        x = self.featurizer.norm_pre(x)
        return x

    def approx_forward(
        self, pixel_values: torch.Tensor, cache_feature: Dict[str, Any], tag_prefix: str
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """First pass on a new image (typically the low-res base layer). Runs every block in approx
        mode, caching per-layer K/V + block-delta-sum for later `.correct()` rounds. Returns the
        patch-only features (prefix tokens stripped, matching stock `get_intermediate_layers`).

        For towers with a CLS token (DINOv2 here; SigLIP has none), also computes the layer-averaged
        CLS->patch attention map (`{tag_prefix}_cls_attn_layermean`, [B, num_patches]) -- AppCorr's
        `cls_attn_prob_layermean` server pscore, for importance-ranked correction-patch selection."""
        collect_cls = self.num_prefix_tokens > 0
        x = self.prepare_full_tokens(pixel_values)
        extracted = None
        for i, blk in enumerate(self.blocks):
            x, cache_feature = blk.approx(x, cache_feature, tag=f"{tag_prefix}_layer{i}",
                                          collect_cls_attn=collect_cls)
            if i == self.extract_block_idx:
                extracted = x
        if collect_cls:
            per_layer = [
                cache_feature[f"{tag_prefix}_layer{i}_cls_attn"] for i in range(len(self.blocks))
            ]
            layermean = torch.stack(per_layer, dim=0).mean(dim=0)  # [B, N]
            cache_feature[f"{tag_prefix}_cls_attn_layermean"] = layermean[:, self.num_prefix_tokens :]
        patch_features = extracted[:, self.num_prefix_tokens :]
        return patch_features, cache_feature

    def correct_forward(
        self,
        pixel_values: torch.Tensor,
        patch_idx: torch.Tensor,
        cache_feature: Dict[str, Any],
        tag_prefix: str,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Subsequent pass once higher-res data has arrived for a subset of patches.

        Args:
            pixel_values: the *current* canvas. In AppCorr-faithful use, non-corrected regions still
                hold base-layer (blurred) data; passing a fully-true image instead only changes the
                *dead* reconstructed stream values at non-corrected positions (never read by any
                consumed output -- attention uses the K/V cache and only `token_idx` rows feed
                norm1/MLP; empirically bit-identical last logits either way, see
                analysis/experiments/audit_progressive_semantics.py). Corrected positions are
                unaffected by this choice because patch_embed is non-overlapping (patch14/stride14),
                so each patch's embedding depends only on its own pixels.
            patch_idx: [Q] long tensor, indices into the *patch grid* (0-indexed, NOT offset by
                `num_prefix_tokens`) that received new data this round.

        Note: prefix tokens (CLS + register, DINOv2 only -- SigLIP has none) are *always* included
        in the corrected/query set, regardless of `patch_idx`. They are not conditioned on pixel data
        directly (fixed learned parameters at block 0), but every later layer's prefix hidden state
        is shaped by attending to the patch tokens, so leaving them permanently frozen at their
        stale (e.g. blurred-image) value would leak into every patch's attention output too (patches
        attend to prefix keys) and break exactness even at 100% patch correction -- confirmed
        empirically: SigLIP (0 prefix tokens) hit exact equality without this; DINOv2 did not, until
        prefix tokens were added to the permanent query set below. Same pattern as Phase 2's LLM
        text-suffix permanent group.
        """
        x = self.prepare_full_tokens(pixel_values)
        patch_token_idx = patch_idx.to(dtype=torch.long, device=x.device) + self.num_prefix_tokens
        prefix_idx = torch.arange(self.num_prefix_tokens, dtype=torch.long, device=x.device)
        token_idx = torch.cat([prefix_idx, patch_token_idx])
        extracted = None
        for i, blk in enumerate(self.blocks):
            x, cache_feature = blk.correct(x, token_idx, cache_feature, tag=f"{tag_prefix}_layer{i}")
            if i == self.extract_block_idx:
                extracted = x
        patch_features = extracted[:, self.num_prefix_tokens :]
        return patch_features, cache_feature
