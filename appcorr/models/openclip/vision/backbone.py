"""
backbone.py

Wraps the stock (pretrained) `transformers.models.clip.modeling_clip.CLIPVisionTransformer` (from
`laion/CLIP-ViT-bigG-14-laion2B-39B-b160k`) with approx/correct forward passes built from
`ApproxCorrectCLIPEncoderLayer`. Mirrors `appcorr/models/openvla/vision/backbone.py`.

Key difference from the OpenVLA vision backbone: CLIP needs the FULL 48-layer forward (no
depth-2 truncation -- there is no downstream LLM consuming intermediate patch features, only the
final CLS state matters), followed by `post_layernorm(CLS) -> visual_projection -> L2-normalize` to
produce the shared image/text embedding space vector used for zero-shot classification and
retrieval. `post_layernorm`/`visual_projection` are cheap (touch only the single CLS row, O(1) per
image) so they are always recomputed fresh from the current CLS hidden state -- no caching needed,
same "cheap non-block ops are always exact" philosophy as `prepare_full_tokens` in the OpenVLA fork.

CLIP has exactly 1 prefix token (CLS, no register tokens), always force-included in the corrected/
query set for the same reason DINOv2's prefix tokens are in the OpenVLA fork: every patch attends to
the CLS key/value too, so leaving CLS permanently stale would leak into every patch's attention
output and break exactness even at 100% patch correction.
"""

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from .block import ApproxCorrectCLIPEncoderLayer


class ApproxCorrectCLIPVisionTower(nn.Module):
    def __init__(self, vision_transformer: nn.Module, visual_projection: nn.Module):
        super().__init__()
        self.embeddings = vision_transformer.embeddings
        self.pre_layrnorm = vision_transformer.pre_layrnorm
        self.post_layernorm = vision_transformer.post_layernorm
        self.visual_projection = visual_projection
        self.blocks = nn.ModuleList(
            [ApproxCorrectCLIPEncoderLayer.from_stock(layer) for layer in vision_transformer.encoder.layers]
        )
        self.num_prefix_tokens = 1  # CLS only, no register tokens
        self.extract_block_idx = len(self.blocks) - 1  # full depth, unlike OpenVLA's depth-2 truncation

    def prepare_full_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """embeddings (patch_embed + CLS + learned pos-embed) -> pre_layrnorm. Always exact --
        tokenization has no notion of "corrected vs. stale"."""
        x = self.embeddings(pixel_values)
        x = self.pre_layrnorm(x)
        return x

    def get_image_embeds(self, x_full: torch.Tensor) -> torch.Tensor:
        """post_layernorm(CLS) -> visual_projection -> L2-normalize. `x_full` is the LAST layer's
        [B, N, C] hidden state (i.e. after `self.blocks[-1]` has run, whether via approx or correct)."""
        cls = x_full[:, 0, :]
        pooled = self.post_layernorm(cls)
        embeds = self.visual_projection(pooled)
        return embeds / embeds.norm(dim=-1, keepdim=True)

    def approx_forward(
        self, pixel_values: torch.Tensor, cache_feature: Dict[str, Any], tag_prefix: str
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """First pass on a new image (typically the low-res base layer). Runs every block in approx
        mode, caching per-layer K/V + block-delta-sum for later `.correct()` rounds. Returns the
        final normalized image embedding.

        Also computes the layer-averaged CLS->patch attention map (`{tag_prefix}_cls_attn_layermean`,
        [B, num_patches]) -- the `cls_attn_prob_layermean` server pscore used for importance-ranked
        correction-patch selection (same recipe validated for the DINOv3 classifier)."""
        x = self.prepare_full_tokens(pixel_values)
        for i, blk in enumerate(self.blocks):
            x, cache_feature = blk.approx(x, cache_feature, tag=f"{tag_prefix}_layer{i}",
                                          collect_cls_attn=True)
        per_layer = [cache_feature[f"{tag_prefix}_layer{i}_cls_attn"] for i in range(len(self.blocks))]
        layermean = torch.stack(per_layer, dim=0).mean(dim=0)  # [B, N]
        cache_feature[f"{tag_prefix}_cls_attn_layermean"] = layermean[:, self.num_prefix_tokens:]
        image_embeds = self.get_image_embeds(x)
        return image_embeds, cache_feature

    def correct_forward(
        self,
        pixel_values: torch.Tensor,
        patch_idx: torch.Tensor,
        cache_feature: Dict[str, Any],
        tag_prefix: str,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Subsequent pass once higher-res data has arrived for a subset of patches.

        Args:
            pixel_values: the *current* canvas (see OpenVLA backbone docstring for why passing a
                fully-true image instead of a faithful partial canvas is harmless here too -- same
                non-overlapping patch_embed argument applies, patch14/stride14).
            patch_idx: [Q] long tensor, indices into the *patch grid* (0-indexed, NOT offset by
                `num_prefix_tokens`) that received new data this round.
        """
        x = self.prepare_full_tokens(pixel_values)
        patch_token_idx = patch_idx.to(dtype=torch.long, device=x.device) + self.num_prefix_tokens
        prefix_idx = torch.arange(self.num_prefix_tokens, dtype=torch.long, device=x.device)
        token_idx = torch.cat([prefix_idx, patch_token_idx])
        for i, blk in enumerate(self.blocks):
            x, cache_feature = blk.correct(x, token_idx, cache_feature, tag=f"{tag_prefix}_layer{i}")
        image_embeds = self.get_image_embeds(x)
        return image_embeds, cache_feature
