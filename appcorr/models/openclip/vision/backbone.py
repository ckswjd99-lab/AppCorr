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
        self,
        x_feature: torch.Tensor,
        start_l: int,
        end_l: int,
        cache_feature: Dict[str, Any],
        tag_prefix: str,
        collect_cls_attn: bool = True,
        collect_attn_mean: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Runs blocks[start_l:end_l] in approx mode, caching per-layer K/V + block-delta-sum for
        later `.correct()` rounds. Layer-range chunked (mirrors the DINOv3 classifier executor's
        `approx_forward(layers=(start_l, end_l))` contract) so the existing `GroupTriggerPolicy`
        scheduling policy can drive this tower directly, interleaving layer chunks with patch
        arrival, with no new scheduling code needed. `x_feature` is `prepare_full_tokens()`'s output
        on the first call (`start_l == 0`); the caller threads the returned tensor through
        subsequent calls."""
        for i in range(start_l, end_l):
            blk = self.blocks[i]
            x_feature, cache_feature = blk.approx(x_feature, cache_feature, tag=f"{tag_prefix}_layer{i}",
                                                  collect_cls_attn=collect_cls_attn,
                                                  collect_attn_mean=collect_attn_mean)
        return x_feature, cache_feature

    def correct_forward(
        self,
        x_feature: torch.Tensor,
        patch_idx: torch.Tensor,
        start_l: int,
        end_l: int,
        cache_feature: Dict[str, Any],
        tag_prefix: str,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Runs blocks[start_l:end_l] in correct mode for the given patch subset (layer-range
        chunked, same contract as `approx_forward` above).

        Args:
            x_feature: current residual stream, restarted from `prepare_full_tokens()`'s output at
                the start of EACH correction round (same invariant as
                `ApproxCorrectCLIPEncoderLayer.correct`/the OpenVLA vision fork).
            patch_idx: [Q] long tensor, indices into the *patch grid* (0-indexed, NOT offset by
                `num_prefix_tokens`) that received new data this round.
        """
        patch_token_idx = patch_idx.to(dtype=torch.long, device=x_feature.device) + self.num_prefix_tokens
        prefix_idx = torch.arange(self.num_prefix_tokens, dtype=torch.long, device=x_feature.device)
        token_idx = torch.cat([prefix_idx, patch_token_idx])
        for i in range(start_l, end_l):
            blk = self.blocks[i]
            x_feature, cache_feature = blk.correct(x_feature, token_idx, cache_feature, tag=f"{tag_prefix}_layer{i}")
        return x_feature, cache_feature

    def finalize_attn_layermean(self, cache_feature: Dict[str, Any], tag_prefix: str) -> Dict[str, Any]:
        """Layer-average of the RECEIVED-attention column mean cached by `approx_forward`'s
        `collect_attn_mean` -- the `patch_attn_prob_layermean` server pscore, the same signal DINOv3
        and Gemma 3 use. Sibling of `finalize_cls_attn_layermean`; same "only approx() writes these,
        so any chunk boundary is a safe place to call it" property, for the same reason."""
        per_layer = [
            cache_feature[k] for k in sorted(
                (k for k in cache_feature
                 if k.startswith(f"{tag_prefix}_layer") and k.endswith("_attn_mean")),
                key=lambda k: int(k[len(f"{tag_prefix}_layer"):-len("_attn_mean")]),
            )
        ]
        if not per_layer:
            return cache_feature
        layermean = torch.stack(per_layer, dim=0).mean(dim=0)  # [B, N]
        cache_feature[f"{tag_prefix}_attn_layermean"] = layermean[:, self.num_prefix_tokens:]
        return cache_feature

    def finalize_cls_attn_layermean(self, cache_feature: Dict[str, Any], tag_prefix: str) -> Dict[str, Any]:
        """Averages the per-layer CLS->patch attention (cached by `approx_forward`'s
        `collect_cls_attn`) across all layers seen SO FAR -- the `cls_attn_prob_layermean` server
        pscore. Only layers that have actually run through `.approx()` have a cached `_cls_attn`
        entry (`.correct()` never writes one), so this is safe to call after ANY approx chunk, not
        just the final one -- e.g. group 0's initial `approx_forward(0, chunk_size)` already gives
        a usable (if less refined than a full-depth average) importance signal for pruning
        subsequent groups' patches, without needing to wait for the whole 48-layer forward."""
        per_layer = [
            cache_feature[k] for k in sorted(
                (k for k in cache_feature if k.startswith(f"{tag_prefix}_layer") and k.endswith("_cls_attn")),
                key=lambda k: int(k[len(f"{tag_prefix}_layer"):-len("_cls_attn")]),
            )
        ]
        if not per_layer:
            return cache_feature
        layermean = torch.stack(per_layer, dim=0).mean(dim=0)  # [B, N]
        cache_feature[f"{tag_prefix}_cls_attn_layermean"] = layermean[:, self.num_prefix_tokens:]
        return cache_feature
