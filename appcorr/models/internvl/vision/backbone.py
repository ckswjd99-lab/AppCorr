"""InternViT tower with an approx/correct split, plus the post-tower path to LLM embeddings.

Mirrors `appcorr/models/sam3/vision/backbone.py`. The pieces a driver needs:

    prepare_tokens(pixel_values)      patch embed + CLS + absolute position embedding
    approx_forward(x, cache, layers)  run a layer RANGE over every token, caching K/V + increment
    correct_forward(x, mask, cache)   recompute the masked tokens over a layer range
    run_projector(hidden)             final norm, drop CLS, pixel shuffle, multi-modal projector
    full_forward(pixel_values)        stock-equivalent path, for the ceiling arm and for checking

Two things about InternVL shape the interface:

**Tiles are the batch dimension.** `pixel_values` is `[num_tiles, 3, 448, 448]`; each tile is an
independent 1025-token sequence and attention never crosses tiles. So a "token" here is
(tile, position), and selections are boolean masks `[num_tiles, 1025]` rather than a shared index
vector -- the patch score ranks patches across the whole image, so one tile can need many
corrections and another none.

**The CLS token is not a patch.** Position 0 of every tile is CLS, and `get_image_features` drops it
before the pixel shuffle, so it never reaches the LLM. It still participates in attention, so
`correct` may select it; but a patch score computed from image residual energy has nothing to say
about it. `patch_mask_to_token_mask` handles the offset in one place so drivers do not each
re-derive it and quietly select patch 0 when they meant CLS.

The layer range arguments exist for interleaved correction: `layers=(a, b)` runs layers [a, b), and
the cache tags are per layer index, so a later round can correct over a shallower prefix than the
approximate frontier has reached. See docs/memo/interleaved_correction_contract.md -- in particular
that a round corrects its OWN group and the input stream is cumulative, which are the driver's
responsibility, not this class's.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from .block import ApproxCorrectInternVLVisionLayer


class ApproxCorrectInternVLVisionTower(nn.Module):
    """Wraps a stock `InternVLModel` (not just its vision tower: the projector lives on the parent)."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        vt = model.vision_tower
        self.embeddings = vt.embeddings
        self.layernorm = vt.layernorm
        self.layers = nn.ModuleList(
            ApproxCorrectInternVLVisionLayer.from_stock(l) for l in vt.encoder.layer
        )
        self._parent = model                      # for pixel_shuffle + multi_modal_projector
        self._vision_tower = vt
        self.downsample_ratio = model.config.downsample_ratio
        self.select_strategy = model.config.vision_feature_select_strategy

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    # ----------------------------------------------------------------------------------------- #

    @torch.no_grad()
    def prepare_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """[tiles, 3, H, W] -> [tiles, 1 + patches, C], position embedding already added."""
        return self.embeddings(pixel_values, bool_masked_pos=None)

    def patch_mask_to_token_mask(self, patch_mask: torch.Tensor) -> torch.Tensor:
        """[tiles, patches] over PATCHES -> [tiles, 1 + patches] over TOKENS, CLS never selected.

        Correcting CLS would be defensible -- it attends to everything and its value changes when
        patches are corrected -- but it is dropped before the projector, so it cannot affect the
        LLM except through other tokens' attention to it, and a patch score has no opinion on it.
        Excluding it keeps the recompute count equal to the patch count the driver asked for.
        """
        tiles, patches = patch_mask.shape
        out = torch.zeros(tiles, patches + 1, dtype=torch.bool, device=patch_mask.device)
        out[:, 1:] = patch_mask
        return out

    def _range(self, layers: Optional[Tuple[int, int]]) -> Tuple[int, int]:
        return (0, self.num_layers) if layers is None else layers

    @torch.no_grad()
    def approx_forward(self, hidden: torch.Tensor, cache_feature: Dict[str, Any],
                       layers: Optional[Tuple[int, int]] = None,
                       collect_attn: bool = False):
        """Run layers [a, b) over every token, caching K/V and each layer's increment."""
        a, b = self._range(layers)
        attn_acc = None
        for i in range(a, b):
            if collect_attn:
                col = self._incoming_attention(hidden, i)
                attn_acc = col if attn_acc is None else attn_acc + col
            hidden, cache_feature = self.layers[i].approx(hidden, cache_feature, f"l{i}")
        if collect_attn:
            key = "vision_layer_patch_attn_layermean"
            prev = cache_feature.get(key)
            total = attn_acc / max(1, b - a)
            cache_feature[key] = total if prev is None else (prev + total) / 2
        return hidden, cache_feature

    @torch.no_grad()
    def correct_forward(self, hidden: torch.Tensor, token_mask: torch.Tensor,
                        cache_feature: Dict[str, Any],
                        layers: Optional[Tuple[int, int]] = None):
        """Recompute `token_mask` [tiles, tokens] through layers [a, b)."""
        a, b = self._range(layers)
        for i in range(a, b):
            hidden, cache_feature = self.layers[i].correct(hidden, token_mask, cache_feature, f"l{i}")
        return hidden, cache_feature

    @torch.no_grad()
    def _incoming_attention(self, hidden: torch.Tensor, layer_idx: int) -> torch.Tensor:
        """Column mass of one layer's attention matrix, head- and query-averaged. [tiles, patches].

        How much the rest of the tile reads FROM each patch, which is the term the default patch
        score multiplies residual energy by. Computed on the approximate pass, from the same
        projections the layer will use. CLS is dropped so the result lines up with patch indices.
        """
        layer = self.layers[layer_idx]
        normed = layer.layernorm_before(hidden)
        q, k, _ = layer._project_qkv(normed)
        scale = layer.attention.scale
        # [tiles, heads, S, S] at 13x1025 in fp32 would be ~3.5 GB; accumulate in query chunks.
        tiles, heads, seq, _ = q.shape
        col = torch.zeros(tiles, seq, device=hidden.device, dtype=torch.float32)
        chunk = 512
        for s in range(0, seq, chunk):
            e = min(s + chunk, seq)
            w = torch.softmax((q[:, :, s:e] @ k.transpose(-1, -2)) * scale, dim=-1)
            col += w.float().sum(dim=2).mean(dim=1)
        return (col / seq)[:, 1:]

    @torch.no_grad()
    def run_projector(self, hidden: torch.Tensor) -> torch.Tensor:
        """Final norm -> drop CLS -> pixel shuffle -> multi-modal projector. [tiles, llm_tokens, D].

        Reproduces `InternVLModel.get_image_features` from the tower output onward, so a driver can
        swap in corrected features and let the model run its own code from the LLM inward.
        """
        feats = self.layernorm(hidden)
        if self.select_strategy == "default":
            feats = feats[:, 1:, :]
        tiles, n, _ = feats.shape
        side = int(n ** 0.5)
        feats = feats.reshape(tiles, side, side, -1)
        feats = self._parent.pixel_shuffle(feats, scale_factor=self.downsample_ratio)
        feats = feats.reshape(tiles, -1, feats.shape[-1])
        return self._parent.multi_modal_projector(feats)

    @torch.no_grad()
    def full_forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Stock-equivalent path: the ceiling arm, and what the fork is checked against."""
        hidden = self.prepare_tokens(pixel_values)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.run_projector(hidden)
