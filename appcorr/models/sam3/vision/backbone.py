"""SAM 3 vision tower with an approx/correct split over its 32 ViT layers.

Wraps a stock `Sam3VisionModel`. The ViT layers are replaced by `ApproxCorrectSam3ViTLayer`; the
patch embedding, the pre-layer norm and the FPN neck are left exactly as they were, because only the
layer stack is expensive enough to be worth splitting and only it carries state between rounds.

The split follows the same contract as the DINOv3 and CLIP forks:

    approx_forward(pixel_values, layers=(a, b))   run layers [a, b) over every token, caching each
                                                  layer's increment and K/V
    correct_forward(x, token_idx, layers=(a, b))  recompute `token_idx` through layers [a, b),
                                                  reconstructing every other position from the cache

so a scheduler can interleave them over layer ranges and over token groups without this module
knowing which policy is driving it.

**The neck is not split, and that is a decision, not an oversight.** `Sam3VisionNeck` is an FPN over
the final hidden state; it is cheap next to 32 layers, and splitting it would mean carrying partial
feature pyramids between rounds for no measurable gain. It runs once, on whatever the layer stack
produced.

**What the tower returns is not the model's answer.** SAM 3's detector consumes the FPN outputs
through a DETR encoder/decoder and a mask decoder. Everything here changes the vision features; the
metric only moves once those features reach the mask decoder, so a correction that looks large here
can be invisible downstream and vice versa. Do not read layer-level error as a proxy for mask AP.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence

import torch
from torch import nn

from .block import ApproxCorrectSam3ViTLayer


class ApproxCorrectSam3VisionTower(nn.Module):
    """`Sam3VisionModel` whose ViT layers can be run approximately and then corrected."""

    def __init__(self, vision_model: nn.Module) -> None:
        super().__init__()
        backbone = vision_model.backbone
        self.embeddings = backbone.embeddings
        self.layer_norm = backbone.layer_norm
        self.layers = nn.ModuleList(
            [ApproxCorrectSam3ViTLayer.from_stock(layer) for layer in backbone.layers]
        )
        self.neck = vision_model.neck
        self.config = vision_model.config
        self.patch_size = vision_model.config.backbone_config.patch_size

    # ------------------------------------------------------------------ setup

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    @property
    def global_layers(self) -> list[int]:
        """Indices whose attention spans the whole image; the rest see only their 24x24 window."""
        return [i for i, layer in enumerate(self.layers) if layer.window_size == 0]

    def prepare_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Patch-embed to the [B, H, W, C] spatial layout the layers expect."""
        hidden = self.embeddings(pixel_values)
        batch = hidden.shape[0]
        height = pixel_values.shape[-2] // self.patch_size
        width = pixel_values.shape[-1] // self.patch_size
        hidden = hidden.view(batch, height, width, hidden.shape[-1])
        return self.layer_norm(hidden)

    # -------------------------------------------------------------- the split

    def approx_forward(
        self,
        hidden: torch.Tensor,
        cache_feature: Dict[str, Any],
        layers: Sequence[int] | None = None,
        tag_prefix: str = "vision_layer",
    ):
        """Run layers [start, end) over all tokens, caching what `correct_forward` will need."""
        start, end = (0, self.num_layers) if layers is None else (int(layers[0]), int(layers[1]))
        for idx in range(start, end):
            hidden, cache_feature = self.layers[idx].approx(
                hidden, cache_feature, f"{tag_prefix}{idx}"
            )
        return hidden, cache_feature

    def correct_forward(
        self,
        hidden: torch.Tensor,
        token_idx: torch.Tensor,
        cache_feature: Dict[str, Any],
        layers: Sequence[int] | None = None,
        tag_prefix: str = "vision_layer",
    ):
        """Recompute `token_idx` through layers [start, end).

        `token_idx` indexes the flattened (H, W) grid and is the same set for every layer in the
        range. Untouched positions come out exactly where the approximate pass left them, so the
        result is a valid input to the next range either way.
        """
        start, end = (0, self.num_layers) if layers is None else (int(layers[0]), int(layers[1]))
        for idx in range(start, end):
            hidden, cache_feature = self.layers[idx].correct(
                hidden, token_idx, cache_feature, f"{tag_prefix}{idx}"
            )
        return hidden, cache_feature

    # -------------------------------------------------------------------- out

    def run_neck(self, hidden: torch.Tensor):
        """FPN over the final hidden state; returns what `Sam3VisionModel.forward` would.

        Mirrors the stock reshape: the layers hand back [B, H, W, C] and the neck wants
        [B, C, H, W].
        """
        spatial = hidden.permute(0, 3, 1, 2).contiguous()
        return self.neck(spatial)

    @torch.no_grad()
    def full_forward(self, pixel_values: torch.Tensor):
        """Stock-equivalent path, for the ceiling arm and for checking the fork against."""
        hidden = self.prepare_tokens(pixel_values)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.run_neck(hidden)
