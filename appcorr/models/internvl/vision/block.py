"""InternViT layer with an approx/correct split, mirroring the SAM 3, DINOv3 and CLIP forks.

`approx` runs the whole layer over every token and caches two things: the layer's total increment
(`{tag}_blocks_out_sum` = attention contribution + MLP contribution) so a later `correct` can
reconstruct untouched positions exactly as `x_in + increment`, and the raw K/V so `correct` can
splice fresh keys and values in for the positions it recomputes.

This is the simplest fork in the repo so far, and the differences from SAM 3 are worth naming
because they are what make it simple:

**Attention is global on every layer.** InternViT has no windows and no RoPE -- position comes from a
learned absolute embedding added once, before layer 0. SAM 3 needed `_locate`, per-window K/V
splicing and window-local rotary phases; none of that exists here.

**Tiles are the batch dimension.** `pixel_values` arrives as `[num_tiles, 3, 448, 448]`, so each
448x448 tile is an independent 1025-token sequence (1 CLS + 32x32 patches) and attention never
crosses tiles. Tiles play SAM 3's window role, but the model already separates them for us.

**Correction is per-batch-element.** The patch score ranks patches across the whole image, so
different tiles need different numbers of tokens corrected. `correct` therefore takes a boolean mask
`[B, N]` rather than a shared index vector, and loops over the batch -- SDPA needs a uniform query
count per call and tiles are at most 13, so a loop is cheaper than padding.

**QK-norm and LayerScale are in the residual path.** `q_norm`/`k_norm` apply to the projections
before the head reshape, and `lambda_1`/`lambda_2` scale each branch before its residual add. Both
have to be reproduced in `correct` in the same order or the recomputed tokens drift from stock in a
way that looks like ordinary float noise.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn


class ApproxCorrectInternVLVisionLayer(nn.Module):
    """Wraps a stock `InternVLVisionLayer`, adding `.approx()` and `.correct()`."""

    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.attention = layer.attention
        self.mlp = layer.mlp
        self.layernorm_before = layer.layernorm_before
        self.layernorm_after = layer.layernorm_after
        self.lambda_1 = layer.lambda_1
        self.lambda_2 = layer.lambda_2
        self.dropout = layer.dropout
        self._stock = layer

    @classmethod
    def from_stock(cls, layer: nn.Module) -> "ApproxCorrectInternVLVisionLayer":
        return cls(layer)

    # ----------------------------------------------------------------------------------------- #

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self._stock(hidden_states)

    def _project_qkv(self, normed: torch.Tensor):
        """q/k/v with QK-norm applied, shaped [B, heads, S, head_dim]."""
        a = self.attention
        b, s, _ = normed.shape
        q = a.q_norm(a.q_proj(normed))
        k = a.k_norm(a.k_proj(normed))
        v = a.v_proj(normed)
        q = q.reshape(b, s, a.num_heads, a.head_dim).transpose(1, 2)
        k = k.reshape(b, s, a.num_heads, a.head_dim).transpose(1, 2)
        v = v.reshape(b, s, a.num_heads, a.head_dim).transpose(1, 2)
        return q, k, v

    def _attn_out(self, attn: torch.Tensor) -> torch.Tensor:
        """[B, heads, S, head_dim] -> projected + dropped output, matching stock ordering."""
        a = self.attention
        b, _, s, _ = attn.shape
        out = attn.transpose(1, 2).reshape(b, s, a.embed_dim)
        return a.projection_dropout(a.projection_layer(out))

    # ----------------------------------------------------------------------------------------- #

    @torch.no_grad()
    def approx(self, x: torch.Tensor, cache_feature: Dict[str, Any], tag: str):
        """Full layer over every token, caching K/V and the total increment."""
        normed = self.layernorm_before(x)
        q, k, v = self._project_qkv(normed)
        attn = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, scale=self.attention.scale
        )
        attn_out = self.lambda_1 * self._attn_out(attn)
        x_attn = x + attn_out

        mlp_out = self.lambda_2 * self.dropout(self.mlp(self.layernorm_after(x_attn)))
        out = x_attn + mlp_out

        cache_feature[f"{tag}_k"] = k
        cache_feature[f"{tag}_v"] = v
        cache_feature[f"{tag}_blocks_out_sum"] = attn_out + mlp_out
        return out, cache_feature

    @torch.no_grad()
    def correct(self, x: torch.Tensor, token_mask: torch.Tensor,
                cache_feature: Dict[str, Any], tag: str):
        """Recompute the tokens selected by `token_mask` [B, N]; reconstruct the rest.

        Positions outside the mask come out identical to an approx-only forward, which is what makes
        partial correction meaningful rather than a different model.
        """
        batch, seq, channels = x.shape
        if token_mask.dim() == 1:                      # shared selection, broadcast over the batch
            token_mask = token_mask.unsqueeze(0).expand(batch, -1)
        token_mask = token_mask.to(x.device)

        k_cache = cache_feature[f"{tag}_k"].clone()
        v_cache = cache_feature[f"{tag}_v"].clone()
        increment = cache_feature[f"{tag}_blocks_out_sum"]
        out = (x + increment.to(x.dtype)).clone()
        new_increment = increment.clone()

        for b in range(batch):
            idx = token_mask[b].nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            x_active = x[b : b + 1, idx]                       # [1, Q, C]
            normed = self.layernorm_before(x_active)
            q_sel, k_sel, v_sel = self._project_qkv(normed)

            # Fresh K/V for the corrected positions, stale cached K/V for the rest of the tile.
            k_cache[b : b + 1, :, idx, :] = k_sel
            v_cache[b : b + 1, :, idx, :] = v_sel

            attn = torch.nn.functional.scaled_dot_product_attention(
                q_sel, k_cache[b : b + 1], v_cache[b : b + 1],
                attn_mask=None, dropout_p=0.0, scale=self.attention.scale,
            )
            attn_out = self.lambda_1 * self._attn_out(attn)
            x_attn_active = x_active + attn_out
            mlp_out = self.lambda_2 * self.dropout(self.mlp(self.layernorm_after(x_attn_active)))

            out[b : b + 1, idx] = (x_attn_active + mlp_out).to(out.dtype)
            # Persist the corrected increment, unconditionally. Without this a LATER round
            # reconstructs every position it is not correcting from the approximate value and
            # discards what earlier rounds fixed -- interleaved correction keeps only its last
            # round, while one-shot looks fine. See docs/memo/interleaved_correction_contract.md.
            new_increment[b : b + 1, idx] = (attn_out + mlp_out).to(new_increment.dtype)

        cache_feature[f"{tag}_k"] = k_cache
        cache_feature[f"{tag}_v"] = v_cache
        cache_feature[f"{tag}_blocks_out_sum"] = new_increment
        return out, cache_feature
