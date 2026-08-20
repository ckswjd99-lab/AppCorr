"""SigLIP encoder layer with an approx/correct split, for Gemma 3's vision tower.

`approx` runs the whole layer over every token and caches two things: the layer's total increment
(`{tag}_blocks_out_sum` = attention contribution + MLP contribution) so a later `correct` can
reconstruct untouched positions exactly as `x_in + increment`, and the raw K/V so `correct` can
splice fresh keys and values in for the positions it recomputes.

This is the simplest fork in the repo, and it is worth saying why so the simplicity is not mistaken
for something missing:

**No CLS token.** SigLIP's vision embeddings are patch embeddings plus a learned absolute position
embedding, nothing prepended. Gemma 3 at 896x896 with patch 14 gives 64x64 = 4096 tokens, all of
them patches, so a patch index IS a token index -- unlike InternVL, where position 0 of every tile is
a CLS token that has to be offset around.

**Global attention on every layer, no RoPE.** Position enters once, before layer 0. There is no
window bookkeeping and no per-query rotary phase, which is what made the SAM 3 fork long.

**No QK-norm, no LayerScale.** Plain pre-norm residual: `x + attn(norm1(x))`, then
`x + mlp(norm2(x))`. Every extra term in a residual path is a chance for `correct` to diverge from
stock in a way that reads as float noise, and this layer has none of them.

Bidirectional attention is the normal case here and is not an obstacle: correction never needs
causality, only a cached K/V it can splice into. What bidirectionality blocks is *streaming*, which
is exactly why Gemma 3 is a model where this technique is needed
(docs/memo/ or memory: streaming vs appcorr scope).
"""

from __future__ import annotations

from typing import Any, Dict

import torch
from torch import nn


class ApproxCorrectSiglipLayer(nn.Module):
    """Wraps a stock `SiglipEncoderLayer`, adding `.approx()` and `.correct()`."""

    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.layer_norm1 = layer.layer_norm1
        self.self_attn = layer.self_attn
        self.layer_norm2 = layer.layer_norm2
        self.mlp = layer.mlp
        self._stock = layer

    @classmethod
    def from_stock(cls, layer: nn.Module) -> "ApproxCorrectSiglipLayer":
        return cls(layer)

    def forward(self, hidden_states: torch.Tensor, attention_mask=None) -> torch.Tensor:
        return self._stock(hidden_states, attention_mask=attention_mask)

    # ----------------------------------------------------------------------------------------- #

    def _qkv(self, normed: torch.Tensor):
        """q/k/v shaped [B, heads, S, head_dim], matching the stock reshape exactly."""
        a = self.self_attn
        shape = (*normed.shape[:-1], -1, a.head_dim)
        q = a.q_proj(normed).view(shape).transpose(1, 2)
        k = a.k_proj(normed).view(shape).transpose(1, 2)
        v = a.v_proj(normed).view(shape).transpose(1, 2)
        return q, k, v

    def _out(self, attn: torch.Tensor, input_shape) -> torch.Tensor:
        return self.self_attn.out_proj(attn.transpose(1, 2).reshape(*input_shape, -1).contiguous())

    # ----------------------------------------------------------------------------------------- #

    @torch.no_grad()
    def approx(self, x: torch.Tensor, cache_feature: Dict[str, Any], tag: str):
        """Full layer over every token, caching K/V and the total increment."""
        normed = self.layer_norm1(x)
        q, k, v = self._qkv(normed)
        attn = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, scale=self.self_attn.scale
        )
        attn_out = self._out(attn, x.shape[:-1])
        x_attn = x + attn_out
        mlp_out = self.mlp(self.layer_norm2(x_attn))

        cache_feature[f"{tag}_k"] = k
        cache_feature[f"{tag}_v"] = v
        cache_feature[f"{tag}_blocks_out_sum"] = attn_out + mlp_out
        return x_attn + mlp_out, cache_feature

    @torch.no_grad()
    def correct(self, x: torch.Tensor, token_mask: torch.Tensor,
                cache_feature: Dict[str, Any], tag: str):
        """Recompute the tokens selected by `token_mask` [B, N]; reconstruct the rest.

        Positions outside the mask come out identical to an approx-only forward, which is what makes
        partial correction a cheaper forward rather than a different model.
        """
        batch = x.shape[0]
        if token_mask.dim() == 1:
            token_mask = token_mask.unsqueeze(0).expand(batch, -1)
        token_mask = token_mask.to(x.device)

        k_cache = cache_feature[f"{tag}_k"].clone()
        v_cache = cache_feature[f"{tag}_v"].clone()
        increment = cache_feature[f"{tag}_blocks_out_sum"]
        out = (x + increment.to(x.dtype)).clone()
        new_increment = increment.clone()

        # Per batch element: the selected count differs between images, and SDPA wants a uniform
        # query count per call. Batches here are small (one image, or a handful).
        for b in range(batch):
            idx = token_mask[b].nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            x_active = x[b : b + 1, idx]
            normed = self.layer_norm1(x_active)
            q_sel, k_sel, v_sel = self._qkv(normed)

            # Fresh K/V for the corrected positions, stale cached K/V everywhere else.
            k_cache[b : b + 1, :, idx, :] = k_sel
            v_cache[b : b + 1, :, idx, :] = v_sel

            attn = torch.nn.functional.scaled_dot_product_attention(
                q_sel, k_cache[b : b + 1], v_cache[b : b + 1],
                attn_mask=None, dropout_p=0.0, scale=self.self_attn.scale,
            )
            attn_out = self._out(attn, x_active.shape[:-1])
            x_attn_active = x_active + attn_out
            mlp_out = self.mlp(self.layer_norm2(x_attn_active))

            out[b : b + 1, idx] = (x_attn_active + mlp_out).to(out.dtype)
            # Persist the corrected increment. Without this a LATER round reconstructs every
            # position it is not correcting from the approximate value and throws away what earlier
            # rounds fixed -- interleaved keeps only its last round, while one-shot looks fine.
            # See docs/memo/interleaved_correction_contract.md.
            new_increment[b : b + 1, idx] = (attn_out + mlp_out).to(new_increment.dtype)

        cache_feature[f"{tag}_k"] = k_cache
        cache_feature[f"{tag}_v"] = v_cache
        cache_feature[f"{tag}_blocks_out_sum"] = new_increment
        return out, cache_feature
