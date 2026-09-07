"""OneVision-Encoder layer with an approx/correct split.

Same contract as the Gemma 3 SigLIP fork: `approx` runs the whole layer and caches the total
increment (so untouched positions reconstruct exactly as `x_in + increment`) plus the post-RoPE K/V
(so `correct` can splice fresh keys and values in for the positions it recomputes).

Three things differ from the SigLIP fork and all three are places to get it wrong:

**Fused QKV.** One `qkv` Linear of width 3*embed_dim, sliced by the stock reshape
`(B, L, 3, H, D).permute(2,0,1,3,4)`. Slicing it any other way silently permutes heads.

**RoPE.** Position enters per layer, not once before layer 0, so `correct` must apply the rotary
phase of the SELECTED positions -- `freqs[:, idx]` -- not of `0..len(idx)`. Getting this wrong
produces a plausible-looking tensor that is wrong everywhere the selection is not a prefix.

**`proj`, not `out_proj`**, and attention output arrives as (B, H, L, D) from SDPA while the stock
path reshapes from (B, L, H, D); the transpose is not optional.

Attention is global and bidirectional over the whole image -- measured, `cu_seqlens` comes back as
[0, N] for a still image -- so there is no window bookkeeping and no mask to thread through.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, Optional

import torch
from torch import nn


class ApproxCorrectOneVisionLayer(nn.Module):
    """Wraps a stock `OneVisionEncoderEncoderLayer`, adding `.approx()` and `.correct()`."""

    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.layer_norm1 = layer.layer_norm1
        self.self_attn = layer.self_attn
        self.layer_norm2 = layer.layer_norm2
        self.mlp = layer.mlp
        self._stock = layer
        # The rotary helper lives in the checkpoint's remote module; take it from the layer's own
        # module rather than re-implementing it, so a change upstream cannot silently diverge.
        self._rope = sys.modules[type(layer).__module__].apply_rotary_pos_emb

    @classmethod
    def from_stock(cls, layer: nn.Module) -> "ApproxCorrectOneVisionLayer":
        return cls(layer)

    def forward(self, hidden_states: torch.Tensor, **kw) -> torch.Tensor:
        return self._stock(hidden_states, **kw)

    # ----------------------------------------------------------------------------------------- #

    def _qkv(self, normed: torch.Tensor, freqs: Optional[torch.Tensor]):
        """q/k/v as (B, H, L, D), post-RoPE, matching the stock reshape exactly."""
        a = self.self_attn
        b, l, _ = normed.shape
        q, k, v = (a.qkv(normed).reshape(b, l, 3, a.num_heads, a.head_dim)
                   .permute(2, 0, 1, 3, 4).unbind(0))
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if freqs is not None:
            q, k = self._rope(q, k, freqs)
        return q, k, v

    def _out(self, attn: torch.Tensor, b: int, l: int) -> torch.Tensor:
        return self.self_attn.proj(attn.transpose(1, 2).reshape(b, l, -1).contiguous())

    # ----------------------------------------------------------------------------------------- #

    @torch.no_grad()
    def approx(self, x: torch.Tensor, cache_feature: Dict[str, Any], tag: str,
               freqs: Optional[torch.Tensor] = None):
        b, l, _ = x.shape
        normed = self.layer_norm1(x)
        q, k, v = self._qkv(normed, freqs)
        attn = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.0, scale=self.self_attn.scale)
        attn_out = self._out(attn, b, l)
        x_attn = x + attn_out
        mlp_out = self.mlp(self.layer_norm2(x_attn))

        cache_feature[f"{tag}_k"] = k
        cache_feature[f"{tag}_v"] = v
        cache_feature[f"{tag}_blocks_out_sum"] = attn_out + mlp_out
        cache_feature[f"{tag}_in_sig"] = (float(x.float().mean()), float(x.float().std()),
                                          tuple(x.shape))
        return x_attn + mlp_out, cache_feature

    @torch.no_grad()
    def correct(self, x: torch.Tensor, token_mask: torch.Tensor,
                cache_feature: Dict[str, Any], tag: str,
                freqs: Optional[torch.Tensor] = None):
        """Recompute the tokens in `token_mask` [B, N]; reconstruct the rest from the increment."""
        batch = x.shape[0]
        if token_mask.dim() == 1:
            token_mask = token_mask.unsqueeze(0).expand(batch, -1)
        token_mask = token_mask.to(x.device)

        k_cache = cache_feature[f"{tag}_k"].clone()
        v_cache = cache_feature[f"{tag}_v"].clone()
        increment = cache_feature[f"{tag}_blocks_out_sum"]
        out = (x + increment.to(x.dtype)).clone()
        new_increment = increment.clone()

        for bi in range(batch):
            idx = token_mask[bi].nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            x_active = x[bi : bi + 1, idx]
            # The rotary phase of the SELECTED positions, not of a fresh 0..n range.
            f_sel = None if freqs is None else freqs[bi : bi + 1, idx]
            normed = self.layer_norm1(x_active)
            q_sel, k_sel, v_sel = self._qkv(normed, f_sel)

            k_cache[bi : bi + 1, :, idx, :] = k_sel
            v_cache[bi : bi + 1, :, idx, :] = v_sel

            attn = torch.nn.functional.scaled_dot_product_attention(
                q_sel, k_cache[bi : bi + 1], v_cache[bi : bi + 1],
                attn_mask=None, dropout_p=0.0, scale=self.self_attn.scale)
            attn_out = self._out(attn, 1, idx.numel())
            x_attn_active = x_active + attn_out
            mlp_out = self.mlp(self.layer_norm2(x_attn_active))

            out[bi : bi + 1, idx] = (x_attn_active + mlp_out).to(out.dtype)
            # Persist the corrected increment, or a later interleaved round rebuilds these
            # positions from the approximate value and discards what this round fixed.
            new_increment[bi : bi + 1, idx] = (attn_out + mlp_out).to(new_increment.dtype)

        cache_feature[f"{tag}_k"] = k_cache
        cache_feature[f"{tag}_v"] = v_cache
        cache_feature[f"{tag}_blocks_out_sum"] = new_increment
        return out, cache_feature
