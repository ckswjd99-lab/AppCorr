"""Gemma 3 decoder layer with an approx/correct split — the LLM half of the unified axis.

The vision half (`appcorr/models/gemma3/vision/block.py`) is the simplest fork in the repo. This one
is the most delicate, and every difference below is a place where copying the SigLIP pattern would
produce numbers that look like float noise and are wrong:

**Four layernorms, sandwich style.** Gemma 3 norms the *branch output* before adding the residual:

    x = residual + post_attention_layernorm(attn(input_layernorm(x)))
    x = residual + post_feedforward_layernorm(mlp(pre_feedforward_layernorm(x)))

SigLIP has two norms and adds the branch raw. Getting this wrong changes the scale of every
correction while leaving the shapes right.

**QK-norm before RoPE.** `q_norm`/`k_norm` apply to the head-shaped projections, and only then does
the rotary embedding go on. Reordering them is invisible in an argmax and wrong everywhere.

**RoPE is per position, so a corrected token needs ITS OWN cos/sin.** The caller passes the full
`(cos, sin)` for the sequence and this slices to the corrected indices. Feeding the first `Q` rows
instead — the obvious mistake, since that is what a contiguous prefill would want — yields finite,
plausible values with every corrected token rotated to the wrong place.

**GQA: 8 query heads over 4 KV heads.** K/V are cached at 4 heads and expanded at use, so the cache
stays small (that is the point of GQA) but the expansion has to happen on both the approx and the
correct path or the head mapping silently differs.

**The attention mask must be sliced on the QUERY axis.** Gemma 3's image tokens are bidirectional
among themselves while text is causal, so mask rows are not interchangeable: correcting query `i`
requires row `i`. A mask sliced wrongly still produces a number.

`sliding_window` is passed through unchanged. At one image plus a short prompt (~277 tokens) the
window (1024) never bites and every layer behaves as full attention, but that is a property of the
prompt, not of the model — longer prompts or several images will separate the two.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from transformers.models.gemma3.modeling_gemma3 import apply_rotary_pos_emb, repeat_kv


class ApproxCorrectGemma3DecoderLayer(nn.Module):
    """Wraps a stock `Gemma3DecoderLayer`, adding `.approx()` and `.correct()`."""

    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.self_attn = layer.self_attn
        self.mlp = layer.mlp
        self.input_layernorm = layer.input_layernorm
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.pre_feedforward_layernorm = layer.pre_feedforward_layernorm
        self.post_feedforward_layernorm = layer.post_feedforward_layernorm
        self.attention_type = getattr(layer, "attention_type", None)
        self._stock = layer

    @classmethod
    def from_stock(cls, layer: nn.Module) -> "ApproxCorrectGemma3DecoderLayer":
        return cls(layer)

    def forward(self, hidden_states, position_embeddings=None, attention_mask=None,
                position_ids=None, **kw):
        return self._stock(hidden_states, position_embeddings=position_embeddings,
                           attention_mask=attention_mask, position_ids=position_ids, **kw)

    # ----------------------------------------------------------------------------------------- #

    def _qkv(self, normed: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """Projections -> QK-norm -> RoPE, in the stock order. K/V stay at kv_heads (GQA)."""
        a = self.self_attn
        shape = (*normed.shape[:-1], -1, a.head_dim)
        q = a.q_proj(normed).view(shape).transpose(1, 2)
        k = a.k_proj(normed).view(shape).transpose(1, 2)
        v = a.v_proj(normed).view(shape).transpose(1, 2)
        q = a.q_norm(q)
        k = a.k_norm(k)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        return q, k, v

    def _attend(self, q, k, v, mask):
        """SDPA with the GQA expansion the stock attention interface performs."""
        a = self.self_attn
        groups = q.shape[1] // k.shape[1]
        k_e, v_e = repeat_kv(k, groups), repeat_kv(v, groups)
        if mask is not None:
            mask = mask[..., : k_e.shape[-2]]
        return torch.nn.functional.scaled_dot_product_attention(
            q, k_e, v_e, attn_mask=mask, dropout_p=0.0, scale=a.scaling
        )

    def _branches(self, x, q, k, v, mask):
        """attn branch + mlp branch, with Gemma 3's sandwich norms. Returns (attn_out, mlp_out)."""
        a = self.self_attn
        attn = self._attend(q, k, v, mask)
        attn = attn.transpose(1, 2).reshape(*x.shape[:-1], -1).contiguous()
        attn_out = self.post_attention_layernorm(a.o_proj(attn))
        x_attn = x + attn_out
        mlp_out = self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(x_attn)))
        return attn_out, mlp_out

    # ----------------------------------------------------------------------------------------- #

    @torch.no_grad()
    def approx(self, x: torch.Tensor, position_embeddings: Tuple[torch.Tensor, torch.Tensor],
               attention_mask: Optional[torch.Tensor], cache_feature: Dict[str, Any], tag: str):
        """Full layer over every token, caching K/V (at kv_heads) and the total increment."""
        cos, sin = position_embeddings
        q, k, v = self._qkv(self.input_layernorm(x), cos, sin)
        attn_out, mlp_out = self._branches(x, q, k, v, attention_mask)

        cache_feature[f"{tag}_k"] = k
        cache_feature[f"{tag}_v"] = v
        cache_feature[f"{tag}_blocks_out_sum"] = attn_out + mlp_out
        cache_feature[f"{tag}_in_sig"] = (float(x.float().mean()), float(x.float().std()),
                                          tuple(x.shape))
        return x + attn_out + mlp_out, cache_feature

    @torch.no_grad()
    def correct(self, x: torch.Tensor, token_mask: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],
                attention_mask: Optional[torch.Tensor],
                cache_feature: Dict[str, Any], tag: str):
        """Recompute the tokens selected by `token_mask` [B, N]; reconstruct the rest."""
        batch = x.shape[0]
        if token_mask.dim() == 1:
            token_mask = token_mask.unsqueeze(0).expand(batch, -1)
        token_mask = token_mask.to(x.device)
        cos, sin = position_embeddings

        k_cache = cache_feature[f"{tag}_k"].clone()
        v_cache = cache_feature[f"{tag}_v"].clone()
        increment = cache_feature[f"{tag}_blocks_out_sum"]
        out = (x + increment.to(x.dtype)).clone()
        new_increment = increment.clone()

        for b in range(batch):
            idx = token_mask[b].nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            x_active = x[b : b + 1, idx]
            # This token's own rotary phase, not the first Q rows of the sequence.
            cos_b = cos[b : b + 1, idx] if cos.dim() == 3 else cos[idx].unsqueeze(0)
            sin_b = sin[b : b + 1, idx] if sin.dim() == 3 else sin[idx].unsqueeze(0)
            q_sel, k_sel, v_sel = self._qkv(self.input_layernorm(x_active), cos_b, sin_b)

            # Fresh K/V for the corrected positions, stale cached K/V everywhere else.
            k_cache[b : b + 1, :, idx, :] = k_sel
            v_cache[b : b + 1, :, idx, :] = v_sel

            # The rows of the mask belonging to these queries.
            m = None
            if attention_mask is not None:
                m = attention_mask[b : b + 1, :, idx, :] if attention_mask.dim() == 4 else attention_mask
            attn_out, mlp_out = self._branches(
                x_active, q_sel, k_cache[b : b + 1], v_cache[b : b + 1], m)

            out[b : b + 1, idx] = (x_active + attn_out + mlp_out).to(out.dtype)
            # Persist the corrected increment, or a LATER round rebuilds these positions from the
            # approximate value and discards what this round fixed. See
            # docs/memo/interleaved_correction_contract.md.
            new_increment[b : b + 1, idx] = (attn_out + mlp_out).to(new_increment.dtype)

        cache_feature[f"{tag}_k"] = k_cache
        cache_feature[f"{tag}_v"] = v_cache
        cache_feature[f"{tag}_blocks_out_sum"] = new_increment
        return out, cache_feature
