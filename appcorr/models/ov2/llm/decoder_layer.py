"""Qwen3 decoder layer with an approx/correct split — the LLM half of the OV2 unified axis.

Ported from the Gemma 3 fork, which shares the delicate parts (QK-norm before RoPE, GQA, per-token
rotary phase, query-axis mask slicing). Two things are genuinely different and both simplify:

**Two layernorms, plain pre-norm.** Qwen3 adds the branch output RAW:

    x = x + attn(input_layernorm(x))
    x = x + mlp(post_attention_layernorm(x))

Gemma 3 instead norms each branch output before the residual (a four-norm sandwich). Copying the
Gemma form here would rescale every correction while leaving all shapes valid.

**Attention is causal everywhere.** Gemma 3 makes image tokens bidirectional among themselves and
text causal, so its mask rows are heterogeneous. Here every row is causal, but the mask still has to
be sliced on the QUERY axis: correcting query `i` needs row `i`, and a selection is generally not a
contiguous prefix. `is_causal=True` cannot be used instead -- SDPA would build a mask for a fresh
0..Q range, which is only correct when the selected tokens happen to be the first Q of the sequence.

Unchanged from the Gemma fork, and still the easy mistakes:

**RoPE is per position.** A corrected token needs ITS OWN cos/sin, sliced to the selected indices.
Feeding the first Q rows is finite, plausible, and wrong everywhere the selection is not a prefix.

**GQA: 32 query heads over 8 KV heads.** K/V are cached at 8 heads and expanded at use, so the cache
stays small; the expansion must happen on both paths or the head mapping silently differs.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv


class ApproxCorrectQwen3DecoderLayer(nn.Module):
    """Wraps a stock `Qwen3DecoderLayer`, adding `.approx()` and `.correct()`."""

    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        self.self_attn = layer.self_attn
        self.mlp = layer.mlp
        self.input_layernorm = layer.input_layernorm
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.attention_type = getattr(layer, "attention_type", None)
        self._stock = layer

    @classmethod
    def from_stock(cls, layer: nn.Module) -> "ApproxCorrectQwen3DecoderLayer":
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
        q = a.q_norm(a.q_proj(normed).view(shape)).transpose(1, 2)
        k = a.k_norm(a.k_proj(normed).view(shape)).transpose(1, 2)
        v = a.v_proj(normed).view(shape).transpose(1, 2)
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
            q, k_e, v_e, attn_mask=mask, dropout_p=0.0, scale=a.scaling)

    def _branches(self, x, q, k, v, mask):
        """attn branch + mlp branch, plain pre-norm. Returns (attn_out, mlp_out)."""
        attn = self._attend(q, k, v, mask)
        attn = attn.transpose(1, 2).reshape(*x.shape[:-1], -1).contiguous()
        attn_out = self.self_attn.o_proj(attn)
        x_attn = x + attn_out
        mlp_out = self.mlp(self.post_attention_layernorm(x_attn))
        return attn_out, mlp_out

    # ----------------------------------------------------------------------------------------- #

    @torch.no_grad()
    def approx(self, x: torch.Tensor, position_embeddings: Tuple[torch.Tensor, torch.Tensor],
               attention_mask: Optional[torch.Tensor], cache_feature: Dict[str, Any], tag: str):
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
            cos_b = cos[b : b + 1, idx] if cos.dim() == 3 else cos[idx].unsqueeze(0)
            sin_b = sin[b : b + 1, idx] if sin.dim() == 3 else sin[idx].unsqueeze(0)
            q_sel, k_sel, v_sel = self._qkv(self.input_layernorm(x_active), cos_b, sin_b)

            k_cache[b : b + 1, :, idx, :] = k_sel
            v_cache[b : b + 1, :, idx, :] = v_sel

            m = None
            if attention_mask is not None:
                m = (attention_mask[b : b + 1, :, idx, :] if attention_mask.dim() == 4
                     else attention_mask)
            attn_out, mlp_out = self._branches(
                x_active, q_sel, k_cache[b : b + 1], v_cache[b : b + 1], m)

            out[b : b + 1, idx] = (x_active + attn_out + mlp_out).to(out.dtype)
            new_increment[b : b + 1, idx] = (attn_out + mlp_out).to(new_increment.dtype)

        cache_feature[f"{tag}_k"] = k_cache
        cache_feature[f"{tag}_v"] = v_cache
        cache_feature[f"{tag}_blocks_out_sum"] = new_increment
        return out, cache_feature
