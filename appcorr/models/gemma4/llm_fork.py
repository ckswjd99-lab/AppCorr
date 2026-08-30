"""Gemma 4 text decoder layer with an approx/correct split -- port-plan step 4's LLM half.

Follows `appcorr/models/gemma3/llm/decoder_layer.py` (the closest relative: same sandwich norms,
same QK-norm-then-RoPE order, same mask-row slicing requirement) with the Gemma4-specific
differences, each verified against `modeling_gemma4.py` source rather than assumed:

  * **RoPE before transpose** (`apply_rotary_pos_emb(..., unsqueeze_dim=2)` on [B, T, H, D]),
    where Gemma3 rotated after the transpose. Copying Gemma3's order runs fine and rotates
    every head wrong.
  * **Two attention variants by layer type.** Sliding layers (50/60): standard GQA, head_dim 256,
    16 KV heads, window+block-bidir mask. Full layers (10/60): `attention_k_eq_v` "alternative
    attention" -- there is NO v_proj; V = v_norm(RAW k_proj output, pre-k_norm, un-roped),
    head_dim 512 (global_head_dim), 4 KV heads, strictly causal mask. The variant is a property
    of the wrapped module (v_proj is None), so the fork branches on that, not on config flags.
  * **`layer_scalar`** multiplies the WHOLE layer output (residual included), so the rule-3 cache
    is the layer DELTA (out - in), not the branch sum: reconstruction of an untouched row is
    x_in + delta, exact because untouched rows re-enter with the same x_in every round.
  * 31B has MoE blocks, per-layer inputs, and KV sharing all OFF -- asserted at wrap time so a
    future checkpoint that enables them fails loudly instead of silently skipping their math.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from torch import nn

from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb, repeat_kv


class ApproxCorrectGemma4TextLayer(nn.Module):
    """Wraps a stock `Gemma4TextDecoderLayer`, adding `.approx()` and `.correct()`."""

    def __init__(self, layer: nn.Module) -> None:
        super().__init__()
        assert not getattr(layer, "enable_moe_block", False), "MoE block active -- fork not built for it"
        assert not getattr(layer, "hidden_size_per_layer_input", 0), "PLE active -- fork not built for it"
        assert not layer.self_attn.is_kv_shared_layer, "KV-shared layer -- fork not built for it"
        self.self_attn = layer.self_attn
        self.mlp = layer.mlp
        self.input_layernorm = layer.input_layernorm
        self.post_attention_layernorm = layer.post_attention_layernorm
        self.pre_feedforward_layernorm = layer.pre_feedforward_layernorm
        self.post_feedforward_layernorm = layer.post_feedforward_layernorm
        self.register_buffer("layer_scalar", layer.layer_scalar, persistent=False)
        self.layer_type = layer.self_attn.layer_type

    @classmethod
    def from_stock(cls, layer: nn.Module) -> "ApproxCorrectGemma4TextLayer":
        return cls(layer)

    # ----------------------------------------------------------------------------------------- #

    def _qkv(self, normed: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """Stock order: project -> (q_norm | k_norm) -> RoPE at unsqueeze_dim=2 -> transpose.
        V: v_proj if present, else the RAW k_proj output (k_eq_v layers); v_norm either way,
        never roped. Returns q,k,v as [B, H(_kv), T, D]."""
        a = self.self_attn
        hidden_shape = (*normed.shape[:-1], -1, a.head_dim)
        q = a.q_proj(normed).view(hidden_shape)
        q = a.q_norm(q)
        q = apply_rotary_pos_emb(q, cos, sin, unsqueeze_dim=2).transpose(1, 2)

        k_raw = a.k_proj(normed).view(hidden_shape)
        v_raw = a.v_proj(normed).view(hidden_shape) if a.v_proj is not None else k_raw
        k = a.k_norm(k_raw)
        k = apply_rotary_pos_emb(k, cos, sin, unsqueeze_dim=2).transpose(1, 2)
        v = a.v_norm(v_raw).transpose(1, 2)
        return q, k, v

    def _attend(self, q, k, v, mask, is_causal=False):
        """mask=None does NOT mean "no mask": the transformers mask builder returns None for
        strictly-causal layer types (Gemma4's 10 full-attention layers) and stock SDPA then
        applies causality via the module's is_causal FLAG. Treating None as bidirectional was
        the walk gate's original failure (q/k/v bitwise equal, attention off by 50)."""
        a = self.self_attn
        groups = q.shape[1] // k.shape[1]
        k_e, v_e = repeat_kv(k, groups), repeat_kv(v, groups)
        if mask is not None:
            mask = mask[..., : k_e.shape[-2]]
        return torch.nn.functional.scaled_dot_product_attention(
            q, k_e, v_e, attn_mask=mask, dropout_p=0.0, scale=a.scaling,
            is_causal=is_causal and mask is None)

    def _layer_out(self, x, q, k, v, mask, is_causal=False):
        """Sandwich-norm branches + layer_scalar, exactly the stock forward's arithmetic."""
        attn = self._attend(q, k, v, mask, is_causal=is_causal)
        attn = attn.transpose(1, 2).reshape(*x.shape[:-1], -1).contiguous()
        attn_out = self.post_attention_layernorm(self.self_attn.o_proj(attn))
        x_attn = x + attn_out
        mlp_out = self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(x_attn)))
        return (x_attn + mlp_out) * self.layer_scalar

    # ----------------------------------------------------------------------------------------- #

    @torch.no_grad()
    def approx(self, x: torch.Tensor, position_embeddings: Tuple[torch.Tensor, torch.Tensor],
               attention_mask: Optional[torch.Tensor], cache_feature: Dict[str, Any], tag: str):
        cos, sin = position_embeddings
        q, k, v = self._qkv(self.input_layernorm(x), cos, sin)
        out = self._layer_out(x, q, k, v, attention_mask,
                              is_causal=(attention_mask is None and self.self_attn.is_causal))
        cache_feature[f"{tag}_k"] = k
        cache_feature[f"{tag}_v"] = v
        # Layer DELTA, not branch sum: layer_scalar scales the whole output, so out-in is the
        # only increment whose "x + delta" reconstruction is exact for untouched rows.
        cache_feature[f"{tag}_delta"] = out - x
        cache_feature[f"{tag}_in_sig"] = (float(x.float().mean()), float(x.float().std()),
                                          tuple(x.shape))
        return out, cache_feature

    @torch.no_grad()
    def correct(self, x: torch.Tensor, token_idx: torch.Tensor,
                position_embeddings: Tuple[torch.Tensor, torch.Tensor],
                attention_mask: Optional[torch.Tensor],
                cache_feature: Dict[str, Any], tag: str):
        """Recompute rows `token_idx` [Q] (B=1) against cached K/V; rule-3 persist the delta."""
        assert x.shape[0] == 1, "fork is B=1 by design (matches the vision fork)"
        cos, sin = position_embeddings
        token_idx = token_idx.to(x.device)

        k_cache = cache_feature[f"{tag}_k"]
        v_cache = cache_feature[f"{tag}_v"]
        delta = cache_feature[f"{tag}_delta"]

        x_active = x[:, token_idx]
        cos_sel = cos[:, token_idx] if cos.dim() == 3 else cos[token_idx].unsqueeze(0)
        sin_sel = sin[:, token_idx] if sin.dim() == 3 else sin[token_idx].unsqueeze(0)
        q_sel, k_sel, v_sel = self._qkv(self.input_layernorm(x_active), cos_sel, sin_sel)

        k_cache = k_cache.clone(); v_cache = v_cache.clone()
        k_cache[:, :, token_idx] = k_sel
        v_cache[:, :, token_idx] = v_sel
        cache_feature[f"{tag}_k"] = k_cache
        cache_feature[f"{tag}_v"] = v_cache

        if attention_mask is not None:
            # Query-axis rows for exactly these tokens -- Gemma4's masks are position-specific
            # (block-bidir islands on sliding layers), so rows are not interchangeable.
            m = attention_mask[:, :, token_idx, :]
        elif self.self_attn.is_causal:
            # None-mask = strictly-causal layer (see _attend). A row subset cannot ride SDPA's
            # is_causal flag (that assumes square alignment), so build the causal rows explicitly.
            N = x.shape[1]
            key_pos = torch.arange(N, device=x.device).view(1, N)
            allowed = key_pos <= token_idx.view(-1, 1)
            m = torch.zeros((token_idx.numel(), N), device=x.device, dtype=x.dtype)
            m.masked_fill_(~allowed, torch.finfo(x.dtype).min)
            m = m.view(1, 1, token_idx.numel(), N)
        else:
            m = None
        out_active = self._layer_out(x_active, q_sel, k_cache, v_cache, m)

        out = (x + delta.to(x.dtype)).clone()
        out[:, token_idx] = out_active.to(out.dtype)
        new_delta = delta.clone()
        new_delta[:, token_idx] = (out_active - x_active).to(new_delta.dtype)
        cache_feature[f"{tag}_delta"] = new_delta
        return out, cache_feature
