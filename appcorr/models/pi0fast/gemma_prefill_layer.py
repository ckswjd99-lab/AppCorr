"""
gemma_prefill_layer.py

Gemma variant of the causal LLM approx/correct/prefill fork (cf.
appcorr/models/openvla/llm/llama_prefill_layer.py), for the PaliGemma language model inside
pi0-FAST. The attention math is identical to Llama's (rotate-half RoPE, GQA repeat_kv, SDPA); the
differences are plumbing, all handled here or by the wrapped stock modules:

  - transformers>=4.5x attention API: RoPE cos/sin are computed once at the *model* level
    (GemmaRotaryEmbedding) and threaded in as `cos,sin` -- there is no per-attention `rotary_emb`.
  - head_dim is `config.head_dim` (256 for Gemma-2B), not hidden/num_heads; num_heads /
    num_key_value_heads are derived from the q/k projection shapes.
  - Gemma RMSNorm uses (1 + weight) and GeGLU MLP -- both live in the wrapped stock
    input_layernorm / post_attention_layernorm / mlp modules, so `.prefill()`/`.correct()` call them
    unchanged.
  - Embedding scaling by sqrt(hidden_size) is applied once when building the input embeddings
    (model level), not in these layers.

Supports `causal=False` (bidirectional) for parity with the OpenVLA fork, though pi-FAST's action
tokens are generated autoregressively (causal); the prefix prefill is the chunked-prefill target.
"""

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.gemma.modeling_gemma import apply_rotary_pos_emb, repeat_kv


class ApproxCorrectGemmaAttention(nn.Module):
    def __init__(self, q_proj, k_proj, v_proj, o_proj, num_heads, num_key_value_heads, head_dim, scaling):
        super().__init__()
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.o_proj = o_proj
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.scaling = scaling
        self.num_key_value_groups = num_heads // num_key_value_heads

    @classmethod
    def from_stock(cls, attn: nn.Module) -> "ApproxCorrectGemmaAttention":
        head_dim = attn.head_dim
        num_heads = attn.q_proj.out_features // head_dim
        num_kv_heads = attn.k_proj.out_features // head_dim
        scaling = getattr(attn, "scaling", head_dim ** -0.5)
        return cls(attn.q_proj, attn.k_proj, attn.v_proj, attn.o_proj,
                   num_heads, num_kv_heads, head_dim, scaling)

    def _project_heads(self, x: torch.Tensor):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    def approx(self, x: torch.Tensor, cache_feature: Dict[str, Any], tag: str,
               cos: torch.Tensor, sin: torch.Tensor):
        """Full causal prefill over all N positions; caches post-RoPE K/V (pre-repeat_kv)."""
        B, N, _ = x.shape
        q, k, v = self._project_heads(x)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        cache_feature[f"{tag}_kv"] = torch.stack([k, v], dim=3)  # [B, H_kv, N, 2, Dh]
        k_full = repeat_kv(k, self.num_key_value_groups)
        v_full = repeat_kv(v, self.num_key_value_groups)
        attn_out = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=True, scale=self.scaling)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)
        return self.o_proj(attn_out), cache_feature

    def correct(self, x_sel: torch.Tensor, token_idx: torch.Tensor, cache_feature: Dict[str, Any], tag: str,
                cos: torch.Tensor, sin: torch.Tensor, key_end: int = None, causal: bool = True):
        """Recompute Q/K/V for `token_idx`, splice K/V into the cache, attend against the (optionally
        prefix-bounded) cache. `causal=False` => bidirectional over the reachable keys."""
        kv = cache_feature[f"{tag}_kv"]  # [B, H_kv, N, 2, Dh]
        B, _, N = kv.shape[0], kv.shape[1], kv.shape[2]
        Q = token_idx.shape[0]
        token_idx = token_idx.to(kv.device)

        q_new, k_new, v_new = self._project_heads(x_sel)
        q_new, k_new = apply_rotary_pos_emb(q_new, k_new, cos, sin)

        kv[:, :, token_idx, 0] = k_new.to(dtype=kv.dtype)
        kv[:, :, token_idx, 1] = v_new.to(dtype=kv.dtype)
        cache_feature[f"{tag}_kv"] = kv

        n_keys = N if key_end is None else min(int(key_end), N)
        k_full, v_full = kv[:, :, :n_keys].unbind(3)
        k_full = repeat_kv(k_full, self.num_key_value_groups)
        v_full = repeat_kv(v_full, self.num_key_value_groups)

        if causal:
            key_positions = torch.arange(n_keys, device=kv.device).view(1, n_keys)
            allowed = key_positions <= token_idx.view(Q, 1)
            attn_mask = torch.zeros((Q, n_keys), device=kv.device, dtype=q_new.dtype)
            attn_mask.masked_fill_(~allowed, torch.finfo(q_new.dtype).min)
            attn_mask = attn_mask.view(1, 1, Q, n_keys)
        else:
            attn_mask = None

        attn_out = F.scaled_dot_product_attention(q_new, k_full, v_full, attn_mask=attn_mask, scale=self.scaling)
        attn_out = attn_out.transpose(1, 2).reshape(B, Q, self.num_heads * self.head_dim)
        return self.o_proj(attn_out), cache_feature


class ApproxCorrectGemmaDecoderLayer(nn.Module):
    def __init__(self, input_layernorm, self_attn, post_attention_layernorm, mlp):
        super().__init__()
        self.input_layernorm = input_layernorm
        self.self_attn = self_attn
        self.post_attention_layernorm = post_attention_layernorm
        self.mlp = mlp

    @classmethod
    def from_stock(cls, layer: nn.Module) -> "ApproxCorrectGemmaDecoderLayer":
        return cls(layer.input_layernorm, ApproxCorrectGemmaAttention.from_stock(layer.self_attn),
                   layer.post_attention_layernorm, layer.mlp)

    def approx(self, x, cache_feature, tag, cos, sin):
        x_attn, cache_feature = self.self_attn.approx(self.input_layernorm(x), cache_feature, tag, cos, sin)
        cache_feature[f"{tag}_blocks_out_sum"] = x_attn.detach().clone()
        x_mid = x + x_attn
        mlp_out = self.mlp(self.post_attention_layernorm(x_mid))
        cache_feature[f"{tag}_blocks_out_sum"] = cache_feature[f"{tag}_blocks_out_sum"] + mlp_out.detach()
        return x_mid + mlp_out, cache_feature

    def prefill(self, x_sel, token_idx, cache_feature, tag, cos, sin, key_end=None, causal=True):
        """O(Q) chunked prefill: operate only on the Q query rows (no [B,N,C] reconstruction)."""
        x_norm_sel = self.input_layernorm(x_sel)
        x_attn_sel, cache_feature = self.self_attn.correct(
            x_norm_sel, token_idx, cache_feature, tag, cos, sin, key_end=key_end, causal=causal)
        x_attn_active = x_sel + x_attn_sel
        mlp_out_new = self.mlp(self.post_attention_layernorm(x_attn_active))
        return x_attn_active + mlp_out_new, cache_feature
