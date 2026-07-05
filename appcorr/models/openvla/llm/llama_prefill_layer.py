"""
llama_prefill_layer.py

Forked `transformers.models.llama.modeling_llama.{LlamaAttention (sdpa), LlamaDecoderLayer}` exposing
`.approx()` / `.correct()`, extending the same partial_token pattern used for the vision towers
(`appcorr/models/openvla/vision/{attention,block}.py`) to a **causal** decoder -- the novel piece of
the progressive-VLA-prefill plan (see /home/nxclab/.claude/plans/async-stargazing-mango.md).

Two differences from the (bidirectional, no-rope) vision-tower fork:

1. **RoPE**: Llama rotates Q/K by each token's *absolute* position before caching/attending. When
   correcting a token subset, we recompute Q/K from that subset's current hidden state and rotate
   using *their own* absolute positions (`token_idx`), matching what a full forward would produce.

2. **Causal masking on a non-contiguous query set**: unlike the vision towers' full bidirectional
   attention, a corrected/queried position `i` may only attend to cached keys at positions `<= i`.
   `.correct()` builds an explicit `[Q, N]` boolean mask per call rather than relying on `is_causal`
   (which assumes a contiguous, sequence-prefix query block).

The corrected/query set is **not** the same as the vision towers' (where corrected == queried).
OpenVLA's multimodal sequence is `[BOS, 256 vision tokens, ~30 text tokens]`; only the vision-token
subset actually receives new pixel data as it progressively arrives. But under causal masking, only
*that* token's own attention output benefits from an isolated K/V patch -- every token after it also
needs its output (and K/V) refreshed to see the new information, or the corrected data never reaches
the action-relevant final hidden state. Rather than recomputing every downstream position (which would
approach the cost of a full forward), we designate a **permanent query group** -- BOS + the entire
text-instruction suffix -- that is *always* included in the corrected/queried set on every round,
regardless of which vision patches changed. This guarantees the text span's (and hence the final
pre-decode position's) hidden state and K/V are always fresh, at the cost of not being exact unless
100% of vision patches are also corrected (exactly mirrors the vision towers' finding that DINOv2's
prefix tokens needed the same permanent treatment for exactness -- see Phase 1's validation notes).
"""

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv


class ApproxCorrectLlamaAttention(nn.Module):
    def __init__(self, q_proj: nn.Linear, k_proj: nn.Linear, v_proj: nn.Linear, o_proj: nn.Linear,
                 rotary_emb: nn.Module, num_heads: int, num_key_value_heads: int, head_dim: int):
        super().__init__()
        self.q_proj = q_proj
        self.k_proj = k_proj
        self.v_proj = v_proj
        self.o_proj = o_proj
        self.rotary_emb = rotary_emb
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.num_key_value_groups = num_heads // num_key_value_heads

    @classmethod
    def from_stock(cls, attn: nn.Module) -> "ApproxCorrectLlamaAttention":
        return cls(
            q_proj=attn.q_proj, k_proj=attn.k_proj, v_proj=attn.v_proj, o_proj=attn.o_proj,
            rotary_emb=attn.rotary_emb, num_heads=attn.num_heads,
            num_key_value_heads=attn.num_key_value_heads, head_dim=attn.head_dim,
        )

    def _project_heads(self, x: torch.Tensor):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        return q, k, v  # [B, H(_kv), T, Dh]

    def approx(self, x: torch.Tensor, cache_feature: Dict[str, Any], tag: str):
        """Full causal self-attention over all N positions (standard prefill); caches post-RoPE K/V
        (pre-repeat_kv, i.e. at `num_key_value_heads`, matching real KV-cache memory semantics) for
        `.correct()` to patch."""
        B, N, _ = x.shape
        q, k, v = self._project_heads(x)

        position_ids = torch.arange(N, device=x.device, dtype=torch.long).unsqueeze(0).expand(B, -1)
        cos, sin = self.rotary_emb(v, position_ids)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        cache_feature[f"{tag}_kv"] = torch.stack([k, v], dim=3)  # [B, H_kv, N, 2, Dh]

        k_full = repeat_kv(k, self.num_key_value_groups)
        v_full = repeat_kv(v, self.num_key_value_groups)
        attn_out = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=True)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)
        return self.o_proj(attn_out), cache_feature

    def correct(self, x_sel: torch.Tensor, token_idx: torch.Tensor, cache_feature: Dict[str, Any], tag: str):
        """
        Args:
            x_sel: [B, Q, C] -- input_layernorm(x) already sliced to the query positions (a superset
                of the corrected vision patches -- see module docstring for the permanent group).
            token_idx: [Q] -- absolute positions `x_sel` corresponds to (used for both RoPE angles
                and the causal mask against the full N-length cache).
        """
        kv = cache_feature[f"{tag}_kv"]  # [B, H_kv, N, 2, Dh]
        B, _, N = kv.shape[0], kv.shape[1], kv.shape[2]
        Q = token_idx.shape[0]
        token_idx = token_idx.to(device=kv.device)

        q_new, k_new, v_new = self._project_heads(x_sel)
        position_ids_sel = token_idx.unsqueeze(0).expand(B, -1)
        cos, sin = self.rotary_emb(v_new, position_ids_sel)
        q_new, k_new = apply_rotary_pos_emb(q_new, k_new, cos, sin)

        kv[:, :, token_idx, 0] = k_new.to(dtype=kv.dtype)
        kv[:, :, token_idx, 1] = v_new.to(dtype=kv.dtype)
        cache_feature[f"{tag}_kv"] = kv

        k_full, v_full = kv.unbind(3)  # each [B, H_kv, N, Dh]
        k_full = repeat_kv(k_full, self.num_key_value_groups)
        v_full = repeat_kv(v_full, self.num_key_value_groups)

        # Causal mask restricted to this (possibly non-contiguous) query set: query i may attend to
        # key j iff j <= token_idx[i]. Shape [1, 1, Q, N] broadcasts over batch/heads.
        key_positions = torch.arange(N, device=kv.device).view(1, N)
        allowed = key_positions <= token_idx.view(Q, 1)  # [Q, N]
        attn_mask = torch.zeros((Q, N), device=kv.device, dtype=q_new.dtype)
        attn_mask.masked_fill_(~allowed, torch.finfo(q_new.dtype).min)
        attn_mask = attn_mask.view(1, 1, Q, N)

        attn_out = F.scaled_dot_product_attention(q_new, k_full, v_full, attn_mask=attn_mask)
        attn_out = attn_out.transpose(1, 2).reshape(B, Q, self.num_heads * self.head_dim)
        return self.o_proj(attn_out), cache_feature


class ApproxCorrectLlamaDecoderLayer(nn.Module):
    def __init__(self, input_layernorm: nn.Module, self_attn: ApproxCorrectLlamaAttention,
                 post_attention_layernorm: nn.Module, mlp: nn.Module):
        super().__init__()
        self.input_layernorm = input_layernorm
        self.self_attn = self_attn
        self.post_attention_layernorm = post_attention_layernorm
        self.mlp = mlp

    @classmethod
    def from_stock(cls, layer: nn.Module) -> "ApproxCorrectLlamaDecoderLayer":
        return cls(
            input_layernorm=layer.input_layernorm,
            self_attn=ApproxCorrectLlamaAttention.from_stock(layer.self_attn),
            post_attention_layernorm=layer.post_attention_layernorm,
            mlp=layer.mlp,
        )

    def approx(self, x: torch.Tensor, cache_feature: Dict[str, Any], tag: str):
        x_attn, cache_feature = self.self_attn.approx(self.input_layernorm(x), cache_feature, tag)
        cache_feature[f"{tag}_blocks_out_sum"] = x_attn.detach().clone()

        x_mid = x + x_attn
        mlp_out = self.mlp(self.post_attention_layernorm(x_mid))
        cache_feature[f"{tag}_blocks_out_sum"] = cache_feature[f"{tag}_blocks_out_sum"] + mlp_out.detach()

        return x_mid + mlp_out, cache_feature

    def correct(self, x: torch.Tensor, token_idx: torch.Tensor, cache_feature: Dict[str, Any], tag: str):
        token_idx = token_idx.to(x.device)
        x_active = x[:, token_idx]
        x_norm_sel = self.input_layernorm(x_active)

        x_attn_sel, cache_feature = self.self_attn.correct(x_norm_sel, token_idx, cache_feature, tag)
        x_attn_active = x_active + x_attn_sel
        mlp_out_new = self.mlp(self.post_attention_layernorm(x_attn_active))

        blocks_out_sum = cache_feature[f"{tag}_blocks_out_sum"]
        x_out = (x + blocks_out_sum.to(dtype=x.dtype)).clone()
        x_out[:, token_idx] = (x_attn_active + mlp_out_new).to(dtype=x_out.dtype)

        return x_out, cache_feature
