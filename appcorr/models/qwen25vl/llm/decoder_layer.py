"""
decoder_layer.py

Forked `transformers.models.qwen2_5_vl.modeling_qwen2_5_vl.{Qwen2_5_VLAttention, Qwen2_5_VLDecoderLayer}`
exposing `.approx()`/`.correct()`, directly adapting
`appcorr/models/openvla/llm/llama_prefill_layer.py`'s pattern (branch
`develop/openvla-progressive-prefill`) to Qwen2.5-VL's causal LLM. Same permanent-query-group
philosophy, same explicit-causal-mask-on-a-non-contiguous-query-set technique, same GQA
(`repeat_kv`) handling -- the one genuinely new piece is **M-RoPE** (3-axis temporal/height/width
rotary position embedding) in place of Llama's plain 1D RoPE:

    - `position_ids` is `[3, B, T]` (not `[B, T]`) -- computed ONCE per request via
      `Qwen2_5_VLModel.get_rope_index(input_ids, mm_token_type_ids, image_grid_thw, ...)` (text-only
      runs get 1D-equivalent positions replicated across all 3 axes; vision-token runs get a
      block-major (t,h,w) spatial layout) -- this needs `input_ids`/`image_grid_thw` only, no vision
      *features*, so it's cheap and can be computed and cached once at request start (analogous to
      how OpenVLA's `self.permanent_group` is computed once in `_finish_setup_after_first_pass`).
    - `Qwen2_5_VLRotaryEmbedding.forward(x, position_ids)` (reused by reference, no fork needed --
      parameter-free) turns `[3,B,T]` position_ids into `cos,sin` of shape `[3,B,T,head_dim]`.
    - `apply_multimodal_rotary_pos_emb` (copied verbatim below -- parameter-free, pure function)
      then splits the channel dim into `mrope_section` chunks and picks one of the 3 axes
      round-robin, producing the single `[B,T,head_dim]` cos/sin actually applied to Q/K via the
      standard rotate-half formula. Like the base rotary computation, this step is parameter-free
      and layer-independent, so it can be hoisted alongside it: `.correct()` accepts pre-gathered,
      pre-interleaved `cos_sel`/`sin_sel` (gathered at `token_idx` from the cached `[3,B,N]` map,
      then already put through the mrope_section interleave) computed ONCE per correction round by
      the caller, not once per layer -- directly extending the Llama fork's RoPE-hoisting
      optimization (there: "the single largest sub-cost... ~0.1ms/layer").
    - `.approx()` intentionally does NOT hoist RoPE (matching the Llama fork's own precedent, which
      only hoists for `.correct()` -- approx() is comparatively rare and correctness/consistency
      with the validated template takes priority over a small extra optimization there).
"""

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def apply_multimodal_rotary_pos_emb(q, k, cos, sin, mrope_section, unsqueeze_dim=1):
    """Copied verbatim from stock (pure function, no learnable params) -- splits the channel dim
    into `mrope_section` (doubled) chunks and interleaves the 3 rope axes round-robin."""
    mrope_section = mrope_section * 2
    cos = torch.cat([m[i % 3] for i, m in enumerate(cos.split(mrope_section, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)
    sin = torch.cat([m[i % 3] for i, m in enumerate(sin.split(mrope_section, dim=-1))], dim=-1).unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class ApproxCorrectQwen25VLAttention(nn.Module):
    def __init__(self, q_proj: nn.Linear, k_proj: nn.Linear, v_proj: nn.Linear, o_proj: nn.Linear,
                 rotary_emb: nn.Module, num_heads: int, num_key_value_heads: int, head_dim: int,
                 mrope_section):
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
        self.mrope_section = list(mrope_section)

    @classmethod
    def from_stock(cls, attn: nn.Module, rotary_emb: nn.Module) -> "ApproxCorrectQwen25VLAttention":
        return cls(
            q_proj=attn.q_proj, k_proj=attn.k_proj, v_proj=attn.v_proj, o_proj=attn.o_proj,
            rotary_emb=rotary_emb, num_heads=attn.num_heads,
            num_key_value_heads=attn.num_key_value_heads, head_dim=attn.head_dim,
            mrope_section=attn.config.rope_parameters["mrope_section"],
        )

    def _project_heads(self, x: torch.Tensor):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        return q, k, v  # [B, H(_kv), T, Dh]

    def approx(self, x: torch.Tensor, position_ids: torch.Tensor, cache_feature: Dict[str, Any], tag: str):
        """Full causal self-attention over all N positions. `position_ids`: [3, B, N], the FULL
        request's M-RoPE position map (computed once via get_rope_index, passed in unchanged)."""
        B, N, _ = x.shape
        q, k, v = self._project_heads(x)

        cos, sin = self.rotary_emb(v, position_ids)  # each [3, B, N, Dh]
        q, k = apply_multimodal_rotary_pos_emb(q, k, cos, sin, self.mrope_section)

        cache_feature[f"{tag}_kv"] = torch.stack([k, v], dim=3)  # [B, H_kv, N, 2, Dh]

        k_full = repeat_kv(k, self.num_key_value_groups)
        v_full = repeat_kv(v, self.num_key_value_groups)
        attn_out = F.scaled_dot_product_attention(q, k_full, v_full, is_causal=True)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)
        return self.o_proj(attn_out), cache_feature

    def correct(
        self,
        x_sel: torch.Tensor,
        token_idx: torch.Tensor,
        cache_feature: Dict[str, Any],
        tag: str,
        cos_sel: torch.Tensor = None,
        sin_sel: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        is_full_causal: bool = None,
        attn_mask: torch.Tensor = None,
    ):
        """
        Args:
            x_sel: [B, Q, C] -- input_layernorm(x) sliced to the query positions (permanent group +
                whichever image merge-groups just arrived -- see the executor/progressive-model
                layer for the permanent-group construction).
            token_idx: [Q] -- absolute positions `x_sel` corresponds to.
            cos_sel, sin_sel: pre-gathered, pre-interleaved (post mrope_section split) RoPE angles
                for `token_idx`, `[1, Q, Dh]` each -- see module docstring. If not provided, computed
                locally from `position_ids[:, :, token_idx]` (standalone/test use).
            is_full_causal, attn_mask: pre-decided/pre-built once per correction round by the
                caller (both are pure functions of `token_idx`/`N`, invariant across every layer in
                the round) -- if not provided, computed locally (standalone/test use). Hoisting these
                avoids a `bool(torch.equal(...))` GPU->CPU sync and an `[Q,N]` mask allocation on
                every one of a round's up-to-64 layer calls.
        """
        kv = cache_feature[f"{tag}_kv"]  # [B, H_kv, N, 2, Dh]
        N = kv.shape[2]
        Q = token_idx.shape[0]
        token_idx = token_idx.to(device=kv.device)

        q_new, k_new, v_new = self._project_heads(x_sel)
        if cos_sel is None or sin_sel is None:
            position_ids_sel = position_ids[:, :, token_idx]
            cos_full, sin_full = self.rotary_emb(v_new, position_ids_sel)
            mrope_section2 = self.mrope_section * 2
            cos_sel = torch.cat([m[i % 3] for i, m in enumerate(cos_full.split(mrope_section2, dim=-1))], dim=-1).unsqueeze(1)
            sin_sel = torch.cat([m[i % 3] for i, m in enumerate(sin_full.split(mrope_section2, dim=-1))], dim=-1).unsqueeze(1)
        q_new = (q_new * cos_sel) + (rotate_half(q_new) * sin_sel)
        k_new = (k_new * cos_sel) + (rotate_half(k_new) * sin_sel)

        kv[:, :, token_idx, 0] = k_new.to(dtype=kv.dtype)
        kv[:, :, token_idx, 1] = v_new.to(dtype=kv.dtype)
        cache_feature[f"{tag}_kv"] = kv

        k_full, v_full = kv.unbind(3)
        k_full = repeat_kv(k_full, self.num_key_value_groups)
        v_full = repeat_kv(v_full, self.num_key_value_groups)

        # When every position is being corrected in this one call (Q == N, token_idx == arange(N)),
        # the explicit allowed-mask below is mathematically identical to a plain causal mask -- but
        # stock/approx() dispatch SDPA via `is_causal=True`, a different fused kernel than the
        # explicit-float-mask path. That kernel mismatch is a real source of bf16 divergence that
        # compounds over many layers and can flip an occasional close-call argmax (confirmed: a
        # single-round, 100%-corrected real run diverged from baseline on 2/20 samples even though
        # this is architecturally a full recomputation). Route the full-coverage case through the
        # same is_causal=True kernel stock uses, for exact kernel-path parity.
        if is_full_causal is None:
            is_full_causal = Q == N and bool(torch.equal(token_idx, torch.arange(N, device=kv.device)))
        if is_full_causal:
            attn_out = F.scaled_dot_product_attention(q_new, k_full, v_full, is_causal=True)
        else:
            if attn_mask is None:
                key_positions = torch.arange(N, device=kv.device).view(1, N)
                allowed = key_positions <= token_idx.view(Q, 1)  # [Q, N]
                attn_mask = torch.zeros((Q, N), device=kv.device, dtype=q_new.dtype)
                attn_mask.masked_fill_(~allowed, torch.finfo(q_new.dtype).min)
                attn_mask = attn_mask.view(1, 1, Q, N)
            attn_out = F.scaled_dot_product_attention(q_new, k_full, v_full, attn_mask=attn_mask)
        attn_out = attn_out.transpose(1, 2).reshape(x_sel.shape[0], Q, self.num_heads * self.head_dim)
        return self.o_proj(attn_out), cache_feature


class ApproxCorrectQwen25VLDecoderLayer(nn.Module):
    def __init__(self, input_layernorm: nn.Module, self_attn: ApproxCorrectQwen25VLAttention,
                 post_attention_layernorm: nn.Module, mlp: nn.Module):
        super().__init__()
        self.input_layernorm = input_layernorm
        self.self_attn = self_attn
        self.post_attention_layernorm = post_attention_layernorm
        self.mlp = mlp

    @classmethod
    def from_stock(cls, layer: nn.Module, rotary_emb: nn.Module) -> "ApproxCorrectQwen25VLDecoderLayer":
        return cls(
            input_layernorm=layer.input_layernorm,
            self_attn=ApproxCorrectQwen25VLAttention.from_stock(layer.self_attn, rotary_emb),
            post_attention_layernorm=layer.post_attention_layernorm,
            mlp=layer.mlp,
        )

    def approx(self, x: torch.Tensor, position_ids: torch.Tensor, cache_feature: Dict[str, Any], tag: str):
        x_attn, cache_feature = self.self_attn.approx(self.input_layernorm(x), position_ids, cache_feature, tag)
        cache_feature[f"{tag}_blocks_out_sum"] = x_attn.detach().clone()

        x_mid = x + x_attn
        mlp_out = self.mlp(self.post_attention_layernorm(x_mid))
        cache_feature[f"{tag}_blocks_out_sum"] = cache_feature[f"{tag}_blocks_out_sum"] + mlp_out.detach()

        return x_mid + mlp_out, cache_feature

    def correct(
        self,
        x: torch.Tensor,
        token_idx: torch.Tensor,
        cache_feature: Dict[str, Any],
        tag: str,
        cos_sel: torch.Tensor = None,
        sin_sel: torch.Tensor = None,
        position_ids: torch.Tensor = None,
        is_full_causal: bool = None,
        attn_mask: torch.Tensor = None,
    ):
        token_idx = token_idx.to(x.device)
        x_active = x[:, token_idx]
        x_norm_sel = self.input_layernorm(x_active)

        x_attn_sel, cache_feature = self.self_attn.correct(
            x_norm_sel, token_idx, cache_feature, tag, cos_sel=cos_sel, sin_sel=sin_sel, position_ids=position_ids,
            is_full_causal=is_full_causal, attn_mask=attn_mask,
        )
        x_attn_active = x_active + x_attn_sel
        mlp_out_new = self.mlp(self.post_attention_layernorm(x_attn_active))

        blocks_out_sum = cache_feature[f"{tag}_blocks_out_sum"]
        residual = blocks_out_sum.to(dtype=x.dtype) if blocks_out_sum.dtype != x.dtype else blocks_out_sum
        x_out = x + residual
        x_out[:, token_idx] = (x_attn_active + mlp_out_new).to(dtype=x_out.dtype)

        return x_out, cache_feature
