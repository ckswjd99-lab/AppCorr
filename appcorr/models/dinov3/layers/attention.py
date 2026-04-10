# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from ._triton_kernels import apply_rope_partial_triton
from ..utils import cat_keep_shapes, uncat_with_shapes


# RoPE-related functions:
def rope_rotate_half(x: Tensor) -> Tensor:
    # x:   [ x0  x1  x2  x3  x4  x5]
    # out: [-x3 -x4 -x5  x0  x1  x2]
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def rope_apply(x: Tensor, sin: Tensor, cos: Tensor) -> Tensor:
    # x:   [..., D], eg [x0,     x1,   x2,   x3,   x4,   x5]
    # sin: [..., D], eg [sin0, sin1, sin2, sin0, sin1, sin2]
    # cos: [..., D], eg [cos0, cos1, cos2, cos0, cos1, cos2]
    return (x * cos) + (rope_rotate_half(x) * sin)


class LinearKMaskedBias(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        o = self.out_features
        assert o % 3 == 0
        if self.bias is not None:
            self.register_buffer("bias_mask", torch.full_like(self.bias, fill_value=math.nan))

    def forward(self, input: Tensor) -> Tensor:
        masked_bias = self.bias * self.bias_mask.to(self.bias.dtype) if self.bias is not None else None
        return F.linear(input, self.weight, masked_bias)


class SelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        mask_k_bias: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        linear_class = LinearKMaskedBias if mask_k_bias else nn.Linear
        self.qkv = linear_class(dim, dim * 3, bias=qkv_bias, device=device)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias, device=device)
        self.proj_drop = nn.Dropout(proj_drop)

    def apply_rope(self, q: Tensor, k: Tensor, rope: Tensor | Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        # All operations will use the dtype of rope, the output is cast back to the dtype of q and k
        q_dtype = q.dtype
        k_dtype = k.dtype
        sin, cos = rope
        rope_dtype = sin.dtype
        q = q.to(dtype=rope_dtype)
        k = k.to(dtype=rope_dtype)
        N = q.shape[-2]
        prefix = N - sin.shape[-2]
        assert prefix >= 0
        q_prefix = q[:, :, :prefix, :]
        q = rope_apply(q[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        q = torch.cat((q_prefix, q), dim=-2)  # [B, head, N, D//head]
        k_prefix = k[:, :, :prefix, :]
        k = rope_apply(k[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        k = torch.cat((k_prefix, k), dim=-2)  # [B, head, N, D//head]
        q = q.to(dtype=q_dtype)
        k = k.to(dtype=k_dtype)
        return q, k

    def forward(self, x: Tensor, attn_bias=None, rope: Tensor = None) -> Tensor:
        qkv = self.qkv(x)
        attn_v = self.compute_attention(qkv=qkv, attn_bias=attn_bias, rope=rope)
        x = self.proj(attn_v)
        x = self.proj_drop(x)
        return x

    def forward_list(self, x_list, attn_bias=None, rope_list=None) -> List[Tensor]:
        assert len(x_list) == len(rope_list)  # should be enforced by the Block
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        qkv_flat = self.qkv(x_flat)
        qkv_list = uncat_with_shapes(qkv_flat, shapes, num_tokens)
        att_out = []
        for _, (qkv, _, rope) in enumerate(zip(qkv_list, shapes, rope_list)):
            att_out.append(self.compute_attention(qkv, attn_bias=attn_bias, rope=rope))
        x_flat, shapes, num_tokens = cat_keep_shapes(att_out)
        x_flat = self.proj(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)

    def compute_attention(
        self, qkv: Tensor, attn_bias=None, rope=None,
    ) -> Tensor:
        assert attn_bias is None
        B, N, _ = qkv.shape
        C = self.qkv.in_features

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        
        # RoPE
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)

        # Attention
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)

        # Reshape back
        x = x.transpose(1, 2)
        return x.reshape([B, N, C])

    def compute_attention_partial_token(
        self,
        qkv: Tensor,
        rope: Tensor,
        cache_feature: Dict,
        tag: str,
        *,
        server_pscore: str = "cls_attn_prob",
    ) -> Tensor:
        B, N, _ = qkv.shape
        C = self.qkv.in_features

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        cache_feature[f"{tag}_kv"] = qkv[:, :, 1:].detach().clone()  # Only k, v, [B, N, 2, H, D//H]
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]

        if rope is not None:
            q, k = self.apply_rope(q, k, rope)

        cache_feature[f"{tag}_kv"][:, :, 0] = k.detach().transpose(1, 2)
        if server_pscore == "patch_attn_prob":
            attn_prob = (q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1)  # [B, H, N, N]
            server_pscore_tensor = attn_prob.mean(dim=1).mean(dim=1).to(dtype=torch.bfloat16)  # [B, N]
        else:
            cls_attn_score = q[:, :, 0:1, :] @ k.transpose(-2, -1) * self.scale
            server_pscore_tensor = cls_attn_score.softmax(-1).mean(dim=1).squeeze(1).to(dtype=torch.float32)
        cache_feature[f"{tag}_server_pscore"] = server_pscore_tensor.detach()

        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2)
        return x.reshape([B, N, C])

    def approx(self, x: Tensor, rope: Tensor, cache_feature: Dict, tag: str, **kwargs) -> Tuple[Tensor, dict]:
        appcorr_method = kwargs.get("appcorr_method", "partial_token")
        if appcorr_method == "partial_token":
            return self.approx_partial_token(
                x,
                rope,
                cache_feature,
                tag,
                server_pscore=kwargs.get("server_pscore", "cls_attn_prob"),
            )
        if appcorr_method == "partial_channel":
            return self.approx_partial_channel(
                x,
                rope,
                cache_feature,
                tag,
                attn_cache_candidates=kwargs.get("attn_cache_candidates"),
                attn_col_alive_ratio=kwargs.get("attn_col_alive_ratio", 1.0),
            )
        raise ValueError(
            f"Unknown SelfAttention.approx method '{appcorr_method}'. "
            "Available methods: partial_channel, partial_token"
        )
    
    def correct(
        self,
        x_sel: Tensor,
        dindice: Tensor,
        rope: Tensor,
        cache_feature: Dict,
        tag: str,
        **kwargs,
    ) -> Tuple[Tensor, dict]:
        appcorr_method = kwargs.get("appcorr_method", "partial_token")
        if appcorr_method == "partial_token":
            return self.correct_partial_token(x_sel, dindice, rope, cache_feature, tag)
        if appcorr_method == "partial_channel":
            return self.correct_partial_channel(x_sel, dindice, rope, cache_feature, tag, **kwargs)
        raise ValueError(
            f"Unknown SelfAttention.correct method '{appcorr_method}'. "
            "Available methods: partial_channel, partial_token"
        )

    def approx_partial_token(
        self,
        x: Tensor,
        rope: Tensor,
        cache_feature: Dict,
        tag: str,
        *,
        server_pscore: str = "cls_attn_prob",
    ) -> Tuple[Tensor, dict]:
        qkv = self.qkv(x)   # [B, N, 3*D]

        attn_v = self.compute_attention_partial_token(
            qkv=qkv,
            rope=rope,
            cache_feature=cache_feature,
            tag=tag,
            server_pscore=server_pscore,
        )
        x = self.proj(attn_v)
        x = self.proj_drop(x)

        return x, cache_feature

    def correct_partial_token(
        self, x_sel: Tensor, dindice: Tensor, rope: Tensor, cache_feature: Dict, tag: str
    ) -> Tuple[Tensor, dict]:
        # Load from cache
        update_indice = cache_feature[f"{tag}_update_indice"]  # [B, num_update]
        kv: Tensor = cache_feature[f"{tag}_kv"]     # [B, N, 2*D]
        
        # Shapes
        B, _, C = x_sel.shape
        N = kv.shape[1]

        gather_idx_x = update_indice.unsqueeze(-1).expand(-1, -1, C)  # [B, num_update, C]
        num_update = gather_idx_x.shape[1]
        gather_idx_qkv = update_indice.view(B, num_update, 1, 1, 1).expand(-1, -1, 3, self.num_heads, C // self.num_heads)  # [B, num_update, 3, H, D//H]

        # Partial update
        qkv_new = self.qkv(x_sel)  # [B, num_update, 3*D]
        qkv_new = qkv_new.reshape(B, num_update, 3, self.num_heads, C // self.num_heads)
        q_new = qkv_new[:, :, 0]
        kv_new = qkv_new[:, :, 1:]

        q = torch.zeros(B, N, self.num_heads, C // self.num_heads, device=x_sel.device, dtype=q_new.dtype)

        q = q.scatter_(1, gather_idx_qkv[:, :, 0], q_new)  # [B, N, H, D//H]
        kv = kv.scatter_(1, gather_idx_qkv[:, :, 1:], kv_new)  # [B, N, 2, H, D//H]

        k, v = torch.unbind(kv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]

        # RoPE
        if rope is not None:
            apply_rope_partial_triton(q, k, rope, update_indice)

        q_sel = q.gather(2, update_indice.view(B, 1, num_update, 1).expand(-1, self.num_heads, -1, C // self.num_heads))
        q_sel = q_sel.contiguous()

        # Attention
        attn_out_new = torch.nn.functional.scaled_dot_product_attention(q_sel, k, v)
        attn_out_new = attn_out_new.transpose(1, 2).reshape(B, num_update, C)

        # Projection
        x_sel = self.proj(attn_out_new)
        x_sel = self.proj_drop(x_sel)

        return x_sel, cache_feature

    def approx_partial_channel(
        self,
        x: Tensor,
        rope: Tensor,
        cache_feature: Dict,
        tag: str,
        attn_cache_candidates: Dict | None = None,
        attn_col_alive_ratio: float = 1.0,
    ) -> Tuple[Tensor, dict]:
        del attn_cache_candidates, attn_col_alive_ratio, tag
        return self.forward(x, rope=rope), cache_feature

    def correct_partial_channel(
        self,
        x_sel: Tensor,
        dindice: Tensor,
        rope: Tensor,
        cache_feature: Dict,
        tag: str,
        appcorr_method: str | None = None,
        fixed_query_state=None,
        attn_col_alive_ratio: float = 1.0,
        attn_cache_key=None,
        all_valid_queries: bool = False,
    ) -> Tuple[Tensor, dict]:
        del dindice, tag, appcorr_method, fixed_query_state, attn_col_alive_ratio, attn_cache_key, all_valid_queries
        return self.forward(x_sel, rope=rope), cache_feature


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def init_weights(
        self, init_attn_std: float | None = None, init_proj_std: float | None = None, factor: float = 1.0
    ) -> None:
        init_attn_std = init_attn_std or (self.dim**-0.5)
        init_proj_std = init_proj_std or init_attn_std * factor
        nn.init.normal_(self.qkv.weight, std=init_attn_std)
        nn.init.normal_(self.proj.weight, std=init_proj_std)
        if self.qkv.bias is not None:
            nn.init.zeros_(self.qkv.bias)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)

    def forward(self, x: Tensor, is_causal: bool = True) -> Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        x = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=self.attn_drop if self.training else 0, is_causal=is_causal
        )
        x = x.transpose(1, 2).contiguous().view(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x
