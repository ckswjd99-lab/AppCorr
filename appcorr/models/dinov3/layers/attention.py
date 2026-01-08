# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import math
from typing import List, Tuple, Dict

import torch
import torch.nn.functional as F
from ..utils import cat_keep_shapes, uncat_with_shapes
from ..utils.hier_token import HierarchicalToken
from torch import Tensor, nn



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

    def apply_rope_partial(self, q: Tensor, k: Tensor, rope: Tensor | Tuple[Tensor, Tensor], dindice: Tensor) -> Tuple[Tensor, Tensor]:
        # All operations will use the dtype of rope, the output is cast back to the dtype of q and k
        q_dtype = q.dtype
        k_dtype = k.dtype
        sin, cos = rope
        rope_dtype = sin.dtype

        N = q.shape[-2]
        prefix = N - sin.shape[-2]
        dindice_real = dindice[prefix:]

        q_sel = q[:, :, dindice_real].to(dtype=rope_dtype)
        k_sel = k[:, :, dindice_real].to(dtype=rope_dtype)

        assert prefix >= 0
        q_prefix = q[:, :, :prefix, :]
        q_sel = rope_apply(q[:, :, dindice_real, :], sin[dindice_real-prefix], cos[dindice_real-prefix])
        q_sel = torch.cat((q_prefix, q_sel), dim=-2)
        
        k_prefix = k[:, :, :prefix, :]
        k_sel = rope_apply(k[:, :, dindice_real, :], sin[dindice_real-prefix], cos[dindice_real-prefix])
        k_sel = torch.cat((k_prefix, k_sel), dim=-2)

        q[:, :, dindice] = q_sel.to(dtype=q_dtype)
        k[:, :, dindice] = k_sel.to(dtype=k_dtype)
        
        return q, k

    def apply_rope_single(self, x: Tensor, rope: Tensor | Tuple[Tensor, Tensor]) -> Tensor:
        # All operations will use the dtype of rope, the output is cast back to the dtype of x
        x_dtype = x.dtype
        sin, cos = rope
        rope_dtype = sin.dtype
        x = x.to(dtype=rope_dtype)
        N = x.shape[-2]
        prefix = N - sin.shape[-2]
        assert prefix >= 0
        x_prefix = x[:, :, :prefix, :]
        x = rope_apply(x[:, :, prefix:, :], sin, cos)  # [B, head, hw, D//head]
        x = torch.cat((x_prefix, x), dim=-2)  # [B, head, N, D//head]
        x = x.to(dtype=x_dtype)
        return x

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

    def compute_attention(self, qkv: Tensor, attn_bias=None, rope=None) -> Tensor:
        assert attn_bias is None
        B, N, _ = qkv.shape
        C = self.qkv.in_features

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        if rope is not None:
            q, k = self.apply_rope(q, k, rope)
        x = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = x.transpose(1, 2)
        return x.reshape([B, N, C])

    def approx(self, x: Tensor, rope: Tensor, cache_feature: Dict, tag: str) -> Tuple[Tensor, dict]:
        qkv = self.qkv(x)
        cache_feature[f"{tag}_qkv"] = qkv.detach()

        attn_v = self.compute_attention(qkv=qkv, attn_bias=None, rope=rope)
        x = self.proj(attn_v)
        x = self.proj_drop(x)

        return x, cache_feature

    def correct(self, x: Tensor, dindice: Tensor, rope: Tensor, cache_feature: Dict, tag: str) -> Tuple[Tensor, dict]:
        qkv_old = cache_feature[f"{tag}_qkv"]
        qkv_new = self.qkv(x[:, dindice])
        qkv = qkv_old
        qkv[:, dindice] = qkv_new

        B, N, C = x.shape
        num_alive = dindice.shape[0]

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]
        if rope is not None:
            q, k = self.apply_rope_partial(q, k, rope, dindice)
        
        attn_out_new = torch.nn.functional.scaled_dot_product_attention(q[:, :, dindice, :], k, v)
        attn_out_new = attn_out_new.transpose(1, 2).reshape(B, num_alive, C)

        x_sel = self.proj(attn_out_new)
        x_sel = self.proj_drop(x_sel)

        x = x.to(dtype=x_sel.dtype) # TEMP
        x[:, dindice] = x_sel

        return x, cache_feature

    def approx_hier(self, x: HierarchicalToken, rope: Tuple[Tensor], cache_feature: Dict, tag: str) -> Tuple[Tensor, dict]:
        x_tensor = x.to_tensor()

        qkv = self.qkv(x_tensor)
        
        B, N, C = x_tensor.shape

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]

        # Cache k, v separately for pretokens, lowres, highres
        k_pretokens = torch.zeros_like(x.pretokens, device=x.pretokens.device)
        v_pretokens = torch.zeros_like(x.pretokens, device=x.pretokens.device)
        k_lowres = torch.zeros_like(x.lowres_tokens, device=x.lowres_tokens.device)
        v_lowres = torch.zeros_like(x.lowres_tokens, device=x.lowres_tokens.device)
        k_highres = torch.zeros_like(x.highres_tokens, device=x.highres_tokens.device)
        v_highres = torch.zeros_like(x.highres_tokens, device=x.highres_tokens.device)

        k_reshaped = k.transpose(1, 2).reshape(B, N, C)
        v_reshaped = v.transpose(1, 2).reshape(B, N, C)

        k_pretokens.copy_(k_reshaped[..., :x.pretokens.shape[-2], :])
        v_pretokens.copy_(v_reshaped[..., :x.pretokens.shape[-2], :])
        k_lowres[..., x.lowres_alive, :].copy_(k_reshaped[..., x.pretokens.shape[-2]:x.pretokens.shape[-2]+x.lowres_alive.sum().item(), :])
        v_lowres[..., x.lowres_alive, :].copy_(v_reshaped[..., x.pretokens.shape[-2]:x.pretokens.shape[-2]+x.lowres_alive.sum().item(), :])
        k_highres[..., x.highres_alive, :].copy_(k_reshaped[..., x.pretokens.shape[-2]+x.lowres_alive.sum().item():, :])
        v_highres[..., x.highres_alive, :].copy_(v_reshaped[..., x.pretokens.shape[-2]+x.lowres_alive.sum().item():, :])

        cache_feature[f"{tag}_k_pretokens"] = k_pretokens.detach()
        cache_feature[f"{tag}_v_pretokens"] = v_pretokens.detach()
        cache_feature[f"{tag}_k_lowres"] = k_lowres.detach()
        cache_feature[f"{tag}_v_lowres"] = v_lowres.detach()
        cache_feature[f"{tag}_k_highres"] = k_highres.detach()
        cache_feature[f"{tag}_v_highres"] = v_highres.detach()

        if rope is not None:
            q, k = self.apply_rope(q, k, rope)

        # TODO: make attention mask
        # -------------------------------------------------------------------------
        # [Log N Correction] Low-res 토큰의 면적만큼 Logit에 가산점 부여
        # -------------------------------------------------------------------------
        # 1. 구간 길이 계산
        n_pre = x.pretokens.shape[-2]
        n_low_alive = int(x.lowres_alive.sum().item())
        seq_len = k.shape[-2]

        # 2. 면적비(N) 계산: (High_H / Low_H)^2 -> 예: (28/14)^2 = 4
        if x.H_high is not None and x.H_low > 0:
            area_ratio = (x.H_high // x.H_low) ** 2
        else:
            area_ratio = 1.0

        # 3. Bias Tensor 생성: [1, 1, 1, Seq_Len] -> (B, Head, N, N)으로 브로드캐스팅됨
        attn_bias = torch.zeros((1, 1, 1, seq_len), device=q.device, dtype=q.dtype)

        # 4. Low-res 구간(Pretoken 직후 ~ Low-res 끝)에만 log(area) 적용
        if area_ratio > 1.0 and n_low_alive > 0:
            attn_bias[..., n_pre : n_pre + n_low_alive] = math.log(area_ratio)

        x_tensor = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        x_tensor = x_tensor.transpose(1, 2).reshape(B, N, C)
        x_tensor = self.proj(x_tensor)
        x_tensor = self.proj_drop(x_tensor)

        x = x.from_tensor(x_tensor)

        return x, cache_feature
    
    def correct_hier(
        self, 
        x_approx: HierarchicalToken, 
        x_correct: HierarchicalToken, 
        rope_approx: Tuple[Tensor],
        rope_correct: Tuple[Tensor], 
        cache_feature: Dict, 
        tag: str
    ) -> Tuple[Tensor, dict]:
        x_cor_tensor = x_correct.to_tensor()
        B, Nq, C = x_cor_tensor.shape

        # Generate q, k, v for the correction tokens
        qkv_new = self.qkv(x_cor_tensor)
        qkv_new = qkv_new.reshape(B, Nq, 3, self.num_heads, C // self.num_heads)
        q_new, k_new, v_new = torch.unbind(qkv_new, 2)
        k_new = k_new.reshape(B, -1, C)
        v_new = v_new.reshape(B, -1, C)

        # Retrieve cached k, v
        k_pretokens, v_pretokens = cache_feature[f"{tag}_k_pretokens"], cache_feature[f"{tag}_v_pretokens"]
        k_lowres, v_lowres = cache_feature[f"{tag}_k_lowres"], cache_feature[f"{tag}_v_lowres"]
        k_highres, v_highres = cache_feature[f"{tag}_k_highres"], cache_feature[f"{tag}_v_highres"]

        # Update cache
        k_pretokens.copy_(k_new[..., :x_correct.num_pretokens, :])
        v_pretokens.copy_(v_new[..., :x_correct.num_pretokens, :])
        k_highres[..., x_correct.highres_alive, :].copy_(k_new[..., x_correct.num_pretokens:, :])
        v_highres[..., x_correct.highres_alive, :].copy_(v_new[..., x_correct.num_pretokens:, :])
        
        # Take only alive tokens from lowres and highres
        k_lowres = k_lowres[..., x_approx.lowres_alive, :]
        v_lowres = v_lowres[..., x_approx.lowres_alive, :]
        k_highres = k_highres[..., x_approx.highres_alive, :]
        v_highres = v_highres[..., x_approx.highres_alive, :]

        k = torch.cat([k_pretokens, k_lowres, k_highres], dim=-2)
        v = torch.cat([v_pretokens, v_lowres, v_highres], dim=-2)

        # Prepare q, k, v for attention computation
        q = q_new.transpose(1, 2)
        k = k.reshape(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.reshape(B, -1, self.num_heads, C // self.num_heads).transpose(1, 2)

        if rope_approx is not None:
            q = self.apply_rope_single(q, rope_correct)
            k = self.apply_rope_single(k, rope_approx)

        # Make attention mask for low-res tokens
        n_pre = k_pretokens.shape[-2]
        n_low = k_lowres.shape[-2]
        n_high = k_highres.shape[-2]
        total_k = n_pre + n_low + n_high

        if x_approx.H_low > 0:
            area_ratio = (x_approx.H_high // x_approx.H_low) ** 2
        else:
            area_ratio = 1.0 

        attn_bias = torch.zeros((1, 1, 1, total_k), device=q.device, dtype=q.dtype)

        if n_low > 0 and area_ratio > 1.0:
            bias_val = math.log(area_ratio)
            attn_bias[..., n_pre : n_pre + n_low] = bias_val

        # Attention computation
        attn_out_new = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_bias)
        attn_out_new = attn_out_new.transpose(1, 2).reshape(B, Nq, C)
        x_sel = self.proj(attn_out_new)
        x_sel = self.proj_drop(x_sel)

        x_correct = x_correct.from_tensor(x_sel)

        return x_correct, cache_feature




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
