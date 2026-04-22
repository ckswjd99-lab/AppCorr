# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from dataclasses import dataclass
import math
from typing import Dict, List, Protocol, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

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


class QueryStateLike(Protocol):
    active_batch_idx: Tensor
    active_pos_idx: Tensor
    active_token_idx: Tensor
    query_valid_mask: Tensor
    active_query_pos_padded: Tensor
    active_query_mask: Tensor
    all_valid: bool


@dataclass
class PackedQueryState:
    active_batch_idx: Tensor
    active_pos_idx: Tensor
    active_token_idx: Tensor
    query_valid_mask: Tensor
    active_query_pos_padded: Tensor
    active_query_mask: Tensor
    all_valid: bool


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

    def _build_sparse_attn_cache(
        self,
        attn_prob: Tensor,
        dindice: Tensor,
        attn_col_alive_ratio: float,
    ) -> Dict[str, Tensor]:
        B, _, num_tok, N = attn_prob.shape[0], attn_prob.shape[1], dindice.shape[1], attn_prob.shape[-1]
        gather_idx = dindice.view(B, 1, num_tok, 1).expand(-1, self.num_heads, -1, N)
        attn_prob_sel = attn_prob.gather(2, gather_idx).contiguous()  # [B, H, Q, N]

        num_alive_cols = max(min(int(N * attn_col_alive_ratio), N), 1)
        col_scores = attn_prob_sel.mean(dim=(1, 2))  # [B, N]
        col_idx = torch.topk(col_scores, k=num_alive_cols, dim=-1, largest=True).indices  # [B, K]
        gather_idx_col = col_idx.view(B, 1, 1, num_alive_cols).expand(-1, self.num_heads, num_tok, -1)
        attn_prob_sel = attn_prob_sel.gather(3, gather_idx_col).contiguous()  # [B, H, Q, K]

        return {
            "query_idx": dindice.detach().clone(),
            "col_idx": col_idx.detach().clone(),
            "attn_prob_sel": attn_prob_sel.to(dtype=torch.bfloat16).detach().clone(),
        }

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
        if server_pscore in {"patch_attn_prob", "patch_attn_prob_layermean"}:
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
                attn_cache_candidates=kwargs["attn_cache_candidates"],
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
            return self.correct_partial_token(x_sel, dindice, rope, cache_feature, tag, **kwargs)
        if appcorr_method == "partial_channel":
            return self.correct_partial_channel(x_sel, dindice, rope, cache_feature, tag, **kwargs)
        raise ValueError(
            f"Unknown SelfAttention.correct method '{appcorr_method}'. "
            "Available methods: partial_channel, partial_token"
        )

    @staticmethod
    def _apply_rope_to_active_tokens(
        q_active: Tensor,
        k_active: Tensor,
        rope: tuple[Tensor, Tensor] | None,
        token_idx: Tensor,
        prefix_len: int,
    ) -> tuple[Tensor, Tensor]:
        if rope is None or token_idx.numel() == 0:
            return q_active, k_active

        patch_mask = token_idx >= prefix_len
        if not bool(patch_mask.any().item()):
            return q_active, k_active

        sin, cos = rope
        rope_idx = token_idx[patch_mask] - prefix_len
        sin_sel = sin.index_select(0, rope_idx).unsqueeze(1)
        cos_sel = cos.index_select(0, rope_idx).unsqueeze(1)

        q_dtype = q_active.dtype
        k_dtype = k_active.dtype
        q_active = q_active.clone()
        k_active = k_active.clone()
        q_active[patch_mask] = rope_apply(
            q_active[patch_mask].to(dtype=sin_sel.dtype),
            sin_sel,
            cos_sel,
        ).to(dtype=q_dtype)
        k_active[patch_mask] = rope_apply(
            k_active[patch_mask].to(dtype=sin_sel.dtype),
            sin_sel,
            cos_sel,
        ).to(dtype=k_dtype)
        return q_active, k_active

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
        self,
        x_sel: Tensor,
        dindice: Tensor,
        rope: Tensor,
        cache_feature: Dict,
        tag: str,
        *,
        fixed_query_state: QueryStateLike,
        **_: Dict,
    ) -> Tuple[Tensor, dict]:
        kv: Tensor = cache_feature[f"{tag}_kv"]  # [B, N, 2, H, Dh]
        if x_sel.numel() == 0:
            return x_sel.new_zeros((0, self.qkv.in_features)), cache_feature

        B = kv.shape[0]
        N = kv.shape[1]
        head_dim = self.qkv.in_features // self.num_heads
        num_active = x_sel.shape[0]
        t_max = fixed_query_state.active_query_pos_padded.shape[1]

        qkv_new = self.qkv(x_sel).reshape(num_active, 3, self.num_heads, head_dim)
        q_new = qkv_new[:, 0]
        kv_new = qkv_new[:, 1:]

        if rope is not None:
            prefix_len = N - rope[0].shape[0]
            q_new, k_new = self._apply_rope_to_active_tokens(
                q_new,
                kv_new[:, 0],
                rope,
                fixed_query_state.active_token_idx,
                prefix_len,
            )
            kv_new = kv_new.clone()
            kv_new[:, 0] = k_new

        kv[
            fixed_query_state.active_batch_idx,
            fixed_query_state.active_token_idx,
        ] = kv_new.to(dtype=kv.dtype)

        q_padded = torch.zeros(
            (B, t_max, self.num_heads, head_dim),
            device=x_sel.device,
            dtype=q_new.dtype,
        )
        q_padded[
            fixed_query_state.active_batch_idx,
            fixed_query_state.active_pos_idx,
        ] = q_new

        q = q_padded.transpose(1, 2).contiguous()
        k, v = torch.unbind(kv, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        attn_out_padded = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn_out_padded = attn_out_padded.transpose(1, 2).contiguous()
        if fixed_query_state.all_valid:
            attn_out_active = attn_out_padded.reshape(num_active, self.qkv.in_features)
        else:
            attn_out_active = attn_out_padded[
                fixed_query_state.active_batch_idx,
                fixed_query_state.active_pos_idx,
            ].reshape(num_active, self.qkv.in_features)

        x_sel = self.proj(attn_out_active)
        x_sel = self.proj_drop(x_sel)
        return x_sel, cache_feature

    def approx_partial_channel(
        self,
        x: Tensor,
        rope: Tensor,
        cache_feature: Dict,
        tag: str,
        attn_cache_candidates: Dict,
        attn_col_alive_ratio: float = 1.0,
    ) -> Tuple[Tensor, dict]:
        B, N, C = x.shape
        head_dim = C // self.num_heads

        qkv = self.qkv(x)   # [B, N, 3*C]

        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = torch.unbind(qkv, 2)
        q, k, v = [t.transpose(1, 2) for t in [q, k, v]]

        if rope is not None:
            q, k = self.apply_rope(q, k, rope)

        attn_score = (q @ k.transpose(-2, -1)) * self.scale
        attn_prob = attn_score.softmax(dim=-1)
        cache_feature[f"{tag}_dv_cache"] = torch.zeros(
            B, self.num_heads, N, head_dim, device=x.device, dtype=torch.bfloat16
        )
        if not attn_cache_candidates:
            raise ValueError("partial_channel requires non-empty attn_cache_candidates")
        for cache_key, dindice in attn_cache_candidates.items():
            cache_feature[f"{tag}_attn_sparse_cache_g{cache_key}"] = self._build_sparse_attn_cache(
                attn_prob, dindice, attn_col_alive_ratio
            )
        attn_v = attn_prob @ v
        attn_v = attn_v.transpose(1, 2).reshape(B, N, C)
        
        x = self.proj(attn_v)
        x = self.proj_drop(x)
        
        return x, cache_feature

    def correct_partial_channel(
        self,
        dx_active: Tensor,
        rope: Tensor,
        cache_feature: Dict,
        tag: str,
        fixed_query_state: QueryStateLike,
        attn_col_alive_ratio: float = 1.0,
        attn_cache_key=None,
        all_valid_queries: bool = False,
    ) -> Tuple[Tensor, dict]:
        # The block has already compacted tokens across the batch. At this point we only
        # need to map those active updates back into the cached sparse-attention layout.
        B = cache_feature[f"{tag}_dv_cache"].shape[0]
        num_active, C = dx_active.shape
        head_dim = C // self.num_heads
        active_batch_idx = fixed_query_state.active_batch_idx
        active_token_idx = fixed_query_state.active_token_idx
        active_query_pos_padded = fixed_query_state.active_query_pos_padded
        active_query_mask = fixed_query_state.active_query_mask
        sparse_cache_key = f"{tag}_attn_sparse_cache_g{attn_cache_key}"

        if num_active == 0:
            if sparse_cache_key not in cache_feature:
                raise KeyError(f"Sparse attention cache miss for key {attn_cache_key!r}")
            del cache_feature[sparse_cache_key]
            return dx_active.new_zeros((0, C)), cache_feature

        # Generate dv_active [T, H, Dh]
        v_weight = self.qkv.weight[2 * C :, :]
        dv_active = F.linear(dx_active, v_weight, bias=None)
        dv_active = dv_active.reshape(num_active, self.num_heads, head_dim).contiguous()

        dv_cache = cache_feature[f"{tag}_dv_cache"]
        dv_active = dv_active.to(dtype=dv_cache.dtype)
        dv_cache[active_batch_idx, :, active_token_idx, :] = dv_active

        sparse_cache = cache_feature.get(sparse_cache_key)
        if sparse_cache is None:
            raise KeyError(f"Sparse attention cache miss for key {attn_cache_key!r}")

        attn_prob_full = sparse_cache["attn_prob_sel"].to(dtype=dv_cache.dtype)  # [B,H,Q,K]
        col_idx = sparse_cache["col_idx"]
        gather_idx_dv = col_idx.view(B, 1, col_idx.shape[1], 1).expand(-1, self.num_heads, -1, head_dim)
        dv_sub = dv_cache.gather(2, gather_idx_dv).contiguous()  # [B, H, K, Dh]

        T_max = active_query_pos_padded.shape[1]
        # When every slot is valid we can slice the packed cache directly; otherwise we
        # gather only the valid query rows and mask padded positions back out.
        if all_valid_queries:
            attn_prob_packed = attn_prob_full[:, :, :T_max, :].contiguous()
        else:
            gather_idx_q = active_query_pos_padded.view(B, 1, T_max, 1).expand(-1, self.num_heads, -1, dv_sub.shape[2])
            attn_prob_packed = attn_prob_full.gather(2, gather_idx_q).contiguous()  # [B,H,Tmax,K]
            active_query_mask_f = active_query_mask.view(B, 1, T_max, 1).to(dtype=attn_prob_packed.dtype)
            attn_prob_packed = attn_prob_packed * active_query_mask_f
        attn_prob_mass_used = attn_prob_packed.float().sum()
        dattn_v_packed = torch.matmul(attn_prob_packed, dv_sub).transpose(1, 2).contiguous()  # [B,Tmax,H,Dh]

        cache_feature["_attn_prob_mass_used_total"] = (
            cache_feature.get("_attn_prob_mass_used_total", dx_active.new_zeros((), dtype=torch.float32))
            + attn_prob_mass_used
        )
        cache_feature["_attn_prob_mass_full_total"] = (
            cache_feature.get("_attn_prob_mass_full_total", dx_active.new_zeros((), dtype=torch.float32))
            + dx_active.new_tensor(float(num_active * self.num_heads), dtype=torch.float32)
        )

        if all_valid_queries:
            dattn_v_active = dattn_v_packed.reshape(num_active, C)
        else:
            dattn_v_active = dattn_v_packed[active_query_mask].reshape(num_active, C)
        dattn_v_active = dattn_v_active.to(dtype=self.proj.weight.dtype)
        dx = F.linear(dattn_v_active, self.proj.weight, bias=None)
        dx = self.proj_drop(dx)

        del cache_feature[sparse_cache_key]
        return dx, cache_feature


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
