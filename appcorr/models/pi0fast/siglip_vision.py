"""
siglip_vision.py

Progressive (approx/correct) fork of the HuggingFace SigLIP vision tower used inside PaliGemma /
pi0-FAST. Same approx/correct/KV-splice pattern as the OpenVLA/CLIP vision forks; adapted to HF
SigLIP's module layout:
  - SiglipEncoderLayer: layer_norm1 -> self_attn -> (residual) -> layer_norm2 -> mlp -> (residual),
    NO LayerScale (like CLIP), pre-LN.
  - SiglipAttention: SEPARATE q_proj/k_proj/v_proj/out_proj (not fused), full bidirectional SDPA,
    scale = head_dim**-0.5.
  - SiglipVisionEmbeddings: patch_embedding conv + position_embedding (no CLS token).
  - Final: post_layernorm over all 256 patch tokens (PaliGemma uses vision_use_head=False, so the
    patch features -- not a pooled vector -- feed the multi_modal_projector).

Progressive vision = approx on a low-res base, then correct only the arriving patches (KV updated
in place); non-corrected patches keep their base K/V (accepted staleness, as in every prior fork).
"""

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ApproxCorrectSiglipAttention(nn.Module):
    def __init__(self, q_proj, k_proj, v_proj, out_proj, num_heads, head_dim, scale):
        super().__init__()
        self.q_proj, self.k_proj, self.v_proj, self.out_proj = q_proj, k_proj, v_proj, out_proj
        self.num_heads, self.head_dim, self.scale = num_heads, head_dim, scale

    @classmethod
    def from_stock(cls, attn: nn.Module) -> "ApproxCorrectSiglipAttention":
        return cls(attn.q_proj, attn.k_proj, attn.v_proj, attn.out_proj,
                   attn.num_heads, attn.head_dim, attn.scale)

    def _heads(self, x):
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    def approx(self, x, cache_feature, tag, pscore=False):
        B, N, _ = x.shape
        q, k, v = self._heads(x)
        cache_feature[f"{tag}_kv"] = torch.stack([k, v], dim=3)  # [B, H, N, 2, Dh]
        if pscore:
            # patch_attn_prob (DINOv3 attention.py): softmax(qk^T*scale) -> mean over heads ->
            # mean over query positions => avg attention RECEIVED by each patch key. [B, N]
            attn_prob = (q @ k.transpose(-2, -1) * self.scale).softmax(dim=-1)  # [B, H, N, N]
            cache_feature[f"{tag}_attn_pscore"] = attn_prob.mean(dim=1).mean(dim=1).detach()  # [B, N]
        o = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        return self.out_proj(o.transpose(1, 2).reshape(B, N, self.num_heads * self.head_dim)), cache_feature

    def correct(self, x_sel, token_idx, cache_feature, tag):
        kv = cache_feature[f"{tag}_kv"]  # [B, H, N, 2, Dh]
        B, _, N = kv.shape[0], kv.shape[1], kv.shape[2]
        Q = token_idx.shape[0]
        token_idx = token_idx.to(kv.device)
        q_new, k_new, v_new = self._heads(x_sel)
        kv[:, :, token_idx, 0] = k_new.to(kv.dtype)
        kv[:, :, token_idx, 1] = v_new.to(kv.dtype)
        cache_feature[f"{tag}_kv"] = kv
        k_full, v_full = kv.unbind(3)  # bidirectional: attend to all N keys
        o = F.scaled_dot_product_attention(q_new, k_full, v_full, scale=self.scale)
        return self.out_proj(o.transpose(1, 2).reshape(B, Q, self.num_heads * self.head_dim)), cache_feature


class ApproxCorrectSiglipLayer(nn.Module):
    def __init__(self, layer_norm1, self_attn, layer_norm2, mlp):
        super().__init__()
        self.layer_norm1, self.self_attn, self.layer_norm2, self.mlp = layer_norm1, self_attn, layer_norm2, mlp

    @classmethod
    def from_stock(cls, layer: nn.Module) -> "ApproxCorrectSiglipLayer":
        return cls(layer.layer_norm1, ApproxCorrectSiglipAttention.from_stock(layer.self_attn),
                   layer.layer_norm2, layer.mlp)

    def approx(self, x, cache_feature, tag, pscore=False):
        x_attn, cache_feature = self.self_attn.approx(self.layer_norm1(x), cache_feature, tag, pscore=pscore)
        cache_feature[f"{tag}_blocks_out_sum"] = x_attn.detach().clone()
        x_mid = x + x_attn
        mlp_out = self.mlp(self.layer_norm2(x_mid))
        cache_feature[f"{tag}_blocks_out_sum"] = cache_feature[f"{tag}_blocks_out_sum"] + mlp_out.detach()
        if pscore:
            # residual magnitude: L2 norm over channels of this block's total output update. [B, N]
            cache_feature[f"{tag}_residual_mag"] = cache_feature[f"{tag}_blocks_out_sum"].norm(dim=-1).detach()
        return x_mid + mlp_out, cache_feature

    def correct(self, x, token_idx, cache_feature, tag):
        token_idx = token_idx.to(x.device)
        x_active = x[:, token_idx]
        x_attn_sel, cache_feature = self.self_attn.correct(self.layer_norm1(x_active), token_idx, cache_feature, tag)
        x_attn_active = x_active + x_attn_sel
        mlp_out_new = self.mlp(self.layer_norm2(x_attn_active))
        x_out = x + cache_feature[f"{tag}_blocks_out_sum"].to(x.dtype)
        x_out[:, token_idx] = (x_attn_active + mlp_out_new).to(x_out.dtype)
        return x_out, cache_feature


class ApproxCorrectSiglipBackbone(nn.Module):
    """Wraps a stock HF SiglipVisionModel (`.vision_model`): embeddings + N forked encoder layers +
    post_layernorm. approx_forward/correct_forward return the [B, 256, D] patch features."""

    def __init__(self, vision_model: nn.Module):
        super().__init__()
        self.embeddings = vision_model.embeddings
        self.post_layernorm = vision_model.post_layernorm
        self.layers = nn.ModuleList([ApproxCorrectSiglipLayer.from_stock(l) for l in vision_model.encoder.layers])

    def _embed(self, pixel_values):
        return self.embeddings(pixel_values)  # [B, 256, D]

    def approx_forward(self, pixel_values, cache_feature, tag_prefix, pscore=False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        x = self._embed(pixel_values)
        attn_acc = None
        res_acc = None
        for i, blk in enumerate(self.layers):
            x, cache_feature = blk.approx(x, cache_feature, f"{tag_prefix}_layer{i}", pscore=pscore)
            if pscore:
                a = cache_feature[f"{tag_prefix}_layer{i}_attn_pscore"]
                r = cache_feature[f"{tag_prefix}_layer{i}_residual_mag"]
                attn_acc = a if attn_acc is None else attn_acc + a
                res_acc = r if res_acc is None else res_acc + r
        if pscore:
            n = len(self.layers)
            cache_feature[f"{tag_prefix}_avg_attn"] = attn_acc / n       # [B, N] mean over layers
            cache_feature[f"{tag_prefix}_residual_mag"] = res_acc / n     # [B, N] mean over layers
        return self.post_layernorm(x), cache_feature

    @staticmethod
    def get_pscore(cache_feature, tag_prefix) -> torch.Tensor:
        """pscore_i = residual_i * avg_attn_i  (ProgVFM contrib_i; both from the approx pass). [B, N]"""
        return cache_feature[f"{tag_prefix}_residual_mag"] * cache_feature[f"{tag_prefix}_avg_attn"]

    def correct_forward(self, pixel_values, patch_idx, cache_feature, tag_prefix) -> Tuple[torch.Tensor, Dict[str, Any]]:
        x = self._embed(pixel_values)
        idx = patch_idx.to(dtype=torch.long, device=x.device)
        for i, blk in enumerate(self.layers):
            x, cache_feature = blk.correct(x, idx, cache_feature, f"{tag_prefix}_layer{i}")
        return self.post_layernorm(x), cache_feature
