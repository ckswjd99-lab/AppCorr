"""
stock.py

A from-code, dependency-free reimplementation of GLM-5.3-Flash's vision tower, used for two
things and only those two:

  * as the WEIGHT CONTAINER `load_stock_vision_tower` builds when this box's `transformers` does
    not carry `models/glm5_next` (the `appcorr` env here ships 5.13.0, which does not; the served
    `appcorr-vllm-main` env ships 5.16.1, which does). The fork wraps whichever of the two it is
    handed -- the module names are the checkpoint's, so they are interchangeable;
  * as the STOCK REFERENCE a gate compares the fork against on a box with no HF class at all.

Ported line for line from vLLM main 658c813 `vllm/models/glm5next/nvidia/multimodal.py`
(`Glm5NextVisionPatchEmbed` :48, `Glm5NextVisionMLP` :76, `Glm5NextVisionAttention` :112,
`Glm5NextVisionBlock` :231, `Glm5NextPatchMerger` :284, `Glm5NextVisionTransformer` :336, forward
:559-606) and cross-checked term by term against transformers 5.16.1
`models/glm5_next/modeling_glm5_next.py` (:1497-1827). The two agree on every tensor op; they
disagree on ONE constant, and the disagreement is the reason this file takes its epsilons as
arguments rather than from the config:

    vision_config.rms_norm_eps   checkpoint ships 1e-5;  vLLM OVERRIDES it to 1e-6
                                 (`vllm/transformers_utils/configs/glm5_next.py:315-319`, with a
                                 comment saying the tower was trained at 1e-6 and that 1e-5
                                 "produces repetitive/degraded image descriptions")
    q_norm / k_norm eps          vLLM hard-codes 1e-5 (`multimodal.py:142-143`), i.e. it does NOT
                                 follow the override; HF reads `config.rms_norm_eps` for these
                                 too, which on this checkpoint is also 1e-5.

So the served tower is `norm1/norm2/post_layernorm @ 1e-6` and `q_norm/k_norm @ 1e-5`, and an HF
`Glm5NextVisionModel` built straight from the checkpoint config is `1e-5` everywhere. We serve
against vLLM, so `NORM_EPS`/`QK_NORM_EPS` below are the defaults and `apply_vllm_eps()` retunes
an HF-built tower to them.

Not ported (deliberately): the video path, the encoder-CUDA-graph metadata path
(`prepare_encoder_metadata`, which recomputes exactly what `rot_pos_emb` does), TP sharding, and
the weight remap `hf_to_vllm_mapper` -- this checkpoint ships a FUSED `attn.qkv.{weight,bias}`
(verified on `model.safetensors.index.json`: 347 `model.visual.*` tensors, all in shard 62,
`blocks.N.attn.qkv.*` and no `blocks.N.attn.q.*`), so the mapper's stacked remap is a no-op here.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# vLLM's two epsilons for this tower (see the module docstring).
NORM_EPS = 1e-6       # norm1 / norm2 / post_layernorm -- vLLM's forced vision rms_norm_eps
QK_NORM_EPS = 1e-5    # per-head q_norm / k_norm -- hard-coded in vllm multimodal.py:142-143


class Glm5NextRMSNorm(nn.Module):
    """`transformers` `Glm5NextRMSNorm` (:1541) == vLLM `RMSNorm` in its non-fused form."""

    def __init__(self, hidden_size: int, eps: float = NORM_EPS):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * x.to(dt)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Glm5NextVisionPatchEmbed(nn.Module):
    """Conv3d over (T, P, P). No post-conv norm and no learned absolute position embedding --
    the two things GLM-4.6V's patch embed has and this one does not (`multimodal.py:380`)."""

    def __init__(self, patch_size: int = 14, temporal_patch_size: int = 2, in_channels: int = 3,
                 hidden_size: int = 1024):
        super().__init__()
        self.patch_size = int(patch_size)
        self.temporal_patch_size = int(temporal_patch_size)
        self.in_channels = int(in_channels)
        self.embed_dim = int(hidden_size)
        k = (self.temporal_patch_size, self.patch_size, self.patch_size)
        self.proj = nn.Conv3d(self.in_channels, self.embed_dim, kernel_size=k, stride=k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        return self.proj(x.to(self.proj.weight.dtype)).view(-1, self.embed_dim)


def _clamped_swiglu(gate: torch.Tensor, up: torch.Tensor, limit: float) -> torch.Tensor:
    """`SiluAndMulWithClamp`: the gate is clamped ABOVE only, the up branch on both sides.
    (vLLM `layers/activation.py`; HF :1509-1513 -- `gate.clamp(min=None, max=limit)` and
    `up.clamp(min=-limit, max=limit)`. The asymmetry is not a typo, it is what both sides do.)"""
    gate = gate.clamp(min=None, max=limit)
    up = up.clamp(min=-limit, max=limit)
    return F.silu(gate) * up


class Glm5NextVisionMLP(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, swiglu_limit: float,
                 bias: bool = True):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)
        self.swiglu_limit = float(swiglu_limit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(_clamped_swiglu(self.gate_proj(x), self.up_proj(x), self.swiglu_limit))


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)


def apply_rotary_pos_emb_vision(q, k, cos, sin):
    qd, kd = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    return ((q * cos) + (rotate_half(q) * sin)).to(qd), ((k * cos) + (rotate_half(k) * sin)).to(kd)


class Glm5NextVisionAttention(nn.Module):
    """Fused qkv WITH bias (`attention_bias: true`), per-head q/k RMSNorm, 2-D RoPE, full
    bidirectional attention split by `cu_seqlens`. No windows anywhere in this tower."""

    def __init__(self, hidden_size: int, num_heads: int, qk_norm_eps: float = QK_NORM_EPS,
                 bias: bool = True):
        super().__init__()
        self.dim = int(hidden_size)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=bias)
        self.proj = nn.Linear(self.dim, self.dim, bias=bias)
        self.scaling = self.head_dim ** -0.5
        self.q_norm = Glm5NextRMSNorm(self.head_dim, eps=qk_norm_eps)
        self.k_norm = Glm5NextRMSNorm(self.head_dim, eps=qk_norm_eps)

    def forward(self, x: torch.Tensor, cu_seqlens, position_embeddings) -> torch.Tensor:
        T = x.shape[0]
        q, k, v = self.qkv(x).reshape(T, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        outs, s = [], 0
        for L in lengths:
            qs, ks, vs = (t[s:s + L].transpose(0, 1).unsqueeze(0) for t in (q, k, v))
            o = F.scaled_dot_product_attention(qs, ks, vs, scale=self.scaling)
            outs.append(o.squeeze(0).transpose(0, 1))
            s += L
        return self.proj(torch.cat(outs, dim=0).reshape(T, -1))


class Glm5NextVisionBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int,
                 swiglu_limit: float, norm_eps: float = NORM_EPS,
                 qk_norm_eps: float = QK_NORM_EPS, bias: bool = True):
        super().__init__()
        self.norm1 = Glm5NextRMSNorm(hidden_size, eps=norm_eps)
        self.norm2 = Glm5NextRMSNorm(hidden_size, eps=norm_eps)
        self.attn = Glm5NextVisionAttention(hidden_size, num_heads, qk_norm_eps, bias=bias)
        self.mlp = Glm5NextVisionMLP(hidden_size, intermediate_size, swiglu_limit, bias=bias)

    def forward(self, x, cu_seqlens, position_embeddings):
        x = x + self.attn(self.norm1(x), cu_seqlens, position_embeddings)
        return x + self.mlp(self.norm2(x))


class Glm5NextVisionPatchMerger(nn.Module):
    """`proj -> LayerNorm -> GELU -> clamped SwiGLU(out_hidden -> projection_intermediate ->
    out_hidden)`, all bias-free except the LayerNorm (checkpoint: only
    `merger.post_projection_norm.bias` exists)."""

    def __init__(self, dim: int, context_dim: int, swiglu_limit: float, bias: bool = False):
        super().__init__()
        self.proj = nn.Linear(dim, dim, bias=bias)
        self.post_projection_norm = nn.LayerNorm(dim)
        self.gate_proj = nn.Linear(dim, context_dim, bias=bias)
        self.up_proj = nn.Linear(dim, context_dim, bias=bias)
        self.down_proj = nn.Linear(context_dim, dim, bias=bias)
        self.act1 = nn.GELU()
        self.swiglu_limit = float(swiglu_limit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        x = self.act1(self.post_projection_norm(x))
        return self.down_proj(_clamped_swiglu(self.gate_proj(x), self.up_proj(x), self.swiglu_limit))


class Glm5NextVisionRotaryEmbedding(nn.Module):
    """`inv_freq` over `head_dim // 2` (= 32 -> 16 frequencies), theta 1e4. `forward(pos_ids)`
    takes the (h, w) ids `[T, 2]` and returns `[T, 32]`; the caller does `cat(f, f)` -> `[T, 64]`,
    i.e. a FULL-head-dim neox rotation whose frequency vector is the two axes concatenated.

    vLLM reaches the identical table through `get_rope(head_size=64, partial_rotary_factor=0.5,
    is_neox_style=True)` (`multimodal.py:390-395`): rotary_dim 32 -> 16 inv_freqs, `cos[pos_ids]
    .flatten(1)` -> `[T, 32]`, and `ApplyRotaryEmb` treats that as the half-vector of a 64-dim
    rotation. Same numbers, two spellings."""

    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))
        self.register_buffer("inv_freq", inv, persistent=False)

    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        return (position_ids.unsqueeze(-1) * self.inv_freq).flatten(1)


class Glm5NextVisionStock(nn.Module):
    """The stock tower: `patch_embed -> 24 blocks -> post_layernorm -> 2x2 downsample conv ->
    merger`. Attribute names are the checkpoint's (`model.visual.*` minus the prefix), so a
    `state_dict` from the safetensors shards loads with `strict=True`."""

    def __init__(self, config, norm_eps: float = NORM_EPS, qk_norm_eps: float = QK_NORM_EPS):
        super().__init__()
        self.config = config
        self.spatial_merge_size = int(config.spatial_merge_size)
        self.patch_size = int(config.patch_size)
        h, heads = int(config.hidden_size), int(config.num_heads)
        limit = float(config.swiglu_limit)
        self.patch_embed = Glm5NextVisionPatchEmbed(config.patch_size, config.temporal_patch_size,
                                                    config.in_channels, h)
        self.rotary_pos_emb = Glm5NextVisionRotaryEmbedding((h // heads) // 2)
        self.blocks = nn.ModuleList([
            Glm5NextVisionBlock(h, heads, int(config.intermediate_size), limit, norm_eps,
                                qk_norm_eps, bias=bool(config.attention_bias))
            for _ in range(int(config.depth))])
        self.merger = Glm5NextVisionPatchMerger(int(config.out_hidden_size),
                                                int(config.projection_intermediate_size), limit)
        self.downsample = nn.Conv2d(h, int(config.out_hidden_size),
                                    kernel_size=self.spatial_merge_size,
                                    stride=self.spatial_merge_size)
        self.post_layernorm = Glm5NextRMSNorm(h, eps=norm_eps)

    @torch.no_grad()
    def forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        from transformers.vision_utils import get_vision_cu_seqlens, get_vision_position_ids
        pos_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size)
        cu = get_vision_cu_seqlens(grid_thw)
        x = self.patch_embed(pixel_values)
        rot = self.rotary_pos_emb(pos_ids)
        emb = torch.cat((rot, rot), dim=-1)
        pe = (emb.cos(), emb.sin())
        for blk in self.blocks:
            x = blk(x, cu, pe)
        x = self.post_layernorm(x)
        m = self.spatial_merge_size
        x = x.view(-1, m, m, x.shape[-1]).permute(0, 3, 1, 2)
        x = self.downsample(x).view(-1, int(self.config.out_hidden_size))
        return self.merger(x)


def apply_vllm_eps(visual: nn.Module, norm_eps: float = NORM_EPS,
                   qk_norm_eps: float = QK_NORM_EPS) -> nn.Module:
    """Retune an HF-built `Glm5NextVisionModel` (which reads ONE eps from the config, 1e-5 on this
    checkpoint) to the two vLLM serves with. Idempotent; returns the module."""
    for blk in visual.blocks:
        blk.norm1.variance_epsilon = float(norm_eps)
        blk.norm2.variance_epsilon = float(norm_eps)
        blk.attn.q_norm.variance_epsilon = float(qk_norm_eps)
        blk.attn.k_norm.variance_epsilon = float(qk_norm_eps)
    visual.post_layernorm.variance_epsilon = float(norm_eps)
    return visual


def tower_eps(visual: nn.Module) -> dict:
    """`{norm, qk_norm}` as actually configured on a tower module, for a gate to print."""
    return {"norm": float(visual.blocks[0].norm1.variance_epsilon),
            "qk_norm": float(visual.blocks[0].attn.q_norm.variance_epsilon),
            "post": float(visual.post_layernorm.variance_epsilon)}
