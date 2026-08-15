# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from appcorr.models.vggt_omega.models.layers import RopePositionEmbedding

# Blocks and the patch-embed ViT come from the *instrumented* DINOv3 tree, not the vendored one.
# They are the same architecture -- verified bit-identical output under VGGT's own weights, with
# state_dict keys matching exactly (`scratchpad/block_swap_check.py`) -- but they additionally carry
# the approx/correct, pscore-selection and precision machinery that correction needs. Building on
# the vendored copies would mean writing a second implementation of all of it.
#
# The only architectural gap was qk-norm, which VGGT's aggregator is trained with and DINOv3 is not;
# it is now a no-op-by-default option on the shared attention.
from appcorr.models.dinov3.layers.block import SelfAttentionBlock
from appcorr.models.dinov3.layers.ffn_layers import Mlp
from appcorr.models.dinov3.models.vision_transformer import DinoVisionTransformer


_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


class Aggregator(nn.Module):
    """Alternating-attention encoder over video frames."""

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 16,
        register_attention_block_indices: list[int] = [2, 6, 9, 14, 20],
        cached_layer_indices: tuple[int, ...] = (4, 11, 17, 23),
    ) -> None:
        super().__init__()

        self.patch_embed = _build_patch_embed(patch_size=patch_size, embed_dim=embed_dim)
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=100,
            normalize_coords="max",
            dtype=torch.float32,
        )

        self.frame_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    ffn_layer=Mlp,
                    init_values=1e-5,
                    use_qk_norm=True,
                    mask_k_bias=True,
                )
                for _ in range(depth)
            ]
        )
        self.inter_frame_blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    ffn_ratio=mlp_ratio,
                    qkv_bias=True,
                    proj_bias=True,
                    ffn_bias=True,
                    ffn_layer=Mlp,
                    init_values=1e-5,
                    use_qk_norm=True,
                    mask_k_bias=True,
                )
                for _ in range(depth)
            ]
        )

        self.depth = depth
        self.patch_size = patch_size
        self.cached_layer_indices = set(cached_layer_indices)
        self.camera_token = nn.Parameter(torch.empty(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.empty(1, 2, num_register_tokens, embed_dim))
        self.patch_token_start = 1 + num_register_tokens

        self.inter_frame_attention_types = ["global"] * depth
        for idx in register_attention_block_indices:
            if idx < 0 or idx >= depth:
                raise ValueError(f"register_attention_block_indices contains invalid block index {idx}")
            self.inter_frame_attention_types[idx] = "register"

        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.init_weights()

    def init_weights(self) -> None:
        nn.init.normal_(self.camera_token, std=1e-3)
        nn.init.normal_(self.register_token, std=1e-3)

    @staticmethod
    def _call_block(blk, x, rope, cache_feature, tag, kwargs, correct=False):
        """Run one block stock, approx, or correct. Every block call in this file goes through here
        so the three modes cannot drift apart.

        Approx mode is numerically the same forward -- it additionally stores the KV cache and the
        per-token scores correction later needs -- so an approx-only pass must reproduce the stock
        forward exactly. Correct mode consumes that cache and recomputes the selected tokens, so at
        100% selection it must reproduce a stock forward on the refined input.

        `dindice` is every token index: with `num_groups = 1` the whole sequence is a candidate and
        the actual pruning happens inside the block from the fused pscore, exactly as in DINOv3.
        """
        if cache_feature is None:
            out = blk(x, rope)
            return (out[0] if isinstance(out, list) else out), None
        if not correct:
            return blk.approx(x, rope, cache_feature, tag=tag, **kwargs)

        rows, n_tok = x.shape[0], x.shape[1]
        dindice = torch.arange(n_tok, device=x.device, dtype=torch.long).unsqueeze(0).expand(rows, -1)
        extra = {} if rope is not None else {"num_pretokens": 0}
        # The mobile residual hint is laid out per stack, since the three stacks read different token
        # axes. Select by stack here, and *do not* mutate `kwargs` -- `run_blocks` hands the same
        # dict to all 72 blocks, so popping from it delivers the hint to the first block only and
        # leaves the other 71 running unhinted. That produced three separate oracle experiments whose
        # results agreed to three decimals, because the hint they varied was reaching one block.
        hints = kwargs.get("mobile_pscore_hints")
        kwargs = {k: v for k, v in kwargs.items() if k != "mobile_pscore_hints"}
        if hints is not None:
            extra["mobile_pscore_hint"] = hints.get(tag.rstrip("0123456789"))
        return blk.correct(x, dindice, rope, cache_feature, tag=tag, **kwargs, **extra)

    def _run_patch_embed(self, images, cache_feature=None, approx_kwargs=None, correct=False):
        """The patch-embed ViT's 24 blocks: stock, approx, or correct.

        This stack is a correction target in its own right, not just a feature extractor. It is
        per-frame and has no cross-frame dependency, so it is the only part of the model that can
        run before every frame has arrived -- which makes it the natural place to hide work under
        the refinement transfer. Its `pe{i}` tags keep its KV cache separate from the aggregator's.

        Not `forward_features_list_appcorr`: that driver builds its own input pyramid by
        interpolating the tensor, which is the standalone/offline path. Here the frames have already
        been degraded by the transmission policy, so degrading them again would double it.
        """
        pe = self.patch_embed
        if cache_feature is None:
            out = pe.forward_features(images)
            return out["x_norm_patchtokens"] if isinstance(out, dict) else out

        x, (grid_h, grid_w) = pe.prepare_tokens_with_masks(images, None)
        for idx, blk in enumerate(pe.blocks):
            rope = pe.rope_embed(H=grid_h, W=grid_w) if pe.rope_embed is not None else None
            x, cache_feature = self._call_block(
                blk, x, rope, cache_feature, f"pe{idx}", approx_kwargs or {}, correct
            )
        return pe.post_features_list([x], [None])[0]["x_norm_patchtokens"]

    def embed(
        self,
        images: torch.Tensor,
        cache_feature: dict | None = None,
        approx_kwargs: dict | None = None,
        correct: bool = False,
    ) -> tuple[torch.Tensor, tuple, dict]:
        """Everything before the block loop: normalize, patch-embed, prepend cam/register, build RoPE.

        Split out of `forward` so the stock, approx and correct paths provably share one prologue.
        `geom` carries the shape metadata the block loop needs, since the token axis is reinterpreted
        between the frame and inter-frame stacks and every caller needs the same numbers.
        """
        batch_size, num_frames, num_channels, height, width = images.shape
        if num_channels != 3:
            raise ValueError(f"Expected 3 input channels, got {num_channels}")

        images = (images - self._resnet_mean) / self._resnet_std
        images = images.view(batch_size * num_frames, num_channels, height, width)

        camera_token = slice_expand_and_flatten(self.camera_token, batch_size, num_frames)
        register_token = slice_expand_and_flatten(self.register_token, batch_size, num_frames)

        # `forward_features`, not `__call__`: on the instrumented ViT `forward` is the *classifier*
        # entry point and returns the CLS token alone, so calling the module directly silently
        # yields [B, D] where [B, N, D] is needed -- caught here only because the following cat
        # happens to fail on rank. `_run_patch_embed` handles that and the approx variant.
        patch_tokens = self._run_patch_embed(images, cache_feature, approx_kwargs, correct)

        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        _, num_tokens, embed_dim = tokens.shape

        patch_grid_size = (height // self.patch_size, width // self.patch_size)
        with torch.no_grad():
            rope_sin, rope_cos = self.rope_embed(H=patch_grid_size[0], W=patch_grid_size[1])
            frame_rope = (
                rope_sin.to(device=patch_tokens.device, dtype=torch.float32),
                rope_cos.to(device=patch_tokens.device, dtype=torch.float32),
            )

        geom = {
            "batch_size": batch_size,
            "num_frames": num_frames,
            "num_tokens": num_tokens,
            "embed_dim": embed_dim,
            "patch_grid_size": patch_grid_size,
        }
        return tokens, frame_rope, geom

    def run_blocks(
        self,
        tokens: torch.Tensor,
        frame_rope: tuple,
        geom: dict,
        block_range: tuple[int, int] | None = None,
        outputs: list | None = None,
        cache_feature: dict | None = None,
        approx_kwargs: dict | None = None,
        correct: bool = False,
    ) -> tuple[list, torch.Tensor]:
        """The block loop over `[start, end)` of the 24 paired blocks. Approx when given a cache.

        Returned `outputs` is indexed by absolute block index so a partial range still lands its
        cached layers in the right slots -- the heads read positions, not order of arrival.

        Tags are `frame{i}` / `inter{i}`, distinct from the patch-embed stack's `pe{i}`. They must
        stay distinct: the three stacks reuse the same block *class* and would otherwise overwrite
        each other's KV caches, which is the kind of failure that still produces plausible numbers.
        """
        start, end = block_range if block_range is not None else (0, self.depth)
        if outputs is None:
            outputs = [None] * self.depth
        approx_kwargs = approx_kwargs or {}

        for block_idx in range(start, end):
            tokens, frame_tokens, cache_feature = self._run_frame_block(
                tokens,
                geom["batch_size"],
                geom["num_frames"],
                geom["num_tokens"],
                geom["embed_dim"],
                block_idx,
                frame_rope,
                cache_feature,
                approx_kwargs,
                correct,
            )
            tokens, cache_feature = self._run_inter_frame_attention_block(
                tokens,
                geom["batch_size"],
                geom["num_frames"],
                geom["num_tokens"],
                geom["embed_dim"],
                block_idx,
                self.inter_frame_attention_types[block_idx],
                cache_feature,
                approx_kwargs,
                correct,
            )
            if block_idx in self.cached_layer_indices:
                outputs[block_idx] = torch.cat([frame_tokens, tokens], dim=-1)

        return outputs, tokens

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[list[torch.Tensor | None], int]:
        tokens, frame_rope, geom = self.embed(images)
        outputs, _ = self.run_blocks(tokens, frame_rope, geom)
        return outputs, self.patch_token_start

    def _run_frame_block(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        embed_dim: int,
        block_idx: int,
        rope_sincos: tuple[torch.Tensor, torch.Tensor],
        cache_feature: dict | None = None,
        approx_kwargs: dict | None = None,
        correct: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, dict | None]:
        tokens = tokens.view(batch_size * num_frames, num_tokens, embed_dim)
        tokens, cache_feature = self._call_block(
            self.frame_blocks[block_idx], tokens, rope_sincos,
            cache_feature, f"frame{block_idx}", approx_kwargs or {}, correct,
        )
        return tokens, tokens.view(batch_size, num_frames, num_tokens, embed_dim), cache_feature

    def _run_inter_frame_attention_block(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        embed_dim: int,
        block_idx: int,
        attention_type: str,
        cache_feature: dict | None = None,
        approx_kwargs: dict | None = None,
        correct: bool = False,
    ) -> tuple[torch.Tensor, dict | None]:
        tokens = tokens.view(batch_size, num_frames, num_tokens, embed_dim)
        # `interg` / `interr`, not one `inter` stack: the global blocks attend over S*N tokens while
        # the register blocks attend over S*17, so their per-token scores are shaped (1, 8392) and
        # (1, 136). Anything that aggregates across a stack -- the layer-mean pscore, for one --
        # must not mix them, and the tag is what carries that distinction.
        tag = f"inter{'g' if attention_type == 'global' else 'r'}{block_idx}"
        approx_kwargs = approx_kwargs or {}

        if attention_type == "global":
            tokens = tokens.view(batch_size, num_frames * num_tokens, embed_dim)
            tokens, cache_feature = self._call_block(
                self.inter_frame_blocks[block_idx], tokens, None, cache_feature, tag,
                approx_kwargs, correct,
            )
            return tokens.view(batch_size, num_frames, num_tokens, embed_dim), cache_feature

        if attention_type != "register":
            raise ValueError(f"Unknown inter-frame attention type: {attention_type}")

        patch_token_start = self.patch_token_start
        camera_and_register_tokens = tokens[:, :, :patch_token_start].reshape(
            batch_size,
            num_frames * patch_token_start,
            embed_dim,
        )
        patch_tokens = tokens[:, :, patch_token_start:].reshape(
            batch_size,
            num_frames * (num_tokens - patch_token_start),
            embed_dim,
        )

        camera_and_register_tokens, cache_feature = self._call_block(
            self.inter_frame_blocks[block_idx], camera_and_register_tokens, None,
            cache_feature, tag, approx_kwargs, correct,
        )
        tokens = torch.cat([camera_and_register_tokens, patch_tokens], dim=1)

        camera_and_register_tokens = tokens[:, : num_frames * patch_token_start].view(
            batch_size,
            num_frames,
            patch_token_start,
            embed_dim,
        )
        patch_tokens = tokens[:, num_frames * patch_token_start :].view(
            batch_size,
            num_frames,
            num_tokens - patch_token_start,
            embed_dim,
        )
        return torch.cat([camera_and_register_tokens, patch_tokens], dim=2), cache_feature


def _build_patch_embed(patch_size: int, embed_dim: int) -> DinoVisionTransformer:
    model = DinoVisionTransformer(
        img_size=224,
        patch_size=patch_size,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="max",
        pos_embed_rope_dtype="fp32",
        embed_dim=embed_dim,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-5,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
    )
    model.init_weights()
    return model


def slice_expand_and_flatten(token_tensor: torch.Tensor, batch_size: int, num_frames: int) -> torch.Tensor:
    first_frame_token = token_tensor[:, 0:1].expand(batch_size, 1, *token_tensor.shape[2:])
    other_frame_tokens = token_tensor[:, 1:].expand(batch_size, num_frames - 1, *token_tensor.shape[2:])
    tokens = torch.cat([first_frame_token, other_frame_tokens], dim=1)
    return tokens.view(batch_size * num_frames, *tokens.shape[2:])
