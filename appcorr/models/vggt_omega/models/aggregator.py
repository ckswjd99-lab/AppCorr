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
        stack = tag.rstrip("0123456789")
        avail = kwargs.get("arrived_masks", {}).get(stack) if kwargs.get("arrived_masks") else None
        if avail is not None and avail.shape == (rows, n_tok):
            # Interleaved rounds carry only part of the residual, so the candidates are the tokens
            # whose refined values have arrived. Row counts differ wildly under per-frame grouping
            # -- the delivered view has ~1049 available tokens and the rest have only their 17
            # camera/register tokens. Rows used to be padded to the widest by repeating the last
            # index; duplicates are numerically idempotent, but the plan keeps `ratio x padded
            # width` per row, so a 17-token row padded to 1049 was CORRECTED at the wide row's
            # width every round. Measured: correction cost g x the intended amount (0.83x of a
            # full forward at keep=0.25, where ~0.2x was the design), because at g=4 per-frame
            # grouping three of every four rows were pure padding. Rows are now bucketed by
            # candidate count and corrected per bucket, with the tag's cache rows gathered for the
            # call and every mutated entry scattered back -- the m2f batch-cache lesson: scatter
            # back EVERYTHING gathered, or the write lands in a temporary and is thrown away.
            counts = avail.sum(1)
            if int(counts.max().item()) <= 0:
                return x, cache_feature
            return Aggregator._correct_row_buckets(blk, x, avail, counts, rope,
                                                   cache_feature, tag, kwargs)
        else:
            dindice = torch.arange(n_tok, device=x.device, dtype=torch.long).unsqueeze(0).expand(rows, -1)
        extra = {} if rope is not None else {"num_pretokens": 0}
        # The mobile residual hint is laid out per stack, since the three stacks read different token
        # axes. Select by stack here, and *do not* mutate `kwargs` -- `run_blocks` hands the same
        # dict to all 72 blocks, so popping from it delivers the hint to the first block only and
        # leaves the other 71 running unhinted. That produced three separate oracle experiments whose
        # results agreed to three decimals, because the hint they varied was reaching one block.
        hints = kwargs.get("mobile_pscore_hints")
        kwargs = {k: v for k, v in kwargs.items()
                  if k not in ("mobile_pscore_hints", "arrived_masks")}
        if hints is not None:
            extra["mobile_pscore_hint"] = hints.get(tag.rstrip("0123456789"))
        return blk.correct(x, dindice, rope, cache_feature, tag=tag, **kwargs, **extra)

    @staticmethod
    def _correct_row_buckets(blk, x, avail, counts, rope, cache_feature, tag, kwargs):
        """Correct ragged rows bucket-by-bucket instead of padding every row to the widest.

        Rows are grouped by candidate count; each bucket gets its own `blk.correct` call at its own
        true width, on a row-slice of the stream and a row-slice of this tag's cache. Numerically
        this changes nothing (the padded duplicates were idempotent); it changes what is COMPUTED.

        Cache handling follows the m2f batch-cache pattern: every tensor under this tag whose
        leading dim matches the row count is gathered for the call and scattered back afterwards --
        including keys the call CREATED. Global (non-row) entries such as the plan-stat totals are
        left in place and accumulate across buckets exactly as they did across padded rows.
        """
        rows, n_tok = x.shape[0], x.shape[1]
        hints = kwargs.get("mobile_pscore_hints")
        call_kwargs = {k: v for k, v in kwargs.items()
                       if k not in ("mobile_pscore_hints", "arrived_masks")}
        extra = {} if rope is not None else {"num_pretokens": 0}
        if hints is not None:
            extra["mobile_pscore_hint"] = hints.get(tag.rstrip("0123456789"))

        x_out = x.clone()
        prefix = f"{tag}_"
        import os as _os
        if _os.environ.get("APPCORR_VGGT_TRACE"):
            print(f"[bkt-entry] tag={tag} rows={rows} counts={counts.tolist()}", flush=True)
        for k in sorted(set(counts.tolist())):
            if k <= 0:
                continue
            rows_sel = torch.nonzero(counts == k, as_tuple=True)[0]
            dindice = torch.stack([
                torch.nonzero(avail[r], as_tuple=True)[0] for r in rows_sel.tolist()
            ])
            # Row-slice EVERY tensor whose leading dim is the row axis, not just this tag's: the
            # plan resolver pools server pscores ACROSS layers (pe0..pe9), so slicing only the
            # current tag left pe0 at the bucket width while pe1..9 stayed full -- "inconsistent
            # shapes" inside the very first bucket call. Tensors of other stacks that happen to
            # share the row count are sliced too; the block never reads them, and scattering an
            # untouched slice back is a no-op.
            sub_cache = {}
            row_keys = []
            for ck, cv in cache_feature.items():
                if torch.is_tensor(cv) and cv.dim() >= 1 and cv.shape[0] == rows:
                    sub_cache[ck] = cv[rows_sel]
                    row_keys.append(ck)
                else:
                    sub_cache[ck] = cv
            rope_sel = rope   # rope is per-token, shared across rows
            hint = extra.get("mobile_pscore_hint")
            extra_b = dict(extra)
            if hint is not None and torch.is_tensor(hint) and hint.dim() >= 1 and hint.shape[0] == rows:
                extra_b["mobile_pscore_hint"] = hint[rows_sel]
            import os as _os
            _dump = _os.environ.get("APPCORR_SEL_DUMP")
            if _dump:
                # Pairs with the block-level dump line that follows: the block records LOCAL row
                # ids (it sees only this bucket's slice), this records the local->global mapping.
                with open(_dump, "a") as _f:
                    _f.write(f"MAP\t{tag}\t{rows_sel.tolist()}\n")
            x_b, sub_cache = blk.correct(x[rows_sel], dindice, rope_sel, sub_cache,
                                         tag=tag, **call_kwargs, **extra_b)
            x_out[rows_sel] = x_b
            # Scatter back EVERY row-shaped entry under this tag -- both the ones gathered and the
            # ones the call created (their dim0 equals this bucket's row count).
            n_b = rows_sel.numel()
            import os as _os
            _trace = _os.environ.get("APPCORR_VGGT_TRACE")
            for ck, cv in sub_cache.items():
                if not torch.is_tensor(cv):
                    if ck not in cache_feature:
                        cache_feature[ck] = cv
                    continue
                if ck in row_keys:
                    base = cache_feature[ck]
                    if cv.shape[0] != n_b or cv.shape[1:] != base.shape[1:]:
                        # The call REPLACED this entry with a different shape -- a derived/plan
                        # cache (they are keyed by dindice content and rebuilt per call), not a
                        # row-axed state tensor. Writing it back raggedly is what produced
                        # "value [1032] cannot broadcast to [1, 0]"; and storing a bucket-local
                        # version globally would hand the next bucket a wrong-shaped entry. The
                        # row-axed state that must persist (k/v, blocks_out_sum, per-layer scores)
                        # always keeps its trailing shape, so it never lands here.
                        continue
                    base = base.clone() if not base.is_contiguous() else base
                    base[rows_sel] = cv
                    cache_feature[ck] = base
                elif cv.dim() >= 1 and cv.shape[0] == n_b and ck not in cache_feature:
                    full = cv.new_zeros((rows,) + tuple(cv.shape[1:]))
                    full[rows_sel] = cv
                    cache_feature[ck] = full
                else:
                    # Neither a gathered row key nor a fresh row-shaped creation: a global entry
                    # (plan stats, scalars) the call updated -- store as-is.
                    cache_feature[ck] = cv
        return x_out, cache_feature

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

    # The network is one 48-stage sequence, not a preprocessor followed by a network:
    #
    #     stages  0..23   patch-embed ViT blocks
    #     stages 24..47   aggregator pairs (frame block + inter-frame block)
    #
    # `patch_embed` is named for what it occupies in the original design -- a projection from image
    # patches -- but here it is a full 24-block DINOv3 ViT and 28.7% of the forward. Treating it as
    # preprocessing means re-running all of it on every interleaved round, which costs more than the
    # network it is supposedly preparing for. It is real computation and is scheduled as such.
    PE_STAGES = 24

    def embed_prologue(self, images: torch.Tensor):
        """Normalise, project to patch tokens, and note the geometry. No transformer blocks."""
        batch_size, num_frames, num_channels, height, width = images.shape
        if num_channels != 3:
            raise ValueError(f"Expected 3 input channels, got {num_channels}")

        flat = ((images - self._resnet_mean) / self._resnet_std).view(
            batch_size * num_frames, num_channels, height, width
        )
        pe_tokens, pe_grid = self.patch_embed.prepare_tokens_with_masks(flat, None)
        geom = {
            "batch_size": batch_size,
            "num_frames": num_frames,
            "patch_grid_size": (height // self.patch_size, width // self.patch_size),
            "pe_grid": pe_grid,
        }
        return pe_tokens, geom

    def run_pe_blocks(self, x, geom, block_range, cache_feature=None, approx_kwargs=None,
                      correct=False):
        """Patch-embed ViT blocks over `[start, end)` of its 24."""
        pe = self.patch_embed
        gh, gw = geom["pe_grid"]
        start, end = block_range
        for idx in range(start, end):
            rope = pe.rope_embed(H=gh, W=gw) if pe.rope_embed is not None else None
            x, cache_feature = self._call_block(
                pe.blocks[idx], x, rope, cache_feature, f"pe{idx}", approx_kwargs or {}, correct
            )
        return x, cache_feature

    def assemble_tokens(self, pe_tokens, geom):
        """Finish the patch-embed stack and prepend the camera and register tokens."""
        batch_size, num_frames = geom["batch_size"], geom["num_frames"]
        patch_tokens = self.patch_embed.post_features_list([pe_tokens], [None])[0][
            "x_norm_patchtokens"
        ]
        camera_token = slice_expand_and_flatten(self.camera_token, batch_size, num_frames)
        register_token = slice_expand_and_flatten(self.register_token, batch_size, num_frames)
        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)

        _, num_tokens, embed_dim = tokens.shape
        grid_h, grid_w = geom["patch_grid_size"]
        with torch.no_grad():
            rope_sin, rope_cos = self.rope_embed(H=grid_h, W=grid_w)
            frame_rope = (
                rope_sin.to(device=tokens.device, dtype=torch.float32),
                rope_cos.to(device=tokens.device, dtype=torch.float32),
            )
        geom = dict(geom, num_tokens=num_tokens, embed_dim=embed_dim)
        return tokens, frame_rope, geom

    def embed(
        self,
        images: torch.Tensor,
        cache_feature: dict | None = None,
        approx_kwargs: dict | None = None,
        correct: bool = False,
    ) -> tuple[torch.Tensor, tuple, dict]:
        """Whole prologue in one call: prologue, all patch-embed blocks, then assembly.

        Kept for the stock forward and for callers that do not stage the depth.
        """
        pe_tokens, geom = self.embed_prologue(images)
        pe_tokens, _ = self.run_pe_blocks(
            pe_tokens, geom, (0, self.PE_STAGES), cache_feature, approx_kwargs, correct
        )
        return self.assemble_tokens(pe_tokens, geom)

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
