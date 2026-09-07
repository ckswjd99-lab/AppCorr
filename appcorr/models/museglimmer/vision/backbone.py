"""
backbone.py

Wraps the stock `MuseGlimmerVisionModel` (plus the surrounding adapter/projection/norm chain from
`MuseGlimmerModel`) with approx/correct forward passes built from
`ApproxCorrectMuseGlimmerVisionBlock`. Descended from
`appcorr/models/qwen25vl/vision/backbone.py` -- MG's vision is the same window-permuted,
per-layer-dispatched family -- with these MG-specific facts baked in:

  - **Row-granular window permutation.** MG runs the tower with `spatial_merge_size=1`: the
    window_index permutes RAW rows (no merge-group-major reshape before permuting, unlike 2.5),
    and the merge is deferred to `pixel_shuffle` after the un-permute.
  - **The LLM-token group is a pixel-shuffle block, not a contiguous row run.** LLM token `g` owns
    the raw rows `shuffle_index[g*unit : (g+1)*unit]` (`unit = merge_size**2 = 4`), where
    `shuffle_index` (stock `get_vision_pixel_shuffle_index`) walks merge_size x merge_size spatial
    blocks in row-major block order. Those rows are NOT contiguous in natural order; correction
    and merging always go through `shuffle_index`, then through `inv_window_index` to land in the
    permuted stream.
  - **ln_pre before the permutation, ln_post + pixel_shuffle + adapter + projection + norm after
    the un-permute.** All of it is per-row / per-group and therefore band-sliceable exactly:
    `merge_groups(g0, g1)` reproduces stock's output rows [g0, g1) bit-for-bit given the same
    stream rows.
  - **Per-layer dispatch from `config.layer_types`** ("full_attention" -> per-image cu_seqlens,
    "window_attention" -> per-window cu_window_seqlens). Received-attention collection is
    restricted to the full-attention layers, same scale argument as the 2.5 fork.
  - **2D RoPE via the stock rotary module**: position_ids = get_vision_position_ids(...).flip(-1)+1
    (stock's "+1 offset" quirk), permuted by window_index, then `rotary_emb(h, position_ids)`
    yields (cos, sin) for the whole request -- pure functions of grid_thw, computed once.

This module is constructed from a `MuseGlimmerForConditionalGeneration` and steals references to
the stock submodules (no copies).
"""

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from transformers.vision_utils import (
    get_vision_cu_seqlens,
    get_vision_position_ids,
    get_vision_window_index,
)
from transformers.models.muse_glimmer.modeling_muse_glimmer import get_vision_pixel_shuffle_index

from .attention import ApproxCorrectMuseGlimmerVisionAttention
from .block import ApproxCorrectMuseGlimmerVisionBlock


class ApproxCorrectMuseGlimmerVisionTower(nn.Module):
    def __init__(self, mg_model: nn.Module):
        """`mg_model` is the inner `MuseGlimmerModel` (model.model on the ForConditionalGeneration)."""
        super().__init__()
        vt = mg_model.vision_tower
        self.patch_embedder = vt.patch_embedder
        self.rotary_emb = vt.rotary_emb
        self.ln_pre = vt.ln_pre
        self.ln_post = vt.ln_post
        self.window_size = vt.window_size
        self.patch_size = vt.patch_size
        self.merge_size = vt.merge_size
        self.spatial_merge_unit = vt.merge_size ** 2
        self.layer_types = list(vt.config.layer_types)
        self.blocks = nn.ModuleList(
            [ApproxCorrectMuseGlimmerVisionBlock.from_stock(b) for b in vt.layers])
        # The chain stock applies AFTER pixel_shuffle, in MuseGlimmerModel.get_image_features:
        self.vision_adapter = mg_model.vision_adapter
        self.vision_projection = mg_model.vision_projection
        self.perception_emb_norm = mg_model.perception_emb_norm

    @torch.no_grad()
    def prepare_full_tokens(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> Dict[str, Any]:
        """patch_embedder (incl. interpolated pos table) -> ln_pre -> window permutation, plus every
        grid-shape-derived tensor the walk needs. Always exact -- nothing here depends on which
        patches have arrived."""
        device = pixel_values.device
        cu_seqlens = get_vision_cu_seqlens(grid_thw)
        window_index, cu_window_seqlens = get_vision_window_index(
            grid_thw, spatial_merge_size=1, window_size=self.window_size, patch_size=self.patch_size)
        inv_window_index = torch.argsort(window_index)
        shuffle_index = get_vision_pixel_shuffle_index(grid_thw, self.merge_size)

        emb = self.patch_embedder(pixel_values, grid_thw)
        h = self.ln_pre(emb)
        seq_len = h.shape[0]
        h = h[window_index]

        position_ids = get_vision_position_ids(grid_thw, spatial_merge_size=1)
        position_ids = position_ids.flip(-1) + 1          # stock's reference-offset quirk
        position_ids = position_ids[window_index]
        cos, sin = self.rotary_emb(h, position_ids)

        cu_seqlens = cu_seqlens.to(device)
        cu_window_seqlens = cu_window_seqlens.to(device)
        A = ApproxCorrectMuseGlimmerVisionAttention
        return {
            "hidden_states": h,
            "position_embeddings": (cos, sin),
            "cu_seqlens_ranges": A._segment_ranges(cu_seqlens),
            "cu_window_seqlens_ranges": A._segment_ranges(cu_window_seqlens),
            "window_index": window_index.to(device),
            "inv_window_index": inv_window_index.to(device),
            "shuffle_index": shuffle_index.to(device),
            "seq_len": seq_len,
        }

    def _segments_for_layer(self, layer_idx: int, ctx: Dict[str, Any]):
        return (ctx["cu_seqlens_ranges"] if self.layer_types[layer_idx] == "full_attention"
                else ctx["cu_window_seqlens_ranges"])

    def group_rows_permuted(self, group_idx: torch.Tensor, ctx: Dict[str, Any]) -> torch.Tensor:
        """LLM-token indices [G] -> the [G*unit] permuted-stream row indices they own."""
        unit = self.spatial_merge_unit
        shuffle_index = ctx["shuffle_index"]
        group_idx = group_idx.to(shuffle_index.device)
        rows_nat = (shuffle_index.reshape(-1, unit)[group_idx]).reshape(-1)
        return ctx["inv_window_index"][rows_nat]

    def approx_forward(self, x_feature: torch.Tensor, start_l: int, end_l: int,
                       ctx: Dict[str, Any], cache_feature: Dict[str, Any], tag_prefix: str,
                       collect_attn_mean: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        full_layers = [i for i in range(len(self.blocks)) if self.layer_types[i] == "full_attention"]
        for i in range(start_l, end_l):
            blk = self.blocks[i]
            segs_now = self._segments_for_layer(i, ctx)
            do_collect = collect_attn_mean and (i in full_layers)
            x_feature, cache_feature = blk.approx(
                x_feature, segs_now, ctx["position_embeddings"], cache_feature,
                tag=f"{tag_prefix}_layer{i}", collect_attn=do_collect)
        return x_feature, cache_feature

    def finalize_attn_layermean(self, cache_feature: Dict[str, Any], tag_prefix: str,
                                n_layers_walked: int) -> Dict[str, Any]:
        acc = cache_feature.pop("vision_patch_attn_layermean_acc", None)
        n = cache_feature.pop("vision_patch_attn_layermean_n", 0)
        if acc is not None and n > 0:
            cache_feature[f"{tag_prefix}_attn_layermean"] = acc / n
        return cache_feature

    def correct_forward(self, x_feature: torch.Tensor, group_idx: torch.Tensor,
                        start_l: int, end_l: int, ctx: Dict[str, Any],
                        cache_feature: Dict[str, Any], tag_prefix: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """`group_idx`: [G] long, LLM-token (pixel-shuffle-group) indices that received new data."""
        token_idx = self.group_rows_permuted(group_idx, ctx)
        cos_full, sin_full = ctx["position_embeddings"]
        pe_sel = (cos_full[token_idx], sin_full[token_idx])
        for i in range(start_l, end_l):
            blk = self.blocks[i]
            segs_now = self._segments_for_layer(i, ctx)
            x_feature, cache_feature = blk.correct(
                x_feature, token_idx, segs_now, pe_sel, cache_feature, tag=f"{tag_prefix}_layer{i}")
        return x_feature, cache_feature

    def merge_groups(self, x_perm: torch.Tensor, ctx: Dict[str, Any], g0: int, g1: int) -> torch.Tensor:
        """The stock post-tower chain for LLM tokens [g0, g1): un-permute + gather the groups' raw
        rows in shuffle order -> ln_post -> pixel_shuffle fold -> adapter -> projection ->
        perception norm. Exactly stock's `get_image_features` restricted to those output rows --
        every op past the gather is per-row/per-group, so band slicing is exact."""
        unit = self.spatial_merge_unit
        shuffle_index = ctx["shuffle_index"]
        rows_nat = shuffle_index[g0 * unit:g1 * unit]
        rows_perm = ctx["inv_window_index"][rows_nat]
        h = self.ln_post(x_perm[rows_perm])
        dim = h.shape[-1]
        h = h.view(-1, unit, dim).permute(0, 2, 1).reshape(-1, dim * unit)
        return self.perception_emb_norm(self.vision_projection(self.vision_adapter(h)))
