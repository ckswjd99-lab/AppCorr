"""
backbone.py

Wraps the stock `Qwen3_5MoeVisionModel` with approx/correct forward passes built from
`ApproxCorrectQwen35VisionBlock`. Same layer-range-chunked contract as every prior fork, so the
existing `GroupTriggerPolicy` drives this tower with no new scheduling code.

Descended from `appcorr/models/qwen25vl/vision/backbone.py`, and mostly SHORTER than it, because
two of that file's three complications do not exist here:

  - **No window-index permutation.** Qwen2.5-VL laid patches out in merge-group-major window order
    and had to permute `hidden_states` and the rotary embeddings before the block loop, then
    un-permute at the merger, carrying `window_index`/`inv_window_index` through everything --
    including `correct_forward`, where an ORIGINAL merge-group index had to be mapped to its
    destination slot in the permuted sequence. Qwen3.5 has `window_size=None`: patches stay in
    natural raster order, so a merge group `g` simply owns rows `g*unit ... g*unit+unit-1`, and the
    merger output needs no reordering.
  - **No per-layer attention dispatch.** With `fullatt_block_indexes=None` every layer is full
    per-image attention, so there is a single `segment_ranges` for the whole tower instead of one
    list per layer kind.

What is NEW relative to 2.5 is the interpolated position embedding: Qwen3.5 adds a learned
`pos_embed` (bilinearly resampled to the image's grid) on top of `patch_embed`'s output. Like the
rotary embedding it is a pure function of `grid_thw`, so it is computed once in
`prepare_full_tokens` and is always exact -- it never depends on which patches have arrived, which
is the same "cheap non-block ops are always exact" rule every other fork follows.

**Correction granularity is the merge group**, as in 2.5: `spatial_merge_unit` (= 4) raw patch rows
are combined nonlinearly by the merger into one LLM token, so a half-corrected merge group is not a
meaningful state. `correct_forward` takes merge-group indices and expands them to raw rows.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from transformers.vision_utils import (
    get_vision_bilinear_indices_and_weights,
    get_vision_cu_seqlens,
    get_vision_position_ids,
)

from .attention import ApproxCorrectQwen35VisionAttention
from .block import ApproxCorrectQwen35VisionBlock


class ApproxCorrectQwen35VisionTower(nn.Module):
    def __init__(self, vision_tower: nn.Module):
        super().__init__()
        self.patch_embed = vision_tower.patch_embed
        self.pos_embed = vision_tower.pos_embed
        self.rotary_pos_emb = vision_tower.rotary_pos_emb
        self.merger = vision_tower.merger
        self.spatial_merge_size = vision_tower.spatial_merge_size
        self.spatial_merge_unit = vision_tower.spatial_merge_unit
        self.num_grid_per_side = vision_tower.num_grid_per_side
        self.blocks = nn.ModuleList(
            [ApproxCorrectQwen35VisionBlock.from_stock(b) for b in vision_tower.blocks]
        )
        # Asserted, not assumed. The whole simplification above rests on this tower being
        # unwindowed; if a future checkpoint reintroduces windowing, this fork would silently
        # compute full attention where stock computes windowed and every number would be wrong
        # while looking fine. Research code -- crash instead.
        cfg = vision_tower.config
        if getattr(cfg, "window_size", None) or getattr(cfg, "fullatt_block_indexes", None):
            raise ValueError(
                f"qwen35 vision fork assumes an unwindowed tower, but config has "
                f"window_size={getattr(cfg, 'window_size', None)!r} "
                f"fullatt_block_indexes={getattr(cfg, 'fullatt_block_indexes', None)!r}. "
                "Port the window permutation from appcorr/models/qwen25vl/vision/backbone.py."
            )

    def prepare_grid(self, grid_thw: torch.Tensor, device) -> Dict[str, Any]:
        """Everything `prepare_full_tokens` derives from `grid_thw` alone: interpolated pos_embed,
        rotary cos/sin, cu_seqlens and the CPU-side segment ranges (the `.tolist()` syncs live
        here). Computed once per request; `embed()` then turns each pixel tensor (full image,
        degraded base) into a residual stream against the same grid context -- the streaming
        axis prepares two images per request and shared none of this before (2026-09-08)."""
        bilinear_indices, bilinear_weights = get_vision_bilinear_indices_and_weights(
            grid_thw, num_grid_per_side=self.num_grid_per_side,
            spatial_merge_size=self.spatial_merge_size,
        )
        position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size)
        cu_seqlens = get_vision_cu_seqlens(grid_thw).to(device)

        pos_embeds = (self.pos_embed(bilinear_indices) * bilinear_weights[:, :, None]).sum(0)
        seq_len = pos_embeds.shape[0]
        rotary_pos_emb = self.rotary_pos_emb(position_ids).reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        # The one GPU->CPU sync, done once per request rather than once per layer per round.
        segment_ranges = ApproxCorrectQwen35VisionAttention.segment_ranges_from_cu_seqlens(cu_seqlens)

        return {
            "pos_embeds": pos_embeds,
            "position_embeddings": position_embeddings,
            "cu_seqlens": cu_seqlens,
            "segment_ranges": segment_ranges,
            "seq_len": seq_len,
        }

    def embed(self, pixel_values: torch.Tensor, gctx: Dict[str, Any]) -> torch.Tensor:
        """patch_embed + the grid's interpolated pos_embed -> [seq_len, dim] layer-0 stream."""
        hidden_states = self.patch_embed(pixel_values)
        hidden_states = hidden_states + gctx["pos_embeds"].to(hidden_states.dtype)
        if hidden_states.shape[0] != gctx["seq_len"]:
            raise ValueError(f"{hidden_states.shape[0]} patch rows for a grid of {gctx['seq_len']}")
        return hidden_states.reshape(gctx["seq_len"], -1)

    def prepare_full_tokens(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor,
                            gctx: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """patch_embed + interpolated pos_embed -> rotary embeddings -> segment boundaries.

        Always exact: every tensor here depends only on `grid_thw`, never on which patches have
        arrived. Mirrors stock `Qwen3_5MoeVisionModel.forward` up to the block loop. Pass a
        `prepare_grid` result as `gctx` to reuse it across images of the same grid.
        """
        if gctx is None:
            gctx = self.prepare_grid(grid_thw, pixel_values.device)
        return {"hidden_states": self.embed(pixel_values, gctx), **gctx}

    def approx_forward(self, x_feature: torch.Tensor, start_l: int, end_l: int, ctx: Dict[str, Any],
                       cache_feature: Dict[str, Any], tag_prefix: str,
                       collect_attn_mean: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Runs blocks[start_l:end_l] in approx mode."""
        for i in range(start_l, end_l):
            x_feature, cache_feature = self.blocks[i].approx(
                x_feature, ctx["segment_ranges"], ctx["position_embeddings"],
                cache_feature, tag=f"{tag_prefix}_layer{i}", collect_attn_mean=collect_attn_mean,
            )
        return x_feature, cache_feature

    def _token_idx(self, group_idx: torch.Tensor, device) -> torch.Tensor:
        unit = self.spatial_merge_unit
        group_idx = group_idx.to(device)
        return (group_idx.unsqueeze(1) * unit + torch.arange(unit, device=device)).flatten()

    def correct_plan(self, token_idx: torch.Tensor, ctx: Dict[str, Any]):
        """Per round, the segment split of the query rows for `attention.correct(plan=)`:
        `"single"` for a one-segment request (no sync at all), else the Qwen2.5-VL shape
        `(order, inv_order, [(start, length, a, b), ...])` from ONE `.tolist()`."""
        segs = ctx["segment_ranges"]
        if len(segs) == 1:
            return "single"
        import bisect
        order = torch.argsort(token_idx)
        inv_order = torch.empty_like(order)
        inv_order[order] = torch.arange(order.numel(), device=order.device)
        pos = token_idx[order].tolist()
        owned = []
        for start, length in segs:
            a = bisect.bisect_left(pos, start)
            b = bisect.bisect_left(pos, start + length)
            if b > a:
                owned.append((start, length, a, b))
        return order, inv_order, owned

    def correct_forward(self, x_feature: torch.Tensor, group_idx: torch.Tensor, start_l: int, end_l: int,
                        ctx: Dict[str, Any], cache_feature: Dict[str, Any],
                        tag_prefix: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            x_feature: current residual stream, restarted from `prepare_full_tokens()`'s output at
                the start of EACH correction round.
            group_idx: [G] long -- merge-group indices that received new data this round, 0-indexed
                into the `seq_len // spatial_merge_unit` groups. In NATURAL order: unlike the 2.5
                fork there is no permutation between a group's index and its rows.
        """
        token_idx = self._token_idx(group_idx, x_feature.device)
        cos_full, sin_full = ctx["position_embeddings"]
        position_embeddings_sel = (cos_full[token_idx], sin_full[token_idx])
        plan = self.correct_plan(token_idx, ctx)

        for i in range(start_l, end_l):
            x_feature, cache_feature = self.blocks[i].correct(
                x_feature, token_idx, ctx["segment_ranges"], position_embeddings_sel,
                cache_feature, tag=f"{tag_prefix}_layer{i}", plan=plan,
            )
        return x_feature, cache_feature

    def correct_rows(self, x0_full: torch.Tensor, group_idx: torch.Tensor, ctx: Dict[str, Any],
                     cache_feature: Dict[str, Any], tag_prefix: str,
                     span=None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Full-depth correction of `group_idx`'s rows only: gathers their layer-0 rows from
        `x0_full` (the [T, dim] layer-0 stream; only these rows are read) and returns the
        corrected last-layer rows [G * unit, dim] in group-major order, i.e. exactly
        `correct_forward(x0_full, group_idx, 0, depth, ...)[0][token_idx]` -- bitwise -- without
        materialising the other T - Q rows at any layer. See `block.correct_rows` for the
        contract (each group corrected once, non-corrected rows never read) that makes the
        omission sound; the streaming axis is the caller that satisfies it.
        """
        token_idx = self._token_idx(group_idx, x0_full.device)
        cos_full, sin_full = ctx["position_embeddings"]
        plan = self.correct_plan(token_idx, ctx)
        if span is not None:
            # `span=(lo, hi)`: the caller knows `group_idx` is the contiguous group range
            # [lo/unit, hi/unit) (a keep=1.0 streaming band), so every row gather here and every
            # K/V scatter in the layers is a plain slice instead of an index kernel. Same rows,
            # same values -- bitwise with the index path.
            lo, hi = span
            position_embeddings_sel = (cos_full[lo:hi], sin_full[lo:hi])
            x_rows = x0_full[lo:hi]
        else:
            position_embeddings_sel = (cos_full[token_idx], sin_full[token_idx])
            x_rows = x0_full[token_idx]
        for i, blk in enumerate(self.blocks):
            x_rows, cache_feature = blk.correct_rows(
                x_rows, token_idx, ctx["segment_ranges"], position_embeddings_sel,
                cache_feature, tag=f"{tag_prefix}_layer{i}", plan=plan, span=span,
            )
        return x_rows, cache_feature

    def finalize_attn_layermean(self, cache_feature: Dict[str, Any], tag_prefix: str,
                                n_layers: int) -> Dict[str, Any]:
        """Average the per-layer received-attention vectors into `{tag_prefix}_attn_layermean` [T].

        Every layer of this tower is full attention, so all of them are comparable and all of them
        are averaged -- there is no windowed subset to exclude (contrast Qwen2.5-VL, where only the
        4 `fullatt_block_indexes` layers may contribute).

        Raises if a layer that should have collected did not: a silently short average would still
        produce a plausible ranking, and the selection would quietly be driven by whichever layers
        happened to run.
        """
        keys = [f"{tag_prefix}_layer{i}_attn_mean" for i in range(n_layers)]
        missing = [k for k in keys if k not in cache_feature]
        if missing:
            raise KeyError(
                f"qwen35 finalize_attn_layermean: {len(missing)} of {n_layers} layers did not "
                f"collect received attention (first missing: {missing[0]}). approx_forward must be "
                "called with collect_attn_mean=True over the FULL depth before this."
            )
        acc = None
        for k in keys:
            v = cache_feature[k]
            acc = v.float() if acc is None else acc + v.float()
        cache_feature[f"{tag_prefix}_attn_layermean"] = acc / n_layers
        return cache_feature

    def get_merged_output(self, x_full: torch.Tensor, ctx: Dict[str, Any]) -> torch.Tensor:
        """merger() over the last layer's [seq_len, dim] hidden state. No un-permutation: rows are
        already in natural order (see the module docstring)."""
        return self.merger(x_full)
