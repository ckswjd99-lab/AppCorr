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

from typing import Any, Dict, Tuple

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

    def prepare_full_tokens(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> Dict[str, Any]:
        """patch_embed + interpolated pos_embed -> rotary embeddings -> segment boundaries.

        Always exact: every tensor here depends only on `grid_thw`, never on which patches have
        arrived. Mirrors stock `Qwen3_5MoeVisionModel.forward` up to the block loop.
        """
        device = pixel_values.device
        bilinear_indices, bilinear_weights = get_vision_bilinear_indices_and_weights(
            grid_thw, num_grid_per_side=self.num_grid_per_side,
            spatial_merge_size=self.spatial_merge_size,
        )
        position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size)
        cu_seqlens = get_vision_cu_seqlens(grid_thw).to(device)

        hidden_states = self.patch_embed(pixel_values)
        pos_embeds = (self.pos_embed(bilinear_indices) * bilinear_weights[:, :, None]).sum(0)
        hidden_states = hidden_states + pos_embeds.to(hidden_states.dtype)

        seq_len = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = self.rotary_pos_emb(position_ids).reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        # The one GPU->CPU sync, done once per request rather than once per layer per round.
        segment_ranges = ApproxCorrectQwen35VisionAttention.segment_ranges_from_cu_seqlens(cu_seqlens)

        return {
            "hidden_states": hidden_states,
            "position_embeddings": position_embeddings,
            "cu_seqlens": cu_seqlens,
            "segment_ranges": segment_ranges,
            "seq_len": seq_len,
        }

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
        unit = self.spatial_merge_unit
        group_idx = group_idx.to(x_feature.device)
        token_idx = (group_idx.unsqueeze(1) * unit
                     + torch.arange(unit, device=group_idx.device)).flatten()

        cos_full, sin_full = ctx["position_embeddings"]
        position_embeddings_sel = (cos_full[token_idx], sin_full[token_idx])

        for i in range(start_l, end_l):
            x_feature, cache_feature = self.blocks[i].correct(
                x_feature, token_idx, ctx["segment_ranges"], position_embeddings_sel,
                cache_feature, tag=f"{tag_prefix}_layer{i}",
            )
        return x_feature, cache_feature

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
