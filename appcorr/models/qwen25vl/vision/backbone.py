"""
backbone.py

Wraps the stock (pretrained) `Qwen2_5_VisionTransformerPretrainedModel` with approx/correct forward
passes built from `ApproxCorrectQwen25VLVisionBlock`. Mirrors
`appcorr/models/openclip/vision/backbone.py`'s layer-chunked contract, adapted for:

  - **Window-index permutation**: patches are laid out in merge-group-major order by `patch_embed`
    (see `transformers.vision_utils.get_vision_window_index` -- reused directly, no fork needed, a
    pure function of `grid_thw` + config constants). `hidden_states`/rotary embeddings are permuted
    by `window_index` BEFORE the block loop and only un-permuted for the final merged output
    (`last_hidden_state` stays in permuted order even in STOCK code -- confirmed against the Phase 0
    oracle). `inv_window_index = argsort(window_index)` serves two purposes here: (a) mapping an
    ORIGINAL merge-group index to its PERMUTED-sequence destination slot (for `.correct()`'s
    `token_idx` construction), and (b) un-permuting the merger's output at the very end -- both are
    literally the same tensor, computed once in `prepare_full_tokens`.
  - **Merge-group granularity**: the smallest independently-correctable unit is a
    `spatial_merge_unit` (=4) run of raw patch rows -- i.e. exactly the group that
    `Qwen2_5_VLPatchMerger` will nonlinearly combine into one output token. `correct_forward` takes
    ORIGINAL (pre-permutation) merge-group indices as `group_idx` and expands each to its 4
    constituent raw-patch rows in the permuted sequence.
  - **No CLS token, no truncation**: this tower has no prefix/CLS token at all (unlike DINOv2's 5 or
    CLIP's 1) and needs its FULL depth (32 layers) -- there is no "second-to-last block" shortcut
    (OpenVLA's DINOv2/SigLIP truncation) and no single CLS row to force-include (CLIP's invariant) --
    every merge-group is either corrected or stale, with no special permanently-corrected token.
  - **Per-layer full-vs-windowed attention dispatch**: `fullatt_block_indexes` (class default
    `(7,15,23,31)` out of 32) use `cu_seqlens` (per-image boundaries); all other layers use
    `cu_window_seqlens` (per-window boundaries) -- both threaded per-layer into approx()/correct().
"""

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from transformers.vision_utils import get_vision_cu_seqlens, get_vision_position_ids, get_vision_window_index

from .block import ApproxCorrectQwen25VLVisionBlock


class ApproxCorrectQwen25VLVisionTower(nn.Module):
    def __init__(self, vision_tower: nn.Module):
        super().__init__()
        self.patch_embed = vision_tower.patch_embed
        self.rotary_pos_emb = vision_tower.rotary_pos_emb
        self.merger = vision_tower.merger
        self.spatial_merge_size = vision_tower.spatial_merge_size
        self.spatial_merge_unit = vision_tower.spatial_merge_unit
        self.window_size = vision_tower.window_size
        self.patch_size = vision_tower.patch_size
        self.fullatt_block_indexes = set(vision_tower.fullatt_block_indexes)
        self.blocks = nn.ModuleList([ApproxCorrectQwen25VLVisionBlock.from_stock(b) for b in vision_tower.blocks])

    def prepare_full_tokens(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> Dict[str, Any]:
        """patch_embed -> window-index permutation of tokens + rotary embeddings -> segment
        boundaries. Always exact -- tokenization/windowing depend only on grid_thw, never on
        which patches have "arrived" yet, matching every prior fork's "cheap non-block ops are
        always exact" philosophy. Returns a dict bundling everything correct_forward/approx_forward
        need, since (unlike prior forks) there are several grid-shape-derived tensors to thread
        through together."""
        device = pixel_values.device
        position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size)
        cu_seqlens = get_vision_cu_seqlens(grid_thw)
        window_index, cu_window_seqlens = get_vision_window_index(
            grid_thw, self.spatial_merge_size, self.window_size, self.patch_size
        )
        inv_window_index = torch.argsort(window_index)

        hidden_states = self.patch_embed(pixel_values)
        seq_len = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        hidden_states = hidden_states[window_index, :, :].reshape(seq_len, -1)

        rotary_pos_emb = self.rotary_pos_emb(position_ids)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
        rotary_pos_emb = rotary_pos_emb[window_index, :, :].reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        return {
            "hidden_states": hidden_states,
            "position_embeddings": position_embeddings,
            "cu_seqlens": cu_seqlens.to(device),
            "cu_window_seqlens": cu_window_seqlens.to(device),
            "window_index": window_index.to(device),
            "inv_window_index": inv_window_index.to(device),
            "seq_len": seq_len,
        }

    def _cu_for_layer(self, layer_idx: int, ctx: Dict[str, Any]) -> torch.Tensor:
        return ctx["cu_seqlens"] if layer_idx in self.fullatt_block_indexes else ctx["cu_window_seqlens"]

    def approx_forward(
        self,
        x_feature: torch.Tensor,
        start_l: int,
        end_l: int,
        ctx: Dict[str, Any],
        cache_feature: Dict[str, Any],
        tag_prefix: str,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Runs blocks[start_l:end_l] in approx mode. Layer-range chunked (start_l/end_l), matching
        the contract every prior fork uses so the existing GroupTriggerPolicy can drive this tower
        with no new scheduling code."""
        for i in range(start_l, end_l):
            blk = self.blocks[i]
            cu_now = self._cu_for_layer(i, ctx)
            x_feature, cache_feature = blk.approx(
                x_feature, cu_now, ctx["position_embeddings"], cache_feature, tag=f"{tag_prefix}_layer{i}"
            )
        return x_feature, cache_feature

    def correct_forward(
        self,
        x_feature: torch.Tensor,
        group_idx: torch.Tensor,
        start_l: int,
        end_l: int,
        ctx: Dict[str, Any],
        cache_feature: Dict[str, Any],
        tag_prefix: str,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Args:
            x_feature: current residual stream (permuted-sequence order), restarted from
                `prepare_full_tokens()`'s output at the start of EACH correction round.
            group_idx: [G] long -- ORIGINAL (pre-window-permutation) merge-group indices that
                received new data this round (0-indexed into the `seq_len // spatial_merge_unit`
                merge groups, in patch_embed's native output order).
        """
        inv_window_index = ctx["inv_window_index"]
        dest_slots = inv_window_index[group_idx.to(inv_window_index.device)]  # permuted-seq group slots
        unit = self.spatial_merge_unit
        token_idx = (dest_slots.unsqueeze(1) * unit + torch.arange(unit, device=dest_slots.device)).flatten()

        cos_full, sin_full = ctx["position_embeddings"]
        position_embeddings_sel = (cos_full[token_idx], sin_full[token_idx])

        for i in range(start_l, end_l):
            blk = self.blocks[i]
            cu_now = self._cu_for_layer(i, ctx)
            x_feature, cache_feature = blk.correct(
                x_feature, token_idx, cu_now, position_embeddings_sel, cache_feature, tag=f"{tag_prefix}_layer{i}"
            )
        return x_feature, cache_feature

    def get_merged_output(self, x_full: torch.Tensor, ctx: Dict[str, Any]) -> torch.Tensor:
        """merger() -> un-permute by inv_window_index (== stock's `reverse_indices`). `x_full` is
        the LAST layer's [seq_len, dim] hidden state, in permuted-sequence order."""
        merged = self.merger(x_full)
        return merged[ctx["inv_window_index"]]
