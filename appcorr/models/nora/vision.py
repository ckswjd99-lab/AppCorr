"""Approximate/correct execution for NORA's Qwen2.5-VL vision encoder.

Qwen groups each 2x2 set of raw ViT patches before window permutation and
merging.  The public correction index is therefore a raster-ordered merged-cell
index.  For NORA's 224x224 input there are 64 cells, each backed by four of the
256 raw tokens.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    apply_rotary_pos_emb_vision,
)


class ApproxCorrectQwenVision(nn.Module):
    """Stateful wrapper around a stock Qwen2.5-VL vision transformer."""

    def __init__(self, visual: nn.Module) -> None:
        super().__init__()
        self.visual = visual
        self.merge_unit = int(visual.spatial_merge_unit)

    def _layout(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x = self.visual.patch_embed(pixel_values)
        rope = self.visual.rot_pos_emb(grid_thw)
        window_index, cu_window = self.visual.get_window_index(grid_thw)
        window_index = window_index.to(x.device)
        num_raw = x.shape[0]
        num_cells = num_raw // self.merge_unit

        x = x.reshape(num_cells, self.merge_unit, -1)
        x = x[window_index].reshape(num_raw, -1)
        rope = rope.reshape(num_cells, self.merge_unit, -1)
        rope = rope[window_index].reshape(num_raw, -1)
        emb = torch.cat((rope, rope), dim=-1)

        cu_window = torch.tensor(
            cu_window,
            dtype=torch.int32,
            device=x.device,
        ).unique_consecutive()
        window_ids = torch.empty(num_raw, dtype=torch.long, device=x.device)
        for window_id, (start, end) in enumerate(
            zip(cu_window[:-1].tolist(), cu_window[1:].tolist())
        ):
            window_ids[start:end] = window_id

        return x, {
            "cos": emb.cos(),
            "sin": emb.sin(),
            "window_index": window_index,
            "raster_to_internal": torch.argsort(window_index),
            "window_ids": window_ids,
        }

    @staticmethod
    def _qkv(
        block: nn.Module,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_heads = block.attn.num_heads
        q, k, v = (
            block.attn.qkv(x)
            .reshape(x.shape[0], 3, num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
        return q, k, v

    @staticmethod
    def _attention(
        block: nn.Module,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        allowed: torch.Tensor | None,
    ) -> torch.Tensor:
        # [tokens, heads, dim] -> [heads, tokens, dim]
        qh = q.transpose(0, 1)
        kh = k.transpose(0, 1)
        vh = v.transpose(0, 1)
        mask = None if allowed is None else allowed.unsqueeze(0)
        out = F.scaled_dot_product_attention(qh, kh, vh, attn_mask=mask)
        out = out.transpose(0, 1).reshape(q.shape[0], -1)
        return block.attn.proj(out)

    def _allowed_full(
        self,
        layout: Dict[str, torch.Tensor],
        query_idx: torch.Tensor | None,
        *,
        full_attention: bool,
    ) -> torch.Tensor | None:
        if full_attention:
            return None
        window_ids = layout["window_ids"]
        query_windows = (
            window_ids if query_idx is None else window_ids[query_idx]
        )
        return query_windows[:, None] == window_ids[None, :]

    def approx(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        cache: Dict[str, Any] | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        cache = {} if cache is None else cache
        x, layout = self._layout(pixel_values, grid_thw)
        cache["layout"] = layout

        for layer_idx, block in enumerate(self.visual.blocks):
            tag = f"layer{layer_idx}"
            norm = block.norm1(x)
            q, k, v = self._qkv(
                block,
                norm,
                layout["cos"],
                layout["sin"],
            )
            cache[f"{tag}_kv"] = torch.stack([k, v], dim=1)
            allowed = self._allowed_full(
                layout,
                None,
                full_attention=layer_idx in self.visual.fullatt_block_indexes,
            )
            attn = self._attention(block, q, k, v, allowed=allowed)
            x_mid = x + attn
            mlp = block.mlp(block.norm2(x_mid))
            cache[f"{tag}_delta"] = (attn + mlp).detach().clone()
            x = x_mid + mlp

        merged_internal = self.visual.merger(x)
        merged = merged_internal[layout["raster_to_internal"]]
        cache["merged"] = merged.detach().clone()
        return merged, cache

    def cell_raw_indices(
        self,
        cell_idx: torch.Tensor,
        cache: Dict[str, Any],
    ) -> torch.Tensor:
        cell_idx = cell_idx.to(
            device=cache["layout"]["raster_to_internal"].device,
            dtype=torch.long,
        )
        internal = cache["layout"]["raster_to_internal"][cell_idx]
        offsets = torch.arange(
            self.merge_unit,
            device=internal.device,
            dtype=torch.long,
        )
        return (internal[:, None] * self.merge_unit + offsets).reshape(-1)

    def correct(
        self,
        pixel_values: torch.Tensor,
        grid_thw: torch.Tensor,
        cell_idx: torch.Tensor,
        cache: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        x, layout_now = self._layout(pixel_values, grid_thw)
        layout = cache["layout"]
        if not torch.equal(layout_now["window_index"], layout["window_index"]):
            raise ValueError("Qwen vision grid/window layout changed within a session")

        cell_idx = cell_idx.to(device=x.device, dtype=torch.long)
        raw_idx = self.cell_raw_indices(cell_idx, cache)
        for layer_idx, block in enumerate(self.visual.blocks):
            tag = f"layer{layer_idx}"
            active = x[raw_idx]
            norm = block.norm1(active)
            q, k, v = self._qkv(
                block,
                norm,
                layout["cos"][raw_idx],
                layout["sin"][raw_idx],
            )
            kv = cache[f"{tag}_kv"]
            kv[raw_idx, 0] = k.to(kv.dtype)
            kv[raw_idx, 1] = v.to(kv.dtype)
            cache[f"{tag}_kv"] = kv
            k_full, v_full = kv.unbind(1)
            allowed = self._allowed_full(
                layout,
                raw_idx,
                full_attention=layer_idx in self.visual.fullatt_block_indexes,
            )
            attn = self._attention(
                block,
                q,
                k_full,
                v_full,
                allowed=allowed,
            )
            active_mid = active + attn
            mlp = block.mlp(block.norm2(active_mid))

            x = x + cache[f"{tag}_delta"].to(x.dtype)
            x[raw_idx] = (active_mid + mlp).to(x.dtype)

        internal_cells = layout["raster_to_internal"][cell_idx]
        cell_hidden = x.reshape(-1, self.merge_unit, x.shape[-1])[
            internal_cells
        ].reshape(-1, x.shape[-1])
        corrected = self.visual.merger(cell_hidden)
        cache["merged"][cell_idx] = corrected.to(cache["merged"].dtype)
        return cache["merged"], cache
