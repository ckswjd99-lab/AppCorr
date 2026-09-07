"""
unified.py

The Qwen2.5-VL progressive-arrival axis as ONE in-process object -- vision tower fork
(`ApproxCorrectQwen25VLVisionTower`, the same fork the offload executor drives) + streaming LLM --
on the shared `QwenVLStreamingAxis` loop. Added 2026-09-07 for the vLLM two-process campaign:
the executor path (`offload/server/model/qwen25vl_executor.py`, transmission policy + GroupTrigger
schedule + HF LLM fork) is what the table's Qwen2.5-VL rows were measured with, but it runs the
LLM inside the same process and decodes its first token by hand; the axis form is what can hand
its chunks to a serving engine. Same vision fork, same "correct band r, merge band r, prefill band
r" schedule, so the vision side of an axis run is the executor's `llm_schedule=streaming`,
`token_keep_ratio=keep` arm with the transmission layer replaced by `degrade()`.

What differs from Qwen3.5 in this file is only the tower's row order: Qwen2.5-VL permutes patch
rows by attention window (`window_index`) before its block loop and un-permutes only at the
merged output, so a merge group's rows sit at `inv_window_index[g] * unit + [0, unit)`. The base
loop asks for rows through `_rows_of_groups`; that one mapping is the whole port.

Received-attention for the keep<1 arms is collected on the 4 full-attention layers only (the
tower's own rule -- windowed layers' per-window mass is not on the same scale; see
`vision/backbone.py`'s `approx_forward` docstring).
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from ..qwen_vl_axis import QwenVLStreamingAxis
from .vision.backbone import ApproxCorrectQwen25VLVisionTower

MODEL_ID_7B = "Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_ID_32B = "Qwen/Qwen2.5-VL-32B-Instruct"   # the table's Qwen2.5-VL (33.5B) rows


class Qwen25VLAxis(QwenVLStreamingAxis):
    """Qwen2.5-VL (3B/7B/32B/72B): windowed tower, rows in window-permuted order."""

    def _make_tower(self, model: nn.Module) -> nn.Module:
        return ApproxCorrectQwen25VLVisionTower(model.model.visual)

    def _approx_base(self, ctx_base: Dict[str, Any], cache: Dict[str, Any],
                     collect_attn: bool) -> Tuple[torch.Tensor, Dict[str, Any]]:
        # Finalisation (divide by the number of full-attention layers walked) happens inside
        # approx_forward when the call covers the full depth, which this one always does.
        return self.tower.approx_forward(
            ctx_base["hidden_states"], 0, len(self.tower.blocks), ctx_base, cache, "v",
            collect_attn=collect_attn)

    def _attn_layermean(self, cache: Dict[str, Any]) -> torch.Tensor:
        return cache["vision_patch_attn_layermean"]

    def _rows_of_groups(self, ctx: Dict[str, Any], group_idx: torch.Tensor) -> torch.Tensor:
        unit = self.tower.spatial_merge_unit
        inv = ctx["inv_window_index"]
        slots = inv[group_idx.to(inv.device)]
        return (slots.unsqueeze(1) * unit + torch.arange(unit, device=slots.device)).flatten()
