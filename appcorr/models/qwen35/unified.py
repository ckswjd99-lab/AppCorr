"""
unified.py

The Qwen3.5-MoE (35B-A3B) progressive-arrival axis: vision tower + MoE decoder as one in-process
object, following `appcorr/models/ov2/unified.py` (whose streaming arm this reuses almost verbatim)
and `appcorr/models/gemma3/unified.py` (whose FLOP-scope conventions it follows).

**The arm this model gets, and why it is not OV2's menu.** OV2 carries both `interleaved_forward`
(approx-then-correct on BOTH halves) and `streaming_forward`, and streaming won (ChartQA 85.0 vs
81.0). Qwen3.5 does not get the choice: 30 of its 40 decoder layers are recurrent
(GatedDeltaNet), so an LLM-side approx-then-correct is not merely worse, it is ill-defined --
correcting token i rewrites the state every later token consumes (see `llm/streaming.py`). So the
LLM half streams, full stop, and only the vision half runs approximate-then-correct:

    base image     -> full 27-layer vision approx, caching K/V for every patch row
    band r arrives -> vision-correct band r's merge groups to full depth against the stored K/V
                      (which already carries earlier bands' corrections), re-merge that band,
                      prefill the LLM for exactly that band's token positions
    trailing text  -> prefilled last, against fully-arrived state, same arrival as the final band

What is given up is the same thing OV2's streaming gives up, stated plainly: vision attention is
bidirectional, so band r's features would keep improving as later bands land -- but band r is
prefilled and consumed before that. The LLM sees a stale view of every band but the last. The
degenerate case g=1 has no staleness at all (one band, corrected after everything arrived), which
is what makes `streaming_forward(groups=1)` an exact identity against `full_forward` -- the gate
this file ships with.

**Chunk contiguity is load-bearing.** Bands must be contiguous runs of image-token positions
(sequential grouping), because an LLM chunk is appended to a cache. This is the documented
constraint from the LLM-interleaved work, and it is asserted here, not assumed.

The streaming loop itself now lives in `appcorr/models/qwen_vl_axis.py` (`QwenVLStreamingAxis`),
shared with Qwen2.5-VL since 2026-09-07; this class supplies the Qwen3.5 tower and its
attention-mean bookkeeping. Refactor gated bitwise on the 35B (greedy tokens + image-embedding
sums, g=4 at keep 1.0 and 0.5, 4 COCO images) -- see docs/memo/vllm_stream_design.md.
"""

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn

from ..qwen_vl_axis import QwenVLStreamingAxis
from .vision.backbone import ApproxCorrectQwen35VisionTower

MODEL_ID_35B = "Qwen/Qwen3.5-35B-A3B"
# The FP8 variant is the only 122B that fits one 183GB device (bf16 would need ~244GB). Same
# vision tower as the 35B (verified identical configs), same qwen3_5_moe architecture -- the
# tower's unwindowed assert and the per-checkpoint gate are what stand between "same in the
# config" and "same in the shipped weights".
MODEL_ID_122B_FP8 = "Qwen/Qwen3.5-122B-A10B-FP8"


class Qwen35Axis(QwenVLStreamingAxis):
    """Qwen3.5 (35B-A3B / 122B-A10B-FP8): unwindowed tower, rows in natural order."""

    def _make_tower(self, model: nn.Module) -> nn.Module:
        return ApproxCorrectQwen35VisionTower(model.model.visual)

    def _chat_template_kwargs(self, think: bool = False, **kw) -> Dict[str, Any]:
        # `think` defaults OFF: Qwen3.5's template opens a `<think>` block when thinking is
        # enabled, and a short greedy decode then spends its whole budget on reasoning preamble
        # without ever reaching the answer -- measured as the 35B scoring 18% on RealWorldQA MCQ,
        # which is a truncated-thought artifact, not a model property. Short-answer evals score
        # the ANSWER, so thinking stays off; pass think=True only from a driver that decodes past
        # the block.
        return {"enable_thinking": think}

    def _approx_base(self, ctx_base: Dict[str, Any], cache: Dict[str, Any],
                     collect_attn: bool) -> Tuple[torch.Tensor, Dict[str, Any]]:
        x, cache = self.tower.approx_forward(
            ctx_base["hidden_states"], 0, len(self.tower.blocks), ctx_base, cache, "v",
            collect_attn_mean=collect_attn)
        if collect_attn:
            cache = self.tower.finalize_attn_layermean(cache, "v", len(self.tower.blocks))
        return x, cache

    def _attn_layermean(self, cache: Dict[str, Any]) -> torch.Tensor:
        return cache["v_attn_layermean"]

    # _rows_of_groups: base-class identity (no window permutation in this tower).
