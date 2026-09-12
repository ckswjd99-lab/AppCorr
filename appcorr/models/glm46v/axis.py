"""
axis.py

The GLM-4.6V (106B-A12B, FP8) progressive-arrival axis: `QwenVLStreamingAxis` with the GLM tower
and the four per-model hooks, exactly as `appcorr/models/qwen35/unified.py::Qwen35Axis` does for
Qwen3.5. Everything the shared loop does -- band splitting, the merge-group row math, the
selection score, the four `llm_schedule`s, the M-RoPE closed form -- is inherited unchanged, and
the survey behind that claim is `docs/memo/glm46v_port_plan.md`. The three things that ARE
GLM-specific:

**Token ids and the prompt.** The image placeholder is `<|image|>` 151363 (`config.image_token_id`
-- the base class reads it from the config, so nothing to parameterise), and the run is FLANKED by
two ordinary text tokens, `<|begin_of_image|>` 151339 at `lo-1` and `<|end_of_image|>` 151340 at
`lo+n`. That is the same shape as Qwen2.5-VL's vision_start/vision_end, so `_image_token_run` and
the band/chunk math need no change -- but it is now checked rather than assumed
(`_image_token_run` below), because a template that dropped the trailing sentinel would leave the
interleaved schedule's last round with no text suffix to release the held-back row. The chat
template is mandatory: it opens with `[gMASK]<sop>` (151331, 151333) at positions 0-1 and closes
with `<|assistant|>`; `enable_thinking=False` appends `/nothink` (151360) to the user turn AND
emits an empty `<think></think>` pair after `<|assistant|>` (chat_template.jinja lines 61 and 140
of the GLM-4.6V-FP8 snapshot -- the memo mentions only the first).

**M-RoPE.** `mrope_section [8, 12, 12]`, partial rotary 0.5, theta 5e5 -- all of that is the
ENGINE's business (vLLM's `get_mrope_input_positions`; this side only produces the (3, N) position
tensor). `Glm4vMoeModel.get_rope_index` (:1106-1198) computes it exactly as Qwen2.5-VL does for a
single image -- t = lo, h = lo + row, w = lo + col over the MERGED grid, text after the run
resuming at lo + max(h_m, w_m) -- so `_positions_fast` is inherited verbatim. `positions_mode =
"check"` asserts that against `get_rope_index` on the real model whenever a gate wants it.

**Base-image resolution.** `Glm46VImageProcessor` has NO `min_pixels` / `max_pixels` kwargs (and
the processor class is `Glm46VProcessor`, not `Glm4vProcessor` -- do not isinstance-check the
latter): the pixel budget is the `size` dict's `shortest_edge` / `longest_edge`, which are AREAS
in pixels (12544 and 9,633,792 in this checkpoint) fed to `smart_resize(factor=28)`. The campaign's
level-2 base keeps the geometry and degrades the content (`qwen35_accuracy.degrade`), so it does
not touch this at all; `low_res_inputs()` below is for the arms that want a genuinely smaller
grid, and it goes through `size`.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..qwen_vl_axis import QwenVLStreamingAxis
from .vision.backbone import ApproxCorrectGlm4vVisionTower

MODEL_ID_GLM46V_FP8 = "zai-org/GLM-4.6V-FP8"

IMAGE_TOKEN_ID = 151363          # <|image|>            == config.image_token_id
IMAGE_START_TOKEN_ID = 151339    # <|begin_of_image|>   == config.image_start_token_id
IMAGE_END_TOKEN_ID = 151340      # <|end_of_image|>     == config.image_end_token_id


class Glm46VAxis(QwenVLStreamingAxis):
    """GLM-4.6V / GLM-4.5V (`Glm4vMoeForConditionalGeneration`): unwindowed 24-layer tower, rows
    in natural merge-group-major order, 46 pure-softmax MoE decoder layers."""

    def _make_tower(self, model: nn.Module) -> nn.Module:
        return ApproxCorrectGlm4vVisionTower(model.model.visual)

    def _chat_template_kwargs(self, think: bool = False, **kw) -> Dict[str, Any]:
        # Thinking defaults OFF for the same reason it does on Qwen3.5: the template opens a
        # reasoning block when it is on, and a 24-token greedy decode then spends its whole
        # budget on preamble without reaching the answer, which scores as a model property and
        # is a truncation artifact. With `enable_thinking=False` GLM's template appends
        # `/nothink` to the user turn and emits `<think></think>` after `<|assistant|>`, i.e.
        # an explicitly CLOSED empty reasoning block -- the answer starts at the first
        # generated token. Pass think=True only from a driver that decodes past the block.
        #
        # Anything else the caller passes (`size=` from `low_res_inputs`) rides through to
        # `apply_chat_template` -> the processor -> `Glm46VImageProcessor`; the Qwen3.5 hook
        # swallows its kwargs, this one does not.
        return {"enable_thinking": think, **kw}

    # --- prompt layout ------------------------------------------------------------------------ #

    def _image_token_run(self, input_ids: torch.Tensor) -> Tuple[int, int]:
        """(start, count) of the single contiguous `<|image|>` run, with GLM's two sentinels and
        the trailing text checked.

        The base class's contiguity check is necessary but not sufficient here. Three extra
        invariants the rest of the axis silently relies on:
          * `lo >= 1` and `ids[lo-1] == <|begin_of_image|>`: the leading-text chunk of the
            streaming schedule is `[0, lo)` and must carry the opening sentinel;
          * `ids[lo+n] == <|end_of_image|>`: the closing sentinel is a TEXT row, so it belongs to
            the trailing chunk, not to the image band -- if it were inside the run the band's
            merge would write over it;
          * there is at least one row after the run: the interleaved schedule holds row `seq-1`
            back and releases it with the final round's text suffix `[lo+G, seq-1)`, which must be
            non-empty (`streaming_forward` raises at that point otherwise -- better to fail here,
            on the prompt, than half way through a request).
        """
        lo, n_tok = super()._image_token_run(input_ids)
        ids = input_ids[0]
        seq = int(ids.shape[0])
        start_id = int(getattr(self.cfg, "image_start_token_id", IMAGE_START_TOKEN_ID))
        end_id = int(getattr(self.cfg, "image_end_token_id", IMAGE_END_TOKEN_ID))
        if lo < 1 or int(ids[lo - 1]) != start_id:
            raise ValueError(f"image run starts at {lo} but ids[{lo - 1}] is "
                             f"{int(ids[lo - 1]) if lo else None}, not <|begin_of_image|> {start_id}")
        if lo + n_tok >= seq or int(ids[lo + n_tok]) != end_id:
            raise ValueError(f"ids[{lo + n_tok}] is not <|end_of_image|> {end_id}: the closing "
                             "sentinel must be a text row after the run")
        if seq - (lo + n_tok) < 2:
            raise ValueError(f"only {seq - lo - n_tok} rows after the image run; the interleaved "
                             "schedule needs a non-empty text suffix plus the held-back last row")
        return lo, n_tok

    @torch.no_grad()
    def low_res_inputs(self, image, question: str, longest_edge: int,
                       shortest_edge: Optional[int] = None, **kw) -> Dict[str, Any]:
        """`build_inputs` at a reduced PIXEL-AREA budget, through the image processor's `size`.

        `Glm46VImageProcessor` takes `size={"shortest_edge": min_area, "longest_edge": max_area}`
        (areas, not side lengths) and feeds them to `smart_resize(factor=28)`; there is no
        `min_pixels` / `max_pixels` kwarg on this family. Lowering `longest_edge` gives a SMALLER
        GRID, i.e. fewer merge groups -- which is NOT what the streaming axis's `px_base` is
        (that keeps the grid and degrades the content, so the two images' rows correspond). Use
        this for arms that deliberately change the token count, never for `px_base`.
        """
        ip = self.processor.image_processor
        size = dict(ip.size)
        size["longest_edge"] = int(longest_edge)
        if shortest_edge is not None:
            size["shortest_edge"] = int(shortest_edge)
        # `processor_kwargs=` rather than a bare `size=`: transformers 5.13 still honours the
        # bare form but warns that it is going away. Checked on a 700x900 image -- default grid
        # 50x64, `longest_edge=50176` -> 18x24 (2026-09-12).
        return self.build_inputs(image, question, processor_kwargs={"size": size}, **kw)

    # --- the four per-model hooks -------------------------------------------------------------- #

    supports_deferred_pscore = True

    def _approx_base(self, ctx_base: Dict[str, Any], cache: Dict[str, Any],
                     collect_attn) -> Tuple[torch.Tensor, Dict[str, Any]]:
        x, cache = self.tower.approx_forward(
            ctx_base["hidden_states"], 0, len(self.tower.blocks), ctx_base, cache, "v",
            collect_attn_mean=collect_attn)
        if collect_attn is True:
            cache = self.tower.finalize_attn_layermean(cache, "v", len(self.tower.blocks))
        return x, cache

    def _attn_layermean(self, cache: Dict[str, Any]) -> torch.Tensor:
        return cache["v_attn_layermean"]

    def _attn_layermean_deferred(self, cache: Dict[str, Any],
                                 ctx_base: Dict[str, Any]) -> torch.Tensor:
        cache = self.tower.deferred_attn_layermean(cache, "v", len(self.tower.blocks), ctx_base)
        return cache["v_attn_layermean"]

    # _rows_of_groups: base-class identity (no window permutation in this tower).

    # --- unified stage axis -------------------------------------------------------------------- #
    #
    # The two cost hooks below are the AXIS's own closed form: they decide where the `groups`
    # round boundaries fall over the `24 + 46` stage axis. The REPORTING closed form (plan item
    # 5) is the engine-side agent's `appcorr/models/glm46v/unified.py` -- the same terms again,
    # as pure functions plus a `Glm46VUnifiedCosts` mixin. Two copies of one formula is a merge
    # artefact, not a design: `tests/test_glm46v_flops_closed_form.py::AxisVsReporting
    # ClosedFormTest` asserts they are equal to the float (and equal to
    # `flops_analytic.Glm46VDecoder` / `Glm46VVision`), so whichever copy survives the merge,
    # changing a term here without changing it there fails that test. Run it after any edit.
    supports_unified_axis = True

    def _approx_range(self, ctx: Dict[str, Any], cache: Dict[str, Any], start_l: int, end_l: int,
                      x=None, collect_attn: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        return self.tower.approx_forward(
            ctx["hidden_states"] if x is None else x, start_l, end_l, ctx, cache, "v",
            collect_attn_mean=bool(collect_attn))

    def _attn_layermean_prefix(self, cache: Dict[str, Any], n_layers: int) -> torch.Tensor:
        return self.tower.prefix_attn_layermean(cache, "v", n_layers)

    def _vision_stage_cost(self, n_rows: int) -> float:
        """FLOPs of ONE tower layer over `n_rows` patch rows: fused qkv + full bidirectional
        attention + out proj + the GATED (SwiGLU) MLP -- three GEMMs, not Qwen3.5's two, because
        `Glm4vMoeisionMlp` is `down(silu(gate(x)) * up(x))` with intermediate = out_hidden_size
        (4096) and hidden 1536. 2 FLOPs per MAC; norms, softmax and the SiLU excluded, the
        convention `appcorr/flops/hooks.py` follows.
        """
        v = self.cfg.vision_config
        h, heads = int(v.hidden_size), int(v.num_heads)
        ffn = int(v.out_hidden_size)          # the vision MLP's intermediate width (NOT
        #                                       intermediate_size, which is the MERGER's 10944)
        return float(2 * n_rows * h * (3 * h)                          # fused qkv
                     + 2 * 2 * heads * n_rows * n_rows * (h // heads)  # QK^T + AV
                     + 2 * n_rows * h * h                              # out proj
                     + 3 * 2 * n_rows * h * ffn)                       # gate + up + down

    def _llm_stage_costs(self, n_prompt: int) -> list:
        """Per decoder layer over an `n_prompt`-token prefill, from the text config alone.

        All 46 layers are PURE SOFTMAX GQA (96 q / 8 kv heads, head_dim 128, partial rotary 0.5,
        no qk-norm, no output gate): there is no Gated DeltaNet anywhere in this model, so unlike
        Qwen3.5's the list is uniform except for layer 0. `first_k_dense_replace = 1` makes layer
        0 a dense SwiGLU at `intermediate_size` (10944) and layers 1-45 MoE (128 routed experts,
        top-8 at `moe_intermediate_size` 1408, plus a shared expert of
        `moe_intermediate_size * n_shared_experts` and the 128-way router).

        `attention_bias: true` adds q/k/v bias vectors; those are adds, not MACs, and are
        excluded here under the same convention that excludes the norms and the softmax.
        """
        t = self.cfg.text_config
        L, H = int(t.num_hidden_layers), int(t.hidden_size)
        heads, kv_heads = int(t.num_attention_heads), int(t.num_key_value_heads)
        dh = int(getattr(t, "head_dim", H // heads))
        n_exp = int(getattr(t, "n_routed_experts", 0) or 0)
        dense_first = int(getattr(t, "first_k_dense_replace", 0) or 0)
        moe_i = int(getattr(t, "moe_intermediate_size", 0) or 0)
        shared = moe_i * int(getattr(t, "n_shared_experts", 0) or 0)
        dense = 3 * 2 * H * int(t.intermediate_size)
        moe = (2 * int(t.num_experts_per_tok) * 3 * moe_i * H     # routed top-k experts
               + 2 * H * n_exp                                   # router
               + 3 * 2 * H * shared)                             # shared expert
        attn = 2 * H * (heads * dh + 2 * kv_heads * dh) + 2 * (heads * dh) * H   # q,k,v + o
        quad = 2 * 2 * heads * n_prompt * n_prompt * dh                          # QK^T + AV
        return [float(n_prompt * (attn + (dense if i < dense_first or not n_exp else moe)) + quad)
                for i in range(L)]
