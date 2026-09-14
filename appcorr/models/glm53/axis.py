"""
axis.py

The GLM-5.3-Flash progressive-arrival axis: `Glm46VAxis` with the GLM-5.3 tower and the five
places where the two GLM families differ. Everything the shared loop does -- band splitting, the
merge-group row math, the selection score, the four `llm_schedule`s -- is inherited unchanged.
Survey: `docs/memo/glm53_vllm_survey.md`; the tower's own differences: `vision/backbone.py`.

**Token ids and the prompt.** The placeholder is `<|image|>` 154854 (`config.image_token_id`),
flanked by `<|begin_of_image|>` 154830 at `lo-1` and `<|end_of_image|>` 154831 at `lo+n`, i.e. the
GLM-4.6V shape at new ids -- so `_image_token_run`'s three extra invariants are inherited, only
the fallback constants change. The chat template (`chat_template.jinja` in the snapshot) opens
with `[gMASK]<sop>` (154822, 154824) and a `<|system|>Reasoning Effort: Max` line, and its
generation prompt is `<|assistant|><think>`.

**Thinking is not a template switch on this model.** GLM-4.6V took `enable_thinking=False`, which
appended `/nothink` and emitted an empty `<think></think>`. GLM-5.3-Flash's template has NO
`enable_thinking` and no `/nothink` (grep of the snapshot's template: 0 hits for either; the only
switches are `clear_thinking`, which rewrites HISTORY only, and `reasoning_effort`, which is
forced to one of low/high/max and always emits its system line). Its generation prompt
unconditionally ends `<|assistant|><think>`, so a short greedy decode spends its whole budget
inside an open reasoning block. The non-thinking form is therefore ours to make: append
`</think>` (154842) to the tokenised prompt, closing the block the template opened, so the answer
starts at the first generated token. `Glm53Composer` does the same thing on the vLLM side, one
level up (on the template TEXT, before the processor runs) -- the two must agree, and
`tests/test_glm53_prompt.py` asserts they do.

**No M-RoPE.** The decoder has no rotary anywhere: the 34 KDA layers ignore `positions`
(`vllm/models/glm5next/nvidia/kda.py:285-289`) and the 11 MLA layers are built with
`skip_rope` (`attention.py:495-520`, `qk_rope_head_dim == 0` in this checkpoint). `positions` is
still LIVE -- the sparse indexer consumes it for tail-pool slots and the short-prefill causal
fill -- but it is the plain 1-D counter the engine already computes from `num_computed_tokens`.
So `uses_mrope = False` and `_positions` returns `(None, 0)`:

  * **what the engine's `open` push expects**: `mrope=None`. Not a broadcast 1-D tensor in (3, T)
    clothing. `GPUModelRunner._init_mrope_positions` is called only under `if self.uses_mrope`
    (vLLM main `v1/worker/gpu_model_runner.py:1343-1345` and `:1676-1677`), and it is the only
    thing that ever fills `CachedRequestState.mrope_positions`; our runner patch then asserts
    `st.mrope_positions is not None` before extending it with a chunk's positions
    (`appcorr/vllm_stream/runner_patch.py:45-48`). Pushing a tensor would therefore assert-fail on
    the FIRST appended chunk of every streaming request. With `None` the runner derives positions
    the stock way and they are exactly the 1-D sequence this model wants.
  * on the HF backend the same flag passes `position_ids=None`, which
    `Glm5NextTextModel.forward` turns into `arange(T) + past_seen` (transformers 5.16.1
    `modeling_glm5_next.py:1450-1453`) -- the reference this axis would otherwise have to
    reproduce. There is no `get_rope_index` on this model class, so `positions_mode="reference"`
    and `"check"` are not available and say so.

**Base-image resolution.** The pixel budget is NOT the `size` dict: `Glm5NextImageProcessor.size`
is `{"longest_edge": 1}` with a `# TODO` saying it is unused, and both HF and vLLM read
`min_image_tokens` / `max_image_tokens` (TOKEN counts) instead. `smart_resize` turns them into
pixels as `tokens * temporal_patch_size * (patch_size * merge_size * patch_expand_factor)**2`
and compares that against `aligned_frames(=2 for a still) * H * W`, so the spatial area cap is
`max_image_tokens * (patch*merge)**2 = 8000 * 784 = 6,272,000` px -- that is the number
`FAMILY_MAX_PX["glm53"]` carries. `low_res_inputs` below goes through `max_image_tokens`.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..glm46v.axis import Glm46VAxis
from .vision.backbone import ApproxCorrectGlm5NextVisionTower

MODEL_ID_GLM53_FP8 = "zai-org/GLM-5.3-Flash"

IMAGE_TOKEN_ID = 154854          # <|image|>            == config.image_token_id
IMAGE_START_TOKEN_ID = 154830    # <|begin_of_image|>   == config.image_start_token_id
IMAGE_END_TOKEN_ID = 154831      # <|end_of_image|>     == config.image_end_token_id
THINK_OPEN_TOKEN_ID = 154841     # <think>   -- the generation prompt's last token
THINK_CLOSE_TOKEN_ID = 154842    # </think>  -- what we append to close it
BOX_TOKEN_IDS = (154852, 154853)  # <|begin_of_box|> / <|end_of_box|>, stripped by `clean_text`

# The image processor's spatial pixel-area cap (see the module docstring).
MAX_IMAGE_TOKENS = 8000
MAX_PIXELS = MAX_IMAGE_TOKENS * (14 * 2) ** 2     # 6_272_000


class Glm53Axis(Glm46VAxis):
    """GLM-5.3-Flash (`Glm5NextForConditionalGeneration`): unwindowed 24-layer tower at hidden
    1024, rows in natural merge-group-major order, 45 hybrid decoder layers (34 KDA + 11 sparse
    MLA, mHC residual streams) with no rotary."""

    # See the module docstring: the text model takes a plain 1-D counter, and a streaming chunk
    # must carry `mrope=None`.
    uses_mrope = False

    # The unified vision+decoder axis needs a per-decoder-layer FLOP closed form and the engine's
    # what the terms are; the engine half is `docs/memo/glm53_correct_design.md` items 1-3, owned
    # elsewhere), so the unified arm is refused rather than run on a guessed cost model.
    supports_unified_axis = True     # the two cost hooks exist (2026-09-13)

    IMAGE_START_ID = IMAGE_START_TOKEN_ID
    IMAGE_END_ID = IMAGE_END_TOKEN_ID

    def _make_tower(self, model: nn.Module) -> nn.Module:
        return ApproxCorrectGlm5NextVisionTower(model.model.visual)

    # --- prompt ------------------------------------------------------------------------------- #

    def _chat_template_kwargs(self, think: bool = False, **kw) -> Dict[str, Any]:
        """GLM-5.3's template has no thinking switch, so `think` is handled in `build_inputs`
        (by appending `</think>`), not here. Anything else the caller passes
        (`processor_kwargs={"max_image_tokens": ...}` from `low_res_inputs`) rides through to
        `apply_chat_template` -> the processor."""
        self._think = bool(think)
        return dict(kw)

    @torch.no_grad()
    def build_inputs(self, image, question: str, **kw) -> Dict[str, Any]:
        """`Glm46VAxis.build_inputs` + the `</think>` suffix when thinking is off.

        Appended as a TOKEN ID rather than re-tokenising the template text: the template's
        generation prompt already ends with `<think>` (154841), so the only edit is one more text
        row, and appending the id cannot perturb the tokenisation of anything before it. Every
        per-token field the axis carries is extended in step; `input_ids` is the only one the
        band math reads, but `attention_mask` / `mm_token_type_ids` would be silently short
        otherwise and `full_forward` passes them to the model.
        """
        inputs = super().build_inputs(image, question, **kw)
        if getattr(self, "_think", False):
            return inputs
        ids = inputs["input_ids"]
        if int(ids[0, -1]) != THINK_OPEN_TOKEN_ID:
            raise ValueError(
                f"prompt ends with token {int(ids[0, -1])}, not <think> {THINK_OPEN_TOKEN_ID}: "
                "GLM-5.3-Flash's chat template is expected to close with `<|assistant|><think>` "
                "(add_generation_prompt=True). Re-check chat_template.jinja before appending.")
        dev = ids.device
        inputs["input_ids"] = torch.cat(
            [ids, torch.full((ids.shape[0], 1), THINK_CLOSE_TOKEN_ID, dtype=ids.dtype, device=dev)],
            dim=1)
        for key in ("attention_mask", "mm_token_type_ids", "token_type_ids"):
            v = inputs.get(key)
            if isinstance(v, torch.Tensor) and v.ndim == 2 and v.shape[1] == ids.shape[1]:
                pad = torch.ones_like(v[:, :1]) if key == "attention_mask" else torch.zeros_like(v[:, :1])
                inputs[key] = torch.cat([v, pad], dim=1)
        return inputs

    # --- positions ----------------------------------------------------------------------------- #

    def _positions(self, inputs: Dict[str, Any], image_run: Optional[Tuple[int, int]] = None,
                   grid_thw: Optional[Tuple[int, int, int]] = None) -> Tuple[None, int]:
        """No M-RoPE on this model (see the module docstring). Returns `(None, 0)`: no position
        tensor is pushed or passed, and the decode counter continues at `seq` (`rope_delta + seq`
        in `streaming_forward`), which is what a 1-D counter does."""
        if self.positions_mode != "fast":
            raise ValueError(
                f"positions_mode {self.positions_mode!r} is not available on GLM-5.3-Flash: the "
                "model class has no `get_rope_index` (it has no rotary at all), so there is no "
                "transformers reference to check a closed form against.")
        return None, 0

    def _positions_reference(self, inputs: Dict[str, Any]):
        raise NotImplementedError(
            "Glm5NextModel has no get_rope_index: the decoder has no rotary and the engine "
            "assigns plain sequential positions. See Glm53Axis._positions.")

    # --- base-image resolution ------------------------------------------------------------------ #

    @torch.no_grad()
    def low_res_inputs(self, image, question: str, max_image_tokens: int,
                       min_image_tokens: Optional[int] = None, **kw) -> Dict[str, Any]:
        """`build_inputs` at a reduced TOKEN budget, through the image processor's
        `max_image_tokens`.

        `Glm5NextImageProcessor` has no `min_pixels` / `max_pixels` kwarg and its `size` dict is a
        placeholder (`{"longest_edge": 1}`, with a `# TODO` in transformers 5.16.1 saying it is
        unused): the budget is `min_image_tokens` / `max_image_tokens`, which `smart_resize`
        multiplies by `temporal_patch_size * (patch_size * merge_size * patch_expand_factor)**2`
        to get a pixel budget. Lowering `max_image_tokens` gives a SMALLER GRID, i.e. fewer merge
        groups -- which is NOT what the streaming axis's `px_base` is (that keeps the grid and
        degrades the content, so the two images' rows correspond). Use this for arms that
        deliberately change the token count, never for `px_base`.
        """
        pk = {"max_image_tokens": int(max_image_tokens)}
        if min_image_tokens is not None:
            pk["min_image_tokens"] = int(min_image_tokens)
        return self.build_inputs(image, question, processor_kwargs=pk, **kw)

    # --- the unified stage axis ------------------------------------------------------------------ #

    def _vision_stage_cost(self, n_rows: int) -> float:
        """FLOPs of ONE tower layer over `n_rows` patch rows: fused qkv + full bidirectional
        attention + out proj + the clamped SwiGLU MLP (three GEMMs). 2 FLOPs per MAC; norms
        (including the per-head q/k RMSNorm), the softmax, the SiLU and the clamps are excluded,
        the convention `appcorr/flops/hooks.py` follows.

        The one term that differs in SHAPE from the GLM-4.6V copy: the vision MLP's intermediate
        width is `vision_config.intermediate_size` (4096) here, whereas on GLM-4.6V the vision MLP
        used `out_hidden_size` and `intermediate_size` was the MERGER's context dim. GLM-5.3 gives
        the merger its own field (`projection_intermediate_size`, 10240), so the two are no longer
        aliased and reading the wrong one would misprice every vision stage by 4096/10240.
        """
        v = self.cfg.vision_config
        h, heads = int(v.hidden_size), int(v.num_heads)
        ffn = int(v.intermediate_size)        # 4096 -- the vision MLP's own width
        return float(2 * n_rows * h * (3 * h)                          # fused qkv (bias excluded)
                     + 2 * 2 * heads * n_rows * n_rows * (h // heads)  # QK^T + AV
                     + 2 * n_rows * h * h                              # out proj
                     + 3 * 2 * n_rows * h * ffn)                       # gate + up + down

    def _llm_stage_costs(self, n_prompt: int) -> list:
        """Per decoder layer over an `n_prompt`-token prefill, from the text config alone -- the
        same convention as the Qwen3.5 / GLM-4.6V axes (2 FLOPs per MAC; norms, activations,
        softmax, the top-k select, the Sinkhorn iterations and biases excluded). Three layer
        kinds, read from `linear_attn_config` (2026-09-13):

          * **KDA** (34 layers): the merged qkv projection [H x 3K] with K = heads*head_dim,
            the per-channel forget gate f [H x K] and output gate g [H x V], the per-head
            beta and decay a [H x 2*heads], three depthwise causal convs of kernel 4 over
            q/k/v, the chunked delta-rule scan (2 * dk * dv MACs per value head per token --
            `hooks._qwen35_deltanet_core_flops`' convention) and o_proj [V x H].
          * **sparse MLA** (11 layers, NoPE): q_a [H x q_lora] + q_b [q_lora x heads*qk_dim],
            kv_a [H x kv_lora] + kv_b [kv_lora x heads*(qk_dim + v_dim)], the attention at the
            query heads against min(index_topk, i+1) keys for row i (top-k, so the quadratic
            term saturates at 2048 keys), o_proj [heads*v_dim x H]; the fp8 indexer: its q/k
            projections [H x index_n_heads*index_head_dim] and [H x index_head_dim], and the
            scoring of every query against the pooled keys (one per index_kpool positions).
          * **MoE / dense MLP** on every layer: the first `first_k_dense_replace` layers a dense
            SwiGLU at `intermediate_size`, the rest 288 routed experts top-8 at
            `moe_intermediate_size` + the shared expert + the 288-way router.
          * **mHC**: the two [24, 4H] fp32 coefficient projections per layer (attn + ffn mix);
            the 4x4 mixing itself and the 20 Sinkhorn iterations are adds/scalars and excluded.

        Not measured against hooks (no HF decoder for this checkpoint runs on a GPU here), so
        this is the closed form the Comp. column is stated on, and it is said so in the notes.
        """
        t = self.cfg.text_config
        L, H = int(t.num_hidden_layers), int(t.hidden_size)
        la = dict(getattr(t, "linear_attn_config", {}) or {})
        kda_layers = set(int(x) for x in la.get("kda_layers", []))
        heads_l, dh_l = int(la.get("num_heads", 64)), int(la.get("head_dim", 128))
        conv_k = int(la.get("short_conv_kernel_size", 4))
        K = V = heads_l * dh_l
        heads = int(t.num_attention_heads)
        qk, vd = int(t.qk_head_dim), int(t.v_head_dim)
        q_lora, kv_lora = int(t.q_lora_rank), int(t.kv_lora_rank)
        topk, kpool = int(t.index_topk), int(t.index_kpool)
        ih, idim = int(t.index_n_heads), int(t.index_head_dim)
        n_exp = int(t.n_routed_experts)
        moe_i = int(t.moe_intermediate_size)
        shared = moe_i * int(getattr(t, "n_shared_experts", 0) or 0)
        dense_first = int(getattr(t, "first_k_dense_replace", 0) or 0)
        dense = 3 * 2 * H * int(t.intermediate_size)
        moe = (2 * int(t.num_experts_per_tok) * 3 * moe_i * H     # routed top-k experts
               + 2 * H * n_exp                                   # router
               + 3 * 2 * H * shared)                             # shared expert
        mhc = 2 * 2 * (4 * H) * 24 if getattr(t, "mhc", False) else 0
        kda = (2 * H * (3 * K + K + V + 2 * heads_l)             # qkv, f, g, beta + a
               + 2 * V * H                                       # o_proj
               + 2 * (3 * K) * conv_k                            # three depthwise convs
               + 2 * heads_l * 2 * dh_l * dh_l)                  # delta-rule scan
        mla_proj = (2 * H * q_lora + 2 * q_lora * heads * qk
                    + 2 * H * kv_lora + 2 * kv_lora * heads * (qk + vd)
                    + 2 * (heads * vd) * H
                    + 2 * H * (ih * idim) + 2 * H * idim + 2 * H * ih)   # indexer q, k, gate
        n = int(n_prompt)
        # top-k attention: row i attends min(topk, i + 1) keys -> sum over the prefill
        keys = sum(min(topk, i + 1) for i in range(n))
        quad = 2 * heads * keys * (qk + vd)
        # indexer scoring: every query against the pooled keys before it (one per kpool rows)
        pooled = sum((i + 1 + kpool - 1) // kpool for i in range(n))
        index_score = 2 * ih * idim * pooled
        out = []
        for i in range(L):
            mlp = dense if i < dense_first else moe
            if i in kda_layers:
                out.append(float(n * (mlp + kda + mhc)))
            else:
                out.append(float(n * (mlp + mla_proj + mhc) + quad + index_score))
        return out
