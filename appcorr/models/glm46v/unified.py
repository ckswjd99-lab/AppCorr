"""GLM-4.6V (106B-A12B, FP8): the unified-axis COST half.

Sibling of `appcorr/models/qwen35/unified.py`, but only its two cost hooks -- the axis class
itself (tower, merger, prompt/positions, `_approx_*`) is `appcorr/models/glm46v/axis.py`, built
in parallel.  To keep ONE definition of each, this module ships them as a MIXIN:

    class Glm46VAxis(Glm46VUnifiedCosts, QwenVLStreamingAxis):   # in axis.py
        ...

As of 2026-09-12 `axis.py` still defines its own `_vision_stage_cost` / `_llm_stage_costs`, which
it wrote independently and which are term-for-term the functions below.
`tests/test_glm46v_flops_closed_form.py::AxisVsReportingClosedFormTest` asserts the two (and the
reporting closed form) equal to the float -- the check `axis.py`'s own comment asks for.  Fold
the mixin in and delete the duplicate when the two halves merge.

`QwenVLStreamingAxis.unified_stage_costs` is `[vision_stage_cost] * len(tower.blocks) +
llm_stage_costs(N)`, so with the mixin in the MRO a GLM axis splits its 24 + 46 = 70 stages by
equal cumulative cost exactly as the Qwen3.5 axis splits its 27 + 40 = 67.

Everything here is config-driven and pure (no model, no GPU): `flops_analytic.Glm46VDecoder` /
`Glm46VVision` price the same terms for the report scripts, and
`tests/test_glm46v_flops_closed_form.py` checks the two against a hand count.

Counting convention -- the one `appcorr/flops/hooks.py` follows, so a closed form and a hooked
run are comparable by construction:
  * 2 FLOPs per multiply-accumulate;
  * norms, activations, softmax, residual adds and **bias adds are NOT counted** (`hooks.
    _linear_flops`), so GLM's q/k/v biases (`attention_bias: true`) and the vision Conv3d/Conv2d
    biases cost nothing here even though they exist;
  * partial rotary (0.5) and M-RoPE are elementwise: not counted;
  * attention is charged at the query heads, `2 * 2 * H_q * Sq * Sk * D`, whether or not the mask
    is causal (`hooks.record_attention`);
  * MoE is charged on the ROUTED count (`top_k` per token) plus the router's own projection;
  * `lm_head` and the embedding table are outside the measured subtree (the hooked runs install
    on `model.visual` + `model.language_model`), so they are reported separately, never added.

Architecture (read off `config.json` of `zai-org/GLM-4.6V-FP8` and the vLLM / HF sources,
2026-09-12; see `docs/memo/glm46v_port_plan.md` and the corrections listed in this file):
  text    46 x `Glm4MoeDecoderLayer`, **all softmax GQA** (no DeltaNet anywhere): 96 q / 8 kv
          heads, head_dim 128 explicit, q/k/v bias, no o bias, no qk-norm, partial rotary 0.5.
          Layer 0 dense SwiGLU 10944 (`first_k_dense_replace: 1`); layers 1-45 MoE with 128
          routed experts, top-8, width 1408, plus ONE shared expert of the same width and a
          128-way fp32 sigmoid router.  `vocab_size` 151552 (`lm_head.weight` is [151552, 4096]
          in the checkpoint -- the port plan's "lm_head 154880" is wrong).
  vision  24 x `Glm4vVisionBlock`, hidden 1536, 12 heads x 128, full bidirectional attention over
          the whole image; the block MLP is a **SwiGLU** (gate/up/down, hidden = out_hidden_size
          4096) -- the port plan says "MLP hidden 4096" without saying gated, which would
          under-count the tower by a third of its MLP.  Patch embed Conv3d(3 -> 1536, 2x14x14),
          2x2/stride-2 Conv2d downsample 1536 -> 4096, then `Glm4vPatchMerger`: proj 4096->4096,
          LayerNorm, GELU, SwiGLU 4096 -> 10944 -> 4096 (`vision_config.intermediate_size`).
"""

from typing import Any, List


# --- pure cost functions (config in, FLOPs out) ------------------------------------------------ #

def llm_stage_costs(cfg: Any, n_prompt: int) -> List[float]:
    """FLOPs of EACH of the 46 decoder layers over an `n_prompt`-token prefill.

    A list, not one number: layer 0 is a dense SwiGLU and layers 1-45 are MoE blocks, so the
    unified axis's cost bound must fall where the cumulative cost says and not where a layer
    count says (the same reason Qwen3.5 returns a list -- there for its hybrid attention).
    """
    t = cfg.text_config
    L, H = int(t.num_hidden_layers), int(t.hidden_size)
    heads, kv_heads = int(t.num_attention_heads), int(t.num_key_value_heads)
    dh = int(getattr(t, "head_dim", H // heads))
    n_dense = int(getattr(t, "first_k_dense_replace", 0))
    n_exp = int(getattr(t, "n_routed_experts", 0) or getattr(t, "num_local_experts", 0) or 0)
    moe_i = int(getattr(t, "moe_intermediate_size", 0) or 0)
    shared = moe_i * int(getattr(t, "n_shared_experts", 0) or 0)
    dense_i = int(t.intermediate_size)

    # q_proj [H x heads*dh] (no output gate, unlike Qwen3.5), k/v [H x kv*dh], o [heads*dh x H]
    proj = 2 * H * (heads * dh + 2 * kv_heads * dh) + 2 * (heads * dh) * H
    quad = 2 * 2 * heads * n_prompt * n_prompt * dh
    dense_mlp = 3 * 2 * H * dense_i
    moe_mlp = (2 * int(t.num_experts_per_tok) * 3 * moe_i * H        # routed experts
               + 2 * H * n_exp                                      # 128-way fp32 router
               + 3 * 2 * H * shared)                                # the one shared expert
    return [float(n_prompt * (proj + (dense_mlp if i < n_dense else moe_mlp)) + quad)
            for i in range(L)]


def vision_stage_cost(cfg: Any, n_rows: int) -> float:
    """One tower block over `n_rows` patch rows: fused qkv + full attention over the whole image
    + out proj + the **gated** (SwiGLU) MLP."""
    v = cfg.vision_config
    h, heads = int(v.hidden_size), int(v.num_heads)
    ffn = int(v.out_hidden_size)          # `Glm4vVisionBlock(mlp_hidden_dim=out_hidden_size)`
    return float(2 * n_rows * h * (3 * h)                               # qkv
                 + 2 * 2 * heads * n_rows * n_rows * (h // heads)       # QK^T + AV
                 + 2 * n_rows * h * h                                   # out proj
                 + 3 * 2 * n_rows * h * ffn)                            # gate + up + down


def vision_pre_cost(cfg: Any, n_rows: int) -> float:
    """Patch embed: Conv3d(in_channels -> hidden, (temporal, patch, patch)), stride = kernel, so
    one output column per patch row (`hooks._conv_flops`: 2 * out.numel() * in_ch * prod(k))."""
    v = cfg.vision_config
    h = int(v.hidden_size)
    k = int(v.temporal_patch_size) * int(v.patch_size) ** 2
    return float(2 * n_rows * h * int(v.in_channels) * k)


def vision_merge_cost(cfg: Any, n_groups: int) -> float:
    """Downsample + merger, per MERGE GROUP (= per image token = 4 patch rows).

    `post_layernorm -> view(-1, m, m, C) -> Conv2d(hidden -> out_hidden, m, stride m)` collapses
    each 2x2 group to one row (2 * out.numel() * in_ch * m^2), then `Glm4vPatchMerger`:
    proj [D x D], LayerNorm, GELU, SwiGLU [D x C] gate + [D x C] up + [C x D] down.
    """
    v = cfg.vision_config
    h, d = int(v.hidden_size), int(v.out_hidden_size)
    ctx, m = int(v.intermediate_size), int(v.spatial_merge_size)
    down = 2 * (n_groups * d) * h * (m * m)
    merger = 2 * n_groups * d * d + 3 * 2 * n_groups * d * ctx
    return float(down + merger)


def lm_head_cost(cfg: Any, rows: int = 1) -> float:
    """Reported, never added: the hooked reference installs on the vision tower + language model,
    so `lm_head` is outside the subtree in every measured number."""
    t = cfg.text_config
    return float(2 * rows * int(t.hidden_size) * int(t.vocab_size))


# --- the axis mixin ---------------------------------------------------------------------------- #

class Glm46VUnifiedCosts:
    """The two `QwenVLStreamingAxis` cost hooks for GLM-4.6V.  Mix into the axis class."""

    supports_unified_axis = True

    def _vision_stage_cost(self, n_rows: int) -> float:
        return vision_stage_cost(self.cfg, n_rows)

    def _llm_stage_costs(self, n_prompt: int) -> List[float]:
        return llm_stage_costs(self.cfg, n_prompt)

    # The per-image one-off terms are NOT axis stages: the unified axis's stages are the 24 tower
    # blocks and the 46 decoder layers (that is what `unified_stage_costs` walks and what the
    # engine's `stage_bounds` index into).  Patch-embed runs once before stage 0 and the
    # downsample+merger once at the tower/decoder crossing, in every arm, so they cancel out of
    # the bound split; `flops_analytic.Glm46VVision` charges them in the absolute totals.
    def _vision_pre_cost(self, n_rows: int) -> float:
        return vision_pre_cost(self.cfg, n_rows)

    def _vision_merge_cost(self, n_groups: int) -> float:
        return vision_merge_cost(self.cfg, n_groups)


MODEL_ID = "zai-org/GLM-4.6V-FP8"
