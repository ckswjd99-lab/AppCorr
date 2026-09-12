"""
backbone.py

Wraps GLM-5.3-Flash's stock vision tower (`Glm5NextVisionModel` in transformers >= 5.16, or the
dependency-free `vision/stock.py::Glm5NextVisionStock` on a box whose transformers predates it)
with approx/correct forward passes, exposing the SAME staged interface every prior AppCorr tower
exposes, so `QwenVLStreamingAxis` drives it with no new scheduling code:

    prepare_grid / prepare_full_tokens / embed        arrival-0 prep, grid-only and exact
    approx_forward(x, start_l, end_l, ...)            approximate walk of a LAYER RANGE
    correct_forward / correct_rows                    per-merge-group correction
    merger(rows)                                      per-band merge (`Glm4vMergeHead`, reused)
    finalize/prefix/deferred_attn_layermean           the keep<1 selection signal

It is the GLM-4.6V tower (`appcorr/models/glm46v/vision/backbone.py`) with the PRE and POST
stages changed and the 24 blocks changed in one place, which is why this class subclasses that
one and overrides `__init__` / `prepare_grid` / `embed` only. Survey: `docs/memo/
glm53_vllm_survey.md` §F; code: vLLM main 658c813 `vllm/models/glm5next/nvidia/multimodal.py:
336-606` and transformers 5.16.1 `models/glm5_next/modeling_glm5_next.py:1733-1827`.

The differences, all verified against the checkpoint's own tensor index rather than the memo:

  1. **No learned absolute position embedding and no post-conv norm.** GLM-4.6V normalises the
     patch-embed output (RMSNorm) and then adds a bicubic-`grid_sample`d 24x24 table; GLM-5.3
     does neither -- `patch_embed`'s output goes straight into block 0 (`multimodal.py:380`
     builds the patch embed with no sibling modules; the checkpoint has no
     `visual.embeddings.*` and no `visual.post_conv_layernorm.*` among its 347 `model.visual.*`
     tensors). So `prepare_grid` has no `pos_embeds` entry and `embed` is one conv.
  2. **Per-head q/k RMSNorm inside the attention** -- `vision/attention.py`.
  3. **Two epsilons, not one.** `norm1`/`norm2`/`post_layernorm` run at 1e-6 (vLLM forces it over
     the checkpoint's 1e-5), the q/k norms at 1e-5 (vLLM hard-codes it). `vision/stock.py` has
     the file:line and the reason; `load_stock_vision_tower` applies both.
  4. **The merge head is the same three modules** -- `post_layernorm -> view(-1,2,2,C) ->
     Conv2d(1024->4096, k=s=2) -> Glm5NextVisionPatchMerger` -- so `Glm4vMergeHead` is reused
     verbatim rather than copied. Only the widths and the merger's internals differ (4096 ->
     10240 -> 4096 clamped SwiGLU against GLM-4.6V's 4096 -> 10944 -> 4096 plain SwiGLU), and
     the head never looks inside the merger. Every stage of it is per-merge-group or per-row,
     which is what makes band slicing exact -- asserted, not assumed, by the merger-band gate.
  5. **Row order is unchanged**: `get_vision_position_ids` lays patches out block-major over 2x2
     merge blocks (the Qwen2-VL convention), and this tower has neither `window_size` nor
     `fullatt_block_indexes` -- every one of its 24 layers is full per-image attention. So
     `_bands`' row math, `_rows_of_groups`' identity mapping and the `span=` fast path carry over.

Correction granularity is the merge group, as in every prior fork.
"""

import json
import os
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from transformers.vision_utils import get_vision_cu_seqlens, get_vision_position_ids

from ...glm46v.vision.backbone import ApproxCorrectGlm4vVisionTower, Glm4vMergeHead
from ...qwen35.vision.attention import ApproxCorrectQwen35VisionAttention
from .block import ApproxCorrectGlm5NextVisionBlock
from .stock import NORM_EPS, QK_NORM_EPS, Glm5NextVisionStock, apply_vllm_eps, tower_eps

MODEL_ID_GLM53_FP8 = "zai-org/GLM-5.3-Flash"

# `Glm5NextVisionConfig`'s own defaults (vllm/transformers_utils/configs/glm5_next.py:274-300),
# used only to fill a field a local config.json omits.
_VISION_DEFAULTS = dict(depth=24, hidden_size=1024, hidden_act="silu", image_size=448,
                        intermediate_size=4096, num_heads=16, out_hidden_size=4096,
                        projection_intermediate_size=10240, in_channels=3, patch_size=14,
                        rms_norm_eps=1e-5, spatial_merge_size=2, temporal_patch_size=2,
                        attention_dropout=0.0, attention_bias=True, swiglu_limit=None)


def resolve_snapshot(model_id: str = MODEL_ID_GLM53_FP8) -> str:
    """The local snapshot directory for `model_id` (a path is returned unchanged)."""
    if os.path.isdir(model_id):
        return model_id
    from huggingface_hub import hf_hub_download
    return os.path.dirname(hf_hub_download(model_id, "config.json"))


def vision_config(model_id: str = MODEL_ID_GLM53_FP8):
    """`config.json`'s `vision_config` as an attribute bag, read WITHOUT `AutoConfig`.

    transformers 5.13 (the `appcorr` env here) has no `glm5_next` model type, so `AutoConfig
    .from_pretrained` raises on this checkpoint; the tower needs eighteen scalars, all of which
    are in the file. `swiglu_limit` falls back to `text_config.swiglu_limit` exactly as vLLM does
    (`multimodal.py:373-378`) -- on this checkpoint both say 10.0.
    """
    from types import SimpleNamespace
    raw = json.load(open(os.path.join(resolve_snapshot(model_id), "config.json")))
    v = dict(_VISION_DEFAULTS)
    v.update(raw.get("vision_config", {}))
    if v.get("swiglu_limit") is None:
        v["swiglu_limit"] = raw.get("text_config", {}).get("swiglu_limit")
    if v["swiglu_limit"] is None:
        raise ValueError("GLM-5.3-Flash vision requires swiglu_limit (vision_config or text_config)")
    return SimpleNamespace(**v)


def load_stock_vision_tower(model_id: str = MODEL_ID_GLM53_FP8, device: str = "cuda:0",
                            dtype: torch.dtype = torch.bfloat16, *, prefer_hf: bool = True,
                            vllm_eps: bool = True) -> nn.Module:
    """The STOCK tower built from the checkpoint's `model.visual.*` tensors alone.

    GLM-5.3-Flash is an FP8 checkpoint (328 GB over 62 shards) whose VISION half is bf16: the
    quantization config's `modules_to_not_convert` covers the decoder modules by name and no
    `visual.*` tensor has a `weight_scale_inv` sibling. The index puts all 347 `model.visual.*`
    tensors in ONE shard (`model-00062-of-00062.safetensors`), so this opens one file, not 62,
    and never touches the 45 decoder layers.

    `prefer_hf`: use transformers' `Glm5NextVisionModel` when this box has it (>= 5.16), else the
    port in `vision/stock.py`. Both carry the checkpoint's module names, so the fork wrapping them
    cannot tell -- which is the point: the gate asserts the two agree.
    `vllm_eps`: retune to what vLLM actually serves (1e-6 block/post, 1e-5 q/k). Pass False to
    reproduce a plain HF `from_pretrained` (1e-5 everywhere) for a comparison.
    """
    from safetensors import safe_open

    snap = resolve_snapshot(model_id)
    cfg = vision_config(model_id)
    index = os.path.join(snap, "model.safetensors.index.json")
    if os.path.exists(index):
        wmap = json.load(open(index))["weight_map"]
        files = sorted({v for k, v in wmap.items() if ".visual." in k or k.startswith("visual.")})
    else:
        files = ["model.safetensors"]
    sd: Dict[str, torch.Tensor] = {}
    for f in files:
        with safe_open(os.path.join(snap, f), framework="pt", device=device) as sf:
            for k in sf.keys():
                for p in ("model.visual.", "visual."):
                    if k.startswith(p):
                        sd[k[len(p):]] = sf.get_tensor(k)
                        break
    if not sd:
        raise KeyError(f"no `visual.*` tensors in {snap}")

    visual = None
    if prefer_hf:
        try:
            from transformers.models.glm5_next.configuration_glm5_next import Glm5NextVisionConfig
            from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel
        except Exception:  # noqa: BLE001 -- transformers < 5.16 has no glm5_next
            visual = None
        else:
            hf_cfg = Glm5NextVisionConfig(**{k: v for k, v in vars(cfg).items()})
            with torch.device(device):
                # fp32 build, then cast PARAMETERS only -- initialising under a bf16 default dtype
                # also computes the non-persistent rotary `inv_freq` in bf16, which
                # `from_pretrained` would have left at its fp32 init (measured on GLM-4.6V as a
                # 3.8% rel-L2 error in the features, vision_only.py:136-145).
                visual = Glm5NextVisionModel._from_config(hf_cfg, dtype=torch.float32)
    if visual is None:
        with torch.device(device):
            visual = Glm5NextVisionStock(cfg, norm_eps=float(cfg.rms_norm_eps),
                                         qk_norm_eps=float(cfg.rms_norm_eps))
    for prm in visual.parameters():
        prm.data = prm.data.to(dtype)
    missing, unexpected = visual.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"glm53 vision tower load mismatch: missing={list(missing)[:5]} "
                           f"unexpected={list(unexpected)[:5]} "
                           f"(of {len(missing)}/{len(unexpected)})")
    if vllm_eps:
        apply_vllm_eps(visual)
    return visual.eval()


class ApproxCorrectGlm5NextVisionTower(ApproxCorrectGlm4vVisionTower):
    """GLM-5.3-Flash (`Glm5NextForConditionalGeneration`): unwindowed 24-layer tower, hidden 1024,
    16 heads x 64, patch 14 / temporal 2, rows in natural merge-group-major order.

    Inherits every staged method from the GLM-4.6V tower (`approx_forward`, `correct_plan`,
    `correct_forward`, `correct_rows`, the three attn-layermean methods, `get_merged_output`,
    `reference_forward`, `stage_names`) and overrides only the three that differ: the constructor
    (different submodules), `prepare_grid` (no absolute position embedding to interpolate) and
    `embed` (no post-conv norm, no position add)."""

    def __init__(self, vision_tower: nn.Module):
        nn.Module.__init__(self)      # NOT super(): the GLM-4.6V constructor wants modules
        #                               (`post_conv_layernorm`, `embeddings`) this tower lacks.
        cfg = getattr(vision_tower, "config", None)
        if cfg is None:
            raise AttributeError("stock tower has no `.config`")
        self.config = cfg
        for gone in ("post_conv_layernorm", "embeddings"):
            if hasattr(vision_tower, gone):
                raise ValueError(
                    f"stock tower has `{gone}`, which GLM-5.3-Flash's does not -- this looks like "
                    "a GLM-4.6V/GLM-OCR tower. Use appcorr/models/glm46v/vision/backbone.py.")
        self.patch_embed = vision_tower.patch_embed
        self.rotary_pos_emb = vision_tower.rotary_pos_emb
        self.spatial_merge_size = int(getattr(vision_tower, "spatial_merge_size",
                                              cfg.spatial_merge_size))
        self.spatial_merge_unit = self.spatial_merge_size ** 2
        self.blocks = nn.ModuleList(
            [ApproxCorrectGlm5NextVisionBlock.from_stock(b) for b in vision_tower.blocks]
        )
        self.merger = Glm4vMergeHead(vision_tower.post_layernorm, vision_tower.downsample,
                                     vision_tower.merger, self.spatial_merge_size,
                                     int(cfg.out_hidden_size))
        # Asserted, not assumed (every fork's rule). This tower's whole simplification -- one
        # `segment_ranges` for all 24 layers, natural row order, every layer's received attention
        # comparable -- rests on there being no windowing. GLM-5.3's vision config has no such
        # field today; a checkpoint that grew one would otherwise silently compute full attention
        # where stock computes windowed.
        if getattr(cfg, "window_size", None) or getattr(cfg, "fullatt_block_indexes", None):
            raise ValueError(
                f"glm53 vision fork assumes an unwindowed tower, but config has "
                f"window_size={getattr(cfg, 'window_size', None)!r} "
                f"fullatt_block_indexes={getattr(cfg, 'fullatt_block_indexes', None)!r}. "
                "Port the window permutation from appcorr/models/qwen25vl/vision/backbone.py.")
        if len(self.blocks) != int(cfg.depth):
            raise ValueError(f"{len(self.blocks)} blocks for a depth-{cfg.depth} config")

    # --- loading ----------------------------------------------------------------------------- #

    @classmethod
    def from_pretrained(cls, model_id: str = MODEL_ID_GLM53_FP8, device: str = "cuda:0",
                        dtype: torch.dtype = torch.bfloat16,
                        **kw) -> "ApproxCorrectGlm5NextVisionTower":
        """The fork around `load_stock_vision_tower(...)`. See that function."""
        return cls(load_stock_vision_tower(model_id, device, dtype, **kw))

    def eps(self) -> Dict[str, float]:
        """The two epsilons this tower is actually running with, for a gate to print."""
        return {"norm": float(self.blocks[0].norm1.variance_epsilon),
                "qk_norm": float(self.blocks[0].attn.q_norm.variance_epsilon),
                "post": float(self.merger.post_layernorm.variance_epsilon)}

    # --- arrival-0 prep (grid only, always exact) --------------------------------------------- #

    def prepare_grid(self, grid_thw: torch.Tensor, device) -> Dict[str, Any]:
        """Everything derivable from `grid_thw` alone: the 2-D RoPE cos/sin, `cu_seqlens` and the
        CPU-side segment ranges (the `.tolist()` syncs live here, once per request).

        The GLM-4.6V version also interpolated a learned absolute position embedding here; this
        tower has none, so the returned context has no `pos_embeds` key and `embed()` adds nothing.
        """
        grid_thw = grid_thw.to(device)
        position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size)
        cu_seqlens = get_vision_cu_seqlens(grid_thw).to(device)

        rotary_pos_emb = self.rotary_pos_emb(position_ids)          # [T, head_dim // 2]
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)   # [T, head_dim]
        position_embeddings = (emb.cos(), emb.sin())

        segment_ranges = ApproxCorrectQwen35VisionAttention.segment_ranges_from_cu_seqlens(cu_seqlens)
        return {
            "position_embeddings": position_embeddings,
            "cu_seqlens": cu_seqlens,
            "segment_ranges": segment_ranges,
            "seq_len": int(position_ids.shape[0]),
        }

    def embed(self, pixel_values: torch.Tensor, gctx: Dict[str, Any]) -> torch.Tensor:
        """`patch_embed` -> [seq_len, dim]. One Conv3d and nothing else: unlike GLM-4.6V there is
        no `post_conv_layernorm` between the conv and the blocks, and no position embedding to
        add (`multimodal.py:559-566`, HF :1810-1813)."""
        hidden_states = self.patch_embed(pixel_values)
        if hidden_states.shape[0] != gctx["seq_len"]:
            raise ValueError(f"{hidden_states.shape[0]} patch rows for a grid of {gctx['seq_len']}")
        return hidden_states.reshape(gctx["seq_len"], -1)
