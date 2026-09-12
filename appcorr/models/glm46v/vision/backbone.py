"""
backbone.py

Wraps the stock `Glm4vMoeVisionModel` (GLM-4.6V / GLM-4.5V family, `transformers.models.
glm4v_moe`) with approx/correct forward passes, exposing the SAME staged interface
`appcorr/models/qwen35/vision/backbone.py` exposes, so `QwenVLStreamingAxis` drives this tower
with no new scheduling code:

    prepare_grid / prepare_full_tokens / embed        arrival-0 prep, grid-only and exact
    approx_forward(x, start_l, end_l, ...)            approximate walk of a LAYER RANGE
    correct_forward / correct_rows                    per-merge-group correction
    merger(rows)                                      per-band merge  (see `Glm4vMergeHead`)
    finalize/prefix/deferred_attn_layermean           the keep<1 selection signal

Structurally this tower is the Qwen3.5 one with four differences, all of them in the PRE and POST
stages rather than in the 24 identical blocks (survey: `docs/memo/glm46v_port_plan.md`; code:
`transformers/models/glm4v_moe/modeling_glm4v_moe.py:736-838` and vLLM 0.28
`vllm/model_executor/models/glm4_1v.py:608-960`):

  1. **`post_conv_layernorm` between the patch embed and the position add.** Qwen3.5 adds its
     interpolated `pos_embed` straight onto `patch_embed`'s output; GLM normalises first
     (RMSNorm), then adds. Getting the order wrong is invisible in a shape check and wrong in
     every number, so `embed()` follows the stock forward line for line.
  2. **Learned absolute position embedding is BICUBIC `grid_sample`d, not bilinear-gathered.**
     A 24x24 (`image_size 336 / patch 14`) table `Embedding(576, 1536)` is resampled to the
     request's (h, w) grid at the block-major patch coordinates -- the same coordinates the
     rotary tables use, which is what keeps the two spatially aligned. It is a pure function of
     `grid_thw`, hence computed once in `prepare_grid` and always exact (the "cheap non-block ops
     are always exact" rule every fork follows). The stock `embeddings` module is CALLED on a
     zero tensor to obtain it, exactly as vLLM's `pos_embeds_interpolate` does
     (`glm4_1v.py:760-823`), so the tensor added here is bitwise the one stock adds.
  3. **The merge head is three modules, not one.** Qwen3.5's `merger` is the whole projector;
     GLM's is `post_layernorm -> view(-1,2,2,C) -> Conv2d(2x2, stride 2, 1536->4096) ->
     Glm4vPatchMerger(proj -> LayerNorm -> GELU -> SwiGLU 4096->10944->4096)`. `Glm4vMergeHead`
     below packs the three into one callable named `merger`, so the axis's two merger call sites
     (`qwen_vl_axis.py` ~567 and ~768) are unchanged. Every stage of it is per-merge-group or
     per-row, which is what makes band slicing exact -- asserted, not assumed, by
     `tests/test_glm46v_merger_bands.py`.
  4. **Row order.** `get_vision_position_ids` lays the patches out block-major over 2x2 merge
     blocks (`vision_utils.py:90-92`), i.e. the Qwen2-VL convention: merge group `g` owns rows
     `g*4 .. g*4+3` in natural order, with no window permutation anywhere in this tower (GLM's
     vision config has neither `window_size` nor `fullatt_block_indexes`). So `_bands`' row math,
     `_rows_of_groups`' identity mapping and the `span=` slice fast path all carry over.

Correction granularity is the merge group, as in every prior fork: four raw patch rows are
combined nonlinearly (2x2 conv + MLP) into one LLM token, so a half-corrected group is not a
meaningful state.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from transformers.vision_utils import get_vision_cu_seqlens, get_vision_position_ids

from ...qwen35.vision.attention import ApproxCorrectQwen35VisionAttention
from .block import ApproxCorrectGlm4vVisionBlock

MODEL_ID_GLM46V_FP8 = "zai-org/GLM-4.6V-FP8"


def load_stock_vision_tower(model_id: str = MODEL_ID_GLM46V_FP8, device: str = "cuda:0",
                            dtype: torch.dtype = torch.bfloat16) -> nn.Module:
    """The STOCK `Glm4vMoeVisionModel` built from the checkpoint's `model.visual.*` tensors alone.

    GLM-4.6V-FP8 is compressed-tensors FP8 over the decoder only: the quantisation config's
    `ignore` list covers every `visual.*` module, and the shards store them as BF16 (verified on
    the index -- all 181 `model.visual.*` tensors are BF16). So the tower needs no dequantisation
    and no decoder, and it lives in 2 of the 41 shards, which the index names: this opens two
    files, not 41.

    Returned unwrapped so a test can gate the fork against stock without a second copy of the
    weights in memory (the fork holds references to these same submodules).
    `appcorr/models/vision_only.py` is the path the campaign driver uses -- it also needs
    `embed_tokens` and the config-only `get_rope_index`, and its generic `model.visual.*` /
    `model.language_model.embed_tokens` collection already covers this checkpoint.
    """
    import json
    import os

    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from transformers import AutoConfig
    from transformers.models.glm4v_moe.modeling_glm4v_moe import Glm4vMoeVisionModel

    config = AutoConfig.from_pretrained(model_id)
    snap = os.path.dirname(hf_hub_download(model_id, "config.json"))
    index = os.path.join(snap, "model.safetensors.index.json")
    if os.path.exists(index):
        wmap = json.load(open(index))["weight_map"]
        files = sorted({v for k, v in wmap.items() if ".visual." in k})
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
    with torch.device(device):
        # fp32 build, then cast PARAMETERS only: initialising under a bf16 default dtype also
        # computes the non-persistent rotary `inv_freq` buffer in bf16, which `from_pretrained`
        # would have left at its fp32 init (vision_only.py:136-145 -- measured there as a 3.8%
        # rel-L2 error in the features).
        visual = Glm4vMoeVisionModel._from_config(config.vision_config, dtype=torch.float32)
    for prm in visual.parameters():
        prm.data = prm.data.to(dtype)
    missing, unexpected = visual.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"glm46v vision tower load mismatch: missing={missing[:5]} "
                           f"unexpected={unexpected[:5]} (of {len(missing)}/{len(unexpected)})")
    return visual.eval()


class Glm4vMergeHead(nn.Module):
    """`post_layernorm -> 2x2 downsample conv -> Glm4vPatchMerger`, over ANY set of whole merge
    groups.

    Stock runs it once over the tower's full output (`Glm4vMoeVisionModel.forward:826-836`);
    this fork runs it per band. That is exact rather than approximate because every stage is
    group-local:

      * `post_layernorm` is an RMSNorm over the feature axis -- per row;
      * the downsample is `Conv2d(kernel=stride=2)` applied to a `(-1, 2, 2, C)` view, i.e. one
        2x2 window per merge group with no overlap and no padding -- per group;
      * the merger is Linear/LayerNorm/GELU/SwiGLU over the feature axis -- per merged token.

    So `head(all_rows)[g0:g1] == head(rows of groups [g0, g1))` bitwise. The test asserts it on
    real weights instead of trusting the reading.
    """

    def __init__(self, post_layernorm: nn.Module, downsample: nn.Module, merger: nn.Module,
                 spatial_merge_size: int, out_hidden_size: int):
        super().__init__()
        self.post_layernorm = post_layernorm
        self.downsample = downsample
        self.merger = merger
        self.spatial_merge_size = int(spatial_merge_size)
        self.out_hidden_size = int(out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[n_groups * merge_unit, hidden] -> [n_groups, out_hidden]."""
        m = self.spatial_merge_size
        if x.shape[0] % (m * m):
            raise ValueError(f"{x.shape[0]} rows is not a whole number of {m * m}-row merge "
                             "groups; bands must be cut on merge-group boundaries")
        x = self.post_layernorm(x)
        x = x.view(-1, m, m, x.shape[-1]).permute(0, 3, 1, 2)
        x = self.downsample(x).view(-1, self.out_hidden_size)
        return self.merger(x)


class ApproxCorrectGlm4vVisionTower(nn.Module):
    def __init__(self, vision_tower: nn.Module):
        super().__init__()
        cfg = vision_tower.config
        self.config = cfg
        self.patch_embed = vision_tower.patch_embed
        self.post_conv_layernorm = vision_tower.post_conv_layernorm
        self.embeddings = vision_tower.embeddings
        self.rotary_pos_emb = vision_tower.rotary_pos_emb
        self.spatial_merge_size = int(vision_tower.spatial_merge_size)
        self.spatial_merge_unit = self.spatial_merge_size ** 2
        self.blocks = nn.ModuleList(
            [ApproxCorrectGlm4vVisionBlock.from_stock(b) for b in vision_tower.blocks]
        )
        self.merger = Glm4vMergeHead(vision_tower.post_layernorm, vision_tower.downsample,
                                     vision_tower.merger, self.spatial_merge_size,
                                     int(cfg.out_hidden_size))
        # Asserted, not assumed (the qwen35 fork's rule). This tower's whole simplification --
        # one `segment_ranges` for all 24 layers, natural row order, every layer's received
        # attention comparable -- rests on there being no windowing. GLM's vision config has no
        # such field today; a future checkpoint that grew one would otherwise silently compute
        # full attention where stock computes windowed.
        if getattr(cfg, "window_size", None) or getattr(cfg, "fullatt_block_indexes", None):
            raise ValueError(
                f"glm46v vision fork assumes an unwindowed tower, but config has "
                f"window_size={getattr(cfg, 'window_size', None)!r} "
                f"fullatt_block_indexes={getattr(cfg, 'fullatt_block_indexes', None)!r}. "
                "Port the window permutation from appcorr/models/qwen25vl/vision/backbone.py.")
        if len(self.blocks) != int(cfg.depth):
            raise ValueError(f"{len(self.blocks)} blocks for a depth-{cfg.depth} config")

    # --- loading ----------------------------------------------------------------------------- #

    @classmethod
    def from_pretrained(cls, model_id: str = MODEL_ID_GLM46V_FP8, device: str = "cuda:0",
                        dtype: torch.dtype = torch.bfloat16) -> "ApproxCorrectGlm4vVisionTower":
        """The fork around `load_stock_vision_tower(...)`. See that function."""
        return cls(load_stock_vision_tower(model_id, device, dtype))

    # --- arrival-0 prep (grid only, always exact) --------------------------------------------- #

    def prepare_grid(self, grid_thw: torch.Tensor, device) -> Dict[str, Any]:
        """Everything `prepare_full_tokens` derives from `grid_thw` alone: the bicubic-resampled
        absolute position embedding, the 2D-RoPE cos/sin, `cu_seqlens` and the CPU-side segment
        ranges (the `.tolist()` syncs live here, once per request). `embed()` then turns each
        pixel tensor (full image, degraded base) into a layer-0 stream against the same context.
        """
        grid_thw = grid_thw.to(device)
        position_ids = get_vision_position_ids(grid_thw, self.spatial_merge_size)
        cu_seqlens = get_vision_cu_seqlens(grid_thw).to(device)

        rotary_pos_emb = self.rotary_pos_emb(position_ids)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        seq_len = int(position_ids.shape[0])
        # The stock forward adds the interpolated absolute embedding INSIDE `self.embeddings`
        # (modeling_glm4v_moe.py:816-823). Feeding it zeros returns the addend itself, which is
        # what vLLM's `pos_embeds_interpolate` does for its CUDA-graph split -- same module, same
        # coordinates (`position_ids[:, 0]` / `[:, 1]`, block-major, identical to the rotary's),
        # so `embed()`'s `x + pos_embeds` is bitwise stock's `embeddings(x, ...)`.
        zeros = torch.zeros(seq_len, int(self.config.hidden_size), device=device,
                            dtype=self.embeddings.position_embedding.weight.dtype)
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        pos_embeds = self.embeddings(zeros, seqlens, grid_thw,
                                     position_ids[:, 0].to(device), position_ids[:, 1].to(device))

        segment_ranges = ApproxCorrectQwen35VisionAttention.segment_ranges_from_cu_seqlens(cu_seqlens)

        return {
            "pos_embeds": pos_embeds,
            "position_embeddings": position_embeddings,
            "cu_seqlens": cu_seqlens,
            "segment_ranges": segment_ranges,
            "seq_len": seq_len,
        }

    def embed(self, pixel_values: torch.Tensor, gctx: Dict[str, Any]) -> torch.Tensor:
        """`patch_embed -> post_conv_layernorm -> + interpolated pos_embed` -> [seq_len, dim].

        The norm sits BETWEEN the conv and the position add (stock forward :812-823); Qwen3.5 has
        no such norm and adds straight onto the conv output.
        """
        hidden_states = self.patch_embed(pixel_values)
        hidden_states = self.post_conv_layernorm(hidden_states)
        hidden_states = hidden_states + gctx["pos_embeds"].to(hidden_states.dtype)
        if hidden_states.shape[0] != gctx["seq_len"]:
            raise ValueError(f"{hidden_states.shape[0]} patch rows for a grid of {gctx['seq_len']}")
        return hidden_states.reshape(gctx["seq_len"], -1)

    def prepare_full_tokens(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor,
                            gctx: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Layer-0 stream + the grid context. Pass a `prepare_grid` result as `gctx` to share it
        across the two images of one request (full + degraded base: same grid, by construction)."""
        if gctx is None:
            gctx = self.prepare_grid(grid_thw, pixel_values.device)
        return {"hidden_states": self.embed(pixel_values, gctx), **gctx}

    # --- the block walk (identical contract to qwen35) ---------------------------------------- #

    def approx_forward(self, x_feature: torch.Tensor, start_l: int, end_l: int, ctx: Dict[str, Any],
                       cache_feature: Dict[str, Any], tag_prefix: str,
                       collect_attn_mean=False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Runs blocks[start_l:end_l] in approx mode. `collect_attn_mean`: False | True | "defer"."""
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
        `"single"` for a one-segment request (no sync at all), else the Qwen2.5-VL shape."""
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

    def correct_forward(self, x_feature: torch.Tensor, group_idx: torch.Tensor, start_l: int,
                        end_l: int, ctx: Dict[str, Any], cache_feature: Dict[str, Any],
                        tag_prefix: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Correct `group_idx`'s rows over blocks[start_l:end_l] against the cached K/V, carrying
        the full [T, dim] residual stream (rule-3 write-back included). `group_idx` is in NATURAL
        order: this tower, like Qwen3.5's, has no window permutation between a group and its rows.
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
        """Full-depth correction of `group_idx`'s rows only -- bitwise
        `correct_forward(x0_full, group_idx, 0, depth, ...)[0][token_idx]` without materialising
        the other T - Q rows at any layer. Valid only for a caller that never reads a
        non-corrected row and never corrects a group twice (the streaming axis); see
        `qwen35/vision/block.py::correct_rows` for the contract."""
        token_idx = self._token_idx(group_idx, x0_full.device)
        cos_full, sin_full = ctx["position_embeddings"]
        plan = self.correct_plan(token_idx, ctx)
        if span is not None:
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

    # --- the keep<1 selection signal ---------------------------------------------------------- #

    def deferred_attn_layermean(self, cache_feature: Dict[str, Any], tag_prefix: str,
                                n_layers: int, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Turn the queries an `approx_forward(..., collect_attn_mean="defer")` stashed into the
        per-layer received-attention vectors and their layer mean."""
        for i in range(n_layers):
            tag = f"{tag_prefix}_layer{i}"
            cache_feature[f"{tag}_attn_mean"] = self.blocks[i].attn.received_attention_from_cache(
                cache_feature, tag, ctx["segment_ranges"])
        return self.finalize_attn_layermean(cache_feature, tag_prefix, n_layers)

    def prefix_attn_layermean(self, cache_feature: Dict[str, Any], tag_prefix: str,
                              n_layers: int) -> torch.Tensor:
        """[T] mean received attention over the FIRST `n_layers` layers (the unified axis's
        progressive signal), returned rather than stored. Eager collection only."""
        acc = None
        for i in range(n_layers):
            v = cache_feature.get(f"{tag_prefix}_layer{i}_attn_mean")
            if v is None:
                raise KeyError(
                    f"glm46v prefix_attn_layermean: layer {i} of {n_layers} did not collect "
                    "received attention -- the chunked approx walk must pass "
                    "collect_attn_mean=True on every range up to the frontier.")
            acc = v.float() if acc is None else acc + v.float()
        return acc / max(1, n_layers)

    def finalize_attn_layermean(self, cache_feature: Dict[str, Any], tag_prefix: str,
                                n_layers: int) -> Dict[str, Any]:
        """Average the per-layer received-attention vectors into `{tag_prefix}_attn_layermean`.

        All 24 layers are full per-image attention, so all of them are comparable and all of them
        are averaged -- there is no windowed subset to exclude (contrast Qwen2.5-VL)."""
        keys = [f"{tag_prefix}_layer{i}_attn_mean" for i in range(n_layers)]
        missing = [k for k in keys if k not in cache_feature]
        if missing:
            raise KeyError(
                f"glm46v finalize_attn_layermean: {len(missing)} of {n_layers} layers did not "
                f"collect received attention (first missing: {missing[0]}). approx_forward must "
                "be called with collect_attn_mean=True over the FULL depth before this.")
        acc = None
        for k in keys:
            v = cache_feature[k]
            acc = v.float() if acc is None else acc + v.float()
        cache_feature[f"{tag_prefix}_attn_layermean"] = acc / n_layers
        return cache_feature

    # --- the merge ---------------------------------------------------------------------------- #

    def get_merged_output(self, x_full: torch.Tensor, ctx: Dict[str, Any]) -> torch.Tensor:
        """The merge head over the last layer's [seq_len, dim] hidden state. No un-permutation:
        rows are already in natural (merge-group-major) order."""
        return self.merger(x_full)

    # --- the unstaged reference (tests / gates) ------------------------------------------------ #

    @torch.no_grad()
    def reference_forward(self, pixel_values: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        """One-shot forward through the fork's own modules -- pre stage, all blocks with their
        plain `forward`, merge head. The staged/approx paths are gated against THIS (and this
        against stock `Glm4vMoeVisionModel.forward`) in `tests/test_glm46v_merger_bands.py`."""
        ctx = self.prepare_grid(grid_thw, pixel_values.device)
        x = self.embed(pixel_values, ctx)
        for blk in self.blocks:
            x = blk(x, ctx["segment_ranges"], ctx["position_embeddings"])
        return self.merger(x)

    def stage_names(self) -> List[str]:
        """The unified axis's vision stages, in order: the pre stage is free (grid-only work that
        never depends on which patches arrived), then one stage per block, then the merge."""
        return ["pre"] + [f"block{i}" for i in range(len(self.blocks))] + ["merge"]
