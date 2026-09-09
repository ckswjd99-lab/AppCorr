"""Vision-only stand-in for the HF model on the AppCorr side of the two-process form.

Once a `StreamSink` takes the chunks, the axis (`qwen_vl_axis.QwenVLStreamingAxis`) touches
exactly four things on the HF model: `model.model.visual` (the tower it wraps), `model.model.
language_model.embed_tokens` (text rows of the prompt), `model.model.get_rope_index` (M-RoPE
positions, config-only) and `model.model.get_image_features` (one-shot floor/ceiling embeds).
Never a decoder layer. Loading the whole checkpoint for that was harmless while the vLLM engine
lived on the other GPU; with one GPU (GPU1 off limits from 2026-09-08) the 122B-FP8 twin
(120 GB) and the engine (120 GB weights + cache) cannot coexist, so this builds only the two
submodules on the device and streams their tensors out of the safetensors shards -- ~3 GB for
122B (tower 0.7B params + a 248k x 4096 embedding), seconds instead of a 33 s full load.

The loaded tensors are the checkpoint's own (the vision modules are in `modules_to_not_convert`
of the FP8 checkpoints, i.e. bf16 as stored), so the tower is bitwise the one the full model
would build; `load_state_dict(strict=True)` guarantees nothing is silently left at init. Any
forward that would need the decoder (`model(...)`) raises instead of computing garbage.

    model = load_vision_only("Qwen/Qwen3.5-122B-A10B-FP8")      # -> VisionOnlyModel
    axis = Qwen35Axis(model, processor)                          # unchanged downstream
"""
from __future__ import annotations

import json
import os
from typing import Dict, List

import torch
import torch.nn as nn


def _snapshot_dir(model_id: str) -> str:
    from huggingface_hub import hf_hub_download
    return os.path.dirname(hf_hub_download(model_id, "config.json"))


def _shard_files(snap: str) -> List[str]:
    idx = os.path.join(snap, "model.safetensors.index.json")
    if os.path.exists(idx):
        wm = json.load(open(idx))["weight_map"]
        return sorted({os.path.join(snap, f) for f in wm.values()})
    return [os.path.join(snap, "model.safetensors")]


# checkpoint prefix -> (submodule name, strip). Two naming generations are in the cache: the
# Qwen2.5-VL files carry `visual.*` / `model.embed_tokens`, Qwen3.5 carries `model.visual.*` /
# `model.language_model.embed_tokens` (transformers 5 maps the old names on a normal load; here
# the mapping is explicit so `strict=True` is meaningful).
_VISUAL_PREFIXES = ("model.visual.", "visual.")
_EMBED_KEYS = ("model.language_model.embed_tokens.weight", "model.embed_tokens.weight")


def _collect(snap: str, device: str) -> Dict[str, torch.Tensor]:
    from safetensors import safe_open
    out: Dict[str, torch.Tensor] = {}
    for f in _shard_files(snap):
        with safe_open(f, framework="pt", device=device) as sf:
            for k in sf.keys():
                if k in _EMBED_KEYS:
                    out["__embed__"] = sf.get_tensor(k)
                    continue
                for p in _VISUAL_PREFIXES:
                    if k.startswith(p):
                        out[k[len(p):]] = sf.get_tensor(k)
                        break
    if "__embed__" not in out:
        raise KeyError(f"no embed_tokens tensor under {_EMBED_KEYS} in {snap}")
    return out


class _Inner(nn.Module):
    """Stands in for `model.model`: the two loaded submodules plus the config-only methods of
    the real inner class (`get_rope_index` / `get_vision_position_ids` read `self.config` and
    nothing else -- checked against transformers 5.13 for Qwen2.5-VL and Qwen3.5)."""

    def __init__(self, config, visual: nn.Module, embed: nn.Embedding, inner_cls: type):
        super().__init__()
        self.config = config
        self.visual = visual
        self.language_model = nn.Module()
        self.language_model.embed_tokens = embed
        self.rope_deltas = None
        self._inner_cls = inner_cls

    def get_rope_index(self, *a, **kw):
        return self._inner_cls.get_rope_index(self, *a, **kw)

    def get_vision_position_ids(self, *a, **kw):
        return self._inner_cls.get_vision_position_ids(self, *a, **kw)

    @torch.no_grad()
    def get_image_features(self, pixel_values: torch.Tensor, image_grid_thw: torch.Tensor = None,
                           **kwargs):
        """The real method minus its return-type decorators: tower -> per-image split. Returns
        the tuple of per-image `[T_i, D]` embeds (what `oneshot_embeds` concatenates)."""
        out = self.visual(pixel_values.type(self.visual.dtype), grid_thw=image_grid_thw,
                          return_dict=True)
        feats = out.pooler_output if hasattr(out, "pooler_output") else out
        if isinstance(feats, (list, tuple)):
            feats = torch.cat(list(feats), dim=0)
        split = (image_grid_thw.prod(-1) // self.visual.spatial_merge_size ** 2).tolist()
        return torch.split(feats, split)

    def forward(self, *a, **kw):
        raise RuntimeError("vision-only model: the decoder layers are not loaded")


class VisionOnlyModel(nn.Module):
    def __init__(self, config, inner: _Inner):
        super().__init__()
        self.config = config
        self.model = inner

    @property
    def dtype(self) -> torch.dtype:
        return self.model.visual.dtype

    @property
    def device(self) -> torch.device:
        return next(self.model.visual.parameters()).device

    def forward(self, *a, **kw):
        raise RuntimeError("vision-only model: the decoder layers are not loaded")


def load_vision_only(model_id: str, device: str = "cuda:0",
                     dtype: torch.dtype = torch.bfloat16) -> VisionOnlyModel:
    from transformers import AutoConfig, AutoModelForImageTextToText
    config = AutoConfig.from_pretrained(model_id)
    # A weightless skeleton on `meta` names the classes (vision tower / inner model) for this
    # checkpoint family without touching the disk or the device.
    with torch.device("meta"):
        skel = AutoModelForImageTextToText.from_config(config)
    inner_cls, vis_cls = type(skel.model), type(skel.model.visual)
    del skel
    tcfg = config.text_config
    with torch.device(device):
        # Build in fp32 and cast PARAMETERS only: initialising under a bf16 default dtype also
        # computes the non-persistent rotary `inv_freq` buffer in bf16 (max|diff| 1.9e-3 on the
        # 7B tower -> block-0 output off by 0.06, features by 3.8% rel-L2), whereas from_pretrained
        # keeps such buffers at their fp32 init and casts only the loaded weights.
        visual = vis_cls._from_config(config.vision_config, dtype=torch.float32)
        embed = nn.Embedding(tcfg.vocab_size, tcfg.hidden_size,
                             padding_idx=getattr(tcfg, "pad_token_id", None), dtype=dtype)
    for prm in visual.parameters():
        prm.data = prm.data.to(dtype)
    sd = _collect(_snapshot_dir(model_id), device)
    embed.weight.data.copy_(sd.pop("__embed__").to(dtype))
    missing, unexpected = visual.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"vision tower load mismatch: missing={missing[:5]} "
                           f"unexpected={unexpected[:5]} (of {len(missing)}/{len(unexpected)})")
    visual = visual.eval()
    model = VisionOnlyModel(config, _Inner(config, visual, embed, inner_cls)).eval()
    return model
