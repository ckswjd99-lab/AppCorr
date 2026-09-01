"""SAM 3 executor: approx/correct on the vision tower, stock model for everything after it.

`Sam3Model.forward` accepts `vision_embeds` in place of `pixel_values`, so the detector -- text
encoder, geometry encoder, DETR encoder/decoder, mask decoder, scoring -- runs completely untouched
on whatever features the tower produced. Nothing downstream of the FPN is reimplemented here, which
is what keeps the comparison honest: the only thing that differs between the floor, the corrected
arms and the ceiling is the vision features.

Prompting is by ground-truth box. That is deliberate for a first benchmark: it makes the run
deterministic and isolates *segmentation* quality, where a text or point prompt would fold the
model's detection or disambiguation behaviour into the same number. `input_boxes` are carried
per-image through the context.

Geometry, from `facebook/sam3`'s config:

    1008 x 1008 input, patch 14  ->  72 x 72 = 5184 tokens
    32 ViT layers, hidden 1024, window 24 (9 windows)
    global attention only at layers [7, 15, 23, 31]

Measured cache cost of the split: **1.90 GB per image** (K/V 1.27 + increments 0.63), which is 4.6x
*smaller* than the DINOv3 ADE20K path despite carrying 1.65x the tokens, because hidden is 1024
rather than 4096. Batch size is not cache-bound here.

The fork is verified against stock to **exactly zero** difference at full resolution with real
weights -- `analysis/experiments/sam3_vision_fork_unittest.py` for the layer, and the tower check
recorded in the memo. Any drift from that is a regression, not a tolerance.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from appcorr.models.sam3.vision.backbone import ApproxCorrectSam3VisionTower

from .base import ModelExecutor
from offload.common.protocol import Task


class Sam3Executor(ModelExecutor):
    """AppCorr executor for SAM 3 image segmentation."""

    def __init__(self, device: torch.device):
        super().__init__(device)
        self.model = None
        self.processor = None
        self.tower: ApproxCorrectSam3VisionTower | None = None
        self.norm_mean = None
        self.norm_std = None

    # ------------------------------------------------------------------ setup

    def load_model(self, model_name: str, config: Any):
        from transformers import Sam3Model, Sam3Processor

        repo = str(getattr(config, "hf_repo", None) or "facebook/sam3")
        dtype = torch.bfloat16 if str(getattr(config, "autocast_dtype", "bfloat16")) == "bfloat16" else torch.float32
        self.model = Sam3Model.from_pretrained(repo, dtype=dtype).to(self.device).eval()
        self.model.requires_grad_(False)
        self.processor = Sam3Processor.from_pretrained(repo)
        self.tower = ApproxCorrectSam3VisionTower(self.model.vision_encoder).eval()

        ip = getattr(self.processor, "image_processor", self.processor)
        mean = torch.tensor(getattr(ip, "image_mean", [0.485, 0.456, 0.406]), device=self.device)
        std = torch.tensor(getattr(ip, "image_std", [0.229, 0.224, 0.225]), device=self.device)
        self.norm_mean = mean.view(1, 3, 1, 1)
        self.norm_std = std.view(1, 3, 1, 1)
        print(
            f"[SAM3] Loaded {repo}: {self.tower.num_layers} ViT layers, "
            f"global attention at {self.tower.global_layers}, patch {self.tower.patch_size}.",
            flush=True,
        )

    def _grid(self, config: Any) -> tuple[int, int]:
        h, w = config.image_shape[0], config.image_shape[1]
        p = self.tower.patch_size
        return h // p, w // p

    # ------------------------------------------------------------- preprocess

    def preprocess(self, batch_data: Any, task: Task, context: Dict[str, Any], config: Any):
        if isinstance(batch_data, torch.Tensor):
            tensor = batch_data.to(device=self.device, non_blocking=True)
            if tensor.shape[1] != 3 and tensor.shape[-1] == 3:
                tensor = tensor.permute(0, 3, 1, 2)
            tensor = tensor.float()
            if batch_data.dtype == torch.uint8:
                tensor = tensor / 255.0
        else:
            tensor = torch.from_numpy(batch_data).to(device=self.device, non_blocking=True)
            tensor = tensor.permute(0, 3, 1, 2).float() / 255.0

        tensor = (tensor - self.norm_mean) / self.norm_std
        context["input_tensor"] = tensor.to(dtype=self.model.dtype)

        height, width = self._grid(config)
        num_patches = height * width
        if "group_map" not in context:
            context["group_map"] = torch.full((1, num_patches), -1, device=self.device, dtype=torch.long)
        group_map = context["group_map"]
        for p in task.payload:
            if 0 <= p.spatial_idx < num_patches:
                group_map[0, p.spatial_idx] = p.group_id

    def prepare_tokens(self, task: Task, context: Dict[str, Any], config: Any):
        if "input_tensor" not in context:
            return
        tokens = self.tower.prepare_tokens(context["input_tensor"])
        context["input_tokens"] = tokens
        context.setdefault("current_feature", tokens)

    # ------------------------------------------------------------ the two arms

    def approx_forward(self, params: Dict[str, Any], context: Dict[str, Any], config: Any):
        start_l, end_l = params.get("layers", (0, self.tower.num_layers))
        x = context["input_tokens"] if start_l == 0 else context.get("current_feature", context["input_tokens"])
        cache = context.get("cache_feature", {})
        x, cache = self.tower.approx_forward(x, cache, layers=(start_l, end_l))
        context["current_feature"] = x
        context["cache_feature"] = cache

    def correct_forward(self, params: Dict[str, Any], context: Dict[str, Any], config: Any):
        start_l, end_l = params.get("layers", (0, self.tower.num_layers))
        group_id = params.get("group_id", 1)
        patch_idx = torch.where(context["group_map"][0] == group_id)[0]
        if patch_idx.numel() == 0:
            return
        # Correction starts from the *approximate* stream, not from the previous round's output:
        # every position outside `patch_idx` is reconstructed from the cached increment, so feeding
        # anything else would double-count what earlier rounds already applied.
        x = context["input_tokens"]
        cache = context.get("cache_feature", {})
        x, cache = self.tower.correct_forward(x, patch_idx, cache, layers=(start_l, end_l))
        context["current_feature"] = x
        context["cache_feature"] = cache

    # -------------------------------------------------------------------- head

    def _run_head(self, context: Dict[str, Any], hidden: torch.Tensor) -> Dict[str, Any]:
        """FPN + the untouched detector stack, prompted by the image's ground-truth boxes."""
        from transformers.models.sam3.modeling_sam3 import Sam3VisionEncoderOutput

        fpn_hidden_states, fpn_position_encoding = self.tower.run_neck(hidden)
        vision_embeds = Sam3VisionEncoderOutput(
            fpn_hidden_states=fpn_hidden_states,
            fpn_position_encoding=fpn_position_encoding,
        )
        boxes = context.get("input_boxes")
        if boxes is None:
            raise ValueError(
                "SAM 3 is prompted by ground-truth boxes; the dataset must supply `input_boxes` "
                "in the request context. An unprompted forward would measure something else."
            )
        with torch.no_grad():
            out = self.model(vision_embeds=vision_embeds, input_boxes=boxes,
                             text_embeds=context.get("text_embeds"))
        context["output"] = out
        return {"num_masks": int(getattr(out, "pred_masks", torch.empty(0)).shape[1])
                if hasattr(out, "pred_masks") else 0}

    def head_inference(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[str, Any]:
        return self._run_head(context, context["current_feature"])

    def full_inference(self, task: Task, context: Dict[str, Any], config: Any):
        """Ceiling arm: the stock tower on the full-resolution image, then the same head."""
        inp = context.get("input_tensor")
        if inp is None:
            return
        with torch.no_grad():
            hidden = self.tower.prepare_tokens(inp)
            for layer in self.tower.layers:
                hidden = layer(hidden)
        self._run_head(context, hidden)

    def get_final_results(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[int, Any]:
        out = context.get("output")
        if out is None:
            return {}
        masks = getattr(out, "pred_masks", None)
        scores = getattr(out, "iou_scores", None)
        return {
            0: {
                "masks": masks.float().cpu().numpy().tolist() if masks is not None else None,
                "scores": scores.float().cpu().numpy().tolist() if scores is not None else None,
            }
        }

    def decide_exit(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[str, Any]:
        return {}
