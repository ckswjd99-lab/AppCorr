from __future__ import annotations

import hashlib
import math
import os
import sys
from functools import partial
from typing import Any, Dict, List

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from offload.common import Task
from .base import ModelExecutor
from .utils import load_weight_mmap


class DINOv3SegmentorExecutor(ModelExecutor):
    def __init__(self, device: torch.device):
        super().__init__(device)
        self.normalize_avg = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.normalize_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.norm_mean = torch.tensor(self.normalize_avg * 255.0).view(1, 3, 1, 1).to(self.device).float()
        self.norm_std = torch.tensor(self.normalize_std * 255.0).view(1, 3, 1, 1).to(self.device).float()
        self.autocast_dtype = torch.bfloat16
        self.num_classes = 150
        self._make_inference = None

    @staticmethod
    def _ensure_dinov3_import_path():
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
        dinov3_parent = os.path.join(repo_root, "appcorr", "models")
        if dinov3_parent not in sys.path:
            sys.path.insert(0, dinov3_parent)

    @staticmethod
    def _dtype_from_config(name: str | torch.dtype) -> torch.dtype:
        if isinstance(name, torch.dtype):
            return name
        normalized = str(name).lower()
        if normalized in {"bf16", "bfloat16", "torch.bfloat16"}:
            return torch.bfloat16
        if normalized in {"fp16", "float16", "half", "torch.float16"}:
            return torch.float16
        if normalized in {"fp32", "float32", "torch.float32"}:
            return torch.float32
        raise ValueError(f"Unsupported autocast dtype: {name}")

    @staticmethod
    def _resize_short_side(image: Image.Image, short_side: int) -> Image.Image:
        width, height = image.size
        if height > width:
            new_width = short_side
            new_height = int(short_side * height / width + 0.5)
        else:
            new_height = short_side
            new_width = int(short_side * width / height + 0.5)
        return image.resize((new_width, new_height), Image.Resampling.BILINEAR)

    @staticmethod
    def _tta_short_side(base_short_side: int, ratio: float) -> int:
        short_side = int(base_short_side * ratio)
        if ratio < 1:
            short_side = int(math.ceil(short_side / 32.0)) * 32
        return short_side

    def load_model(self, model_name: str, config: Any):
        print(f"[Executor] Loading Segmentor Model (MMap): {model_name}...")
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()

        self._ensure_dinov3_import_path()
        from dinov3.eval.segmentation.inference import make_inference
        from dinov3.hub.segmentors import dinov3_vit7b16_ms

        profile_config = config.get_input_profile_config()
        self.autocast_dtype = self._dtype_from_config(profile_config.get("autocast_dtype", "bfloat16"))
        self.num_classes = int(profile_config.get("num_classes", 150))

        self.model = dinov3_vit7b16_ms(pretrained=False, autocast_dtype=self.autocast_dtype)
        # Keep module parameters in their original dtype. The M2F adapter has
        # BatchNorm layers that expect float weights; bf16 is handled by the
        # decoder's autocast context during inference.
        self.model.to(device=self.device)

        try:
            backbone_path = profile_config.get(
                "backbone_weights_path",
                "~/cjpark/weights/dinov3/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
            )
            print(f"[Executor] Loading Segmentor Backbone Weights from {backbone_path}")
            vit_backbone = self.model.segmentation_model[0].backbone
            vit_backbone.load_state_dict(load_weight_mmap(backbone_path), strict=True)

            head_path = profile_config.get(
                "segmentor_head_weights_path",
                "~/cjpark/weights/dinov3/dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth",
            )
            print(f"[Executor] Loading Segmentor Head Weights from {head_path}")
            head_ckpt = load_weight_mmap(head_path)
            state_dict = head_ckpt.get("model", head_ckpt)
            missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
            bad_missing = [key for key in missing_keys if "backbone" not in key]
            if bad_missing or unexpected_keys:
                raise RuntimeError(
                    "Unexpected segmentor weight load result: "
                    f"missing(non-backbone)={bad_missing}, unexpected={unexpected_keys}"
                )
            del head_ckpt
        except Exception as e:
            print(f"!!! [Executor] Failed to load segmentor weights: {e}")
            raise e

        self.model.eval()
        self._make_inference = make_inference

    def preprocess(self, batch_data: Any, task: Task, context: Dict[str, Any], config: Any):
        images, target_shapes = self._as_image_list(batch_data)
        active_indices = context.get("active_indices")
        if active_indices is not None:
            indices = active_indices.detach().cpu().tolist()
            filtered_images = []
            filtered_target_shapes = []
            for idx in indices:
                idx = int(idx)
                if idx < len(images):
                    filtered_images.append(images[idx])
                    filtered_target_shapes.append(target_shapes[idx] if idx < len(target_shapes) else None)
            images = filtered_images
            target_shapes = filtered_target_shapes
        context["input_images_uint8"] = [np.ascontiguousarray(image) for image in images]
        context["target_shapes"] = target_shapes

    def _as_image_list(self, batch_data: Any) -> tuple[List[np.ndarray], List[tuple[int, int] | None]]:
        if isinstance(batch_data, list):
            image_list = batch_data
        elif torch.is_tensor(batch_data):
            tensor = batch_data.detach().cpu()
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim != 4:
                raise ValueError(f"Expected 3D/4D image tensor input, got {tuple(tensor.shape)}")
            if tensor.shape[1] == 3:
                tensor = tensor.permute(0, 2, 3, 1)
            image_list = [tensor[idx].numpy() for idx in range(tensor.shape[0])]
        elif isinstance(batch_data, np.ndarray):
            if batch_data.ndim == 3:
                image_list = [batch_data]
            elif batch_data.ndim == 4:
                image_list = [batch_data[idx] for idx in range(batch_data.shape[0])]
            else:
                raise ValueError(f"Expected 3D/4D numpy image input, got {batch_data.shape}")
        else:
            raise TypeError(f"Unsupported segmentor input type: {type(batch_data)}")

        normalized = []
        target_shapes = []
        for item in image_list:
            image = item
            target_shape = None
            if isinstance(item, dict):
                image = item.get("image")
                target_shape = item.get("target_shape")
            if image is None:
                raise ValueError("Segmentor received an empty image slot.")
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"Expected HWC image with 3 channels, got {image.shape}")
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            normalized.append(image)
            target_shapes.append(tuple(int(v) for v in target_shape) if target_shape is not None else None)
        return normalized, target_shapes

    def _pil_to_normalized_tensor(self, image: Image.Image) -> torch.Tensor:
        image_np = np.array(image, dtype=np.uint8, copy=True)
        tensor = torch.from_numpy(image_np).to(device=self.device, non_blocking=True)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).float()
        tensor = (tensor - self.norm_mean) / self.norm_std
        return tensor.to(dtype=self.autocast_dtype)

    def _build_tta_inputs(
        self,
        image_np: np.ndarray,
        profile_config: Dict[str, Any],
    ) -> tuple[List[torch.Tensor], List[bool], tuple[int, int]]:
        base_image = Image.fromarray(image_np)
        base_h, base_w = image_np.shape[:2]
        eval_mode = str(profile_config.get("server_eval_mode", "tta")).lower()
        if eval_mode not in {"tta", "single"}:
            raise ValueError(f"Unsupported ADE20K segmentor server_eval_mode: {eval_mode}")
        use_tta = eval_mode == "tta" and bool(profile_config.get("server_use_tta", True))

        if not use_tta:
            return [self._pil_to_normalized_tensor(base_image)], [False], (base_h, base_w)

        base_short_side = int(profile_config.get("mobile_resize_short_side", min(base_h, base_w)))
        tta_ratios = list(profile_config.get("server_tta_ratios", [1.0]))
        resized_images = [
            self._resize_short_side(base_image, self._tta_short_side(base_short_side, float(ratio)))
            for ratio in tta_ratios
        ]
        augmented_images = resized_images + [
            image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            for image in resized_images
        ]
        flip_flags = [False] * len(resized_images) + [True] * len(resized_images)
        tensors = [self._pil_to_normalized_tensor(image) for image in augmented_images]
        return tensors, flip_flags, (base_h, base_w)

    @torch.inference_mode()
    def full_inference(self, task: Task, context: Dict[str, Any], config: Any):
        if self._make_inference is None:
            raise RuntimeError("Segmentor inference helper is not loaded.")

        images = context.get("input_images_uint8")
        if images is None:
            raise RuntimeError("Missing context['input_images_uint8'] for segmentor full_inference().")
        target_shapes = context.get("target_shapes") or [None] * len(images)

        profile_config = config.get_input_profile_config()
        inference_mode = str(profile_config.get("server_inference_mode", "slide"))
        crop_size = int(profile_config.get("server_crop_size", 896))
        stride = int(profile_config.get("server_stride", 596))
        decoder_head_type = str(profile_config.get("decoder_head_type", "m2f"))
        rescale_mode = str(profile_config.get("server_rescale_to", "input")).lower()
        if rescale_mode not in {"input", "original"}:
            raise ValueError(f"Unsupported ADE20K segmentor server_rescale_to: {rescale_mode}")
        outputs = []

        for image_idx, image_np in enumerate(images):
            tta_tensors, flip_flags, rescale_to = self._build_tta_inputs(image_np, profile_config)
            if rescale_mode == "original" and image_idx < len(target_shapes) and target_shapes[image_idx] is not None:
                rescale_to = target_shapes[image_idx]
            aggregated_preds = torch.zeros(1, self.num_classes, *rescale_to, dtype=torch.float32)
            for img_tensor, apply_flip in zip(tta_tensors, flip_flags):
                pred = self._make_inference(
                    img_tensor,
                    self.model,
                    inference_mode=inference_mode,
                    decoder_head_type=decoder_head_type,
                    rescale_to=rescale_to,
                    n_output_channels=self.num_classes,
                    crop_size=(crop_size, crop_size),
                    stride=(stride, stride),
                    apply_horizontal_flip=apply_flip,
                    output_activation=partial(F.softmax, dim=1),
                )
                aggregated_preds += pred.float()
                del pred

            pred_label = (aggregated_preds / len(tta_tensors)).argmax(dim=1)[0].to(torch.uint8)
            outputs.append(pred_label.cpu())
            del aggregated_preds, tta_tensors

        context["seg_outputs"] = outputs

    def get_final_results(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[int, Any]:
        outputs = context.get("seg_outputs")
        if outputs is None:
            return {}

        active_indices = context.get("active_indices")
        if active_indices is None:
            indices = list(range(len(outputs)))
        else:
            indices = active_indices.detach().cpu().tolist()

        results = {}
        for local_idx, orig_idx in enumerate(indices):
            if local_idx >= len(outputs):
                break
            results[int(orig_idx)] = self._format_segmentation_output(outputs[local_idx])
        return results

    def _format_segmentation_output(self, pred_label: torch.Tensor) -> Dict[str, Any]:
        pred_np = pred_label.detach().cpu().numpy().astype(np.uint8, copy=False)
        histogram = torch.bincount(pred_label.long().flatten(), minlength=self.num_classes)[:self.num_classes]
        return {
            "shape": [int(pred_np.shape[0]), int(pred_np.shape[1])],
            "num_classes": int(self.num_classes),
            "sha256": hashlib.sha256(np.ascontiguousarray(pred_np).tobytes()).hexdigest(),
            "histogram": histogram.cpu().tolist(),
            "mask": np.ascontiguousarray(pred_np),
        }

    def prepare_tokens(self, task: Task, context: Dict[str, Any], config: Any):
        return {}

    def approx_forward(self, params: Dict[str, Any], context: Dict[str, Any], config: Any):
        return {}

    def correct_forward(self, params: Dict[str, Any], context: Dict[str, Any], config: Any):
        return {}

    def head_inference(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[str, Any]:
        return {}

    def decide_exit(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[str, Any]:
        return {"num_exits": 0}
