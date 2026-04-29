from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from offload.common import Task
from .base import ModelExecutor
from .utils import load_weight_mmap


class DINOv3DeptherExecutor(ModelExecutor):
    def __init__(self, device: torch.device):
        super().__init__(device)
        self.normalize_avg = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.normalize_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.norm_mean = torch.tensor(self.normalize_avg * 255.0).view(1, 3, 1, 1).to(self.device).float()
        self.norm_std = torch.tensor(self.normalize_std * 255.0).view(1, 3, 1, 1).to(self.device).float()
        self.autocast_dtype = torch.bfloat16

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

    def load_model(self, model_name: str, config: Any):
        print(f"[Executor] Loading Depther Model (MMap): {model_name}...")
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()

        self._ensure_dinov3_import_path()
        from dinov3.hub.depthers import dinov3_vit7b16_dd, _get_depther_config
        from dinov3.hub.backbones import dinov3_vit7b16
        from dinov3.eval.depth.models import make_depther_from_config

        profile_config = config.get_input_profile_config()
        self.autocast_dtype = self._dtype_from_config(profile_config.get("autocast_dtype", "bfloat16"))

        backbone = dinov3_vit7b16(pretrained=False)
        depther_config = _get_depther_config("dinov3_vit7b16")
        self.model = make_depther_from_config(
            backbone,
            config=depther_config,
            autocast_dtype=self.autocast_dtype,
        )
        self.model.to(device=self.device)

        try:
            backbone_path = profile_config.get(
                "backbone_weights_path",
                "~/cjpark/weights/dinov3/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
            )
            print(f"[Executor] Loading Depther Backbone Weights from {backbone_path}")
            vit_backbone = self.model.encoder.backbone
            vit_backbone.load_state_dict(load_weight_mmap(backbone_path), strict=True)

            head_path = profile_config.get(
                "depther_head_weights_path",
                "~/cjpark/weights/dinov3/dinov3_vit7b16_synthmix_dpt_head-02040be1.pth",
            )
            print(f"[Executor] Loading Depther Head Weights from {head_path}")
            head_ckpt = load_weight_mmap(head_path)
            self.model.decoder.load_state_dict(head_ckpt, strict=True)
            del head_ckpt
        except Exception as e:
            print(f"!!! [Executor] Failed to load depther weights: {e}")
            raise e

        self.model.eval()

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
            raise TypeError(f"Unsupported depther input type: {type(batch_data)}")

        normalized = []
        target_shapes = []
        for item in image_list:
            image = item
            target_shape = None
            if isinstance(item, dict):
                image = item.get("image")
                target_shape = item.get("target_shape")
            if image is None:
                raise ValueError("Depther received an empty image slot.")
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError(f"Expected HWC image with 3 channels, got {image.shape}")
            if image.dtype != np.uint8:
                image = np.clip(image, 0, 255).astype(np.uint8)
            normalized.append(image)
            target_shapes.append(tuple(int(v) for v in target_shape) if target_shape is not None else None)
        return normalized, target_shapes

    def _pil_to_normalized_tensor(self, image_np: np.ndarray) -> torch.Tensor:
        tensor = torch.from_numpy(image_np).to(device=self.device, non_blocking=True)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).float()
        tensor = (tensor - self.norm_mean) / self.norm_std
        return tensor.to(dtype=self.autocast_dtype)

    @torch.inference_mode()
    def full_inference(self, task: Task, context: Dict[str, Any], config: Any):
        images = context.get("input_images_uint8")
        if images is None:
            raise RuntimeError("Missing context['input_images_uint8'] for depther full_inference().")
        target_shapes = context.get("target_shapes") or [None] * len(images)

        profile_config = config.get_input_profile_config()
        eval_size = int(profile_config.get("depther_eval_size", 768))
        use_tta = bool(profile_config.get("depther_use_tta", True))

        outputs = []
        for image_idx, image_np in enumerate(images):
            target_shape = target_shapes[image_idx] if image_idx < len(target_shapes) else None
            img_tensor = self._pil_to_normalized_tensor(image_np)
            input_tensor = F.interpolate(img_tensor, size=(eval_size, eval_size), mode="bilinear", align_corners=False)

            with torch.autocast("cuda", self.autocast_dtype):
                if use_tta:
                    flipped = torch.flip(input_tensor, [-1])
                    pred = self.model(input_tensor)
                    pred_flip = self.model(flipped)
                    pred_flip = torch.flip(pred_flip, [-1])
                    depth = (pred + pred_flip) / 2.0
                else:
                    depth = self.model(input_tensor)

            rescale_to = target_shape if target_shape is not None else image_np.shape[:2]
            if depth.shape[-2:] != rescale_to:
                depth = F.interpolate(depth, size=rescale_to, mode="bilinear", align_corners=False)

            outputs.append(depth[0, 0].cpu())
            del img_tensor, input_tensor, depth

        context["depth_outputs"] = outputs

    def get_final_results(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[int, Any]:
        outputs = context.get("depth_outputs")
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
            results[int(orig_idx)] = self._format_depth_output(outputs[local_idx])
        return results

    def _format_depth_output(self, depth_map: torch.Tensor) -> Dict[str, Any]:
        depth_np = depth_map.detach().cpu().float().numpy().astype(np.float32, copy=True)
        return {
            "shape": [int(depth_np.shape[0]), int(depth_np.shape[1])],
            "min": float(depth_np.min()),
            "max": float(depth_np.max()),
            "mean": float(depth_np.mean()),
            "depth": np.ascontiguousarray(depth_np),
        }

    def prepare_tokens(self, task: Task, context: Dict[str, Any], config: Any):
        raise NotImplementedError

    def approx_forward(self, params: Dict[str, Any], context: Dict[str, Any], config: Any):
        raise NotImplementedError

    def correct_forward(self, params: Dict[str, Any], context: Dict[str, Any], config: Any):
        raise NotImplementedError

    def head_inference(self, task: Task, context: Dict[str, Any], config: Any):
        raise NotImplementedError

    def decide_exit(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[str, Any]:
        return {"num_exits": 0}
