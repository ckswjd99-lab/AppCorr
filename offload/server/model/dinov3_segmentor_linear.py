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


class DINOv3SegmentorLinearExecutor(ModelExecutor):
    """Segmentation executor using a frozen ViT backbone + linear probe head.

    Unlike DINOv3SegmentorExecutor which uses the M2F adapter + Mask2FormerHead,
    this executor uses a simple LinearHead (SyncBatchNorm + 1x1 Conv) on top of
    the last ViT block output — significantly faster at head inference time.
    """

    def __init__(self, device: torch.device):
        super().__init__(device)
        self.normalize_avg = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.normalize_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.norm_mean = torch.tensor(self.normalize_avg * 255.0).view(1, 3, 1, 1).to(self.device).float()
        self.norm_std = torch.tensor(self.normalize_std * 255.0).view(1, 3, 1, 1).to(self.device).float()
        self.autocast_dtype = torch.bfloat16
        self.num_classes = 150
        self.vit_backbone = None
        self.linear_head = None

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
            raise ValueError(f"Unsupported segmentor-linear server_eval_mode: {eval_mode}")
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

    # ── Model loading ──────────────────────────────────────────────────

    def load_model(self, model_name: str, config: Any):
        print(f"[Executor] Loading Segmentor-Linear Model (MMap): {model_name}...")
        if self.model is not None:
            del self.model
            torch.cuda.empty_cache()

        self._ensure_dinov3_import_path()
        from dinov3.hub.backbones import dinov3_vit7b16

        profile_config = config.get_input_profile_config()
        self.autocast_dtype = torch.bfloat16
        self.num_classes = int(profile_config.get("num_classes", 150))

        backbone = dinov3_vit7b16(pretrained=False)
        backbone.to(device=self.device)
        backbone.eval()
        backbone.requires_grad_(False)

        linear_head = torch.nn.Module()
        linear_head.batchnorm_layer = torch.nn.BatchNorm2d(backbone.embed_dim)
        linear_head.conv = torch.nn.Conv2d(backbone.embed_dim, self.num_classes, kernel_size=1)
        linear_head.to(device=self.device)

        try:
            backbone_path = profile_config.get(
                "backbone_weights_path",
                "~/cjpark/weights/dinov3/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
            )
            print(f"[Executor] Loading Segmentor-Linear Backbone Weights from {backbone_path}")
            backbone.load_state_dict(load_weight_mmap(backbone_path), strict=True)

            head_path = profile_config.get(
                "segmentor_head_weights_path",
                "~/cjpark/weights/dinov3/dinov3_vit7b16_ade20k_linear_head-custom.pth",
            )
            print(f"[Executor] Loading Segmentor-Linear Head Weights from {head_path}")
            head_ckpt = load_weight_mmap(head_path)
            raw_sd = head_ckpt.get("model", head_ckpt)
            # Strip "segmentation_model.1." prefix from checkpoint keys
            head_sd = {}
            for k, v in raw_sd.items():
                new_key = k.removeprefix("segmentation_model.1.")
                head_sd[new_key] = v
            missing_keys, unexpected_keys = linear_head.load_state_dict(head_sd, strict=True)
            if missing_keys or unexpected_keys:
                raise RuntimeError(
                    "Unexpected linear head weight load result: "
                    f"missing={missing_keys}, unexpected={unexpected_keys}"
                )
            del head_ckpt
        except Exception as e:
            print(f"!!! [Executor] Failed to load segmentor-linear weights: {e}")
            raise e

        linear_head.eval()
        self.vit_backbone = backbone
        self.linear_head = linear_head

    # ── Preprocess / output ────────────────────────────────────────────

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

    def get_final_results(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[int, Any]:
        outputs = context.get("slin_outputs")
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

    # ── Full inference (baseline) ──────────────────────────────────────

    @torch.inference_mode()
    def full_inference(self, task: Task, context: Dict[str, Any], config: Any):
        images = context.get("input_images_uint8")
        if images is None:
            raise RuntimeError("Missing context['input_images_uint8'] for segmentor-linear full_inference().")
        target_shapes = context.get("target_shapes") or [None] * len(images)

        profile_config = config.get_input_profile_config()
        inference_mode = str(profile_config.get("server_inference_mode", "slide"))
        crop_size = int(profile_config.get("server_crop_size", 512))
        stride = int(profile_config.get("server_stride", 341))
        rescale_mode = str(profile_config.get("server_rescale_to", "input")).lower()
        if rescale_mode not in {"input", "original"}:
            raise ValueError(f"Unsupported segmentor-linear server_rescale_to: {rescale_mode}")

        vit_backbone = self.vit_backbone
        linear_head = self.linear_head

        outputs = []

        for image_idx, image_np in enumerate(images):
            tta_tensors, flip_flags, rescale_to = self._build_tta_inputs(image_np, profile_config)
            if rescale_mode == "original" and image_idx < len(target_shapes) and target_shapes[image_idx] is not None:
                rescale_to = target_shapes[image_idx]
            aggregated_preds = torch.zeros(1, self.num_classes, *rescale_to, dtype=torch.float32, device=self.device)

            for img_tensor, apply_flip in zip(tta_tensors, flip_flags):
                if inference_mode == "slide":
                    h_stride = w_stride = stride
                    h_crop = w_crop = crop_size
                    _, _, h_img, w_img = img_tensor.shape
                    if h_crop > h_img and w_crop > w_img:
                        h_crop, w_crop = min(h_img, w_img), min(h_img, w_img)
                    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
                    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

                    slide_preds = torch.zeros(1, self.num_classes, h_img, w_img, dtype=torch.float32, device=self.device)
                    slide_count = torch.zeros(1, 1, h_img, w_img, dtype=torch.int8, device=self.device)

                    for h_idx in range(h_grids):
                        for w_idx in range(w_grids):
                            y1 = h_idx * h_stride
                            x1 = w_idx * w_stride
                            y2 = min(y1 + h_crop, h_img)
                            x2 = min(x1 + w_crop, w_img)
                            y1 = max(y2 - h_crop, 0)
                            x1 = max(x2 - w_crop, 0)
                            crop_img = img_tensor[:, :, y1:y2, x1:x2]

                            # Backbone forward in bf16
                            with torch.autocast("cuda", torch.bfloat16):
                                x_tokens, (tok_H, tok_W) = vit_backbone.prepare_tokens_with_masks(crop_img)
                                rope = vit_backbone.rope_embed(H=tok_H, W=tok_W) if vit_backbone.rope_embed else None
                                for blk in vit_backbone.blocks:
                                    x_tokens = blk(x_tokens, rope)

                            # Head in fp32
                            if vit_backbone.untie_cls_and_patch_norms:
                                normed_cls_reg = vit_backbone.cls_norm(x_tokens[:, :vit_backbone.n_storage_tokens + 1])
                                normed_patch = vit_backbone.norm(x_tokens[:, vit_backbone.n_storage_tokens + 1:])
                                normed_out = torch.cat((normed_cls_reg, normed_patch), dim=1)
                            else:
                                normed_out = vit_backbone.norm(x_tokens)

                            patch_tokens = normed_out[:, vit_backbone.n_storage_tokens + 1:]
                            B, N, C = patch_tokens.shape
                            patch_spatial = patch_tokens.view(B, tok_H, tok_W, C).permute(0, 3, 1, 2).contiguous()
                            pred = linear_head.batchnorm_layer(patch_spatial.float())
                            pred = linear_head.conv(pred)
                            pred = F.interpolate(pred.float(), size=(y2 - y1, x2 - x1), mode="bilinear", align_corners=False)

                            slide_preds[:, :, y1:y2, x1:x2] += pred
                            slide_count[:, :, y1:y2, x1:x2] += 1
                            del crop_img, pred

                    assert (slide_count == 0).sum() == 0
                    pred = slide_preds / slide_count
                    pred = F.interpolate(pred, size=rescale_to, mode="bilinear", align_corners=False)
                    del slide_preds, slide_count
                else:
                    resized = F.interpolate(img_tensor, size=(512, 512), mode="bilinear", align_corners=False)

                    # Backbone forward in bf16
                    with torch.autocast("cuda", torch.bfloat16):
                        x_tokens, (tok_H, tok_W) = vit_backbone.prepare_tokens_with_masks(resized)
                        rope = vit_backbone.rope_embed(H=tok_H, W=tok_W) if vit_backbone.rope_embed else None
                        for blk in vit_backbone.blocks:
                            x_tokens = blk(x_tokens, rope)

                    # Head in fp32
                    if vit_backbone.untie_cls_and_patch_norms:
                        normed_cls_reg = vit_backbone.cls_norm(x_tokens[:, :vit_backbone.n_storage_tokens + 1])
                        normed_patch = vit_backbone.norm(x_tokens[:, vit_backbone.n_storage_tokens + 1:])
                        normed_out = torch.cat((normed_cls_reg, normed_patch), dim=1)
                    else:
                        normed_out = vit_backbone.norm(x_tokens)

                    patch_tokens = normed_out[:, vit_backbone.n_storage_tokens + 1:]
                    B, N, C = patch_tokens.shape
                    patch_spatial = patch_tokens.view(B, tok_H, tok_W, C).permute(0, 3, 1, 2).contiguous()
                    pred = linear_head.batchnorm_layer(patch_spatial.float())
                    pred = linear_head.conv(pred)
                    pred = F.interpolate(pred.float(), size=rescale_to, mode="bilinear", align_corners=False)

                if apply_flip:
                    pred = F.hflip(pred)
                pred = F.softmax(pred, dim=1)
                aggregated_preds += pred.float()
                del pred

            pred_label = (aggregated_preds / len(tta_tensors)).argmax(dim=1)[0].to(torch.uint8)
            outputs.append(pred_label.cpu())
            del aggregated_preds

        context["slin_outputs"] = outputs

    # ── Decomposed inference ───────────────────────────────────────────

    @torch.inference_mode()
    def prepare_tokens(self, task: Task, context: Dict[str, Any], config: Any):
        images = context.get("input_images_uint8")
        if images is None:
            raise RuntimeError("Missing context['input_images_uint8'] for segmentor-linear prepare_tokens().")
        target_shapes = context.get("target_shapes") or [None] * len(images)

        profile_config = config.get_input_profile_config()
        inference_mode = str(profile_config.get("server_inference_mode", "slide"))
        crop_size = int(profile_config.get("server_crop_size", 512))
        stride = int(profile_config.get("server_stride", 341))
        rescale_mode = str(profile_config.get("server_rescale_to", "input")).lower()
        if rescale_mode not in {"input", "original"}:
            raise ValueError(f"Unsupported segmentor-linear server_rescale_to: {rescale_mode}")

        vit_backbone = self.vit_backbone

        all_x_backbones = []
        all_rope_sincos = []
        all_token_shapes = []
        all_crop_hw = []

        image_metas = []

        for image_idx, image_np in enumerate(images):
            with torch.cuda.nvtx.range(f"slin_prepare_tta_img{image_idx}"):
                tta_tensors, flip_flags, rescale_to = self._build_tta_inputs(image_np, profile_config)
            if rescale_mode == "original" and image_idx < len(target_shapes) and target_shapes[image_idx] is not None:
                rescale_to = target_shapes[image_idx]

            tta_source_ranges = []

            for tta_idx, (img_tensor, _apply_flip) in enumerate(zip(tta_tensors, flip_flags)):
                src_start = len(all_x_backbones)

                if inference_mode == "slide":
                    h_stride = w_stride = stride
                    h_crop = w_crop = crop_size
                    _, _, h_img, w_img = img_tensor.shape
                    if h_crop > h_img and w_crop > w_img:
                        h_crop, w_crop = min(h_img, w_img), min(h_img, w_img)
                    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
                    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

                    slide_crops = []
                    for h_idx in range(h_grids):
                        for w_idx in range(w_grids):
                            y1 = h_idx * h_stride
                            x1 = w_idx * w_stride
                            y2 = min(y1 + h_crop, h_img)
                            x2 = min(x1 + w_crop, w_img)
                            y1 = max(y2 - h_crop, 0)
                            x1 = max(x2 - w_crop, 0)
                            crop_img = img_tensor[:, :, y1:y2, x1:x2]

                            with torch.cuda.nvtx.range(f"slin_prepare_src{len(all_x_backbones)}"):
                                with torch.autocast("cuda", self.autocast_dtype):
                                    x_tokens, (tok_H, tok_W) = vit_backbone.prepare_tokens_with_masks(crop_img)
                                    rope = vit_backbone.rope_embed(H=tok_H, W=tok_W) if vit_backbone.rope_embed else None

                            all_x_backbones.append(x_tokens)
                            all_rope_sincos.append(rope)
                            all_token_shapes.append((tok_H, tok_W))
                            all_crop_hw.append((crop_img.shape[2], crop_img.shape[3]))
                            slide_crops.append((y1, y2, x1, x2))

                    tta_source_ranges.append({
                        "src_start": src_start,
                        "src_end": len(all_x_backbones),
                        "mode": "slide",
                        "slide_crops": slide_crops,
                        "h_img": h_img,
                        "w_img": w_img,
                    })
                else:
                    resized = F.interpolate(img_tensor, size=(512, 512), mode="bilinear", align_corners=False)

                    with torch.cuda.nvtx.range(f"slin_prepare_src{len(all_x_backbones)}"):
                        with torch.autocast("cuda", self.autocast_dtype):
                            x_tokens, (tok_H, tok_W) = vit_backbone.prepare_tokens_with_masks(resized)
                            rope = vit_backbone.rope_embed(H=tok_H, W=tok_W) if vit_backbone.rope_embed else None

                    all_x_backbones.append(x_tokens)
                    all_rope_sincos.append(rope)
                    all_token_shapes.append((tok_H, tok_W))
                    all_crop_hw.append((resized.shape[2], resized.shape[3]))

                    tta_source_ranges.append({
                        "src_start": src_start,
                        "src_end": len(all_x_backbones),
                        "mode": "whole",
                    })

            image_metas.append({
                "tta_source_ranges": tta_source_ranges,
                "flip_flags": flip_flags,
                "rescale_to": rescale_to,
                "n_tta": len(tta_tensors),
            })

        context["slin_x_backbones"] = all_x_backbones
        context["slin_rope_sincos"] = all_rope_sincos
        context["slin_token_shapes"] = all_token_shapes
        context["slin_crop_hw"] = all_crop_hw
        context["slin_image_metas"] = image_metas
        context["slin_inference_mode"] = inference_mode
        return {}

    @torch.inference_mode()
    def approx_forward(self, params: Dict[str, Any], context: Dict[str, Any], config: Any):
        start_l, end_l = params.get("layers", (0, 40))

        vit_backbone = self.vit_backbone

        all_x_backbones = context.get("slin_x_backbones")
        all_rope_sincos = context.get("slin_rope_sincos")
        if all_x_backbones is None or all_rope_sincos is None:
            return {}

        if "slin_current_features" not in context:
            context["slin_current_features"] = [x.clone() for x in all_x_backbones]

        current_features = context["slin_current_features"]

        with torch.autocast("cuda", self.autocast_dtype):
            for src_idx in range(len(all_x_backbones)):
                x_tokens = current_features[src_idx] if start_l > 0 else all_x_backbones[src_idx].clone()
                rope = all_rope_sincos[src_idx]

                with torch.cuda.nvtx.range(f"slin_vit_src{src_idx}_L{start_l}-{end_l}"):
                    for lidx in range(start_l, end_l):
                        x_tokens = vit_backbone.blocks[lidx](x_tokens, rope)

                current_features[src_idx] = x_tokens

        context["slin_current_features"] = current_features
        return {}

    @torch.inference_mode()
    def correct_forward(self, params: Dict[str, Any], context: Dict[str, Any], config: Any):
        layers = params.get("layers", (0, 40))
        start_l, end_l = layers[0], layers[1]
        source_idx = params.get("group_id", 0)

        vit_backbone = self.vit_backbone

        all_x_backbones = context.get("slin_x_backbones")
        all_rope_sincos = context.get("slin_rope_sincos")
        if all_x_backbones is None or all_rope_sincos is None:
            return {}
        if source_idx >= len(all_x_backbones):
            return {}

        if "slin_current_features" not in context:
            context["slin_current_features"] = [x.clone() for x in all_x_backbones]

        current_features = context["slin_current_features"]

        x_tokens = all_x_backbones[source_idx].clone()
        rope = all_rope_sincos[source_idx]

        with torch.autocast("cuda", self.autocast_dtype):
            for lidx in range(start_l, end_l):
                x_tokens = vit_backbone.blocks[lidx](x_tokens, rope)

        current_features[source_idx] = x_tokens
        return {}

    @torch.inference_mode()
    def head_inference(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[str, Any]:
        vit_backbone = self.vit_backbone
        linear_head = self.linear_head

        current_features = context.get("slin_current_features")
        token_shapes = context.get("slin_token_shapes")
        crop_hws = context.get("slin_crop_hw")
        image_metas = context.get("slin_image_metas")

        if current_features is None or token_shapes is None or image_metas is None:
            raise RuntimeError("Missing context for segmentor-linear head_inference().")

        total_sources = len(current_features)

        # Phase 1: Batch sources by (tok_H, tok_W) shape, run norm + LinearHead
        shape_groups = {}
        for si in range(total_sources):
            shape = token_shapes[si]
            shape_groups.setdefault(shape, []).append(si)

        source_preds = [None] * total_sources

        for shape_key, src_indices in shape_groups.items():
            tok_H, tok_W = shape_key
            with torch.cuda.nvtx.range(f"slin_head_batch{len(src_indices)}src"):
                # Batch norm
                raw_outs = [current_features[si] for si in src_indices]
                batched_raw = torch.cat(raw_outs, dim=0)

                with torch.autocast("cuda", self.autocast_dtype):
                    if vit_backbone.untie_cls_and_patch_norms:
                        normed_cls_reg = vit_backbone.cls_norm(batched_raw[:, :vit_backbone.n_storage_tokens + 1])
                        normed_patch = vit_backbone.norm(batched_raw[:, vit_backbone.n_storage_tokens + 1:])
                        normed_out = torch.cat((normed_cls_reg, normed_patch), dim=1)
                    else:
                        normed_out = vit_backbone.norm(batched_raw)

                patch_tokens = normed_out[:, vit_backbone.n_storage_tokens + 1:]
                N_batch = patch_tokens.shape[0]
                C = patch_tokens.shape[2]
                patch_spatial = patch_tokens.view(N_batch, tok_H, tok_W, C).permute(0, 3, 1, 2).contiguous()

                # LinearHead in fp32
                pred = linear_head.batchnorm_layer(patch_spatial.float())
                pred = linear_head.conv(pred)

                for local_i, si in enumerate(src_indices):
                    crop_hw = crop_hws[si]
                    p = pred[local_i].unsqueeze(0)
                    p = F.interpolate(p.float(), size=crop_hw, mode="bilinear", align_corners=False)
                    source_preds[si] = p

                del batched_raw, patch_tokens, patch_spatial, pred

        # Phase 2: Per-image slide/TTA aggregation
        outputs = []

        for image_idx, image_meta in enumerate(image_metas):
            rescale_to = image_meta["rescale_to"]
            flip_flags = image_meta["flip_flags"]
            n_tta = image_meta["n_tta"]
            tta_source_ranges = image_meta["tta_source_ranges"]

            aggregated_preds = torch.zeros(1, self.num_classes, *rescale_to, dtype=torch.float32, device=self.device)

            for tta_idx, tta_range in enumerate(tta_source_ranges):
                src_start = tta_range["src_start"]
                src_end = tta_range["src_end"]
                apply_flip = flip_flags[tta_idx]

                if tta_range["mode"] == "slide":
                    h_img = tta_range["h_img"]
                    w_img = tta_range["w_img"]
                    slide_crops = tta_range["slide_crops"]

                    slide_preds = torch.zeros(1, self.num_classes, h_img, w_img, dtype=torch.float32, device=self.device)
                    slide_count = torch.zeros(1, 1, h_img, w_img, dtype=torch.int8, device=self.device)

                    for crop_local_idx, (y1, y2, x1, x2) in enumerate(slide_crops):
                        si = src_start + crop_local_idx
                        slide_preds[:, :, y1:y2, x1:x2] += source_preds[si]
                        slide_count[:, :, y1:y2, x1:x2] += 1

                    assert (slide_count == 0).sum() == 0
                    pred = slide_preds / slide_count
                    pred = F.interpolate(pred, size=rescale_to, mode="bilinear", align_corners=False)
                    del slide_preds, slide_count
                else:
                    src_idx = src_start
                    pred = source_preds[src_idx]

                if apply_flip:
                    pred = F.hflip(pred)
                pred = F.softmax(pred, dim=1)
                aggregated_preds += pred.float()
                del pred

            pred_label = (aggregated_preds / n_tta).argmax(dim=1)[0].to(torch.uint8)
            outputs.append(pred_label.cpu())
            del aggregated_preds

        context["slin_outputs"] = outputs
        return {}

        # Phase 2: Per-image slide/TTA aggregation
        outputs = []

        for image_idx, image_meta in enumerate(image_metas):
            rescale_to = image_meta["rescale_to"]
            flip_flags = image_meta["flip_flags"]
            n_tta = image_meta["n_tta"]
            tta_source_ranges = image_meta["tta_source_ranges"]

            aggregated_preds = torch.zeros(1, self.num_classes, *rescale_to, dtype=torch.float32, device=self.device)

            for tta_idx, tta_range in enumerate(tta_source_ranges):
                src_start = tta_range["src_start"]
                src_end = tta_range["src_end"]
                apply_flip = flip_flags[tta_idx]

                if tta_range["mode"] == "slide":
                    h_img = tta_range["h_img"]
                    w_img = tta_range["w_img"]
                    slide_crops = tta_range["slide_crops"]

                    slide_preds = torch.zeros(1, self.num_classes, h_img, w_img, dtype=torch.float32, device=self.device)
                    slide_count = torch.zeros(1, 1, h_img, w_img, dtype=torch.int8, device=self.device)

                    for crop_local_idx, (y1, y2, x1, x2) in enumerate(slide_crops):
                        si = src_start + crop_local_idx
                        slide_preds[:, :, y1:y2, x1:x2] += source_preds[si]
                        slide_count[:, :, y1:y2, x1:x2] += 1

                    assert (slide_count == 0).sum() == 0
                    pred = slide_preds / slide_count
                    pred = F.interpolate(pred, size=rescale_to, mode="bilinear", align_corners=False)
                    del slide_preds, slide_count
                else:
                    src_idx = src_start
                    pred = source_preds[src_idx]

                if apply_flip:
                    pred = F.hflip(pred)
                pred = F.softmax(pred, dim=1)
                aggregated_preds += pred.float()
                del pred

            pred_label = (aggregated_preds / n_tta).argmax(dim=1)[0].to(torch.uint8)
            outputs.append(pred_label.cpu())
            del aggregated_preds

        context["slin_outputs"] = outputs
        return {}

    def decide_exit(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[str, Any]:
        return {"num_exits": 0}
