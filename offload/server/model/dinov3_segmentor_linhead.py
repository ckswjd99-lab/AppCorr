from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from offload.common import Task
from offload.common.protocol import normalize_appcorr_kwargs
from .base import ModelExecutor
from .utils import load_weight_mmap

from appcorr.models.dinov3.models.vision_transformer import create_group_index


@dataclass
class QueryState:
    query_pos_idx: torch.Tensor
    query_valid_mask: torch.Tensor
    active_batch_idx: torch.Tensor
    active_pos_idx: torch.Tensor
    active_token_idx: torch.Tensor
    active_query_pos: torch.Tensor
    active_query_pos_padded: torch.Tensor
    active_query_mask: torch.Tensor
    all_valid: bool


@dataclass
class GroupCorrectionPlan:
    num_pretokens: int
    prefix_dindice: torch.Tensor
    group_patch_dindice: torch.Tensor
    group_patch_keep_local_idx: torch.Tensor
    full_dindice: torch.Tensor
    pruned_dindice: torch.Tensor
    query_state: QueryState
    kept_patch_count: torch.Tensor
    full_patch_count: torch.Tensor
    kept_residual_mass: torch.Tensor
    full_residual_mass: torch.Tensor


class DINOv3SegmentorLinheadExecutor(ModelExecutor):
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
            raise ValueError(f"Unsupported segmentor-linhead server_eval_mode: {eval_mode}")
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
        print(f"[Executor] Loading Segmentor-Linhead Model (MMap): {model_name}...")
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
            print(f"[Executor] Loading Segmentor-Linhead Backbone Weights from {backbone_path}")
            backbone.load_state_dict(load_weight_mmap(backbone_path), strict=True)

            head_path = profile_config.get(
                "segmentor_head_weights_path",
                "~/cjpark/weights/dinov3/dinov3_vit7b16_ade20k_linear_head-custom.pth",
            )
            print(f"[Executor] Loading Segmentor-Linhead Head Weights from {head_path}")
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
            print(f"!!! [Executor] Failed to load segmentor-linhead weights: {e}")
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
        return {
            "mask": np.ascontiguousarray(pred_np),
        }

    def get_final_results(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[int, Any]:
        outputs = context.get("slinhead_outputs")
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
            raise RuntimeError("Missing context['input_images_uint8'] for segmentor-linhead full_inference().")
        target_shapes = context.get("target_shapes") or [None] * len(images)

        profile_config = config.get_input_profile_config()
        inference_mode = str(profile_config.get("server_inference_mode", "slide"))
        crop_size = int(profile_config.get("server_crop_size", 512))
        stride = int(profile_config.get("server_stride", 341))
        rescale_mode = str(profile_config.get("server_rescale_to", "input")).lower()
        if rescale_mode not in {"input", "original"}:
            raise ValueError(f"Unsupported segmentor-linhead server_rescale_to: {rescale_mode}")

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

        context["slinhead_outputs"] = outputs

    # ── Decomposed inference ───────────────────────────────────────────

    def prepare_tokens(self, task: Task, context: Dict[str, Any], config: Any):
        images = context.get("input_images_uint8")
        if images is None:
            raise RuntimeError("Missing context['input_images_uint8'] for segmentor-linhead prepare_tokens().")
        target_shapes = context.get("target_shapes") or [None] * len(images)

        profile_config = config.get_input_profile_config()
        inference_mode = str(profile_config.get("server_inference_mode", "slide"))
        crop_size = int(profile_config.get("server_crop_size", 512))
        stride = int(profile_config.get("server_stride", 341))
        rescale_mode = str(profile_config.get("server_rescale_to", "input")).lower()
        if rescale_mode not in {"input", "original"}:
            raise ValueError(f"Unsupported segmentor-linhead server_rescale_to: {rescale_mode}")

        vit_backbone = self.vit_backbone

        all_x_backbones = []
        all_rope_sincos = []
        all_token_shapes = []
        all_crop_hw = []

        image_metas = []

        for image_idx, image_np in enumerate(images):
            with torch.cuda.nvtx.range(f"slinhead_prepare_tta_img{image_idx}"):
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

                            with torch.cuda.nvtx.range(f"slinhead_prepare_src{len(all_x_backbones)}"):
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

                    with torch.cuda.nvtx.range(f"slinhead_prepare_src{len(all_x_backbones)}"):
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

        context["slinhead_x_backbones"] = all_x_backbones
        context["slinhead_rope_sincos"] = all_rope_sincos
        context["slinhead_token_shapes"] = all_token_shapes
        context["slinhead_crop_hw"] = all_crop_hw
        context["slinhead_image_metas"] = image_metas
        context["slinhead_inference_mode"] = inference_mode
        context.pop("slinhead_group_maps", None)
        context.pop("slinhead_group_plans", None)
        context.pop("slinhead_cached_dindices", None)
        context.pop("slinhead_current_features", None)
        self._ensure_group_maps_and_plans(context, config)
        return {}

    def approx_forward(self, params: Dict[str, Any], context: Dict[str, Any], config: Any):
        start_l, end_l = params.get("layers", (0, 40))

        vit_backbone = self.vit_backbone

        all_x_backbones = context.get("slinhead_x_backbones")
        all_rope_sincos = context.get("slinhead_rope_sincos")
        if all_x_backbones is None or all_rope_sincos is None:
            return {}

        if "slinhead_current_features" not in context:
            context["slinhead_current_features"] = [x.clone() for x in all_x_backbones]

        current_features = context["slinhead_current_features"]

        all_cache_features = context.get("slinhead_cache_features")
        if all_cache_features is None:
            all_cache_features = [dict() for _ in range(len(all_x_backbones))]

        appcorr_options = normalize_appcorr_kwargs(config.appcorr_kwargs, config.transmission_kwargs)
        appcorr_method = appcorr_options["method"]

        self._ensure_group_maps_and_plans(context, config)
        all_group_plans = context.get("slinhead_group_plans")

        with torch.autocast("cuda", self.autocast_dtype):
            for src_idx in range(len(all_x_backbones)):
                x_tokens = current_features[src_idx] if start_l > 0 else all_x_backbones[src_idx].clone()
                rope = all_rope_sincos[src_idx]
                cache = all_cache_features[src_idx]

                group_plans = all_group_plans[src_idx] if appcorr_method == "partial_channel" and all_group_plans is not None else None
                attn_cache_candidates = (
                    {gid: plan.full_dindice for gid, plan in group_plans.items()}
                    if group_plans is not None else None
                )

                with torch.cuda.nvtx.range(f"slinhead_vit_src{src_idx}_L{start_l}-{end_l}"):
                    for lidx in range(start_l, end_l):
                        blk = vit_backbone.blocks[lidx]
                        x_tokens, cache = blk.approx(
                            x_tokens, rope, cache, tag=f"src{src_idx}_layer{lidx}",
                            appcorr_method=appcorr_method,
                            attn_cache_candidates=attn_cache_candidates,
                            group_plans=group_plans,
                            server_pscore=appcorr_options["server_pscore"],
                            attn_col_alive_ratio=appcorr_options["attn_col_alive_ratio"],
                            debug=False,
                        )

                current_features[src_idx] = x_tokens
                all_cache_features[src_idx] = cache

        context["slinhead_current_features"] = current_features
        context["slinhead_cache_features"] = all_cache_features
        context["cache_feature"] = self._aggregate_cache_features(all_cache_features)
        return {}

    def correct_forward(self, params: Dict[str, Any], context: Dict[str, Any], config: Any):
        layers = params.get("layers", (0, 40))
        group_id = params.get("group_id", 0)
        start_l, end_l = layers[0], layers[1]

        vit_backbone = self.vit_backbone

        all_x_backbones = context.get("slinhead_x_backbones")
        all_rope_sincos = context.get("slinhead_rope_sincos")
        if all_x_backbones is None or all_rope_sincos is None:
            return {}

        if "slinhead_current_features" not in context:
            context["slinhead_current_features"] = [x.clone() for x in all_x_backbones]

        current_features = context["slinhead_current_features"]

        all_cache_features = context.get("slinhead_cache_features")
        if all_cache_features is None:
            all_cache_features = [dict() for _ in range(len(all_x_backbones))]

        all_cached_dindices = context.get("slinhead_cached_dindices")
        all_group_plans = context.get("slinhead_group_plans")

        self._ensure_group_maps_and_plans(context, config)
        all_cached_dindices = context.get("slinhead_cached_dindices")
        all_group_plans = context.get("slinhead_group_plans")

        appcorr_options = normalize_appcorr_kwargs(config.appcorr_kwargs, config.transmission_kwargs)
        appcorr_method = appcorr_options["method"]
        token_keep_ratio = appcorr_options["token_keep_ratio"]
        token_keep_thres = appcorr_options["token_keep_thres"]
        sdpa_query_bucket_size = appcorr_options["sdpa_query_bucket_size"]

        new_current_features = []
        new_cache_features = []

        for src_idx in range(len(all_x_backbones)):
            x_feature = current_features[src_idx]
            input_tokens = all_x_backbones[src_idx]
            rope = all_rope_sincos[src_idx]
            cache = all_cache_features[src_idx]

            cached_dindices = all_cached_dindices[src_idx] if all_cached_dindices is not None and src_idx < len(all_cached_dindices) else {}
            src_group_plans = all_group_plans[src_idx] if all_group_plans is not None and src_idx < len(all_group_plans) else {}

            # Resolve the appcorr group to correct: prefer the requested group_id,
            # fall back to all available groups if not found (transmission group_id
            # != appcorr group_id when num_groups=1).
            if group_id in cached_dindices:
                target_gids = [group_id]
            else:
                target_gids = sorted(cached_dindices.keys())

            if not target_gids:
                new_current_features.append(x_feature)
                new_cache_features.append(cache)
                continue

            # Correct each target group, concatenating dindices for batched correction
            all_dindices_for_src = []
            for gid in target_gids:
                d = cached_dindices.get(gid)
                if d is not None:
                    all_dindices_for_src.append(d)

            if not all_dindices_for_src:
                new_current_features.append(x_feature)
                new_cache_features.append(cache)
                continue

            # Use the first (or only) group's dindice; for single-group this is all tokens
            dindice = all_dindices_for_src[0] if len(all_dindices_for_src) == 1 else torch.cat(all_dindices_for_src, dim=1)
            plan = src_group_plans.get(target_gids[0]) if appcorr_method == "partial_channel" else None

            if dindice is None:
                new_current_features.append(x_feature)
                new_cache_features.append(cache)
                continue

            if appcorr_method == "partial_channel":
                if plan is None:
                    new_current_features.append(x_feature)
                    new_cache_features.append(cache)
                    continue
                dindice = plan.pruned_dindice.to(device=self.device, non_blocking=True)
                plan.pruned_dindice = dindice
                fixed_query_state = plan.query_state
                attn_col_alive_ratio = appcorr_options["attn_col_alive_ratio"]
                cache["_token_prune_kept_patch_total"] = (
                    cache.get("_token_prune_kept_patch_total", input_tokens.new_zeros((), dtype=torch.float32))
                    + plan.kept_patch_count.sum(dtype=torch.float32)
                )
                cache["_token_prune_full_patch_total"] = (
                    cache.get("_token_prune_full_patch_total", input_tokens.new_zeros((), dtype=torch.float32))
                    + plan.full_patch_count.sum(dtype=torch.float32)
                )
                cache["_token_prune_kept_residual_mass_total"] = (
                    cache.get("_token_prune_kept_residual_mass_total", input_tokens.new_zeros((), dtype=torch.float32))
                    + plan.kept_residual_mass.sum(dtype=torch.float32)
                )
                cache["_token_prune_full_residual_mass_total"] = (
                    cache.get("_token_prune_full_residual_mass_total", input_tokens.new_zeros((), dtype=torch.float32))
                    + plan.full_residual_mass.sum(dtype=torch.float32)
                )
            else:
                fixed_query_state = None
                attn_col_alive_ratio = 1.0

            mobile_pscore_hint = None
            all_mobile_pscore_hints = context.get("slinhead_mobile_pscore_hints")
            if isinstance(all_mobile_pscore_hints, list) and src_idx < len(all_mobile_pscore_hints):
                mobile_pscore_hint = all_mobile_pscore_hints[src_idx]

            x_temp = input_tokens

            with torch.autocast("cuda", self.autocast_dtype):
                for lidx in range(start_l, end_l):
                    blk = vit_backbone.blocks[lidx]
                    if appcorr_method == "partial_channel":
                        x_temp, cache = blk.correct(
                            x_temp, dindice, rope, cache, tag=f"src{src_idx}_layer{lidx}",
                            appcorr_method=appcorr_method,
                            token_keep_ratio=token_keep_ratio,
                            token_keep_thres=token_keep_thres,
                            mobile_pscore=appcorr_options["mobile_pscore"],
                            mobile_pscore_weight=appcorr_options["mobile_pscore_weight"],
                            mobile_pscore_hint=mobile_pscore_hint,
                            server_pscore=appcorr_options["server_pscore"],
                            server_pscore_weight=appcorr_options["server_pscore_weight"],
                            pscore_fusion=appcorr_options["pscore_fusion"],
                            sdpa_query_bucket_size=sdpa_query_bucket_size,
                            attn_col_alive_ratio=attn_col_alive_ratio,
                            fixed_query_state=fixed_query_state,
                            group_plan=plan,
                            attn_cache_key=group_id,
                            debug=False,
                        )
                    else:
                        x_temp, cache = blk.correct(
                            x_temp, dindice, rope, cache, tag=f"src{src_idx}_layer{lidx}",
                            appcorr_method=appcorr_method,
                            token_keep_ratio=token_keep_ratio,
                            token_keep_thres=token_keep_thres,
                            mobile_pscore=appcorr_options["mobile_pscore"],
                            mobile_pscore_weight=appcorr_options["mobile_pscore_weight"],
                            mobile_pscore_hint=mobile_pscore_hint,
                            server_pscore=appcorr_options["server_pscore"],
                            server_pscore_weight=appcorr_options["server_pscore_weight"],
                            pscore_fusion=appcorr_options["pscore_fusion"],
                            sdpa_query_bucket_size=sdpa_query_bucket_size,
                            attn_col_alive_ratio=attn_col_alive_ratio,
                            debug=False,
                        )

            new_current_features.append(x_temp)
            new_cache_features.append(cache)

        context["slinhead_current_features"] = new_current_features
        context["slinhead_cache_features"] = new_cache_features
        context["cache_feature"] = self._aggregate_cache_features(new_cache_features)
        return {}

    @torch.inference_mode()
    def head_inference(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[str, Any]:
        vit_backbone = self.vit_backbone
        linear_head = self.linear_head

        current_features = context.get("slinhead_current_features")
        token_shapes = context.get("slinhead_token_shapes")
        crop_hws = context.get("slinhead_crop_hw")
        image_metas = context.get("slinhead_image_metas")

        if current_features is None or token_shapes is None or image_metas is None:
            raise RuntimeError("Missing context for segmentor-linhead head_inference().")

        total_sources = len(current_features)

        # Phase 1: Batch sources by (tok_H, tok_W) shape, run norm + LinearHead
        shape_groups = {}
        for si in range(total_sources):
            shape = token_shapes[si]
            shape_groups.setdefault(shape, []).append(si)

        source_preds = [None] * total_sources

        for shape_key, src_indices in shape_groups.items():
            tok_H, tok_W = shape_key
            with torch.cuda.nvtx.range(f"slinhead_head_batch{len(src_indices)}src"):
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

        context["slinhead_outputs"] = outputs
        return {}

    def decide_exit(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[str, Any]:
        return {"num_exits": 0}

    # ── Cache aggregation ──────────────────────────────────────────────

    @staticmethod
    def _aggregate_cache_features(all_cache_features: Optional[List[Dict[str, Any]]]) -> Dict[str, Any]:
        if not all_cache_features:
            return {}
        total_keys = {
            "_attn_prob_mass_used_total",
            "_attn_prob_mass_full_total",
            "_token_prune_kept_patch_total",
            "_token_prune_full_patch_total",
            "_token_prune_kept_residual_mass_total",
            "_token_prune_full_residual_mass_total",
            "_token_pscore_kept_mass_total",
            "_token_pscore_full_mass_total",
            "_partial_token_kept_patch_total",
            "_partial_token_full_patch_total",
            "_partial_token_sample_total",
        }
        merged: Dict[str, Any] = {}
        total_values: Dict[str, Any] = {}
        for src_cache in all_cache_features:
            for key, value in src_cache.items():
                if key in total_keys:
                    total_values[key] = total_values.get(key, 0.0) + value
                else:
                    merged[key] = value
        merged.update(total_values)
        return merged

    # ── Group map / dindice helpers ────────────────────────────────────

    def _ensure_group_maps_and_plans(self, context: Dict[str, Any], config: Any) -> None:
        all_input_tokens = context.get("slinhead_x_backbones")
        if all_input_tokens is None:
            return

        all_group_maps = context.get("slinhead_group_maps")
        valid_group_maps = (
            isinstance(all_group_maps, list)
            and len(all_group_maps) == len(all_input_tokens)
            and all(
                torch.is_tensor(gm)
                and gm.shape[0] == it.shape[0]
                and gm.shape[1] == (it.shape[1] - (1 + self.vit_backbone.n_storage_tokens))
                for gm, it in zip(all_group_maps, all_input_tokens)
            )
        )
        rebuilt_group_maps = False
        if not valid_group_maps:
            rebuilt_group_maps = self._build_all_group_maps(context, config)
            if rebuilt_group_maps is not None:
                context["slinhead_group_maps"] = rebuilt_group_maps
                rebuilt_group_maps = True

        all_group_plans = context.get("slinhead_group_plans")
        valid_group_plans = (
            not rebuilt_group_maps
            and isinstance(all_group_plans, list)
            and len(all_group_plans) == len(all_input_tokens)
            and all(isinstance(sp, dict) for sp in all_group_plans)
        )
        if not valid_group_plans:
            self.prepare_group_maps_and_dindices(None, context, config)

    def _build_all_group_maps(self, context: Dict[str, Any], config: Any) -> List[torch.Tensor] | None:
        all_input_tokens = context.get("slinhead_x_backbones")
        token_shapes = context.get("slinhead_token_shapes")
        if all_input_tokens is None or token_shapes is None:
            return None

        appcorr_options = normalize_appcorr_kwargs(config.appcorr_kwargs, config.transmission_kwargs)
        if not appcorr_options.get("generated_from_client", False):
            return None

        num_groups = appcorr_options.get("num_groups", 1)
        if num_groups < 1:
            return None

        all_group_maps = []
        for input_tokens, (tok_H, tok_W) in zip(all_input_tokens, token_shapes):
            N = tok_H * tok_W
            if num_groups == 1:
                group_map = torch.zeros(input_tokens.shape[0], N, dtype=torch.long, device=self.device)
            else:
                group_map = create_group_index(N, num_groups, "grid", self.device)
                group_map = group_map.unsqueeze(0).expand(input_tokens.shape[0], -1)
            all_group_maps.append(group_map)
        return all_group_maps

    def prepare_group_maps_and_dindices(self, task: Task | None, context: Dict[str, Any], config: Any):
        all_input_tokens = context.get("slinhead_x_backbones")
        if all_input_tokens is None:
            return

        all_group_maps = context.get("slinhead_group_maps")
        if all_group_maps is None:
            return
        if not isinstance(all_group_maps, list) or len(all_group_maps) != len(all_input_tokens):
            return

        all_cached_dindices = []
        all_group_plans = []
        num_pretokens = 1 + self.vit_backbone.n_storage_tokens
        appcorr_options = normalize_appcorr_kwargs(config.appcorr_kwargs, config.transmission_kwargs)
        token_prune_enabled = appcorr_options["token_prune_enabled"]
        token_prune_threshold = appcorr_options["token_prune_threshold"]
        token_prune_min_keep = appcorr_options["token_prune_min_keep"]

        for src_idx, (input_tokens, group_map) in enumerate(zip(all_input_tokens, all_group_maps)):
            if not torch.is_tensor(group_map):
                return
            if group_map.ndim != 2 or group_map.shape[0] != input_tokens.shape[0]:
                return
            expected_tokens = input_tokens.shape[1] - num_pretokens
            if group_map.shape[1] != expected_tokens:
                return

            src_cached_dindices = {}
            src_group_plans = {}
            group_ids = torch.unique(group_map)
            group_ids = group_ids[group_ids >= 0]

            for gid_tensor in group_ids:
                gid = int(gid_tensor.item())
                nonzero_indices = torch.nonzero(group_map == gid, as_tuple=False)
                if nonzero_indices.numel() == 0:
                    continue

                B = input_tokens.shape[0]
                try:
                    spatial_indices = nonzero_indices[:, 1].view(B, -1)
                except RuntimeError:
                    return

                patch_indices = spatial_indices + num_pretokens
                pre_indices = torch.arange(
                    num_pretokens, device=input_tokens.device, dtype=torch.long,
                ).unsqueeze(0).expand(B, -1)
                dindice = torch.cat([pre_indices, patch_indices], dim=1)
                src_cached_dindices[gid] = dindice
                src_group_plans[gid] = self._build_group_plan(
                    dindice, spatial_indices, None,
                    num_pretokens, token_prune_enabled,
                    token_prune_threshold, token_prune_min_keep,
                )

            all_cached_dindices.append(src_cached_dindices)
            all_group_plans.append(src_group_plans)

        context["slinhead_cached_dindices"] = all_cached_dindices
        context["slinhead_group_plans"] = all_group_plans

    def _build_group_plan(
        self,
        dindice: torch.Tensor,
        spatial_indices: torch.Tensor,
        patch_residual_rms: torch.Tensor | None,
        num_pretokens: int,
        token_prune_enabled: bool,
        token_prune_threshold: float,
        token_prune_min_keep: int,
    ) -> GroupCorrectionPlan:
        B = spatial_indices.shape[0]
        kept_patch_count = torch.full((B,), spatial_indices.shape[1], device=self.device, dtype=torch.int32)
        full_patch_count = kept_patch_count.clone()
        kept_residual_mass = torch.zeros((B,), device=self.device, dtype=torch.float32)
        full_residual_mass = torch.zeros((B,), device=self.device, dtype=torch.float32)
        group_patch_keep_local_idx = torch.arange(
            spatial_indices.shape[1], device=self.device, dtype=torch.long,
        ).unsqueeze(0).expand(B, -1)
        pruned_dindice = dindice

        if token_prune_enabled and patch_residual_rms is not None:
            (
                pruned_dindice,
                kept_patch_count,
                full_patch_count,
                kept_residual_mass,
                full_residual_mass,
                group_patch_keep_local_idx,
            ) = self._apply_image_residual_token_pruning(
                dindice, spatial_indices, patch_residual_rms,
                token_prune_threshold, token_prune_min_keep,
            )

        return GroupCorrectionPlan(
            num_pretokens=num_pretokens,
            prefix_dindice=dindice[:, :num_pretokens],
            group_patch_dindice=dindice[:, num_pretokens:],
            group_patch_keep_local_idx=group_patch_keep_local_idx,
            full_dindice=dindice,
            pruned_dindice=pruned_dindice,
            query_state=self._build_fixed_query_state(pruned_dindice, kept_patch_count, num_pretokens),
            kept_patch_count=kept_patch_count,
            full_patch_count=full_patch_count,
            kept_residual_mass=kept_residual_mass,
            full_residual_mass=full_residual_mass,
        )

    @staticmethod
    def _build_fixed_query_state(
        dindice: torch.Tensor,
        kept_patch_count: torch.Tensor,
        num_pretokens: int,
    ) -> QueryState:
        B, q = dindice.shape
        device = dindice.device
        query_pos_idx = torch.arange(q, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)
        valid_lengths = kept_patch_count.to(device=device, dtype=torch.long) + num_pretokens
        query_valid_mask = query_pos_idx < valid_lengths.unsqueeze(1)
        active_batch_idx, active_pos_idx = query_valid_mask.nonzero(as_tuple=True)
        active_token_idx = dindice[active_batch_idx, active_pos_idx]
        max_active = int(valid_lengths.max().item()) if valid_lengths.numel() > 0 else 0
        if max_active > 0:
            active_query_pos_padded = torch.arange(max_active, device=device, dtype=torch.long).unsqueeze(0).expand(B, -1)
            active_query_mask = active_query_pos_padded < valid_lengths.unsqueeze(1)
        else:
            active_query_pos_padded = torch.empty((B, 0), device=device, dtype=torch.long)
            active_query_mask = torch.empty((B, 0), device=device, dtype=torch.bool)
        return QueryState(
            query_pos_idx=query_pos_idx,
            query_valid_mask=query_valid_mask,
            active_batch_idx=active_batch_idx,
            active_pos_idx=active_pos_idx,
            active_token_idx=active_token_idx,
            active_query_pos=active_pos_idx,
            active_query_pos_padded=active_query_pos_padded,
            active_query_mask=active_query_mask,
            all_valid=bool(torch.all(valid_lengths == q).item()) if valid_lengths.numel() > 0 else True,
        )

    def _apply_image_residual_token_pruning(
        self,
        dindice: torch.Tensor,
        spatial_indices: torch.Tensor,
        patch_residual_rms: torch.Tensor,
        threshold: float,
        min_keep: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B = spatial_indices.shape[0]
        kept_patch_count = torch.zeros(B, device=self.device, dtype=torch.int32)
        full_patch_count = torch.full(B, spatial_indices.shape[1], device=self.device, dtype=torch.int32)
        kept_residual_mass = torch.zeros(B, device=self.device, dtype=torch.float32)
        full_residual_mass = torch.zeros(B, device=self.device, dtype=torch.float32)

        pruned_rows = []
        keep_local_rows = []
        for b in range(B):
            rms = patch_residual_rms[b, spatial_indices[b]]
            full_mass = rms.sum()
            sorted_idx = torch.argsort(rms, descending=True)
            cumulative = torch.cumsum(rms[sorted_idx], dim=0)
            keep_count = max(int((cumulative < full_mass * (1.0 - threshold)).sum().item()) + 1, min_keep)
            keep_count = min(keep_count, len(sorted_idx))
            kept_local = sorted_idx[:keep_count].sort()[0]
            pruned_rows.append(dindice[b, torch.cat([
                torch.arange(dindice.shape[1] - spatial_indices.shape[1], dindice.shape[1], device=self.device),
                spatial_indices[b, kept_local] + (dindice.shape[1] - spatial_indices.shape[1]),
            ])])
            keep_local_rows.append(kept_local)
            kept_patch_count[b] = keep_count
            kept_residual_mass[b] = rms[kept_local].sum()
            full_residual_mass[b] = full_mass

        num_pretokens = dindice.shape[1] - spatial_indices.shape[1]
        max_cols = max(r.shape[0] for r in pruned_rows)
        pruned_dindice = dindice.new_zeros(B, max_cols)
        group_patch_keep_local_idx = torch.arange(
            spatial_indices.shape[1], device=self.device, dtype=torch.long,
        ).unsqueeze(0).expand(B, -1).clone()
        for b in range(B):
            pruned_dindice[b, :pruned_rows[b].shape[0]] = pruned_rows[b]
            group_patch_keep_local_idx[b, :keep_local_rows[b].shape[0]] = keep_local_rows[b]
        return pruned_dindice, kept_patch_count, full_patch_count, kept_residual_mass, full_residual_mass, group_patch_keep_local_idx
