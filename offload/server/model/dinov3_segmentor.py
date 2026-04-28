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
            aggregated_preds = torch.zeros(1, self.num_classes, *rescale_to, dtype=torch.float32, device=self.device)
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

    @torch.inference_mode()
    def _full_inference_analyzed(self, task: Task, context: Dict[str, Any], config: Any):
        """Decomposed version of full_inference. All steps are inlined so that
        the function can later be split into prepare_tokens, approx_forward,
        and head_inference."""
        from dinov3.eval.segmentation.models.backbone.dinov3_adapter import deform_inputs

        images = context.get("input_images_uint8")
        if images is None:
            raise RuntimeError("Missing context['input_images_uint8'] for segmentor _full_inference_analyzed().")
        target_shapes = context.get("target_shapes") or [None] * len(images)

        profile_config = config.get_input_profile_config()
        inference_mode = str(profile_config.get("server_inference_mode", "slide"))
        crop_size = int(profile_config.get("server_crop_size", 896))
        stride = int(profile_config.get("server_stride", 596))
        decoder_head_type = str(profile_config.get("decoder_head_type", "m2f"))
        rescale_mode = str(profile_config.get("server_rescale_to", "input")).lower()
        if rescale_mode not in {"input", "original"}:
            raise ValueError(f"Unsupported ADE20K segmentor server_rescale_to: {rescale_mode}")

        # Model components
        adapter = self.model.segmentation_model[0]   # DINOv3_Adapter
        m2f_head = self.model.segmentation_model[1]  # Mask2FormerHead
        vit_backbone = adapter.backbone               # DinoVisionTransformer
        interaction_indexes = adapter.interaction_indexes

        outputs = []

        for image_idx, image_np in enumerate(images):
            # ── TTA input preparation ──
            tta_tensors, flip_flags, rescale_to = self._build_tta_inputs(image_np, profile_config)
            if rescale_mode == "original" and image_idx < len(target_shapes) and target_shapes[image_idx] is not None:
                rescale_to = target_shapes[image_idx]
            aggregated_preds = torch.zeros(1, self.num_classes, *rescale_to, dtype=torch.float32, device=self.device)

            for img_tensor, apply_flip in zip(tta_tensors, flip_flags):
                if inference_mode == "slide":
                    # ══════════════════════════════════════════════════════════
                    # Sliding-window inference (inlined from slide_inference)
                    # ══════════════════════════════════════════════════════════
                    h_stride = w_stride = stride
                    h_crop = w_crop = crop_size
                    batch_size, C, h_img, w_img = img_tensor.shape
                    if h_crop > h_img and w_crop > w_img:
                        h_crop, w_crop = min(h_img, w_img), min(h_img, w_img)
                    assert batch_size == 1
                    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
                    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

                    slide_preds = img_tensor.new_zeros((1, self.num_classes, h_img, w_img))
                    slide_count = img_tensor.new_zeros((1, 1, h_img, w_img)).to(torch.int8)

                    for h_idx in range(h_grids):
                        for w_idx in range(w_grids):
                            y1 = h_idx * h_stride
                            x1 = w_idx * w_stride
                            y2 = min(y1 + h_crop, h_img)
                            x2 = min(x1 + w_crop, w_img)
                            y1 = max(y2 - h_crop, 0)
                            x1 = max(x2 - w_crop, 0)
                            crop_img = img_tensor[:, :, y1:y2, x1:x2]

                            # ── DINOv3_Adapter.forward(crop_img) inlined ──
                            with torch.autocast("cuda", torch.bfloat16):
                                with torch.no_grad():
                                    # A) deform_inputs for InteractionBlocks
                                    crop_deform_in1, crop_deform_in2 = deform_inputs(crop_img, adapter.patch_size)

                                    # B) SPM forward
                                    spm_c1, spm_c2, spm_c3, spm_c4 = adapter.spm(crop_img)
                                    spm_c2, spm_c3, spm_c4 = adapter._add_level_embed(spm_c2, spm_c3, spm_c4)
                                    spm_c2_len = spm_c2.shape[1]
                                    spm_c3_len = spm_c3.shape[1]
                                    spm_c_cat = torch.cat([spm_c2, spm_c3, spm_c4], dim=1)

                                    crop_H_c = crop_img.shape[2] // 16
                                    crop_W_c = crop_img.shape[3] // 16
                                    crop_H_toks = crop_img.shape[2] // adapter.patch_size
                                    crop_W_toks = crop_img.shape[3] // adapter.patch_size
                                    crop_bs = crop_img.shape[0]

                                    # C) ViT Backbone forward
                                    # C.1) prepare_tokens_with_masks
                                    x_tokens, (tok_H, tok_W) = vit_backbone.prepare_tokens_with_masks(crop_img)

                                    # C.2) Run all ViT blocks, collecting intermediate outputs
                                    intermediate_raw = []
                                    for blk_idx, blk in enumerate(vit_backbone.blocks):
                                        rope_sincos = vit_backbone.rope_embed(H=tok_H, W=tok_W) if vit_backbone.rope_embed else None
                                        x_tokens = blk(x_tokens, rope_sincos)
                                        if blk_idx in interaction_indexes:
                                            intermediate_raw.append(x_tokens)

                                    # C.3) Apply norms and split (from get_intermediate_layers)
                                    intermediate_normed = []
                                    for raw_out in intermediate_raw:
                                        if vit_backbone.untie_cls_and_patch_norms:
                                            normed_cls_reg = vit_backbone.cls_norm(raw_out[:, :vit_backbone.n_storage_tokens + 1])
                                            normed_patch = vit_backbone.norm(raw_out[:, vit_backbone.n_storage_tokens + 1:])
                                            intermediate_normed.append(torch.cat((normed_cls_reg, normed_patch), dim=1))
                                        else:
                                            intermediate_normed.append(vit_backbone.norm(raw_out))

                                    backbone_cls_tokens = [out[:, 0] for out in intermediate_normed]
                                    backbone_patch_tokens = [out[:, vit_backbone.n_storage_tokens + 1:] for out in intermediate_normed]
                                    all_backbone_layers = list(zip(backbone_patch_tokens, backbone_cls_tokens))

                                    # D) InteractionBlocks (combine SPM + backbone features)
                                    x_for_shape, _ = all_backbone_layers[0]
                                    _, _, feat_dim = x_for_shape.shape
                                    del x_for_shape

                                    interaction_outs = []
                                    for i, interaction_layer in enumerate(adapter.interactions):
                                        layer_x, layer_cls = all_backbone_layers[i]
                                        _, spm_c_cat, _ = interaction_layer(
                                            layer_x, spm_c_cat, layer_cls,
                                            crop_deform_in1, crop_deform_in2,
                                            crop_H_c, crop_W_c, crop_H_toks, crop_W_toks,
                                        )
                                        interaction_outs.append(
                                            layer_x.transpose(1, 2).view(crop_bs, feat_dim, crop_H_toks, crop_W_toks).contiguous()
                                        )

                                    # E) Split & Reshape SPM features
                                    final_c2 = spm_c_cat[:, 0:spm_c2_len, :].transpose(1, 2).view(crop_bs, feat_dim, crop_H_c * 2, crop_W_c * 2).contiguous()
                                    final_c3 = spm_c_cat[:, spm_c2_len:spm_c2_len + spm_c3_len, :].transpose(1, 2).view(crop_bs, feat_dim, crop_H_c, crop_W_c).contiguous()
                                    final_c4 = spm_c_cat[:, spm_c2_len + spm_c3_len:, :].transpose(1, 2).view(crop_bs, feat_dim, crop_H_c // 2, crop_W_c // 2).contiguous()
                                    final_c1 = adapter.up(final_c2) + spm_c1

                                    # F) Add ViT features (from InteractionBlocks)
                                    if adapter.add_vit_feature:
                                        vit_x1, vit_x2, vit_x3, vit_x4 = interaction_outs
                                        vit_x1 = F.interpolate(vit_x1, size=(4 * crop_H_c, 4 * crop_W_c), mode="bilinear", align_corners=False)
                                        vit_x2 = F.interpolate(vit_x2, size=(2 * crop_H_c, 2 * crop_W_c), mode="bilinear", align_corners=False)
                                        vit_x3 = F.interpolate(vit_x3, size=(1 * crop_H_c, 1 * crop_W_c), mode="bilinear", align_corners=False)
                                        vit_x4 = F.interpolate(vit_x4, size=(crop_H_c // 2, crop_W_c // 2), mode="bilinear", align_corners=False)
                                        final_c1 = final_c1 + vit_x1
                                        final_c2 = final_c2 + vit_x2
                                        final_c3 = final_c3 + vit_x3
                                        final_c4 = final_c4 + vit_x4

                                    # G) Final Norm
                                    f1 = adapter.norm1(final_c1)
                                    f2 = adapter.norm2(final_c2)
                                    f3 = adapter.norm3(final_c3)
                                    f4 = adapter.norm4(final_c4)
                                    adapter_features = {"1": f1, "2": f2, "3": f3, "4": f4}

                                    # ── Mask2FormerHead.predict ──
                                    crop_pred_dict = m2f_head.predict(adapter_features, rescale_to=crop_img.shape[2:])

                            # M2F post-processing (from slide_inference)
                            if decoder_head_type == "m2f":
                                mask_pred = crop_pred_dict["pred_masks"]
                                mask_cls = crop_pred_dict["pred_logits"]
                                mask_cls_softmax = F.softmax(mask_cls, dim=-1)[..., :-1]
                                mask_pred_sigmoid = mask_pred.sigmoid()
                                crop_pred = torch.einsum(
                                    "bqc,bqhw->bchw",
                                    mask_cls_softmax.to(torch.bfloat16),
                                    mask_pred_sigmoid.to(torch.bfloat16),
                                )
                                del mask_cls, mask_pred, mask_cls_softmax, mask_pred_sigmoid

                            # Slide aggregation
                            slide_preds += F.pad(
                                crop_pred,
                                (int(x1), int(slide_preds.shape[-1] - x2), int(y1), int(slide_preds.shape[-2] - y2)),
                            )
                            slide_count[:, :, y1:y2, x1:x2] += 1
                            del crop_img, crop_pred

                    assert (slide_count == 0).sum() == 0
                    pred = slide_preds / slide_count
                    # Rescale to target size (from make_inference for slide mode)
                    pred = F.interpolate(pred, size=rescale_to, mode="bilinear", align_corners=False)

                else:
                    # ══════════════════════════════════════════════════════════
                    # Whole-image inference (inlined from make_inference)
                    # ══════════════════════════════════════════════════════════
                    resized_input = F.interpolate(img_tensor, size=(512, 512), mode="bilinear", align_corners=False)

                    with torch.autocast("cuda", torch.bfloat16):
                        with torch.no_grad():
                            whole_deform_in1, whole_deform_in2 = deform_inputs(resized_input, adapter.patch_size)
                            spm_c1, spm_c2, spm_c3, spm_c4 = adapter.spm(resized_input)
                            spm_c2, spm_c3, spm_c4 = adapter._add_level_embed(spm_c2, spm_c3, spm_c4)
                            spm_c2_len = spm_c2.shape[1]
                            spm_c3_len = spm_c3.shape[1]
                            spm_c_cat = torch.cat([spm_c2, spm_c3, spm_c4], dim=1)

                            whole_H_c = resized_input.shape[2] // 16
                            whole_W_c = resized_input.shape[3] // 16
                            whole_H_toks = resized_input.shape[2] // adapter.patch_size
                            whole_W_toks = resized_input.shape[3] // adapter.patch_size
                            whole_bs = resized_input.shape[0]

                            # ViT Backbone forward
                            x_tokens, (tok_H, tok_W) = vit_backbone.prepare_tokens_with_masks(resized_input)
                            intermediate_raw = []
                            for blk_idx, blk in enumerate(vit_backbone.blocks):
                                rope_sincos = vit_backbone.rope_embed(H=tok_H, W=tok_W) if vit_backbone.rope_embed else None
                                x_tokens = blk(x_tokens, rope_sincos)
                                if blk_idx in interaction_indexes:
                                    intermediate_raw.append(x_tokens)

                            intermediate_normed = []
                            for raw_out in intermediate_raw:
                                if vit_backbone.untie_cls_and_patch_norms:
                                    normed_cls_reg = vit_backbone.cls_norm(raw_out[:, :vit_backbone.n_storage_tokens + 1])
                                    normed_patch = vit_backbone.norm(raw_out[:, vit_backbone.n_storage_tokens + 1:])
                                    intermediate_normed.append(torch.cat((normed_cls_reg, normed_patch), dim=1))
                                else:
                                    intermediate_normed.append(vit_backbone.norm(raw_out))

                            backbone_cls_tokens = [out[:, 0] for out in intermediate_normed]
                            backbone_patch_tokens = [out[:, vit_backbone.n_storage_tokens + 1:] for out in intermediate_normed]
                            all_backbone_layers = list(zip(backbone_patch_tokens, backbone_cls_tokens))

                            x_for_shape, _ = all_backbone_layers[0]
                            _, _, feat_dim = x_for_shape.shape
                            del x_for_shape

                            interaction_outs = []
                            for i, interaction_layer in enumerate(adapter.interactions):
                                layer_x, layer_cls = all_backbone_layers[i]
                                _, spm_c_cat, _ = interaction_layer(
                                    layer_x, spm_c_cat, layer_cls,
                                    whole_deform_in1, whole_deform_in2,
                                    whole_H_c, whole_W_c, whole_H_toks, whole_W_toks,
                                )
                                interaction_outs.append(
                                    layer_x.transpose(1, 2).view(whole_bs, feat_dim, whole_H_toks, whole_W_toks).contiguous()
                                )

                            final_c2 = spm_c_cat[:, 0:spm_c2_len, :].transpose(1, 2).view(whole_bs, feat_dim, whole_H_c * 2, whole_W_c * 2).contiguous()
                            final_c3 = spm_c_cat[:, spm_c2_len:spm_c2_len + spm_c3_len, :].transpose(1, 2).view(whole_bs, feat_dim, whole_H_c, whole_W_c).contiguous()
                            final_c4 = spm_c_cat[:, spm_c2_len + spm_c3_len:, :].transpose(1, 2).view(whole_bs, feat_dim, whole_H_c // 2, whole_W_c // 2).contiguous()
                            final_c1 = adapter.up(final_c2) + spm_c1

                            if adapter.add_vit_feature:
                                vit_x1, vit_x2, vit_x3, vit_x4 = interaction_outs
                                vit_x1 = F.interpolate(vit_x1, size=(4 * whole_H_c, 4 * whole_W_c), mode="bilinear", align_corners=False)
                                vit_x2 = F.interpolate(vit_x2, size=(2 * whole_H_c, 2 * whole_W_c), mode="bilinear", align_corners=False)
                                vit_x3 = F.interpolate(vit_x3, size=(1 * whole_H_c, 1 * whole_W_c), mode="bilinear", align_corners=False)
                                vit_x4 = F.interpolate(vit_x4, size=(whole_H_c // 2, whole_W_c // 2), mode="bilinear", align_corners=False)
                                final_c1 = final_c1 + vit_x1
                                final_c2 = final_c2 + vit_x2
                                final_c3 = final_c3 + vit_x3
                                final_c4 = final_c4 + vit_x4

                            f1 = adapter.norm1(final_c1)
                            f2 = adapter.norm2(final_c2)
                            f3 = adapter.norm3(final_c3)
                            f4 = adapter.norm4(final_c4)
                            adapter_features = {"1": f1, "2": f2, "3": f3, "4": f4}

                            pred_dict = m2f_head.predict(adapter_features, rescale_to=rescale_to)

                    # M2F post-processing (float for whole mode, per make_inference)
                    if decoder_head_type == "m2f":
                        mask_pred = pred_dict["pred_masks"]
                        mask_cls = pred_dict["pred_logits"]
                        mask_cls_softmax = F.softmax(mask_cls, dim=-1)[..., :-1]
                        mask_pred_sigmoid = mask_pred.sigmoid()
                        pred = torch.einsum(
                            "bqc,bqhw->bchw",
                            mask_cls_softmax.to(torch.float),
                            mask_pred_sigmoid.to(torch.float),
                        )
                        del mask_cls, mask_pred, mask_cls_softmax, mask_pred_sigmoid

                # ── TTA post-processing ──
                if apply_flip:
                    pred = F.hflip(pred)
                pred = F.softmax(pred, dim=1)
                aggregated_preds += pred.float()
                del pred

            # Average TTA predictions
            pred_label = (aggregated_preds / len(tta_tensors)).argmax(dim=1)[0].to(torch.uint8)
            outputs.append(pred_label.cpu())
            del aggregated_preds

        return outputs

    def _prepare_single_source(self, input_tensor: torch.Tensor, adapter, vit_backbone) -> Dict[str, Any]:
        """Compute SPM features, deform inputs, and prepare ViT tokens for one input."""
        from dinov3.eval.segmentation.models.backbone.dinov3_adapter import deform_inputs

        with torch.autocast("cuda", self.autocast_dtype):
            d_in1, d_in2 = deform_inputs(input_tensor, adapter.patch_size)
            spm_c1, spm_c2, spm_c3, spm_c4 = adapter.spm(input_tensor)
            spm_c2, spm_c3, spm_c4 = adapter._add_level_embed(spm_c2, spm_c3, spm_c4)
            c2_len = spm_c2.shape[1]
            c3_len = spm_c3.shape[1]
            spm_c_cat = torch.cat([spm_c2, spm_c3, spm_c4], dim=1)
            x_backbone, (tok_H, tok_W) = vit_backbone.prepare_tokens_with_masks(input_tensor)
            rope = vit_backbone.rope_embed(H=tok_H, W=tok_W) if vit_backbone.rope_embed else None

        H_c = input_tensor.shape[2] // 16
        W_c = input_tensor.shape[3] // 16
        H_toks = input_tensor.shape[2] // adapter.patch_size
        W_toks = input_tensor.shape[3] // adapter.patch_size

        return {
            "x_backbone": x_backbone,
            "rope_sincos": rope,
            "deform_in1": d_in1,
            "deform_in2": d_in2,
            "spm_c1_raw": spm_c1,
            "spm_c_cat": spm_c_cat,
            "spm_c2_len": c2_len,
            "spm_c3_len": c3_len,
            "source_shape": (H_c, W_c, H_toks, W_toks, input_tensor.shape[0]),
            "crop_hw": (input_tensor.shape[2], input_tensor.shape[3]),
        }

    @torch.inference_mode()
    def prepare_tokens(self, task: Task, context: Dict[str, Any], config: Any):
        images = context.get("input_images_uint8")
        if images is None:
            raise RuntimeError("Missing context['input_images_uint8'] for segmentor prepare_tokens().")
        target_shapes = context.get("target_shapes") or [None] * len(images)

        profile_config = config.get_input_profile_config()
        inference_mode = str(profile_config.get("server_inference_mode", "slide"))
        crop_size = int(profile_config.get("server_crop_size", 896))
        stride = int(profile_config.get("server_stride", 596))
        decoder_head_type = str(profile_config.get("decoder_head_type", "m2f"))
        rescale_mode = str(profile_config.get("server_rescale_to", "input")).lower()
        if rescale_mode not in {"input", "original"}:
            raise ValueError(f"Unsupported ADE20K segmentor server_rescale_to: {rescale_mode}")

        adapter = self.model.segmentation_model[0]
        vit_backbone = adapter.backbone

        all_x_backbones = []
        all_rope_sincos = []
        all_deform_in1 = []
        all_deform_in2 = []
        all_spm_c1_raw = []
        all_spm_c_cat = []
        all_spm_c2_len = []
        all_spm_c3_len = []
        all_source_shapes = []
        all_crop_hw = []

        image_metas = []

        for image_idx, image_np in enumerate(images):
            with torch.cuda.nvtx.range(f"seg_prepare_tta_img{image_idx}"):
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

                            with torch.cuda.nvtx.range(f"seg_prepare_src{len(all_x_backbones)}"):
                                src = self._prepare_single_source(crop_img, adapter, vit_backbone)
                            all_x_backbones.append(src["x_backbone"])
                            all_rope_sincos.append(src["rope_sincos"])
                            all_deform_in1.append(src["deform_in1"])
                            all_deform_in2.append(src["deform_in2"])
                            all_spm_c1_raw.append(src["spm_c1_raw"])
                            all_spm_c_cat.append(src["spm_c_cat"])
                            all_spm_c2_len.append(src["spm_c2_len"])
                            all_spm_c3_len.append(src["spm_c3_len"])
                            all_source_shapes.append(src["source_shape"])
                            all_crop_hw.append(src["crop_hw"])
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

                    with torch.cuda.nvtx.range(f"seg_prepare_src{len(all_x_backbones)}"):
                        src = self._prepare_single_source(resized, adapter, vit_backbone)
                    all_x_backbones.append(src["x_backbone"])
                    all_rope_sincos.append(src["rope_sincos"])
                    all_deform_in1.append(src["deform_in1"])
                    all_deform_in2.append(src["deform_in2"])
                    all_spm_c1_raw.append(src["spm_c1_raw"])
                    all_spm_c_cat.append(src["spm_c_cat"])
                    all_spm_c2_len.append(src["spm_c2_len"])
                    all_spm_c3_len.append(src["spm_c3_len"])
                    all_source_shapes.append(src["source_shape"])
                    all_crop_hw.append(src["crop_hw"])

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

        context["seg_x_backbones"] = all_x_backbones
        context["seg_rope_sincos"] = all_rope_sincos
        context["seg_deform_in1"] = all_deform_in1
        context["seg_deform_in2"] = all_deform_in2
        context["seg_spm_c1_raw"] = all_spm_c1_raw
        context["seg_spm_c_cat"] = all_spm_c_cat
        context["seg_spm_c2_len"] = all_spm_c2_len
        context["seg_spm_c3_len"] = all_spm_c3_len
        context["seg_source_shapes"] = all_source_shapes
        context["seg_crop_hw"] = all_crop_hw
        context["seg_image_metas"] = image_metas
        context["seg_inference_mode"] = inference_mode
        context["seg_decoder_head_type"] = decoder_head_type
        return {}

    @torch.inference_mode()
    def approx_forward(self, params: Dict[str, Any], context: Dict[str, Any], config: Any):
        start_l, end_l = params.get("layers", (0, 40))

        adapter = self.model.segmentation_model[0]
        vit_backbone = adapter.backbone
        interaction_indexes = set(adapter.interaction_indexes)

        all_x_backbones = context.get("seg_x_backbones")
        all_rope_sincos = context.get("seg_rope_sincos")
        if all_x_backbones is None or all_rope_sincos is None:
            return {}

        if "seg_current_features" not in context:
            context["seg_current_features"] = [x.clone() for x in all_x_backbones]
        if "seg_intermediate_raw" not in context:
            context["seg_intermediate_raw"] = [[] for _ in range(len(all_x_backbones))]

        current_features = context["seg_current_features"]
        intermediate_raw = context["seg_intermediate_raw"]

        with torch.autocast("cuda", self.autocast_dtype):
            for src_idx in range(len(all_x_backbones)):
                x_tokens = current_features[src_idx] if start_l > 0 else all_x_backbones[src_idx].clone()
                rope = all_rope_sincos[src_idx]

                with torch.cuda.nvtx.range(f"seg_vit_src{src_idx}_L{start_l}-{end_l}"):
                    for lidx in range(start_l, end_l):
                        blk = vit_backbone.blocks[lidx]
                        with torch.no_grad():
                            x_tokens = blk(x_tokens, rope)
                        if lidx in interaction_indexes:
                            intermediate_raw[src_idx].append(x_tokens)

                current_features[src_idx] = x_tokens

        context["seg_current_features"] = current_features
        context["seg_intermediate_raw"] = intermediate_raw
        return {}

    @torch.inference_mode()
    def correct_forward(self, params: Dict[str, Any], context: Dict[str, Any], config: Any):
        layers = params.get("layers", (0, 40))
        start_l, end_l = layers[0], layers[1]
        source_idx = params.get("group_id", 0)

        adapter = self.model.segmentation_model[0]
        vit_backbone = adapter.backbone
        interaction_indexes = set(adapter.interaction_indexes)

        all_x_backbones = context.get("seg_x_backbones")
        all_rope_sincos = context.get("seg_rope_sincos")
        if all_x_backbones is None or all_rope_sincos is None:
            return {}
        if source_idx >= len(all_x_backbones):
            return {}

        if "seg_current_features" not in context:
            context["seg_current_features"] = [x.clone() for x in all_x_backbones]
        if "seg_intermediate_raw" not in context:
            context["seg_intermediate_raw"] = [[] for _ in range(len(all_x_backbones))]

        current_features = context["seg_current_features"]
        intermediate_raw = context["seg_intermediate_raw"]

        # Re-run from the original input tokens for the specified source
        x_tokens = all_x_backbones[source_idx].clone()
        rope = all_rope_sincos[source_idx]
        intermediate_raw[source_idx] = []

        with torch.autocast("cuda", self.autocast_dtype):
            for lidx in range(start_l, end_l):
                blk = vit_backbone.blocks[lidx]
                with torch.no_grad():
                    x_tokens = blk(x_tokens, rope)
                if lidx in interaction_indexes:
                    intermediate_raw[source_idx].append(x_tokens)

        current_features[source_idx] = x_tokens
        return {}

    def _run_adapter_postprocess(self, src_idx: int, context: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Run InteractionBlocks + feature assembly for a single source.
        Returns the multi-scale feature dict {"1": f1, ..., "4": f4}."""
        adapter = self.model.segmentation_model[0]
        vit_backbone = adapter.backbone

        intermediate_raw = context["seg_intermediate_raw"][src_idx]
        spm_c_cat = context["seg_spm_c_cat"][src_idx].clone()
        spm_c1 = context["seg_spm_c1_raw"][src_idx]
        c2_len = context["seg_spm_c2_len"][src_idx]
        c3_len = context["seg_spm_c3_len"][src_idx]
        H_c, W_c, H_toks, W_toks, bs = context["seg_source_shapes"][src_idx]
        deform_in1 = context["seg_deform_in1"][src_idx]
        deform_in2 = context["seg_deform_in2"][src_idx]

        with torch.cuda.nvtx.range(f"seg_adapter_norm_src{src_idx}"):
            intermediate_normed = []
            for raw_out in intermediate_raw:
                if vit_backbone.untie_cls_and_patch_norms:
                    normed_cls_reg = vit_backbone.cls_norm(raw_out[:, :vit_backbone.n_storage_tokens + 1])
                    normed_patch = vit_backbone.norm(raw_out[:, vit_backbone.n_storage_tokens + 1:])
                    intermediate_normed.append(torch.cat((normed_cls_reg, normed_patch), dim=1))
                else:
                    intermediate_normed.append(vit_backbone.norm(raw_out))

            backbone_cls_tokens = [out[:, 0] for out in intermediate_normed]
            backbone_patch_tokens = [out[:, vit_backbone.n_storage_tokens + 1:] for out in intermediate_normed]
            all_backbone_layers = list(zip(backbone_patch_tokens, backbone_cls_tokens))

            x_for_shape, _ = all_backbone_layers[0]
            _, _, feat_dim = x_for_shape.shape
            del x_for_shape

        with torch.cuda.nvtx.range(f"seg_adapter_interaction_src{src_idx}"):
            interaction_outs = []
            for i, interaction_layer in enumerate(adapter.interactions):
                layer_x, layer_cls = all_backbone_layers[i]
                _, spm_c_cat, _ = interaction_layer(
                    layer_x, spm_c_cat, layer_cls,
                    deform_in1, deform_in2,
                    H_c, W_c, H_toks, W_toks,
                )
                interaction_outs.append(
                    layer_x.transpose(1, 2).view(bs, feat_dim, H_toks, W_toks).contiguous()
                )

        with torch.cuda.nvtx.range(f"seg_adapter_feat_assembly_src{src_idx}"):
            final_c2 = spm_c_cat[:, 0:c2_len, :].transpose(1, 2).view(bs, feat_dim, H_c * 2, W_c * 2).contiguous()
            final_c3 = spm_c_cat[:, c2_len:c2_len + c3_len, :].transpose(1, 2).view(bs, feat_dim, H_c, W_c).contiguous()
            final_c4 = spm_c_cat[:, c2_len + c3_len:, :].transpose(1, 2).view(bs, feat_dim, H_c // 2, W_c // 2).contiguous()
            final_c1 = adapter.up(final_c2) + spm_c1

            if adapter.add_vit_feature:
                vit_x1, vit_x2, vit_x3, vit_x4 = interaction_outs
                vit_x1 = F.interpolate(vit_x1, size=(4 * H_c, 4 * W_c), mode="bilinear", align_corners=False)
                vit_x2 = F.interpolate(vit_x2, size=(2 * H_c, 2 * W_c), mode="bilinear", align_corners=False)
                vit_x3 = F.interpolate(vit_x3, size=(1 * H_c, 1 * W_c), mode="bilinear", align_corners=False)
                vit_x4 = F.interpolate(vit_x4, size=(H_c // 2, W_c // 2), mode="bilinear", align_corners=False)
                final_c1 = final_c1 + vit_x1
                final_c2 = final_c2 + vit_x2
                final_c3 = final_c3 + vit_x3
                final_c4 = final_c4 + vit_x4

            f1 = adapter.norm1(final_c1)
            f2 = adapter.norm2(final_c2)
            f3 = adapter.norm3(final_c3)
            f4 = adapter.norm4(final_c4)
        return {"1": f1, "2": f2, "3": f3, "4": f4}

    def _run_adapter_postprocess_batch(
        self, src_indices: List[int], context: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """Batched version of _run_adapter_postprocess for sources with identical spatial shapes.
        Returns the multi-scale feature dict with batch dim = len(src_indices)."""
        adapter = self.model.segmentation_model[0]
        vit_backbone = adapter.backbone

        N = len(src_indices)
        s0 = src_indices[0]
        H_c, W_c, H_toks, W_toks, _ = context["seg_source_shapes"][s0]

        # Batch intermediate_raw: stack per interaction_index
        n_interactions = len(adapter.interaction_indexes)
        batched_raw = []
        for inter_i in range(n_interactions):
            raws = [context["seg_intermediate_raw"][si][inter_i] for si in src_indices]
            batched_raw.append(torch.cat(raws, dim=0))

        # Batch spm_c_cat and spm_c1
        spm_c_cat = torch.cat([context["seg_spm_c_cat"][si].clone() for si in src_indices], dim=0)
        spm_c1 = torch.cat([context["seg_spm_c1_raw"][si] for si in src_indices], dim=0)
        c2_len = context["seg_spm_c2_len"][s0]
        c3_len = context["seg_spm_c3_len"][s0]

        # deform_inputs are identical for same-shape sources, use first
        deform_in1 = context["seg_deform_in1"][s0]
        deform_in2 = context["seg_deform_in2"][s0]

        # Norm + split backbone outputs
        intermediate_normed = []
        for raw_out in batched_raw:
            if vit_backbone.untie_cls_and_patch_norms:
                normed_cls_reg = vit_backbone.cls_norm(raw_out[:, :vit_backbone.n_storage_tokens + 1])
                normed_patch = vit_backbone.norm(raw_out[:, vit_backbone.n_storage_tokens + 1:])
                intermediate_normed.append(torch.cat((normed_cls_reg, normed_patch), dim=1))
            else:
                intermediate_normed.append(vit_backbone.norm(raw_out))

        backbone_cls_tokens = [out[:, 0] for out in intermediate_normed]
        backbone_patch_tokens = [out[:, vit_backbone.n_storage_tokens + 1:] for out in intermediate_normed]
        all_backbone_layers = list(zip(backbone_patch_tokens, backbone_cls_tokens))

        x_for_shape, _ = all_backbone_layers[0]
        _, _, feat_dim = x_for_shape.shape
        del x_for_shape

        # InteractionBlocks
        interaction_outs = []
        for i, interaction_layer in enumerate(adapter.interactions):
            layer_x, layer_cls = all_backbone_layers[i]
            _, spm_c_cat, _ = interaction_layer(
                layer_x, spm_c_cat, layer_cls,
                deform_in1, deform_in2,
                H_c, W_c, H_toks, W_toks,
            )
            interaction_outs.append(
                layer_x.transpose(1, 2).view(N, feat_dim, H_toks, W_toks).contiguous()
            )

        # Split & Reshape SPM features
        final_c2 = spm_c_cat[:, 0:c2_len, :].transpose(1, 2).view(N, feat_dim, H_c * 2, W_c * 2).contiguous()
        final_c3 = spm_c_cat[:, c2_len:c2_len + c3_len, :].transpose(1, 2).view(N, feat_dim, H_c, W_c).contiguous()
        final_c4 = spm_c_cat[:, c2_len + c3_len:, :].transpose(1, 2).view(N, feat_dim, H_c // 2, W_c // 2).contiguous()
        final_c1 = adapter.up(final_c2) + spm_c1

        if adapter.add_vit_feature:
            vit_x1, vit_x2, vit_x3, vit_x4 = interaction_outs
            vit_x1 = F.interpolate(vit_x1, size=(4 * H_c, 4 * W_c), mode="bilinear", align_corners=False)
            vit_x2 = F.interpolate(vit_x2, size=(2 * H_c, 2 * W_c), mode="bilinear", align_corners=False)
            vit_x3 = F.interpolate(vit_x3, size=(1 * H_c, 1 * W_c), mode="bilinear", align_corners=False)
            vit_x4 = F.interpolate(vit_x4, size=(H_c // 2, W_c // 2), mode="bilinear", align_corners=False)
            final_c1 = final_c1 + vit_x1
            final_c2 = final_c2 + vit_x2
            final_c3 = final_c3 + vit_x3
            final_c4 = final_c4 + vit_x4

        f1 = adapter.norm1(final_c1)
        f2 = adapter.norm2(final_c2)
        f3 = adapter.norm3(final_c3)
        f4 = adapter.norm4(final_c4)
        return {"1": f1, "2": f2, "3": f3, "4": f4}

    @torch.inference_mode()
    def head_inference(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[str, Any]:
        m2f_head = self.model.segmentation_model[1]

        intermediate_raw = context.get("seg_intermediate_raw")
        image_metas = context.get("seg_image_metas")
        inference_mode = context.get("seg_inference_mode", "slide")
        decoder_head_type = context.get("seg_decoder_head_type", "m2f")

        if intermediate_raw is None or image_metas is None:
            raise RuntimeError("Missing context for segmentor head_inference().")

        total_sources = len(intermediate_raw)

        # Phase 1: Batch all sources by shape, run adapter + M2F predict
        shape_groups = {}
        for si in range(total_sources):
            shape = tuple(context["seg_source_shapes"][si])
            shape_groups.setdefault(shape, []).append(si)

        source_preds = [None] * total_sources

        with torch.autocast("cuda", self.autocast_dtype):
            for shape_key, src_indices in shape_groups.items():
                is_single = len(src_indices) == 1
                with torch.cuda.nvtx.range(f"seg_adapter_batch_{len(src_indices)}src"):
                    if is_single:
                        adapter_features = self._run_adapter_postprocess(src_indices[0], context)
                    else:
                        adapter_features = self._run_adapter_postprocess_batch(src_indices, context)

                with torch.cuda.nvtx.range(f"seg_m2f_predict_batch{len(src_indices)}"):
                    crop_hw = context["seg_crop_hw"][src_indices[0]]
                    batch_pred_dict = m2f_head.predict(adapter_features, rescale_to=crop_hw)

                with torch.cuda.nvtx.range(f"seg_m2f_postprocess_batch{len(src_indices)}"):
                    if decoder_head_type == "m2f":
                        mask_pred = batch_pred_dict["pred_masks"]
                        mask_cls = batch_pred_dict["pred_logits"]
                        mask_cls_sm = F.softmax(mask_cls, dim=-1)[..., :-1]
                        mask_pred_sig = mask_pred.sigmoid()
                        batch_pred = torch.einsum(
                            "bqc,bqhw->bchw",
                            mask_cls_sm.to(torch.bfloat16),
                            mask_pred_sig.to(torch.bfloat16),
                        )
                        del mask_cls, mask_pred, mask_cls_sm, mask_pred_sig

                for local_i, si in enumerate(src_indices):
                    source_preds[si] = batch_pred[local_i].unsqueeze(0)
                del batch_pred

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

                # TTA post-processing
                if apply_flip:
                    pred = F.hflip(pred)
                pred = F.softmax(pred, dim=1)
                aggregated_preds += pred.float()
                del pred

            pred_label = (aggregated_preds / n_tta).argmax(dim=1)[0].to(torch.uint8)
            outputs.append(pred_label.cpu())
            del aggregated_preds

        context["seg_outputs"] = outputs
        return {}

    def decide_exit(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[str, Any]:
        return {"num_exits": 0}
