from __future__ import annotations

import os
import sys
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from offload.common import Task
from offload.common.protocol import normalize_appcorr_kwargs
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

    # ── Full inference (single-pass) ────────────────────────────────────

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

    # ── Decomposed inference ─────────────────────────────────────────────

    def prepare_tokens(self, task: Task, context: Dict[str, Any], config: Any):
        images = context.get("input_images_uint8")
        if images is None:
            raise RuntimeError("Missing context['input_images_uint8'] for depther prepare_tokens().")
        target_shapes = context.get("target_shapes") or [None] * len(images)

        profile_config = config.get_input_profile_config()
        eval_size = int(profile_config.get("depther_eval_size", 768))
        use_tta = bool(profile_config.get("depther_use_tta", True))

        vit_backbone = self.model.encoder.backbone
        out_indices = self.model.encoder.backbone_out_indices

        all_x_backbones = []
        all_rope_sincos = []
        all_token_shapes = []
        image_metas = []

        for image_idx, image_np in enumerate(images):
            target_shape = target_shapes[image_idx] if image_idx < len(target_shapes) else None
            img_tensor = self._pil_to_normalized_tensor(image_np)
            input_tensor = F.interpolate(img_tensor, size=(eval_size, eval_size), mode="bilinear", align_corners=False)

            tta_tensors = [input_tensor]
            flip_flags = [False]
            if use_tta:
                tta_tensors.append(torch.flip(input_tensor, [-1]))
                flip_flags.append(True)

            rescale_to = target_shape if target_shape is not None else image_np.shape[:2]

            tta_source_ranges = []
            for tta_idx, (img_t, _apply_flip) in enumerate(zip(tta_tensors, flip_flags)):
                src_start = len(all_x_backbones)

                with torch.cuda.nvtx.range(f"depther_prepare_src{len(all_x_backbones)}"):
                    with torch.autocast("cuda", self.autocast_dtype):
                        x_tokens, (tok_H, tok_W) = vit_backbone.prepare_tokens_with_masks(img_t)
                        rope = vit_backbone.rope_embed(H=tok_H, W=tok_W) if vit_backbone.rope_embed else None

                all_x_backbones.append(x_tokens)
                all_rope_sincos.append(rope)
                all_token_shapes.append((tok_H, tok_W))

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

            del img_tensor, input_tensor

        context["depther_x_backbones"] = all_x_backbones
        context["depther_rope_sincos"] = all_rope_sincos
        context["depther_token_shapes"] = all_token_shapes
        context["depther_image_metas"] = image_metas
        context["depther_out_indices"] = out_indices
        context.pop("depther_group_maps", None)
        context.pop("depther_group_plans", None)
        context.pop("depther_cached_dindices", None)
        self._ensure_group_maps_and_plans(context, config)
        return {}

    def approx_forward(self, params: Dict[str, Any], context: Dict[str, Any], config: Any):
        start_l, end_l = params.get("layers", (0, 40))

        vit_backbone = self.model.encoder.backbone
        out_indices = context.get("depther_out_indices", [9, 19, 29, 39])

        all_x_backbones = context.get("depther_x_backbones")
        all_rope_sincos = context.get("depther_rope_sincos")
        if all_x_backbones is None or all_rope_sincos is None:
            return {}

        if "depther_current_features" not in context:
            context["depther_current_features"] = [x.clone() for x in all_x_backbones]

        current_features = context["depther_current_features"]

        all_cache_features = context.get("depther_cache_features")
        if all_cache_features is None:
            all_cache_features = [dict() for _ in range(len(all_x_backbones))]

        all_intermediate_outputs = context.get("depther_intermediate_outputs")
        if all_intermediate_outputs is None:
            all_intermediate_outputs = [{} for _ in range(len(all_x_backbones))]

        appcorr_options = normalize_appcorr_kwargs(config.appcorr_kwargs, config.transmission_kwargs)
        appcorr_method = appcorr_options["method"]

        self._ensure_group_maps_and_plans(context, config)
        all_group_plans = context.get("depther_group_plans")

        with torch.autocast("cuda", self.autocast_dtype):
            for src_idx in range(len(all_x_backbones)):
                x_tokens = current_features[src_idx] if start_l > 0 else all_x_backbones[src_idx].clone()
                rope = all_rope_sincos[src_idx]
                cache = all_cache_features[src_idx]
                intermediates = all_intermediate_outputs[src_idx]

                group_plans = all_group_plans[src_idx] if appcorr_method == "partial_channel" and all_group_plans is not None else None
                attn_cache_candidates = (
                    {gid: plan.full_dindice for gid, plan in group_plans.items()}
                    if group_plans is not None else None
                )

                with torch.cuda.nvtx.range(f"depther_vit_src{src_idx}_L{start_l}-{end_l}"):
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

                        # Collect intermediate outputs at out_indices for DPT head
                        if lidx in out_indices:
                            intermediates[lidx] = x_tokens.clone()

                current_features[src_idx] = x_tokens
                all_cache_features[src_idx] = cache
                all_intermediate_outputs[src_idx] = intermediates

        context["depther_current_features"] = current_features
        context["depther_cache_features"] = all_cache_features
        context["depther_intermediate_outputs"] = all_intermediate_outputs
        context["cache_feature"] = self._aggregate_cache_features(all_cache_features)
        return {}

    def correct_forward(self, params: Dict[str, Any], context: Dict[str, Any], config: Any):
        layers = params.get("layers", (0, 40))
        group_id = params.get("group_id", 0)
        start_l, end_l = layers[0], layers[1]

        vit_backbone = self.model.encoder.backbone
        out_indices = context.get("depther_out_indices", [9, 19, 29, 39])

        all_x_backbones = context.get("depther_x_backbones")
        all_rope_sincos = context.get("depther_rope_sincos")
        if all_x_backbones is None or all_rope_sincos is None:
            return {}

        if "depther_current_features" not in context:
            context["depther_current_features"] = [x.clone() for x in all_x_backbones]

        current_features = context["depther_current_features"]

        all_cache_features = context.get("depther_cache_features")
        if all_cache_features is None:
            all_cache_features = [dict() for _ in range(len(all_x_backbones))]

        all_intermediate_outputs = context.get("depther_intermediate_outputs")
        if all_intermediate_outputs is None:
            all_intermediate_outputs = [{} for _ in range(len(all_x_backbones))]

        all_cached_dindices = context.get("depther_cached_dindices")
        all_group_plans = context.get("depther_group_plans")

        self._ensure_group_maps_and_plans(context, config)
        all_cached_dindices = context.get("depther_cached_dindices")
        all_group_plans = context.get("depther_group_plans")

        appcorr_options = normalize_appcorr_kwargs(config.appcorr_kwargs, config.transmission_kwargs)
        appcorr_method = appcorr_options["method"]
        token_keep_ratio = appcorr_options["token_keep_ratio"]
        token_keep_thres = appcorr_options["token_keep_thres"]
        sdpa_query_bucket_size = appcorr_options["sdpa_query_bucket_size"]
        skip_patch_correction = self._partial_token_threshold_forces_no_patch_keep(appcorr_options)

        new_current_features = []
        new_cache_features = []
        new_intermediate_outputs = []

        for src_idx in range(len(all_x_backbones)):
            input_tokens = all_x_backbones[src_idx]
            rope = all_rope_sincos[src_idx]
            cache = all_cache_features[src_idx]
            intermediates = dict(all_intermediate_outputs[src_idx])

            src_cached_dindices = (
                all_cached_dindices[src_idx]
                if isinstance(all_cached_dindices, list) and src_idx < len(all_cached_dindices)
                else {}
            )
            src_group_plans = (
                all_group_plans[src_idx]
                if isinstance(all_group_plans, list) and src_idx < len(all_group_plans)
                else {}
            )

            if group_id in src_cached_dindices:
                target_gids = [group_id]
            else:
                target_gids = sorted(src_cached_dindices.keys())

            if not target_gids:
                new_current_features.append(current_features[src_idx])
                new_cache_features.append(cache)
                new_intermediate_outputs.append(intermediates)
                continue

            all_dindices_for_src = []
            for gid in target_gids:
                d = src_cached_dindices.get(gid)
                if d is not None:
                    all_dindices_for_src.append(d)

            if not all_dindices_for_src:
                new_current_features.append(current_features[src_idx])
                new_cache_features.append(cache)
                new_intermediate_outputs.append(intermediates)
                continue

            dindice = all_dindices_for_src[0] if len(all_dindices_for_src) == 1 else torch.cat(all_dindices_for_src, dim=1)
            dindice = dindice.to(device=self.device, non_blocking=True)
            plan = src_group_plans.get(target_gids[0]) if appcorr_method == "partial_channel" else None

            if dindice is None:
                new_current_features.append(current_features[src_idx])
                new_cache_features.append(cache)
                new_intermediate_outputs.append(intermediates)
                continue

            if appcorr_method == "partial_channel":
                if plan is None:
                    new_current_features.append(current_features[src_idx])
                    new_cache_features.append(cache)
                    new_intermediate_outputs.append(intermediates)
                    continue
                dindice = plan.pruned_dindice.to(device=self.device, non_blocking=True)
                plan.pruned_dindice = dindice
                fixed_query_state = plan.query_state
                attn_col_alive_ratio = appcorr_options["attn_col_alive_ratio"]
            else:
                fixed_query_state = None
                attn_col_alive_ratio = 1.0

            if skip_patch_correction:
                self._record_zero_patch_correction_stats(
                    cache,
                    dindice,
                    input_tokens,
                    rope,
                    num_layers=end_l - start_l,
                )
                new_current_features.append(current_features[src_idx])
                new_cache_features.append(cache)
                new_intermediate_outputs.append(intermediates)
                continue

            x_feature = input_tokens

            with torch.autocast("cuda", self.autocast_dtype):
                with torch.cuda.nvtx.range(f"depther_correct_src{src_idx}_g{group_id}_L{start_l}-{end_l}"):
                    for lidx in range(start_l, end_l):
                        blk = vit_backbone.blocks[lidx]

                        if appcorr_method == "partial_channel":
                            x_feature, cache = blk.correct(
                                x_feature, dindice, rope, cache, tag=f"src{src_idx}_layer{lidx}",
                                appcorr_method=appcorr_method,
                                token_keep_ratio=token_keep_ratio,
                                token_keep_thres=token_keep_thres,
                                mobile_pscore=appcorr_options["mobile_pscore"],
                                mobile_pscore_weight=appcorr_options["mobile_pscore_weight"],
                                mobile_pscore_hint=None,
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
                            x_feature, cache = blk.correct(
                                x_feature, dindice, rope, cache, tag=f"src{src_idx}_layer{lidx}",
                                appcorr_method=appcorr_method,
                                token_keep_ratio=token_keep_ratio,
                                token_keep_thres=token_keep_thres,
                                mobile_pscore=appcorr_options["mobile_pscore"],
                                mobile_pscore_weight=appcorr_options["mobile_pscore_weight"],
                                mobile_pscore_hint=None,
                                server_pscore=appcorr_options["server_pscore"],
                                server_pscore_weight=appcorr_options["server_pscore_weight"],
                                pscore_fusion=appcorr_options["pscore_fusion"],
                                sdpa_query_bucket_size=sdpa_query_bucket_size,
                                attn_col_alive_ratio=attn_col_alive_ratio,
                                debug=False,
                            )

                        if lidx in out_indices:
                            intermediates[lidx] = x_feature.clone()

                new_current_features.append(x_feature)
                new_cache_features.append(cache)
                new_intermediate_outputs.append(intermediates)

        context["depther_current_features"] = new_current_features
        context["depther_cache_features"] = new_cache_features
        context["depther_intermediate_outputs"] = new_intermediate_outputs
        context["cache_feature"] = self._aggregate_cache_features(new_cache_features)
        return {}

    @staticmethod
    def _partial_token_threshold_forces_no_patch_keep(appcorr_options: Dict[str, Any]) -> bool:
        if appcorr_options.get("method") != "partial_token":
            return False
        token_keep_thres = appcorr_options.get("token_keep_thres")
        if token_keep_thres in {None, "", "null", "None"}:
            try:
                return float(appcorr_options.get("token_keep_ratio", 0.2)) <= 0.0
            except (TypeError, ValueError):
                return False
        try:
            threshold = float(token_keep_thres)
        except (TypeError, ValueError):
            return False

        # Depther currently has no mobile pscore hint, so partial-token scores are
        # bounded by the server attention-probability score and its configured weight.
        server_weight = max(float(appcorr_options.get("server_pscore_weight", 1.0)), 0.0)
        return threshold > server_weight

    @staticmethod
    def _record_zero_patch_correction_stats(
        cache: Dict[str, Any],
        dindice: torch.Tensor,
        input_tokens: torch.Tensor,
        rope: Any,
        *,
        num_layers: int,
    ) -> None:
        if dindice.ndim != 2 or num_layers <= 0:
            return

        if rope is not None:
            num_pretokens = input_tokens.shape[1] - rope[0].shape[0]
        else:
            num_pretokens = 0
        num_patch_candidates = max(int(dindice.shape[1]) - int(num_pretokens), 0)
        batch_size = int(dindice.shape[0])

        zero = input_tokens.new_zeros((), dtype=torch.float32)
        full_patch_total = input_tokens.new_tensor(
            float(batch_size * num_patch_candidates * num_layers),
            dtype=torch.float32,
        )
        sample_total = input_tokens.new_tensor(
            float(batch_size * num_layers),
            dtype=torch.float32,
        )

        cache["_partial_token_kept_patch_total"] = cache.get("_partial_token_kept_patch_total", zero) + zero
        cache["_partial_token_full_patch_total"] = cache.get("_partial_token_full_patch_total", zero) + full_patch_total
        cache["_partial_token_sample_total"] = cache.get("_partial_token_sample_total", zero) + sample_total

    @torch.inference_mode()
    def head_inference(self, task: Task, context: Dict[str, Any], config: Any):
        vit_backbone = self.model.encoder.backbone
        encoder_wrapper = self.model.encoder
        decoder = self.model.decoder
        features_to_depth = self.model.features_to_depth

        current_features = context.get("depther_current_features")
        token_shapes = context.get("depther_token_shapes")
        image_metas = context.get("depther_image_metas")
        out_indices = context.get("depther_out_indices", [9, 19, 29, 39])
        all_intermediate_outputs = context.get("depther_intermediate_outputs")

        if current_features is None or token_shapes is None or image_metas is None:
            raise RuntimeError("Missing context for depther head_inference().")

        outputs = []
        for image_idx, image_meta in enumerate(image_metas):
            rescale_to = image_meta["rescale_to"]
            flip_flags = image_meta["flip_flags"]
            n_tta = image_meta["n_tta"]
            tta_source_ranges = image_meta["tta_source_ranges"]

            aggregated_depth = torch.zeros(1, 1, *rescale_to, dtype=torch.float32, device=self.device)

            for tta_idx, tta_range in enumerate(tta_source_ranges):
                src_start = tta_range["src_start"]
                src_end = tta_range["src_end"]
                apply_flip = flip_flags[tta_idx]

                # Collect intermediate features from all sources in this TTA pass
                # and run DPT head per source, then average
                for si in range(src_start, src_end):
                    x_tokens = current_features[si]
                    intermediates = all_intermediate_outputs[si] if all_intermediate_outputs else {}

                    # Apply backbone norm + reshape, mimicking DinoVisionTransformerWrapper.forward
                    with torch.autocast("cuda", self.autocast_dtype):
                        layer_outputs = []
                        for layer_idx in out_indices:
                            if layer_idx in intermediates:
                                out = intermediates[layer_idx]
                            else:
                                out = x_tokens

                            # Norm
                            if encoder_wrapper.final_norm:
                                if vit_backbone.untie_cls_and_patch_norms:
                                    x_norm_cls_reg = vit_backbone.cls_norm(out[:, :vit_backbone.n_storage_tokens + 1])
                                    x_norm_patch = vit_backbone.norm(out[:, vit_backbone.n_storage_tokens + 1:])
                                    out = torch.cat((x_norm_cls_reg, x_norm_patch), dim=1)
                                else:
                                    out = vit_backbone.norm(out)

                            # Extract patch tokens and cls token
                            cls_token = out[:, 0]
                            patch_out = out[:, vit_backbone.n_storage_tokens + 1:]

                            tok_H, tok_W = token_shapes[si]
                            patch_spatial = patch_out.reshape(1, tok_H, tok_W, -1).permute(0, 3, 1, 2).contiguous()

                            layer_outputs.append((patch_spatial, cls_token))

                        # DPT decoder
                        depth_logits = decoder(layer_outputs)
                        depth = features_to_depth(depth_logits)

                    if apply_flip:
                        depth = depth.flip([-1])

                    if depth.shape[-2:] != rescale_to:
                        depth = F.interpolate(depth, size=rescale_to, mode="bilinear", align_corners=False)

                    aggregated_depth += depth.float()
                    del depth, layer_outputs

            depth_avg = aggregated_depth / (n_tta * (src_end - src_start))
            outputs.append(depth_avg[0, 0].cpu())
            del aggregated_depth

        context["depth_outputs"] = outputs
        return {}

    def decide_exit(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[str, Any]:
        return {"num_exits": 0}

    def get_final_results(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[int, Any]:
        outputs = context.get("depth_outputs")
        if outputs is None:
            return {}

        results = {}
        for local_idx, output in enumerate(outputs):
            results[local_idx] = self._format_depth_output(output)
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

    # ── AppCorr group map / plan helpers ────────────────────────────────

    def _ensure_group_maps_and_plans(self, context: Dict[str, Any], config: Any):
        from appcorr.models.dinov3.models.vision_transformer import create_group_index

        all_x_backbones = context.get("depther_x_backbones")
        if all_x_backbones is None:
            return

        appcorr_options = normalize_appcorr_kwargs(config.appcorr_kwargs, config.transmission_kwargs)
        appcorr_method = appcorr_options["method"]
        num_pretokens = 1 + self.model.encoder.backbone.n_storage_tokens

        all_group_maps = context.get("depther_group_maps")
        all_group_plans = context.get("depther_group_plans")
        all_cached_dindices = context.get("depther_cached_dindices")

        valid_group_maps = (
            isinstance(all_group_maps, list)
            and len(all_group_maps) == len(all_x_backbones)
            and all(
                torch.is_tensor(group_map)
                and group_map.ndim == 2
                and group_map.shape[0] == input_tokens.shape[0]
                and group_map.shape[1] == input_tokens.shape[1] - num_pretokens
                for group_map, input_tokens in zip(all_group_maps, all_x_backbones)
            )
        )
        valid_cached_dindices = (
            isinstance(all_cached_dindices, list)
            and len(all_cached_dindices) == len(all_x_backbones)
            and all(isinstance(src_dindices, dict) for src_dindices in all_cached_dindices)
        )
        valid_group_plans = (
            appcorr_method != "partial_channel"
            or (
                isinstance(all_group_plans, list)
                and len(all_group_plans) == len(all_x_backbones)
                and all(isinstance(src_plans, dict) for src_plans in all_group_plans)
            )
        )
        if valid_group_maps and valid_cached_dindices and valid_group_plans:
            return

        if not valid_group_maps:
            all_group_maps = [None] * len(all_x_backbones)
        if not isinstance(all_group_plans, list) or len(all_group_plans) != len(all_x_backbones):
            all_group_plans = [None] * len(all_x_backbones)
        if not valid_cached_dindices:
            all_cached_dindices = [None] * len(all_x_backbones)

        num_groups = appcorr_options.get("num_groups", 1)

        for src_idx in range(len(all_x_backbones)):
            if all_group_maps[src_idx] is not None and all_cached_dindices[src_idx] is not None:
                continue
            x_tokens = all_x_backbones[src_idx]
            tok_H, tok_W = context["depther_token_shapes"][src_idx]
            N = tok_H * tok_W
            B = x_tokens.shape[0]

            if num_groups < 1:
                all_group_maps[src_idx] = None
                all_group_plans[src_idx] = {}
                all_cached_dindices[src_idx] = {}
            elif num_groups == 1:
                group_map = torch.zeros(B, N, dtype=torch.long, device=x_tokens.device)
                all_group_maps[src_idx] = group_map

                pre_indices = torch.arange(
                    num_pretokens, device=x_tokens.device, dtype=torch.long,
                ).unsqueeze(0).expand(B, -1)
                patch_indices = torch.arange(
                    num_pretokens,
                    num_pretokens + N,
                    device=x_tokens.device,
                    dtype=torch.long,
                ).unsqueeze(0).expand(B, -1)
                all_cached_dindices[src_idx] = {
                    0: torch.cat([pre_indices, patch_indices], dim=1),
                }

                if appcorr_method == "partial_channel":
                    all_group_plans[src_idx] = self._build_group_plans(
                        x_tokens, group_map, num_groups, appcorr_options
                    )
                else:
                    all_group_plans[src_idx] = {}
            else:
                group_map = create_group_index(N, num_groups, "grid", self.device, token_hw=(tok_H, tok_W))
                group_map = group_map.unsqueeze(0).expand(B, -1)
                all_group_maps[src_idx] = group_map

                src_cached_dindices = {}
                group_ids = torch.unique(group_map)
                group_ids = group_ids[group_ids >= 0]
                for gid_tensor in group_ids:
                    gid = int(gid_tensor.item())
                    nonzero_indices = torch.nonzero(group_map == gid, as_tuple=False)
                    if nonzero_indices.numel() == 0:
                        continue
                    try:
                        spatial_indices = nonzero_indices[:, 1].view(B, -1)
                    except RuntimeError:
                        continue
                    patch_indices = spatial_indices + num_pretokens
                    pre_indices = torch.arange(
                        num_pretokens, device=x_tokens.device, dtype=torch.long,
                    ).unsqueeze(0).expand(B, -1)
                    dindice = torch.cat([pre_indices, patch_indices], dim=1)
                    src_cached_dindices[gid] = dindice

                all_cached_dindices[src_idx] = src_cached_dindices

                if appcorr_method == "partial_channel":
                    all_group_plans[src_idx] = self._build_group_plans(
                        x_tokens, group_map, num_groups, appcorr_options
                    )
                else:
                    all_group_plans[src_idx] = {}

        context["depther_group_maps"] = all_group_maps
        context["depther_group_plans"] = all_group_plans
        context["depther_cached_dindices"] = all_cached_dindices

    @staticmethod
    def _build_group_plans(x_tokens, group_map, num_groups, appcorr_options):
        from offload.server.model.dinov3_segmentor_linhead import GroupCorrectionPlan
        plans = {}
        for gid in range(1, num_groups + 1):
            plans[gid] = GroupCorrectionPlan(
                num_pretokens=0,
                prefix_dindice=torch.tensor([], device=x_tokens.device),
                group_patch_dindice=torch.tensor([], device=x_tokens.device),
                full_dindice=torch.tensor([], device=x_tokens.device),
                pruned_dindice=torch.tensor([], device=x_tokens.device),
                query_state=None,
                kept_patch_count=torch.tensor(0, device=x_tokens.device),
                full_patch_count=torch.tensor(0, device=x_tokens.device),
                kept_residual_mass=torch.tensor(1.0, device=x_tokens.device),
                full_residual_mass=torch.tensor(1.0, device=x_tokens.device),
            )
        return plans

    @staticmethod
    def _aggregate_cache_features(all_cache_features):
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
        merged = {}
        total_values = {}
        for src_cache in all_cache_features:
            for key, value in src_cache.items():
                if key in total_keys:
                    total_values[key] = total_values.get(key, 0.0) + value
                else:
                    merged[key] = value
        merged.update(total_values)
        return merged
