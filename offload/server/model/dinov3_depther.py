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
        self._sdpa_warmup_done = False

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
        self._sdpa_warmup_done = False

    def _maybe_warmup_sdpa_buckets(self, config: Any):
        if self._sdpa_warmup_done or not torch.cuda.is_available():
            return

        appcorr_options = normalize_appcorr_kwargs(config.appcorr_kwargs, config.transmission_kwargs)
        bucket_size = int(appcorr_options.get("sdpa_query_bucket_size", 0) or 0)
        if bucket_size <= 0 or not bool(appcorr_options.get("sdpa_warmup", True)):
            self._sdpa_warmup_done = True
            return

        vit_backbone = self.model.encoder.backbone
        first_block = vit_backbone.blocks[0]
        num_heads = int(first_block.attn.num_heads)
        hidden_dim = int(first_block.attn.qkv.in_features)
        head_dim = hidden_dim // num_heads
        num_pretokens = 1 + int(getattr(vit_backbone, "n_storage_tokens", 0))

        profile_config = config.get_input_profile_config()
        eval_size = int(profile_config.get("depther_eval_size", 768))
        model_patch_size = getattr(vit_backbone, "patch_size", None)
        if model_patch_size is None and hasattr(vit_backbone, "patch_embed"):
            model_patch_size = getattr(vit_backbone.patch_embed, "patch_size", None)
        if isinstance(model_patch_size, (tuple, list)):
            ph, pw = (int(v) for v in model_patch_size)
        else:
            ph = pw = int(model_patch_size or 16)
        patch_tokens = (eval_size // ph) * (eval_size // pw)
        num_groups = max(int(config.transmission_kwargs.get("num_groups", 1)), 1)
        group_tokens = (patch_tokens + num_groups - 1) // num_groups
        max_query_tokens = num_pretokens + group_tokens
        max_bucket = ((max_query_tokens + bucket_size - 1) // bucket_size) * bucket_size
        key_tokens = num_pretokens + patch_tokens

        with torch.cuda.device(self.device):
            with torch.cuda.nvtx.range("depther_sdpa_bucket_warmup"):
                k = torch.empty(
                    (1, num_heads, key_tokens, head_dim),
                    device=self.device,
                    dtype=self.autocast_dtype,
                )
                v = torch.empty_like(k)
                for query_tokens in range(bucket_size, max_bucket + 1, bucket_size):
                    with torch.cuda.nvtx.range(f"sdpa_bucket_T{query_tokens}"):
                        q = torch.empty(
                            (1, num_heads, query_tokens, head_dim),
                            device=self.device,
                            dtype=self.autocast_dtype,
                        )
                        torch.nn.functional.scaled_dot_product_attention(q, k, v)
                torch.cuda.synchronize(self.device)

        print(
            f"[Executor] Warmed SDPA correction buckets: "
            f"T={bucket_size}..{max_bucket} step {bucket_size}, K={key_tokens}, H={num_heads}, D={head_dim}"
        )
        self._sdpa_warmup_done = True

    @staticmethod
    def _bool_option(config: Any, key: str, default: bool = False) -> bool:
        appcorr_kwargs = getattr(config, "appcorr_kwargs", {}) or {}
        transmission_kwargs = getattr(config, "transmission_kwargs", {}) or {}
        if key in appcorr_kwargs:
            return bool(appcorr_kwargs[key])
        if key in transmission_kwargs:
            return bool(transmission_kwargs[key])
        return bool(default)

    @staticmethod
    def _profile_prepare_detail(config: Any) -> bool:
        return DINOv3DeptherExecutor._bool_option(config, "profile_prepare_detail", False)

    @staticmethod
    def _stage_input_in_preprocess(config: Any) -> bool:
        return DINOv3DeptherExecutor._bool_option(config, "depther_stage_input_in_preprocess", True)

    @staticmethod
    def _incremental_input_in_preprocess(config: Any) -> bool:
        return DINOv3DeptherExecutor._bool_option(config, "depther_incremental_input_in_preprocess", False)

    @staticmethod
    def _nvtx_range(name: str):
        return torch.cuda.nvtx.range(name)

    def preprocess(self, batch_data: Any, task: Task, context: Dict[str, Any], config: Any):
        images, target_shapes = self._as_image_list(batch_data)
        images = [np.ascontiguousarray(image) for image in images]
        context["input_images_uint8"] = images
        context["target_shapes"] = target_shapes
        stage_full_input = self._stage_input_in_preprocess(config)
        stage_incremental_input = self._incremental_input_in_preprocess(config)
        if not (stage_full_input or stage_incremental_input):
            context.pop("depther_input_batch", None)
        else:
            profile_config = config.get_input_profile_config()
            eval_size = int(profile_config.get("depther_eval_size", 768))
            profile_detail = self._profile_prepare_detail(config)
            updated_input = None
            if stage_incremental_input:
                updated_input = self._try_update_input_batch_from_patches(
                    images,
                    task,
                    context,
                    config,
                    eval_size,
                )
            if updated_input is None:
                context["depther_input_batch"] = self._build_input_batch_from_images(
                    images,
                    eval_size,
                    profile_detail,
                    "Preprocess",
                    batch_source=self._get_batch_source(batch_data, images, context),
                )
        context["depther_mobile_pscore_hint_maps"] = self._build_mobile_pscore_hint_maps(
            task,
            images,
            config,
        )

    def _build_mobile_pscore_hint_maps(
        self,
        task: Task,
        images: List[np.ndarray],
        config: Any,
    ) -> List[tuple[torch.Tensor, tuple[int, int]] | None] | None:
        appcorr_options = normalize_appcorr_kwargs(config.appcorr_kwargs, config.transmission_kwargs)
        if appcorr_options["mobile_pscore"] == "none" or appcorr_options["mobile_pscore_weight"] == 0.0:
            return None

        if not task.payload:
            return None

        if isinstance(config.patch_size, int):
            ph = pw = int(config.patch_size)
        else:
            ph, pw = (int(v) for v in config.patch_size)

        target_res_level = min(config.transmission_kwargs.get("pyramid_levels", [0]))
        hint_maps_cpu: List[np.ndarray | None] = []
        hint_shapes: List[tuple[int, int] | None] = []
        shape_to_indices: dict[tuple[int, int], List[int]] = {}
        for image_idx, image in enumerate(images):
            img_h, img_w = image.shape[:2]
            grid_h = img_h // ph
            grid_w = img_w // pw
            if grid_h <= 0 or grid_w <= 0:
                hint_maps_cpu.append(None)
                hint_shapes.append(None)
                continue
            hint_shape = (grid_h, grid_w)
            hint_maps_cpu.append(np.zeros((grid_h * grid_w,), dtype=np.float32))
            hint_shapes.append(hint_shape)
            shape_to_indices.setdefault(hint_shape, []).append(image_idx)

        for patch in task.payload:
            image_idx = int(getattr(patch, "image_idx", -1))
            if image_idx < 0 or image_idx >= len(hint_maps_cpu):
                continue
            if int(getattr(patch, "res_level", target_res_level)) != target_res_level:
                continue
            hint_map = hint_maps_cpu[image_idx]
            hint_shape = hint_shapes[image_idx]
            if hint_map is None or hint_shape is None:
                continue
            spatial_idx = int(getattr(patch, "spatial_idx", -1))
            if 0 <= spatial_idx < hint_map.shape[0]:
                hint_map[spatial_idx] = float(getattr(patch, "pscore_hint", 0.0))

        projected_input_maps: List[tuple[torch.Tensor, tuple[int, int]] | None] = [None] * len(images)
        for hint_shape, image_indices in shape_to_indices.items():
            stacked_cpu = np.stack(
                [hint_maps_cpu[image_idx] for image_idx in image_indices],
                axis=0,
            )
            hint_batch = torch.from_numpy(stacked_cpu).to(
                device=self.device,
                dtype=torch.float32,
                non_blocking=True,
            )
            normalized_batch = self._normalize_patch_score_map(hint_batch)
            for row_idx, image_idx in enumerate(image_indices):
                projected_input_maps[image_idx] = (normalized_batch[row_idx:row_idx + 1], hint_shape)
        return projected_input_maps

    def _project_mobile_pscore_hint_to_tokens(
        self,
        hint_entry: tuple[torch.Tensor, tuple[int, int]] | None,
        token_hw: tuple[int, int],
        *,
        apply_flip: bool,
    ) -> torch.Tensor | None:
        if hint_entry is None:
            return None

        hint_map, hint_hw = hint_entry
        hint_h, hint_w = hint_hw
        tok_h, tok_w = token_hw
        if hint_h <= 0 or hint_w <= 0 or tok_h <= 0 or tok_w <= 0:
            return None

        hint_2d = hint_map.view(1, 1, hint_h, hint_w)
        if apply_flip:
            hint_2d = torch.flip(hint_2d, dims=(-1,))
        if (hint_h, hint_w) != (tok_h, tok_w):
            hint_2d = F.interpolate(hint_2d, size=(tok_h, tok_w), mode="bilinear", align_corners=False)

        source_hint = hint_2d.reshape(1, tok_h * tok_w).contiguous()
        return self._normalize_patch_score_map(source_hint)

    def _project_mobile_pscore_hints_to_source(
        self,
        hint_entries: List[tuple[torch.Tensor, tuple[int, int]] | None] | None,
        token_hw: tuple[int, int],
        *,
        apply_flip: bool,
        batch_size: int,
        ref_tensor: torch.Tensor,
    ) -> torch.Tensor | None:
        if not isinstance(hint_entries, list) or batch_size <= 0:
            return None

        grouped_entries: dict[tuple[int, int], List[tuple[int, torch.Tensor]]] = {}
        for image_idx in range(batch_size):
            hint_entry = hint_entries[image_idx] if image_idx < len(hint_entries) else None
            if hint_entry is None:
                continue
            hint_map, hint_hw = hint_entry
            if hint_map is None or hint_hw is None:
                continue
            grouped_entries.setdefault(hint_hw, []).append((image_idx, hint_map))

        if not grouped_entries:
            return None

        tok_h, tok_w = token_hw
        source_hint = ref_tensor.new_zeros((batch_size, tok_h * tok_w), dtype=torch.float32)
        for (hint_h, hint_w), entries in grouped_entries.items():
            rows = [image_idx for image_idx, _hint_map in entries]
            hint_batch = torch.cat(
                [
                    hint_map.to(device=self.device, dtype=torch.float32, non_blocking=True)
                    for _image_idx, hint_map in entries
                ],
                dim=0,
            )
            hint_2d = hint_batch.view(len(entries), 1, hint_h, hint_w)
            if apply_flip:
                hint_2d = torch.flip(hint_2d, dims=(-1,))
            if (hint_h, hint_w) != (tok_h, tok_w):
                hint_2d = F.interpolate(hint_2d, size=(tok_h, tok_w), mode="bilinear", align_corners=False)

            projected = hint_2d.reshape(len(entries), tok_h * tok_w).contiguous()
            projected = self._normalize_patch_score_map(projected)
            row_indices = torch.as_tensor(rows, device=self.device, dtype=torch.long)
            source_hint.index_copy_(0, row_indices, projected)
        return source_hint

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
        cpu_tensor = torch.from_numpy(image_np)
        if hasattr(cpu_tensor, "pin_memory"):
            cpu_tensor = cpu_tensor.pin_memory()
        tensor = cpu_tensor.to(device=self.device, non_blocking=True)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).float()
        tensor = (tensor - self.norm_mean) / self.norm_std
        return tensor.to(dtype=self.autocast_dtype)

    def _pil_to_normalized_tensor_profiled(
        self,
        image_np: np.ndarray,
        prefix: str = "Prepare",
    ) -> torch.Tensor:
        cpu_tensor = torch.from_numpy(image_np)
        if hasattr(cpu_tensor, "pin_memory"):
            cpu_tensor = cpu_tensor.pin_memory()
        with self._nvtx_range(f"{prefix}::ImageH2D"):
            tensor = cpu_tensor.to(device=self.device, non_blocking=True)
        with self._nvtx_range(f"{prefix}::ImageCastFloat"):
            tensor = tensor.permute(2, 0, 1).unsqueeze(0).float()
        with self._nvtx_range(f"{prefix}::ImageNormalize"):
            tensor = (tensor - self.norm_mean) / self.norm_std
        with self._nvtx_range(f"{prefix}::ImageCastAutocast"):
            tensor = tensor.to(dtype=self.autocast_dtype)
        return tensor

    @staticmethod
    def _get_batch_source(
        batch_data: Any,
        images: List[np.ndarray],
        context: Dict[str, Any],
    ) -> np.ndarray | None:
        if not images:
            return None

        candidates = []
        if isinstance(batch_data, np.ndarray):
            candidates.append(batch_data)
        input_hr_np = context.get("input_hr_np")
        if (batch_data is None or isinstance(batch_data, list)) and isinstance(input_hr_np, np.ndarray):
            candidates.append(input_hr_np)

        expected_shape = images[0].shape
        for candidate in candidates:
            if (
                isinstance(candidate, np.ndarray)
                and candidate.ndim == 4
                and candidate.shape[0] == len(images)
                and candidate.shape[1:] == expected_shape
                and all(image.shape == expected_shape for image in images)
            ):
                return np.ascontiguousarray(candidate)
        return None

    def _normalize_input_batch_tensor(
        self,
        cpu_tensor: torch.Tensor,
        eval_size: int,
        prefix: str,
    ) -> torch.Tensor:
        with self._nvtx_range(f"{prefix}::BatchH2D"):
            tensor = cpu_tensor.to(device=self.device, non_blocking=True)
        with self._nvtx_range(f"{prefix}::BatchCastFloat"):
            tensor = tensor.permute(0, 3, 1, 2).float()
        with self._nvtx_range(f"{prefix}::BatchNormalize"):
            tensor = (tensor - self.norm_mean) / self.norm_std
        with self._nvtx_range(f"{prefix}::BatchCastAutocast"):
            tensor = tensor.to(dtype=self.autocast_dtype)
        with self._nvtx_range(f"{prefix}::ResizeInput"):
            if tuple(tensor.shape[-2:]) != (eval_size, eval_size):
                tensor = F.interpolate(tensor, size=(eval_size, eval_size), mode="bilinear", align_corners=False)
        return tensor.contiguous()

    def _build_input_batch_from_images(
        self,
        images: List[np.ndarray],
        eval_size: int,
        profile_prepare: bool,
        prefix: str,
        batch_source: np.ndarray | None = None,
    ) -> torch.Tensor:
        if not images:
            return torch.empty((0, 3, eval_size, eval_size), dtype=self.autocast_dtype, device=self.device)

        if batch_source is not None:
            with self._nvtx_range(f"{prefix}::BatchToTensor"):
                cpu_tensor = torch.from_numpy(batch_source)
                if hasattr(cpu_tensor, "pin_memory"):
                    cpu_tensor = cpu_tensor.pin_memory()
            return self._normalize_input_batch_tensor(cpu_tensor, eval_size, prefix)

        same_shape = all(image.shape == images[0].shape for image in images)
        if same_shape:
            with self._nvtx_range(f"{prefix}::BatchStack"):
                batch_np = np.ascontiguousarray(np.stack(images, axis=0))
            with self._nvtx_range(f"{prefix}::BatchToTensor"):
                cpu_tensor = torch.from_numpy(batch_np)
                if hasattr(cpu_tensor, "pin_memory"):
                    cpu_tensor = cpu_tensor.pin_memory()
            return self._normalize_input_batch_tensor(cpu_tensor, eval_size, prefix)

        normalized_inputs = []
        for image_np in images:
            with torch.cuda.nvtx.range(f"{prefix}::ImageToTensor"):
                if profile_prepare:
                    img_tensor = self._pil_to_normalized_tensor_profiled(
                        image_np,
                        prefix=prefix,
                    )
                else:
                    img_tensor = self._pil_to_normalized_tensor(image_np)
            with self._nvtx_range(f"{prefix}::ResizeInput"):
                input_tensor = F.interpolate(img_tensor, size=(eval_size, eval_size), mode="bilinear", align_corners=False)
            normalized_inputs.append(input_tensor)
        with self._nvtx_range(f"{prefix}::StackInput"):
            return torch.cat(normalized_inputs, dim=0).contiguous()

    def _try_update_input_batch_from_patches(
        self,
        images: List[np.ndarray],
        task: Task,
        context: Dict[str, Any],
        config: Any,
        eval_size: int,
    ) -> torch.Tensor | None:
        input_batch = context.get("depther_input_batch")
        if not torch.is_tensor(input_batch):
            return None
        if input_batch.ndim != 4 or tuple(input_batch.shape[-2:]) != (eval_size, eval_size):
            return None
        if input_batch.shape[0] != len(images):
            return None
        if not task.payload:
            return input_batch

        if isinstance(config.patch_size, int):
            ph = pw = int(config.patch_size)
        else:
            ph, pw = (int(v) for v in config.patch_size)
        if ph <= 0 or pw <= 0:
            return None

        channels = int(input_batch.shape[1])
        if channels != 3:
            return None

        for image in images:
            if image.shape[:2] != (eval_size, eval_size) or image.shape[2] != channels:
                return None

        image_indices = []
        rows = []
        cols = []
        grid_w = (eval_size + pw - 1) // pw
        for patch in task.payload:
            image_idx = int(getattr(patch, "image_idx", -1))
            if image_idx < 0 or image_idx >= len(images):
                continue
            if int(getattr(patch, "group_id", -1)) == 0:
                return None
            if int(getattr(patch, "res_level", 0)) != 0:
                return None
            spatial_idx = int(getattr(patch, "spatial_idx", -1))
            if spatial_idx < 0:
                return None
            row, col = divmod(spatial_idx, grid_w)
            y, x = row * ph, col * pw
            if y < 0 or x < 0 or y + ph > eval_size or x + pw > eval_size:
                return None
            image_indices.append(image_idx)
            rows.append(row)
            cols.append(col)

        if not image_indices:
            return input_batch

        images_np = context.get("input_hr_np")
        if not (
            isinstance(images_np, np.ndarray)
            and images_np.ndim == 4
            and images_np.shape[0] == len(images)
            and images_np.shape[1:4] == (eval_size, eval_size, channels)
        ):
            images_np = np.stack(images, axis=0)
        images_np = np.ascontiguousarray(images_np)
        image_indices = np.asarray(image_indices, dtype=np.int64)
        rows = np.asarray(rows, dtype=np.int64)
        cols = np.asarray(cols, dtype=np.int64)
        grid_h = eval_size // ph
        grid_w = eval_size // pw
        s_b, s_h, s_w, s_c = images_np.strides
        patch_view_np = np.lib.stride_tricks.as_strided(
            images_np,
            shape=(images_np.shape[0], grid_h, grid_w, ph, pw, channels),
            strides=(s_b, ph * s_h, pw * s_w, s_h, s_w, s_c),
            writeable=False,
        )
        crops = np.ascontiguousarray(patch_view_np[image_indices, rows, cols])

        crops_cpu = torch.from_numpy(crops)
        if hasattr(crops_cpu, "pin_memory"):
            crops_cpu = crops_cpu.pin_memory()
        with self._nvtx_range("Preprocess::PatchBatchH2D"):
            crop_tensor = crops_cpu.to(device=self.device, non_blocking=True)
        with self._nvtx_range("Preprocess::PatchNormalize"):
            crop_tensor = crop_tensor.permute(0, 3, 1, 2).float()
            crop_tensor = (crop_tensor - self.norm_mean) / self.norm_std
            crop_tensor = crop_tensor.to(dtype=input_batch.dtype)

        image_idx_t = torch.from_numpy(image_indices).to(device=self.device, non_blocking=True)
        row_t = torch.from_numpy(rows).to(device=self.device, non_blocking=True)
        col_t = torch.from_numpy(cols).to(device=self.device, non_blocking=True)
        with self._nvtx_range("Preprocess::PatchScatter"):
            s_b, s_c, s_h, s_w = input_batch.stride()
            patch_view = input_batch.as_strided(
                (input_batch.shape[0], eval_size // ph, eval_size // pw, channels, ph, pw),
                (s_b, ph * s_h, pw * s_w, s_c, s_h, s_w),
            )
            patch_view[image_idx_t, row_t, col_t] = crop_tensor

        return input_batch

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
        input_batch = context.get("depther_input_batch")
        if not torch.is_tensor(input_batch):
            input_batch = self._build_input_batch_from_images(
                images,
                eval_size,
                profile_prepare=False,
                prefix="FullInference",
                batch_source=self._get_batch_source(None, images, context),
            )

        with torch.autocast("cuda", self.autocast_dtype):
            if use_tta:
                flipped = torch.flip(input_batch, [-1]).contiguous()
                pred = self.model(input_batch)
                pred_flip = self.model(flipped)
                pred_flip = torch.flip(pred_flip, [-1])
                depth_batch = (pred + pred_flip) / 2.0
            else:
                depth_batch = self.model(input_batch)

        for image_idx, image_np in enumerate(images):
            target_shape = target_shapes[image_idx] if image_idx < len(target_shapes) else None
            rescale_to = target_shape if target_shape is not None else image_np.shape[:2]
            depth = depth_batch[image_idx:image_idx + 1]
            if depth.shape[-2:] != rescale_to:
                depth = F.interpolate(depth, size=rescale_to, mode="bilinear", align_corners=False)

            outputs.append(depth[0, 0].detach())

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
        profile_prepare = self._profile_prepare_detail(config)

        vit_backbone = self.model.encoder.backbone
        out_indices = self.model.encoder.backbone_out_indices
        self._maybe_warmup_sdpa_buckets(config)

        image_metas = []
        mobile_pscore_hint_maps = context.get("depther_mobile_pscore_hint_maps")
        rope_cache = context.setdefault("depther_rope_cache", {})

        for image_idx, image_np in enumerate(images):
            target_shape = target_shapes[image_idx] if image_idx < len(target_shapes) else None
            rescale_to = target_shape if target_shape is not None else image_np.shape[:2]
            image_metas.append({
                "rescale_to": rescale_to,
                "tta_entries": [],
            })

        input_batch = context.get("depther_input_batch")
        if input_batch is None:
            input_batch = self._build_input_batch_from_images(
                images,
                eval_size,
                profile_prepare,
                "Prepare",
                batch_source=self._get_batch_source(None, images, context),
            )

        source_inputs = [(input_batch, False)]
        if use_tta:
            with self._nvtx_range("Prepare::FlipInput"):
                source_inputs.append((torch.flip(input_batch, [-1]).contiguous(), True))

        all_x_backbones = []
        all_rope_sincos = []
        all_token_shapes = []
        all_mobile_pscore_hints = []
        all_source_flip_flags = []

        for src_idx, (src_tensor, apply_flip) in enumerate(source_inputs):
            with torch.cuda.nvtx.range(f"depther_prepare_src{src_idx}"):
                with torch.autocast("cuda", self.autocast_dtype):
                    with self._nvtx_range("Prepare::PatchEmbed"):
                        x_tokens, (tok_H, tok_W) = vit_backbone.prepare_tokens_with_masks(src_tensor)
                    if vit_backbone.rope_embed:
                        rope_key = (tok_H, tok_W)
                        rope = rope_cache.get(rope_key)
                        if rope is None:
                            with self._nvtx_range("Prepare::RopeMiss"):
                                rope = vit_backbone.rope_embed(H=tok_H, W=tok_W)
                            rope_cache[rope_key] = rope
                    else:
                        rope = None

            all_x_backbones.append(x_tokens)
            all_rope_sincos.append(rope)
            all_token_shapes.append((tok_H, tok_W))
            all_source_flip_flags.append(bool(apply_flip))

            with self._nvtx_range("Prepare::HintProject"):
                all_mobile_pscore_hints.append(
                    self._project_mobile_pscore_hints_to_source(
                        mobile_pscore_hint_maps,
                        (tok_H, tok_W),
                        apply_flip=bool(apply_flip),
                        batch_size=len(images),
                        ref_tensor=x_tokens,
                    )
                )
            for image_idx in range(len(images)):
                image_metas[image_idx]["tta_entries"].append({
                    "src_idx": src_idx,
                    "batch_idx": image_idx,
                    "apply_flip": bool(apply_flip),
                })

        del input_batch

        context["depther_x_backbones"] = all_x_backbones
        context["depther_rope_sincos"] = all_rope_sincos
        context["depther_token_shapes"] = all_token_shapes
        context["depther_mobile_pscore_hints"] = all_mobile_pscore_hints
        context["depther_source_flip_flags"] = all_source_flip_flags
        context["depther_image_metas"] = image_metas
        context["depther_out_indices"] = out_indices
        with self._nvtx_range("Prepare::GroupPlan"):
            self._ensure_group_maps_and_plans(context, config)

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
        all_mobile_pscore_hints = context.get("depther_mobile_pscore_hints")

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

            mobile_pscore_hint = None
            if isinstance(all_mobile_pscore_hints, list) and src_idx < len(all_mobile_pscore_hints):
                mobile_pscore_hint = all_mobile_pscore_hints[src_idx]

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
                                mobile_pscore_hint=mobile_pscore_hint,
                                server_pscore=appcorr_options["server_pscore"],
                                server_pscore_weight=appcorr_options["server_pscore_weight"],
                                pscore_fusion=appcorr_options["pscore_fusion"],
                                sdpa_query_bucket_size=sdpa_query_bucket_size,
                                attn_col_alive_ratio=attn_col_alive_ratio,
                                fixed_query_state=fixed_query_state,
                                group_plan=plan,
                                attn_cache_key=group_id,
                                partial_token_plan_stat_scale=end_l - start_l,
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
                                mobile_pscore_hint=mobile_pscore_hint,
                                server_pscore=appcorr_options["server_pscore"],
                                server_pscore_weight=appcorr_options["server_pscore_weight"],
                                pscore_fusion=appcorr_options["pscore_fusion"],
                                sdpa_query_bucket_size=sdpa_query_bucket_size,
                                attn_col_alive_ratio=attn_col_alive_ratio,
                                partial_token_plan_stat_scale=end_l - start_l,
                                inplace_residual_add=lidx > start_l,
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

        server_weight = max(float(appcorr_options.get("server_pscore_weight", 1.0)), 0.0)
        mobile_weight = max(float(appcorr_options.get("mobile_pscore_weight", 0.0)), 0.0)
        has_mobile_score = appcorr_options.get("mobile_pscore") != "none" and mobile_weight != 0.0
        if not has_mobile_score:
            max_score = server_weight
        else:
            pscore_fusion = appcorr_options.get("pscore_fusion", "add")
            if pscore_fusion == "multiply":
                max_score = server_weight * mobile_weight
            elif pscore_fusion == "geo_mean":
                max_score = (server_weight * mobile_weight) ** 0.5
            else:
                max_score = server_weight + mobile_weight
        return threshold > max_score

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
            tta_entries = image_meta["tta_entries"]
            if not tta_entries:
                continue

            aggregated_depth = torch.zeros(1, 1, *rescale_to, dtype=torch.float32, device=self.device)

            for tta_entry in tta_entries:
                si = int(tta_entry["src_idx"])
                batch_idx = int(tta_entry["batch_idx"])
                apply_flip = bool(tta_entry["apply_flip"])
                x_tokens = current_features[si]
                intermediates = all_intermediate_outputs[si] if all_intermediate_outputs else {}

                with torch.autocast("cuda", self.autocast_dtype):
                    layer_outputs = []
                    for layer_idx in out_indices:
                        if layer_idx in intermediates:
                            out = intermediates[layer_idx]
                        else:
                            out = x_tokens
                        out = out[batch_idx:batch_idx + 1]

                        if encoder_wrapper.final_norm:
                            if vit_backbone.untie_cls_and_patch_norms:
                                x_norm_cls_reg = vit_backbone.cls_norm(out[:, :vit_backbone.n_storage_tokens + 1])
                                x_norm_patch = vit_backbone.norm(out[:, vit_backbone.n_storage_tokens + 1:])
                                out = torch.cat((x_norm_cls_reg, x_norm_patch), dim=1)
                            else:
                                out = vit_backbone.norm(out)

                        cls_token = out[:, 0]
                        patch_out = out[:, vit_backbone.n_storage_tokens + 1:]

                        tok_H, tok_W = token_shapes[si]
                        patch_spatial = patch_out.reshape(1, tok_H, tok_W, -1).permute(0, 3, 1, 2).contiguous()

                        layer_outputs.append((patch_spatial, cls_token))

                    depth_logits = decoder(layer_outputs)
                    depth = features_to_depth(depth_logits)

                if apply_flip:
                    depth = depth.flip([-1])

                if depth.shape[-2:] != rescale_to:
                    depth = F.interpolate(depth, size=rescale_to, mode="bilinear", align_corners=False)

                aggregated_depth += depth.float()
                del depth, layer_outputs

            depth_avg = aggregated_depth / len(tta_entries)
            outputs.append(depth_avg[0, 0].detach())
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
        source_flip_flags = context.get("depther_source_flip_flags")
        if not isinstance(source_flip_flags, list) or len(source_flip_flags) != len(all_x_backbones):
            source_flip_flags = [False] * len(all_x_backbones)

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
                s = int(num_groups ** 0.5)
                pattern = torch.arange(1, num_groups + 1, dtype=torch.long).view(s, s)
                rep_h = (tok_H + s - 1) // s
                rep_w = (tok_W + s - 1) // s
                group_map_2d_cpu = pattern.repeat(rep_h, rep_w)[:tok_H, :tok_W]
                if bool(source_flip_flags[src_idx]):
                    group_map_2d_cpu = torch.flip(group_map_2d_cpu, dims=(-1,))
                group_map_1d_cpu = group_map_2d_cpu.flatten().contiguous()
                group_map_1d = group_map_1d_cpu.to(device=x_tokens.device, non_blocking=True)
                group_map = group_map_1d.unsqueeze(0).expand(B, -1)
                all_group_maps[src_idx] = group_map

                src_cached_dindices = {}
                pre_indices = torch.arange(
                    num_pretokens, device=x_tokens.device, dtype=torch.long,
                ).unsqueeze(0).expand(B, -1)
                for gid in range(1, num_groups + 1):
                    spatial_indices_cpu = torch.nonzero(group_map_1d_cpu == gid, as_tuple=False).flatten()
                    if spatial_indices_cpu.numel() == 0:
                        continue
                    spatial_indices = spatial_indices_cpu.to(device=x_tokens.device, non_blocking=True)
                    spatial_indices = spatial_indices.unsqueeze(0).expand(B, -1)
                    patch_indices = spatial_indices + num_pretokens
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
