from dataclasses import dataclass
from typing import Any, Dict, List
import gc
import math
import os

import torch
import torch.nn.functional as F
import numpy as np

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


class DINOv3SegmentorExecutor(ModelExecutor):
    OFFICIAL_ADE20K_EVAL_SIZE = 896
    OFFICIAL_ADE20K_EVAL_STRIDE = 596
    OFFICIAL_ADE20K_TTA_RATIOS = [0.9, 0.95, 1.0, 1.05, 1.1]
    OFFICIAL_ADE20K_SLIDE_OOM_FALLBACKS = (
        ((768, 768), (512, 512)),
        ((512, 512), (341, 341)),
    )

    def __init__(self, device: torch.device):
        super().__init__(device)
        self.normalize_avg = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.normalize_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.norm_mean = torch.tensor(self.normalize_avg).view(1, 3, 1, 1).to(self.device).float()
        self.norm_std = torch.tensor(self.normalize_std).view(1, 3, 1, 1).to(self.device).float()

    @staticmethod
    def _get_group_plans(context: Dict[str, Any]) -> Dict[int, GroupCorrectionPlan]:
        return context.setdefault("group_plans", {})

    def _get_active_batch_indices(self, context: Dict[str, Any], full_batch_size: int) -> torch.Tensor:
        if "active_indices" in context and context["active_indices"] is not None:
            active = context["active_indices"]
            if len(active) < full_batch_size:
                return active.to(device=self.device, dtype=torch.long)
        return torch.arange(full_batch_size, device=self.device, dtype=torch.long)

    def _compute_patch_residual_rms(
        self,
        context: Dict[str, Any],
        config: Any,
        active_indices: torch.Tensor,
    ) -> torch.Tensor | None:
        curr_np = context.get("input_hr_np")
        prev_np = context.get("prev_input_hr_np")
        if curr_np is None or prev_np is None:
            return None

        curr_full = torch.from_numpy(curr_np).to(device=self.device, non_blocking=True).float()
        prev_full = torch.from_numpy(prev_np).to(device=self.device, non_blocking=True).float()
        if curr_full.ndim != 4 or prev_full.ndim != 4:
            return None

        curr = curr_full[active_indices]
        prev = prev_full[active_indices]
        residual = curr - prev

        H, W = config.image_shape[:2]
        if isinstance(config.patch_size, int):
            ph = pw = config.patch_size
        else:
            ph, pw = config.patch_size
        gh, gw = H // ph, W // pw

        residual = residual.view(curr.shape[0], gh, ph, gw, pw, curr.shape[-1])
        residual = residual.permute(0, 1, 3, 2, 4, 5).contiguous()
        residual_sq_mean = residual.square().mean(dim=(3, 4, 5))
        return torch.sqrt(residual_sq_mean.view(curr.shape[0], gh * gw))

    def _apply_image_residual_token_pruning(
        self,
        dindice: torch.Tensor,
        spatial_indices: torch.Tensor,
        patch_residual_rms: torch.Tensor | None,
        threshold: float,
        min_keep: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        B, total_selected = dindice.shape
        num_pretokens = total_selected - spatial_indices.shape[1]
        full_patch_count = torch.full((B,), spatial_indices.shape[1], device=dindice.device, dtype=torch.int32)
        full_residual_mass = torch.zeros((B,), device=dindice.device, dtype=torch.float32)
        kept_residual_mass = torch.zeros((B,), device=dindice.device, dtype=torch.float32)
        full_keep_local_idx = torch.arange(
            spatial_indices.shape[1],
            device=dindice.device,
            dtype=torch.long,
        ).unsqueeze(0).expand(B, -1)

        if patch_residual_rms is None or spatial_indices.shape[1] == 0:
            kept_patch_count = full_patch_count.clone()
            return (
                dindice,
                kept_patch_count,
                full_patch_count,
                kept_residual_mass,
                full_residual_mass,
                full_keep_local_idx,
            )

        residual_sel = patch_residual_rms.gather(1, spatial_indices)
        full_residual_mass = residual_sel.sum(dim=1, dtype=torch.float32)
        keep_mask = residual_sel >= threshold

        if min_keep > 0 and spatial_indices.shape[1] > 0:
            k = min(min_keep, spatial_indices.shape[1])
            topk_idx = torch.topk(residual_sel, k=k, dim=1, largest=True).indices
            keep_mask.scatter_(1, topk_idx, True)

        kept_patch_count = keep_mask.sum(dim=1, dtype=torch.int32)
        kept_residual_mass = (residual_sel * keep_mask.to(dtype=residual_sel.dtype)).sum(dim=1, dtype=torch.float32)
        max_keep = int(kept_patch_count.max().item()) if kept_patch_count.numel() > 0 else 0
        if max_keep == spatial_indices.shape[1]:
            return (
                dindice,
                kept_patch_count,
                full_patch_count,
                kept_residual_mass,
                full_residual_mass,
                full_keep_local_idx,
            )

        kept_spatial = torch.zeros((B, max_keep), device=dindice.device, dtype=spatial_indices.dtype)
        kept_local_idx = torch.zeros((B, max_keep), device=dindice.device, dtype=torch.long)
        if max_keep > 0:
            batch_idx, patch_idx = keep_mask.nonzero(as_tuple=True)
            counts = torch.bincount(batch_idx, minlength=B)
            batch_start = counts.cumsum(0) - counts
            slot_idx = torch.arange(patch_idx.shape[0], device=dindice.device, dtype=torch.long)
            slot_idx = slot_idx - torch.repeat_interleave(batch_start, counts)
            kept_spatial[batch_idx, slot_idx] = spatial_indices[batch_idx, patch_idx]
            kept_local_idx[batch_idx, slot_idx] = patch_idx

        pruned_dindice = dindice[:, :num_pretokens]
        if max_keep > 0:
            pruned_dindice = torch.cat([pruned_dindice, kept_spatial + num_pretokens], dim=1)
        return (
            pruned_dindice,
            kept_patch_count,
            full_patch_count,
            kept_residual_mass,
            full_residual_mass,
            kept_local_idx,
        )

    def _build_fixed_query_state(
        self,
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
            spatial_indices.shape[1],
            device=self.device,
            dtype=torch.long,
        ).unsqueeze(0).expand(B, -1)
        pruned_dindice = dindice

        if token_prune_enabled:
            (
                pruned_dindice,
                kept_patch_count,
                full_patch_count,
                kept_residual_mass,
                full_residual_mass,
                group_patch_keep_local_idx,
            ) = self._apply_image_residual_token_pruning(
                dindice,
                spatial_indices,
                patch_residual_rms,
                token_prune_threshold,
                token_prune_min_keep,
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

    def _unwrap_backbone_for_blocks(self):
        return self._get_vit_backbone()

    def _get_segmentation_backbone(self):
        return self.model.segmentation_model[0]

    def _get_vit_backbone(self):
        return self._get_segmentation_backbone().backbone

    def _get_segmentation_head(self):
        return self.model.segmentation_model[1]

    @staticmethod
    def _keep_batchnorm_in_fp32(module: torch.nn.Module) -> None:
        batchnorm_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
        )
        for submodule in module.modules():
            if isinstance(submodule, batchnorm_types):
                submodule.float()

    def load_model(self, model_name: str, config: Any):
        print(f"[Executor] Loading Segmentor Model (MMap): {model_name}...")
        if self.model is not None:
            del self.model
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        from dinov3.hub.segmentors import dinov3_vit7b16_ms

        backbone_path = "/home/nxc/cjpark/weights/dinov3/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
        head_path = "/home/nxc/cjpark/weights/dinov3/dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth"

        print(f"[Executor] Loading Backbone Weights from {backbone_path}")
        print(f"[Executor] Loading Segmentation Weights from {head_path}")

        self.model = dinov3_vit7b16_ms(
            pretrained=False,
            weights="ADE20K",
            backbone_weights="LVD1689M",
            autocast_dtype=torch.bfloat16,
        )
        # Keep most weights in bf16 so the 7B segmentor fits on a 24 GB GPU,
        # but restore normalization layers and the decoder head to fp32 for stability.
        if self.device.type == "cuda":
            self.model.to(dtype=torch.bfloat16)
            self._keep_batchnorm_in_fp32(self.model)
            self._get_segmentation_head().float()
        self.model.to(device=self.device)

        backbone_state = load_weight_mmap(backbone_path)
        self._get_vit_backbone().load_state_dict(backbone_state, strict=True)
        del backbone_state

        head_state = load_weight_mmap(head_path)
        missing_keys, unexpected_keys = self.model.load_state_dict(head_state, strict=False)
        non_backbone_missing = [k for k in missing_keys if "backbone" not in k]
        if non_backbone_missing:
            raise RuntimeError(f"Unexpected non-backbone missing_keys: {non_backbone_missing[:20]}")
        if unexpected_keys:
            raise RuntimeError(f"Unexpected keys in segmentation head checkpoint: {unexpected_keys[:20]}")
        del head_state

        self.model.eval()

    def preprocess(self, batch_data: Any, task: Task, context: Dict[str, Any], config: Any):
        if isinstance(batch_data, torch.Tensor):
            with torch.cuda.nvtx.range("Preprocess::ToDevice"):
                tensor = batch_data
                if tensor.ndim != 4:
                    raise ValueError(f"Expected 4D tensor input, got {tensor.shape}")

                if tensor.shape[1] == 3:
                    pass
                elif tensor.shape[-1] == 3:
                    tensor = tensor.permute(0, 3, 1, 2).contiguous()
                else:
                    raise ValueError(f"Expected channel dimension of size 3, got {tensor.shape}")

                full_batch = tensor.shape[0]
                tensor = tensor.to(device=self.device, non_blocking=True).float()

                if batch_data.dtype == torch.uint8 or tensor.max() > 1.5:
                    tensor = tensor / 255.0

                tensor = (tensor - self.norm_mean) / self.norm_std
        else:
            with torch.cuda.nvtx.range("Preprocess::PinMemory"):
                tensor = torch.from_numpy(batch_data)
                if hasattr(tensor, "pin_memory"):
                    tensor = tensor.pin_memory()

            with torch.cuda.nvtx.range("Preprocess::ToDevice"):
                if tensor.ndim != 4:
                    raise ValueError(f"Expected 4D numpy input, got {tensor.shape}")

                if tensor.shape[-1] == 3:
                    tensor = tensor.permute(0, 3, 1, 2).contiguous()
                elif tensor.shape[1] == 3:
                    pass
                else:
                    raise ValueError(f"Expected NHWC or NCHW with 3 channels, got {tensor.shape}")

                full_batch = tensor.shape[0]
                tensor = tensor.to(device=self.device, non_blocking=True).float()

                if tensor.max() > 1.5:
                    tensor = tensor / 255.0

                tensor = (tensor - self.norm_mean) / self.norm_std

        with torch.cuda.nvtx.range("Preprocess::Slicing"):
            if "active_indices" in context and context["active_indices"] is not None:
                active_indices = context["active_indices"]
                if len(active_indices) < full_batch:
                    tensor = tensor[active_indices]
            else:
                context["active_indices"] = torch.arange(
                    full_batch, device=self.device, dtype=torch.long
                )

            context["input_tensor"] = tensor

        with torch.cuda.nvtx.range("Preprocess::GroupMap"):
            context.pop("cached_dindices", None)

            H, W = config.image_shape[:2]
            if isinstance(config.patch_size, int):
                ph = pw = config.patch_size
            else:
                ph, pw = config.patch_size

            num_patches = (H // ph) * (W // pw)
            curr_B = tensor.shape[0]
            group_map = torch.full((curr_B, num_patches), -1, device=self.device, dtype=torch.long)
            context["group_map"] = group_map

            if "active_indices" in context and len(context["active_indices"]) < full_batch:
                active_list = context["active_indices"].tolist()
                idx_map = {orig: local for local, orig in enumerate(active_list)}
                valid_payload = [p for p in task.payload if p.image_idx in idx_map]

                if valid_payload:
                    b_t = torch.tensor(
                        [idx_map[p.image_idx] for p in valid_payload],
                        device=self.device,
                        dtype=torch.long,
                    )
                    s_t = torch.tensor(
                        [p.spatial_idx for p in valid_payload],
                        device=self.device,
                        dtype=torch.long,
                    )
                    g_t = torch.tensor(
                        [p.group_id for p in valid_payload],
                        device=self.device,
                        dtype=torch.long,
                    )
                    group_map[b_t, s_t] = g_t
            else:
                if task.payload:
                    b_t = torch.tensor(
                        [p.image_idx for p in task.payload],
                        device=self.device,
                        dtype=torch.long,
                    )
                    s_t = torch.tensor(
                        [p.spatial_idx for p in task.payload],
                        device=self.device,
                        dtype=torch.long,
                    )
                    g_t = torch.tensor(
                        [p.group_id for p in task.payload],
                        device=self.device,
                        dtype=torch.long,
                    )
                    group_map[b_t, s_t] = g_t

        with torch.cuda.nvtx.range("Preprocess::Dindices"):
            vit_backbone = self._get_vit_backbone()
            num_pretokens = 1 + getattr(vit_backbone, "n_storage_tokens", 0)
            B = group_map.shape[0]

            appcorr_options = normalize_appcorr_kwargs(config.appcorr_kwargs)
            appcorr_method = appcorr_options["method"]
            grouping_strategy = config.transmission_kwargs.get("grouping_strategy", "uniform_diff")
            num_groups = config.transmission_kwargs.get("num_groups", 4)
            active_indices = self._get_active_batch_indices(context, config.batch_size)
            patch_residual_rms = self._compute_patch_residual_rms(context, config, active_indices)
            token_prune_enabled = appcorr_options["token_prune_enabled"]
            token_prune_threshold = appcorr_options["token_prune_threshold"]
            token_prune_min_keep = appcorr_options["token_prune_min_keep"]

            group_plans = self._get_group_plans(context)
            group_plans.clear()

            b_list = []
            s_list = []
            g_list = []
            b_t = s_t = g_t = None

            if "active_indices" in context and len(context["active_indices"]) < full_batch:
                active_list = context["active_indices"].tolist()
                idx_map = {orig: local for local, orig in enumerate(active_list)}
                valid_payload = [p for p in task.payload if p.image_idx in idx_map]
                if valid_payload:
                    b_list = [idx_map[p.image_idx] for p in valid_payload]
                    s_list = [p.spatial_idx for p in valid_payload]
                    g_list = [p.group_id for p in valid_payload]
                    b_t = torch.tensor(b_list, device=self.device, dtype=torch.long)
                    s_t = torch.tensor(s_list, device=self.device, dtype=torch.long)
                    g_t = torch.tensor(g_list, device=self.device, dtype=torch.long)
            else:
                if task.payload:
                    b_list = [p.image_idx for p in task.payload]
                    s_list = [p.spatial_idx for p in task.payload]
                    g_list = [p.group_id for p in task.payload]
                    b_t = torch.tensor(b_list, device=self.device, dtype=torch.long)
                    s_t = torch.tensor(s_list, device=self.device, dtype=torch.long)
                    g_t = torch.tensor(g_list, device=self.device, dtype=torch.long)

            if appcorr_method == "partial_channel" and grouping_strategy in {"grid", "uniform", "geometric"}:
                num_tokens = group_map.shape[1]
                group_idx = create_group_index(
                    num_tokens,
                    num_groups,
                    grouping_strategy,
                    device=self.device,
                )
                if group_idx.ndim == 1:
                    group_idx = group_idx.unsqueeze(0).expand(B, -1)

                pre_indices = torch.arange(num_pretokens, device=self.device).unsqueeze(0).expand(B, -1)
                for gid in range(1, num_groups + 1):
                    spatial_indices = torch.where(group_idx == gid)[1].view(B, -1)
                    patch_indices = spatial_indices + num_pretokens
                    dindice = torch.cat([pre_indices, patch_indices], dim=1)
                    plan = self._build_group_plan(
                        dindice,
                        spatial_indices,
                        patch_residual_rms,
                        num_pretokens,
                        token_prune_enabled,
                        token_prune_threshold,
                        token_prune_min_keep,
                    )
                    group_plans[gid] = plan

            elif b_list:
                involved_groups = set(g_list)
                for gid in involved_groups:
                    mask = g_t == gid
                    if not mask.any():
                        continue
                    try:
                        spatial_indices = s_t[mask].view(B, -1)
                    except RuntimeError:
                        print(f"!!! [Executor] Non-uniform group {gid} size detected. Fallback skip.")
                        continue

                    patch_indices = spatial_indices + num_pretokens
                    pre_indices = torch.arange(num_pretokens, device=self.device).unsqueeze(0).expand(B, -1)
                    dindice = torch.cat([pre_indices, patch_indices], dim=1)
                    plan = self._build_group_plan(
                        dindice,
                        spatial_indices,
                        patch_residual_rms,
                        num_pretokens,
                        token_prune_enabled,
                        token_prune_threshold,
                        token_prune_min_keep,
                    )
                    group_plans[gid] = plan

    def prepare_tokens(self, task: Task, context: Dict[str, Any], config: Any):
        if "input_tensor" not in context:
            return

        tensor = context["input_tensor"]
        backbone = self._get_vit_backbone()

        if not hasattr(backbone, "prepare_tokens_with_masks"):
            raise RuntimeError("Backbone does not implement prepare_tokens_with_masks().")

        t2_x, hw_tuple = backbone.prepare_tokens_with_masks(tensor, None)
        context["input_tokens"] = t2_x
        context["hw_tuple"] = hw_tuple

        if "current_feature" not in context:
            context["current_feature"] = t2_x

        if getattr(backbone, "rope_embed", None) is not None:
            prev_hw = context.get("rope_hw_tuple")
            if prev_hw != hw_tuple:
                context["rope_sincos"] = backbone.rope_embed(
                    H=hw_tuple[0], W=hw_tuple[1]
                )
                context["rope_hw_tuple"] = hw_tuple

    def approx_forward(self, params: Dict[str, Any], context: Dict[str, Any], config: Any):
        pass

    def correct_forward(self, params: Dict[str, Any], context: Dict[str, Any], config: Any):
        vit_backbone = self._unwrap_backbone_for_blocks()
        all_x_backbones = context.get("all_x_backbones")
        all_rope_sincos = context.get("all_rope_sincos")

        if all_x_backbones is None or all_rope_sincos is None:
            raise RuntimeError("Missing context['all_x_backbones'] or context['all_rope_sincos'].")

        layers_to_use = getattr(vit_backbone, "layers_to_use", None)
        blocks = vit_backbone.blocks
        if layers_to_use is None:
            raise RuntimeError(
                "Backbone does not expose layers_to_use needed for segmentation feature extraction."
            )

        blocks_to_take = (
            range(len(blocks) - layers_to_use, len(blocks))
            if isinstance(layers_to_use, int)
            else layers_to_use
        )

        all_outputs = []
        for x_backbone, rope_sincos in zip(all_x_backbones, all_rope_sincos):
            output = []
            for i, blk in enumerate(blocks):
                x_backbone = blk(x_backbone, rope_sincos)
                if i in blocks_to_take:
                    output.append(x_backbone)
            all_outputs.append(output)

        context["all_outputs"] = all_outputs

        if len(all_outputs) == 1:
            context["outputs"] = all_outputs[0]

    def _process_outputs(
        self,
        outputs: List[torch.Tensor],
        x: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        B, _, h, w = x.shape
        vit_backbone = self._unwrap_backbone_for_blocks()
        dense_backbone = self._get_segmentation_backbone()

        ps = vit_backbone.patch_size
        if isinstance(ps, int):
            ph = pw = ps
        else:
            ph, pw = ps

        expected = (h // ph) * (w // pw)
        special = getattr(vit_backbone, "n_storage_tokens", 0) + 1

        xs = []
        for i, out in enumerate(outputs):
            n = out.shape[1]
            if n == expected:
                out = vit_backbone.norm(out)
            elif n == expected + special:
                if getattr(vit_backbone, "untie_cls_and_patch_norms", False):
                    out = torch.cat(
                        [
                            vit_backbone.cls_norm(out[:, :special]),
                            vit_backbone.norm(out[:, special:]),
                        ],
                        dim=1,
                    )
                else:
                    out = vit_backbone.norm(out)
                out = out[:, special:]
            else:
                raise RuntimeError(
                    f"Unexpected token count at layer {i}: got {n}, "
                    f"expected {expected} or {expected + special}"
                )

            feat = out.reshape(B, h // ph, w // pw, -1).permute(0, 3, 1, 2).contiguous()
            xs.append(feat)

        if hasattr(dense_backbone, "use_layernorm") and dense_backbone.use_layernorm:
            xs = [ln(feat).contiguous() for ln, feat in zip(dense_backbone.layer_norms, xs)]

        return {str(i + 1): feat for i, feat in enumerate(xs)}

    def _extract_seg_logits(self, seg_out: Any) -> torch.Tensor:
        if torch.is_tensor(seg_out):
            return seg_out

        if isinstance(seg_out, dict):
            if "pred_masks" in seg_out and "pred_logits" in seg_out:
                mask_pred = seg_out["pred_masks"]
                mask_cls = seg_out["pred_logits"]
                mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
                mask_pred = mask_pred.sigmoid()
                return torch.einsum(
                    "bqc,bqhw->bchw",
                    mask_cls.to(torch.float32),
                    mask_pred.to(torch.float32),
                )
            for key in ("preds", "logits", "seg_logits", "out", "sem_seg"):
                if key in seg_out and torch.is_tensor(seg_out[key]):
                    return seg_out[key]

        if isinstance(seg_out, (list, tuple)):
            for item in seg_out:
                try:
                    return self._extract_seg_logits(item)
                except Exception:
                    pass

        if hasattr(seg_out, "pred_sem_seg") and hasattr(seg_out.pred_sem_seg, "data"):
            return seg_out.pred_sem_seg.data
        if hasattr(seg_out, "seg_logits") and hasattr(seg_out.seg_logits, "data"):
            return seg_out.seg_logits.data

        raise RuntimeError(
            f"Could not extract segmentation logits from output type={type(seg_out)}"
        )

    @staticmethod
    def _normalize_hw(value: Any, default_hw: tuple[int, int]) -> tuple[int, int]:
        if value is None:
            return default_hw
        if isinstance(value, int):
            return (value, value)
        if isinstance(value, (list, tuple)) and len(value) == 2:
            return (int(value[0]), int(value[1]))
        raise ValueError(f"Expected int or pair for spatial size, got {value!r}")

    @staticmethod
    def _is_cuda_oom(exc: BaseException) -> bool:
        return isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()

    def _get_slide_oom_fallbacks(
        self,
        crop_size: tuple[int, int],
        stride: tuple[int, int],
    ) -> List[tuple[tuple[int, int], tuple[int, int]]]:
        seen = {(crop_size, stride)}
        fallbacks: List[tuple[tuple[int, int], tuple[int, int]]] = []
        for candidate_crop, candidate_stride in self.OFFICIAL_ADE20K_SLIDE_OOM_FALLBACKS:
            if candidate_crop[0] > crop_size[0] or candidate_crop[1] > crop_size[1]:
                continue
            key = (candidate_crop, candidate_stride)
            if key in seen:
                continue
            fallbacks.append(key)
            seen.add(key)
        return fallbacks

    def _get_segmentation_inference_cfg(self, config: Any) -> Dict[str, Any]:
        raw = dict(getattr(config, "scheduler_kwargs", {}) or {})
        crop_size = self._normalize_hw(
            raw.get("seg_crop_size", self.OFFICIAL_ADE20K_EVAL_SIZE),
            (self.OFFICIAL_ADE20K_EVAL_SIZE, self.OFFICIAL_ADE20K_EVAL_SIZE),
        )
        stride = self._normalize_hw(
            raw.get("seg_stride", self.OFFICIAL_ADE20K_EVAL_STRIDE),
            (self.OFFICIAL_ADE20K_EVAL_STRIDE, self.OFFICIAL_ADE20K_EVAL_STRIDE),
        )
        tta_ratios = raw.get("seg_tta_ratios", self.OFFICIAL_ADE20K_TTA_RATIOS)
        return {
            "mode": str(raw.get("seg_inference_mode", "whole")),
            "crop_size": crop_size,
            "stride": stride,
            "use_tta": bool(raw.get("seg_use_tta", False)),
            "tta_ratios": [float(r) for r in tta_ratios],
            "oom_fallbacks": self._get_slide_oom_fallbacks(crop_size, stride),
        }

    def _predict_from_feature_maps(
        self,
        feature_maps: Dict[str, torch.Tensor],
        rescale_to: tuple[int, int],
    ) -> torch.Tensor:
        head = self._get_segmentation_head()
        seg_out = head.predict(feature_maps, rescale_to=rescale_to)
        seg_logits = self._extract_seg_logits(seg_out)
        if seg_logits.shape[-2:] != rescale_to:
            seg_logits = F.interpolate(
                seg_logits,
                size=rescale_to,
                mode="bilinear",
                align_corners=False,
            )
        return seg_logits

    def _predict_from_seg_output(
        self,
        seg_out: Any,
        rescale_to: tuple[int, int],
    ) -> torch.Tensor:
        seg_logits = self._extract_seg_logits(seg_out)
        if seg_logits.shape[-2:] != rescale_to:
            seg_logits = F.interpolate(
                seg_logits,
                size=rescale_to,
                mode="bilinear",
                align_corners=False,
            )
        return seg_logits

    def _forward_segmentation_backbone(self, input_tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self._get_segmentation_backbone()(input_tensor)

    def _build_tta_views(self, input_tensor: torch.Tensor, tta_ratios: List[float]) -> List[torch.Tensor]:
        _, _, in_h, in_w = input_tensor.shape
        views: List[torch.Tensor] = []

        for ratio in tta_ratios:
            small_size = int(self.OFFICIAL_ADE20K_EVAL_SIZE * ratio)
            if ratio < 1.0:
                small_size = int(math.ceil(small_size / 32.0) * 32)

            if in_h <= in_w:
                resized_h = small_size
                resized_w = int(small_size * in_w / in_h + 0.5)
            else:
                resized_w = small_size
                resized_h = int(small_size * in_h / in_w + 0.5)

            view = F.interpolate(
                input_tensor,
                size=(resized_h, resized_w),
                mode="bilinear",
                align_corners=False,
            )
            views.append(view)

        return views

    def _slide_inference(
        self,
        input_tensor: torch.Tensor,
        rescale_to: tuple[int, int],
        crop_size: tuple[int, int],
        stride: tuple[int, int],
    ) -> torch.Tensor:
        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = input_tensor.shape
        if h_crop > h_img and w_crop > w_img:
            h_crop, w_crop = min(h_img, w_img), min(h_img, w_img)

        assert batch_size == 1, "Slide inference currently expects a single image per forward."

        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        if input_tensor.device.type == "cuda":
            torch.cuda.empty_cache()
        preds = torch.zeros(
            (1, 150, h_img, w_img),
            device="cpu",
            dtype=torch.float32,
        )
        count_mat = torch.zeros(
            (1, 1, h_img, w_img),
            device="cpu",
            dtype=torch.float32,
        )

        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)

                crop_img = input_tensor[:, :, y1:y2, x1:x2]
                crop_features = self._forward_segmentation_backbone(crop_img)
                crop_pred = self._predict_from_feature_maps(crop_features, crop_img.shape[-2:])
                preds[:, :, y1:y2, x1:x2] += crop_pred.to(device="cpu", dtype=torch.float32)
                count_mat[:, :, y1:y2, x1:x2] += 1.0
                del crop_img, crop_features, crop_pred

        pred = (preds / torch.clamp(count_mat, min=1.0)).to(
            device=input_tensor.device,
            non_blocking=True,
        )
        if pred.shape[-2:] != rescale_to:
            pred = F.interpolate(
                pred,
                size=rescale_to,
                mode="bilinear",
                align_corners=False,
            )
        return pred

    def _predict_single_input(
        self,
        input_tensor: torch.Tensor,
        rescale_to: tuple[int, int],
        inference_cfg: Dict[str, Any],
    ) -> torch.Tensor:
        if inference_cfg["mode"] == "slide":
            slide_attempts = [
                (inference_cfg["crop_size"], inference_cfg["stride"]),
                *inference_cfg.get("oom_fallbacks", []),
            ]
            for attempt_idx, (crop_size, stride) in enumerate(slide_attempts):
                try:
                    if attempt_idx > 0:
                        print(
                            "[SegmentorExecutor] Retrying slide inference "
                            f"with crop_size={crop_size}, stride={stride}."
                        )
                    return self._slide_inference(
                        input_tensor,
                        rescale_to=rescale_to,
                        crop_size=crop_size,
                        stride=stride,
                    )
                except Exception as exc:
                    if (
                        input_tensor.device.type != "cuda"
                        or not self._is_cuda_oom(exc)
                        or attempt_idx == len(slide_attempts) - 1
                    ):
                        raise
                    print(
                        "[SegmentorExecutor] CUDA OOM during slide inference "
                        f"with crop_size={crop_size}, stride={stride}. "
                        "Clearing cache before retry."
                    )
                    gc.collect()
                    torch.cuda.empty_cache()

        resized_input = F.interpolate(
            input_tensor,
            size=(512, 512),
            mode="bilinear",
            align_corners=False,
        )
        feature_maps = self._forward_segmentation_backbone(resized_input)
        return self._predict_from_feature_maps(feature_maps, rescale_to)

    def _predict_input_tensor(
        self,
        input_tensor: torch.Tensor,
        config: Any,
        rescale_to: tuple[int, int],
    ) -> torch.Tensor:
        inference_cfg = self._get_segmentation_inference_cfg(config)
        if not inference_cfg["use_tta"] and inference_cfg["mode"] != "slide":
            resized_input = F.interpolate(
                input_tensor,
                size=(512, 512),
                mode="bilinear",
                align_corners=False,
            )
            feature_maps = self._forward_segmentation_backbone(resized_input)
            return self._predict_from_feature_maps(feature_maps, rescale_to)

        preds = []
        for batch_idx in range(input_tensor.shape[0]):
            sample = input_tensor[batch_idx : batch_idx + 1]
            sample_rescale_to = tuple(int(v) for v in rescale_to)

            if not inference_cfg["use_tta"]:
                preds.append(self._predict_single_input(sample, sample_rescale_to, inference_cfg))
                continue

            accum = None
            count = 0
            for view in self._build_tta_views(sample, inference_cfg["tta_ratios"]):
                pred = self._predict_single_input(view, sample_rescale_to, inference_cfg)
                pred = torch.softmax(pred, dim=1)
                accum = pred if accum is None else accum + pred
                count += 1

                flipped_view = torch.flip(view, dims=(-1,))
                flipped_pred = self._predict_single_input(flipped_view, sample_rescale_to, inference_cfg)
                flipped_pred = torch.flip(flipped_pred, dims=(-1,))
                flipped_pred = torch.softmax(flipped_pred, dim=1)
                accum = accum + flipped_pred
                count += 1

            preds.append(accum / max(count, 1))

        return torch.cat(preds, dim=0)

    def predict(
        self,
        *,
        input_tensor: torch.Tensor | None = None,
        seg_output: Any | None = None,
        feature_maps: Dict[str, torch.Tensor] | None = None,
        config: Any | None = None,
        rescale_to: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        if rescale_to is None:
            if input_tensor is None:
                raise ValueError("predict() needs rescale_to when input_tensor is not provided.")
            rescale_to = tuple(int(v) for v in input_tensor.shape[-2:])

        if feature_maps is not None:
            return self._predict_from_feature_maps(feature_maps, rescale_to)

        if seg_output is not None and input_tensor is None:
            return self._predict_from_seg_output(seg_output, rescale_to)

        if input_tensor is None:
            raise ValueError("predict() requires input_tensor or feature_maps/seg_output.")

        if seg_output is not None:
            inference_cfg = self._get_segmentation_inference_cfg(config)
            if not inference_cfg["use_tta"] and inference_cfg["mode"] != "slide":
                return self._predict_from_seg_output(seg_output, rescale_to)

        return self._predict_input_tensor(input_tensor, config, rescale_to)

    def head_inference(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[str, Any]:
        outputs = context.get("outputs")
        input_tensor = context.get("input_tensor")

        if outputs is None:
            all_outputs = context.get("all_outputs")
            if all_outputs is not None and len(all_outputs) == 1:
                outputs = all_outputs[0]

        if outputs is None:
            raise RuntimeError("Missing context['outputs'] for head_inference().")
        if input_tensor is None:
            raise RuntimeError("Missing context['input_tensor'] for head_inference().")

        feature_maps = self._process_outputs(outputs, input_tensor)
        seg_logits = self.predict(
            feature_maps=feature_maps,
            input_tensor=input_tensor,
            config=config,
            rescale_to=tuple(int(v) for v in input_tensor.shape[-2:]),
        )

        context["seg_logits"] = seg_logits
        context["seg_pred"] = seg_logits.argmax(dim=1)

        active_indices = context.get(
            "active_indices",
            torch.arange(seg_logits.shape[0], device=self.device, dtype=torch.long),
        )

        return {
            "active_indices": active_indices.detach().cpu().numpy().tolist(),
        }

    @torch.inference_mode()
    def full_inference(self, task: Task, context: Dict[str, Any], config: Any):
        inp = context.get("input_tensor")
        if inp is None:
            return

        inference_cfg = self._get_segmentation_inference_cfg(config)
        rescale_to = tuple(int(v) for v in inp.shape[-2:])

        if not inference_cfg["use_tta"] and inference_cfg["mode"] != "slide":
            resized_input = F.interpolate(
                inp,
                size=(512, 512),
                mode="bilinear",
                align_corners=False,
            )
            context["seg_outputs"] = self._forward_segmentation_backbone(resized_input)
            context["seg_output"] = context["seg_outputs"]
            context["seg_logits"] = self.predict(
                feature_maps=context["seg_outputs"],
                config=config,
                rescale_to=rescale_to,
            )
        else:
            context["seg_outputs"] = None
            context["seg_output"] = None
            context["seg_logits"] = self.predict(
                input_tensor=inp,
                config=config,
                rescale_to=rescale_to,
            )
        context["seg_pred"] = context["seg_logits"].argmax(dim=1)

    def get_final_results(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[int, Any]:
        results = {}
        seg_pred = context.get("seg_pred")
        seg_logits = context.get("seg_logits")

        if seg_pred is None:
            return results

        indices = context.get("active_indices")
        if indices is None:
            indices = torch.arange(seg_pred.shape[0], device=seg_pred.device, dtype=torch.long)

        indices_np = indices.detach().cpu().numpy()
        for i, orig_idx in enumerate(indices_np):
            payload = {
                "segmentation": seg_pred[i].to(torch.int32).detach().cpu().numpy(),
            }
            if seg_logits is not None:
                payload["logits"] = seg_logits[i].float().detach().cpu().numpy()
            results[int(orig_idx)] = payload

        return results

    def decide_exit(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[str, Any]:
        # Segmentation early-exit is not implemented yet. Keep all active samples
        # alive so the pipeline remains correct and the executor is instantiable.
        if config.early_exit_enabled() and not context.get("_warned_segmentation_early_exit", False):
            print("[SegmentorExecutor] Early exit requested, but segmentation early-exit is not implemented. Continuing without exiting samples.")
            context["_warned_segmentation_early_exit"] = True

        return {
            "num_exits": 0,
            "supported": False,
        }
