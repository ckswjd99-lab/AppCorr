from dataclasses import dataclass
from typing import Any, Dict, List
import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as v2
import math
import numpy as np
from offload.common import Task
from .base import ModelExecutor
from .utils import load_weight_mmap

from appcorr.models.dinov3.eval.detection.util.misc import nested_tensor_from_tensor_list, NestedTensor
from appcorr.models.dinov3.eval.detection.util import box_ops
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


class DINOv3DetectorExecutor(ModelExecutor):
    def __init__(self, device: torch.device):
        super().__init__(device)
        self.normalize_avg = np.array([0.485, 0.456, 0.406])
        self.normalize_std = np.array([0.229, 0.224, 0.225])
        self.norm_mean = torch.tensor(self.normalize_avg).view(1, 3, 1, 1).to(self.device).float()
        self.norm_std = torch.tensor(self.normalize_std).view(1, 3, 1, 1).to(self.device).float()

    def _get_active_batch_indices(self, context: Dict[str, Any], full_batch_size: int) -> torch.Tensor:
        if 'active_indices' in context and len(context['active_indices']) < full_batch_size:
            return context['active_indices'].to(device=self.device, dtype=torch.long)
        return torch.arange(full_batch_size, device=self.device, dtype=torch.long)

    def _compute_patch_residual_rms_for_source(
        self,
        curr_source: torch.Tensor,
        prev_source: torch.Tensor | None,
        patch_size: int,
    ) -> torch.Tensor | None:
        if prev_source is None:
            return None

        residual = curr_source.float() - prev_source.float()
        B, _, H, W = residual.shape
        gh, gw = H // patch_size, W // patch_size
        residual = residual.view(B, 3, gh, patch_size, gw, patch_size)
        residual = residual.permute(0, 2, 4, 3, 5, 1).contiguous()
        residual_sq_mean = residual.square().mean(dim=(3, 4, 5))
        return torch.sqrt(residual_sq_mean.view(B, gh * gw))

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
        kept_residual_mass = torch.zeros((B,), device=dindice.device, dtype=torch.float32)
        full_residual_mass = torch.zeros((B,), device=dindice.device, dtype=torch.float32)
        pruned_dindice = dindice
        group_patch_keep_local_idx = torch.arange(
            spatial_indices.shape[1],
            device=dindice.device,
            dtype=torch.long,
        ).unsqueeze(0).expand(B, -1)
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

    def _aggregate_cache_features(self, all_cache_features: List[Dict[str, Any]] | None) -> Dict[str, Any]:
        if not all_cache_features:
            return {}

        merged: Dict[str, Any] = {}
        total_keys = {
            "_attn_prob_mass_used_total",
            "_attn_prob_mass_full_total",
            "_token_prune_kept_patch_total",
            "_token_prune_full_patch_total",
            "_token_prune_kept_residual_mass_total",
            "_token_prune_full_residual_mass_total",
        }
        total_values: Dict[str, Any] = {}
        for src_cache in all_cache_features:
            for key, value in src_cache.items():
                if key in total_keys:
                    total_values[key] = total_values.get(key, 0.0) + value
                else:
                    merged[key] = value
        merged.update(total_values)
        return merged

    def _build_all_group_maps(self, context: Dict[str, Any], config: Any) -> List[torch.Tensor] | None:
        all_input_tokens = context.get('all_x_backbones')
        if all_input_tokens is None:
            return None

        grouping_strategy = config.transmission_kwargs.get('grouping_strategy', 'uniform_diff')
        num_groups = config.transmission_kwargs.get('num_groups', 4)
        if grouping_strategy not in {'grid', 'uniform', 'geometric'}:
            return None

        all_group_maps = []
        num_pretokens = 1 + getattr(self._get_vit_backbone(), 'n_storage_tokens', 0)
        for input_tokens in all_input_tokens:
            B = input_tokens.shape[0]
            num_tokens = input_tokens.shape[1] - num_pretokens
            group_idx = create_group_index(
                num_tokens,
                num_groups,
                grouping_strategy,
                device=input_tokens.device,
            )
            if group_idx.ndim == 1:
                group_idx = group_idx.unsqueeze(0).expand(B, -1)
            all_group_maps.append(group_idx)
        return all_group_maps

    def _ensure_group_maps_and_plans(self, context: Dict[str, Any], config: Any) -> None:
        all_input_tokens = context.get('all_x_backbones')
        if all_input_tokens is None:
            return

        all_group_maps = context.get('all_group_maps')
        valid_group_maps = (
            isinstance(all_group_maps, list)
            and len(all_group_maps) == len(all_input_tokens)
            and all(
                torch.is_tensor(group_map) and group_map.shape[0] == input_tokens.shape[0]
                for group_map, input_tokens in zip(all_group_maps, all_input_tokens)
            )
        )
        rebuilt_group_maps = False
        if not valid_group_maps:
            rebuilt_group_maps = self._build_all_group_maps(context, config)
            if rebuilt_group_maps is not None:
                context['all_group_maps'] = rebuilt_group_maps
                rebuilt_group_maps = True

        all_group_plans = context.get('all_group_plans')
        valid_group_plans = (
            not rebuilt_group_maps
            and isinstance(all_group_plans, list)
            and len(all_group_plans) == len(all_input_tokens)
            and all(isinstance(src_plans, dict) for src_plans in all_group_plans)
        )
        if not valid_group_plans:
            self.prepare_group_maps_and_dindices(None, context, config)

    def load_model(self, model_name: str, config: Any):
        print(f"[Executor] Loading Detector Model (MMap): {model_name}...")
        if self.model is not None:
             del self.model
             torch.cuda.empty_cache()

        from appcorr.models.dinov3.hub.detectors import dinov3_vit7b16_de
        self.model = dinov3_vit7b16_de(pretrained=False, weights="COCO2017", backbone_weights="LVD1689M")
        self.model.to(dtype=torch.bfloat16, device=self.device)
        
        try:
             backbone_path = "~/cjpark/weights/dinov3/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
             print(f"[Executor] Loading Backbone Weights from {backbone_path}")
             vit_backbone = self._get_vit_backbone()
             vit_backbone.load_state_dict(load_weight_mmap(backbone_path), strict=True)
             
             head_path = "~/cjpark/weights/dinov3/dinov3_vit7b16_coco_detr_head-b0235ff7.pth"
             print(f"[Executor] Loading Detector Head Weights from {head_path}")
             head_ckpt = load_weight_mmap(head_path)
             self.model.detector.load_state_dict(head_ckpt.get("model", head_ckpt), strict=False)
             del head_ckpt
        except Exception as e:
            print(f"!!! [Executor] Failed to load detector weights: {e}")
            raise e

        appcorr_kwargs = config.appcorr_kwargs.copy()
        appcorr_kwargs.pop('generated_from_client', None)
        if appcorr_kwargs:
            print(f"[Executor] Enabling AppCorr Mode: {appcorr_kwargs}")
            self._get_vit_backbone().set_appcorr_mode(**appcorr_kwargs)
        else:
            self._get_vit_backbone().set_appcorr_mode(enabled=False)
            
        self.model.eval()

    def _get_vit_backbone(self):
        inner = self.model.detector.backbone[0]
        return inner._backbone.backbone if hasattr(inner, "_backbone") else inner.backbone
        

    def preprocess(self, batch_data: Any, task: Task, context: Dict[str, Any], config: Any):
        if isinstance(batch_data, torch.Tensor):
            with torch.cuda.nvtx.range("Preprocess::ToDevice"):
                tensor = batch_data.to(device=self.device, non_blocking=True)
                if tensor.ndim != 4:
                    raise ValueError(f"Expected 4D tensor input, got {tensor.shape}")
                if tensor.shape[1] != 3:
                    if tensor.shape[-1] == 3:
                        tensor = tensor.permute(0, 3, 1, 2)
                    else:
                        raise ValueError(f"Expected channel dimension of size 3, got {tensor.shape}")
                tensor = tensor.float()
                if batch_data.dtype == torch.uint8:
                    tensor = tensor / 255.0
                tensor = (tensor - self.norm_mean) / self.norm_std
        else:
            with torch.cuda.nvtx.range("Preprocess::PinMemory"):
                tensor = torch.from_numpy(batch_data)
                if hasattr(tensor, 'pin_memory'):
                    tensor = tensor.pin_memory()

            with torch.cuda.nvtx.range("Preprocess::ToDevice"):
                tensor = tensor.to(device=self.device, non_blocking=True).permute(0, 3, 1, 2).float() / 255.0
                tensor = (tensor - self.norm_mean) / self.norm_std

        with torch.cuda.nvtx.range("Preprocess::Slicing"):
            idx = context.get('active_indices')
            if idx is not None and len(idx) < config.batch_size:
                tensor = tensor[idx]
            context['input_tensor'] = tensor


    def prepare_group_maps_and_dindices(self, task: Task, context: Dict[str, Any], config: Any):
        all_input_tokens = context.get('all_x_backbones')
        if all_input_tokens is None:
            return

        all_group_maps = context.get('all_group_maps')
        if all_group_maps is None:
            return

        if not isinstance(all_group_maps, list) or len(all_group_maps) != len(all_input_tokens):
            print("!!! [DetectorExecutor] all_group_maps must be a list aligned with all_x_backbones.")
            return

        all_cached_dindices = []
        all_group_plans = []
        num_pretokens = 1 + getattr(self._get_vit_backbone(), 'n_storage_tokens', 0)
        all_patch_residual_rms = context.get('all_patch_residual_rms')
        token_prune_enabled = getattr(self._get_vit_backbone(), 'appcorr_token_prune_enabled', False)
        token_prune_threshold = float(getattr(self._get_vit_backbone(), 'appcorr_token_prune_threshold', 0.0))
        token_prune_min_keep = int(getattr(self._get_vit_backbone(), 'appcorr_token_prune_min_keep', 1))

        for src_idx, (input_tokens, group_map) in enumerate(zip(all_input_tokens, all_group_maps)):
            if not torch.is_tensor(group_map):
                print(f"!!! [DetectorExecutor] all_group_maps[{src_idx}] must be a tensor.")
                return
            if group_map.ndim != 2:
                print(f"!!! [DetectorExecutor] all_group_maps[{src_idx}] must have shape [B, N].")
                return
            if group_map.shape[0] != input_tokens.shape[0]:
                print(
                    f"!!! [DetectorExecutor] Batch mismatch for all_group_maps[{src_idx}]: "
                    f"{group_map.shape[0]} != {input_tokens.shape[0]}"
                )
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
                    print(f"!!! [DetectorExecutor] Non-uniform group size detected at src={src_idx}, group={gid}.")
                    return

                patch_indices = spatial_indices + num_pretokens
                pre_indices = torch.arange(
                    num_pretokens,
                    device=input_tokens.device,
                    dtype=torch.long,
                ).unsqueeze(0).expand(B, -1)
                dindice = torch.cat([pre_indices, patch_indices], dim=1)
                src_cached_dindices[gid] = dindice
                patch_residual_rms = (
                    all_patch_residual_rms[src_idx]
                    if isinstance(all_patch_residual_rms, list) and src_idx < len(all_patch_residual_rms)
                    else None
                )
                src_group_plans[gid] = self._build_group_plan(
                    dindice,
                    spatial_indices,
                    patch_residual_rms,
                    num_pretokens,
                    token_prune_enabled,
                    token_prune_threshold,
                    token_prune_min_keep,
                )

            all_cached_dindices.append(src_cached_dindices)
            all_group_plans.append(src_group_plans)

        context['all_cached_dindices'] = all_cached_dindices
        context['all_group_plans'] = all_group_plans


    def prepare_tokens(self, task: Task, context: Dict[str, Any], config: Any):
        if 'input_tensor' not in context: return {}

        input_tensor = context['input_tensor']
        context['sizes_tensor'] = torch.tensor([s.shape[1:] for s in input_tensor], device=input_tensor[0].device)
        
        if not isinstance(input_tensor, NestedTensor):
            input_tensor = nested_tensor_from_tensor_list(input_tensor)

        detector = self.model.detector
        win_wrapper = detector.backbone[0]
        dino_bb = win_wrapper._backbone.backbone
        tensors, mask = input_tensor.tensors, input_tensor.mask
        context['input_tensor_mask'] = mask
        
        orig_h, orig_w = tensors.shape[2:]
        win_h = math.ceil((orig_h // win_wrapper._n_windows_h) / win_wrapper._patch_size) * win_wrapper._patch_size
        win_w = math.ceil((orig_w // win_wrapper._n_windows_w) / win_wrapper._patch_size) * win_wrapper._patch_size
        
        all_h = [win_h] * (win_wrapper._n_windows_h - 1) + [orig_h - win_h * (win_wrapper._n_windows_h - 1)]
        all_w = [win_w] * (win_wrapper._n_windows_w - 1) + [orig_w - win_w * (win_wrapper._n_windows_w - 1)]
        all_h_cum, all_w_cum = [0] + list(np.cumsum(all_h)), [0] + list(np.cumsum(all_w))

        context['window_patch_tensors'], context['window_patch_masks'], context['window_patch_tokens'] = [], [], []
        all_x_backbones, all_rope_sincos = [], []
        all_patch_residual_rms = []

        def _prep(x):
            xb, (H, W) = dino_bb.prepare_tokens_with_masks(x)
            rs = dino_bb.rope_embed(H=H, W=W) if dino_bb.rope_embed else None
            return xb, rs

        curr_hr_np = context.get('input_hr_np')
        prev_hr_np = context.get('prev_input_hr_np')
        curr_hr_tensor = None
        prev_hr_tensor = None
        if curr_hr_np is not None:
            curr_hr_tensor = torch.from_numpy(curr_hr_np).to(device=self.device, non_blocking=True).permute(0, 3, 1, 2).float()
        if prev_hr_np is not None:
            prev_hr_tensor = torch.from_numpy(prev_hr_np).to(device=self.device, non_blocking=True).permute(0, 3, 1, 2).float()
        if curr_hr_tensor is not None:
            active_indices = self._get_active_batch_indices(context, curr_hr_tensor.shape[0])
            curr_hr_tensor = curr_hr_tensor[active_indices]
            if prev_hr_tensor is not None:
                prev_hr_tensor = prev_hr_tensor[active_indices]

        for ih in range(win_wrapper._n_windows_h):
            row_t, row_m, row_x = [], [], []
            for iw in range(win_wrapper._n_windows_w):
                wt = v2.functional.crop(tensors, all_h_cum[ih], all_w_cum[iw], all_h[ih], all_w[iw])
                wm = v2.functional.crop(mask, all_h_cum[ih], all_w_cum[iw], all_h[ih], all_w[iw])
                x = NestedTensor(wt, wm).tensors
                
                row_t.append(wt); row_m.append(wm); row_x.append(x)
                xb, rs = _prep(x)
                all_x_backbones.append(xb); all_rope_sincos.append(rs)
                if curr_hr_tensor is not None:
                    curr_src = v2.functional.crop(curr_hr_tensor, all_h_cum[ih], all_w_cum[iw], all_h[ih], all_w[iw])
                    prev_src = None
                    if prev_hr_tensor is not None:
                        prev_src = v2.functional.crop(prev_hr_tensor, all_h_cum[ih], all_w_cum[iw], all_h[ih], all_w[iw])
                    all_patch_residual_rms.append(
                        self._compute_patch_residual_rms_for_source(curr_src, prev_src, dino_bb.patch_size)
                    )
                
            context['window_patch_tensors'].append(row_t)
            context['window_patch_masks'].append(row_m)
            context['window_patch_tokens'].append(row_x)

        context['global_x'] = NestedTensor(v2.functional.resize(tensors, size=(win_h, win_w)), mask).tensors
        g_xb, g_rs = _prep(context['global_x'])
        all_x_backbones.append(g_xb); all_rope_sincos.append(g_rs)
        if curr_hr_tensor is not None:
            curr_global = v2.functional.resize(curr_hr_tensor, size=(win_h, win_w))
            prev_global = v2.functional.resize(prev_hr_tensor, size=(win_h, win_w)) if prev_hr_tensor is not None else None
            all_patch_residual_rms.append(
                self._compute_patch_residual_rms_for_source(curr_global, prev_global, dino_bb.patch_size)
            )

        context['all_x_backbones'], context['all_rope_sincos'] = all_x_backbones, all_rope_sincos
        context['all_patch_residual_rms'] = all_patch_residual_rms
        context.pop('all_group_plans', None)
        context.pop('all_cached_dindices', None)

        self._ensure_group_maps_and_plans(context, config)

    def approx_forward(self, params: Dict[str, Any], context: Dict[str, Any], config: Any):
        layers = params.get('layers', (0, 40))
        start_l, end_l = layers[0], layers[1]

        dino_backbone = self.model.detector.backbone[0]._backbone
        blocks = dino_backbone.backbone.blocks

        if 'current_features' not in context:
            if 'all_x_backbones' in context:
                context['current_features'] = [x for x in context['all_x_backbones']]
            else:
                return

        all_current_features = context['current_features']
        all_input_tokens = context.get('all_x_backbones')
        all_rope_sincos = context.get('all_rope_sincos')
        all_cache_features = context.get('all_cache_features')
        all_outputs = context.get('all_outputs')

        self._ensure_group_maps_and_plans(context, config)
        all_group_plans = context.get('all_group_plans')
        appcorr_method = getattr(self._get_vit_backbone(), 'appcorr_method', None)

        if all_input_tokens is None or all_rope_sincos is None:
            return

        if all_cache_features is None:
            all_cache_features = [dict() for _ in range(len(all_input_tokens))]

        if all_outputs is None or start_l == 0:
            all_outputs = [[] for _ in range(len(all_input_tokens))]

        if start_l == 0:
            all_current_features = [x for x in all_input_tokens]

        new_current_features = []
        new_cache_features = []
        new_all_outputs = []

        if appcorr_method == 'partial_channel':
            if all_group_plans is None or not isinstance(all_group_plans, list):
                raise ValueError("partial_channel requires per-source group plans")
            if len(all_group_plans) != len(all_input_tokens):
                raise ValueError("all_group_plans must align with all_x_backbones")

        for src_idx, (x_feature, rope_sincos, cache, prev_outputs) in enumerate(
            zip(all_current_features, all_rope_sincos, all_cache_features, all_outputs)
        ):
            collected_outputs = [] if start_l == 0 else list(prev_outputs)
            group_plans = all_group_plans[src_idx] if appcorr_method == 'partial_channel' else None
            attn_cache_candidates = (
                {gid: plan.full_dindice for gid, plan in group_plans.items()}
                if group_plans is not None else None
            )

            for lidx in range(start_l, end_l):
                blk = blocks[lidx]

                with torch.no_grad():
                    x_feature, cache = blk.approx(
                        x_feature,
                        rope_sincos,
                        cache,
                        tag=f"src{src_idx}_layer{lidx}",
                        attn_cache_candidates=attn_cache_candidates,
                        group_plans=group_plans,
                        attn_col_alive_ratio=getattr(self._get_vit_backbone(), 'appcorr_attn_col_alive_ratio', 1.0),
                        debug=False
                    )

                if lidx in dino_backbone.layers_to_use:
                    collected_outputs.append(x_feature)

            new_current_features.append(x_feature)
            new_cache_features.append(cache)
            new_all_outputs.append(collected_outputs)

        context['current_features'] = new_current_features
        context['all_cache_features'] = new_cache_features
        context['all_outputs'] = new_all_outputs
        context['cache_feature'] = self._aggregate_cache_features(new_cache_features)

    def correct_forward(self, params: Dict[str, Any], context: Dict[str, Any], config: Any):
        layers = params.get('layers', (0, 40))
        group_id = params.get('group_id', 1)
        start_l, end_l = layers[0], layers[1]

        dino_backbone = self.model.detector.backbone[0]._backbone
        blocks = self._get_vit_backbone().blocks

        all_input_tokens = context.get('all_x_backbones')
        all_rope_sincos = context.get('all_rope_sincos')
        all_group_maps = context.get('all_group_maps')
        all_cache_features = context.get('all_cache_features')
        all_cached_dindices = context.get('all_cached_dindices')
        all_outputs = context.get('all_outputs')

        self._ensure_group_maps_and_plans(context, config)
        all_group_plans = context.get('all_group_plans')
        appcorr_method = getattr(self._get_vit_backbone(), 'appcorr_method', None)

        if all_input_tokens is None or all_rope_sincos is None:
            return
        if all_group_maps is None or all_cached_dindices is None:
            return
        if not isinstance(all_group_maps, list) or not isinstance(all_cached_dindices, list):
            return
        if len(all_group_maps) != len(all_input_tokens) or len(all_cached_dindices) != len(all_input_tokens):
            return

        if all_cache_features is None:
            all_cache_features = [dict() for _ in range(len(all_input_tokens))]

        if all_outputs is None or start_l == 0:   
            all_outputs = [[] for _ in range(len(all_input_tokens))]

        alive_ratio = getattr(dino_backbone, 'appcorr_cls_alive_ratio', 0.2)

        if appcorr_method == 'partial_channel':
            if all_group_plans is None or not isinstance(all_group_plans, list):
                raise ValueError("partial_channel requires per-source group plans")
            if len(all_group_plans) != len(all_input_tokens):
                raise ValueError("all_group_plans must align with all_x_backbones")

        new_current_features = []
        new_cache_features = []
        new_all_outputs = []

        for src_idx, (input_tokens, rope_sincos, group_map, cached_dindices, cache, prev_outputs) in enumerate(
            zip(all_input_tokens, all_rope_sincos, all_group_maps, all_cached_dindices, all_cache_features, all_outputs)
        ):
            if not torch.is_tensor(group_map) or not isinstance(cached_dindices, dict):
                return

            dindice = cached_dindices.get(group_id)
            group_plans = all_group_plans[src_idx] if appcorr_method == 'partial_channel' else None
            plan = group_plans.get(group_id) if group_plans is not None else None
            collected_outputs = [] if start_l == 0 else list(prev_outputs)

            if dindice is None:
                new_current_features.append(input_tokens)
                new_cache_features.append(cache)
                new_all_outputs.append(collected_outputs)
                continue

            if appcorr_method == 'partial_channel':
                if plan is None:
                    new_current_features.append(input_tokens)
                    new_cache_features.append(cache)
                    new_all_outputs.append(collected_outputs)
                    continue
                dindice = plan.pruned_dindice.to(device=self.device, non_blocking=True)
                plan.pruned_dindice = dindice
                fixed_query_state = plan.query_state
                attn_col_alive_ratio = getattr(self._get_vit_backbone(), 'appcorr_attn_col_alive_ratio', 1.0)
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

            x_temp = input_tokens
            for lidx in range(start_l, end_l):
                blk = blocks[lidx]
                with torch.no_grad():
                    if appcorr_method == 'partial_channel':
                        x_temp, cache = blk.correct(
                            x_temp,
                            dindice,
                            rope_sincos,
                            cache,
                            tag=f"src{src_idx}_layer{lidx}",
                            cls_alive_ratio=alive_ratio,
                            attn_col_alive_ratio=attn_col_alive_ratio,
                            fixed_query_state=fixed_query_state,
                            group_plan=plan,
                            attn_cache_key=group_id,
                            debug=False,
                        )
                    else:
                        x_temp, cache = blk.correct(
                            x_temp,
                            dindice,
                            rope_sincos,
                            cache,
                            tag=f"src{src_idx}_layer{lidx}",
                            cls_alive_ratio=alive_ratio,
                            debug=False,
                        )

                if lidx in dino_backbone.layers_to_use:   
                    collected_outputs.append(x_temp)

            new_current_features.append(x_temp)
            new_cache_features.append(cache)
            new_all_outputs.append(collected_outputs)   

        context['current_features'] = new_current_features
        context['all_cache_features'] = new_cache_features
        context['all_outputs'] = new_all_outputs
        context['cache_feature'] = self._aggregate_cache_features(new_cache_features)

    def _process_outputs(self, outputs, x, dt_backbone):
        B, _, h, w = x.shape
        ps = dt_backbone.backbone.patch_size
        expected = (h // ps) * (w // ps)
        special = dt_backbone.backbone.n_storage_tokens + 1

        xs = []
        for i, out in enumerate(outputs):
            n = out.shape[1]

            if n == expected:
                out = dt_backbone.backbone.norm(out)
            elif n == expected + special:
                if dt_backbone.backbone.untie_cls_and_patch_norms:
                    out = torch.cat([
                        dt_backbone.backbone.cls_norm(out[:, :special]),
                        dt_backbone.backbone.norm(out[:, special:])
                    ], dim=1)
                else:
                    out = dt_backbone.backbone.norm(out)
                out = out[:, special:]
            else:
                raise RuntimeError(
                    f"Unexpected token count at layer {i}: got {n}, "
                    f"expected {expected} or {expected + special}"
                )

            xs.append(
                out.reshape(B, h // ps, w // ps, -1).permute(0, 3, 1, 2).contiguous()
            )

        if dt_backbone.use_layernorm:
            xs = [ln(x).contiguous() for ln, x in zip(dt_backbone.layer_norms, xs)]

        return torch.cat(xs, dim=1)

    def _run_transformer_and_postprocess(self, detector, features, sizes_tensor):
        pos = [detector.backbone[1][idx](x).to(x.tensors.dtype) for idx, x in enumerate(features)]
        srcs, masks = [], []
        for layer, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(detector.input_proj[layer](src))
            masks.append(mask)

        query_embeds = detector.query_embed.weight[:detector.num_queries] if not detector.two_stage or detector.mixed_selection else None
        
        self_attn_mask = torch.zeros([detector.num_queries, detector.num_queries], dtype=bool, device=src.device)
        self_attn_mask[detector.num_queries_one2one :, : detector.num_queries_one2one] = True
        self_attn_mask[: detector.num_queries_one2one, detector.num_queries_one2one :] = True

        (hs, init_ref, inter_refs, enc_out_cls, enc_out_coord, enc_out_delta, prop, max_shape) = detector.transformer(
            srcs, masks, pos, query_embeds, self_attn_mask
        )

        o_cls_1, o_coords_1, o_cls_m, o_coords_m = [], [], [], []
        o_coords_old_1, o_deltas_1, o_coords_old_m, o_deltas_m = [], [], [], []
        n1 = detector.num_queries_one2one

        for lvl in range(hs.shape[0]):
            ref = init_ref if lvl == 0 else inter_refs[lvl - 1]
            out_class = detector.class_embed[lvl](hs[lvl])
            tmp = detector.bbox_embed[lvl](hs[lvl])
            out_coord = box_ops.box_xyxy_to_cxcywh(box_ops.delta2bbox(ref, tmp, max_shape))

            o_cls_1.append(out_class[:, :n1]); o_cls_m.append(out_class[:, n1:])
            o_coords_1.append(out_coord[:, :n1]); o_coords_m.append(out_coord[:, n1:])
            o_coords_old_1.append(ref[:, :n1]); o_coords_old_m.append(ref[:, n1:])
            o_deltas_1.append(tmp[:, :n1]); o_deltas_m.append(tmp[:, n1:])

        out = {
            "pred_logits": torch.stack(o_cls_1)[-1], "pred_boxes": torch.stack(o_coords_1)[-1],
            "pred_logits_one2many": torch.stack(o_cls_m)[-1], "pred_boxes_one2many": torch.stack(o_coords_m)[-1],
            "pred_boxes_old": o_coords_old_1[-1], "pred_deltas": o_deltas_1[-1],
            "pred_boxes_old_one2many": o_coords_old_m[-1], "pred_deltas_one2many": o_deltas_m[-1],
        }

        if detector.aux_loss:
            out["aux_outputs"] = detector._set_aux_loss(o_cls_1, o_coords_1, o_coords_old_1, o_deltas_1)
            out["aux_outputs_one2many"] = detector._set_aux_loss(o_cls_m, o_coords_m, o_coords_old_m, o_deltas_m)

        if detector.two_stage:
            out["enc_outputs"] = {
                "pred_logits": enc_out_cls, "pred_boxes": enc_out_coord, "pred_boxes_old": prop, "pred_deltas": enc_out_delta
            }
        
        return self.model.postprocessor(out, target_sizes=sizes_tensor, original_target_sizes=sizes_tensor)

    def head_inference(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[str, Any]:
        detector = self.model.detector
        win_wrapper = detector.backbone[0]
        dino_bb = win_wrapper._backbone

        all_outputs = context.get('all_outputs')
        if all_outputs is None:
            raise RuntimeError("Missing context['all_outputs'] for head_inference().")

        global_output, window_outputs = all_outputs[-1], all_outputs[:-1]

        win_patch_tensors, win_patch_masks = context.get('window_patch_tensors'), context.get('window_patch_masks')
        win_patch_tokens = context.get('window_patch_tokens')

        win_patch_features = []
        idx = 0
        for ih in range(win_wrapper._n_windows_h):
            row_features = []
            for iw in range(win_wrapper._n_windows_w):
                x = self._process_outputs(window_outputs[idx], win_patch_tokens[ih][iw], dino_bb)
                m = NestedTensor(win_patch_tensors[ih][iw], win_patch_masks[ih][iw]).mask
                mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
                row_features.append(NestedTensor(x, mask))
                idx += 1
            win_patch_features.append(row_features)

        window_features = torch.cat([torch.cat([el.tensors for el in row], dim=-1) for row in win_patch_features], dim=-2)
        global_x_tensor = self._process_outputs(global_output, context.get('global_x'), dino_bb)
        
        input_tensor_mask = context.get('input_tensor_mask')
        
        concat_tensors = torch.cat([v2.functional.resize(global_x_tensor, size=window_features.shape[-2:]), window_features], dim=1)
        global_mask = F.interpolate(input_tensor_mask[None].float(), size=concat_tensors.shape[-2:]).to(torch.bool)[0]
        features = [NestedTensor(tensors=concat_tensors, mask=global_mask)]

        context['det_outputs'] = self._run_transformer_and_postprocess(detector, features, context.get('sizes_tensor'))
        return {}
        

    @torch.inference_mode()
    def full_inference(self, task: Task, context: Dict[str, Any], config: Any):
        inp = context.get('input_tensor')
        if inp is not None:
            context['det_outputs'] = self.model(inp)
            context['det_output'] = context['det_outputs']


    def get_final_results(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[int, Any]:
        results = {}
        outputs = context.get('det_outputs')
        if outputs is None:
            outputs = context.get('det_output')
        if outputs is not None and 'active_indices' in context:
            indices = context.get('active_indices').cpu().numpy()
            for i, orig_idx in enumerate(indices):
                if i < len(outputs):
                    pred = outputs[i]
                    results[int(orig_idx)] = {
                        'scores': pred['scores'].float().cpu().tolist(),
                        'labels': pred['labels'].long().cpu().tolist(),
                        'boxes': pred['boxes'].float().cpu().tolist()
                    }
        return results

    def decide_exit(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[str, Any]:
        return {'num_exits': 0}

    def _full_inference_analyzed(self, input_tensor: list[torch.Tensor]):
        detector = self.model.detector

        sizes_tensor = torch.tensor([sample.shape[1:] for sample in input_tensor], device=input_tensor[0].device)  # N * [3, H, W]
        
        if not isinstance(input_tensor, NestedTensor):
            input_tensor = nested_tensor_from_tensor_list(input_tensor)
            
        windows_wrapper = detector.backbone[0]
        tensors = input_tensor.tensors
        original_h, original_w = tensors.shape[2], tensors.shape[3]
        # Get height and width of the windows, such that it is a multiple of the patch size
        window_h = math.ceil((original_h // windows_wrapper._n_windows_h) / windows_wrapper._patch_size) * windows_wrapper._patch_size
        window_w = math.ceil((original_w // windows_wrapper._n_windows_w) / windows_wrapper._patch_size) * windows_wrapper._patch_size
        all_h = [window_h] * (windows_wrapper._n_windows_h - 1) + [original_h - window_h * (windows_wrapper._n_windows_h - 1)]
        all_w = [window_w] * (windows_wrapper._n_windows_w - 1) + [original_w - window_w * (windows_wrapper._n_windows_w - 1)]
        all_h_cumsum = [0] + list(np.cumsum(all_h))
        all_w_cumsum = [0] + list(np.cumsum(all_w))
        window_patch_features = [[0 for _ in range(windows_wrapper._n_windows_w)] for _ in range(windows_wrapper._n_windows_h)]

        for ih in range(windows_wrapper._n_windows_h):
            for iw in range(windows_wrapper._n_windows_w):
                window_tensor = v2.functional.crop(
                    tensors, top=all_h_cumsum[ih], left=all_w_cumsum[iw], height=all_h[ih], width=all_w[iw]
                )
                window_mask = v2.functional.crop(
                    input_tensor.mask, top=all_h_cumsum[ih], left=all_w_cumsum[iw], height=all_h[ih], width=all_w[iw]
                )

                dino_backbone = windows_wrapper._backbone
                x = NestedTensor(tensors=window_tensor, mask=window_mask).tensors
                n = dino_backbone.layers_to_use

                x_backbone = x
                x_backbone, (H, W) = dino_backbone.backbone.prepare_tokens_with_masks(x_backbone)
                # If n is an int, take the n last blocks. If it's a list, take them
                output, total_block_len = [], len(dino_backbone.backbone.blocks)
                blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
                for i, blk in enumerate(dino_backbone.backbone.blocks):
                    if dino_backbone.backbone.rope_embed is not None:
                        rope_sincos = dino_backbone.backbone.rope_embed(H=H, W=W)
                    else:
                        rope_sincos = None
                    x_backbone = blk(x_backbone, rope_sincos)
                    if i in blocks_to_take:
                        output.append(x_backbone)
                assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
                outputs = output

                outputs_normed = []
                for out in outputs:
                    if dino_backbone.backbone.untie_cls_and_patch_norms:
                        x_norm_cls_reg = dino_backbone.backbone.cls_norm(out[:, : dino_backbone.backbone.n_storage_tokens + 1])
                        x_norm_patch = dino_backbone.backbone.norm(out[:, dino_backbone.backbone.n_storage_tokens + 1 :])
                        outputs_normed.append(torch.cat((x_norm_cls_reg, x_norm_patch), dim=1))
                    else:
                        outputs_normed.append(dino_backbone.backbone.norm(out))
                outputs = [out[:, dino_backbone.backbone.n_storage_tokens + 1 :] for out in outputs_normed]

                B, _, h, w = x.shape
                xs = [
                    out.reshape(B, h // dino_backbone.backbone.patch_size, w // dino_backbone.backbone.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                    for out in outputs
                ]

                if dino_backbone.use_layernorm:
                    xs = [ln(x).contiguous() for ln, x in zip(dino_backbone.layer_norms, xs)]

                x = torch.cat(xs, axis=1)

                m = NestedTensor(tensors=window_tensor, mask=window_mask).mask
                assert m is not None
                mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
                window_patch_features[ih][iw] = NestedTensor(x, mask)

        window_tensors = torch.cat(
            [
                torch.cat([el.tensors for el in window_patch_features[ih]], dim=-1)  # type: ignore
                for ih in range(len(window_patch_features))
            ],
            dim=-2,
        )
        # Also compute the global features in a "preferential" setting, of lower resolution
        resized_global_tensor = v2.functional.resize(tensors, size=(window_h, window_w))
        global_features = windows_wrapper._backbone(
            NestedTensor(tensors=resized_global_tensor, mask=input_tensor.mask)
        )  # mask is not used

        concat_tensors = torch.cat(
            [v2.functional.resize(global_features[0].tensors, size=window_tensors.shape[-2:]), window_tensors], dim=1
        )
        global_mask = F.interpolate(input_tensor.mask[None].float(), size=concat_tensors.shape[-2:]).to(torch.bool)[0]
        features = [NestedTensor(tensors=concat_tensors, mask=global_mask)]

        pos = [detector.backbone[1][idx](x).to(x.tensors.dtype) for idx, x in enumerate(features)]

        srcs = []
        masks = []
        for layer, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(detector.input_proj[layer](src))
            masks.append(mask)
            assert mask is not None

        query_embeds = None
        if not detector.two_stage or detector.mixed_selection:
            query_embeds = detector.query_embed.weight[0 : detector.num_queries, :]

        # make attn mask
        """ attention mask to prevent information leakage
        """
        self_attn_mask = torch.zeros(
            [
                detector.num_queries,
                detector.num_queries,
            ],
            dtype=bool,
            device=src.device,
        )
        self_attn_mask[
            detector.num_queries_one2one :,
            0 : detector.num_queries_one2one,
        ] = True
        self_attn_mask[
            0 : detector.num_queries_one2one,
            detector.num_queries_one2one :,
        ] = True

        (
            hs,
            init_reference,
            inter_references,
            enc_outputs_class,
            enc_outputs_coord_unact,
            enc_outputs_delta,
            output_proposals,
            max_shape,
        ) = detector.transformer(srcs, masks, pos, query_embeds, self_attn_mask)

        outputs_classes_one2one = []
        outputs_coords_one2one = []
        outputs_classes_one2many = []
        outputs_coords_one2many = []

        outputs_coords_old_one2one = []
        outputs_deltas_one2one = []
        outputs_coords_old_one2many = []
        outputs_deltas_one2many = []

        for lvl in range(hs.shape[0]):
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            outputs_class = detector.class_embed[lvl](hs[lvl])
            tmp = detector.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                outputs_coord = box_ops.box_xyxy_to_cxcywh(box_ops.delta2bbox(reference, tmp, max_shape))
            else:
                raise NotImplementedError

            outputs_classes_one2one.append(outputs_class[:, 0 : detector.num_queries_one2one])
            outputs_classes_one2many.append(outputs_class[:, detector.num_queries_one2one :])

            outputs_coords_one2one.append(outputs_coord[:, 0 : detector.num_queries_one2one])
            outputs_coords_one2many.append(outputs_coord[:, detector.num_queries_one2one :])

            outputs_coords_old_one2one.append(reference[:, : detector.num_queries_one2one])
            outputs_coords_old_one2many.append(reference[:, detector.num_queries_one2one :])
            outputs_deltas_one2one.append(tmp[:, : detector.num_queries_one2one])
            outputs_deltas_one2many.append(tmp[:, detector.num_queries_one2one :])

        outputs_classes_one2one = torch.stack(outputs_classes_one2one)
        outputs_coords_one2one = torch.stack(outputs_coords_one2one)

        outputs_classes_one2many = torch.stack(outputs_classes_one2many)
        outputs_coords_one2many = torch.stack(outputs_coords_one2many)

        out = {
            "pred_logits": outputs_classes_one2one[-1],
            "pred_boxes": outputs_coords_one2one[-1],
            "pred_logits_one2many": outputs_classes_one2many[-1],
            "pred_boxes_one2many": outputs_coords_one2many[-1],
            "pred_boxes_old": outputs_coords_old_one2one[-1],
            "pred_deltas": outputs_deltas_one2one[-1],
            "pred_boxes_old_one2many": outputs_coords_old_one2many[-1],
            "pred_deltas_one2many": outputs_deltas_one2many[-1],
        }

        if detector.aux_loss:
            out["aux_outputs"] = detector._set_aux_loss(
                outputs_classes_one2one, outputs_coords_one2one, outputs_coords_old_one2one, outputs_deltas_one2one
            )
            out["aux_outputs_one2many"] = detector._set_aux_loss(
                outputs_classes_one2many, outputs_coords_one2many, outputs_coords_old_one2many, outputs_deltas_one2many
            )

        if detector.two_stage:
            out["enc_outputs"] = {
                "pred_logits": enc_outputs_class,
                "pred_boxes": enc_outputs_coord_unact,
                "pred_boxes_old": output_proposals,
                "pred_deltas": enc_outputs_delta,
            }
        
        outputs = self.model.postprocessor(out, target_sizes=sizes_tensor, original_target_sizes=sizes_tensor)
        return outputs
        
