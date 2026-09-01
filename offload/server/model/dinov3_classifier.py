from dataclasses import dataclass
from typing import Any, Dict
import torch
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

class DINOv3ClassifierExecutor(ModelExecutor):
    def __init__(self, device: torch.device):
        super().__init__(device)
        # ImageNet normalization constants
        self.normalize_avg = np.array([0.485, 0.456, 0.406])
        self.normalize_std = np.array([0.229, 0.224, 0.225])
        self.norm_mean = torch.tensor(self.normalize_avg).view(1, 3, 1, 1).to(self.device).float()
        self.norm_std = torch.tensor(self.normalize_std).view(1, 3, 1, 1).to(self.device).float()

    def _get_active_batch_indices(self, context: Dict[str, Any], full_batch_size: int) -> torch.Tensor:
        if 'active_indices' in context and len(context['active_indices']) < full_batch_size:
            return context['active_indices'].to(device=self.device, dtype=torch.long)
        return torch.arange(full_batch_size, device=self.device, dtype=torch.long)

    @staticmethod
    def _uses_l2l1l0_transmission(config: Any) -> bool:
        return (
            getattr(config, "transmission_policy_name", None)
            == "L2L1L0ProgressiveLaplacian"
        )

    @classmethod
    def _model_spatial_indices_for_patch(
        cls,
        patch,
        config: Any,
    ) -> tuple[int, ...]:
        if not cls._uses_l2l1l0_transmission(config):
            return (int(patch.spatial_idx),)

        levels = list(config.transmission_kwargs.get("pyramid_levels", [2, 1, 0]))
        target_level = min(levels)
        patch_level = int(patch.res_level)
        if patch_level < target_level:
            raise ValueError(
                f"Patch level {patch_level} is below target level {target_level}"
            )

        patch_h, patch_w = config.patch_size
        image_h, image_w = config.image_shape[:2]
        coarse_h = image_h // (2 ** patch_level)
        coarse_w = image_w // (2 ** patch_level)
        coarse_grid_h = coarse_h // patch_h
        coarse_grid_w = coarse_w // patch_w
        spatial_idx = int(patch.spatial_idx)
        if spatial_idx < 0 or spatial_idx >= coarse_grid_h * coarse_grid_w:
            raise IndexError(
                f"Patch spatial_idx={spatial_idx} is outside level-{patch_level} "
                f"grid {coarse_grid_h}x{coarse_grid_w}"
            )

        scale = 2 ** (patch_level - target_level)
        target_w = image_w // (2 ** target_level)
        target_grid_w = target_w // patch_w
        coarse_row, coarse_col = divmod(spatial_idx, coarse_grid_w)
        return tuple(
            fine_row * target_grid_w + fine_col
            for fine_row in range(
                coarse_row * scale,
                (coarse_row + 1) * scale,
            )
            for fine_col in range(
                coarse_col * scale,
                (coarse_col + 1) * scale,
            )
        )

    def _expand_payload_token_records(
        self,
        task: Task,
        context: Dict[str, Any],
        config: Any,
    ) -> tuple[list[int], list[int], list[int]]:
        if (
            "active_indices" in context
            and len(context["active_indices"]) < config.batch_size
        ):
            active_list = context["active_indices"].tolist()
            image_idx_map = {
                original_idx: local_idx
                for local_idx, original_idx in enumerate(active_list)
            }
        else:
            image_idx_map = None

        batch_indices = []
        spatial_indices = []
        group_ids = []
        for patch in task.payload:
            if image_idx_map is not None:
                if patch.image_idx not in image_idx_map:
                    continue
                batch_idx = image_idx_map[patch.image_idx]
            else:
                batch_idx = int(patch.image_idx)

            for spatial_idx in self._model_spatial_indices_for_patch(
                patch,
                config,
            ):
                batch_indices.append(batch_idx)
                spatial_indices.append(spatial_idx)
                group_ids.append(int(patch.group_id))
        return batch_indices, spatial_indices, group_ids

    def _build_mobile_pscore_hint_map(
        self,
        task: Task,
        context: Dict[str, Any],
        config: Any,
        batch_size: int,
        num_patches: int,
    ) -> torch.Tensor:
        hint_map = torch.zeros((batch_size, num_patches), device=self.device, dtype=torch.float32)
        if self._uses_l2l1l0_transmission(config):
            if (
                "active_indices" in context
                and len(context["active_indices"]) < config.batch_size
            ):
                active_list = context["active_indices"].tolist()
                image_idx_map = {
                    original_idx: local_idx
                    for local_idx, original_idx in enumerate(active_list)
                }
            else:
                image_idx_map = None

            batch_indices = []
            spatial_indices = []
            hint_values = []
            for patch in task.payload:
                if image_idx_map is not None:
                    if patch.image_idx not in image_idx_map:
                        continue
                    batch_idx = image_idx_map[patch.image_idx]
                else:
                    batch_idx = int(patch.image_idx)
                patch_spatial_indices = self._model_spatial_indices_for_patch(
                    patch,
                    config,
                )
                batch_indices.extend([batch_idx] * len(patch_spatial_indices))
                spatial_indices.extend(patch_spatial_indices)
                hint_values.extend(
                    [float(getattr(patch, "pscore_hint", 0.0))]
                    * len(patch_spatial_indices)
                )
            if spatial_indices:
                batch_tensor = torch.tensor(
                    batch_indices,
                    device=self.device,
                    dtype=torch.long,
                )
                spatial_tensor = torch.tensor(
                    spatial_indices,
                    device=self.device,
                    dtype=torch.long,
                )
                hint_tensor = torch.tensor(
                    hint_values,
                    device=self.device,
                    dtype=torch.float32,
                )
                hint_map[batch_tensor, spatial_tensor] = hint_tensor
            return self._normalize_patch_score_map(hint_map)

        target_res_level = min(config.transmission_kwargs.get('pyramid_levels', [0]))

        if 'active_indices' in context and len(context['active_indices']) < config.batch_size:
            active_list = context['active_indices'].tolist()
            idx_map = {orig: local for local, orig in enumerate(active_list)}
            valid_payload = [
                p for p in task.payload
                if p.image_idx in idx_map and p.res_level == target_res_level and 0 <= p.spatial_idx < num_patches
            ]
            if not valid_payload:
                return hint_map
            b_list = [idx_map[p.image_idx] for p in valid_payload]
        else:
            valid_payload = [
                p for p in task.payload
                if p.res_level == target_res_level and 0 <= p.spatial_idx < num_patches
            ]
            if not valid_payload:
                return hint_map
            b_list = [p.image_idx for p in valid_payload]

        s_list = [p.spatial_idx for p in valid_payload]
        h_list = [float(getattr(p, 'pscore_hint', 0.0)) for p in valid_payload]
        b_t = torch.tensor(b_list, device=self.device, dtype=torch.long)
        s_t = torch.tensor(s_list, device=self.device, dtype=torch.long)
        h_t = torch.tensor(h_list, device=self.device, dtype=torch.float32)
        hint_map[b_t, s_t] = h_t
        return self._normalize_patch_score_map(hint_map)

    def _compute_patch_residual_rms(
        self,
        context: Dict[str, Any],
        config: Any,
        active_indices: torch.Tensor,
    ) -> torch.Tensor | None:
        curr_np = context.get('input_hr_np')
        prev_np = context.get('prev_input_hr_np')
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

        # Residual RMS per image patch: [B, gh * gw]
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
            ) = (
                self._apply_image_residual_token_pruning(
                    dindice,
                    spatial_indices,
                    patch_residual_rms,
                    token_prune_threshold,
                    token_prune_min_keep,
                )
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
    def _get_group_plans(context: Dict[str, Any]) -> Dict[int, GroupCorrectionPlan]:
        return context.setdefault('group_plans', {})

    def load_model(self, model_name: str, config: Any):
        print(f"[Executor] Loading Model (MMap): {model_name}...")
        if self.model is not None:
             del self.model
             torch.cuda.empty_cache()

        from appcorr.models.dinov3.hub.classifiers import dinov3_vit7b16_lc

        # Init empty model (random weights)
        self.model = dinov3_vit7b16_lc(
            pretrained=False,
            # We must provide some valid enum or string, but it won't be used for loading
            weights="IMAGENET1K", 
            backbone_weights="LVD1689M",
        )

        # Move to target device/dtype BEFORE loading weights to save RAM
        self.model.to(dtype=torch.bfloat16)
        self.model.to(self.device)
        
        try:
            # Load Backbone Weights
            backbone_path = "~/cjpark/weights/dinov3/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
            print(f"[Executor] Loading Backbone Weights from {backbone_path}")
            backbone_state = load_weight_mmap(backbone_path)
            self.model.backbone.load_state_dict(backbone_state, strict=True)
            del backbone_state
            
            # Load Head Weights
            head_path = "~/cjpark/weights/dinov3/dinov3_vit7b16_imagenet1k_linear_head-90d8ed92.pth"
            print(f"[Executor] Loading Head Weights from {head_path}")
            head_state = load_weight_mmap(head_path)
            self.model.linear_head.load_state_dict(head_state, strict=True)
            del head_state
            
        except Exception as e:
            print(f"!!! [Executor] Failed to load weights: {e}")
            raise e

        self.model.eval()
        self.configure_dinov3_approx_precision(self.model.backbone, config)
        self.configure_dinov3_correct_precision(self.model.backbone, config)

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
            # Preprocess
            with torch.cuda.nvtx.range("Preprocess::PinMemory"):
                tensor = torch.from_numpy(batch_data)
                if hasattr(tensor, 'pin_memory'):
                    tensor = tensor.pin_memory()
                    
            with torch.cuda.nvtx.range("Preprocess::ToDevice"):
                tensor = tensor.to(device=self.device, non_blocking=True).permute(0, 3, 1, 2).float()
                tensor = tensor / 255.0
                tensor = (tensor - self.norm_mean) / self.norm_std

        # Sliced Update Handling
        with torch.cuda.nvtx.range("Preprocess::Slicing"):
            # Early-exit may shrink the active batch. We slice the decoded image batch first so
            # every downstream tensor uses the same compact batch indexing.
            if 'active_indices' in context and len(context['active_indices']) < config.batch_size:
                active_indices = context['active_indices']
                tensor = tensor[active_indices]
            context['input_tensor'] = tensor

        with torch.cuda.nvtx.range("Preprocess::GroupMap"):
            # Group IDs arrive incrementally from the client. We keep a per-request map from
            # patch index to transmission group so correction plans can be rebuilt after exits.
            self._get_group_plans(context)

            # Initialize or Retrieve group_map: [CurrentBatch, Max_Tokens]
            if 'group_map' not in context:
                H, W = config.image_shape[:2]
                if isinstance(config.patch_size, int):
                    ph = pw = config.patch_size
                else:
                    ph, pw = config.patch_size
                num_patches = (H // ph) * (W // pw)
                # Init with -1 (no group)
                curr_B = tensor.shape[0]
                context['group_map'] = torch.full((curr_B, num_patches), -1, device=self.device, dtype=torch.long)
            
            group_map = context['group_map']
            
            b_list, s_list, g_list = self._expand_payload_token_records(
                task,
                context,
                config,
            )
            if b_list:
                b_t = torch.tensor(b_list, device=self.device, dtype=torch.long)
                s_t = torch.tensor(s_list, device=self.device, dtype=torch.long)
                g_t = torch.tensor(g_list, device=self.device, dtype=torch.long)
                group_map[b_t, s_t] = g_t

            num_patches = group_map.shape[1]
            context['mobile_pscore_hint_map'] = self._build_mobile_pscore_hint_map(
                task,
                context,
                config,
                tensor.shape[0],
                num_patches,
            )
                    
        with torch.cuda.nvtx.range("Preprocess::Dindices"):
            # Build one correction plan per group. Each plan owns:
            # 1) the full token set for approx sparse-cache capture,
            # 2) the pruned token set for correction,
            # 3) the packed query metadata reused by every correction layer.
            num_pretokens = 1 + self.model.backbone.n_storage_tokens
            B = group_map.shape[0]
            appcorr_options = normalize_appcorr_kwargs(config.appcorr_kwargs, config.transmission_kwargs)
            appcorr_method = appcorr_options["method"]
            grouping_strategy = config.transmission_kwargs.get('grouping_strategy', 'uniform_diff')
            num_groups = config.transmission_kwargs.get('num_groups', 4)
            active_indices = self._get_active_batch_indices(context, config.batch_size)
            patch_residual_rms = self._compute_patch_residual_rms(context, config, active_indices)
            token_prune_enabled = appcorr_options["token_prune_enabled"]
            token_prune_threshold = appcorr_options["token_prune_threshold"]
            token_prune_min_keep = appcorr_options["token_prune_min_keep"]
            group_plans = self._get_group_plans(context)
            group_plans.clear()

            if appcorr_method == 'partial_channel' and grouping_strategy in {'grid', 'uniform', 'geometric'}:
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
                    kept_total = int(plan.kept_patch_count.sum().item())
                    full_total = int(plan.full_patch_count.sum().item())
                    keep_ratio = (kept_total / full_total) if full_total > 0 else 1.0
                    kept_mass_total = float(plan.kept_residual_mass.sum().item())
                    full_mass_total = float(plan.full_residual_mass.sum().item())
                    mass_ratio = (kept_mass_total / full_mass_total) if full_mass_total > 0 else 1.0
                    rms_mean = (
                        float(patch_residual_rms.gather(1, spatial_indices).mean().item())
                        if patch_residual_rms is not None and spatial_indices.numel() > 0
                        else -1.0
                    )
                    # print(
                    #     f"[TokenPrune][gid={gid}] kept={kept_total}/{full_total} "
                    #     f"ratio={keep_ratio:.4f} thr={token_prune_threshold:.3f} "
                    #     f"min_keep={token_prune_min_keep} rms_mean={rms_mean:.4f} "
                    #     f"res_mass={kept_mass_total:.2f}/{full_mass_total:.2f} "
                    #     f"mass_ratio={mass_ratio:.4f}"
                    # )

            elif b_list:  # Fallback for payload-dependent grouping.
                involved_groups = set(g_list)
                for gid in involved_groups:
                    mask = (g_t == gid)
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
                    kept_total = int(plan.kept_patch_count.sum().item())
                    full_total = int(plan.full_patch_count.sum().item())
                    keep_ratio = (kept_total / full_total) if full_total > 0 else 1.0
                    kept_mass_total = float(plan.kept_residual_mass.sum().item())
                    full_mass_total = float(plan.full_residual_mass.sum().item())
                    mass_ratio = (kept_mass_total / full_mass_total) if full_mass_total > 0 else 1.0
                    rms_mean = (
                        float(patch_residual_rms.gather(1, spatial_indices).mean().item())
                        if patch_residual_rms is not None and spatial_indices.numel() > 0
                        else -1.0
                    )
    

    def prepare_tokens(self, task: Task, context: Dict[str, Any], config: Any):
        if 'input_tensor' not in context: return

        tensor = context['input_tensor']
        backbone = self.model.backbone
            
        if hasattr(backbone, 'prepare_tokens_with_masks'):
            t2_x, hw_tuple = backbone.prepare_tokens_with_masks(tensor, None)
            context['input_tokens'] = t2_x
            context['hw_tuple'] = hw_tuple
            
            if 'current_feature' not in context:
                context['current_feature'] = t2_x

            if self.model.backbone.rope_embed is not None and 'rope_sincos' not in context:
                rope_sincos = self.model.backbone.rope_embed(H=hw_tuple[0], W=hw_tuple[1])
                context['rope_sincos'] = rope_sincos

    def approx_forward(self, params: Dict[str, Any], context: Dict[str, Any], config: Any):
        layers = params.get('layers', range(0, 40))
        cache_mode = str(params.get('cache_mode', 'correction'))
        if cache_mode not in {'correction', 'none'}:
            raise ValueError(
                f"Unknown approx cache_mode '{cache_mode}'. "
                "Available values: correction, none"
            )
        # Ensure context has required items
        if 'current_feature' not in context:
             if 'input_tokens' in context:
                 context['current_feature'] = context['input_tokens']
             else:
                 # Should not happen if flow is correct
                 return

        x_feature = context['current_feature']
        rope_sincos = context.get('rope_sincos')
        cache = context.get('cache_feature', {})
        
        start_l, end_l = layers[0], layers[1]
        appcorr_options = normalize_appcorr_kwargs(config.appcorr_kwargs, config.transmission_kwargs)
        appcorr_method = appcorr_options["method"]
        
        if start_l == 0:
            x_feature = context['input_tokens']

        if cache_mode == 'none':
            for lidx in range(start_l, end_l):
                blk = self.model.backbone.blocks[lidx]
                x_feature = blk(x_feature, rope_sincos)
            context['current_feature'] = x_feature
            return {
                'cache_mode': cache_mode,
                'layer_count': end_l - start_l,
            }

        group_plans = self._get_group_plans(context)
        attn_cache_candidates = {
            gid: plan.full_dindice
            for gid, plan in group_plans.items()
        } if appcorr_method == 'partial_channel' else None

        self.begin_dinov3_approx_event()
        for lidx in range(start_l, end_l):
            x_feature, cache = self.run_dinov3_approx_block(
                lidx, x_feature, rope_sincos, cache, tag=f"layer{lidx}",
                appcorr_method=appcorr_method,
                attn_cache_candidates=attn_cache_candidates,
                group_plans=group_plans if appcorr_method == 'partial_channel' else None,
                server_pscore=appcorr_options["server_pscore"],
                server_pscore_weight=appcorr_options["server_pscore_weight"],
                attn_col_alive_ratio=appcorr_options["attn_col_alive_ratio"],
                debug=False
            )
        
        context['current_feature'] = x_feature
        context['cache_feature'] = cache
        return self.dinov3_approx_event_metadata()

    def correct_forward(self, params: Dict[str, Any], context: Dict[str, Any], config: Any):
        layers = params.get('layers', range(0, 40))
        group_id = params.get('group_id', 1)
        start_l, end_l = layers[0], layers[1]

        group_plans = self._get_group_plans(context)
        plan = group_plans.get(group_id)
        if plan is None:
            print(f"!!! [Executor] Missing group correction plan for group {group_id}.")
            return

        dindice = plan.pruned_dindice.to(device=self.device, non_blocking=True)
        plan.pruned_dindice = dindice

        cache = context.get('cache_feature', {})
        rope_sincos = context.get('rope_sincos')
        
        # logic from worker.py: x_temp starts from input_tokens
        x_temp = context.get('input_tokens')
        
        appcorr_options = normalize_appcorr_kwargs(config.appcorr_kwargs, config.transmission_kwargs)
        token_keep_ratio = appcorr_options["token_keep_ratio"]
        token_keep_thres = appcorr_options["token_keep_thres"]
        sdpa_query_bucket_size = appcorr_options["sdpa_query_bucket_size"]
        mobile_pscore_hint = context.get('mobile_pscore_hint_map')
        attn_col_alive_ratio = appcorr_options["attn_col_alive_ratio"]
        fixed_query_state = plan.query_state
        if fixed_query_state is not None:
            cache["_token_prune_kept_patch_total"] = (
                cache.get("_token_prune_kept_patch_total", x_temp.new_zeros((), dtype=torch.float32))
                + plan.kept_patch_count.sum(dtype=torch.float32)
            )
            cache["_token_prune_full_patch_total"] = (
                cache.get("_token_prune_full_patch_total", x_temp.new_zeros((), dtype=torch.float32))
                + plan.full_patch_count.sum(dtype=torch.float32)
            )
            cache["_token_prune_kept_residual_mass_total"] = (
                cache.get("_token_prune_kept_residual_mass_total", x_temp.new_zeros((), dtype=torch.float32))
                + plan.kept_residual_mass.sum(dtype=torch.float32)
            )
            cache["_token_prune_full_residual_mass_total"] = (
                cache.get("_token_prune_full_residual_mass_total", x_temp.new_zeros((), dtype=torch.float32))
                + plan.full_residual_mass.sum(dtype=torch.float32)
            )

        # dindice must be passed to correct()
        if 'dindice' not in locals(): 
             return 

        self.begin_dinov3_correct_event()
        for lidx in range(start_l, end_l):
            x_temp, cache = self.run_dinov3_correct_block(
                lidx, x_temp, dindice, rope_sincos, cache, f"layer{lidx}",
                source_key=f"layer{lidx}",
                appcorr_method=appcorr_options["method"],
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
                debug=False
            )
        
        context['current_feature'] = x_temp
        context['cache_feature'] = cache

    def head_inference(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[str, Any]:
        feat = context.get('current_feature')
        backbone = self.model.backbone
        
        # Apply Norm
        x_norm = backbone.norm(feat)
        x_norm_clstoken = x_norm[:, 0]
        x_norm_patchtokens = x_norm[:, backbone.n_storage_tokens + 1 :]
        
        # Simulating Classifier Wrapper
        linear_input = torch.cat(
            [x_norm_clstoken, x_norm_patchtokens.mean(dim=1)],
            dim=1,
        )
        output_logits = self.model.linear_head(linear_input)
        context['output'] = output_logits
        
        # Analysis Logging
        probs = torch.softmax(output_logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        top5_probs, top5_indices = torch.topk(probs, k=5, dim=1)
        
        return {
            'entropy': entropy.cpu().numpy().tolist(),
            'top5_probs': top5_probs.cpu().numpy().tolist(),
            'top5_indices': top5_indices.cpu().numpy().tolist(),
            'active_indices': context.get('active_indices', torch.arange(len(probs), device=self.device)).cpu().numpy().tolist()
        }

    def full_inference(self, task: Task, context: Dict[str, Any], config: Any):
        inp = context.get('input_tensor')
        if inp is not None:
            # The stock model call has no per-block precision hook, so `precision` reaches it only
            # by substituting the quantized blocks into the backbone for the duration of the call.
            with self.dinov3_full_inference_precision():
                context['output'] = self.model(inp)

    def get_final_results(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[int, Any]:
        """Extracts Top-5 predictions from context['output'] for all active indices."""
        results = {}
        if 'output' in context and 'active_indices' in context:
            output_logits = context['output']
            _, top5 = torch.topk(output_logits, k=5, dim=1)
            
            current_active_indices = context['active_indices']
            preds_list = top5.cpu().numpy().tolist()
            
            for orig_idx, pred_list in zip(current_active_indices.cpu().numpy(), preds_list):
                results[int(orig_idx)] = pred_list
        return results

    def decide_exit(self, task: Task, context: Dict[str, Any], config: Any) -> Dict[str, Any]:
        if 'output' not in context: return {}

        # Get Criteria
        early_exit_config = config.get_early_exit_config()
        metric = early_exit_config.get('metric', 'max_prob')
        threshold = early_exit_config.get('threshold', 0.9)
        
        output_logits = context['output']
        probs = torch.softmax(output_logits, dim=1)
        
        # Compute Top-5 for potential exits
        _, top5 = torch.topk(output_logits, k=5, dim=1)
        
        # Identify Exits
        if metric == 'max_prob':
            max_probs, _ = torch.max(probs, dim=1)
            exit_mask = max_probs >= threshold
        elif metric == 'top2_margin':
            top2_vals, _ = torch.topk(probs, k=2, dim=1)
            margin = top2_vals[:, 0] - top2_vals[:, 1]
            exit_mask = margin >= threshold
        elif metric == 'entropy':
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            exit_mask = entropy <= threshold
        else:
            # Default/Fallback
            max_probs, _ = torch.max(probs, dim=1)
            exit_mask = torch.zeros_like(max_probs, dtype=torch.bool)
            
        num_exits = exit_mask.sum().item()
        
        if num_exits > 0:
            print(f"[Executor] exiting {num_exits} samples (metric {metric}, thresh {threshold})")
            
            # Store Results for Exited Samples
            if 'final_results' not in context:
                context['final_results'] = {}
            
            # Map active index to original request index
            current_active_indices = context['active_indices']
            
            exited_original_indices = current_active_indices[exit_mask]
            exited_preds = top5[exit_mask].cpu().numpy().tolist()
            
            for orig_idx, pred_list in zip(exited_original_indices.cpu().numpy(), exited_preds):
                context['final_results'][int(orig_idx)] = pred_list
                
            # Slice State
            keep_mask = ~exit_mask
            
            if keep_mask.sum() == 0:
                context['active_indices'] = torch.empty(0, device=self.device, dtype=torch.long)
                if 'current_feature' in context: del context['current_feature']
                if 'input_tokens' in context: del context['input_tokens']
            else:
                 # Helper to slice
                 self._slice_context(context, keep_mask)

        return {
            'num_exits': num_exits,
            'threshold': threshold
        }

    def _slice_context(self, context, keep_mask):
        # Metadata
        context['active_indices'] = context['active_indices'][keep_mask]
        
        # Group Map
        if 'group_map' in context:
            context['group_map'] = context['group_map'][keep_mask]
            
        # Input Tensors
        if 'input_tokens' in context:
            context['input_tokens'] = context['input_tokens'][keep_mask]
            
        if 'input_tensor' in context:
            context['input_tensor'] = context['input_tensor'][keep_mask]
            
        # Current Feature
        if 'current_feature' in context:
            context['current_feature'] = context['current_feature'][keep_mask]
            
        # Cache (KV Cache)
        if 'cache_feature' in context:
            cache = context['cache_feature']
            for k in list(cache.keys()):
                cache[k] = self._slice_cache_value(cache[k], keep_mask)
                
        # Clear derived structures
        if 'group_plans' in context:
            context['group_plans'] = {}

        # Output (Logits)
        if 'output' in context:
            context['output'] = context['output'][keep_mask]

    def _slice_cache_value(self, value, keep_mask):
        if isinstance(value, dict):
            sliced = {}
            for key, item in value.items():
                if key == 'key_to_slot':
                    sliced[key] = item.copy()
                elif key == 'query_count':
                    # Packed sparse-cache metadata is indexed by group slot, not batch.
                    sliced[key] = item
                elif key in {'query_idx', 'col_idx', 'attn_prob_sel'} and isinstance(item, torch.Tensor):
                    # Packed sparse attention caches use shape [G, B, ...]; slice only the batch axis.
                    if item.ndim > 1 and item.shape[1] == keep_mask.shape[0]:
                        sliced[key] = item[:, keep_mask].contiguous()
                    else:
                        sliced[key] = self._slice_cache_value(item, keep_mask)
                else:
                    sliced[key] = self._slice_cache_value(item, keep_mask)
            return sliced

        if isinstance(value, torch.Tensor):
            if value.ndim > 0 and value.shape[0] == keep_mask.shape[0]:
                return value[keep_mask]
            if value.ndim > 1 and value.shape[1] == keep_mask.shape[0]:
                return value[:, keep_mask]
            return value

        if isinstance(value, tuple):
            return tuple(self._slice_cache_value(v, keep_mask) for v in value)

        if isinstance(value, list):
            return [self._slice_cache_value(v, keep_mask) for v in value]

        if isinstance(value, dict):
            return {k: self._slice_cache_value(v, keep_mask) for k, v in value.items()}

        return value
