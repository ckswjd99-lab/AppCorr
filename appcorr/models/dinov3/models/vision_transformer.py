# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
from functools import partial
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn.init
from torch import Tensor, nn

from ..layers import LayerScale, Mlp, PatchEmbed, RMSNorm, RopePositionEmbedding, SelfAttentionBlock, SwiGLUFFN
from ..layers._triton_kernels import token_prune_select_compact_triton
from ..utils import named_apply
from ..utils.hier_token import HierarchicalToken

logger = logging.getLogger("dinov3")

ffn_layer_dict = {
    "mlp": Mlp,
    "swiglu": SwiGLUFFN,
    "swiglu32": partial(SwiGLUFFN, align_to=32),
    "swiglu64": partial(SwiGLUFFN, align_to=64),
    "swiglu128": partial(SwiGLUFFN, align_to=128),
}

norm_layer_dict = {
    "layernorm": partial(nn.LayerNorm, eps=1e-6),
    "layernormbf16": partial(nn.LayerNorm, eps=1e-5),
    "rmsnorm": RMSNorm,
}

dtype_dict = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def init_weights_vit(module: nn.Module, name: str = ""):
    if isinstance(module, nn.Linear):
        torch.nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        if hasattr(module, "bias_mask") and module.bias_mask is not None:
            o = module.out_features
            module.bias_mask.fill_(1)
            module.bias_mask[o // 3 : 2 * o // 3].fill_(0)
    if isinstance(module, nn.LayerNorm):
        module.reset_parameters()
    if isinstance(module, LayerScale):
        module.reset_parameters()
    if isinstance(module, PatchEmbed):
        module.reset_parameters()
    if isinstance(module, RMSNorm):
        module.reset_parameters()

def create_group_index(num_tokens: int, num_groups: int, strategy: str, device: torch.device, **kwargs) -> torch.Tensor:
    if strategy == "uniform":
        group_idx = torch.randint(1, num_groups + 1, (num_tokens,), device=device)
    elif strategy == "grid":
        s = int(num_groups ** 0.5)  # sqrt(num_groups), assumes perfect square
        H = W = int((num_tokens) ** 0.5)  # assumes square input

        pattern = torch.arange(1, num_groups + 1, device=device).view(s, s)

        rep_h = (H + s - 1) // s
        rep_w = (W + s - 1) // s
        grid_2d = pattern.repeat(rep_h, rep_w)[:H, :W]

        group_idx = grid_2d.flatten()
    elif strategy == "geometric":
        probs = torch.rand(num_tokens, device=device)

        group_idx = torch.floor(-torch.log2(1 - probs)) + 1
        group_idx = torch.clamp(group_idx, max=num_groups).long()
    elif strategy == "uniform_diff":
        token_diffs: torch.Tensor = kwargs.get("token_diffs", None)
        if token_diffs is None:
            raise ValueError("token_diffs must be provided for 'uniform_diff' grouping strategy.")

        # token_diffs: [B, N, C]
        diffs_norm = torch.norm(token_diffs, p=1, dim=-1)
        B, N = diffs_norm.shape
        
        # Sort norms independently for each batch -> [B, N]
        sorted_norms, sorted_indices = torch.sort(diffs_norm, dim=1)

        # Determine Group Split Indices based on AVERAGE distribution
        avg_sorted_norms = sorted_norms.mean(dim=0)  # [N]
        
        cumsum_norms = torch.cumsum(avg_sorted_norms, dim=0)
        total_norm = cumsum_norms[-1]
        
        if total_norm == 0:
            return torch.randint(1, num_groups + 1, (B * N,), device=device).view(B, N)

        target_sum = total_norm / num_groups
        boundaries = torch.arange(1, num_groups, device=device, dtype=torch.float32) * target_sum

        # Create a "Template" Group Assignment -> [N]
        rank_to_group_id = torch.bucketize(cumsum_norms, boundaries) + 1

        # Expand to Batch -> [B, N]
        batch_group_ids = rank_to_group_id.unsqueeze(0).expand(B, -1)

        # Map back to original token order -> [B, N]
        group_idx = torch.zeros_like(diffs_norm, dtype=torch.long)
        group_idx.scatter_(1, sorted_indices, batch_group_ids)

    else:
        raise NotImplementedError(f"Unknown grouping strategy: {strategy}")

    return group_idx

class DinoVisionTransformer(nn.Module):
    def __init__(
        self,
        *,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        pos_embed_rope_base: float = 100.0,
        pos_embed_rope_min_period: float | None = None,
        pos_embed_rope_max_period: float | None = None,
        pos_embed_rope_normalize_coords: Literal["min", "max", "separate"] = "separate",
        pos_embed_rope_shift_coords: float | None = None,
        pos_embed_rope_jitter_coords: float | None = None,
        pos_embed_rope_rescale_coords: float | None = None,
        pos_embed_rope_dtype: str = "bf16",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = True,
        drop_path_rate: float = 0.0,
        layerscale_init: float | None = None,
        norm_layer: str = "layernorm",
        ffn_layer: str = "mlp",
        ffn_bias: bool = True,
        proj_bias: bool = True,
        n_storage_tokens: int = 0,
        mask_k_bias: bool = False,
        untie_cls_and_patch_norms: bool = False,
        untie_global_and_local_cls_norm: bool = False,
        device: Any | None = None,
        **ignored_kwargs,
    ):
        super().__init__()
        if len(ignored_kwargs) > 0:
            logger.warning(f"Ignored kwargs: {ignored_kwargs}")
        del ignored_kwargs

        norm_layer_cls = norm_layer_dict[norm_layer]

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            flatten_embedding=False,
        )

        self.cls_token = nn.Parameter(torch.empty(1, 1, embed_dim, device=device))
        self.n_storage_tokens = n_storage_tokens
        if self.n_storage_tokens > 0:
            self.storage_tokens = nn.Parameter(torch.empty(1, n_storage_tokens, embed_dim, device=device))
        logger.info(f"using base={pos_embed_rope_base} for rope new")
        logger.info(f"using min_period={pos_embed_rope_min_period} for rope new")
        logger.info(f"using max_period={pos_embed_rope_max_period} for rope new")
        logger.info(f"using normalize_coords={pos_embed_rope_normalize_coords} for rope new")
        logger.info(f"using shift_coords={pos_embed_rope_shift_coords} for rope new")
        logger.info(f"using rescale_coords={pos_embed_rope_rescale_coords} for rope new")
        logger.info(f"using jitter_coords={pos_embed_rope_jitter_coords} for rope new")
        logger.info(f"using dtype={pos_embed_rope_dtype} for rope new")
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=pos_embed_rope_base,
            min_period=pos_embed_rope_min_period,
            max_period=pos_embed_rope_max_period,
            normalize_coords=pos_embed_rope_normalize_coords,
            shift_coords=pos_embed_rope_shift_coords,
            jitter_coords=pos_embed_rope_jitter_coords,
            rescale_coords=pos_embed_rope_rescale_coords,
            dtype=dtype_dict[pos_embed_rope_dtype],
            device=device,
        )
        logger.info(f"using {ffn_layer} layer as FFN")
        ffn_layer_cls = ffn_layer_dict[ffn_layer]
        ffn_ratio_sequence = [ffn_ratio] * depth
        blocks_list = [
            SelfAttentionBlock(
                dim=embed_dim,
                num_heads=num_heads,
                ffn_ratio=ffn_ratio_sequence[i],
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=drop_path_rate,
                norm_layer=norm_layer_cls,
                act_layer=nn.GELU,
                ffn_layer=ffn_layer_cls,
                init_values=layerscale_init,
                mask_k_bias=mask_k_bias,
                device=device,
            )
            for i in range(depth)
        ]

        self.chunked_blocks = False
        self.blocks = nn.ModuleList(blocks_list)

        # This norm is applied to everything, or when untying, to patch and mask tokens.
        self.norm = norm_layer_cls(embed_dim)

        self.untie_cls_and_patch_norms = untie_cls_and_patch_norms
        if untie_cls_and_patch_norms:
            # When untying, this norm is applied to CLS tokens and registers.
            self.cls_norm = norm_layer_cls(embed_dim)
        else:
            self.cls_norm = None

        self.untie_global_and_local_cls_norm = untie_global_and_local_cls_norm
        if untie_global_and_local_cls_norm:
            # When untying, this norm is applied to local CLS tokens and registers.
            # This norm is never used during eval.
            self.local_cls_norm = norm_layer_cls(embed_dim)
        else:
            self.local_cls_norm = None
        self.head = nn.Identity()
        self.mask_token = nn.Parameter(torch.empty(1, embed_dim, device=device))

        # AppCorr settings
        self.appcorr_enabled = False
        self.appcorr_update_attn = False
        self.appcorr_pyramid_levels = [0]
        self.appcorr_token_res = [1.0]
        self.appcorr_plan = []
        self.appcorr_group = 0
        self.appcorr_grouping_strategy = "uniform"
        self.appcorr_cls_alive_ratio = 0.2
        self.appcorr_attn_col_alive_ratio = 1.0
        self.appcorr_token_prune_enabled = False
        self.appcorr_token_prune_threshold = 0.0
        self.appcorr_token_prune_min_keep = 1
        self.appcorr_method = "partial_token"

        self.appcorr_debug = False
    
    def set_appcorr_mode(
        self,
        enabled: bool | None = None,
        update_attn: bool | None = None,
        pyramid_levels: List[int] | None = None,
        token_res: List[float] | None = None,
        plan: List[Tuple[str, int, range, Optional[int]]] | None = None,
        num_groups: int | None = None,
        group_strategy: str | None = None,
        cls_alive_ratio: float | None = None,
        attn_col_alive_ratio: float | None = None,
        token_prune_enabled: bool | None = None,
        token_prune_threshold: float | None = None,
        token_prune_min_keep: int | None = None,
        method: str | None = None,
        debug: bool | None = None,
    ):
        if enabled is not None: self.appcorr_enabled = enabled
        if update_attn is not None: self.appcorr_update_attn = update_attn
        if pyramid_levels is not None: self.appcorr_pyramid_levels = pyramid_levels
        if token_res is not None: self.appcorr_token_res = token_res
        if plan is not None: self.appcorr_plan = plan
        if num_groups is not None: self.appcorr_group = num_groups
        if group_strategy is not None: self.appcorr_grouping_strategy = group_strategy
        if cls_alive_ratio is not None: self.appcorr_cls_alive_ratio = cls_alive_ratio
        if attn_col_alive_ratio is not None: self.appcorr_attn_col_alive_ratio = attn_col_alive_ratio
        if token_prune_enabled is not None: self.appcorr_token_prune_enabled = token_prune_enabled
        if token_prune_threshold is not None: self.appcorr_token_prune_threshold = token_prune_threshold
        if token_prune_min_keep is not None: self.appcorr_token_prune_min_keep = token_prune_min_keep
        if method is not None: self.appcorr_method = method
        if debug is not None: self.appcorr_debug = debug

        for blk in self.blocks:
            if hasattr(blk, "set_appcorr_method"):
                blk.set_appcorr_method(method=self.appcorr_method)


    def init_weights(self):
        self.rope_embed._init_weights()
        nn.init.normal_(self.cls_token, std=0.02)
        if self.n_storage_tokens > 0:
            nn.init.normal_(self.storage_tokens, std=0.02)
        nn.init.zeros_(self.mask_token)
        named_apply(init_weights_vit, self)

    def prepare_tokens_with_masks(self, x: Tensor, masks=None) -> Tuple[Tensor, Tuple[int]]:
        x = self.patch_embed(x)
        B, H, W, _ = x.shape
        x = x.flatten(1, 2)

        if masks is not None:
            x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype).unsqueeze(0), x)
            cls_token = self.cls_token
        else:
            cls_token = self.cls_token + 0 * self.mask_token
        if self.n_storage_tokens > 0:
            storage_tokens = self.storage_tokens
        else:
            storage_tokens = torch.empty(
                1,
                0,
                cls_token.shape[-1],
                dtype=cls_token.dtype,
                device=cls_token.device,
            )

        x = torch.cat(
            [
                cls_token.expand(B, -1, -1),
                storage_tokens.expand(B, -1, -1),
                x,
            ],
            dim=1,
        )

        return x, (H, W)

    def forward_features_list(self, x_list: List[Tensor], masks_list: List[Tensor]) -> List[Dict[str, Tensor]]:
        # AppCorr-specific forward_features
        if self.appcorr_enabled:
            return self.forward_features_list_appcorr(x_list, masks_list)
            
        x = []
        rope = []

        # Patch embed and concat additional tokens
        for t_x, t_masks in zip(x_list, masks_list):
            t2_x, hw_tuple = self.prepare_tokens_with_masks(t_x, t_masks)
            x.append(t2_x)
            rope.append(hw_tuple)
        
        # Run through transformer blocks
        for _, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = [self.rope_embed(H=H, W=W) for H, W in rope]
            else:
                rope_sincos = [None for r in rope]
            x = blk(x, rope_sincos)
        
        # Post normalization and output formatting
        output = self.post_features_list(x, masks_list)

        return output

    def forward_features_list_appcorr(self, x_list: List[Tensor], masks_list: List[Tensor]) -> List[Dict[str, Tensor]]:
        if len(x_list) != 1:
            raise NotImplementedError("AppCorr forward_features_list currently only supports single input.")

        B = x_list[0].shape[0]

        x_tensor = x_list[0]
        mask = masks_list[0]

        num_pretokens = 1 + self.n_storage_tokens  # CLS + storage tokens

        x_pyramid = []
        x_groups = []
        rope_pyramid = []

        # Prepare pyramid of inputs
        for level in self.appcorr_pyramid_levels:
            # downsample and upsample
            x_temp = x_tensor
            x_temp = nn.functional.interpolate(x_temp, scale_factor=2**(-level), mode='bicubic', align_corners=False)
            x_temp = nn.functional.interpolate(x_temp, scale_factor=2**level, mode='bicubic', align_corners=False)

            t2_x, hw_tuple = self.prepare_tokens_with_masks(x_temp, mask)
            x_pyramid.append(t2_x)
            rope_pyramid.append(hw_tuple)

            num_tokens = t2_x.shape[1] - num_pretokens  # Exclude pre-tokens
            group_idx = create_group_index(
                num_tokens, self.appcorr_group, self.appcorr_grouping_strategy, device=t2_x.device,
                token_diffs=(x_pyramid[0]-t2_x)[:, num_pretokens:, :]
            )   # [N] or [B, N]
            if len(group_idx.shape) == 1:
                group_idx = group_idx.unsqueeze(0).expand(B, -1)  # [B, N]
            
            pre_groups = torch.zeros(B, num_pretokens, device=t2_x.device, dtype=group_idx.dtype)
            full_group_idx = torch.cat([pre_groups, group_idx], dim=1)
            x_groups.append(full_group_idx)

        # Run through transformer blocks
        cache_feature = {}
        attn_cache_candidates = {}
        for (op_type, level, layers, group_idx) in self.appcorr_plan:
            if op_type != "C":
                continue
            dmask = (x_groups[level] == group_idx)
            dmask[:, :num_pretokens] = True
            attn_cache_candidates[(level, group_idx)] = torch.where(dmask)[1].view(B, -1)

        x_feature = x_pyramid[0]
        for (op_type, level, layers, group_idx) in self.appcorr_plan:
            if op_type == "A":
                # Approx
                for lidx in layers:
                    with torch.cuda.nvtx.range(f"approx_{lidx}"):
                        blk = self.blocks[lidx]
                        rope_sincos = self.rope_embed(H=rope_pyramid[level][0], W=rope_pyramid[level][1]) if self.rope_embed is not None else None
                        x_feature, cache_feature = blk.approx(
                            x_feature, rope_sincos, cache_feature, tag=f"layer{lidx}",
                            attn_cache_candidates=attn_cache_candidates if self.appcorr_method == "partial_channel" else None,
                            attn_col_alive_ratio=self.appcorr_attn_col_alive_ratio,
                            debug=self.appcorr_debug
                        )

                        if self.appcorr_debug:
                            torch.cuda.synchronize()
            elif op_type == "C":
                # Correct
                dmask = (x_groups[level] == group_idx)  # [B, N]
                dmask[:, :num_pretokens] = True  # Always keep pre-tokens
                dindice = torch.where(dmask)[1].view(B, -1)

                x_temp = x_pyramid[level]
                rope_sincos = self.rope_embed(H=rope_pyramid[level][0], W=rope_pyramid[level][1]) if self.rope_embed is not None else None
                fixed_token_prune_state = None
                if self.appcorr_token_prune_enabled:
                    B_level, _, C = x_temp.shape
                    num_level_pretokens = x_temp.shape[1] - rope_sincos[0].shape[0] if rope_sincos is not None else num_pretokens
                    gather_idx_x_full = dindice.unsqueeze(-1).expand(-1, -1, C)
                    dx_sel_full = (x_feature - x_temp).gather(1, gather_idx_x_full).contiguous()
                    dindice_sel, query_pos_idx, query_valid_mask, kept_patch_count = token_prune_select_compact_triton(
                        dx_sel_full,
                        dindice,
                        num_pretokens=num_level_pretokens,
                        token_prune_threshold=self.appcorr_token_prune_threshold,
                        token_prune_min_keep=self.appcorr_token_prune_min_keep,
                    )
                    active_batch_idx, active_pos_idx = query_valid_mask.nonzero(as_tuple=True)
                    active_token_idx = dindice_sel[active_batch_idx, active_pos_idx]
                    active_query_pos = query_pos_idx[active_batch_idx, active_pos_idx]
                    active_counts = torch.bincount(active_batch_idx, minlength=B_level)
                    max_active = int(active_counts.max().item()) if active_counts.numel() > 0 else 0
                    active_query_pos_padded = torch.zeros((B_level, max_active), device=x_temp.device, dtype=torch.long)
                    active_query_mask = torch.zeros((B_level, max_active), device=x_temp.device, dtype=torch.bool)
                    if active_query_pos.numel() > 0:
                        batch_start = active_counts.cumsum(0) - active_counts
                        slot_idx = torch.arange(active_query_pos.shape[0], device=x_temp.device, dtype=torch.long)
                        slot_idx = slot_idx - torch.repeat_interleave(batch_start, active_counts)
                        active_query_pos_padded[active_batch_idx, slot_idx] = active_query_pos
                        active_query_mask[active_batch_idx, slot_idx] = True
                    fixed_token_prune_state = {
                        "dindice_sel": dindice_sel,
                        "query_pos_idx": query_pos_idx,
                        "query_valid_mask": query_valid_mask,
                        "kept_patch_count": kept_patch_count,
                        "active_batch_idx": active_batch_idx,
                        "active_pos_idx": active_pos_idx,
                        "active_token_idx": active_token_idx,
                        "active_query_pos": active_query_pos,
                        "active_query_pos_padded": active_query_pos_padded,
                        "active_query_mask": active_query_mask,
                    }
                
                for lidx in layers:
                    with torch.cuda.nvtx.range(f"correct_{lidx}"):
                        blk = self.blocks[lidx]
                        x_temp, cache_feature = blk.correct(
                            x_temp, dindice, rope_sincos, cache_feature, tag=f"layer{lidx}",
                            cls_alive_ratio=self.appcorr_cls_alive_ratio,
                            attn_col_alive_ratio=self.appcorr_attn_col_alive_ratio,
                            token_prune_enabled=self.appcorr_token_prune_enabled,
                            token_prune_threshold=self.appcorr_token_prune_threshold,
                            token_prune_min_keep=self.appcorr_token_prune_min_keep,
                            fixed_token_prune_state=fixed_token_prune_state,
                            attn_cache_key=(level, group_idx),
                            debug=self.appcorr_debug
                        )

                        if self.appcorr_debug:
                            torch.cuda.synchronize()
                
                x_feature = x_temp.to(x_feature.dtype)
            else:
                raise NotImplementedError(f"Unknown op_type {op_type} in AppCorr plan.")

        # Post normalization and output formatting
        output = self.post_features_list([x_feature], masks_list)
        return output
    
    def post_features_list(self, x: List[Tensor], masks_list: List[Tensor]) -> List[Dict[str, Tensor]]:
        # Post normalization and output formatting
        all_x = x
        output = []
        for idx, (x, masks) in enumerate(zip(all_x, masks_list)):
            if self.untie_cls_and_patch_norms or self.untie_global_and_local_cls_norm:
                if self.untie_global_and_local_cls_norm and self.training and idx == 1:
                    # Assume second entry of list corresponds to local crops.
                    # We only ever apply this during training.
                    x_norm_cls_reg = self.local_cls_norm(x[:, : self.n_storage_tokens + 1])
                elif self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(x[:, : self.n_storage_tokens + 1])
                else:
                    x_norm_cls_reg = self.norm(x[:, : self.n_storage_tokens + 1])
                x_norm_patch = self.norm(x[:, self.n_storage_tokens + 1 :])
            else:
                x_norm = self.norm(x)
                x_norm_cls_reg = x_norm[:, : self.n_storage_tokens + 1]
                x_norm_patch = x_norm[:, self.n_storage_tokens + 1 :]
            output.append(
                {
                    "x_norm_clstoken": x_norm_cls_reg[:, 0],
                    "x_storage_tokens": x_norm_cls_reg[:, 1:],
                    "x_norm_patchtokens": x_norm_patch,
                    "x_prenorm": x,
                    "masks": masks,
                }
            )

        return output

    def forward_features(self, x: Tensor | List[Tensor], masks: Optional[Tensor] = None) -> List[Dict[str, Tensor]]:
        if isinstance(x, torch.Tensor):
            return self.forward_features_list([x], [masks])[0]
        else:
            return self.forward_features_list(x, masks)

    def _get_intermediate_layers_not_chunked(self, x: Tensor, n: int = 1) -> List[Tensor]:
        x, (H, W) = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            if self.rope_embed is not None:
                rope_sincos = self.rope_embed(H=H, W=W)
            else:
                rope_sincos = None
            x = blk(x, rope_sincos)
            if i in blocks_to_take:
                output.append(x)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output

    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        *,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        return_extra_tokens: bool = False,
        norm: bool = True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor, ...]]]:
        outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs_normed = []
            for out in outputs:
                if self.untie_cls_and_patch_norms:
                    x_norm_cls_reg = self.cls_norm(out[:, : self.n_storage_tokens + 1])
                    x_norm_patch = self.norm(out[:, self.n_storage_tokens + 1 :])
                    outputs_normed.append(torch.cat((x_norm_cls_reg, x_norm_patch), dim=1))
                else:
                    outputs_normed.append(self.norm(out))
            outputs = outputs_normed
        class_tokens = [out[:, 0] for out in outputs]
        extra_tokens = [out[:, 1 : self.n_storage_tokens + 1] for out in outputs]
        outputs = [out[:, self.n_storage_tokens + 1 :] for out in outputs]
        if reshape:
            B, _, h, w = x.shape
            outputs = [
                out.reshape(B, h // self.patch_size, w // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if not return_class_token and not return_extra_tokens:
            return tuple(outputs)
        elif return_class_token and not return_extra_tokens:
            return tuple(zip(outputs, class_tokens))
        elif not return_class_token and return_extra_tokens:
            return tuple(zip(outputs, extra_tokens))
        elif return_class_token and return_extra_tokens:
            return tuple(zip(outputs, class_tokens, extra_tokens))

    def forward(self, *args, is_training: bool = False, **kwargs) -> List[Dict[str, Tensor]] | Tensor:
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return self.head(ret["x_norm_clstoken"])


def vit_small(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=384,
        depth=12,
        num_heads=6,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_base(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=768,
        depth=12,
        num_heads=12,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_large(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_so400m(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1152,
        depth=27,
        num_heads=18,
        ffn_ratio=3.777777778,
        **kwargs,
    )
    return model


def vit_huge2(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1280,
        depth=32,
        num_heads=20,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_giant2(patch_size=16, **kwargs):
    """
    Close to ViT-giant, with embed-dim 1536 and 24 heads => embed-dim per head 64
    """
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=1536,
        depth=40,
        num_heads=24,
        ffn_ratio=4,
        **kwargs,
    )
    return model


def vit_7b(patch_size=16, **kwargs):
    model = DinoVisionTransformer(
        patch_size=patch_size,
        embed_dim=4096,
        depth=40,
        num_heads=32,
        ffn_ratio=3,
        **kwargs,
    )
    return model
