# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from ..utils import cat_keep_shapes, uncat_with_shapes

from ._triton_kernels import (
    fused_layerscale_add,
    masked_residual_add_triton,
    masked_token_update_triton,
)
from .attention import CausalSelfAttention, PackedQueryState, QueryStateLike, SelfAttention
from .ffn_layers import Mlp, SwiGLUFFN
from .layer_scale import LayerScale  # , DropPath

from ..utils.hier_token import HierarchicalToken

torch._dynamo.config.automatic_dynamic_shapes = False
torch._dynamo.config.accumulated_cache_size_limit = 1024


class SelfAttentionBlock(nn.Module):
    _MOBILE_HINT_PSCORE_ALIASES = frozenset({
        "residual_rms",
        "patch_residual_rms",
        "residual_l2",
        "residual_l2_energy",
        "residual_energy",
        "patch_residual_l2",
        "patch_residual_energy",
    })

    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = SelfAttention,
        ffn_layer: Callable[..., nn.Module] = SwiGLUFFN,
        mask_k_bias: bool = False,
        device=None,
    ) -> None:
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        self.attn: SelfAttention = attn_class(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            proj_bias=proj_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
            mask_k_bias=mask_k_bias,
            device=device,
        )
        self.ls1 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * ffn_ratio)
        self.mlp: SwiGLUFFN = ffn_layer(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
            bias=ffn_bias,
            device=device,
        )
        self.ls2 = LayerScale(dim, init_values=init_values, device=device) if init_values else nn.Identity()

        self.sample_drop_ratio = drop_path

    @staticmethod
    def _maybe_index_rope(rope: tuple[Tensor, Tensor] | None, indices: Tensor) -> tuple[Tensor, Tensor] | None:
        if rope is None:
            return None

        sin, cos = rope
        assert sin.ndim == cos.ndim
        if sin.ndim == 4:
            # If the rope embedding has a batch dimension (is different for each batch element), index into it
            return sin[indices], cos[indices]  # [batch, heads, patches, embed_dim]
        else:
            # No batch dimension, do not index
            return sin, cos  # [heads, patches, embed_dim] or [patches, embed_dim]

    def _forward(self, x: Tensor, rope=None) -> Tensor:
        """
        This is the reference implementation for a single tensor, matching what is done below for a list.
        We call the list op on [x] instead of this function.
        """
        b, _, _ = x.shape
        sample_subset_size = max(int(b * (1 - self.sample_drop_ratio)), 1)
        residual_scale_factor = b / sample_subset_size

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1 = (torch.randperm(b, device=x.device))[:sample_subset_size]

            x_subset_1 = x[indices_1]
            rope_subset = self._maybe_index_rope(rope, indices_1)
            residual_1 = self.attn(self.norm1(x_subset_1), rope=rope_subset)

            x_attn = torch.index_add(
                x,
                dim=0,
                source=self.ls1(residual_1),
                index=indices_1,
                alpha=residual_scale_factor,
            )

            indices_2 = (torch.randperm(b, device=x.device))[:sample_subset_size]

            x_subset_2 = x_attn[indices_2]
            residual_2 = self.mlp(self.norm2(x_subset_2))

            x_ffn = torch.index_add(
                x_attn,
                dim=0,
                source=self.ls2(residual_2),
                index=indices_2,
                alpha=residual_scale_factor,
            )
        else:
            x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
            x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))

        return x_ffn

    def _forward_list(self, x_list: List[Tensor], rope_list=None) -> List[Tensor]:
        """
        This list operator concatenates the tokens from the list of inputs together to save
        on the elementwise operations. Torch-compile memory-planning allows hiding the overhead
        related to concat ops.
        """
        b_list = [x.shape[0] for x in x_list]
        sample_subset_sizes = [max(int(b * (1 - self.sample_drop_ratio)), 1) for b in b_list]
        residual_scale_factors = [b / sample_subset_size for b, sample_subset_size in zip(b_list, sample_subset_sizes)]

        if self.training and self.sample_drop_ratio > 0.0:
            indices_1_list = [
                (torch.randperm(b, device=x.device))[:sample_subset_size]
                for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
            ]
            x_subset_1_list = [x[indices_1] for x, indices_1 in zip(x_list, indices_1_list)]

            if rope_list is not None:
                rope_subset_list = [
                    self._maybe_index_rope(rope, indices_1) for rope, indices_1 in zip(rope_list, indices_1_list)
                ]
            else:
                rope_subset_list = rope_list

            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_1_list)
            norm1 = uncat_with_shapes(self.norm1(flattened), shapes, num_tokens)
            residual_1_list = self.attn.forward_list(norm1, rope_list=rope_subset_list)

            x_attn_list = [
                torch.index_add(
                    x,
                    dim=0,
                    source=self.ls1(residual_1),
                    index=indices_1,
                    alpha=residual_scale_factor,
                )
                for x, residual_1, indices_1, residual_scale_factor in zip(
                    x_list, residual_1_list, indices_1_list, residual_scale_factors
                )
            ]

            indices_2_list = [
                (torch.randperm(b, device=x.device))[:sample_subset_size]
                for x, b, sample_subset_size in zip(x_list, b_list, sample_subset_sizes)
            ]
            x_subset_2_list = [x[indices_2] for x, indices_2 in zip(x_attn_list, indices_2_list)]
            flattened, shapes, num_tokens = cat_keep_shapes(x_subset_2_list)
            norm2_flat = self.norm2(flattened)
            norm2_list = uncat_with_shapes(norm2_flat, shapes, num_tokens)

            residual_2_list = self.mlp.forward_list(norm2_list)

            x_ffn = [
                torch.index_add(
                    x_attn,
                    dim=0,
                    source=self.ls2(residual_2),
                    index=indices_2,
                    alpha=residual_scale_factor,
                )
                for x_attn, residual_2, indices_2, residual_scale_factor in zip(
                    x_attn_list, residual_2_list, indices_2_list, residual_scale_factors
                )
            ]
        else:
            x_out = []
            for x, rope in zip(x_list, rope_list):
                x_attn = x + self.ls1(self.attn(self.norm1(x), rope=rope))
                x_ffn = x_attn + self.ls2(self.mlp(self.norm2(x_attn)))
                x_out.append(x_ffn)
            x_ffn = x_out

        return x_ffn

    def forward(self, x_or_x_list, rope_or_rope_list=None) -> List[Tensor]:
        if isinstance(x_or_x_list, Tensor):
            # for reference:
            # return self._forward(x_or_x_list, rope=rope_or_rope_list)
            # in order to match implementations we call the list op:
            return self._forward_list([x_or_x_list], rope_list=[rope_or_rope_list])[0]
        elif isinstance(x_or_x_list, list):
            if rope_or_rope_list is None:
                rope_or_rope_list = [None for x in x_or_x_list]
            # return [self._forward(x, rope=rope) for x, rope in zip(x_or_x_list, rope_or_rope_list)]
            return self._forward_list(x_or_x_list, rope_list=rope_or_rope_list)
        else:
            raise AssertionError
    
    def approx(
        self, x: torch.Tensor, rope: Tuple[torch.Tensor], cache_feature: Dict, tag: str, **kwargs
    ) -> List[Tensor]:
        with torch.cuda.nvtx.range("approx"):
            appcorr_method = kwargs.get("appcorr_method", "partial_token")
            if appcorr_method == "partial_token":
                return self.approx_partial_token(x, rope, cache_feature, tag, **kwargs)
            if appcorr_method == "partial_channel":
                return self.approx_partial_channel(x, rope, cache_feature, tag, **kwargs)
            raise ValueError(
                f"Unknown SelfAttentionBlock.approx method '{appcorr_method}'. "
                "Available methods: partial_channel, partial_token"
            )
    
    def correct(
            self, x: torch.Tensor, dindice: List[int], rope: Tuple[torch.Tensor], cache_feature: Dict, tag: str, **kwargs
    ) -> List[Tensor]:
        with torch.cuda.nvtx.range("correct"):
            appcorr_method = kwargs.get("appcorr_method", "partial_token")
            if appcorr_method == "partial_token":
                return self.correct_partial_token(x, dindice, rope, cache_feature, tag, **kwargs)
            if appcorr_method == "partial_channel":
                return self.correct_partial_channel(
                    x,
                    dindice,
                    rope,
                    cache_feature,
                    tag,
                    fixed_query_state=kwargs["fixed_query_state"],
                    group_plan=kwargs["group_plan"],
                    attn_col_alive_ratio=kwargs.get("attn_col_alive_ratio", 1.0),
                    attn_cache_key=kwargs.get("attn_cache_key"),
                )
            raise ValueError(
                f"Unknown SelfAttentionBlock.correct method '{appcorr_method}'. "
                "Available methods: partial_channel, partial_token"
            )

    def approx_partial_token(
        self, x: torch.Tensor, rope: Tuple[torch.Tensor], cache_feature: Dict, tag: str, **kwargs
    ) -> List[Tensor]:
        # check debug
        debug = kwargs.get("debug", False)
        server_pscore = str(kwargs.get("server_pscore", "cls_attn_prob"))

        with torch.cuda.nvtx.range("approx_attn"):
            x_attn, cache_feature = self.attn.approx(
                self.norm1(x),
                rope=rope,
                cache_feature=cache_feature,
                tag=tag,
                appcorr_method="partial_token",
                server_pscore=server_pscore,
            )
            x_attn = self.ls1(x_attn)  # [B, N, C]
            cache_feature[f"{tag}_blocks_out_sum"] = x_attn.detach().clone()
            
            x_attn = x + x_attn

            if debug:
                torch.cuda.synchronize()

        with torch.cuda.nvtx.range("approx_ffn"):
            mlp_out = self.ls2(self.mlp(self.norm2(x_attn)))  # [B, N, C]
            cache_feature[f"{tag}_blocks_out_sum"] += mlp_out.detach()

            x_ffn = x_attn + mlp_out

            if debug:
                torch.cuda.synchronize()

        return x_ffn, cache_feature

    @staticmethod
    def _resolve_token_keep_threshold(kwargs: Dict) -> float | None:
        token_keep_thres = kwargs.get("token_keep_thres")
        if token_keep_thres in {None, "", "null", "None"}:
            return None
        return float(token_keep_thres)

    @staticmethod
    def _select_patch_keep_mask(
        combined_patch_scores: torch.Tensor,
        token_keep_ratio: float,
        token_keep_thres: float | None,
    ) -> torch.Tensor:
        B, num_patch_candidates = combined_patch_scores.shape
        keep_patch_mask = torch.zeros(
            (B, num_patch_candidates),
            device=combined_patch_scores.device,
            dtype=torch.bool,
        )
        if num_patch_candidates == 0:
            return keep_patch_mask

        if token_keep_thres is not None:
            keep_patch_mask = combined_patch_scores >= token_keep_thres
            return keep_patch_mask

        k_refined = min(int(num_patch_candidates * token_keep_ratio), num_patch_candidates)
        if k_refined <= 0:
            return keep_patch_mask

        topk_local_idx = torch.topk(
            combined_patch_scores,
            k=k_refined,
            dim=1,
            largest=True,
        ).indices
        keep_patch_mask.scatter_(1, topk_local_idx, True)
        return keep_patch_mask

    @staticmethod
    def _build_packed_query_state(
        dindice_pre: torch.Tensor,
        dindice_patches: torch.Tensor,
        keep_patch_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, PackedQueryState]:
        B = dindice_pre.shape[0]
        num_pretokens = dindice_pre.shape[1]
        kept_patch_count = keep_patch_mask.sum(dim=1, dtype=torch.int32)
        max_keep = int(kept_patch_count.max().item()) if kept_patch_count.numel() > 0 else 0
        max_active = num_pretokens + max_keep

        update_indice = torch.zeros(
            (B, max_active),
            device=dindice_pre.device,
            dtype=dindice_pre.dtype,
        )
        if num_pretokens > 0:
            update_indice[:, :num_pretokens] = dindice_pre

        if max_keep > 0:
            batch_idx, patch_idx = keep_patch_mask.nonzero(as_tuple=True)
            counts = torch.bincount(batch_idx, minlength=B)
            batch_start = counts.cumsum(0) - counts
            slot_idx = torch.arange(patch_idx.shape[0], device=dindice_pre.device, dtype=torch.long)
            slot_idx = slot_idx - torch.repeat_interleave(batch_start, counts)
            update_indice[batch_idx, num_pretokens + slot_idx] = dindice_patches[batch_idx, patch_idx]

        if max_active > 0:
            active_query_pos_padded = torch.arange(
                max_active,
                device=dindice_pre.device,
                dtype=torch.long,
            ).unsqueeze(0).expand(B, -1)
            valid_lengths = kept_patch_count.to(dtype=torch.long) + num_pretokens
            query_valid_mask = active_query_pos_padded < valid_lengths.unsqueeze(1)
        else:
            active_query_pos_padded = torch.empty((B, 0), device=dindice_pre.device, dtype=torch.long)
            query_valid_mask = torch.empty((B, 0), device=dindice_pre.device, dtype=torch.bool)

        active_batch_idx, active_pos_idx = query_valid_mask.nonzero(as_tuple=True)
        active_token_idx = update_indice[active_batch_idx, active_pos_idx]
        query_state = PackedQueryState(
            active_batch_idx=active_batch_idx,
            active_pos_idx=active_pos_idx,
            active_token_idx=active_token_idx,
            query_valid_mask=query_valid_mask,
            active_query_pos_padded=active_query_pos_padded,
            active_query_mask=query_valid_mask,
            all_valid=bool(torch.all(query_valid_mask).item()) if query_valid_mask.numel() > 0 else True,
        )
        return update_indice, query_state

    @staticmethod
    def _pad_active_tokens(
        x_active: torch.Tensor,
        x_template: torch.Tensor,
        fixed_query_state: QueryStateLike,
    ) -> torch.Tensor:
        if fixed_query_state.all_valid:
            return x_active.to(dtype=x_template.dtype).view(x_template.shape)

        x_padded = torch.zeros_like(x_template)
        x_padded[fixed_query_state.active_batch_idx, fixed_query_state.active_pos_idx] = x_active.to(
            dtype=x_template.dtype
        )
        return x_padded

    def _resolve_mobile_patch_scores(
        self,
        mobile_pscore: str,
        mobile_pscore_hint: torch.Tensor | None,
        dindice_patches: torch.Tensor,
        *,
        num_pretokens: int,
        num_tokens: int,
    ) -> torch.Tensor | None:
        if mobile_pscore not in self._MOBILE_HINT_PSCORE_ALIASES or mobile_pscore_hint is None:
            return None

        if mobile_pscore_hint.shape[0] != dindice_patches.shape[0]:
            raise RuntimeError(
                f"Mobile pscore hint batch mismatch: hint={tuple(mobile_pscore_hint.shape)} "
                f"candidates={tuple(dindice_patches.shape)}"
            )

        if mobile_pscore_hint.shape[1] == (num_tokens - num_pretokens):
            gather_idx = dindice_patches - num_pretokens
            return mobile_pscore_hint.gather(1, gather_idx)

        if mobile_pscore_hint.shape[1] == num_tokens:
            return mobile_pscore_hint.gather(1, dindice_patches)

        raise RuntimeError(
            f"Unsupported mobile pscore hint shape {tuple(mobile_pscore_hint.shape)} for "
            f"num_tokens={num_tokens}, num_pretokens={num_pretokens}"
        )

    @staticmethod
    def _combine_patch_scores(
        server_patch_scores: torch.Tensor,
        server_pscore_weight: float,
        mobile_patch_scores: torch.Tensor | None,
        mobile_pscore_weight: float,
        pscore_fusion: str,
    ) -> torch.Tensor:
        combined_patch_scores = server_pscore_weight * server_patch_scores
        if mobile_patch_scores is None or mobile_pscore_weight == 0.0:
            return combined_patch_scores

        mobile_term = mobile_pscore_weight * mobile_patch_scores.to(dtype=combined_patch_scores.dtype)
        if pscore_fusion == "multiply":
            return combined_patch_scores * mobile_term
        if pscore_fusion == "geo_mean":
            return torch.sqrt(torch.clamp(combined_patch_scores * mobile_term, min=0.0))
        return combined_patch_scores + mobile_term

    @staticmethod
    def _resolve_server_token_scores(
        cache_feature: Dict,
        tag: str,
        server_pscore: str,
    ) -> torch.Tensor:
        if server_pscore in {"patch_attn_prob_layermean", "cls_attn_prob_layermean"}:
            cache_key = "_shared_server_pscore_mean_all_layers"
            signature_key = "_shared_server_pscore_mean_all_layers_keys"
            server_pscore_keys = tuple(
                sorted(
                    key
                    for key, value in cache_feature.items()
                    if key.endswith("_server_pscore") and isinstance(value, torch.Tensor)
                )
            )
            if not server_pscore_keys:
                raise KeyError("Missing cached *_server_pscore entries. They must be produced during approx.")

            cached_server_pscore = cache_feature.get(cache_key)
            cached_signature = cache_feature.get(signature_key)
            if cached_server_pscore is not None and cached_signature == server_pscore_keys:
                return cached_server_pscore

            server_pscore_tensors = [cache_feature[key].to(dtype=torch.float32) for key in server_pscore_keys]
            base_shape = server_pscore_tensors[0].shape
            if any(tensor.shape != base_shape for tensor in server_pscore_tensors[1:]):
                raise RuntimeError(
                    f"Cannot average server pscores across layers with inconsistent shapes: "
                    f"{[(key, tuple(cache_feature[key].shape)) for key in server_pscore_keys]}"
                )

            shared_server_pscore = torch.stack(server_pscore_tensors, dim=0).mean(dim=0)
            cache_feature[cache_key] = shared_server_pscore.detach()
            cache_feature[signature_key] = server_pscore_keys
            return shared_server_pscore

        server_token_scores = cache_feature.get(f"{tag}_server_pscore")
        if server_token_scores is None:
            raise KeyError(
                f"Missing cached {tag}_server_pscore. "
                "It must be produced during approx."
            )
        return server_token_scores

    @staticmethod
    def _accumulate_token_pscore_coverage(
        cache_feature: Dict,
        combined_patch_scores: torch.Tensor,
        keep_patch_mask: torch.Tensor,
    ) -> None:
        kept_pscore_mass = (
            combined_patch_scores * keep_patch_mask.to(dtype=combined_patch_scores.dtype)
        ).sum(dtype=torch.float32)
        full_pscore_mass = combined_patch_scores.sum(dtype=torch.float32)
        cache_feature["_token_pscore_kept_mass_total"] = (
            cache_feature.get(
                "_token_pscore_kept_mass_total",
                kept_pscore_mass.new_zeros((), dtype=torch.float32),
            )
            + kept_pscore_mass
        )
        cache_feature["_token_pscore_full_mass_total"] = (
            cache_feature.get(
                "_token_pscore_full_mass_total",
                full_pscore_mass.new_zeros((), dtype=torch.float32),
            )
            + full_pscore_mass
        )

    @staticmethod
    def _accumulate_partial_token_patch_stats(
        cache_feature: Dict,
        keep_patch_mask: torch.Tensor,
    ) -> None:
        kept_patch_total = keep_patch_mask.sum(dtype=torch.float32)
        full_patch_total = keep_patch_mask.new_full(
            (),
            float(keep_patch_mask.shape[0] * keep_patch_mask.shape[1]),
            dtype=torch.float32,
        )
        sample_total = keep_patch_mask.new_full(
            (),
            float(keep_patch_mask.shape[0]),
            dtype=torch.float32,
        )
        cache_feature["_partial_token_kept_patch_total"] = (
            cache_feature.get(
                "_partial_token_kept_patch_total",
                kept_patch_total.new_zeros((), dtype=torch.float32),
            )
            + kept_patch_total
        )
        cache_feature["_partial_token_full_patch_total"] = (
            cache_feature.get(
                "_partial_token_full_patch_total",
                full_patch_total.new_zeros((), dtype=torch.float32),
            )
            + full_patch_total
        )
        cache_feature["_partial_token_sample_total"] = (
            cache_feature.get(
                "_partial_token_sample_total",
                sample_total.new_zeros((), dtype=torch.float32),
            )
            + sample_total
        )

    def correct_partial_token(
            self, x: torch.Tensor, dindice: List[int], rope: Tuple[torch.Tensor], cache_feature: Dict, tag: str, **kwargs
    ) -> List[Tensor]:
        debug = kwargs.get("debug", False)
        token_keep_ratio = kwargs.get("token_keep_ratio", 0.2)
        token_keep_thres = self._resolve_token_keep_threshold(kwargs)
        server_pscore_weight = float(kwargs.get("server_pscore_weight", 1.0))
        server_pscore = str(kwargs.get("server_pscore", "cls_attn_prob"))
        mobile_pscore = str(kwargs.get("mobile_pscore", "none"))
        mobile_pscore_weight = float(kwargs.get("mobile_pscore_weight", 0.0))
        mobile_pscore_hint = kwargs.get("mobile_pscore_hint")
        pscore_fusion = str(kwargs.get("pscore_fusion", "add")).lower()
        if pscore_fusion not in {"add", "multiply", "geo_mean"}:
            pscore_fusion = "add"
        valid_server_pscores = {
            "cls_attn_prob",
            "patch_attn_prob",
            "patch_attn_prob_layermean",
            "cls_attn_prob_layermean",
        }
        if server_pscore not in valid_server_pscores:
            raise ValueError(
                f"Unknown server_pscore '{server_pscore}'. "
                f"Available values: {sorted(valid_server_pscores)}"
            )

        # create update index
        B, N, C = x.shape
        num_pretokens = N - (rope[0].shape[0])
        server_token_scores = self._resolve_server_token_scores(
            cache_feature,
            tag,
            server_pscore,
        )

        dindice_pre = dindice[:, :num_pretokens]      # [B, 5] Shared pretokens
        dindice_patches = dindice[:, num_pretokens:]  # [B, M] Shared candidate patches

        server_patch_scores = server_token_scores.gather(1, dindice_patches) # [B, M]

        mobile_patch_scores = None
        if mobile_pscore != "none" and mobile_pscore_weight != 0.0:
            mobile_patch_scores = self._resolve_mobile_patch_scores(
                mobile_pscore,
                mobile_pscore_hint,
                dindice_patches,
                num_pretokens=num_pretokens,
                num_tokens=N,
            )
            if mobile_patch_scores is not None:
                cache_feature[f"{tag}_mobile_pscore_sel"] = mobile_patch_scores.detach()
        combined_patch_scores = self._combine_patch_scores(
            server_patch_scores,
            server_pscore_weight,
            mobile_patch_scores,
            mobile_pscore_weight,
            pscore_fusion,
        )
        cache_feature[f"{tag}_server_pscore_sel"] = server_patch_scores.detach()
        cache_feature[f"{tag}_combined_pscore"] = combined_patch_scores.detach()

        keep_patch_mask = self._select_patch_keep_mask(
            combined_patch_scores,
            token_keep_ratio,
            token_keep_thres,
        )
        self._accumulate_token_pscore_coverage(
            cache_feature,
            combined_patch_scores,
            keep_patch_mask,
        )
        self._accumulate_partial_token_patch_stats(
            cache_feature,
            keep_patch_mask,
        )
        update_indice, fixed_query_state = self._build_packed_query_state(
            dindice_pre,
            dindice_patches,
            keep_patch_mask,
        )
        cache_feature[f"{tag}_update_indice"] = update_indice.detach().clone()
        
        with torch.cuda.nvtx.range("correct_attn"):
            gather_idx_x = update_indice.unsqueeze(-1).expand(-1, -1, C)  # [B, num_update, C]
            x_sel = x.gather(1, gather_idx_x).contiguous()  # [B, num_update, C]
            x_norm_sel = self.norm1(self._select_active_tokens(x_sel, fixed_query_state))

            x_attn_sel, cache_feature = self.attn.correct(
                x_norm_sel,
                dindice=dindice,
                rope=rope,
                cache_feature=cache_feature,
                tag=tag,
                appcorr_method="partial_token",
                fixed_query_state=fixed_query_state,
            )
            x_sel = x_sel + self._pad_active_tokens(
                self.ls1(x_attn_sel),
                x_sel,
                fixed_query_state,
            )
            
            if debug:
                torch.cuda.synchronize()

        with torch.cuda.nvtx.range("correct_ffn"):
            blocks_out_sum = cache_feature[f"{tag}_blocks_out_sum"]
            x_attn_active = self._select_active_tokens(x_sel, fixed_query_state)
            mlp_out_new = self.ls2(self.mlp(self.norm2(x_attn_active)))
            x_ls2 = self._pad_active_tokens(
                mlp_out_new,
                x_sel,
                fixed_query_state,
            )

            x = masked_token_update_triton(
                x + blocks_out_sum.to(x.dtype),
                update_indice,
                x_sel,
                x_ls2,
                fixed_query_state.query_valid_mask,
            )

            if debug:
                torch.cuda.synchronize()

        return x, cache_feature

    def approx_partial_channel(
        self, x: torch.Tensor, rope: Tuple[torch.Tensor], cache_feature: Dict, tag: str, **kwargs
    ) -> List[Tensor]:
        attn_cache_candidates = kwargs.get("attn_cache_candidates")
        group_plans = kwargs.get("group_plans")
        attn_col_alive_ratio = kwargs.get("attn_col_alive_ratio", 1.0)

        with torch.cuda.nvtx.range("approx_attn"):
            shortcut1 = x
            x_norm1 = self.norm1(x)
            if group_plans:
                self._cache_group_slices(cache_feature, tag, "x_norm1", x_norm1, group_plans)

            x_attn, cache_feature = self.attn.approx_partial_channel(
                x_norm1,
                rope,
                cache_feature,
                tag,
                attn_cache_candidates=attn_cache_candidates,
                attn_col_alive_ratio=attn_col_alive_ratio,
            )
            x_ls1 = self.ls1(x_attn)
            
            if group_plans:
                self._cache_group_slices(cache_feature, tag, "x_ls1", x_ls1, group_plans)
            cache_feature[f"{tag}_blocks_out_sum"] = x_ls1.detach().clone()

            x = shortcut1 + x_ls1

        with torch.cuda.nvtx.range("approx_ffn"):
            shortcut2 = x
            x_norm2 = self.norm2(x)
            x_mlp, cache_feature = self.mlp.approx_partial_channel(x_norm2, cache_feature, tag)
            x_ls2 = self.ls2(x_mlp)
            cache_feature[f"{tag}_blocks_out_sum"] += x_ls2.detach()
            x = shortcut2 + x_ls2

        return x, cache_feature

    @staticmethod
    def _cache_group_slices(
        cache_feature: Dict,
        tag: str,
        cache_name: str,
        x_cache: torch.Tensor,
        group_plans: Dict[int, object],
    ) -> None:
        if torch.is_floating_point(x_cache) and x_cache.dtype != torch.bfloat16:
            x_cache = x_cache.to(dtype=torch.bfloat16)
        first_plan = next(iter(group_plans.values()))
        cache_feature[f"{tag}_{cache_name}_prefix"] = x_cache[:, :first_plan.num_pretokens].detach().clone()
        for gid, plan in group_plans.items():
            cache_feature[f"{tag}_{cache_name}_full_dindice_g{gid}"] = plan.full_dindice.detach().clone()
            if plan.group_patch_dindice.shape[1] > 0:
                group_idx = plan.group_patch_dindice.unsqueeze(-1).expand(-1, -1, x_cache.shape[-1])
                cache_feature[f"{tag}_{cache_name}_g{gid}"] = x_cache.gather(1, group_idx).detach().clone()
            else:
                cache_feature[f"{tag}_{cache_name}_g{gid}"] = x_cache[:, :0].detach().clone()

    @staticmethod
    def _pop_group_cached_tensor(
        cache_feature: Dict,
        tag: str,
        cache_name: str,
        group_id: int,
        group_plan: object,
    ) -> torch.Tensor | None:
        prefix_key = f"{tag}_{cache_name}_prefix"
        group_key = f"{tag}_{cache_name}_g{group_id}"
        dindice_key = f"{tag}_{cache_name}_full_dindice_g{group_id}"
        if prefix_key not in cache_feature or group_key not in cache_feature or dindice_key not in cache_feature:
            missing_keys = [
                key
                for key in (prefix_key, group_key, dindice_key)
                if key not in cache_feature
            ]
            raise KeyError(
                f"Missing split cache for {tag}/{cache_name}/group {group_id}: {missing_keys}"
            )

        cached_dindice = cache_feature[dindice_key]
        if cached_dindice.shape != group_plan.full_dindice.shape or not torch.equal(cached_dindice, group_plan.full_dindice):
            raise RuntimeError(
                f"Stale split cache for {tag}/{cache_name}/group {group_id}: "
                f"cached full dindice shape={tuple(cached_dindice.shape)} "
                f"current full shape={tuple(group_plan.full_dindice.shape)}"
            )

        group_tensor = cache_feature.pop(group_key)
        cache_feature.pop(dindice_key, None)
        keep_local_idx = group_plan.group_patch_keep_local_idx
        if keep_local_idx.shape[1] > 0:
            gather_idx = keep_local_idx.unsqueeze(-1).expand(-1, -1, group_tensor.shape[-1])
            group_tensor = group_tensor.gather(1, gather_idx)
        else:
            group_tensor = group_tensor[:, :0]
        return torch.cat([cache_feature[prefix_key], group_tensor], dim=1)

    @staticmethod
    def _select_active_tokens(x: torch.Tensor, fixed_query_state: QueryStateLike) -> torch.Tensor:
        if fixed_query_state.all_valid:
            return x.reshape(-1, x.shape[-1])
        return x[fixed_query_state.active_batch_idx, fixed_query_state.active_pos_idx]

    def _apply_attn_delta(
        self,
        x_sel: torch.Tensor,
        x_ls1_sel_old: torch.Tensor,
        dx_attn: torch.Tensor,
        fixed_query_state: QueryStateLike,
    ) -> torch.Tensor:
        if fixed_query_state.all_valid:
            dx_ls1 = self.ls1(dx_attn).view(x_sel.shape)
            return x_sel + x_ls1_sel_old.to(dtype=x_sel.dtype) + dx_ls1.to(dtype=x_sel.dtype)

        dx_ls1 = torch.zeros_like(x_sel)
        dx_ls1[fixed_query_state.active_batch_idx, fixed_query_state.active_pos_idx] = self.ls1(dx_attn)
        return masked_residual_add_triton(x_sel, x_ls1_sel_old, dx_ls1, fixed_query_state.query_valid_mask)

    def _build_ffn_delta(
        self,
        x_attn_sel: torch.Tensor,
        cache_feature: Dict,
        tag: str,
        fixed_query_state: QueryStateLike,
    ) -> tuple[torch.Tensor, Dict]:
        x_attn_active = self._select_active_tokens(x_attn_sel, fixed_query_state)
        x_mlp_active, cache_feature = self.mlp.correct_partial_channel(
            self.norm2(x_attn_active),
            cache_feature,
            tag,
        )
        if fixed_query_state.all_valid:
            return self.ls2(x_mlp_active).view(x_attn_sel.shape), cache_feature

        x_ls2 = torch.zeros_like(x_attn_sel)
        x_ls2[fixed_query_state.active_batch_idx, fixed_query_state.active_pos_idx] = self.ls2(x_mlp_active)
        return x_ls2, cache_feature
    
    def correct_partial_channel(
            self,
            x: torch.Tensor,
            dindice: List[int],
            rope: Tuple[torch.Tensor],
            cache_feature: Dict,
            tag: str,
            *,
            fixed_query_state: QueryStateLike,
            group_plan: object,
            attn_col_alive_ratio: float = 1.0,
            attn_cache_key=None,
    ) -> List[Tensor]:
        with torch.cuda.nvtx.range("correct_attn"):
            blocks_out_sum = cache_feature[f"{tag}_blocks_out_sum"]
            x_base = x + blocks_out_sum.to(x.dtype)

            dindice_sel = dindice
            x_norm1_old = self._pop_group_cached_tensor(cache_feature, tag, "x_norm1", attn_cache_key, group_plan)
            x_ls1_old = self._pop_group_cached_tensor(cache_feature, tag, "x_ls1", attn_cache_key, group_plan)
            gather_idx_x = dindice_sel.unsqueeze(-1).expand(-1, -1, x.shape[-1])
            x_sel = x.gather(1, gather_idx_x).contiguous()
            x_norm1_sel = self.norm1(x_sel)
            x_norm1_sel_old = x_norm1_old.to(dtype=x_norm1_sel.dtype)
            dx_norm1 = x_norm1_sel - x_norm1_sel_old
            dx_norm1_active = self._select_active_tokens(dx_norm1, fixed_query_state)

            dx_attn, cache_feature = self.attn.correct_partial_channel(
                dx_norm1_active,
                rope,
                cache_feature,
                tag,
                fixed_query_state,
                attn_col_alive_ratio=attn_col_alive_ratio,
                attn_cache_key=attn_cache_key,
                all_valid_queries=fixed_query_state.all_valid,
            )
            if x_ls1_old.shape[1] != dindice_sel.shape[1]:
                raise RuntimeError(
                    f"Split x_ls1 cache shape mismatch for {tag}/group {attn_cache_key}: "
                    f"cached={tuple(x_ls1_old.shape)} current={tuple(x_sel.shape)}"
                )
            x_ls1_sel_old = x_ls1_old.to(dtype=x_sel.dtype)
            x_attn_sel = self._apply_attn_delta(x_sel, x_ls1_sel_old, dx_attn, fixed_query_state)

        with torch.cuda.nvtx.range("correct_ffn"):
            x_ls2, cache_feature = self._build_ffn_delta(
                x_attn_sel,
                cache_feature,
                tag,
                fixed_query_state,
            )
            x = masked_token_update_triton(
                x_base,
                dindice_sel,
                x_attn_sel,
                x_ls2,
                fixed_query_state.query_valid_mask,
            )

        return x, cache_feature


class CausalSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        ffn_ratio: float = 4.0,
        ls_init_value: Optional[float] = None,
        is_causal: bool = True,
        act_layer: Callable = nn.GELU,
        norm_layer: Callable = nn.LayerNorm,
        dropout_prob: float = 0.0,
    ):
        super().__init__()

        self.dim = dim
        self.is_causal = is_causal
        self.ls1 = LayerScale(dim, init_values=ls_init_value) if ls_init_value else nn.Identity()
        self.attention_norm = norm_layer(dim)
        self.attention = CausalSelfAttention(dim, num_heads, attn_drop=dropout_prob, proj_drop=dropout_prob)

        self.ffn_norm = norm_layer(dim)
        ffn_hidden_dim = int(dim * ffn_ratio)
        self.feed_forward = Mlp(
            in_features=dim,
            hidden_features=ffn_hidden_dim,
            drop=dropout_prob,
            act_layer=act_layer,
        )

        self.ls2 = LayerScale(dim, init_values=ls_init_value) if ls_init_value else nn.Identity()

    def init_weights(
        self,
        init_attn_std: float | None = None,
        init_proj_std: float | None = None,
        init_fc_std: float | None = None,
        factor: float = 1.0,
    ) -> None:
        init_attn_std = init_attn_std or (self.dim**-0.5)
        init_proj_std = init_proj_std or init_attn_std * factor
        init_fc_std = init_fc_std or (2 * self.dim) ** -0.5
        self.attention.init_weights(init_attn_std, init_proj_std)
        self.attention_norm.reset_parameters()
        nn.init.normal_(self.feed_forward.fc1.weight, std=init_fc_std)
        nn.init.normal_(self.feed_forward.fc2.weight, std=init_proj_std)
        self.ffn_norm.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
    ):

        x_attn = x + self.ls1(self.attention(self.attention_norm(x), self.is_causal))
        x_ffn = x_attn + self.ls2(self.feed_forward(self.ffn_norm(x_attn)))
        return x_ffn
