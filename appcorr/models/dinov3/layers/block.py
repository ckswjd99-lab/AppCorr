# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import re
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor, nn

from ..utils import cat_keep_shapes, uncat_with_shapes

from .attention import CausalSelfAttention, SelfAttention
from .ffn_layers import Mlp, SwiGLUFFN
from .learned_correction import supports_learned_block_layer
from .layer_scale import LayerScale  # , DropPath

torch._dynamo.config.automatic_dynamic_shapes = False
torch._dynamo.config.accumulated_cache_size_limit = 1024


class SelfAttentionBlock(nn.Module):
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
        self.learned_block_delta = None

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

    @staticmethod
    def _resolve_layer_idx(tag: str, kwargs: Dict) -> int:
        explicit_layer_idx = kwargs.get("layer_idx")
        if explicit_layer_idx is not None:
            return int(explicit_layer_idx)

        match = re.search(r"layer(\d+)$", tag)
        if match is None:
            return -1
        return int(match.group(1))

    def _should_use_learned_block_pair(self, tag: str, kwargs: Dict) -> bool:
        correction_mode = str(kwargs.get("correction_mode", "exact"))
        if correction_mode not in {"learned_block", "none"}:
            return False

        layer_idx = self._resolve_layer_idx(tag, kwargs)
        return supports_learned_block_layer(
            layer_idx,
            {"learned_correction_layers": kwargs.get("learned_correction_layers", [0])},
        )

    def forward_with_branch_outputs(self, x: Tensor, rope=None) -> Dict[str, Tensor]:
        ln1 = self.norm1(x)
        attn_out = self.ls1(self.attn(ln1, rope=rope))
        h = x + attn_out
        ln2 = self.norm2(h)
        ffn_out = self.ls2(self.mlp(ln2))
        out = h + ffn_out
        return {
            "x": x,
            "ln1": ln1,
            "attn_out": attn_out,
            "h": h,
            "ln2": ln2,
            "ffn_out": ffn_out,
            "out": out,
        }

    def predict_learned_block_delta(
        self,
        x_old: Tensor,
        x_new: Tensor,
        attn_out_old: Tensor,
        *,
        ln1_old: Tensor | None = None,
        ln2_old: Tensor | None = None,
        h_old: Tensor | None = None,
    ) -> Dict[str, Tensor]:
        if self.learned_block_delta is None:
            raise RuntimeError("learned_block_delta is not initialized for this block.")

        return self.learned_block_delta(
            x_old=x_old,
            dx_in=x_new - x_old,
            attn_out_old=attn_out_old,
            norm1=self.norm1,
            norm2=self.norm2,
            ln1_old=ln1_old,
            ln2_old=ln2_old,
            h_old=h_old,
        )

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
            if self._should_use_learned_block_pair(tag, kwargs):
                return self.approx_learned_block(x, rope, cache_feature, tag, **kwargs)
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
            if self._should_use_learned_block_pair(tag, kwargs):
                return self.correct_learned_block(x, dindice, rope, cache_feature, tag, **kwargs)
            appcorr_method = kwargs.get("appcorr_method", "partial_token")
            if appcorr_method == "partial_token":
                return self.correct_partial_token(x, dindice, rope, cache_feature, tag, **kwargs)
            if appcorr_method == "partial_channel":
                return self.correct_partial_channel(x, dindice, rope, cache_feature, tag, **kwargs)
            raise ValueError(
                f"Unknown SelfAttentionBlock.correct method '{appcorr_method}'. "
                "Available methods: partial_channel, partial_token"
            )

    def approx_learned_block(
        self,
        x: torch.Tensor,
        rope: Tuple[torch.Tensor],
        cache_feature: Dict,
        tag: str,
        **kwargs,
    ) -> List[Tensor]:
        outputs = self.forward_with_branch_outputs(x, rope=rope)
        cache_feature[f"{tag}_learned_x_old"] = outputs["x"].detach().clone()
        cache_feature[f"{tag}_learned_attn_out_old"] = outputs["attn_out"].detach().clone()
        cache_feature[f"{tag}_learned_h_old"] = outputs["h"].detach().clone()
        cache_feature[f"{tag}_learned_ln1_old"] = outputs["ln1"].detach().clone()
        cache_feature[f"{tag}_learned_ln2_old"] = outputs["ln2"].detach().clone()
        cache_feature[f"{tag}_learned_block_out_old"] = outputs["out"].detach().clone()
        return outputs["out"], cache_feature

    def correct_learned_block(
        self,
        x: torch.Tensor,
        dindice: List[int],
        rope: Tuple[torch.Tensor],
        cache_feature: Dict,
        tag: str,
        **kwargs,
    ) -> List[Tensor]:
        x_old = cache_feature[f"{tag}_learned_x_old"].to(device=x.device, dtype=x.dtype)
        attn_out_old = cache_feature[f"{tag}_learned_attn_out_old"].to(device=x.device, dtype=x.dtype)
        block_out_old = cache_feature[f"{tag}_learned_block_out_old"].to(device=x.device, dtype=x.dtype)

        if str(kwargs.get("correction_mode", "learned_block")) == "none":
            return block_out_old, cache_feature

        h_old = cache_feature.get(f"{tag}_learned_h_old")
        if h_old is not None:
            h_old = h_old.to(device=x.device, dtype=x.dtype)
        ln1_old = cache_feature.get(f"{tag}_learned_ln1_old")
        if ln1_old is not None:
            ln1_old = ln1_old.to(device=x.device, dtype=x.dtype)
        ln2_old = cache_feature.get(f"{tag}_learned_ln2_old")
        if ln2_old is not None:
            ln2_old = ln2_old.to(device=x.device, dtype=x.dtype)

        pred = self.predict_learned_block_delta(
            x_old=x_old,
            x_new=x,
            attn_out_old=attn_out_old,
            ln1_old=ln1_old,
            ln2_old=ln2_old,
            h_old=h_old,
        )
        return block_out_old + pred["dx_out_hat"], cache_feature

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

    def correct_partial_token(
            self, x: torch.Tensor, dindice: List[int], rope: Tuple[torch.Tensor], cache_feature: Dict, tag: str, **kwargs
    ) -> List[Tensor]:
        debug = kwargs.get("debug", False)
        token_keep_ratio = kwargs.get("token_keep_ratio", 0.2)
        server_pscore_weight = float(kwargs.get("server_pscore_weight", 1.0))
        mobile_pscore = str(kwargs.get("mobile_pscore", "none"))
        mobile_pscore_weight = float(kwargs.get("mobile_pscore_weight", 0.0))

        # create update index
        B, N, C = x.shape
        num_pretokens = N - (rope[0].shape[0])
        server_token_scores = cache_feature.get(f"{tag}_server_pscore")
        if server_token_scores is None:
            raise KeyError(
                f"Missing cached {tag}_server_pscore. "
                "It must be produced during approx."
            )

        dindice_pre = dindice[:, :num_pretokens]      # [B, 5] Shared pretokens
        dindice_patches = dindice[:, num_pretokens:]  # [B, M] Shared candidate patches

        server_patch_scores = server_token_scores.gather(1, dindice_patches) # [B, M]

        combined_patch_scores = server_pscore_weight * server_patch_scores
        if mobile_pscore != "none" and mobile_pscore_weight != 0.0:
            mobile_patch_scores = torch.zeros_like(server_patch_scores)
            combined_patch_scores = combined_patch_scores + (mobile_pscore_weight * mobile_patch_scores)
        cache_feature[f"{tag}_server_pscore_sel"] = server_patch_scores.detach()
        cache_feature[f"{tag}_combined_pscore"] = combined_patch_scores.detach()
        
        num_patch_candidates = dindice_patches.shape[1]
        k_refined = min(int(num_patch_candidates * token_keep_ratio), num_patch_candidates)
        if k_refined <= 0:
            selected_patch_indices = dindice_patches[:, :0]
        else:
            _, topk_local_idx = torch.topk(combined_patch_scores, k=k_refined, dim=1, largest=True) # [B, k_refined]
            selected_patch_indices = dindice_patches.gather(1, topk_local_idx)  # [B, k_refined]

        update_indice = torch.cat([dindice_pre, selected_patch_indices], dim=1)
        cache_feature[f"{tag}_update_indice"] = update_indice.detach().clone()
        
        with torch.cuda.nvtx.range("correct_attn"):
            gather_idx_x = update_indice.unsqueeze(-1).expand(-1, -1, C)  # [B, num_update, C]
            x_sel = x.gather(1, gather_idx_x).contiguous()  # [B, num_update, C]
            x_norm_sel = self.norm1(x_sel)

            x_attn_sel, cache_feature = self.attn.correct(
                x_norm_sel, dindice=dindice, rope=rope, cache_feature=cache_feature, tag=tag, appcorr_method="partial_token"
            )
            x_sel = x_sel + self.ls1(x_attn_sel)
            
            if debug:
                torch.cuda.synchronize()

        with torch.cuda.nvtx.range("correct_ffn"):
            blocks_out_sum = cache_feature[f"{tag}_blocks_out_sum"]

            mlp_out_new = self.ls2(self.mlp(self.norm2(x_sel)))

            x_sel = x_sel + mlp_out_new
            x = x + blocks_out_sum.to(x.dtype)
            x.scatter_(1, gather_idx_x, x_sel)

            if debug:
                torch.cuda.synchronize()

        return x, cache_feature

    def approx_partial_channel(
        self, x: torch.Tensor, rope: Tuple[torch.Tensor], cache_feature: Dict, tag: str, **kwargs
    ) -> List[Tensor]:
        with torch.cuda.nvtx.range("approx_attn"):
            x_norm1 = self.norm1(x)
            x_attn, cache_feature = self.attn.approx_partial_channel(
                x_norm1,
                rope,
                cache_feature,
                tag
            )
            x = x + self.ls1(x_attn)

        with torch.cuda.nvtx.range("approx_ffn"):
            x_mlp, cache_feature = self.mlp.approx_partial_channel(self.norm2(x), cache_feature, tag)
            x = x + self.ls2(x_mlp)

        return x, cache_feature

    def correct_partial_channel(
            self,
            x: torch.Tensor,
            dindice: List[int],
            rope: Tuple[torch.Tensor],
            cache_feature: Dict,
            tag: str,
            **kwargs
    ) -> List[Tensor]:
        with torch.cuda.nvtx.range("correct_attn"):
            x_norm1 = self.norm1(x)
            x_attn, cache_feature = self.attn.correct_partial_channel(
                x_norm1,
                dindice,
                rope,
                cache_feature,
                tag
            )
            x = x + self.ls1(x_attn)

        with torch.cuda.nvtx.range("correct_ffn"):
            x_mlp, cache_feature = self.mlp.correct_partial_channel(
                self.norm2(x),
                cache_feature,
                tag,
            )
            x = x + self.ls2(x_mlp)

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
