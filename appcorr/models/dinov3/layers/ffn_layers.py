# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from typing import Callable, List, Optional, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..utils import cat_keep_shapes, uncat_with_shapes


class ListForwardMixin(object):
    def forward(self, x: Tensor):
        raise NotImplementedError

    def forward_list(self, x_list: List[Tensor]) -> List[Tensor]:
        x_flat, shapes, num_tokens = cat_keep_shapes(x_list)
        x_flat = self.forward(x_flat)
        return uncat_with_shapes(x_flat, shapes, num_tokens)


class OutputProjL2MaskMixin(object):
    def _get_output_proj_weight(self) -> Tensor:
        raise NotImplementedError

    def _get_hidden_keep_mask(self, hidden_prune_ratio: float) -> Optional[Tensor]:
        if hidden_prune_ratio <= 0.0:
            return None

        output_proj_weight = self._get_output_proj_weight()
        cached_ratio = getattr(self, "_approx_hidden_prune_ratio", None)
        cached_mask = getattr(self, "_approx_hidden_keep_mask", None)
        if (
            cached_mask is None
            or cached_ratio is None
            or abs(float(cached_ratio) - float(hidden_prune_ratio)) > 1e-12
            or cached_mask.shape[0] != output_proj_weight.shape[1]
            or cached_mask.device != output_proj_weight.device
        ):
            # Use output projection column norms as a cheap hidden-channel importance proxy.
            channel_norms = output_proj_weight.detach().float().norm(p=2, dim=0)
            num_channels = channel_norms.numel()
            num_pruned = min(max(int(num_channels * hidden_prune_ratio), 0), num_channels)

            keep_mask = torch.ones(num_channels, device=output_proj_weight.device, dtype=torch.bool)
            if num_pruned > 0:
                prune_idx = torch.topk(channel_norms, k=num_pruned, largest=False).indices
                keep_mask[prune_idx] = False

            self._approx_hidden_prune_ratio = float(hidden_prune_ratio)
            self._approx_hidden_keep_mask = keep_mask

        return self._approx_hidden_keep_mask

    def _apply_hidden_prune(self, hidden: Tensor, hidden_prune_ratio: float) -> Tensor:
        keep_mask = self._get_hidden_keep_mask(hidden_prune_ratio)
        if keep_mask is None:
            return hidden
        return hidden * keep_mask.to(device=hidden.device, dtype=hidden.dtype)


class Mlp(nn.Module, ListForwardMixin, OutputProjL2MaskMixin):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
        device=None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias, device=device)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias, device=device)
        self.drop = nn.Dropout(drop)

    def _get_output_proj_weight(self) -> Tensor:
        return self.fc2.weight

    def _forward_impl(self, x: Tensor, hidden_prune_ratio: float = 0.0) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self._apply_hidden_prune(x, hidden_prune_ratio)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    def correct_partial_channel(
        self,
        x: Tensor,
        cache_feature: Dict,
        tag: str,
    ) -> Tuple[Tensor, dict]:
        return self.forward(x), cache_feature

    def approx_partial_channel(
        self,
        x: Tensor,
        cache_feature: Dict,
        tag: str,
        hidden_prune_ratio: float = 0.0,
    ) -> Tuple[Tensor, dict]:
        return self._forward_impl(x, hidden_prune_ratio=hidden_prune_ratio), cache_feature


class SwiGLUFFN(nn.Module, ListForwardMixin, OutputProjL2MaskMixin):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Optional[Callable[..., nn.Module]] = None,
        drop: float = 0.0,
        bias: bool = True,
        align_to: int = 8,
        device=None,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        d = int(hidden_features * 2 / 3)
        swiglu_hidden_features = d + (-d % align_to)
        self.w1 = nn.Linear(in_features, swiglu_hidden_features, bias=bias, device=device)
        self.w2 = nn.Linear(in_features, swiglu_hidden_features, bias=bias, device=device)
        self.w3 = nn.Linear(swiglu_hidden_features, out_features, bias=bias, device=device)

    def _get_output_proj_weight(self) -> Tensor:
        return self.w3.weight

    def _forward_impl(self, x: Tensor, hidden_prune_ratio: float = 0.0) -> Tensor:
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2
        hidden = self._apply_hidden_prune(hidden, hidden_prune_ratio)
        output = self.w3(hidden)
        return output

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
    
    def approx_partial_channel(
        self,
        x: Tensor,
        cache_feature: Dict,
        tag: str,
        hidden_prune_ratio: float = 0.0,
    ) -> Tuple[Tensor, dict]:
        return self._forward_impl(x, hidden_prune_ratio=hidden_prune_ratio), cache_feature
    
    def correct_partial_channel(
        self,
        x: Tensor,
        cache_feature: Dict,
        tag: str,
    ) -> Tuple[Tensor, dict]:
        return self.forward(x), cache_feature
