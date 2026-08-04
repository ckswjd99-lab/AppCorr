# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from typing import Callable, List, Optional, Dict, Tuple

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


class Mlp(nn.Module, ListForwardMixin):
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

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

    def correct_partial_channel(
        self,
        x: Tensor,
        cache_feature: Dict,
        tag: str,
    ) -> Tuple[Tensor, dict]:
        return self.forward(x), cache_feature


class SwiGLUFFN(nn.Module, ListForwardMixin):
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

    def _fused_swiglu_hidden(self, x: Tensor):
        """silu(w1(x)) * w2(x) with both FP4 epilogues folded in, or None if not applicable.

        w1 and w2 are the one pair in the block that share an output shape and are combined
        immediately, so their `out * scale + bias` epilogues can ride along with the SiLU and the
        multiply -- four kernels per block collapsing into one. FP4's per-kernel fixed cost is what
        makes that worth doing; see fused_swiglu_epilogue_triton.
        """
        raw1 = getattr(self.w1, "forward_raw", None)
        raw2 = getattr(self.w2, "forward_raw", None)
        if raw1 is None or raw2 is None:
            return None
        r1, r2 = raw1(x), raw2(x)
        if r1 is None or r2 is None:
            return None
        from .triton_kernels import fused_swiglu_epilogue_triton

        o1, s1, b1, shape = r1
        o2, s2, b2, _ = r2
        out = fused_swiglu_epilogue_triton(o1, s1, b1, o2, s2, b2)
        return None if out is None else out.reshape(*shape[:-1], out.shape[-1])

    def forward(self, x: Tensor) -> Tensor:
        hidden = self._fused_swiglu_hidden(x)
        if hidden is not None:
            return self.w3(hidden)
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2
        output = self.w3(hidden)

        return output
    
    def approx_partial_channel(self, x: Tensor, cache_feature: Dict, tag: str) -> Tuple[Tensor, dict]:
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2
        output = self.w3(hidden)

        return output, cache_feature
    
    def correct_partial_channel(
        self,
        x: Tensor,
        cache_feature: Dict,
        tag: str,
    ) -> Tuple[Tensor, dict]:
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2
        output = self.w3(hidden)
        return output, cache_feature
