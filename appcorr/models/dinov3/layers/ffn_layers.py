# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

from typing import Callable, List, Optional

import torch.nn.functional as F
from torch import Tensor, nn
import torch

from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig

from ..utils import cat_keep_shapes, uncat_with_shapes
from ._triton_kernels import fused_swiglu_quant, quantize_input


def get_weight_and_scale(linear_module):
    """Float8Linear에서 Weight(FP8)와 Scale(Expanded)을 추출"""
    w_tensor = linear_module.weight
    w_data = w_tensor.qdata  # (Out, In)
    w_scale = w_tensor.scale # (1, 1) or (1, N)
    
    # Row-Wise Scaling 규칙(A_scale: Mx1 -> B_scale: 1xN) 맞추기
    if w_scale.numel() == 1:
        out_features = w_data.size(0)
        # 메모리 연속성 보장된 복사본 생성
        w_scale = w_scale.expand(1, out_features).contiguous()
        
    return w_data, w_scale

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

        self.is_fp8_prepared = False

    def prepare_fp8_weights(self):
        # Copy original weights to new Linear modules
        w1_temp = nn.Linear(self.w1.in_features, self.w1.out_features, bias=self.w1.bias is not None, device=self.w1.weight.device)
        w1_temp.weight.data.copy_(self.w1.weight.data)
        if self.w1.bias is not None: w1_temp.bias.data.copy_(self.w1.bias.data)

        w2_temp = nn.Linear(self.w2.in_features, self.w2.out_features, bias=self.w2.bias is not None, device=self.w2.weight.device)
        w2_temp.weight.data.copy_(self.w2.weight.data)
        if self.w2.bias is not None: w2_temp.bias.data.copy_(self.w2.bias.data)

        w3_temp = nn.Linear(self.w3.in_features, self.w3.out_features, bias=self.w3.bias is not None, device=self.w3.weight.device)
        w3_temp.weight.data.copy_(self.w3.weight.data)
        if self.w3.bias is not None: w3_temp.bias.data.copy_(self.w3.bias.data)

        # Quantization
        container = nn.ModuleDict({
            'w1': w1_temp,
            'w2': w2_temp,
            'w3': w3_temp
        })

        quantize_(container, Float8DynamicActivationFloat8WeightConfig())

        # Assign quantized modules
        self.w1_fp8 = container['w1']
        self.w2_fp8 = container['w2']
        self.w3_fp8 = container['w3']

        self.is_fp8_prepared = True

    def forward(self, x: Tensor) -> Tensor:
        with torch.cuda.nvtx.range("SwiGLUFFN"):
            x1 = self.w1(x)
            x2 = self.w2(x)
            hidden = F.silu(x1) * x2
            output = self.w3(hidden)

            torch.cuda.synchronize()
        return output

    def forward_fp8(self, x: torch.Tensor) -> torch.Tensor:
        with torch.cuda.nvtx.range("SwiGLUFFN-FP8-Full-Optimized"):
            if not self.is_fp8_prepared:
                self.prepare_fp8_weights()
            
            B, N, C = x.shape
            target_dtype = torch.bfloat16
            
            # Quantize input activation to FP8
            x_fp8, x_scale = quantize_input(x)
            
            # Load quantized weights and scales
            w1_data, w1_scale = get_weight_and_scale(self.w1_fp8)
            w2_data, w2_scale = get_weight_and_scale(self.w2_fp8)
            
            # w1 GEMM
            x1_flat = torch._scaled_mm(
                x_fp8, w1_data.t(), x_scale, w1_scale,
                out_dtype=target_dtype, use_fast_accum=True
            )
            if self.w1_fp8.bias is not None:
                x1_flat += self.w1_fp8.bias.to(target_dtype)

            # w2 GEMM
            x2_flat = torch._scaled_mm(
                x_fp8, w2_data.t(), x_scale, w2_scale,
                out_dtype=target_dtype, use_fast_accum=True
            )
            if self.w2_fp8.bias is not None:
                x2_flat += self.w2_fp8.bias.to(target_dtype)

            # SiLU + Elementwise Mul + Quantization
            x1_view = x1_flat.view(B, N, -1)
            x2_view = x2_flat.view(B, N, -1)
            hidden_fp8, hidden_scale = fused_swiglu_quant(x1_view, x2_view)
            
            # w3 GEMM
            w3_data, w3_scale = get_weight_and_scale(self.w3_fp8)
            
            output_flat = torch._scaled_mm(
                hidden_fp8, w3_data.t(), hidden_scale, w3_scale,
                out_dtype=target_dtype, use_fast_accum=True
            )
            if self.w3_fp8.bias is not None:
                output_flat += self.w3_fp8.bias.to(target_dtype)
            
            # Reshape output
            output = output_flat.view(B, N, -1)

        return output
