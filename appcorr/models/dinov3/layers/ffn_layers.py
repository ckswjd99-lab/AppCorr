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

    def approx_csr(self, x: Tensor, cache_feature: Dict, tag: str) -> Tuple[Tensor, dict]:
        # Same CSR scheme as SwiGLUFFN, single-gate variant: hidden = act(fc1 x).
        hidden = self.act(self.fc1(x))                            # [B, N, H]
        output = self.fc2(hidden)                                # [B, N, out]

        Hd = hidden.shape[-1]
        num_keep = max(1, int(Hd * 0.50))
        topk = hidden.abs().topk(num_keep, dim=-1).indices
        mask = torch.zeros_like(hidden, dtype=torch.bool)
        mask.scatter_(-1, topk, True)

        cache_feature[f"{tag}_ffn_mask"] = mask.detach()
        cache_feature[f"{tag}_ffn_hidden"] = hidden.detach()
        cache_feature[f"{tag}_ffn_output"] = output.detach()
        return output, cache_feature

    def correct_csr(
        self, x: Tensor, cache_feature: Dict, tag: str, active_index=None
    ) -> Tuple[Tensor, dict]:
        mask_full = cache_feature[f"{tag}_ffn_mask"]
        hidden_full = cache_feature[f"{tag}_ffn_hidden"]
        output_full = cache_feature[f"{tag}_ffn_output"]
        Hd = hidden_full.shape[-1]
        in_f = x.shape[-1]
        lead_shape = x.shape[:-1]

        if active_index is not None:
            abi, ati = active_index
            mask2d = mask_full[abi, ati]
            hidden_cache2d = hidden_full[abi, ati]
            output_cache2d = output_full[abi, ati]
        else:
            mask2d = mask_full.reshape(-1, Hd)
            hidden_cache2d = hidden_full.reshape(-1, Hd)
            output_cache2d = output_full.reshape(-1, output_full.shape[-1])
        M = mask2d.shape[0]

        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.reshape(M, in_f).float()
            mask_csr = mask2d.to(torch.float32).to_sparse_csr()
            w1t = self.fc1.weight.t().float()                     # [in, H]

            h_sp = torch.sparse.sampled_addmm(mask_csr, xf, w1t, beta=0.0, alpha=1.0)
            col_idx = mask_csr.col_indices()
            crow = mask_csr.crow_indices()
            row_idx = torch.repeat_interleave(
                torch.arange(M, device=x.device), crow[1:] - crow[:-1]
            )

            h_vals = h_sp.values()
            if self.fc1.bias is not None:
                h_vals = h_vals + self.fc1.bias.float()[col_idx]
            hidden_new_vals = self.act(h_vals)
            hidden_cache_vals = hidden_cache2d.float()[row_idx, col_idx]
            delta_vals = hidden_new_vals - hidden_cache_vals

            delta_coo = torch.sparse_coo_tensor(
                torch.stack([row_idx, col_idx], dim=0), delta_vals, (M, Hd)
            ).coalesce()
            w2t = self.fc2.weight.t().float()                     # [H, out]
            delta_out = torch.sparse.mm(delta_coo, w2t)

        out_dim = output_cache2d.shape[-1]
        output = output_cache2d.float() + delta_out
        output = output.to(x.dtype).reshape(*lead_shape, out_dim)
        return output, cache_feature


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

    def forward(self, x: Tensor) -> Tensor:
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

    def approx_csr(self, x: Tensor, cache_feature: Dict, tag: str) -> Tuple[Tensor, dict]:
        # Full SwiGLU forward, plus predict the sparse support of the hidden activation: per token
        # keep the top-50% hidden units by |hidden| (= |silu(w1 x) * (w2 x)|). Cache the boolean mask,
        # the full hidden, and the output; correct() only refreshes those selected units and adds the
        # delta through w3 (analogue of attention's CSR: mask over keys -> here over hidden units).
        x1 = self.w1(x)
        x2 = self.w2(x)
        hidden = F.silu(x1) * x2                                  # [B, N, H]
        output = self.w3(hidden)                                 # [B, N, out]

        Hd = hidden.shape[-1]
        num_keep = max(1, int(Hd * 0.50))
        topk = hidden.abs().topk(num_keep, dim=-1).indices        # [B, N, num_keep]
        mask = torch.zeros_like(hidden, dtype=torch.bool)         # [B, N, H]
        mask.scatter_(-1, topk, True)

        cache_feature[f"{tag}_ffn_mask"] = mask.detach()
        cache_feature[f"{tag}_ffn_hidden"] = hidden.detach()
        cache_feature[f"{tag}_ffn_output"] = output.detach()
        return output, cache_feature

    def correct_csr(
        self, x: Tensor, cache_feature: Dict, tag: str, active_index=None
    ) -> Tuple[Tensor, dict]:
        # Recompute only the selected top-50% hidden units (cached mask) from the fresh input x, via
        # sampled w1 / w2 matmuls, then push the delta vs the cached hidden through w3 as a sparse.mm:
        #   out_new = out_cache + sparse.mm( (hidden_new - hidden_cache)|_sel , w3ᵀ )
        # The unselected hidden units keep their cached contribution (already inside out_cache), so
        # only w3's selected columns are touched -- exactly like attention's sparse.mm(probs, v).
        # When active_index=(batch_idx, token_idx) is given (token pruning), x is the packed active set
        # [num_active, C] and the cached mask/hidden/output rows are gathered at those token positions.
        mask_full = cache_feature[f"{tag}_ffn_mask"]              # [B, N, H] bool
        hidden_full = cache_feature[f"{tag}_ffn_hidden"]         # [B, N, H]
        output_full = cache_feature[f"{tag}_ffn_output"]         # [B, N, out]
        Hd = hidden_full.shape[-1]
        in_f = x.shape[-1]
        lead_shape = x.shape[:-1]

        if active_index is not None:
            abi, ati = active_index
            mask2d = mask_full[abi, ati]                          # [M, H]
            hidden_cache2d = hidden_full[abi, ati]                # [M, H]
            output_cache2d = output_full[abi, ati]               # [M, out]
        else:
            mask2d = mask_full.reshape(-1, Hd)
            hidden_cache2d = hidden_full.reshape(-1, Hd)
            output_cache2d = output_full.reshape(-1, output_full.shape[-1])
        M = mask2d.shape[0]

        # torch.sparse CUDA kernels are fp32/fp64 only, so disable the surrounding bf16 autocast.
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.reshape(M, in_f).float()
            mask_csr = mask2d.to(torch.float32).to_sparse_csr()
            w1t = self.w1.weight.t().float()                      # [in, H]
            w2t = self.w2.weight.t().float()

            # Sampled pre-activations at the selected (token, hidden-unit) positions. Both share the
            # mask's CSR layout, so their .values() align positionally with mask_csr.col_indices().
            x1_sp = torch.sparse.sampled_addmm(mask_csr, xf, w1t, beta=0.0, alpha=1.0)
            x2_sp = torch.sparse.sampled_addmm(mask_csr, xf, w2t, beta=0.0, alpha=1.0)

            col_idx = mask_csr.col_indices()                      # [nnz]
            crow = mask_csr.crow_indices()
            row_idx = torch.repeat_interleave(
                torch.arange(M, device=x.device), crow[1:] - crow[:-1]
            )                                                     # [nnz]

            x1_vals = x1_sp.values()
            x2_vals = x2_sp.values()
            if self.w1.bias is not None:
                x1_vals = x1_vals + self.w1.bias.float()[col_idx]
            if self.w2.bias is not None:
                x2_vals = x2_vals + self.w2.bias.float()[col_idx]

            hidden_new_vals = F.silu(x1_vals) * x2_vals            # [nnz]
            hidden_cache_vals = hidden_cache2d.float()[row_idx, col_idx]
            delta_vals = hidden_new_vals - hidden_cache_vals

            delta_coo = torch.sparse_coo_tensor(
                torch.stack([row_idx, col_idx], dim=0), delta_vals, (M, Hd)
            ).coalesce()
            w3t = self.w3.weight.t().float()                      # [H, out]
            delta_out = torch.sparse.mm(delta_coo, w3t)           # [M, out]

        out_dim = output_cache2d.shape[-1]
        output = output_cache2d.float() + delta_out
        output = output.to(x.dtype).reshape(*lead_shape, out_dim)
        return output, cache_feature
