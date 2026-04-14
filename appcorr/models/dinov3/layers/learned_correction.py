from __future__ import annotations

import os
from typing import Any, Dict, Iterator, List, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LightweightTokenMixer(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.pointwise = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.norm(x)
        x = self.depthwise(x.transpose(1, 2)).transpose(1, 2)
        x = self.act(x)
        x = self.pointwise(x)
        x = self.drop(x)
        return residual + x


class AttnDeltaPredictor(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        bottleneck_dim: int,
        mixer_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        proj_dim = max(int(bottleneck_dim), 1)
        mixer_layers = max(int(mixer_layers), 1)

        self.in_proj = nn.Linear(dim * 2, proj_dim)
        self.mixers = nn.ModuleList(
            LightweightTokenMixer(proj_dim, dropout=dropout) for _ in range(mixer_layers)
        )
        self.hidden = nn.Linear(proj_dim, max(int(hidden_dim), proj_dim))
        self.out_proj = nn.Linear(max(int(hidden_dim), proj_dim), dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, ln_old: Tensor, dln: Tensor) -> Tensor:
        x = torch.cat([ln_old, dln], dim=-1)
        x = F.gelu(self.in_proj(x))
        for mixer in self.mixers:
            x = mixer(x)
        x = self.drop(F.gelu(self.hidden(x)))
        return self.out_proj(x)


class FFNDeltaPredictor(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        bottleneck_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = max(int(hidden_dim), int(bottleneck_dim), 1)
        self.norm = nn.LayerNorm(dim * 2)
        self.fc1 = nn.Linear(dim * 2, inner_dim)
        self.fc2 = nn.Linear(inner_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, ln_old: Tensor, dln: Tensor) -> Tensor:
        x = torch.cat([ln_old, dln], dim=-1)
        x = self.norm(x)
        x = self.drop(F.gelu(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class BlockDeltaPredictor(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        bottleneck_dim: int,
        mixer_layers: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn_predictor = AttnDeltaPredictor(
            dim=dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            mixer_layers=mixer_layers,
            dropout=dropout,
        )
        self.ffn_predictor = FFNDeltaPredictor(
            dim=dim,
            hidden_dim=hidden_dim,
            bottleneck_dim=bottleneck_dim,
            dropout=dropout,
        )

    def forward(
        self,
        x_old: Tensor,
        dx_in: Tensor,
        attn_out_old: Tensor,
        *,
        norm1: nn.Module,
        norm2: nn.Module,
        ln1_old: Tensor | None = None,
        ln2_old: Tensor | None = None,
        h_old: Tensor | None = None,
    ) -> Dict[str, Tensor]:
        input_dtype = x_old.dtype
        model_dtype = self.attn_predictor.in_proj.weight.dtype

        x_new = x_old + dx_in
        if ln1_old is None:
            ln1_old = norm1(x_old)
        dln1 = norm1(x_new) - ln1_old
        dA_hat = self.attn_predictor(ln1_old.to(model_dtype), dln1.to(model_dtype)).to(dtype=input_dtype)

        if h_old is None:
            h_old = x_old + attn_out_old
        dh_hat = dx_in + dA_hat

        if ln2_old is None:
            ln2_old = norm2(h_old)
        dln2 = norm2(h_old + dh_hat) - ln2_old
        dM_hat = self.ffn_predictor(ln2_old.to(model_dtype), dln2.to(model_dtype)).to(dtype=input_dtype)
        dx_out_hat = dh_hat + dM_hat

        return {
            "dA_hat": dA_hat,
            "dh_hat": dh_hat,
            "dM_hat": dM_hat,
            "dx_out_hat": dx_out_hat,
            "dLN1": dln1,
            "dLN2": dln2,
        }


def learned_block_requested(appcorr_options: Dict[str, Any]) -> bool:
    correction_mode = str(appcorr_options.get("correction_mode", "exact"))
    has_ckpt = bool(
        appcorr_options.get("learned_checkpoint_path")
        or appcorr_options.get("learned_checkpoint_load_path")
        or appcorr_options.get("learned_checkpoint_save_path")
    )
    return correction_mode == "learned_block" or bool(appcorr_options.get("learned_train", False)) or has_ckpt


def get_learned_block_layers(
    appcorr_options: Dict[str, Any],
    *,
    max_layers: int | None = None,
) -> List[int]:
    raw_layers = appcorr_options.get("learned_correction_layers", [0])
    layers: List[int] = []
    for raw_layer in raw_layers:
        try:
            layer_idx = int(raw_layer)
        except (TypeError, ValueError):
            continue
        if layer_idx < 0:
            continue
        if max_layers is not None and layer_idx >= max_layers:
            continue
        if layer_idx not in layers:
            layers.append(layer_idx)
    return sorted(layers)


def supports_learned_block_layer(layer_idx: int, appcorr_options: Dict[str, Any]) -> bool:
    enabled_layers = get_learned_block_layers(appcorr_options)
    return int(layer_idx) in enabled_layers


def ensure_learned_block_predictors(vit_backbone: nn.Module, appcorr_options: Dict[str, Any]) -> None:
    if not learned_block_requested(appcorr_options):
        return

    hidden_dim = int(appcorr_options.get("learned_hidden_size", 512))
    bottleneck_dim = int(appcorr_options.get("learned_bottleneck_size", 256))
    mixer_layers = int(appcorr_options.get("learned_attn_mixer_layers", 1))
    dropout = float(appcorr_options.get("learned_dropout", 0.0))
    device = next(vit_backbone.parameters()).device

    enabled_layers = set(get_learned_block_layers(appcorr_options, max_layers=len(vit_backbone.blocks)))

    for layer_idx, block in enumerate(vit_backbone.blocks):
        if layer_idx not in enabled_layers:
            continue
        if getattr(block, "learned_block_delta", None) is None:
            dim = int(block.norm1.normalized_shape[0])
            block.learned_block_delta = BlockDeltaPredictor(
                dim=dim,
                hidden_dim=hidden_dim,
                bottleneck_dim=bottleneck_dim,
                mixer_layers=mixer_layers,
                dropout=dropout,
            )
        block.learned_block_delta.to(device=device, dtype=torch.float32)


def iter_learned_block_modules(vit_backbone: nn.Module) -> Iterator[Tuple[int, nn.Module]]:
    for layer_idx, block in enumerate(vit_backbone.blocks):
        module = getattr(block, "learned_block_delta", None)
        if module is not None:
            yield layer_idx, module


def collect_learned_block_state_dict(vit_backbone: nn.Module) -> Dict[str, Tensor]:
    state_dict: Dict[str, Tensor] = {}
    for layer_idx, module in iter_learned_block_modules(vit_backbone):
        prefix = f"blocks.{layer_idx}."
        for key, value in module.state_dict().items():
            state_dict[f"{prefix}{key}"] = value.detach().cpu()
    return state_dict


def load_learned_block_checkpoint(
    vit_backbone: nn.Module,
    checkpoint_path: str,
    *,
    strict: bool = True,
    map_location: str | torch.device = "cpu",
) -> Dict[str, Any]:
    checkpoint_path = os.path.expanduser(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("state_dict")
        if state_dict is None:
            state_dict = checkpoint.get("learned_correction_state_dict", checkpoint)
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported learned correction checkpoint format: {type(state_dict)}")

    for layer_idx, module in iter_learned_block_modules(vit_backbone):
        prefix = f"blocks.{layer_idx}."
        module_state = {
            key[len(prefix):]: value
            for key, value in state_dict.items()
            if key.startswith(prefix)
        }
        if module_state:
            module.load_state_dict(module_state, strict=strict)

    if isinstance(checkpoint, dict):
        return checkpoint
    return {"state_dict": state_dict}
