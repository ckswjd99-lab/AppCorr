#!/usr/bin/env python3
"""Evaluate low-rank predicted FFN block correction on ADE20K.

Modes are deliberately gated:

* ``parity`` checks that 100% FFN support reproduces dense token correction;
* ``diagnose`` compares static/rank-16/rank-32 selectors with an exact oracle;
* ``evaluate`` compares L0 full, token-50 dense, and token-50 predicted-block
  correction.  Dense masking emulates the sparse FFN arithmetic for accuracy;
  FLOP accounting describes a selected-block implementation.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from appcorr.models.dinov3.layers.ffn_block_selector import (
    LowRankSwiGLUFactors,
    build_joint_swiglu_low_rank_factors,
    exact_swiglu_delta_selected_blocks,
    low_rank_swiglu_channel_score,
    mask_diagnostics,
    select_ffn_block_mask,
    select_ffn_2to4_mask,
    select_ffn_row_topk_mask,
    static_swiglu_channel_score,
)
from offload.common.protocol import ExperimentConfig
from offload.server.model.dinov3_segmentor_m2f import DINOv3SegmentorM2FExecutor


INTERMEDIATE_LAYERS = (9, 19, 29, 39)
FULL_PREFIX_LAYERS = 2
FULL_SUFFIX_LAYERS = 2
SPECIAL_TOKENS = 5
NUM_CLASSES = 150
IGNORE_INDEX = 255
BASE_CANVAS_MODE = "native_pyramid_then_scale"


@dataclass
class SelectorTotals:
    selected_channels: int = 0
    oracle_channels: int = 0
    intersection_channels: int = 0
    total_energy: float = 0.0
    selected_energy: float = 0.0
    oracle_energy: float = 0.0
    delta_error_sq: float = 0.0
    delta_reference_sq: float = 0.0
    delta_dot: float = 0.0
    delta_selected_sq: float = 0.0
    calls: int = 0

    def update(
        self,
        mask: torch.Tensor,
        oracle_mask: torch.Tensor,
        oracle_score: torch.Tensor,
        selected_delta: torch.Tensor,
        dense_delta: torch.Tensor,
    ) -> None:
        predicted = mask.bool()
        oracle = oracle_mask.bool()
        self.selected_channels += int(predicted.sum().item())
        self.oracle_channels += int(oracle.sum().item())
        self.intersection_channels += int((predicted & oracle).sum().item())
        self.total_energy += float(oracle_score.double().sum().item())
        self.selected_energy += float(
            oracle_score.masked_fill(~predicted, 0).double().sum().item()
        )
        self.oracle_energy += float(
            oracle_score.masked_fill(~oracle, 0).double().sum().item()
        )
        selected_float = selected_delta.float()
        dense_float = dense_delta.float()
        self.delta_error_sq += float((selected_float - dense_float).square().sum().item())
        self.delta_reference_sq += float(dense_float.square().sum().item())
        self.delta_dot += float((selected_float * dense_float).sum().item())
        self.delta_selected_sq += float(selected_float.square().sum().item())
        self.calls += 1

    def summary(self) -> dict[str, float]:
        eps = torch.finfo(torch.float64).eps
        return {
            "calls": float(self.calls),
            "oracle_block_recall": self.intersection_channels / max(self.oracle_channels, 1),
            "total_energy_retained": self.selected_energy / max(self.total_energy, eps),
            "oracle_energy_retained": self.oracle_energy / max(self.total_energy, eps),
            "retained_energy_vs_oracle": self.selected_energy / max(self.oracle_energy, eps),
            "delta_y_relative_l2": math.sqrt(
                self.delta_error_sq / max(self.delta_reference_sq, eps)
            ),
            "delta_y_cosine": self.delta_dot
            / max(math.sqrt(self.delta_selected_sq * self.delta_reference_sq), eps),
        }


class AreaMetrics:
    def __init__(self) -> None:
        self.intersection = torch.zeros(NUM_CLASSES, dtype=torch.float64)
        self.union = torch.zeros(NUM_CLASSES, dtype=torch.float64)
        self.predicted = torch.zeros(NUM_CLASSES, dtype=torch.float64)
        self.label = torch.zeros(NUM_CLASSES, dtype=torch.float64)

    def add_areas(self, areas: Iterable[Iterable[float]]) -> None:
        values = [torch.as_tensor(value, dtype=torch.float64) for value in areas]
        self.intersection += values[0]
        self.union += values[1]
        self.predicted += values[2]
        self.label += values[3]

    def update(self, prediction: torch.Tensor, annotation: np.ndarray) -> list[list[float]]:
        pred = prediction.cpu().long()
        label = torch.from_numpy(np.array(annotation, dtype=np.uint8, copy=True)).long()
        if pred.shape != label.shape:
            pred = F.interpolate(
                pred[None, None].float(),
                size=tuple(label.shape),
                mode="nearest",
            )[0, 0].long()
        label = label.clone()
        label[label == IGNORE_INDEX] += 1
        label -= 1
        label[label == -1] = IGNORE_INDEX
        valid = label != IGNORE_INDEX
        pred = pred[valid]
        label = label[valid]
        intersect = pred[pred == label]
        area_intersection = torch.bincount(intersect, minlength=NUM_CLASSES)[:NUM_CLASSES].double()
        area_predicted = torch.bincount(pred, minlength=NUM_CLASSES)[:NUM_CLASSES].double()
        area_label = torch.bincount(label, minlength=NUM_CLASSES)[:NUM_CLASSES].double()
        area_union = area_predicted + area_label - area_intersection
        areas = [area_intersection, area_union, area_predicted, area_label]
        self.add_areas(areas)
        return [value.tolist() for value in areas]

    def summary(self) -> dict[str, float]:
        eps = torch.finfo(torch.float64).eps
        iou = self.intersection / self.union.clamp_min(eps)
        iou[self.union == 0] = torch.nan
        aacc = self.intersection.sum() / self.label.sum().clamp_min(eps)
        return {
            "mIoU": float(torch.nanmean(iou).item() * 100),
            "aAcc": float(aacc.item() * 100),
        }


def parse_ints(value: str) -> list[int]:
    result = [int(item) for item in value.split(",") if item.strip()]
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected positive comma-separated integers")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=(
            "parity",
            "diagnose",
            "pooled_diagnose",
            "evaluate",
            "oracle_evaluate",
            "predictor_evaluate",
            "pooled_evaluate",
        ),
        required=True,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-images", type=int)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--short-side", type=int, default=896)
    parser.add_argument("--crop-size", type=int, default=896)
    parser.add_argument("--stride", type=int, default=596)
    parser.add_argument("--base-level", type=int, default=2)
    parser.add_argument("--token-rate", type=float, default=0.5)
    parser.add_argument("--ffn-keep", type=float, default=0.5)
    parser.add_argument("--ffn-token-block", type=int, default=8)
    parser.add_argument("--ffn-channel-block", type=int, default=128)
    parser.add_argument("--ranks", type=parse_ints, default=parse_ints("16,32"))
    parser.add_argument("--selected-rank", type=int, default=16)
    parser.add_argument("--factor-oversample", type=int, default=8)
    parser.add_argument("--factor-power-iterations", type=int, default=1)
    parser.add_argument("--factor-seed", type=int, default=20260801)
    parser.add_argument(
        "--ffn-backend",
        choices=("dense_mask", "selected_reference"),
        default="dense_mask",
    )
    parser.add_argument("--max-crops-per-image", type=int)
    parser.add_argument("--bootstrap-samples", type=int, default=1000)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/ade20k_ffn_lowrank_block_selector"),
    )
    parser.add_argument(
        "--backbone-weights",
        default="~/cjpark/weights/dinov3/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
    )
    parser.add_argument(
        "--head-weights",
        default="~/cjpark/weights/dinov3/dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth",
    )
    args = parser.parse_args()
    defaults = {
        "parity": 2,
        "diagnose": 16,
        "pooled_diagnose": 16,
        "evaluate": 100,
        "oracle_evaluate": 100,
        "predictor_evaluate": 100,
        "pooled_evaluate": 100,
    }
    if args.num_images is None:
        args.num_images = defaults[args.mode]
    if not 0 < args.token_rate <= 1 or not 0 < args.ffn_keep <= 1:
        parser.error("token-rate and ffn-keep must be in (0, 1]")
    if args.mode == "parity":
        args.ffn_keep = 1.0
    if args.mode in {"diagnose", "pooled_diagnose"} and args.max_crops_per_image is None:
        args.max_crops_per_image = 1
    return args


def scaled_native_pyramid_level(
    original: np.ndarray,
    level: int,
    target_hw: tuple[int, int],
) -> np.ndarray:
    current = np.ascontiguousarray(original.astype(np.uint8, copy=False))
    for _ in range(level):
        current = cv2.pyrDown(current)
        current = np.ascontiguousarray(current.astype(np.uint8, copy=False))
    target_h, target_w = target_hw
    if current.shape[:2] != target_hw:
        current = np.asarray(
            Image.fromarray(current).resize(
                (target_w, target_h),
                Image.Resampling.BILINEAR,
            ),
            dtype=np.uint8,
        )
    return np.ascontiguousarray(current)


def load_executor(args: argparse.Namespace) -> DINOv3SegmentorM2FExecutor:
    config = ExperimentConfig(
        model_name="dinov3_segmentor_m2f",
        dataset_name="ade20k",
        batch_size=1,
        image_shape=(args.crop_size, args.crop_size, 3),
        patch_size=(16, 16),
        input_profile_name="dinov3_ade20k_m2f_official",
        input_profile_kwargs={
            "server_eval_mode": "single",
            "server_rescale_to": "original",
            "m2f_slide_merge_stage": "pre_head",
            "mobile_resize_short_side": args.short_side,
            "server_crop_size": args.crop_size,
            "server_stride": args.stride,
            "backbone_weights_path": str(Path(args.backbone_weights).expanduser()),
            "segmentor_head_weights_path": str(Path(args.head_weights).expanduser()),
            "autocast_dtype": "bfloat16",
        },
    )
    executor = DINOv3SegmentorM2FExecutor(torch.device(args.device))
    executor.load_model(config.model_name, config)
    return executor


def sliding_crops(height: int, width: int, crop_size: int, stride: int) -> list[tuple[int, int, int, int]]:
    crop_h = min(crop_size, height)
    crop_w = min(crop_size, width)
    h_grids = max(height - crop_h + stride - 1, 0) // stride + 1
    w_grids = max(width - crop_w + stride - 1, 0) // stride + 1
    result = []
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * stride
            x1 = w_idx * stride
            y2 = min(y1 + crop_h, height)
            x2 = min(x1 + crop_w, width)
            y1 = max(y2 - crop_h, 0)
            x1 = max(x2 - crop_w, 0)
            result.append((y1, y2, x1, x2))
    return result


def qkv(block: Any, state: torch.Tensor, rope: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    projected = block.attn.qkv(block.norm1(state))
    batch, tokens, _ = projected.shape
    hidden = block.attn.qkv.in_features
    projected = projected.reshape(
        batch,
        tokens,
        3,
        block.attn.num_heads,
        hidden // block.attn.num_heads,
    )
    query, key, value = (item.transpose(1, 2) for item in torch.unbind(projected, dim=2))
    if rope is not None:
        query, key = block.attn.apply_rope(query, key, rope)
    return query, key, value


def gather_tokens(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return value.gather(1, indices.unsqueeze(-1).expand(-1, -1, value.shape[-1]))


def gather_heads(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return value.gather(
        2,
        indices[:, None, :, None].expand(-1, value.shape[1], -1, value.shape[-1]),
    )


def scatter_tokens(base: torch.Tensor, indices: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    return base.clone().scatter(1, indices.unsqueeze(-1).expand_as(value), value.to(base.dtype))


def run_base(
    backbone: Any,
    state: torch.Tensor,
    rope: Any,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, dict[int, torch.Tensor]]:
    states = [state]
    attention_states = []
    score_sum = torch.zeros(state.shape[:2], device=state.device, dtype=torch.float32)
    raw_intermediates: dict[int, torch.Tensor] = {}
    cache: dict[str, Any] = {}
    for layer_index, block in enumerate(backbone.blocks):
        tag = f"base_layer{layer_index}"
        attention_output, cache = block.attn.approx(
            block.norm1(state),
            rope=rope,
            cache_feature=cache,
            tag=tag,
            appcorr_method="partial_token",
            server_pscore="patch_attn_prob_layermean",
        )
        attention_state = state + block.ls1(attention_output)
        state = attention_state + block.ls2(block.mlp(block.norm2(attention_state)))
        attention_states.append(attention_state)
        score_sum += cache.pop(f"{tag}_server_pscore").float()
        cache.pop(f"{tag}_kv", None)
        states.append(state)
        if layer_index in INTERMEDIATE_LAYERS:
            raw_intermediates[layer_index] = state
    return states, attention_states, score_sum / len(backbone.blocks), raw_intermediates


def run_full(backbone: Any, state: torch.Tensor, rope: Any) -> dict[int, torch.Tensor]:
    outputs: dict[int, torch.Tensor] = {}
    for layer_index, block in enumerate(backbone.blocks):
        state = block(state, rope)
        if layer_index in INTERMEDIATE_LAYERS:
            outputs[layer_index] = state
    return outputs


def token_support(
    full_crop: np.ndarray,
    base_crop: np.ndarray,
    attention_score: torch.Tensor,
    token_rate: float,
    patch_size: int = 16,
) -> torch.Tensor:
    height, width = full_crop.shape[:2]
    grid_h, grid_w = height // patch_size, width // patch_size
    residual = torch.from_numpy(
        full_crop[: grid_h * patch_size, : grid_w * patch_size].astype(np.float32)
        - base_crop[: grid_h * patch_size, : grid_w * patch_size].astype(np.float32)
    )
    energy = residual.square().mean(dim=-1)
    energy = energy.reshape(grid_h, patch_size, grid_w, patch_size).mean(dim=(1, 3)).flatten()
    received = attention_score[0, SPECIAL_TOKENS:].detach().cpu()
    if received.numel() != energy.numel():
        raise RuntimeError(
            f"patch score mismatch: attention={received.numel()} residual={energy.numel()}"
        )
    energy = energy / energy.max().clamp_min(torch.finfo(torch.float32).eps)
    received = received / received.max().clamp_min(torch.finfo(torch.float32).eps)
    combined = energy * received
    keep = max(1, min(combined.numel(), math.ceil(combined.numel() * token_rate)))
    selected = torch.topk(combined, k=keep).indices.to(attention_score.device) + SPECIAL_TOKENS
    special = torch.arange(SPECIAL_TOKENS, device=attention_score.device)
    return torch.cat((special, selected.sort().values)).unsqueeze(0)


def corrected_attention_dense(
    block: Any,
    base_state: torch.Tensor,
    base_attention_state: torch.Tensor,
    current_state: torch.Tensor,
    active_indices: torch.Tensor,
    rope: Any,
) -> torch.Tensor:
    q0, k0, v0 = qkv(block, base_state, rope)
    q1, k1, v1 = qkv(block, current_state, rope)
    q0 = gather_heads(q0, active_indices)
    q1 = gather_heads(q1, active_indices)
    scale = float(block.attn.scale)
    logits0 = torch.matmul(q0.float(), k0.float().transpose(-2, -1)) * scale
    logits1 = torch.matmul(q1.float(), k1.float().transpose(-2, -1)) * scale
    probability0 = torch.softmax(logits0, dim=-1).to(v0.dtype)
    probability1 = torch.softmax(logits1, dim=-1).to(v1.dtype)
    delta_raw = torch.matmul(probability1, v1) - torch.matmul(probability0, v0)
    delta_raw = delta_raw.transpose(1, 2).flatten(2)
    active_state = gather_tokens(current_state, active_indices)
    base_active = gather_tokens(base_state, active_indices)
    base_attn = gather_tokens(base_attention_state, active_indices)
    delta_projected = F.linear(delta_raw, block.attn.proj.weight, bias=None)
    return base_attn + (active_state - base_active) + block.ls1(delta_projected)


def exact_hidden_delta(block: Any, base_x: torch.Tensor, corrected_x: torch.Tensor) -> torch.Tensor:
    base_hidden = F.silu(block.mlp.w1(base_x)) * block.mlp.w2(base_x)
    corrected_hidden = F.silu(block.mlp.w1(corrected_x)) * block.mlp.w2(corrected_x)
    return corrected_hidden - base_hidden


def oracle_score_from_hidden(hidden_delta: torch.Tensor, down_weight: torch.Tensor) -> torch.Tensor:
    return hidden_delta.float().square() * down_weight.float().square().sum(dim=0)


def selected_delta_dense_mask(
    hidden_delta: torch.Tensor,
    down_weight: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    return F.linear(hidden_delta.masked_fill(~mask, 0), down_weight, bias=None)


def build_factors(
    backbone: Any,
    args: argparse.Namespace,
) -> dict[int, LowRankSwiGLUFactors]:
    max_rank = max([*args.ranks, args.selected_rank])
    result = {}
    middle_end = len(backbone.blocks) - FULL_SUFFIX_LAYERS
    for layer_index in range(FULL_PREFIX_LAYERS, middle_end):
        block = backbone.blocks[layer_index]
        result[layer_index] = build_joint_swiglu_low_rank_factors(
            block.mlp.w1.weight,
            block.mlp.w2.weight,
            rank=max_rank,
            oversample=args.factor_oversample,
            power_iterations=args.factor_power_iterations,
            seed=args.factor_seed + layer_index,
        )
        print(json.dumps({"factor_layer": layer_index, "rank": max_rank}), flush=True)
    return result


def selector_masks(
    block: Any,
    base_x: torch.Tensor,
    corrected_x: torch.Tensor,
    factors: LowRankSwiGLUFactors,
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    score, _ = low_rank_swiglu_channel_score(
        base_x,
        corrected_x,
        factors,
        block.mlp.w3.weight,
        gate_bias=block.mlp.w1.bias,
        up_bias=block.mlp.w2.bias,
    )
    return select_ffn_block_mask(
        score,
        keep_ratio=args.ffn_keep,
        token_block_size=args.ffn_token_block,
        channel_block_size=args.ffn_channel_block,
    )


def predictor_2to4_mask(
    block: Any,
    base_x: torch.Tensor,
    corrected_x: torch.Tensor,
    factors: LowRankSwiGLUFactors,
) -> torch.Tensor:
    """Predict a per-token 2:4 support from low-rank gate/up changes."""

    score, _ = low_rank_swiglu_channel_score(
        base_x,
        corrected_x,
        factors,
        block.mlp.w3.weight,
        gate_bias=block.mlp.w1.bias,
        up_bias=block.mlp.w2.bias,
    )
    return select_ffn_2to4_mask(score)


def patch_grid_side(total_tokens: int) -> int:
    patch_tokens = total_tokens - SPECIAL_TOKENS
    side = math.isqrt(patch_tokens)
    if side * side != patch_tokens or side % 2:
        raise ValueError(
            f"expected an even square patch grid, got {patch_tokens} patch tokens"
        )
    return side


def pool_patch_tokens_2x2(value: torch.Tensor) -> torch.Tensor:
    """Average prenorm patch-token inputs on a fixed spatial 2x2 grid."""

    side = patch_grid_side(value.shape[1])
    patch = value[:, SPECIAL_TOKENS:]
    patch_map = patch.reshape(
        value.shape[0],
        side,
        side,
        value.shape[-1],
    ).permute(0, 3, 1, 2)
    pooled = F.avg_pool2d(patch_map, kernel_size=2, stride=2)
    return pooled.permute(0, 2, 3, 1).reshape(
        value.shape[0],
        (side // 2) ** 2,
        value.shape[-1],
    )


def force_special_full(
    mask: torch.Tensor,
    active_indices: torch.Tensor,
) -> torch.Tensor:
    special = active_indices < SPECIAL_TOKENS
    return mask.masked_fill(special.unsqueeze(-1), True)


def expand_coarse_mask_to_active(
    coarse_mask: torch.Tensor,
    active_indices: torch.Tensor,
    total_tokens: int,
) -> torch.Tensor:
    side = patch_grid_side(total_tokens)
    coarse_side = side // 2
    channels = coarse_mask.shape[-1]
    coarse_map = coarse_mask.reshape(
        coarse_mask.shape[0],
        coarse_side,
        coarse_side,
        channels,
    ).permute(0, 3, 1, 2)
    fine_map = coarse_map.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)
    fine = fine_map.permute(0, 2, 3, 1).reshape(
        coarse_mask.shape[0],
        side * side,
        channels,
    )
    patch_indices = (active_indices - SPECIAL_TOKENS).clamp_min(0)
    gathered = fine.gather(
        1,
        patch_indices.unsqueeze(-1).expand(-1, -1, channels),
    )
    return force_special_full(gathered, active_indices)


def oracle_2x2_shared_mask(
    oracle_score: torch.Tensor,
    active_indices: torch.Tensor,
    total_tokens: int,
) -> torch.Tensor:
    """Choose one exact 2:4 mask shared by active tokens in each spatial 2x2."""

    side = patch_grid_side(total_tokens)
    coarse_side = side // 2
    patch_indices = active_indices - SPECIAL_TOKENS
    valid = patch_indices >= 0
    safe = patch_indices.clamp_min(0)
    row = safe // side
    column = safe % side
    block_index = (row // 2) * coarse_side + column // 2
    coarse_score = torch.zeros(
        oracle_score.shape[0],
        coarse_side * coarse_side,
        oracle_score.shape[-1],
        device=oracle_score.device,
        dtype=oracle_score.dtype,
    )
    coarse_score.scatter_add_(
        1,
        block_index.unsqueeze(-1).expand_as(oracle_score),
        oracle_score * valid.unsqueeze(-1),
    )
    coarse_mask = select_ffn_2to4_mask(coarse_score)
    return expand_coarse_mask_to_active(
        coarse_mask,
        active_indices,
        total_tokens,
    )


def pooled_input_2to4_masks(
    block: Any,
    base_x_full: torch.Tensor,
    corrected_x_full: torch.Tensor,
    active_indices: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Predict shared and interpolated per-token 2:4 supports on a 2x2 grid."""

    base_coarse = pool_patch_tokens_2x2(base_x_full)
    corrected_coarse = pool_patch_tokens_2x2(corrected_x_full)
    coarse_delta = exact_hidden_delta(block, base_coarse, corrected_coarse)
    coarse_score = oracle_score_from_hidden(coarse_delta, block.mlp.w3.weight)

    coarse_mask = select_ffn_2to4_mask(coarse_score)
    shared_mask = expand_coarse_mask_to_active(
        coarse_mask,
        active_indices,
        base_x_full.shape[1],
    )

    side = patch_grid_side(base_x_full.shape[1])
    coarse_side = side // 2
    channels = coarse_score.shape[-1]
    coarse_map = coarse_score.reshape(
        coarse_score.shape[0],
        coarse_side,
        coarse_side,
        channels,
    ).permute(0, 3, 1, 2)
    fine_map = F.interpolate(
        coarse_map,
        size=(side, side),
        mode="bilinear",
        align_corners=False,
    )
    fine_score = fine_map.permute(0, 2, 3, 1).reshape(
        coarse_score.shape[0],
        side * side,
        channels,
    )
    patch_indices = (active_indices - SPECIAL_TOKENS).clamp_min(0)
    active_score = fine_score.gather(
        1,
        patch_indices.unsqueeze(-1).expand(-1, -1, channels),
    )
    interpolated_mask = force_special_full(
        select_ffn_2to4_mask(active_score),
        active_indices,
    )
    return shared_mask, interpolated_mask



def run_policy(
    backbone: Any,
    full_initial: torch.Tensor,
    base_states: list[torch.Tensor],
    base_attention_states: list[torch.Tensor],
    rope: Any,
    active_indices: torch.Tensor,
    *,
    mode: str,
    factors: dict[int, LowRankSwiGLUFactors] | None,
    args: argparse.Namespace,
    diagnostics: dict[str, SelectorTotals] | None = None,
) -> dict[int, torch.Tensor]:
    state = full_initial
    outputs: dict[int, torch.Tensor] = {}
    middle_end = len(backbone.blocks) - FULL_SUFFIX_LAYERS
    for layer_index, block in enumerate(backbone.blocks):
        if layer_index < FULL_PREFIX_LAYERS or layer_index >= middle_end:
            state = block(state, rope)
        else:
            corrected_attn = corrected_attention_dense(
                block,
                base_states[layer_index],
                base_attention_states[layer_index],
                state,
                active_indices,
                rope,
            )
            base_attn = gather_tokens(base_attention_states[layer_index], active_indices)
            base_x = block.norm2(base_attn)
            corrected_x = block.norm2(corrected_attn)

            if mode == "dense":
                # Use the same finite-difference reconstruction as the sparse
                # path. At 100% support this is its exact arithmetic oracle; a
                # production dense correction can still evaluate f(x1)
                # directly with the three ordinary FFN projections.
                hidden_delta = exact_hidden_delta(block, base_x, corrected_x)
                delta = F.linear(hidden_delta, block.mlp.w3.weight, bias=None)
                base_next_active = gather_tokens(
                    base_states[layer_index + 1],
                    active_indices,
                )
                base_ffn_residual = base_next_active - base_attn
                corrected_next = (
                    corrected_attn + base_ffn_residual + block.ls2(delta)
                )
            else:
                hidden_delta = None
                if (
                    mode in {"diagnose", "pooled_diagnose"}
                    or mode.startswith(("oracle_", "pooled_"))
                    or args.ffn_backend == "dense_mask"
                ):
                    hidden_delta = exact_hidden_delta(block, base_x, corrected_x)

                if mode == "diagnose":
                    assert diagnostics is not None and factors is not None
                    assert hidden_delta is not None
                    oracle_score = oracle_score_from_hidden(
                        hidden_delta,
                        block.mlp.w3.weight,
                    )
                    oracle_block_mask, _ = select_ffn_block_mask(
                        oracle_score,
                        keep_ratio=args.ffn_keep,
                        token_block_size=args.ffn_token_block,
                        channel_block_size=args.ffn_channel_block,
                    )
                    oracle_2to4_mask = select_ffn_2to4_mask(oracle_score)
                    oracle_row_mask = select_ffn_row_topk_mask(
                        oracle_score,
                        keep_ratio=args.ffn_keep,
                    )
                    dense_delta = F.linear(
                        hidden_delta,
                        block.mlp.w3.weight,
                        bias=None,
                    )
                    masks = {
                        "static": select_ffn_block_mask(
                            static_swiglu_channel_score(
                                base_x,
                                block.mlp.w1.weight,
                                block.mlp.w2.weight,
                                block.mlp.w3.weight,
                            ),
                            keep_ratio=args.ffn_keep,
                            token_block_size=args.ffn_token_block,
                            channel_block_size=args.ffn_channel_block,
                        )[0],
                        "oracle_8x128": oracle_block_mask,
                        "oracle_2to4": oracle_2to4_mask,
                        "oracle_row50": oracle_row_mask,
                    }
                    for rank in args.ranks:
                        rank_factors = factors[layer_index].truncate(rank)
                        masks[f"rank{rank}_8x128"] = selector_masks(
                            block,
                            base_x,
                            corrected_x,
                            rank_factors,
                            args,
                        )[0]
                        masks[f"rank{rank}_2to4"] = predictor_2to4_mask(
                            block,
                            base_x,
                            corrected_x,
                            rank_factors,
                        )
                    for name, mask in masks.items():
                        selected_delta = selected_delta_dense_mask(
                            hidden_delta,
                            block.mlp.w3.weight,
                            mask,
                        )
                        if name.startswith("oracle_"):
                            reference_mask = mask
                        elif name.endswith("_2to4"):
                            reference_mask = oracle_2to4_mask
                        else:
                            reference_mask = oracle_block_mask
                        diagnostics[name].update(
                            mask,
                            reference_mask,
                            oracle_score,
                            selected_delta,
                            dense_delta,
                        )
                    base_next_active = gather_tokens(
                        base_states[layer_index + 1],
                        active_indices,
                    )
                    base_ffn_residual = base_next_active - base_attn
                    corrected_next = (
                        corrected_attn
                        + base_ffn_residual
                        + block.ls2(dense_delta)
                    )
                elif mode == "pooled_diagnose":
                    assert diagnostics is not None and hidden_delta is not None
                    oracle_score = oracle_score_from_hidden(
                        hidden_delta,
                        block.mlp.w3.weight,
                    )
                    oracle_token_mask = force_special_full(
                        select_ffn_2to4_mask(oracle_score),
                        active_indices,
                    )
                    oracle_shared_mask = oracle_2x2_shared_mask(
                        oracle_score,
                        active_indices,
                        base_states[layer_index].shape[1],
                    )
                    corrected_attn_full = scatter_tokens(
                        base_attention_states[layer_index],
                        active_indices,
                        corrected_attn,
                    )
                    base_x_full = block.norm2(base_attention_states[layer_index])
                    corrected_x_full = block.norm2(corrected_attn_full)
                    pooled_shared_mask, pooled_interpolated_mask = (
                        pooled_input_2to4_masks(
                            block,
                            base_x_full,
                            corrected_x_full,
                            active_indices,
                        )
                    )
                    dense_delta = F.linear(
                        hidden_delta,
                        block.mlp.w3.weight,
                        bias=None,
                    )
                    masks = {
                        "oracle_token_2to4": oracle_token_mask,
                        "oracle_2x2_shared": oracle_shared_mask,
                        "pooled_2x2_shared": pooled_shared_mask,
                        "pooled_2x2_interpolated": pooled_interpolated_mask,
                    }
                    for name, mask in masks.items():
                        selected_delta = selected_delta_dense_mask(
                            hidden_delta,
                            block.mlp.w3.weight,
                            mask,
                        )
                        reference_mask = (
                            oracle_shared_mask
                            if name.endswith("_shared")
                            else oracle_token_mask
                        )
                        patch = slice(SPECIAL_TOKENS, None)
                        diagnostics[name].update(
                            mask[:, patch],
                            reference_mask[:, patch],
                            oracle_score[:, patch],
                            selected_delta[:, patch],
                            dense_delta[:, patch],
                        )
                    base_next_active = gather_tokens(
                        base_states[layer_index + 1],
                        active_indices,
                    )
                    base_ffn_residual = base_next_active - base_attn
                    corrected_next = (
                        corrected_attn
                        + base_ffn_residual
                        + block.ls2(dense_delta)
                    )

                elif mode.startswith("oracle_"):
                    assert hidden_delta is not None
                    oracle_score = oracle_score_from_hidden(
                        hidden_delta,
                        block.mlp.w3.weight,
                    )
                    if mode == "oracle_8x128":
                        mask, _ = select_ffn_block_mask(
                            oracle_score,
                            keep_ratio=args.ffn_keep,
                            token_block_size=args.ffn_token_block,
                            channel_block_size=args.ffn_channel_block,
                        )
                    elif mode == "oracle_2to4":
                        mask = select_ffn_2to4_mask(oracle_score)
                    elif mode == "oracle_2to4_special_full":
                        mask = force_special_full(
                            select_ffn_2to4_mask(oracle_score),
                            active_indices,
                        )
                    elif mode == "oracle_2x2_shared":
                        mask = oracle_2x2_shared_mask(
                            oracle_score,
                            active_indices,
                            base_states[layer_index].shape[1],
                        )
                    elif mode == "oracle_row50":
                        mask = select_ffn_row_topk_mask(
                            oracle_score,
                            keep_ratio=args.ffn_keep,
                        )
                    else:
                        raise ValueError(f"unknown oracle policy: {mode}")
                    delta = selected_delta_dense_mask(
                        hidden_delta,
                        block.mlp.w3.weight,
                        mask,
                    )
                    base_next_active = gather_tokens(
                        base_states[layer_index + 1],
                        active_indices,
                    )
                    base_ffn_residual = base_next_active - base_attn
                    corrected_next = (
                        corrected_attn + base_ffn_residual + block.ls2(delta)
                    )
                elif mode in {
                    "pooled_2x2_shared",
                    "pooled_2x2_interpolated",
                }:
                    assert hidden_delta is not None
                    corrected_attn_full = scatter_tokens(
                        base_attention_states[layer_index],
                        active_indices,
                        corrected_attn,
                    )
                    base_x_full = block.norm2(base_attention_states[layer_index])
                    corrected_x_full = block.norm2(corrected_attn_full)
                    shared_mask, interpolated_mask = pooled_input_2to4_masks(
                        block,
                        base_x_full,
                        corrected_x_full,
                        active_indices,
                    )
                    mask = (
                        shared_mask
                        if mode == "pooled_2x2_shared"
                        else interpolated_mask
                    )
                    delta = selected_delta_dense_mask(
                        hidden_delta,
                        block.mlp.w3.weight,
                        mask,
                    )
                    base_next_active = gather_tokens(
                        base_states[layer_index + 1],
                        active_indices,
                    )
                    base_ffn_residual = base_next_active - base_attn
                    corrected_next = (
                        corrected_attn + base_ffn_residual + block.ls2(delta)
                    )

                elif mode == "predictor_2to4":
                    assert factors is not None and hidden_delta is not None
                    mask = predictor_2to4_mask(
                        block,
                        base_x,
                        corrected_x,
                        factors[layer_index].truncate(args.selected_rank),
                    )
                    delta = selected_delta_dense_mask(
                        hidden_delta,
                        block.mlp.w3.weight,
                        mask,
                    )
                    base_next_active = gather_tokens(
                        base_states[layer_index + 1],
                        active_indices,
                    )
                    base_ffn_residual = base_next_active - base_attn
                    corrected_next = (
                        corrected_attn + base_ffn_residual + block.ls2(delta)
                    )
                else:
                    assert factors is not None
                    mask, _ = selector_masks(
                        block,
                        base_x,
                        corrected_x,
                        factors[layer_index].truncate(args.selected_rank),
                        args,
                    )
                    if args.ffn_backend == "dense_mask":
                        assert hidden_delta is not None
                        delta = selected_delta_dense_mask(
                            hidden_delta,
                            block.mlp.w3.weight,
                            mask,
                        )
                    else:
                        delta = exact_swiglu_delta_selected_blocks(
                            base_x,
                            corrected_x,
                            block.mlp.w1.weight,
                            block.mlp.w2.weight,
                            block.mlp.w3.weight,
                            mask,
                            gate_bias=block.mlp.w1.bias,
                            up_bias=block.mlp.w2.bias,
                            token_block_size=args.ffn_token_block,
                        )
                    base_next_active = gather_tokens(
                        base_states[layer_index + 1],
                        active_indices,
                    )
                    base_ffn_residual = base_next_active - base_attn
                    corrected_next = (
                        corrected_attn + base_ffn_residual + block.ls2(delta)
                    )
            state = scatter_tokens(base_states[layer_index + 1], active_indices, corrected_next)
        if layer_index in INTERMEDIATE_LAYERS:
            outputs[layer_index] = state
    return outputs


def source_context(source: dict[str, Any], raw: dict[int, torch.Tensor]) -> dict[str, Any]:
    return {
        "m2f_intermediate_raw": [[raw[index] for index in INTERMEDIATE_LAYERS]],
        "m2f_spm_c_cat": [source["spm_c_cat"]],
        "m2f_spm_c1_raw": [source["spm_c1_raw"]],
        "m2f_spm_c2_len": [source["spm_c2_len"]],
        "m2f_spm_c3_len": [source["spm_c3_len"]],
        "m2f_source_shapes": [source["source_shape"]],
        "m2f_deform_in1": [source["deform_in1"]],
        "m2f_deform_in2": [source["deform_in2"]],
    }


def add_crop_features(
    executor: DINOv3SegmentorM2FExecutor,
    sums: dict[str, torch.Tensor],
    counts: dict[str, torch.Tensor],
    source: dict[str, Any],
    raw: dict[int, torch.Tensor],
    crop: tuple[int, int, int, int],
    image_hw: tuple[int, int],
) -> None:
    features = executor._run_adapter_postprocess(0, source_context(source, raw))
    executor._accumulate_m2f_adapter_feature_crop(sums, counts, features, crop, image_hw)


def finish_prediction(
    executor: DINOv3SegmentorM2FExecutor,
    sums: dict[str, torch.Tensor],
    counts: dict[str, torch.Tensor],
    image_hw: tuple[int, int],
    original_hw: tuple[int, int],
) -> torch.Tensor:
    features = {key: sums[key] / counts[key].clamp_min(1) for key in sums}
    logits = executor._run_m2f_head_predict(
        executor.model.segmentation_model[1],
        features,
        "m2f",
        rescale_to=image_hw,
    )
    logits = F.interpolate(logits.float(), size=original_hw, mode="bilinear", align_corners=False)
    return torch.softmax(logits, dim=1).argmax(dim=1)[0].to(torch.uint8)


def logical_flops(backbone: Any, args: argparse.Namespace, tokens: int) -> dict[str, float]:
    block = backbone.blocks[0]
    hidden = int(block.mlp.w1.in_features)
    channels = int(block.mlp.w1.out_features)
    active = SPECIAL_TOKENS + math.ceil((tokens - SPECIAL_TOKENS) * args.token_rate)
    middle = len(backbone.blocks) - FULL_PREFIX_LAYERS - FULL_SUFFIX_LAYERS
    dense_ffn = middle * 6 * active * hidden * channels
    predictor = middle * (
        2 * active * hidden * args.selected_rank
        + 8 * active * channels * args.selected_rank
    )
    # Without caching full base gate/up, selected exact finite difference needs
    # base+corrected gate/up (four projections) and one selected down projection.
    selected_exact = middle * 10 * active * hidden * channels * args.ffn_keep
    ours = predictor + selected_exact
    return {
        "middle_dense_ffn_correction": float(dense_ffn),
        "middle_ours_predictor": float(predictor),
        "middle_ours_exact_selected": float(selected_exact),
        "middle_ours_total": float(ours),
        "ours_vs_dense_ffn": float(ours / dense_ffn),
        "ours_ffn_savings_percent": float((1 - ours / dense_ffn) * 100),
        "active_tokens": float(active),
    }


def cache_bytes(backbone: Any, args: argparse.Namespace, tokens: int) -> dict[str, float]:
    hidden = int(backbone.blocks[0].mlp.w1.in_features)
    channels = int(backbone.blocks[0].mlp.w1.out_features)
    middle = len(backbone.blocks) - FULL_PREFIX_LAYERS - FULL_SUFFIX_LAYERS
    bytes_per = 2
    return {
        "z0_rank_cache": float(middle * tokens * args.selected_rank * bytes_per),
        "base_x_cache": float(middle * tokens * hidden * bytes_per),
        "base_y_cache": float(middle * tokens * hidden * bytes_per),
        "avoided_full_gate_up_cache": float(middle * tokens * 2 * channels * bytes_per),
    }


def write_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()


def bootstrap_deltas(
    rows: list[dict[str, Any]],
    *,
    samples: int,
    seed: int,
) -> dict[str, Any]:
    if not rows or samples <= 0:
        return {}
    generator = np.random.default_rng(seed)
    names = tuple(rows[0]["areas"])
    methods = tuple(
        name for name in names if name not in {"full", "token50_dense"}
    )
    deltas = {
        method: {metric: [] for metric in ("mIoU", "aAcc")}
        for method in methods
    }
    for _ in range(samples):
        sampled = generator.integers(0, len(rows), size=len(rows))
        summaries = {}
        for name in names:
            metric = AreaMetrics()
            for index in sampled:
                metric.add_areas(rows[int(index)]["areas"][name])
            summaries[name] = metric.summary()
        for method in methods:
            for metric_name in ("mIoU", "aAcc"):
                deltas[method][metric_name].append(
                    summaries[method][metric_name]
                    - summaries["token50_dense"][metric_name]
                )
    return {
        method: {
            metric_name: {
                "mean": float(np.mean(values)),
                "ci95": [
                    float(np.quantile(values, 0.025)),
                    float(np.quantile(values, 0.975)),
                ],
            }
            for metric_name, values in metrics.items()
        }
        for method, metrics in deltas.items()
    }


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    executor = load_executor(args)
    adapter = executor.model.segmentation_model[0]
    backbone = adapter.backbone
    factors = (
        None
        if args.mode in {"oracle_evaluate", "pooled_diagnose", "pooled_evaluate"}
        else build_factors(backbone, args)
    )

    from datasets import load_dataset

    split = "train" if args.mode in {"diagnose", "pooled_diagnose"} else "validation"
    dataset = load_dataset("merve/scene_parse_150", split=split)
    end_index = min(args.start_index + args.num_images, len(dataset))
    metrics: dict[str, AreaMetrics] = defaultdict(AreaMetrics)
    diagnostic_totals: dict[str, SelectorTotals] = defaultdict(SelectorTotals)
    result_rows: list[dict[str, Any]] = []
    parity_errors = []
    start_time = time.time()

    for index in range(args.start_index, end_index):
        sample = dataset[index]
        original = sample["image"].convert("RGB")
        annotation = np.asarray(sample["annotation"], dtype=np.uint8)
        original_np = np.asarray(original, dtype=np.uint8)
        resized = executor._resize_short_side(original, args.short_side)
        resized_np = np.asarray(resized, dtype=np.uint8)
        image_hw = tuple(resized_np.shape[:2])
        base_canvas = scaled_native_pyramid_level(original_np, args.base_level, image_hw)
        original_hw = (original.height, original.width)
        crops = sliding_crops(*image_hw, args.crop_size, args.stride)
        if args.max_crops_per_image is not None:
            crops = crops[: args.max_crops_per_image]

        if args.mode in {"diagnose", "pooled_diagnose"}:
            names = []
        elif args.mode == "oracle_evaluate":
            names = [
                "full",
                "token50_dense",
                "token50_oracle_8x128",
                "token50_oracle_2to4",
                "token50_oracle_row50",
            ]
        elif args.mode == "predictor_evaluate":
            names = [
                "full",
                "token50_dense",
                "token50_oracle_2to4",
                "token50_predictor_2to4",
            ]
        elif args.mode == "pooled_evaluate":
            names = [
                "full",
                "token50_dense",
                "token50_oracle_2to4_special_full",
                "token50_oracle_2x2_shared",
                "token50_pooled_2x2_shared",
                "token50_pooled_2x2_interpolated",
            ]
        else:
            names = ["full", "token50_dense", "token50_ours"]
        feature_sums = {name: {} for name in names}
        feature_counts = {name: {} for name in names}

        for crop_index, crop in enumerate(crops):
            y1, y2, x1, x2 = crop
            full_np = np.ascontiguousarray(resized_np[y1:y2, x1:x2])
            base_np = np.ascontiguousarray(base_canvas[y1:y2, x1:x2])
            full_tensor = executor._pil_to_normalized_tensor(Image.fromarray(full_np))
            base_tensor = executor._pil_to_normalized_tensor(Image.fromarray(base_np))
            with torch.autocast("cuda", executor.autocast_dtype):
                source = executor._prepare_single_source(full_tensor, adapter, backbone)
                full_initial = source["x_backbone"]
                base_initial, _ = backbone.prepare_tokens_with_masks(base_tensor)
                base_states, base_attention_states, attention_score, _ = run_base(
                    backbone,
                    base_initial,
                    source["rope_sincos"],
                )
                active = token_support(full_np, base_np, attention_score, args.token_rate)

                if args.mode in {"diagnose", "pooled_diagnose"}:
                    run_policy(
                        backbone,
                        full_initial,
                        base_states,
                        base_attention_states,
                        source["rope_sincos"],
                        active,
                        mode=args.mode,
                        factors=factors,
                        args=args,
                        diagnostics=diagnostic_totals,
                    )
                else:
                    raw_by_name = {
                        "full": run_full(
                            backbone,
                            full_initial,
                            source["rope_sincos"],
                        ),
                        "token50_dense": run_policy(
                            backbone,
                            full_initial,
                            base_states,
                            base_attention_states,
                            source["rope_sincos"],
                            active,
                            mode="dense",
                            factors=None,
                            args=args,
                        ),
                    }
                    if args.mode == "oracle_evaluate":
                        for output_name, policy_mode in (
                            ("token50_oracle_8x128", "oracle_8x128"),
                            ("token50_oracle_2to4", "oracle_2to4"),
                            ("token50_oracle_row50", "oracle_row50"),
                        ):
                            raw_by_name[output_name] = run_policy(
                                backbone,
                                full_initial,
                                base_states,
                                base_attention_states,
                                source["rope_sincos"],
                                active,
                                mode=policy_mode,
                                factors=None,
                                args=args,
                            )
                    elif args.mode == "predictor_evaluate":
                        assert factors is not None
                        raw_by_name["token50_oracle_2to4"] = run_policy(
                            backbone,
                            full_initial,
                            base_states,
                            base_attention_states,
                            source["rope_sincos"],
                            active,
                            mode="oracle_2to4",
                            factors=None,
                            args=args,
                        )
                        raw_by_name["token50_predictor_2to4"] = run_policy(
                            backbone,
                            full_initial,
                            base_states,
                            base_attention_states,
                            source["rope_sincos"],
                            active,
                            mode="predictor_2to4",
                            factors=factors,
                            args=args,
                        )
                    elif args.mode == "pooled_evaluate":
                        for output_name, policy_mode in (
                            (
                                "token50_oracle_2to4_special_full",
                                "oracle_2to4_special_full",
                            ),
                            (
                                "token50_oracle_2x2_shared",
                                "oracle_2x2_shared",
                            ),
                            (
                                "token50_pooled_2x2_shared",
                                "pooled_2x2_shared",
                            ),
                            (
                                "token50_pooled_2x2_interpolated",
                                "pooled_2x2_interpolated",
                            ),
                        ):
                            raw_by_name[output_name] = run_policy(
                                backbone,
                                full_initial,
                                base_states,
                                base_attention_states,
                                source["rope_sincos"],
                                active,
                                mode=policy_mode,
                                factors=None,
                                args=args,
                            )
                    else:
                        assert factors is not None
                        ours_raw = run_policy(
                            backbone,
                            full_initial,
                            base_states,
                            base_attention_states,
                            source["rope_sincos"],
                            active,
                            mode="ours",
                            factors=factors,
                            args=args,
                        )
                        raw_by_name["token50_ours"] = ours_raw
                        if args.mode == "parity":
                            dense_raw = raw_by_name["token50_dense"]
                            for layer_index in INTERMEDIATE_LAYERS:
                                actual = ours_raw[layer_index].float()
                                expected = dense_raw[layer_index].float()
                                parity_errors.append(
                                    {
                                        "image": index,
                                        "crop": crop_index,
                                        "layer": layer_index,
                                        "max_abs": float(
                                            (actual - expected).abs().max().item()
                                        ),
                                        "relative_l2": float(
                                            (actual - expected).norm().item()
                                            / expected.norm().clamp_min(1e-12).item()
                                        ),
                                    }
                                )
                    for name, raw in raw_by_name.items():
                        add_crop_features(
                            executor,
                            feature_sums[name],
                            feature_counts[name],
                            source,
                            raw,
                            crop,
                            image_hw,
                        )
            del base_states, base_attention_states, source
            torch.cuda.empty_cache()

        if args.mode not in {"diagnose", "pooled_diagnose"}:
            areas = {}
            predictions = {}
            with torch.autocast("cuda", executor.autocast_dtype):
                for name in names:
                    predictions[name] = finish_prediction(
                        executor,
                        feature_sums[name],
                        feature_counts[name],
                        image_hw,
                        original_hw,
                    )
                    areas[name] = metrics[name].update(predictions[name], annotation)
            row = {"index": index, "areas": areas, "num_crops": len(crops)}
            result_rows.append(row)
            write_jsonl(args.output / "results.jsonl", row)

        progress = {
            "completed_index": index,
            "elapsed_seconds": time.time() - start_time,
        }
        if args.mode in {"diagnose", "pooled_diagnose"}:
            progress["selector"] = {
                name: totals.summary() for name, totals in diagnostic_totals.items()
            }
        else:
            progress["metrics"] = {name: value.summary() for name, value in metrics.items()}
        print(json.dumps(progress, sort_keys=True), flush=True)

    token_count = SPECIAL_TOKENS + (args.crop_size // 16) ** 2
    summary = {
        "configuration": {
            **vars(args),
            "output": str(args.output),
            "base_canvas_mode": BASE_CANVAS_MODE,
            "split": split,
            "full_prefix_layers": FULL_PREFIX_LAYERS,
            "full_suffix_layers": FULL_SUFFIX_LAYERS,
            "intermediate_layers": list(INTERMEDIATE_LAYERS),
        },
        "metrics": {name: value.summary() for name, value in metrics.items()},
        "selector": {
            name: totals.summary() for name, totals in diagnostic_totals.items()
        },
        "parity": {
            "errors": parity_errors,
            "max_relative_l2": max(
                (item["relative_l2"] for item in parity_errors),
                default=0.0,
            ),
            "max_abs": max((item["max_abs"] for item in parity_errors), default=0.0),
        },
        "logical_flops_per_crop": logical_flops(backbone, args, token_count),
        "cache_per_crop": cache_bytes(backbone, args, token_count),
        "bootstrap_ours_minus_dense": bootstrap_deltas(
            result_rows,
            samples=args.bootstrap_samples,
            seed=args.factor_seed,
        ),
        "elapsed_seconds": time.time() - start_time,
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "result": summary}, sort_keys=True))


if __name__ == "__main__":
    main()
