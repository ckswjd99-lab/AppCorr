#!/usr/bin/env python3
"""Evaluate low-rank weight factors in DINOv3 correction deltas.

The low-resolution approximate pass uses every original dense weight.  In the
middle transformer layers, the full-resolution correction reconstructs QKV as

    qkv_1 ~= qkv_0 + low_rank(norm1(x_1) - norm1(x_0)).

For FFN variants, the approximate gate/up/output values are reused:

    [gate_1, up_1] ~= [gate_0, up_0] + LR_gate_up(delta_norm2)
    hidden_1 = silu(gate_1) * up_1
    output_1 ~= output_0 + LR_down(hidden_1 - hidden_0).

All tokens are corrected. Attention, softmax, and output projection remain
dense. The first and last two transformer layers also remain fully dense.
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

from appcorr.models.dinov3.layers.low_rank import (
    LowRankLinearFactors,
    dense_linear_flop_ratio,
    factorize_linear_weight,
    factorize_linear_weight_activation_aware,
)
from offload.common.protocol import ExperimentConfig
from offload.server.model.dinov3_segmentor_m2f import DINOv3SegmentorM2FExecutor


INTERMEDIATE_LAYERS = (9, 19, 29, 39)
FULL_PREFIX_LAYERS = 2
FULL_SUFFIX_LAYERS = 2
NUM_CLASSES = 150
IGNORE_INDEX = 255
BASE_CANVAS_MODE = "native_pyramid_then_scale"
VALID_VARIANTS = ("qkv", "ffn", "qkv_ffn")


@dataclass(frozen=True)
class BaseFFNCache:
    norm: torch.Tensor
    gate: torch.Tensor
    up: torch.Tensor
    output: torch.Tensor


@dataclass
class ErrorAccumulator:
    squared_error: float = 0.0
    squared_reference: float = 0.0
    cosine_weighted: float = 0.0
    elements: int = 0
    samples: int = 0

    def update(self, actual: torch.Tensor, expected: torch.Tensor) -> None:
        actual_float = actual.float()
        expected_float = expected.float()
        error = actual_float - expected_float
        self.squared_error += float(error.square().sum().item())
        self.squared_reference += float(expected_float.square().sum().item())
        self.cosine_weighted += float(
            F.cosine_similarity(
                actual_float.flatten(),
                expected_float.flatten(),
                dim=0,
                eps=1e-12,
            ).item()
        )
        self.elements += expected.numel()
        self.samples += 1

    def summary(self) -> dict[str, float]:
        return {
            "relative_l2": math.sqrt(
                self.squared_error / max(self.squared_reference, 1e-24)
            ),
            "rmse": math.sqrt(self.squared_error / max(self.elements, 1)),
            "cosine": self.cosine_weighted / max(self.samples, 1),
            "samples": float(self.samples),
        }


class AreaMetrics:
    def __init__(self) -> None:
        self.intersection = torch.zeros(NUM_CLASSES, dtype=torch.float64)
        self.union = torch.zeros(NUM_CLASSES, dtype=torch.float64)
        self.predicted = torch.zeros(NUM_CLASSES, dtype=torch.float64)
        self.label = torch.zeros(NUM_CLASSES, dtype=torch.float64)

    def add_areas(self, areas: Iterable[Iterable[float]]) -> None:
        tensors = [torch.as_tensor(value, dtype=torch.float64) for value in areas]
        self.intersection += tensors[0]
        self.union += tensors[1]
        self.predicted += tensors[2]
        self.label += tensors[3]

    def update(
        self,
        prediction: torch.Tensor,
        annotation: np.ndarray,
    ) -> list[list[float]]:
        pred = prediction.cpu().long()
        label = torch.from_numpy(
            np.array(annotation, dtype=np.uint8, copy=True)
        ).long()
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
        area_intersection = torch.bincount(
            intersect,
            minlength=NUM_CLASSES,
        )[:NUM_CLASSES].double()
        area_predicted = torch.bincount(
            pred,
            minlength=NUM_CLASSES,
        )[:NUM_CLASSES].double()
        area_label = torch.bincount(
            label,
            minlength=NUM_CLASSES,
        )[:NUM_CLASSES].double()
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
            "mIoU": float(torch.nanmean(iou).item() * 100.0),
            "aAcc": float(aacc.item() * 100.0),
        }


def parse_ranks(value: str) -> list[int]:
    ranks = sorted({int(item) for item in value.split(",") if item.strip()})
    if not ranks or any(rank <= 0 for rank in ranks):
        raise argparse.ArgumentTypeError("ranks must be positive comma-separated integers")
    return ranks


def parse_variants(value: str) -> list[str]:
    variants = list(dict.fromkeys(item.strip() for item in value.split(",")))
    if not variants or any(item not in VALID_VARIANTS for item in variants):
        raise argparse.ArgumentTypeError(
            "variants must be comma-separated values from "
            f"{','.join(VALID_VARIANTS)}"
        )
    return variants


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-images", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--short-side", type=int, default=896)
    parser.add_argument("--crop-size", type=int, default=896)
    parser.add_argument("--stride", type=int, default=596)
    parser.add_argument("--base-level", type=int, default=2)
    parser.add_argument("--ranks", type=parse_ranks, default=parse_ranks("64,128"))
    parser.add_argument(
        "--variants",
        type=parse_variants,
        default=parse_variants("qkv"),
        help="Evaluate qkv, ffn, and/or qkv_ffn low-rank correction.",
    )
    parser.add_argument(
        "--factor-mode",
        choices=("weight_svd", "delta_rms"),
        default="weight_svd",
    )
    parser.add_argument("--factor-oversample", type=int, default=16)
    parser.add_argument("--factor-power-iters", type=int, default=1)
    parser.add_argument("--factor-rms-floor-ratio", type=float, default=1e-3)
    parser.add_argument(
        "--factor-dtype",
        choices=("bfloat16", "float32"),
        default="bfloat16",
    )
    parser.add_argument("--factor-seed", type=int, default=20260731)
    parser.add_argument("--calibration-split", default="train")
    parser.add_argument("--calibration-start-index", type=int, default=0)
    parser.add_argument("--calibration-images", type=int, default=4)
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument("--benchmark-warmup", type=int, default=3)
    parser.add_argument("--benchmark-iters", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/ade20k_qkv_low_rank"),
    )
    parser.add_argument(
        "--backbone-weights",
        default="~/cjpark/weights/dinov3/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
    )
    parser.add_argument(
        "--head-weights",
        default="~/cjpark/weights/dinov3/dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth",
    )
    return parser.parse_args()


def scaled_native_pyramid_level(
    original: np.ndarray,
    level: int,
    target_hw: tuple[int, int],
) -> np.ndarray:
    """Construct native Lk first, then resize the completed level."""

    if original.ndim != 3:
        raise ValueError(f"expected HWC image, got {tuple(original.shape)}")
    if level < 0:
        raise ValueError(f"pyramid level must be non-negative, got {level}")
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
    return np.ascontiguousarray(current.astype(np.uint8, copy=False))


def load_executor(
    args: argparse.Namespace,
) -> tuple[DINOv3SegmentorM2FExecutor, ExperimentConfig]:
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
    return executor, config


def sliding_crops(
    height: int,
    width: int,
    crop_size: int,
    stride: int,
) -> list[tuple[int, int, int, int]]:
    crop_h = min(crop_size, height)
    crop_w = min(crop_size, width)
    h_grids = max(height - crop_h + stride - 1, 0) // stride + 1
    w_grids = max(width - crop_w + stride - 1, 0) // stride + 1
    crops = []
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * stride
            x1 = w_idx * stride
            y2 = min(y1 + crop_h, height)
            x2 = min(x1 + crop_w, width)
            y1 = max(y2 - crop_h, 0)
            x1 = max(x2 - crop_w, 0)
            crops.append((y1, y2, x1, x2))
    return crops


def replace_spm_features(
    full_source: dict[str, Any],
    base_source: dict[str, Any],
) -> dict[str, Any]:
    result = dict(full_source)
    for key in ("spm_c_cat", "spm_c1_raw", "spm_c2_len", "spm_c3_len"):
        result[key] = base_source[key]
    return result


def run_base(
    backbone: Any,
    state: torch.Tensor,
    rope: Any,
) -> tuple[
    list[torch.Tensor],
    list[torch.Tensor],
    list[BaseFFNCache],
    dict[int, torch.Tensor],
]:
    states = [state]
    qkv_values = []
    ffn_values = []
    intermediates: dict[int, torch.Tensor] = {}
    for layer_index, block in enumerate(backbone.blocks):
        qkv0 = block.attn.qkv(block.norm1(state))
        attention_raw = block.attn.compute_attention(qkv0, rope=rope)
        attention_output = block.attn.proj_drop(block.attn.proj(attention_raw))
        attention_state = state + block.ls1(attention_output)
        ffn_norm = block.norm2(attention_state)
        gate = block.mlp.w1(ffn_norm)
        up = block.mlp.w2(ffn_norm)
        hidden = F.silu(gate) * up
        output = block.mlp.w3(hidden)
        state = attention_state + block.ls2(output)
        qkv_values.append(qkv0.detach())
        ffn_values.append(
            BaseFFNCache(
                norm=ffn_norm.detach(),
                gate=gate.detach(),
                up=up.detach(),
                output=output.detach(),
            )
        )
        states.append(state)
        if layer_index in INTERMEDIATE_LAYERS:
            intermediates[layer_index] = state
    return states, qkv_values, ffn_values, intermediates


def run_full(
    backbone: Any,
    state: torch.Tensor,
    rope: Any,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    intermediates: dict[int, torch.Tensor] = {}
    for layer_index, block in enumerate(backbone.blocks):
        state = block(state, rope)
        if layer_index in INTERMEDIATE_LAYERS:
            intermediates[layer_index] = state
    return state, intermediates


def run_qkv_delta_correction(
    backbone: Any,
    full_initial: torch.Tensor,
    base_states: list[torch.Tensor],
    base_qkv: list[torch.Tensor],
    rope: Any,
    factors: dict[int, LowRankLinearFactors],
    rank: int | None,
    error_accumulator: ErrorAccumulator | None = None,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    state = full_initial
    intermediates: dict[int, torch.Tensor] = {}
    middle_end = len(backbone.blocks) - FULL_SUFFIX_LAYERS
    for layer_index, block in enumerate(backbone.blocks):
        if layer_index < FULL_PREFIX_LAYERS or layer_index >= middle_end:
            state = block(state, rope)
        else:
            current_norm = block.norm1(state)
            base_norm = block.norm1(base_states[layer_index])
            delta_norm = current_norm - base_norm
            if rank is None:
                delta_qkv = F.linear(
                    delta_norm,
                    block.attn.qkv.weight,
                    bias=None,
                )
            else:
                delta_qkv = factors[layer_index].apply(delta_norm, rank)
                if error_accumulator is not None:
                    exact_delta = F.linear(
                        delta_norm,
                        block.attn.qkv.weight,
                        bias=None,
                    )
                    error_accumulator.update(delta_qkv, exact_delta)
            corrected_qkv = base_qkv[layer_index] + delta_qkv
            attention_raw = block.attn.compute_attention(corrected_qkv, rope=rope)
            attention_output = block.attn.proj_drop(block.attn.proj(attention_raw))
            attention_state = state + block.ls1(attention_output)
            state = attention_state + block.ls2(
                block.mlp(block.norm2(attention_state))
            )
        if layer_index in INTERMEDIATE_LAYERS:
            intermediates[layer_index] = state
    return state, intermediates


def run_low_rank_delta_correction(
    backbone: Any,
    full_initial: torch.Tensor,
    base_states: list[torch.Tensor],
    base_qkv: list[torch.Tensor],
    base_ffn: list[BaseFFNCache],
    rope: Any,
    *,
    qkv_factors: dict[int, LowRankLinearFactors],
    ffn_gate_factors: dict[int, LowRankLinearFactors],
    ffn_up_factors: dict[int, LowRankLinearFactors],
    ffn_down_factors: dict[int, LowRankLinearFactors],
    qkv_rank: int | None,
    ffn_rank: int | None,
    qkv_error_accumulator: ErrorAccumulator | None = None,
    ffn_gate_error_accumulator: ErrorAccumulator | None = None,
    ffn_up_error_accumulator: ErrorAccumulator | None = None,
    ffn_down_error_accumulator: ErrorAccumulator | None = None,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Run dense or low-rank QKV/FFN correction on the middle layers."""

    state = full_initial
    intermediates: dict[int, torch.Tensor] = {}
    middle_end = len(backbone.blocks) - FULL_SUFFIX_LAYERS
    for layer_index, block in enumerate(backbone.blocks):
        if layer_index < FULL_PREFIX_LAYERS or layer_index >= middle_end:
            state = block(state, rope)
        else:
            current_norm1 = block.norm1(state)
            base_norm1 = block.norm1(base_states[layer_index])
            delta_norm1 = current_norm1 - base_norm1
            if qkv_rank is None:
                delta_qkv = F.linear(
                    delta_norm1,
                    block.attn.qkv.weight,
                    bias=None,
                )
            else:
                delta_qkv = qkv_factors[layer_index].apply(
                    delta_norm1,
                    qkv_rank,
                )
                if qkv_error_accumulator is not None:
                    exact_delta_qkv = F.linear(
                        delta_norm1,
                        block.attn.qkv.weight,
                        bias=None,
                    )
                    qkv_error_accumulator.update(delta_qkv, exact_delta_qkv)
            corrected_qkv = base_qkv[layer_index] + delta_qkv
            attention_raw = block.attn.compute_attention(corrected_qkv, rope=rope)
            attention_output = block.attn.proj_drop(block.attn.proj(attention_raw))
            attention_state = state + block.ls1(attention_output)

            current_norm2 = block.norm2(attention_state)
            if ffn_rank is None:
                ffn_output = block.mlp(current_norm2)
            else:
                cache = base_ffn[layer_index]
                delta_norm2 = current_norm2 - cache.norm
                delta_gate = ffn_gate_factors[layer_index].apply(
                    delta_norm2,
                    ffn_rank,
                )
                delta_up = ffn_up_factors[layer_index].apply(
                    delta_norm2,
                    ffn_rank,
                )
                if ffn_gate_error_accumulator is not None:
                    ffn_gate_error_accumulator.update(
                        delta_gate,
                        F.linear(
                            delta_norm2,
                            block.mlp.w1.weight,
                            bias=None,
                        ),
                    )
                if ffn_up_error_accumulator is not None:
                    ffn_up_error_accumulator.update(
                        delta_up,
                        F.linear(
                            delta_norm2,
                            block.mlp.w2.weight,
                            bias=None,
                        ),
                    )
                corrected_gate = cache.gate + delta_gate
                corrected_up = cache.up + delta_up
                base_hidden = F.silu(cache.gate) * cache.up
                corrected_hidden = F.silu(corrected_gate) * corrected_up
                delta_hidden = corrected_hidden - base_hidden
                delta_output = ffn_down_factors[layer_index].apply(
                    delta_hidden,
                    ffn_rank,
                )
                if ffn_down_error_accumulator is not None:
                    exact_delta_output = F.linear(
                        delta_hidden,
                        block.mlp.w3.weight,
                        bias=None,
                    )
                    ffn_down_error_accumulator.update(
                        delta_output,
                        exact_delta_output,
                    )
                ffn_output = cache.output + delta_output
            state = attention_state + block.ls2(ffn_output)
        if layer_index in INTERMEDIATE_LAYERS:
            intermediates[layer_index] = state
    return state, intermediates


def accumulate_dense_delta_rms(
    backbone: Any,
    full_initial: torch.Tensor,
    base_states: list[torch.Tensor],
    base_qkv: list[torch.Tensor],
    base_ffn: list[BaseFFNCache],
    rope: Any,
    qkv_squared_sums: dict[int, torch.Tensor],
    ffn_squared_sums: dict[int, torch.Tensor],
    down_squared_sums: dict[int, torch.Tensor],
    sample_counts: dict[int, int],
) -> torch.Tensor:
    """Collect RMS for QKV, gate/up, and down correction inputs."""

    state = full_initial
    middle_end = len(backbone.blocks) - FULL_SUFFIX_LAYERS
    for layer_index, block in enumerate(backbone.blocks):
        if layer_index < FULL_PREFIX_LAYERS or layer_index >= middle_end:
            state = block(state, rope)
            continue

        current_norm = block.norm1(state)
        base_norm = block.norm1(base_states[layer_index])
        delta_norm = current_norm - base_norm
        delta_float = delta_norm.float()
        qkv_squared_sums[layer_index].add_(
            delta_float.square().sum(dim=(0, 1)).to(torch.float64)
        )

        delta_qkv = F.linear(
            delta_norm,
            block.attn.qkv.weight,
            bias=None,
        )
        corrected_qkv = base_qkv[layer_index] + delta_qkv
        attention_raw = block.attn.compute_attention(corrected_qkv, rope=rope)
        attention_output = block.attn.proj_drop(block.attn.proj(attention_raw))
        attention_state = state + block.ls1(attention_output)
        current_norm2 = block.norm2(attention_state)
        delta_norm2 = current_norm2 - base_ffn[layer_index].norm
        ffn_squared_sums[layer_index].add_(
            delta_norm2.float().square().sum(dim=(0, 1)).to(torch.float64)
        )

        delta_gate = F.linear(delta_norm2, block.mlp.w1.weight, bias=None)
        delta_up = F.linear(delta_norm2, block.mlp.w2.weight, bias=None)
        corrected_gate = base_ffn[layer_index].gate + delta_gate
        corrected_up = base_ffn[layer_index].up + delta_up
        base_hidden = (
            F.silu(base_ffn[layer_index].gate) * base_ffn[layer_index].up
        )
        corrected_hidden = F.silu(corrected_gate) * corrected_up
        delta_hidden = corrected_hidden - base_hidden
        down_squared_sums[layer_index].add_(
            delta_hidden.float().square().sum(dim=(0, 1)).to(torch.float64)
        )
        sample_counts[layer_index] += int(delta_norm.numel() // delta_norm.shape[-1])
        state = attention_state + block.ls2(
            base_ffn[layer_index].output
            + F.linear(delta_hidden, block.mlp.w3.weight, bias=None)
        )
    return state


@torch.no_grad()
def collect_delta_rms(
    executor: DINOv3SegmentorM2FExecutor,
    adapter: Any,
    backbone: Any,
    args: argparse.Namespace,
    dataset: Any,
) -> tuple[dict[str, dict[int, torch.Tensor]], dict[str, Any]]:
    if args.calibration_images <= 0:
        raise ValueError("--calibration-images must be positive for delta_rms")
    middle_end = len(backbone.blocks) - FULL_SUFFIX_LAYERS
    hidden = int(backbone.blocks[0].attn.qkv.in_features)
    ffn_hidden = int(backbone.blocks[0].mlp.w3.in_features)
    device = backbone.blocks[0].attn.qkv.weight.device
    qkv_squared_sums = {
        layer_index: torch.zeros(hidden, device=device, dtype=torch.float64)
        for layer_index in range(FULL_PREFIX_LAYERS, middle_end)
    }
    ffn_squared_sums = {
        layer_index: torch.zeros(hidden, device=device, dtype=torch.float64)
        for layer_index in range(FULL_PREFIX_LAYERS, middle_end)
    }
    down_squared_sums = {
        layer_index: torch.zeros(ffn_hidden, device=device, dtype=torch.float64)
        for layer_index in range(FULL_PREFIX_LAYERS, middle_end)
    }
    sample_counts = {
        layer_index: 0
        for layer_index in range(FULL_PREFIX_LAYERS, middle_end)
    }
    end_index = min(
        args.calibration_start_index + args.calibration_images,
        len(dataset),
    )
    crop_count = 0
    started = time.time()
    for index in range(args.calibration_start_index, end_index):
        sample = dataset[index]
        original = sample["image"].convert("RGB")
        original_np = np.asarray(original, dtype=np.uint8)
        resized = executor._resize_short_side(original, args.short_side)
        resized_np = np.asarray(resized, dtype=np.uint8)
        image_hw = (resized_np.shape[0], resized_np.shape[1])
        base_canvas = scaled_native_pyramid_level(
            original_np,
            args.base_level,
            image_hw,
        )
        crops = sliding_crops(*image_hw, args.crop_size, args.stride)
        for crop in crops:
            y1, y2, x1, x2 = crop
            full_np = np.ascontiguousarray(resized_np[y1:y2, x1:x2])
            base_np = np.ascontiguousarray(base_canvas[y1:y2, x1:x2])
            full_tensor = executor._pil_to_normalized_tensor(
                Image.fromarray(full_np)
            )
            base_tensor = executor._pil_to_normalized_tensor(
                Image.fromarray(base_np)
            )
            with torch.autocast("cuda", executor.autocast_dtype):
                full_source = executor._prepare_single_source(
                    full_tensor,
                    adapter,
                    backbone,
                )
                base_source = executor._prepare_single_source(
                    base_tensor,
                    adapter,
                    backbone,
                )
                full_initial = full_source["x_backbone"]
                base_initial = base_source["x_backbone"]
                rope = full_source["rope_sincos"]
                base_states, base_qkv, base_ffn, _ = run_base(
                    backbone,
                    base_initial,
                    rope,
                )
                accumulate_dense_delta_rms(
                    backbone,
                    full_initial,
                    base_states,
                    base_qkv,
                    base_ffn,
                    rope,
                    qkv_squared_sums,
                    ffn_squared_sums,
                    down_squared_sums,
                    sample_counts,
                )
            crop_count += 1
            del base_states, base_qkv, base_ffn, base_source, full_source
            torch.cuda.empty_cache()
        print(
            json.dumps(
                {
                    "calibration_index": index,
                    "calibration_crops": crop_count,
                    "elapsed_seconds": time.time() - started,
                },
                sort_keys=True,
            ),
            flush=True,
        )

    rms = {
        "qkv": {
            layer_index: (
                qkv_squared_sums[layer_index]
                / max(sample_counts[layer_index], 1)
            ).sqrt().float()
            for layer_index in qkv_squared_sums
        },
        "ffn_input": {
            layer_index: (
                ffn_squared_sums[layer_index]
                / max(sample_counts[layer_index], 1)
            ).sqrt().float()
            for layer_index in ffn_squared_sums
        },
        "ffn_down": {
            layer_index: (
                down_squared_sums[layer_index]
                / max(sample_counts[layer_index], 1)
            ).sqrt().float()
            for layer_index in down_squared_sums
        },
    }
    layer_rows = []
    for projection, projection_rms in rms.items():
        for layer_index in sorted(projection_rms):
            values = projection_rms[layer_index]
            layer_rows.append(
                {
                    "projection": projection,
                    "layer": float(layer_index),
                    "samples": float(sample_counts[layer_index]),
                    "rms_min": float(values.min().item()),
                    "rms_mean": float(values.mean().item()),
                    "rms_max": float(values.max().item()),
                    "rms_max_to_mean": float(
                        values.max().item()
                        / max(values.mean().item(), 1e-12)
                    ),
                }
            )
    return rms, {
        "split": args.calibration_split,
        "start_index": args.calibration_start_index,
        "images": end_index - args.calibration_start_index,
        "crops": crop_count,
        "seconds": time.time() - started,
        "layers": layer_rows,
    }


@torch.no_grad()
def factorize_backbone_projections(
    backbone: Any,
    args: argparse.Namespace,
    input_rms: dict[str, dict[int, torch.Tensor]] | None = None,
) -> tuple[
    dict[str, dict[int, LowRankLinearFactors]],
    list[dict[str, float]],
]:
    factor_dtype = {
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.factor_dtype]
    max_rank = max(args.ranks)
    middle_end = len(backbone.blocks) - FULL_SUFFIX_LAYERS
    needs_qkv = any(variant in ("qkv", "qkv_ffn") for variant in args.variants)
    needs_ffn = any(variant in ("ffn", "qkv_ffn") for variant in args.variants)
    factors: dict[str, dict[int, LowRankLinearFactors]] = {
        "qkv": {},
        "ffn_gate": {},
        "ffn_up": {},
        "ffn_down": {},
    }
    rows: list[dict[str, float]] = []
    for layer_index in range(FULL_PREFIX_LAYERS, middle_end):
        block = backbone.blocks[layer_index]
        projection_weights: list[tuple[str, torch.Tensor]] = []
        if needs_qkv:
            projection_weights.append(("qkv", block.attn.qkv.weight))
        if needs_ffn:
            projection_weights.extend(
                (
                    ("ffn_gate", block.mlp.w1.weight),
                    ("ffn_up", block.mlp.w2.weight),
                    ("ffn_down", block.mlp.w3.weight),
                )
            )
        for projection_index, (projection, weight) in enumerate(
            projection_weights
        ):
            seed = args.factor_seed + layer_index + projection_index * 10_000
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
                torch.cuda.synchronize()
            started = time.perf_counter()
            if args.factor_mode == "delta_rms":
                rms_projection = (
                    "ffn_input"
                    if projection in ("ffn_gate", "ffn_up")
                    else projection
                )
                if (
                    input_rms is None
                    or rms_projection not in input_rms
                    or layer_index not in input_rms[rms_projection]
                ):
                    raise ValueError(
                        "missing delta RMS calibration for "
                        f"{projection} layer {layer_index}"
                    )
                factor = factorize_linear_weight_activation_aware(
                    weight,
                    input_rms[rms_projection][layer_index],
                    max_rank=max_rank,
                    oversample=args.factor_oversample,
                    power_iterations=args.factor_power_iters,
                    factor_dtype=factor_dtype,
                    rms_floor_ratio=args.factor_rms_floor_ratio,
                )
            else:
                factor = factorize_linear_weight(
                    weight,
                    max_rank=max_rank,
                    oversample=args.factor_oversample,
                    power_iterations=args.factor_power_iters,
                    factor_dtype=factor_dtype,
                )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            factors[projection][layer_index] = factor
            row = {
                "projection": projection,
                "layer": float(layer_index),
                "seconds": elapsed,
                "mode": args.factor_mode,
                "max_rank_energy": factor.spectral_energy_fraction(max_rank),
                "factor_bytes": float(factor.factor_bytes(max_rank)),
            }
            rows.append(row)
            print(json.dumps({"factorized": row}, sort_keys=True), flush=True)
    return factors, rows


def representative_delta(
    backbone: Any,
    full_initial: torch.Tensor,
    base_states: list[torch.Tensor],
    layer_index: int,
    rope: Any,
) -> torch.Tensor:
    state = full_initial
    for index in range(layer_index):
        state = backbone.blocks[index](state, rope)
    block = backbone.blocks[layer_index]
    return block.norm1(state) - block.norm1(base_states[layer_index])


def benchmark_qkv_projection(
    delta_norm: torch.Tensor,
    weight: torch.Tensor,
    factors: LowRankLinearFactors,
    ranks: list[int],
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    return benchmark_projection(
        delta_norm,
        weight,
        factors,
        ranks,
        warmup,
        iterations,
    )


def benchmark_projection(
    delta_input: torch.Tensor,
    weight: torch.Tensor,
    factors: LowRankLinearFactors,
    ranks: list[int],
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    if not delta_input.is_cuda or iterations <= 0:
        return {}

    operations = {
        "dense": lambda: F.linear(delta_input, weight, bias=None),
        **{
            f"rank_{rank}": (
                lambda selected_rank=rank: factors.apply(
                    delta_input,
                    selected_rank,
                )
            )
            for rank in ranks
        },
    }
    timings: dict[str, float] = {}
    for name, operation in operations.items():
        for _ in range(max(warmup, 0)):
            operation()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            operation()
        end.record()
        end.synchronize()
        timings[name] = float(start.elapsed_time(end) / iterations)
    dense_ms = timings["dense"]
    for rank in ranks:
        timings[f"rank_{rank}_speedup"] = dense_ms / timings[f"rank_{rank}"]
    return timings


def representative_projection_deltas(
    backbone: Any,
    full_initial: torch.Tensor,
    base_states: list[torch.Tensor],
    base_qkv: list[torch.Tensor],
    base_ffn: list[BaseFFNCache],
    layer_index: int,
    rope: Any,
) -> dict[str, torch.Tensor]:
    state = full_initial
    for index in range(layer_index):
        state = backbone.blocks[index](state, rope)
    block = backbone.blocks[layer_index]
    delta_norm1 = (
        block.norm1(state) - block.norm1(base_states[layer_index])
    )
    corrected_qkv = base_qkv[layer_index] + F.linear(
        delta_norm1,
        block.attn.qkv.weight,
        bias=None,
    )
    attention_raw = block.attn.compute_attention(corrected_qkv, rope=rope)
    attention_output = block.attn.proj_drop(block.attn.proj(attention_raw))
    attention_state = state + block.ls1(attention_output)
    delta_norm2 = block.norm2(attention_state) - base_ffn[layer_index].norm
    corrected_gate = base_ffn[layer_index].gate + F.linear(
        delta_norm2,
        block.mlp.w1.weight,
        bias=None,
    )
    corrected_up = base_ffn[layer_index].up + F.linear(
        delta_norm2,
        block.mlp.w2.weight,
        bias=None,
    )
    base_hidden = F.silu(base_ffn[layer_index].gate) * base_ffn[layer_index].up
    corrected_hidden = F.silu(corrected_gate) * corrected_up
    return {
        "qkv": delta_norm1,
        "ffn_gate": delta_norm2,
        "ffn_up": delta_norm2,
        "ffn_down": corrected_hidden - base_hidden,
    }


def source_context(
    source: dict[str, Any],
    raw: dict[int, torch.Tensor],
) -> dict[str, Any]:
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
    executor._accumulate_m2f_adapter_feature_crop(
        sums,
        counts,
        features,
        crop,
        image_hw,
    )


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
    logits = F.interpolate(
        logits.float(),
        size=original_hw,
        mode="bilinear",
        align_corners=False,
    )
    return torch.softmax(logits, dim=1).argmax(dim=1)[0].to(torch.uint8)


def correction_qkv_flops(
    backbone: Any,
    tokens: int,
    ranks: list[int],
) -> dict[str, dict[str, float]]:
    block = backbone.blocks[0]
    in_features = int(block.attn.qkv.in_features)
    out_features = int(block.attn.qkv.out_features)
    depth = len(backbone.blocks)
    middle = depth - FULL_PREFIX_LAYERS - FULL_SUFFIX_LAYERS
    boundary = FULL_PREFIX_LAYERS + FULL_SUFFIX_LAYERS
    dense_layer = 2 * tokens * in_features * out_features
    dense_total = depth * dense_layer
    result = {
        "dense_delta": {
            "qkv_correction_flops": float(dense_total),
            "ratio_to_dense": 1.0,
        }
    }
    for rank in ranks:
        low_rank_layer = 2 * tokens * rank * (in_features + out_features)
        total = boundary * dense_layer + middle * low_rank_layer
        result[f"rank_{rank}"] = {
            "qkv_correction_flops": float(total),
            "ratio_to_dense": float(total / dense_total),
            "middle_layer_ratio": dense_linear_flop_ratio(
                in_features,
                out_features,
                rank,
            ),
        }
    return result


def correction_projection_flops(
    backbone: Any,
    tokens: int,
    ranks: list[int],
    variants: list[str],
) -> dict[str, dict[str, float]]:
    block = backbone.blocks[0]
    qkv_in = int(block.attn.qkv.in_features)
    qkv_out = int(block.attn.qkv.out_features)
    ffn_in = int(block.mlp.w1.in_features)
    ffn_hidden = int(block.mlp.w1.out_features)
    depth = len(backbone.blocks)
    middle = depth - FULL_PREFIX_LAYERS - FULL_SUFFIX_LAYERS
    boundary = FULL_PREFIX_LAYERS + FULL_SUFFIX_LAYERS
    dense_per_layer = {
        "qkv": 2 * tokens * qkv_in * qkv_out,
        "ffn_gate": 2 * tokens * ffn_in * ffn_hidden,
        "ffn_up": 2 * tokens * ffn_in * ffn_hidden,
        "ffn_down": 2 * tokens * ffn_hidden * ffn_in,
    }
    dense_total = depth * sum(dense_per_layer.values())
    result: dict[str, dict[str, float]] = {
        "dense_delta": {
            "projection_flops": float(dense_total),
            "ratio_to_dense": 1.0,
        }
    }
    for variant in variants:
        for rank in ranks:
            total = boundary * sum(dense_per_layer.values())
            projection_ratios = {}
            for projection, dense_layer in dense_per_layer.items():
                low_rank_enabled = (
                    (projection == "qkv" and variant in ("qkv", "qkv_ffn"))
                    or (
                        projection.startswith("ffn")
                        and variant in ("ffn", "qkv_ffn")
                    )
                )
                if low_rank_enabled:
                    if projection == "qkv":
                        in_features, out_features = qkv_in, qkv_out
                    elif projection in ("ffn_gate", "ffn_up"):
                        in_features, out_features = ffn_in, ffn_hidden
                    else:
                        in_features, out_features = ffn_hidden, ffn_in
                    low_rank_layer = (
                        2 * tokens * rank * (in_features + out_features)
                    )
                    total += middle * low_rank_layer
                    projection_ratios[projection] = dense_linear_flop_ratio(
                        in_features,
                        out_features,
                        rank,
                    )
                else:
                    total += middle * dense_layer
                    projection_ratios[projection] = 1.0
            result[f"{variant}_rank_{rank}"] = {
                "projection_flops": float(total),
                "ratio_to_dense": float(total / dense_total),
                **{
                    f"{projection}_middle_ratio": float(ratio)
                    for projection, ratio in projection_ratios.items()
                },
            }
    return result


def ffn_cache_bytes(
    base_ffn: list[BaseFFNCache],
    *,
    middle_only: bool = True,
) -> int:
    start = FULL_PREFIX_LAYERS if middle_only else 0
    end = (
        len(base_ffn) - FULL_SUFFIX_LAYERS
        if middle_only
        else len(base_ffn)
    )
    return int(
        sum(
            tensor.numel() * tensor.element_size()
            for cache in base_ffn[start:end]
            for tensor in (cache.norm, cache.gate, cache.up, cache.output)
        )
    )


def write_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.num_images <= 0:
        raise ValueError("--num-images must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / "results.jsonl"
    summary_path = args.output / "summary.json"
    if result_path.exists() or summary_path.exists():
        raise FileExistsError(
            f"output already exists under {args.output}; choose a new --output"
        )

    torch.set_grad_enabled(False)
    executor, _ = load_executor(args)
    adapter = executor.model.segmentation_model[0]
    backbone = adapter.backbone

    from datasets import load_dataset

    calibration_summary: dict[str, Any] = {
        "mode": args.factor_mode,
        "images": 0,
        "crops": 0,
        "layers": [],
    }
    input_rms = None
    if args.factor_mode == "delta_rms":
        calibration_dataset = load_dataset(
            "merve/scene_parse_150",
            split=args.calibration_split,
        )
        input_rms, calibration_summary = collect_delta_rms(
            executor,
            adapter,
            backbone,
            args,
            calibration_dataset,
        )
        del calibration_dataset
    factors, factorization_rows = factorize_backbone_projections(
        backbone,
        args,
        input_rms=input_rms,
    )
    del input_rms

    names = [
        "full_oracle",
        "approx_l2",
        "dense_delta",
        *[
            f"{variant}_rank_{rank}"
            for variant in args.variants
            for rank in args.ranks
        ],
    ]
    metrics = {name: AreaMetrics() for name in names}
    projection_errors = {
        name: {
            "qkv": ErrorAccumulator(),
            "ffn_gate": ErrorAccumulator(),
            "ffn_up": ErrorAccumulator(),
            "ffn_down": ErrorAccumulator(),
        }
        for name in names
        if name not in ("full_oracle", "approx_l2", "dense_delta")
    }
    rank_endpoint_errors = {
        name: ErrorAccumulator() for name in projection_errors
    }
    dense_endpoint_error = ErrorAccumulator()
    correction_times_ms: dict[str, list[float]] = defaultdict(list)
    benchmarks: dict[str, dict[str, float]] = {}
    measured_ffn_cache_bytes: int | None = None

    dataset = load_dataset("merve/scene_parse_150", split=args.eval_split)
    end_index = min(args.start_index + args.num_images, len(dataset))
    started = time.time()
    for index in range(args.start_index, end_index):
        sample = dataset[index]
        original = sample["image"].convert("RGB")
        annotation = np.asarray(sample["annotation"], dtype=np.uint8)
        original_np = np.asarray(original, dtype=np.uint8)
        resized = executor._resize_short_side(original, args.short_side)
        resized_np = np.asarray(resized, dtype=np.uint8)
        image_hw = (resized_np.shape[0], resized_np.shape[1])
        original_hw = (original.height, original.width)
        base_canvas = scaled_native_pyramid_level(
            original_np,
            args.base_level,
            image_hw,
        )
        crops = sliding_crops(*image_hw, args.crop_size, args.stride)
        feature_sums = {name: {} for name in names}
        feature_counts = {name: {} for name in names}

        for crop_index, crop in enumerate(crops):
            y1, y2, x1, x2 = crop
            full_np = np.ascontiguousarray(resized_np[y1:y2, x1:x2])
            base_np = np.ascontiguousarray(base_canvas[y1:y2, x1:x2])
            full_tensor = executor._pil_to_normalized_tensor(Image.fromarray(full_np))
            base_tensor = executor._pil_to_normalized_tensor(Image.fromarray(base_np))
            with torch.autocast("cuda", executor.autocast_dtype):
                full_source = executor._prepare_single_source(
                    full_tensor,
                    adapter,
                    backbone,
                )
                base_source = executor._prepare_single_source(
                    base_tensor,
                    adapter,
                    backbone,
                )
                corrected_source = replace_spm_features(full_source, base_source)
                full_initial = full_source["x_backbone"]
                base_initial = base_source["x_backbone"]
                rope = full_source["rope_sincos"]

                base_states, base_qkv, base_ffn, base_raw = run_base(
                    backbone,
                    base_initial,
                    rope,
                )
                if measured_ffn_cache_bytes is None:
                    measured_ffn_cache_bytes = ffn_cache_bytes(base_ffn)
                full_final, full_raw = run_full(backbone, full_initial, rope)

                add_crop_features(
                    executor,
                    feature_sums["full_oracle"],
                    feature_counts["full_oracle"],
                    full_source,
                    full_raw,
                    crop,
                    image_hw,
                )
                add_crop_features(
                    executor,
                    feature_sums["approx_l2"],
                    feature_counts["approx_l2"],
                    corrected_source,
                    base_raw,
                    crop,
                    image_hw,
                )

                if full_initial.is_cuda:
                    event_start = torch.cuda.Event(enable_timing=True)
                    event_end = torch.cuda.Event(enable_timing=True)
                    event_start.record()
                dense_final, dense_raw = run_low_rank_delta_correction(
                    backbone,
                    full_initial,
                    base_states,
                    base_qkv,
                    base_ffn,
                    rope,
                    qkv_factors=factors["qkv"],
                    ffn_gate_factors=factors["ffn_gate"],
                    ffn_up_factors=factors["ffn_up"],
                    ffn_down_factors=factors["ffn_down"],
                    qkv_rank=None,
                    ffn_rank=None,
                )
                if full_initial.is_cuda:
                    event_end.record()
                    event_end.synchronize()
                    correction_times_ms["dense_delta"].append(
                        float(event_start.elapsed_time(event_end))
                    )
                dense_endpoint_error.update(dense_final, full_final)
                add_crop_features(
                    executor,
                    feature_sums["dense_delta"],
                    feature_counts["dense_delta"],
                    corrected_source,
                    dense_raw,
                    crop,
                    image_hw,
                )

                for variant in args.variants:
                    for rank in args.ranks:
                        name = f"{variant}_rank_{rank}"
                        collect_projection_error = (
                            index == args.start_index and crop_index == 0
                        )
                        if full_initial.is_cuda and not collect_projection_error:
                            event_start = torch.cuda.Event(enable_timing=True)
                            event_end = torch.cuda.Event(enable_timing=True)
                            event_start.record()
                        rank_final, rank_raw = run_low_rank_delta_correction(
                            backbone,
                            full_initial,
                            base_states,
                            base_qkv,
                            base_ffn,
                            rope,
                            qkv_factors=factors["qkv"],
                            ffn_gate_factors=factors["ffn_gate"],
                            ffn_up_factors=factors["ffn_up"],
                            ffn_down_factors=factors["ffn_down"],
                            qkv_rank=(
                                rank
                                if variant in ("qkv", "qkv_ffn")
                                else None
                            ),
                            ffn_rank=(
                                rank
                                if variant in ("ffn", "qkv_ffn")
                                else None
                            ),
                            qkv_error_accumulator=(
                                projection_errors[name]["qkv"]
                                if (
                                    collect_projection_error
                                    and variant in ("qkv", "qkv_ffn")
                                )
                                else None
                            ),
                            ffn_gate_error_accumulator=(
                                projection_errors[name]["ffn_gate"]
                                if (
                                    collect_projection_error
                                    and variant in ("ffn", "qkv_ffn")
                                )
                                else None
                            ),
                            ffn_up_error_accumulator=(
                                projection_errors[name]["ffn_up"]
                                if (
                                    collect_projection_error
                                    and variant in ("ffn", "qkv_ffn")
                                )
                                else None
                            ),
                            ffn_down_error_accumulator=(
                                projection_errors[name]["ffn_down"]
                                if (
                                    collect_projection_error
                                    and variant in ("ffn", "qkv_ffn")
                                )
                                else None
                            ),
                        )
                        if full_initial.is_cuda and not collect_projection_error:
                            event_end.record()
                            event_end.synchronize()
                            correction_times_ms[name].append(
                                float(event_start.elapsed_time(event_end))
                            )
                        rank_endpoint_errors[name].update(
                            rank_final,
                            dense_final,
                        )
                        add_crop_features(
                            executor,
                            feature_sums[name],
                            feature_counts[name],
                            corrected_source,
                            rank_raw,
                            crop,
                            image_hw,
                        )

                if not benchmarks:
                    layer_index = FULL_PREFIX_LAYERS
                    representative = representative_projection_deltas(
                        backbone,
                        full_initial,
                        base_states,
                        base_qkv,
                        base_ffn,
                        layer_index,
                        rope,
                    )
                    block = backbone.blocks[layer_index]
                    weights = {
                        "qkv": block.attn.qkv.weight,
                        "ffn_gate": block.mlp.w1.weight,
                        "ffn_up": block.mlp.w2.weight,
                        "ffn_down": block.mlp.w3.weight,
                    }
                    for projection, projection_factors in factors.items():
                        if not projection_factors:
                            continue
                        benchmarks[projection] = benchmark_projection(
                            representative[projection],
                            weights[projection],
                            projection_factors[layer_index],
                            args.ranks,
                            args.benchmark_warmup,
                            args.benchmark_iters,
                        )
            del base_states, base_qkv, base_ffn, base_raw, full_raw, dense_raw
            del base_source, corrected_source, full_source
            torch.cuda.empty_cache()

        areas = {}
        with torch.autocast("cuda", executor.autocast_dtype):
            for name in names:
                prediction = finish_prediction(
                    executor,
                    feature_sums[name],
                    feature_counts[name],
                    image_hw,
                    original_hw,
                )
                areas[name] = metrics[name].update(prediction, annotation)
        write_jsonl(
            result_path,
            {
                "index": index,
                "areas": areas,
                "image_hw": image_hw,
                "original_hw": original_hw,
                "num_crops": len(crops),
            },
        )
        print(
            json.dumps(
                {
                    "completed_index": index,
                    "elapsed_seconds": time.time() - started,
                    "metrics": {
                        name: metric.summary()
                        for name, metric in metrics.items()
                    },
                },
                sort_keys=True,
            ),
            flush=True,
        )

    token_count = 5 + (args.crop_size // 16) ** 2
    factor_stats: dict[str, dict[str, dict[str, float]]] = {}
    for projection, projection_factors in factors.items():
        if not projection_factors:
            continue
        factor_stats[projection] = {}
        for rank in args.ranks:
            bytes_total = sum(
                factor.factor_bytes(rank)
                for factor in projection_factors.values()
            )
            energy = [
                factor.spectral_energy_fraction(rank)
                for factor in projection_factors.values()
            ]
            factor_stats[projection][f"rank_{rank}"] = {
                "bytes": float(bytes_total),
                "mean_weight_spectral_energy": float(np.mean(energy)),
                "min_weight_spectral_energy": float(np.min(energy)),
                "max_weight_spectral_energy": float(np.max(energy)),
            }
    timing_summary = {
        name: {
            "mean_ms": float(np.mean(values)),
            "min_ms": float(np.min(values)),
            "max_ms": float(np.max(values)),
            "samples": float(len(values)),
        }
        for name, values in correction_times_ms.items()
    }
    summary = {
        "configuration": {
            **vars(args),
            "output": str(args.output),
            "ranks": list(args.ranks),
            "base_canvas_mode": BASE_CANVAS_MODE,
            "full_prefix_layers": FULL_PREFIX_LAYERS,
            "full_suffix_layers": FULL_SUFFIX_LAYERS,
            "intermediate_layers": list(INTERMEDIATE_LAYERS),
        },
        "metrics": {name: metric.summary() for name, metric in metrics.items()},
        "dense_delta_endpoint_error": dense_endpoint_error.summary(),
        "projection_delta_errors": {
            name: {
                projection: accumulator.summary()
                for projection, accumulator in errors.items()
                if accumulator.samples > 0
            }
            for name, errors in projection_errors.items()
        },
        "rank_endpoint_errors": {
            name: accumulator.summary()
            for name, accumulator in rank_endpoint_errors.items()
        },
        "factorization": {
            "layers": factorization_rows,
            "total_seconds": float(
                sum(row["seconds"] for row in factorization_rows)
            ),
            "rank_stats": factor_stats,
        },
        "calibration": calibration_summary,
        "projection_benchmark_ms": benchmarks,
        "correction_wall_time": timing_summary,
        "correction_projection_flops": correction_projection_flops(
            backbone,
            token_count,
            args.ranks,
            args.variants,
        ),
        "ffn_approx_cache_bytes_per_crop": float(
            measured_ffn_cache_bytes or 0
        ),
        "elapsed_seconds": time.time() - started,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "metrics": summary["metrics"],
                "projection_delta_errors": summary["projection_delta_errors"],
                "benchmark_ms": summary["projection_benchmark_ms"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
