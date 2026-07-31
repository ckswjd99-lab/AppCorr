#!/usr/bin/env python3
"""Evaluate input-aware low-rank approximation plus 50% token correction.

Approximation runs on the native-L2 canvas. Selected projections use factors
calibrated against their approximate-pass input RMS. Correction uses the
original dense weights for one pscore-selected token pattern and applies no
attention-edge or FFN-channel sparsity. The first and last two layers remain
fully corrected, matching the earlier ADE20K block-sparse experiments.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.experiments.ade20k_qkv_low_rank_eval import (
    AreaMetrics,
    FULL_PREFIX_LAYERS,
    FULL_SUFFIX_LAYERS,
    INTERMEDIATE_LAYERS,
    add_crop_features,
    finish_prediction,
    load_executor,
    replace_spm_features,
    run_full,
    scaled_native_pyramid_level,
    sliding_crops,
)
from appcorr.models.dinov3.layers.low_rank import (
    LowRankLinearFactors,
    dense_linear_flop_ratio,
    factorize_linear_weight_activation_aware,
)


SPECIAL_TOKENS = 5
PROJECTIONS = ("qkv", "attn_out", "ffn_gate", "ffn_up", "ffn_down")
VARIANT_TARGETS: dict[str, frozenset[str]] = {
    "qkv": frozenset({"qkv"}),
    "attn_out": frozenset({"attn_out"}),
    "attn_all": frozenset({"qkv", "attn_out"}),
    "ffn_gate": frozenset({"ffn_gate"}),
    "ffn_up": frozenset({"ffn_up"}),
    "ffn_down": frozenset({"ffn_down"}),
    "ffn_all": frozenset({"ffn_gate", "ffn_up", "ffn_down"}),
    "qkv_ffn_gate": frozenset({"qkv", "ffn_gate"}),
    "qkv_ffn_up": frozenset({"qkv", "ffn_up"}),
    "qkv_ffn_down": frozenset({"qkv", "ffn_down"}),
    "qkv_ffn_gate_up": frozenset({"qkv", "ffn_gate", "ffn_up"}),
    "qkv_ffn_all": frozenset({"qkv", "ffn_gate", "ffn_up", "ffn_down"}),
    "attn_out_ffn_gate": frozenset({"attn_out", "ffn_gate"}),
    "attn_out_ffn_up": frozenset({"attn_out", "ffn_up"}),
    "attn_out_ffn_gate_up": frozenset(
        {"attn_out", "ffn_gate", "ffn_up"}
    ),
    "attn_out_ffn_all": frozenset(
        {"attn_out", "ffn_gate", "ffn_up", "ffn_down"}
    ),
    "attn_all_ffn_gate": frozenset({"qkv", "attn_out", "ffn_gate"}),
    "attn_all_ffn_up": frozenset({"qkv", "attn_out", "ffn_up"}),
    "attn_all_ffn_down": frozenset({"qkv", "attn_out", "ffn_down"}),
    "all_except_down": frozenset(
        {"qkv", "attn_out", "ffn_gate", "ffn_up"}
    ),
    "all": frozenset(PROJECTIONS),
}


@dataclass
class ApproxCache:
    states: list[torch.Tensor]
    qkv: list[torch.Tensor]
    raw: dict[int, torch.Tensor]
    received_attention: torch.Tensor


def parse_csv(value: str) -> list[str]:
    values = list(dict.fromkeys(item.strip() for item in value.split(",")))
    if not values or any(item not in VARIANT_TARGETS for item in values):
        raise argparse.ArgumentTypeError(
            "variants must be chosen from " + ",".join(VARIANT_TARGETS)
        )
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-images", type=int, default=1)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--short-side", type=int, default=896)
    parser.add_argument("--crop-size", type=int, default=896)
    parser.add_argument("--stride", type=int, default=596)
    parser.add_argument("--base-level", type=int, default=2)
    parser.add_argument("--rank", type=int, default=1536)
    parser.add_argument("--token-rate", type=float, default=0.5)
    parser.add_argument(
        "--variants",
        type=parse_csv,
        default=parse_csv(",".join(VARIANT_TARGETS)),
    )
    parser.add_argument("--factor-oversample", type=int, default=16)
    parser.add_argument("--factor-power-iters", type=int, default=1)
    parser.add_argument("--factor-rms-floor-ratio", type=float, default=1e-3)
    parser.add_argument("--factor-seed", type=int, default=20260801)
    parser.add_argument("--calibration-split", default="train")
    parser.add_argument("--calibration-start-index", type=int, default=0)
    parser.add_argument("--calibration-images", type=int, default=16)
    parser.add_argument("--eval-split", default="validation")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/ade20k_low_rank_approx_token50"),
    )
    parser.add_argument(
        "--backbone-weights",
        default="~/cjpark/weights/dinov3/"
        "dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
    )
    parser.add_argument(
        "--head-weights",
        default="~/cjpark/weights/dinov3/"
        "dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth",
    )
    return parser.parse_args()


def projection_module(block: Any, projection: str) -> torch.nn.Linear:
    return {
        "qkv": block.attn.qkv,
        "attn_out": block.attn.proj,
        "ffn_gate": block.mlp.w1,
        "ffn_up": block.mlp.w2,
        "ffn_down": block.mlp.w3,
    }[projection]


def projection_rms_key(projection: str) -> str:
    if projection == "qkv":
        return "norm1"
    if projection == "attn_out":
        return "attn_raw"
    if projection in ("ffn_gate", "ffn_up"):
        return "norm2"
    return "ffn_hidden"


def apply_projection(
    module: torch.nn.Linear,
    value: torch.Tensor,
    factor: LowRankLinearFactors | None,
    rank: int,
) -> torch.Tensor:
    if factor is None:
        return module(value)
    output = factor.apply(value, rank)
    if module.bias is not None:
        output = output + module.bias
    return output


def accumulate_channel_rms(
    squared_sums: dict[str, dict[int, torch.Tensor]],
    sample_counts: dict[str, dict[int, int]],
    key: str,
    layer_index: int,
    value: torch.Tensor,
) -> None:
    squared_sums[key][layer_index].add_(
        value.float().square().sum(dim=(0, 1)).to(torch.float64)
    )
    sample_counts[key][layer_index] += int(value.numel() // value.shape[-1])


def received_attention_score(
    block: Any,
    qkv: torch.Tensor,
    rope: Any,
) -> torch.Tensor:
    batch, tokens, _ = qkv.shape
    hidden = block.attn.qkv.in_features
    qkv_heads = qkv.reshape(
        batch,
        tokens,
        3,
        block.attn.num_heads,
        hidden // block.attn.num_heads,
    )
    query, key, _ = torch.unbind(qkv_heads, dim=2)
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    if rope is not None:
        query, key = block.attn.apply_rope(query, key, rope)
    probability = (
        query @ key.transpose(-2, -1) * float(block.attn.scale)
    ).softmax(dim=-1)
    return probability.mean(dim=1).mean(dim=1).float()


def run_approx(
    backbone: Any,
    state: torch.Tensor,
    rope: Any,
    factors: dict[str, dict[int, LowRankLinearFactors]],
    targets: frozenset[str],
    rank: int,
    *,
    collect_score: bool,
    squared_sums: dict[str, dict[int, torch.Tensor]] | None = None,
    sample_counts: dict[str, dict[int, int]] | None = None,
) -> ApproxCache:
    states = [state]
    qkv_values = []
    raw: dict[int, torch.Tensor] = {}
    score_sum = torch.zeros(state.shape[:2], device=state.device, dtype=torch.float32)
    for layer_index, block in enumerate(backbone.blocks):
        norm1 = block.norm1(state)
        if squared_sums is not None and sample_counts is not None:
            accumulate_channel_rms(
                squared_sums,
                sample_counts,
                "norm1",
                layer_index,
                norm1,
            )
        qkv_factor = factors["qkv"].get(layer_index) if "qkv" in targets else None
        qkv = apply_projection(block.attn.qkv, norm1, qkv_factor, rank)
        qkv_values.append(qkv.detach())
        if collect_score:
            score_sum += received_attention_score(block, qkv, rope)
        attention_raw = block.attn.compute_attention(qkv, rope=rope)
        if squared_sums is not None and sample_counts is not None:
            accumulate_channel_rms(
                squared_sums,
                sample_counts,
                "attn_raw",
                layer_index,
                attention_raw,
            )
        out_factor = (
            factors["attn_out"].get(layer_index)
            if "attn_out" in targets
            else None
        )
        attention_output = block.attn.proj_drop(
            apply_projection(block.attn.proj, attention_raw, out_factor, rank)
        )
        attention_state = state + block.ls1(attention_output)
        norm2 = block.norm2(attention_state)
        if squared_sums is not None and sample_counts is not None:
            accumulate_channel_rms(
                squared_sums,
                sample_counts,
                "norm2",
                layer_index,
                norm2,
            )
        gate_factor = (
            factors["ffn_gate"].get(layer_index)
            if "ffn_gate" in targets
            else None
        )
        up_factor = (
            factors["ffn_up"].get(layer_index)
            if "ffn_up" in targets
            else None
        )
        gate = apply_projection(block.mlp.w1, norm2, gate_factor, rank)
        up = apply_projection(block.mlp.w2, norm2, up_factor, rank)
        hidden = F.silu(gate) * up
        if squared_sums is not None and sample_counts is not None:
            accumulate_channel_rms(
                squared_sums,
                sample_counts,
                "ffn_hidden",
                layer_index,
                hidden,
            )
        down_factor = (
            factors["ffn_down"].get(layer_index)
            if "ffn_down" in targets
            else None
        )
        output = apply_projection(block.mlp.w3, hidden, down_factor, rank)
        state = attention_state + block.ls2(output)
        states.append(state)
        if layer_index in INTERMEDIATE_LAYERS:
            raw[layer_index] = state
    return ApproxCache(
        states=states,
        qkv=qkv_values,
        raw=raw,
        received_attention=score_sum / len(backbone.blocks),
    )


def gather_tokens(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return value.gather(
        1,
        indices.unsqueeze(-1).expand(-1, -1, value.shape[-1]),
    )


def scatter_tokens(
    base: torch.Tensor,
    indices: torch.Tensor,
    update: torch.Tensor,
) -> torch.Tensor:
    result = base.clone()
    result.scatter_(
        1,
        indices.unsqueeze(-1).expand(-1, -1, result.shape[-1]),
        update,
    )
    return result


def gather_head_tokens(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return value.gather(
        2,
        indices[:, None, :, None].expand(
            -1,
            value.shape[1],
            -1,
            value.shape[3],
        ),
    )


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
    energy = energy.reshape(
        grid_h,
        patch_size,
        grid_w,
        patch_size,
    ).mean(dim=(1, 3)).flatten()
    received = attention_score[0, SPECIAL_TOKENS:].detach().cpu()
    if received.numel() != energy.numel():
        raise RuntimeError(
            f"patch score mismatch: attention={received.numel()} residual={energy.numel()}"
        )
    eps = torch.finfo(torch.float32).eps
    combined = (
        energy / energy.max().clamp_min(eps)
    ) * (
        received / received.max().clamp_min(eps)
    )
    keep = max(1, min(combined.numel(), math.ceil(combined.numel() * token_rate)))
    selected = torch.topk(combined, k=keep).indices.sort().values
    selected = selected.to(attention_score.device) + SPECIAL_TOKENS
    prefix = torch.arange(SPECIAL_TOKENS, device=attention_score.device)
    return torch.cat((prefix, selected)).unsqueeze(0)


def run_token_correction(
    backbone: Any,
    full_initial: torch.Tensor,
    approx: ApproxCache,
    active_indices: torch.Tensor,
    rope: Any,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    state = full_initial
    raw: dict[int, torch.Tensor] = {}
    middle_end = len(backbone.blocks) - FULL_SUFFIX_LAYERS
    for layer_index, block in enumerate(backbone.blocks):
        if layer_index < FULL_PREFIX_LAYERS or layer_index >= middle_end:
            state = block(state, rope)
        else:
            active_state = gather_tokens(state, active_indices)
            active_qkv = block.attn.qkv(block.norm1(active_state))
            hybrid_qkv = scatter_tokens(
                approx.qkv[layer_index],
                active_indices,
                active_qkv,
            )
            batch, tokens, _ = hybrid_qkv.shape
            hidden = block.attn.qkv.in_features
            qkv_heads = hybrid_qkv.reshape(
                batch,
                tokens,
                3,
                block.attn.num_heads,
                hidden // block.attn.num_heads,
            )
            query, key, value = torch.unbind(qkv_heads, dim=2)
            query, key, value = [item.transpose(1, 2) for item in (query, key, value)]
            if rope is not None:
                query, key = block.attn.apply_rope(query, key, rope)
            active_query = gather_head_tokens(query, active_indices)
            attention_raw = F.scaled_dot_product_attention(active_query, key, value)
            attention_raw = attention_raw.transpose(1, 2).reshape(
                batch,
                active_indices.shape[1],
                hidden,
            )
            attention_output = block.attn.proj_drop(block.attn.proj(attention_raw))
            attention_state = active_state + block.ls1(attention_output)
            corrected_active = attention_state + block.ls2(
                block.mlp(block.norm2(attention_state))
            )
            state = scatter_tokens(
                approx.states[layer_index + 1],
                active_indices,
                corrected_active,
            )
        if layer_index in INTERMEDIATE_LAYERS:
            raw[layer_index] = state
    return state, raw


@torch.no_grad()
def collect_approx_rms(
    executor: Any,
    adapter: Any,
    backbone: Any,
    args: argparse.Namespace,
    dataset: Any,
) -> tuple[dict[str, dict[int, torch.Tensor]], dict[str, Any]]:
    device = backbone.blocks[0].attn.qkv.weight.device
    dimensions = {
        "norm1": int(backbone.blocks[0].attn.qkv.in_features),
        "attn_raw": int(backbone.blocks[0].attn.proj.in_features),
        "norm2": int(backbone.blocks[0].mlp.w1.in_features),
        "ffn_hidden": int(backbone.blocks[0].mlp.w3.in_features),
    }
    squared_sums = {
        key: {
            layer: torch.zeros(width, device=device, dtype=torch.float64)
            for layer in range(len(backbone.blocks))
        }
        for key, width in dimensions.items()
    }
    sample_counts = {
        key: {layer: 0 for layer in range(len(backbone.blocks))}
        for key in dimensions
    }
    empty_factors = {projection: {} for projection in PROJECTIONS}
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
        image_hw = resized_np.shape[:2]
        base_canvas = scaled_native_pyramid_level(
            original_np,
            args.base_level,
            image_hw,
        )
        for y1, y2, x1, x2 in sliding_crops(
            *image_hw,
            args.crop_size,
            args.stride,
        ):
            base_np = np.ascontiguousarray(base_canvas[y1:y2, x1:x2])
            base_tensor = executor._pil_to_normalized_tensor(Image.fromarray(base_np))
            with torch.autocast("cuda", executor.autocast_dtype):
                base_source = executor._prepare_single_source(
                    base_tensor,
                    adapter,
                    backbone,
                )
                run_approx(
                    backbone,
                    base_source["x_backbone"],
                    base_source["rope_sincos"],
                    empty_factors,
                    frozenset(),
                    args.rank,
                    collect_score=False,
                    squared_sums=squared_sums,
                    sample_counts=sample_counts,
                )
            del base_source
            crop_count += 1
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
        key: {
            layer: (
                squared_sums[key][layer] / max(sample_counts[key][layer], 1)
            ).sqrt().float()
            for layer in squared_sums[key]
        }
        for key in squared_sums
    }
    return rms, {
        "split": args.calibration_split,
        "images": end_index - args.calibration_start_index,
        "crops": crop_count,
        "seconds": time.time() - started,
    }


@torch.no_grad()
def factorize_backbone(
    backbone: Any,
    rms: dict[str, dict[int, torch.Tensor]],
    args: argparse.Namespace,
) -> tuple[dict[str, dict[int, LowRankLinearFactors]], list[dict[str, Any]]]:
    needed = frozenset().union(*(VARIANT_TARGETS[name] for name in args.variants))
    factors = {projection: {} for projection in PROJECTIONS}
    rows = []
    for layer_index, block in enumerate(backbone.blocks):
        for projection_index, projection in enumerate(PROJECTIONS):
            if projection not in needed:
                continue
            seed = args.factor_seed + layer_index + projection_index * 10_000
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.cuda.synchronize()
            started = time.perf_counter()
            module = projection_module(block, projection)
            factor = factorize_linear_weight_activation_aware(
                module.weight,
                rms[projection_rms_key(projection)][layer_index],
                args.rank,
                oversample=args.factor_oversample,
                power_iterations=args.factor_power_iters,
                factor_dtype=torch.bfloat16,
                rms_floor_ratio=args.factor_rms_floor_ratio,
            )
            torch.cuda.synchronize()
            row = {
                "layer": layer_index,
                "projection": projection,
                "seconds": time.perf_counter() - started,
                "factor_bytes": factor.factor_bytes(args.rank),
                "energy": factor.spectral_energy_fraction(args.rank),
            }
            factors[projection][layer_index] = factor
            rows.append(row)
            print(json.dumps({"factorized": row}, sort_keys=True), flush=True)
    return factors, rows


def active_jaccard(first: torch.Tensor, second: torch.Tensor) -> float:
    first_set = set(first[0, SPECIAL_TOKENS:].tolist())
    second_set = set(second[0, SPECIAL_TOKENS:].tolist())
    return len(first_set & second_set) / max(len(first_set | second_set), 1)


def timed_cuda(operation: Any) -> tuple[Any, float]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = operation()
    end.record()
    end.synchronize()
    return result, float(start.elapsed_time(end))


def flop_estimates(
    backbone: Any,
    tokens: int,
    rank: int,
    token_rate: float,
    variants: list[str],
) -> dict[str, dict[str, float]]:
    block = backbone.blocks[0]
    hidden = int(block.attn.qkv.in_features)
    ffn = int(block.mlp.w1.out_features)
    heads = int(block.attn.num_heads)
    depth = len(backbone.blocks)
    boundary = FULL_PREFIX_LAYERS + FULL_SUFFIX_LAYERS
    middle = depth - boundary
    active = SPECIAL_TOKENS + math.ceil((tokens - SPECIAL_TOKENS) * token_rate)
    dense = {
        "qkv": 6 * tokens * hidden * hidden,
        "attn_qk": 2 * tokens * tokens * hidden,
        "attn_softmax": 5 * heads * tokens * tokens,
        "attn_pv": 2 * tokens * tokens * hidden,
        "attn_out": 2 * tokens * hidden * hidden,
        "ffn_gate": 2 * tokens * hidden * ffn,
        "ffn_up": 2 * tokens * hidden * ffn,
        "ffn_down": 2 * tokens * ffn * hidden,
    }
    dense_layer = sum(dense.values())
    correction_boundary = boundary * dense_layer
    correction_middle = middle * (
        6 * active * hidden * hidden
        + 2 * active * tokens * hidden
        + 5 * heads * active * tokens
        + 2 * active * tokens * hidden
        + 2 * active * hidden * hidden
        + 6 * active * hidden * ffn
    )
    correction = correction_boundary + correction_middle
    result = {}
    for name, targets in {"dense": frozenset(), **{
        variant: VARIANT_TARGETS[variant] for variant in variants
    }}.items():
        approx_layer = dense_layer
        for projection in targets:
            module = projection_module(block, projection)
            dense_projection = dense[projection]
            low_rank_projection = (
                2 * tokens * rank * (module.in_features + module.out_features)
            )
            approx_layer += low_rank_projection - dense_projection
        approx = depth * approx_layer
        result[name] = {
            "approx_flops": float(approx),
            "approx_ratio_to_dense": float(approx / (depth * dense_layer)),
            "correction_flops": float(correction),
            "total_flops": float(approx + correction),
            "total_ratio_to_dense_approx_token50": 0.0,
            "active_tokens": float(active),
        }
    baseline = result["dense"]["total_flops"]
    for value in result.values():
        value["total_ratio_to_dense_approx_token50"] = float(
            value["total_flops"] / baseline
        )
    return result


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if not 0 < args.token_rate <= 1:
        raise ValueError("--token-rate must be in (0, 1]")
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / "results.jsonl"
    summary_path = args.output / "summary.json"
    if result_path.exists() or summary_path.exists():
        raise FileExistsError(f"output already exists under {args.output}")
    executor, _ = load_executor(args)
    adapter = executor.model.segmentation_model[0]
    backbone = adapter.backbone

    from datasets import load_dataset

    calibration_dataset = load_dataset(
        "merve/scene_parse_150",
        split=args.calibration_split,
    )
    rms, calibration = collect_approx_rms(
        executor,
        adapter,
        backbone,
        args,
        calibration_dataset,
    )
    del calibration_dataset
    factors, factorization_rows = factorize_backbone(backbone, rms, args)
    del rms

    approx_names = ["dense", *args.variants]
    metric_names = ["full_oracle"]
    for name in approx_names:
        metric_names.extend((f"{name}_approx", f"{name}_token50"))
    metrics = {name: AreaMetrics() for name in metric_names}
    timings: dict[str, list[float]] = defaultdict(list)
    jaccards: dict[str, list[float]] = defaultdict(list)
    realized_rates: dict[str, list[float]] = defaultdict(list)

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
        image_hw = resized_np.shape[:2]
        original_hw = (original.height, original.width)
        base_canvas = scaled_native_pyramid_level(
            original_np,
            args.base_level,
            image_hw,
        )
        crops = sliding_crops(*image_hw, args.crop_size, args.stride)
        feature_sums = {name: {} for name in metric_names}
        feature_counts = {name: {} for name in metric_names}

        for crop in crops:
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
                _, full_raw = run_full(backbone, full_initial, rope)
                add_crop_features(
                    executor,
                    feature_sums["full_oracle"],
                    feature_counts["full_oracle"],
                    full_source,
                    full_raw,
                    crop,
                    image_hw,
                )

                dense_active = None
                for name in approx_names:
                    targets = (
                        frozenset() if name == "dense" else VARIANT_TARGETS[name]
                    )
                    approx, approx_ms = timed_cuda(
                        lambda selected_targets=targets: run_approx(
                            backbone,
                            base_initial,
                            rope,
                            factors,
                            selected_targets,
                            args.rank,
                            collect_score=True,
                        )
                    )
                    timings[f"{name}_approx"].append(approx_ms)
                    active = token_support(
                        full_np,
                        base_np,
                        approx.received_attention,
                        args.token_rate,
                    )
                    realized_rates[name].append(
                        (active.shape[1] - SPECIAL_TOKENS)
                        / (base_initial.shape[1] - SPECIAL_TOKENS)
                    )
                    if dense_active is None:
                        dense_active = active
                    else:
                        jaccards[name].append(active_jaccard(active, dense_active))
                    (_, corrected_raw), correction_ms = timed_cuda(
                        lambda current_approx=approx, current_active=active: (
                            run_token_correction(
                                backbone,
                                full_initial,
                                current_approx,
                                current_active,
                                rope,
                            )
                        )
                    )
                    timings[f"{name}_correction"].append(correction_ms)
                    add_crop_features(
                        executor,
                        feature_sums[f"{name}_approx"],
                        feature_counts[f"{name}_approx"],
                        corrected_source,
                        approx.raw,
                        crop,
                        image_hw,
                    )
                    add_crop_features(
                        executor,
                        feature_sums[f"{name}_token50"],
                        feature_counts[f"{name}_token50"],
                        corrected_source,
                        corrected_raw,
                        crop,
                        image_hw,
                    )
                    del approx, corrected_raw
                del full_raw, base_source, corrected_source, full_source
                torch.cuda.empty_cache()

        areas = {}
        with torch.autocast("cuda", executor.autocast_dtype):
            for name in metric_names:
                prediction = finish_prediction(
                    executor,
                    feature_sums[name],
                    feature_counts[name],
                    image_hw,
                    original_hw,
                )
                areas[name] = metrics[name].update(prediction, annotation)
        with result_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"index": index, "areas": areas}) + "\n")
        print(
            json.dumps(
                {
                    "completed_index": index,
                    "elapsed_seconds": time.time() - started,
                    "metrics": {
                        name: metric.summary() for name, metric in metrics.items()
                    },
                },
                sort_keys=True,
            ),
            flush=True,
        )

    token_count = SPECIAL_TOKENS + (args.crop_size // 16) ** 2
    timing_summary = {
        name: {
            "mean_ms": float(np.mean(values)),
            "min_ms": float(np.min(values)),
            "max_ms": float(np.max(values)),
            "samples": len(values),
        }
        for name, values in timings.items()
    }
    for name in approx_names:
        timing_summary[f"{name}_total"] = {
            "mean_ms": (
                timing_summary[f"{name}_approx"]["mean_ms"]
                + timing_summary[f"{name}_correction"]["mean_ms"]
            )
        }
    factor_stats = {
        projection: {
            "bytes": float(
                sum(factor.factor_bytes(args.rank) for factor in layer_factors.values())
            ),
            "middle_flop_ratio": dense_linear_flop_ratio(
                projection_module(backbone.blocks[0], projection).in_features,
                projection_module(backbone.blocks[0], projection).out_features,
                args.rank,
            ),
        }
        for projection, layer_factors in factors.items()
        if layer_factors
    }
    summary = {
        "configuration": {
            **vars(args),
            "output": str(args.output),
            "variants": list(args.variants),
            "selection": "residual_energy_x_layermean_received_attention",
            "first_last_full_layers": 2,
        },
        "metrics": {name: metric.summary() for name, metric in metrics.items()},
        "timings": timing_summary,
        "selection": {
            "realized_rates": {
                name: float(np.mean(values)) for name, values in realized_rates.items()
            },
            "jaccard_to_dense": {
                name: float(np.mean(values)) for name, values in jaccards.items()
            },
        },
        "flops": flop_estimates(
            backbone,
            token_count,
            args.rank,
            args.token_rate,
            args.variants,
        ),
        "calibration": calibration,
        "factorization": {
            "seconds": float(sum(row["seconds"] for row in factorization_rows)),
            "rows": factorization_rows,
            "stats": factor_stats,
        },
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
                "timings": summary["timings"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
