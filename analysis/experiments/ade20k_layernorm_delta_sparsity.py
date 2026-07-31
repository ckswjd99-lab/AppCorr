#!/usr/bin/env python3
"""Measure pre-attention/pre-FFN LayerNorm delta sparsity on ADE20K."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.experiments import ade20k_block_sparse_exact_eval as experiment


ABSOLUTE_THRESHOLDS = (
    0.0,
    1e-6,
    1e-5,
    1e-4,
    1e-3,
    1e-2,
    5e-2,
    1e-1,
)
RMS_THRESHOLDS = (0.01, 0.05, 0.1, 0.2, 0.5)
KEEP_RATIOS = (0.25, 0.5, 0.75)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-images", type=int, default=4)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--crops-per-image", type=int, default=1)
    parser.add_argument("--short-side", type=int, default=896)
    parser.add_argument("--crop-size", type=int, default=896)
    parser.add_argument("--stride", type=int, default=596)
    parser.add_argument("--base-level", type=int, default=2)
    parser.add_argument("--token-rate", type=float, default=0.5)
    parser.add_argument("--attention-rate", type=float, default=0.5)
    parser.add_argument("--ffn-rate", type=float, default=0.5)
    parser.add_argument("--query-block", type=int, default=8)
    parser.add_argument("--key-block", type=int, default=16)
    parser.add_argument("--head-group", type=int, default=4)
    parser.add_argument("--query-chunk", type=int, default=64)
    parser.add_argument("--ffn-token-block", type=int, default=8)
    parser.add_argument("--ffn-channel-block", type=int, default=128)
    parser.set_defaults(correction_backend="dense_mask")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/ade20k_layernorm_delta_sparsity/summary.json"),
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


def ratio_key(value: float) -> str:
    return f"{value:.2f}"


class DeltaStatistics:
    """Streaming element and structured block-energy statistics."""

    def __init__(self, token_block_size: int, channel_block_size: int) -> None:
        self.token_block_size = token_block_size
        self.channel_block_size = channel_block_size
        self.aggregate: dict[str, dict[str, Any]] = defaultdict(self._new_stats)
        self.layers: dict[str, dict[int, dict[str, Any]]] = defaultdict(
            lambda: defaultdict(self._new_stats)
        )
        self.fixed_channel_energy: dict[str, dict[int, torch.Tensor]] = defaultdict(dict)

    @staticmethod
    def _new_stats() -> dict[str, Any]:
        return {
            "calls": 0,
            "elements": 0,
            "sum_abs": 0.0,
            "sum_sq": 0.0,
            "max_abs": 0.0,
            "absolute_counts": {str(value): 0 for value in ABSOLUTE_THRESHOLDS},
            "rms_counts": {str(value): 0 for value in RMS_THRESHOLDS},
            "local_kept_energy": {ratio_key(value): 0.0 for value in KEEP_RATIOS},
            "local_total_energy": 0.0,
        }

    @staticmethod
    def _merge_scalar_stats(
        stats: dict[str, Any],
        *,
        elements: int,
        sum_abs: float,
        sum_sq: float,
        max_abs: float,
        absolute_counts: dict[str, int],
        rms_counts: dict[str, int],
        local_kept_energy: dict[str, float],
        local_total_energy: float,
    ) -> None:
        stats["calls"] += 1
        stats["elements"] += elements
        stats["sum_abs"] += sum_abs
        stats["sum_sq"] += sum_sq
        stats["max_abs"] = max(stats["max_abs"], max_abs)
        for key, value in absolute_counts.items():
            stats["absolute_counts"][key] += value
        for key, value in rms_counts.items():
            stats["rms_counts"][key] += value
        for key, value in local_kept_energy.items():
            stats["local_kept_energy"][key] += value
        stats["local_total_energy"] += local_total_energy

    def update(self, kind: str, layer_index: int, delta: torch.Tensor) -> None:
        value = delta.detach().float()
        absolute = value.abs()
        elements = value.numel()
        sum_abs = float(absolute.sum(dtype=torch.float64).item())
        sum_sq_tensor = value.square().sum(dtype=torch.float64)
        sum_sq = float(sum_sq_tensor.item())
        rms = math.sqrt(sum_sq / max(elements, 1))
        max_abs = float(absolute.max().item())
        absolute_counts = {
            str(threshold): int((absolute <= threshold).sum().item())
            for threshold in ABSOLUTE_THRESHOLDS
        }
        rms_counts = {
            str(threshold): int((absolute <= threshold * rms).sum().item())
            for threshold in RMS_THRESHOLDS
        }

        batch, tokens, channels = value.shape
        token_blocks = math.ceil(tokens / self.token_block_size)
        channel_blocks = math.ceil(channels / self.channel_block_size)
        padded = F.pad(
            value,
            (
                0,
                channel_blocks * self.channel_block_size - channels,
                0,
                token_blocks * self.token_block_size - tokens,
            ),
        )
        block_energy = (
            padded.square()
            .reshape(
                batch,
                token_blocks,
                self.token_block_size,
                channel_blocks,
                self.channel_block_size,
            )
            .sum(dim=(2, 4), dtype=torch.float64)
        )
        sorted_energy = block_energy.sort(dim=-1, descending=True).values
        local_total_energy = float(block_energy.sum().item())
        local_kept_energy = {}
        for keep_ratio in KEEP_RATIOS:
            keep = max(1, math.ceil(channel_blocks * keep_ratio))
            local_kept_energy[ratio_key(keep_ratio)] = float(
                sorted_energy[..., :keep].sum().item()
            )

        channel_energy = (
            value.square()
            .sum(dim=(0, 1), dtype=torch.float64)
            .reshape(channel_blocks, self.channel_block_size)
            .sum(dim=-1)
            .cpu()
        )
        existing = self.fixed_channel_energy[kind].get(layer_index)
        if existing is None:
            self.fixed_channel_energy[kind][layer_index] = channel_energy
        else:
            existing.add_(channel_energy)

        for stats in (
            self.aggregate[kind],
            self.layers[kind][layer_index],
        ):
            self._merge_scalar_stats(
                stats,
                elements=elements,
                sum_abs=sum_abs,
                sum_sq=sum_sq,
                max_abs=max_abs,
                absolute_counts=absolute_counts,
                rms_counts=rms_counts,
                local_kept_energy=local_kept_energy,
                local_total_energy=local_total_energy,
            )

    def _summarize_stats(self, stats: dict[str, Any]) -> dict[str, Any]:
        elements = max(int(stats["elements"]), 1)
        total_energy = max(float(stats["local_total_energy"]), torch.finfo(torch.float64).eps)
        return {
            "calls": int(stats["calls"]),
            "elements": int(stats["elements"]),
            "mean_abs": float(stats["sum_abs"]) / elements,
            "rms": math.sqrt(float(stats["sum_sq"]) / elements),
            "max_abs": float(stats["max_abs"]),
            "absolute_cdf": {
                key: value / elements
                for key, value in stats["absolute_counts"].items()
            },
            "relative_to_call_rms_cdf": {
                key: value / elements
                for key, value in stats["rms_counts"].items()
            },
            "dynamic_8x128_block_energy_retained": {
                key: value / total_energy
                for key, value in stats["local_kept_energy"].items()
            },
        }

    def summary(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for kind, stats in self.aggregate.items():
            fixed_kept = {ratio_key(value): 0.0 for value in KEEP_RATIOS}
            fixed_total = 0.0
            for channel_energy in self.fixed_channel_energy[kind].values():
                sorted_energy = channel_energy.sort(descending=True).values
                fixed_total += float(sorted_energy.sum().item())
                for keep_ratio in KEEP_RATIOS:
                    keep = max(1, math.ceil(sorted_energy.numel() * keep_ratio))
                    fixed_kept[ratio_key(keep_ratio)] += float(
                        sorted_energy[:keep].sum().item()
                    )
            fixed_total = max(fixed_total, torch.finfo(torch.float64).eps)
            kind_summary = self._summarize_stats(stats)
            kind_summary["fixed_per_layer_128_channel_block_energy_retained"] = {
                key: value / fixed_total for key, value in fixed_kept.items()
            }
            kind_summary["layers"] = {
                str(layer_index): self._summarize_stats(layer_stats)
                for layer_index, layer_stats in sorted(self.layers[kind].items())
            }
            result[kind] = kind_summary
        return result


def run_correction_with_observer(
    backbone: Any,
    full_initial: torch.Tensor,
    base_states: list[torch.Tensor],
    base_attention_states: list[torch.Tensor],
    rope: Any,
    active_indices: torch.Tensor,
    policy: experiment.Policy,
    args: argparse.Namespace,
    ffn_weight_scores: list[torch.Tensor],
    statistics: DeltaStatistics,
) -> None:
    state = full_initial
    depth = len(backbone.blocks)
    middle_end = depth - experiment.FULL_SUFFIX_LAYERS
    support = experiment.SupportAccumulator()
    for layer_index, block in enumerate(backbone.blocks):
        if (
            layer_index < experiment.FULL_PREFIX_LAYERS
            or layer_index >= middle_end
        ):
            state = block(state, rope)
            continue

        normalized_base = experiment.gather_tokens(
            block.norm1(base_states[layer_index]),
            active_indices,
        )
        normalized_current = experiment.gather_tokens(
            block.norm1(state),
            active_indices,
        )
        statistics.update(
            "pre_attention_norm_delta",
            layer_index,
            normalized_current - normalized_base,
        )
        corrected_attn = experiment.corrected_attention(
            block,
            base_states[layer_index],
            base_attention_states[layer_index],
            state,
            active_indices,
            rope,
            policy.attention_rate,
            args,
            support,
        )

        base_attn = experiment.gather_tokens(
            base_attention_states[layer_index],
            active_indices,
        )
        statistics.update(
            "pre_ffn_norm_delta",
            layer_index,
            block.norm2(corrected_attn) - block.norm2(base_attn),
        )
        corrected_next = experiment.corrected_ffn(
            block,
            base_attention_states[layer_index],
            corrected_attn,
            base_states[layer_index + 1],
            active_indices,
            policy.ffn_rate,
            args,
            support,
            ffn_weight_scores[layer_index],
        )
        state = experiment.scatter_tokens(
            base_states[layer_index + 1],
            active_indices,
            corrected_next,
        )


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    if args.crops_per_image <= 0:
        raise ValueError("crops-per-image must be positive")
    executor, _ = experiment.load_executor(args)
    adapter = executor.model.segmentation_model[0]
    backbone = adapter.backbone
    ffn_weight_scores = [
        (
            block.mlp.w2.weight.float().norm(dim=1)
            * block.mlp.w3.weight.float().norm(dim=0)
        ).to(block.mlp.w2.weight.dtype)
        for block in backbone.blocks
    ]
    statistics = DeltaStatistics(
        args.ffn_token_block,
        args.ffn_channel_block,
    )
    policy = experiment.Policy(
        "probe",
        args.token_rate,
        args.attention_rate,
        args.ffn_rate,
    )

    from datasets import load_dataset

    dataset = load_dataset("merve/scene_parse_150", split="validation")
    end_index = min(args.start_index + args.num_images, len(dataset))
    measured_crops = 0
    for index in range(args.start_index, end_index):
        sample = dataset[index]
        original = sample["image"].convert("RGB")
        original_np = np.asarray(original, dtype=np.uint8)
        resized = executor._resize_short_side(original, args.short_side)
        resized_np = np.asarray(resized, dtype=np.uint8)
        image_hw = (resized_np.shape[0], resized_np.shape[1])
        base_canvas_np = experiment.scaled_native_pyramid_level(
            original_np,
            args.base_level,
            image_hw,
        )
        crops = experiment.sliding_crops(
            *image_hw,
            args.crop_size,
            args.stride,
        )[: args.crops_per_image]
        for crop in crops:
            y1, y2, x1, x2 = crop
            full_np = np.ascontiguousarray(resized_np[y1:y2, x1:x2])
            base_np = np.ascontiguousarray(base_canvas_np[y1:y2, x1:x2])
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
                full_initial = full_source["x_backbone"]
                base_initial, _ = backbone.prepare_tokens_with_masks(
                    base_tensor
                )
                (
                    base_states,
                    base_attention_states,
                    attention_score,
                    _base_raw,
                ) = experiment.run_base(
                    backbone,
                    base_initial,
                    full_source["rope_sincos"],
                )
                active = experiment.token_support(
                    full_np,
                    base_np,
                    attention_score,
                    policy.token_rate,
                )
                run_correction_with_observer(
                    backbone,
                    full_initial,
                    base_states,
                    base_attention_states,
                    full_source["rope_sincos"],
                    active,
                    policy,
                    args,
                    ffn_weight_scores,
                    statistics,
                )
            measured_crops += 1
            print(
                json.dumps(
                    {
                        "completed_index": index,
                        "measured_crops": measured_crops,
                    }
                ),
                flush=True,
            )

    result = {
        "configuration": {
            "start_index": args.start_index,
            "num_images": args.num_images,
            "measured_crops": measured_crops,
            "base_level": args.base_level,
            "token_rate": args.token_rate,
            "attention_rate": args.attention_rate,
            "ffn_rate": args.ffn_rate,
            "token_block_size": args.ffn_token_block,
            "channel_block_size": args.ffn_channel_block,
        },
        "statistics": statistics.summary(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"summary": str(args.output)}))


if __name__ == "__main__":
    main()
