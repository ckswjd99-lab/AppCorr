#!/usr/bin/env python3
"""Merge resume-safe ADE20K block-sparse result shards."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import torch


NUM_CLASSES = 150
HIDDEN_SIZE = 4096
NUM_HEADS = 32
MIDDLE_LAYERS = 36
SPECIAL_TOKENS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--expected-start", type=int, default=0)
    parser.add_argument("--expected-count", type=int, default=200)
    parser.add_argument("--flop-summary", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows_by_index: dict[int, dict[str, Any]] = {}
    for path in args.results:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                index = int(row["index"])
                if index in rows_by_index:
                    raise RuntimeError(f"Duplicate dataset index {index} in {path}")
                rows_by_index[index] = row

    expected = set(range(args.expected_start, args.expected_start + args.expected_count))
    actual = set(rows_by_index)
    if actual != expected:
        raise RuntimeError(
            f"Dataset index mismatch: missing={sorted(expected - actual)}, "
            f"unexpected={sorted(actual - expected)}"
        )

    areas = defaultdict(
        lambda: [torch.zeros(NUM_CLASSES, dtype=torch.float64) for _ in range(4)]
    )
    for row in rows_by_index.values():
        for policy_name, policy_areas in row["areas"].items():
            for area_index, value in enumerate(policy_areas):
                areas[policy_name][area_index] += torch.as_tensor(
                    value,
                    dtype=torch.float64,
                )

    metrics = {}
    for policy_name, (intersection, union, _predicted, label) in areas.items():
        iou = torch.where(union > 0, intersection / union, torch.nan)
        metrics[policy_name] = {
            "mIoU": float(torch.nanmean(iou).item() * 100.0),
            "aAcc": float((intersection.sum() / label.sum()).item() * 100.0),
        }

    flops = {}
    cache = {}
    if args.flop_summary is not None:
        source = json.loads(args.flop_summary.read_text(encoding="utf-8"))
        dense_reference = next(iter(source["flops_per_crop"].values()))[
            "dense_full_reference"
        ]
        for policy_name, value in source["flops_per_crop"].items():
            correction_ratio = value["correction_total"] / dense_reference
            flops[policy_name] = {
                "correction_tflops_per_crop": value["correction_total"] / 1e12,
                "correction_vs_dense_full": correction_ratio,
                "correction_savings_percent": (1.0 - correction_ratio) * 100.0,
                "approx_plus_correction_tflops_per_crop": value["total"] / 1e12,
            }
        crop_size = int(source["configuration"]["crop_size"])
        tokens = SPECIAL_TOKENS + (crop_size // 16) ** 2
        bytes_per = 2
        boundary_states = (
            (MIDDLE_LAYERS + 1) * tokens * HIDDEN_SIZE * bytes_per
        )
        ffn_prenorm_states = MIDDLE_LAYERS * tokens * HIDDEN_SIZE * bytes_per
        cache = {
            "layer_boundary_state_bytes": boundary_states,
            "ffn_prenorm_state_bytes": ffn_prenorm_states,
            "base_state_total_bytes": boundary_states + ffn_prenorm_states,
            "cached_pre_softmax_logits_bytes": (
                MIDDLE_LAYERS * NUM_HEADS * tokens * tokens * bytes_per
            ),
            "alternative_cached_kv_bytes": (
                MIDDLE_LAYERS * 2 * tokens * HIDDEN_SIZE * bytes_per
            ),
            "ffn_extra_gate_up_cache_bytes": 0,
        }

    result = {
        "dataset_indices": [args.expected_start, args.expected_start + args.expected_count - 1],
        "num_images": args.expected_count,
        "metrics": dict(sorted(metrics.items())),
        "flops": dict(sorted(flops.items())),
        "cache": cache,
        "source_results": [str(path) for path in args.results],
    }
    encoded = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
