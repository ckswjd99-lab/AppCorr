#!/usr/bin/env python3
"""Merge complete ImageNet policy-evaluation shards into a compact summary."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.experiments.jacobian_support_oracle import (  # noqa: E402
    add_sums,
    finish_tensor_sums,
)


SUM_KEYS = ("token_feature_sums", "pooled_feature_sums", "logit_sums")


def git_output(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("shards", nargs="+", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            Path(__file__).resolve().parent
            / "results"
            / "jacobian_policy_imagenet1k_summary.json"
        ),
    )
    return parser.parse_args()


def validate_shards(shards: list[dict]) -> None:
    if any(shard.get("status") != "complete" for shard in shards):
        raise ValueError("All shards must have status=complete")
    first = shards[0]["manifest"]
    invariant_keys = (
        "git_commit",
        "data_root",
        "total_dataset_samples",
        "num_shards",
        "base_level",
        "image_size",
        "keep_special_tokens",
        "policy_json",
        "targets",
        "policy_metadata",
    )
    for index, shard in enumerate(shards[1:], start=1):
        mismatches = [
            key
            for key in invariant_keys
            if shard["manifest"].get(key) != first.get(key)
        ]
        if mismatches:
            raise ValueError(
                f"Shard {index} manifest mismatch: {', '.join(mismatches)}"
            )
    expected_shards = int(first["num_shards"])
    shard_indices = {int(shard["manifest"]["shard_index"]) for shard in shards}
    if len(shards) != expected_shards or shard_indices != set(range(expected_shards)):
        raise ValueError(
            f"Expected shard indices 0..{expected_shards - 1}, got "
            f"{sorted(shard_indices)}"
        )
    total = sum(int(shard["processed_samples"]) for shard in shards)
    if total != int(first["total_dataset_samples"]):
        raise ValueError(
            f"Shards contain {total} samples, expected "
            f"{first['total_dataset_samples']}"
        )


def merge_methods(shards: list[dict]) -> dict[str, dict]:
    merged = {}
    for name in shards[0]["methods"]:
        result = {
            "samples": 0,
            "top1_correct": 0,
            "top5_correct": 0,
            "l0_top1_match": 0,
        }
        for key in SUM_KEYS:
            if key in shards[0]["methods"][name]:
                result[key] = {}
        for shard in shards:
            source = shard["methods"][name]
            for key in (
                "samples",
                "top1_correct",
                "top5_correct",
                "l0_top1_match",
            ):
                result[key] += int(source[key])
            for key in SUM_KEYS:
                if key in result:
                    add_sums(result[key], source[key])
        samples = result["samples"]
        result["top1_accuracy"] = result["top1_correct"] / samples
        result["top5_accuracy"] = result["top5_correct"] / samples
        result["l0_top1_match_rate"] = result["l0_top1_match"] / samples
        for key in SUM_KEYS:
            if key in result:
                result[key.removesuffix("_sums")] = finish_tensor_sums(result[key])
                del result[key]
        merged[name] = result
    return merged


def main() -> None:
    args = parse_args()
    shards = [
        json.loads(path.read_text(encoding="utf-8")) for path in args.shards
    ]
    validate_shards(shards)
    methods = merge_methods(shards)
    full = methods["l0_full"]
    approx = methods["l2_approx"]
    for method in methods.values():
        method["top1_delta_vs_l0"] = (
            method["top1_accuracy"] - full["top1_accuracy"]
        )
        method["top5_delta_vs_l0"] = (
            method["top5_accuracy"] - full["top5_accuracy"]
        )
        method["top1_gain_vs_l2"] = (
            method["top1_accuracy"] - approx["top1_accuracy"]
        )

    runtimes = [float(shard["runtime_seconds"]) for shard in shards]
    total_samples = sum(int(shard["processed_samples"]) for shard in shards)
    parallel_seconds = max(runtimes)
    first = shards[0]["manifest"]
    summary = {
        "schema_version": 1,
        "experiment": "ImageNet-1K L2-to-L0 exact-difference pruning policy",
        "implementation": (
            "dense PyTorch structured-support oracle; accuracy validation, "
            "not optimized-kernel latency"
        ),
        "samples": total_samples,
        "base_level": first["base_level"],
        "image_size": first["image_size"],
        "keep_special_tokens": first["keep_special_tokens"],
        "pruning_targets": first["targets"],
        "policy_metadata": first["policy_metadata"],
        "evaluation_source_commit": git_output("rev-parse", "HEAD"),
        "launch_base_commit": first["git_commit"],
        "hardware": first["cuda_device"],
        "execution": {
            "num_shards": first["num_shards"],
            "batch_size_per_gpu": first["batch_size"],
            "shard_runtime_seconds": runtimes,
            "parallel_wall_clock_seconds": parallel_seconds,
            "aggregate_gpu_seconds": sum(runtimes),
            "aggregate_images_per_parallel_second": (
                total_samples / max(parallel_seconds, 1e-12)
            ),
        },
        "methods": methods,
        "notes": [
            "L2 is produced by two pyrDown and two pyrUp operations at 256x256.",
            (
                "CLS/register tokens are always corrected; input-token keep "
                "ratios apply only to the 256 patch tokens."
                if first["keep_special_tokens"]
                else "Input-token keep ratios apply to all backbone tokens."
            ),
            (
                "Attention and FFN corrections use exact nonlinear output "
                "differences on block-structured support."
            ),
            (
                "Reported pruning targets use the equal-component work model; "
                "they are not measured kernel-FLOP or latency reductions."
            ),
        ],
    }
    if not math.isfinite(summary["execution"]["parallel_wall_clock_seconds"]):
        raise RuntimeError("Non-finite runtime")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary["methods"], indent=2, sort_keys=True))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
