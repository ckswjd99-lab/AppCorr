#!/usr/bin/env python3
"""Evaluate layerwise exact-difference pruning policies on ImageNet-1K.

The driver is deterministic, shardable, and resume-safe.  It evaluates the
stock L0 full-resolution path, the stock L2 approximate path, and all requested
mixed policies in the same model pass so every comparison uses identical
inputs and classifier weights.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.experiments.jacobian_support_oracle import (
    Dinov3JacobianOracle,
    add_sums,
    finish_tensor_sums,
    low_resolution_canvas,
)


DEFAULT_POLICY = (
    Path(__file__).resolve().parent
    / "results"
    / "jacobian_pruning_policy_equal_work.json"
)
SUM_KEYS = ("token_feature_sums", "pooled_feature_sums", "logit_sums")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("~/data/imagenet_val").expanduser(),
    )
    parser.add_argument("--policy-json", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--targets", default="0.25,0.5,0.75")
    parser.add_argument("--base-level", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument(
        "--keep-special-tokens",
        action="store_true",
        help=(
            "Always correct the CLS/register prefix; pruning ratios then apply "
            "only to patch tokens."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum total samples in this shard; zero evaluates the full shard.",
    )
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/analysis/jacobian_policy_imagenet_shard.json"),
    )
    return parser.parse_args()


def git_output(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_policies(
    path: Path,
    targets: list[float],
) -> tuple[dict[str, list[dict[str, float]]], dict[str, dict]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    policies = {}
    metadata = {}
    for target in targets:
        candidates = [
            row
            for row in payload["policies"]
            if math.isclose(
                float(row["target_pruning_rate"]),
                target,
                rel_tol=0,
                abs_tol=1e-9,
            )
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one policy for pruning target {target}, got "
                f"{len(candidates)}"
            )
        name = f"prune_{round(target * 100):02d}"
        policies[name] = candidates[0]["schedule"]
        metadata[name] = {
            key: value
            for key, value in candidates[0].items()
            if key != "schedule"
        }
    return policies, metadata


def empty_method_stats(has_reference_sums: bool) -> dict:
    stats = {
        "samples": 0,
        "top1_correct": 0,
        "top5_correct": 0,
        "l0_top1_match": 0,
    }
    if has_reference_sums:
        for key in SUM_KEYS:
            stats[key] = {
                "error_sq": 0.0,
                "reference_sq": 0.0,
                "actual_sq": 0.0,
                "dot": 0.0,
            }
    return stats


def make_state(
    args: argparse.Namespace,
    *,
    total_dataset_samples: int,
    shard_samples: int,
    policy_metadata: dict[str, dict],
) -> dict:
    method_names = ["l0_full", "l2_approx", *policy_metadata]
    return {
        "schema_version": 1,
        "status": "running",
        "manifest": {
            "git_commit": git_output("rev-parse", "HEAD"),
            "git_branch": git_output("branch", "--show-current"),
            "data_root": str(args.data_root.resolve()),
            "total_dataset_samples": total_dataset_samples,
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "shard_samples": shard_samples,
            "base_level": args.base_level,
            "image_size": args.image_size,
            "keep_special_tokens": args.keep_special_tokens,
            "batch_size": args.batch_size,
            "device": args.device,
            "policy_json": str(args.policy_json.resolve()),
            "targets": [
                float(item) for item in args.targets.split(",") if item.strip()
            ],
            "policy_metadata": policy_metadata,
            "torch_version": torch.__version__,
            "cuda_device": (
                torch.cuda.get_device_name(torch.device(args.device))
                if torch.device(args.device).type == "cuda"
                else None
            ),
        },
        "processed_samples": 0,
        "runtime_seconds": 0.0,
        "methods": {
            name: empty_method_stats(name != "l0_full")
            for name in method_names
        },
    }


def validate_resume(state: dict, expected: dict) -> None:
    old = state["manifest"]
    new = expected["manifest"]
    keys = (
        "git_commit",
        "data_root",
        "total_dataset_samples",
        "shard_index",
        "num_shards",
        "shard_samples",
        "base_level",
        "image_size",
        "keep_special_tokens",
        "policy_json",
        "targets",
    )
    mismatches = [key for key in keys if old.get(key) != new.get(key)]
    if mismatches:
        raise RuntimeError(
            "Refusing incompatible resume; mismatched fields: "
            + ", ".join(mismatches)
        )


def finish_state(state: dict) -> dict:
    result = json.loads(json.dumps(state))
    for method in result["methods"].values():
        samples = max(int(method["samples"]), 1)
        method["top1_accuracy"] = method["top1_correct"] / samples
        method["top5_accuracy"] = method["top5_correct"] / samples
        method["l0_top1_match_rate"] = method["l0_top1_match"] / samples
        for key in SUM_KEYS:
            if key in method:
                method[key.removesuffix("_sums")] = finish_tensor_sums(method[key])
    return result


def update_metrics(
    state: dict,
    outputs: dict[str, dict[str, object]],
    labels: torch.Tensor,
) -> None:
    labels = labels.to(outputs["l0_full"]["logits"].device)
    full_prediction = outputs["l0_full"]["logits"].argmax(dim=-1)
    for name, output in outputs.items():
        logits = output["logits"]
        prediction = logits.argmax(dim=-1)
        top5 = logits.topk(k=min(5, logits.shape[-1]), dim=-1).indices
        stats = state["methods"][name]
        stats["samples"] += labels.numel()
        stats["top1_correct"] += int((prediction == labels).sum().item())
        stats["top5_correct"] += int(
            (top5 == labels[:, None]).any(dim=-1).sum().item()
        )
        stats["l0_top1_match"] += int(
            (prediction == full_prediction).sum().item()
        )
        for key in SUM_KEYS:
            if key in output:
                add_sums(stats[key], output[key])


def make_l2_batch(images: torch.Tensor, level: int) -> torch.Tensor:
    canvases = []
    for image in images:
        image_bhwc = image.permute(1, 2, 0).numpy()
        canvas = low_resolution_canvas(image_bhwc, level)
        canvases.append(torch.from_numpy(np.ascontiguousarray(canvas)).permute(2, 0, 1))
    return torch.stack(canvases)


def main() -> None:
    args = parse_args()
    if args.num_shards <= 0 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    targets = [
        float(item) for item in args.targets.split(",") if item.strip()
    ]
    if not targets or any(not 0 <= target <= 1 for target in targets):
        raise ValueError("targets must contain pruning rates in [0, 1]")

    policies, policy_metadata = load_policies(args.policy_json, targets)
    transform = transforms.Compose([
        transforms.Resize(args.image_size),
        transforms.CenterCrop(args.image_size),
        transforms.PILToTensor(),
    ])
    dataset = datasets.ImageFolder(str(args.data_root), transform=transform)
    indices = list(range(args.shard_index, len(dataset), args.num_shards))
    if args.max_samples > 0:
        indices = indices[: args.max_samples]
    expected = make_state(
        args,
        total_dataset_samples=len(dataset),
        shard_samples=len(indices),
        policy_metadata=policy_metadata,
    )
    if args.output.exists():
        state = json.loads(args.output.read_text(encoding="utf-8"))
        validate_resume(state, expected)
        print(
            f"Resuming {args.output} at "
            f"{state['processed_samples']}/{len(indices)} samples"
        )
    else:
        state = expected
    processed = int(state["processed_samples"])
    if processed > len(indices):
        raise RuntimeError("Resume position exceeds this shard")

    loader = DataLoader(
        Subset(dataset, indices[processed:]),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    oracle = Dinov3JacobianOracle(
        device=torch.device(args.device),
        image_size=args.image_size,
        layers=None,
        query_chunk=16,
        support_ratios=[0.5],
        tail_epsilons=[0.1],
        query_block=8,
        key_block=16,
        head_group=4,
        ffn_channel_block=128,
        ffn_token_block=8,
    )

    start = time.perf_counter()
    progress = tqdm(
        loader,
        initial=processed // max(args.batch_size, 1),
        total=math.ceil(len(indices) / args.batch_size),
        dynamic_ncols=True,
    )
    try:
        for batch_index, (images, labels) in enumerate(progress, start=1):
            l2_images = make_l2_batch(images, args.base_level)
            outputs = oracle.exact_policy_classification_batch(
                l2_images,
                images,
                policies,
                always_keep_special_tokens=args.keep_special_tokens,
            )
            update_metrics(state, outputs, labels)
            state["processed_samples"] += labels.numel()
            if batch_index % args.save_every == 0:
                state["runtime_seconds"] += time.perf_counter() - start
                atomic_write_json(args.output, finish_state(state))
                start = time.perf_counter()
            full_acc = (
                state["methods"]["l0_full"]["top1_correct"]
                / max(state["processed_samples"], 1)
            )
            progress.set_postfix(
                samples=state["processed_samples"],
                full_top1=f"{full_acc:.3f}",
            )
    finally:
        state["runtime_seconds"] += time.perf_counter() - start
        state["status"] = (
            "complete"
            if state["processed_samples"] == len(indices)
            else "interrupted"
        )
        atomic_write_json(args.output, finish_state(state))


if __name__ == "__main__":
    main()
