"""Evaluate DINOv3 tail-full and L2-L1-L0 AppCorr through the real runtime.

This driver uses SchedulerModule and WorkerModule directly through multiprocessing
queues. It avoids TCP setup while preserving transmission grouping, scheduling,
incremental decode, executor state, and CUDA-event timing.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import platform
import queue
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DINOV3_DIM = 4096
DINOV3_FFN_DIM = 8192
DINOV3_PATCH_TOKENS = 256
DINOV3_PREFIX_TOKENS = 5
DINOV3_TOTAL_TOKENS = DINOV3_PREFIX_TOKENS + DINOV3_PATCH_TOKENS
DINOV3_LAYERS = 40
BOOTSTRAP_SEED = 20260728
MODEL_COMPUTE_EVENT_TYPES = {
    "LOAD_INPUT",
    "PREPARE_TOKENS",
    "FULL_INFERENCE",
    "APPROX_FORWARD",
    "CORRECT_FORWARD",
    "FINAL_FULL_FORWARD",
    "HEAD_INFERENCE",
}
BACKBONE_COMPUTE_EVENT_TYPES = {
    "FULL_INFERENCE",
    "APPROX_FORWARD",
    "CORRECT_FORWARD",
    "FINAL_FULL_FORWARD",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Offload JSON config.")
    parser.add_argument("--data-root", required=True, help="ImageNet validation root.")
    parser.add_argument("--num-samples", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--warmup-batches", type=int, default=1)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--final-full-layers", type=int, default=None)
    parser.add_argument("--token-keep-ratio", type=float, default=None)
    parser.add_argument(
        "--energy-attn-pscore",
        action="store_true",
        help="Use residual_energy x layer-mean CLS attention for partial-token selection.",
    )
    parser.add_argument(
        "--uplink-mbps",
        type=float,
        default=0.0,
        help="Pace each transmitted group by exact serialized bytes; 0 disables pacing.",
    )
    parser.add_argument("--result-timeout", type=float, default=900.0)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--label", type=str, default=None)
    return parser.parse_args()


def load_config(args: argparse.Namespace):
    from offload.common.protocol import ExperimentConfig

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    if args.batch_size is not None:
        raw["batch_size"] = int(args.batch_size)
    raw["device"] = args.device
    raw.setdefault("scheduler_kwargs", {})
    raw.setdefault("transmission_kwargs", {})
    raw.setdefault("appcorr_kwargs", {})

    if args.final_full_layers is not None:
        raw["scheduler_kwargs"]["final_full_layers"] = int(
            args.final_full_layers
        )
    if args.token_keep_ratio is not None:
        raw["appcorr_kwargs"]["token_keep_ratio"] = float(
            args.token_keep_ratio
        )
        raw["appcorr_kwargs"]["token_keep_thres"] = None
    if args.energy_attn_pscore:
        raw["appcorr_kwargs"].update(
            {
                "method": "partial_token",
                "mobile_pscore": "residual_energy",
                "mobile_pscore_weight": 1.0,
                "server_pscore": "cls_attn_prob_layermean",
                "server_pscore_weight": 1.0,
                "pscore_fusion": "multiply",
            }
        )

    raw["image_shape"] = tuple(raw.get("image_shape", (256, 256, 3)))
    raw["patch_size"] = tuple(raw.get("patch_size", (16, 16)))
    return ExperimentConfig(**raw), raw, config_path


def block_flops(query_tokens: int, key_value_tokens: int) -> int:
    """Dominant attention/FFN FLOPs for one ViT-7B block and one sample."""
    query_tokens = int(query_tokens)
    key_value_tokens = int(key_value_tokens)
    projection_flops = 8 * query_tokens * DINOV3_DIM * DINOV3_DIM
    attention_flops = (
        4 * query_tokens * key_value_tokens * DINOV3_DIM
    )
    ffn_flops = (
        6 * query_tokens * DINOV3_DIM * DINOV3_FFN_DIM
    )
    return projection_flops + attention_flops + ffn_flops


def full_backbone_flops() -> int:
    return DINOV3_LAYERS * block_flops(
        DINOV3_TOTAL_TOKENS,
        DINOV3_TOTAL_TOKENS,
    )


def correction_candidates(config, group_id: int) -> int:
    if (
        config.transmission_policy_name
        == "L2L1L0ProgressiveLaplacian"
    ):
        if group_id == 1:
            return DINOV3_PATCH_TOKENS
        l0_num_groups = int(
            config.transmission_kwargs.get("l0_num_groups", 4)
        )
        return DINOV3_PATCH_TOKENS // l0_num_groups

    num_groups = int(config.transmission_kwargs.get("num_groups", 4))
    return DINOV3_PATCH_TOKENS // num_groups


def estimate_request_flops(
    events: list[dict[str, Any]],
    config,
) -> dict[str, int | float]:
    appcorr = config.appcorr_kwargs
    token_keep_ratio = float(appcorr.get("token_keep_ratio", 1.0))
    token_keep_threshold = appcorr.get("token_keep_thres")
    if token_keep_threshold not in {None, "", "null", "None"}:
        raise ValueError(
            "FLOPs estimation requires ratio-based selection "
            "(appcorr_kwargs.token_keep_thres=null)"
        )

    approx_flops = 0
    final_full_flops = 0
    correction_flops = 0
    full_inference_flops = 0

    for event in events:
        event_type = str(event.get("type", ""))
        params = event.get("params") or {}
        if event_type == "FULL_INFERENCE":
            full_inference_flops += full_backbone_flops()
            continue
        if event_type == "APPROX_FORWARD":
            start_layer, end_layer = params.get("layers", (0, 0))
            value = (end_layer - start_layer) * block_flops(
                DINOV3_TOTAL_TOKENS,
                DINOV3_TOTAL_TOKENS,
            )
            if params.get("phase") == "final_full":
                final_full_flops += value
            else:
                approx_flops += value
            continue
        if event_type != "CORRECT_FORWARD":
            continue

        start_layer, end_layer = params.get("layers", (0, 0))
        group_id = int(params.get("group_id", 1))
        candidates = correction_candidates(config, group_id)
        selected_patches = min(
            int(candidates * token_keep_ratio),
            candidates,
        )
        query_tokens = DINOV3_PREFIX_TOKENS + selected_patches
        correction_flops += (end_layer - start_layer) * block_flops(
            query_tokens,
            DINOV3_TOTAL_TOKENS,
        )

    total = (
        approx_flops
        + final_full_flops
        + correction_flops
        + full_inference_flops
    )
    return {
        "approx_flops": approx_flops,
        "correction_flops": correction_flops,
        "final_full_flops": final_full_flops,
        "full_inference_flops": full_inference_flops,
        "total_dominant_flops": total,
        "full_backbone_flops": full_backbone_flops(),
        "ratio_to_full_backbone": total / full_backbone_flops(),
    }


def event_category(event: dict[str, Any]) -> str:
    event_type = str(event.get("type", ""))
    params = event.get("params") or {}
    if (
        event_type == "APPROX_FORWARD"
        and params.get("phase") == "final_full"
    ):
        return "FINAL_FULL_FORWARD"
    return event_type


def exact_patch_wire_bytes(patch) -> int:
    return 4 + len(pickle.dumps(patch))


def prepare_image_batch(
    images: torch.Tensor,
    configured_batch_size: int,
    image_shape: tuple[int, int, int],
) -> tuple[np.ndarray, int]:
    real_batch_size = int(images.shape[0])
    images_hwc = (
        images.permute(0, 2, 3, 1)
        .contiguous()
        .cpu()
        .numpy()
    )
    full_batch = np.zeros(
        (configured_batch_size, *image_shape),
        dtype=np.uint8,
    )
    full_batch[:real_batch_size] = images_hwc
    return full_batch, real_batch_size


def send_request(
    scheduler_queue,
    result_queue,
    policy,
    images: torch.Tensor,
    config,
    *,
    uplink_mbps: float,
    result_timeout: float,
) -> dict[str, Any]:
    image_batch, real_batch_size = prepare_image_batch(
        images,
        config.batch_size,
        config.image_shape,
    )

    payload_bytes = 0
    wire_bytes = 0
    group_stats = {}
    encode_ms = 0.0
    paced_uplink_ms = 0.0
    request_start = time.perf_counter()

    encode_generator = policy.encode(image_batch, config)
    while True:
        encode_start = time.perf_counter()
        try:
            group = next(encode_generator)
        except StopIteration:
            break
        encode_ms += (time.perf_counter() - encode_start) * 1000.0

        group_payload_bytes = sum(len(patch.data) for patch in group)
        group_wire_bytes = sum(exact_patch_wire_bytes(patch) for patch in group)
        payload_bytes += group_payload_bytes
        wire_bytes += group_wire_bytes

        if uplink_mbps > 0:
            transfer_seconds = (
                group_wire_bytes * 8.0 / (uplink_mbps * 1_000_000.0)
            )
            time.sleep(transfer_seconds)
            paced_uplink_ms += transfer_seconds * 1000.0

        arrival_time = time.time()
        for patch in group:
            patch.arrival_time = arrival_time
            scheduler_queue.put(patch)

        group_id = int(group[0].group_id)
        group_stats[str(group_id)] = {
            "patch_count": len(group),
            "payload_bytes": group_payload_bytes,
            "wire_bytes": group_wire_bytes,
        }

    try:
        result = result_queue.get(timeout=result_timeout)
    except queue.Empty as exc:
        raise TimeoutError(
            f"Timed out after {result_timeout:.1f}s waiting for inference"
        ) from exc

    wall_ms = (time.perf_counter() - request_start) * 1000.0
    event_ms = defaultdict(float)
    for event in result.server_events:
        event_ms[event_category(event)] += (
            float(event["end"]) - float(event["start"])
        ) * 1000.0

    return {
        "result": result,
        "real_batch_size": real_batch_size,
        "payload_bytes": payload_bytes,
        "wire_bytes": wire_bytes,
        "group_stats": group_stats,
        "encode_ms": encode_ms,
        "paced_uplink_ms": paced_uplink_ms,
        "wall_ms": wall_ms,
        "event_ms": dict(event_ms),
        "flops": estimate_request_flops(result.server_events, config),
    }


def wait_until_worker_ready(
    control_queue,
    result_queue,
    config,
    *,
    timeout: float,
) -> float:
    """Configure the runtime and wait until model loading reaches the GPU worker."""
    control_queue.put(("CONFIG", config))
    control_queue.put(("TIME_SYNC", None))
    try:
        ready_timestamp = result_queue.get(timeout=timeout)
    except queue.Empty as exc:
        raise TimeoutError(
            f"Timed out after {timeout:.1f}s waiting for worker setup"
        ) from exc
    if not isinstance(ready_timestamp, float):
        raise RuntimeError(
            "Expected a TIME_SYNC readiness timestamp, got "
            f"{type(ready_timestamp).__name__}"
        )
    return ready_timestamp


def percentile_summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "mean": 0.0,
            "p50": 0.0,
            "p95": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def bootstrap_accuracy_summary(
    correct_flags: list[bool],
    *,
    num_resamples: int = 2_000,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    """Return deterministic non-parametric uncertainty for an accuracy."""
    values = np.asarray(correct_flags, dtype=np.float64)
    if values.size == 0:
        return {
            "num_samples": 0,
            "accuracy_percent": 0.0,
            "bootstrap_std_percent": 0.0,
            "ci95_low_percent": 0.0,
            "ci95_high_percent": 0.0,
            "num_resamples": int(num_resamples),
        }
    if num_resamples <= 0:
        raise ValueError("num_resamples must be positive")

    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(num_resamples, dtype=np.float64)
    # Chunking avoids a large [num_resamples, num_samples] index array for
    # full ImageNet validation runs.
    chunk_size = max(1, min(256, num_resamples))
    for start in range(0, num_resamples, chunk_size):
        end = min(start + chunk_size, num_resamples)
        indices = rng.integers(
            0,
            values.size,
            size=(end - start, values.size),
        )
        bootstrap_means[start:end] = values[indices].mean(axis=1)

    low, high = np.percentile(bootstrap_means, (2.5, 97.5))
    return {
        "num_samples": int(values.size),
        "accuracy_percent": float(values.mean() * 100.0),
        "bootstrap_std_percent": float(
            bootstrap_means.std(ddof=1) * 100.0
        ),
        "ci95_low_percent": float(low * 100.0),
        "ci95_high_percent": float(high * 100.0),
        "num_resamples": int(num_resamples),
    }


def runtime_manifest(device: str) -> dict[str, Any]:
    manifest = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_request": device,
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
    }
    if torch.cuda.is_available():
        device_index = torch.device(device).index
        if device_index is None:
            device_index = torch.cuda.current_device()
        manifest.update(
            {
                "gpu_name": torch.cuda.get_device_name(device_index),
                "gpu_capability": list(
                    torch.cuda.get_device_capability(device_index)
                ),
            }
        )
    return manifest


def git_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def build_subset_loaders(args, config):
    from offload.mobile.dataset import get_dataset_loader
    from torch.utils.data import DataLoader, Subset

    metric_loader = get_dataset_loader(
        "imagenet-1k",
        args.data_root,
        batch_size=config.batch_size,
        image_size=config.image_shape[0],
        num_workers=args.num_workers,
    )
    full_loader = metric_loader.get_loader()
    dataset = full_loader.dataset
    num_samples = min(int(args.num_samples), len(dataset))
    if num_samples <= 0:
        raise ValueError("--num-samples must be positive")

    indices = np.linspace(
        0,
        len(dataset) - 1,
        num=num_samples,
        dtype=np.int64,
    ).tolist()
    measured_loader = DataLoader(
        Subset(dataset, indices),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    warmup_count = min(
        max(args.warmup_batches, 0) * config.batch_size,
        len(indices),
    )
    warmup_loader = DataLoader(
        Subset(dataset, indices[:warmup_count]),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return metric_loader, warmup_loader, measured_loader, indices


def run(args: argparse.Namespace) -> dict[str, Any]:
    from offload.policies import get_transmission
    from offload.server.scheduler import SchedulerModule
    from offload.server.worker import WorkerModule

    config, raw_config, config_path = load_config(args)
    (
        metric_loader,
        warmup_loader,
        measured_loader,
        sample_indices,
    ) = build_subset_loaders(args, config)

    scheduler_queue = multiprocessing.Queue()
    worker_queue = multiprocessing.Queue()
    result_queue = multiprocessing.Queue()
    control_queue = multiprocessing.Queue()
    feedback_queue = multiprocessing.Queue()
    scheduler = SchedulerModule(
        scheduler_queue,
        worker_queue,
        control_queue,
        feedback_queue,
    )
    worker = WorkerModule(worker_queue, result_queue, feedback_queue)
    scheduler.start()
    worker.start()
    policy = get_transmission(config.transmission_policy_name)

    requests = []
    event_values = defaultdict(list)
    top1_flags = []
    top5_flags = []
    started_at = time.time()

    try:
        worker_ready_at = wait_until_worker_ready(
            control_queue,
            result_queue,
            config,
            timeout=args.result_timeout,
        )

        for images, _ in warmup_loader:
            send_request(
                scheduler_queue,
                result_queue,
                policy,
                images,
                config,
                uplink_mbps=args.uplink_mbps,
                result_timeout=args.result_timeout,
            )

        processed_samples = 0
        for request_idx, (images, labels) in enumerate(measured_loader):
            request = send_request(
                scheduler_queue,
                result_queue,
                policy,
                images,
                config,
                uplink_mbps=args.uplink_mbps,
                result_timeout=args.result_timeout,
            )
            real_batch_size = request["real_batch_size"]
            predictions = request["result"].output[:real_batch_size]
            label_values = [
                int(value)
                for value in labels[:real_batch_size].tolist()
            ]
            metric_loader.evaluate_batch(predictions, label_values)

            for prediction, label in zip(predictions, label_values):
                top1_flags.append(
                    bool(prediction and prediction[0] == label)
                )
                top5_flags.append(bool(label in prediction))

            for key, value in request["event_ms"].items():
                event_values[key].append(float(value))

            result = request.pop("result")
            request["backbone_compute_ms"] = sum(
                value
                for key, value in request["event_ms"].items()
                if key in BACKBONE_COMPUTE_EVENT_TYPES
            )
            request["model_compute_ms"] = sum(
                value
                for key, value in request["event_ms"].items()
                if key in MODEL_COMPUTE_EVENT_TYPES
            )
            request["cache_size_bytes"] = int(result.cache_size_bytes)
            request["partial_token_kept_patch"] = float(
                result.partial_token_kept_patch
            )
            request["partial_token_full_patch"] = float(
                result.partial_token_full_patch
            )
            request["token_pscore_kept_mass"] = float(
                result.token_pscore_kept_mass
            )
            request["token_pscore_full_mass"] = float(
                result.token_pscore_full_mass
            )
            request["top1_correct"] = int(
                sum(top1_flags[processed_samples:])
            )
            request["top5_correct"] = int(
                sum(top5_flags[processed_samples:])
            )
            requests.append(request)
            processed_samples += real_batch_size

            summary = metric_loader.get_summary()
            print(
                f"[{request_idx + 1}/{len(measured_loader)}] "
                f"samples={processed_samples} "
                f"top1={summary['top1_acc']:.2f}% "
                f"wall={request['wall_ms']:.1f}ms",
                flush=True,
            )
    finally:
        control_queue.put(("STOP", None))
        for process in (scheduler, worker):
            process.join(timeout=30)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)

    dataset_summary = metric_loader.get_summary()
    wall_values = [float(request["wall_ms"]) for request in requests]
    encode_values = [float(request["encode_ms"]) for request in requests]
    cache_values = [
        float(request["cache_size_bytes"]) for request in requests
    ]
    backbone_compute_values = [
        float(request["backbone_compute_ms"]) for request in requests
    ]
    model_compute_values = [
        float(request["model_compute_ms"]) for request in requests
    ]
    per_sample_payload = (
        sum(request["payload_bytes"] for request in requests)
        / max(dataset_summary["total_samples"], 1)
    )
    per_sample_wire = (
        sum(request["wire_bytes"] for request in requests)
        / max(dataset_summary["total_samples"], 1)
    )
    total_kept = sum(
        request["partial_token_kept_patch"] for request in requests
    )
    total_full = sum(
        request["partial_token_full_patch"] for request in requests
    )
    flops_values = [
        float(request["flops"]["ratio_to_full_backbone"])
        for request in requests
    ]

    label = args.label or (
        f"{config_path.stem}_n"
        f"{config.scheduler_kwargs.get('final_full_layers', 0)}_"
        f"k{config.appcorr_kwargs.get('token_keep_ratio', 1.0)}"
    )
    output = {
        "label": label,
        "git_commit": git_revision(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_seconds": time.time() - started_at,
        "config_path": str(config_path),
        "config": asdict(config),
        "sample_indices": sample_indices,
        "dataset_summary": dataset_summary,
        "top1_flags": top1_flags,
        "top5_flags": top5_flags,
        "top1_bootstrap": bootstrap_accuracy_summary(top1_flags),
        "top5_bootstrap": bootstrap_accuracy_summary(
            top5_flags,
            seed=BOOTSTRAP_SEED + 1,
        ),
        "request_latency_ms": percentile_summary(wall_values),
        "backbone_compute_ms": percentile_summary(
            backbone_compute_values
        ),
        "model_compute_ms": percentile_summary(model_compute_values),
        "mobile_encode_ms": percentile_summary(encode_values),
        "event_duration_ms": {
            key: percentile_summary(values)
            for key, values in sorted(event_values.items())
        },
        "cache_size_bytes": percentile_summary(cache_values),
        "avg_payload_bytes_per_sample": per_sample_payload,
        "avg_wire_bytes_per_sample": per_sample_wire,
        "avg_partial_token_keep_ratio": (
            total_kept / total_full if total_full > 0 else 1.0
        ),
        "dominant_flops_ratio": percentile_summary(flops_values),
        "uplink_mbps": float(args.uplink_mbps),
        "num_warmup_batches": int(args.warmup_batches),
        "num_measured_requests": len(requests),
        "worker_ready_at": worker_ready_at,
        "runtime_manifest": runtime_manifest(args.device),
        "requests": requests,
        "raw_config": raw_config,
    }

    print(json.dumps(
        {
            "label": label,
            "dataset_summary": dataset_summary,
            "request_latency_ms": output["request_latency_ms"],
            "event_duration_ms": output["event_duration_ms"],
            "dominant_flops_ratio": output["dominant_flops_ratio"],
            "avg_payload_bytes_per_sample": per_sample_payload,
            "avg_wire_bytes_per_sample": per_sample_wire,
        },
        indent=2,
    ))
    if args.output_json:
        output_path = Path(args.output_json)
        if not output_path.is_absolute():
            output_path = REPO_ROOT / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(output, handle, indent=2)
        print(f"[eval] wrote {output_path}")
    return output


def main() -> None:
    args = parse_args()
    if args.num_samples <= 0:
        raise SystemExit("--num-samples must be positive")
    if args.batch_size is not None and args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.warmup_batches < 0:
        raise SystemExit("--warmup-batches must be non-negative")
    if args.uplink_mbps < 0:
        raise SystemExit("--uplink-mbps must be non-negative")
    run(args)


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
