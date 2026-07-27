#!/usr/bin/env python3
"""B200 microbenchmark for split JVP and product-delta attention correction.

The benchmark uses the token counts and batch sizes of the checked-in
ImageNet/COCO/ADE20K/NYUv2 configurations.  Query count is independently
bounded because correction operates on an arrived/selected query bucket.
Selector+packing is timed separately and can also be included in the reported
end-to-end sparse time.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path
import statistics
import sys
from typing import Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.shared.cuda_environment import configure_triton_cuda_environment

CUDA_ENV = configure_triton_cuda_environment()

import torch

from appcorr.models.dinov3.layers.triton_kernels.jacobian_attention import (
    packed_attention_delta_triton,
)
from appcorr.models.dinov3.layers.triton_kernels.jacobian_probability import (
    selected_softmax_jvp_triton,
)


PRESETS = {
    "imagenet": {"batch": 32, "tokens": 261},
    "coco": {"batch": 1, "tokens": 4101},
    "ade20k": {"batch": 1, "tokens": 3141},
    "nyuv2": {"batch": 8, "tokens": 2309},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--presets", default="imagenet,coco,ade20k,nyuv2")
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--tokens", type=int, default=None)
    parser.add_argument("--queries", type=int, default=64)
    parser.add_argument("--heads", type=int, default=32)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--support", default="0.125,0.25,0.5,0.75,1.0")
    parser.add_argument("--key-block", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def cuda_times_ms(
    operation: Callable[[], torch.Tensor | tuple[torch.Tensor, ...]],
    *,
    warmup: int,
    iterations: int,
) -> list[float]:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    times = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = operation()
        end.record()
        end.synchronize()
        times.append(float(start.elapsed_time(end)))
        del result
    return times


def bootstrap_median_ci(
    values: list[float],
    *,
    repetitions: int = 2_000,
    seed: int = 19,
) -> tuple[float, float]:
    data = np.asarray(values, dtype=np.float64)
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(data), size=(repetitions, len(data)))
    medians = np.median(data[indices], axis=1)
    low, high = np.quantile(medians, (0.025, 0.975))
    return float(low), float(high)


def summarize(values: list[float]) -> dict[str, float]:
    ci_low, ci_high = bootstrap_median_ci(values)
    return {
        "p50_ms": float(statistics.median(values)),
        "p95_ms": float(np.quantile(values, 0.95)),
        "mean_ms": float(statistics.fmean(values)),
        "p50_ci95_low_ms": ci_low,
        "p50_ci95_high_ms": ci_high,
    }


def make_key_index(tokens: int, ratio: float, block: int, device: torch.device) -> torch.Tensor:
    total_blocks = math.ceil(tokens / block)
    kept_blocks = max(1, min(total_blocks, math.ceil(total_blocks * ratio)))
    # Deterministic interleaving avoids benchmarking only a contiguous prefix.
    block_index = torch.linspace(
        0,
        total_blocks - 1,
        kept_blocks,
        device=device,
        dtype=torch.float64,
    ).round().long().unique(sorted=True)
    offsets = torch.arange(block, device=device)
    index = (block_index[:, None] * block + offsets[None, :]).flatten()
    return index[index < tokens]


def benchmark_case(
    *,
    name: str,
    batch: int,
    tokens: int,
    queries: int,
    heads: int,
    head_dim: int,
    support_ratio: float,
    key_block: int,
    dtype: torch.dtype,
    device: torch.device,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    generator = torch.Generator(device=device).manual_seed(23)
    probability = torch.rand(
        batch,
        heads,
        queries,
        tokens,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    probability /= probability.sum(dim=-1, keepdim=True)
    delta_probability = 0.001 * torch.randn(
        probability.shape,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    value = torch.randn(
        batch,
        heads,
        tokens,
        head_dim,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    delta_value = 0.01 * torch.randn(
        value.shape,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    query = torch.randn(
        batch,
        heads,
        queries,
        head_dim,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    delta_query = 0.01 * torch.randn(
        query.shape,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    key = torch.randn(
        batch,
        heads,
        tokens,
        head_dim,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    delta_key = 0.01 * torch.randn(
        key.shape,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    key_index = make_key_index(tokens, support_ratio, key_block, device)
    cached_full_base_output = probability @ value

    def pack() -> tuple[torch.Tensor, ...]:
        return (
            probability.index_select(-1, key_index),
            delta_probability.index_select(-1, key_index),
            value.index_select(-2, key_index),
            delta_value.index_select(-2, key_index),
        )

    packed = pack()
    packed_probability, packed_delta_probability, packed_value, packed_delta_value = packed
    cached_base_support_output = packed_probability @ packed_value
    key_index_batch = key_index.unsqueeze(0).expand(batch, -1).contiguous()
    scale = head_dim**-0.5

    def dense_full_pipeline() -> torch.Tensor:
        delta_logits = (
            delta_query @ key.transpose(-2, -1)
            + query @ delta_key.transpose(-2, -1)
        ) * scale
        delta_probability_full = probability * (
            delta_logits
            - (probability * delta_logits).sum(dim=-1, keepdim=True)
        )
        return (
            (probability + delta_probability_full) @ (value + delta_value)
            - cached_full_base_output
        )

    def sparse_indirect_pipeline() -> torch.Tensor:
        selected_delta_probability = selected_softmax_jvp_triton(
            packed_probability,
            query,
            delta_query,
            key,
            delta_key,
            key_index_batch,
            scale=scale,
        )
        return packed_attention_delta_triton(
            packed_probability,
            selected_delta_probability,
            value,
            delta_value,
            backend="product_delta",
            base_support_output=cached_base_support_output,
            key_index=key_index_batch,
        )

    operations: dict[str, Callable[[], torch.Tensor | tuple[torch.Tensor, ...]]] = {
        "selector_pack": pack,
        "dense_full_split": lambda: (
            probability @ delta_value
            + delta_probability @ value
        ),
        "dense_full_product_cached": lambda: (
            (probability + delta_probability)
            @ (value + delta_value)
            - cached_full_base_output
        ),
        "dense_full_pipeline": dense_full_pipeline,
        "dense_split": lambda: (
            packed_probability @ packed_delta_value
            + packed_delta_probability @ packed_value
        ),
        "dense_product": lambda: (
            (packed_probability + packed_delta_probability)
            @ (packed_value + packed_delta_value)
            - packed_probability @ packed_value
        ),
        "dense_product_cached": lambda: (
            (packed_probability + packed_delta_probability)
            @ (packed_value + packed_delta_value)
            - cached_base_support_output
        ),
        "triton_split": lambda: packed_attention_delta_triton(
            *packed, backend="split_jvp"
        ),
        "triton_product": lambda: packed_attention_delta_triton(
            *packed, backend="product_delta"
        ),
        "triton_product_cached": lambda: packed_attention_delta_triton(
            *packed,
            backend="product_delta",
            base_support_output=cached_base_support_output,
        ),
        "triton_split_indirect": lambda: packed_attention_delta_triton(
            packed_probability,
            packed_delta_probability,
            value,
            delta_value,
            backend="split_jvp",
            key_index=key_index_batch,
        ),
        "triton_product_cached_indirect": lambda: packed_attention_delta_triton(
            packed_probability,
            packed_delta_probability,
            value,
            delta_value,
            backend="product_delta",
            base_support_output=cached_base_support_output,
            key_index=key_index_batch,
        ),
        "triton_product_pipeline_indirect": sparse_indirect_pipeline,
    }
    timings = {
        operation_name: summarize(
            cuda_times_ms(operation, warmup=warmup, iterations=iterations)
        )
        for operation_name, operation in operations.items()
    }
    pack_ms = timings["selector_pack"]["p50_ms"]
    for operation_name in (
        "dense_full_split",
        "dense_full_product_cached",
        "dense_full_pipeline",
        "dense_split",
        "dense_product",
        "dense_product_cached",
        "triton_split",
        "triton_product",
        "triton_product_cached",
        "triton_split_indirect",
        "triton_product_cached_indirect",
        "triton_product_pipeline_indirect",
    ):
        timings[operation_name]["with_pack_p50_ms"] = (
            timings[operation_name]["p50_ms"] + pack_ms
        )
        timings[operation_name]["packing_fraction"] = pack_ms / max(
            timings[operation_name]["with_pack_p50_ms"], 1e-12
        )
    for operation_name in (
        "dense_full_split",
        "dense_full_product_cached",
        "dense_full_pipeline",
        "triton_split_indirect",
        "triton_product_cached_indirect",
        "triton_product_pipeline_indirect",
    ):
        # The key descriptor and base support are draft-cache products.  The
        # correction kernel reads full V/dV indirectly and performs no gather.
        timings[operation_name]["with_pack_p50_ms"] = timings[operation_name][
            "p50_ms"
        ]
        timings[operation_name]["packing_fraction"] = 0.0
        timings[operation_name]["explicit_correction_pack"] = False

    del probability, delta_probability, value, delta_value, packed
    del query, delta_query, key, delta_key
    torch.cuda.empty_cache()
    return {
        "preset": name,
        "batch": batch,
        "heads": heads,
        "queries": queries,
        "tokens": tokens,
        "head_dim": head_dim,
        "support_requested": support_ratio,
        "support_actual": key_index.numel() / tokens,
        "selected_keys": key_index.numel(),
        "key_block": key_block,
        "timings": timings,
    }


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    dtype = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[args.dtype]
    device = torch.device(args.device)
    preset_names = [value.strip() for value in args.presets.split(",") if value.strip()]
    if args.batch is not None or args.tokens is not None:
        if args.batch is None or args.tokens is None:
            raise ValueError("--batch and --tokens must be specified together")
        cases = [("custom", {"batch": args.batch, "tokens": args.tokens})]
    else:
        unknown = sorted(set(preset_names) - set(PRESETS))
        if unknown:
            raise ValueError(f"Unknown presets: {unknown}")
        cases = [(name, PRESETS[name]) for name in preset_names]
    ratios = [float(value) for value in args.support.split(",")]

    rows = []
    for name, case in cases:
        for ratio in ratios:
            print(
                f"[benchmark] {name} B={case['batch']} N={case['tokens']} "
                f"Q={args.queries} support={ratio:.3f}",
                flush=True,
            )
            rows.append(
                benchmark_case(
                    name=name,
                    batch=case["batch"],
                    tokens=case["tokens"],
                    queries=min(args.queries, case["tokens"]),
                    heads=args.heads,
                    head_dim=args.head_dim,
                    support_ratio=ratio,
                    key_block=args.key_block,
                    dtype=dtype,
                    device=device,
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
            )

    payload = {
        "schema_version": 1,
        "hardware": {
            "device": str(device),
            "name": torch.cuda.get_device_name(device),
            "capability": torch.cuda.get_device_capability(device),
        },
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "dtype": args.dtype,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "scope_note": (
            "triton_product_pipeline_indirect includes the selected-logit "
            "softmax-JVP producer and product consumer. Other indirect rows are "
            "kernel-only and assume selected delta_probability is available."
        ),
        "cuda_environment_added": CUDA_ENV,
        "results": rows,
    }
    rendered = json.dumps(payload, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
        print(f"[saved] {args.output}")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
