"""Summarize and plot a DINOv3 tail-full/L2-L1-L0 calibration sweep."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np


LABEL_PATTERN = re.compile(
    r"^(?P<mode>l2l0|l2l1l0)_n(?P<tail>\d+)_k(?P<keep>[0-9.]+)$"
)
BOOTSTRAP_SEED = 20260728


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="logs/dinov3_l2l1l0/calibration",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/results/dinov3_l2l1l0",
    )
    parser.add_argument(
        "--isolated-dir",
        default=None,
        help="Optional directory containing isolated full/L2L0/L2L1L0 runs.",
    )
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    return parser.parse_args()


def load_results(input_dir: Path) -> list[dict[str, Any]]:
    results = []
    for path in sorted(input_dir.glob("*.json")):
        if path.name == "calibration_index.json":
            continue
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        results.append(result)
    if not results:
        raise FileNotFoundError(f"No result JSON files found under {input_dir}")
    return results


def paired_accuracy_delta(
    flags: list[bool],
    baseline_flags: list[bool],
    *,
    num_resamples: int,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int]:
    values = np.asarray(flags, dtype=np.float64)
    baseline = np.asarray(baseline_flags, dtype=np.float64)
    if values.shape != baseline.shape:
        raise ValueError(
            f"Paired flags have different shapes: {values.shape} vs {baseline.shape}"
        )
    if values.size == 0:
        raise ValueError("Paired accuracy requires at least one sample")
    if num_resamples <= 0:
        raise ValueError("num_resamples must be positive")

    per_sample_delta = (values - baseline) * 100.0
    rng = np.random.default_rng(seed)
    bootstrap_means = np.empty(num_resamples, dtype=np.float64)
    chunk_size = max(1, min(256, num_resamples))
    for start in range(0, num_resamples, chunk_size):
        end = min(start + chunk_size, num_resamples)
        indices = rng.integers(
            0,
            values.size,
            size=(end - start, values.size),
        )
        bootstrap_means[start:end] = per_sample_delta[indices].mean(axis=1)
    low, high = np.percentile(bootstrap_means, (2.5, 97.5))
    return {
        "num_samples": int(values.size),
        "delta_top1_points": float(per_sample_delta.mean()),
        "bootstrap_std_points": float(bootstrap_means.std(ddof=1)),
        "ci95_low_points": float(low),
        "ci95_high_points": float(high),
        "num_resamples": int(num_resamples),
    }


def sum_request_event_ms(result: dict[str, Any], event_names: set[str]) -> float:
    request_values = [
        sum(
            float(value)
            for name, value in request.get("event_ms", {}).items()
            if name in event_names
        )
        for request in result.get("requests", [])
    ]
    return float(np.median(request_values)) if request_values else 0.0


def paired_latency_summary(
    result: dict[str, Any],
    baseline: dict[str, Any],
    *,
    num_resamples: int,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, float | int | str]:
    result_values = np.asarray(
        [
            float(request["wall_ms"])
            for request in result.get("requests", [])
        ],
        dtype=np.float64,
    )
    baseline_values = np.asarray(
        [
            float(request["wall_ms"])
            for request in baseline.get("requests", [])
        ],
        dtype=np.float64,
    )
    if result_values.shape != baseline_values.shape:
        raise ValueError(
            "Isolated latency runs must have matching request counts, got "
            f"{result_values.shape} and {baseline_values.shape}"
        )
    if result_values.size == 0:
        raise ValueError("Isolated latency runs contain no requests")

    rng = np.random.default_rng(seed)
    median_delta = np.empty(num_resamples, dtype=np.float64)
    speedup = np.empty(num_resamples, dtype=np.float64)
    for sample_idx in range(num_resamples):
        indices = rng.integers(
            0,
            result_values.size,
            size=result_values.size,
        )
        result_median = np.median(result_values[indices])
        baseline_median = np.median(baseline_values[indices])
        median_delta[sample_idx] = result_median - baseline_median
        speedup[sample_idx] = baseline_median / result_median
    delta_low, delta_high = np.percentile(
        median_delta,
        (2.5, 97.5),
    )
    speedup_low, speedup_high = np.percentile(
        speedup,
        (2.5, 97.5),
    )

    event_names = {
        "FULL_INFERENCE",
        "APPROX_FORWARD",
        "CORRECT_FORWARD",
        "FINAL_FULL_FORWARD",
    }
    return {
        "label": result["label"],
        "num_requests": int(result_values.size),
        "top1_percent": float(result["dataset_summary"]["top1_acc"]),
        "dominant_flops_ratio": float(
            result["dominant_flops_ratio"]["p50"]
        ),
        "request_p50_ms": float(np.median(result_values)),
        "request_p95_ms": float(np.percentile(result_values, 95)),
        "paired_median_delta_ms": float(
            np.median(result_values) - np.median(baseline_values)
        ),
        "paired_median_delta_ci95_low_ms": float(delta_low),
        "paired_median_delta_ci95_high_ms": float(delta_high),
        "speedup_vs_full": float(
            np.median(baseline_values) / np.median(result_values)
        ),
        "speedup_ci95_low": float(speedup_low),
        "speedup_ci95_high": float(speedup_high),
        "mobile_encode_p50_ms": float(
            result["mobile_encode_ms"]["p50"]
        ),
        "decode_p50_ms": sum_request_event_ms(result, {"Decode"}),
        "backbone_p50_ms": sum_request_event_ms(result, event_names),
        "model_compute_p50_ms": float(
            result["model_compute_ms"]["p50"]
        ),
        "wire_kib_per_image": float(
            result["avg_wire_bytes_per_sample"] / 1024.0
        ),
        "num_resamples": int(num_resamples),
    }


def build_isolated_rows(
    isolated_dir: Path,
    *,
    bootstrap_resamples: int,
) -> list[dict[str, Any]]:
    results = load_results(isolated_dir)
    by_label = {result["label"]: result for result in results}
    baseline_label = "full_sequential_isolated"
    if baseline_label not in by_label:
        raise ValueError(
            f"Isolated results must contain {baseline_label}"
        )
    baseline = by_label[baseline_label]
    baseline_indices = baseline["sample_indices"]
    rows = []
    for result in results:
        if result["sample_indices"] != baseline_indices:
            raise ValueError(
                f"{result['label']} does not use isolated baseline indices"
            )
        rows.append(
            paired_latency_summary(
                result,
                baseline,
                num_resamples=bootstrap_resamples,
            )
        )
    preferred_order = {
        "full_sequential_isolated": 0,
        "l2l0_n3_k0.25_isolated": 1,
        "l2l1l0_n3_k0.25_isolated": 2,
    }
    return sorted(
        rows,
        key=lambda row: (
            preferred_order.get(row["label"], 99),
            row["label"],
        ),
    )


def build_rows(
    results: list[dict[str, Any]],
    *,
    bootstrap_resamples: int,
) -> list[dict[str, Any]]:
    by_label = {result["label"]: result for result in results}
    if "full_sequential" not in by_label:
        raise ValueError("The sweep must contain the full_sequential baseline")
    baseline = by_label["full_sequential"]
    baseline_indices = baseline["sample_indices"]
    baseline_flags = baseline["top1_flags"]

    rows = []
    for result in results:
        if result["sample_indices"] != baseline_indices:
            raise ValueError(
                f"{result['label']} does not use the baseline sample indices"
            )
        match = LABEL_PATTERN.match(result["label"])
        paired = paired_accuracy_delta(
            result["top1_flags"],
            baseline_flags,
            num_resamples=bootstrap_resamples,
            seed=BOOTSTRAP_SEED,
        )
        top1_bootstrap = result["top1_bootstrap"]
        first_request = result.get("requests", [{}])[0]
        first_flops = first_request.get("flops", {})
        correction_flops = float(first_flops.get("correction_flops", 0.0))
        approx_flops = float(first_flops.get("approx_flops", 0.0))
        final_full_flops = float(first_flops.get("final_full_flops", 0.0))
        row = {
            "label": result["label"],
            "mode": match.group("mode") if match else result["label"],
            "final_full_layers": int(match.group("tail")) if match else None,
            "token_keep_ratio": float(match.group("keep")) if match else None,
            "num_samples": int(result["dataset_summary"]["total_samples"]),
            "top1_percent": float(result["dataset_summary"]["top1_acc"]),
            "top1_ci95_low": float(
                top1_bootstrap["ci95_low_percent"]
            ),
            "top1_ci95_high": float(
                top1_bootstrap["ci95_high_percent"]
            ),
            **paired,
            "dominant_flops_ratio": float(
                result["dominant_flops_ratio"]["p50"]
            ),
            "correction_to_approx_flops": (
                correction_flops / approx_flops
                if approx_flops > 0
                else 0.0
            ),
            "final_full_to_approx_flops": (
                final_full_flops / approx_flops
                if approx_flops > 0
                else 0.0
            ),
            "backbone_compute_p50_ms": float(
                result["backbone_compute_ms"]["p50"]
            ),
            "model_compute_p50_ms": float(
                result["model_compute_ms"]["p50"]
            ),
            "request_p50_ms": float(
                result["request_latency_ms"]["p50"]
            ),
            "correction_p50_ms": sum_request_event_ms(
                result,
                {"CORRECT_FORWARD"},
            ),
            "final_full_p50_ms": sum_request_event_ms(
                result,
                {"FINAL_FULL_FORWARD"},
            ),
            "avg_wire_bytes_per_sample": float(
                result["avg_wire_bytes_per_sample"]
            ),
            "avg_actual_patch_keep_ratio": float(
                result["avg_partial_token_keep_ratio"]
            ),
        }
        rows.append(row)
    return sorted(rows, key=lambda row: row["label"])


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    rows: list[dict[str, Any]],
    path: Path,
    isolated_rows: list[dict[str, Any]] | None = None,
) -> None:
    by_label = {row["label"]: row for row in rows}
    full = by_label["full_sequential"]
    low_res = by_label["l2_approx_only"]
    l2l0_n0_k25 = by_label["l2l0_n0_k0.25"]
    l2l0_n3_k25 = by_label["l2l0_n3_k0.25"]
    l2l1l0_n3_k25 = by_label["l2l1l0_n3_k0.25"]
    tail_overhead_reduction = (
        (
            (l2l0_n0_k25["dominant_flops_ratio"] - 1.0)
            - (l2l0_n3_k25["dominant_flops_ratio"] - 1.0)
        )
        / (l2l0_n0_k25["dominant_flops_ratio"] - 1.0)
        * 100.0
    )
    lines = [
        "# DINOv3 tail-full and L2-L1-L0 calibration",
        "",
        "## Implemented semantics",
        "",
        "- `final_full_layers=N` partitions only layers `[0, 40-N)` "
        "across progressive correction groups. The final group first "
        "corrects that prefix, then runs `[40-N, 40)` once through the "
        "stock block forward without creating correction caches.",
        "- `L2L1L0ProgressiveLaplacian` emits L2 base, one complete L1 "
        "residual group, then four L0 residual groups. A coarse L1 patch "
        "maps to its 2×2 fine ViT token cells. Selection uses L1/L0 "
        "residual energy multiplied by layer-mean CLS attention.",
        "- CLS and four register tokens remain mandatory correction "
        "queries. Existing `ProgressiveLaplacian` and N=0 scheduling "
        "remain available unchanged.",
        "",
        "## Evaluation protocol",
        "",
        "DINOv3 ViT-7B/16 (BF16, 40 layers, hidden size 4096) was run "
        "on one or two NVIDIA B200 GPUs. Calibration uses the same "
        "deterministic 1,024 ImageNet validation samples for every "
        "configuration, batch size 32, and 10,000 paired bootstrap "
        "resamples. Dominant FLOPs include ViT projection, attention, "
        "and SwiGLU matmuls; they exclude codec and host work.",
        "",
        "## Main findings",
        "",
        f"- Full resolution reaches {full['top1_percent']:.2f}% top-1; "
        f"L2-only reaches {low_res['top1_percent']:.2f}% "
        f"({low_res['delta_top1_points']:+.2f} points).",
        f"- On the existing L2-L0 path, N=3/K=25% reaches "
        f"{l2l0_n3_k25['top1_percent']:.2f}% "
        f"({l2l0_n3_k25['delta_top1_points']:+.2f} points) at "
        f"{l2l0_n3_k25['dominant_flops_ratio']:.3f}× dominant FLOPs. "
        f"Compared with N=0/K=25%, tail deferral reduces correction "
        f"overhead by {tail_overhead_reduction:.1f}% while preserving "
        "the measured top-1.",
        f"- The matching L2-L1-L0 N=3/K=25% point reaches "
        f"{l2l1l0_n3_k25['top1_percent']:.2f}% at "
        f"{l2l1l0_n3_k25['dominant_flops_ratio']:.3f}× FLOPs and "
        f"{l2l1l0_n3_k25['avg_wire_bytes_per_sample'] / 1024.0:.1f} "
        f"KiB/image. It is dominated by L2-L0 "
        f"({l2l0_n3_k25['avg_wire_bytes_per_sample'] / 1024.0:.1f} "
        "KiB/image).",
        "- Every progressive point exceeds 1.0× dominant backbone "
        "FLOPs because all layers still run one approximate/full pass "
        "and correction is additional work. Any end-to-end gain must "
        "therefore come from pipeline overlap or codec behavior, not "
        "compute reduction.",
        "",
        "## Full calibration table",
        "",
        "All rows use identical deterministic ImageNet sample indices. "
        "Accuracy deltas and confidence intervals are paired against the "
        "`full_sequential` predictions. FLOPs count dominant ViT block "
        "matmuls; measured timings include selector and correction runtime.",
        "",
        "| setting | top-1 | Δ full (95% CI) | FLOPs/full | "
        "backbone p50 | correction p50 | wire/image |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        rows,
        key=lambda item: (
            -item["top1_percent"],
            item["dominant_flops_ratio"],
        ),
    ):
        lines.append(
            f"| {row['label']} | {row['top1_percent']:.2f}% | "
            f"{row['delta_top1_points']:+.2f} "
            f"[{row['ci95_low_points']:+.2f}, "
            f"{row['ci95_high_points']:+.2f}] | "
            f"{row['dominant_flops_ratio']:.3f}× | "
            f"{row['backbone_compute_p50_ms']:.1f} ms | "
            f"{row['correction_p50_ms']:.1f} ms | "
            f"{row['avg_wire_bytes_per_sample'] / 1024.0:.1f} KiB |"
        )
    lines.extend(
        [
            "",
            "The calibration set ranks configurations; it is not the final "
            "ImageNet validation claim. Promote Pareto candidates to the "
            "full 50,000-image validation set before reporting final accuracy.",
            "",
        ]
    )
    if isolated_rows:
        lines.extend(
            [
                "## Isolated B200 latency",
                "",
                "These runs use one B200 with no simultaneous model loading, "
                "three warm-up batches, and 32 measured batch-32 requests. "
                "Component times overlap and therefore must not be summed.",
                "",
                "| setting | top-1 | FLOPs/full | request p50 / p95 | "
                "speedup vs full (95% CI) | encode | decode | backbone |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in isolated_rows:
            lines.append(
                f"| {row['label']} | {row['top1_percent']:.2f}% | "
                f"{row['dominant_flops_ratio']:.3f}× | "
                f"{row['request_p50_ms']:.1f} / "
                f"{row['request_p95_ms']:.1f} ms | "
                f"{row['speedup_vs_full']:.3f}× "
                f"[{row['speedup_ci95_low']:.3f}, "
                f"{row['speedup_ci95_high']:.3f}] | "
                f"{row['mobile_encode_p50_ms']:.1f} ms | "
                f"{row['decode_p50_ms']:.1f} ms | "
                f"{row['backbone_p50_ms']:.1f} ms |"
            )
        lines.extend(
            [
                "",
                "The progressive request speedup is a pipeline/codec effect, "
                "not compute reduction: its backbone time and dominant FLOPs "
                "are both higher than full inference.",
                "",
                "## Limitations",
                "",
                "- The 1,024-image calibration confidence intervals are "
                "wide; these are ranking results, not final ImageNet claims.",
                "- Calibration p95 latency is affected by simultaneous "
                "checkpoint mmap on the other GPU. Only the isolated table "
                "is used for latency conclusions.",
                "- The isolated run has no imposed uplink delay. Finite-"
                "bandwidth crossover must be measured separately because "
                "progressive streams transmit more bytes.",
                "- Progressive decode remains CPU-heavy. The L1 stage adds "
                "both decode work and wire bytes, explaining much of its "
                "end-to-end regression.",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def write_plots(
    rows: list[dict[str, Any]],
    output_dir: Path,
    isolated_rows: list[dict[str, Any]] | None = None,
) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str((output_dir / ".mplconfig").resolve()),
    )
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    progressive_rows = [
        row for row in rows if row["mode"] in {"l2l0", "l2l1l0"}
    ]
    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    colors = {"l2l0": "#4C78A8", "l2l1l0": "#E45756"}
    markers = {0: "o", 2: "s", 3: "^"}
    for row in progressive_rows:
        low_error = (
            row["delta_top1_points"] - row["ci95_low_points"]
        )
        high_error = (
            row["ci95_high_points"] - row["delta_top1_points"]
        )
        axis.errorbar(
            row["dominant_flops_ratio"],
            row["delta_top1_points"],
            yerr=[[low_error], [high_error]],
            color=colors[row["mode"]],
            marker=markers[row["final_full_layers"]],
            alpha=0.8,
            capsize=2,
            linestyle="none",
        )
    axis.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
    axis.set_xlabel("Dominant ViT FLOPs / full-resolution backbone")
    axis.set_ylabel("Paired top-1 delta vs full (points)")
    axis.grid(alpha=0.2)
    axis.set_title("Paired accuracy–compute calibration")
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=colors[mode],
            marker="o",
            linestyle="none",
            label=mode.upper(),
        )
        for mode in ("l2l0", "l2l1l0")
    ] + [
        Line2D(
            [0],
            [0],
            color="black",
            marker=markers[tail],
            linestyle="none",
            label=f"tail N={tail}",
        )
        for tail in (0, 2, 3)
    ]
    axis.legend(handles=legend_handles, ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "accuracy_vs_flops.png", dpi=180)
    fig.savefig(output_dir / "accuracy_vs_flops.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for axis, mode in zip(axes, ("l2l0", "l2l1l0")):
        for tail in (0, 2, 3):
            selected = sorted(
                [
                    row
                    for row in progressive_rows
                    if row["mode"] == mode
                    and row["final_full_layers"] == tail
                ],
                key=lambda row: row["token_keep_ratio"],
            )
            if not selected:
                continue
            x = np.asarray(
                [row["token_keep_ratio"] for row in selected]
            )
            y = np.asarray(
                [row["delta_top1_points"] for row in selected]
            )
            low = np.asarray(
                [row["ci95_low_points"] for row in selected]
            )
            high = np.asarray(
                [row["ci95_high_points"] for row in selected]
            )
            axis.plot(x, y, marker=markers[tail], label=f"tail N={tail}")
            axis.fill_between(x, low, high, alpha=0.12)
        axis.set_title(mode.upper())
        axis.set_xlabel("Correction patch keep ratio")
        axis.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
        axis.grid(alpha=0.2)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, labels)
    axes[0].set_ylabel("Paired top-1 delta vs full (points)")
    fig.tight_layout()
    fig.savefig(output_dir / "keep_sweep.png", dpi=180)
    fig.savefig(output_dir / "keep_sweep.pdf")
    plt.close(fig)

    if isolated_rows:
        fig, axis = plt.subplots(figsize=(8.2, 4.6))
        labels = [
            row["label"]
            .replace("_isolated", "")
            .replace("full_sequential", "full")
            for row in isolated_rows
        ]
        x = np.arange(len(labels))
        width = 0.18
        components = [
            ("mobile_encode_p50_ms", "mobile encode"),
            ("decode_p50_ms", "server decode"),
            ("backbone_p50_ms", "backbone"),
            ("request_p50_ms", "request p50"),
        ]
        for component_idx, (key, label) in enumerate(components):
            axis.bar(
                x + (component_idx - 1.5) * width,
                [row[key] for row in isolated_rows],
                width,
                label=label,
            )
        axis.set_xticks(x, labels, rotation=10)
        axis.set_ylabel("Latency (ms)")
        axis.set_title("Isolated B200 pipeline components (overlapping)")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output_dir / "isolated_latency.png", dpi=180)
        fig.savefig(output_dir / "isolated_latency.pdf")
        plt.close(fig)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(
        load_results(input_dir),
        bootstrap_resamples=args.bootstrap_resamples,
    )
    isolated_rows = None
    if args.isolated_dir:
        isolated_rows = build_isolated_rows(
            Path(args.isolated_dir).resolve(),
            bootstrap_resamples=args.bootstrap_resamples,
        )
    write_csv(rows, output_dir / "calibration_summary.csv")
    (output_dir / "calibration_summary.json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )
    if isolated_rows:
        write_csv(
            isolated_rows,
            output_dir / "isolated_latency_summary.csv",
        )
        (output_dir / "isolated_latency_summary.json").write_text(
            json.dumps(isolated_rows, indent=2),
            encoding="utf-8",
        )
    write_report(
        rows,
        output_dir / "REPORT.md",
        isolated_rows=isolated_rows,
    )
    write_plots(
        rows,
        output_dir,
        isolated_rows=isolated_rows,
    )
    print(f"[summary] wrote {output_dir}")


if __name__ == "__main__":
    main()
