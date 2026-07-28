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
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    return parser.parse_args()


def load_results(input_dir: Path) -> list[dict[str, Any]]:
    results = []
    for path in sorted(input_dir.glob("*.json")):
        if path.name == "calibration_index.json":
            continue
        with path.open("r", encoding="utf-8") as handle:
            result = json.load(handle)
        result["_source_path"] = str(path)
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
            "source_path": result["_source_path"],
        }
        rows.append(row)
    return sorted(rows, key=lambda row: row["label"])


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(rows: list[dict[str, Any]], path: Path) -> None:
    lines = [
        "# DINOv3 tail-full and L2-L1-L0 calibration",
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
    path.write_text("\n".join(lines), encoding="utf-8")


def write_plots(rows: list[dict[str, Any]], output_dir: Path) -> None:
    os.environ.setdefault(
        "MPLCONFIGDIR",
        str((output_dir / ".mplconfig").resolve()),
    )
    import matplotlib.pyplot as plt

    progressive_rows = [
        row for row in rows if row["mode"] in {"l2l0", "l2l1l0"}
    ]
    fig, axis = plt.subplots(figsize=(7.2, 4.6))
    colors = {"l2l0": "#4C78A8", "l2l1l0": "#E45756"}
    markers = {0: "o", 2: "s", 3: "^"}
    for row in progressive_rows:
        low_error = row["top1_percent"] - row["top1_ci95_low"]
        high_error = row["top1_ci95_high"] - row["top1_percent"]
        axis.errorbar(
            row["dominant_flops_ratio"],
            row["top1_percent"],
            yerr=[[low_error], [high_error]],
            color=colors[row["mode"]],
            marker=markers[row["final_full_layers"]],
            alpha=0.8,
            capsize=2,
            linestyle="none",
        )
    axis.set_xlabel("Dominant ViT FLOPs / full-resolution backbone")
    axis.set_ylabel("ImageNet top-1 (%)")
    axis.grid(alpha=0.2)
    axis.set_title("Accuracy–compute calibration")
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
            y = np.asarray([row["top1_percent"] for row in selected])
            low = np.asarray([row["top1_ci95_low"] for row in selected])
            high = np.asarray([row["top1_ci95_high"] for row in selected])
            axis.plot(x, y, marker=markers[tail], label=f"tail N={tail}")
            axis.fill_between(x, low, high, alpha=0.12)
        axis.set_title(mode.upper())
        axis.set_xlabel("Correction patch keep ratio")
        axis.grid(alpha=0.2)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, labels)
    axes[0].set_ylabel("ImageNet top-1 (%)")
    fig.tight_layout()
    fig.savefig(output_dir / "keep_sweep.png", dpi=180)
    fig.savefig(output_dir / "keep_sweep.pdf")
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
    write_csv(rows, output_dir / "calibration_summary.csv")
    (output_dir / "calibration_summary.json").write_text(
        json.dumps(rows, indent=2),
        encoding="utf-8",
    )
    write_report(rows, output_dir / "REPORT.md")
    write_plots(rows, output_dir)
    print(f"[summary] wrote {output_dir}")


if __name__ == "__main__":
    main()
