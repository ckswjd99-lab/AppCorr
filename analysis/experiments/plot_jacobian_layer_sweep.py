#!/usr/bin/env python3
"""Plot final-feature sensitivity from exact layer/component sweep outputs."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COMPONENTS = (
    ("input_token", "Input-token state"),
    ("attention_edge", "Attention edges"),
    ("ffn_channel", "FFN channels"),
)
DISPLAY_RATIOS = (0.2, 0.5, 0.8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--band",
        choices=("std", "sem", "none"),
        default="std",
        help="Translucent uncertainty band around the sample mean.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=None,
        help="Optional compact JSON with per-layer and per-stage statistics.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values: dict[tuple[int, str, float], list[float]] = defaultdict(list)
    sample_records: list[dict[tuple[int, str, float], float]] = []
    sample_count = 0
    labels: set[int] = set()
    full_endpoint_count = 0
    full_endpoint_max = 0.0
    source_manifests = []
    for input_path in args.inputs:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        sample_count += len(payload["samples"])
        source_manifests.append({
            "input": str(input_path),
            "experiment": payload["experiment"],
        })
        for sample in payload["samples"]:
            if sample["label"] is not None:
                labels.add(int(sample["label"]))
            sample_record: dict[tuple[int, str, float], float] = {}
            sweeps = sample["exact_layer_component_sweeps"]
            for layer_text, components in sweeps.items():
                layer = int(layer_text)
                for component, rows in components.items():
                    for row in rows:
                        ratio = float(row["requested_ratio"])
                        relative_l2 = row["normalized_token_feature"][
                            "relative_l2_error"
                        ]
                        if ratio == 1:
                            full_endpoint_count += 1
                            full_endpoint_max = max(
                                full_endpoint_max, relative_l2
                            )
                        if ratio in DISPLAY_RATIOS:
                            key = (layer, component, ratio)
                            values[key].append(relative_l2)
                            sample_record[key] = relative_l2
            sample_records.append(sample_record)

    layers = sorted({key[0] for key in values})
    if not layers:
        raise RuntimeError("No exact layer/component sweep rows found")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), constrained_layout=True)
    colors = {0.2: "#d95f02", 0.5: "#7570b3", 0.8: "#1b9e77"}
    for axis, (component, title) in zip(axes, COMPONENTS):
        for ratio in DISPLAY_RATIOS:
            sample_values = [
                np.asarray(values[(layer, component, ratio)], dtype=np.float64)
                for layer in layers
            ]
            means = np.asarray([values.mean() for values in sample_values])
            deviations = np.asarray([
                values.std(ddof=1) if len(values) > 1 else 0.0
                for values in sample_values
            ])
            if args.band == "sem":
                deviations = deviations / np.sqrt(
                    np.asarray([len(values) for values in sample_values])
                )
            axis.plot(
                layers,
                means,
                color=colors[ratio],
                linewidth=1.8,
                marker="o",
                markersize=2.7,
                label=f"{ratio:.0%} keep",
            )
            if args.band != "none":
                axis.fill_between(
                    layers,
                    np.maximum(means - deviations, 0),
                    means + deviations,
                    color=colors[ratio],
                    alpha=0.12,
                    linewidth=0,
                )
        axis.axvspan(-0.5, 12.5, color="#4c78a8", alpha=0.06)
        axis.axvspan(12.5, 26.5, color="#f2cf5b", alpha=0.08)
        axis.axvspan(26.5, 39.5, color="#e45756", alpha=0.05)
        axis.axvline(12.5, color="0.75", linewidth=0.8)
        axis.axvline(26.5, color="0.75", linewidth=0.8)
        axis.set_title(title)
        axis.set_xlabel("Target layer")
        axis.grid(axis="y", alpha=0.22)
    axes[0].set_ylabel("Final normalized-token relative L2")
    axes[-1].legend(frameon=False, fontsize=8)
    band_text = (
        f"mean with ±1 {args.band.upper()} band"
        if args.band != "none"
        else "sample mean"
    )
    fig.suptitle(
        "Exact finite-difference support: one component in one layer\n"
        f"{band_text}, n={sample_count}",
        fontsize=12,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(f"[saved] {args.output}")

    if args.summary_output is not None:
        stage_ranges = {
            "early_layers_0_12": (0, 12),
            "middle_layers_13_26": (13, 26),
            "late_layers_27_39": (27, 39),
        }
        layer_summary = {}
        stage_summary = {}
        stage_contrasts = {}
        for component, _title in COMPONENTS:
            component_layers = {}
            component_stages = {}
            component_contrasts = {}
            for ratio in DISPLAY_RATIOS:
                ratio_key = f"{ratio:g}"
                rows = []
                for layer in layers:
                    layer_values = np.asarray(
                        values[(layer, component, ratio)],
                        dtype=np.float64,
                    )
                    std = float(layer_values.std(ddof=1))
                    sem = std / np.sqrt(len(layer_values))
                    rows.append({
                        "layer": layer,
                        "mean": float(layer_values.mean()),
                        "std": std,
                        "ci95_half_width": float(1.96 * sem),
                    })
                component_layers[ratio_key] = rows

                ratio_stages = {}
                stage_samples = {}
                for stage_name, (start, end) in stage_ranges.items():
                    per_sample = np.asarray([
                        np.mean([
                            record[(layer, component, ratio)]
                            for layer in layers
                            if start <= layer <= end
                        ])
                        for record in sample_records
                    ])
                    stage_samples[stage_name] = per_sample
                    std = float(per_sample.std(ddof=1))
                    sem = std / np.sqrt(len(per_sample))
                    ratio_stages[stage_name] = {
                        "mean": float(per_sample.mean()),
                        "std": std,
                        "ci95_half_width": float(1.96 * sem),
                    }
                component_stages[ratio_key] = ratio_stages
                ratio_contrasts = {}
                for contrast_name, first, second in (
                    (
                        "early_minus_middle",
                        "early_layers_0_12",
                        "middle_layers_13_26",
                    ),
                    (
                        "early_minus_late",
                        "early_layers_0_12",
                        "late_layers_27_39",
                    ),
                    (
                        "late_minus_middle",
                        "late_layers_27_39",
                        "middle_layers_13_26",
                    ),
                ):
                    difference = stage_samples[first] - stage_samples[second]
                    std = float(difference.std(ddof=1))
                    sem = std / np.sqrt(len(difference))
                    ratio_contrasts[contrast_name] = {
                        "mean_difference": float(difference.mean()),
                        "std": std,
                        "ci95_half_width": float(1.96 * sem),
                    }
                component_contrasts[ratio_key] = ratio_contrasts
            layer_summary[component] = component_layers
            stage_summary[component] = component_stages
            stage_contrasts[component] = component_contrasts

        summary = {
            "schema_version": 1,
            "sample_count": sample_count,
            "unique_label_count": len(labels),
            "display_ratios": list(DISPLAY_RATIOS),
            "uncertainty_definition": "sample standard deviation; CI uses 1.96*SEM",
            "full_support_endpoint": {
                "count": full_endpoint_count,
                "max_relative_l2": full_endpoint_max,
            },
            "source_manifests": source_manifests,
            "per_layer": layer_summary,
            "per_stage": stage_summary,
            "paired_stage_contrasts": stage_contrasts,
        }
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"[saved] {args.summary_output}")


if __name__ == "__main__":
    main()
