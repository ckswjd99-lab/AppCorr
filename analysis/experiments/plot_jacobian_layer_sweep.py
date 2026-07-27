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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    values: dict[tuple[int, str, float], list[float]] = defaultdict(list)
    for input_path in args.inputs:
        payload = json.loads(input_path.read_text(encoding="utf-8"))
        for sample in payload["samples"]:
            sweeps = sample["exact_layer_component_sweeps"]
            for layer_text, components in sweeps.items():
                layer = int(layer_text)
                for component, rows in components.items():
                    for row in rows:
                        ratio = float(row["requested_ratio"])
                        if ratio in DISPLAY_RATIOS:
                            values[(layer, component, ratio)].append(
                                row["normalized_token_feature"][
                                    "relative_l2_error"
                                ]
                            )

    layers = sorted({key[0] for key in values})
    if not layers:
        raise RuntimeError("No exact layer/component sweep rows found")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), constrained_layout=True)
    colors = {0.2: "#d95f02", 0.5: "#7570b3", 0.8: "#1b9e77"}
    for axis, (component, title) in zip(axes, COMPONENTS):
        for ratio in DISPLAY_RATIOS:
            means = [
                np.mean(values[(layer, component, ratio)])
                for layer in layers
            ]
            axis.plot(
                layers,
                means,
                color=colors[ratio],
                linewidth=1.8,
                marker="o",
                markersize=2.7,
                label=f"{ratio:.0%} keep",
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
    fig.suptitle(
        "Exact finite-difference support: one component in one layer",
        fontsize=12,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
