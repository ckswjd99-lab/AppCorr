#!/usr/bin/env python3
"""Fit a lightweight layer/component pruning allocator from sensitivity curves."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


COMPONENTS = ("input_token", "attention_edge", "ffn_channel")
STAGES = {
    "early_layers_0_12": range(0, 13),
    "middle_layers_13_26": range(13, 27),
    "late_layers_27_39": range(27, 40),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stats",
        type=Path,
        default=Path(
            "analysis/experiments/results/"
            "jacobian_layer_support_sweep_stats.json"
        ),
    )
    parser.add_argument("--target-pruning", default="0.25,0.5,0.75")
    parser.add_argument(
        "--risk-sigma",
        type=float,
        default=0.25,
        help="Fit mean + risk_sigma * sample STD before monotonic regression.",
    )
    parser.add_argument(
        "--cost-mode",
        choices=("equal", "vit7b_flops"),
        default="equal",
        help="Definition of the global pruning-rate budget.",
    )
    parser.add_argument(
        "--component-costs",
        default=None,
        help="Optional input_token=...,attention_edge=...,ffn_channel=... override.",
    )
    parser.add_argument(
        "--grid-step",
        type=float,
        default=0.1,
        help="Policy keep-ratio grid; 0.1 uses only measured support points.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "analysis/experiments/results/"
            "jacobian_pruning_policy_equal_work.json"
        ),
    )
    return parser.parse_args()


def decreasing_isotonic(values: np.ndarray) -> np.ndarray:
    """Unweighted pool-adjacent-violators fit constrained to decrease."""

    blocks: list[dict[str, float | int]] = []
    for index, value in enumerate(values):
        blocks.append({
            "start": index,
            "end": index + 1,
            "weight": 1.0,
            "sum": float(value),
        })
        while len(blocks) >= 2:
            previous = blocks[-2]
            current = blocks[-1]
            previous_mean = float(previous["sum"]) / float(previous["weight"])
            current_mean = float(current["sum"]) / float(current["weight"])
            if previous_mean >= current_mean:
                break
            blocks[-2:] = [{
                "start": int(previous["start"]),
                "end": int(current["end"]),
                "weight": float(previous["weight"]) + float(current["weight"]),
                "sum": float(previous["sum"]) + float(current["sum"]),
            }]

    fitted = np.empty_like(values, dtype=np.float64)
    for block in blocks:
        fitted[int(block["start"]):int(block["end"])] = (
            float(block["sum"]) / float(block["weight"])
        )
    return fitted


def parse_component_costs(args: argparse.Namespace) -> dict[str, float]:
    if args.component_costs is not None:
        costs = {}
        for item in args.component_costs.split(","):
            name, value = item.split("=", maxsplit=1)
            costs[name.strip()] = float(value)
        if set(costs) != set(COMPONENTS):
            raise ValueError(
                f"--component-costs must define exactly {COMPONENTS}"
            )
    elif args.cost_mode == "equal":
        costs = {component: 1.0 for component in COMPONENTS}
    else:
        tokens = 261
        hidden = 4096
        ffn_hidden = 8192
        costs = {
            "input_token": float(4 * tokens * hidden * hidden),
            "attention_edge": float(2 * tokens * tokens * hidden),
            "ffn_channel": float(3 * tokens * hidden * ffn_hidden),
        }
    if any(value <= 0 for value in costs.values()):
        raise ValueError("Component costs must be positive")
    scale = sum(costs.values()) / len(costs)
    return {name: value / scale for name, value in costs.items()}


def fit_curves(
    stats: dict,
    *,
    risk_sigma: float,
    grid_step: float,
) -> tuple[np.ndarray, dict[tuple[int, str], np.ndarray]]:
    if risk_sigma < 0:
        raise ValueError("--risk-sigma must be non-negative")
    if not 0 < grid_step <= 0.1:
        raise ValueError("--grid-step must be in (0, 0.1]")
    grid_size = round(1 / grid_step)
    if not math.isclose(grid_size * grid_step, 1.0):
        raise ValueError("--grid-step must divide 1.0 exactly")

    source_ratios = np.asarray(stats["fit_ratios"], dtype=np.float64)
    grid = np.linspace(0, 1, grid_size + 1)
    curves = {}
    for component in COMPONENTS:
        ratio_rows = stats["per_layer"][component]
        layers = [int(row["layer"]) for row in ratio_rows["0"]]
        for layer in layers:
            adjusted = []
            for ratio in source_ratios:
                rows = ratio_rows[f"{ratio:g}"]
                row = next(value for value in rows if value["layer"] == layer)
                adjusted.append(
                    float(row["mean"]) + risk_sigma * float(row["std"])
                )
            monotone = decreasing_isotonic(
                np.asarray(adjusted, dtype=np.float64)
            )
            curves[(layer, component)] = np.interp(
                grid, source_ratios, monotone
            )
    return grid, curves


def predicted_rms(
    indices: dict[tuple[int, str], int],
    curves: dict[tuple[int, str], np.ndarray],
) -> float:
    squared_errors = [
        float(curves[item][index]) ** 2
        for item, index in indices.items()
    ]
    return math.sqrt(sum(squared_errors) / len(squared_errors))


def uniform_predicted_rms(
    items: list[tuple[int, str]],
    grid: np.ndarray,
    curves: dict[tuple[int, str], np.ndarray],
    target_pruning: float,
) -> float:
    target_keep = 1 - target_pruning
    squared_errors = [
        float(np.interp(target_keep, grid, curves[item])) ** 2
        for item in items
    ]
    return math.sqrt(sum(squared_errors) / len(squared_errors))


def allocate(
    *,
    target_pruning: float,
    grid: np.ndarray,
    curves: dict[tuple[int, str], np.ndarray],
    component_costs: dict[str, float],
) -> dict:
    if not 0 <= target_pruning <= 1:
        raise ValueError("Target pruning rates must be in [0, 1]")
    items = sorted(curves)
    indices = {item: len(grid) - 1 for item in items}
    full_cost = sum(component_costs[component] for _, component in items)
    target_cost = (1 - target_pruning) * full_cost
    current_cost = full_cost

    while current_cost > target_cost + full_cost * 1e-9:
        candidates = []
        for item in items:
            current_index = indices[item]
            if current_index == 0:
                continue
            next_index = current_index - 1
            component_cost = component_costs[item[1]]
            saved_cost = (
                float(grid[current_index] - grid[next_index])
                * component_cost
            )
            current_error = float(curves[item][current_index])
            next_error = float(curves[item][next_index])
            damage = max(next_error**2 - current_error**2, 0.0)
            candidates.append((
                damage / saved_cost,
                damage,
                item,
                next_index,
                saved_cost,
            ))
        if not candidates:
            break
        _score, _damage, item, next_index, saved_cost = min(candidates)
        indices[item] = next_index
        current_cost -= saved_cost

    optimized_rms = predicted_rms(indices, curves)
    uniform_rms = uniform_predicted_rms(
        items, grid, curves, target_pruning
    )
    achieved_pruning = 1 - current_cost / full_cost

    schedule = []
    for layer in range(40):
        row = {"layer": layer}
        for component in COMPONENTS:
            keep = float(grid[indices[(layer, component)]])
            row[f"{component}_keep"] = keep
            row[f"{component}_pruning"] = 1 - keep
        schedule.append(row)

    component_summary = {}
    for component in COMPONENTS:
        keeps = np.asarray([
            grid[indices[(layer, component)]]
            for layer in range(40)
        ])
        component_summary[component] = {
            "mean_keep": float(keeps.mean()),
            "mean_pruning": float(1 - keeps.mean()),
            "min_keep": float(keeps.min()),
            "max_keep": float(keeps.max()),
        }

    stage_summary = {}
    for stage_name, stage_layers in STAGES.items():
        stage_summary[stage_name] = {}
        for component in COMPONENTS:
            keeps = np.asarray([
                grid[indices[(layer, component)]]
                for layer in stage_layers
            ])
            stage_summary[stage_name][component] = {
                "mean_keep": float(keeps.mean()),
                "mean_pruning": float(1 - keeps.mean()),
            }

    return {
        "target_pruning_rate": target_pruning,
        "achieved_pruning_rate": achieved_pruning,
        "optimized_predicted_rms": optimized_rms,
        "uniform_predicted_rms": uniform_rms,
        "predicted_rms_reduction_vs_uniform": (
            1 - optimized_rms / max(uniform_rms, 1e-12)
        ),
        "component_summary": component_summary,
        "stage_summary": stage_summary,
        "schedule": schedule,
    }


def main() -> None:
    args = parse_args()
    stats = json.loads(args.stats.read_text(encoding="utf-8"))
    target_pruning_rates = [
        float(value)
        for value in args.target_pruning.split(",")
        if value.strip()
    ]
    component_costs = parse_component_costs(args)
    grid, curves = fit_curves(
        stats,
        risk_sigma=args.risk_sigma,
        grid_step=args.grid_step,
    )
    policies = [
        allocate(
            target_pruning=target,
            grid=grid,
            curves=curves,
            component_costs=component_costs,
        )
        for target in target_pruning_rates
    ]
    payload = {
        "schema_version": 1,
        "model": {
            "type": "monotone_isotonic_curves_plus_marginal_greedy_allocator",
            "source_stats": str(args.stats),
            "sample_count": stats["sample_count"],
            "risk_sigma": args.risk_sigma,
            "grid_step": args.grid_step,
            "support_grid_semantics": (
                "requested structured support ratio; realized block keep can differ"
            ),
            "cost_mode": args.cost_mode,
            "normalized_component_costs": component_costs,
            "objective": "root mean square of isolated final-feature L2",
            "caveat": (
                "Isolated layer/component errors are treated as additive in "
                "squared error; mixed-policy interaction is not calibrated."
            ),
        },
        "policies": policies,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    for policy in policies:
        components = policy["component_summary"]
        print(
            f"target={policy['target_pruning_rate']:.0%} "
            f"achieved={policy['achieved_pruning_rate']:.1%} "
            f"predicted_rms={policy['optimized_predicted_rms']:.4f} "
            f"uniform={policy['uniform_predicted_rms']:.4f} "
            f"token/attn/ffn pruning="
            f"{components['input_token']['mean_pruning']:.0%}/"
            f"{components['attention_edge']['mean_pruning']:.0%}/"
            f"{components['ffn_channel']['mean_pruning']:.0%}"
        )
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
