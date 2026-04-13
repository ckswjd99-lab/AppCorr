import argparse
import csv
import math
import os
import sys
import textwrap
from contextlib import redirect_stderr, redirect_stdout
from contextlib import nullcontext
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from statistics import fmean, stdev
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from appcorr.models.dinov3.hub import backbones as dinov3_backbones  # noqa: E402
from offload.server.model.utils import load_weight_mmap  # noqa: E402


MODEL_ALIASES = {
    "small": "dinov3_vits16",
    "s": "dinov3_vits16",
    "vits16": "dinov3_vits16",
    "dinov3_vits16": "dinov3_vits16",
    "small_plus": "dinov3_vits16plus",
    "vits16plus": "dinov3_vits16plus",
    "dinov3_vits16plus": "dinov3_vits16plus",
    "base": "dinov3_vitb16",
    "b": "dinov3_vitb16",
    "vitb16": "dinov3_vitb16",
    "dinov3_vitb16": "dinov3_vitb16",
    "large": "dinov3_vitl16",
    "l": "dinov3_vitl16",
    "vitl16": "dinov3_vitl16",
    "dinov3_vitl16": "dinov3_vitl16",
    "large_plus": "dinov3_vitl16plus",
    "vitl16plus": "dinov3_vitl16plus",
    "dinov3_vitl16plus": "dinov3_vitl16plus",
    "huge": "dinov3_vith16plus",
    "h": "dinov3_vith16plus",
    "vith16plus": "dinov3_vith16plus",
    "dinov3_vith16plus": "dinov3_vith16plus",
    "7b": "dinov3_vit7b16",
    "vit7b16": "dinov3_vit7b16",
    "dinov3_vit7b16": "dinov3_vit7b16",
}

MODEL_WEIGHT_CANDIDATES = {
    "dinov3_vits16": [
        "~/cjpark/weights/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
        "~/.cache/torch/hub/checkpoints/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    ],
    "dinov3_vits16plus": [
        "~/cjpark/weights/dinov3/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
        "~/.cache/torch/hub/checkpoints/dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth",
    ],
    "dinov3_vitb16": [
        "~/cjpark/weights/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        "~/.cache/torch/hub/checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    ],
    "dinov3_vitl16": [
        "~/cjpark/weights/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
        "~/.cache/torch/hub/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    ],
    "dinov3_vitl16plus": [
        "~/cjpark/weights/dinov3/dinov3_vitl16plus_pretrain_lvd1689m-46503df0.pth",
        "~/.cache/torch/hub/checkpoints/dinov3_vitl16plus_pretrain_lvd1689m-46503df0.pth",
    ],
    "dinov3_vith16plus": [
        "~/cjpark/weights/dinov3/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
        "~/.cache/torch/hub/checkpoints/dinov3_vith16plus_pretrain_lvd1689m-7c1da9a5.pth",
    ],
    "dinov3_vit7b16": [
        "~/cjpark/weights/dinov3/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
        "~/.cache/torch/hub/checkpoints/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
    ],
}

MODEL_COMPUTE_DTYPES = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}

IMAGE_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGE_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
DEFAULT_IMAGE_SIZES = {
    "imagenet-1k": 256,
    "coco": 1024,
}


def normalize_dataset_name(name: str) -> str:
    value = str(name).strip().lower()
    aliases = {
        "imagenet": "imagenet-1k",
        "imnet-1k": "imagenet-1k",
        "imagenet-1k": "imagenet-1k",
        "coco": "coco",
        "coco2017": "coco",
        "coco-2017": "coco",
    }
    if value not in aliases:
        raise ValueError(f"Unsupported dataset: {name}")
    return aliases[value]


def default_data_root_for_dataset(dataset_name: str) -> str:
    if normalize_dataset_name(dataset_name) == "imagenet-1k":
        return "~/data/imagenet_val"
    return ""


def resolve_output_dir(prefix: str, out_dir: str | None) -> Path:
    if out_dir is not None:
        path = Path(out_dir).expanduser()
    else:
        from datetime import datetime

        path = Path("logs") / "analysis" / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict) -> None:
    import json

    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


@dataclass(frozen=True)
class ExperimentCondition:
    family: str
    label: str
    budget: int | None
    k: int | None
    m: int | None
    tail_prune_ratio: float | None = None


CORRECTION_VARIANTS = (
    ("sdv_only", False, "SdV"),
    ("sdv_plus_dsv", True, "SdV+dSV"),
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run AppCorr SdV partitioning experiments on DINOv3 backbones.")
    parser.add_argument("--dataset", type=str, default="imagenet-1k", help="Dataset: imagenet-1k or coco.")
    parser.add_argument("--data-root", type=str, default=None, help="Dataset root. Ignored for COCO.")
    parser.add_argument("--model", type=str, default="base", help="Backbone size alias: small, base, large, huge, or 7b.")
    parser.add_argument("--weights-path", type=str, default=None, help="Optional explicit checkpoint path.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size.")
    parser.add_argument("--max-batches", type=int, default=1, help="Maximum number of batches to process.")
    parser.add_argument("--image-size", type=int, default=None, help="Square analysis size. Defaults: 256 for ImageNet, 1024 for COCO.")
    parser.add_argument("--source-level", type=int, default=2, help="Pyramid source level used for the approximate input.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--device", type=str, default=None, help="Torch device. Default: cuda if available else cpu.")
    parser.add_argument("--autocast-dtype", type=str, choices=tuple(MODEL_COMPUTE_DTYPES.keys()), default="bf16", help="Autocast dtype on CUDA.")
    parser.add_argument("--feature-scope", type=str, choices=("all_tokens", "patch_tokens", "cls_token"), default="all_tokens", help="Feature scope used for final metrics.")
    parser.add_argument("--query-chunk-size", type=int, default=128, help="Query chunk size for score aggregation and partitioned SdV.")
    parser.add_argument("--metric-chunk-size", type=int, default=1_048_576, help="Feature chunk size used during metric computation.")
    parser.add_argument(
        "--conservative-tail-prune-ratio",
        type=float,
        default=0.5,
        help="For prune+merge conservative sink: prune this bottom fraction of the post-top-K tail before merging the remainder.",
    )
    parser.add_argument("--skip-dashboard", action="store_true", help="Skip PNG dashboard rendering to avoid extra host-memory use.")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory. Defaults to logs/analysis/sdv_partition_<timestamp>.")
    return parser.parse_args()


def resolve_device(value: str | None) -> torch.device:
    if value is not None:
        return torch.device(value)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_model_name(value: str) -> str:
    key = str(value).strip().lower()
    if key not in MODEL_ALIASES:
        supported = ", ".join(sorted(MODEL_ALIASES))
        raise ValueError(f"Unsupported model alias '{value}'. Supported aliases: {supported}")
    return MODEL_ALIASES[key]


def resolve_weights_path(model_name: str, explicit_path: str | None) -> Path:
    if explicit_path is not None:
        path = Path(explicit_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    candidates = MODEL_WEIGHT_CANDIDATES.get(model_name, [])
    for candidate in candidates:
        path = Path(candidate).expanduser()
        if path.exists():
            return path

    pattern = f"{model_name}_pretrain_lvd1689m-*.pth"
    search_roots = [
        Path("~/cjpark/weights/dinov3").expanduser(),
        Path("~/.cache/torch/hub/checkpoints").expanduser(),
    ]
    for root in search_roots:
        matches = sorted(root.glob(pattern))
        if matches:
            return matches[0]

    joined_candidates = ", ".join(str(Path(path).expanduser()) for path in candidates)
    raise FileNotFoundError(
        f"Could not find local weights for {model_name}. "
        f"Looked for: {joined_candidates if joined_candidates else pattern}"
    )


def build_conditions(conservative_tail_prune_ratio: float) -> list[ExperimentCondition]:
    conditions = [
        ExperimentCondition(family="no_correction", label="NoCorr", budget=None, k=None, m=None),
        ExperimentCondition(family="dense", label="Dense", budget=None, k=None, m=None),
    ]
    grid = {
        64: [(63, 1), (60, 4), (48, 16), (32, 32)],
        128: [(127, 1), (120, 8), (96, 32), (64, 64)],
    }
    prune_suffix = f"P{int(round(conservative_tail_prune_ratio * 100.0))}"
    for budget, pairs in grid.items():
        conditions.append(
            ExperimentCondition(
                family="topk_prune",
                label=f"C{budget}_K{budget}_M0",
                budget=budget,
                k=budget,
                m=0,
            )
        )
        for k, m in pairs:
            family = "topk_1sink" if m == 1 else "topk_msink"
            conditions.append(
                ExperimentCondition(
                    family=family,
                    label=f"C{budget}_K{k}_M{m}",
                    budget=budget,
                    k=k,
                    m=m,
                )
            )
            conditions.append(
                ExperimentCondition(
                    family="topk_conservative_msink",
                    label=f"C{budget}_K{k}_M{m}_{prune_suffix}_MinW",
                    budget=budget,
                    k=k,
                    m=m,
                    tail_prune_ratio=conservative_tail_prune_ratio,
                )
            )
    return conditions


def write_rows_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(raw_rows: list[dict]) -> list[dict]:
    metrics = ("l1_error", "l2_mse", "relative_l2_error", "cosine_similarity")
    grouped: dict[tuple, dict] = {}

    for row in raw_rows:
        key = (
            row["dataset"],
            row["model_name"],
            row["feature_scope"],
            row["source_level"],
            row["correction_variant"],
            row["condition_label"],
            row["family"],
            row["budget"],
            row["k"],
            row["m"],
            row.get("tail_prune_ratio"),
        )
        if key not in grouped:
            grouped[key] = {
                "dataset": row["dataset"],
                "model_name": row["model_name"],
                "feature_scope": row["feature_scope"],
                "source_level": row["source_level"],
                "correction_variant": row["correction_variant"],
                "condition_label": row["condition_label"],
                "family": row["family"],
                "budget": row["budget"],
                "k": row["k"],
                "m": row["m"],
                "tail_prune_ratio": row.get("tail_prune_ratio"),
                "_values": {metric: [] for metric in metrics},
            }
        for metric in metrics:
            grouped[key]["_values"][metric].append(float(row[metric]))

    summary_rows = []
    for grouped_row in grouped.values():
        values = grouped_row.pop("_values")
        summary = dict(grouped_row)
        sample_count = len(values["l1_error"])
        summary["sample_count"] = sample_count
        for metric in metrics:
            metric_values = values[metric]
            summary[f"{metric}_mean"] = fmean(metric_values)
            summary[f"{metric}_std"] = stdev(metric_values) if len(metric_values) > 1 else 0.0
        summary_rows.append(summary)

    return sorted(
        summary_rows,
        key=lambda row: (
            float("-inf") if row["budget"] is None else row["budget"],
            row["correction_variant"],
            row["family"],
            -1 if row["k"] is None else row["k"],
            -1 if row["m"] is None else row["m"],
            row["condition_label"],
        ),
    )


def format_table(rows: list[dict], columns: list[str]) -> str:
    if not rows:
        return ""

    def _format_value(value):
        if isinstance(value, float):
            return f"{value:.6e}"
        if value is None:
            return "-"
        return str(value)

    formatted_rows = [[_format_value(row.get(column)) for column in columns] for row in rows]
    widths = [
        max(len(column), max(len(row[col_idx]) for row in formatted_rows))
        for col_idx, column in enumerate(columns)
    ]

    def _join(values):
        return " | ".join(value.ljust(width) for value, width in zip(values, widths))

    header = _join(columns)
    separator = "-+-".join("-" * width for width in widths)
    body = "\n".join(_join(row) for row in formatted_rows)
    return f"{header}\n{separator}\n{body}"


def maybe_print_pandas_table(summary_rows: list[dict]) -> bool:
    sink = StringIO()
    try:
        with redirect_stdout(sink), redirect_stderr(sink):
            import pandas as pd  # type: ignore
    except Exception as exc:
        print(f"[Warning] pandas import failed, falling back to plain-text summary: {exc}")
        return False

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    return True


def correction_variant_display(variant: str) -> str:
    if variant == "sdv_only":
        return "SdV"
    if variant == "sdv_plus_dsv":
        return "SdV+dSV"
    return variant


def lighten_color(color: tuple[int, int, int], amount: float = 0.55) -> tuple[int, int, int]:
    amount = min(max(float(amount), 0.0), 1.0)
    return tuple(int(round(channel + (255 - channel) * amount)) for channel in color)


def hex_to_rgb(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    return tuple(int(value[idx : idx + 2], 16) for idx in (0, 2, 4))


def _summary_sort_key(row: dict) -> tuple:
    return (
        -1 if row["budget"] is None else row["budget"],
        -1 if row["k"] is None else row["k"],
        -1 if row["m"] is None else row["m"],
        row["plot_label"],
    )


def build_variant_overlay_rows(summary_rows: list[dict], mean_key: str, std_key: str) -> list[dict]:
    grouped: dict[tuple, dict] = {}
    for row in summary_rows:
        key = (
            row["family"],
            row["budget"],
            row["k"],
            row["m"],
            row["condition_label"],
            row.get("tail_prune_ratio"),
        )
        entry = grouped.setdefault(
            key,
            {
                "plot_label": "Dense" if row["family"] == "dense" else row["condition_label"],
                "family": row["family"],
                "budget": row["budget"],
                "k": row["k"],
                "m": row["m"],
                "tail_prune_ratio": row.get("tail_prune_ratio"),
                "sdv_only_mean": None,
                "sdv_only_std": 0.0,
                "sdv_plus_dsv_mean": None,
                "sdv_plus_dsv_std": 0.0,
            },
        )
        variant = row["correction_variant"]
        entry[f"{variant}_mean"] = row[mean_key]
        entry[f"{variant}_std"] = row[std_key]

    rows = list(grouped.values())
    rows.sort(key=_summary_sort_key)
    return rows


def find_summary_row(
    summary_rows: list[dict],
    *,
    family: str,
    correction_variant: str,
    budget: int | None = None,
    condition_label: str | None = None,
) -> dict:
    for row in summary_rows:
        if row["family"] != family:
            continue
        if row["correction_variant"] != correction_variant:
            continue
        if budget is not None and row["budget"] != budget:
            continue
        if condition_label is not None and row["condition_label"] != condition_label:
            continue
        return row
    raise KeyError(
        f"No summary row for family={family}, correction_variant={correction_variant}, "
        f"budget={budget}, condition_label={condition_label}"
    )


def _dashboard_palette() -> dict[str, tuple[int, int, int]]:
    return {
        "bg": (245, 247, 250),
        "panel": (255, 255, 255),
        "panel_border": (221, 226, 235),
        "text": (28, 35, 49),
        "muted": (102, 112, 128),
        "grid": (231, 235, 242),
        "blue": (59, 130, 246),
        "teal": (20, 184, 166),
        "orange": (249, 115, 22),
        "red": (239, 68, 68),
        "purple": (139, 92, 246),
        "gray": (107, 114, 128),
        "slate": (51, 65, 85),
    }


def _load_dashboard_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size=size)
        except Exception:
            continue
    return ImageFont.load_default()


def _format_dashboard_number(value: float) -> str:
    value = float(value)
    if value == 0.0:
        return "0"
    if abs(value) < 0.01 or abs(value) >= 1000:
        return f"{value:.2e}"
    return f"{value:.4f}"


def _text_size(draw, text: str, font) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def _fit_text(draw, text: str, font, max_width: int) -> str:
    if _text_size(draw, text, font)[0] <= max_width:
        return text
    for width in range(len(text), 0, -1):
        candidate = textwrap.shorten(text, width=width, placeholder="...")
        if _text_size(draw, candidate, font)[0] <= max_width:
            return candidate
    return "..."


def _draw_panel(draw, rect: tuple[int, int, int, int], title: str, subtitle: str, title_font, small_font, colors: dict[str, tuple[int, int, int]]) -> tuple[int, int, int, int]:
    left, top, right, bottom = rect
    draw.rounded_rectangle(rect, radius=24, fill=colors["panel"], outline=colors["panel_border"], width=2)
    draw.text((left + 24, top + 18), title, font=title_font, fill=colors["text"])
    if subtitle:
        draw.text((left + 24, top + 52), subtitle, font=small_font, fill=colors["muted"])
    return left + 72, top + 92, right - 28, bottom - 54


def _draw_legend(draw, anchor: tuple[int, int], items: list[tuple[str, tuple[int, int, int]]], font, colors: dict[str, tuple[int, int, int]]) -> None:
    x, y = anchor
    cursor_x = x
    for label, color in items:
        draw.rounded_rectangle((cursor_x, y + 4, cursor_x + 18, y + 22), radius=4, fill=color)
        draw.text((cursor_x + 26, y), label, font=font, fill=colors["muted"])
        cursor_x += 26 + _text_size(draw, label, font)[0] + 20


def _draw_grouped_bar_panel(
    draw,
    rect: tuple[int, int, int, int],
    title: str,
    subtitle: str,
    categories: list[str],
    series: list[dict],
    title_font,
    small_font,
    tick_font,
    colors: dict[str, tuple[int, int, int]],
) -> None:
    plot_left, plot_top, plot_right, plot_bottom = _draw_panel(draw, rect, title, subtitle, title_font, small_font, colors)
    values = [value for item in series for value in item["values"]]
    overlay_values = [value for item in series for value in item.get("overlay_values", []) if value is not None]
    values.extend(overlay_values)
    y_max = max(values) if values else 1.0
    y_max = max(y_max * 1.15, 1e-8)
    axis_left = plot_left + 48
    axis_bottom = plot_bottom - 34
    axis_top = plot_top + 14
    axis_right = plot_right - 20
    ticks = 5
    for tick_idx in range(ticks + 1):
        frac = tick_idx / ticks
        y = axis_bottom - frac * (axis_bottom - axis_top)
        draw.line((axis_left, y, axis_right, y), fill=colors["grid"], width=1)
        label = _format_dashboard_number(y_max * frac)
        label_w, label_h = _text_size(draw, label, tick_font)
        draw.text((axis_left - label_w - 10, y - label_h / 2), label, font=tick_font, fill=colors["muted"])
    draw.line((axis_left, axis_top, axis_left, axis_bottom), fill=colors["muted"], width=2)
    draw.line((axis_left, axis_bottom, axis_right, axis_bottom), fill=colors["muted"], width=2)
    if categories:
        group_width = (axis_right - axis_left) / len(categories)
        bar_width = max(group_width * 0.56 / max(len(series), 1), 10)
        for cat_idx, category in enumerate(categories):
            center_x = axis_left + group_width * (cat_idx + 0.5)
            total_bar_width = len(series) * bar_width
            start_x = center_x - total_bar_width / 2
            for series_idx, item in enumerate(series):
                value = item["values"][cat_idx]
                bar_height = 0.0 if y_max <= 0 else (value / y_max) * (axis_bottom - axis_top)
                x0 = start_x + series_idx * bar_width + 6
                x1 = start_x + (series_idx + 1) * bar_width - 6
                y0 = axis_bottom - bar_height
                base_color = item.get("base_color", item["color"])
                overlay_color = item.get("overlay_color", item["color"])
                draw.rounded_rectangle((x0, y0, x1, axis_bottom), radius=8, fill=base_color)
                overlay = item.get("overlay_values")
                if overlay is not None:
                    overlay_value = overlay[cat_idx]
                    overlay_height = 0.0 if y_max <= 0 else (overlay_value / y_max) * (axis_bottom - axis_top)
                    ox0 = x0 + (x1 - x0) * 0.18
                    ox1 = x1 - (x1 - x0) * 0.18
                    oy0 = axis_bottom - overlay_height
                    draw.rounded_rectangle((ox0, oy0, ox1, axis_bottom), radius=6, fill=overlay_color)
            label = _fit_text(draw, category, tick_font, int(group_width - 12))
            label_w, label_h = _text_size(draw, label, tick_font)
            draw.text((center_x - label_w / 2, axis_bottom + 12), label, font=tick_font, fill=colors["text"])
    legend_width = sum(26 + _text_size(draw, label, small_font)[0] + 20 for label, _ in [(item["label"], item["color"]) for item in series])
    _draw_legend(
        draw,
        (max(plot_left + 12, plot_right - legend_width - 16), rect[1] + 18),
        [(item["label"], item["color"]) for item in series],
        small_font,
        colors,
    )


def _draw_line_panel(
    draw,
    rect: tuple[int, int, int, int],
    title: str,
    subtitle: str,
    series: list[dict],
    title_font,
    small_font,
    tick_font,
    colors: dict[str, tuple[int, int, int]],
) -> None:
    plot_left, plot_top, plot_right, plot_bottom = _draw_panel(draw, rect, title, subtitle, title_font, small_font, colors)
    x_values = sorted({int(x) for item in series for x in item["x"]})
    y_values = [float(y) for item in series for y in item["y"]]
    if not x_values or not y_values:
        draw.text((plot_left, plot_top), "No data", font=small_font, fill=colors["muted"])
        return
    x_min = min(x_values)
    x_max = max(x_values)
    if x_min == x_max:
        x_min -= 1
        x_max += 1
    y_min = min(0.0, min(y_values))
    y_max = max(y_values)
    if math.isclose(y_min, y_max):
        y_max = y_min + 1.0
    y_pad = (y_max - y_min) * 0.15
    y_max += y_pad
    axis_left = plot_left + 48
    axis_bottom = plot_bottom - 34
    axis_top = plot_top + 14
    axis_right = plot_right - 20
    ticks = 5
    for tick_idx in range(ticks + 1):
        frac = tick_idx / ticks
        y_value = y_min + (y_max - y_min) * frac
        y = axis_bottom - frac * (axis_bottom - axis_top)
        draw.line((axis_left, y, axis_right, y), fill=colors["grid"], width=1)
        label = _format_dashboard_number(y_value)
        label_w, label_h = _text_size(draw, label, tick_font)
        draw.text((axis_left - label_w - 10, y - label_h / 2), label, font=tick_font, fill=colors["muted"])
    draw.line((axis_left, axis_top, axis_left, axis_bottom), fill=colors["muted"], width=2)
    draw.line((axis_left, axis_bottom, axis_right, axis_bottom), fill=colors["muted"], width=2)

    def map_x(value: float) -> float:
        return axis_left + (value - x_min) / (x_max - x_min) * (axis_right - axis_left)

    def map_y(value: float) -> float:
        return axis_bottom - (value - y_min) / (y_max - y_min) * (axis_bottom - axis_top)

    for tick_value in x_values:
        x = map_x(tick_value)
        draw.line((x, axis_bottom, x, axis_bottom + 6), fill=colors["muted"], width=2)
        label = str(tick_value)
        label_w, _ = _text_size(draw, label, tick_font)
        draw.text((x - label_w / 2, axis_bottom + 10), label, font=tick_font, fill=colors["text"])

    legend_width = sum(26 + _text_size(draw, item["label"], small_font)[0] + 20 for item in series)
    _draw_legend(
        draw,
        (max(plot_left + 12, plot_right - legend_width - 16), rect[1] + 18),
        [(item["label"], item["color"]) for item in series],
        small_font,
        colors,
    )

    for item in series:
        points = [(map_x(float(x)), map_y(float(y))) for x, y in zip(item["x"], item["y"])]
        if len(points) >= 2:
            if item.get("linestyle") == "dashed":
                for point_idx in range(len(points) - 1):
                    x0, y0 = points[point_idx]
                    x1, y1 = points[point_idx + 1]
                    steps = max(1, int(max(abs(x1 - x0), abs(y1 - y0)) // 12))
                    for step_idx in range(steps):
                        if step_idx % 2 == 1:
                            continue
                        t0 = step_idx / steps
                        t1 = min((step_idx + 1) / steps, 1.0)
                        sx0 = x0 + (x1 - x0) * t0
                        sy0 = y0 + (y1 - y0) * t0
                        sx1 = x0 + (x1 - x0) * t1
                        sy1 = y0 + (y1 - y0) * t1
                        draw.line((sx0, sy0, sx1, sy1), fill=item["color"], width=4)
            else:
                draw.line(points, fill=item["color"], width=4)
        for x, y in points:
            if item.get("linestyle") == "dashed":
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=(255, 255, 255), outline=item["color"], width=3)
            else:
                draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=item["color"], outline=(255, 255, 255), width=2)


def _draw_horizontal_bar_panel(
    draw,
    rect: tuple[int, int, int, int],
    title: str,
    subtitle: str,
    rows: list[dict],
    base_mean_key: str,
    base_std_key: str,
    overlay_mean_key: str,
    overlay_std_key: str,
    title_font,
    small_font,
    tick_font,
    colors: dict[str, tuple[int, int, int]],
) -> None:
    plot_left, plot_top, plot_right, plot_bottom = _draw_panel(draw, rect, title, subtitle, title_font, small_font, colors)
    if not rows:
        draw.text((plot_left, plot_top), "No data", font=small_font, fill=colors["muted"])
        return
    label_width = 220
    axis_left = plot_left + label_width
    axis_right = plot_right - 100
    axis_top = plot_top + 10
    axis_bottom = plot_bottom - 20
    metric_values = []
    for row in rows:
        if row[base_mean_key] is not None:
            metric_values.append((float(row[base_mean_key]), float(row[base_std_key])))
        if row[overlay_mean_key] is not None:
            metric_values.append((float(row[overlay_mean_key]), float(row[overlay_std_key])))
    lower = min(max(0.0, mean - std) for mean, std in metric_values)
    upper = max(mean + std for mean, std in metric_values)
    if math.isclose(lower, upper):
        upper = lower + 0.05
    margin = max((upper - lower) * 0.15, 0.02 if upper <= 1.0 else (upper - lower) * 0.05)
    x_min = max(0.0, lower - margin)
    x_max = upper + margin
    if math.isclose(x_min, x_max):
        x_min = max(0.0, x_min - 0.05)
        x_max = x_max + 0.05
    ticks = 5
    for tick_idx in range(ticks + 1):
        frac = tick_idx / ticks
        x = axis_left + frac * (axis_right - axis_left)
        draw.line((x, axis_top, x, axis_bottom), fill=colors["grid"], width=1)
        value = x_min + frac * (x_max - x_min)
        label = f"{value:.3f}"
        label_w, _ = _text_size(draw, label, tick_font)
        draw.text((x - label_w / 2, axis_bottom + 10), label, font=tick_font, fill=colors["muted"])
    row_height = (axis_bottom - axis_top) / len(rows)

    def map_x(value: float) -> float:
        return axis_left + (value - x_min) / (x_max - x_min) * (axis_right - axis_left)

    color_by_family = {
        "no_correction": colors["slate"],
        "dense": colors["gray"],
        "topk_prune": colors["purple"],
        "topk_1sink": colors["blue"],
        "topk_msink": colors["orange"],
        "topk_conservative_msink": colors["teal"],
    }
    for idx, row in enumerate(rows):
        y = axis_top + row_height * (idx + 0.5)
        label = _fit_text(draw, row["plot_label"], tick_font, label_width - 16)
        draw.text((plot_left, y - 9), label, font=tick_font, fill=colors["text"])
        base_color = lighten_color(color_by_family.get(row["family"], colors["blue"]), amount=0.5)
        overlay_color = color_by_family.get(row["family"], colors["blue"])
        base_mean = row[base_mean_key]
        base_std = row[base_std_key]
        overlay_mean = row[overlay_mean_key]
        overlay_std = row[overlay_std_key]
        if base_mean is not None:
            base_mean = float(base_mean)
            base_std = float(base_std)
            base_right = map_x(base_mean)
            draw.rounded_rectangle((axis_left, y - 10, base_right, y + 10), radius=8, fill=base_color)
            err_left = map_x(max(x_min, base_mean - base_std))
            err_right = map_x(min(x_max, base_mean + base_std))
            draw.line((err_left, y, err_right, y), fill=colors["text"], width=2)
            draw.line((err_left, y - 6, err_left, y + 6), fill=colors["text"], width=2)
            draw.line((err_right, y - 6, err_right, y + 6), fill=colors["text"], width=2)
        if overlay_mean is not None:
            overlay_mean = float(overlay_mean)
            overlay_std = float(overlay_std)
            overlay_right = map_x(overlay_mean)
            draw.rounded_rectangle((axis_left, y - 6, overlay_right, y + 6), radius=6, fill=overlay_color)
            oerr_left = map_x(max(x_min, overlay_mean - overlay_std))
            oerr_right = map_x(min(x_max, overlay_mean + overlay_std))
            draw.line((oerr_left, y, oerr_right, y), fill=colors["text"], width=2)
            draw.line((oerr_left, y - 4, oerr_left, y + 4), fill=colors["text"], width=2)
            draw.line((oerr_right, y - 4, oerr_right, y + 4), fill=colors["text"], width=2)
        value_text = f"{(overlay_mean if overlay_mean is not None else base_mean):.4f}"
        draw.text((axis_right + 12, y - 9), value_text, font=tick_font, fill=colors["muted"])
    _draw_legend(
        draw,
        (rect[0] + 24, rect[1] + 18),
        [
            ("No Correction", colors["slate"]),
            ("Dense", colors["gray"]),
            ("Top-K Prune", colors["purple"]),
            ("Top-K + 1 Sink", colors["blue"]),
            ("Top-K + M Sink", colors["orange"]),
            ("Prune + Min Sink", colors["teal"]),
        ],
        small_font,
        colors,
    )


def write_dashboard_png(summary_rows: list[dict], output_path: Path) -> None:
    if not summary_rows:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
        from matplotlib.gridspec import GridSpec  # type: ignore
        from matplotlib.lines import Line2D  # type: ignore
        from matplotlib.patches import Patch  # type: ignore
    except Exception as exc:
        print(f"[Warning] matplotlib import failed, falling back to PIL dashboard renderer: {exc}")
    else:
        dense_row = find_summary_row(summary_rows, family="dense", correction_variant="sdv_only")
        nocorr_row = {
            "base": find_summary_row(summary_rows, family="no_correction", correction_variant="sdv_only"),
            "overlay": find_summary_row(summary_rows, family="no_correction", correction_variant="sdv_plus_dsv"),
        }
        algorithm_rows = []
        for budget in (64, 128):
            prune_label = find_summary_row(summary_rows, family="topk_prune", correction_variant="sdv_only", budget=budget)["condition_label"]
            sink1_label = find_summary_row(summary_rows, family="topk_1sink", correction_variant="sdv_only", budget=budget)["condition_label"]
            msink_rows = [row for row in summary_rows if row["family"] == "topk_msink" and row["budget"] == budget and row["correction_variant"] == "sdv_only"]
            best_msink_label = min(msink_rows, key=lambda row: row["l2_mse_mean"])["condition_label"]
            conservative_rows = [row for row in summary_rows if row["family"] == "topk_conservative_msink" and row["budget"] == budget and row["correction_variant"] == "sdv_only"]
            best_conservative_label = min(conservative_rows, key=lambda row: row["l2_mse_mean"])["condition_label"]
            algorithm_rows.append(
                {
                    "budget": budget,
                    "rows": [
                        nocorr_row,
                        {
                            "base": find_summary_row(summary_rows, family="dense", correction_variant="sdv_only"),
                            "overlay": find_summary_row(summary_rows, family="dense", correction_variant="sdv_plus_dsv"),
                        },
                        {
                            "base": find_summary_row(summary_rows, family="topk_prune", correction_variant="sdv_only", budget=budget, condition_label=prune_label),
                            "overlay": find_summary_row(summary_rows, family="topk_prune", correction_variant="sdv_plus_dsv", budget=budget, condition_label=prune_label),
                        },
                        {
                            "base": find_summary_row(summary_rows, family="topk_1sink", correction_variant="sdv_only", budget=budget, condition_label=sink1_label),
                            "overlay": find_summary_row(summary_rows, family="topk_1sink", correction_variant="sdv_plus_dsv", budget=budget, condition_label=sink1_label),
                        },
                        {
                            "base": find_summary_row(summary_rows, family="topk_msink", correction_variant="sdv_only", budget=budget, condition_label=best_msink_label),
                            "overlay": find_summary_row(summary_rows, family="topk_msink", correction_variant="sdv_plus_dsv", budget=budget, condition_label=best_msink_label),
                        },
                        {
                            "base": find_summary_row(summary_rows, family="topk_conservative_msink", correction_variant="sdv_only", budget=budget, condition_label=best_conservative_label),
                            "overlay": find_summary_row(summary_rows, family="topk_conservative_msink", correction_variant="sdv_plus_dsv", budget=budget, condition_label=best_conservative_label),
                        },
                    ],
                }
            )

        tradeoff_styles = {
            ("topk_1sink", 64): ("1 Sink / C64", "#3b82f6"),
            ("topk_1sink", 128): ("1 Sink / C128", "#14b8a6"),
            ("topk_msink", 64): ("M Sink / C64", "#f97316"),
            ("topk_msink", 128): ("M Sink / C128", "#ef4444"),
            ("topk_conservative_msink", 64): ("Prune+Min / C64", "#8b5cf6"),
            ("topk_conservative_msink", 128): ("Prune+Min / C128", "#7c3aed"),
        }
        tradeoff_series_l1 = []
        tradeoff_series_l2 = []
        for family in ("topk_1sink", "topk_msink", "topk_conservative_msink"):
            for budget in (64, 128):
                label, color = tradeoff_styles[(family, budget)]
                for correction_variant, _, variant_label in CORRECTION_VARIANTS:
                    subset = sorted(
                        [
                            row
                            for row in summary_rows
                            if row["family"] == family and row["budget"] == budget and row["correction_variant"] == correction_variant
                        ],
                        key=lambda row: row["k"],
                    )
                    if not subset:
                        continue
                    line_style = "-" if correction_variant == "sdv_only" else "--"
                    tradeoff_series_l1.append(
                        {
                            "label": f"{label} ({variant_label})",
                            "color": color,
                            "linestyle": line_style,
                            "x": [row["k"] for row in subset],
                            "y": [row["l1_error_mean"] for row in subset],
                        }
                    )
                    tradeoff_series_l2.append(
                        {
                            "label": f"{label} ({variant_label})",
                            "color": color,
                            "linestyle": line_style,
                            "x": [row["k"] for row in subset],
                            "y": [row["l2_mse_mean"] for row in subset],
                        }
                    )

        cosine_rows = build_variant_overlay_rows(summary_rows, "cosine_similarity_mean", "cosine_similarity_std")
        relative_rows = build_variant_overlay_rows(summary_rows, "relative_l2_error_mean", "relative_l2_error_std")

        plt.style.use("seaborn-v0_8-whitegrid")
        fig = plt.figure(figsize=(16, 18), dpi=160)
        fig.patch.set_facecolor("#f5f7fa")
        grid = GridSpec(4, 2, figure=fig, height_ratios=[1.0, 1.0, 1.5, 1.5], hspace=0.38, wspace=0.18)
        ax_l1 = fig.add_subplot(grid[0, 0])
        ax_l2 = fig.add_subplot(grid[0, 1])
        ax_tradeoff_l1 = fig.add_subplot(grid[1, 0])
        ax_tradeoff_l2 = fig.add_subplot(grid[1, 1])
        ax_cosine = fig.add_subplot(grid[2, :])
        ax_relative = fig.add_subplot(grid[3, :])

        axes = [ax_l1, ax_l2, ax_tradeoff_l1, ax_tradeoff_l2, ax_cosine, ax_relative]
        for ax in axes:
            ax.set_facecolor("white")
            ax.grid(axis="y", color="#e5e7eb", linewidth=0.8)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.spines["left"].set_color("#cbd5e1")
            ax.spines["bottom"].set_color("#cbd5e1")
        for ax in (ax_cosine, ax_relative):
            ax.grid(axis="x", color="#e5e7eb", linewidth=0.8)

        categories = ["NoCorr", "Dense", "Prune", "1 Sink", "M Sink", "Prune+Min"]
        bar_x = np.arange(len(categories))
        bar_width = 0.34
        color_by_budget = {64: "#3b82f6", 128: "#f97316"}
        for offset_idx, group in enumerate(algorithm_rows):
            offset = (-0.5 + offset_idx) * bar_width
            base_color = color_by_budget[group["budget"]]
            l1_values = [pair["base"]["l1_error_mean"] for pair in group["rows"]]
            l1_overlay = [pair["overlay"]["l1_error_mean"] for pair in group["rows"]]
            l2_values = [pair["base"]["l2_mse_mean"] for pair in group["rows"]]
            l2_overlay = [pair["overlay"]["l2_mse_mean"] for pair in group["rows"]]
            ax_l1.bar(bar_x + offset, l1_values, width=bar_width, color=base_color, alpha=0.28)
            ax_l1.bar(bar_x + offset, l1_overlay, width=bar_width * 0.58, color=base_color, alpha=0.95)
            ax_l2.bar(bar_x + offset, l2_values, width=bar_width, color=base_color, alpha=0.28)
            ax_l2.bar(bar_x + offset, l2_overlay, width=bar_width * 0.58, color=base_color, alpha=0.95)

        ax_l1.set_title("Algorithm Comparison: L1", fontsize=13, fontweight="bold")
        ax_l2.set_title("Algorithm Comparison: L2 (MSE)", fontsize=13, fontweight="bold")
        ax_l1.set_xticks(bar_x, categories)
        ax_l2.set_xticks(bar_x, categories)
        ax_l1.set_ylabel("Mean L1 Error")
        ax_l2.set_ylabel("Mean L2 Error (MSE)")
        budget_handles = [
            Patch(facecolor=color_by_budget[64], edgecolor="none", alpha=0.95, label="C=64"),
            Patch(facecolor=color_by_budget[128], edgecolor="none", alpha=0.95, label="C=128"),
        ]
        mode_handles = [
            Patch(facecolor="#334155", edgecolor="none", alpha=0.28, label="SdV"),
            Patch(facecolor="#334155", edgecolor="none", alpha=0.95, label="SdV+dSV"),
        ]
        legend_budget_l1 = ax_l1.legend(handles=budget_handles, frameon=False, loc="upper right")
        ax_l1.add_artist(legend_budget_l1)
        ax_l1.legend(handles=mode_handles, frameon=False, loc="upper left")
        legend_budget_l2 = ax_l2.legend(handles=budget_handles, frameon=False, loc="upper right")
        ax_l2.add_artist(legend_budget_l2)
        ax_l2.legend(handles=mode_handles, frameon=False, loc="upper left")

        for item in tradeoff_series_l1:
            ax_tradeoff_l1.plot(
                item["x"],
                item["y"],
                marker="o",
                linewidth=2.4,
                markersize=6,
                color=item["color"],
                linestyle=item["linestyle"],
                label=item["label"],
            )
        for item in tradeoff_series_l2:
            ax_tradeoff_l2.plot(
                item["x"],
                item["y"],
                marker="o",
                linewidth=2.4,
                markersize=6,
                color=item["color"],
                linestyle=item["linestyle"],
                label=item["label"],
            )

        ax_tradeoff_l1.set_title("K vs. M Trade-off: L1", fontsize=13, fontweight="bold")
        ax_tradeoff_l2.set_title("K vs. M Trade-off: L2 (MSE)", fontsize=13, fontweight="bold")
        ax_tradeoff_l1.set_xlabel("K (lossless tokens)")
        ax_tradeoff_l2.set_xlabel("K (lossless tokens)")
        ax_tradeoff_l1.set_ylabel("Mean L1 Error")
        ax_tradeoff_l2.set_ylabel("Mean L2 Error (MSE)")
        family_handles = [
            Line2D([0], [0], color=color, lw=3, label=label)
            for label, color in [
                ("1 Sink / C64", "#3b82f6"),
                ("1 Sink / C128", "#14b8a6"),
                ("M Sink / C64", "#f97316"),
                ("M Sink / C128", "#ef4444"),
                ("Prune+Min / C64", "#8b5cf6"),
                ("Prune+Min / C128", "#7c3aed"),
            ]
        ]
        line_mode_handles = [
            Line2D([0], [0], color="#334155", lw=3, linestyle="-", label="SdV"),
            Line2D([0], [0], color="#334155", lw=3, linestyle="--", label="SdV+dSV"),
        ]
        legend_family_l1 = ax_tradeoff_l1.legend(handles=family_handles, frameon=False, loc="upper right", fontsize=8)
        ax_tradeoff_l1.add_artist(legend_family_l1)
        ax_tradeoff_l1.legend(handles=line_mode_handles, frameon=False, loc="upper left", fontsize=9)
        legend_family_l2 = ax_tradeoff_l2.legend(handles=family_handles, frameon=False, loc="upper right", fontsize=8)
        ax_tradeoff_l2.add_artist(legend_family_l2)
        ax_tradeoff_l2.legend(handles=line_mode_handles, frameon=False, loc="upper left", fontsize=9)

        cosine_labels = [row["plot_label"] for row in cosine_rows]
        cosine_values = np.array([float(row["sdv_only_mean"]) for row in cosine_rows], dtype=float)
        cosine_errors = np.array([float(row["sdv_only_std"]) for row in cosine_rows], dtype=float)
        cosine_overlay_values = np.array([float(row["sdv_plus_dsv_mean"]) for row in cosine_rows], dtype=float)
        cosine_overlay_errors = np.array([float(row["sdv_plus_dsv_std"]) for row in cosine_rows], dtype=float)
        cosine_colors = {
            "no_correction": "#334155",
            "dense": "#6b7280",
            "topk_prune": "#8b5cf6",
            "topk_1sink": "#3b82f6",
            "topk_msink": "#f97316",
            "topk_conservative_msink": "#14b8a6",
        }
        y_pos = np.arange(len(cosine_rows))
        ax_cosine.barh(
            y_pos,
            cosine_values,
            xerr=cosine_errors,
            height=0.72,
            color=[cosine_colors.get(row["family"], "#3b82f6") for row in cosine_rows],
            alpha=0.28,
            error_kw={"elinewidth": 1.0, "ecolor": "#334155", "capsize": 2},
        )
        ax_cosine.barh(
            y_pos,
            cosine_overlay_values,
            xerr=cosine_overlay_errors,
            height=0.42,
            color=[cosine_colors.get(row["family"], "#3b82f6") for row in cosine_rows],
            error_kw={"elinewidth": 1.3, "ecolor": "#334155", "capsize": 3},
        )
        ax_cosine.set_yticks(y_pos, cosine_labels)
        ax_cosine.invert_yaxis()
        ax_cosine.set_title("Feature Direction Preservation", fontsize=13, fontweight="bold")
        ax_cosine.set_xlabel("Cosine Similarity")
        cosine_low = max(0.0, float(np.min(np.minimum(cosine_values - cosine_errors, cosine_overlay_values - cosine_overlay_errors))) - 0.03)
        cosine_high = min(1.0, float(np.max(np.maximum(cosine_values + cosine_errors, cosine_overlay_values + cosine_overlay_errors))) + 0.03)
        if math.isclose(cosine_low, cosine_high):
            cosine_high = min(1.0, cosine_low + 0.05)
        ax_cosine.set_xlim(cosine_low, cosine_high)
        cosine_handles = [
            Line2D([0], [0], color=color, lw=8, label=label)
            for label, color in [
                ("No Correction", cosine_colors["no_correction"]),
                ("Dense", cosine_colors["dense"]),
                ("Top-K Prune", cosine_colors["topk_prune"]),
                ("Top-K + 1 Sink", cosine_colors["topk_1sink"]),
                ("Top-K + M Sink", cosine_colors["topk_msink"]),
                ("Prune + Min Sink", cosine_colors["topk_conservative_msink"]),
            ]
        ]
        overlay_mode_handles = [
            Patch(facecolor="#334155", edgecolor="none", alpha=0.28, label="SdV"),
            Patch(facecolor="#334155", edgecolor="none", alpha=0.95, label="SdV+dSV"),
        ]
        legend_family_cos = ax_cosine.legend(handles=cosine_handles, frameon=False, loc="lower right", ncol=3)
        ax_cosine.add_artist(legend_family_cos)
        ax_cosine.legend(handles=overlay_mode_handles, frameon=False, loc="lower left")

        relative_labels = [row["plot_label"] for row in relative_rows]
        relative_values = np.array([float(row["sdv_only_mean"]) for row in relative_rows], dtype=float)
        relative_errors = np.array([float(row["sdv_only_std"]) for row in relative_rows], dtype=float)
        relative_overlay_values = np.array([float(row["sdv_plus_dsv_mean"]) for row in relative_rows], dtype=float)
        relative_overlay_errors = np.array([float(row["sdv_plus_dsv_std"]) for row in relative_rows], dtype=float)
        relative_y_pos = np.arange(len(relative_rows))
        ax_relative.barh(
            relative_y_pos,
            relative_values,
            xerr=relative_errors,
            height=0.72,
            color=[cosine_colors.get(row["family"], "#3b82f6") for row in relative_rows],
            alpha=0.28,
            error_kw={"elinewidth": 1.0, "ecolor": "#334155", "capsize": 2},
        )
        ax_relative.barh(
            relative_y_pos,
            relative_overlay_values,
            xerr=relative_overlay_errors,
            height=0.42,
            color=[cosine_colors.get(row["family"], "#3b82f6") for row in relative_rows],
            error_kw={"elinewidth": 1.3, "ecolor": "#334155", "capsize": 3},
        )
        ax_relative.set_yticks(relative_y_pos, relative_labels)
        ax_relative.invert_yaxis()
        ax_relative.set_title("Relative Feature Error", fontsize=13, fontweight="bold")
        ax_relative.set_xlabel("||error||_2 / ||feature||_2")
        relative_low = max(0.0, float(np.min(np.minimum(relative_values - relative_errors, relative_overlay_values - relative_overlay_errors))) - 0.03)
        relative_high = float(np.max(np.maximum(relative_values + relative_errors, relative_overlay_values + relative_overlay_errors))) + 0.03
        if math.isclose(relative_low, relative_high):
            relative_high = relative_low + 0.05
        ax_relative.set_xlim(relative_low, relative_high)
        legend_family_rel = ax_relative.legend(handles=cosine_handles, frameon=False, loc="lower right", ncol=3)
        ax_relative.add_artist(legend_family_rel)
        ax_relative.legend(handles=overlay_mode_handles, frameon=False, loc="lower left")

        title = "SdV Partitioning Dashboard"
        meta_line = (
            f"Dataset: {dense_row['dataset']} | Model: {dense_row['model_name']} | "
            f"Feature: {dense_row['feature_scope']} | Source Level: L{dense_row['source_level']} | "
            f"Samples: {dense_row['sample_count']}"
        )
        note_line = "Wide pale bars / solid lines: SdV only. Narrow dark bars / dashed lines: SdV+dSV. L2 is per-element MSE."
        fig.suptitle(title, fontsize=20, fontweight="bold", x=0.02, y=0.985, ha="left")
        fig.text(0.02, 0.955, meta_line, fontsize=11, color="#475569", ha="left")
        fig.text(0.02, 0.935, note_line, fontsize=10.5, color="#64748b", ha="left")

        fig.tight_layout(rect=(0, 0, 1, 0.9))
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        return

    colors = _dashboard_palette()
    title_font = _load_dashboard_font(26, bold=True)
    panel_title_font = _load_dashboard_font(20, bold=True)
    body_font = _load_dashboard_font(16)
    small_font = _load_dashboard_font(14)
    tick_font = _load_dashboard_font(13)

    dense_row = find_summary_row(summary_rows, family="dense", correction_variant="sdv_only")
    nocorr_row = {
        "base": find_summary_row(summary_rows, family="no_correction", correction_variant="sdv_only"),
        "overlay": find_summary_row(summary_rows, family="no_correction", correction_variant="sdv_plus_dsv"),
    }
    algorithm_rows = []
    for budget in (64, 128):
        prune_label = find_summary_row(summary_rows, family="topk_prune", correction_variant="sdv_only", budget=budget)["condition_label"]
        sink1_label = find_summary_row(summary_rows, family="topk_1sink", correction_variant="sdv_only", budget=budget)["condition_label"]
        msink_rows = [row for row in summary_rows if row["family"] == "topk_msink" and row["budget"] == budget and row["correction_variant"] == "sdv_only"]
        best_msink_label = min(msink_rows, key=lambda row: row["l2_mse_mean"])["condition_label"]
        conservative_rows = [row for row in summary_rows if row["family"] == "topk_conservative_msink" and row["budget"] == budget and row["correction_variant"] == "sdv_only"]
        best_conservative_label = min(conservative_rows, key=lambda row: row["l2_mse_mean"])["condition_label"]
        algorithm_rows.append(
            {
                "budget": budget,
                "rows": [
                    nocorr_row,
                    {
                        "base": find_summary_row(summary_rows, family="dense", correction_variant="sdv_only"),
                        "overlay": find_summary_row(summary_rows, family="dense", correction_variant="sdv_plus_dsv"),
                    },
                    {
                        "base": find_summary_row(summary_rows, family="topk_prune", correction_variant="sdv_only", budget=budget, condition_label=prune_label),
                        "overlay": find_summary_row(summary_rows, family="topk_prune", correction_variant="sdv_plus_dsv", budget=budget, condition_label=prune_label),
                    },
                    {
                        "base": find_summary_row(summary_rows, family="topk_1sink", correction_variant="sdv_only", budget=budget, condition_label=sink1_label),
                        "overlay": find_summary_row(summary_rows, family="topk_1sink", correction_variant="sdv_plus_dsv", budget=budget, condition_label=sink1_label),
                    },
                    {
                        "base": find_summary_row(summary_rows, family="topk_msink", correction_variant="sdv_only", budget=budget, condition_label=best_msink_label),
                        "overlay": find_summary_row(summary_rows, family="topk_msink", correction_variant="sdv_plus_dsv", budget=budget, condition_label=best_msink_label),
                    },
                    {
                        "base": find_summary_row(summary_rows, family="topk_conservative_msink", correction_variant="sdv_only", budget=budget, condition_label=best_conservative_label),
                        "overlay": find_summary_row(summary_rows, family="topk_conservative_msink", correction_variant="sdv_plus_dsv", budget=budget, condition_label=best_conservative_label),
                    },
                ],
            }
        )

    tradeoff_series_l1 = []
    tradeoff_series_l2 = []
    tradeoff_styles = {
        ("topk_1sink", 64): ("1 Sink / C64", colors["blue"]),
        ("topk_1sink", 128): ("1 Sink / C128", colors["teal"]),
        ("topk_msink", 64): ("M Sink / C64", colors["orange"]),
        ("topk_msink", 128): ("M Sink / C128", colors["red"]),
        ("topk_conservative_msink", 64): ("Prune+Min / C64", colors["purple"]),
        ("topk_conservative_msink", 128): ("Prune+Min / C128", colors["blue"]),
    }
    for family in ("topk_1sink", "topk_msink", "topk_conservative_msink"):
        for budget in (64, 128):
            label, color = tradeoff_styles[(family, budget)]
            for correction_variant, _, variant_label in CORRECTION_VARIANTS:
                subset = sorted(
                    [
                        row
                        for row in summary_rows
                        if row["family"] == family and row["budget"] == budget and row["correction_variant"] == correction_variant
                    ],
                    key=lambda row: row["k"],
                )
                if not subset:
                    continue
                linestyle = "solid" if correction_variant == "sdv_only" else "dashed"
                tradeoff_series_l1.append(
                    {
                        "label": f"{label} ({variant_label})",
                        "color": color,
                        "linestyle": linestyle,
                        "x": [row["k"] for row in subset],
                        "y": [row["l1_error_mean"] for row in subset],
                    }
                )
                tradeoff_series_l2.append(
                    {
                        "label": f"{label} ({variant_label})",
                        "color": color,
                        "linestyle": linestyle,
                        "x": [row["k"] for row in subset],
                        "y": [row["l2_mse_mean"] for row in subset],
                    }
                )

    cosine_rows = build_variant_overlay_rows(summary_rows, "cosine_similarity_mean", "cosine_similarity_std")
    relative_rows = build_variant_overlay_rows(summary_rows, "relative_l2_error_mean", "relative_l2_error_std")

    width = 1800
    height = 2200
    image = Image.new("RGB", (width, height), color=colors["bg"])
    draw = ImageDraw.Draw(image)
    draw.text((40, 28), "SdV Partitioning Dashboard", font=title_font, fill=colors["text"])
    meta_line = (
        f"Dataset: {dense_row['dataset']}   |   Model: {dense_row['model_name']}   |   "
        f"Feature: {dense_row['feature_scope']}   |   Source Level: L{dense_row['source_level']}   |   "
        f"Samples: {dense_row['sample_count']}"
    )
    draw.text((40, 68), meta_line, font=body_font, fill=colors["muted"])
    draw.text((40, 96), "Wide pale bars / solid lines: SdV only. Narrow dark bars / dashed lines: SdV+dSV.", font=body_font, fill=colors["muted"])

    margin = 36
    gap = 24
    header_h = 150
    half_w = (width - margin * 2 - gap) // 2
    row_h = 380
    remaining_h = height - header_h - margin * 2 - gap * 3 - row_h * 2
    row3_h = remaining_h // 2
    row4_h = remaining_h - row3_h

    rect_l1 = (margin, header_h, margin + half_w, header_h + row_h)
    rect_l2 = (margin + half_w + gap, header_h, width - margin, header_h + row_h)
    rect_tradeoff_l1 = (margin, header_h + row_h + gap, margin + half_w, header_h + row_h * 2 + gap)
    rect_tradeoff_l2 = (margin + half_w + gap, header_h + row_h + gap, width - margin, header_h + row_h * 2 + gap)
    rect_cosine = (margin, header_h + row_h * 2 + gap * 2, width - margin, header_h + row_h * 2 + gap * 2 + row3_h)
    rect_relative = (margin, rect_cosine[3] + gap, width - margin, rect_cosine[3] + gap + row4_h)

    categories = ["NoCorr", "Dense", "Prune", "1 Sink", "M Sink", "Prune+Min"]
    series_l1 = [
        {
            "label": "C=64" if group["budget"] == 64 else "C=128",
            "color": colors["blue"] if group["budget"] == 64 else colors["orange"],
            "base_color": lighten_color(colors["blue"] if group["budget"] == 64 else colors["orange"], 0.5),
            "overlay_color": colors["blue"] if group["budget"] == 64 else colors["orange"],
            "values": [pair["base"]["l1_error_mean"] for pair in group["rows"]],
            "overlay_values": [pair["overlay"]["l1_error_mean"] for pair in group["rows"]],
        }
        for group in algorithm_rows
    ]
    series_l2 = [
        {
            "label": "C=64" if group["budget"] == 64 else "C=128",
            "color": colors["blue"] if group["budget"] == 64 else colors["orange"],
            "base_color": lighten_color(colors["blue"] if group["budget"] == 64 else colors["orange"], 0.5),
            "overlay_color": colors["blue"] if group["budget"] == 64 else colors["orange"],
            "values": [pair["base"]["l2_mse_mean"] for pair in group["rows"]],
            "overlay_values": [pair["overlay"]["l2_mse_mean"] for pair in group["rows"]],
        }
        for group in algorithm_rows
    ]
    _draw_grouped_bar_panel(
        draw,
        rect_l1,
        "Algorithm Comparison: L1",
        "Grouped means by total budget",
        categories,
        series_l1,
        panel_title_font,
        small_font,
        tick_font,
        colors,
    )
    _draw_grouped_bar_panel(
        draw,
        rect_l2,
        "Algorithm Comparison: L2 (MSE)",
        "Grouped means by total budget",
        categories,
        series_l2,
        panel_title_font,
        small_font,
        tick_font,
        colors,
    )
    _draw_line_panel(
        draw,
        rect_tradeoff_l1,
        "K vs. M Trade-off: L1",
        "Solid: SdV only, dashed: SdV+dSV",
        tradeoff_series_l1,
        panel_title_font,
        small_font,
        tick_font,
        colors,
    )
    _draw_line_panel(
        draw,
        rect_tradeoff_l2,
        "K vs. M Trade-off: L2 (MSE)",
        "Solid: SdV only, dashed: SdV+dSV",
        tradeoff_series_l2,
        panel_title_font,
        small_font,
        tick_font,
        colors,
    )
    _draw_horizontal_bar_panel(
        draw,
        rect_cosine,
        "Feature Direction Preservation",
        "Wide bar: SdV only, narrow overlay: SdV+dSV",
        cosine_rows,
        "sdv_only_mean",
        "sdv_only_std",
        "sdv_plus_dsv_mean",
        "sdv_plus_dsv_std",
        panel_title_font,
        small_font,
        tick_font,
        colors,
    )
    _draw_horizontal_bar_panel(
        draw,
        rect_relative,
        "Relative Feature Error",
        "Wide bar: SdV only, narrow overlay: SdV+dSV",
        relative_rows,
        "sdv_only_mean",
        "sdv_only_std",
        "sdv_plus_dsv_mean",
        "sdv_plus_dsv_std",
        panel_title_font,
        small_font,
        tick_font,
        colors,
    )
    image.save(output_path, format="PNG")


def resize_longest_side_and_pad(image: Image.Image, image_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    width, height = image.size
    scale = image_size / max(height, width)
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))

    resampling = Image.Resampling if hasattr(Image, "Resampling") else Image
    resized = image.resize((resized_width, resized_height), resample=resampling.BILINEAR)
    canvas = Image.new("RGB", (image_size, image_size), color=(0, 0, 0))

    pad_left = (image_size - resized_width) // 2
    pad_top = (image_size - resized_height) // 2
    canvas.paste(resized, (pad_left, pad_top))

    image_tensor = torch.from_numpy(np.asarray(canvas, dtype=np.uint8).copy()).permute(2, 0, 1).contiguous()

    valid_mask = torch.zeros((image_size, image_size), dtype=torch.bool)
    valid_mask[pad_top:pad_top + resized_height, pad_left:pad_left + resized_width] = True
    return image_tensor, valid_mask


class ImageNetAnalysisDataset(Dataset):
    def __init__(self, root: str, image_size: int):
        self.root = Path(root).expanduser()
        self.image_size = image_size
        if not self.root.exists():
            raise FileNotFoundError(f"ImageNet root not found: {self.root}")

        classes = sorted(path.name for path in self.root.iterdir() if path.is_dir())
        class_to_idx = {name: idx for idx, name in enumerate(classes)}
        samples = []
        for class_name in classes:
            class_dir = self.root / class_name
            for path in sorted(class_dir.iterdir()):
                if path.is_file():
                    samples.append((path, class_to_idx[class_name]))
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        image_tensor, valid_mask = resize_longest_side_and_pad(image, self.image_size)
        return image_tensor, {
            "sample_key": f"{idx:06d}:{path.name}",
            "path": str(path),
            "label": int(label),
            "valid_mask": valid_mask,
        }


class COCOAnalysisDataset(Dataset):
    def __init__(self, image_size: int):
        import fiftyone.zoo as foz

        self.image_size = image_size
        dataset = foz.load_zoo_dataset("coco-2017", split="validation")
        self.filepaths = dataset.values("filepath")

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx: int):
        path = Path(self.filepaths[idx])
        image = Image.open(path).convert("RGB")
        image_tensor, valid_mask = resize_longest_side_and_pad(image, self.image_size)
        return image_tensor, {
            "sample_key": f"{idx:06d}:{path.name}",
            "path": str(path),
            "label": -1,
            "valid_mask": valid_mask,
        }


def collate_analysis_batch(batch):
    images = torch.stack([item[0] for item in batch], dim=0)
    metadata = [item[1] for item in batch]
    return {
        "images": images,
        "valid_masks": torch.stack([item["valid_mask"] for item in metadata], dim=0),
        "sample_keys": [item["sample_key"] for item in metadata],
        "paths": [item["path"] for item in metadata],
        "labels": [item["label"] for item in metadata],
    }


def build_analysis_loader(
    dataset_name: str,
    data_root: str,
    batch_size: int,
    image_size: int,
    num_workers: int,
) -> DataLoader:
    normalized = normalize_dataset_name(dataset_name)
    if normalized == "imagenet-1k":
        dataset = ImageNetAnalysisDataset(root=data_root, image_size=image_size)
    else:
        dataset = COCOAnalysisDataset(image_size=image_size)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_analysis_batch,
    )


def make_pyramid_source(images_bchw_uint8: torch.Tensor, level: int) -> torch.Tensor:
    if level <= 0:
        return images_bchw_uint8.clone()

    images_bhwc = images_bchw_uint8.permute(0, 2, 3, 1).contiguous().cpu().numpy()
    outputs = []
    for image in images_bhwc:
        current = image
        sizes = []
        for _ in range(level):
            sizes.append((current.shape[1], current.shape[0]))
            current = cv2.pyrDown(current)
        for target_w, target_h in reversed(sizes):
            current = cv2.pyrUp(current, dstsize=(target_w, target_h))
            current = current.astype(image.dtype, copy=False)
        outputs.append(current)

    source = torch.from_numpy(np.stack(outputs, axis=0)).permute(0, 3, 1, 2).contiguous()
    return source.to(dtype=images_bchw_uint8.dtype)


def build_valid_token_mask(
    pixel_valid_masks: torch.Tensor,
    patch_start: int,
    patch_size: int,
    feature_scope: str,
) -> torch.Tensor:
    batch, height, width = pixel_valid_masks.shape
    patch_mask = pixel_valid_masks.reshape(
        batch,
        height // patch_size,
        patch_size,
        width // patch_size,
        patch_size,
    ).any(dim=2).any(dim=3).reshape(batch, -1)

    if feature_scope == "cls_token":
        return torch.ones((batch, 1), dtype=torch.bool, device=pixel_valid_masks.device)
    if feature_scope == "patch_tokens":
        return patch_mask
    pretoken_mask = torch.ones((batch, patch_start), dtype=torch.bool, device=pixel_valid_masks.device)
    return torch.cat([pretoken_mask, patch_mask], dim=1)


def feature_tokens_from_state(backbone, x: torch.Tensor) -> torch.Tensor:
    patch_start = 1 + backbone.n_storage_tokens
    if backbone.untie_cls_and_patch_norms or backbone.untie_global_and_local_cls_norm:
        if backbone.untie_cls_and_patch_norms:
            cls_tokens = backbone.cls_norm(x[:, :patch_start])
        else:
            cls_tokens = backbone.norm(x[:, :patch_start])
        patch_tokens = backbone.norm(x[:, patch_start:])
        return torch.cat([cls_tokens, patch_tokens], dim=1)
    return backbone.norm(x)


def select_feature_tensor(tokens: torch.Tensor, token_mask: torch.Tensor, feature_scope: str, patch_start: int) -> torch.Tensor:
    if feature_scope == "cls_token":
        return tokens[:, :1]
    if feature_scope == "patch_tokens":
        return tokens[patch_start:][token_mask].reshape(1, -1)
    return tokens[token_mask].reshape(1, -1)


def compute_feature_metrics(
    ref_vec: torch.Tensor,
    cand_vec: torch.Tensor,
    chunk_size: int,
) -> tuple[float, float, float, float]:
    ref_flat = ref_vec.reshape(-1)
    cand_flat = cand_vec.reshape(-1)
    if ref_flat.numel() != cand_flat.numel():
        raise ValueError(f"Mismatched feature shapes: {ref_flat.shape} vs {cand_flat.shape}")

    total = ref_flat.numel()
    l1_sum = torch.zeros((), device=ref_flat.device, dtype=torch.float64)
    l2_sum = torch.zeros((), device=ref_flat.device, dtype=torch.float64)
    dot_sum = torch.zeros((), device=ref_flat.device, dtype=torch.float64)
    ref_norm_sum = torch.zeros((), device=ref_flat.device, dtype=torch.float64)
    cand_norm_sum = torch.zeros((), device=ref_flat.device, dtype=torch.float64)

    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        ref_chunk = ref_flat[start:end].float()
        cand_chunk = cand_flat[start:end].float()
        diff = cand_chunk - ref_chunk
        l1_sum += diff.abs().sum(dtype=torch.float64)
        l2_sum += diff.square().sum(dtype=torch.float64)
        dot_sum += (ref_chunk * cand_chunk).sum(dtype=torch.float64)
        ref_norm_sum += ref_chunk.square().sum(dtype=torch.float64)
        cand_norm_sum += cand_chunk.square().sum(dtype=torch.float64)

    error_l2_norm = torch.sqrt(l2_sum)
    ref_l2_norm = torch.sqrt(ref_norm_sum)
    l1_error = float((l1_sum / total).item())
    l2_mse = float((l2_sum / total).item())
    relative_l2_error = float((error_l2_norm / (ref_l2_norm + 1e-12)).item())
    cosine = float((dot_sum / (ref_l2_norm * torch.sqrt(cand_norm_sum) + 1e-12)).item())
    return l1_error, l2_mse, relative_l2_error, cosine


def iter_query_ranges(length: int, chunk_size: int) -> Iterable[tuple[int, int]]:
    for start in range(0, length, chunk_size):
        yield start, min(start + chunk_size, length)


def compute_qkv(block, x: torch.Tensor, rope):
    x_norm = block.norm1(x)
    qkv = block.attn.qkv(x_norm)
    batch, num_tokens, _ = qkv.shape
    embed_dim = block.attn.qkv.in_features
    head_dim = embed_dim // block.attn.num_heads
    qkv = qkv.reshape(batch, num_tokens, 3, block.attn.num_heads, head_dim)
    q, k, v = torch.unbind(qkv, dim=2)
    q, k, v = [tensor.transpose(1, 2).contiguous() for tensor in (q, k, v)]
    if rope is not None:
        q, k = block.attn.apply_rope(q, k, rope)
    return q, k, v


def project_attention_heads(attn_module, heads: torch.Tensor) -> torch.Tensor:
    batch, num_heads, num_tokens, head_dim = heads.shape
    tokens = heads.transpose(1, 2).reshape(batch, num_tokens, num_heads * head_dim)
    tokens = tokens.to(dtype=attn_module.proj.weight.dtype)
    tokens = attn_module.proj(tokens)
    return attn_module.proj_drop(tokens)


def compute_column_scores(q: torch.Tensor, k: torch.Tensor, scale: float, query_chunk_size: int) -> torch.Tensor:
    batch, num_heads, num_tokens, _ = q.shape
    k_t = k.transpose(-2, -1).float()
    scores = torch.zeros((batch, num_heads, num_tokens), device=q.device, dtype=torch.float32)
    for start, end in iter_query_ranges(num_tokens, query_chunk_size):
        q_chunk = q[:, :, start:end, :].float()
        logits = torch.matmul(q_chunk, k_t) * scale
        probs = logits.softmax(dim=-1)
        scores += probs.sum(dim=-2)
        del q_chunk
        del logits
        del probs
    return scores


def build_partition_plan(scores: torch.Tensor, k_budget: int, m_budget: int, tail_prune_ratio: float = 0.0):
    num_tokens = scores.shape[-1]
    sorted_idx = scores.argsort(dim=-1, descending=True)
    k_eff = min(max(int(k_budget), 0), num_tokens)
    top_idx = sorted_idx[..., :k_eff]
    tail_idx_full = sorted_idx[..., k_eff:]
    tail_scores_full = scores.gather(-1, tail_idx_full) if tail_idx_full.numel() > 0 else None
    tail_len = tail_idx_full.shape[-1]
    if tail_len > 0 and tail_prune_ratio > 0.0:
        tail_keep = int(math.ceil(tail_len * max(0.0, 1.0 - float(tail_prune_ratio))))
        if m_budget > 0:
            tail_keep = max(tail_keep, 1)
        tail_keep = min(tail_keep, tail_len)
        tail_idx = tail_idx_full[..., :tail_keep]
        tail_scores = tail_scores_full[..., :tail_keep]
        pruned_idx = tail_idx_full[..., tail_keep:]
    else:
        tail_idx = tail_idx_full
        tail_scores = tail_scores_full
        pruned_idx = None
    if m_budget <= 0 or tail_idx.shape[-1] == 0:
        return {
            "top_idx": top_idx,
            "tail_idx": None,
            "pruned_idx": pruned_idx,
            "bin_ids": None,
            "num_bins": 0,
        }

    cumulative = tail_scores.cumsum(dim=-1)
    num_bins = max(int(m_budget), 1)
    if num_bins == 1:
        bin_ids = torch.zeros_like(tail_idx)
    else:
        boundaries = (
            tail_scores.sum(dim=-1, keepdim=True)
            * torch.arange(1, num_bins, device=scores.device, dtype=scores.dtype).view(1, 1, -1)
            / float(num_bins)
        )
        bin_ids = (cumulative.unsqueeze(-1) > boundaries.unsqueeze(-2)).sum(dim=-1)

    return {
        "top_idx": top_idx,
        "tail_idx": tail_idx,
        "pruned_idx": pruned_idx,
        "bin_ids": bin_ids,
        "num_bins": num_bins,
    }


def aggregate_probs_by_bins(
    probs_tail: torch.Tensor,
    sink_bin_ids: torch.Tensor,
    num_sink_bins: int,
    reduce_mode: str,
) -> torch.Tensor:
    if num_sink_bins <= 0:
        raise ValueError(f"num_sink_bins must be positive, got {num_sink_bins}")
    if reduce_mode == "sum":
        if num_sink_bins == 1:
            return probs_tail.sum(dim=-1, keepdim=True)
        probs_sink = torch.zeros(
            (*probs_tail.shape[:-1], num_sink_bins),
            device=probs_tail.device,
            dtype=probs_tail.dtype,
        )
        probs_sink.scatter_add_(
            -1,
            sink_bin_ids.unsqueeze(-2).expand(-1, -1, probs_tail.shape[-2], -1),
            probs_tail,
        )
        return probs_sink
    if reduce_mode == "min":
        if num_sink_bins == 1:
            return probs_tail.amin(dim=-1, keepdim=True)
        probs_sink = torch.full(
            (*probs_tail.shape[:-1], num_sink_bins),
            fill_value=float("inf"),
            device=probs_tail.device,
            dtype=probs_tail.dtype,
        )
        probs_sink.scatter_reduce_(
            -1,
            sink_bin_ids.unsqueeze(-2).expand(-1, -1, probs_tail.shape[-2], -1),
            probs_tail,
            reduce="amin",
            include_self=True,
        )
        return torch.where(torch.isinf(probs_sink), torch.zeros_like(probs_sink), probs_sink)
    raise ValueError(f"Unsupported reduce_mode: {reduce_mode}")


def apply_partitioned_sdv_(
    approx_heads: torch.Tensor,
    q_exact: torch.Tensor,
    k_exact: torch.Tensor,
    q_corr: torch.Tensor,
    k_corr: torch.Tensor,
    v_corr: torch.Tensor,
    d_v: torch.Tensor,
    condition: ExperimentCondition,
    query_chunk_size: int,
    scale: float,
    apply_dsv: bool,
) -> None:
    batch, num_heads, num_tokens, head_dim = d_v.shape
    k_exact_t = k_exact.transpose(-2, -1).float()
    k_corr_t = k_corr.transpose(-2, -1).float()
    v_corr_float = v_corr.float()
    d_v_float = d_v.float()
    is_conservative_msink = condition.family == "topk_conservative_msink"
    sink_reduce_mode = "min" if is_conservative_msink else "sum"

    if condition.family == "dense":
        plan = None
        top_dv = None
        sink_dv = None
        sink_idx = None
        sink_bin_ids = None
        num_sink_bins = 0
    else:
        scores = compute_column_scores(q_corr, k_corr, scale, query_chunk_size)
        plan = build_partition_plan(
            scores,
            condition.k or 0,
            condition.m or 0,
            tail_prune_ratio=condition.tail_prune_ratio or 0.0,
        )
        top_idx = plan["top_idx"]
        top_dv = None
        if top_idx.numel() > 0:
            top_dv = d_v_float.gather(-2, top_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim))

        sink_idx = plan["tail_idx"]
        sink_bin_ids = plan["bin_ids"]
        num_sink_bins = int(plan["num_bins"])
        sink_dv = None
        if sink_idx is not None and sink_idx.numel() > 0 and num_sink_bins > 0:
            sink_dv_tokens = d_v_float.gather(-2, sink_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim))
            sink_dv = torch.zeros((batch, num_heads, num_sink_bins, head_dim), device=d_v.device, dtype=torch.float32)
            sink_dv.scatter_add_(
                -2,
                sink_bin_ids.unsqueeze(-1).expand(-1, -1, -1, head_dim),
                sink_dv_tokens,
            )
            sink_counts = torch.zeros((batch, num_heads, num_sink_bins), device=d_v.device, dtype=torch.float32)
            sink_counts.scatter_add_(
                -1,
                sink_bin_ids,
                torch.ones_like(sink_bin_ids, dtype=torch.float32),
            )
            sink_dv = sink_dv / sink_counts.clamp_min(1.0).unsqueeze(-1)

        del scores

    for start, end in iter_query_ranges(num_tokens, query_chunk_size):
        q_corr_chunk = q_corr[:, :, start:end, :].float()
        logits_corr = torch.matmul(q_corr_chunk, k_corr_t) * scale
        probs_corr = logits_corr.softmax(dim=-1)

        if condition.family == "dense":
            corr_chunk = torch.matmul(probs_corr, d_v_float)
        else:
            corr_chunk = torch.zeros((batch, num_heads, end - start, head_dim), device=d_v.device, dtype=torch.float32)

            if top_dv is not None and top_dv.shape[-2] > 0:
                top_idx = plan["top_idx"]
                probs_top = probs_corr.gather(-1, top_idx.unsqueeze(-2).expand(-1, -1, end - start, -1))
                corr_chunk += torch.matmul(probs_top, top_dv)
                del probs_top

            if sink_dv is not None and sink_idx is not None and sink_idx.shape[-1] > 0:
                probs_tail = probs_corr.gather(-1, sink_idx.unsqueeze(-2).expand(-1, -1, end - start, -1))
                probs_sink = aggregate_probs_by_bins(
                    probs_tail=probs_tail,
                    sink_bin_ids=sink_bin_ids,
                    num_sink_bins=num_sink_bins,
                    reduce_mode=sink_reduce_mode,
                )
                corr_chunk += torch.matmul(probs_sink, sink_dv)
                del probs_sink
                del probs_tail

        if apply_dsv:
            q_exact_chunk = q_exact[:, :, start:end, :].float()
            logits_exact = torch.matmul(q_exact_chunk, k_exact_t) * scale
            probs_exact = logits_exact.softmax(dim=-1)
            d_probs = probs_exact - probs_corr
            corr_chunk += torch.matmul(d_probs, v_corr_float)
            del q_exact_chunk
            del logits_exact
            del probs_exact
            del d_probs

        approx_heads[:, :, start:end, :] += corr_chunk.to(dtype=approx_heads.dtype)
        del q_corr_chunk
        del logits_corr
        del probs_corr
        del corr_chunk

    # del k_t
    del d_v_float
    del top_dv
    del sink_dv


class SdVPartitionExperiment:
    def __init__(self, args):
        self.args = args
        self.dataset_name = normalize_dataset_name(args.dataset)
        self.image_size = args.image_size or DEFAULT_IMAGE_SIZES[self.dataset_name]
        self.device = resolve_device(args.device)
        self.autocast_dtype = MODEL_COMPUTE_DTYPES[args.autocast_dtype]
        self.output_dir = resolve_output_dir("sdv_partition", args.out_dir)
        self.model_name = resolve_model_name(args.model)
        self.weights_path = resolve_weights_path(self.model_name, args.weights_path)
        self.conditions = build_conditions(args.conservative_tail_prune_ratio)
        self.data_root = args.data_root or default_data_root_for_dataset(self.dataset_name)
        self.loader = build_analysis_loader(
            dataset_name=self.dataset_name,
            data_root=self.data_root,
            batch_size=args.batch_size,
            image_size=self.image_size,
            num_workers=args.num_workers,
        )
        self.model = self._load_backbone()
        self.patch_start = 1 + self.model.n_storage_tokens
        self.patch_size = int(self.model.patch_size)
        self.norm_mean = IMAGE_MEAN.to(self.device)
        self.norm_std = IMAGE_STD.to(self.device)

    def _load_backbone(self):
        builder = getattr(dinov3_backbones, self.model_name)
        model = builder(pretrained=False)
        compute_dtype = torch.float32 if self.device.type == "cpu" else self.autocast_dtype
        model.to(device=self.device, dtype=compute_dtype)
        state_dict = load_weight_mmap(str(self.weights_path))
        model.load_state_dict(state_dict, strict=True)
        del state_dict
        model.eval()
        return model

    def _autocast_context(self):
        if self.device.type != "cuda" or self.autocast_dtype == torch.float32:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.autocast_dtype)

    def _normalize_inputs(self, images_uint8: torch.Tensor) -> torch.Tensor:
        tensor = images_uint8.to(device=self.device, non_blocking=True).float() / 255.0
        return (tensor - self.norm_mean) / self.norm_std

    def _prepare_batch_inputs(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        images = batch["images"]
        source_uint8 = make_pyramid_source(images, self.args.source_level)
        full_input = self._normalize_inputs(images)
        source_input = self._normalize_inputs(source_uint8)
        valid_masks = batch["valid_masks"].to(device=self.device, non_blocking=True)
        del source_uint8
        return full_input, source_input, valid_masks

    def _forward_condition(
        self,
        full_input: torch.Tensor,
        source_input: torch.Tensor,
        condition: ExperimentCondition,
        apply_dsv: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        with self._autocast_context():
            x_exact, hw_exact = self.model.prepare_tokens_with_masks(full_input, None)
            x_corr, hw_corr = self.model.prepare_tokens_with_masks(source_input, None)
            if hw_exact != hw_corr:
                raise ValueError(f"Token grid mismatch: {hw_exact} vs {hw_corr}")
            rope = self.model.rope_embed(H=hw_exact[0], W=hw_exact[1]) if self.model.rope_embed is not None else None

            for block in self.model.blocks:
                q_exact, k_exact, v_exact = compute_qkv(block, x_exact, rope)
                attn_exact_heads = F.scaled_dot_product_attention(q_exact, k_exact, v_exact)
                x_exact_attn = x_exact + block.ls1(project_attention_heads(block.attn, attn_exact_heads))
                del attn_exact_heads

                q_corr, k_corr, v_corr = compute_qkv(block, x_corr, rope)
                attn_corr_heads = F.scaled_dot_product_attention(q_corr, k_corr, v_corr)
                d_v = None
                if condition.family != "no_correction":
                    d_v = v_exact.float() - v_corr.float()
                    apply_partitioned_sdv_(
                        approx_heads=attn_corr_heads,
                        q_exact=q_exact,
                        k_exact=k_exact,
                        q_corr=q_corr,
                        k_corr=k_corr,
                        v_corr=v_corr,
                        d_v=d_v,
                        condition=condition,
                        query_chunk_size=self.args.query_chunk_size,
                        scale=float(block.attn.scale),
                        apply_dsv=apply_dsv,
                    )
                x_corr_attn = x_corr + block.ls1(project_attention_heads(block.attn, attn_corr_heads))

                del q_exact
                del k_exact
                del q_corr
                del k_corr
                del v_exact
                del v_corr
                if d_v is not None:
                    del d_v
                del attn_corr_heads

                x_exact = x_exact_attn + block.ls2(block.mlp(block.norm2(x_exact_attn)))
                x_corr = x_corr_attn + block.ls2(block.mlp(block.norm2(x_corr_attn)))

                del x_exact_attn
                del x_corr_attn

        return x_exact, x_corr

    @torch.no_grad()
    def evaluate_condition(
        self,
        batch_idx: int,
        batch: dict,
        full_input: torch.Tensor,
        source_input: torch.Tensor,
        valid_masks: torch.Tensor,
        condition: ExperimentCondition,
        correction_variant: str,
        apply_dsv: bool,
    ) -> list[dict]:
        x_exact, x_corr = self._forward_condition(full_input, source_input, condition, apply_dsv=apply_dsv)
        exact_tokens = feature_tokens_from_state(self.model, x_exact)
        corr_tokens = feature_tokens_from_state(self.model, x_corr)

        token_mask = build_valid_token_mask(
            pixel_valid_masks=valid_masks,
            patch_start=self.patch_start,
            patch_size=self.patch_size,
            feature_scope=self.args.feature_scope,
        )

        rows = []
        for sample_offset, sample_key in enumerate(batch["sample_keys"]):
            sample_mask = token_mask[sample_offset]
            if self.args.feature_scope == "cls_token":
                ref_vec = exact_tokens[sample_offset : sample_offset + 1, :1]
                cand_vec = corr_tokens[sample_offset : sample_offset + 1, :1]
            else:
                ref_vec = select_feature_tensor(
                    exact_tokens[sample_offset],
                    sample_mask,
                    self.args.feature_scope,
                    self.patch_start,
                )
                cand_vec = select_feature_tensor(
                    corr_tokens[sample_offset],
                    sample_mask,
                    self.args.feature_scope,
                    self.patch_start,
                )

            l1_error, l2_mse, relative_l2_error, cosine = compute_feature_metrics(
                ref_vec=ref_vec,
                cand_vec=cand_vec,
                chunk_size=self.args.metric_chunk_size,
            )
            rows.append(
                {
                    "dataset": self.dataset_name,
                    "model_name": self.model_name,
                    "feature_scope": self.args.feature_scope,
                    "source_level": self.args.source_level,
                    "correction_variant": correction_variant,
                    "batch_idx": batch_idx,
                    "sample_key": sample_key,
                    "condition_label": condition.label,
                    "family": condition.family,
                    "budget": condition.budget,
                    "k": condition.k,
                    "m": condition.m,
                    "tail_prune_ratio": condition.tail_prune_ratio,
                    "l1_error": l1_error,
                    "l2_mse": l2_mse,
                    "relative_l2_error": relative_l2_error,
                    "cosine_similarity": cosine,
                }
            )

        del x_exact
        del x_corr
        del exact_tokens
        del corr_tokens
        del token_mask
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        return rows

    def run(self):
        raw_rows = []
        total_batches = min(len(self.loader), self.args.max_batches)

        for batch_idx, batch in enumerate(self.loader):
            if batch_idx >= self.args.max_batches:
                break
            print(f"[Batch {batch_idx + 1}/{total_batches}] Preparing inputs...")
            full_input, source_input, valid_masks = self._prepare_batch_inputs(batch)

            for condition in self.conditions:
                for correction_variant, apply_dsv, variant_label in CORRECTION_VARIANTS:
                    print(f"[Batch {batch_idx + 1}/{total_batches}] {condition.label} [{variant_label}]")
                    raw_rows.extend(
                        self.evaluate_condition(
                            batch_idx=batch_idx,
                            batch=batch,
                            full_input=full_input,
                            source_input=source_input,
                            valid_masks=valid_masks,
                            condition=condition,
                            correction_variant=correction_variant,
                            apply_dsv=apply_dsv,
                        )
                    )

            del full_input
            del source_input
            del valid_masks
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        if not raw_rows:
            raise RuntimeError("No results were produced.")

        summary_rows = summarize_rows(raw_rows)
        self._save_outputs(raw_rows, summary_rows)
        print("\n**Summary**")
        if not maybe_print_pandas_table(summary_rows):
            columns = [
                "condition_label",
                "family",
                "budget",
                "k",
                "m",
                "correction_variant",
                "sample_count",
                "l1_error_mean",
                "l2_mse_mean",
                "relative_l2_error_mean",
                "cosine_similarity_mean",
            ]
            print(format_table(summary_rows, columns))

    def _save_outputs(self, raw_rows: list[dict], summary_rows: list[dict]) -> None:
        raw_path = self.output_dir / "results_raw.csv"
        summary_path = self.output_dir / "results.csv"
        write_rows_csv(raw_path, raw_rows)
        write_rows_csv(summary_path, summary_rows)

        config_payload = {
            "dataset": self.dataset_name,
            "data_root": self.data_root,
            "model_name": self.model_name,
            "weights_path": str(self.weights_path),
            "batch_size": self.args.batch_size,
            "max_batches": self.args.max_batches,
            "image_size": self.image_size,
            "source_level": self.args.source_level,
            "feature_scope": self.args.feature_scope,
            "query_chunk_size": self.args.query_chunk_size,
            "metric_chunk_size": self.args.metric_chunk_size,
            "conservative_tail_prune_ratio": self.args.conservative_tail_prune_ratio,
            "skip_dashboard": self.args.skip_dashboard,
            "correction_variants": [name for name, _, _ in CORRECTION_VARIANTS],
            "device": str(self.device),
            "autocast_dtype": self.args.autocast_dtype,
        }
        write_json(self.output_dir / "config.json", config_payload)
        if not self.args.skip_dashboard:
            try:
                write_dashboard_png(summary_rows, self.output_dir / "dashboard.png")
            except Exception as exc:
                print(f"[Warning] dashboard rendering failed, but CSV results were saved: {exc}")

    def _plot_algorithm_comparison(self, summary_rows: list[dict]) -> None:
        plot_rows = []
        dense_row = next(row for row in summary_rows if row["family"] == "dense")
        for budget in (64, 128):
            plot_rows.append(
                {
                    "budget": budget,
                    "algorithm": "Dense",
                    "l1_error_mean": dense_row["l1_error_mean"],
                    "l2_mse_mean": dense_row["l2_mse_mean"],
                }
            )

            prune_row = next(row for row in summary_rows if row["family"] == "topk_prune" and row["budget"] == budget)
            sink1_row = next(row for row in summary_rows if row["family"] == "topk_1sink" and row["budget"] == budget)
            msink_rows = [row for row in summary_rows if row["family"] == "topk_msink" and row["budget"] == budget]
            best_msink = min(msink_rows, key=lambda row: row["l2_mse_mean"])

            plot_rows.extend(
                [
                    {
                        "budget": budget,
                        "algorithm": "Top-K Prune",
                        "l1_error_mean": prune_row["l1_error_mean"],
                        "l2_mse_mean": prune_row["l2_mse_mean"],
                    },
                    {
                        "budget": budget,
                        "algorithm": "Top-K + 1 Sink",
                        "l1_error_mean": sink1_row["l1_error_mean"],
                        "l2_mse_mean": sink1_row["l2_mse_mean"],
                    },
                    {
                        "budget": budget,
                        "algorithm": "Top-K + M Sink",
                        "l1_error_mean": best_msink["l1_error_mean"],
                        "l2_mse_mean": best_msink["l2_mse_mean"],
                    },
                ]
            )

        try:
            import plotly.graph_objects as go  # type: ignore
            from plotly.subplots import make_subplots  # type: ignore
        except Exception as exc:
            print(f"[Warning] plotly import failed, skipping algorithm comparison plot: {exc}")
            return

        budgets = [64, 128]
        algorithms = ["Dense", "Top-K Prune", "Top-K + 1 Sink", "Top-K + M Sink"]
        colors = {64: "#1f77b4", 128: "#ff7f0e"}
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Mean L1 Error", "Mean L2 Error (MSE)"))

        for col_idx, metric in enumerate(("l1_error_mean", "l2_mse_mean"), start=1):
            for budget in budgets:
                subset = [row for row in plot_rows if row["budget"] == budget]
                subset_by_name = {row["algorithm"]: row[metric] for row in subset}
                fig.add_trace(
                    go.Bar(
                        x=algorithms,
                        y=[subset_by_name[name] for name in algorithms],
                        name=f"C={budget}",
                        marker_color=colors[budget],
                        offsetgroup=str(budget),
                        showlegend=(col_idx == 1),
                    ),
                    row=1,
                    col=col_idx,
                )

        fig.update_layout(
            title="Algorithm Comparison by Budget",
            barmode="group",
            template="plotly_white",
            width=1200,
            height=500,
        )
        fig.write_html(self.output_dir / "plot_algorithm_comparison.html", include_plotlyjs="cdn")

    def _plot_tradeoff(self, summary_rows: list[dict]) -> None:
        tradeoff_rows = [row for row in summary_rows if row["family"] in {"topk_1sink", "topk_msink"}]
        if not tradeoff_rows:
            return

        try:
            import plotly.graph_objects as go  # type: ignore
            from plotly.subplots import make_subplots  # type: ignore
        except Exception as exc:
            print(f"[Warning] plotly import failed, skipping trade-off plot: {exc}")
            return

        metrics = [
            ("l1_error_mean", "Mean L1 Error"),
            ("l2_mse_mean", "Mean L2 Error (MSE)"),
        ]
        styles = {
            ("topk_1sink", 64): ("#1f77b4", "solid"),
            ("topk_1sink", 128): ("#1f77b4", "dash"),
            ("topk_msink", 64): ("#ff7f0e", "solid"),
            ("topk_msink", 128): ("#ff7f0e", "dash"),
        }

        fig = make_subplots(rows=1, cols=2, subplot_titles=[title for _, title in metrics])
        for col_idx, (metric, _) in enumerate(metrics, start=1):
            for budget in (64, 128):
                for family in ("topk_1sink", "topk_msink"):
                    subset = sorted(
                        [row for row in tradeoff_rows if row["budget"] == budget and row["family"] == family],
                        key=lambda row: row["k"],
                    )
                    if not subset:
                        continue
                    color, dash = styles[(family, budget)]
                    fig.add_trace(
                        go.Scatter(
                            x=[row["k"] for row in subset],
                            y=[row[metric] for row in subset],
                            mode="lines+markers",
                            name=f"{family} (C={budget})",
                            line=dict(color=color, dash=dash),
                            showlegend=(col_idx == 1),
                        ),
                        row=1,
                        col=col_idx,
                    )

        fig.update_xaxes(title_text="K (lossless tokens)")
        fig.update_layout(
            title="K vs. M Trade-off",
            template="plotly_white",
            width=1200,
            height=500,
        )
        fig.write_html(self.output_dir / "plot_km_tradeoff.html", include_plotlyjs="cdn")

    def _plot_cosine(self, summary_rows: list[dict]) -> None:
        plot_rows = []
        for row in summary_rows:
            plot_rows.append(
                {
                    "plot_label": "Dense" if row["family"] == "dense" else row["condition_label"],
                    "cosine_similarity_mean": row["cosine_similarity_mean"],
                    "cosine_similarity_std": row["cosine_similarity_std"],
                    "sort_budget": -1 if row["budget"] is None else row["budget"],
                    "sort_k": -1 if row["k"] is None else row["k"],
                    "sort_m": -1 if row["m"] is None else row["m"],
                }
            )
        plot_rows.sort(key=lambda row: (row["sort_budget"], row["sort_k"], row["sort_m"], row["plot_label"]))

        try:
            import plotly.graph_objects as go  # type: ignore
        except Exception as exc:
            print(f"[Warning] plotly import failed, skipping cosine plot: {exc}")
            return

        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=[row["plot_label"] for row in plot_rows],
                y=[row["cosine_similarity_mean"] for row in plot_rows],
                error_y=dict(
                    type="data",
                    array=[row["cosine_similarity_std"] for row in plot_rows],
                    visible=True,
                ),
                marker_color="#4c72b0",
            )
        )
        fig.update_layout(
            title="Feature Direction Preservation",
            xaxis_title="Condition",
            yaxis_title="Cosine Similarity",
            template="plotly_white",
            width=1400,
            height=500,
        )
        fig.write_html(self.output_dir / "plot_cosine_similarity.html", include_plotlyjs="cdn")


def main():
    args = parse_args()
    experiment = SdVPartitionExperiment(args)
    experiment.run()


if __name__ == "__main__":
    main()
