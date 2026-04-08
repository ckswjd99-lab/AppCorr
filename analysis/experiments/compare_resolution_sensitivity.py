import argparse
import csv
import importlib
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from numbers import Real
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.shared.dinov3_probe import Dinov3ResolutionProbe
from offload.common import ExperimentConfig

DEFAULT_CONFIG = "offload/config/imnet_interleaved_g4.json"
VARIANT_ORDER = [("L0", 0), ("L1", 1), ("L2", 2)]
COMPARISON_ORDER = [("L0", "L1"), ("L1", "L2"), ("L0", "L2")]
PROBE_SUPPORTED_MODEL_NAMES = {"dinov3_classifier", "dinov3_detector"}
PRIMARY_METRIC = {
    ("attn_prob_mean", "patch_to_patch"): "js_divergence",
    ("attn_block_output", "patch_tokens"): "normalized_l2_error",
    ("ffn_block_output", "patch_tokens"): "normalized_l2_error",
}
PANEL_SPECS = [
    ("attn_prob_mean", "patch_to_patch", "js_divergence", "Attention Probability", "JS divergence"),
    ("attn_block_output", "patch_tokens", "normalized_l2_error", "Attention Block Output", "Normalized L2"),
    ("ffn_block_output", "patch_tokens", "normalized_l2_error", "FFN Block Output", "Normalized L2"),
]

_TORCHVISION_NMS_STUB_LIB = None


def _purge_torchvision_modules() -> None:
    to_delete = [name for name in sys.modules if name == "torchvision" or name.startswith("torchvision.")]
    for name in to_delete:
        del sys.modules[name]


def _ensure_torchvision_nms_stub() -> None:
    global _TORCHVISION_NMS_STUB_LIB
    if _TORCHVISION_NMS_STUB_LIB is not None:
        return
    _TORCHVISION_NMS_STUB_LIB = torch.library.Library("torchvision", "DEF")
    _TORCHVISION_NMS_STUB_LIB.define("nms(Tensor dets, Tensor scores, float iou_threshold) -> Tensor")


def get_dataset_loader_factory():
    try:
        module = importlib.import_module("offload.mobile.dataset")
        return module.get_dataset_loader
    except RuntimeError as exc:
        message = str(exc)
        if "operator torchvision::nms does not exist" not in message:
            raise
        _purge_torchvision_modules()
        _ensure_torchvision_nms_stub()
        module = importlib.import_module("offload.mobile.dataset")
        return module.get_dataset_loader


def default_data_root_for_dataset(dataset_name: str) -> str:
    if dataset_name == "imagenet-1k":
        return "~/data/imagenet_val"
    return ""


def resolve_output_dir(prefix: str, out_dir: str | None) -> Path:
    if out_dir is not None:
        path = Path(out_dir).expanduser()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path("logs") / "analysis" / f"{prefix}_{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        with path.open("w", encoding="utf-8") as f:
            f.write("")
        return

    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        formatted_rows = []
        for row in rows:
            formatted = {}
            for key, value in row.items():
                if isinstance(value, Real) and not isinstance(value, bool):
                    if isinstance(value, int):
                        formatted[key] = value
                    else:
                        formatted[key] = f"{float(value):+0.6e}"
                else:
                    formatted[key] = value
            formatted_rows.append(formatted)
        writer.writerows(formatted_rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Measure how DINOv3 7B intermediate signals change across L0, L1, and L2 "
            "images built with the same preprocessing and Gaussian pyramid logic as offload. "
            "COCO detector configs are also supported; they reuse the DINOv3 classifier backbone for probing."
        )
    )
    parser.add_argument("--config", type=str, default=DEFAULT_CONFIG, help="Path to offload config JSON.")
    parser.add_argument("--dataset", type=str, default=None, help="Optional dataset override.")
    parser.add_argument("--data-root", type=str, default=None, help="Dataset root override.")
    parser.add_argument("--batch-size", type=int, default=1, help="Loader batch size. Samples are still processed one-by-one.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers.")
    parser.add_argument("--max-batches", type=int, default=None, help="Stop after this many loader batches.")
    parser.add_argument("--max-samples", type=int, default=None, help="Stop after this many images.")
    parser.add_argument("--layers", type=str, default=None, help="Comma-separated layer indices. Default: all layers.")
    parser.add_argument("--device", type=str, default=None, help="Torch device. Default: cuda if available else cpu.")
    parser.add_argument("--cache-dtype", type=str, default="fp16", help="CPU cache dtype for block outputs: fp16, bf16, or fp32.")
    parser.add_argument("--example-images", type=int, default=4, help="Number of sample rows to render in variant_examples.png.")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory.")
    return parser.parse_args()


def normalize_offload_dataset_name(name: str) -> str:
    value = str(name).strip().lower()
    aliases = {
        "imagenet": "imagenet-1k",
        "imnet": "imagenet-1k",
        "imagenet-1k": "imagenet-1k",
        "coco": "coco2017",
        "coco2017": "coco2017",
        "coco-2017": "coco2017",
    }
    if value not in aliases:
        raise ValueError(f"Unsupported dataset: {name}")
    return aliases[value]


def parse_layers(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    layers = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    return layers or None


def parse_cache_dtype(value: str) -> torch.dtype:
    normalized = value.strip().lower()
    mapping = {
        "fp16": torch.float16,
        "float16": torch.float16,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported cache dtype: {value}")
    return mapping[normalized]


def load_config(path: str) -> tuple[ExperimentConfig, dict]:
    config_path = Path(path).expanduser()
    with config_path.open("r", encoding="utf-8") as f:
        raw_config = json.load(f)
    return ExperimentConfig(**raw_config), raw_config


def determine_data_root(dataset_name: str, data_root: str | None) -> str:
    if data_root is not None:
        return str(Path(data_root).expanduser())
    if dataset_name == "imagenet-1k":
        return str(Path(default_data_root_for_dataset("imagenet-1k")).expanduser())
    return ""


def build_coco_loader_without_fiftyone(batch_size: int, image_size: int, num_workers: int):
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset
    from torchvision.transforms import v2

    class CocoFallbackDataset(Dataset):
        def __init__(self, image_size_value: int):
            base_dir = Path("~/fiftyone/coco-2017/validation/data").expanduser()
            if not base_dir.exists():
                raise FileNotFoundError(
                    "COCO fallback directory not found: ~/fiftyone/coco-2017/validation/data"
                )

            self.filepaths = sorted(
                str(path)
                for path in base_dir.iterdir()
                if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            )
            self.transform = v2.Compose([
                v2.ToImage(),
                v2.Resize((image_size_value, image_size_value), antialias=True),
                v2.ToDtype(torch.uint8, scale=False),
            ])

        def __len__(self) -> int:
            return len(self.filepaths)

        def __getitem__(self, idx: int):
            filepath = self.filepaths[idx]
            image = Image.open(filepath).convert("RGB")
            width, height = image.size
            image_tensor = self.transform(image)
            image_id = int(Path(filepath).stem)
            return image_tensor, torch.tensor([image_id, width, height], dtype=torch.long)

    dataset = CocoFallbackDataset(image_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


def build_loader_with_fallback(
    dataset_name: str,
    data_root: str,
    batch_size: int,
    image_size: int,
    loader_kwargs: dict[str, object],
):
    get_dataset_loader = get_dataset_loader_factory()
    try:
        dataset_loader = get_dataset_loader(
            dataset_name,
            data_root,
            batch_size=batch_size,
            image_size=image_size,
            **loader_kwargs,
        )
        return dataset_loader.get_loader()
    except ModuleNotFoundError as exc:
        if dataset_name != "coco2017" or exc.name != "fiftyone":
            raise
        return build_coco_loader_without_fiftyone(batch_size, image_size, int(loader_kwargs.get("num_workers", 4)))


def dataset_sample_count(dataset, batch_size: int, max_batches: int | None, max_samples: int | None) -> int:
    total = len(dataset)
    if max_batches is not None:
        total = min(total, max_batches * batch_size)
    if max_samples is not None:
        total = min(total, max_samples)
    return total


def dataset_paths(dataset, indices: list[int]) -> list[str]:
    if hasattr(dataset, "samples"):
        return [str(dataset.samples[idx][0]) for idx in indices]
    if hasattr(dataset, "filepaths"):
        return [str(dataset.filepaths[idx]) for idx in indices]
    return ["" for _ in indices]


def infer_sample_label(dataset_name: str, label_tensor: torch.Tensor | int | list[int]) -> tuple[int, int | None, int | None]:
    if dataset_name == "imagenet-1k":
        if isinstance(label_tensor, torch.Tensor):
            return int(label_tensor.item()), None, None
        return int(label_tensor), None, None

    if isinstance(label_tensor, torch.Tensor):
        values = label_tensor.tolist()
    else:
        values = list(label_tensor)
    image_id, width, height = values
    return int(image_id), int(width), int(height)


def build_gaussians(image: np.ndarray, max_level: int) -> dict[int, np.ndarray]:
    gaussians = {0: image}
    curr = image
    for level in range(1, max_level + 1):
        curr = cv2.pyrDown(curr)
        gaussians[level] = curr
    return gaussians


def iterative_upsample(image: np.ndarray, start_level: int, end_level: int, image_shape: tuple[int, int, int]) -> np.ndarray:
    if end_level > start_level:
        raise ValueError(f"end_level={end_level} cannot be greater than start_level={start_level}")

    image_h, image_w, _ = image_shape
    curr = image
    gap = start_level - end_level
    for offset in range(gap):
        next_level = start_level - 1 - offset
        target_h = image_h // (2 ** next_level)
        target_w = image_w // (2 ** next_level)
        curr = cv2.pyrUp(curr, dstsize=(target_w, target_h))
        curr = curr.astype(np.uint8)
    return curr


def make_resolution_variants(image_bhwc_uint8: np.ndarray, image_shape: tuple[int, int, int]) -> dict[str, np.ndarray]:
    gaussians = build_gaussians(image_bhwc_uint8, max_level=2)
    variants = {"L0": image_bhwc_uint8}
    for variant_name, level in VARIANT_ORDER[1:]:
        variants[variant_name] = iterative_upsample(gaussians[level], level, 0, image_shape)
    return variants


def _safe_distribution(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    arr = np.clip(arr, 0.0, None)
    total = arr.sum()
    if total <= 0:
        return np.full_like(arr, 1.0 / max(arr.size, 1))
    return arr / total


def mean_rowwise_js_divergence(ref: np.ndarray, cand: np.ndarray) -> float:
    ref_arr = np.asarray(ref, dtype=np.float32)
    cand_arr = np.asarray(cand, dtype=np.float32)
    if ref_arr.ndim == 1:
        ref_arr = ref_arr[None, :]
        cand_arr = cand_arr[None, :]

    p = np.clip(ref_arr, 0.0, None)
    q = np.clip(cand_arr, 0.0, None)
    p_sum = p.sum(axis=-1, keepdims=True)
    q_sum = q.sum(axis=-1, keepdims=True)
    uniform = 1.0 / max(p.shape[-1], 1)
    p = np.divide(p, p_sum, out=np.full_like(p, uniform), where=p_sum > 0)
    q = np.divide(q, q_sum, out=np.full_like(q, uniform), where=q_sum > 0)
    m = 0.5 * (p + q)
    eps = 1e-12
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    m = np.clip(m, eps, None)
    kl_pm = np.sum(p * np.log(p / m), axis=-1)
    kl_qm = np.sum(q * np.log(q / m), axis=-1)
    return float(np.mean(0.5 * (kl_pm + kl_qm)))


def normalized_l2_error(ref: np.ndarray, cand: np.ndarray) -> float:
    ref_arr = np.asarray(ref, dtype=np.float32)
    cand_arr = np.asarray(cand, dtype=np.float32)
    numerator = np.linalg.norm((cand_arr - ref_arr).reshape(-1))
    denominator = np.linalg.norm(ref_arr.reshape(-1))
    if denominator == 0:
        return 0.0 if numerator == 0 else float("inf")
    return float(numerator / denominator)


def l2_mean(ref: np.ndarray, cand: np.ndarray) -> float:
    diff = np.asarray(cand, dtype=np.float32) - np.asarray(ref, dtype=np.float32)
    return float(np.sqrt(np.mean(diff * diff)))


def flat_cosine_similarity(ref: np.ndarray, cand: np.ndarray) -> float | None:
    x = np.asarray(ref, dtype=np.float32).reshape(-1)
    y = np.asarray(cand, dtype=np.float32).reshape(-1)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return None
    return float(np.dot(x, y) / denom)


def mean_token_cosine(ref: np.ndarray, cand: np.ndarray) -> float | None:
    x = np.asarray(ref, dtype=np.float32)
    y = np.asarray(cand, dtype=np.float32)
    if x.ndim != 2 or y.ndim != 2:
        return None
    numer = np.sum(x * y, axis=-1)
    denom = np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1)
    valid = denom > 0
    if not np.any(valid):
        return None
    values = numer[valid] / denom[valid]
    return float(np.mean(values))


def mean_row_cosine(ref: np.ndarray, cand: np.ndarray) -> float | None:
    x = np.asarray(ref, dtype=np.float32)
    y = np.asarray(cand, dtype=np.float32)
    if x.ndim != 2 or y.ndim != 2:
        return None
    numer = np.sum(x * y, axis=-1)
    denom = np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1)
    valid = denom > 0
    if not np.any(valid):
        return None
    values = numer[valid] / denom[valid]
    return float(np.mean(values))


def topk_overlap_ratio(ref: np.ndarray, cand: np.ndarray, ratio: float = 0.1) -> float:
    x = np.asarray(ref, dtype=np.float32).reshape(-1)
    y = np.asarray(cand, dtype=np.float32).reshape(-1)
    k = max(1, min(x.size, int(np.ceil(x.size * ratio))))
    x_idx = set(np.argpartition(x, -k)[-k:].tolist())
    y_idx = set(np.argpartition(y, -k)[-k:].tolist())
    return float(len(x_idx & y_idx) / k)


def attention_metrics(ref: np.ndarray, cand: np.ndarray) -> dict[str, float | None]:
    ref_arr = np.asarray(ref, dtype=np.float32)
    cand_arr = np.asarray(cand, dtype=np.float32)
    return {
        "js_divergence": mean_rowwise_js_divergence(ref_arr, cand_arr),
        "normalized_l2_error": normalized_l2_error(ref_arr, cand_arr),
        "mean_row_cosine": mean_row_cosine(ref_arr, cand_arr),
        "topk_overlap_0p1": topk_overlap_ratio(ref_arr, cand_arr, ratio=0.1),
        "l2_mean": l2_mean(ref_arr, cand_arr),
        "cosine_similarity": flat_cosine_similarity(ref_arr, cand_arr),
        "mean_token_cosine": None,
    }


def output_metrics(ref: np.ndarray, cand: np.ndarray) -> dict[str, float | None]:
    ref_arr = np.asarray(ref, dtype=np.float32)
    cand_arr = np.asarray(cand, dtype=np.float32)
    return {
        "js_divergence": None,
        "normalized_l2_error": normalized_l2_error(ref_arr, cand_arr),
        "mean_row_cosine": None,
        "topk_overlap_0p1": None,
        "l2_mean": l2_mean(ref_arr, cand_arr),
        "cosine_similarity": flat_cosine_similarity(ref_arr, cand_arr),
        "mean_token_cosine": mean_token_cosine(ref_arr, cand_arr),
    }


def compare_attention_pair(
    ref_signal: np.ndarray,
    cand_signal: np.ndarray,
    patch_start: int,
    base_row: dict[str, object],
) -> list[dict[str, object]]:
    rows = []
    scopes = {
        "all_tokens": (ref_signal, cand_signal),
        "patch_to_patch": (ref_signal[patch_start:, patch_start:], cand_signal[patch_start:, patch_start:]),
    }
    for scope, (ref_view, cand_view) in scopes.items():
        row = dict(base_row)
        row["scope"] = scope
        row.update(attention_metrics(ref_view, cand_view))
        rows.append(row)
    return rows


def compare_output_pair(
    ref_signal: np.ndarray,
    cand_signal: np.ndarray,
    patch_start: int,
    base_row: dict[str, object],
) -> list[dict[str, object]]:
    rows = []
    scopes = {
        "all_tokens": (ref_signal, cand_signal),
        "patch_tokens": (ref_signal[patch_start:], cand_signal[patch_start:]),
    }
    for scope, (ref_view, cand_view) in scopes.items():
        row = dict(base_row)
        row["scope"] = scope
        row.update(output_metrics(ref_view, cand_view))
        rows.append(row)
    return rows


def summarize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], dict[str, list[float]]] = {}
    key_fields = ("dataset", "signal_name", "scope", "comparison", "src_variant", "dst_variant", "layer")
    metric_fields = (
        "js_divergence",
        "normalized_l2_error",
        "mean_row_cosine",
        "topk_overlap_0p1",
        "l2_mean",
        "cosine_similarity",
        "mean_token_cosine",
    )

    for row in rows:
        key = tuple(row[field] for field in key_fields)
        if key not in grouped:
            grouped[key] = {field: [] for field in metric_fields}
        for field in metric_fields:
            value = row.get(field)
            if value is not None:
                grouped[key][field].append(float(value))

    summary = []
    for key in sorted(grouped):
        data = grouped[key]
        item = dict(zip(key_fields, key))
        item["num_samples"] = max((len(values) for values in data.values()), default=0)
        for field in metric_fields:
            values = data[field]
            item[field] = float(np.mean(values)) if values else None
        primary_metric = PRIMARY_METRIC.get((item["signal_name"], item["scope"]))
        item["primary_metric_name"] = primary_metric
        item["primary_metric_value"] = item.get(primary_metric) if primary_metric is not None else None
        summary.append(item)
    return summary


def summarize_global(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], dict[str, list[float]]] = {}
    key_fields = ("dataset", "signal_name", "scope", "comparison", "src_variant", "dst_variant")
    metric_fields = (
        "js_divergence",
        "normalized_l2_error",
        "mean_row_cosine",
        "topk_overlap_0p1",
        "l2_mean",
        "cosine_similarity",
        "mean_token_cosine",
        "primary_metric_value",
    )

    for row in summary_rows:
        key = tuple(row[field] for field in key_fields)
        if key not in grouped:
            grouped[key] = {field: [] for field in metric_fields}
        for field in metric_fields:
            value = row.get(field)
            if value is not None:
                grouped[key][field].append(float(value))

    rows = []
    for key in sorted(grouped):
        item = dict(zip(key_fields, key))
        for field in metric_fields:
            values = grouped[key][field]
            item[field] = float(np.mean(values)) if values else None
        item["primary_metric_name"] = PRIMARY_METRIC.get((item["signal_name"], item["scope"]))
        rows.append(item)
    return rows


def build_step_sensitivity_summary(global_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    indexed = {
        (row["signal_name"], row["scope"], row["comparison"]): row
        for row in global_rows
    }

    rows = []
    for signal_name, scope in PRIMARY_METRIC:
        earlier = indexed.get((signal_name, scope, "L0_to_L1"))
        later = indexed.get((signal_name, scope, "L1_to_L2"))
        if earlier is None or later is None:
            continue

        early_value = earlier.get("primary_metric_value")
        late_value = later.get("primary_metric_value")
        ratio = None
        difference = None
        if early_value is not None and late_value is not None:
            early_float = float(early_value)
            late_float = float(late_value)
            ratio = late_float / max(abs(early_float), 1e-12)
            difference = late_float - early_float

        rows.append({
            "dataset": earlier["dataset"],
            "signal_name": signal_name,
            "scope": scope,
            "primary_metric_name": PRIMARY_METRIC[(signal_name, scope)],
            "l0_to_l1_value": early_value,
            "l1_to_l2_value": late_value,
            "l1_to_l2_over_l0_to_l1": ratio,
            "l1_to_l2_minus_l0_to_l1": difference,
        })
    return rows


def write_visual_summary(
    out_dir: Path,
    dataset_name: str,
    processed_samples: int,
    summary_rows: list[dict[str, object]],
    global_rows: list[dict[str, object]],
    sensitivity_rows: list[dict[str, object]],
) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str((out_dir / ".mplconfig").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    comparisons = [f"{src}_to_{dst}" for src, dst in COMPARISON_ORDER]
    fig = plt.figure(figsize=(19, 15), dpi=180, constrained_layout=True)
    gs = fig.add_gridspec(3, 3, height_ratios=[1.0, 1.0, 0.75])

    fig.suptitle(
        f"Resolution Sensitivity Dashboard: {dataset_name} | samples={processed_samples}",
        fontsize=18,
        fontweight="bold",
    )

    for col_idx, (signal_name, scope, metric_name, title, colorbar_label) in enumerate(PANEL_SPECS):
        ax = fig.add_subplot(gs[0, col_idx])
        relevant = [
            row for row in summary_rows
            if row["signal_name"] == signal_name and row["scope"] == scope
        ]
        layers = sorted({int(row["layer"]) for row in relevant})
        if not layers:
            ax.axis("off")
            ax.text(0.5, 0.5, f"{title}\nno data", ha="center", va="center", fontsize=12)
            continue
        matrix = np.full((len(layers), len(comparisons)), np.nan, dtype=np.float64)
        for row in relevant:
            layer_idx = layers.index(int(row["layer"]))
            comp_idx = comparisons.index(str(row["comparison"]))
            value = row.get(metric_name)
            if value is not None:
                matrix[layer_idx, comp_idx] = float(value)

        im = ax.imshow(matrix, aspect="auto", cmap="viridis")
        finite_mask = np.isfinite(matrix)
        finite_mean = float(np.mean(matrix[finite_mask])) if np.any(finite_mask) else 0.0
        ax.set_title(f"{title}\n{colorbar_label}")
        ax.set_xticks(np.arange(len(comparisons)), comparisons, rotation=20)
        ax.set_yticks(np.arange(len(layers)), layers)
        ax.set_ylabel("Layer")
        for layer_idx, _ in enumerate(layers):
            for comp_idx, _ in enumerate(comparisons):
                value = matrix[layer_idx, comp_idx]
                if np.isfinite(value):
                    ax.text(
                        comp_idx,
                        layer_idx,
                        f"{value:.3f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if value > finite_mean else "#1f1f1f",
                    )
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(colorbar_label)

    for col_idx, (signal_name, scope, metric_name, title, colorbar_label) in enumerate(PANEL_SPECS):
        ax = fig.add_subplot(gs[1, col_idx])
        relevant = [
            row for row in summary_rows
            if row["signal_name"] == signal_name and row["scope"] == scope
        ]
        layers = sorted({int(row["layer"]) for row in relevant})
        for comparison in comparisons:
            xs = []
            ys = []
            for layer in layers:
                row = next(
                    (
                        item for item in relevant
                        if int(item["layer"]) == layer and item["comparison"] == comparison
                    ),
                    None,
                )
                if row is None or row.get(metric_name) is None:
                    continue
                xs.append(layer)
                ys.append(float(row[metric_name]))
            if xs:
                ax.plot(xs, ys, marker="o", linewidth=2.0, label=comparison)

        ax.set_title(f"{title}\nLayer profile ({colorbar_label})")
        ax.set_xlabel("Layer")
        ax.set_ylabel(colorbar_label)
        ax.grid(True, alpha=0.25)
        if layers:
            ax.set_xticks(layers[:: max(1, len(layers) // 8)] if len(layers) > 8 else layers)
        if col_idx == 0:
            ax.legend(frameon=False, fontsize=9)

    summary_ax = fig.add_subplot(gs[2, 0])
    ratio_ax = fig.add_subplot(gs[2, 1])
    notes_ax = fig.add_subplot(gs[2, 2])

    signal_labels = []
    bar_comparisons = ["L0_to_L1", "L1_to_L2", "L0_to_L2"]
    x = np.arange(len(PANEL_SPECS))
    width = 0.22
    palette = {
        "L0_to_L1": "#5B8FF9",
        "L1_to_L2": "#F6BD16",
        "L0_to_L2": "#5AD8A6",
    }

    for signal_name, scope, _, title, _ in PANEL_SPECS:
        signal_labels.append(title)

    for idx, comparison in enumerate(bar_comparisons):
        values = []
        for signal_name, scope, _, _, _ in PANEL_SPECS:
            row = next(
                (
                    item for item in global_rows
                    if item["signal_name"] == signal_name and item["scope"] == scope and item["comparison"] == comparison
                ),
                None,
            )
            values.append(float(row["primary_metric_value"]) if row and row["primary_metric_value"] is not None else np.nan)
        offset = (idx - 1) * width
        summary_ax.bar(x + offset, values, width=width, label=comparison, color=palette[comparison], edgecolor="#3a3a3a", linewidth=0.7)

    summary_ax.set_xticks(x, signal_labels, rotation=10)
    summary_ax.set_title("Average Sensitivity By Resolution Step")
    summary_ax.set_ylabel("Primary metric")
    summary_ax.grid(True, axis="y", alpha=0.25)
    summary_ax.legend(frameon=False)

    ratio_values = []
    ratio_labels = []
    ratio_colors = []
    for signal_name, scope, _, title, _ in PANEL_SPECS:
        row = next(
            (
                item for item in sensitivity_rows
                if item["signal_name"] == signal_name and item["scope"] == scope
            ),
            None,
        )
        ratio_labels.append(title)
        ratio = float(row["l1_to_l2_over_l0_to_l1"]) if row and row["l1_to_l2_over_l0_to_l1"] is not None else np.nan
        ratio_values.append(ratio)
        ratio_colors.append("#D96C75" if np.isfinite(ratio) and ratio > 1.0 else "#6C9BD2")

    ratio_ax.bar(np.arange(len(ratio_labels)), ratio_values, color=ratio_colors, edgecolor="#3a3a3a", linewidth=0.7)
    ratio_ax.axhline(1.0, color="#444444", linestyle="--", linewidth=1.0)
    ratio_ax.set_xticks(np.arange(len(ratio_labels)), ratio_labels, rotation=10)
    ratio_ax.set_title("Later-Step Sensitivity Ratio")
    ratio_ax.set_ylabel("L1->L2 / L0->L1")
    ratio_ax.grid(True, axis="y", alpha=0.25)
    for idx, value in enumerate(ratio_values):
        if np.isfinite(value):
            ratio_ax.text(idx, value, f"{value:.2f}x", ha="center", va="bottom", fontsize=9)

    notes_ax.axis("off")
    notes = []
    for signal_name, scope, _, title, metric_label in PANEL_SPECS:
        row = next(
            (
                item for item in sensitivity_rows
                if item["signal_name"] == signal_name and item["scope"] == scope
            ),
            None,
        )
        if row is None or row["l0_to_l1_value"] is None or row["l1_to_l2_value"] is None:
            notes.append(f"{title}: insufficient data")
            continue
        early_value = float(row["l0_to_l1_value"])
        late_value = float(row["l1_to_l2_value"])
        direction = "L1->L2" if late_value > early_value else "L0->L1"
        ratio = row["l1_to_l2_over_l0_to_l1"]
        if ratio is None:
            notes.append(f"{title}: {direction} is larger")
        else:
            notes.append(
                f"{title}: {direction} more sensitive "
                f"({metric_label}: {late_value:.3f} vs {early_value:.3f}, ratio={float(ratio):.2f}x)"
            )
    notes_ax.text(
        0.0,
        0.98,
        "Step-wise takeaways\n\n" + "\n\n".join(notes),
        ha="left",
        va="top",
        fontsize=11,
        color="#2d2d2d",
    )

    fig.text(
        0.5,
        0.01,
        "Top: layer heatmaps. Middle: per-layer profiles. Bottom: dataset averages, later-step ratio, and a short textual takeaway.",
        ha="center",
        fontsize=10,
        color="#666666",
    )
    fig.savefig(out_dir / "dashboard.png", bbox_inches="tight")
    plt.close(fig)


def write_variant_examples(
    out_dir: Path,
    examples: list[dict[str, object]],
) -> None:
    if not examples:
        return

    os.environ.setdefault("MPLCONFIGDIR", str((out_dir / ".mplconfig").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns = ["L0", "L1", "L2", "|L0-L1|", "|L1-L2|"]
    num_rows = len(examples)
    fig, axes = plt.subplots(
        num_rows,
        len(columns),
        figsize=(3.6 * len(columns), 3.2 * num_rows),
        dpi=180,
        constrained_layout=True,
    )
    axes = np.asarray(axes, dtype=object)
    if axes.ndim == 1:
        axes = axes[None, :]

    for row_idx, example in enumerate(examples):
        l0 = np.asarray(example["variants"]["L0"], dtype=np.uint8)
        l1 = np.asarray(example["variants"]["L1"], dtype=np.uint8)
        l2 = np.asarray(example["variants"]["L2"], dtype=np.uint8)
        diff_01 = np.mean(np.abs(l0.astype(np.float32) - l1.astype(np.float32)), axis=-1)
        diff_12 = np.mean(np.abs(l1.astype(np.float32) - l2.astype(np.float32)), axis=-1)
        render_items = [l0, l1, l2, diff_01, diff_12]

        for col_idx, item in enumerate(render_items):
            ax = axes[row_idx, col_idx]
            if item.ndim == 3:
                ax.imshow(item)
            else:
                ax.imshow(item, cmap="inferno", vmin=0.0, vmax=max(1.0, float(item.max())))
            if row_idx == 0:
                ax.set_title(columns[col_idx], fontsize=11, fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                label = f"{example['sample_key']}\nlabel={example['label_or_image_id']}"
                ax.set_ylabel(label, rotation=0, ha="right", va="center", labelpad=36, fontsize=9)

    fig.suptitle("Resolution Variants And Pixel-Space Differences", fontsize=16, fontweight="bold")
    fig.savefig(out_dir / "variant_examples.png", bbox_inches="tight")
    plt.close(fig)


def print_quick_summary(rows: list[dict[str, object]]) -> None:
    def fmt(value: object) -> str:
        if value is None:
            return "n/a"
        return f"{float(value):.6f}"

    print("[ResolutionSensitivity] Quick summary")
    for row in rows:
        ratio = row.get("l1_to_l2_over_l0_to_l1")
        if ratio is None:
            ratio_str = "n/a"
        else:
            ratio_str = f"{float(ratio):.3f}x"
        print(
            "  - "
            f"{row['signal_name']} ({row['scope']}): "
            f"L0->L1={fmt(row['l0_to_l1_value'])}, "
            f"L1->L2={fmt(row['l1_to_l2_value'])}, "
            f"L1->L2 / L0->L1={ratio_str}"
        )


def main():
    args = parse_args()
    started_at = time.time()

    config, raw_config = load_config(args.config)
    config.dataset_name = normalize_offload_dataset_name(args.dataset or config.dataset_name)
    if config.model_name not in PROBE_SUPPORTED_MODEL_NAMES:
        raise ValueError(
            "This preliminary experiment currently supports configs based on the "
            f"DINOv3 classifier or detector pipelines only, got {config.model_name}"
        )

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.image_shape[0] != config.image_shape[1]:
        raise ValueError(f"Expected square inputs, got image_shape={config.image_shape}")

    image_h, image_w, channels = map(int, config.image_shape)
    image_shape = (image_h, image_w, channels)
    data_root = determine_data_root(config.dataset_name, args.data_root)
    layers = parse_layers(args.layers)
    cache_dtype = parse_cache_dtype(args.cache_dtype)
    out_dir = resolve_output_dir("resolution_sensitivity", args.out_dir)

    loader_kwargs = dict(config.dataset_kwargs)
    loader_kwargs["num_workers"] = args.num_workers
    loader = build_loader_with_fallback(
        config.dataset_name,
        data_root,
        args.batch_size,
        image_shape[0],
        loader_kwargs,
    )

    planned_samples = dataset_sample_count(loader.dataset, args.batch_size, args.max_batches, args.max_samples)
    if planned_samples <= 0:
        raise ValueError("No samples selected. Check max-batches/max-samples.")

    probe = Dinov3ResolutionProbe(
        device=device,
        image_size=image_shape[0],
        layers=layers,
        output_cpu_dtype=cache_dtype,
    )

    per_image_rows: list[dict[str, object]] = []
    samples_rows: list[dict[str, object]] = []
    example_rows: list[dict[str, object]] = []
    processed = 0
    progress = tqdm(loader, total=len(loader), leave=False)

    for batch_idx, (images, labels) in enumerate(progress):
        if args.max_batches is not None and batch_idx >= args.max_batches:
            break
        if processed >= planned_samples:
            break

        current_batch = images.shape[0]
        remaining = planned_samples - processed
        if current_batch > remaining:
            images = images[:remaining]
            labels = labels[:remaining]
            current_batch = remaining

        if images.dtype != torch.uint8:
            images = images.to(torch.uint8)

        images_bhwc = images.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        batch_indices = list(range(processed, processed + current_batch))
        paths = dataset_paths(loader.dataset, batch_indices)

        for local_idx in range(current_batch):
            sample_index = processed + local_idx
            path = paths[local_idx]
            label_or_image_id, orig_width, orig_height = infer_sample_label(config.dataset_name, labels[local_idx])
            sample_key = f"{sample_index:06d}:{Path(path).name}" if path else f"{sample_index:06d}"
            samples_rows.append({
                "dataset": config.dataset_name,
                "sample_index": sample_index,
                "sample_key": sample_key,
                "path": path,
                "label_or_image_id": label_or_image_id,
                "orig_width": orig_width,
                "orig_height": orig_height,
            })

            image_bhwc = images_bhwc[local_idx]
            variants = make_resolution_variants(image_bhwc, image_shape)
            if len(example_rows) < max(0, args.example_images):
                example_rows.append({
                    "sample_index": sample_index,
                    "sample_key": sample_key,
                    "label_or_image_id": label_or_image_id,
                    "variants": {name: variant.copy() for name, variant in variants.items()},
                })
            variant_generators: dict[str, object] = {}
            try:
                for variant_name, _ in VARIANT_ORDER:
                    variant_tensor = torch.from_numpy(variants[variant_name]).permute(2, 0, 1).contiguous().unsqueeze(0)
                    variant_generators[variant_name] = probe.iter_layer_signals(variant_tensor)

                while True:
                    layer_outputs: dict[str, dict[str, torch.Tensor]] = {}
                    current_layer_idx: int | None = None
                    try:
                        for variant_name, _ in VARIANT_ORDER:
                            layer_idx, signals = next(variant_generators[variant_name])
                            if current_layer_idx is None:
                                current_layer_idx = int(layer_idx)
                            elif int(layer_idx) != current_layer_idx:
                                raise RuntimeError(
                                    f"Layer mismatch across variants: expected {current_layer_idx}, got {layer_idx}"
                                )
                            layer_outputs[variant_name] = signals
                    except StopIteration:
                        break

                    assert current_layer_idx is not None
                    for src_variant, dst_variant in COMPARISON_ORDER:
                        comparison_name = f"{src_variant}_to_{dst_variant}"
                        src_signals = layer_outputs[src_variant]
                        dst_signals = layer_outputs[dst_variant]
                        base_row = {
                            "dataset": config.dataset_name,
                            "sample_index": sample_index,
                            "sample_key": sample_key,
                            "path": path,
                            "label_or_image_id": label_or_image_id,
                            "orig_width": orig_width,
                            "orig_height": orig_height,
                            "comparison": comparison_name,
                            "src_variant": src_variant,
                            "dst_variant": dst_variant,
                            "layer": current_layer_idx,
                        }

                        attn_row_base = dict(base_row)
                        attn_row_base["signal_name"] = "attn_prob_mean"
                        per_image_rows.extend(compare_attention_pair(
                            src_signals["attn_prob_mean"][0].numpy(),
                            dst_signals["attn_prob_mean"][0].numpy(),
                            probe.patch_start,
                            attn_row_base,
                        ))

                        attn_block_row_base = dict(base_row)
                        attn_block_row_base["signal_name"] = "attn_block_output"
                        per_image_rows.extend(compare_output_pair(
                            src_signals["attn_block_output"][0].numpy(),
                            dst_signals["attn_block_output"][0].numpy(),
                            probe.patch_start,
                            attn_block_row_base,
                        ))

                        ffn_block_row_base = dict(base_row)
                        ffn_block_row_base["signal_name"] = "ffn_block_output"
                        per_image_rows.extend(compare_output_pair(
                            src_signals["ffn_block_output"][0].numpy(),
                            dst_signals["ffn_block_output"][0].numpy(),
                            probe.patch_start,
                            ffn_block_row_base,
                        ))

                    del layer_outputs
            finally:
                for generator in variant_generators.values():
                    close_fn = getattr(generator, "close", None)
                    if callable(close_fn):
                        close_fn()

            processed += 1
            progress.set_description(f"samples={processed}")
            if device.type == "cuda":
                torch.cuda.empty_cache()

            if args.max_samples is not None and processed >= args.max_samples:
                break

    progress.close()

    summary_rows = summarize_rows(per_image_rows)
    global_rows = summarize_global(summary_rows)
    sensitivity_rows = build_step_sensitivity_summary(global_rows)

    visualization_path = out_dir / "dashboard.png"
    examples_path = out_dir / "variant_examples.png"
    visualization_error = None
    examples_error = None
    try:
        write_visual_summary(out_dir, config.dataset_name, processed, summary_rows, global_rows, sensitivity_rows)
    except Exception as exc:
        visualization_error = f"{type(exc).__name__}: {exc}"
    try:
        write_variant_examples(out_dir, example_rows)
    except Exception as exc:
        examples_error = f"{type(exc).__name__}: {exc}"

    config_payload = {
        "config_path": str(Path(args.config).expanduser()),
        "dataset_name": config.dataset_name,
        "input_config_model_name": config.model_name,
        "probe_model_name": "dinov3_classifier",
        "data_root": data_root,
        "image_shape": list(image_shape),
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "processed_samples": processed,
        "planned_samples": planned_samples,
        "device": str(device),
        "layers": layers,
        "cache_dtype": args.cache_dtype,
        "variants": [name for name, _ in VARIANT_ORDER],
        "comparisons": [f"{src}_to_{dst}" for src, dst in COMPARISON_ORDER],
        "duration_sec": time.time() - started_at,
        "visualization_path": str(visualization_path),
        "visualization_error": visualization_error,
        "example_images": args.example_images,
        "examples_path": str(examples_path),
        "examples_error": examples_error,
        "raw_config": raw_config,
    }

    write_json(out_dir / "config.json", config_payload)
    write_csv(out_dir / "samples.csv", samples_rows)
    write_csv(out_dir / "per_image.csv", per_image_rows)
    write_csv(out_dir / "summary.csv", summary_rows)
    write_csv(out_dir / "global_summary.csv", global_rows)
    write_csv(out_dir / "step_sensitivity_summary.csv", sensitivity_rows)

    print_quick_summary(sensitivity_rows)
    print(f"[ResolutionSensitivity] Wrote config to {out_dir / 'config.json'}")
    print(f"[ResolutionSensitivity] Wrote samples to {out_dir / 'samples.csv'}")
    print(f"[ResolutionSensitivity] Wrote per-image metrics to {out_dir / 'per_image.csv'}")
    print(f"[ResolutionSensitivity] Wrote layer summary to {out_dir / 'summary.csv'}")
    print(f"[ResolutionSensitivity] Wrote global summary to {out_dir / 'global_summary.csv'}")
    print(f"[ResolutionSensitivity] Wrote step sensitivity summary to {out_dir / 'step_sensitivity_summary.csv'}")
    if visualization_error is None:
        print(f"[ResolutionSensitivity] Wrote dashboard to {visualization_path}")
    else:
        print(f"[ResolutionSensitivity] Dashboard skipped: {visualization_error}")
    if examples_error is None and example_rows:
        print(f"[ResolutionSensitivity] Wrote variant examples to {examples_path}")
    elif examples_error is not None:
        print(f"[ResolutionSensitivity] Variant examples skipped: {examples_error}")


if __name__ == "__main__":
    main()
