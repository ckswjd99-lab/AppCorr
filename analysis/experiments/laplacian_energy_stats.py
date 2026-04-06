import argparse
import csv
import importlib
import json
import math
import sys
import time
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

from offload.common import ExperimentConfig

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

DEFAULT_CONFIG = "offload/config/imnet_interleaved_g4.json"
DEFAULT_PYRAMID_LEVELS = [2, 1, 0]
RATIO_THRESHOLDS = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]


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
            "Measure Laplacian-pyramid signal energies and dx-versus-x ratios "
            "using the same preprocessing path as offload."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG,
        help="Path to an offload experiment config JSON.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Optional dataset override: imagenet, imagenet-1k, coco, or coco2017.",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Dataset root override. Ignored by the current COCO loader.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Loader batch size override. Default: config.batch_size.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="DataLoader workers.",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Stop after this many batches.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Stop after this many samples.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Square image size override. Default: config.image_shape[0].",
    )
    parser.add_argument(
        "--patch-size",
        type=str,
        default=None,
        help="Patch size override. Either '16' or '16,16'. Default: config.patch_size.",
    )
    parser.add_argument(
        "--pyramid-levels",
        type=str,
        default="2,1,0",
        help="Comma-separated pyramid levels, coarse to fine. Default: 2,1,0.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory. Default: logs/analysis/laplacian_energy_<timestamp>.",
    )
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


def parse_pyramid_levels(value: str | None) -> list[int]:
    if value is None or value.strip() == "":
        levels = list(DEFAULT_PYRAMID_LEVELS)
    else:
        levels = [int(part.strip()) for part in value.split(",") if part.strip()]

    if not levels:
        raise ValueError("At least one pyramid level is required.")
    if any(level < 0 for level in levels):
        raise ValueError(f"Invalid pyramid levels: {levels}")
    if len(set(levels)) != len(levels):
        raise ValueError(f"Duplicate pyramid levels are not allowed: {levels}")

    return sorted(levels, reverse=True)


def parse_patch_size(value: str | None, fallback: tuple[int, int]) -> tuple[int, int]:
    if value is None or value.strip() == "":
        patch_h, patch_w = fallback
    elif "," in value:
        parts = [int(part.strip()) for part in value.split(",") if part.strip()]
        if len(parts) != 2:
            raise ValueError(f"Invalid patch-size value: {value}")
        patch_h, patch_w = parts
    else:
        side = int(value.strip())
        patch_h = side
        patch_w = side

    if patch_h <= 0 or patch_w <= 0:
        raise ValueError(f"Patch size must be positive, got {(patch_h, patch_w)}")
    return patch_h, patch_w


def load_config(path: str) -> tuple[ExperimentConfig, dict]:
    config_path = Path(path).expanduser()
    with config_path.open("r", encoding="utf-8") as f:
        raw_config = json.load(f)
    return ExperimentConfig(**raw_config), raw_config


def validate_geometry(image_shape: tuple[int, int, int], patch_size: tuple[int, int], levels: list[int]) -> None:
    image_h, image_w, _ = image_shape
    patch_h, patch_w = patch_size

    if image_h <= 0 or image_w <= 0:
        raise ValueError(f"Invalid image shape: {image_shape}")

    if image_h % patch_h != 0 or image_w % patch_w != 0:
        raise ValueError(
            f"Full-resolution image shape {(image_h, image_w)} is not divisible by patch size {(patch_h, patch_w)}"
        )

    for level in levels:
        scale = 2 ** level
        if image_h % scale != 0 or image_w % scale != 0:
            raise ValueError(
                f"image_shape={image_shape[:2]} is not divisible by 2**{level}={scale}"
            )


def determine_data_root(dataset_name: str, data_root: str | None) -> str:
    if data_root is not None:
        return str(Path(data_root).expanduser())
    if dataset_name == "imagenet-1k":
        return str(Path(default_data_root_for_dataset("imagenet-1k")).expanduser())
    return ""


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


def align_signal_to_fullres(signal: np.ndarray, source_level: int, image_shape: tuple[int, int, int]) -> np.ndarray:
    if source_level == 0:
        return signal

    if signal.dtype == np.uint8:
        return iterative_upsample(signal, source_level, 0, image_shape)

    image_h, image_w, _ = image_shape
    curr = signal.astype(np.float32)
    for next_level in range(source_level - 1, -1, -1):
        target_h = image_h // (2 ** next_level) if next_level > 0 else image_h
        target_w = image_w // (2 ** next_level) if next_level > 0 else image_w
        curr = cv2.pyrUp(curr, dstsize=(target_w, target_h))
    return curr


def patch_energy(signal: np.ndarray, patch_size: tuple[int, int]) -> np.ndarray:
    patch_h, patch_w = patch_size
    height, width, channels = signal.shape
    grid_h = height // patch_h
    grid_w = width // patch_w

    patches = signal.reshape(grid_h, patch_h, grid_w, patch_w, channels)
    patches = patches.transpose(0, 2, 1, 3, 4).reshape(-1, patch_h * patch_w * channels)
    patches_f64 = patches.astype(np.float64)
    energies = np.einsum("nc,nc->n", patches_f64, patches_f64)
    return energies


def signal_energy_stats(signal: np.ndarray) -> tuple[float, float]:
    signal_f64 = signal.astype(np.float64)
    sq = signal_f64 * signal_f64
    return float(sq.mean()), float(sq.sum())


def safe_ratio(numer: np.ndarray | float, denom: np.ndarray | float) -> np.ndarray:
    numer_arr = np.asarray(numer, dtype=np.float64)
    denom_arr = np.asarray(denom, dtype=np.float64)
    out = np.full(np.broadcast_shapes(numer_arr.shape, denom_arr.shape), np.inf, dtype=np.float64)
    numer_b, denom_b = np.broadcast_arrays(numer_arr, denom_arr)
    positive = denom_b > 0
    out[positive] = numer_b[positive] / denom_b[positive]
    both_zero = (~positive) & (numer_b == 0)
    out[both_zero] = 0.0
    return out


def summarize_values(values: np.ndarray, prefix: str) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = arr[np.isfinite(arr)]
    summary: dict[str, float | int | None] = {
        f"{prefix}_count": int(arr.size),
        f"{prefix}_finite_count": int(finite.size),
        f"{prefix}_inf_count": int(arr.size - finite.size),
    }

    stats = ["mean", "std", "min", "q05", "q25", "q50", "q75", "q95", "q99", "max"]
    if finite.size == 0:
        for name in stats:
            summary[f"{prefix}_{name}"] = None
        return summary

    summary[f"{prefix}_mean"] = float(finite.mean())
    summary[f"{prefix}_std"] = float(finite.std())
    summary[f"{prefix}_min"] = float(finite.min())
    summary[f"{prefix}_q05"] = float(np.quantile(finite, 0.05))
    summary[f"{prefix}_q25"] = float(np.quantile(finite, 0.25))
    summary[f"{prefix}_q50"] = float(np.quantile(finite, 0.50))
    summary[f"{prefix}_q75"] = float(np.quantile(finite, 0.75))
    summary[f"{prefix}_q95"] = float(np.quantile(finite, 0.95))
    summary[f"{prefix}_q99"] = float(np.quantile(finite, 0.99))
    summary[f"{prefix}_max"] = float(finite.max())
    return summary


def summarize_ratio_thresholds(values: np.ndarray, prefix: str) -> dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    summary = {}
    if arr.size == 0:
        for threshold in RATIO_THRESHOLDS:
            summary[f"{prefix}_frac_le_{format_threshold(threshold)}"] = 0.0
        summary[f"{prefix}_frac_gt_1"] = 0.0
        return summary

    for threshold in RATIO_THRESHOLDS:
        summary[f"{prefix}_frac_le_{format_threshold(threshold)}"] = float(np.mean(arr <= threshold))
    summary[f"{prefix}_frac_gt_1"] = float(np.mean(arr > 1.0))
    return summary


def format_threshold(value: float) -> str:
    if value >= 1:
        return str(int(value))
    return str(value).replace(".", "p")


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


def level_shape(image_shape: tuple[int, int, int], level: int) -> tuple[int, int, int]:
    image_h, image_w, channels = image_shape
    scale = 2 ** level
    return image_h // scale, image_w // scale, channels


def to_uint32_exact(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    if values.size == 0:
        return values.astype(np.uint32)
    max_value = values.max()
    if max_value > np.iinfo(np.uint32).max:
        raise ValueError(f"Patch energy exceeds uint32 range: {max_value}")
    return values.astype(np.uint32)


def initialize_level_store(
    num_samples: int,
    levels: list[int],
    image_shape: tuple[int, int, int],
    patch_size: tuple[int, int],
) -> dict[int, dict[str, np.ndarray | str | int]]:
    patch_h, patch_w = patch_size
    image_h, image_w, _ = image_shape
    store: dict[int, dict[str, np.ndarray | str | int]] = {}
    base_level = levels[0]
    num_patches = (image_h // patch_h) * (image_w // patch_w)

    for level in levels:
        source_level_h, source_level_w, _ = level_shape(image_shape, level)
        signal_kind = "base_gaussian" if level == base_level else "residual"

        row: dict[str, np.ndarray | str | int] = {
            "signal_kind": signal_kind,
            "analysis_space": "aligned_fullres",
            "source_level_h": source_level_h,
            "source_level_w": source_level_w,
            "analysis_h": image_h,
            "analysis_w": image_w,
            "num_patches": num_patches,
            "signal_patch_energy": np.empty((num_samples, num_patches), dtype=np.uint32),
            "signal_energy_mean": np.empty(num_samples, dtype=np.float64),
            "signal_energy_sum": np.empty(num_samples, dtype=np.float64),
        }

        if signal_kind == "residual":
            row["reference_energy_mean"] = np.empty(num_samples, dtype=np.float64)
            row["reference_energy_sum"] = np.empty(num_samples, dtype=np.float64)
            row["target_energy_mean"] = np.empty(num_samples, dtype=np.float64)
            row["target_energy_sum"] = np.empty(num_samples, dtype=np.float64)
            row["ratio_energy_to_reference"] = np.empty(num_samples, dtype=np.float64)
            row["ratio_energy_to_target"] = np.empty(num_samples, dtype=np.float64)
            row["ratio_patch_to_reference"] = np.empty((num_samples, num_patches), dtype=np.float32)

        store[level] = row

    return store


def image_level_row(
    dataset_name: str,
    sample_meta: dict[str, object],
    level: int,
    signal_kind: str,
    signal_energy_mean_value: float,
    signal_energy_sum_value: float,
    signal_patch_energy: np.ndarray,
    reference_energy_mean_value: float | None = None,
    reference_energy_sum_value: float | None = None,
    target_energy_mean_value: float | None = None,
    target_energy_sum_value: float | None = None,
    patch_ratio_to_reference: np.ndarray | None = None,
    ratio_energy_to_reference: float | None = None,
    ratio_energy_to_target: float | None = None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "dataset": dataset_name,
        "sample_index": sample_meta["sample_index"],
        "sample_key": sample_meta["sample_key"],
        "path": sample_meta["path"],
        "label_or_image_id": sample_meta["label_or_image_id"],
        "orig_width": sample_meta["orig_width"],
        "orig_height": sample_meta["orig_height"],
        "level": level,
        "signal_kind": signal_kind,
        "signal_energy_mean": signal_energy_mean_value,
        "signal_energy_sum": signal_energy_sum_value,
    }
    row["analysis_space"] = sample_meta["analysis_space"]
    row["source_level_h"] = sample_meta["source_level_h"]
    row["source_level_w"] = sample_meta["source_level_w"]
    row["analysis_h"] = sample_meta["analysis_h"]
    row["analysis_w"] = sample_meta["analysis_w"]
    row.update(summarize_values(signal_patch_energy, "patch_energy"))

    if signal_kind == "residual":
        assert reference_energy_mean_value is not None
        assert reference_energy_sum_value is not None
        assert target_energy_mean_value is not None
        assert target_energy_sum_value is not None
        assert patch_ratio_to_reference is not None
        assert ratio_energy_to_reference is not None
        assert ratio_energy_to_target is not None

        row["reference_energy_mean"] = reference_energy_mean_value
        row["reference_energy_sum"] = reference_energy_sum_value
        row["target_energy_mean"] = target_energy_mean_value
        row["target_energy_sum"] = target_energy_sum_value
        row["ratio_energy_to_reference"] = ratio_energy_to_reference
        row["ratio_energy_to_target"] = ratio_energy_to_target
        row.update(summarize_values(patch_ratio_to_reference, "patch_ratio_to_reference"))
        row.update(summarize_ratio_thresholds(patch_ratio_to_reference, "patch_ratio_to_reference"))

    return row


def dataset_level_row(
    dataset_name: str,
    level: int,
    level_store: dict[str, np.ndarray | str | int],
) -> dict[str, object]:
    signal_kind = str(level_store["signal_kind"])
    signal_patch_energy = np.asarray(level_store["signal_patch_energy"])
    row: dict[str, object] = {
        "dataset": dataset_name,
        "level": level,
        "signal_kind": signal_kind,
        "analysis_space": str(level_store["analysis_space"]),
        "num_images": int(signal_patch_energy.shape[0]),
        "source_level_h": int(level_store["source_level_h"]),
        "source_level_w": int(level_store["source_level_w"]),
        "analysis_h": int(level_store["analysis_h"]),
        "analysis_w": int(level_store["analysis_w"]),
        "num_patches_per_image": int(level_store["num_patches"]),
        "signal_energy_mean_avg": float(np.mean(level_store["signal_energy_mean"])),
        "signal_energy_mean_std": float(np.std(level_store["signal_energy_mean"])),
        "signal_energy_sum_avg": float(np.mean(level_store["signal_energy_sum"])),
        "signal_energy_sum_std": float(np.std(level_store["signal_energy_sum"])),
    }
    row.update(summarize_values(signal_patch_energy, "patch_energy"))

    if signal_kind == "residual":
        ratio_energy_to_reference = np.asarray(level_store["ratio_energy_to_reference"])
        ratio_energy_to_target = np.asarray(level_store["ratio_energy_to_target"])
        ratio_patch_to_reference = np.asarray(level_store["ratio_patch_to_reference"])

        row["reference_energy_mean_avg"] = float(np.mean(level_store["reference_energy_mean"]))
        row["reference_energy_mean_std"] = float(np.std(level_store["reference_energy_mean"]))
        row["target_energy_mean_avg"] = float(np.mean(level_store["target_energy_mean"]))
        row["target_energy_mean_std"] = float(np.std(level_store["target_energy_mean"]))
        row.update(summarize_values(ratio_energy_to_reference, "ratio_energy_to_reference"))
        row.update(summarize_values(ratio_energy_to_target, "ratio_energy_to_target"))
        row.update(summarize_values(ratio_patch_to_reference, "patch_ratio_to_reference"))
        row.update(summarize_ratio_thresholds(ratio_patch_to_reference, "patch_ratio_to_reference"))

    return row


def level_label(level: int, signal_kind: str) -> str:
    return f"L{level} base" if signal_kind == "base_gaussian" else f"L{level} dx"


def sci_text(value: float) -> str:
    if not np.isfinite(value):
        return "inf"
    return f"{value:.2e}"


def sample_for_plot(values: np.ndarray, max_points: int = 50000) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size <= max_points:
        return arr
    indices = np.linspace(0, arr.size - 1, num=max_points, dtype=np.int64)
    return arr[indices]


def finite_positive(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr) & (arr > 0)]


def finite_values(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def create_canvas(width: int, height: int) -> np.ndarray:
    return np.full((height, width, 3), 247, dtype=np.uint8)


def draw_text(image: np.ndarray, text: str, x: int, y: int, scale: float = 0.6, color=(30, 30, 30), thickness: int = 1) -> None:
    cv2.putText(
        image,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_panel_frame(image: np.ndarray, rect: tuple[int, int, int, int], title: str, subtitle: str | None = None) -> None:
    x, y, w, h = rect
    cv2.rectangle(image, (x, y), (x + w, y + h), (210, 214, 220), 2)
    draw_text(image, title, x + 16, y + 28, scale=0.75, color=(20, 20, 20), thickness=2)
    if subtitle:
        draw_text(image, subtitle, x + 16, y + 54, scale=0.48, color=(95, 95, 95), thickness=1)


def fit_log_range(values: list[np.ndarray], default_min: float = 1e-4, default_max: float = 1.0) -> tuple[float, float]:
    positives = [finite_positive(v) for v in values]
    positives = [v for v in positives if v.size > 0]
    if not positives:
        return default_min, default_max
    combined = np.concatenate(positives)
    min_v = float(combined.min())
    max_v = float(combined.max())
    min_v = 10 ** math.floor(math.log10(max(min_v, 1e-12)))
    max_v = 10 ** math.ceil(math.log10(max(max_v, min_v * 10)))
    if min_v == max_v:
        max_v = min_v * 10
    return min_v, max_v


def map_log_y(value: float, y0: int, y1: int, min_v: float, max_v: float) -> int:
    value = min(max(value, min_v), max_v)
    lo = math.log10(min_v)
    hi = math.log10(max_v)
    t = 0.0 if hi == lo else (math.log10(value) - lo) / (hi - lo)
    return int(round(y1 - t * (y1 - y0)))


def draw_log_grid(image: np.ndarray, x0: int, y0: int, x1: int, y1: int, min_v: float, max_v: float, tick_x: int) -> None:
    start_exp = int(math.floor(math.log10(min_v)))
    end_exp = int(math.ceil(math.log10(max_v)))
    for exp in range(start_exp, end_exp + 1):
        tick = 10 ** exp
        if tick < min_v or tick > max_v:
            continue
        y = map_log_y(tick, y0, y1, min_v, max_v)
        cv2.line(image, (x0, y), (x1, y), (228, 231, 236), 1)
        draw_text(image, f"1e{exp}", tick_x, y + 4, scale=0.42, color=(110, 110, 110))


def draw_energy_bar_panel(
    image: np.ndarray,
    rect: tuple[int, int, int, int],
    pyramid_levels: list[int],
    level_store: dict[int, dict[str, np.ndarray | str | int]],
    colors: dict[int, tuple[int, int, int]],
) -> None:
    x, y, w, h = rect
    draw_panel_frame(image, rect, "Signal Energy By Level", "Mean per-pixel energy, log scale")

    plot_x0 = x + 72
    plot_y0 = y + 78
    plot_x1 = x + w - 24
    plot_y1 = y + h - 52

    means = [float(np.mean(level_store[level]["signal_energy_mean"])) for level in pyramid_levels]
    min_v, max_v = fit_log_range([np.asarray(means)], default_min=1.0, default_max=1e5)
    draw_log_grid(image, plot_x0, plot_y0, plot_x1, plot_y1, min_v, max_v, x + 10)
    cv2.line(image, (plot_x0, plot_y0), (plot_x0, plot_y1), (90, 90, 90), 2)
    cv2.line(image, (plot_x0, plot_y1), (plot_x1, plot_y1), (90, 90, 90), 2)

    num_levels = len(pyramid_levels)
    slot_w = (plot_x1 - plot_x0) / max(num_levels, 1)
    bar_w = max(20, int(slot_w * 0.55))

    for idx, level in enumerate(pyramid_levels):
        mean_value = means[idx]
        signal_kind = str(level_store[level]["signal_kind"])
        cx = int(plot_x0 + slot_w * (idx + 0.5))
        y_top = map_log_y(max(mean_value, min_v), plot_y0, plot_y1, min_v, max_v)
        color = colors[level]
        cv2.rectangle(image, (cx - bar_w // 2, y_top), (cx + bar_w // 2, plot_y1), color, -1)
        cv2.rectangle(image, (cx - bar_w // 2, y_top), (cx + bar_w // 2, plot_y1), (70, 70, 70), 1)
        draw_text(image, level_label(level, signal_kind), cx - 34, plot_y1 + 24, scale=0.46, color=(50, 50, 50))
        draw_text(image, sci_text(mean_value), cx - 34, max(y_top - 8, plot_y0 + 18), scale=0.42, color=(50, 50, 50))


def draw_ratio_box_panel(
    image: np.ndarray,
    rect: tuple[int, int, int, int],
    residual_levels: list[int],
    level_store: dict[int, dict[str, np.ndarray | str | int]],
    colors: dict[int, tuple[int, int, int]],
) -> None:
    x, y, w, h = rect
    draw_panel_frame(image, rect, "Image-Level dx/x Ratio", "Each box is ratio_energy_to_reference = ||dx||^2 / ||x||^2")

    plot_x0 = x + 72
    plot_y0 = y + 78
    plot_x1 = x + w - 24
    plot_y1 = y + h - 52

    series = [np.asarray(level_store[level]["ratio_energy_to_reference"]) for level in residual_levels]
    min_v, max_v = fit_log_range(series, default_min=1e-4, default_max=1.0)
    draw_log_grid(image, plot_x0, plot_y0, plot_x1, plot_y1, min_v, max_v, x + 10)
    cv2.line(image, (plot_x0, plot_y0), (plot_x0, plot_y1), (90, 90, 90), 2)
    cv2.line(image, (plot_x0, plot_y1), (plot_x1, plot_y1), (90, 90, 90), 2)

    slot_w = (plot_x1 - plot_x0) / max(len(residual_levels), 1)
    box_w = max(26, int(slot_w * 0.38))

    for idx, level in enumerate(residual_levels):
        values = finite_positive(level_store[level]["ratio_energy_to_reference"])
        if values.size == 0:
            continue
        q05, q25, q50, q75, q95 = np.quantile(values, [0.05, 0.25, 0.50, 0.75, 0.95])
        mean_v = float(np.mean(values))
        cx = int(plot_x0 + slot_w * (idx + 0.5))
        color = colors[level]
        y_q05 = map_log_y(q05, plot_y0, plot_y1, min_v, max_v)
        y_q25 = map_log_y(q25, plot_y0, plot_y1, min_v, max_v)
        y_q50 = map_log_y(q50, plot_y0, plot_y1, min_v, max_v)
        y_q75 = map_log_y(q75, plot_y0, plot_y1, min_v, max_v)
        y_q95 = map_log_y(q95, plot_y0, plot_y1, min_v, max_v)
        y_mean = map_log_y(mean_v, plot_y0, plot_y1, min_v, max_v)

        cv2.line(image, (cx, y_q05), (cx, y_q95), (70, 70, 70), 2)
        cv2.rectangle(image, (cx - box_w // 2, y_q75), (cx + box_w // 2, y_q25), color, -1)
        cv2.rectangle(image, (cx - box_w // 2, y_q75), (cx + box_w // 2, y_q25), (70, 70, 70), 1)
        cv2.line(image, (cx - box_w // 2, y_q50), (cx + box_w // 2, y_q50), (20, 20, 20), 2)
        cv2.line(image, (cx - box_w // 3, y_q05), (cx + box_w // 3, y_q05), (70, 70, 70), 2)
        cv2.line(image, (cx - box_w // 3, y_q95), (cx + box_w // 3, y_q95), (70, 70, 70), 2)
        cv2.circle(image, (cx, y_mean), 4, (20, 20, 20), -1)

        draw_text(image, f"L{level}", cx - 10, plot_y1 + 24, scale=0.5, color=(50, 50, 50))
        draw_text(image, sci_text(q50), cx - 24, max(y_q50 - 8, plot_y0 + 18), scale=0.42, color=(50, 50, 50))


def draw_patch_cdf_panel(
    image: np.ndarray,
    rect: tuple[int, int, int, int],
    residual_levels: list[int],
    level_store: dict[int, dict[str, np.ndarray | str | int]],
    colors: dict[int, tuple[int, int, int]],
) -> None:
    x, y, w, h = rect
    draw_panel_frame(image, rect, "Patch CDF of dx/x", "ECDF of patch_ratio_to_reference = ||dx||^2 / ||x||^2")

    plot_x0 = x + 72
    plot_y0 = y + 78
    plot_x1 = x + w - 24
    plot_y1 = y + h - 52

    raw_series = [sample_for_plot(level_store[level]["ratio_patch_to_reference"], max_points=50000) for level in residual_levels]
    min_v, max_v = fit_log_range(raw_series, default_min=1e-5, default_max=1.0)
    min_log = math.log10(min_v)
    max_log = math.log10(max_v)

    for exp in range(int(math.floor(min_log)), int(math.ceil(max_log)) + 1):
        tick = 10 ** exp
        if tick < min_v or tick > max_v:
            continue
        t = (math.log10(tick) - min_log) / max(max_log - min_log, 1e-12)
        px = int(round(plot_x0 + t * (plot_x1 - plot_x0)))
        cv2.line(image, (px, plot_y0), (px, plot_y1), (228, 231, 236), 1)
        draw_text(image, f"1e{exp}", px - 16, plot_y1 + 24, scale=0.42, color=(110, 110, 110))

    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        py = int(round(plot_y1 - frac * (plot_y1 - plot_y0)))
        cv2.line(image, (plot_x0, py), (plot_x1, py), (228, 231, 236), 1)
        draw_text(image, f"{frac:.2f}", x + 10, py + 4, scale=0.42, color=(110, 110, 110))

    cv2.line(image, (plot_x0, plot_y0), (plot_x0, plot_y1), (90, 90, 90), 2)
    cv2.line(image, (plot_x0, plot_y1), (plot_x1, plot_y1), (90, 90, 90), 2)

    for marker in [0.01, 0.1, 1.0]:
        if marker < min_v or marker > max_v:
            continue
        t = (math.log10(marker) - min_log) / max(max_log - min_log, 1e-12)
        px = int(round(plot_x0 + t * (plot_x1 - plot_x0)))
        cv2.line(image, (px, plot_y0), (px, plot_y1), (176, 196, 255), 1)

    legend_x = plot_x1 - 118
    legend_y = plot_y0 + 16
    for idx, level in enumerate(residual_levels):
        values = finite_positive(sample_for_plot(level_store[level]["ratio_patch_to_reference"], max_points=50000))
        if values.size == 0:
            continue
        sorted_values = np.sort(values)
        ys = np.linspace(0.0, 1.0, num=sorted_values.size)
        logs = np.log10(np.clip(sorted_values, min_v, max_v))
        xs = plot_x0 + ((logs - min_log) / max(max_log - min_log, 1e-12)) * (plot_x1 - plot_x0)
        pts = np.stack([xs, plot_y1 - ys * (plot_y1 - plot_y0)], axis=1).astype(np.int32)
        cv2.polylines(image, [pts], False, colors[level], 2, cv2.LINE_AA)
        ly = legend_y + idx * 22
        cv2.line(image, (legend_x, ly), (legend_x + 18, ly), colors[level], 3)
        draw_text(image, f"L{level}", legend_x + 26, ly + 4, scale=0.46, color=(50, 50, 50))


def draw_threshold_panel(
    image: np.ndarray,
    rect: tuple[int, int, int, int],
    residual_levels: list[int],
    level_store: dict[int, dict[str, np.ndarray | str | int]],
    colors: dict[int, tuple[int, int, int]],
) -> None:
    x, y, w, h = rect
    thresholds = [0.01, 0.05, 0.1, 0.25]
    draw_panel_frame(image, rect, "How Often Is dx Small?", "Fraction of patches with ||dx||^2 / ||x||^2 below threshold")

    plot_x0 = x + 72
    plot_y0 = y + 78
    plot_x1 = x + w - 24
    plot_y1 = y + h - 72
    cv2.line(image, (plot_x0, plot_y0), (plot_x0, plot_y1), (90, 90, 90), 2)
    cv2.line(image, (plot_x0, plot_y1), (plot_x1, plot_y1), (90, 90, 90), 2)

    for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
        py = int(round(plot_y1 - frac * (plot_y1 - plot_y0)))
        cv2.line(image, (plot_x0, py), (plot_x1, py), (228, 231, 236), 1)
        draw_text(image, f"{frac:.2f}", x + 10, py + 4, scale=0.42, color=(110, 110, 110))

    level_slot_w = (plot_x1 - plot_x0) / max(len(residual_levels), 1)
    inner_slot_w = level_slot_w / (len(thresholds) + 1)
    threshold_colors = [
        (217, 217, 217),
        (196, 222, 255),
        (158, 198, 255),
        (106, 153, 245),
    ]

    for level_idx, level in enumerate(residual_levels):
        values = finite_values(level_store[level]["ratio_patch_to_reference"])
        level_x = plot_x0 + level_slot_w * level_idx
        for th_idx, threshold in enumerate(thresholds):
            frac = float(np.mean(values <= threshold)) if values.size else 0.0
            bar_left = int(round(level_x + inner_slot_w * (th_idx + 0.35)))
            bar_right = int(round(bar_left + inner_slot_w * 0.58))
            bar_top = int(round(plot_y1 - frac * (plot_y1 - plot_y0)))
            cv2.rectangle(image, (bar_left, bar_top), (bar_right, plot_y1), threshold_colors[th_idx], -1)
            cv2.rectangle(image, (bar_left, bar_top), (bar_right, plot_y1), (90, 90, 90), 1)
        center_x = int(round(level_x + level_slot_w * 0.5))
        draw_text(image, f"L{level}", center_x - 10, plot_y1 + 26, scale=0.5, color=colors[level])

    legend_x = plot_x0
    legend_y = y + h - 32
    for idx, threshold in enumerate(thresholds):
        lx = legend_x + idx * 116
        cv2.rectangle(image, (lx, legend_y - 12), (lx + 16, legend_y + 4), threshold_colors[idx], -1)
        cv2.rectangle(image, (lx, legend_y - 12), (lx + 16, legend_y + 4), (90, 90, 90), 1)
        draw_text(image, f"<= {threshold:g}", lx + 24, legend_y, scale=0.44, color=(60, 60, 60))


def write_visual_summary(
    out_dir: Path,
    dataset_name: str,
    processed: int,
    pyramid_levels: list[int],
    level_store: dict[int, dict[str, np.ndarray | str | int]],
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required for visualization. Install it in the active environment and rerun."
        ) from exc

    residual_levels = [level for level in pyramid_levels if str(level_store[level]["signal_kind"]) == "residual"]
    palette = ["#5B8FF9", "#F6BD16", "#5AD8A6", "#E8684A", "#6DC8EC"]
    colors = {level: palette[idx % len(palette)] for idx, level in enumerate(pyramid_levels)}

    fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=180, constrained_layout=True)
    fig.suptitle(
        f"Laplacian Energy Summary: {dataset_name} | samples={processed}",
        fontsize=18,
        fontweight="bold",
    )

    ax = axes[0, 0]
    energy_means = [float(np.mean(level_store[level]["signal_energy_mean"])) for level in pyramid_levels]
    labels = [level_label(level, str(level_store[level]["signal_kind"])) for level in pyramid_levels]
    bars = ax.bar(
        np.arange(len(pyramid_levels)),
        energy_means,
        color=[colors[level] for level in pyramid_levels],
        edgecolor="#2f2f2f",
        linewidth=0.8,
    )
    ax.set_yscale("log")
    ax.set_title("Signal Energy By Level (Aligned To Full Resolution)")
    ax.set_ylabel("Mean per-pixel energy")
    ax.set_xticks(np.arange(len(pyramid_levels)), labels, rotation=0)
    ax.grid(True, axis="y", which="both", alpha=0.25)
    for bar, value in zip(bars, energy_means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            sci_text(value),
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax = axes[0, 1]
    ax.set_title("Image-Level dx/x Ratio (Aligned To Full Resolution)")
    ax.set_ylabel(r"$||dx||^2 / ||x||^2$")
    if residual_levels:
        series = [finite_positive(level_store[level]["ratio_energy_to_reference"]) for level in residual_levels]
        box = ax.boxplot(
            series,
            labels=[f"L{level}" for level in residual_levels],
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#202020", "linewidth": 1.8},
            whiskerprops={"color": "#4a4a4a"},
            capprops={"color": "#4a4a4a"},
        )
        for patch, level in zip(box["boxes"], residual_levels):
            patch.set_facecolor(colors[level])
            patch.set_alpha(0.7)
            patch.set_edgecolor("#3a3a3a")
        ax.set_yscale("log")
        ax.axhline(1.0, color="#d9534f", linestyle="--", linewidth=1.0, alpha=0.9)
        ax.grid(True, axis="y", which="both", alpha=0.25)
    else:
        ax.text(0.5, 0.5, "No residual levels", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])

    ax = axes[1, 0]
    ax.set_title("Patch CDF of dx/x (Aligned To Full Resolution)")
    ax.set_xlabel(r"patch $||dx||^2 / ||x||^2$")
    ax.set_ylabel("CDF")
    if residual_levels:
        for level in residual_levels:
            values = finite_positive(sample_for_plot(level_store[level]["ratio_patch_to_reference"], max_points=50000))
            if values.size == 0:
                continue
            values = np.sort(values)
            ys = np.linspace(0.0, 1.0, num=values.size)
            ax.plot(values, ys, label=f"L{level}", color=colors[level], linewidth=2.0)
        ax.set_xscale("log")
        ax.set_ylim(0.0, 1.0)
        for threshold in [0.01, 0.1, 1.0]:
            ax.axvline(threshold, color="#9aa5b1", linestyle="--", linewidth=1.0, alpha=0.9)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "No residual levels", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])

    ax = axes[1, 1]
    ax.set_title("How Often Is dx Small? (Aligned To Full Resolution)")
    ax.set_ylabel("Fraction of patches")
    thresholds = [0.01, 0.05, 0.1, 0.25]
    if residual_levels:
        x = np.arange(len(residual_levels), dtype=np.float64)
        width = 0.18
        threshold_palette = ["#d9d9d9", "#b3d4ff", "#7fb3ff", "#4f86f7"]
        for idx, threshold in enumerate(thresholds):
            fracs = []
            for level in residual_levels:
                values = finite_values(level_store[level]["ratio_patch_to_reference"])
                fracs.append(float(np.mean(values <= threshold)) if values.size else 0.0)
            offset = (idx - (len(thresholds) - 1) / 2) * width
            ax.bar(x + offset, fracs, width=width, label=f"<= {threshold:g}", color=threshold_palette[idx], edgecolor="#3a3a3a", linewidth=0.6)
        ax.set_xticks(x, [f"L{level}" for level in residual_levels])
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(frameon=False, ncol=2)
    else:
        ax.text(0.5, 0.5, "No residual levels", ha="center", va="center", transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.savefig(out_dir / "summary_dashboard.png", bbox_inches="tight")
    plt.close(fig)

def main():
    args = parse_args()
    started_at = time.time()

    config, raw_config = load_config(args.config)
    config.dataset_name = normalize_offload_dataset_name(args.dataset or config.dataset_name)

    image_h = args.image_size or int(config.image_shape[0])
    image_w = args.image_size or int(config.image_shape[1])
    channels = int(config.image_shape[2])
    image_shape = (image_h, image_w, channels)

    config.image_shape = image_shape
    patch_size = parse_patch_size(args.patch_size, tuple(config.patch_size))
    config.patch_size = patch_size

    pyramid_levels = parse_pyramid_levels(args.pyramid_levels)
    validate_geometry(image_shape, patch_size, pyramid_levels)

    batch_size = args.batch_size or int(config.batch_size)
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    data_root = determine_data_root(config.dataset_name, args.data_root)
    loader_kwargs = dict(config.dataset_kwargs)
    loader_kwargs["num_workers"] = args.num_workers

    loader = build_loader_with_fallback(
        config.dataset_name,
        data_root,
        batch_size,
        image_shape[0],
        loader_kwargs,
    )

    planned_samples = dataset_sample_count(loader.dataset, batch_size, args.max_batches, args.max_samples)
    if planned_samples <= 0:
        raise ValueError("No samples selected. Check max-batches/max-samples.")

    out_dir = resolve_output_dir("laplacian_energy", args.out_dir)
    level_store = initialize_level_store(planned_samples, pyramid_levels, image_shape, patch_size)

    sample_rows: list[dict[str, object]] = []
    image_level_rows: list[dict[str, object]] = []
    sample_keys: list[str] = []
    sample_paths: list[str] = []
    sample_indices: list[int] = []

    processed = 0
    max_level = max(pyramid_levels)
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
            dataset_index = processed + local_idx
            path = paths[local_idx]
            label_or_image_id, orig_width, orig_height = infer_sample_label(
                config.dataset_name,
                labels[local_idx],
            )
            sample_key = f"{dataset_index:06d}:{Path(path).name}" if path else f"{dataset_index:06d}"
            sample_meta = {
                "sample_index": dataset_index,
                "sample_key": sample_key,
                "path": path,
                "label_or_image_id": label_or_image_id,
                "orig_width": orig_width,
                "orig_height": orig_height,
            }

            sample_rows.append({
                "dataset": config.dataset_name,
                "sample_index": dataset_index,
                "sample_key": sample_key,
                "path": path,
                "label_or_image_id": label_or_image_id,
                "orig_width": orig_width,
                "orig_height": orig_height,
            })
            sample_keys.append(sample_key)
            sample_paths.append(path)
            sample_indices.append(dataset_index)

            image = images_bhwc[local_idx]
            gaussians = build_gaussians(image, max_level)
            prev_level = pyramid_levels[0]
            prev_gaussian = gaussians[prev_level]

            base_signal = align_signal_to_fullres(prev_gaussian, prev_level, image_shape)
            base_patch_energy = patch_energy(base_signal, patch_size)
            base_signal_energy_mean, base_signal_energy_sum = signal_energy_stats(base_signal)
            level_store[prev_level]["signal_patch_energy"][dataset_index] = to_uint32_exact(base_patch_energy)
            level_store[prev_level]["signal_energy_mean"][dataset_index] = base_signal_energy_mean
            level_store[prev_level]["signal_energy_sum"][dataset_index] = base_signal_energy_sum
            sample_meta["analysis_space"] = str(level_store[prev_level]["analysis_space"])
            sample_meta["source_level_h"] = int(level_store[prev_level]["source_level_h"])
            sample_meta["source_level_w"] = int(level_store[prev_level]["source_level_w"])
            sample_meta["analysis_h"] = int(level_store[prev_level]["analysis_h"])
            sample_meta["analysis_w"] = int(level_store[prev_level]["analysis_w"])
            image_level_rows.append(
                image_level_row(
                    config.dataset_name,
                    sample_meta,
                    prev_level,
                    "base_gaussian",
                    base_signal_energy_mean,
                    base_signal_energy_sum,
                    base_patch_energy,
                )
            )

            for level in pyramid_levels[1:]:
                target = gaussians[level]
                reference = iterative_upsample(prev_gaussian, prev_level, level, image_shape)
                target_aligned = align_signal_to_fullres(target, level, image_shape)
                reference_aligned = align_signal_to_fullres(reference, level, image_shape)
                residual_aligned = target_aligned.astype(np.int16) - reference_aligned.astype(np.int16)

                residual_patch_energy = patch_energy(residual_aligned, patch_size)
                reference_patch_energy = patch_energy(reference_aligned, patch_size)

                residual_energy_mean, residual_energy_sum = signal_energy_stats(residual_aligned)
                reference_energy_mean, reference_energy_sum = signal_energy_stats(reference_aligned)
                target_energy_mean, target_energy_sum = signal_energy_stats(target_aligned)

                patch_ratio_to_reference = safe_ratio(residual_patch_energy, reference_patch_energy)
                ratio_energy_to_reference = float(safe_ratio(residual_energy_sum, reference_energy_sum))
                ratio_energy_to_target = float(safe_ratio(residual_energy_sum, target_energy_sum))

                level_store[level]["signal_patch_energy"][dataset_index] = to_uint32_exact(residual_patch_energy)
                level_store[level]["signal_energy_mean"][dataset_index] = residual_energy_mean
                level_store[level]["signal_energy_sum"][dataset_index] = residual_energy_sum
                level_store[level]["reference_energy_mean"][dataset_index] = reference_energy_mean
                level_store[level]["reference_energy_sum"][dataset_index] = reference_energy_sum
                level_store[level]["target_energy_mean"][dataset_index] = target_energy_mean
                level_store[level]["target_energy_sum"][dataset_index] = target_energy_sum
                level_store[level]["ratio_energy_to_reference"][dataset_index] = ratio_energy_to_reference
                level_store[level]["ratio_energy_to_target"][dataset_index] = ratio_energy_to_target
                level_store[level]["ratio_patch_to_reference"][dataset_index] = patch_ratio_to_reference.astype(np.float32)
                sample_meta["analysis_space"] = str(level_store[level]["analysis_space"])
                sample_meta["source_level_h"] = int(level_store[level]["source_level_h"])
                sample_meta["source_level_w"] = int(level_store[level]["source_level_w"])
                sample_meta["analysis_h"] = int(level_store[level]["analysis_h"])
                sample_meta["analysis_w"] = int(level_store[level]["analysis_w"])

                image_level_rows.append(
                    image_level_row(
                        config.dataset_name,
                        sample_meta,
                        level,
                        "residual",
                        residual_energy_mean,
                        residual_energy_sum,
                        residual_patch_energy,
                        reference_energy_mean_value=reference_energy_mean,
                        reference_energy_sum_value=reference_energy_sum,
                        target_energy_mean_value=target_energy_mean,
                        target_energy_sum_value=target_energy_sum,
                        patch_ratio_to_reference=patch_ratio_to_reference,
                        ratio_energy_to_reference=ratio_energy_to_reference,
                        ratio_energy_to_target=ratio_energy_to_target,
                    )
                )

                prev_gaussian = target
                prev_level = level

            progress.set_description(f"samples={dataset_index + 1}")

        processed += current_batch

        if args.max_samples is not None and processed >= args.max_samples:
            break

    progress.close()

    sample_rows = sample_rows[:processed]
    image_level_rows = image_level_rows[: processed * len(pyramid_levels)]

    for level, store in level_store.items():
        for key, value in list(store.items()):
            if isinstance(value, np.ndarray):
                store[key] = value[:processed]

        file_name = (
            f"patch_energy_level{level}_base.npz"
            if store["signal_kind"] == "base_gaussian"
            else f"patch_energy_level{level}_residual.npz"
        )
        payload = {
            "sample_indices": np.asarray(sample_indices, dtype=np.int32),
            "sample_keys": np.asarray(sample_keys),
            "paths": np.asarray(sample_paths),
            "signal_patch_energy": np.asarray(store["signal_patch_energy"]),
        }
        if store["signal_kind"] == "residual":
            payload["ratio_patch_to_reference"] = np.asarray(store["ratio_patch_to_reference"])
        np.savez_compressed(out_dir / file_name, **payload)

    dataset_level_rows = [
        dataset_level_row(config.dataset_name, level, level_store[level])
        for level in pyramid_levels
    ]

    visualization_path = out_dir / "summary_dashboard.png"
    visualization_error = None
    try:
        write_visual_summary(out_dir, config.dataset_name, processed, pyramid_levels, level_store)
    except Exception as exc:
        visualization_error = f"{type(exc).__name__}: {exc}"

    run_summary = {
        "config_path": str(Path(args.config).expanduser()),
        "dataset_name": config.dataset_name,
        "data_root": data_root,
        "batch_size": batch_size,
        "num_workers": args.num_workers,
        "processed_samples": processed,
        "planned_samples": planned_samples,
        "image_shape": list(image_shape),
        "patch_size": list(patch_size),
        "pyramid_levels": pyramid_levels,
        "analysis_space": "aligned_fullres",
        "duration_sec": time.time() - started_at,
        "visualization_path": str(visualization_path),
        "visualization_error": visualization_error,
        "raw_config": raw_config,
    }

    write_json(out_dir / "run_config.json", run_summary)
    write_csv(out_dir / "samples.csv", sample_rows)
    write_csv(out_dir / "image_level_summary.csv", image_level_rows)
    write_csv(out_dir / "level_summary.csv", dataset_level_rows)

    print(f"[LaplacianEnergy] Wrote run config to {out_dir / 'run_config.json'}")
    print(f"[LaplacianEnergy] Wrote sample list to {out_dir / 'samples.csv'}")
    print(f"[LaplacianEnergy] Wrote image summaries to {out_dir / 'image_level_summary.csv'}")
    print(f"[LaplacianEnergy] Wrote level summaries to {out_dir / 'level_summary.csv'}")
    if visualization_error is None:
        print(f"[LaplacianEnergy] Wrote visualization to {visualization_path}")
    else:
        print(f"[LaplacianEnergy] Visualization skipped: {visualization_error}")


if __name__ == "__main__":
    main()
