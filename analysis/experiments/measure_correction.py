import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import sys
import zipfile
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

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


try:
    from analysis.shared.common import (
        build_analysis_loader,
        default_data_root_for_dataset,
        normalize_dataset_name,
    )
except RuntimeError as exc:
    if "operator torchvision::nms does not exist" not in str(exc):
        raise
    _purge_torchvision_modules()
    _ensure_torchvision_nms_stub()
    from analysis.shared.common import (
        build_analysis_loader,
        default_data_root_for_dataset,
        normalize_dataset_name,
    )
from appcorr.models.dinov3.hub.backbones import (
    dinov3_vit7b16,
    dinov3_vitb16,
    dinov3_vitl16,
    dinov3_vits16,
)

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
TOP_RANKS = (1, 2, 4, 8, 16, 32, 64)
TOP_PCTS = (0.01, 0.05, 0.10, 0.20)
CORRECTION_PCTS = (0.01, 0.05, 0.10, 0.20, 0.30, 0.50)
SPLIT_NAMES = ("analysis", "holdout")
DEPTH_GROUP_NAMES = ("early", "mid", "late")
MODEL_ALIASES = {
    "small": "dinov3_vits16",
    "s": "dinov3_vits16",
    "dinov3_vits16": "dinov3_vits16",
    "base": "dinov3_vitb16",
    "b": "dinov3_vitb16",
    "dinov3_vitb16": "dinov3_vitb16",
    "large": "dinov3_vitl16",
    "l": "dinov3_vitl16",
    "dinov3_vitl16": "dinov3_vitl16",
    "7b": "dinov3_vit7b16",
    "giant": "dinov3_vit7b16",
    "dinov3_vit7b16": "dinov3_vit7b16",
}
DEFAULT_LOCAL_WEIGHTS = {
    "dinov3_vits16": "~/cjpark/weights/dinov3/dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    "dinov3_vitb16": "~/cjpark/weights/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    "dinov3_vitl16": "~/cjpark/weights/dinov3/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
    "dinov3_vit7b16": "~/cjpark/weights/dinov3/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
}
@dataclass
class ProbeRun:
    patch_start: int
    patch_hw: tuple[int, int]
    layer_outputs: dict[int, dict[str, torch.Tensor]]
    final_feature_tokens: torch.Tensor
    pooled_output: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure correction structure between full-resolution and low-resolution inputs "
            "for DINOv3 backbones without applying any correction."
        )
    )
    parser.add_argument("--dataset", type=str, default="imagenet-1k", help="Dataset name: imagenet-1k or coco.")
    parser.add_argument("--data-root", type=str, default=None, help="Dataset root for ImageNet.")
    parser.add_argument("--image-size", type=int, default=256, help="Square image size after resize-and-pad preprocessing.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for paired full/low forwards.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional batch limit.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional sample limit after split filtering.")
    parser.add_argument(
        "--model",
        type=str,
        default="small",
        help="DINOv3 ViT scale alias or canonical name. Supported: small/base/large/7b.",
    )
    parser.add_argument(
        "--backbone-weights",
        type=str,
        default=None,
        help="Optional local backbone weights path. Defaults to common local weight files when present.",
    )
    parser.add_argument("--no-pretrained", action="store_true", help="Use randomly initialized weights instead of loading checkpoints.")
    parser.add_argument("--device", type=str, default=None, help="Torch device. Default: cuda if available else cpu.")
    parser.add_argument(
        "--model-dtype",
        type=str,
        default="bf16",
        help="Resident dtype for the loaded backbone on CUDA: bf16 or fp32. CPU falls back to fp32.",
    )
    parser.add_argument(
        "--autocast-dtype",
        type=str,
        default="bf16",
        help="Autocast dtype on CUDA: bf16, fp16, or fp32.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Deterministic seed.")
    parser.add_argument("--layers", type=str, default=None, help="Comma-separated layer indices. Default: all layers.")
    parser.add_argument(
        "--run-mode",
        type=str,
        default="debug",
        choices=("debug", "stats"),
        help="debug saves tensor dumps and attention maps for a small subset; stats focuses on summaries only.",
    )
    parser.add_argument(
        "--debug-dump-limit",
        type=int,
        default=4,
        help="Number of samples per split allowed to save full debug tensor dumps.",
    )
    parser.add_argument(
        "--save-hidden-states",
        action="store_true",
        help="Save per-sample full/low/delta hidden states for every selected layer.",
    )
    parser.add_argument(
        "--downstream-oracle-methods",
        type=str,
        default="oracle,fixed",
        help="Comma-separated downstream intervention methods to evaluate from saved hidden states: oracle,fixed,predictor,none.",
    )
    parser.add_argument(
        "--debug-representative-layers",
        type=int,
        default=3,
        help="In debug mode, number of representative early/mid/late layers used for downstream re-forward validation.",
    )
    parser.add_argument(
        "--low-mode",
        type=str,
        default="gaussian_pyr",
        choices=("gaussian_pyr", "bicubic"),
        help="Low-resolution construction rule. gaussian_pyr matches the existing L2-style pyramid path.",
    )
    parser.add_argument("--low-level", type=int, default=2, help="Downsample pyramid level for gaussian_pyr mode.")
    parser.add_argument("--downscale", type=int, default=4, help="Downscale factor for bicubic mode.")
    parser.add_argument(
        "--split",
        type=str,
        default="analysis",
        choices=("analysis", "holdout", "both"),
        help="Which split to process. Split routing is deterministic by sample id.",
    )
    parser.add_argument(
        "--analysis-ratio",
        type=float,
        default=0.8,
        help="Fraction of samples routed to analysis_split. The remainder goes to holdout_split.",
    )
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory.")
    return parser.parse_args()


def parse_layers(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    layers = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    return layers or None


def parse_autocast_dtype(value: str) -> torch.dtype | None:
    normalized = value.strip().lower()
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": None,
        "float32": None,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported autocast dtype: {value}")
    return mapping[normalized]


def parse_model_dtype(value: str) -> torch.dtype:
    normalized = value.strip().lower()
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported model dtype: {value}")
    return mapping[normalized]


def parse_downstream_methods(value: str) -> tuple[str, ...]:
    methods = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    if not methods:
        return ()
    if "none" in methods:
        return ()
    valid = {"oracle", "fixed", "predictor"}
    invalid = sorted(set(methods) - valid)
    if invalid:
        raise ValueError(f"Unsupported downstream intervention methods: {', '.join(invalid)}")
    return tuple(dict.fromkeys(methods))


def resolve_output_dir(prefix: str, out_dir: str | None) -> Path:
    if out_dir is not None:
        path = Path(out_dir).expanduser()
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path("logs") / "analysis" / f"{prefix}_{stamp}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        with path.open("w", encoding="utf-8") as handle:
            handle.write("")
        return

    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def set_determinism(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha1_int(text: str) -> int:
    return int(hashlib.sha1(text.encode("utf-8")).hexdigest(), 16)


def assign_split(sample_id: str, analysis_ratio: float) -> str:
    bucket = sha1_int(sample_id) % 10_000
    boundary = int(round(analysis_ratio * 10_000))
    return "analysis" if bucket < boundary else "holdout"


def normalize_model_name(name: str) -> str:
    key = str(name).strip().lower()
    if key not in MODEL_ALIASES:
        raise ValueError(f"Unsupported model alias: {name}")
    return MODEL_ALIASES[key]


def maybe_expand(path: str | None) -> str | None:
    if path is None:
        return None
    return str(Path(path).expanduser())


def resolve_backbone_weights(model_name: str, override: str | None, pretrained: bool) -> str | bool:
    if not pretrained:
        return False
    if override is not None:
        return maybe_expand(override)
    candidate = Path(DEFAULT_LOCAL_WEIGHTS[model_name]).expanduser()
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError(
        f"No local weights found for {model_name}. Pass --backbone-weights explicitly or use --no-pretrained."
    )


def build_gaussians(image: np.ndarray, max_level: int) -> dict[int, np.ndarray]:
    gaussians = {0: image}
    current = image
    for level in range(1, max_level + 1):
        current = cv2.pyrDown(current)
        gaussians[level] = current
    return gaussians


def iterative_upsample(image: np.ndarray, start_level: int, end_level: int, image_shape: tuple[int, int, int]) -> np.ndarray:
    if end_level > start_level:
        raise ValueError(f"end_level={end_level} cannot exceed start_level={start_level}")
    image_h, image_w, _ = image_shape
    current = image
    gap = start_level - end_level
    for offset in range(gap):
        next_level = start_level - 1 - offset
        target_h = image_h // (2 ** next_level)
        target_w = image_w // (2 ** next_level)
        current = cv2.pyrUp(current, dstsize=(target_w, target_h)).astype(np.uint8)
    return current


def make_low_inputs(batch_bchw_uint8: torch.Tensor, mode: str, low_level: int, downscale: int) -> torch.Tensor:
    images = batch_bchw_uint8.permute(0, 2, 3, 1).contiguous().cpu().numpy()
    image_h, image_w = images.shape[1:3]
    low_images = []
    for image in images:
        if mode == "gaussian_pyr":
            gaussians = build_gaussians(image, max_level=low_level)
            low_image = iterative_upsample(gaussians[low_level], low_level, 0, image.shape)
        elif mode == "bicubic":
            target_hw = (max(1, image_w // downscale), max(1, image_h // downscale))
            low_small = cv2.resize(image, target_hw, interpolation=cv2.INTER_AREA)
            low_image = cv2.resize(low_small, (image_w, image_h), interpolation=cv2.INTER_CUBIC)
        else:
            raise ValueError(f"Unsupported low mode: {mode}")
        low_images.append(low_image)
    low_np = np.stack(low_images, axis=0)
    return torch.from_numpy(low_np).permute(0, 3, 1, 2).contiguous()


def build_valid_patch_mask(pixel_valid_mask: torch.Tensor, patch_size: int) -> np.ndarray:
    patch_mask = pixel_valid_mask.reshape(
        pixel_valid_mask.shape[0] // patch_size,
        patch_size,
        pixel_valid_mask.shape[1] // patch_size,
        patch_size,
    ).any(dim=1).any(dim=2)
    return patch_mask.reshape(-1).cpu().numpy().astype(bool)


def cosine_similarity(x: np.ndarray, y: np.ndarray) -> float | None:
    x_flat = np.asarray(x, dtype=np.float64).reshape(-1)
    y_flat = np.asarray(y, dtype=np.float64).reshape(-1)
    denom = np.linalg.norm(x_flat) * np.linalg.norm(y_flat)
    if denom <= 0:
        return None
    return float(np.dot(x_flat, y_flat) / denom)


def relative_error(ref: np.ndarray, cand: np.ndarray) -> float:
    ref_arr = np.asarray(ref, dtype=np.float64)
    cand_arr = np.asarray(cand, dtype=np.float64)
    numerator = np.linalg.norm((cand_arr - ref_arr).reshape(-1))
    denominator = np.linalg.norm(ref_arr.reshape(-1))
    if denominator <= 0:
        return 0.0 if numerator <= 0 else float("inf")
    return float(numerator / denominator)


def pearson_corr(x_values: list[float], y_values: list[float]) -> float | None:
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if x.size < 2:
        return None
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom <= 0:
        return None
    return float(np.dot(x, y) / denom)


def spearman_corr(x_values: list[float], y_values: list[float]) -> float | None:
    x = np.asarray(x_values)
    y = np.asarray(y_values)
    if x.size < 2:
        return None
    x_rank = np.argsort(np.argsort(x, kind="stable"), kind="stable")
    y_rank = np.argsort(np.argsort(y, kind="stable"), kind="stable")
    return pearson_corr(x_rank.tolist(), y_rank.tolist())


def summarize_numeric(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "q25": None,
            "median": None,
            "q75": None,
        }
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "median": float(np.quantile(arr, 0.50)),
        "q75": float(np.quantile(arr, 0.75)),
    }


def percentile_suffix(top_p: float) -> str:
    return f"{int(round(top_p * 100)):02d}pct"


def topk_count(num_channels: int, top_p: float) -> int:
    return max(1, int(math.ceil(num_channels * top_p)))


def topk_indices_from_scores(scores: np.ndarray, top_p: float) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.int64)
    k = topk_count(arr.size, top_p)
    partition = np.argpartition(arr, -k)[-k:]
    values = arr[partition]
    order = np.argsort(-values, kind="stable")
    return partition[order].astype(np.int64)


def jaccard_from_indices(lhs: np.ndarray, rhs: np.ndarray) -> float:
    lhs_set = set(int(value) for value in lhs.tolist())
    rhs_set = set(int(value) for value in rhs.tolist())
    union = len(lhs_set | rhs_set)
    if union == 0:
        return 1.0
    return float(len(lhs_set & rhs_set) / union)


def mean_pairwise_jaccard(index_sets: list[np.ndarray]) -> float | None:
    if len(index_sets) < 2:
        return None
    total = 0.0
    count = 0
    for lhs, rhs in itertools.combinations(index_sets, 2):
        total += jaccard_from_indices(lhs, rhs)
        count += 1
    if count == 0:
        return None
    return float(total / count)


def safe_divide(numerator: float, denominator: float, *, default: float = 0.0) -> float:
    if abs(denominator) <= 1e-12:
        return default
    return float(numerator / denominator)


def assign_depth_group(layer_idx: int, ordered_layers: list[int]) -> str:
    if not ordered_layers:
        return "unknown"
    position = ordered_layers.index(layer_idx)
    total = len(ordered_layers)
    boundary_1 = max(1, math.ceil(total / 3))
    boundary_2 = max(boundary_1 + 1, math.ceil((2 * total) / 3))
    if position < boundary_1:
        return "early"
    if position < boundary_2:
        return "mid"
    return "late"


def select_representative_layers(ordered_layers: list[int], requested_count: int) -> list[int]:
    if not ordered_layers:
        return []
    if requested_count <= 1 or len(ordered_layers) == 1:
        return [ordered_layers[0]]
    anchors = np.linspace(0, len(ordered_layers) - 1, num=min(requested_count, len(ordered_layers)))
    indices = sorted({int(round(anchor)) for anchor in anchors})
    return [ordered_layers[idx] for idx in indices]


def mass_at_top_p(energies: np.ndarray, top_p: float) -> float:
    flat = np.asarray(energies, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return 0.0
    positive = np.clip(flat, 0.0, None)
    total = positive.sum()
    if total <= 0:
        return 0.0
    k = max(1, int(math.ceil(flat.size * top_p)))
    values = np.partition(positive, -k)[-k:]
    return float(values.sum() / total)


def matrix_for_svd(tensor: np.ndarray) -> tuple[np.ndarray, str]:
    arr = np.asarray(tensor, dtype=np.float64)
    if arr.ndim == 2:
        return arr, "transformer_tc"
    if arr.ndim == 3:
        if arr.shape[0] <= 16 and arr.shape[1] > 16 and arr.shape[2] > 16:
            matrix = np.transpose(arr, (1, 2, 0)).reshape(-1, arr.shape[0])
            return matrix, "featuremap_hw_c"
        if arr.shape[-1] <= 16 and arr.shape[0] > 16 and arr.shape[1] > 16:
            matrix = arr.reshape(-1, arr.shape[-1])
            return matrix, "featuremap_hw_c"
    raise ValueError(f"Unsupported tensor shape for SVD convention: {arr.shape}")


def svd_spectrum(matrix: np.ndarray) -> tuple[list[float], dict[str, float]]:
    if matrix.size == 0:
        ratios = {f"top{rank}_explained": None for rank in TOP_RANKS}
        return [], ratios
    gram = matrix @ matrix.T
    eigvals = np.linalg.eigvalsh(gram)
    eigvals = np.clip(eigvals, 0.0, None)
    eigvals = np.sort(eigvals)[::-1]
    singular_values = np.sqrt(eigvals, dtype=np.float64).tolist()
    total = float(np.sum(eigvals))
    ratios: dict[str, float] = {}
    running = np.cumsum(eigvals)
    for rank in TOP_RANKS:
        if total <= 0:
            ratios[f"top{rank}_explained"] = 0.0
        else:
            idx = min(rank, eigvals.shape[0]) - 1
            ratios[f"top{rank}_explained"] = float(running[idx] / total)
    return singular_values, ratios


def tensor_energy_stats(
    full_tensor: np.ndarray,
    low_tensor: np.ndarray,
    delta_tensor: np.ndarray,
    patch_start: int,
    patch_hw: tuple[int, int],
    valid_patch_mask: np.ndarray,
) -> dict[str, float | list[float]]:
    full_norm = float(np.linalg.norm(full_tensor.reshape(-1)))
    delta_norm = float(np.linalg.norm(delta_tensor.reshape(-1)))
    relative_norm = delta_norm / max(full_norm, 1e-12)

    token_energy = np.sum(np.square(delta_tensor), axis=-1)
    token_norms = np.sqrt(token_energy)
    channel_energy = np.sum(np.square(delta_tensor), axis=0)
    channel_norms = np.sqrt(channel_energy)

    patch_energy = token_energy[patch_start:]
    if patch_energy.size != patch_hw[0] * patch_hw[1]:
        raise ValueError(
            f"Patch token count mismatch: got {patch_energy.size}, expected {patch_hw[0] * patch_hw[1]}"
        )
    patch_energy_map = patch_energy.reshape(patch_hw)
    valid_patch_energy = patch_energy[valid_patch_mask] if valid_patch_mask.size == patch_energy.size else patch_energy

    stats: dict[str, float | list[float]] = {
        "abs_correction_norm": delta_norm,
        "relative_correction_norm": relative_norm,
        "tokenwise_mean_correction_norm": float(np.mean(token_norms)),
        "channelwise_mean_correction_norm": float(np.mean(channel_norms)),
        "full_low_cosine_similarity": cosine_similarity(full_tensor, low_tensor),
        "layer_relative_error": relative_norm,
    }
    for top_p in TOP_PCTS:
        suffix = f"{int(round(top_p * 100)):02d}pct"
        stats[f"token_mass_top_{suffix}"] = mass_at_top_p(token_energy, top_p)
        stats[f"channel_mass_top_{suffix}"] = mass_at_top_p(channel_energy, top_p)
        stats[f"spatial_mass_top_{suffix}"] = mass_at_top_p(valid_patch_energy, top_p)
    stats["spatial_energy_map"] = patch_energy_map.astype(np.float32).tolist()
    stats["token_energy"] = token_energy.astype(np.float32).tolist()
    stats["channel_energy"] = channel_energy.astype(np.float32).tolist()
    return stats


def compute_qkv_tensors(
    block: Any,
    norm1_output: torch.Tensor,
    rope: tuple[torch.Tensor, torch.Tensor] | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    qkv = block.attn.qkv(norm1_output)
    batch_size, num_tokens, _ = qkv.shape
    embed_dim = block.attn.qkv.in_features
    qkv = qkv.reshape(batch_size, num_tokens, 3, block.attn.num_heads, embed_dim // block.attn.num_heads)
    q, k, v = torch.unbind(qkv, dim=2)
    q, k, v = [tensor_.transpose(1, 2) for tensor_ in (q, k, v)]
    if rope is not None:
        q, k = block.attn.apply_rope(q, k, rope)
    return q, k, v


def attn_output_from_qkv(block: Any, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    attn_out = F.scaled_dot_product_attention(q, k, v)
    batch_size, _, num_tokens, head_dim = attn_out.shape
    attn_out = attn_out.transpose(1, 2).reshape(batch_size, num_tokens, block.attn.num_heads * head_dim)
    attn_out = block.attn.proj(attn_out)
    attn_out = block.attn.proj_drop(attn_out)
    attn_out = block.ls1(attn_out)
    return attn_out


class Dinov3CorrectionProbe:
    def __init__(
        self,
        *,
        device: torch.device,
        image_size: int,
        model_name: str,
        backbone_weights: str | bool,
        pretrained: bool,
        model_dtype: torch.dtype,
        autocast_dtype: torch.dtype | None,
        layers: list[int] | None,
    ):
        self.device = device
        self.image_size = image_size
        self.model_name = model_name
        self.requested_model_dtype = model_dtype
        self.model_dtype = model_dtype if device.type == "cuda" else torch.float32
        self.autocast_dtype = autocast_dtype
        self.selected_layers = None if layers is None else set(layers)
        constructor_map = {
            "dinov3_vits16": dinov3_vits16,
            "dinov3_vitb16": dinov3_vitb16,
            "dinov3_vitl16": dinov3_vitl16,
            "dinov3_vit7b16": dinov3_vit7b16,
        }
        self.backbone = constructor_map[model_name](
            pretrained=pretrained,
            weights=backbone_weights,
        ).to(device=device, dtype=self.model_dtype)

        self.backbone.eval()
        self.norm_mean = IMAGENET_MEAN.to(device)
        self.norm_std = IMAGENET_STD.to(device)
        self.patch_start = 1 + int(getattr(self.backbone, "n_storage_tokens", 0))
        patch_size = getattr(self.backbone, "patch_size")
        self.patch_size = int(patch_size if isinstance(patch_size, int) else patch_size[0])

    def _autocast_context(self):
        if self.device.type != "cuda" or self.autocast_dtype is None:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=self.autocast_dtype)

    def _prepare_input(self, batch_bchw_uint8: torch.Tensor) -> torch.Tensor:
        tensor = batch_bchw_uint8.to(device=self.device, non_blocking=True).float() / 255.0
        tensor = (tensor - self.norm_mean) / self.norm_std
        return tensor

    def _selected(self, layer_idx: int) -> bool:
        return self.selected_layers is None or layer_idx in self.selected_layers

    def _collect_ffn_output(self, block: Any, x_norm2: torch.Tensor) -> torch.Tensor:
        return block.mlp(x_norm2)

    def _run_block(
        self,
        x: torch.Tensor,
        block: Any,
        rope: tuple[torch.Tensor, torch.Tensor] | None,
        *,
        collect_attention_map: bool,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor | None]]:
        x_norm1 = block.norm1(x)
        qkv = block.attn.qkv(x_norm1)
        batch_size, num_tokens, _ = qkv.shape
        embed_dim = block.attn.qkv.in_features
        qkv = qkv.reshape(batch_size, num_tokens, 3, block.attn.num_heads, embed_dim // block.attn.num_heads)
        q, k, v = torch.unbind(qkv, dim=2)
        q, k, v = [tensor_.transpose(1, 2) for tensor_ in (q, k, v)]
        if rope is not None:
            q, k = block.attn.apply_rope(q, k, rope)

        attn_map_mean = None
        if collect_attention_map:
            attn_logits = (q @ k.transpose(-2, -1)) * block.attn.scale
            attn_map_mean = torch.softmax(attn_logits.float(), dim=-1).mean(dim=1)

        attn_output = F.scaled_dot_product_attention(q, k, v)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, num_tokens, embed_dim)
        attn_output = block.attn.proj(attn_output)
        attn_output = block.attn.proj_drop(attn_output)
        attn_block_output = block.ls1(attn_output)
        x_attn = x + attn_block_output

        x_norm2 = block.norm2(x_attn)
        mlp_out = self._collect_ffn_output(block, x_norm2)
        ffn_block_output = block.ls2(mlp_out)
        x_out = x_attn + ffn_block_output
        return x_out, {
            "attn_block_output": attn_block_output,
            "ffn_block_output": ffn_block_output,
            "norm1_output": x_norm1,
            "norm2_output": x_norm2,
            "attn_map_mean": attn_map_mean,
        }

    def _norm_final_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if getattr(self.backbone, "untie_cls_and_patch_norms", False):
            x_norm_cls_reg = self.backbone.cls_norm(x[:, : self.patch_start])
            x_norm_patch = self.backbone.norm(x[:, self.patch_start :])
        else:
            x_norm = self.backbone.norm(x)
            x_norm_cls_reg = x_norm[:, : self.patch_start]
            x_norm_patch = x_norm[:, self.patch_start :]
        cls_token = x_norm_cls_reg[:, 0]
        final_tokens = torch.cat([x_norm_cls_reg, x_norm_patch], dim=1)
        pooled = torch.cat([cls_token, x_norm_patch.mean(dim=1)], dim=1)
        return final_tokens, pooled

    @torch.no_grad()
    def run(self, batch_bchw_uint8: torch.Tensor, *, collect_attention_maps: bool, collect_debug_tensors: bool) -> ProbeRun:
        tensor = self._prepare_input(batch_bchw_uint8)
        layer_outputs: dict[int, dict[str, torch.Tensor]] = {}

        with self._autocast_context():
            x, hw_tuple = self.backbone.prepare_tokens_with_masks(tensor, None)
            rope = self.backbone.rope_embed(H=hw_tuple[0], W=hw_tuple[1]) if self.backbone.rope_embed is not None else None
            patch_hw = (int(hw_tuple[0]), int(hw_tuple[1]))

            for layer_idx, block in enumerate(self.backbone.blocks):
                block_input = x
                x, internals = self._run_block(
                    x,
                    block,
                    rope,
                    collect_attention_map=collect_attention_maps,
                )

                if self._selected(layer_idx):
                    record = {
                        "block_output": x.detach().to(device="cpu", dtype=torch.float16),
                        "attn_block_output": internals["attn_block_output"].detach().to(device="cpu", dtype=torch.float16),
                        "ffn_block_output": internals["ffn_block_output"].detach().to(device="cpu", dtype=torch.float16),
                        "norm1_output": internals["norm1_output"].detach().to(device="cpu", dtype=torch.float16),
                        "norm2_output": internals["norm2_output"].detach().to(device="cpu", dtype=torch.float16),
                    }
                    if collect_debug_tensors:
                        record["block_input"] = block_input.detach().to(device="cpu", dtype=torch.float16)
                    if internals["attn_map_mean"] is not None:
                        record["attn_map_mean"] = internals["attn_map_mean"].detach().to(device="cpu", dtype=torch.float16)
                    layer_outputs[layer_idx] = record

            final_feature_tokens, pooled_output = self._norm_final_features(x)
            final_feature_tokens = final_feature_tokens.detach().to(device="cpu", dtype=torch.float32)
            pooled_output = pooled_output.detach().to(device="cpu", dtype=torch.float32)

        return ProbeRun(
            patch_start=self.patch_start,
            patch_hw=patch_hw,
            layer_outputs=layer_outputs,
            final_feature_tokens=final_feature_tokens,
            pooled_output=pooled_output,
        )

    @torch.no_grad()
    def continue_from_layer(
        self,
        hidden_state: torch.Tensor,
        *,
        layer_idx: int,
        patch_hw: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = hidden_state.to(device=self.device, dtype=self.model_dtype, non_blocking=True)
        rope = self.backbone.rope_embed(H=patch_hw[0], W=patch_hw[1]) if self.backbone.rope_embed is not None else None
        with self._autocast_context():
            for next_idx in range(layer_idx + 1, len(self.backbone.blocks)):
                x, _ = self._run_block(
                    x,
                    self.backbone.blocks[next_idx],
                    rope,
                    collect_attention_map=False,
                )
            final_feature_tokens, pooled_output = self._norm_final_features(x)
        return (
            final_feature_tokens.detach().to(device="cpu", dtype=torch.float32),
            pooled_output.detach().to(device="cpu", dtype=torch.float32),
        )


def save_debug_dump(
    output_dir: Path,
    *,
    sample_id: str,
    sample_metadata: dict[str, Any],
    full_run: ProbeRun,
    low_run: ProbeRun,
) -> None:
    dump = {
        "sample_metadata": sample_metadata,
        "full": {
            "patch_start": full_run.patch_start,
            "patch_hw": full_run.patch_hw,
            "final_feature_tokens": full_run.final_feature_tokens,
            "pooled_output": full_run.pooled_output,
            "layers": full_run.layer_outputs,
        },
        "low": {
            "patch_start": low_run.patch_start,
            "patch_hw": low_run.patch_hw,
            "final_feature_tokens": low_run.final_feature_tokens,
            "pooled_output": low_run.pooled_output,
            "layers": low_run.layer_outputs,
        },
    }
    safe_name = sample_id.replace("/", "_").replace(":", "_")
    torch.save(dump, output_dir / f"{safe_name}.pt")


def save_hidden_state_dump(
    output_dir: Path,
    *,
    sample_id: str,
    sample_metadata: dict[str, Any],
    full_run: ProbeRun,
    low_run: ProbeRun,
) -> Path:
    layers: dict[int, dict[str, torch.Tensor]] = {}
    for layer_idx in sorted(full_run.layer_outputs):
        full_block = full_run.layer_outputs[layer_idx]["block_output"].detach().cpu().to(torch.float16)
        low_block = low_run.layer_outputs[layer_idx]["block_output"].detach().cpu().to(torch.float16)
        delta_block = (full_block.float() - low_block.float()).to(torch.float16)
        layers[layer_idx] = {
            "full": full_block,
            "low": low_block,
            "delta": delta_block,
        }

    payload = {
        "sample_metadata": sample_metadata,
        "patch_start": full_run.patch_start,
        "patch_hw": full_run.patch_hw,
        "final_feature_tokens_full": full_run.final_feature_tokens.detach().cpu().to(torch.float32),
        "final_feature_tokens_low": low_run.final_feature_tokens.detach().cpu().to(torch.float32),
        "pooled_output_full": full_run.pooled_output.detach().cpu().to(torch.float32),
        "pooled_output_low": low_run.pooled_output.detach().cpu().to(torch.float32),
        "layers": layers,
    }
    safe_name = sample_id.replace("/", "_").replace(":", "_")
    output_path = output_dir / f"{safe_name}.pt"
    torch.save(payload, output_path)
    return output_path


def build_sample_id(sample_key: str, path: str) -> str:
    if path:
        return f"{sample_key}|{Path(path).name}"
    return sample_key


def build_sample_metadata(batch: dict[str, Any], batch_offset: int) -> dict[str, Any]:
    return {
        "sample_key": batch["sample_keys"][batch_offset],
        "path": batch["paths"][batch_offset],
        "label": int(batch["labels"][batch_offset]),
    }


def format_tensor_shape(array: np.ndarray) -> str:
    return "x".join(str(dim) for dim in array.shape)


def compute_layer_rows(
    *,
    split_name: str,
    sample_id: str,
    model_name: str,
    low_input_id: str,
    precision: str,
    seed: int,
    full_run: ProbeRun,
    low_run: ProbeRun,
    valid_patch_mask: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[int, np.ndarray], list[dict[str, Any]]]:
    layer_rows: list[dict[str, Any]] = []
    spectrum_rows: list[dict[str, Any]] = []
    aggregated_grams: dict[int, np.ndarray] = {}
    attention_rows: list[dict[str, Any]] = []

    full_layer_indices = sorted(full_run.layer_outputs.keys())
    prev_relative_error: float | None = None
    for layer_idx in full_layer_indices:
        full_layer = full_run.layer_outputs[layer_idx]
        low_layer = low_run.layer_outputs[layer_idx]
        full_block = full_layer["block_output"].numpy().astype(np.float32)[0]
        low_block = low_layer["block_output"].numpy().astype(np.float32)[0]
        delta_block = full_block - low_block

        metrics = tensor_energy_stats(
            full_tensor=full_block,
            low_tensor=low_block,
            delta_tensor=delta_block,
            patch_start=full_run.patch_start,
            patch_hw=full_run.patch_hw,
            valid_patch_mask=valid_patch_mask,
        )
        matrix, svd_convention = matrix_for_svd(delta_block)
        singular_values, explained = svd_spectrum(matrix)
        aggregated_grams[layer_idx] = matrix @ matrix.T

        attn_full = full_layer["attn_block_output"].numpy().astype(np.float32)[0]
        attn_low = low_layer["attn_block_output"].numpy().astype(np.float32)[0]
        ffn_full = full_layer["ffn_block_output"].numpy().astype(np.float32)[0]
        ffn_low = low_layer["ffn_block_output"].numpy().astype(np.float32)[0]
        norm1_full = full_layer["norm1_output"].numpy().astype(np.float32)[0]
        norm1_low = low_layer["norm1_output"].numpy().astype(np.float32)[0]
        norm2_full = full_layer["norm2_output"].numpy().astype(np.float32)[0]
        norm2_low = low_layer["norm2_output"].numpy().astype(np.float32)[0]

        attn_rel = relative_error(attn_full, attn_low)
        ffn_rel = relative_error(ffn_full, ffn_low)
        norm1_rel = relative_error(norm1_full, norm1_low)
        norm2_rel = relative_error(norm2_full, norm2_low)
        current_relative_error = float(metrics["relative_correction_norm"])
        delta_error = None if prev_relative_error is None else current_relative_error - prev_relative_error
        prev_relative_error = current_relative_error

        layer_row = {
            "sample_id": sample_id,
            "split_name": split_name,
            "model_name": model_name,
            "low_input_id": low_input_id,
            "layer_idx": layer_idx,
            "tensor_shape": format_tensor_shape(full_block),
            "precision": precision,
            "seed": seed,
            "svd_convention": svd_convention,
            **{key: value for key, value in metrics.items() if key not in {"spatial_energy_map", "token_energy", "channel_energy"}},
            **explained,
            "attn_relative_correction_norm": attn_rel,
            "ffn_relative_correction_norm": ffn_rel,
            "norm1_relative_correction_norm": norm1_rel,
            "norm2_relative_correction_norm": norm2_rel,
            "error_increment": delta_error,
        }
        layer_rows.append(layer_row)

        spectrum_rows.append({
            "sample_id": sample_id,
            "split_name": split_name,
            "model_name": model_name,
            "low_input_id": low_input_id,
            "layer_idx": layer_idx,
            "tensor_shape": format_tensor_shape(full_block),
            "svd_convention": svd_convention,
            "singular_values": singular_values,
            "cumulative_explained_energy": np.cumsum(np.square(np.asarray(singular_values, dtype=np.float64))).tolist()
            if singular_values
            else [],
            "top_r_explained": explained,
            "spatial_energy_map": metrics["spatial_energy_map"],
            "token_energy": metrics["token_energy"],
            "channel_energy": metrics["channel_energy"],
        })

        if "attn_map_mean" in full_layer and "attn_map_mean" in low_layer:
            attn_map_full = full_layer["attn_map_mean"].numpy().astype(np.float32)[0]
            attn_map_low = low_layer["attn_map_mean"].numpy().astype(np.float32)[0]
            attention_rows.append({
                "sample_id": sample_id,
                "split_name": split_name,
                "model_name": model_name,
                "low_input_id": low_input_id,
                "layer_idx": layer_idx,
                "attn_map_mean_l2": float(np.linalg.norm((attn_map_full - attn_map_low).reshape(-1))),
                "attn_map_mean_relative_error": relative_error(attn_map_full, attn_map_low),
                "attn_map_mean_cosine_similarity": cosine_similarity(attn_map_full, attn_map_low),
            })

    return layer_rows, spectrum_rows, aggregated_grams, attention_rows


def compute_channel_records(
    *,
    split_name: str,
    sample_id: str,
    model_name: str,
    low_input_id: str,
    precision: str,
    seed: int,
    full_run: ProbeRun,
    low_run: ProbeRun,
    hidden_state_path: Path | None,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    requested_pcts = sorted(set(TOP_PCTS) | set(CORRECTION_PCTS))
    ordered_layers = sorted(full_run.layer_outputs)
    for layer_idx in ordered_layers:
        full_block = full_run.layer_outputs[layer_idx]["block_output"].numpy().astype(np.float32)[0]
        low_block = low_run.layer_outputs[layer_idx]["block_output"].numpy().astype(np.float32)[0]
        delta_block = full_block - low_block
        channel_energy = np.sum(np.square(delta_block), axis=0).astype(np.float64)
        low_channel_norm = np.linalg.norm(low_block.astype(np.float64), axis=0)
        low_channel_abs_mean = np.mean(np.abs(low_block.astype(np.float64)), axis=0)
        low_channel_variance = np.var(low_block.astype(np.float64), axis=0)
        top_indices = {
            percentile_suffix(top_p): topk_indices_from_scores(channel_energy, top_p).tolist()
            for top_p in requested_pcts
        }
        records.append({
            "sample_id": sample_id,
            "split_name": split_name,
            "model_name": model_name,
            "low_input_id": low_input_id,
            "precision": precision,
            "seed": seed,
            "layer_idx": layer_idx,
            "depth_group": assign_depth_group(layer_idx, ordered_layers),
            "tensor_shape": format_tensor_shape(full_block),
            "num_tokens": int(full_block.shape[0]),
            "num_channels": int(full_block.shape[1]),
            "delta_norm_sq": float(channel_energy.sum()),
            "channel_energy": channel_energy.astype(np.float32).tolist(),
            "low_channel_norm": low_channel_norm.astype(np.float32).tolist(),
            "low_channel_abs_mean": low_channel_abs_mean.astype(np.float32).tolist(),
            "low_channel_variance": low_channel_variance.astype(np.float32).tolist(),
            "top_channel_indices": top_indices,
            "hidden_state_path": "" if hidden_state_path is None else str(hidden_state_path),
        })
    return records


def compute_attention_decomposition_rows(
    *,
    split_name: str,
    sample_id: str,
    model_name: str,
    low_input_id: str,
    precision: str,
    seed: int,
    probe: Dinov3CorrectionProbe,
    full_run: ProbeRun,
    low_run: ProbeRun,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rope = probe.backbone.rope_embed(H=full_run.patch_hw[0], W=full_run.patch_hw[1]) if probe.backbone.rope_embed is not None else None
    if rope is not None:
        rope = tuple(item.to(device=probe.device) for item in rope)

    for layer_idx in sorted(full_run.layer_outputs):
        full_layer = full_run.layer_outputs[layer_idx]
        low_layer = low_run.layer_outputs[layer_idx]
        block = probe.backbone.blocks[layer_idx]
        full_norm1 = full_layer["norm1_output"].to(device=probe.device, dtype=next(block.parameters()).dtype)
        low_norm1 = low_layer["norm1_output"].to(device=probe.device, dtype=next(block.parameters()).dtype)

        with torch.no_grad():
            q_full, k_full, v_full = compute_qkv_tensors(block, full_norm1, rope)
            q_low, k_low, v_low = compute_qkv_tensors(block, low_norm1, rope)
            out_low = attn_output_from_qkv(block, q_low, k_low, v_low)
            out_v_only = attn_output_from_qkv(block, q_low, k_low, v_full)
            out_qk_only = attn_output_from_qkv(block, q_full, k_full, v_low)
            out_full = attn_output_from_qkv(block, q_full, k_full, v_full)

        out_low_np = out_low.detach().float().cpu().numpy()[0]
        out_v_only_np = out_v_only.detach().float().cpu().numpy()[0]
        out_qk_only_np = out_qk_only.detach().float().cpu().numpy()[0]
        out_full_np = out_full.detach().float().cpu().numpy()[0]
        baseline_distance = relative_error(out_full_np, out_low_np)
        v_only_distance = relative_error(out_full_np, out_v_only_np)
        qk_only_distance = relative_error(out_full_np, out_qk_only_np)
        rows.append({
            "sample_id": sample_id,
            "split_name": split_name,
            "model_name": model_name,
            "low_input_id": low_input_id,
            "layer_idx": layer_idx,
            "precision": precision,
            "seed": seed,
            "attn_low_to_full_relative_error": baseline_distance,
            "attn_v_only_to_full_relative_error": v_only_distance,
            "attn_qk_only_to_full_relative_error": qk_only_distance,
            "attn_v_only_improvement": None if baseline_distance <= 0 else 1.0 - (v_only_distance / baseline_distance),
            "attn_qk_only_improvement": None if baseline_distance <= 0 else 1.0 - (qk_only_distance / baseline_distance),
        })
    return rows


def compute_final_row(
    *,
    split_name: str,
    sample_id: str,
    model_name: str,
    low_input_id: str,
    precision: str,
    seed: int,
    full_run: ProbeRun,
    low_run: ProbeRun,
) -> dict[str, Any]:
    full_tokens = full_run.final_feature_tokens.numpy().astype(np.float32)[0]
    low_tokens = low_run.final_feature_tokens.numpy().astype(np.float32)[0]
    full_pooled = full_run.pooled_output.numpy().astype(np.float32)[0]
    low_pooled = low_run.pooled_output.numpy().astype(np.float32)[0]
    full_cls = full_tokens[0]
    low_cls = low_tokens[0]
    full_patch = full_tokens[full_run.patch_start :]
    low_patch = low_tokens[low_run.patch_start :]
    row = {
        "sample_id": sample_id,
        "split_name": split_name,
        "model_name": model_name,
        "low_input_id": low_input_id,
        "precision": precision,
        "seed": seed,
        "final_feature_tokens_l2": float(np.linalg.norm((full_tokens - low_tokens).reshape(-1))),
        "final_feature_tokens_relative_error": relative_error(full_tokens, low_tokens),
        "final_feature_tokens_cosine_similarity": cosine_similarity(full_tokens, low_tokens),
        "final_cls_token_l2": float(np.linalg.norm((full_cls - low_cls).reshape(-1))),
        "final_cls_token_relative_error": relative_error(full_cls, low_cls),
        "final_cls_token_cosine_similarity": cosine_similarity(full_cls, low_cls),
        "final_patch_tokens_l2": float(np.linalg.norm((full_patch - low_patch).reshape(-1))),
        "final_patch_tokens_relative_error": relative_error(full_patch, low_patch),
        "final_patch_tokens_cosine_similarity": cosine_similarity(full_patch, low_patch),
        "pooled_output_l2": float(np.linalg.norm((full_pooled - low_pooled).reshape(-1))),
        "pooled_output_relative_error": relative_error(full_pooled, low_pooled),
        "pooled_output_cosine_similarity": cosine_similarity(full_pooled, low_pooled),
    }
    return row


def compare_final_representations(
    *,
    sample_id: str,
    split_name: str,
    model_name: str,
    low_input_id: str,
    precision: str,
    seed: int,
    layer_idx: int,
    depth_group: str,
    method: str,
    pct: float,
    patch_start: int,
    final_feature_tokens_full: np.ndarray,
    final_feature_tokens_low: np.ndarray,
    pooled_output_full: np.ndarray,
    pooled_output_low: np.ndarray,
    final_feature_tokens_hat: np.ndarray,
    pooled_output_hat: np.ndarray,
) -> dict[str, Any]:
    suffix = percentile_suffix(pct)
    full_cls = final_feature_tokens_full[0]
    low_cls = final_feature_tokens_low[0]
    hat_cls = final_feature_tokens_hat[0]
    full_patch = final_feature_tokens_full[patch_start:]
    low_patch = final_feature_tokens_low[patch_start:]
    hat_patch = final_feature_tokens_hat[patch_start:]

    low_ref_distance = relative_error(final_feature_tokens_full, final_feature_tokens_low)
    hat_ref_distance = relative_error(final_feature_tokens_full, final_feature_tokens_hat)
    low_cls_distance = relative_error(full_cls, low_cls)
    hat_cls_distance = relative_error(full_cls, hat_cls)
    low_patch_distance = relative_error(full_patch, low_patch)
    hat_patch_distance = relative_error(full_patch, hat_patch)
    low_pooled_distance = relative_error(pooled_output_full, pooled_output_low)
    hat_pooled_distance = relative_error(pooled_output_full, pooled_output_hat)

    return {
        "sample_id": sample_id,
        "split_name": split_name,
        "model_name": model_name,
        "low_input_id": low_input_id,
        "precision": precision,
        "seed": seed,
        "layer_idx": layer_idx,
        "depth_group": depth_group,
        "method": method,
        "pct": pct,
        "pct_label": suffix,
        "final_feature_tokens_relative_error_to_full": hat_ref_distance,
        "final_feature_tokens_cosine_similarity_to_full": cosine_similarity(final_feature_tokens_full, final_feature_tokens_hat),
        "final_cls_token_relative_error_to_full": hat_cls_distance,
        "final_cls_token_cosine_similarity_to_full": cosine_similarity(full_cls, hat_cls),
        "final_patch_tokens_relative_error_to_full": hat_patch_distance,
        "final_patch_tokens_cosine_similarity_to_full": cosine_similarity(full_patch, hat_patch),
        "pooled_output_relative_error_to_full": hat_pooled_distance,
        "pooled_output_cosine_similarity_to_full": cosine_similarity(pooled_output_full, pooled_output_hat),
        "final_feature_tokens_recovery_vs_low": 1.0 - safe_divide(hat_ref_distance, low_ref_distance, default=1.0),
        "final_cls_token_recovery_vs_low": 1.0 - safe_divide(hat_cls_distance, low_cls_distance, default=1.0),
        "final_patch_tokens_recovery_vs_low": 1.0 - safe_divide(hat_patch_distance, low_patch_distance, default=1.0),
        "pooled_output_recovery_vs_low": 1.0 - safe_divide(hat_pooled_distance, low_pooled_distance, default=1.0),
    }


def build_fixed_subsets(channel_records: list[dict[str, Any]]) -> dict[tuple[int, str], list[int]]:
    grouped: dict[tuple[int, str], Counter[int]] = defaultdict(Counter)
    num_channels_by_layer: dict[int, int] = {}
    for record in channel_records:
        layer_idx = int(record["layer_idx"])
        num_channels_by_layer[layer_idx] = int(record["num_channels"])
        top_channel_indices = record["top_channel_indices"]
        for top_p in CORRECTION_PCTS:
            suffix = percentile_suffix(top_p)
            grouped[(layer_idx, suffix)].update(int(value) for value in top_channel_indices[suffix])

    fixed_subsets: dict[tuple[int, str], list[int]] = {}
    for (layer_idx, suffix), counter in grouped.items():
        pct = next(top_p for top_p in CORRECTION_PCTS if percentile_suffix(top_p) == suffix)
        k = topk_count(num_channels_by_layer[layer_idx], pct)
        fixed_subsets[(layer_idx, suffix)] = [channel for channel, _ in counter.most_common(k)]
    return fixed_subsets


def compute_overlap_analysis(channel_records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    overlap_rows: list[dict[str, Any]] = []
    frequency_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    grouped_by_layer: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in channel_records:
        grouped_by_layer[int(record["layer_idx"])].append(record)

    for layer_idx in sorted(grouped_by_layer):
        records = grouped_by_layer[layer_idx]
        depth_group = str(records[0]["depth_group"])
        num_channels = int(records[0]["num_channels"])
        for top_p in TOP_PCTS:
            suffix = percentile_suffix(top_p)
            index_sets = [np.asarray(record["top_channel_indices"][suffix], dtype=np.int64) for record in records]
            mean_jaccard = mean_pairwise_jaccard(index_sets)
            counter: Counter[int] = Counter()
            for indices in index_sets:
                counter.update(int(value) for value in indices.tolist())
            total_occurrences = sum(counter.values())
            k = topk_count(num_channels, top_p)
            consensus_channels = [channel for channel, _ in counter.most_common(k)]
            consensus_recall_values = []
            consensus_set = set(consensus_channels)
            for indices in index_sets:
                oracle_set = set(int(value) for value in indices.tolist())
                consensus_recall_values.append(safe_divide(len(consensus_set & oracle_set), len(oracle_set)))

            overlap_rows.append({
                "layer_idx": layer_idx,
                "depth_group": depth_group,
                "pct": top_p,
                "pct_label": suffix,
                "num_samples": len(records),
                "topk_channels": k,
                "average_jaccard": mean_jaccard,
                "consensus_recall_at_k": float(np.mean(consensus_recall_values)) if consensus_recall_values else None,
                "unique_top_channels": len(counter),
            })

            running = 0
            for rank, (channel_idx, freq) in enumerate(counter.most_common(), start=1):
                running += freq
                frequency_rows.append({
                    "layer_idx": layer_idx,
                    "depth_group": depth_group,
                    "pct": top_p,
                    "pct_label": suffix,
                    "channel_idx": channel_idx,
                    "frequency": freq,
                    "rank": rank,
                })
                coverage_rows.append({
                    "layer_idx": layer_idx,
                    "depth_group": depth_group,
                    "pct": top_p,
                    "pct_label": suffix,
                    "topk_frequency_rank": rank,
                    "cumulative_coverage": safe_divide(running, total_occurrences),
                    "total_occurrences": total_occurrences,
                })
    return overlap_rows, frequency_rows, coverage_rows


def compute_channel_policy_metrics(
    *,
    channel_records: list[dict[str, Any]],
    fixed_subsets: dict[tuple[int, str], list[int]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in channel_records:
        layer_idx = int(record["layer_idx"])
        depth_group = str(record["depth_group"])
        num_channels = int(record["num_channels"])
        channel_energy = np.asarray(record["channel_energy"], dtype=np.float64)
        low_channel_norm = np.asarray(record["low_channel_norm"], dtype=np.float64)
        total_energy = float(channel_energy.sum())
        for top_p in CORRECTION_PCTS:
            suffix = percentile_suffix(top_p)
            oracle_indices = np.asarray(record["top_channel_indices"][suffix], dtype=np.int64)
            fixed_indices = np.asarray(fixed_subsets.get((layer_idx, suffix), []), dtype=np.int64)
            predictor_indices = topk_indices_from_scores(low_channel_norm, top_p)

            method_to_indices = {
                "oracle": oracle_indices,
                "fixed": fixed_indices,
                "predictor": predictor_indices,
            }
            oracle_set = set(int(value) for value in oracle_indices.tolist())
            for method, selected_indices in method_to_indices.items():
                if method == "fixed" and selected_indices.size == 0:
                    continue
                k = topk_count(num_channels, top_p)
                selected_set = set(int(value) for value in selected_indices.tolist())
                kept_energy = float(channel_energy[selected_indices].sum()) if selected_indices.size else 0.0
                energy_recovery = safe_divide(kept_energy, total_energy)
                truncation_error = float(math.sqrt(max(0.0, 1.0 - energy_recovery)))
                intersection = len(selected_set & oracle_set)
                union = len(selected_set | oracle_set)
                rows.append({
                    "sample_id": record["sample_id"],
                    "split_name": record["split_name"],
                    "model_name": record["model_name"],
                    "low_input_id": record["low_input_id"],
                    "precision": record["precision"],
                    "seed": record["seed"],
                    "layer_idx": layer_idx,
                    "depth_group": depth_group,
                    "pct": top_p,
                    "pct_label": suffix,
                    "method": method,
                    "topk_channels": k,
                    "energy_recovery": energy_recovery,
                    "truncation_error": truncation_error,
                    "precision_at_k": safe_divide(intersection, len(selected_set)),
                    "recall_at_k": safe_divide(intersection, len(oracle_set)),
                    "jaccard_with_oracle": safe_divide(intersection, union, default=1.0),
                })
    return rows


def summarize_policy_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (int(row["layer_idx"]), str(row["depth_group"]), str(row["pct_label"]), str(row["method"]))
        for field in ("energy_recovery", "truncation_error", "precision_at_k", "recall_at_k", "jaccard_with_oracle"):
            value = row.get(field)
            if value is not None:
                grouped[key][field].append(float(value))

    summary_rows: list[dict[str, Any]] = []
    for (layer_idx, depth_group, pct_label, method), metrics in sorted(grouped.items()):
        pct = next(top_p for top_p in CORRECTION_PCTS if percentile_suffix(top_p) == pct_label)
        row = {
            "layer_idx": layer_idx,
            "depth_group": depth_group,
            "pct": pct,
            "pct_label": pct_label,
            "method": method,
        }
        for field, values in metrics.items():
            stats = summarize_numeric(values)
            for stat_name, stat_value in stats.items():
                row[f"{field}_{stat_name}"] = stat_value
        summary_rows.append(row)
    return summary_rows


def summarize_overlap_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    return rows


def summarize_depth_group_rows(
    overlap_rows: list[dict[str, Any]],
    policy_rows: list[dict[str, Any]],
    downstream_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    overlap_grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in overlap_rows:
        if row["average_jaccard"] is not None:
            overlap_grouped[(str(row["depth_group"]), str(row["pct_label"]))].append(float(row["average_jaccard"]))
    for (depth_group, pct_label), values in sorted(overlap_grouped.items()):
        summary_rows.append({
            "depth_group": depth_group,
            "pct_label": pct_label,
            "section": "overlap",
            "metric_name": "average_jaccard",
            **summarize_numeric(values),
        })

    policy_grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in policy_rows:
        for field in ("energy_recovery", "truncation_error", "recall_at_k", "jaccard_with_oracle"):
            value = row.get(field)
            if value is not None:
                policy_grouped[(str(row["depth_group"]), str(row["pct_label"]), str(row["method"]), field)].append(float(value))
    for (depth_group, pct_label, method, field), values in sorted(policy_grouped.items()):
        summary_rows.append({
            "depth_group": depth_group,
            "pct_label": pct_label,
            "section": "policy",
            "method": method,
            "metric_name": field,
            **summarize_numeric(values),
        })

    downstream_grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in downstream_rows:
        for field in (
            "final_feature_tokens_recovery_vs_low",
            "final_cls_token_recovery_vs_low",
            "pooled_output_recovery_vs_low",
        ):
            value = row.get(field)
            if value is not None:
                downstream_grouped[(str(row["depth_group"]), str(row["pct_label"]), str(row["method"]), field)].append(float(value))
    for (depth_group, pct_label, method, field), values in sorted(downstream_grouped.items()):
        summary_rows.append({
            "depth_group": depth_group,
            "pct_label": pct_label,
            "section": "downstream",
            "method": method,
            "metric_name": field,
            **summarize_numeric(values),
        })
    return summary_rows


def select_downstream_layers(
    channel_records: list[dict[str, Any]],
    *,
    run_mode: str,
    representative_count: int,
) -> list[int]:
    ordered_layers = sorted({int(record["layer_idx"]) for record in channel_records})
    if run_mode == "debug":
        return select_representative_layers(ordered_layers, representative_count)
    return ordered_layers


def build_masked_hidden(
    low_hidden: torch.Tensor,
    delta_hidden: torch.Tensor,
    selected_indices: np.ndarray,
) -> torch.Tensor:
    masked = torch.zeros_like(delta_hidden, dtype=torch.float32)
    if selected_indices.size:
        masked[..., selected_indices.tolist()] = delta_hidden.float()[..., selected_indices.tolist()]
    return low_hidden.float() + masked


def compute_downstream_recovery_rows(
    *,
    probe: Dinov3CorrectionProbe,
    channel_records: list[dict[str, Any]],
    fixed_subsets: dict[tuple[int, str], list[int]],
    methods: tuple[str, ...],
    run_mode: str,
    representative_count: int,
) -> list[dict[str, Any]]:
    if not methods or not channel_records:
        return []

    records_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in channel_records:
        hidden_state_path = str(record["hidden_state_path"])
        if not hidden_state_path:
            continue
        records_by_sample[hidden_state_path].append(record)

    downstream_rows: list[dict[str, Any]] = []
    for hidden_state_path, sample_records in tqdm(
        sorted(records_by_sample.items()),
        desc="downstream_recovery",
        unit="sample",
        leave=False,
    ):
        payload = torch.load(hidden_state_path, map_location="cpu")
        selected_layers = set(select_downstream_layers(sample_records, run_mode=run_mode, representative_count=representative_count))
        final_feature_tokens_full = payload["final_feature_tokens_full"].numpy().astype(np.float32)[0]
        final_feature_tokens_low = payload["final_feature_tokens_low"].numpy().astype(np.float32)[0]
        pooled_output_full = payload["pooled_output_full"].numpy().astype(np.float32)[0]
        pooled_output_low = payload["pooled_output_low"].numpy().astype(np.float32)[0]
        patch_start = int(payload["patch_start"])
        patch_hw = tuple(int(value) for value in payload["patch_hw"])
        record_by_layer = {int(record["layer_idx"]): record for record in sample_records}

        for layer_idx in sorted(selected_layers):
            record = record_by_layer[layer_idx]
            layer_payload = payload["layers"][layer_idx]
            low_hidden = layer_payload["low"]
            delta_hidden = layer_payload["delta"]
            channel_energy = np.asarray(record["channel_energy"], dtype=np.float64)
            low_channel_norm = np.asarray(record["low_channel_norm"], dtype=np.float64)
            for top_p in CORRECTION_PCTS:
                suffix = percentile_suffix(top_p)
                candidate_sets: dict[str, np.ndarray] = {}
                if "oracle" in methods:
                    candidate_sets["oracle"] = np.asarray(record["top_channel_indices"][suffix], dtype=np.int64)
                if "fixed" in methods:
                    candidate_sets["fixed"] = np.asarray(fixed_subsets.get((layer_idx, suffix), []), dtype=np.int64)
                if "predictor" in methods:
                    candidate_sets["predictor"] = topk_indices_from_scores(low_channel_norm, top_p)

                for method, indices in candidate_sets.items():
                    if method == "fixed" and indices.size == 0:
                        continue
                    h_hat = build_masked_hidden(low_hidden, delta_hidden, indices)
                    final_feature_tokens_hat, pooled_output_hat = probe.continue_from_layer(
                        h_hat,
                        layer_idx=layer_idx,
                        patch_hw=patch_hw,
                    )
                    downstream_rows.append(compare_final_representations(
                        sample_id=str(record["sample_id"]),
                        split_name=str(record["split_name"]),
                        model_name=str(record["model_name"]),
                        low_input_id=str(record["low_input_id"]),
                        precision=str(record["precision"]),
                        seed=int(record["seed"]),
                        layer_idx=layer_idx,
                        depth_group=str(record["depth_group"]),
                        method=method,
                        pct=top_p,
                        patch_start=patch_start,
                        final_feature_tokens_full=final_feature_tokens_full,
                        final_feature_tokens_low=final_feature_tokens_low,
                        pooled_output_full=pooled_output_full,
                        pooled_output_low=pooled_output_low,
                        final_feature_tokens_hat=final_feature_tokens_hat.numpy().astype(np.float32)[0],
                        pooled_output_hat=pooled_output_hat.numpy().astype(np.float32)[0],
                    ))
    return downstream_rows


def summarize_downstream_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (int(row["layer_idx"]), str(row["depth_group"]), str(row["pct_label"]), str(row["method"]))
        for field in (
            "final_feature_tokens_relative_error_to_full",
            "final_feature_tokens_cosine_similarity_to_full",
            "final_cls_token_relative_error_to_full",
            "final_cls_token_cosine_similarity_to_full",
            "pooled_output_relative_error_to_full",
            "pooled_output_cosine_similarity_to_full",
            "final_feature_tokens_recovery_vs_low",
            "final_cls_token_recovery_vs_low",
            "pooled_output_recovery_vs_low",
        ):
            value = row.get(field)
            if value is not None:
                grouped[key][field].append(float(value))

    summary_rows: list[dict[str, Any]] = []
    for (layer_idx, depth_group, pct_label, method), metrics in sorted(grouped.items()):
        pct = next(top_p for top_p in CORRECTION_PCTS if percentile_suffix(top_p) == pct_label)
        row = {
            "layer_idx": layer_idx,
            "depth_group": depth_group,
            "pct": pct,
            "pct_label": pct_label,
            "method": method,
        }
        for field, values in metrics.items():
            stats = summarize_numeric(values)
            for stat_name, stat_value in stats.items():
                row[f"{field}_{stat_name}"] = stat_value
        summary_rows.append(row)
    return summary_rows


def summarize_by_layer(rows: list[dict[str, Any]], value_fields: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[float]] = defaultdict(list)
    for row in rows:
        for field in value_fields:
            value = row.get(field)
            if value is not None:
                grouped[(int(row["layer_idx"]), field)].append(float(value))

    summary_rows = []
    for (layer_idx, field_name), values in sorted(grouped.items()):
        stats = summarize_numeric(values)
        summary_rows.append({
            "layer_idx": layer_idx,
            "metric_name": field_name,
            **stats,
        })
    return summary_rows


def summarize_final_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    numeric_fields = [
        key for key in rows[0].keys()
        if key not in {"sample_id", "split_name", "model_name", "low_input_id", "precision", "seed"}
    ]
    summary = []
    for field in numeric_fields:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        summary.append({
            "metric_name": field,
            **summarize_numeric(values),
        })
    return summary


def compute_correlations(layer_rows: list[dict[str, Any]], final_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not layer_rows or not final_rows:
        return []
    final_by_sample = {row["sample_id"]: row for row in final_rows}
    excluded_layer_fields = {"sample_id", "split_name", "model_name", "low_input_id", "layer_idx", "tensor_shape", "precision", "seed", "svd_convention"}
    excluded_final_fields = {"sample_id", "split_name", "model_name", "low_input_id", "precision", "seed"}

    layer_metric_fields = [key for key in layer_rows[0].keys() if key not in excluded_layer_fields]
    final_metric_fields = [key for key in final_rows[0].keys() if key not in excluded_final_fields]

    results: list[dict[str, Any]] = []
    for layer_idx in sorted({int(row["layer_idx"]) for row in layer_rows}):
        subset = [row for row in layer_rows if int(row["layer_idx"]) == layer_idx and row["sample_id"] in final_by_sample]
        for layer_field in layer_metric_fields:
            x_values = []
            sample_ids = []
            for row in subset:
                value = row.get(layer_field)
                if value is None:
                    continue
                x_values.append(float(value))
                sample_ids.append(row["sample_id"])
            if len(x_values) < 2:
                continue
            for final_field in final_metric_fields:
                y_values = []
                filtered_x = []
                for sample_id, x_value in zip(sample_ids, x_values):
                    final_value = final_by_sample[sample_id].get(final_field)
                    if final_value is None:
                        continue
                    filtered_x.append(x_value)
                    y_values.append(float(final_value))
                if len(filtered_x) < 2:
                    continue
                results.append({
                    "layer_idx": layer_idx,
                    "layer_metric": layer_field,
                    "final_metric": final_field,
                    "num_samples": len(filtered_x),
                    "pearson": pearson_corr(filtered_x, y_values),
                    "spearman": spearman_corr(filtered_x, y_values),
                })
    return results


def aggregated_spectrum_rows(gram_store: dict[int, list[np.ndarray]]) -> list[dict[str, Any]]:
    rows = []
    for layer_idx in sorted(gram_store):
        gram_list = gram_store[layer_idx]
        if not gram_list:
            continue
        mean_gram = np.mean(np.stack(gram_list, axis=0), axis=0)
        eigvals = np.linalg.eigvalsh(mean_gram)
        eigvals = np.clip(eigvals, 0.0, None)
        eigvals = np.sort(eigvals)[::-1]
        singular_values = np.sqrt(eigvals, dtype=np.float64).tolist()
        total = float(np.sum(eigvals))
        running = np.cumsum(eigvals)
        row = {
            "layer_idx": layer_idx,
            "aggregation_method": "mean_row_gram",
            "singular_values": singular_values,
            "cumulative_explained_energy": (running / max(total, 1e-12)).tolist() if eigvals.size else [],
        }
        for rank in TOP_RANKS:
            if eigvals.size == 0 or total <= 0:
                row[f"top{rank}_explained"] = 0.0
            else:
                idx = min(rank, eigvals.shape[0]) - 1
                row[f"top{rank}_explained"] = float(running[idx] / total)
        rows.append(row)
    return rows


def write_plots(
    output_dir: Path,
    *,
    layer_summary_rows: list[dict[str, Any]],
    attention_summary_rows: list[dict[str, Any]],
) -> None:
    if not layer_summary_rows and not attention_summary_rows:
        return

    os.environ.setdefault("MPLCONFIGDIR", str((output_dir / ".mplconfig").resolve()))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "plotting_skipped.txt").write_text(
            f"Plot generation skipped because matplotlib import failed: {exc}\n",
            encoding="utf-8",
        )
        return

    def build_metric_index(rows: list[dict[str, Any]]) -> dict[str, dict[str, list[float]]]:
        index: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"layer_idx": [], "mean": [], "q25": [], "q75": []})
        for row in rows:
            name = str(row["metric_name"])
            index[name]["layer_idx"].append(int(row["layer_idx"]))
            index[name]["mean"].append(float(row["mean"]) if row["mean"] is not None else np.nan)
            index[name]["q25"].append(float(row["q25"]) if row["q25"] is not None else np.nan)
            index[name]["q75"].append(float(row["q75"]) if row["q75"] is not None else np.nan)
        return index

    layer_index = build_metric_index(layer_summary_rows)
    attn_index = build_metric_index(attention_summary_rows)

    def plot_metric_lines(index: dict[str, dict[str, list[float]]], metric_names: list[str], title: str, ylabel: str, filename: str) -> None:
        available = [name for name in metric_names if name in index]
        if not available:
            return
        fig, ax = plt.subplots(figsize=(11, 6), dpi=180)
        palette = ["#175676", "#D62839", "#F77F00", "#2A9D8F", "#6A4C93", "#5C677D"]
        for color, metric_name in zip(palette, available):
            record = index[metric_name]
            xs = np.asarray(record["layer_idx"], dtype=np.int32)
            order = np.argsort(xs)
            xs = xs[order]
            mean = np.asarray(record["mean"], dtype=np.float64)[order]
            q25 = np.asarray(record["q25"], dtype=np.float64)[order]
            q75 = np.asarray(record["q75"], dtype=np.float64)[order]
            ax.plot(xs, mean, linewidth=2.0, label=metric_name, color=color)
            ax.fill_between(xs, q25, q75, alpha=0.18, color=color)
        ax.set_title(title)
        ax.set_xlabel("Layer depth")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.savefig(output_dir / filename, bbox_inches="tight")
        plt.close(fig)

    plot_metric_lines(
        layer_index,
        ["relative_correction_norm"],
        "Layerwise Relative Correction Norm",
        "Relative correction norm",
        "layerwise_relative_correction_norm.png",
    )
    plot_metric_lines(
        layer_index,
        [f"top{rank}_explained" for rank in TOP_RANKS],
        "Layerwise Top-r Explained Energy",
        "Explained energy",
        "layerwise_topr_explained_energy.png",
    )
    plot_metric_lines(
        layer_index,
        ["spatial_mass_top_10pct", "token_mass_top_10pct", "channel_mass_top_10pct"],
        "Layerwise Sparsity Summary",
        "Mass captured by top 10%",
        "layerwise_sparsity_summary.png",
    )
    plot_metric_lines(
        layer_index,
        [
            "attn_relative_correction_norm",
            "ffn_relative_correction_norm",
            "norm1_relative_correction_norm",
            "norm2_relative_correction_norm",
        ],
        "Attention vs FFN vs Norm Deviation",
        "Relative deviation",
        "block_contribution_summary.png",
    )
    plot_metric_lines(
        attn_index,
        [
            "attn_low_to_full_relative_error",
            "attn_v_only_to_full_relative_error",
            "attn_qk_only_to_full_relative_error",
        ],
        "Attention Local Intervention Summary",
        "Relative distance to full attention output",
        "attention_decomposition_summary.png",
    )


def write_channel_policy_plots(
    output_dir: Path,
    *,
    overlap_rows: list[dict[str, Any]],
    policy_summary_rows: list[dict[str, Any]],
    frequency_rows: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
    depth_summary_rows: list[dict[str, Any]],
    downstream_summary_rows: list[dict[str, Any]],
) -> None:
    if not any((overlap_rows, policy_summary_rows, frequency_rows, depth_summary_rows, downstream_summary_rows)):
        return

    os.environ.setdefault("MPLCONFIGDIR", str((output_dir / ".mplconfig").resolve()))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "plotting_skipped.txt").write_text(
            f"Plot generation skipped because matplotlib import failed: {exc}\n",
            encoding="utf-8",
        )
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    def policy_index(method: str, metric_prefix: str) -> dict[str, tuple[list[int], list[float]]]:
        index: dict[str, tuple[list[int], list[float]]] = {}
        for top_p in CORRECTION_PCTS:
            suffix = percentile_suffix(top_p)
            subset = [
                row for row in policy_summary_rows
                if row["method"] == method and row["pct_label"] == suffix and row.get(f"{metric_prefix}_mean") is not None
            ]
            subset.sort(key=lambda row: int(row["layer_idx"]))
            index[suffix] = (
                [int(row["layer_idx"]) for row in subset],
                [float(row[f"{metric_prefix}_mean"]) for row in subset],
            )
        return index

    def overlap_index() -> dict[str, tuple[list[int], list[float]]]:
        index: dict[str, tuple[list[int], list[float]]] = {}
        for top_p in TOP_PCTS:
            suffix = percentile_suffix(top_p)
            subset = [row for row in overlap_rows if row["pct_label"] == suffix and row["average_jaccard"] is not None]
            subset.sort(key=lambda row: int(row["layer_idx"]))
            index[suffix] = (
                [int(row["layer_idx"]) for row in subset],
                [float(row["average_jaccard"]) for row in subset],
            )
        return index

    def plot_lines(index: dict[str, tuple[list[int], list[float]]], title: str, ylabel: str, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(11, 6), dpi=180)
        palette = ["#175676", "#D62839", "#F77F00", "#2A9D8F", "#6A4C93", "#5C677D"]
        plotted = False
        for color, (suffix, (xs, ys)) in zip(palette, sorted(index.items())):
            if not xs:
                continue
            ax.plot(xs, ys, linewidth=2.0, label=suffix, color=color)
            plotted = True
        if not plotted:
            plt.close(fig)
            return
        ax.set_title(title)
        ax.set_xlabel("Layer depth")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, title="Top-p")
        fig.savefig(output_dir / filename, bbox_inches="tight")
        plt.close(fig)

    plot_lines(overlap_index(), "Layer vs Average Jaccard@p", "Average Jaccard", "layer_vs_average_jaccard.png")
    plot_lines(policy_index("oracle", "energy_recovery"), "Layer vs Oracle Energy Recovery@p", "Energy recovery", "layer_vs_oracle_energy_recovery.png")
    plot_lines(policy_index("fixed", "energy_recovery"), "Layer vs Fixed-Subset Energy Recovery@p", "Energy recovery", "layer_vs_fixed_energy_recovery.png")
    plot_lines(policy_index("predictor", "recall_at_k"), "Layer vs Predictor Recall@p", "Recall@k", "layer_vs_predictor_recall.png")

    oracle_index = policy_index("oracle", "energy_recovery")
    fixed_index = policy_index("fixed", "energy_recovery")
    gap_index: dict[str, tuple[list[int], list[float]]] = {}
    for suffix in oracle_index:
        oracle_xs, oracle_ys = oracle_index[suffix]
        fixed_xs, fixed_ys = fixed_index.get(suffix, ([], []))
        if oracle_xs != fixed_xs:
            continue
        gap_index[suffix] = (oracle_xs, [oracle - fixed for oracle, fixed in zip(oracle_ys, fixed_ys)])
    plot_lines(gap_index, "Layer vs Oracle-Fixed Energy Gap", "Energy gap", "layer_vs_oracle_minus_fixed_gap.png")

    representative_layers = []
    for depth_group in DEPTH_GROUP_NAMES:
        candidates = sorted({int(row["layer_idx"]) for row in frequency_rows if row["depth_group"] == depth_group and row["pct_label"] == "10pct"})
        if candidates:
            representative_layers.append(candidates[len(candidates) // 2])
    representative_layers = sorted(set(representative_layers))
    if representative_layers:
        fig, axes = plt.subplots(len(representative_layers), 1, figsize=(11, 3.5 * len(representative_layers)), dpi=180)
        if len(representative_layers) == 1:
            axes = [axes]
        for ax, layer_idx in zip(axes, representative_layers):
            subset = [row for row in frequency_rows if int(row["layer_idx"]) == layer_idx and row["pct_label"] == "10pct"][:50]
            ax.bar([int(row["channel_idx"]) for row in subset], [int(row["frequency"]) for row in subset], color="#175676")
            ax.set_title(f"Layer {layer_idx} Channel Frequency Histogram @10pct")
            ax.set_xlabel("Channel index")
            ax.set_ylabel("Top-set frequency")
        fig.tight_layout()
        fig.savefig(output_dir / "representative_layer_channel_frequency_histograms.png", bbox_inches="tight")
        plt.close(fig)

    grouped_depth = [
        row for row in depth_summary_rows
        if row["section"] == "policy" and row.get("method") in {"oracle", "fixed", "predictor"} and row["metric_name"] == "energy_recovery"
    ]
    if grouped_depth:
        fig, ax = plt.subplots(figsize=(11, 6), dpi=180)
        colors = {"early": "#175676", "mid": "#D62839", "late": "#2A9D8F"}
        for depth_group in DEPTH_GROUP_NAMES:
            subset = [row for row in grouped_depth if row["depth_group"] == depth_group and row.get("method") == "oracle" and row["mean"] is not None]
            subset.sort(key=lambda row: row["pct_label"])
            if not subset:
                continue
            ax.plot(
                [row["pct_label"] for row in subset],
                [float(row["mean"]) for row in subset],
                marker="o",
                linewidth=2.0,
                color=colors[depth_group],
                label=depth_group,
            )
        ax.set_title("Early/Mid/Late Oracle Energy Recovery Summary")
        ax.set_xlabel("Top-p")
        ax.set_ylabel("Mean energy recovery")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.savefig(output_dir / "depth_grouped_summaries.png", bbox_inches="tight")
        plt.close(fig)

    if downstream_summary_rows:
        downstream_index: dict[str, tuple[list[int], list[float]]] = {}
        for top_p in CORRECTION_PCTS:
            suffix = percentile_suffix(top_p)
            subset = [
                row for row in downstream_summary_rows
                if row["method"] == "oracle"
                and row["pct_label"] == suffix
                and row.get("final_feature_tokens_recovery_vs_low_mean") is not None
            ]
            subset.sort(key=lambda row: int(row["layer_idx"]))
            downstream_index[suffix] = (
                [int(row["layer_idx"]) for row in subset],
                [float(row["final_feature_tokens_recovery_vs_low_mean"]) for row in subset],
            )
        plot_lines(
            downstream_index,
            "Layer vs Oracle Downstream Recovery@p",
            "Final embedding recovery vs low",
            "layer_vs_oracle_downstream_recovery.png",
        )


def create_key_results_zip(split_output_dir: Path) -> Path:
    key_files = [
        "config.json",
        "layer_summary.csv",
        "channel_overlap_summary.csv",
        "channel_policy_summary.csv",
        "downstream_recovery_summary.csv",
        "depth_group_summary.csv",
        "aggregated_spectrum.jsonl",
        "attention_decomposition_summary.csv",
        "final_output_correlations.csv",
    ]
    zip_path = split_output_dir / f"{split_output_dir.name}_key_results.zip"
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename in key_files:
            file_path = split_output_dir / filename
            if file_path.exists():
                archive.write(file_path, arcname=filename)
    return zip_path


def create_report_bundle_zip(split_output_dir: Path) -> Path:
    key_zip_name = f"{split_output_dir.name}_key_results.zip"
    report_files = [
        "config.json",
        "channel_overlap_summary.csv",
        "channel_policy_summary.csv",
        "downstream_recovery_summary.csv",
        "depth_group_summary.csv",
        "final_output_summary.csv",
        "layer_summary.csv",
        "attention_decomposition_summary.csv",
        "attention_map_summary.csv",
        key_zip_name,
    ]
    zip_path = split_output_dir / f"{split_output_dir.name}_report_bundle.zip"
    plots_dir = split_output_dir / "plots"
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename in report_files:
            file_path = split_output_dir / filename
            if file_path.exists():
                archive.write(file_path, arcname=filename)
        if plots_dir.exists():
            for plot_path in sorted(plots_dir.rglob("*")):
                if plot_path.is_file():
                    archive.write(plot_path, arcname=str(plot_path.relative_to(split_output_dir)))
    return zip_path


def finalize_split_outputs(
    split_output_dir: Path,
    *,
    config_payload: dict[str, Any],
    layer_rows: list[dict[str, Any]],
    spectrum_rows: list[dict[str, Any]],
    final_rows: list[dict[str, Any]],
    attention_rows: list[dict[str, Any]],
    attention_map_rows: list[dict[str, Any]],
    aggregated_grams: dict[int, list[np.ndarray]],
    channel_records: list[dict[str, Any]],
    overlap_rows: list[dict[str, Any]],
    frequency_rows: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
    policy_rows: list[dict[str, Any]],
    downstream_rows: list[dict[str, Any]],
) -> None:
    split_output_dir.mkdir(parents=True, exist_ok=True)
    layer_summary = summarize_by_layer(
        layer_rows,
        [
            "abs_correction_norm",
            "relative_correction_norm",
            "tokenwise_mean_correction_norm",
            "channelwise_mean_correction_norm",
            "full_low_cosine_similarity",
            "layer_relative_error",
            "error_increment",
            "attn_relative_correction_norm",
            "ffn_relative_correction_norm",
            "norm1_relative_correction_norm",
            "norm2_relative_correction_norm",
            *[f"top{rank}_explained" for rank in TOP_RANKS],
            "spatial_mass_top_10pct",
            "token_mass_top_10pct",
            "channel_mass_top_10pct",
        ],
    )
    attention_summary = summarize_by_layer(
        attention_rows,
        [
            "attn_low_to_full_relative_error",
            "attn_v_only_to_full_relative_error",
            "attn_qk_only_to_full_relative_error",
            "attn_v_only_improvement",
            "attn_qk_only_improvement",
        ],
    )
    attention_map_summary = summarize_by_layer(
        attention_map_rows,
        [
            "attn_map_mean_l2",
            "attn_map_mean_relative_error",
            "attn_map_mean_cosine_similarity",
        ],
    )
    final_summary = summarize_final_rows(final_rows)
    correlations = compute_correlations(layer_rows, final_rows)
    aggregated_spectra = aggregated_spectrum_rows(aggregated_grams)
    policy_summary = summarize_policy_rows(policy_rows)
    downstream_summary = summarize_downstream_rows(downstream_rows)
    depth_summary = summarize_depth_group_rows(overlap_rows, policy_rows, downstream_rows)

    write_json(split_output_dir / "config.json", config_payload)
    write_csv(split_output_dir / "per_sample_layer_metrics.csv", layer_rows)
    write_jsonl(split_output_dir / "per_sample_spectra.jsonl", spectrum_rows)
    write_csv(split_output_dir / "per_sample_final_metrics.csv", final_rows)
    write_jsonl(split_output_dir / "per_sample_channel_records.jsonl", channel_records)
    write_csv(split_output_dir / "per_sample_attention_decomposition.csv", attention_rows)
    write_csv(split_output_dir / "per_sample_attention_map_metrics.csv", attention_map_rows)
    write_csv(split_output_dir / "channel_overlap_summary.csv", summarize_overlap_rows(overlap_rows))
    write_csv(split_output_dir / "channel_frequency_histogram.csv", frequency_rows)
    write_csv(split_output_dir / "channel_cumulative_coverage.csv", coverage_rows)
    write_csv(split_output_dir / "per_sample_channel_policy_metrics.csv", policy_rows)
    write_csv(split_output_dir / "channel_policy_summary.csv", policy_summary)
    write_csv(split_output_dir / "per_sample_downstream_recovery.csv", downstream_rows)
    write_csv(split_output_dir / "downstream_recovery_summary.csv", downstream_summary)
    write_csv(split_output_dir / "depth_group_summary.csv", depth_summary)
    write_csv(split_output_dir / "layer_summary.csv", layer_summary)
    write_csv(split_output_dir / "final_output_summary.csv", final_summary)
    write_csv(split_output_dir / "attention_decomposition_summary.csv", attention_summary)
    write_csv(split_output_dir / "attention_map_summary.csv", attention_map_summary)
    write_csv(split_output_dir / "final_output_correlations.csv", correlations)
    write_jsonl(split_output_dir / "aggregated_spectrum.jsonl", aggregated_spectra)
    write_plots(
        split_output_dir / "plots",
        layer_summary_rows=layer_summary,
        attention_summary_rows=attention_summary,
    )
    write_channel_policy_plots(
        split_output_dir / "plots",
        overlap_rows=overlap_rows,
        policy_summary_rows=policy_summary,
        frequency_rows=frequency_rows,
        coverage_rows=coverage_rows,
        depth_summary_rows=depth_summary,
        downstream_summary_rows=downstream_summary,
    )
    create_key_results_zip(split_output_dir)
    create_report_bundle_zip(split_output_dir)



def main() -> None:
    args = parse_args()
    set_determinism(args.seed)

    dataset_name = normalize_dataset_name(args.dataset)
    model_name = normalize_model_name(args.model)
    data_root = args.data_root or default_data_root_for_dataset(dataset_name)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layers = parse_layers(args.layers)
    out_dir = resolve_output_dir("measure_correction", args.out_dir)
    autocast_dtype = parse_autocast_dtype(args.autocast_dtype)
    model_dtype = parse_model_dtype(args.model_dtype)
    downstream_methods = parse_downstream_methods(args.downstream_oracle_methods)
    pretrained = not args.no_pretrained
    backbone_weights = resolve_backbone_weights(model_name, args.backbone_weights, pretrained)
    low_input_id = f"{args.low_mode}_level{args.low_level}" if args.low_mode == "gaussian_pyr" else f"{args.low_mode}_x{args.downscale}"
    split_targets = list(SPLIT_NAMES) if args.split == "both" else [args.split]
    save_hidden_states = True

    loader = build_analysis_loader(
        dataset_name=dataset_name,
        data_root=data_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
    )

    probe = Dinov3CorrectionProbe(
        device=device,
        image_size=args.image_size,
        model_name=model_name,
        backbone_weights=backbone_weights,
        pretrained=pretrained,
        model_dtype=model_dtype,
        autocast_dtype=autocast_dtype,
        layers=layers,
    )

    split_state = {
        split_name: {
            "processed": 0,
            "debug_saved": 0,
            "layer_rows": [],
            "spectrum_rows": [],
            "final_rows": [],
            "attention_rows": [],
            "attention_map_rows": [],
            "aggregated_grams": defaultdict(list),
            "channel_records": [],
        }
        for split_name in split_targets
    }

    total_batches = len(loader)
    if args.max_batches is not None:
        total_batches = min(total_batches, args.max_batches)

    iterator = enumerate(loader)
    if args.max_batches is not None:
        iterator = ((batch_idx, batch) for batch_idx, batch in iterator if batch_idx < args.max_batches)
    progress = tqdm(iterator, total=total_batches, desc="measure_correction", unit="batch")

    for batch_idx, batch in progress:
        full_images = batch["images"]
        low_images = make_low_inputs(full_images, mode=args.low_mode, low_level=args.low_level, downscale=args.downscale)

        batch_sample_indices: list[tuple[int, str, dict[str, Any]]] = []
        for sample_offset, sample_key in enumerate(batch["sample_keys"]):
            sample_metadata = build_sample_metadata(batch, sample_offset)
            sample_id = build_sample_id(sample_metadata["sample_key"], sample_metadata["path"])
            split_name = assign_split(sample_id, args.analysis_ratio)
            if split_name not in split_targets:
                continue
            if args.max_samples is not None and split_state[split_name]["processed"] >= args.max_samples:
                continue
            batch_sample_indices.append((sample_offset, split_name, sample_metadata))

        if not batch_sample_indices:
            continue

        collect_attention_maps = args.run_mode == "debug"
        collect_debug_tensors = args.run_mode == "debug"
        full_run = probe.run(full_images, collect_attention_maps=collect_attention_maps, collect_debug_tensors=collect_debug_tensors)
        low_run = probe.run(low_images, collect_attention_maps=collect_attention_maps, collect_debug_tensors=collect_debug_tensors)

        for sample_offset, split_name, sample_metadata in batch_sample_indices:
            state = split_state[split_name]
            sample_id = build_sample_id(sample_metadata["sample_key"], sample_metadata["path"])
            valid_patch_mask = build_valid_patch_mask(batch["valid_masks"][sample_offset], probe.patch_size)

            single_full = ProbeRun(
                patch_start=full_run.patch_start,
                patch_hw=full_run.patch_hw,
                layer_outputs={
                    layer_idx: {key: value[sample_offset : sample_offset + 1] for key, value in layer.items()}
                    for layer_idx, layer in full_run.layer_outputs.items()
                },
                final_feature_tokens=full_run.final_feature_tokens[sample_offset : sample_offset + 1],
                pooled_output=full_run.pooled_output[sample_offset : sample_offset + 1],
            )
            single_low = ProbeRun(
                patch_start=low_run.patch_start,
                patch_hw=low_run.patch_hw,
                layer_outputs={
                    layer_idx: {key: value[sample_offset : sample_offset + 1] for key, value in layer.items()}
                    for layer_idx, layer in low_run.layer_outputs.items()
                },
                final_feature_tokens=low_run.final_feature_tokens[sample_offset : sample_offset + 1],
                pooled_output=low_run.pooled_output[sample_offset : sample_offset + 1],
            )

            layer_rows, spectrum_rows, grams, attention_map_rows = compute_layer_rows(
                split_name=split_name,
                sample_id=sample_id,
                model_name=model_name,
                low_input_id=low_input_id,
                precision=str(probe.model_dtype),
                seed=args.seed,
                full_run=single_full,
                low_run=single_low,
                valid_patch_mask=valid_patch_mask,
            )
            final_row = compute_final_row(
                split_name=split_name,
                sample_id=sample_id,
                model_name=model_name,
                low_input_id=low_input_id,
                precision=str(probe.model_dtype),
                seed=args.seed,
                full_run=single_full,
                low_run=single_low,
            )
            hidden_state_path = None
            if save_hidden_states:
                hidden_dir = out_dir / f"{split_name}_split" / "hidden_states"
                hidden_dir.mkdir(parents=True, exist_ok=True)
                hidden_state_path = save_hidden_state_dump(
                    hidden_dir,
                    sample_id=sample_id,
                    sample_metadata={
                        **sample_metadata,
                        "split_name": split_name,
                        "batch_idx": batch_idx,
                        "model_name": model_name,
                        "low_input_id": low_input_id,
                    },
                    full_run=single_full,
                    low_run=single_low,
                )
            channel_records = compute_channel_records(
                split_name=split_name,
                sample_id=sample_id,
                model_name=model_name,
                low_input_id=low_input_id,
                precision=str(probe.model_dtype),
                seed=args.seed,
                full_run=single_full,
                low_run=single_low,
                hidden_state_path=hidden_state_path,
            )
            attention_rows = compute_attention_decomposition_rows(
                split_name=split_name,
                sample_id=sample_id,
                model_name=model_name,
                low_input_id=low_input_id,
                precision=str(probe.model_dtype),
                seed=args.seed,
                probe=probe,
                full_run=single_full,
                low_run=single_low,
            )

            state["layer_rows"].extend(layer_rows)
            state["spectrum_rows"].extend(spectrum_rows)
            state["final_rows"].append(final_row)
            state["attention_rows"].extend(attention_rows)
            state["attention_map_rows"].extend(attention_map_rows)
            state["channel_records"].extend(channel_records)
            for layer_idx, gram in grams.items():
                state["aggregated_grams"][layer_idx].append(gram)

            if args.run_mode == "debug" and state["debug_saved"] < args.debug_dump_limit:
                debug_dir = out_dir / f"{split_name}_split" / "debug_tensors"
                debug_dir.mkdir(parents=True, exist_ok=True)
                save_debug_dump(
                    debug_dir,
                    sample_id=sample_id,
                    sample_metadata={
                        **sample_metadata,
                        "split_name": split_name,
                        "batch_idx": batch_idx,
                        "model_name": model_name,
                        "low_input_id": low_input_id,
                    },
                    full_run=single_full,
                    low_run=single_low,
                )
                state["debug_saved"] += 1

            state["processed"] += 1
            progress.set_postfix({name: split_state[name]["processed"] for name in split_targets})

        if args.max_samples is not None and all(split_state[name]["processed"] >= args.max_samples for name in split_targets):
            break

    for split_name, state in split_state.items():
        analysis_channel_records = split_state["analysis"]["channel_records"] if "analysis" in split_state else []
        fixed_subsets = build_fixed_subsets(analysis_channel_records) if analysis_channel_records else {}
        overlap_rows, frequency_rows, coverage_rows = compute_overlap_analysis(state["channel_records"])
        policy_rows = compute_channel_policy_metrics(
            channel_records=state["channel_records"],
            fixed_subsets=fixed_subsets,
        )
        downstream_rows = compute_downstream_recovery_rows(
            probe=probe,
            channel_records=state["channel_records"],
            fixed_subsets=fixed_subsets,
            methods=downstream_methods,
            run_mode=args.run_mode,
            representative_count=args.debug_representative_layers,
        )
        split_output_dir = out_dir / f"{split_name}_split"
        config_payload = {
            "dataset": dataset_name,
            "data_root": str(Path(data_root).expanduser()) if data_root else "",
            "image_size": args.image_size,
            "model_name": model_name,
            "backbone_weights": backbone_weights,
            "pretrained": pretrained,
            "model_dtype_requested": str(model_dtype),
            "model_dtype_runtime": str(probe.model_dtype),
            "low_mode": args.low_mode,
            "low_level": args.low_level,
            "downscale": args.downscale,
            "low_input_id": low_input_id,
            "run_mode": args.run_mode,
            "split_name": split_name,
            "analysis_ratio": args.analysis_ratio,
            "layers": layers,
            "selected_layer_count": len(sorted({int(record["layer_idx"]) for record in state["channel_records"]})),
            "device": str(device),
            "autocast_dtype": None if autocast_dtype is None else str(autocast_dtype),
            "seed": args.seed,
            "processed_samples": state["processed"],
            "save_hidden_states": save_hidden_states,
            "downstream_methods": list(downstream_methods),
            "debug_representative_layers": args.debug_representative_layers,
            "fixed_subset_source_split": "analysis" if "analysis" in split_state and split_state["analysis"]["channel_records"] else None,
            "svd_convention_transformer": "use tensor as [tokens, channels]",
            "svd_convention_feature_map": "flatten [channels, H, W] to [(H*W), channels]",
            "attention_map_storage_policy": "debug_only" if args.run_mode == "debug" else "disabled",
            "split_safe_protocol": {
                "full_and_low_forward": "independent_per_sample",
                "full_activation_reuse_for_low": False,
                "cross_sample_activation_mixing": False,
                "downstream_intervention": "same-sample same-layer local intervention only",
                "fixed_subset_fit_split": "analysis_only" if "analysis" in split_state and split_state["analysis"]["channel_records"] else "unavailable",
            },
        }
        finalize_split_outputs(
            split_output_dir,
            config_payload=config_payload,
            layer_rows=state["layer_rows"],
            spectrum_rows=state["spectrum_rows"],
            final_rows=state["final_rows"],
            attention_rows=state["attention_rows"],
            attention_map_rows=state["attention_map_rows"],
            aggregated_grams=state["aggregated_grams"],
            channel_records=state["channel_records"],
            overlap_rows=overlap_rows,
            frequency_rows=frequency_rows,
            coverage_rows=coverage_rows,
            policy_rows=policy_rows,
            downstream_rows=downstream_rows,
        )

    print("[measure_correction] Done")
    for split_name in split_targets:
        print(f"  - {split_name}: {split_state[split_name]['processed']} samples")


if __name__ == "__main__":
    main()
