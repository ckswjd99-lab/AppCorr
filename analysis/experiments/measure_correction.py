import argparse
import csv
import hashlib
import json
import math
import os
import sys
import zipfile
from collections import defaultdict
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
SPLIT_NAMES = ("analysis", "holdout")
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
                if collect_attention_maps:
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
                x = x_attn + ffn_block_output

                if self._selected(layer_idx):
                    record = {
                        "block_output": x.detach().to(device="cpu", dtype=torch.float16),
                        "attn_block_output": attn_block_output.detach().to(device="cpu", dtype=torch.float16),
                        "ffn_block_output": ffn_block_output.detach().to(device="cpu", dtype=torch.float16),
                        "norm1_output": x_norm1.detach().to(device="cpu", dtype=torch.float16),
                        "norm2_output": x_norm2.detach().to(device="cpu", dtype=torch.float16),
                    }
                    if collect_debug_tensors:
                        record["block_input"] = block_input.detach().to(device="cpu", dtype=torch.float16)
                    if attn_map_mean is not None:
                        record["attn_map_mean"] = attn_map_mean.detach().to(device="cpu", dtype=torch.float16)
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
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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


def create_key_results_zip(split_output_dir: Path) -> Path:
    key_files = [
        "config.json",
        "layer_summary.csv",
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

    write_json(split_output_dir / "config.json", config_payload)
    write_csv(split_output_dir / "per_sample_layer_metrics.csv", layer_rows)
    write_jsonl(split_output_dir / "per_sample_spectra.jsonl", spectrum_rows)
    write_csv(split_output_dir / "per_sample_final_metrics.csv", final_rows)
    write_csv(split_output_dir / "per_sample_attention_decomposition.csv", attention_rows)
    write_csv(split_output_dir / "per_sample_attention_map_metrics.csv", attention_map_rows)
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
    create_key_results_zip(split_output_dir)



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
    pretrained = not args.no_pretrained
    backbone_weights = resolve_backbone_weights(model_name, args.backbone_weights, pretrained)
    low_input_id = f"{args.low_mode}_level{args.low_level}" if args.low_mode == "gaussian_pyr" else f"{args.low_mode}_x{args.downscale}"
    split_targets = list(SPLIT_NAMES) if args.split == "both" else [args.split]

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
            "device": str(device),
            "autocast_dtype": None if autocast_dtype is None else str(autocast_dtype),
            "seed": args.seed,
            "processed_samples": state["processed"],
            "svd_convention_transformer": "use tensor as [tokens, channels]",
            "svd_convention_feature_map": "flatten [channels, H, W] to [(H*W), channels]",
            "attention_map_storage_policy": "debug_only" if args.run_mode == "debug" else "disabled",
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
        )

    print("[measure_correction] Done")
    for split_name in split_targets:
        print(f"  - {split_name}: {split_state[split_name]['processed']} samples")


if __name__ == "__main__":
    main()
