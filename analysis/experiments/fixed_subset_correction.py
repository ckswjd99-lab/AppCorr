import argparse
import json
import math
import os
import sys
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
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
from analysis.experiments.measure_correction import (
    Dinov3CorrectionProbe,
    assign_split,
    build_sample_id,
    build_sample_metadata,
    cosine_similarity,
    make_low_inputs,
    normalize_model_name,
    parse_autocast_dtype,
    parse_model_dtype,
    percentile_suffix,
    relative_error,
    resolve_backbone_weights,
    resolve_output_dir,
    safe_divide,
    set_determinism,
    topk_count,
    topk_indices_from_scores,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate fixed channel subset correction heads on frozen DINOv3 backbones."
    )
    parser.add_argument("--dataset", type=str, default="imagenet-1k", help="Dataset name: imagenet-1k or coco.")
    parser.add_argument("--data-root", type=str, default=None, help="Dataset root for ImageNet.")
    parser.add_argument("--image-size", type=int, default=256, help="Square image size after resize-and-pad preprocessing.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for frozen full/low forward collection.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional train/analysis sample cap.")
    parser.add_argument("--max-eval-samples", type=int, default=None, help="Optional holdout sample cap.")
    parser.add_argument("--max-mask-samples", type=int, default=None, help="Optional analysis sample cap for fitting fixed masks.")
    parser.add_argument("--max-batches", type=int, default=None, help="Optional global batch limit per loop.")
    parser.add_argument(
        "--model",
        type=str,
        default="small",
        help="DINOv3 ViT scale alias or canonical name. Supported: small/base/large/7b.",
    )
    parser.add_argument("--backbone-weights", type=str, default=None, help="Optional local backbone weights path.")
    parser.add_argument("--no-pretrained", action="store_true", help="Use randomly initialized weights instead of loading checkpoints.")
    parser.add_argument("--device", type=str, default=None, help="Torch device. Default: cuda if available else cpu.")
    parser.add_argument("--model-dtype", type=str, default="bf16", help="Resident dtype for the loaded backbone on CUDA.")
    parser.add_argument("--autocast-dtype", type=str, default="bf16", help="Autocast dtype on CUDA.")
    parser.add_argument("--seed", type=int, default=1234, help="Deterministic seed.")
    parser.add_argument("--target-layers", type=str, default="4,6,8", help="Comma-separated target layer indices.")
    parser.add_argument("--mask-pcts", type=str, default="10,20", help="Comma-separated mask percentages.")
    parser.add_argument("--low-mode", type=str, default="gaussian_pyr", choices=("gaussian_pyr", "bicubic"), help="Low-resolution construction rule.")
    parser.add_argument("--low-level", type=int, default=2, help="Downsample pyramid level for gaussian_pyr mode.")
    parser.add_argument("--downscale", type=int, default=4, help="Downscale factor for bicubic mode.")
    parser.add_argument("--analysis-ratio", type=float, default=0.8, help="Fraction of samples routed to analysis/train split.")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs for the small correction heads.")
    parser.add_argument("--lr", type=float, default=1e-3, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument("--head-hidden-dim", type=int, default=128, help="Hidden dimension inside the tokenwise correction MLP.")
    parser.add_argument("--sketch-size", type=int, default=8, help="Residual sketch adaptive pooled size.")
    parser.add_argument("--sketch-proj-dim", type=int, default=32, help="Residual sketch projection dimension.")
    parser.add_argument("--run-mode", type=str, default="full", choices=("debug", "full"), help="debug narrows to p=10 and tiny sample counts; full runs the requested configuration.")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory.")
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    items = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not items:
        raise ValueError("Expected at least one integer value.")
    return items


def parse_pct_list(value: str) -> list[float]:
    items = []
    for part in value.split(","):
        stripped = part.strip()
        if not stripped:
            continue
        pct = float(stripped)
        if pct > 1.0:
            pct = pct / 100.0
        items.append(pct)
    unique = sorted(set(items))
    if not unique:
        raise ValueError("Expected at least one percentage value.")
    return unique


@dataclass(frozen=True)
class ComboSpec:
    layer_idx: int
    pct: float
    pct_label: str

    @property
    def name(self) -> str:
        return f"layer{self.layer_idx}_{self.pct_label}"


class TokenwiseCorrectionHead(nn.Module):
    def __init__(
        self,
        *,
        selected_dim: int,
        sketch_dim: int,
        sketch_proj_dim: int,
        hidden_dim: int,
        use_sketch: bool,
    ):
        super().__init__()
        self.selected_dim = selected_dim
        self.sketch_dim = sketch_dim
        self.sketch_proj_dim = sketch_proj_dim if use_sketch else 0
        self.use_sketch = use_sketch
        self.sketch_proj = None
        if self.use_sketch:
            self.sketch_proj = nn.Sequential(
                nn.Linear(sketch_dim, sketch_proj_dim),
                nn.GELU(),
            )
        input_dim = selected_dim + self.sketch_proj_dim
        self.token_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, selected_dim),
        )

    def forward(self, low_selected: torch.Tensor, residual_sketch: torch.Tensor | None) -> torch.Tensor:
        features = [low_selected]
        if self.use_sketch:
            if residual_sketch is None:
                raise ValueError("Residual sketch is required when use_sketch=True.")
            sketch_feature = self.sketch_proj(residual_sketch).unsqueeze(1).expand(-1, low_selected.shape[1], -1)
            features.append(sketch_feature)
        x = torch.cat(features, dim=-1)
        return self.token_mlp(x)


def make_combo_specs(target_layers: list[int], mask_pcts: list[float]) -> list[ComboSpec]:
    return [ComboSpec(layer_idx=layer_idx, pct=pct, pct_label=percentile_suffix(pct)) for layer_idx in target_layers for pct in mask_pcts]


def compute_residual_sketch(full_images: torch.Tensor, low_images: torch.Tensor, sketch_size: int) -> torch.Tensor:
    residual = (full_images.float() - low_images.float()) / 255.0
    pooled = F.adaptive_avg_pool2d(residual, output_size=(sketch_size, sketch_size))
    return pooled.flatten(start_dim=1).contiguous()


def estimate_head_flops(
    *,
    num_tokens: int,
    selected_dim: int,
    full_dim: int,
    sketch_dim: int,
    sketch_proj_dim: int,
    hidden_dim: int,
    use_sketch: bool,
) -> tuple[int, int]:
    input_dim = selected_dim + (sketch_proj_dim if use_sketch else 0)
    sketch_flops = sketch_dim * sketch_proj_dim if use_sketch else 0
    token_flops = num_tokens * ((input_dim * hidden_dim) + (hidden_dim * selected_dim))
    dense_input_dim = full_dim + (sketch_proj_dim if use_sketch else 0)
    dense_token_flops = num_tokens * ((dense_input_dim * hidden_dim) + (hidden_dim * full_dim))
    return sketch_flops + token_flops, sketch_flops + dense_token_flops


def parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def select_split_subset(
    batch: dict[str, Any],
    *,
    split_name: str,
    analysis_ratio: float,
    processed: int,
    max_samples: int | None,
) -> tuple[list[int], list[str], list[dict[str, Any]]]:
    indices: list[int] = []
    sample_ids: list[str] = []
    metadata_rows: list[dict[str, Any]] = []
    for offset, _ in enumerate(batch["sample_keys"]):
        if max_samples is not None and processed + len(indices) >= max_samples:
            break
        metadata = build_sample_metadata(batch, offset)
        sample_id = build_sample_id(metadata["sample_key"], metadata["path"])
        if assign_split(sample_id, analysis_ratio) != split_name:
            continue
        indices.append(offset)
        sample_ids.append(sample_id)
        metadata_rows.append(metadata)
    return indices, sample_ids, metadata_rows


def build_fixed_masks(
    *,
    loader: torch.utils.data.DataLoader,
    probe: Dinov3CorrectionProbe,
    combo_specs: list[ComboSpec],
    analysis_ratio: float,
    low_mode: str,
    low_level: int,
    downscale: int,
    max_samples: int | None,
    max_batches: int | None,
) -> dict[str, list[int]]:
    counters: dict[str, Counter[int]] = {combo.name: Counter() for combo in combo_specs}
    num_channels_by_layer: dict[int, int] = {}
    processed = 0
    iterator = enumerate(loader)
    total_batches = len(loader) if max_batches is None else min(len(loader), max_batches)
    progress = tqdm(iterator, total=total_batches, desc="fit_fixed_masks", unit="batch")
    for batch_idx, batch in progress:
        if max_batches is not None and batch_idx >= max_batches:
            break
        indices, sample_ids, _ = select_split_subset(
            batch,
            split_name="analysis",
            analysis_ratio=analysis_ratio,
            processed=processed,
            max_samples=max_samples,
        )
        if not indices:
            if max_samples is not None and processed >= max_samples:
                break
            continue
        full_images = batch["images"][indices]
        low_images = make_low_inputs(full_images, mode=low_mode, low_level=low_level, downscale=downscale)
        full_run = probe.run(full_images, collect_attention_maps=False, collect_debug_tensors=False)
        low_run = probe.run(low_images, collect_attention_maps=False, collect_debug_tensors=False)

        for local_idx, _sample_id in enumerate(sample_ids):
            for combo in combo_specs:
                full_hidden = full_run.layer_outputs[combo.layer_idx]["block_output"][local_idx].numpy().astype(np.float32)
                low_hidden = low_run.layer_outputs[combo.layer_idx]["block_output"][local_idx].numpy().astype(np.float32)
                channel_energy = np.sum(np.square(full_hidden - low_hidden), axis=0)
                num_channels_by_layer.setdefault(combo.layer_idx, int(channel_energy.shape[0]))
                counters[combo.name].update(int(idx) for idx in topk_indices_from_scores(channel_energy, combo.pct).tolist())
        processed += len(indices)
        progress.set_postfix({"analysis_samples": processed})
        if max_samples is not None and processed >= max_samples:
            break

    masks: dict[str, list[int]] = {}
    for combo in combo_specs:
        num_channels = num_channels_by_layer.get(combo.layer_idx)
        if num_channels is None:
            raise RuntimeError(f"No analysis samples were collected for layer {combo.layer_idx}.")
        k = topk_count(num_channels, combo.pct)
        masks[combo.name] = [channel for channel, _ in counters[combo.name].most_common(k)]
    return masks


def build_heads(
    *,
    combo_specs: list[ComboSpec],
    fixed_masks: dict[str, list[int]],
    num_channels_by_layer: dict[int, int],
    sketch_dim: int,
    sketch_proj_dim: int,
    hidden_dim: int,
    num_tokens_by_layer: dict[int, int],
    device: torch.device,
) -> tuple[nn.ModuleDict, dict[str, dict[str, float | int]]]:
    heads = nn.ModuleDict()
    cost_rows: dict[str, dict[str, float | int]] = {}
    for combo in combo_specs:
        selected_dim = len(fixed_masks[combo.name])
        full_dim = num_channels_by_layer[combo.layer_idx]
        num_tokens = num_tokens_by_layer[combo.layer_idx]
        for baseline_type, use_sketch in (("learned_fixed", True), ("low_hidden_only", False)):
            key = f"{baseline_type}:{combo.name}"
            head = TokenwiseCorrectionHead(
                selected_dim=selected_dim,
                sketch_dim=sketch_dim,
                sketch_proj_dim=sketch_proj_dim,
                hidden_dim=hidden_dim,
                use_sketch=use_sketch,
            ).to(device=device)
            heads[key] = head
            flops, dense_flops = estimate_head_flops(
                num_tokens=num_tokens,
                selected_dim=selected_dim,
                full_dim=full_dim,
                sketch_dim=sketch_dim,
                sketch_proj_dim=sketch_proj_dim,
                hidden_dim=hidden_dim,
                use_sketch=use_sketch,
            )
            cost_rows[key] = {
                "selected_channel_count": selected_dim,
                "selected_channel_ratio": selected_dim / max(full_dim, 1),
                "head_param_count": parameter_count(head),
                "rough_head_flops": flops,
                "relative_cost_vs_dense_head": safe_divide(flops, dense_flops, default=0.0),
            }
    return heads, cost_rows


def initialize_shape_metadata(
    *,
    loader: torch.utils.data.DataLoader,
    probe: Dinov3CorrectionProbe,
    target_layers: list[int],
    analysis_ratio: float,
    low_mode: str,
    low_level: int,
    downscale: int,
) -> tuple[dict[int, int], dict[int, int]]:
    for batch in loader:
        indices, _, _ = select_split_subset(
            batch,
            split_name="analysis",
            analysis_ratio=analysis_ratio,
            processed=0,
            max_samples=1,
        )
        if not indices:
            continue
        full_images = batch["images"][indices]
        low_images = make_low_inputs(full_images, mode=low_mode, low_level=low_level, downscale=downscale)
        full_run = probe.run(full_images, collect_attention_maps=False, collect_debug_tensors=False)
        low_run = probe.run(low_images, collect_attention_maps=False, collect_debug_tensors=False)
        del low_run
        num_channels_by_layer = {}
        num_tokens_by_layer = {}
        for layer_idx in target_layers:
            tensor = full_run.layer_outputs[layer_idx]["block_output"][0]
            num_tokens_by_layer[layer_idx] = int(tensor.shape[0])
            num_channels_by_layer[layer_idx] = int(tensor.shape[1])
        return num_channels_by_layer, num_tokens_by_layer
    raise RuntimeError("Could not initialize layer shape metadata because no analysis samples were found.")


def train_heads(
    *,
    loader: torch.utils.data.DataLoader,
    probe: Dinov3CorrectionProbe,
    combo_specs: list[ComboSpec],
    fixed_masks: dict[str, list[int]],
    heads: nn.ModuleDict,
    optimizer: torch.optim.Optimizer,
    analysis_ratio: float,
    low_mode: str,
    low_level: int,
    downscale: int,
    sketch_size: int,
    max_samples: int | None,
    max_batches: int | None,
    epochs: int,
    device: torch.device,
) -> list[dict[str, Any]]:
    history_rows: list[dict[str, Any]] = []
    for epoch_idx in range(epochs):
        epoch_losses: dict[str, list[float]] = defaultdict(list)
        processed = 0
        iterator = enumerate(loader)
        total_batches = len(loader) if max_batches is None else min(len(loader), max_batches)
        progress = tqdm(iterator, total=total_batches, desc=f"train_heads epoch={epoch_idx + 1}", unit="batch")
        for batch_idx, batch in progress:
            if max_batches is not None and batch_idx >= max_batches:
                break
            indices, _, _ = select_split_subset(
                batch,
                split_name="analysis",
                analysis_ratio=analysis_ratio,
                processed=processed,
                max_samples=max_samples,
            )
            if not indices:
                if max_samples is not None and processed >= max_samples:
                    break
                continue
            full_images = batch["images"][indices]
            low_images = make_low_inputs(full_images, mode=low_mode, low_level=low_level, downscale=downscale)
            residual_sketch = compute_residual_sketch(full_images, low_images, sketch_size=sketch_size).to(device=device)

            with torch.no_grad():
                full_run = probe.run(full_images, collect_attention_maps=False, collect_debug_tensors=False)
                low_run = probe.run(low_images, collect_attention_maps=False, collect_debug_tensors=False)

            optimizer.zero_grad(set_to_none=True)
            total_loss = None
            num_terms = 0
            for combo in combo_specs:
                selected_channels = fixed_masks[combo.name]
                full_hidden = full_run.layer_outputs[combo.layer_idx]["block_output"].to(device=device, dtype=torch.float32)
                low_hidden = low_run.layer_outputs[combo.layer_idx]["block_output"].to(device=device, dtype=torch.float32)
                target_delta = full_hidden[..., selected_channels] - low_hidden[..., selected_channels]
                for baseline_type, use_sketch in (("learned_fixed", True), ("low_hidden_only", False)):
                    key = f"{baseline_type}:{combo.name}"
                    prediction = heads[key](
                        low_hidden[..., selected_channels],
                        residual_sketch if use_sketch else None,
                    )
                    loss = F.mse_loss(prediction, target_delta)
                    total_loss = loss if total_loss is None else total_loss + loss
                    num_terms += 1
                    epoch_losses[key].append(float(loss.detach().cpu()))
            if total_loss is None or num_terms == 0:
                continue
            total_loss = total_loss / num_terms
            total_loss.backward()
            optimizer.step()
            processed += len(indices)
            progress.set_postfix({"analysis_samples": processed, "loss": float(total_loss.detach().cpu())})
            if max_samples is not None and processed >= max_samples:
                break

        for key, values in sorted(epoch_losses.items()):
            baseline_type, combo_name = key.split(":")
            history_rows.append({
                "epoch": epoch_idx + 1,
                "baseline_type": baseline_type,
                "combo_name": combo_name,
                "train_loss_mean": float(np.mean(values)) if values else None,
                "train_loss_std": float(np.std(values)) if values else None,
                "num_batches": len(values),
            })
    return history_rows


def build_corrected_hidden(low_hidden: torch.Tensor, selected_channels: list[int], predicted_delta: torch.Tensor) -> torch.Tensor:
    corrected_hidden = low_hidden.clone()
    corrected_hidden[..., selected_channels] = corrected_hidden[..., selected_channels] + predicted_delta
    return corrected_hidden


def distance_improvement(metric_value: float, low_value: float) -> float:
    return 1.0 - safe_divide(metric_value, low_value, default=1.0)


def similarity_improvement(metric_value: float | None, low_value: float | None) -> float | None:
    if metric_value is None or low_value is None:
        return None
    return float(metric_value - low_value)


def per_sample_result_row(
    *,
    sample_id: str,
    baseline_type: str,
    combo: ComboSpec,
    selected_channel_count: int,
    selected_channel_ratio: float,
    head_param_count: int,
    rough_head_flops: int,
    relative_cost_vs_dense_head: float,
    selected_channel_mse: float,
    hidden_relative_error_to_full: float,
    hidden_relative_error_low: float,
    hidden_relative_error_oracle: float,
    final_feature_relative_error: float,
    final_feature_relative_error_low: float,
    final_feature_relative_error_oracle: float,
    pooled_relative_error: float,
    pooled_relative_error_low: float,
    pooled_relative_error_oracle: float,
    final_feature_cosine: float | None,
    final_feature_cosine_low: float | None,
    final_feature_cosine_oracle: float | None,
    pooled_cosine: float | None,
    pooled_cosine_low: float | None,
    pooled_cosine_oracle: float | None,
) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "layer_idx": combo.layer_idx,
        "pct": combo.pct,
        "pct_label": combo.pct_label,
        "baseline_type": baseline_type,
        "selected_channel_mse": selected_channel_mse,
        "hidden_relative_error_to_full": hidden_relative_error_to_full,
        "hidden_recovery_vs_low": distance_improvement(hidden_relative_error_to_full, hidden_relative_error_low),
        "hidden_gap_to_oracle_fixed": hidden_relative_error_to_full - hidden_relative_error_oracle,
        "final_feature_relative_error_to_full": final_feature_relative_error,
        "final_feature_recovery_vs_low": distance_improvement(final_feature_relative_error, final_feature_relative_error_low),
        "final_feature_gap_to_oracle_fixed": final_feature_relative_error - final_feature_relative_error_oracle,
        "final_feature_cosine_to_full": final_feature_cosine,
        "final_feature_cosine_improvement_vs_low": similarity_improvement(final_feature_cosine, final_feature_cosine_low),
        "final_feature_cosine_gap_to_oracle_fixed": None if final_feature_cosine is None or final_feature_cosine_oracle is None else float(final_feature_cosine_oracle - final_feature_cosine),
        "pooled_output_relative_error_to_full": pooled_relative_error,
        "pooled_output_recovery_vs_low": distance_improvement(pooled_relative_error, pooled_relative_error_low),
        "pooled_output_gap_to_oracle_fixed": pooled_relative_error - pooled_relative_error_oracle,
        "pooled_output_cosine_to_full": pooled_cosine,
        "pooled_output_cosine_improvement_vs_low": similarity_improvement(pooled_cosine, pooled_cosine_low),
        "pooled_output_cosine_gap_to_oracle_fixed": None if pooled_cosine is None or pooled_cosine_oracle is None else float(pooled_cosine_oracle - pooled_cosine),
        "selected_channel_count": selected_channel_count,
        "selected_channel_ratio": selected_channel_ratio,
        "head_param_count": head_param_count,
        "rough_head_flops": rough_head_flops,
        "relative_cost_vs_dense_head": relative_cost_vs_dense_head,
    }


def evaluate_baselines(
    *,
    loader: torch.utils.data.DataLoader,
    probe: Dinov3CorrectionProbe,
    combo_specs: list[ComboSpec],
    fixed_masks: dict[str, list[int]],
    heads: nn.ModuleDict,
    cost_rows: dict[str, dict[str, float | int]],
    analysis_ratio: float,
    low_mode: str,
    low_level: int,
    downscale: int,
    sketch_size: int,
    max_samples: int | None,
    max_batches: int | None,
    device: torch.device,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    processed = 0
    iterator = enumerate(loader)
    total_batches = len(loader) if max_batches is None else min(len(loader), max_batches)
    progress = tqdm(iterator, total=total_batches, desc="eval_holdout", unit="batch")
    for batch_idx, batch in progress:
        if max_batches is not None and batch_idx >= max_batches:
            break
        indices, sample_ids, _ = select_split_subset(
            batch,
            split_name="holdout",
            analysis_ratio=analysis_ratio,
            processed=processed,
            max_samples=max_samples,
        )
        if not indices:
            if max_samples is not None and processed >= max_samples:
                break
            continue
        full_images = batch["images"][indices]
        low_images = make_low_inputs(full_images, mode=low_mode, low_level=low_level, downscale=downscale)
        residual_sketch = compute_residual_sketch(full_images, low_images, sketch_size=sketch_size).to(device=device)

        with torch.no_grad():
            full_run = probe.run(full_images, collect_attention_maps=False, collect_debug_tensors=False)
            low_run = probe.run(low_images, collect_attention_maps=False, collect_debug_tensors=False)

        final_feature_full = full_run.final_feature_tokens.numpy().astype(np.float32)
        final_feature_low = low_run.final_feature_tokens.numpy().astype(np.float32)
        pooled_full = full_run.pooled_output.numpy().astype(np.float32)
        pooled_low = low_run.pooled_output.numpy().astype(np.float32)

        for combo in combo_specs:
            selected_channels = fixed_masks[combo.name]
            layer_idx = combo.layer_idx
            full_hidden = full_run.layer_outputs[layer_idx]["block_output"].to(device=device, dtype=torch.float32)
            low_hidden = low_run.layer_outputs[layer_idx]["block_output"].to(device=device, dtype=torch.float32)
            true_delta_sel = full_hidden[..., selected_channels] - low_hidden[..., selected_channels]

            oracle_hidden = build_corrected_hidden(low_hidden, selected_channels, true_delta_sel)
            learned_prediction = heads[f"learned_fixed:{combo.name}"](low_hidden[..., selected_channels], residual_sketch)
            low_only_prediction = heads[f"low_hidden_only:{combo.name}"](low_hidden[..., selected_channels], None)

            corrected_batches = {
                "low_only": low_hidden,
                "oracle_fixed": oracle_hidden,
                "learned_fixed": build_corrected_hidden(low_hidden, selected_channels, learned_prediction),
                "low_hidden_only": build_corrected_hidden(low_hidden, selected_channels, low_only_prediction),
            }
            selected_mse_batches = {
                "low_only": torch.mean(true_delta_sel.square(), dim=(1, 2)).detach().cpu().numpy(),
                "oracle_fixed": np.zeros((low_hidden.shape[0],), dtype=np.float32),
                "learned_fixed": torch.mean((learned_prediction - true_delta_sel).square(), dim=(1, 2)).detach().cpu().numpy(),
                "low_hidden_only": torch.mean((low_only_prediction - true_delta_sel).square(), dim=(1, 2)).detach().cpu().numpy(),
            }

            final_feature_by_baseline: dict[str, np.ndarray] = {"low_only": final_feature_low}
            pooled_by_baseline: dict[str, np.ndarray] = {"low_only": pooled_low}
            for baseline_type in ("oracle_fixed", "learned_fixed", "low_hidden_only"):
                final_hat, pooled_hat = probe.continue_from_layer(
                    corrected_batches[baseline_type].detach().cpu(),
                    layer_idx=layer_idx,
                    patch_hw=full_run.patch_hw,
                )
                final_feature_by_baseline[baseline_type] = final_hat.numpy().astype(np.float32)
                pooled_by_baseline[baseline_type] = pooled_hat.numpy().astype(np.float32)

            for local_idx, sample_id in enumerate(sample_ids):
                full_hidden_np = full_hidden[local_idx].detach().cpu().numpy().astype(np.float32)
                low_hidden_np = low_hidden[local_idx].detach().cpu().numpy().astype(np.float32)
                oracle_hidden_np = corrected_batches["oracle_fixed"][local_idx].detach().cpu().numpy().astype(np.float32)

                hidden_low_error = relative_error(full_hidden_np, low_hidden_np)
                hidden_oracle_error = relative_error(full_hidden_np, oracle_hidden_np)
                final_low_error = relative_error(final_feature_full[local_idx], final_feature_low[local_idx])
                final_oracle_error = relative_error(final_feature_full[local_idx], final_feature_by_baseline["oracle_fixed"][local_idx])
                pooled_low_error = relative_error(pooled_full[local_idx], pooled_low[local_idx])
                pooled_oracle_error = relative_error(pooled_full[local_idx], pooled_by_baseline["oracle_fixed"][local_idx])
                final_low_cos = cosine_similarity(final_feature_full[local_idx], final_feature_low[local_idx])
                final_oracle_cos = cosine_similarity(final_feature_full[local_idx], final_feature_by_baseline["oracle_fixed"][local_idx])
                pooled_low_cos = cosine_similarity(pooled_full[local_idx], pooled_low[local_idx])
                pooled_oracle_cos = cosine_similarity(pooled_full[local_idx], pooled_by_baseline["oracle_fixed"][local_idx])

                baseline_costs = {
                    "low_only": {
                        "selected_channel_count": len(selected_channels),
                        "selected_channel_ratio": len(selected_channels) / max(full_hidden_np.shape[-1], 1),
                        "head_param_count": 0,
                        "rough_head_flops": 0,
                        "relative_cost_vs_dense_head": 0.0,
                    },
                    "oracle_fixed": {
                        "selected_channel_count": len(selected_channels),
                        "selected_channel_ratio": len(selected_channels) / max(full_hidden_np.shape[-1], 1),
                        "head_param_count": 0,
                        "rough_head_flops": 0,
                        "relative_cost_vs_dense_head": 0.0,
                    },
                    "learned_fixed": cost_rows[f"learned_fixed:{combo.name}"],
                    "low_hidden_only": cost_rows[f"low_hidden_only:{combo.name}"],
                }

                for baseline_type in ("low_only", "oracle_fixed", "learned_fixed", "low_hidden_only"):
                    corrected_hidden_np = corrected_batches[baseline_type][local_idx].detach().cpu().numpy().astype(np.float32)
                    final_feature_np = final_feature_by_baseline[baseline_type][local_idx]
                    pooled_np = pooled_by_baseline[baseline_type][local_idx]
                    rows.append(per_sample_result_row(
                        sample_id=sample_id,
                        baseline_type=baseline_type,
                        combo=combo,
                        selected_channel_count=int(baseline_costs[baseline_type]["selected_channel_count"]),
                        selected_channel_ratio=float(baseline_costs[baseline_type]["selected_channel_ratio"]),
                        head_param_count=int(baseline_costs[baseline_type]["head_param_count"]),
                        rough_head_flops=int(baseline_costs[baseline_type]["rough_head_flops"]),
                        relative_cost_vs_dense_head=float(baseline_costs[baseline_type]["relative_cost_vs_dense_head"]),
                        selected_channel_mse=float(selected_mse_batches[baseline_type][local_idx]),
                        hidden_relative_error_to_full=relative_error(full_hidden_np, corrected_hidden_np),
                        hidden_relative_error_low=hidden_low_error,
                        hidden_relative_error_oracle=hidden_oracle_error,
                        final_feature_relative_error=relative_error(final_feature_full[local_idx], final_feature_np),
                        final_feature_relative_error_low=final_low_error,
                        final_feature_relative_error_oracle=final_oracle_error,
                        pooled_relative_error=relative_error(pooled_full[local_idx], pooled_np),
                        pooled_relative_error_low=pooled_low_error,
                        pooled_relative_error_oracle=pooled_oracle_error,
                        final_feature_cosine=cosine_similarity(final_feature_full[local_idx], final_feature_np),
                        final_feature_cosine_low=final_low_cos,
                        final_feature_cosine_oracle=final_oracle_cos,
                        pooled_cosine=cosine_similarity(pooled_full[local_idx], pooled_np),
                        pooled_cosine_low=pooled_low_cos,
                        pooled_cosine_oracle=pooled_oracle_cos,
                    ))
        processed += len(indices)
        progress.set_postfix({"holdout_samples": processed})
        if max_samples is not None and processed >= max_samples:
            break
    return rows


def summarize_results(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    cost_fields = {
        "selected_channel_count",
        "selected_channel_ratio",
        "head_param_count",
        "rough_head_flops",
        "relative_cost_vs_dense_head",
    }
    for row in rows:
        key = (int(row["layer_idx"]), str(row["pct_label"]), str(row["baseline_type"]))
        for field, value in row.items():
            if field in {"sample_id", "layer_idx", "pct", "pct_label", "baseline_type"}:
                continue
            if value is None:
                continue
            grouped[key][field].append(float(value))

    summary_rows: list[dict[str, Any]] = []
    for (layer_idx, pct_label, baseline_type), metrics in sorted(grouped.items()):
        pct = next(float(row["pct"]) for row in rows if int(row["layer_idx"]) == layer_idx and row["pct_label"] == pct_label)
        summary_row = {
            "layer_idx": layer_idx,
            "pct": pct,
            "pct_label": pct_label,
            "baseline_type": baseline_type,
        }
        for field, values in metrics.items():
            if field in cost_fields:
                summary_row[field] = float(np.mean(values))
                continue
            summary_row[f"{field}_mean"] = float(np.mean(values))
            summary_row[f"{field}_std"] = float(np.std(values))
        summary_row["num_samples"] = len(next(iter(metrics.values()))) if metrics else 0
        summary_rows.append(summary_row)
    return summary_rows


def summarize_b2_vs_b3(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    indexed = {
        (int(row["layer_idx"]), str(row["pct_label"]), str(row["baseline_type"])): row
        for row in summary_rows
    }
    comparison_rows: list[dict[str, Any]] = []
    keys = sorted({(int(row["layer_idx"]), str(row["pct_label"])) for row in summary_rows})
    for layer_idx, pct_label in keys:
        b2 = indexed.get((layer_idx, pct_label, "learned_fixed"))
        b3 = indexed.get((layer_idx, pct_label, "low_hidden_only"))
        if b2 is None or b3 is None:
            continue
        comparison_rows.append({
            "layer_idx": layer_idx,
            "pct": float(b2["pct"]),
            "pct_label": pct_label,
            "selected_channel_count": b2["selected_channel_count"],
            "selected_channel_ratio": b2["selected_channel_ratio"],
            "selected_channel_mse_delta_b2_minus_b3": float(b2["selected_channel_mse_mean"] - b3["selected_channel_mse_mean"]),
            "hidden_relative_error_delta_b2_minus_b3": float(b2["hidden_relative_error_to_full_mean"] - b3["hidden_relative_error_to_full_mean"]),
            "hidden_recovery_gain_b2_minus_b3": float(b2["hidden_recovery_vs_low_mean"] - b3["hidden_recovery_vs_low_mean"]),
            "hidden_oracle_gap_gain_b3_minus_b2": float(b3["hidden_gap_to_oracle_fixed_mean"] - b2["hidden_gap_to_oracle_fixed_mean"]),
            "final_feature_relative_error_delta_b2_minus_b3": float(b2["final_feature_relative_error_to_full_mean"] - b3["final_feature_relative_error_to_full_mean"]),
            "final_feature_recovery_gain_b2_minus_b3": float(b2["final_feature_recovery_vs_low_mean"] - b3["final_feature_recovery_vs_low_mean"]),
            "final_feature_oracle_gap_gain_b3_minus_b2": float(b3["final_feature_gap_to_oracle_fixed_mean"] - b2["final_feature_gap_to_oracle_fixed_mean"]),
            "pooled_output_relative_error_delta_b2_minus_b3": float(b2["pooled_output_relative_error_to_full_mean"] - b3["pooled_output_relative_error_to_full_mean"]),
            "pooled_output_recovery_gain_b2_minus_b3": float(b2["pooled_output_recovery_vs_low_mean"] - b3["pooled_output_recovery_vs_low_mean"]),
            "pooled_output_oracle_gap_gain_b3_minus_b2": float(b3["pooled_output_gap_to_oracle_fixed_mean"] - b2["pooled_output_gap_to_oracle_fixed_mean"]),
        })
    return comparison_rows


def write_plots(output_dir: Path, summary_rows: list[dict[str, Any]], comparison_rows: list[dict[str, Any]]) -> None:
    if not summary_rows and not comparison_rows:
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

    def subset_rows(baseline_type: str, pct_label: str | None = None) -> list[dict[str, Any]]:
        subset = [row for row in summary_rows if row["baseline_type"] == baseline_type]
        if pct_label is not None:
            subset = [row for row in subset if row["pct_label"] == pct_label]
        return sorted(subset, key=lambda row: int(row["layer_idx"]))

    def line_plot(rows_by_label: dict[str, list[dict[str, Any]]], metric_name: str, title: str, ylabel: str, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
        palette = ["#175676", "#D62839", "#2A9D8F", "#F77F00"]
        plotted = False
        for color, (label, rows_) in zip(palette, rows_by_label.items()):
            xs = [int(row["layer_idx"]) for row in rows_ if row.get(metric_name) is not None]
            ys = [float(row[metric_name]) for row in rows_ if row.get(metric_name) is not None]
            if not xs:
                continue
            ax.plot(xs, ys, marker="o", linewidth=2.0, color=color, label=label)
            plotted = True
        if not plotted:
            plt.close(fig)
            return
        ax.set_title(title)
        ax.set_xlabel("Target layer")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.savefig(output_dir / filename, bbox_inches="tight")
        plt.close(fig)

    line_plot(
        {
            "learned_fixed@10%": subset_rows("learned_fixed", "10pct"),
            "learned_fixed@20%": subset_rows("learned_fixed", "20pct"),
        },
        "pooled_output_recovery_vs_low_mean",
        "Learned Fixed-Subset Recovery by Layer",
        "Pooled recovery vs low",
        "layerwise_learned_fixed_recovery.png",
    )
    line_plot(
        {
            "oracle_fixed@10%": subset_rows("oracle_fixed", "10pct"),
            "learned_fixed@10%": subset_rows("learned_fixed", "10pct"),
            "oracle_fixed@20%": subset_rows("oracle_fixed", "20pct"),
            "learned_fixed@20%": subset_rows("learned_fixed", "20pct"),
        },
        "pooled_output_relative_error_to_full_mean",
        "Oracle-Fixed vs Learned-Fixed Distance",
        "Pooled relative error to full",
        "oracle_vs_learned_fixed.png",
    )
    line_plot(
        {
            "with_residual_sketch@10%": subset_rows("learned_fixed", "10pct"),
            "without_sketch@10%": subset_rows("low_hidden_only", "10pct"),
            "with_residual_sketch@20%": subset_rows("learned_fixed", "20pct"),
            "without_sketch@20%": subset_rows("low_hidden_only", "20pct"),
        },
        "pooled_output_recovery_vs_low_mean",
        "Residual Sketch Contribution (B2 vs B3)",
        "Pooled recovery vs low",
        "residual_sketch_ablation.png",
    )
    line_plot(
        {
            "learned_fixed": sorted([row for row in summary_rows if row["baseline_type"] == "learned_fixed"], key=lambda row: (row["pct"], row["layer_idx"])),
        },
        "pooled_output_gap_to_oracle_fixed_mean",
        "Gap from Oracle-Fixed Upper Bound",
        "Pooled oracle gap",
        "oracle_gap_learned_fixed.png",
    )

    if comparison_rows:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=180)
        palette = {"10pct": "#175676", "20pct": "#D62839"}
        for pct_label in sorted({row["pct_label"] for row in comparison_rows}):
            rows_ = sorted([row for row in comparison_rows if row["pct_label"] == pct_label], key=lambda row: int(row["layer_idx"]))
            ax.plot(
                [int(row["layer_idx"]) for row in rows_],
                [float(row["pooled_output_recovery_gain_b2_minus_b3"]) for row in rows_],
                marker="o",
                linewidth=2.0,
                color=palette.get(pct_label, "#2A9D8F"),
                label=pct_label,
            )
        ax.set_title("Residual Sketch Gain: B2 minus B3")
        ax.set_xlabel("Target layer")
        ax.set_ylabel("Pooled recovery gain")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        fig.savefig(output_dir / "b2_vs_b3_recovery_gain.png", bbox_inches="tight")
        plt.close(fig)

    scatter_subset = [row for row in summary_rows if row["baseline_type"] in {"learned_fixed", "low_hidden_only"}]
    if scatter_subset:
        fig, ax = plt.subplots(figsize=(9, 6), dpi=180)
        color_map = {"learned_fixed": "#175676", "low_hidden_only": "#D62839"}
        for row in scatter_subset:
            x_value = float(row["relative_cost_vs_dense_head"])
            y_value = float(row.get("pooled_output_recovery_vs_low_mean", 0.0))
            label = f"L{int(row['layer_idx'])}-{row['pct_label']}"
            ax.scatter(x_value, y_value, color=color_map[row["baseline_type"]], s=70, alpha=0.85)
            ax.text(x_value, y_value, label, fontsize=8)
        ax.set_title("Recovery vs Cost Proxy")
        ax.set_xlabel("Relative cost vs dense head")
        ax.set_ylabel("Pooled recovery vs low")
        ax.grid(True, alpha=0.25)
        handles = [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color_map[name], markersize=9, label=name)
            for name in ("learned_fixed", "low_hidden_only")
        ]
        ax.legend(handles=handles, frameon=False)
        fig.savefig(output_dir / "recovery_vs_cost_tradeoff.png", bbox_inches="tight")
        plt.close(fig)


def create_report_bundle(output_dir: Path) -> Path:
    report_files = [
        "config.json",
        "fixed_masks.json",
        "training_history.csv",
        "per_sample_eval.csv",
        "summary.csv",
        "b2_vs_b3_summary.csv",
        "heads.pt",
    ]
    zip_path = output_dir / f"{output_dir.name}_report_bundle.zip"
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename in report_files:
            file_path = output_dir / filename
            if file_path.exists():
                archive.write(file_path, arcname=filename)
        plots_dir = output_dir / "plots"
        if plots_dir.exists():
            for plot_path in sorted(plots_dir.rglob("*")):
                if plot_path.is_file():
                    archive.write(plot_path, arcname=str(plot_path.relative_to(output_dir)))
    return zip_path


def main() -> None:
    args = parse_args()
    set_determinism(args.seed)

    dataset_name = normalize_dataset_name(args.dataset)
    model_name = normalize_model_name(args.model)
    data_root = args.data_root or default_data_root_for_dataset(dataset_name)
    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dtype = parse_model_dtype(args.model_dtype)
    autocast_dtype = parse_autocast_dtype(args.autocast_dtype)
    pretrained = not args.no_pretrained
    backbone_weights = resolve_backbone_weights(model_name, args.backbone_weights, pretrained)
    target_layers = parse_int_list(args.target_layers)
    mask_pcts = parse_pct_list(args.mask_pcts)

    if args.run_mode == "debug":
        mask_pcts = [0.10]
        args.epochs = min(args.epochs, 1)
        args.max_mask_samples = 16 if args.max_mask_samples is None else min(args.max_mask_samples, 16)
        args.max_train_samples = 16 if args.max_train_samples is None else min(args.max_train_samples, 16)
        args.max_eval_samples = 16 if args.max_eval_samples is None else min(args.max_eval_samples, 16)

    combo_specs = make_combo_specs(target_layers=target_layers, mask_pcts=mask_pcts)
    out_dir = resolve_output_dir("fixed_subset_correction", args.out_dir)
    low_input_id = f"{args.low_mode}_level{args.low_level}" if args.low_mode == "gaussian_pyr" else f"{args.low_mode}_x{args.downscale}"

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
        layers=target_layers,
    )

    num_channels_by_layer, num_tokens_by_layer = initialize_shape_metadata(
        loader=loader,
        probe=probe,
        target_layers=target_layers,
        analysis_ratio=args.analysis_ratio,
        low_mode=args.low_mode,
        low_level=args.low_level,
        downscale=args.downscale,
    )
    fixed_masks = build_fixed_masks(
        loader=loader,
        probe=probe,
        combo_specs=combo_specs,
        analysis_ratio=args.analysis_ratio,
        low_mode=args.low_mode,
        low_level=args.low_level,
        downscale=args.downscale,
        max_samples=args.max_mask_samples if args.max_mask_samples is not None else args.max_train_samples,
        max_batches=args.max_batches,
    )

    sketch_dim = 3 * args.sketch_size * args.sketch_size
    heads, cost_rows = build_heads(
        combo_specs=combo_specs,
        fixed_masks=fixed_masks,
        num_channels_by_layer=num_channels_by_layer,
        sketch_dim=sketch_dim,
        sketch_proj_dim=args.sketch_proj_dim,
        hidden_dim=args.head_hidden_dim,
        num_tokens_by_layer=num_tokens_by_layer,
        device=device,
    )
    optimizer = torch.optim.AdamW(heads.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    training_history = train_heads(
        loader=loader,
        probe=probe,
        combo_specs=combo_specs,
        fixed_masks=fixed_masks,
        heads=heads,
        optimizer=optimizer,
        analysis_ratio=args.analysis_ratio,
        low_mode=args.low_mode,
        low_level=args.low_level,
        downscale=args.downscale,
        sketch_size=args.sketch_size,
        max_samples=args.max_train_samples,
        max_batches=args.max_batches,
        epochs=args.epochs,
        device=device,
    )
    eval_rows = evaluate_baselines(
        loader=loader,
        probe=probe,
        combo_specs=combo_specs,
        fixed_masks=fixed_masks,
        heads=heads,
        cost_rows=cost_rows,
        analysis_ratio=args.analysis_ratio,
        low_mode=args.low_mode,
        low_level=args.low_level,
        downscale=args.downscale,
        sketch_size=args.sketch_size,
        max_samples=args.max_eval_samples,
        max_batches=args.max_batches,
        device=device,
    )
    summary_rows = summarize_results(eval_rows)
    comparison_rows = summarize_b2_vs_b3(summary_rows)

    config_payload = {
        "dataset": dataset_name,
        "data_root": str(Path(data_root).expanduser()) if data_root else "",
        "image_size": args.image_size,
        "model_name": model_name,
        "backbone_weights": backbone_weights,
        "pretrained": pretrained,
        "device": str(device),
        "model_dtype_requested": str(model_dtype),
        "model_dtype_runtime": str(probe.model_dtype),
        "autocast_dtype": None if autocast_dtype is None else str(autocast_dtype),
        "seed": args.seed,
        "analysis_ratio": args.analysis_ratio,
        "target_layers": target_layers,
        "mask_pcts": mask_pcts,
        "low_mode": args.low_mode,
        "low_level": args.low_level,
        "downscale": args.downscale,
        "low_input_id": low_input_id,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "head_hidden_dim": args.head_hidden_dim,
        "sketch_size": args.sketch_size,
        "sketch_proj_dim": args.sketch_proj_dim,
        "run_mode": args.run_mode,
        "max_mask_samples": args.max_mask_samples,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "split_safe_protocol": {
            "mask_fit_split": "analysis_only",
            "head_training_split": "analysis_only",
            "evaluation_split": "holdout_only",
            "backbone_frozen": True,
            "same_sample_same_layer_intervention_only": True,
            "cross_sample_activation_reuse": False,
            "full_path_downstream_activation_reuse": False,
        },
    }

    write_json(out_dir / "config.json", config_payload)
    write_json(
        out_dir / "fixed_masks.json",
        {
            combo.name: {
                "layer_idx": combo.layer_idx,
                "pct": combo.pct,
                "pct_label": combo.pct_label,
                "channels": fixed_masks[combo.name],
            }
            for combo in combo_specs
        },
    )
    write_csv(out_dir / "training_history.csv", training_history)
    write_csv(out_dir / "per_sample_eval.csv", eval_rows)
    write_csv(out_dir / "summary.csv", summary_rows)
    write_csv(out_dir / "b2_vs_b3_summary.csv", comparison_rows)
    torch.save(
        {
            "fixed_masks": fixed_masks,
            "heads": heads.state_dict(),
            "cost_rows": cost_rows,
            "config": config_payload,
        },
        out_dir / "heads.pt",
    )
    write_plots(out_dir / "plots", summary_rows, comparison_rows)
    create_report_bundle(out_dir)

    print("[fixed_subset_correction] Done")
    print(f"  - output_dir: {out_dir}")
    print(f"  - combos: {[combo.name for combo in combo_specs]}")
    print(f"  - eval_rows: {len(eval_rows)}")


if __name__ == "__main__":
    main()
