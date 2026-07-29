from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image


def ensure_cuda_toolchain_on_path() -> None:
    """Make Triton's ptxas lookup robust on hosts with versioned CUDA installs."""
    discovered = shutil.which("ptxas")
    candidates = (
        Path(discovered) if discovered else Path("__missing_ptxas__"),
        Path(os.environ.get("CUDA_HOME", "")) / "bin" / "ptxas",
        Path("/usr/local/cuda/bin/ptxas"),
        Path("/usr/local/cuda-13.1/bin/ptxas"),
        Path("/usr/local/cuda-13/bin/ptxas"),
    )
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            cuda_bin = str(candidate.parent)
            os.environ["PATH"] = f"{cuda_bin}:{os.environ.get('PATH', '')}"
            os.environ.setdefault("CUDA_HOME", str(candidate.parent.parent))
            os.environ["TRITON_PTXAS_PATH"] = str(candidate.resolve())
            return
    raise RuntimeError(
        "Triton correction kernels require ptxas, but no executable was found. "
        "Set CUDA_HOME or TRITON_PTXAS_PATH."
    )


ensure_cuda_toolchain_on_path()

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from tqdm import tqdm  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.shared.lowres_cache_lift import (  # noqa: E402
    lift_partial_token_cache,
    lift_token_grid,
    tensor_cache_nbytes,
)
from offload.common import ExperimentConfig  # noqa: E402
from offload.mobile.dataset import get_dataset_loader  # noqa: E402
from offload.server.model import get_model_executor  # noqa: E402


DEFAULT_CONFIG = "offload/config/ade20k_m2f_interleaved_static.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Probe DINOv3 low-resolution token-grid approximate forward followed by "
            "spatial cache lifting and existing partial-token correction."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", default="")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--high-size", type=int, default=512)
    parser.add_argument("--low-scale", type=float, default=0.5)
    parser.add_argument("--keep-ratios", default="0.25,0.5,0.75,1.0")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--include-image-upsample-baseline",
        action="store_true",
        help="Also run the existing low-image-upsample/full-token-grid approximate path.",
    )
    parser.add_argument(
        "--skip-head",
        action="store_true",
        help="Measure backbone feature quality only; skip M2F prediction and mIoU.",
    )
    parser.add_argument(
        "--output",
        default="logs/analysis/dinov3_lowres_cache_lift_probe.json",
    )
    return parser.parse_args()


def load_config(path: str) -> tuple[ExperimentConfig, dict]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    return ExperimentConfig(**raw), raw


def cuda_elapsed_ms(function):
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = function()
    end.record()
    torch.cuda.synchronize()
    return result, float(start.elapsed_time(end))


def warmup_cuda(function) -> None:
    result = function()
    torch.cuda.synchronize()
    del result


def clone_tensor_cache(cache: Dict[str, Any]) -> Dict[str, Any]:
    cloned: Dict[str, Any] = {}
    for key, value in cache.items():
        if torch.is_tensor(value):
            cloned[key] = value.clone()
        else:
            cloned[key] = copy.deepcopy(value)
    return cloned


def feature_metrics(reference: torch.Tensor, candidate: torch.Tensor, num_prefix: int) -> dict:
    reference = reference[:, num_prefix:].float()
    candidate = candidate[:, num_prefix:].float()
    difference = candidate - reference
    denominator = torch.linalg.vector_norm(reference).clamp_min(torch.finfo(torch.float32).eps)
    return {
        "relative_l2": float((torch.linalg.vector_norm(difference) / denominator).item()),
        "cosine_similarity": float(
            F.cosine_similarity(reference.reshape(1, -1), candidate.reshape(1, -1)).item()
        ),
    }


def run_full_backbone(backbone, tokens, rope, interaction_indexes):
    x = tokens
    intermediates = []
    for layer_idx, block in enumerate(backbone.blocks):
        x = block(x, rope)
        if layer_idx in interaction_indexes:
            intermediates.append(x)
    return x, intermediates


def run_approx_backbone(
    backbone,
    tokens,
    rope,
    interaction_indexes,
    layer_tags,
    *,
    cache_pre_rope_k: bool,
):
    x = tokens
    cache: Dict[str, Any] = {}
    intermediates = []
    for layer_idx, (block, tag) in enumerate(zip(backbone.blocks, layer_tags)):
        x, cache = block.approx(
            x,
            rope,
            cache,
            tag=tag,
            appcorr_method="partial_token",
            server_pscore="cls_attn_prob",
            server_pscore_weight=1.0,
            cache_pre_rope_k=cache_pre_rope_k,
            debug=False,
        )
        if layer_idx in interaction_indexes:
            intermediates.append(x)
    return x, intermediates, cache


def run_partial_correction(
    backbone,
    high_tokens,
    high_rope,
    cache,
    interaction_indexes,
    layer_tags,
    keep_ratio: float,
):
    x = high_tokens
    intermediates = []
    dindice = torch.arange(
        high_tokens.shape[1],
        device=high_tokens.device,
        dtype=torch.long,
    ).unsqueeze(0).expand(high_tokens.shape[0], -1)
    for layer_idx, (block, tag) in enumerate(zip(backbone.blocks, layer_tags)):
        x, cache = block.correct(
            x,
            dindice,
            high_rope,
            cache,
            tag=tag,
            appcorr_method="partial_token",
            token_keep_ratio=float(keep_ratio),
            token_keep_thres=None,
            mobile_pscore="none",
            mobile_pscore_weight=0.0,
            server_pscore="cls_attn_prob",
            server_pscore_weight=1.0,
            pscore_fusion="add",
            sdpa_query_bucket_size=0,
            # M2F consumes snapshots from layers 9/19/29/39. In-place residual
            # addition would mutate previously appended snapshots in later layers.
            inplace_residual_add=False,
            debug=False,
        )
        if layer_idx in interaction_indexes:
            intermediates.append(x)
    return x, intermediates, cache


def build_head_context(source: Dict[str, Any], intermediates: list[torch.Tensor]) -> dict:
    return {
        "m2f_intermediate_raw": [intermediates],
        "m2f_spm_c_cat": [source["spm_c_cat"]],
        "m2f_spm_c1_raw": [source["spm_c1_raw"]],
        "m2f_spm_c2_len": [source["spm_c2_len"]],
        "m2f_spm_c3_len": [source["spm_c3_len"]],
        "m2f_source_shapes": [source["source_shape"]],
        "m2f_deform_in1": [source["deform_in1"]],
        "m2f_deform_in2": [source["deform_in2"]],
    }


def predict_mask(executor, source, intermediates, output_hw):
    context = build_head_context(source, intermediates)
    features = executor._run_adapter_postprocess(0, context)
    prediction = executor._run_m2f_head_predict(
        executor.model.segmentation_model[1],
        features,
        "m2f",
        output_hw,
    )
    return prediction.argmax(dim=1)[0].to(torch.uint8).cpu()


def add_areas(accumulator: torch.Tensor, loader, prediction, label) -> None:
    ground_truth = label.get("orig_mask")
    if ground_truth is None:
        ground_truth = label["mask"]
    accumulator += loader._intersect_and_union(prediction, ground_truth)


def metrics_from_areas(loader, areas: torch.Tensor) -> dict:
    return loader._metrics_from_areas(areas[0], areas[1], areas[2], areas[3])


def prepare_inputs(image_tensor: torch.Tensor, executor, high_size: int, low_size: int):
    if image_tensor.ndim != 3:
        raise ValueError(f"Expected CHW image, got {tuple(image_tensor.shape)}")
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy()
    high_image = Image.fromarray(image_np).resize(
        (high_size, high_size),
        Image.Resampling.BILINEAR,
    )
    high_input = executor._pil_to_normalized_tensor(high_image)
    low_input = F.interpolate(
        high_input.float(),
        size=(low_size, low_size),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).to(dtype=executor.autocast_dtype)
    low_up_input = F.interpolate(
        low_input.float(),
        size=(high_size, high_size),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    ).to(dtype=executor.autocast_dtype)
    return high_input, low_input, low_up_input


def summarize_rows(rows: list[dict]) -> dict[str, dict]:
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_variant[row["variant"]].append(row)
    summary = {}
    for variant, variant_rows in sorted(by_variant.items()):
        summary[variant] = {
            "num_samples": len(variant_rows),
            "relative_l2_mean": float(np.mean([row["relative_l2"] for row in variant_rows])),
            "cosine_similarity_mean": float(
                np.mean([row["cosine_similarity"] for row in variant_rows])
            ),
            "approx_ms_mean": float(np.mean([row["approx_ms"] for row in variant_rows])),
            "cache_lift_ms_mean": float(
                np.mean([row["cache_lift_ms"] for row in variant_rows])
            ),
            "correction_ms_mean": float(
                np.mean([row["correction_ms"] for row in variant_rows])
            ),
            "compute_ms_mean": float(np.mean([row["compute_ms"] for row in variant_rows])),
            "cache_mib_mean": float(np.mean([row["cache_mib"] for row in variant_rows])),
            "source_cache_mib_mean": float(
                np.mean([row["source_cache_mib"] for row in variant_rows])
            ),
            "full_ms_mean": float(np.mean([row["full_ms"] for row in variant_rows])),
            "speedup_vs_full_mean": float(
                np.mean(
                    [
                        row["full_ms"] / max(row["compute_ms"], 1e-12)
                        for row in variant_rows
                    ]
                )
            ),
        }
    return summary


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This probe requires CUDA.")
    keep_ratios = [float(value) for value in args.keep_ratios.split(",") if value.strip()]
    if not keep_ratios or any(not 0.0 <= ratio <= 1.0 for ratio in keep_ratios):
        raise ValueError("--keep-ratios must contain values in [0, 1]")
    low_size = int(round(args.high_size * args.low_scale))
    if low_size <= 0 or low_size % 16 != 0 or args.high_size % 16 != 0:
        raise ValueError("High and low input sizes must be positive multiples of patch size 16")

    config, raw_config = load_config(args.config)
    device = torch.device(args.device)
    executor = get_model_executor(config.model_name, device)
    executor.load_model(config.model_name, config)
    adapter = executor.model.segmentation_model[0]
    backbone = adapter.backbone
    interaction_indexes = set(adapter.interaction_indexes)
    layer_tags = [f"cache_lift_layer{idx}" for idx in range(len(backbone.blocks))]
    num_prefix = 1 + backbone.n_storage_tokens

    dataset_kwargs = dict(raw_config.get("dataset_kwargs", {}))
    dataset_kwargs["emit_original_image"] = True
    loader = get_dataset_loader(
        "ade20k",
        args.data_root,
        batch_size=1,
        image_size=int(raw_config.get("image_shape", [896])[0]),
        num_workers=args.num_workers,
        **dataset_kwargs,
    )
    dataloader = loader.get_loader()

    rows: list[dict] = []
    area_totals: dict[str, torch.Tensor] = defaultdict(
        lambda: torch.zeros(4, 150, dtype=torch.float64)
    )

    for sample_idx, (image_batch, label_batch) in enumerate(
        tqdm(dataloader, total=min(len(dataloader), args.max_samples))
    ):
        if sample_idx >= args.max_samples:
            break
        image_tensor = image_batch[0] if isinstance(image_batch, list) else image_batch[0]
        label = label_batch[0]
        high_input, low_input, low_up_input = prepare_inputs(
            image_tensor,
            executor,
            args.high_size,
            low_size,
        )

        with torch.inference_mode(), torch.autocast("cuda", executor.autocast_dtype):
            high_source = executor._prepare_single_source(high_input, adapter, backbone)
            low_tokens, low_hw = backbone.prepare_tokens_with_masks(low_input)
            low_rope = backbone.rope_embed(H=low_hw[0], W=low_hw[1])
            low_up_tokens, low_up_hw = backbone.prepare_tokens_with_masks(low_up_input)
            low_up_rope = backbone.rope_embed(H=low_up_hw[0], W=low_up_hw[1])
            high_tokens = high_source["x_backbone"]
            high_rope = high_source["rope_sincos"]
            high_hw = high_source["token_shape"]

            full_call = lambda: run_full_backbone(
                backbone,
                high_tokens.clone(),
                high_rope,
                interaction_indexes,
            )
            warmup_cuda(full_call)
            (full_output, full_intermediates), full_ms = cuda_elapsed_ms(full_call)
            rows.append(
                {
                    "sample_idx": sample_idx,
                    "variant": "full",
                    "relative_l2": 0.0,
                    "cosine_similarity": 1.0,
                    "approx_ms": full_ms,
                    "cache_lift_ms": 0.0,
                    "correction_ms": 0.0,
                    "compute_ms": full_ms,
                    "cache_mib": 0.0,
                    "source_cache_mib": 0.0,
                    "full_ms": full_ms,
                }
            )
            if not args.skip_head:
                pred = predict_mask(
                    executor,
                    high_source,
                    full_intermediates,
                    tuple(np.asarray(label["orig_mask"]).shape),
                )
                add_areas(area_totals["full"], loader, pred, label)

            low_approx_call = lambda: run_approx_backbone(
                backbone,
                low_tokens.clone(),
                low_rope,
                interaction_indexes,
                layer_tags,
                cache_pre_rope_k=True,
            )
            warmup_cuda(low_approx_call)
            (low_output, low_intermediates, low_cache), low_approx_ms = cuda_elapsed_ms(
                low_approx_call
            )
            lift_call = lambda: lift_partial_token_cache(
                low_cache,
                layer_tags,
                low_hw,
                high_hw,
                num_prefix,
                high_rope,
            )
            warmup_cuda(lift_call)
            lifted_cache, lift_ms = cuda_elapsed_ms(lift_call)
            lifted_output = lift_token_grid(
                low_output,
                low_hw,
                high_hw,
                num_prefix,
            )
            lifted_intermediates = [
                lift_token_grid(value, low_hw, high_hw, num_prefix)
                for value in low_intermediates
            ]
            approx_metrics = feature_metrics(full_output, lifted_output, num_prefix)
            low_cache_mib = tensor_cache_nbytes(low_cache) / (1024**2)
            lifted_cache_mib = tensor_cache_nbytes(lifted_cache) / (1024**2)
            rows.append(
                {
                    "sample_idx": sample_idx,
                    "variant": "low_token_grid_approx",
                    **approx_metrics,
                    "approx_ms": low_approx_ms,
                    "cache_lift_ms": lift_ms,
                    "correction_ms": 0.0,
                    "compute_ms": low_approx_ms + lift_ms,
                    "cache_mib": lifted_cache_mib,
                    "source_cache_mib": low_cache_mib,
                    "full_ms": full_ms,
                }
            )
            if not args.skip_head:
                pred = predict_mask(
                    executor,
                    high_source,
                    lifted_intermediates,
                    tuple(np.asarray(label["orig_mask"]).shape),
                )
                add_areas(area_totals["low_token_grid_approx"], loader, pred, label)

            for keep_ratio in keep_ratios:
                warmup_cache = clone_tensor_cache(lifted_cache)
                warmup_cuda(
                    lambda ratio=keep_ratio, cache=warmup_cache: run_partial_correction(
                        backbone,
                        high_tokens.clone(),
                        high_rope,
                        cache,
                        interaction_indexes,
                        layer_tags,
                        ratio,
                    )
                )
                del warmup_cache
                correction_cache = clone_tensor_cache(lifted_cache)
                (corrected_output, corrected_intermediates, _), correction_ms = cuda_elapsed_ms(
                    lambda ratio=keep_ratio, cache=correction_cache: run_partial_correction(
                        backbone,
                        high_tokens.clone(),
                        high_rope,
                        cache,
                        interaction_indexes,
                        layer_tags,
                        ratio,
                    )
                )
                variant = f"low_token_grid_correct_{keep_ratio:.2f}"
                rows.append(
                    {
                        "sample_idx": sample_idx,
                        "variant": variant,
                        **feature_metrics(full_output, corrected_output, num_prefix),
                        "approx_ms": low_approx_ms,
                        "cache_lift_ms": lift_ms,
                        "correction_ms": correction_ms,
                        "compute_ms": low_approx_ms + lift_ms + correction_ms,
                        "cache_mib": lifted_cache_mib,
                        "source_cache_mib": low_cache_mib,
                        "full_ms": full_ms,
                    }
                )
                if not args.skip_head:
                    pred = predict_mask(
                        executor,
                        high_source,
                        corrected_intermediates,
                        tuple(np.asarray(label["orig_mask"]).shape),
                    )
                    add_areas(area_totals[variant], loader, pred, label)
                del correction_cache, corrected_output, corrected_intermediates

            if args.include_image_upsample_baseline:
                image_up_approx_call = lambda: run_approx_backbone(
                    backbone,
                    low_up_tokens.clone(),
                    low_up_rope,
                    interaction_indexes,
                    layer_tags,
                    cache_pre_rope_k=False,
                )
                warmup_cuda(image_up_approx_call)
                (
                    image_up_output,
                    image_up_intermediates,
                    image_up_cache,
                ), image_up_approx_ms = cuda_elapsed_ms(
                    image_up_approx_call
                )
                image_up_cache_mib = tensor_cache_nbytes(image_up_cache) / (1024**2)
                rows.append(
                    {
                        "sample_idx": sample_idx,
                        "variant": "image_upsample_approx",
                        **feature_metrics(full_output, image_up_output, num_prefix),
                        "approx_ms": image_up_approx_ms,
                        "cache_lift_ms": 0.0,
                        "correction_ms": 0.0,
                        "compute_ms": image_up_approx_ms,
                        "cache_mib": image_up_cache_mib,
                        "source_cache_mib": image_up_cache_mib,
                        "full_ms": full_ms,
                    }
                )
                if not args.skip_head:
                    pred = predict_mask(
                        executor,
                        high_source,
                        image_up_intermediates,
                        tuple(np.asarray(label["orig_mask"]).shape),
                    )
                    add_areas(area_totals["image_upsample_approx"], loader, pred, label)
                for keep_ratio in keep_ratios:
                    warmup_cache = clone_tensor_cache(image_up_cache)
                    warmup_cuda(
                        lambda ratio=keep_ratio, cache=warmup_cache: run_partial_correction(
                            backbone,
                            high_tokens.clone(),
                            high_rope,
                            cache,
                            interaction_indexes,
                            layer_tags,
                            ratio,
                        )
                    )
                    del warmup_cache
                    correction_cache = clone_tensor_cache(image_up_cache)
                    (
                        corrected_output,
                        corrected_intermediates,
                        _,
                    ), correction_ms = cuda_elapsed_ms(
                        lambda ratio=keep_ratio, cache=correction_cache: run_partial_correction(
                            backbone,
                            high_tokens.clone(),
                            high_rope,
                            cache,
                            interaction_indexes,
                            layer_tags,
                            ratio,
                        )
                    )
                    variant = f"image_upsample_correct_{keep_ratio:.2f}"
                    rows.append(
                        {
                            "sample_idx": sample_idx,
                            "variant": variant,
                            **feature_metrics(full_output, corrected_output, num_prefix),
                            "approx_ms": image_up_approx_ms,
                            "cache_lift_ms": 0.0,
                            "correction_ms": correction_ms,
                            "compute_ms": image_up_approx_ms + correction_ms,
                            "cache_mib": image_up_cache_mib,
                            "source_cache_mib": image_up_cache_mib,
                            "full_ms": full_ms,
                        }
                    )
                    if not args.skip_head:
                        pred = predict_mask(
                            executor,
                            high_source,
                            corrected_intermediates,
                            tuple(np.asarray(label["orig_mask"]).shape),
                        )
                        add_areas(area_totals[variant], loader, pred, label)
                    del correction_cache, corrected_output, corrected_intermediates

        del high_source, low_cache, lifted_cache
        torch.cuda.empty_cache()

    summary = summarize_rows(rows)
    for variant, areas in area_totals.items():
        summary.setdefault(variant, {})["segmentation"] = metrics_from_areas(loader, areas)

    payload = {
        "config": args.config,
        "device": args.device,
        "high_size": args.high_size,
        "low_size": low_size,
        "low_scale": args.low_scale,
        "keep_ratios": keep_ratios,
        "num_samples": min(args.max_samples, len(dataloader)),
        "note": (
            "Whole-image 512-style probe. M2F SPM/deform features are computed at high "
            "resolution for every variant; reported compute_ms covers ViT approx, cache "
            "lifting, and ViT correction only."
        ),
        "summary": summary,
        "rows": rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(payload["summary"], indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
