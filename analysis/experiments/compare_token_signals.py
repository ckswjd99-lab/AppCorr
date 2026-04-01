import argparse
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.shared.common import (
    build_analysis_loader,
    default_data_root_for_dataset,
    make_lowres_and_bicubic,
    normalize_dataset_name,
    resolve_output_dir,
    run_sr_model_in_chunks,
    write_csv,
    write_json,
)
LOWER_IS_BETTER = {"js_divergence", "l1_mean", "l2_mean"}
HIGHER_IS_BETTER = {"pearson", "spearman", "topk_overlap"}


def parse_args():
    parser = argparse.ArgumentParser(description="Compare DINOv3 attention and FFN signals across L0, L2 bicubic, and L2SR.")
    parser.add_argument("--dataset", type=str, default="imagenet-1k", help="Dataset: imagenet-1k or coco.")
    parser.add_argument("--data-root", type=str, default=None, help="Dataset root. Ignored for COCO.")
    parser.add_argument("--batch-size", type=int, default=4, help="Analysis batch size.")
    parser.add_argument("--max-batches", type=int, default=1, help="Maximum number of batches to process.")
    parser.add_argument("--image-size", type=int, default=256, help="Square input size for DINOv3.")
    parser.add_argument("--downscale", type=int, default=4, help="Downsample factor for L2.")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers.")
    parser.add_argument("--device", type=str, default=None, help="Torch device. Default: cuda if available else cpu.")
    parser.add_argument("--dtype", type=str, default="fp16", help="SR dtype: fp16, bf16, or fp32.")
    parser.add_argument("--weights-dir", type=str, default="~/cjpark/weights/realesrgan", help="Real-ESRGAN checkpoint directory.")
    parser.add_argument("--sr-batch-size", type=int, default=None, help="Batch size used for SR inference. Default: batch-size.")
    parser.add_argument("--layers", type=str, default=None, help="Comma-separated layer indices. Default: all 40 layers.")
    parser.add_argument("--topk-ratio", type=float, default=0.1, help="Ratio for top-k overlap.")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory. Default: logs/analysis/token_signals_<timestamp>.")
    return parser.parse_args()


def parse_layers(value: str | None) -> list[int] | None:
    if value is None or value.strip() == "":
        return None
    layers = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not layers:
        return None
    return layers


def _safe_distribution(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    arr = np.clip(arr, 0.0, None)
    total = arr.sum()
    if total <= 0:
        return np.full_like(arr, 1.0 / max(arr.size, 1))
    return arr / total


def js_divergence(ref: np.ndarray, cand: np.ndarray) -> float:
    p = _safe_distribution(ref)
    q = _safe_distribution(cand)
    m = 0.5 * (p + q)
    eps = 1e-12
    p = np.clip(p, eps, None)
    q = np.clip(q, eps, None)
    m = np.clip(m, eps, None)
    return float(0.5 * (np.sum(p * np.log(p / m)) + np.sum(q * np.log(q / m))))


def pearson_corr(ref: np.ndarray, cand: np.ndarray) -> float | None:
    x = np.asarray(ref, dtype=np.float64).reshape(-1)
    y = np.asarray(cand, dtype=np.float64).reshape(-1)
    if x.size < 2:
        return None
    x = x - x.mean()
    y = y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        return None
    return float(np.dot(x, y) / denom)


def spearman_corr(ref: np.ndarray, cand: np.ndarray) -> float | None:
    x = np.asarray(ref).reshape(-1)
    y = np.asarray(cand).reshape(-1)
    if x.size < 2:
        return None
    x_rank = np.argsort(np.argsort(x, kind="stable"), kind="stable")
    y_rank = np.argsort(np.argsort(y, kind="stable"), kind="stable")
    return pearson_corr(x_rank, y_rank)


def topk_overlap(ref: np.ndarray, cand: np.ndarray, ratio: float) -> float:
    x = np.asarray(ref).reshape(-1)
    y = np.asarray(cand).reshape(-1)
    k = max(1, min(x.size, int(math.ceil(x.size * ratio))))
    x_idx = set(np.argpartition(x, -k)[-k:].tolist())
    y_idx = set(np.argpartition(y, -k)[-k:].tolist())
    return float(len(x_idx & y_idx) / k)


def l1_mean(ref: np.ndarray, cand: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(ref, dtype=np.float64) - np.asarray(cand, dtype=np.float64))))


def l2_mean(ref: np.ndarray, cand: np.ndarray) -> float:
    diff = np.asarray(ref, dtype=np.float64) - np.asarray(cand, dtype=np.float64)
    return float(np.sqrt(np.mean(diff * diff)))


def normalized_l2_error(ref: np.ndarray, cand: np.ndarray) -> float:
    ref_arr = np.asarray(ref, dtype=np.float64)
    cand_arr = np.asarray(cand, dtype=np.float64)
    numerator = np.linalg.norm((cand_arr - ref_arr).reshape(-1))
    denominator = np.linalg.norm(ref_arr.reshape(-1))
    if denominator == 0:
        return 0.0 if numerator == 0 else float("inf")
    return float(numerator / denominator)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    row_sums = matrix.sum(axis=-1, keepdims=True)
    return np.divide(matrix, np.clip(row_sums, 1e-12, None))


def build_valid_token_masks(pixel_valid_masks: torch.Tensor, patch_start: int, patch_size: tuple[int, int]) -> np.ndarray:
    batch, height, width = pixel_valid_masks.shape
    patch_h, patch_w = patch_size
    patch_mask = pixel_valid_masks.reshape(
        batch,
        height // patch_h,
        patch_h,
        width // patch_w,
        patch_w,
    ).any(dim=2).any(dim=3).reshape(batch, -1)
    pretoken_mask = torch.ones((batch, patch_start), dtype=torch.bool)
    return torch.cat([pretoken_mask, patch_mask], dim=1).cpu().numpy()


def vector_metrics(ref: np.ndarray, cand: np.ndarray, topk_ratio: float) -> dict[str, float | None]:
    return {
        "js_divergence": js_divergence(ref, cand),
        "pearson": pearson_corr(ref, cand),
        "spearman": spearman_corr(ref, cand),
        "topk_overlap": topk_overlap(ref, cand, topk_ratio),
        "l1_mean": l1_mean(ref, cand),
        "l2_mean": l2_mean(ref, cand),
    }


def matrix_rowwise_metrics(ref: np.ndarray, cand: np.ndarray, topk_ratio: float) -> dict[str, float | None]:
    row_metrics = defaultdict(list)
    for ref_row, cand_row in zip(ref, cand):
        metrics = vector_metrics(ref_row, cand_row, topk_ratio)
        for name, value in metrics.items():
            if value is not None:
                row_metrics[name].append(value)

    return {
        key: float(np.mean(values)) if values else None
        for key, values in row_metrics.items()
    }


def _row_template(
    family: str,
    signal_name: str,
    scope: str,
    variant: str,
    layer: int,
    batch_idx: int,
    sample_key: str,
    dataset: str,
) -> dict[str, object]:
    return {
        "family": family,
        "signal_name": signal_name,
        "scope": scope,
        "variant": variant,
        "layer": layer,
        "batch_idx": batch_idx,
        "sample_key": sample_key,
        "dataset": dataset,
        "js_divergence": None,
        "pearson": None,
        "spearman": None,
        "topk_overlap": None,
        "l1_mean": None,
        "l2_mean": None,
        "normalized_l2_error": None,
    }


def compare_attention_rows(
    ref_signals: dict[int, dict[str, torch.Tensor]],
    cand_signals: dict[int, dict[str, torch.Tensor]],
    valid_token_masks: np.ndarray,
    sample_keys: list[str],
    batch_idx: int,
    dataset: str,
    variant: str,
    topk_ratio: float,
) -> list[dict[str, object]]:
    rows = []
    for layer in sorted(ref_signals):
        ref_attn = ref_signals[layer]["attn_mean"].numpy()
        cand_attn = cand_signals[layer]["attn_mean"].numpy()

        for sample_offset, sample_key in enumerate(sample_keys):
            valid_mask = valid_token_masks[sample_offset].astype(bool)
            full_all_ref = ref_attn[sample_offset][np.ix_(valid_mask, valid_mask)]
            full_all_cand = cand_attn[sample_offset][np.ix_(valid_mask, valid_mask)]
            full_all_ref = normalize_rows(full_all_ref)
            full_all_cand = normalize_rows(full_all_cand)

            row = _row_template("attention", "attn_prob", "all_tokens", variant, layer, batch_idx, sample_key, dataset)
            row.update(matrix_rowwise_metrics(full_all_ref, full_all_cand, topk_ratio))
            row["normalized_l2_error"] = normalized_l2_error(full_all_ref, full_all_cand)
            rows.append(row)

    return rows


def compare_ffn_rows(
    ref_signals: dict[int, dict[str, torch.Tensor]],
    cand_signals: dict[int, dict[str, torch.Tensor]],
    valid_token_masks: np.ndarray,
    sample_keys: list[str],
    batch_idx: int,
    dataset: str,
    variant: str,
    topk_ratio: float,
) -> list[dict[str, object]]:
    rows = []
    for layer in sorted(ref_signals):
        for signal_name in ("gate_score", "effective_gate_score"):
            ref_score = ref_signals[layer][signal_name].numpy()
            cand_score = cand_signals[layer][signal_name].numpy()

            for sample_offset, sample_key in enumerate(sample_keys):
                valid_mask = valid_token_masks[sample_offset].astype(bool)
                ref_valid = ref_score[sample_offset][valid_mask]
                cand_valid = cand_score[sample_offset][valid_mask]
                row = _row_template("ffn", signal_name, "all_tokens", variant, layer, batch_idx, sample_key, dataset)
                row.update(vector_metrics(ref_valid, cand_valid, topk_ratio))
                row["normalized_l2_error"] = normalized_l2_error(ref_valid, cand_valid)
                rows.append(row)

    return rows


def summarize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], dict[str, list[float]]] = {}
    key_fields = ("family", "signal_name", "scope", "variant", "layer", "dataset")
    metric_fields = ("js_divergence", "pearson", "spearman", "topk_overlap", "l1_mean", "l2_mean")

    for row in rows:
        key = tuple(row[field] for field in key_fields)
        if key not in grouped:
            grouped[key] = {field: [] for field in metric_fields}
        for field in metric_fields:
            value = row[field]
            if value is not None:
                grouped[key][field].append(float(value))

    summary = []
    for key in sorted(grouped):
        metric_lists = grouped[key]
        item = dict(zip(key_fields, key))
        item["num_samples"] = max((len(values) for values in metric_lists.values()), default=0)
        for field in metric_fields:
            values = metric_lists[field]
            item[field] = float(np.mean(values)) if values else None
        summary.append(item)
    return summary


def build_improvement_summary(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    baseline_variant = "bicubic_x4"
    paired = {}
    for row in rows:
        key = (
            row["family"],
            row["signal_name"],
            row["scope"],
            row["layer"],
            row["dataset"],
            row["sample_key"],
        )
        paired.setdefault(key, {})[row["variant"]] = row

    grouped: dict[tuple[object, ...], dict[str, list[float]]] = {}
    metric_fields = ("js_divergence", "pearson", "spearman", "topk_overlap", "l1_mean", "l2_mean")

    def metric_to_error(metric: str, value: float) -> float:
        if metric in LOWER_IS_BETTER:
            return value
        return 1.0 - value

    for key, variants in paired.items():
        baseline = variants.get(baseline_variant)
        if baseline is None:
            continue
        for variant_name, row in variants.items():
            if variant_name == baseline_variant:
                continue
            group_key = (
                row["family"],
                row["signal_name"],
                row["scope"],
                row["variant"],
                row["dataset"],
                row["layer"],
            )
            if group_key not in grouped:
                grouped[group_key] = defaultdict(list)

            for metric in metric_fields:
                base_value = baseline[metric]
                cand_value = row[metric]
                if base_value is None or cand_value is None:
                    continue
                base_value = float(base_value)
                cand_value = float(cand_value)
                bicubic_error = metric_to_error(metric, base_value)
                sr_error = metric_to_error(metric, cand_value)
                ratio = sr_error / max(abs(bicubic_error), 1e-12)
                grouped[group_key][f"{metric}_ratio"].append(ratio)
                grouped[group_key][f"{metric}_bicubic_error"].append(bicubic_error)
                grouped[group_key][f"{metric}_sr_error"].append(sr_error)

            base_norm_l2 = baseline.get("normalized_l2_error")
            cand_norm_l2 = row.get("normalized_l2_error")
            if base_norm_l2 is not None and cand_norm_l2 is not None:
                grouped[group_key]["bicubic_normalized_l2_error"].append(float(base_norm_l2))
                grouped[group_key]["sr_normalized_l2_error"].append(float(cand_norm_l2))

    summary = []
    for key in sorted(grouped):
        data = grouped[key]
        item = {
            "family": key[0],
            "signal_name": key[1],
            "scope": key[2],
            "variant": key[3],
            "dataset": key[4],
            "layer": key[5],
        }
        for metric in metric_fields:
            ratios = data.get(f"{metric}_ratio", [])
            item[f"{metric}_sr_error_over_bicubic_error"] = float(np.mean(ratios)) if ratios else None
        for metric in metric_fields:
            bicubic_errors = data.get(f"{metric}_bicubic_error", [])
            sr_errors = data.get(f"{metric}_sr_error", [])
            item[f"{metric}_bicubic_error"] = float(np.mean(bicubic_errors)) if bicubic_errors else None
            item[f"{metric}_sr_error"] = float(np.mean(sr_errors)) if sr_errors else None
        bicubic_norm_l2 = data.get("bicubic_normalized_l2_error", [])
        sr_norm_l2 = data.get("sr_normalized_l2_error", [])
        item["bicubic_error_l2_over_reference_l2"] = float(np.mean(bicubic_norm_l2)) if bicubic_norm_l2 else None
        item["sr_error_l2_over_reference_l2"] = float(np.mean(sr_norm_l2)) if sr_norm_l2 else None
        summary.append(item)
    return summary


def print_quick_summary(summary_rows: list[dict[str, object]]) -> None:
    focus = [
        ("attention", "attn_prob", "all_tokens", "js_divergence"),
        ("ffn", "gate_score", "all_tokens", "js_divergence"),
        ("ffn", "effective_gate_score", "all_tokens", "js_divergence"),
    ]
    print("[TokenSignals] Quick summary")
    for family, signal_name, scope, metric in focus:
        target_rows = [
            row for row in summary_rows
            if row["family"] == family and row["signal_name"] == signal_name and row["scope"] == scope
        ]
        if not target_rows:
            continue
        by_variant = defaultdict(list)
        for row in target_rows:
            value = row.get(metric)
            if value is not None:
                by_variant[row["variant"]].append(float(value))
        parts = []
        for variant, values in sorted(by_variant.items()):
            parts.append(f"{variant}={np.mean(values):.5f}")
        print(f"  - {family}/{signal_name}/{scope}/{metric}: " + ", ".join(parts))


def main():
    args = parse_args()
    from analysis.shared.dinov3_probe import Dinov3SignalProbe

    dataset_name = normalize_dataset_name(args.dataset)
    if args.image_size % args.downscale != 0:
        raise ValueError(f"image_size={args.image_size} must be divisible by downscale={args.downscale}")

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sr_batch_size = args.sr_batch_size or args.batch_size
    data_root = args.data_root if args.data_root is not None else default_data_root_for_dataset(dataset_name)
    layers = parse_layers(args.layers)

    out_dir = resolve_output_dir("token_signals", args.out_dir)
    write_json(out_dir / "config.json", {
        "dataset": dataset_name,
        "data_root": data_root,
        "batch_size": args.batch_size,
        "max_batches": args.max_batches,
        "image_size": args.image_size,
        "downscale": args.downscale,
        "num_workers": args.num_workers,
        "device": str(device),
        "sr_dtype": args.dtype,
        "sr_batch_size": sr_batch_size,
        "weights_dir": args.weights_dir,
        "layers": layers,
        "topk_ratio": args.topk_ratio,
        "variants": ["l0", "bicubic_x4", "realesr_general_x4v3", "realesrgan_x4plus"],
    })

    loader = build_analysis_loader(
        dataset_name=dataset_name,
        data_root=data_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        num_workers=args.num_workers,
    )
    probe = Dinov3SignalProbe(device=device, image_size=args.image_size, layers=layers)
    patch_size = probe.config.patch_size if isinstance(probe.config.patch_size, tuple) else (probe.config.patch_size, probe.config.patch_size)

    processed_samples = []
    attention_rows: list[dict[str, object]] = []
    ffn_rows: list[dict[str, object]] = []

    for batch_idx, batch in enumerate(loader):
        if batch_idx >= args.max_batches:
            break

        images = batch["images"].to(dtype=torch.uint8)
        valid_masks = batch["valid_masks"].to(dtype=torch.bool)
        sample_keys = batch["sample_keys"]
        paths = batch["paths"]
        labels = batch["labels"]
        processed_samples.extend([
            {
                "batch_idx": batch_idx,
                "sample_key": sample_key,
                "path": path,
                "label": label,
            }
            for sample_key, path, label in zip(sample_keys, paths, labels)
        ])

        print(f"[TokenSignals] Batch {batch_idx + 1}/{args.max_batches}: probing L0 on {len(sample_keys)} images")
        lowres, bicubic = make_lowres_and_bicubic(images, args.downscale)
        ref_signals = probe.run(images)
        valid_token_masks = build_valid_token_masks(valid_masks, probe.patch_start, patch_size)

        candidates = [("bicubic_x4", bicubic)]
        for model_name in ("realesr_general_x4v3", "realesrgan_x4plus"):
            print(f"[TokenSignals] Batch {batch_idx + 1}: running SR model {model_name}")
            sr_images = run_sr_model_in_chunks(
                lowres_bchw=lowres,
                target_hw=(args.image_size, args.image_size),
                model_name=model_name,
                weights_dir=args.weights_dir,
                dtype=args.dtype,
                device=device,
                batch_size=sr_batch_size,
            )
            candidates.append((model_name, sr_images))

        for variant_name, candidate_images in candidates:
            print(f"[TokenSignals] Batch {batch_idx + 1}: probing {variant_name}")
            candidate_signals = probe.run(candidate_images)
            attention_rows.extend(compare_attention_rows(
                ref_signals=ref_signals,
                cand_signals=candidate_signals,
                valid_token_masks=valid_token_masks,
                sample_keys=sample_keys,
                batch_idx=batch_idx,
                dataset=dataset_name,
                variant=variant_name,
                topk_ratio=args.topk_ratio,
            ))
            ffn_rows.extend(compare_ffn_rows(
                ref_signals=ref_signals,
                cand_signals=candidate_signals,
                valid_token_masks=valid_token_masks,
                sample_keys=sample_keys,
                batch_idx=batch_idx,
                dataset=dataset_name,
                variant=variant_name,
                topk_ratio=args.topk_ratio,
            ))
            del candidate_signals

        del ref_signals
        if device.type == "cuda":
            torch.cuda.empty_cache()

    attention_summary = summarize_rows(attention_rows)
    ffn_summary = summarize_rows(ffn_rows)
    improvement_summary = build_improvement_summary(attention_rows + ffn_rows)

    write_json(out_dir / "samples.json", {"samples": processed_samples})
    write_csv(out_dir / "attention_per_image.csv", attention_rows)
    write_csv(out_dir / "attention_summary.csv", attention_summary)
    write_csv(out_dir / "ffn_per_image.csv", ffn_rows)
    write_csv(out_dir / "ffn_summary.csv", ffn_summary)
    write_csv(out_dir / "improvement_summary.csv", improvement_summary)

    print_quick_summary(attention_summary + ffn_summary)
    print(f"[TokenSignals] Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
