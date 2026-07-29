from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.experiments.dinov3_lowres_cache_lift_probe import (  # noqa: E402
    cuda_elapsed_ms,
    feature_metrics,
    load_config,
    predict_mask,
    prepare_inputs,
    run_full_backbone,
    warmup_cuda,
)
from analysis.shared.lowres_cache_lift import (  # noqa: E402
    apply_rope_to_cached_key,
    lift_token_grid,
    tensor_cache_nbytes,
)
from appcorr.models.dinov3.layers.attention import PackedQueryState  # noqa: E402
from appcorr.models.dinov3.models.vision_transformer import create_group_index  # noqa: E402
from offload.mobile.dataset import get_dataset_loader  # noqa: E402
from offload.server.model import get_model_executor  # noqa: E402


DEFAULT_CONFIG = "offload/config/ade20k_m2f_interleaved_static.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate G=4 interleaved expand-once correction. The low-resolution "
            "frontier advances 10 layers at a time; each arriving fine group is "
            "corrected only to that frontier and remains fine in later chunks."
        )
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--data-root", default="")
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--high-size", type=int, default=512)
    parser.add_argument("--low-scale", type=float, default=0.5)
    parser.add_argument("--num-groups", type=int, default=4)
    parser.add_argument(
        "--group-strategy",
        choices=("grid", "block_grid"),
        default="grid",
        help="Spatial grouping used for arriving high-resolution patches.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--skip-head", action="store_true")
    parser.add_argument(
        "--output",
        default="logs/analysis/dinov3_expand_once_interleaved_ade100.json",
    )
    return parser.parse_args()


def build_parent_token_index(
    low_hw: tuple[int, int],
    high_hw: tuple[int, int],
    num_prefix: int,
    device: torch.device,
) -> torch.Tensor:
    low_h, low_w = low_hw
    high_h, high_w = high_hw
    if high_h % low_h != 0 or high_w % low_w != 0:
        raise ValueError(f"High grid {high_hw} must be an integer multiple of low grid {low_hw}")
    scale_h = high_h // low_h
    scale_w = high_w // low_w
    rows = torch.arange(high_h, device=device, dtype=torch.long) // scale_h
    cols = torch.arange(high_w, device=device, dtype=torch.long) // scale_w
    patch_parent = (rows[:, None] * low_w + cols[None, :]).reshape(-1) + num_prefix
    return torch.cat(
        (
            torch.arange(num_prefix, device=device, dtype=torch.long),
            patch_parent,
        )
    )


def build_group_dindices(
    high_hw: tuple[int, int],
    num_groups: int,
    num_prefix: int,
    device: torch.device,
    group_strategy: str = "grid",
) -> list[torch.Tensor]:
    high_h, high_w = high_hw
    group_map = create_group_index(
        high_h * high_w,
        num_groups,
        group_strategy,
        device,
        token_hw=high_hw,
    )
    prefix = torch.arange(num_prefix, device=device, dtype=torch.long)
    return [
        torch.cat(
            (
                prefix,
                torch.nonzero(group_map == group_id, as_tuple=False).flatten() + num_prefix,
            )
        )
        for group_id in range(1, num_groups + 1)
    ]


def validate_group_partition(
    group_dindices: list[torch.Tensor],
    high_hw: tuple[int, int],
    num_prefix: int,
) -> None:
    num_patches = high_hw[0] * high_hw[1]
    patch_indices = torch.cat(
        [indices[num_prefix:].cpu() - num_prefix for indices in group_dindices]
    )
    if patch_indices.numel() != num_patches:
        raise RuntimeError("Groups do not contain exactly one entry per patch")
    if not torch.equal(
        patch_indices.sort().values,
        torch.arange(num_patches, dtype=patch_indices.dtype),
    ):
        raise RuntimeError("Groups must be disjoint and cover the high-resolution grid")
    group_sizes = [indices.numel() - num_prefix for indices in group_dindices]
    if max(group_sizes) != min(group_sizes):
        raise RuntimeError(f"Groups must have equal patch counts, got {group_sizes}")


def validate_parent_group_alignment(
    group_dindices: list[torch.Tensor],
    low_hw: tuple[int, int],
    high_hw: tuple[int, int],
    num_prefix: int,
) -> None:
    low_h, low_w = low_hw
    high_h, high_w = high_hw
    scale_h = high_h // low_h
    scale_w = high_w // low_w
    memberships = torch.full((high_h * high_w,), -1, dtype=torch.long)
    for group_idx, indices in enumerate(group_dindices):
        memberships[indices[num_prefix:].cpu() - num_prefix] = group_idx
    memberships = memberships.view(high_h, high_w)
    for low_row in range(low_h):
        for low_col in range(low_w):
            children = memberships[
                low_row * scale_h:(low_row + 1) * scale_h,
                low_col * scale_w:(low_col + 1) * scale_w,
            ]
            if children.unique().numel() != scale_h * scale_w:
                raise RuntimeError("Grid groups do not split each low-resolution parent evenly")


def build_query_state(active_indices: torch.Tensor, num_prefix: int) -> PackedQueryState:
    num_active = active_indices.numel()
    device = active_indices.device
    active_pos = torch.arange(num_active, device=device, dtype=torch.long)
    valid_mask = torch.ones((1, num_active), device=device, dtype=torch.bool)
    return PackedQueryState(
        active_batch_idx=torch.zeros(num_active, device=device, dtype=torch.long),
        active_pos_idx=active_pos,
        active_token_idx=active_indices,
        query_valid_mask=valid_mask,
        active_query_pos_padded=active_pos.unsqueeze(0),
        active_query_mask=valid_mask,
        all_valid=True,
        active_patch_mask=active_indices >= num_prefix,
        active_rope_idx=(active_indices - num_prefix).clamp_min(0),
    )


def run_low_approx_range(
    backbone,
    x: torch.Tensor,
    rope,
    cache: Dict[str, Any],
    layer_tags: list[str],
    interaction_indexes: set[int],
    start_layer: int,
    end_layer: int,
):
    captured: Dict[int, torch.Tensor] = {}
    for layer_idx in range(start_layer, end_layer):
        block = backbone.blocks[layer_idx]
        x, cache = block.approx(
            x,
            rope,
            cache,
            tag=layer_tags[layer_idx],
            appcorr_method="partial_token",
            server_pscore="cls_attn_prob",
            server_pscore_weight=1.0,
            cache_pre_rope_k=True,
            debug=False,
        )
        if layer_idx in interaction_indexes:
            captured[layer_idx] = x
    return x, cache, captured


def resolve_logical_high_kv(
    low_cache: Dict[str, Any],
    tag: str,
    parent_token_index: torch.Tensor,
    high_rope,
    num_prefix: int,
    layer_override: Dict[str, torch.Tensor] | None,
) -> torch.Tensor:
    pre_rope_k = low_cache[f"{tag}_pre_rope_k"].index_select(1, parent_token_index)
    high_k = apply_rope_to_cached_key(pre_rope_k, high_rope, num_prefix)
    high_v = low_cache[f"{tag}_kv"][:, :, 1].index_select(1, parent_token_index)
    logical_kv = torch.stack((high_k, high_v), dim=2)
    if layer_override is not None:
        logical_kv[:, layer_override["indices"]] = layer_override["kv"]
    return logical_kv


def update_layer_override(
    previous: Dict[str, torch.Tensor] | None,
    active_indices: torch.Tensor,
    active_kv: torch.Tensor,
    num_prefix: int,
) -> Dict[str, torch.Tensor]:
    if previous is None:
        return {
            "indices": active_indices.detach().clone(),
            "kv": active_kv.detach().clone(),
        }
    # Prefix state is global and is replaced on every correction/advance. Patch
    # groups are disjoint, while an advance can refresh already-stored patches.
    previous_indices = previous["indices"]
    keep = torch.ones_like(previous_indices, dtype=torch.bool)
    keep &= previous_indices >= num_prefix
    if active_indices.numel() > num_prefix:
        keep &= ~torch.isin(previous_indices, active_indices[num_prefix:])
    return {
        "indices": torch.cat((previous_indices[keep], active_indices)),
        "kv": torch.cat((previous["kv"][:, keep], active_kv), dim=1),
    }


def run_active_range(
    backbone,
    x_active: torch.Tensor,
    active_indices: torch.Tensor,
    high_rope,
    low_cache: Dict[str, Any],
    layer_tags: list[str],
    interaction_indexes: set[int],
    parent_token_index: torch.Tensor,
    num_prefix: int,
    overrides: Dict[str, Dict[str, torch.Tensor]],
    start_layer: int,
    end_layer: int,
):
    query_state = build_query_state(active_indices, num_prefix)
    captured: Dict[int, torch.Tensor] = {}
    for layer_idx in range(start_layer, end_layer):
        block = backbone.blocks[layer_idx]
        tag = layer_tags[layer_idx]
        logical_kv = resolve_logical_high_kv(
            low_cache,
            tag,
            parent_token_index,
            high_rope,
            num_prefix,
            overrides.get(tag),
        )
        attention_cache = {f"{tag}_kv": logical_kv}
        attention_out, attention_cache = block.attn.correct(
            block.norm1(x_active),
            dindice=active_indices.unsqueeze(0),
            rope=high_rope,
            cache_feature=attention_cache,
            tag=tag,
            appcorr_method="partial_token",
            fixed_query_state=query_state,
            sdpa_query_bucket_size=0,
        )
        x_attn = x_active + block.ls1(attention_out).to(dtype=x_active.dtype)
        x_active = x_attn + block.ls2(block.mlp(block.norm2(x_attn))).to(dtype=x_active.dtype)
        active_kv = attention_cache[f"{tag}_kv"][:, active_indices]
        overrides[tag] = update_layer_override(
            overrides.get(tag),
            active_indices,
            active_kv,
            num_prefix,
        )
        if layer_idx in interaction_indexes:
            captured[layer_idx] = x_active
    return x_active, captured, overrides


def scatter_active(
    base: torch.Tensor,
    active_indices: torch.Tensor,
    active_values: torch.Tensor,
) -> torch.Tensor:
    output = base.clone()
    output[0, active_indices] = active_values.to(dtype=output.dtype)
    return output


def install_low_captures(
    high_intermediates: Dict[int, torch.Tensor],
    low_captures: Dict[int, torch.Tensor],
    low_hw: tuple[int, int],
    high_hw: tuple[int, int],
    num_prefix: int,
) -> None:
    for layer_idx, value in low_captures.items():
        high_intermediates[layer_idx] = lift_token_grid(
            value,
            low_hw,
            high_hw,
            num_prefix,
            mode="nearest",
        )


def install_active_captures(
    high_intermediates: Dict[int, torch.Tensor],
    active_captures: Dict[int, torch.Tensor],
    active_indices: torch.Tensor,
) -> None:
    for layer_idx, value in active_captures.items():
        high_intermediates[layer_idx] = scatter_active(
            high_intermediates[layer_idx],
            active_indices,
            value,
        )


def combine_arrived_states(
    prefix_state: torch.Tensor,
    patch_states: Dict[int, torch.Tensor],
    group_dindices: list[torch.Tensor],
    arrived_groups: int,
    num_prefix: int,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
    indices = [torch.arange(num_prefix, device=prefix_state.device, dtype=torch.long)]
    values = [prefix_state]
    spans = []
    cursor = num_prefix
    for group_id in range(1, arrived_groups + 1):
        group_patch_indices = group_dindices[group_id - 1][num_prefix:]
        group_values = patch_states[group_id]
        indices.append(group_patch_indices)
        values.append(group_values)
        spans.append((cursor, cursor + group_values.shape[0]))
        cursor += group_values.shape[0]
    return torch.cat(values, dim=0), torch.cat(indices, dim=0), spans


def split_arrived_states(
    combined: torch.Tensor,
    spans: list[tuple[int, int]],
    num_prefix: int,
) -> tuple[torch.Tensor, Dict[int, torch.Tensor]]:
    prefix_state = combined[:num_prefix]
    patch_states = {
        group_id: combined[start:end]
        for group_id, (start, end) in enumerate(spans, start=1)
    }
    return prefix_state, patch_states


def override_nbytes(overrides: Dict[str, Dict[str, torch.Tensor]]) -> int:
    return sum(
        value["indices"].numel() * value["indices"].element_size()
        + value["kv"].numel() * value["kv"].element_size()
        for value in overrides.values()
    )


def add_areas(accumulator: torch.Tensor, loader, prediction, label) -> None:
    ground_truth = label.get("orig_mask")
    if ground_truth is None:
        ground_truth = label["mask"]
    accumulator += loader._intersect_and_union(prediction, ground_truth)


def metrics_from_areas(loader, areas: torch.Tensor) -> dict:
    return loader._metrics_from_areas(areas[0], areas[1], areas[2], areas[3])


def summarize(rows: list[dict]) -> dict:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)
    summary = {}
    for variant, values in sorted(grouped.items()):
        steady_values = values[1:] if len(values) > 1 else values
        total_work = [row["total_vit_work_ms"] for row in values]
        steady_total_work = [row["total_vit_work_ms"] for row in steady_values]
        item = {
            "num_samples": len(values),
            "relative_l2_mean": float(np.mean([row["relative_l2"] for row in values])),
            "relative_l2_std": float(np.std([row["relative_l2"] for row in values])),
            "cosine_similarity_mean": float(
                np.mean([row["cosine_similarity"] for row in values])
            ),
            "cosine_similarity_std": float(
                np.std([row["cosine_similarity"] for row in values])
            ),
            "total_vit_work_ms_mean": float(
                np.mean(total_work)
            ),
            "total_vit_work_ms_median": float(np.median(total_work)),
            "total_vit_work_ms_p95": float(np.percentile(total_work, 95)),
            "steady_state_total_vit_work_ms_mean": float(np.mean(steady_total_work)),
            "full_ms_mean": float(np.mean([row["full_ms"] for row in values])),
            "work_ratio_vs_full_mean": float(
                np.mean(
                    [
                        row["total_vit_work_ms"] / max(row["full_ms"], 1e-12)
                        for row in values
                    ]
                )
            ),
            "steady_state_work_ratio_vs_full_mean": float(
                np.mean(
                    [
                        row["total_vit_work_ms"] / max(row["full_ms"], 1e-12)
                        for row in steady_values
                    ]
                )
            ),
            "state_mib_mean": float(np.mean([row["state_mib"] for row in values])),
        }
        for key in (
            "low_approx_ms",
            "new_group_correction_ms",
            "arrived_group_advance_ms",
        ):
            if key in values[0]:
                item[f"{key}_mean"] = float(np.mean([row[key] for row in values]))
                item[f"steady_state_{key}_mean"] = float(
                    np.mean([row[key] for row in steady_values])
                )
        summary[variant] = item
    return summary


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This probe requires CUDA")
    if args.num_groups != 4:
        raise ValueError("The interleaved reference currently requires G=4")
    low_size = int(round(args.high_size * args.low_scale))
    if args.high_size % 16 != 0 or low_size % 16 != 0:
        raise ValueError("High and low sizes must be multiples of 16")

    config, raw_config = load_config(args.config)
    device = torch.device(args.device)
    executor = get_model_executor(config.model_name, device)
    executor.load_model(config.model_name, config)
    adapter = executor.model.segmentation_model[0]
    backbone = adapter.backbone
    interaction_indexes = set(adapter.interaction_indexes)
    ordered_interactions = sorted(interaction_indexes)
    total_layers = len(backbone.blocks)
    chunk_size = total_layers // args.num_groups
    layer_tags = [f"expand_once_layer{idx}" for idx in range(total_layers)]
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
    processed_samples = 0

    for sample_idx, (image_batch, label_batch) in enumerate(
        tqdm(dataloader, total=min(len(dataloader), args.max_samples))
    ):
        if sample_idx >= args.max_samples:
            break
        processed_samples += 1
        image_tensor = image_batch[0] if isinstance(image_batch, list) else image_batch[0]
        label = label_batch[0]
        high_input, low_input, _ = prepare_inputs(
            image_tensor,
            executor,
            args.high_size,
            low_size,
        )

        with torch.inference_mode(), torch.autocast("cuda", executor.autocast_dtype):
            high_source = executor._prepare_single_source(high_input, adapter, backbone)
            high_tokens = high_source["x_backbone"]
            high_rope = high_source["rope_sincos"]
            high_hw = high_source["token_shape"]
            low_tokens, low_hw = backbone.prepare_tokens_with_masks(low_input)
            low_rope = backbone.rope_embed(H=low_hw[0], W=low_hw[1])
            parent_token_index = build_parent_token_index(
                low_hw,
                high_hw,
                num_prefix,
                device,
            )
            group_dindices = build_group_dindices(
                high_hw,
                args.num_groups,
                num_prefix,
                device,
                args.group_strategy,
            )
            if sample_idx == 0:
                validate_group_partition(
                    group_dindices,
                    high_hw,
                    num_prefix,
                )
                if args.group_strategy == "grid":
                    validate_parent_group_alignment(
                        group_dindices,
                        low_hw,
                        high_hw,
                        num_prefix,
                    )

            full_call = lambda: run_full_backbone(
                backbone,
                high_tokens.clone(),
                high_rope,
                interaction_indexes,
            )
            warmup_cuda(full_call)
            (full_output, full_intermediates), full_ms = cuda_elapsed_ms(full_call)

            low_x = low_tokens.clone()
            low_cache: Dict[str, Any] = {}
            high_intermediates: Dict[int, torch.Tensor] = {}
            low_approx_ms = 0.0
            new_group_correction_ms = 0.0
            arrived_group_advance_ms = 0.0
            overrides: Dict[str, Dict[str, torch.Tensor]] = {}
            patch_states: Dict[int, torch.Tensor] = {}
            prefix_state: torch.Tensor | None = None

            # Group 0/base: approximate only the first 10-layer chunk.
            (low_x, low_cache, low_captures), chunk_ms = cuda_elapsed_ms(
                lambda: run_low_approx_range(
                    backbone,
                    low_x,
                    low_rope,
                    low_cache,
                    layer_tags,
                    interaction_indexes,
                    0,
                    chunk_size,
                )
            )
            low_approx_ms += chunk_ms
            install_low_captures(
                high_intermediates,
                low_captures,
                low_hw,
                high_hw,
                num_prefix,
            )

            # C_g(0, frontier), then A(frontier, frontier+10). Fine groups that
            # have already arrived advance through the new chunk together.
            for group_id in range(1, args.num_groups + 1):
                frontier = group_id * chunk_size
                group_indices = group_dindices[group_id - 1]
                group_input = high_tokens[0, group_indices].contiguous()
                (
                    group_output,
                    group_captures,
                    overrides,
                ), correction_ms = cuda_elapsed_ms(
                    lambda x=group_input, indices=group_indices, end=frontier: run_active_range(
                        backbone,
                        x,
                        indices,
                        high_rope,
                        low_cache,
                        layer_tags,
                        interaction_indexes,
                        parent_token_index,
                        num_prefix,
                        overrides,
                        0,
                        end,
                    )
                )
                new_group_correction_ms += correction_ms
                prefix_state = group_output[:num_prefix]
                patch_states[group_id] = group_output[num_prefix:]
                install_active_captures(
                    high_intermediates,
                    group_captures,
                    group_indices,
                )

                if group_id == args.num_groups:
                    continue

                next_frontier = frontier + chunk_size
                (low_x, low_cache, low_captures), chunk_ms = cuda_elapsed_ms(
                    lambda start=frontier, end=next_frontier: run_low_approx_range(
                        backbone,
                        low_x,
                        low_rope,
                        low_cache,
                        layer_tags,
                        interaction_indexes,
                        start,
                        end,
                    )
                )
                low_approx_ms += chunk_ms
                install_low_captures(
                    high_intermediates,
                    low_captures,
                    low_hw,
                    high_hw,
                    num_prefix,
                )

                combined_input, combined_indices, spans = combine_arrived_states(
                    prefix_state,
                    patch_states,
                    group_dindices,
                    group_id,
                    num_prefix,
                )
                (
                    combined_output,
                    advance_captures,
                    overrides,
                ), advance_ms = cuda_elapsed_ms(
                    lambda x=combined_input, indices=combined_indices, start=frontier, end=next_frontier: run_active_range(
                        backbone,
                        x,
                        indices,
                        high_rope,
                        low_cache,
                        layer_tags,
                        interaction_indexes,
                        parent_token_index,
                        num_prefix,
                        overrides,
                        start,
                        end,
                    )
                )
                arrived_group_advance_ms += advance_ms
                prefix_state, advanced_patch_states = split_arrived_states(
                    combined_output,
                    spans,
                    num_prefix,
                )
                patch_states.update(advanced_patch_states)
                install_active_captures(
                    high_intermediates,
                    advance_captures,
                    combined_indices,
                )

            expand_output = lift_token_grid(
                low_x,
                low_hw,
                high_hw,
                num_prefix,
                mode="nearest",
            )
            expand_output[0, :num_prefix] = prefix_state
            for group_id, indices in enumerate(group_dindices, start=1):
                expand_output[0, indices[num_prefix:]] = patch_states[group_id]

            expand_intermediates = [
                high_intermediates[layer_idx]
                for layer_idx in ordered_interactions
            ]
            total_work_ms = (
                low_approx_ms
                + new_group_correction_ms
                + arrived_group_advance_ms
            )
            state_mib = (
                tensor_cache_nbytes(low_cache) + override_nbytes(overrides)
            ) / (1024**2)
            rows.extend(
                [
                    {
                        "sample_idx": sample_idx,
                        "variant": "full",
                        "relative_l2": 0.0,
                        "cosine_similarity": 1.0,
                        "low_approx_ms": 0.0,
                        "new_group_correction_ms": 0.0,
                        "arrived_group_advance_ms": 0.0,
                        "total_vit_work_ms": full_ms,
                        "full_ms": full_ms,
                        "state_mib": 0.0,
                    },
                    {
                        "sample_idx": sample_idx,
                        "variant": "expand_once_interleaved",
                        **feature_metrics(full_output, expand_output, num_prefix),
                        "low_approx_ms": low_approx_ms,
                        "new_group_correction_ms": new_group_correction_ms,
                        "arrived_group_advance_ms": arrived_group_advance_ms,
                        "total_vit_work_ms": total_work_ms,
                        "full_ms": full_ms,
                        "state_mib": state_mib,
                    },
                ]
            )
            if not args.skip_head:
                output_hw = tuple(np.asarray(label["orig_mask"]).shape)
                full_prediction = predict_mask(
                    executor,
                    high_source,
                    full_intermediates,
                    output_hw,
                )
                expand_prediction = predict_mask(
                    executor,
                    high_source,
                    expand_intermediates,
                    output_hw,
                )
                add_areas(area_totals["full"], loader, full_prediction, label)
                add_areas(
                    area_totals["expand_once_interleaved"],
                    loader,
                    expand_prediction,
                    label,
                )

        del high_source, low_cache, overrides
        torch.cuda.empty_cache()

    summary = summarize(rows)
    for variant, areas in area_totals.items():
        summary.setdefault(variant, {})["segmentation"] = metrics_from_areas(loader, areas)

    payload = {
        "config": args.config,
        "device": args.device,
        "high_size": args.high_size,
        "low_size": low_size,
        "num_groups": args.num_groups,
        "group_strategy": args.group_strategy,
        "num_samples": processed_samples,
        "schedule": [
            "A_low(0,10)",
            "C_new_g1(0,10)",
            "A_low(10,20)+A_fine_g1(10,20)",
            "C_new_g2(0,20)",
            "A_low(20,30)+A_fine_g1,g2(20,30)",
            "C_new_g3(0,30)",
            "A_low(30,40)+A_fine_g1,g2,g3(30,40)",
            "C_new_g4(0,40)",
        ],
        "semantics": (
            "Each patch becomes fine once. It is corrected from layer 0 to the "
            "current low-resolution frontier, then advances as a fine token through "
            "later approximate chunks. Earlier-layer queries are never revisited "
            "when later groups arrive."
        ),
        "timing_note": (
            "CUDA-event total GPU work; preprocessing, SPM, M2F head, transmission, "
            "and overlap are excluded."
        ),
        "summary": summary,
        "rows": rows,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
