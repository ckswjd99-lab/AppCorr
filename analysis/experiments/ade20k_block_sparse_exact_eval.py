#!/usr/bin/env python3
"""ADE20K exact-delta block-sparse DINOv3 correction experiment.

The default backend emulates sparse support with dense PyTorch masks.  The
optional Triton backend executes selected attention products and FFN projection
blocks directly. FLOP/cache estimates describe the structured sparse algorithm;
wall time also includes the remaining dense operations and evaluation pipeline.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.shared.cuda_environment import configure_triton_cuda_environment

configure_triton_cuda_environment()

from appcorr.models.dinov3.layers.jacobian_support import (
    attention_block_index_from_mask,
    ffn_block_index_from_mask,
    select_attention_block_support,
    select_ffn_block_support,
)
from appcorr.models.dinov3.layers.triton_kernels import (
    block_product_delta_triton,
    block_sparse_ffn_delta_triton,
)
from offload.common.protocol import ExperimentConfig
from offload.server.model.dinov3_segmentor_m2f import DINOv3SegmentorM2FExecutor


INTERMEDIATE_LAYERS = (9, 19, 29, 39)
FULL_PREFIX_LAYERS = 2
FULL_SUFFIX_LAYERS = 2
SPECIAL_TOKENS = 5
NUM_CLASSES = 150
IGNORE_INDEX = 255
BASE_CANVAS_MODE = "native_pyramid_then_scale"


@dataclass(frozen=True)
class Policy:
    name: str
    token_rate: float
    attention_rate: float
    ffn_rate: float


@dataclass
class SupportAccumulator:
    attention_requested: float = 0.0
    attention_realized: float = 0.0
    attention_mass: float = 0.0
    attention_samples: int = 0
    ffn_requested: float = 0.0
    ffn_realized: float = 0.0
    ffn_samples: int = 0
    token_realized: float = 0.0
    token_samples: int = 0

    def add_attention(self, requested: float, realized: float, mass: float) -> None:
        self.attention_requested += requested
        self.attention_realized += realized
        self.attention_mass += mass
        self.attention_samples += 1

    def add_ffn(self, requested: float, realized: float) -> None:
        self.ffn_requested += requested
        self.ffn_realized += realized
        self.ffn_samples += 1

    def add_token(self, realized: float) -> None:
        self.token_realized += realized
        self.token_samples += 1

    def summary(self) -> dict[str, float]:
        return {
            "token_realized": self.token_realized / max(self.token_samples, 1),
            "attention_requested": self.attention_requested / max(self.attention_samples, 1),
            "attention_realized": self.attention_realized / max(self.attention_samples, 1),
            "attention_probability_mass": self.attention_mass / max(self.attention_samples, 1),
            "ffn_requested": self.ffn_requested / max(self.ffn_samples, 1),
            "ffn_realized": self.ffn_realized / max(self.ffn_samples, 1),
        }


class AreaMetrics:
    def __init__(self) -> None:
        self.intersection = torch.zeros(NUM_CLASSES, dtype=torch.float64)
        self.union = torch.zeros(NUM_CLASSES, dtype=torch.float64)
        self.predicted = torch.zeros(NUM_CLASSES, dtype=torch.float64)
        self.label = torch.zeros(NUM_CLASSES, dtype=torch.float64)

    def add_areas(self, areas: Iterable[Iterable[float]]) -> None:
        tensors = [torch.as_tensor(value, dtype=torch.float64) for value in areas]
        self.intersection += tensors[0]
        self.union += tensors[1]
        self.predicted += tensors[2]
        self.label += tensors[3]

    def update(self, prediction: torch.Tensor, annotation: np.ndarray) -> list[list[float]]:
        pred = prediction.cpu().long()
        label = torch.from_numpy(np.array(annotation, dtype=np.uint8, copy=True)).long()
        if pred.shape != label.shape:
            pred = F.interpolate(
                pred[None, None].float(),
                size=tuple(label.shape),
                mode="nearest",
            )[0, 0].long()
        label = label.clone()
        label[label == IGNORE_INDEX] += 1
        label -= 1
        label[label == -1] = IGNORE_INDEX
        valid = label != IGNORE_INDEX
        pred = pred[valid]
        label = label[valid]
        intersect = pred[pred == label]
        area_intersection = torch.bincount(intersect, minlength=NUM_CLASSES)[:NUM_CLASSES].double()
        area_predicted = torch.bincount(pred, minlength=NUM_CLASSES)[:NUM_CLASSES].double()
        area_label = torch.bincount(label, minlength=NUM_CLASSES)[:NUM_CLASSES].double()
        area_union = area_predicted + area_label - area_intersection
        areas = [area_intersection, area_union, area_predicted, area_label]
        self.add_areas(areas)
        return [value.tolist() for value in areas]

    def summary(self) -> dict[str, float]:
        eps = torch.finfo(torch.float64).eps
        iou = self.intersection / self.union.clamp_min(eps)
        iou[self.union == 0] = torch.nan
        aacc = self.intersection.sum() / self.label.sum().clamp_min(eps)
        return {
            "mIoU": float(torch.nanmean(iou).item() * 100.0),
            "aAcc": float(aacc.item() * 100.0),
        }


def parse_rates(value: str) -> list[float]:
    rates = [float(item) for item in value.split(",") if item.strip()]
    if not rates or any(rate <= 0 or rate > 1 for rate in rates):
        raise argparse.ArgumentTypeError("rates must be comma-separated values in (0, 1]")
    return rates


def scaled_native_pyramid_level(
    original: np.ndarray,
    level: int,
    target_hw: tuple[int, int],
) -> np.ndarray:
    """Build a Gaussian level in native coordinates, then scale to the model canvas."""
    if original.ndim != 3:
        raise ValueError(f"expected HWC image, got shape {original.shape}")
    if level < 0:
        raise ValueError(f"pyramid level must be non-negative, got {level}")
    target_h, target_w = target_hw
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"target shape must be positive, got {target_hw}")

    current = np.ascontiguousarray(original.astype(np.uint8, copy=False))
    for _ in range(level):
        current = cv2.pyrDown(current)
        current = np.ascontiguousarray(current.astype(np.uint8, copy=False))

    if current.shape[:2] != target_hw:
        current = np.asarray(
            Image.fromarray(current).resize(
                (target_w, target_h),
                Image.Resampling.BILINEAR,
            ),
            dtype=np.uint8,
        )
    return np.ascontiguousarray(current.astype(np.uint8, copy=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-images", type=int, default=200)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--short-side", type=int, default=896)
    parser.add_argument("--crop-size", type=int, default=896)
    parser.add_argument("--stride", type=int, default=596)
    parser.add_argument("--base-level", type=int, default=2)
    parser.add_argument(
        "--correction-backend",
        choices=("dense_mask", "triton"),
        default="dense_mask",
        help="Execute correction products with dense masks or block-sparse Triton kernels.",
    )
    parser.add_argument(
        "--spm-input",
        choices=("full", "base"),
        default="full",
        help=(
            "RGB source for SPM features used by approx/correction outputs. "
            "'base' keeps the full-resolution token grid but feeds the L2 canvas to SPM."
        ),
    )
    parser.add_argument("--token-rate", type=float, default=0.5)
    parser.add_argument("--attention-rates", type=parse_rates, default=parse_rates("0.25,0.5,0.75"))
    parser.add_argument("--ffn-rates", type=parse_rates, default=parse_rates("0.25,0.5,0.75"))
    parser.add_argument("--query-block", type=int, default=8)
    parser.add_argument("--key-block", type=int, default=16)
    parser.add_argument("--head-group", type=int, default=4)
    parser.add_argument("--ffn-token-block", type=int, default=8)
    parser.add_argument("--ffn-channel-block", type=int, default=128)
    parser.add_argument("--query-chunk", type=int, default=64)
    parser.add_argument("--policies", default="sweep,token_only,full,approx")
    parser.add_argument("--endpoint-only", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("logs/ade20k_block_sparse_exact"))
    parser.add_argument(
        "--resume-from",
        type=Path,
        action="append",
        default=[],
        help="Additional result JSONL files whose completed indices and areas are reused.",
    )
    parser.add_argument(
        "--backbone-weights",
        default="~/cjpark/weights/dinov3/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth",
    )
    parser.add_argument(
        "--head-weights",
        default="~/cjpark/weights/dinov3/dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth",
    )
    return parser.parse_args()


def make_policies(args: argparse.Namespace) -> list[Policy]:
    requested = {item.strip() for item in args.policies.split(",") if item.strip()}
    policies: list[Policy] = []
    if "sweep" in requested:
        for attention_rate in args.attention_rates:
            for ffn_rate in args.ffn_rates:
                policies.append(
                    Policy(
                        f"a{int(attention_rate * 100):02d}_f{int(ffn_rate * 100):02d}",
                        args.token_rate,
                        attention_rate,
                        ffn_rate,
                    )
                )
    if "token_only" in requested:
        policies.append(Policy("token_only", args.token_rate, 1.0, 1.0))
    if args.endpoint_only:
        return [Policy("endpoint_all_support", 1.0, 1.0, 1.0)]
    return policies


def load_executor(args: argparse.Namespace) -> tuple[DINOv3SegmentorM2FExecutor, ExperimentConfig]:
    config = ExperimentConfig(
        model_name="dinov3_segmentor_m2f",
        dataset_name="ade20k",
        batch_size=1,
        image_shape=(args.crop_size, args.crop_size, 3),
        patch_size=(16, 16),
        input_profile_name="dinov3_ade20k_m2f_official",
        input_profile_kwargs={
            "server_eval_mode": "single",
            "server_rescale_to": "original",
            "m2f_slide_merge_stage": "pre_head",
            "mobile_resize_short_side": args.short_side,
            "server_crop_size": args.crop_size,
            "server_stride": args.stride,
            "backbone_weights_path": str(Path(args.backbone_weights).expanduser()),
            "segmentor_head_weights_path": str(Path(args.head_weights).expanduser()),
            "autocast_dtype": "bfloat16",
        },
    )
    executor = DINOv3SegmentorM2FExecutor(torch.device(args.device))
    executor.load_model(config.model_name, config)
    return executor, config


def sliding_crops(height: int, width: int, crop_size: int, stride: int) -> list[tuple[int, int, int, int]]:
    crop_h = min(crop_size, height)
    crop_w = min(crop_size, width)
    h_grids = max(height - crop_h + stride - 1, 0) // stride + 1
    w_grids = max(width - crop_w + stride - 1, 0) // stride + 1
    result = []
    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * stride
            x1 = w_idx * stride
            y2 = min(y1 + crop_h, height)
            x2 = min(x1 + crop_w, width)
            y1 = max(y2 - crop_h, 0)
            x1 = max(x2 - crop_w, 0)
            result.append((y1, y2, x1, x2))
    return result


def qkv(block: Any, state: torch.Tensor, rope: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    projected = block.attn.qkv(block.norm1(state))
    batch, tokens, _ = projected.shape
    hidden = block.attn.qkv.in_features
    projected = projected.reshape(
        batch,
        tokens,
        3,
        block.attn.num_heads,
        hidden // block.attn.num_heads,
    )
    query, key, value = (
        item.transpose(1, 2)
        for item in torch.unbind(projected, dim=2)
    )
    if rope is not None:
        query, key = block.attn.apply_rope(query, key, rope)
    return query, key, value


def gather_tokens(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return value.gather(1, indices.unsqueeze(-1).expand(-1, -1, value.shape[-1]))


def gather_heads(value: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return value.gather(
        2,
        indices[:, None, :, None].expand(-1, value.shape[1], -1, value.shape[-1]),
    )


def scatter_tokens(base: torch.Tensor, indices: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    result = base.clone()
    return result.scatter(1, indices.unsqueeze(-1).expand_as(value), value.to(base.dtype))


def run_base(
    backbone: Any,
    state: torch.Tensor,
    rope: Any,
) -> tuple[list[torch.Tensor], list[torch.Tensor], torch.Tensor, dict[int, torch.Tensor]]:
    states = [state]
    attention_states = []
    score_sum = torch.zeros(state.shape[:2], device=state.device, dtype=torch.float32)
    raw_intermediates: dict[int, torch.Tensor] = {}
    cache: dict[str, Any] = {}
    for layer_index, block in enumerate(backbone.blocks):
        tag = f"base_layer{layer_index}"
        attention_output, cache = block.attn.approx(
            block.norm1(state),
            rope=rope,
            cache_feature=cache,
            tag=tag,
            appcorr_method="partial_token",
            server_pscore="patch_attn_prob_layermean",
        )
        attention_state = state + block.ls1(attention_output)
        state = attention_state + block.ls2(block.mlp(block.norm2(attention_state)))
        attention_states.append(attention_state)
        score_sum += cache.pop(f"{tag}_server_pscore").float()
        cache.pop(f"{tag}_kv", None)
        states.append(state)
        if layer_index in INTERMEDIATE_LAYERS:
            raw_intermediates[layer_index] = state
    return states, attention_states, score_sum / len(backbone.blocks), raw_intermediates


def run_base_block(
    block: Any,
    state: torch.Tensor,
    rope: Any,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference one-layer base values without collecting a pscore."""

    attention_state = state + block.ls1(block.attn(block.norm1(state), rope=rope))
    next_state = attention_state + block.ls2(block.mlp(block.norm2(attention_state)))
    return attention_state, next_state


def run_full(
    backbone: Any,
    state: torch.Tensor,
    rope: Any,
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    intermediates: dict[int, torch.Tensor] = {}
    for layer_index, block in enumerate(backbone.blocks):
        state = block(state, rope)
        if layer_index in INTERMEDIATE_LAYERS:
            intermediates[layer_index] = state
    return state, intermediates


def token_support(
    full_crop: np.ndarray,
    base_crop: np.ndarray,
    attention_score: torch.Tensor,
    token_rate: float,
    patch_size: int = 16,
) -> torch.Tensor:
    height, width = full_crop.shape[:2]
    grid_h, grid_w = height // patch_size, width // patch_size
    residual = torch.from_numpy(
        full_crop[: grid_h * patch_size, : grid_w * patch_size].astype(np.float32)
        - base_crop[: grid_h * patch_size, : grid_w * patch_size].astype(np.float32)
    )
    energy = residual.square().mean(dim=-1)
    energy = energy.reshape(grid_h, patch_size, grid_w, patch_size).mean(dim=(1, 3)).flatten()
    received = attention_score[0, SPECIAL_TOKENS:].detach().cpu()
    if received.numel() != energy.numel():
        raise RuntimeError(f"Patch score mismatch: attention={received.numel()}, residual={energy.numel()}")
    energy = energy / energy.max().clamp_min(torch.finfo(torch.float32).eps)
    received = received / received.max().clamp_min(torch.finfo(torch.float32).eps)
    combined = energy * received
    keep = max(1, min(combined.numel(), math.ceil(combined.numel() * token_rate)))
    selected_patch = torch.topk(combined, k=keep).indices.to(attention_score.device) + SPECIAL_TOKENS
    prefix = torch.arange(SPECIAL_TOKENS, device=attention_score.device)
    return torch.cat((prefix, selected_patch.sort().values)).unsqueeze(0)


def corrected_attention(
    block: Any,
    base_state: torch.Tensor,
    base_attention_state: torch.Tensor,
    current_state: torch.Tensor,
    active_indices: torch.Tensor,
    rope: Any,
    keep_rate: float,
    args: argparse.Namespace,
    support: SupportAccumulator,
) -> torch.Tensor:
    q0, k0, v0 = qkv(block, base_state, rope)
    q1, k1, v1 = qkv(block, current_state, rope)
    q0 = gather_heads(q0, active_indices)
    q1 = gather_heads(q1, active_indices)
    scale = float(block.attn.scale)
    active_count = q0.shape[2]
    raw_chunks = []
    forced_keys = torch.zeros(
        (base_state.shape[0], base_state.shape[1]),
        dtype=torch.bool,
        device=base_state.device,
    )
    forced_keys[:, :SPECIAL_TOKENS] = True
    chunk_size = max(args.query_block, (args.query_chunk // args.query_block) * args.query_block)
    for start in range(0, active_count, chunk_size):
        end = min(start + chunk_size, active_count)
        logits0 = torch.matmul(q0[:, :, start:end].float(), k0.float().transpose(-2, -1)) * scale
        logits1 = torch.matmul(q1[:, :, start:end].float(), k1.float().transpose(-2, -1)) * scale
        probability0 = torch.softmax(logits0, dim=-1)
        mask, stats = select_attention_block_support(
            probability0,
            keep_ratio=keep_rate,
            key_block_size=args.key_block,
            query_block_size=args.query_block,
            head_group_size=args.head_group,
            residual_key_mask=forced_keys,
        )
        hybrid_probability = torch.softmax(torch.where(mask, logits1, logits0), dim=-1)
        if getattr(args, "correction_backend", "dense_mask") == "triton":
            block_index = attention_block_index_from_mask(
                mask,
                key_block_size=args.key_block,
                query_block_size=args.query_block,
                head_group_size=args.head_group,
            )
            delta_raw = block_product_delta_triton(
                probability0.to(v0.dtype),
                hybrid_probability.to(v1.dtype),
                v0,
                v1,
                block_index,
                head_group_size=args.head_group,
                query_block_size=args.query_block,
                key_block_size=args.key_block,
            )
        else:
            selected0 = probability0.masked_fill(~mask, 0)
            selected1 = hybrid_probability.masked_fill(~mask, 0)
            delta_raw = (
                torch.matmul(selected1.to(v1.dtype), v1)
                - torch.matmul(selected0.to(v0.dtype), v0)
            )
        raw_chunks.append(delta_raw)
        support.add_attention(keep_rate, stats.kept_fraction, stats.probability_mass)
    delta_raw = torch.cat(raw_chunks, dim=2).transpose(1, 2).flatten(2)
    active_state = gather_tokens(current_state, active_indices)
    base_active = gather_tokens(base_state, active_indices)
    base_attn = gather_tokens(base_attention_state, active_indices)
    delta_projected = F.linear(delta_raw, block.attn.proj.weight, bias=None)
    corrected_attn = (
        base_attn
        + (active_state - base_active)
        + block.ls1(delta_projected)
    )
    return corrected_attn


def corrected_ffn(
    block: Any,
    base_attention_state: torch.Tensor,
    corrected_attn: torch.Tensor,
    base_next: torch.Tensor,
    active_indices: torch.Tensor,
    keep_rate: float,
    args: argparse.Namespace,
    support: SupportAccumulator,
    weight_score: torch.Tensor,
) -> torch.Tensor:
    base_attn = gather_tokens(base_attention_state, active_indices)
    x0 = block.norm2(base_attn)
    x1 = block.norm2(corrected_attn)
    gate1_pre = block.mlp.w1(x1)
    gate1 = F.silu(gate1_pre)
    channel_score = gate1.float().abs() * weight_score[None, None, :]
    mask = select_ffn_block_support(
        channel_score,
        keep_ratio=keep_rate,
        channel_block_size=args.ffn_channel_block,
        token_block_size=args.ffn_token_block,
    )
    # This is a difference of two down projections, so the shared bias cancels.
    if getattr(args, "correction_backend", "dense_mask") == "triton":
        block_index = ffn_block_index_from_mask(
            mask,
            channel_block_size=args.ffn_channel_block,
            token_block_size=args.ffn_token_block,
        )
        kernel_tensors = getattr(block.mlp, "_appcorr_sparse_ffn_tensors", None)
        kernel_dtype = gate1.dtype
        if (
            kernel_tensors is None
            or kernel_tensors[0].device != x0.device
            or kernel_tensors[0].dtype != kernel_dtype
        ):
            # The official M2F model retains FP32 parameters and relies on
            # autocast. Triton reads parameters directly, so cache the
            # equivalent model-static autocast copies once per layer.
            def kernel_copy(tensor: torch.Tensor | None) -> torch.Tensor | None:
                if tensor is None:
                    return None
                return (
                    tensor.detach()
                    .to(device=x0.device, dtype=kernel_dtype)
                    .contiguous()
                )

            kernel_tensors = (
                kernel_copy(block.mlp.w1.weight),
                kernel_copy(block.mlp.w2.weight),
                kernel_copy(block.mlp.w3.weight),
                kernel_copy(block.mlp.w1.bias),
                kernel_copy(block.mlp.w2.bias),
            )
            block.mlp._appcorr_sparse_ffn_tensors = kernel_tensors
        gate_weight, up_weight, down_weight, gate_bias, up_bias = kernel_tensors
        delta = block_sparse_ffn_delta_triton(
            x0.to(dtype=kernel_dtype),
            x1.to(dtype=kernel_dtype),
            gate1,
            gate_weight,
            up_weight,
            down_weight,
            block_index,
            gate_bias=gate_bias,
            up_bias=up_bias,
            token_block_size=args.ffn_token_block,
            channel_block_size=args.ffn_channel_block,
        )
    else:
        gate0 = F.silu(block.mlp.w1(x0))
        up0 = block.mlp.w2(x0)
        up1 = block.mlp.w2(x1)
        hidden_delta = gate1 * up1 - gate0 * up0
        delta = F.linear(
            hidden_delta.masked_fill(~mask, 0),
            block.mlp.w3.weight,
            bias=None,
        )
    base_next_active = gather_tokens(base_next, active_indices)
    base_ffn_residual = base_next_active - base_attn
    corrected_next = corrected_attn + base_ffn_residual + block.ls2(delta)
    support.add_ffn(keep_rate, float(mask.float().mean().item()))
    return corrected_next


def run_policy(
    backbone: Any,
    full_initial: torch.Tensor,
    base_states: list[torch.Tensor],
    base_attention_states: list[torch.Tensor],
    rope: Any,
    active_indices: torch.Tensor,
    policy: Policy,
    args: argparse.Namespace,
    support: SupportAccumulator,
    ffn_weight_scores: list[torch.Tensor],
) -> dict[int, torch.Tensor]:
    state = full_initial
    outputs: dict[int, torch.Tensor] = {}
    depth = len(backbone.blocks)
    middle_end = depth - FULL_SUFFIX_LAYERS
    for layer_index, block in enumerate(backbone.blocks):
        if layer_index < FULL_PREFIX_LAYERS or layer_index >= middle_end:
            state = block(state, rope)
        else:
            corrected_attn = corrected_attention(
                block,
                base_states[layer_index],
                base_attention_states[layer_index],
                state,
                active_indices,
                rope,
                policy.attention_rate,
                args,
                support,
            )
            corrected_next = corrected_ffn(
                block,
                base_attention_states[layer_index],
                corrected_attn,
                base_states[layer_index + 1],
                active_indices,
                policy.ffn_rate,
                args,
                support,
                ffn_weight_scores[layer_index],
            )
            state = scatter_tokens(base_states[layer_index + 1], active_indices, corrected_next)
        if layer_index in INTERMEDIATE_LAYERS:
            outputs[layer_index] = state
    return outputs


def source_context(source: dict[str, Any], raw: dict[int, torch.Tensor]) -> dict[str, Any]:
    return {
        "m2f_intermediate_raw": [[raw[index] for index in INTERMEDIATE_LAYERS]],
        "m2f_spm_c_cat": [source["spm_c_cat"]],
        "m2f_spm_c1_raw": [source["spm_c1_raw"]],
        "m2f_spm_c2_len": [source["spm_c2_len"]],
        "m2f_spm_c3_len": [source["spm_c3_len"]],
        "m2f_source_shapes": [source["source_shape"]],
        "m2f_deform_in1": [source["deform_in1"]],
        "m2f_deform_in2": [source["deform_in2"]],
    }


def replace_spm_features(
    source: dict[str, Any],
    spm_source: dict[str, Any],
) -> dict[str, Any]:
    """Use SPM tensors from another same-shaped source without changing ViT state."""
    result = dict(source)
    for key in ("spm_c_cat", "spm_c1_raw", "spm_c2_len", "spm_c3_len"):
        result[key] = spm_source[key]
    return result


def add_crop_features(
    executor: DINOv3SegmentorM2FExecutor,
    sums: dict[str, torch.Tensor],
    counts: dict[str, torch.Tensor],
    source: dict[str, Any],
    raw: dict[int, torch.Tensor],
    crop: tuple[int, int, int, int],
    image_hw: tuple[int, int],
) -> None:
    features = executor._run_adapter_postprocess(0, source_context(source, raw))
    executor._accumulate_m2f_adapter_feature_crop(sums, counts, features, crop, image_hw)


def finish_prediction(
    executor: DINOv3SegmentorM2FExecutor,
    sums: dict[str, torch.Tensor],
    counts: dict[str, torch.Tensor],
    image_hw: tuple[int, int],
    original_hw: tuple[int, int],
) -> torch.Tensor:
    features = {key: sums[key] / counts[key].clamp_min(1) for key in sums}
    logits = executor._run_m2f_head_predict(
        executor.model.segmentation_model[1],
        features,
        "m2f",
        rescale_to=image_hw,
    )
    logits = F.interpolate(logits.float(), size=original_hw, mode="bilinear", align_corners=False)
    return torch.softmax(logits, dim=1).argmax(dim=1)[0].to(torch.uint8)


def flop_estimate(backbone: Any, tokens: int, policy: Policy, args: argparse.Namespace) -> dict[str, float]:
    block = backbone.blocks[0]
    hidden = int(block.attn.qkv.in_features)
    heads = int(block.attn.num_heads)
    ffn = int(block.mlp.w1.out_features)
    active = SPECIAL_TOKENS + math.ceil((tokens - SPECIAL_TOKENS) * policy.token_rate)
    depth = len(backbone.blocks)
    middle = depth - FULL_PREFIX_LAYERS - FULL_SUFFIX_LAYERS
    boundary = FULL_PREFIX_LAYERS + FULL_SUFFIX_LAYERS
    full_qkv = 6 * tokens * hidden * hidden
    full_qk = 2 * tokens * tokens * hidden
    full_softmax = 5 * heads * tokens * tokens
    full_pv = 2 * tokens * tokens * hidden
    full_out = 2 * tokens * hidden * hidden
    full_ffn = 6 * tokens * hidden * ffn
    edge_rate = min(
        1.0,
        math.ceil(math.ceil(tokens / args.key_block) * policy.attention_rate)
        / math.ceil(tokens / args.key_block)
        + math.ceil(SPECIAL_TOKENS / args.key_block) / math.ceil(tokens / args.key_block),
    )
    correction = {
        "boundary_full_qkv": boundary * full_qkv,
        "boundary_full_qk": boundary * full_qk,
        "boundary_full_softmax": boundary * full_softmax,
        "boundary_full_pv": boundary * full_pv,
        "boundary_full_output_projection": boundary * full_out,
        "boundary_full_ffn": boundary * full_ffn,
        "middle_qkv_generation": middle * 6 * active * hidden * hidden,
        "middle_qk_selected": middle * 2 * active * tokens * hidden * edge_rate,
        "middle_softmax_full_denominator": middle * 5 * heads * active * tokens,
        "middle_pv_selected": middle * 2 * active * tokens * hidden * edge_rate,
        "middle_output_projection": middle * 2 * active * hidden * hidden,
        "middle_gate1_full": middle * 2 * active * hidden * ffn,
        "middle_selected_gate0_up0_up1_down": (
            middle * 8 * active * hidden * ffn * policy.ffn_rate
        ),
    }
    approx = {
        "approx_qkv": depth * full_qkv,
        "approx_qk": depth * full_qk,
        "approx_softmax": depth * full_softmax,
        "approx_pv": depth * full_pv,
        "approx_output_projection": depth * full_out,
        "approx_ffn": depth * full_ffn,
    }
    return {
        **{key: float(value) for key, value in approx.items()},
        **{key: float(value) for key, value in correction.items()},
        "approx_total": float(sum(approx.values())),
        "correction_total": float(sum(correction.values())),
        "total": float(sum(approx.values()) + sum(correction.values())),
        "dense_full_reference": float(depth * (full_qkv + full_qk + full_softmax + full_pv + full_out + full_ffn)),
        "active_tokens": float(active),
        "estimated_attention_edge_rate": float(edge_rate),
    }


def cache_estimate(backbone: Any, tokens: int) -> dict[str, float]:
    block = backbone.blocks[0]
    hidden = int(block.attn.qkv.in_features)
    heads = int(block.attn.num_heads)
    depth = len(backbone.blocks)
    middle = depth - FULL_PREFIX_LAYERS - FULL_SUFFIX_LAYERS
    bytes_per = 2
    boundary_states = (middle + 1) * tokens * hidden * bytes_per
    ffn_prenorm_states = middle * tokens * hidden * bytes_per
    logits = middle * heads * tokens * tokens * bytes_per
    kv = middle * 2 * tokens * hidden * bytes_per
    return {
        "layer_boundary_state_bytes": float(boundary_states),
        "ffn_prenorm_state_bytes": float(ffn_prenorm_states),
        "base_state_total_bytes": float(boundary_states + ffn_prenorm_states),
        "cached_pre_softmax_logits_bytes": float(logits),
        "alternative_cached_kv_bytes": float(kv),
        "ffn_extra_gate_up_cache_bytes": 0.0,
    }


def write_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()


def load_existing(paths: Iterable[Path]) -> tuple[set[int], dict[str, AreaMetrics]]:
    completed: set[int] = set()
    metrics: dict[str, AreaMetrics] = defaultdict(AreaMetrics)
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                index = int(row["index"])
                if index in completed:
                    continue
                completed.add(index)
                for name, areas in row["areas"].items():
                    metrics[name].add_areas(areas)
    return completed, metrics


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    torch.set_grad_enabled(False)
    policies = make_policies(args)
    include_full = ("full" in args.policies and not args.endpoint_only) or args.endpoint_only
    include_approx = "approx" in args.policies and not args.endpoint_only
    args.output.mkdir(parents=True, exist_ok=True)
    result_path = args.output / "results.jsonl"
    completed, metrics = load_existing([*args.resume_from, result_path])

    executor, _ = load_executor(args)
    adapter = executor.model.segmentation_model[0]
    backbone = adapter.backbone
    ffn_weight_scores = [
        (
            block.mlp.w2.weight.float().norm(dim=1)
            * block.mlp.w3.weight.float().norm(dim=0)
        ).to(block.mlp.w2.weight.dtype)
        for block in backbone.blocks
    ]
    support = {policy.name: SupportAccumulator() for policy in policies}

    from datasets import load_dataset

    dataset = load_dataset("merve/scene_parse_150", split="validation")
    end_index = min(args.start_index + args.num_images, len(dataset))
    endpoint_errors: list[dict[str, float]] = []
    start_time = time.time()
    for index in range(args.start_index, end_index):
        if index in completed:
            continue
        sample = dataset[index]
        original = sample["image"].convert("RGB")
        annotation = np.asarray(sample["annotation"], dtype=np.uint8)
        original_np = np.asarray(original, dtype=np.uint8)
        resized = executor._resize_short_side(original, args.short_side)
        resized_np = np.asarray(resized, dtype=np.uint8)
        image_hw = (resized_np.shape[0], resized_np.shape[1])
        # Preserve true resolution differences: derive Lk from the native image
        # before scaling every level onto the common model/crop coordinate system.
        base_canvas_np = scaled_native_pyramid_level(
            original_np,
            args.base_level,
            image_hw,
        )
        original_hw = (original.height, original.width)
        crops = sliding_crops(*image_hw, args.crop_size, args.stride)
        names = [policy.name for policy in policies]
        if include_full:
            names.append("full")
        if include_approx:
            names.append("approx")
        feature_sums = {name: {} for name in names}
        feature_counts = {name: {} for name in names}

        for crop_index, crop in enumerate(crops):
            y1, y2, x1, x2 = crop
            full_np = np.ascontiguousarray(resized_np[y1:y2, x1:x2])
            base_np = np.ascontiguousarray(base_canvas_np[y1:y2, x1:x2])
            full_tensor = executor._pil_to_normalized_tensor(Image.fromarray(full_np))
            base_tensor = executor._pil_to_normalized_tensor(Image.fromarray(base_np))
            with torch.autocast("cuda", executor.autocast_dtype):
                full_source = executor._prepare_single_source(full_tensor, adapter, backbone)
                source = full_source
                if args.spm_input == "base":
                    base_source = executor._prepare_single_source(base_tensor, adapter, backbone)
                    source = replace_spm_features(full_source, base_source)
                full_initial = full_source["x_backbone"]
                base_initial, _ = backbone.prepare_tokens_with_masks(base_tensor)
                base_states, base_attention_states, attention_score, base_raw = run_base(
                    backbone,
                    base_initial,
                    source["rope_sincos"],
                )
                full_raw = None
                if include_full:
                    _, full_raw = run_full(
                        backbone,
                        full_initial,
                        full_source["rope_sincos"],
                    )

                if include_full:
                    assert full_raw is not None
                    add_crop_features(
                        executor,
                        feature_sums["full"],
                        feature_counts["full"],
                        full_source,
                        full_raw,
                        crop,
                        image_hw,
                    )
                if include_approx:
                    add_crop_features(
                        executor,
                        feature_sums["approx"],
                        feature_counts["approx"],
                        source,
                        base_raw,
                        crop,
                        image_hw,
                    )

                for policy in policies:
                    active = token_support(
                        full_np,
                        base_np,
                        attention_score,
                        policy.token_rate,
                    )
                    support[policy.name].add_token(
                        (active.shape[1] - SPECIAL_TOKENS)
                        / max(full_initial.shape[1] - SPECIAL_TOKENS, 1)
                    )
                    policy_raw = run_policy(
                        backbone,
                        full_initial,
                        base_states,
                        base_attention_states,
                        source["rope_sincos"],
                        active,
                        policy,
                        args,
                        support[policy.name],
                        ffn_weight_scores,
                    )
                    if args.endpoint_only:
                        for layer_index in INTERMEDIATE_LAYERS:
                            actual = policy_raw[layer_index].float()
                            expected = full_raw[layer_index].float()
                            endpoint_errors.append(
                                {
                                    "image": float(index),
                                    "crop": float(crop_index),
                                    "layer": float(layer_index),
                                    "max_abs": float((actual - expected).abs().max().item()),
                                    "relative_l2": float(
                                        (actual - expected).norm().item()
                                        / expected.norm().clamp_min(1e-12).item()
                                    ),
                                }
                            )
                    add_crop_features(
                        executor,
                        feature_sums[policy.name],
                        feature_counts[policy.name],
                        source,
                        policy_raw,
                        crop,
                        image_hw,
                    )
            del base_states, base_attention_states, base_raw, full_raw, source, full_source
            torch.cuda.empty_cache()

        predictions = {}
        areas = {}
        with torch.autocast("cuda", executor.autocast_dtype):
            for name in names:
                predictions[name] = finish_prediction(
                    executor,
                    feature_sums[name],
                    feature_counts[name],
                    image_hw,
                    original_hw,
                )
                areas[name] = metrics[name].update(predictions[name], annotation)
        endpoint_pixel_disagreement = None
        if args.endpoint_only:
            endpoint_prediction = predictions["endpoint_all_support"]
            full_prediction = predictions["full"]
            endpoint_pixel_disagreement = float(
                (endpoint_prediction != full_prediction).float().mean().item()
            )
        write_jsonl(
            result_path,
            {
                "index": index,
                "areas": areas,
                "image_hw": image_hw,
                "original_hw": original_hw,
                "num_crops": len(crops),
                "endpoint_pixel_disagreement": endpoint_pixel_disagreement,
            },
        )
        elapsed = time.time() - start_time
        print(
            json.dumps(
                {
                    "completed_index": index,
                    "elapsed_seconds": elapsed,
                    "metrics": {name: metrics[name].summary() for name in names},
                },
                sort_keys=True,
            ),
            flush=True,
        )

    token_count = SPECIAL_TOKENS + (args.crop_size // 16) ** 2
    summary = {
        "configuration": {
            **vars(args),
            "output": str(args.output),
            "resume_from": [str(path) for path in args.resume_from],
            "policies": [asdict(policy) for policy in policies],
            "intermediate_layers": list(INTERMEDIATE_LAYERS),
            "full_prefix_layers": FULL_PREFIX_LAYERS,
            "full_suffix_layers": FULL_SUFFIX_LAYERS,
            "base_canvas_mode": BASE_CANVAS_MODE,
        },
        "metrics": {name: value.summary() for name, value in metrics.items()},
        "support": {name: value.summary() for name, value in support.items()},
        "flops_per_crop": {
            policy.name: flop_estimate(backbone, token_count, policy, args)
            for policy in policies
        },
        "cache_per_crop": cache_estimate(backbone, token_count),
        "endpoint_errors": endpoint_errors,
        "elapsed_seconds": time.time() - start_time,
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "metrics": summary["metrics"]}, sort_keys=True))


if __name__ == "__main__":
    main()
