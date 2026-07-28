#!/usr/bin/env python3
"""Measure draft-guided attention and SwiGLU support on real DINOv3 states.

This is a streaming local oracle: base and full-resolution states advance
through the stock blocks, while selected layers are audited in FP32 query
chunks.  It never retains a full ``[B,H,N,N]`` probability tensor across a
layer and never writes raw activations to disk.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
from typing import Iterable

import numpy as np
try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.shared.cuda_environment import configure_triton_cuda_environment

configure_triton_cuda_environment()

import torch
import torch.nn.functional as F
from tqdm import tqdm

from analysis.shared.dinov3_probe import Dinov3SignalProbe
from appcorr.models.dinov3.layers.jacobian_support import (
    attention_delta,
    attention_edge_energy,
    select_attention_block_support,
    select_ffn_block_support,
    silu_derivative,
)
from offload.common import ExperimentConfig


def load_config(path: str) -> tuple[ExperimentConfig, dict]:
    raw = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    return ExperimentConfig(**raw), raw


def normalize_offload_dataset_name(name: str) -> str:
    aliases = {
        "imagenet": "imagenet-1k",
        "imnet": "imagenet-1k",
        "imagenet-1k": "imagenet-1k",
        "coco": "coco2017",
        "coco2017": "coco2017",
        "ade20k": "ade20k",
        "nyu": "nyu_depth",
        "nyuv2": "nyu_depth",
        "nyu_depth": "nyu_depth",
    }
    normalized = str(name).strip().lower()
    if normalized not in aliases:
        raise ValueError(f"Unknown dataset: {name}")
    return aliases[normalized]


def parse_layers(value: str | None) -> list[int] | None:
    if value is None or not value.strip():
        return None
    return sorted({int(item.strip()) for item in value.split(",") if item.strip()})


def determine_data_root(dataset_name: str, override: str | None) -> str:
    if override is not None:
        return str(Path(override).expanduser())
    if dataset_name == "imagenet-1k":
        return str(Path("~/data/imagenet_val").expanduser())
    return ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="offload/config/imnet_interleaved_g4.json",
        help="Checked-in offload config used for model input shape and dataset.",
    )
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--image", action="append", default=[])
    parser.add_argument(
        "--stratified-classes",
        type=int,
        default=0,
        help=(
            "For ImageNet, select one deterministic random image from this many "
            "distinct classes instead of iterating ImageFolder order."
        ),
    )
    parser.add_argument("--sample-seed", type=int, default=20260727)
    parser.add_argument("--sample-shard-index", type=int, default=0)
    parser.add_argument("--sample-num-shards", type=int, default=1)
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use one deterministic synthetic image; intended only for smoke tests.",
    )
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--layers", default=None, help="Comma-separated; default all 40.")
    parser.add_argument("--base-level", type=int, default=2)
    parser.add_argument(
        "--num-groups",
        type=int,
        default=4,
        help="Split full-resolution patches into cumulative correction steps.",
    )
    parser.add_argument("--group-strategy", default="grid")
    parser.add_argument("--query-chunk", type=int, default=16)
    parser.add_argument("--support", default="0.125,0.25,0.5")
    parser.add_argument("--tail-epsilon", default="0.05,0.1")
    parser.add_argument("--query-block", type=int, default=8)
    parser.add_argument("--key-block", type=int, default=16)
    parser.add_argument("--head-group", type=int, default=4)
    parser.add_argument("--ffn-channel-block", type=int, default=128)
    parser.add_argument("--ffn-token-block", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dense-gate",
        action="store_true",
        help="Propagate dense correction backends through all layers and compare logits.",
    )
    parser.add_argument(
        "--exact-support-sweep",
        action="store_true",
        help="Sweep exact nonlinear delta support and compare final backbone features.",
    )
    parser.add_argument(
        "--exact-component-sweep",
        action="store_true",
        help=(
            "Sweep input-token, attention-edge, and FFN-channel support separately "
            "while holding the other exact-delta components at 100%."
        ),
    )
    parser.add_argument(
        "--exact-layer-component-sweep",
        action="store_true",
        help=(
            "Sweep each exact-difference component in one target layer at a time, "
            "then propagate the perturbation through the remaining stock layers."
        ),
    )
    parser.add_argument(
        "--target-layers",
        default=None,
        help="Comma-separated layer sweep targets; default all backbone layers.",
    )
    parser.add_argument(
        "--sweep-ratios",
        default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("logs/analysis/jacobian_support_oracle.json"),
    )
    return parser.parse_args()


def low_resolution_canvas(image_bhwc: np.ndarray, level: int) -> np.ndarray:
    if cv2 is not None:
        current = image_bhwc
        for _ in range(level):
            current = cv2.pyrDown(current)
        for target_level in reversed(range(level)):
            target_h = image_bhwc.shape[0] // (2**target_level)
            target_w = image_bhwc.shape[1] // (2**target_level)
            current = cv2.pyrUp(current, dstsize=(target_w, target_h)).astype(np.uint8)
        return current

    tensor = (
        torch.from_numpy(image_bhwc)
        .permute(2, 0, 1)
        .unsqueeze(0)
        .float()
    )
    height, width = image_bhwc.shape[:2]
    low_height = max(1, height // (2**level))
    low_width = max(1, width // (2**level))
    low = F.interpolate(
        tensor,
        size=(low_height, low_width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    restored = F.interpolate(
        low,
        size=(height, width),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    return (
        restored.clamp(0, 255)
        .round()
        .to(torch.uint8)[0]
        .permute(1, 2, 0)
        .numpy()
    )


def progressive_canvases(
    base_bhwc: np.ndarray,
    full_bhwc: np.ndarray,
    *,
    patch_size: tuple[int, int],
    num_groups: int,
    group_strategy: str,
) -> list[np.ndarray]:
    if num_groups <= 1:
        return [base_bhwc.copy(), full_bhwc.copy()]
    from appcorr.models.dinov3.models.vision_transformer import create_group_index

    height, width = full_bhwc.shape[:2]
    patch_h, patch_w = patch_size
    grid_h, grid_w = height // patch_h, width // patch_w
    group_index = create_group_index(
        grid_h * grid_w,
        num_groups,
        group_strategy,
        device=torch.device("cpu"),
        token_hw=(grid_h, grid_w),
    ).reshape(grid_h, grid_w)
    canvases = [base_bhwc.copy()]
    current = base_bhwc.copy()
    for group_id in range(1, num_groups + 1):
        mask = group_index == group_id
        for row, col in mask.nonzero(as_tuple=False).tolist():
            y0, y1 = row * patch_h, (row + 1) * patch_h
            x0, x1 = col * patch_w, (col + 1) * patch_w
            current[y0:y1, x0:x1] = full_bhwc[y0:y1, x0:x1]
        canvases.append(current.copy())
    if not np.array_equal(canvases[-1], full_bhwc):
        raise RuntimeError("Progressive grouping did not reconstruct the full image")
    return canvases


def _git_output(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def tensor_sums(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    actual = actual.float()
    reference = reference.float()
    return {
        "error_sq": float((actual - reference).square().sum().item()),
        "reference_sq": float(reference.square().sum().item()),
        "actual_sq": float(actual.square().sum().item()),
        "dot": float((actual * reference).sum().item()),
    }


def finish_tensor_sums(values: dict[str, float]) -> dict[str, float]:
    reference_norm = math.sqrt(max(values["reference_sq"], 0))
    actual_norm = math.sqrt(max(values["actual_sq"], 0))
    error_norm = math.sqrt(max(values["error_sq"], 0))
    denominator = max(reference_norm, 1e-12)
    cosine_denominator = max(reference_norm * actual_norm, 1e-12)
    return {
        "relative_l2_error": error_norm / denominator,
        "gap_recovery": 1 - error_norm / denominator,
        "cosine": values["dot"] / cosine_denominator,
        "actual_norm": actual_norm,
        "reference_norm": reference_norm,
    }


def add_sums(target: dict[str, float], values: dict[str, float]) -> None:
    for key, value in values.items():
        target[key] = target.get(key, 0.0) + value


class Dinov3JacobianOracle(Dinov3SignalProbe):
    def __init__(
        self,
        *,
        device: torch.device,
        image_size: int,
        layers: list[int] | None,
        query_chunk: int,
        support_ratios: list[float],
        tail_epsilons: list[float],
        query_block: int,
        key_block: int,
        head_group: int,
        ffn_channel_block: int,
        ffn_token_block: int,
    ):
        super().__init__(device=device, image_size=image_size, layers=layers)
        self.query_chunk = query_chunk
        self.support_ratios = support_ratios
        self.tail_epsilons = tail_epsilons
        self.query_block = query_block
        self.key_block = key_block
        self.head_group = head_group
        self.ffn_channel_block = ffn_channel_block
        self.ffn_token_block = ffn_token_block

    def _attention_qkv(
        self,
        block,
        x: torch.Tensor,
        rope,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = block.attn.qkv(block.norm1(x))
        batch, tokens, _ = qkv.shape
        embed_dim = block.attn.qkv.in_features
        qkv = qkv.reshape(
            batch,
            tokens,
            3,
            block.attn.num_heads,
            embed_dim // block.attn.num_heads,
        )
        q, k, v = (value.transpose(1, 2) for value in torch.unbind(qkv, dim=2))
        if rope is not None:
            q, k = block.attn.apply_rope(q, k, rope)
        return q, k, v

    def _audit_attention(
        self,
        block,
        x_base: torch.Tensor,
        x_full: torch.Tensor,
        rope,
    ) -> dict[str, object]:
        q_base, k_base, v_base = self._attention_qkv(block, x_base, rope)
        q_full, k_full, v_full = self._attention_qkv(block, x_full, rope)
        k_base = k_base.float()
        v_base = v_base.float()
        dk = k_full.float() - k_base
        dv = v_full.float() - v_base
        scale = float(block.attn.scale)
        tokens = q_base.shape[-2]

        metric_sums: dict[str, dict[str, float]] = {
            "split_linearized": {},
            "product_linearized": {},
            "product_exact_probability": {},
            "cross_term": {},
        }
        support_sums: dict[str, dict[str, float]] = {}
        residual_score = dv.square().sum(dim=-1).mean(dim=1)

        for query_start in range(0, tokens, self.query_chunk):
            query_end = min(tokens, query_start + self.query_chunk)
            q0 = q_base[:, :, query_start:query_end].float()
            dq = q_full[:, :, query_start:query_end].float() - q0
            linearized = attention_delta(
                q0,
                k_base,
                v_base,
                dq,
                dk,
                dv,
                scale=scale,
                backend="split_jvp",
                probability_mode="linearized",
            )
            exact_probability = attention_delta(
                q0,
                k_base,
                v_base,
                dq,
                dk,
                dv,
                scale=scale,
                backend="product_delta",
                probability_mode="exact",
            )
            exact = exact_probability.delta
            add_sums(
                metric_sums["split_linearized"],
                tensor_sums(linearized.delta, exact),
            )
            add_sums(
                metric_sums["product_linearized"],
                tensor_sums(linearized.delta + linearized.cross_term, exact),
            )
            add_sums(
                metric_sums["product_exact_probability"],
                tensor_sums(exact_probability.delta, exact),
            )
            add_sums(
                metric_sums["cross_term"],
                tensor_sums(linearized.cross_term, exact),
            )

            oracle_edge_energy = attention_edge_energy(
                exact_probability.base_probability,
                exact_probability.delta_probability,
                v_base,
                dv,
                backend="product_delta",
            )
            total_energy = float(oracle_edge_energy.sum().item())

            for ratio in self.support_ratios:
                residual_blocks = max(1, math.ceil(tokens * ratio / 2))
                residual_idx = torch.topk(
                    residual_score,
                    k=min(residual_blocks, tokens),
                    dim=-1,
                ).indices
                residual_mask = torch.zeros_like(residual_score, dtype=torch.bool)
                residual_mask.scatter_(-1, residual_idx, True)
                for support_name, residual_key_mask in (
                    ("base_attention", None),
                    ("residual_union", residual_mask),
                ):
                    mask, stats = select_attention_block_support(
                        linearized.base_probability,
                        keep_ratio=ratio,
                        key_block_size=self.key_block,
                        query_block_size=self.query_block,
                        head_group_size=self.head_group,
                        residual_key_mask=residual_key_mask,
                    )
                    kept_energy = float(oracle_edge_energy.masked_select(mask).sum().item())
                    sparse_delta = (
                        (exact_probability.base_probability + exact_probability.delta_probability)
                        * mask
                    ) @ (v_base + dv) - (
                        exact_probability.base_probability * mask
                    ) @ v_base
                    key = f"{support_name}:ratio={ratio:g}"
                    sums = support_sums.setdefault(key, {})
                    sums["oracle_energy"] = sums.get("oracle_energy", 0) + total_energy
                    sums["kept_oracle_energy"] = (
                        sums.get("kept_oracle_energy", 0) + kept_energy
                    )
                    sums["kept_fraction_weighted"] = (
                        sums.get("kept_fraction_weighted", 0)
                        + stats.kept_fraction * mask.numel()
                    )
                    sums["mask_elements"] = sums.get("mask_elements", 0) + mask.numel()
                    add_sums(sums, {
                        f"sparse_{name}": value
                        for name, value in tensor_sums(sparse_delta, exact).items()
                    })

            for epsilon in self.tail_epsilons:
                mask, stats = select_attention_block_support(
                    linearized.base_probability,
                    tail_epsilon=epsilon,
                    key_block_size=self.key_block,
                    query_block_size=self.query_block,
                    head_group_size=self.head_group,
                )
                kept_energy = float(oracle_edge_energy.masked_select(mask).sum().item())
                key = f"adaptive_tail:epsilon={epsilon:g}"
                sums = support_sums.setdefault(key, {})
                sums["oracle_energy"] = sums.get("oracle_energy", 0) + total_energy
                sums["kept_oracle_energy"] = (
                    sums.get("kept_oracle_energy", 0) + kept_energy
                )
                sums["kept_fraction_weighted"] = (
                    sums.get("kept_fraction_weighted", 0)
                    + stats.kept_fraction * mask.numel()
                )
                sums["mask_elements"] = sums.get("mask_elements", 0) + mask.numel()

            del linearized, exact_probability, oracle_edge_energy

        metrics = {
            name: finish_tensor_sums(values)
            for name, values in metric_sums.items()
        }
        support = {}
        for name, values in support_sums.items():
            item = {
                "kept_fraction": values["kept_fraction_weighted"]
                / max(values["mask_elements"], 1),
                "oracle_energy_recovery": values["kept_oracle_energy"]
                / max(values["oracle_energy"], 1e-12),
            }
            if "sparse_error_sq" in values:
                item.update(
                    finish_tensor_sums({
                        "error_sq": values["sparse_error_sq"],
                        "reference_sq": values["sparse_reference_sq"],
                        "actual_sq": values["sparse_actual_sq"],
                        "dot": values["sparse_dot"],
                    })
                )
            support[name] = item
        return {"metrics": metrics, "support": support}

    def _audit_ffn(
        self,
        block,
        x_base_attn: torch.Tensor,
        x_full_attn: torch.Tensor,
    ) -> dict[str, object]:
        mlp = block.mlp
        if not all(hasattr(mlp, name) for name in ("w1", "w2", "w3")):
            return {"unsupported": type(mlp).__name__}

        norm_base = block.norm2(x_base_attn)
        norm_full = block.norm2(x_full_attn)
        dx = norm_full - norm_base
        gate = mlp.w1(norm_base).float()
        up = mlp.w2(norm_base).float()
        delta_gate = mlp.w1(norm_full).float() - gate
        delta_up = mlp.w2(norm_full).float() - up
        jvp_hidden = (
            silu_derivative(gate) * up * delta_gate
            + F.silu(gate) * delta_up
        )
        exact_hidden = (
            F.silu(gate + delta_gate) * (up + delta_up)
            - F.silu(gate) * up
        )
        down_weight = mlp.w3.weight.float()
        jvp_output = F.linear(jvp_hidden, down_weight)
        exact_output = F.linear(exact_hidden, down_weight)

        dx_norm = dx.float().square().sum(dim=-1, keepdim=True).sqrt()
        gate_row_norm = mlp.w1.weight.float().square().sum(dim=-1).sqrt()
        up_row_norm = mlp.w2.weight.float().square().sum(dim=-1).sqrt()
        down_column_norm = down_weight.square().sum(dim=0).sqrt()
        training_free_score = dx_norm * (
            (silu_derivative(gate) * up).abs() * gate_row_norm
            + F.silu(gate).abs() * up_row_norm
        ) * down_column_norm
        oracle_channel_energy = exact_hidden.abs() * down_column_norm

        support = {}
        for ratio in self.support_ratios:
            mask = select_ffn_block_support(
                training_free_score,
                keep_ratio=ratio,
                channel_block_size=self.ffn_channel_block,
                token_block_size=self.ffn_token_block,
            )
            sparse_output = F.linear(jvp_hidden.masked_fill(~mask, 0), down_weight)
            values = finish_tensor_sums(tensor_sums(sparse_output, exact_output))
            values["kept_fraction"] = float(mask.float().mean().item())
            values["oracle_energy_recovery"] = float(
                oracle_channel_energy.masked_select(mask).sum().item()
                / max(oracle_channel_energy.sum().item(), 1e-12)
            )
            support[f"derivative_bound:ratio={ratio:g}"] = values

        result = {
            "metrics": {
                "split_jvp": finish_tensor_sums(
                    tensor_sums(jvp_output, exact_output)
                )
            },
            "support": support,
        }
        del down_weight
        return result

    @torch.no_grad()
    def analyze_pair(
        self,
        base_bchw_uint8: torch.Tensor,
        full_bchw_uint8: torch.Tensor,
    ) -> list[dict[str, object]]:
        base_input = self._prepare_input(base_bchw_uint8)
        full_input = self._prepare_input(full_bchw_uint8)
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        rows = []
        with context:
            x_base, hw_base = self.backbone.prepare_tokens_with_masks(base_input, None)
            x_full, hw_full = self.backbone.prepare_tokens_with_masks(full_input, None)
            if hw_base != hw_full:
                raise RuntimeError(f"Token grid mismatch: {hw_base} vs {hw_full}")
            rope = (
                self.backbone.rope_embed(H=hw_base[0], W=hw_base[1])
                if self.backbone.rope_embed is not None
                else None
            )

            for layer_index, block in enumerate(self.backbone.blocks):
                selected = self.layers is None or layer_index in self.layers
                attention = (
                    self._audit_attention(block, x_base, x_full, rope)
                    if selected
                    else None
                )
                x_base_attn = x_base + block.ls1(
                    block.attn(block.norm1(x_base), rope=rope)
                )
                x_full_attn = x_full + block.ls1(
                    block.attn(block.norm1(x_full), rope=rope)
                )
                ffn = (
                    self._audit_ffn(block, x_base_attn, x_full_attn)
                    if selected
                    else None
                )
                x_base = x_base_attn + block.ls2(block.mlp(block.norm2(x_base_attn)))
                x_full = x_full_attn + block.ls2(block.mlp(block.norm2(x_full_attn)))
                if selected:
                    rows.append({
                        "layer": layer_index,
                        "tokens": x_base.shape[1],
                        "attention": attention,
                        "ffn": ffn,
                    })
        return rows

    def _classifier_logits(self, x: torch.Tensor) -> torch.Tensor:
        output = self.backbone.post_features_list([x], [None])[0]
        linear_input = torch.cat([
            output["x_norm_clstoken"],
            output["x_norm_patchtokens"].mean(dim=1),
        ], dim=-1)
        return self.model.linear_head(linear_input)

    @torch.no_grad()
    def dense_propagation_gate(
        self,
        base_bchw_uint8: torch.Tensor,
        full_bchw_uint8: torch.Tensor,
    ) -> dict[str, object]:
        """Propagate dense formulas, rather than only measuring local errors."""

        variants = {
            "split_jvp": {
                "attention_backend": "split_jvp",
                "probability_mode": "linearized",
                "exact_ffn": False,
            },
            "linearized_product": {
                "attention_backend": "product_delta",
                "probability_mode": "linearized",
                "exact_ffn": False,
            },
            "exact_probability_product_jvp_ffn": {
                "attention_backend": "product_delta",
                "probability_mode": "exact",
                "exact_ffn": False,
            },
            "exact_probability_product_exact_ffn": {
                "attention_backend": "product_delta",
                "probability_mode": "exact",
                "exact_ffn": True,
            },
        }
        base_input = self._prepare_input(base_bchw_uint8)
        full_input = self._prepare_input(full_bchw_uint8)
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with context:
            x_base, hw = self.backbone.prepare_tokens_with_masks(base_input, None)
            x_full, full_hw = self.backbone.prepare_tokens_with_masks(full_input, None)
            if hw != full_hw:
                raise RuntimeError("Dense gate token grids must match")
            rope = (
                self.backbone.rope_embed(H=hw[0], W=hw[1])
                if self.backbone.rope_embed is not None
                else None
            )
            variant_states = {
                name: x_full.clone()
                for name in variants
            }
            for block in self.backbone.blocks:
                q_base, k_base, v_base = self._attention_qkv(
                    block, x_base, rope
                )
                q_base_f = q_base.float()
                k_base_f = k_base.float()
                v_base_f = v_base.float()
                x_base_attn = x_base + block.ls1(
                    block.attn(block.norm1(x_base), rope=rope)
                )
                base_norm2 = block.norm2(x_base_attn)
                base_gate = block.mlp.w1(base_norm2).float()
                base_up = block.mlp.w2(base_norm2).float()
                base_hidden = F.silu(base_gate) * base_up

                next_states = {}
                for name, options in variants.items():
                    state = variant_states[name]
                    q_new, k_new, v_new = self._attention_qkv(
                        block, state, rope
                    )
                    result = attention_delta(
                        q_base_f,
                        k_base_f,
                        v_base_f,
                        q_new.float() - q_base_f,
                        k_new.float() - k_base_f,
                        v_new.float() - v_base_f,
                        scale=float(block.attn.scale),
                        backend=options["attention_backend"],
                        probability_mode=options["probability_mode"],
                    )
                    corrected_raw = (
                        result.corrected_output.transpose(1, 2)
                        .reshape(state.shape)
                        .to(dtype=state.dtype)
                    )
                    corrected_attention = block.attn.proj_drop(
                        block.attn.proj(corrected_raw)
                    )
                    state_attn = state + block.ls1(corrected_attention)
                    corrected_norm2 = block.norm2(state_attn)
                    corrected_gate = block.mlp.w1(corrected_norm2).float()
                    corrected_up = block.mlp.w2(corrected_norm2).float()
                    delta_gate = corrected_gate - base_gate
                    delta_up = corrected_up - base_up
                    if options["exact_ffn"]:
                        delta_hidden = (
                            F.silu(corrected_gate) * corrected_up
                            - base_hidden
                        )
                    else:
                        delta_hidden = (
                            silu_derivative(base_gate) * base_up * delta_gate
                            + F.silu(base_gate) * delta_up
                        )
                    corrected_hidden = (base_hidden + delta_hidden).to(
                        dtype=state.dtype
                    )
                    corrected_ffn = block.mlp.w3(corrected_hidden)
                    next_states[name] = state_attn + block.ls2(corrected_ffn)

                variant_states = next_states
                x_base = x_base_attn + block.ls2(block.mlp(base_norm2))
                x_full = block(x_full, rope)

            logits_base = self._classifier_logits(x_base).float()
            logits_full = self._classifier_logits(x_full).float()
            results = {
                "base_only": {
                    **finish_tensor_sums(tensor_sums(logits_base, logits_full)),
                    "top1_match": bool(
                        torch.equal(logits_base.argmax(-1), logits_full.argmax(-1))
                    ),
                    "predicted_top1": int(logits_base.argmax(-1).item()),
                }
            }
            for name, state in variant_states.items():
                logits = self._classifier_logits(state).float()
                results[name] = {
                    **finish_tensor_sums(tensor_sums(logits, logits_full)),
                    "top1_match": bool(
                        torch.equal(logits.argmax(-1), logits_full.argmax(-1))
                    ),
                    "predicted_top1": int(logits.argmax(-1).item()),
                }
            results["stock_top1"] = int(logits_full.argmax(-1).item())
            return results

    def _normalized_backbone_features(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.backbone.post_features_list([x], [None])[0]
        cls = output["x_norm_clstoken"]
        patches = output["x_norm_patchtokens"]
        tokens = torch.cat([cls.unsqueeze(1), patches], dim=1)
        pooled = torch.cat([cls, patches.mean(dim=1)], dim=-1)
        return tokens, pooled

    @torch.no_grad()
    def exact_support_feature_sweep(
        self,
        base_bchw_uint8: torch.Tensor,
        full_bchw_uint8: torch.Tensor,
        ratios: list[float],
        *,
        component: str = "joint",
    ) -> list[dict[str, object]]:
        """Propagate exact nonlinear deltas at a variable structured support."""

        if any(ratio < 0 or ratio > 1 for ratio in ratios):
            raise ValueError("Sweep ratios must be in [0, 1]")
        valid_components = {
            "joint",
            "input_token",
            "attention_edge",
            "ffn_channel",
        }
        if component not in valid_components:
            raise ValueError(
                f"Unknown sweep component {component!r}; expected {valid_components}"
            )

        def component_ratio(name: str, requested_ratio: float) -> float:
            return requested_ratio if component in {"joint", name} else 1.0

        base_input = self._prepare_input(base_bchw_uint8)
        full_input = self._prepare_input(full_bchw_uint8)
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with context:
            x_base, hw = self.backbone.prepare_tokens_with_masks(base_input, None)
            x_full, full_hw = self.backbone.prepare_tokens_with_masks(full_input, None)
            if hw != full_hw:
                raise RuntimeError("Exact support sweep token grids must match")
            rope = (
                self.backbone.rope_embed(H=hw[0], W=hw[1])
                if self.backbone.rope_embed is not None
                else None
            )

            input_delta = x_full - x_base
            input_score = input_delta.float().square().sum(dim=-1)
            token_count = x_base.shape[1]
            states: dict[float, torch.Tensor] = {}
            realized: dict[float, dict[str, float]] = {}
            for ratio in ratios:
                input_ratio = component_ratio("input_token", ratio)
                if input_ratio <= 0:
                    token_mask = torch.zeros_like(input_score, dtype=torch.bool)
                elif input_ratio >= 1:
                    token_mask = torch.ones_like(input_score, dtype=torch.bool)
                else:
                    keep = max(1, math.ceil(token_count * input_ratio))
                    indices = torch.topk(input_score, k=keep, dim=-1).indices
                    token_mask = torch.zeros_like(input_score, dtype=torch.bool)
                    token_mask.scatter_(-1, indices, True)
                states[ratio] = x_base + input_delta * token_mask.unsqueeze(-1)
                realized[ratio] = {
                    "input_token_keep": float(token_mask.float().mean().item()),
                    "attention_keep_sum": 0.0,
                    "ffn_keep_sum": 0.0,
                    "layers": 0.0,
                }

            for block in self.backbone.blocks:
                q_base, k_base, v_base = self._attention_qkv(
                    block, x_base, rope
                )
                q_base_f = q_base.float()
                k_base_f = k_base.float()
                v_base_f = v_base.float()
                x_base_attn = x_base + block.ls1(
                    block.attn(block.norm1(x_base), rope=rope)
                )
                base_norm2 = block.norm2(x_base_attn)
                base_gate = block.mlp.w1(base_norm2).float()
                base_up = block.mlp.w2(base_norm2).float()
                base_hidden = F.silu(base_gate) * base_up
                down_norm = block.mlp.w3.weight.float().square().sum(dim=0).sqrt()

                next_states = {}
                for ratio, state in states.items():
                    if component == "joint" and ratio <= 0:
                        next_states[ratio] = (
                            x_base_attn + block.ls2(block.mlp(base_norm2))
                        )
                        realized[ratio]["layers"] += 1
                        continue

                    attention_ratio = component_ratio("attention_edge", ratio)
                    ffn_ratio = component_ratio("ffn_channel", ratio)
                    q_new, k_new, v_new = self._attention_qkv(
                        block, state, rope
                    )
                    exact_attention = attention_delta(
                        q_base_f,
                        k_base_f,
                        v_base_f,
                        q_new.float() - q_base_f,
                        k_new.float() - k_base_f,
                        v_new.float() - v_base_f,
                        scale=float(block.attn.scale),
                        backend="product_delta",
                        probability_mode="exact",
                    )
                    if attention_ratio <= 0:
                        attention_mask = torch.zeros_like(
                            exact_attention.base_probability,
                            dtype=torch.bool,
                        )
                        corrected_attention_raw = exact_attention.base_output
                    elif attention_ratio >= 1:
                        attention_mask = torch.ones_like(
                            exact_attention.base_probability,
                            dtype=torch.bool,
                        )
                        corrected_attention_raw = exact_attention.corrected_output
                    else:
                        attention_mask, _ = select_attention_block_support(
                            exact_attention.base_probability,
                            keep_ratio=attention_ratio,
                            key_block_size=self.key_block,
                            query_block_size=self.query_block,
                            head_group_size=self.head_group,
                        )
                        masked_attention = attention_delta(
                            q_base_f,
                            k_base_f,
                            v_base_f,
                            q_new.float() - q_base_f,
                            k_new.float() - k_base_f,
                            v_new.float() - v_base_f,
                            scale=float(block.attn.scale),
                            backend="product_delta",
                            probability_mode="exact",
                            support_mask=attention_mask,
                        )
                        corrected_attention_raw = masked_attention.corrected_output
                    corrected_attention_raw = (
                        corrected_attention_raw.transpose(1, 2)
                        .reshape(state.shape)
                        .to(dtype=state.dtype)
                    )
                    corrected_attention = block.attn.proj_drop(
                        block.attn.proj(corrected_attention_raw)
                    )
                    state_attn = state + block.ls1(corrected_attention)

                    corrected_norm2 = block.norm2(state_attn)
                    corrected_gate = block.mlp.w1(corrected_norm2).float()
                    corrected_up = block.mlp.w2(corrected_norm2).float()
                    exact_hidden_delta = (
                        F.silu(corrected_gate) * corrected_up
                        - base_hidden
                    )
                    if ffn_ratio <= 0:
                        ffn_mask = torch.zeros_like(
                            exact_hidden_delta,
                            dtype=torch.bool,
                        )
                    elif ffn_ratio >= 1:
                        ffn_mask = torch.ones_like(
                            exact_hidden_delta,
                            dtype=torch.bool,
                        )
                    else:
                        channel_score = exact_hidden_delta.abs() * down_norm
                        ffn_mask = select_ffn_block_support(
                            channel_score,
                            keep_ratio=ffn_ratio,
                            channel_block_size=self.ffn_channel_block,
                            token_block_size=self.ffn_token_block,
                        )
                    corrected_hidden = (
                        base_hidden
                        + exact_hidden_delta.masked_fill(~ffn_mask, 0)
                    ).to(dtype=state.dtype)
                    corrected_ffn = block.mlp.w3(corrected_hidden)
                    next_states[ratio] = state_attn + block.ls2(corrected_ffn)
                    realized[ratio]["attention_keep_sum"] += float(
                        attention_mask.float().mean().item()
                    )
                    realized[ratio]["ffn_keep_sum"] += float(
                        ffn_mask.float().mean().item()
                    )
                    realized[ratio]["layers"] += 1

                states = next_states
                x_base = x_base_attn + block.ls2(block.mlp(base_norm2))
                x_full = block(x_full, rope)

            full_tokens, full_pooled = self._normalized_backbone_features(x_full)
            base_tokens, base_pooled = self._normalized_backbone_features(x_base)
            rows = []
            for ratio in ratios:
                tokens, pooled = self._normalized_backbone_features(states[ratio])
                layer_count = max(realized[ratio]["layers"], 1)
                rows.append({
                    "component": component,
                    "requested_ratio": ratio,
                    "input_token_ratio": component_ratio("input_token", ratio),
                    "attention_edge_ratio": component_ratio(
                        "attention_edge", ratio
                    ),
                    "ffn_channel_ratio": component_ratio("ffn_channel", ratio),
                    "realized_input_token_keep": realized[ratio][
                        "input_token_keep"
                    ],
                    "mean_attention_edge_keep": (
                        realized[ratio]["attention_keep_sum"] / layer_count
                    ),
                    "mean_ffn_channel_keep": (
                        realized[ratio]["ffn_keep_sum"] / layer_count
                    ),
                    "normalized_token_feature": finish_tensor_sums(
                        tensor_sums(tokens, full_tokens)
                    ),
                    "pooled_cls_mean_feature": finish_tensor_sums(
                        tensor_sums(pooled, full_pooled)
                    ),
                })
            if component in {"joint", "input_token"}:
                rows[0]["base_endpoint_check"] = finish_tensor_sums(
                    tensor_sums(base_tokens, full_tokens)
                )
            return rows

    @torch.no_grad()
    def exact_layer_component_sweep(
        self,
        base_bchw_uint8: torch.Tensor,
        full_bchw_uint8: torch.Tensor,
        ratios: list[float],
        target_layers: list[int] | None = None,
    ) -> dict[str, dict[str, list[dict[str, object]]]]:
        """Isolate one component in one layer and measure its final effect.

        The target layer receives stock base/full states. Only the requested
        component is support-masked in that layer; all downstream blocks run
        their stock full computation. Thus each row measures the final feature
        sensitivity to a single layer/component support decision.
        """

        if any(ratio < 0 or ratio > 1 for ratio in ratios):
            raise ValueError("Sweep ratios must be in [0, 1]")
        num_layers = len(self.backbone.blocks)
        targets = (
            list(range(num_layers))
            if target_layers is None
            else sorted(set(target_layers))
        )
        if any(layer < 0 or layer >= num_layers for layer in targets):
            raise ValueError(
                f"Target layers must be in [0, {num_layers - 1}], got {targets}"
            )

        base_input = self._prepare_input(base_bchw_uint8)
        full_input = self._prepare_input(full_bchw_uint8)
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with context:
            x_base, hw = self.backbone.prepare_tokens_with_masks(base_input, None)
            x_full, full_hw = self.backbone.prepare_tokens_with_masks(full_input, None)
            if hw != full_hw:
                raise RuntimeError("Layer sweep token grids must match")
            rope = (
                self.backbone.rope_embed(H=hw[0], W=hw[1])
                if self.backbone.rope_embed is not None
                else None
            )

            base_states = [x_base]
            full_states = [x_full]
            for block in self.backbone.blocks:
                base_states.append(block(base_states[-1], rope))
                full_states.append(block(full_states[-1], rope))
            full_tokens, full_pooled = self._normalized_backbone_features(
                full_states[-1]
            )

            results: dict[str, dict[str, list[dict[str, object]]]] = {}
            for target_layer in targets:
                block = self.backbone.blocks[target_layer]
                base_state = base_states[target_layer]
                full_state = full_states[target_layer]
                full_layer_output = full_states[target_layer + 1]
                layer_results: dict[str, list[dict[str, object]]] = {}

                for component in (
                    "input_token",
                    "attention_edge",
                    "ffn_channel",
                ):
                    states: dict[float, torch.Tensor] = {}
                    realized_keep: dict[float, float] = {}

                    if component == "input_token":
                        input_delta = full_state - base_state
                        input_score = input_delta.float().square().sum(dim=-1)
                        token_count = full_state.shape[1]
                        for ratio in ratios:
                            if ratio <= 0:
                                token_mask = torch.zeros_like(
                                    input_score, dtype=torch.bool
                                )
                            elif ratio >= 1:
                                token_mask = torch.ones_like(
                                    input_score, dtype=torch.bool
                                )
                            else:
                                keep = max(1, math.ceil(token_count * ratio))
                                indices = torch.topk(
                                    input_score, k=keep, dim=-1
                                ).indices
                                token_mask = torch.zeros_like(
                                    input_score, dtype=torch.bool
                                )
                                token_mask.scatter_(-1, indices, True)
                            mixed_state = (
                                full_state
                                if ratio >= 1
                                else base_state
                                + input_delta * token_mask.unsqueeze(-1)
                            )
                            states[ratio] = block(mixed_state, rope)
                            realized_keep[ratio] = float(
                                token_mask.float().mean().item()
                            )

                    elif component == "attention_edge":
                        q_base, k_base, v_base = self._attention_qkv(
                            block, base_state, rope
                        )
                        q_full, k_full, v_full = self._attention_qkv(
                            block, full_state, rope
                        )
                        exact_attention = attention_delta(
                            q_base.float(),
                            k_base.float(),
                            v_base.float(),
                            q_full.float() - q_base.float(),
                            k_full.float() - k_base.float(),
                            v_full.float() - v_base.float(),
                            scale=float(block.attn.scale),
                            backend="product_delta",
                            probability_mode="exact",
                        )
                        base_attention = block.attn(
                            block.norm1(base_state), rope=rope
                        )
                        full_attention = block.attn(
                            block.norm1(full_state), rope=rope
                        )
                        for ratio in ratios:
                            if ratio <= 0:
                                corrected_attention = base_attention
                                realized_keep[ratio] = 0.0
                            elif ratio >= 1:
                                corrected_attention = full_attention
                                realized_keep[ratio] = 1.0
                            else:
                                attention_mask, stats = (
                                    select_attention_block_support(
                                        exact_attention.base_probability,
                                        keep_ratio=ratio,
                                        key_block_size=self.key_block,
                                        query_block_size=self.query_block,
                                        head_group_size=self.head_group,
                                    )
                                )
                                masked_attention = attention_delta(
                                    q_base.float(),
                                    k_base.float(),
                                    v_base.float(),
                                    q_full.float() - q_base.float(),
                                    k_full.float() - k_base.float(),
                                    v_full.float() - v_base.float(),
                                    scale=float(block.attn.scale),
                                    backend="product_delta",
                                    probability_mode="exact",
                                    support_mask=attention_mask,
                                )
                                corrected_raw = (
                                    masked_attention.corrected_output.transpose(
                                        1, 2
                                    )
                                    .reshape(full_state.shape)
                                    .to(dtype=full_state.dtype)
                                )
                                corrected_attention = block.attn.proj_drop(
                                    block.attn.proj(corrected_raw)
                                )
                                realized_keep[ratio] = stats.kept_fraction
                            state_attn = full_state + block.ls1(
                                corrected_attention
                            )
                            states[ratio] = state_attn + block.ls2(
                                block.mlp(block.norm2(state_attn))
                            )

                    else:
                        base_state_attn = base_state + block.ls1(
                            block.attn(block.norm1(base_state), rope=rope)
                        )
                        full_state_attn = full_state + block.ls1(
                            block.attn(block.norm1(full_state), rope=rope)
                        )
                        base_norm2 = block.norm2(base_state_attn)
                        full_norm2 = block.norm2(full_state_attn)
                        base_gate = block.mlp.w1(base_norm2).float()
                        base_up = block.mlp.w2(base_norm2).float()
                        full_gate = block.mlp.w1(full_norm2).float()
                        full_up = block.mlp.w2(full_norm2).float()
                        base_hidden = F.silu(base_gate) * base_up
                        exact_hidden_delta = (
                            F.silu(full_gate) * full_up - base_hidden
                        )
                        down_norm = (
                            block.mlp.w3.weight.float()
                            .square()
                            .sum(dim=0)
                            .sqrt()
                        )
                        channel_score = exact_hidden_delta.abs() * down_norm
                        for ratio in ratios:
                            if ratio <= 0:
                                states[ratio] = full_state_attn + block.ls2(
                                    block.mlp(base_norm2)
                                )
                                realized_keep[ratio] = 0.0
                                continue
                            if ratio >= 1:
                                states[ratio] = full_state_attn + block.ls2(
                                    block.mlp(full_norm2)
                                )
                                realized_keep[ratio] = 1.0
                                continue
                            ffn_mask = select_ffn_block_support(
                                channel_score,
                                keep_ratio=ratio,
                                channel_block_size=self.ffn_channel_block,
                                token_block_size=self.ffn_token_block,
                            )
                            corrected_hidden = (
                                base_hidden
                                + exact_hidden_delta.masked_fill(~ffn_mask, 0)
                            ).to(dtype=full_state.dtype)
                            states[ratio] = full_state_attn + block.ls2(
                                block.mlp.w3(corrected_hidden)
                            )
                            realized_keep[ratio] = float(
                                ffn_mask.float().mean().item()
                            )

                    target_outputs = dict(states)
                    for downstream_block in self.backbone.blocks[
                        target_layer + 1:
                    ]:
                        states = {
                            ratio: downstream_block(state, rope)
                            for ratio, state in states.items()
                        }

                    rows = []
                    for ratio in ratios:
                        tokens, pooled = self._normalized_backbone_features(
                            states[ratio]
                        )
                        rows.append({
                            "target_layer": target_layer,
                            "component": component,
                            "requested_ratio": ratio,
                            "realized_keep": realized_keep[ratio],
                            "target_layer_output": finish_tensor_sums(
                                tensor_sums(
                                    target_outputs[ratio],
                                    full_layer_output,
                                )
                            ),
                            "normalized_token_feature": finish_tensor_sums(
                                tensor_sums(tokens, full_tokens)
                            ),
                            "pooled_cls_mean_feature": finish_tensor_sums(
                                tensor_sums(pooled, full_pooled)
                            ),
                        })
                    layer_results[component] = rows
                    del target_outputs, states

                results[str(target_layer)] = layer_results
            return results

    @torch.no_grad()
    def exact_policy_classification_batch(
        self,
        base_bchw_uint8: torch.Tensor,
        full_bchw_uint8: torch.Tensor,
        policies: dict[str, list[dict[str, float]]],
    ) -> dict[str, dict[str, object]]:
        """Apply mixed layer/component policies in one exact-difference pass.

        Token support gates the current residual state against the stock L2
        state at every layer. Attention and FFN use exact nonlinear output
        differences on structured support. The returned L2-only and policy
        tensors are compared against the same stock L0 full forward.
        """

        num_layers = len(self.backbone.blocks)
        policy_rows = {}
        for name, schedule in policies.items():
            if len(schedule) != num_layers:
                raise ValueError(
                    f"Policy {name!r} has {len(schedule)} layers; "
                    f"expected {num_layers}"
                )
            rows = {int(row["layer"]): row for row in schedule}
            if set(rows) != set(range(num_layers)):
                raise ValueError(
                    f"Policy {name!r} must define every layer exactly once"
                )
            for row in rows.values():
                for component in (
                    "input_token_keep",
                    "attention_edge_keep",
                    "ffn_channel_keep",
                ):
                    value = float(row[component])
                    if not 0 <= value <= 1:
                        raise ValueError(
                            f"{name} layer {row['layer']} {component}={value} "
                            "is outside [0, 1]"
                        )
            policy_rows[name] = rows

        base_input = self._prepare_input(base_bchw_uint8)
        full_input = self._prepare_input(full_bchw_uint8)
        context = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.device.type == "cuda"
            else nullcontext()
        )
        with context:
            x_base, hw = self.backbone.prepare_tokens_with_masks(base_input, None)
            x_full, full_hw = self.backbone.prepare_tokens_with_masks(
                full_input, None
            )
            if hw != full_hw:
                raise RuntimeError("Policy correction token grids must match")
            rope = (
                self.backbone.rope_embed(H=hw[0], W=hw[1])
                if self.backbone.rope_embed is not None
                else None
            )
            states = {name: x_full.clone() for name in policies}

            for layer_index, block in enumerate(self.backbone.blocks):
                q_base, k_base, v_base = self._attention_qkv(
                    block, x_base, rope
                )
                q_base_f = q_base.float()
                k_base_f = k_base.float()
                v_base_f = v_base.float()
                base_attention = block.attn(block.norm1(x_base), rope=rope)
                x_base_attn = x_base + block.ls1(base_attention)
                base_norm2 = block.norm2(x_base_attn)
                base_gate = block.mlp.w1(base_norm2).float()
                base_up = block.mlp.w2(base_norm2).float()
                base_hidden = F.silu(base_gate) * base_up
                # Preserve the stock BF16 operation order at the 0% endpoint.
                # The FP32 hidden value above is used only by an intermediate
                # exact-difference correction.
                base_ffn = block.mlp(base_norm2)
                down_norm = (
                    block.mlp.w3.weight.float()
                    .square()
                    .sum(dim=0)
                    .sqrt()
                )

                next_states = {}
                for name, state in states.items():
                    row = policy_rows[name][layer_index]
                    token_ratio = float(row["input_token_keep"])
                    attention_ratio = float(row["attention_edge_keep"])
                    ffn_ratio = float(row["ffn_channel_keep"])

                    if token_ratio <= 0:
                        supported_state = x_base
                    elif token_ratio >= 1:
                        supported_state = state
                    else:
                        state_delta = state - x_base
                        token_score = state_delta.float().square().sum(dim=-1)
                        keep = max(
                            1,
                            math.ceil(state.shape[1] * token_ratio),
                        )
                        token_indices = torch.topk(
                            token_score, k=keep, dim=-1
                        ).indices
                        token_mask = torch.zeros_like(
                            token_score, dtype=torch.bool
                        )
                        token_mask.scatter_(-1, token_indices, True)
                        supported_state = (
                            x_base
                            + state_delta * token_mask.unsqueeze(-1)
                        )

                    if attention_ratio <= 0:
                        corrected_attention = base_attention
                    elif attention_ratio >= 1:
                        corrected_attention = block.attn(
                            block.norm1(supported_state), rope=rope
                        )
                    else:
                        q_new, k_new, v_new = self._attention_qkv(
                            block, supported_state, rope
                        )
                        dense_attention = attention_delta(
                            q_base_f,
                            k_base_f,
                            v_base_f,
                            q_new.float() - q_base_f,
                            k_new.float() - k_base_f,
                            v_new.float() - v_base_f,
                            scale=float(block.attn.scale),
                            backend="product_delta",
                            probability_mode="exact",
                        )
                        attention_mask, _ = select_attention_block_support(
                            dense_attention.base_probability,
                            keep_ratio=attention_ratio,
                            key_block_size=self.key_block,
                            query_block_size=self.query_block,
                            head_group_size=self.head_group,
                        )
                        sparse_attention = attention_delta(
                            q_base_f,
                            k_base_f,
                            v_base_f,
                            q_new.float() - q_base_f,
                            k_new.float() - k_base_f,
                            v_new.float() - v_base_f,
                            scale=float(block.attn.scale),
                            backend="product_delta",
                            probability_mode="exact",
                            support_mask=attention_mask,
                        )
                        corrected_raw = (
                            sparse_attention.corrected_output.transpose(1, 2)
                            .reshape(supported_state.shape)
                            .to(dtype=supported_state.dtype)
                        )
                        corrected_attention = block.attn.proj_drop(
                            block.attn.proj(corrected_raw)
                        )

                    state_attn = supported_state + block.ls1(
                        corrected_attention
                    )
                    if ffn_ratio <= 0:
                        corrected_ffn = base_ffn
                    elif ffn_ratio >= 1:
                        corrected_ffn = block.mlp(block.norm2(state_attn))
                    else:
                        corrected_norm2 = block.norm2(state_attn)
                        corrected_gate = block.mlp.w1(
                            corrected_norm2
                        ).float()
                        corrected_up = block.mlp.w2(
                            corrected_norm2
                        ).float()
                        exact_hidden_delta = (
                            F.silu(corrected_gate) * corrected_up
                            - base_hidden
                        )
                        channel_score = exact_hidden_delta.abs() * down_norm
                        ffn_mask = select_ffn_block_support(
                            channel_score,
                            keep_ratio=ffn_ratio,
                            channel_block_size=self.ffn_channel_block,
                            token_block_size=self.ffn_token_block,
                        )
                        corrected_hidden = (
                            base_hidden
                            + exact_hidden_delta.masked_fill(~ffn_mask, 0)
                        ).to(dtype=state_attn.dtype)
                        corrected_ffn = block.mlp.w3(corrected_hidden)
                    next_states[name] = state_attn + block.ls2(corrected_ffn)

                states = next_states
                x_base = x_base_attn + block.ls2(base_ffn)
                x_full = block(x_full, rope)

            full_tokens, full_pooled = self._normalized_backbone_features(x_full)
            logits_full = self._classifier_logits(x_full).float()
            outputs: dict[str, dict[str, object]] = {
                "l0_full": {"logits": logits_full}
            }
            comparison_states = {"l2_approx": x_base, **states}
            for name, state in comparison_states.items():
                tokens, pooled = self._normalized_backbone_features(state)
                logits = self._classifier_logits(state).float()
                outputs[name] = {
                    "logits": logits,
                    "token_feature_sums": tensor_sums(tokens, full_tokens),
                    "pooled_feature_sums": tensor_sums(pooled, full_pooled),
                    "logit_sums": tensor_sums(logits, logits_full),
                }
            return outputs


def load_images(
    args: argparse.Namespace,
    image_size: int,
    dataset_name: str,
) -> Iterable[tuple[torch.Tensor, int | None]]:
    if args.synthetic:
        generator = torch.Generator().manual_seed(31)
        yield torch.randint(
            0,
            256,
            (3, image_size, image_size),
            generator=generator,
            dtype=torch.uint8,
        ), None
        return
    if args.image:
        from PIL import Image

        for filename in args.image[: args.max_samples]:
            image = Image.open(filename).convert("RGB").resize((image_size, image_size))
            yield (
                torch.from_numpy(np.asarray(image, dtype=np.uint8).copy()).permute(2, 0, 1),
                None,
            )
        return
    if args.stratified_classes:
        if dataset_name != "imagenet-1k":
            raise ValueError("--stratified-classes currently supports ImageNet only")
        if args.stratified_classes < 1:
            raise ValueError("--stratified-classes must be positive")
        if args.sample_num_shards < 1:
            raise ValueError("--sample-num-shards must be positive")
        if not 0 <= args.sample_shard_index < args.sample_num_shards:
            raise ValueError(
                "--sample-shard-index must be in "
                f"[0, {args.sample_num_shards - 1}]"
            )

        from PIL import Image
        from torchvision.datasets import ImageFolder

        data_root = determine_data_root(dataset_name, args.data_root)
        dataset = ImageFolder(root=data_root)
        samples_by_class: dict[int, list[str]] = {}
        for filename, label in dataset.samples:
            samples_by_class.setdefault(label, []).append(filename)
        class_ids = np.asarray(sorted(samples_by_class), dtype=np.int64)
        if args.stratified_classes > len(class_ids):
            raise ValueError(
                f"Requested {args.stratified_classes} classes, "
                f"but only {len(class_ids)} are available"
            )

        generator = np.random.default_rng(args.sample_seed)
        selected_classes = sorted(
            int(value)
            for value in generator.choice(
                class_ids,
                size=args.stratified_classes,
                replace=False,
            )
        )
        selected_samples = []
        for label in selected_classes:
            candidates = samples_by_class[label]
            sample_index = int(generator.integers(0, len(candidates)))
            selected_samples.append((candidates[sample_index], label))
        selected_samples = selected_samples[
            args.sample_shard_index::args.sample_num_shards
        ]
        for filename, label in selected_samples[: args.max_samples]:
            image = Image.open(filename).convert("RGB").resize(
                (image_size, image_size)
            )
            yield (
                torch.from_numpy(np.asarray(image, dtype=np.uint8).copy())
                .permute(2, 0, 1),
                label,
            )
        return

    config, _ = load_config(args.config)
    data_root = determine_data_root(dataset_name, args.data_root)
    loader_kwargs = dict(config.dataset_kwargs)
    loader_kwargs["num_workers"] = args.num_workers
    from offload.mobile.dataset import get_dataset_loader

    dataset_loader = get_dataset_loader(
        dataset_name,
        data_root,
        batch_size=1,
        image_size=image_size,
        **loader_kwargs,
    )
    loader = dataset_loader.get_loader()
    emitted = 0
    for images, _labels in loader:
        for image, label in zip(images, _labels):
            yield image.to(torch.uint8), int(label.item())
            emitted += 1
            if emitted >= args.max_samples:
                return


def main() -> None:
    args = parse_args()
    config, _ = load_config(args.config)
    dataset_name = normalize_offload_dataset_name(args.dataset or config.dataset_name)
    image_h, image_w, _ = config.image_shape
    if image_h != image_w:
        raise ValueError("The current oracle expects a square token grid")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise RuntimeError("The DINOv3 ViT-7B oracle requires CUDA")
    layers = parse_layers(args.layers)
    target_layers = parse_layers(args.target_layers)
    support_ratios = [float(value) for value in args.support.split(",")]
    tail_epsilons = [float(value) for value in args.tail_epsilon.split(",")]
    sweep_ratios = [
        float(value) for value in args.sweep_ratios.split(",") if value.strip()
    ]

    oracle = Dinov3JacobianOracle(
        device=device,
        image_size=image_h,
        layers=layers,
        query_chunk=args.query_chunk,
        support_ratios=support_ratios,
        tail_epsilons=tail_epsilons,
        query_block=args.query_block,
        key_block=args.key_block,
        head_group=args.head_group,
        ffn_channel_block=args.ffn_channel_block,
        ffn_token_block=args.ffn_token_block,
    )
    samples = []
    images = load_images(args, image_h, dataset_name)
    for sample_index, (full_chw, label) in enumerate(
        tqdm(images, total=args.max_samples, desc="oracle")
    ):
        full_bhwc = full_chw.permute(1, 2, 0).contiguous().numpy()
        base_bhwc = low_resolution_canvas(full_bhwc, args.base_level)
        patch_size = (
            (config.patch_size, config.patch_size)
            if isinstance(config.patch_size, int)
            else tuple(config.patch_size)
        )
        canvases = progressive_canvases(
            base_bhwc,
            full_bhwc,
            patch_size=patch_size,
            num_groups=args.num_groups,
            group_strategy=args.group_strategy,
        )
        corrections = []
        for group_index, (previous, current) in enumerate(
            zip(canvases, canvases[1:]),
            start=1,
        ):
            rows = oracle.analyze_pair(
                torch.from_numpy(previous).permute(2, 0, 1).unsqueeze(0),
                torch.from_numpy(current).permute(2, 0, 1).unsqueeze(0),
            )
            corrections.append({
                "group": group_index,
                "changed_pixel_fraction": float(
                    np.any(current != previous, axis=-1).mean()
                ),
                "layers": rows,
            })
        dense_gate = (
            oracle.dense_propagation_gate(
                torch.from_numpy(base_bhwc).permute(2, 0, 1).unsqueeze(0),
                full_chw.unsqueeze(0),
            )
            if args.dense_gate
            else None
        )
        exact_support_sweep = (
            oracle.exact_support_feature_sweep(
                torch.from_numpy(base_bhwc).permute(2, 0, 1).unsqueeze(0),
                full_chw.unsqueeze(0),
                sweep_ratios,
            )
            if args.exact_support_sweep
            else None
        )
        exact_component_sweeps = (
            {
                component: oracle.exact_support_feature_sweep(
                    torch.from_numpy(base_bhwc).permute(2, 0, 1).unsqueeze(0),
                    full_chw.unsqueeze(0),
                    sweep_ratios,
                    component=component,
                )
                for component in (
                    "input_token",
                    "attention_edge",
                    "ffn_channel",
                )
            }
            if args.exact_component_sweep
            else None
        )
        exact_layer_component_sweeps = (
            oracle.exact_layer_component_sweep(
                torch.from_numpy(base_bhwc).permute(2, 0, 1).unsqueeze(0),
                full_chw.unsqueeze(0),
                sweep_ratios,
                target_layers,
            )
            if args.exact_layer_component_sweep
            else None
        )
        if dense_gate is not None and label is not None:
            dense_gate["label"] = label
            dense_gate["stock_correct"] = dense_gate["stock_top1"] == label
            for value in dense_gate.values():
                if isinstance(value, dict) and "predicted_top1" in value:
                    value["correct"] = value["predicted_top1"] == label
        samples.append({
            "sample_index": sample_index,
            "label": label,
            "corrections": corrections,
            "dense_gate": dense_gate,
            "exact_support_sweep": exact_support_sweep,
            "exact_component_sweeps": exact_component_sweeps,
            "exact_layer_component_sweeps": exact_layer_component_sweeps,
        })
        if sample_index + 1 >= args.max_samples:
            break

    payload = {
        "schema_version": 1,
        "manifest": {
            "git_commit": _git_output("rev-parse", "HEAD"),
            "git_branch": _git_output("branch", "--show-current"),
            "dirty": bool(_git_output("status", "--porcelain")),
            "device": torch.cuda.get_device_name(device),
            "device_capability": torch.cuda.get_device_capability(device),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "dtype": "bfloat16 model / float32 local oracle",
            "low_resolution_impl": "opencv_pyramid" if cv2 is not None else "torch_bicubic",
        },
        "experiment": {
            "dataset": dataset_name,
            "stratified_classes": args.stratified_classes,
            "sample_seed": args.sample_seed,
            "sample_shard_index": args.sample_shard_index,
            "sample_num_shards": args.sample_num_shards,
            "image_shape": list(config.image_shape),
            "base_level": args.base_level,
            "num_groups": args.num_groups,
            "group_strategy": args.group_strategy,
            "query_chunk": args.query_chunk,
            "layers": layers,
            "support_ratios": support_ratios,
            "tail_epsilons": tail_epsilons,
            "query_block": args.query_block,
            "key_block": args.key_block,
            "head_group": args.head_group,
            "ffn_channel_block": args.ffn_channel_block,
            "ffn_token_block": args.ffn_token_block,
            "dense_gate": args.dense_gate,
            "exact_support_sweep": args.exact_support_sweep,
            "exact_component_sweep": args.exact_component_sweep,
            "exact_layer_component_sweep": args.exact_layer_component_sweep,
            "target_layers": target_layers,
            "sweep_ratios": sweep_ratios,
        },
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
