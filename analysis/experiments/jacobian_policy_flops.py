#!/usr/bin/env python3
"""Estimate sparse-correction FLOPs for generated Jacobian policies.

This models the intended structured kernels, not the dense PyTorch accuracy
oracle. One multiply-accumulate is counted as two FLOPs.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy-json",
        type=Path,
        default=Path(
            "analysis/experiments/results/"
            "jacobian_pruning_policy_token50_attnffn.json"
        ),
    )
    parser.add_argument("--tokens", type=int, default=261)
    parser.add_argument("--special-tokens", type=int, default=5)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--ffn-hidden-size", type=int, default=8192)
    parser.add_argument("--layers", type=int, default=40)
    parser.add_argument("--attention-key-block", type=int, default=16)
    parser.add_argument("--ffn-channel-block", type=int, default=128)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "analysis/experiments/results/"
            "jacobian_policy_token50_special_attnffn_flops.json"
        ),
    )
    return parser.parse_args()


def structured_keep(keep: float, blocks: int) -> float:
    """Return padded block workload keep after top-k count rounding."""

    if keep <= 0:
        return 0.0
    if keep >= 1:
        return 1.0
    return math.ceil(blocks * keep) / blocks


def estimate(
    payload: dict,
    *,
    tokens: int,
    special_tokens: int,
    hidden_size: int,
    ffn_hidden_size: int,
    layers: int,
    attention_key_block: int,
    ffn_channel_block: int,
) -> dict:
    if not 0 <= special_tokens <= tokens:
        raise ValueError("special_tokens must be in [0, tokens]")
    patch_tokens = tokens - special_tokens
    attention_key_blocks = math.ceil(tokens / attention_key_block)
    ffn_channel_blocks = math.ceil(
        ffn_hidden_size / ffn_channel_block
    )

    # The same decomposition is used by fit_jacobian_pruning_policy.py's
    # vit7b_flops mode: four D->D projections, two N^2*D attention products,
    # and three N*D*M SwiGLU projections. Convert MACs to FLOPs with x2.
    component_per_layer = {
        "input_token_projection": (
            2 * 4 * tokens * hidden_size * hidden_size
        ),
        "attention_products": (
            2 * 2 * tokens * tokens * hidden_size
        ),
        "ffn_projections": (
            2 * 3 * tokens * hidden_size * ffn_hidden_size
        ),
    }
    approx_per_layer = sum(component_per_layer.values())
    approx_total = approx_per_layer * layers
    one_token_projection = 2 * tokens * hidden_size * hidden_size
    one_attention_product = 2 * tokens * tokens * hidden_size
    one_ffn_projection = 2 * tokens * hidden_size * ffn_hidden_size

    policies = {}
    dense_oracle_policy_flops = {}
    for policy in payload["policies"]:
        schedule = policy["schedule"]
        if len(schedule) != layers:
            raise ValueError(
                f"Policy has {len(schedule)} layers; expected {layers}"
            )
        nominal_components = {
            component: 0.0 for component in component_per_layer
        }
        structured_components = {
            component: 0.0 for component in component_per_layer
        }
        layer_rows = []
        for row in schedule:
            token_keep = float(row["input_token_keep"])
            attention_keep = float(row["attention_edge_keep"])
            ffn_keep = float(row["ffn_channel_keep"])

            nominal_token_keep = (
                special_tokens + patch_tokens * token_keep
            ) / tokens
            structured_token_keep = (
                special_tokens + math.ceil(patch_tokens * token_keep)
            ) / tokens
            structured_attention_keep = structured_keep(
                attention_keep,
                attention_key_blocks,
            )
            structured_ffn_keep = structured_keep(
                ffn_keep,
                ffn_channel_blocks,
            )
            nominal_keeps = {
                "input_token_projection": nominal_token_keep,
                "attention_products": attention_keep,
                "ffn_projections": ffn_keep,
            }
            structured_keeps = {
                "input_token_projection": structured_token_keep,
                "attention_products": structured_attention_keep,
                "ffn_projections": structured_ffn_keep,
            }
            for component, full_flops in component_per_layer.items():
                nominal_components[component] += (
                    full_flops * nominal_keeps[component]
                )
                structured_components[component] += (
                    full_flops * structured_keeps[component]
                )
            layer_rows.append({
                "layer": int(row["layer"]),
                "requested_keep": {
                    "patch_tokens": token_keep,
                    "attention_edges": attention_keep,
                    "ffn_channels": ffn_keep,
                },
                "structured_workload_keep": {
                    "all_token_projection": structured_token_keep,
                    "attention_key_blocks": structured_attention_keep,
                    "ffn_channel_blocks": structured_ffn_keep,
                },
            })

        nominal_total = sum(nominal_components.values())
        structured_total = sum(structured_components.values())
        name = f"attn_ffn_budget_{policy['target_pruning_rate']:.0%}"
        dense_projection_flops = 0
        dense_attention_product_flops = 0
        dense_ffn_flops = 0
        dense_path_counts = {
            "attention_zero_layers": 0,
            "attention_intermediate_layers": 0,
            "attention_full_layers": 0,
            "ffn_zero_layers": 0,
            "ffn_dense_layers": 0,
        }
        for row in schedule:
            attention_keep = float(row["attention_edge_keep"])
            ffn_keep = float(row["ffn_channel_keep"])
            if attention_keep <= 0:
                dense_path_counts["attention_zero_layers"] += 1
            elif attention_keep >= 1:
                dense_path_counts["attention_full_layers"] += 1
                dense_projection_flops += 4 * one_token_projection
                dense_attention_product_flops += (
                    2 * one_attention_product
                )
            else:
                dense_path_counts["attention_intermediate_layers"] += 1
                dense_projection_flops += 4 * one_token_projection
                # exact_policy_classification_batch calls attention_delta
                # twice. Each product-delta call performs four QK-like and
                # six probability-value products.
                dense_attention_product_flops += (
                    20 * one_attention_product
                )
            if ffn_keep <= 0:
                dense_path_counts["ffn_zero_layers"] += 1
            else:
                dense_path_counts["ffn_dense_layers"] += 1
                # Intermediate support is applied after dense w1/w2 and the
                # masked tensor still goes through a dense w3.
                dense_ffn_flops += 3 * one_ffn_projection
        dense_policy_total = (
            dense_projection_flops
            + dense_attention_product_flops
            + dense_ffn_flops
        )
        dense_oracle_policy_flops[name] = {
            "leading_gemm_flops": dense_policy_total,
            "over_approx": dense_policy_total / approx_total,
            "projection_flops": dense_projection_flops,
            "attention_product_flops": dense_attention_product_flops,
            "ffn_flops": dense_ffn_flops,
            "path_counts": dense_path_counts,
        }
        policies[name] = {
            "target_pruning_rate": policy["target_pruning_rate"],
            "requested_overall_equal_work_pruning": policy[
                "overall_achieved_pruning_rate"
            ],
            "nominal_sparse_correction_flops": nominal_total,
            "nominal_correction_over_approx": nominal_total / approx_total,
            "structured_sparse_correction_flops": structured_total,
            "structured_correction_over_approx": (
                structured_total / approx_total
            ),
            "approx_plus_correction_over_approx": (
                1 + structured_total / approx_total
            ),
            "structured_component_flops": structured_components,
            "structured_component_over_approx": {
                component: flops / approx_total
                for component, flops in structured_components.items()
            },
            "structured_component_share_of_correction": {
                component: flops / structured_total
                for component, flops in structured_components.items()
            },
            "layers": layer_rows,
        }

    # Besides a normal base/approx forward, the oracle separately materializes
    # base QKV and base gate/up tensors for scoring. It also advances a stock
    # L0 reference path. Both are shared when several policies run together.
    dense_oracle_shared_support = layers * (
        3 * one_token_projection + 2 * one_ffn_projection
    )
    dense_oracle_all_policies = (
        approx_total
        + dense_oracle_shared_support
        + approx_total
        + sum(
            row["leading_gemm_flops"]
            for row in dense_oracle_policy_flops.values()
        )
    )

    return {
        "schema_version": 1,
        "scope": "DINOv3 ViT-7B transformer backbone, batch 1, one L2-to-L0 correction",
        "counting": "one multiply-accumulate equals two FLOPs",
        "model": {
            "tokens": tokens,
            "patch_tokens": patch_tokens,
            "always_corrected_special_tokens": special_tokens,
            "hidden_size": hidden_size,
            "ffn_hidden_size": ffn_hidden_size,
            "layers": layers,
            "attention_key_block": attention_key_block,
            "attention_key_blocks": attention_key_blocks,
            "ffn_channel_block": ffn_channel_block,
            "ffn_channel_blocks": ffn_channel_blocks,
        },
        "approx": {
            "component_flops_per_layer": component_per_layer,
            "flops_per_layer": approx_per_layer,
            "flops_total": approx_total,
            "component_share": {
                component: flops / approx_per_layer
                for component, flops in component_per_layer.items()
            },
        },
        "policies": policies,
        "assumptions": [
            (
                "Q, K, V, and attention-output projection work scales "
                "linearly with effective token support, matching the "
                "repository vit7b_flops component decomposition."
            ),
            (
                "Product-delta reuses cached base SV and evaluates one "
                "corrected QK and probability-value product on selected "
                "attention blocks."
            ),
            (
                "Gate, up, and down projection work scales linearly with "
                "selected FFN channel blocks."
            ),
        ],
        "dense_accuracy_oracle": {
            "description": (
                "Leading-GEMM estimate for exact_policy_classification_batch; "
                "this is the path used to obtain accuracy, not a sparse kernel."
            ),
            "normal_approx_flops": approx_total,
            "shared_support_producer_flops": dense_oracle_shared_support,
            "stock_l0_reference_flops": approx_total,
            "policy_branches": dense_oracle_policy_flops,
            "all_policy_eval_flops": dense_oracle_all_policies,
            "all_policy_eval_over_approx": (
                dense_oracle_all_policies / approx_total
            ),
        },
        "excluded": [
            "patch embedding and classifier head",
            "layer normalization, softmax, SiLU, RoPE, residual adds",
            "selector, top-k, packing, scatter, and memory traffic",
            "padding overhead in query/token dimensions",
        ],
        "caveat": (
            "The ImageNet evaluator computes dense tensors and masks them. "
            "These are intended sparse-kernel FLOPs, not FLOPs actually "
            "executed by that accuracy oracle."
        ),
    }


def main() -> None:
    args = parse_args()
    payload = json.loads(args.policy_json.read_text(encoding="utf-8"))
    result = estimate(
        payload,
        tokens=args.tokens,
        special_tokens=args.special_tokens,
        hidden_size=args.hidden_size,
        ffn_hidden_size=args.ffn_hidden_size,
        layers=args.layers,
        attention_key_block=args.attention_key_block,
        ffn_channel_block=args.ffn_channel_block,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    approx = result["approx"]["flops_total"]
    print(f"approx={approx / 1e12:.6f} TFLOPs")
    for name, policy in result["policies"].items():
        correction = policy["structured_sparse_correction_flops"]
        ratio = policy["structured_correction_over_approx"]
        dense_ratio = result["dense_accuracy_oracle"]["policy_branches"][
            name
        ]["over_approx"]
        print(
            f"{name}: correction={correction / 1e12:.6f} TFLOPs "
            f"({ratio:.2%} of approx), "
            f"approx+correction={1 + ratio:.2%}, "
            f"dense-oracle-branch={dense_ratio:.2%}"
        )
    dense_oracle = result["dense_accuracy_oracle"]
    print(
        "dense accuracy evaluator, all policies="
        f"{dense_oracle['all_policy_eval_flops'] / 1e12:.6f} TFLOPs "
        f"({dense_oracle['all_policy_eval_over_approx']:.2f}x approx)"
    )
    print(f"[saved] {args.output}")


if __name__ == "__main__":
    main()
