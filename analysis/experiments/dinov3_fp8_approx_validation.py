#!/usr/bin/env python3
"""Validate and benchmark stage-selective DINOv3 FP8/FP4 approximate inference."""

from __future__ import annotations

import argparse
import statistics
from types import SimpleNamespace

import torch
from torch import nn
from torch.profiler import ProfilerActivity, profile

from appcorr.models.dinov3.layers.block import SelfAttentionBlock
from offload.common.protocol import ExperimentConfig
from offload.server.model.dinov3_precision import DINOv3ApproxPrecisionController


def _make_block(
    dim: int,
    num_heads: int,
    *,
    ffn_ratio: float = 4.0,
) -> SelfAttentionBlock:
    block = SelfAttentionBlock(
        dim,
        num_heads,
        ffn_ratio=ffn_ratio,
        qkv_bias=True,
        mask_k_bias=True,
        init_values=None,
    ).to("cuda", dtype=torch.bfloat16).eval()
    qkv = block.attn.qkv
    if qkv.bias is not None:
        qkv.bias_mask.fill_(1)
        qkv.bias_mask[qkv.out_features // 3 : 2 * qkv.out_features // 3].fill_(0)
    return block


def _rope(num_tokens: int, num_pretokens: int, head_dim: int):
    num_patches = num_tokens - num_pretokens
    return (
        torch.zeros(num_patches, head_dim, device="cuda"),
        torch.ones(num_patches, head_dim, device="cuda"),
    )


def _partial_token_kwargs():
    return {
        "appcorr_method": "partial_token",
        "server_pscore": "cls_attn_prob",
        "server_pscore_weight": 1.0,
        "debug": False,
    }


def _partial_token_correct_kwargs():
    return {
        "appcorr_method": "partial_token",
        "token_keep_ratio": 1.0,
        "token_keep_thres": None,
        "mobile_pscore": "none",
        "mobile_pscore_weight": 0.0,
        "server_pscore": "cls_attn_prob",
        "server_pscore_weight": 1.0,
        "pscore_fusion": "add",
        "sdpa_query_bucket_size": 0,
        "debug": False,
    }


def _group_plans(
    block: SelfAttentionBlock,
    batch_size: int,
    num_tokens: int,
    num_pretokens: int,
    num_groups: int,
):
    all_tokens = torch.arange(num_tokens, device="cuda")
    prefix = all_tokens[:num_pretokens].view(1, -1).expand(batch_size, -1)
    patch_groups = torch.tensor_split(all_tokens[num_pretokens:], num_groups)
    plans = {}
    query_states = {}
    for group_id, patch_group in enumerate(patch_groups):
        patches = patch_group.view(1, -1).expand(batch_size, -1)
        full = torch.cat((prefix, patches), dim=1)
        keep_mask = torch.ones_like(patches, dtype=torch.bool)
        _, query_state = block._build_packed_query_state(prefix, patches, keep_mask)
        plans[group_id] = SimpleNamespace(
            num_pretokens=num_pretokens,
            prefix_dindice=prefix,
            group_patch_dindice=patches,
            group_patch_keep_local_idx=torch.arange(
                patches.shape[1],
                device="cuda",
            ).view(1, -1).expand(batch_size, -1),
            full_dindice=full,
        )
        query_states[group_id] = query_state
    return plans, query_states


def validate_config() -> None:
    for precision in ("bf16", "fp8", "fp4", "auto"):
        config = ExperimentConfig(precision=precision)
        assert config.precision == precision
        assert config.fp8_auto_min_rows == 3072

    for invalid in (
        {"precision": "float8"},
        {"fp8_auto_min_rows": 0},
        {"fp8_auto_min_rows": -1},
    ):
        try:
            ExperimentConfig(**invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid precision config was accepted: {invalid}")


def validate_cuda_paths(dim: int = 1024, num_heads: int = 16) -> None:
    torch.manual_seed(0)
    block = _make_block(dim, num_heads)
    controller = DINOv3ApproxPrecisionController(
        nn.ModuleList([block]),
        precision="fp8",
        auto_min_rows=3072,
        device=torch.device("cuda"),
    )
    fp8_linears = list(controller.iter_fp8_linears())
    assert len(fp8_linears) == 5
    assert all(type(module.weight).__name__ == "Float8Tensor" for _, module in fp8_linears)
    assert all(
        type(module.weight).__name__ == "Parameter"
        for module in block.modules()
        if isinstance(module, nn.Linear)
    )
    assert controller._effective_precision(3071) == "fp8"
    controller.precision = "auto"
    assert controller._effective_precision(3071) == "bf16"
    assert controller._effective_precision(3072) == "fp8"
    controller.precision = "fp8"

    batch_size, num_tokens, num_pretokens = 1, 64, 5
    x_low = torch.randn(batch_size, num_tokens, dim, device="cuda", dtype=torch.bfloat16)
    x_high = torch.randn_like(x_low)
    rope = _rope(num_tokens, num_pretokens, dim // num_heads)
    dindice = torch.arange(num_tokens, device="cuda").view(1, -1)

    profile_kwargs = {
        "appcorr_method": "partial_token",
        "server_pscore": "none",
        "server_pscore_weight": 0.0,
        "debug": False,
    }
    with torch.inference_mode():
        controller.run_block(0, x_low, rope, {}, "profile", **profile_kwargs)
        torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as fp8_prof:
        with torch.inference_mode():
            controller.run_block(0, x_low, rope, {}, "profile", **profile_kwargs)
            torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as bf16_prof:
        with torch.inference_mode():
            block.approx(x_low, rope, {}, "profile", **profile_kwargs)
            torch.cuda.synchronize()
    assert any("scaled_mm" in event.key for event in fp8_prof.key_averages())
    assert not any("scaled_mm" in event.key for event in bf16_prof.key_averages())

    with torch.inference_mode():
        _, bf16_cache = block.approx(
            x_low,
            rope,
            {},
            "layer0",
            **_partial_token_kwargs(),
        )
        controller.begin_event()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fp8_approx_output, fp8_cache = controller.run_block(
                0,
                x_low,
                rope,
                {},
                "layer0",
                **_partial_token_kwargs(),
            )
        assert fp8_approx_output.dtype == torch.bfloat16
        assert fp8_cache["layer0_kv"].dtype == torch.bfloat16
        assert fp8_cache["layer0_blocks_out_sum"].dtype == torch.bfloat16
        mixed_output, _ = block.correct(
            x_high,
            dindice,
            rope,
            fp8_cache,
            "layer0",
            **_partial_token_correct_kwargs(),
        )
        bf16_output, _ = block.correct(
            x_high,
            dindice,
            rope,
            bf16_cache,
            "layer0",
            **_partial_token_correct_kwargs(),
        )
        torch.cuda.synchronize()

    assert torch.equal(mixed_output, bf16_output)
    assert torch.isfinite(mixed_output).all()
    metadata = controller.event_metadata()
    assert metadata["approx_precision_effective"] == "fp8"
    assert metadata["approx_precision_rows"] == [batch_size * num_tokens]

    plans, query_states = _group_plans(
        block,
        batch_size,
        num_tokens,
        num_pretokens,
        num_groups=4,
    )
    partial_channel_kwargs = {
        "appcorr_method": "partial_channel",
        "attn_cache_candidates": {
            group_id: plan.full_dindice
            for group_id, plan in plans.items()
        },
        "group_plans": plans,
        "attn_col_alive_ratio": 1.0,
        "debug": False,
    }
    with torch.inference_mode():
        _, bf16_cache = block.approx(
            x_low,
            rope,
            {},
            "layer0",
            **partial_channel_kwargs,
        )
        _, fp8_cache = controller.run_block(
            0,
            x_low,
            rope,
            {},
            "layer0",
            **partial_channel_kwargs,
        )
        for group_id, plan in plans.items():
            correct_kwargs = {
                "appcorr_method": "partial_channel",
                "fixed_query_state": query_states[group_id],
                "group_plan": plan,
                "attn_col_alive_ratio": 1.0,
                "attn_cache_key": group_id,
            }
            mixed_output, fp8_cache = block.correct(
                x_high,
                plan.full_dindice,
                rope,
                fp8_cache,
                "layer0",
                **correct_kwargs,
            )
            bf16_output, bf16_cache = block.correct(
                x_high,
                plan.full_dindice,
                rope,
                bf16_cache,
                "layer0",
                **correct_kwargs,
            )
            assert torch.isfinite(mixed_output).all()
        torch.cuda.synchronize()

    assert torch.equal(mixed_output, bf16_output)
    assert torch.isfinite(mixed_output).all()


def validate_fp4_cuda_path(dim: int = 1024, num_heads: int = 16) -> None:
    if torch.cuda.get_device_capability() < (10, 0):
        print("SKIP: FP4 validation requires SM100+")
        return

    torch.manual_seed(0)
    block = _make_block(dim, num_heads, ffn_ratio=3.0)
    controller = DINOv3ApproxPrecisionController(
        nn.ModuleList([block]),
        precision="fp4",
        auto_min_rows=3072,
        device=torch.device("cuda"),
    )
    fp4_linears = list(controller.iter_fp4_linears())
    assert len(fp4_linears) == 5
    assert all(
        type(module.weight).__name__ == "NVFP4Tensor"
        for _, module in fp4_linears
    )
    assert controller._effective_precision(1) == "fp4"

    batch_size, num_tokens, num_pretokens = 1, 64, 5
    x_low = torch.randn(
        batch_size,
        num_tokens,
        dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    x_high = torch.randn_like(x_low)
    rope = _rope(num_tokens, num_pretokens, dim // num_heads)
    dindice = torch.arange(num_tokens, device="cuda").view(1, -1)

    with torch.inference_mode():
        controller.run_block(
            0,
            x_low,
            rope,
            {},
            "profile",
            **_partial_token_kwargs(),
        )
        torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as fp4_prof:
        with torch.inference_mode():
            fp4_output, fp4_cache = controller.run_block(
                0,
                x_low,
                rope,
                {},
                "layer0",
                **_partial_token_kwargs(),
            )
            torch.cuda.synchronize()
    assert any("scaled_mm" in event.key for event in fp4_prof.key_averages())
    assert fp4_output.dtype == torch.bfloat16
    assert fp4_cache["layer0_kv"].dtype == torch.bfloat16

    with torch.inference_mode():
        mixed_output, _ = block.correct(
            x_high,
            dindice,
            rope,
            fp4_cache,
            "layer0",
            **_partial_token_correct_kwargs(),
        )
        _, bf16_cache = block.approx(
            x_low,
            rope,
            {},
            "layer0",
            **_partial_token_kwargs(),
        )
        bf16_output, _ = block.correct(
            x_high,
            dindice,
            rope,
            bf16_cache,
            "layer0",
            **_partial_token_correct_kwargs(),
        )
        torch.cuda.synchronize()

    assert torch.equal(mixed_output, bf16_output)
    assert torch.isfinite(mixed_output).all()
    metadata = controller.event_metadata()
    assert metadata["approx_precision_effective"] == "fp4"
    assert metadata["approx_fp4_sources"] == 1


def _measure_ms(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def benchmark(args) -> None:
    block = _make_block(args.dim, args.num_heads)
    controller = DINOv3ApproxPrecisionController(
        nn.ModuleList([block]),
        precision="fp8",
        auto_min_rows=args.auto_min_rows,
        device=torch.device("cuda"),
    )
    print(
        "allocated_gib",
        round(torch.cuda.memory_allocated() / 1024**3, 3),
        "peak_gib",
        round(torch.cuda.max_memory_allocated() / 1024**3, 3),
    )
    cases = ((224, 1), (224, 16), (448, 4), (896, 1))
    print("image batch rows bf16_ms fp8_ms speedup auto")
    for image_size, batch_size in cases:
        num_tokens = (image_size // 16) ** 2 + 5
        rows = batch_size * num_tokens
        x = torch.randn(
            batch_size,
            num_tokens,
            args.dim,
            device="cuda",
            dtype=torch.bfloat16,
        )
        rope = _rope(num_tokens, 5, args.dim // args.num_heads)
        approx_kwargs = {
            "appcorr_method": "partial_token",
            "server_pscore": "none",
            "server_pscore_weight": 0.0,
            "debug": False,
        }

        def bf16_call():
            with torch.inference_mode():
                block.approx(x, rope, {}, "layer0", **approx_kwargs)

        def fp8_call():
            with torch.inference_mode():
                controller.run_block(0, x, rope, {}, "layer0", **approx_kwargs)

        bf16_ms = _measure_ms(bf16_call, args.warmup, args.iterations)
        fp8_ms = _measure_ms(fp8_call, args.warmup, args.iterations)
        effective = "fp8" if rows >= args.auto_min_rows else "bf16"
        print(
            f"{image_size:>5} {batch_size:>5} {rows:>5} "
            f"{bf16_ms:>8.3f} {fp8_ms:>7.3f} {bf16_ms / fp8_ms:>7.3f}x {effective}"
        )
        del x


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--dim", type=int, default=4096)
    parser.add_argument("--num-heads", type=int, default=32)
    parser.add_argument("--auto-min-rows", type=int, default=3072)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    validate_config()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for FP8/FP4 validation")
    validate_cuda_paths()
    validate_fp4_cuda_path()
    print(
        "PASS: config, FP8/FP4 weights, auto routing, and full-correction parity"
    )
    if args.benchmark:
        benchmark(args)


if __name__ == "__main__":
    main()
