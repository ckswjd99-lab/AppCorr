"""Where does CORRECT_FORWARD time actually go?

Motivation: `docs/memo/dinov3_nvfp4_speedup_gate.md` established that the five weight-GEMMs are only
~28% of the correction stage — quantizing them to NVFP4 buys 1.13x at ImageNet bs=128 and nothing at
all below that. The other ~70% is untouched by any quantization work, and nobody has measured what
it is. This script does.

It reconstructs one realistic correction round outside the client/server pipeline (approx pass to
populate the cache, then `correct_partial_token` over all 40 blocks) and reports:

  1. a torch.profiler breakdown by CUDA kernel, so the actual cost centres are named;
  2. a phase-level CUDA-event breakdown (gather / attention / GEMM / scatter / residual), so the
     shares line up with the ~28% GEMM figure;
  3. the count and cost of host syncs, since `_build_packed_query_state` calls `.item()` once per
     correction and that stalls the launch pipeline.

Run:
    PYTHONPATH="$PWD/appcorr/models:$PWD" python analysis/experiments/dinov3_correct_profile.py \
        --batch-size 128 --num-groups 4
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch

for _p in (Path(__file__).resolve().parents[2],):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

APPROX_KWARGS = {
    "appcorr_method": "partial_token",
    "server_pscore": "cls_attn_prob",
    "server_pscore_weight": 1.0,
    "debug": False,
}
CORRECT_KWARGS = {
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


def load_backbone(device):
    from appcorr.models.dinov3.hub.classifiers import dinov3_vit7b16_lc
    from offload.server.model.utils import load_weight_mmap

    model = dinov3_vit7b16_lc(pretrained=False, weights="IMAGENET1K", backbone_weights="LVD1689M")
    model.to(dtype=torch.bfloat16).to(device).eval().requires_grad_(False)
    path = "~/cjpark/weights/dinov3/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
    print(f"[load] {path}", flush=True)
    model.backbone.load_state_dict(load_weight_mmap(path), strict=True)
    return model.backbone


@torch.inference_mode()
def build_state(backbone, B, image_size, num_groups, device):
    """Run the approx pass to populate the cache, and build a one-group dindice."""
    img = torch.randn(B, 3, image_size, image_size, device=device, dtype=torch.bfloat16)
    x, (H, W) = backbone.prepare_tokens_with_masks(img)
    rope = backbone.rope_embed(H=H, W=W) if backbone.rope_embed is not None else None
    cache: dict = {}
    xa = x
    for i, blk in enumerate(backbone.blocks):
        xa, cache = blk.approx(xa, rope, cache, f"L{i}", **APPROX_KWARGS)

    N = x.shape[1]
    num_pre = backbone.n_storage_tokens + 1
    n_patch = N - num_pre
    # one grid group out of num_groups, mirroring imnet_interleaved_g4's grouping
    grp = torch.arange(n_patch, device=device)[::num_groups] + num_pre
    dindice = torch.cat([torch.arange(num_pre, device=device), grp]).unsqueeze(0).expand(B, -1).contiguous()
    print(f"[state] B={B} N={N} num_pre={num_pre} -> dindice {tuple(dindice.shape)}, "
          f"expected M = {B * dindice.shape[1]}", flush=True)
    return x, rope, cache, dindice


@torch.inference_mode()
def run_correct(backbone, x, rope, cache, dindice):
    xc = x
    for i, blk in enumerate(backbone.blocks):
        xc, cache = blk.correct(xc, dindice, rope, cache, f"L{i}", **CORRECT_KWARGS)
    return xc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--num-groups", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--rows", type=int, default=28, help="profiler rows to print")
    ap.add_argument("--server-pscore", default="cls_attn_prob",
                    help="cls_attn_prob = plan cache OFF (imnet); patch_attn_prob_layermean = plan cache ON (ade20k)")
    args = ap.parse_args()
    device = torch.device(args.device)
    APPROX_KWARGS["server_pscore"] = args.server_pscore
    CORRECT_KWARGS["server_pscore"] = args.server_pscore

    backbone = load_backbone(device)
    x, rope, cache0, dindice = build_state(
        backbone, args.batch_size, args.image_size, args.num_groups, device
    )

    import copy as _copy

    def fresh():
        return _copy.copy(cache0)

    # ---- warm up ----
    for _ in range(2):
        run_correct(backbone, x, rope, fresh(), dindice)
    torch.cuda.synchronize()

    # ---- total wall time ----
    ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
    ev0.record()
    for _ in range(3):
        run_correct(backbone, x, rope, fresh(), dindice)
    ev1.record()
    torch.cuda.synchronize()
    total_ms = ev0.elapsed_time(ev1) / 3
    print(f"\n[total] one full 40-block correction pass: {total_ms:.1f} ms\n", flush=True)

    # ---- profiler ----
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        run_correct(backbone, x, rope, fresh(), dindice)
        torch.cuda.synchronize()

    evts = [e for e in prof.key_averages() if e.self_device_time_total > 0]
    evts.sort(key=lambda e: e.self_device_time_total, reverse=True)
    tot = sum(e.self_device_time_total for e in evts)
    print(f"===== CUDA kernels by self time (total {tot/1e3:.1f} ms) =====")
    print(f"{'kernel':<62}{'self ms':>10}{'%':>8}{'calls':>8}")
    for e in evts[: args.rows]:
        ms = e.self_device_time_total / 1e3
        print(f"{e.key[:60]:<62}{ms:>10.2f}{100*ms*1e3/tot:>7.1f}%{e.count:>8}")

    # ---- coarse attribution ----
    BUCKETS = {
        "GEMM (Linear / _scaled_mm)": ("gemm", "sm90", "sm100", "cutlass", "ampere", "nvjet",
                                       "scaled_mm", "tensorop", "s16816", "gett"),
        "attention / SDPA": ("attention", "sdpa", "flash", "fmha", "softmax"),
        "elementwise / residual / norm": ("elementwise", "vectorized_elementwise", "add", "mul",
                                          "layer_norm", "native_layer_norm", "silu", "reduce_kernel"),
        "gather / scatter / index": ("index", "gather", "scatter", "take", "copy", "clone",
                                     "token_update", "masked"),
        "triton (appcorr kernels)": ("triton",),
    }
    agg = defaultdict(float)
    for e in evts:
        k = e.key.lower()
        for label, pats in BUCKETS.items():
            if any(p in k for p in pats):
                agg[label] += e.self_device_time_total / 1e3
                break
        else:
            agg["other"] += e.self_device_time_total / 1e3
    print(f"\n===== coarse attribution =====")
    for label, ms in sorted(agg.items(), key=lambda kv: -kv[1]):
        print(f"  {label:<34}{ms:>9.2f} ms{100*ms/(tot/1e3):>8.1f}%")

    # ---- host syncs ----
    cpu = [e for e in prof.key_averages()
           if any(s in e.key.lower() for s in ("item", "synchronize", "memcpydtoh", "nonzero"))]
    if cpu:
        print(f"\n===== host syncs / D2H (launch-pipeline stalls) =====")
        for e in sorted(cpu, key=lambda e: -e.cpu_time_total)[:8]:
            print(f"  {e.key[:52]:<54}{e.cpu_time_total/1e3:>9.2f} ms cpu{e.count:>7} calls")


if __name__ == "__main__":
    main()
