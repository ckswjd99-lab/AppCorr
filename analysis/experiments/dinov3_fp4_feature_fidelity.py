"""Feature fidelity of NVFP4 under an EXACT approx/correct decomposition of the DINOv3 block.

Motivation. Accuracy metrics were too coarse to separate FP4 placements: at full-2000 ADE20K both
"FP4 on correction only" and "FP4 on the whole forward" landed inside a ~0.2pp noise floor
(docs/memo/dinov3_correct_low_precision_status.md). This script drops to the representation itself,
and — more importantly — replaces the current correction mechanism with one whose *arithmetic* is
exact, so that the only error left to measure is the quantization.

The decomposition. Carry every activation as a pair `(a, d)` whose true value is `a + d`
(`a` = approx/base path, `d` = correction path), and push it through the block op by op:

  linear  W x + b :  a' = W a + b            d' = W d          <- bias cancels in the difference,
                                                                  so the correction GEMM only ever
                                                                  sees the small delta
  nonlinear   g   :  a' = g(a)               d' = g(a + d) - g(a)   <- recomputed exactly

Both lines are identities, so the pair telescopes: `a' + d' = f(a + d)` exactly, for *any* `a`.
There is no Taylor truncation. The only requirement is that the a-path and d-path use the same
weights — which is precisely what breaks when the two paths are given different precisions, and the
size of that break is what this script measures.

Why this should favour FP4. The five weight-GEMMs (qkv/proj/w1/w2/w3) are the expensive ops and the
only ones quantized. In the correction path they consume `d`, not `x`, so the absolute quantization
error scales with `|d|` instead of `|x|`. Non-linearities (LayerNorm, softmax/attention core,
SiLU⊙product) are recomputed in BF16 and contribute no approximation error at all.

Conditions (reference = `bf16_full`, the plain BF16 forward):
  fp4_full          plain forward, all 5 Linears NVFP4 — the naive "quantize the absolute
                    activations" baseline.
  exact_bf16_bf16   decomposition, both paths BF16   -> must reproduce the reference (control).
  exact_bf16_fp4    BF16 base, NVFP4 correction GEMMs -> FP4 applied only to the delta.
  exact_fp4_bf16    NVFP4 base, BF16 correction GEMMs -> FP4 applied only to the base.
  exact_fp4_fp4     both NVFP4                        -> should match `fp4_full` (control).

Metrics per image over patch tokens (cls/storage excluded): relative L2 ||f-ref||/||ref|| and mean
per-token cosine similarity.

Run:
    PYTHONPATH="$PWD/appcorr/models:$PWD" python analysis/experiments/dinov3_fp4_feature_fidelity.py \
        --dataset imagenet-1k --fraction 0.1
    PYTHONPATH="$PWD/appcorr/models:$PWD" python analysis/experiments/dinov3_fp4_feature_fidelity.py \
        --dataset coco2017 --fraction 0.1
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

for _p in (Path(__file__).resolve().parents[2],):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from offload.server.model.dinov3_precision import _eligible_fp4_linears  # noqa: E402


# --------------------------------------------------------------------------------------
# model / data
# --------------------------------------------------------------------------------------

def quantize_blocks_fp4(blocks: nn.ModuleList) -> nn.ModuleList:
    """Deep-copy the blocks and convert their 5 eligible Linears to NVFP4.

    Same settings as DINOv3ApproxPrecisionController._initialize_fp4 minus torch.compile: eager and
    accurate mode, because the fused Triton NVFP4 kernel needs the uninstalled MSLK package and this
    script measures numerics, not latency.
    """
    from torchao.prototype.mx_formats import NVFP4DynamicActivationNVFP4WeightConfig
    from torchao.quantization import quantize_

    cfg = NVFP4DynamicActivationNVFP4WeightConfig(
        use_triton_kernel=False, use_dynamic_per_tensor_scale=True
    )
    out = nn.ModuleList()
    for block in blocks:
        qb = copy.deepcopy(block).to(dtype=torch.bfloat16).eval().requires_grad_(False)
        names = {n for n, _ in _eligible_fp4_linears(qb)}
        quantize_(qb, cfg, filter_fn=lambda m, fqn, _n=names: isinstance(m, nn.Linear) and fqn in _n)
        converted = [
            n for n, m in qb.named_modules()
            if n in names and type(m.weight).__name__ == "NVFP4Tensor"
        ]
        if len(converted) != len(names):
            raise RuntimeError(f"NVFP4 conversion incomplete: {converted} vs {sorted(names)}")
        out.append(qb)
    return out


def load_backbone(device: torch.device) -> nn.Module:
    from appcorr.models.dinov3.hub.classifiers import dinov3_vit7b16_lc
    from offload.server.model.utils import load_weight_mmap

    model = dinov3_vit7b16_lc(pretrained=False, weights="IMAGENET1K", backbone_weights="LVD1689M")
    model.to(dtype=torch.bfloat16).to(device).eval().requires_grad_(False)
    path = "~/cjpark/weights/dinov3/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth"
    print(f"[load] backbone weights: {path}", flush=True)
    model.backbone.load_state_dict(load_weight_mmap(path), strict=True)
    return model.backbone


def build_loader(dataset: str, batch_size: int, image_size: int):
    from offload.mobile.dataset import get_dataset_loader

    root = {"imagenet-1k": "~/data/imagenet_val", "coco2017": None}[dataset]
    kwargs = {"image_size": image_size, "num_workers": 4}
    if dataset == "coco2017":
        kwargs.update({"num_workers": 0, "download_if_necessary": False})
    return get_dataset_loader(dataset, root, batch_size, **kwargs).get_loader()


def to_model_input(images, device: torch.device, image_size: int) -> torch.Tensor:
    if isinstance(images, (list, tuple)):
        images = torch.stack([i if torch.is_tensor(i) else torch.as_tensor(i) for i in images])
    x = images.to(device)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.shape[-1] in (1, 3) and x.shape[1] not in (1, 3):
        x = x.permute(0, 3, 1, 2)
    x = x.float()
    if x.max() > 1.5:
        x = x / 255.0
    x = F.interpolate(x, size=(image_size, image_size), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
    return ((x - mean) / std).to(torch.bfloat16)


def l2_base(x: torch.Tensor) -> torch.Tensor:
    """AppCorr's L2 approximation: downsample by 4, upsample back to the same canvas."""
    h, w = x.shape[-2:]
    small = F.interpolate(x.float(), size=(h // 4, w // 4), mode="area")
    return F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False).to(x.dtype)


# --------------------------------------------------------------------------------------
# exact approx/correct decomposition
# --------------------------------------------------------------------------------------

def _lin_delta(mod: nn.Linear, d: torch.Tensor, book: torch.dtype) -> torch.Tensor:
    """W @ d, bias excluded — the exact differential of an affine layer.

    The GEMM itself always runs in the module's own precision (BF16 or NVFP4) — that is the
    experimental variable. `book` only controls the dtype the result is carried in.
    """
    return F.linear(d.to(torch.bfloat16), mod.weight, None).to(book)


def _lin_base(mod: nn.Linear, a: torch.Tensor, book: torch.dtype) -> torch.Tensor:
    return mod(a.to(torch.bfloat16)).to(book)


@torch.inference_mode()
def exact_block(blk_a, blk_d, a: torch.Tensor, d: torch.Tensor, rope, book: torch.dtype):
    """One block of the (a, d) decomposition.

    `blk_a` supplies the weights for the base path, `blk_d` for the correction path; passing the
    BF16 block for one and the NVFP4 block for the other is how a precision is assigned per path.
    Non-linearities carry no weights and are recomputed exactly.

    `book` is the dtype the (a, d) pair is carried in between ops. With float32 the decomposition's
    own rounding — the telescoping is exact only in exact arithmetic — stops contaminating the
    measurement, leaving the five weight-GEMMs as the only source of error.
    """
    attn_a, attn_d = blk_a.attn, blk_d.attn
    mlp_a, mlp_d = blk_a.mlp, blk_d.mlp

    def nl(fn, x):  # nonlinear op evaluated in the bookkeeping dtype
        return fn(x.to(torch.bfloat16)).to(book)

    # --- norm1 (nonlinear: exact difference) ---
    a_h = nl(blk_a.norm1, a)
    d_h = nl(blk_a.norm1, a + d) - a_h

    # --- qkv (linear) ---
    a_qkv = _lin_base(attn_a.qkv, a_h, book)
    d_qkv = _lin_delta(attn_d.qkv, d_h, book)

    # --- attention core: softmax + two activation-activation matmuls (nonlinear) ---
    core = lambda t: attn_a.compute_attention(t.to(torch.bfloat16), rope=rope).to(book)
    a_att = core(a_qkv)
    d_att = core(a_qkv + d_qkv) - a_att

    # --- output projection (linear) ---
    a_u = _lin_base(attn_a.proj, a_att, book)
    d_u = _lin_delta(attn_d.proj, d_att, book)

    # --- residual + LayerScale (linear, no bias) ---
    g1 = blk_a.ls1.gamma.to(book) if hasattr(blk_a.ls1, "gamma") else 1.0
    a_x = a + a_u * g1
    d_x = d + d_u * g1

    # --- norm2 (nonlinear) ---
    a_h2 = nl(blk_a.norm2, a_x)
    d_h2 = nl(blk_a.norm2, a_x + d_x) - a_h2

    # --- SwiGLU: w1 = W_g, w2 = W_u, w3 = W_d ---
    a_z1 = _lin_base(mlp_a.w1, a_h2, book)
    a_z2 = _lin_base(mlp_a.w2, a_h2, book)
    d_z1 = _lin_delta(mlp_d.w1, d_h2, book)
    d_z2 = _lin_delta(mlp_d.w2, d_h2, book)

    a_hid = F.silu(a_z1) * a_z2                                   # nonlinear
    d_hid = F.silu(a_z1 + d_z1) * (a_z2 + d_z2) - a_hid

    a_m = _lin_base(mlp_a.w3, a_hid, book)
    d_m = _lin_delta(mlp_d.w3, d_hid, book)

    g2 = blk_a.ls2.gamma.to(book) if hasattr(blk_a.ls2, "gamma") else 1.0
    return a_x + a_m * g2, d_x + d_m * g2


def _prep(backbone, img):
    x, (H, W) = backbone.prepare_tokens_with_masks(img)
    rope = backbone.rope_embed(H=H, W=W) if backbone.rope_embed is not None else None
    return x, rope


@torch.inference_mode()
def plain_forward(backbone, blocks, img) -> torch.Tensor:
    x, rope = _prep(backbone, img)
    for blk in blocks:
        x = blk(x, rope)
    return x


@torch.inference_mode()
def exact_decomposed_forward(backbone, blocks_a, blocks_d, base_img, full_img,
                             book: torch.dtype) -> torch.Tensor:
    """Base path on the L2 image, correction path carrying the exact delta to the real image."""
    a, rope = _prep(backbone, base_img)
    x_full, _ = _prep(backbone, full_img)
    a = a.to(book)
    d = (x_full.to(book) - a)
    for blk_a, blk_d in zip(blocks_a, blocks_d):
        a, d = exact_block(blk_a, blk_d, a, d, rope, book)
    return (a + d).to(torch.bfloat16)


# --------------------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------------------

def feature_metrics(f: torch.Tensor, ref: torch.Tensor, num_pre: int) -> tuple[float, float]:
    a = f[:, num_pre:].float().reshape(-1, f.shape[-1])
    b = ref[:, num_pre:].float().reshape(-1, ref.shape[-1])
    rel = ((a - b).norm() / b.norm().clamp_min(1e-12)).item()
    cos = F.cosine_similarity(a, b, dim=-1).mean().item()
    return rel, cos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["imagenet-1k", "coco2017"], default="imagenet-1k")
    ap.add_argument("--fraction", type=float, default=0.1)
    ap.add_argument("--image-size", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--bookkeep", choices=["fp32", "bf16"], default="fp32",
                    help="dtype the (a,d) pair is carried in; the 5 weight-GEMMs keep their own "
                         "precision either way. fp32 removes the decomposition's own rounding so "
                         "the GEMM precision is the only variable.")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()
    book = torch.float32 if args.bookkeep == "fp32" else torch.bfloat16

    image_size = args.image_size or (256 if args.dataset == "imagenet-1k" else 1024)
    device = torch.device(args.device)

    backbone = load_backbone(device)
    bf16 = backbone.blocks
    num_pre = backbone.n_storage_tokens + 1

    print("[init] quantizing blocks to NVFP4 ...", flush=True)
    fp4 = quantize_blocks_fp4(bf16)

    loader = build_loader(args.dataset, batch_size=1, image_size=image_size)
    total = len(loader.dataset)
    n_target = max(1, int(round(total * args.fraction)))
    print(f"[init] {args.dataset}: {total} samples, first {n_target} ({args.fraction:.0%}), "
          f"image_size={image_size}", flush=True)

    # (name, base-path blocks, correction-path blocks); None = plain forward variant
    conds = [
        ("fp4_full", None, None),
        ("exact_bf16_bf16", bf16, bf16),
        ("exact_bf16_fp4", bf16, fp4),
        ("exact_fp4_bf16", fp4, bf16),
        ("exact_fp4_fp4", fp4, fp4),
    ]
    acc = {c[0]: {"rel_l2": [], "cos": []} for c in conds}

    t0, done = time.time(), 0
    for images, _labels in loader:
        if done >= n_target:
            break
        img = to_model_input(images, device, image_size)
        base = l2_base(img)

        ref = plain_forward(backbone, bf16, img)
        for name, ba, bd in conds:
            out = (plain_forward(backbone, fp4, img) if ba is None
                   else exact_decomposed_forward(backbone, ba, bd, base, img, book))
            r, cs = feature_metrics(out, ref, num_pre)
            acc[name]["rel_l2"].append(r)
            acc[name]["cos"].append(cs)

        done += 1
        if done % 25 == 0 or done == n_target:
            msg = "  ".join(
                f"{n}: L2={np.mean(acc[n]['rel_l2']):.6f}" for n, _, _ in conds
            )
            print(f"[{done}/{n_target}] {time.time()-t0:.0f}s  {msg}", flush=True)

    print(f"\n===== FP4 FEATURE FIDELITY, EXACT DECOMPOSITION "
          f"({args.dataset}, N={done}, image_size={image_size}, bookkeep={args.bookkeep}) =====")
    print("reference = bf16_full (plain BF16 forward), patch tokens only")
    print("naming: exact_<base path precision>_<correction path precision>\n")
    print(f"{'condition':<20}{'rel L2':>13}{'cosine sim':>15}")
    summary = {"dataset": args.dataset, "n": done, "image_size": image_size,
               "bookkeep": args.bookkeep, "conditions": {}}
    for name, _, _ in conds:
        r, cs = float(np.mean(acc[name]["rel_l2"])), float(np.mean(acc[name]["cos"]))
        print(f"{name:<20}{r:>13.6f}{cs:>15.6f}")
        summary["conditions"][name] = {"rel_l2": r, "cosine": cs}

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(summary, indent=2))
        print(f"\n[out] {args.out_json}")


if __name__ == "__main__":
    main()
