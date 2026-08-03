"""Two follow-ups to the exact (a, d) decomposition study.

A. **Channel-axis spread of the delta vs the base.** NVFP4 quantizes along the reduction (channel)
   dimension — a per-16-element E4M3 block scale plus a per-tensor scale — so what governs its
   relative error is how much dynamic range the input carries along that axis. This measures, at
   the input of every quantized Linear (qkv / proj / w1 / w2 / w3, all 40 blocks), the per-token
   channel-axis variance of the base activation `a` and of the correction delta `d`, plus the
   per-tensor amax that actually sets NVFP4's outer scale.

B. **Is the control error subtractable?** The `exact_bf16_bf16` control has a small nonzero
   rel L2 against the plain BF16 forward. Rather than assume how to remove it, this measures the
   FP4 contribution *directly*, by using the control itself as the reference:

       e_ctrl = ctrl - ref          (control vs plain BF16 forward)
       e_fp4  = fp4d - ref          (FP4-on-delta vs plain BF16 forward)
       e_iso  = fp4d - ctrl         (FP4-on-delta vs the control)  <- assumption-free

   and reports `e_iso` next to what linear subtraction (`|e_fp4| - |e_ctrl|`) and quadrature
   subtraction (`sqrt(|e_fp4|^2 - |e_ctrl|^2)`) would have predicted, plus cos(e_ctrl, e_fp4),
   which says which of the two — if either — is the right model.

Run:
    PYTHONPATH="$PWD/appcorr/models:$PWD" python analysis/experiments/dinov3_exact_decomposition_stats.py \
        --dataset imagenet-1k --num-images 50
    PYTHONPATH="$PWD/appcorr/models:$PWD" python analysis/experiments/dinov3_exact_decomposition_stats.py \
        --dataset coco2017 --num-images 50
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

for _p in (Path(__file__).resolve().parents[2],):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import analysis.experiments.dinov3_fp4_feature_fidelity as M  # noqa: E402
from offload.server.model.dinov3_precision import _eligible_fp4_linears  # noqa: E402

LIN_ORDER = ["attn.qkv", "attn.proj", "mlp.w1", "mlp.w2", "mlp.w3"]


def channel_stats(x: torch.Tensor) -> tuple[float, float]:
    """(mean per-token variance along the channel axis, per-tensor amax)."""
    xf = x.detach().float()
    return xf.var(dim=-1).mean().item(), xf.abs().max().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["imagenet-1k", "coco2017"], default="imagenet-1k")
    ap.add_argument("--num-images", type=int, default=50)
    ap.add_argument("--image-size", type=int, default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    image_size = args.image_size or (256 if args.dataset == "imagenet-1k" else 1024)
    device = torch.device(args.device)

    backbone = M.load_backbone(device)
    bf16 = backbone.blocks
    num_pre = backbone.n_storage_tokens + 1
    print("[init] quantizing blocks to NVFP4 ...", flush=True)
    fp4 = M.quantize_blocks_fp4(bf16)

    # id(module) -> (block_idx, linear name), so the monkeypatched hooks can label themselves.
    id2name = {}
    for bi, blk in enumerate(bf16):
        for name, mod in _eligible_fp4_linears(blk):
            id2name[id(mod)] = (bi, name)

    # ---- Part A instrumentation: exact_block resolves _lin_base/_lin_delta from module globals
    # at call time, so patching the module attributes is enough (no code duplication).
    stats = defaultdict(lambda: {"var": [], "amax": []})
    collecting = False
    orig_base, orig_delta = M._lin_base, M._lin_delta

    def rec_base(mod, a, book):
        if collecting and id(mod) in id2name:
            v, mx = channel_stats(a)
            key = ("a", id2name[id(mod)][1])
            stats[key]["var"].append(v)
            stats[key]["amax"].append(mx)
        return orig_base(mod, a, book)

    def rec_delta(mod, d, book):
        if collecting and id(mod) in id2name:
            v, mx = channel_stats(d)
            key = ("d", id2name[id(mod)][1])
            stats[key]["var"].append(v)
            stats[key]["amax"].append(mx)
        return orig_delta(mod, d, book)

    M._lin_base, M._lin_delta = rec_base, rec_delta

    loader = M.build_loader(args.dataset, batch_size=1, image_size=image_size)
    n = min(args.num_images, len(loader.dataset))
    print(f"[init] {args.dataset}: {n} images, image_size={image_size}", flush=True)

    iso, lin_pred, quad_pred, ctrl_rel, fp4_rel, cosines = [], [], [], [], [], []
    book = torch.float32
    done = 0
    for images, _ in loader:
        if done >= n:
            break
        img = M.to_model_input(images, device, image_size)
        base = M.l2_base(img)

        ref = M.plain_forward(backbone, bf16, img)

        collecting = True   # stats only from the bf16/bf16 pass
        ctrl = M.exact_decomposed_forward(backbone, bf16, bf16, base, img, book)
        collecting = False

        fp4d = M.exact_decomposed_forward(backbone, bf16, fp4, base, img, book)

        r = ref[:, num_pre:].float()
        e_ctrl = (ctrl[:, num_pre:].float() - r).flatten()
        e_fp4 = (fp4d[:, num_pre:].float() - r).flatten()
        e_iso = (fp4d[:, num_pre:].float() - ctrl[:, num_pre:].float()).flatten()
        rn = r.norm().clamp_min(1e-12)

        a_ctrl, a_fp4 = e_ctrl.norm().item(), e_fp4.norm().item()
        ctrl_rel.append(a_ctrl / rn.item())
        fp4_rel.append(a_fp4 / rn.item())
        iso.append(e_iso.norm().item() / rn.item())
        lin_pred.append((a_fp4 - a_ctrl) / rn.item())
        quad_pred.append(np.sqrt(max(a_fp4**2 - a_ctrl**2, 0.0)) / rn.item())
        cosines.append(F.cosine_similarity(e_ctrl, e_fp4, dim=0).item())

        done += 1
        if done % 10 == 0 or done == n:
            print(f"  [{done}/{n}]", flush=True)

    M._lin_base, M._lin_delta = orig_base, orig_delta

    # ---------------- Part A report ----------------
    print(f"\n===== A. CHANNEL-AXIS SPREAD, base a vs delta d ({args.dataset}, N={done}) =====")
    print("input of each quantized Linear, all 40 blocks pooled")
    print("var = mean per-token variance along the channel axis; amax = per-tensor max |.|\n")
    print(f"{'linear':<12}{'var(a)':>12}{'var(d)':>12}{'var d/a':>10}"
          f"{'amax(a)':>11}{'amax(d)':>11}{'amax d/a':>10}")
    part_a = {}
    for name in LIN_ORDER:
        va = float(np.mean(stats[("a", name)]["var"]))
        vd = float(np.mean(stats[("d", name)]["var"]))
        ma = float(np.mean(stats[("a", name)]["amax"]))
        md = float(np.mean(stats[("d", name)]["amax"]))
        print(f"{name:<12}{va:>12.4f}{vd:>12.4f}{vd/va:>10.4f}{ma:>11.3f}{md:>11.3f}{md/ma:>10.4f}")
        part_a[name] = {"var_a": va, "var_d": vd, "var_ratio": vd / va,
                        "amax_a": ma, "amax_d": md, "amax_ratio": md / ma}
    allv_a = float(np.mean([part_a[k]["var_a"] for k in LIN_ORDER]))
    allv_d = float(np.mean([part_a[k]["var_d"] for k in LIN_ORDER]))
    print(f"{'ALL':<12}{allv_a:>12.4f}{allv_d:>12.4f}{allv_d/allv_a:>10.4f}")

    # ---------------- Part B report ----------------
    m = lambda v: float(np.mean(v))
    print(f"\n===== B. IS THE CONTROL ERROR SUBTRACTABLE? ({args.dataset}, N={done}) =====")
    print(f"  |e_ctrl| rel (control vs bf16_full)        : {m(ctrl_rel):.6f}")
    print(f"  |e_fp4|  rel (fp4-on-delta vs bf16_full)   : {m(fp4_rel):.6f}")
    print(f"  |e_iso|  rel (fp4-on-delta vs CONTROL)     : {m(iso):.6f}   <- measured, assumption-free")
    print(f"     linear subtraction would predict        : {m(lin_pred):.6f}")
    print(f"     quadrature subtraction would predict    : {m(quad_pred):.6f}")
    print(f"  cos(e_ctrl, e_fp4)                         : {m(cosines):.4f}")

    summary = {
        "dataset": args.dataset, "n": done, "image_size": image_size,
        "channel_stats": part_a,
        "error_isolation": {
            "ctrl_rel": m(ctrl_rel), "fp4_rel": m(fp4_rel), "iso_rel": m(iso),
            "linear_pred": m(lin_pred), "quadrature_pred": m(quad_pred),
            "cos_ctrl_fp4": m(cosines),
        },
    }
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(summary, indent=2))
        print(f"\n[out] {args.out_json}")


if __name__ == "__main__":
    main()
