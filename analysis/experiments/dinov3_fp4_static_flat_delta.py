"""Does the correction delta tolerate a static flat FP4 scale where the activation does not?

The hypothesis under test: NVFP4's 16-element block scale is what makes FP4 survivable, and it is
needed because *activations* have huge per-block dynamic range. The correction delta `d` should have
less of it, so a single scale fixed offline -- which removes the per-call amax reduction and the
scale tensor entirely -- might cost little on `d` while wrecking `a`.

Three quantizers per site, applied to the Linear's input only (the weight is always block16, since
weight scales are computed offline and cost nothing at inference):

    block16   per-16-element scale from this tensor       -- what ships today
    flat_dyn  one scale for the tensor, from this tensor  -- no block scale, still a runtime reduction
    flat_stat one scale for the tensor, FROZEN from calibration -- no runtime reduction at all

`flat_stat` is the deployable one: it is the only variant whose quantization needs no reduction over
the activation, so it is the only one that can save the preprocessing latency. `flat_dyn` is here to
separate "coarse scale hurts" from "stale scale hurts".

**The decision metric is error injected into the total output**, because that is what the two designs
actually differ on:

    today     recompute W(a+d) in FP4       -> ||err(W(a+d))|| / ||W(a+d)||
    proposed  keep Wa, recompute W d in FP4 -> ||err(W d)||    / ||W(a+d)||

Normalising the delta's error against its *own* norm instead is the trap: a 50% error on a delta that
carries a tenth of the output injects 5%, not 50%. Both views are printed, the total-output one first.

Sites are where the five Linears take their input:

    qkv <- norm1 output          proj <- attention core output
    w1/w2 <- norm2 output        w3 <- SwiGLU hidden

    python -m analysis.experiments.dinov3_fp4_static_flat_delta --blocks 0 20 39
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import torch

import analysis.experiments.dinov3_fp4_feature_fidelity as fid
from offload.server.model.fp4_granularity_linear import _round_to_e2m1, fake_quantize_fp4

_E2M1_MAX = 6.0


def _flat_quant(x: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Quantize-dequantize with one given scale for the whole tensor (E2M1 grid, E4M3 scale)."""
    s = scale.to(torch.float8_e4m3fn).to(torch.float32).clamp_min(1e-30)
    return (_round_to_e2m1(x.float() / s) * s).to(x.dtype)


def outlier_ratio(x: torch.Tensor) -> float:
    """Global amax over the mean per-16-block amax -- the range a flat scale must span."""
    flat = x.reshape(-1, x.shape[-1]).float()
    g = flat.reshape(-1, 16)
    return float(flat.abs().amax() / g.abs().amax(dim=-1).mean().clamp_min(1e-30))


def _mean(v):
    return sum(v) / len(v) if v else float("nan")


def _median(v):
    w = sorted(v)
    n = len(w)
    if not n:
        return float("nan")
    return w[n // 2] if n % 2 else (w[n // 2 - 1] + w[n // 2]) / 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["imagenet-1k", "coco2017"], default="imagenet-1k")
    ap.add_argument("--blocks", type=int, nargs="+", default=[0, 20, 39])
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--cal-batches", type=int, default=2)
    ap.add_argument("--eval-batches", type=int, default=3)
    ap.add_argument("--image-size", type=int, default=224)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device)
    backbone = fid.load_backbone(device)
    blocks = backbone.blocks
    keep = set(args.blocks)

    # Module identity -> "<block>.<site>". blocks_a and blocks_d are the same BF16 stack: the
    # experimental variable is the quantizer applied to the captured input, not the block's own
    # precision, so both paths must run exact BF16 or the two are not comparable.
    site_of: dict[int, str] = {}
    for bi, blk in enumerate(blocks):
        if bi not in keep:
            continue
        for name, mod in (
            ("qkv", blk.attn.qkv), ("proj", blk.attn.proj),
            ("w1", blk.mlp.w1), ("w3", blk.mlp.w3),
        ):
            site_of[id(mod)] = f"{bi}.{name}"

    cal_amax: dict[tuple[str, str], float] = defaultdict(float)
    _KEYS = ("outlier_a", "outlier_d", "d_share",
             "self_a_blk", "self_a_fdyn", "self_a_fstat",
             "self_d_blk", "self_d_fdyn", "self_d_fstat",
             "tot_a_blk", "tot_a_fstat", "tot_d_blk", "tot_d_fstat")
    stats: dict[str, dict[str, list]] = defaultdict(lambda: {k: [] for k in _KEYS})
    mode = {"phase": "cal"}
    # `exact_block` calls the base Linear and then immediately the delta Linear for the same site, so
    # the a-side result is still pending when the d-side arrives and the true ||W(a+d)|| can be formed
    # from the two exact outputs -- no triangle-inequality bound, which would be wrong wherever Wa and
    # Wd partly cancel.
    pending: dict[str, dict] = {}

    def _quantized(x2d, wq, cal_key):
        static_scale = torch.tensor(
            cal_amax[cal_key] / _E2M1_MAX, device=x2d.device, dtype=torch.float32
        )
        return {
            "blk": fake_quantize_fp4(x2d, "block16").float() @ wq,
            "fdyn": fake_quantize_fp4(x2d, "tensor").float() @ wq,
            "fstat": _flat_quant(x2d, static_scale).float() @ wq,
        }

    def record(mod, x, path):
        site = site_of.get(id(mod))
        # w2 shares w1's input; leaving it out of `site_of` keeps each distinct input counted once.
        if site is None or x.numel() == 0:
            return
        x2d = x.reshape(-1, x.shape[-1])
        if mode["phase"] == "cal":
            cal_amax[(site, path)] = max(cal_amax[(site, path)], float(x2d.abs().amax()))
            # The "today" arm's static scale is calibrated on `a + d`, the tensor it actually
            # quantizes. Pairing works here for the same reason as in the eval phase: the base and
            # delta Linears for a site are called back to back.
            if path == "a":
                pending[site] = x2d
            else:
                x_a = pending.pop(site, None)
                if x_a is not None:
                    key = (site, "ad")
                    cal_amax[key] = max(cal_amax[key], float((x_a + x2d).abs().amax()))
            return

        w = mod.weight.detach()
        wq = fake_quantize_fp4(w, "block16").float().t()
        ref = x2d.float() @ w.float().t()
        q = _quantized(x2d, wq, (site, path))

        if path == "a":
            pending[site] = {"x": x2d, "ref": ref, "q": q, "outlier": outlier_ratio(x2d)}
            return

        prev = pending.pop(site, None)
        if prev is None:
            return
        s = stats[site]
        ref_a, q_a, x_a = prev["ref"], prev["q"], prev["x"]
        total = ref_a + ref                      # exact W(a+d), no triangle-inequality bound
        tn = max(float(total.norm()), 1e-30)

        # Today's correction recomputes the FULL activation for the selected tokens, so its arm has
        # to quantize `a + d` -- not `a`. Quantizing `a` alone would understate today's error by
        # leaving the delta out of the tensor whose amax sets the scale.
        x_ad = x_a + x2d
        q_ad = _quantized(x_ad, wq, (site, "ad"))

        s["outlier_a"].append(outlier_ratio(x_ad))
        s["outlier_d"].append(outlier_ratio(x2d))
        s["d_share"].append(float(ref.norm()) / tn)

        adn, dn = max(float(total.norm()), 1e-30), max(float(ref.norm()), 1e-30)
        for tag in ("blk", "fdyn", "fstat"):
            s[f"self_a_{tag}"].append(float((q_ad[tag] - total).norm()) / adn)
            s[f"self_d_{tag}"].append(float((q[tag] - ref).norm()) / dn)
        for tag in ("blk", "fstat"):
            # today: quantize the whole activation. proposed: quantize only the delta.
            s[f"tot_a_{tag}"].append(float((q_ad[tag] - total).norm()) / tn)
            s[f"tot_d_{tag}"].append(float((q[tag] - ref).norm()) / tn)

    # Monkeypatch rather than copy `exact_block`: the decomposition's op order is subtle (RoPE,
    # LayerScale, the SwiGLU cross term) and a second copy here would drift out of agreement with the
    # harness it is meant to be measuring.
    orig_base, orig_delta = fid._lin_base, fid._lin_delta

    def traced_base(mod, a, book):
        record(mod, a, "a")
        return orig_base(mod, a, book)

    def traced_delta(mod, d, book):
        record(mod, d, "d")
        return orig_delta(mod, d, book)

    fid._lin_base, fid._lin_delta = traced_base, traced_delta
    try:
        loader = fid.build_loader(args.dataset, args.batch, args.image_size)
        it = iter(loader)
        total_batches = args.cal_batches + args.eval_batches
        for i in range(total_batches):
            mode["phase"] = "cal" if i < args.cal_batches else "eval"
            pending.clear()
            images = next(it)
            if isinstance(images, (list, tuple)):
                images = images[0]
            img = fid.to_model_input(images, device, args.image_size)
            fid.exact_decomposed_forward(
                backbone, blocks, blocks, fid.l2_base(img), img, torch.float32
            )
            print(f"  [{mode['phase']}] batch {i + 1}/{total_batches}", flush=True)
    finally:
        fid._lin_base, fid._lin_delta = orig_base, orig_delta

    sites = sorted(stats, key=lambda k: (int(k.split(".")[0]), k))
    print(f"\ncalibration: {args.cal_batches} batches, eval: {args.eval_batches} batches, "
          f"dataset={args.dataset}, sites={len(sites)}\n")

    print("=== THE DECISION: error injected into the total output W(a+d) ===")
    print("    a+d = recompute the whole activation (today) | delta = recompute only d (proposed)")
    hdr = (f"{'site':<10}{'d_share':>9}{'a+d blk16':>11}{'a+d flat':>10}"
           f"{'delta blk16':>13}{'delta flat':>12}{'dflat/adflat':>14}")
    print(hdr)
    print("-" * len(hdr))
    for site in sites:
        s = stats[site]
        adf, df = _mean(s["tot_a_fstat"]), _mean(s["tot_d_fstat"])
        print(f"{site:<10}{_mean(s['d_share']):>9.3f}{_mean(s['tot_a_blk']):>11.4f}"
              f"{adf:>10.4f}{_mean(s['tot_d_blk']):>13.4f}{df:>12.4f}{df / adf:>13.2f}x")
    print("-" * len(hdr))
    m = {k: _mean([_mean(stats[s][k]) for s in sites]) for k in _KEYS}
    print(f"{'MEAN':<10}{m['d_share']:>9.3f}{m['tot_a_blk']:>11.4f}{m['tot_a_fstat']:>10.4f}"
          f"{m['tot_d_blk']:>13.4f}{m['tot_d_fstat']:>12.4f}"
          f"{m['tot_d_fstat'] / m['tot_a_fstat']:>13.2f}x")
    win_vs_flat = sum(
        1 for s in sites if _mean(stats[s]["tot_d_fstat"]) < _mean(stats[s]["tot_a_fstat"])
    )
    win_vs_blk = sum(
        1 for s in sites if _mean(stats[s]["tot_d_fstat"]) < _mean(stats[s]["tot_a_blk"])
    )
    print(f"\nflat-static on the delta beats flat-static on a+d : {win_vs_flat}/{len(sites)} sites")
    print(f"flat-static on the delta beats BLOCK16 on a+d      : {win_vs_blk}/{len(sites)} sites"
          "   <- the bar that matters; block16 on a+d is what ships")

    print("\n=== mechanism: each path's error relative to its OWN output ===")
    hdr2 = (f"{'site':<10}{'out_a':>9}{'out_d':>9}{'a blk16':>9}{'a fdyn':>9}{'a fstat':>9}"
            f"{'d blk16':>9}{'d fdyn':>9}{'d fstat':>9}")
    print(hdr2)
    print("-" * len(hdr2))
    for site in sites:
        s = stats[site]
        print(f"{site:<10}{_mean(s['outlier_a']):>9.1f}{_mean(s['outlier_d']):>9.1f}"
              f"{_mean(s['self_a_blk']):>9.4f}{_mean(s['self_a_fdyn']):>9.4f}"
              f"{_mean(s['self_a_fstat']):>9.4f}{_mean(s['self_d_blk']):>9.4f}"
              f"{_mean(s['self_d_fdyn']):>9.4f}{_mean(s['self_d_fstat']):>9.4f}")
    print("-" * len(hdr2))
    for label, fn in (("MEAN", _mean), ("MEDIAN", _median)):
        vals = {k: fn([_mean(stats[s][k]) for s in sites]) for k in _KEYS}
        print(f"{label:<10}{vals['outlier_a']:>9.1f}{vals['outlier_d']:>9.1f}"
              f"{vals['self_a_blk']:>9.4f}{vals['self_a_fdyn']:>9.4f}{vals['self_a_fstat']:>9.4f}"
              f"{vals['self_d_blk']:>9.4f}{vals['self_d_fdyn']:>9.4f}{vals['self_d_fstat']:>9.4f}")
    print("\nMedian is shown alongside the mean because a single site can carry an outlier ratio two "
          "orders above the rest\nand drag the mean with it. Where the two disagree, the median "
          "describes the typical site.")


if __name__ == "__main__":
    main()
