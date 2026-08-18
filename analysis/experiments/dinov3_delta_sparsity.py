"""Is the correction delta concentrated enough that sparsifying it beats quantizing it?

Grid-based quantization produced no asymmetry: the delta is *harder* to quantize per unit norm than
`a+d` (block16 self-relative 0.1085 vs 0.0865 on COCO 1024) and carries the same outlier ratio. One
distribution shape explains both that and why sparsification might work anyway -- a spiky delta, a
few large entries among many near-zero ones, is the worst case for a shared block scale and the best
case for keeping the top-k and dropping the rest.

So this measures the shape directly, and then the thing that actually decides it: the error each
sparsity level injects into the layer output, for the delta and for the full activation side by side.

    today     nothing to sparsify -- `a+d` IS the answer, dropping entries destroys information
    proposed  keep the top p% of `d` by magnitude, zero the rest, and let the untouched entries stay
              at their approximate value, which is a valid answer rather than a hole

The `a+d` column is measured anyway, as the control that says whether concentration is a property of
the delta or of the activations generally. If both sparsify equally well, the technique is not
delta-specific and does not support the method.

Reported per site:
  mean/rms   how far off zero-centred the tensor is (a symmetric grid wastes range on a DC offset)
  kurtosis   spikiness; 3.0 is Gaussian, higher is heavier-tailed
  top1/10%   fraction of squared norm held by the largest 1% / 10% of entries by magnitude
  err@p      output-relative L2 error from keeping only the top p% (weight stays BF16 and exact, so
             this isolates sparsification from quantization)

    python -m analysis.experiments.dinov3_delta_sparsity --dataset coco2017 --image-size 1024
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import torch

import analysis.experiments.dinov3_fp4_feature_fidelity as fid

KEEP_FRACTIONS = (0.5, 0.25, 0.10, 0.05)


def _stats(x2d: torch.Tensor) -> tuple[float, float, float, float]:
    v = x2d.reshape(-1).float()
    rms = v.pow(2).mean().sqrt().clamp_min(1e-30)
    mean_over_rms = float(v.mean() / rms)
    centred = v - v.mean()
    kurt = float((centred.pow(4).mean() / centred.pow(2).mean().clamp_min(1e-30) ** 2))
    sq = v.pow(2)
    total = sq.sum().clamp_min(1e-30)
    n = v.numel()
    top1 = float(sq.topk(max(1, n // 100)).values.sum() / total)
    top10 = float(sq.topk(max(1, n // 10)).values.sum() / total)
    return mean_over_rms, kurt, top1, top10


def _sparsify(x: torch.Tensor, keep: float) -> torch.Tensor:
    """Zero all but the largest `keep` fraction of entries by magnitude, per row.

    Per row rather than per tensor on purpose: a global threshold would drop whole tokens, which is
    the token selection AppCorr already does upstream. The question here is what is left to gain
    *within* a token that the token-level selection has already decided to correct.
    """
    flat = x.reshape(-1, x.shape[-1]).float()
    k = max(1, int(round(flat.shape[-1] * keep)))
    thresh = flat.abs().topk(k, dim=-1).values[:, -1:]
    return (flat * (flat.abs() >= thresh)).reshape(x.shape).to(x.dtype)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["imagenet-1k", "coco2017"], default="coco2017")
    ap.add_argument("--blocks", type=int, nargs="+", default=[0, 10, 20, 30, 39])
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--batches", type=int, default=3)
    ap.add_argument("--image-size", type=int, default=1024)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device)
    backbone = fid.load_backbone(device)
    blocks = backbone.blocks
    keep_blocks = set(args.blocks)

    site_of: dict[int, str] = {}
    for bi, blk in enumerate(blocks):
        if bi not in keep_blocks:
            continue
        for name, mod in (("qkv", blk.attn.qkv), ("proj", blk.attn.proj),
                          ("w1", blk.mlp.w1), ("w3", blk.mlp.w3)):
            site_of[id(mod)] = f"{bi}.{name}"

    stats: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    pending: dict[str, dict] = {}

    def record(mod, x, path):
        site = site_of.get(id(mod))
        if site is None or x.numel() == 0:
            return
        x2d = x.reshape(-1, x.shape[-1])
        w = mod.weight.detach().float().t()
        if path == "a":
            pending[site] = {"x": x2d, "w": w}
            return
        prev = pending.pop(site, None)
        if prev is None:
            return
        x_a = prev["x"]
        x_ad = x_a + x2d
        total = (x_ad.float() @ w)
        tn = max(float(total.norm()), 1e-30)
        s = stats[site]

        for tag, t in (("ad", x_ad), ("d", x2d)):
            m, kurt, t1, t10 = _stats(t)
            s[f"{tag}_mean"].append(m)
            s[f"{tag}_kurt"].append(kurt)
            s[f"{tag}_top1"].append(t1)
            s[f"{tag}_top10"].append(t10)
            for keep in KEEP_FRACTIONS:
                # Both errors are normalised by ||W(a+d)||: sparsifying `d` leaves the base intact,
                # so its error enters the output at the delta's scale, not the delta's own norm.
                err = float((_sparsify(t, keep).float() @ w - (t.float() @ w)).norm()) / tn
                s[f"{tag}_err{keep}"].append(err)

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
        for i in range(args.batches):
            pending.clear()
            images = next(it)
            if isinstance(images, (list, tuple)):
                images = images[0]
            img = fid.to_model_input(images, device, args.image_size)
            fid.exact_decomposed_forward(
                backbone, blocks, blocks, fid.l2_base(img), img, torch.float32
            )
            print(f"  batch {i + 1}/{args.batches}", flush=True)
    finally:
        fid._lin_base, fid._lin_delta = orig_base, orig_delta

    def m(site, key):
        v = stats[site][key]
        return sum(v) / len(v) if v else float("nan")

    sites = sorted(stats, key=lambda k: (int(k.split(".")[0]), k))
    print(f"\n{args.dataset} @ {args.image_size}px, {args.batches} batches, {len(sites)} sites\n")

    print("=== shape ===")
    hdr = (f"{'site':<10}{'mean/rms':>19}{'kurtosis':>19}{'top1% energy':>21}"
           f"{'top10% energy':>21}")
    print(f"{'':<10}{'a+d':>9}{'d':>10}{'a+d':>9}{'d':>10}{'a+d':>10}{'d':>11}{'a+d':>10}{'d':>11}")
    for site in sites:
        print(f"{site:<10}{m(site,'ad_mean'):>9.3f}{m(site,'d_mean'):>10.3f}"
              f"{m(site,'ad_kurt'):>9.1f}{m(site,'d_kurt'):>10.1f}"
              f"{m(site,'ad_top1'):>10.3f}{m(site,'d_top1'):>11.3f}"
              f"{m(site,'ad_top10'):>10.3f}{m(site,'d_top10'):>11.3f}")
    print(f"{'MEAN':<10}"
          f"{sum(m(s,'ad_mean') for s in sites)/len(sites):>9.3f}"
          f"{sum(m(s,'d_mean') for s in sites)/len(sites):>10.3f}"
          f"{sum(m(s,'ad_kurt') for s in sites)/len(sites):>9.1f}"
          f"{sum(m(s,'d_kurt') for s in sites)/len(sites):>10.1f}"
          f"{sum(m(s,'ad_top1') for s in sites)/len(sites):>10.3f}"
          f"{sum(m(s,'d_top1') for s in sites)/len(sites):>11.3f}"
          f"{sum(m(s,'ad_top10') for s in sites)/len(sites):>10.3f}"
          f"{sum(m(s,'d_top10') for s in sites)/len(sites):>11.3f}")

    print("\n=== THE DECISION: output-relative error from keeping only the top p% ===")
    print("    (weight exact BF16 throughout, so this is sparsification alone, no quantization)")
    cols = "".join(f"{f'keep {int(k*100)}%':>21}" for k in KEEP_FRACTIONS)
    print(f"{'site':<10}{cols}")
    print(f"{'':<10}" + "".join(f"{'a+d':>10}{'d':>11}" for _ in KEEP_FRACTIONS))
    for site in sites:
        row = f"{site:<10}"
        for k in KEEP_FRACTIONS:
            row += f"{m(site, f'ad_err{k}'):>10.4f}{m(site, f'd_err{k}'):>11.4f}"
        print(row)
    print("-" * (10 + 21 * len(KEEP_FRACTIONS)))
    row = f"{'MEAN':<10}"
    for k in KEEP_FRACTIONS:
        a = sum(m(s, f'ad_err{k}') for s in sites) / len(sites)
        d = sum(m(s, f'd_err{k}') for s in sites) / len(sites)
        row += f"{a:>10.4f}{d:>11.4f}"
    print(row)
    row = f"{'a+d / d':<10}"
    for k in KEEP_FRACTIONS:
        a = sum(m(s, f'ad_err{k}') for s in sites) / len(sites)
        d = sum(m(s, f'd_err{k}') for s in sites) / len(sites)
        row += f"{'':>10}{a / d if d else float('nan'):>10.2f}x"
    print(row)
    print("\nThe last row is the asymmetry. Above ~2x means sparsification is delta-specific and "
          "worth building;\nnear 1x means it works equally well on the plain activation and proves "
          "nothing about the method.")


if __name__ == "__main__":
    main()
