"""
spatial_dependence.py -- Phase A: measure the FUNCTIONAL spatial dependence structure of the
Qwen2.5-VL vision encoder, to test whether each visual token depends on only a FEW other patches
(which would let a dependence-aware schedule pick WHICH high-res patches to send first / WHICH
patches to include when recomputing a token under cheap correction, recovering the grounding
overhead -- see README Phase 5b, the -2.56pp cheap-correction overhead on RefCOCO).

Functional dependence D[t, s] = how much target block t's merged tokens change when source block s's
input is degraded to the coarse base (rest full). This is the signal that actually governs
correction error (a stale patch j hurts token i exactly as much as i functionally depends on j),
unlike raw attention probability (a cheap online proxy tested separately in Phase B).

Partition: the merged-token grid is split into a P x Q grid of 2D blocks (so "top-left" vs
"bottom-right" dependence is captured, per the hypothesis). For each source block s we re-encode the
image with ONLY block s staled to base, and measure the per-token output delta vs the full encoding.

Architectural caveat printed with the results: 28/32 vision layers use WINDOWED attention
(window=112px), 4 use full attention -- so some locality is imposed by the architecture, not content.

Run:
    python qwen_vl_prefill/spatial_dependence.py --dataset refcoco --num-images 12 --P 4 --Q 4 \
        --device cuda:0 --out-fig qwen_vl_prefill/figs/dep_refcoco.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen_vl_prefill import introspect as I
from qwen_vl_prefill import progressive as PR
from qwen_vl_prefill import datasets_eval as DE


def block_layout(merged_h, merged_w, P, Q, merged_px=28):
    """P row-splits x Q col-splits of the merged grid. Each block -> (token indices [raster],
    pixel rect, grid-center (r,c)). Fixed P*Q blocks regardless of image size (comparable across
    images); distances measured in block-grid units."""
    r_edges = [(merged_h * p) // P for p in range(P + 1)]
    c_edges = [(merged_w * q) // Q for q in range(Q + 1)]
    blocks = []
    for bp in range(P):
        for bq in range(Q):
            r0, r1 = r_edges[bp], r_edges[bp + 1]
            c0, c1 = c_edges[bq], c_edges[bq + 1]
            if r1 <= r0 or c1 <= c0:
                blocks.append(None); continue
            toks = np.array([r * merged_w + c for r in range(r0, r1) for c in range(c0, c1)], dtype=np.int64)
            blocks.append({
                "toks": toks,
                "px": (r0 * merged_px, r1 * merged_px, c0 * merged_px, c1 * merged_px),
                "center": (bp, bq),
            })
    return blocks


def stale_block(full_np, base_np, px):
    r0, r1, c0, c1 = px
    img = full_np.copy()
    img[r0:r1, c0:c1] = base_np[r0:r1, c0:c1]
    return img


@torch.inference_mode()
def dependence_matrix(model, processor, full_np, base_np, P, Q, device):
    """Returns D [B, B] (B=P*Q): D[t,s] = mean L2 change of block t's tokens when block s is staled,
    normalized by the image's mean token norm. Diagonal = self-dependence."""
    full_e, grid = PR.encode_image(model, processor, full_np, device)  # [Nv, H], raster
    _, hp, wp = [int(x) for x in grid[0].tolist()]
    mh, mw = hp // 2, wp // 2  # merge_size 2
    blocks = block_layout(mh, mw, P, Q)
    tok_norm = full_e.float().norm(dim=-1).mean().item() + 1e-9

    B = len(blocks)
    D = np.full((B, B), np.nan, dtype=np.float64)
    for s, bs in enumerate(blocks):
        if bs is None:
            continue
        emb_s, _ = PR.encode_image(model, processor, stale_block(full_np, base_np, bs["px"]), device)
        delta = (full_e.float() - emb_s.float()).norm(dim=-1).cpu().numpy()  # [Nv]
        for t, bt in enumerate(blocks):
            if bt is None:
                continue
            D[t, s] = delta[bt["toks"]].mean() / tok_norm
    return D, blocks


def analyze(D, blocks):
    B = D.shape[0]
    valid = [i for i, b in enumerate(blocks) if b is not None]
    centers = {i: blocks[i]["center"] for i in valid}
    diag = np.array([D[i, i] for i in valid])
    off = np.array([D[t, s] for t in valid for s in valid if t != s])
    # distance decay (target!=source), distance in block-grid units
    dist_bins = {}
    for t in valid:
        for s in valid:
            if t == s:
                continue
            d = round(np.hypot(centers[t][0] - centers[s][0], centers[t][1] - centers[s][1]), 3)
            dist_bins.setdefault(d, []).append(D[t, s])
    # top-k concentration: for each target, fraction of total cross-block dependence in top-k sources
    def topk_frac(k):
        fr = []
        for t in valid:
            row = np.array([D[t, s] for s in valid if s != t])
            row = row[~np.isnan(row)]
            if row.sum() <= 0:
                continue
            fr.append(np.sort(row)[::-1][:k].sum() / row.sum())
        return float(np.mean(fr))
    # ACTIONABLE metric: refreshing block t itself (always) + its top-k cross sources, what fraction
    # of t's TOTAL dependence (incl. self) is captured => how much correction error is removed.
    def selfplus_topk_frac(k):
        fr = []
        for t in valid:
            cross = np.array([D[t, s] for s in valid if s != t]); cross = cross[~np.isnan(cross)]
            total = D[t, t] + cross.sum()
            if total <= 0:
                continue
            fr.append((D[t, t] + np.sort(cross)[::-1][:k].sum()) / total)
        return float(np.mean(fr))
    # hub imbalance: column sums (how much each source is depended-on), coeff of variation
    col = np.array([np.nansum([D[t, s] for t in valid if t != s]) for s in valid])
    cv = float(col.std() / (col.mean() + 1e-9))
    return {
        "diag_mean": float(diag.mean()), "off_mean": float(off.mean()),
        "diag_over_off": float(diag.mean() / (off.mean() + 1e-9)),
        "dist_decay": {d: float(np.mean(v)) for d, v in sorted(dist_bins.items())},
        "topk_frac": {k: topk_frac(k) for k in (1, 2, 3)},
        "selfplus_topk": {k: selfplus_topk_frac(k) for k in (0, 1, 2, 3)},
        "hub_cv": cv,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--dataset", default="refcoco", choices=list(DE.SPECS))
    ap.add_argument("--num-images", type=int, default=12)
    ap.add_argument("--P", type=int, default=4)
    ap.add_argument("--Q", type=int, default=4)
    ap.add_argument("--base-factor", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-fig", default=None)
    args = ap.parse_args()
    device = args.device

    print(f"[dep] loading {args.model_id} ...")
    model, processor = I.load_model(args.model_id, device=device)
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    from datasets import load_dataset

    ip = processor.image_processor
    factor = ip.patch_size * ip.merge_size * 4
    min_px, max_px = ip.size["shortest_edge"], ip.size["longest_edge"]
    spec = DE.get_spec(args.dataset)
    ds = spec.load(load_dataset)
    stride = max(len(ds) // args.num_images, 1)
    indices = list(range(0, len(ds), stride))[:args.num_images]

    Dsum = None; ncnt = 0
    for c, idx in enumerate(indices):
        image_r, _, _ = spec.prepare(ds[idx], smart_resize, factor, min_px, max_px)
        full_np = np.array(image_r, dtype=np.uint8)
        base_np = PR.make_base(full_np, args.base_factor)
        D, blocks = dependence_matrix(model, processor, full_np, base_np, args.P, args.Q, device)
        Dsum = D if Dsum is None else np.nansum(np.stack([Dsum, D]), axis=0)
        ncnt += 1
        if (c + 1) % 4 == 0:
            print(f"  [{c+1}/{len(indices)}] processed", flush=True)
    Davg = Dsum / ncnt
    stats = analyze(Davg, blocks)

    print(f"\n===== FUNCTIONAL SPATIAL DEPENDENCE ({args.dataset}, {args.model_id.split('/')[-1]}, "
          f"{args.P}x{args.Q} blocks, N={ncnt} images, base factor {args.base_factor}) =====")
    print(f"  self-dependence (diag) mean:      {stats['diag_mean']:.4f}")
    print(f"  cross-block (off-diag) mean:      {stats['off_mean']:.4f}")
    print(f"  diag / off-diag ratio:            {stats['diag_over_off']:.2f}x  (self >> cross => strong locality)")
    print(f"  distance decay (block-grid units, cross-block only):")
    for d, v in stats["dist_decay"].items():
        print(f"      dist {d:>5}: {v:.4f}")
    print(f"  top-k dependence concentration (per target, fraction of cross-block dep in top-k sources):")
    for k, f in stats["topk_frac"].items():
        print(f"      top-{k}: {100*f:.1f}%")
    print(f"  ACTIONABLE -- refresh self + top-k neighbors, fraction of TOTAL dependence captured")
    print(f"  (=how much cheap-correction error removed; k=0 is self-only = current per-band scheme):")
    for k, f in stats["selfplus_topk"].items():
        print(f"      self+top-{k}: {100*f:.1f}%")
    print(f"  hub imbalance (source column-sum CV): {stats['hub_cv']:.3f}  (higher => some blocks are hubs)")
    print(f"\n  NOTE: 28/32 vision layers are WINDOWED (window=112px) -> some locality is architectural,")
    print(f"  not content. Compare distance-decay slope against that window scale when interpreting.")

    if args.out_fig:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, (a0, a1) = plt.subplots(1, 2, figsize=(11, 4.2))
            im = a0.imshow(Davg, cmap="viridis"); a0.set_title(f"D[target,source]  {args.dataset}")
            a0.set_xlabel("source block"); a0.set_ylabel("target block"); fig.colorbar(im, ax=a0)
            dd = stats["dist_decay"]
            a1.plot(list(dd.keys()), list(dd.values()), "o-")
            a1.set_xlabel("block-grid distance"); a1.set_ylabel("mean cross-block dependence")
            a1.set_title("distance decay"); a1.grid(True, alpha=0.3)
            Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)
            fig.tight_layout(); fig.savefig(args.out_fig, dpi=120)
            print(f"\n  figure saved -> {args.out_fig}")
        except Exception as e:
            print(f"  (fig skipped: {e})")


if __name__ == "__main__":
    main()
