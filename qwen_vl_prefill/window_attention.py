"""
window_attention.py -- measure cross-WINDOW attention in the ONLY 4 layers that can create
cross-window dependence: the full-attention layers (fullatt_block_indexes = 7,15,23,31). The other
28 layers are windowed (window=112px) so their attention is confined within a window by construction
-- ALL long-range/global mixing in the Qwen2.5-VL vision encoder happens in these 4 layers.

For each full-attention layer we compute the real attention matrix softmax(QK^T * scale) over the
whole image (SDPA hides it, so we recompute it explicitly), then aggregate to window x window:
    A_win[a, b] = mean_heads mean_{q in window a} sum_{k in window b} attn[q, k]
so each row sums to 1 (a query window's attention distributed over key windows). Windows are the
contiguous token runs in the window-permuted sequence (cu_window_seqlens boundaries). Window spatial
centroids (for distance decay) come from mapping permuted tokens back to original merged-grid coords.

Reports, per full layer + averaged:
  - within-window fraction  (diag of A_win): how much attention STAYS in the query's own window
  - cross-window top-k concentration: of the OFF-window mass, fraction in the top-k key windows
  - cross-window distance decay: off-window A_win vs window-centroid distance
This tells us whether cross-window dependence is sparse/local enough to schedule (send/refresh a
window's few strongly-linked windows), which is exactly what the correction selector would exploit.

Run: python qwen_vl_prefill/window_attention.py --dataset refcoco --num-images 12 --device cuda:0 \
        --out-fig qwen_vl_prefill/figs/winattn_refcoco.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen_vl_prefill import introspect as I
from qwen_vl_prefill import progressive as PR
from qwen_vl_prefill import progressive_correct as PC
from qwen_vl_prefill import datasets_eval as DE
from appcorr.models.qwen25vl.vision.attention import apply_rotary_pos_emb_vision


def window_bounds(cu_window_seqlens, T):
    """Contiguous token ranges [start, end) per window (permuted order)."""
    cw = [int(x) for x in cu_window_seqlens.tolist()]
    cw = sorted(set(cw))
    assert cw[0] == 0 and cw[-1] == T, (cw[:3], cw[-3:], T)
    return [(cw[i], cw[i + 1]) for i in range(len(cw) - 1) if cw[i + 1] > cw[i]]


def window_centroids(wins, ctx, merged_w, unit):
    """Each window's centroid in original merged-grid (row,col), via window_index un-permute."""
    win_index = ctx["window_index"].cpu().numpy()  # permuted merge-group -> original merge-group
    cents = []
    for (a, b) in wins:
        mg = np.arange(a, b) // unit          # permuted merge-group ids for this window's tokens
        orig = win_index[np.unique(mg)]       # original merge-group ids
        rc = np.stack([orig // merged_w, orig % merged_w], 1).astype(np.float64)
        cents.append(rc.mean(0))
    return np.array(cents)  # [W, 2]


@torch.inference_mode()
def window_attn_maps(model, tower, processor, full_np, device):
    """Returns {layer_idx: A_win [W,W]} for the 4 full-attention layers, plus window centroids."""
    pv, grid = PC.encode_pixels(processor, full_np, device, model.dtype)
    ctx = tower.prepare_full_tokens(pv, grid)
    _, hp, wp = [int(x) for x in grid[0].tolist()]
    merged_w = wp // 2
    unit = tower.spatial_merge_unit
    T = ctx["seq_len"]
    wins = window_bounds(ctx["cu_window_seqlens"], T)
    W = len(wins)
    # token -> window id (permuted order)
    tok2win = np.empty(T, dtype=np.int64)
    for w, (a, b) in enumerate(wins):
        tok2win[a:b] = w
    tok2win_t = torch.from_numpy(tok2win).to(device)
    cents = window_centroids(wins, ctx, merged_w, unit)

    cos, sin = ctx["position_embeddings"]
    h = ctx["hidden_states"]
    maps = {}
    for i, blk in enumerate(tower.blocks):
        segs = tower._segments_for_layer(i, ctx)
        if i in tower.fullatt_block_indexes:
            x = blk.norm1(h)
            q, k, v = blk.attn._qkv_heads(x)            # [T, H, Dh]
            q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
            q = q.transpose(0, 1).float()               # [H, T, Dh]
            k = k.transpose(0, 1).float()
            attn = torch.softmax((q @ k.transpose(-1, -2)) * blk.attn.scaling, dim=-1)  # [H, T, T]
            attn = attn.mean(0)                          # [T, T] head-averaged
            # aggregate keys -> key windows: sum attn over k in each window
            keyagg = torch.zeros(T, W, device=device, dtype=attn.dtype)
            keyagg.index_add_(1, tok2win_t, attn)        # [T, W]
            # aggregate queries -> query windows: mean over q in each window
            Awin = torch.zeros(W, W, device=device, dtype=attn.dtype)
            Awin.index_add_(0, tok2win_t, keyagg)        # [W, W] summed over q
            counts = torch.bincount(tok2win_t, minlength=W).clamp(min=1).float().unsqueeze(1)
            Awin = (Awin / counts).cpu().numpy()         # row = query window, sums to 1
            maps[i] = Awin
        h = blk.forward(h, segs, ctx["position_embeddings"])
    return maps, cents


def analyze_layer(Awin, cents):
    W = Awin.shape[0]
    within = float(np.mean(np.diag(Awin)))               # attention staying in own window
    # cross-window (off-diagonal), renormalized per row to the cross mass
    topk = {1: [], 2: [], 3: []}
    dist_bins = {}
    for a in range(W):
        cross = np.delete(Awin[a], a)
        cs = cross.sum()
        if cs > 0:
            sc = np.sort(cross)[::-1]
            for k in topk:
                topk[k].append(sc[:k].sum() / cs)
        for b in range(W):
            if a == b:
                continue
            d = round(float(np.hypot(*(cents[a] - cents[b]))), 1)
            dist_bins.setdefault(d, []).append(Awin[a, b])
    return {
        "within": within,
        "topk": {k: float(np.mean(v)) for k, v in topk.items()},
        "dist_decay": {d: float(np.mean(v)) for d, v in sorted(dist_bins.items())},
        "W": W,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--dataset", default="refcoco", choices=list(DE.SPECS))
    ap.add_argument("--num-images", type=int, default=12)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-fig", default=None)
    args = ap.parse_args()
    device = args.device

    print(f"[winattn] loading {args.model_id} ...")
    model, processor = I.load_model(args.model_id, device=device)
    tower = PC.build_tower(model)
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    from datasets import load_dataset

    ip = processor.image_processor
    factor = ip.patch_size * ip.merge_size * 4
    min_px, max_px = ip.size["shortest_edge"], ip.size["longest_edge"]
    spec = DE.get_spec(args.dataset)
    ds = spec.load(load_dataset)
    stride = max(len(ds) // args.num_images, 1)
    indices = list(range(0, len(ds), stride))[:args.num_images]

    full_layers = sorted(tower.fullatt_block_indexes)
    acc = {L: {"within": [], "topk": {1: [], 2: [], 3: []}, "dd": {}, "W": []} for L in full_layers}
    last_maps = None
    for c, idx in enumerate(indices):
        image_r, _, _ = spec.prepare(ds[idx], smart_resize, factor, min_px, max_px)
        full_np = np.array(image_r, dtype=np.uint8)
        maps, cents = window_attn_maps(model, tower, processor, full_np, device)
        last_maps = maps
        for L in full_layers:
            st = analyze_layer(maps[L], cents)
            acc[L]["within"].append(st["within"])
            acc[L]["W"].append(st["W"])
            for k in (1, 2, 3):
                acc[L]["topk"][k].append(st["topk"][k])
            for d, v in st["dist_decay"].items():
                acc[L]["dd"].setdefault(d, []).append(v)
        if (c + 1) % 4 == 0:
            print(f"  [{c+1}/{len(indices)}] processed", flush=True)

    print(f"\n===== CROSS-WINDOW ATTENTION in the 4 full-attention layers "
          f"({args.dataset}, {args.model_id.split('/')[-1]}, N={len(indices)} images) =====")
    print(f"  (window=112px=4x4 merged tokens; within = attention staying in query's own window)")
    Wmean = np.mean([np.mean(acc[L]["W"]) for L in full_layers])
    print(f"  avg #windows/image = {Wmean:.0f}  ->  UNIFORM within-window baseline = {100/Wmean:.1f}% "
          f"(own window is 1 of {Wmean:.0f})")
    for L in full_layers:
        within = np.mean(acc[L]["within"]); W = np.mean(acc[L]["W"])
        tk = {k: 100 * np.mean(acc[L]["topk"][k]) for k in (1, 2, 3)}
        print(f"  layer {L:2d}:  within-window {100*within:5.1f}% ({within*W:.1f}x uniform)   |  "
              f"of CROSS mass  top-1 {tk[1]:.0f}%  top-2 {tk[2]:.0f}%  top-3 {tk[3]:.0f}%")
    allwithin = np.mean([np.mean(acc[L]["within"]) for L in full_layers])
    print(f"  --> avg within-window across the 4 full layers: {100*allwithin:.1f}%  "
          f"({allwithin*Wmean:.1f}x uniform -- >1 means locality even in the global layers)")
    print(f"\n  cross-window distance decay (layer 31, the deepest/most semantic):")
    dd = {d: np.mean(v) for d, v in sorted(acc[31]["dd"].items())}
    for d, v in list(dd.items())[:10]:
        print(f"      dist {d:>5}: {v:.4f}")

    if args.out_fig and last_maps is not None:
        try:
            import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 4, figsize=(17, 4.3))
            for ax, L in zip(axes, full_layers):
                im = ax.imshow(np.log10(last_maps[L] + 1e-4), cmap="magma")
                ax.set_title(f"full layer {L}  (log A_win)"); ax.set_xlabel("key window"); ax.set_ylabel("query window")
            fig.colorbar(im, ax=axes.tolist()); Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(args.out_fig, dpi=110, bbox_inches="tight")
            print(f"\n  figure saved -> {args.out_fig}  (last image's 4 window x window maps)")
        except Exception as e:
            print(f"  (fig skipped: {e})")


if __name__ == "__main__":
    main()
