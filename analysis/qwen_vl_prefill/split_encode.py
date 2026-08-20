"""
split_encode.py -- are the Qwen2.5-VL visual tokens location-agnostic enough to encode image TILES
independently and stitch? Split the image into a P x Q grid, run EACH tile through the vision encoder
SEPARATELY, then place the resulting merged tokens back into full-image raster order and feed the LLM
(which re-assigns global 2D positions via M-RoPE). vs encoding the whole image once.

Why it can work: Qwen2.5-VL vision uses 2D RoPE (relative positions) and NO learned absolute position
embedding; patch_embed/merger are position-agnostic. So a tile's tokens depend on its within-tile
relative structure, identical whether encoded standalone or as part of the full image -- the ONLY
thing lost by separate encoding is CROSS-TILE attention (which happens only in the 4 full-attention
layers; the 28 windowed layers are already local). Tiles are cut on 28px (merge-unit) boundaries so
merge groups stay intact. P=Q=1 == full image (sanity: bit-exact).

Run: python qwen_vl_prefill/split_encode.py --dataset realworldqa --splits 1x1,2x2,4x4 --num-samples 200 --device cuda:0
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

for _p in Path(__file__).resolve().parents[1:3]:  # analysis/ (qwen_vl_prefill) + repo root (appcorr)
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from qwen_vl_prefill import introspect as I
from qwen_vl_prefill import progressive as PR
from qwen_vl_prefill import datasets_eval as DE

MERGE_PX = 28  # patch_size(14) * merge_size(2)


def _edges(total, n):
    """n+1 split boundaries (pixels), each a multiple of MERGE_PX so merge groups stay intact."""
    e = [round(total * k / n / MERGE_PX) * MERGE_PX for k in range(n + 1)]
    e[0], e[-1] = 0, total
    return e


@torch.inference_mode()
def encode_splits(model, processor, full_np, P, Q, device):
    """Encode each P x Q tile separately, stitch merged tokens into full-image raster order [Nv, H]."""
    H, W = full_np.shape[:2]
    assert H % MERGE_PX == 0 and W % MERGE_PX == 0, (H, W)
    mw = W // MERGE_PX
    r_px, c_px = _edges(H, P), _edges(W, Q)
    stitched = None
    for i in range(P):
        for j in range(Q):
            crop = full_np[r_px[i]:r_px[i + 1], c_px[j]:c_px[j + 1]]
            emb, grid = PR.encode_image(model, processor, crop, device)  # [nt, dim], tile raster
            _, hp, wp = [int(x) for x in grid[0].tolist()]
            lh, lw = hp // 2, wp // 2
            assert lh == (r_px[i + 1] - r_px[i]) // MERGE_PX and lw == (c_px[j + 1] - c_px[j]) // MERGE_PX, \
                (lh, lw, (r_px[i + 1] - r_px[i]) // MERGE_PX, (c_px[j + 1] - c_px[j]) // MERGE_PX)
            if stitched is None:
                stitched = torch.empty((mw * (H // MERGE_PX), emb.shape[1]), dtype=emb.dtype, device=device)
            gr0, gc0 = r_px[i] // MERGE_PX, c_px[j] // MERGE_PX
            loc = torch.arange(lh * lw, device=device)
            gidx = (gr0 + loc // lw) * mw + (gc0 + loc % lw)
            stitched[gidx] = emb
    return stitched


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--dataset", default="realworldqa", choices=list(DE.SPECS))
    ap.add_argument("--splits", default="1x1,2x2,4x4", help="comma list like 1x1,1x2,2x2,4x4")
    ap.add_argument("--num-samples", type=int, default=200)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--shard", default=None)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--out-jsonl", default=None)
    args = ap.parse_args()
    dev = args.device
    splits = [(int(s.split("x")[0]), int(s.split("x")[1])) for s in args.splits.split(",")]
    names = args.splits.split(",")

    print(f"[split] loading {args.model_id} ...")
    model, proc = I.load_model(args.model_id, device=dev)
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    from datasets import load_dataset
    ip = proc.image_processor
    factor = ip.patch_size * ip.merge_size * 4
    minp, maxp = ip.size["shortest_edge"], ip.size["longest_edge"]
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos) if isinstance(eos, (list, tuple)) else {eos}

    spec = DE.get_spec(args.dataset)
    ds = spec.load(load_dataset)
    if args.full:
        indices = list(range(len(ds)))
    else:
        stride = max(len(ds) // args.num_samples, 1)
        indices = list(range(0, len(ds), stride))[:args.num_samples]
    if args.shard is not None:
        k, N = (int(x) for x in args.shard.split("/"))
        per = (len(indices) + N - 1) // N
        indices = indices[k * per:(k + 1) * per]

    import json as _json
    out_f = open(args.out_jsonl, "a", encoding="utf-8") if args.out_jsonl else None
    correct = {nm: 0 for nm in names}
    sc_sum = {nm: 0.0 for nm in names}

    for c_i, di in enumerate(indices):
        img, prompt, gold = spec.prepare(ds[di], smart_resize, factor, minp, maxp)
        full_np = np.array(img, dtype=np.uint8)
        prepared = I.prepare_inputs(model, proc, img, prompt, device=dev)
        pid = I.compute_position_ids(model, prepared)
        row = {"idx": di}
        for (P, Q), nm in zip(splits, names):
            emb = encode_splits(model, proc, full_np, P, Q, dev)
            assert emb.shape[0] == prepared.n_visual_tokens, (emb.shape, prepared.n_visual_tokens)
            ie = I.build_inputs_embeds(model, prepared, emb)
            ids = PR.greedy_generate_from_embeds(model, ie, pid, args.max_new_tokens, eos_ids, dev)
            ok, sc = spec.score(proc.tokenizer.decode(ids, skip_special_tokens=True), gold)
            correct[nm] += ok; sc_sum[nm] += sc; row[nm + "_ok"] = ok; row[nm + "_sc"] = sc
        if out_f:
            out_f.write(_json.dumps(row) + "\n"); out_f.flush()
        if (c_i + 1) % 32 == 0:
            print(f"  [{c_i+1}/{len(indices)}] " + "  ".join(f"{nm}={100*correct[nm]/(c_i+1):.1f}" for nm in names), flush=True)

    n = len(indices)
    metric = "iou" if args.dataset == "refcoco" else "score"
    print(f"\n===== SPLIT-ENCODE ({args.dataset}, N={n}) =====   (1x1 = full-image baseline)")
    base = correct[names[0]]
    for nm in names:
        print(f"  {nm:6s}: acc {100*correct[nm]/n:6.2f}%  {metric} {sc_sum[nm]/n:.4f}  vs 1x1 {100*(correct[nm]-base)/n:+.2f}pp")


if __name__ == "__main__":
    main()
