"""
causal_order.py -- does PERMUTING the causal order of the visual-token prefill (while keeping each
token's M-RoPE position correct) hurt? Motivated by the 2D-dependence finding: if benign, the
streaming system can prefill visual tokens in arrival/dependence order, not locked to raster.

M-RoPE makes attention depend on relative 2D POSITION, not sequence index, so reordering the visual
tokens in the sequence AND carrying their position_ids along keeps every token at its true position;
only the causal MASK (who may attend to whom) changes. Text tokens stay last (they always see all
visual tokens). identity == the stock raster order (sanity/baseline).

Permutations (on the merged grid, raster index i = r*mw + c):
  identity  : raster (stock)                    reverse  : raster reversed (max disruption)
  colmajor  : column-major (== the "1-3-2-4" example)
  random    : fixed-seed shuffle                block<bs>: bs x bs block-raster (2D-locality-preserving)

Run: python qwen_vl_prefill/causal_order.py --dataset realworldqa --full --device cuda:0 \
        --out-jsonl /tmp/co_rwqa.jsonl
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


def build_perms(mh, mw, block_size=4):
    n = mh * mw
    idx = np.arange(n)
    r, c = idx // mw, idx % mw
    colmajor = np.lexsort((r, c))                       # sort by (c, then r)
    rng = np.random.RandomState(0).permutation(n)
    # bs x bs block-raster: order by (block_row, block_col, in-block r, in-block c)
    br, bc = r // block_size, c // block_size
    block = np.lexsort((c, r, bc, br))
    return {"identity": idx, "reverse": idx[::-1].copy(),
            "colmajor": colmajor, "random": rng, f"block{block_size}": block}


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--dataset", default="realworldqa", choices=list(DE.SPECS))
    ap.add_argument("--num-samples", type=int, default=64)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--shard", default=None)
    ap.add_argument("--block-size", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--out-jsonl", default=None)
    args = ap.parse_args()
    dev = args.device

    print(f"[causal-order] loading {args.model_id} ...")
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
    order_names = ["identity", "reverse", "colmajor", "random", f"block{args.block_size}"]
    correct = {o: 0 for o in order_names}
    score = {o: 0.0 for o in order_names}

    for c_i, di in enumerate(indices):
        img, prompt, gold = spec.prepare(ds[di], smart_resize, factor, minp, maxp)
        prepared = I.prepare_inputs(model, proc, img, prompt, device=dev)
        pid0 = I.compute_position_ids(model, prepared)
        vemb, _ = PR.encode_image(model, proc, np.array(img, np.uint8), dev)
        ie0 = I.build_inputs_embeds(model, prepared, vemb)
        _, hp, wp = [int(x) for x in prepared.image_grid_thw[0].tolist()]
        mh, mw = hp // 2, wp // 2
        span = prepared.image_mask[0].nonzero(as_tuple=True)[0]
        a, b = int(span[0]), int(span[-1]) + 1
        perms = build_perms(mh, mw, args.block_size)

        row = {"idx": di}
        for name in order_names:
            pt = torch.as_tensor(perms[name].copy(), device=dev)
            ie = ie0.clone(); ie[:, a:b] = ie0[:, a:b][:, pt]
            pid = pid0.clone(); pid[:, :, a:b] = pid0[:, :, a:b][:, :, pt]
            ids = PR.greedy_generate_from_embeds(model, ie, pid, args.max_new_tokens, eos_ids, dev)
            out = proc.tokenizer.decode(ids, skip_special_tokens=True)
            ok, sc = spec.score(out, gold)
            correct[name] += ok; score[name] += sc
            row[name + "_ok"] = ok; row[name + "_sc"] = sc
        if out_f:
            out_f.write(_json.dumps(row) + "\n"); out_f.flush()
        if (c_i + 1) % 32 == 0:
            msg = "  ".join(f"{o}={100*correct[o]/(c_i+1):.1f}" for o in order_names)
            print(f"  [{c_i+1}/{len(indices)}] {msg}", flush=True)

    n = len(indices)
    metric = "mean_iou" if args.dataset == "refcoco" else "mean_score"
    print(f"\n===== CAUSAL-ORDER PERMUTATION ({args.dataset}, N={n}) =====")
    base = correct["identity"]
    for o in order_names:
        print(f"  {o:9s}: acc {100*correct[o]/n:6.2f}%   {metric} {score[o]/n:.4f}   "
              f"vs identity {100*(correct[o]-base)/n:+.2f}pp")


if __name__ == "__main__":
    main()
