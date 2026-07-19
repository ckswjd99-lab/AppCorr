"""
block_accuracy.py -- compare 1D-band vs 2D-block progressive correction (with spatial-neighbor
overlap) on a dataset. Does 2D-block granularity + a 2D-NEAREST-neighbor refresh recover more of the
grounding cheap-correction overhead than the 1D trailing-band scheme?

Runs a fixed sweep of configs per image (sharing the encode + base approx pass), all through the
IDENTICAL raster LLM prefill so only the vision correction differs:
  band_o{0,1,2}  : P=4 Q=1 horizontal bands, trailing overlap (== progressive_correct baseline)
  blk44_o{0,1,2,3}: P=4 Q=4 2D blocks, nearest-neighbor overlap

Run: python qwen_vl_prefill/block_accuracy.py --dataset refcoco --num-samples 1000 --device cuda:0
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen_vl_prefill import introspect as I
from qwen_vl_prefill import progressive as PR
from qwen_vl_prefill import progressive_correct as PC
from qwen_vl_prefill import block_correct as BC
from qwen_vl_prefill import datasets_eval as DE

# (name, P, Q, overlap, policy)
CONFIGS = [
    ("band_o0", 4, 1, 0, "trailing"),
    ("band_o1", 4, 1, 1, "trailing"),
    ("band_o2", 4, 1, 2, "trailing"),
    ("blk44_o0", 4, 4, 0, "nearest"),
    ("blk44_o1", 4, 4, 1, "nearest"),
    ("blk44_o2", 4, 4, 2, "nearest"),
    ("blk44_o3", 4, 4, 3, "nearest"),
]


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--dataset", default="refcoco", choices=list(DE.SPECS))
    ap.add_argument("--num-samples", type=int, default=1000)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--shard", default=None)
    ap.add_argument("--base-factor", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--out-jsonl", default=None)
    ap.add_argument("--configs", default=None, help="comma-separated config names to run (default all)")
    args = ap.parse_args()
    dev = args.device

    global CONFIGS
    if args.configs:
        keep = set(args.configs.split(","))
        CONFIGS = [c for c in CONFIGS if c[0] in keep]

    print(f"[block-acc] loading {args.model_id} ...")
    model, proc = I.load_model(args.model_id, device=dev)
    tower = PC.build_tower(model)
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    from datasets import load_dataset
    ip = proc.image_processor
    merge_size = ip.merge_size
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

    out_f = open(args.out_jsonl, "a", encoding="utf-8") if args.out_jsonl else None
    modes = ["full", "base"] + [c[0] for c in CONFIGS]
    correct = {m: 0 for m in modes}
    sc_sum = {m: 0.0 for m in modes}

    def gen_score(embeds, prepared, pid, gold):
        ie = I.build_inputs_embeds(model, prepared, embeds)
        ids = PR.greedy_generate_from_embeds(model, ie, pid, args.max_new_tokens, eos_ids, dev)
        return spec.score(proc.tokenizer.decode(ids, skip_special_tokens=True), gold)

    for c_i, di in enumerate(indices):
        img, prompt, gold = spec.prepare(ds[di], smart_resize, factor, minp, maxp)
        full_np = np.array(img, dtype=np.uint8)
        base_np = PR.make_base(full_np, args.base_factor)
        prepared = I.prepare_inputs(model, proc, img, prompt, device=dev)
        pid = I.compute_position_ids(model, prepared)
        setup = BC.setup_correction(model, tower, proc, full_np, base_np, dev)

        row = {"idx": di}
        for name, e in (("full", setup["full_embeds"]), ("base", setup["base_embeds"])):
            ok, sc = gen_score(e, prepared, pid, gold)
            correct[name] += ok; sc_sum[name] += sc; row[name + "_ok"] = ok; row[name + "_sc"] = sc
        for (name, P, Q, ov, pol) in CONFIGS:
            groups, centers = BC.block_groups(setup["grid"], merge_size, P, Q)
            rs = BC.build_refresh_sets(centers, ov, pol)
            prog = BC.run_correction_config(tower, setup, groups, rs, dev)
            ok, sc = gen_score(prog, prepared, pid, gold)
            correct[name] += ok; sc_sum[name] += sc; row[name + "_ok"] = ok; row[name + "_sc"] = sc
        if out_f:
            out_f.write(json.dumps(row) + "\n"); out_f.flush()
        if (c_i + 1) % 32 == 0:
            msg = "  ".join(f"{m}={100*correct[m]/(c_i+1):.1f}" for m in modes[:4])
            print(f"  [{c_i+1}/{len(indices)}] {msg}", flush=True)

    n = len(indices)
    metric = "iou" if args.dataset == "refcoco" else "score"
    print(f"\n===== BLOCK vs BAND CORRECTION ({args.dataset}, N={n}) =====")
    fa = correct["full"]
    for m in modes:
        print(f"  {m:9s}: acc {100*correct[m]/n:6.2f}%   {metric} {sc_sum[m]/n:.4f}   "
              f"vs full {100*(correct[m]-fa)/n:+.2f}pp")


if __name__ == "__main__":
    main()
