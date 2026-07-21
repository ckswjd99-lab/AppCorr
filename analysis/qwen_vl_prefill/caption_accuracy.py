"""
caption_accuracy.py -- image captioning (COCO Captions / Flickr30k) under progressive correction.
Generates a caption for each image via full / base(approx-only) / progressive(appcorr) visual embeds
(shared vision correction machinery), then scores each mode with CIDEr-D at the end (corpus-level, so
it can't use the per-sample driver). Reports all three (approx-only / appcorr / baseline) + delta.

Run: python qwen_vl_prefill/caption_accuracy.py --dataset coco --device cuda:0 --out-jsonl /tmp/coco.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

for _p in Path(__file__).resolve().parents[1:3]:  # analysis/ (qwen_vl_prefill) + repo root (appcorr)
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from qwen_vl_prefill import introspect as I
from qwen_vl_prefill import progressive as PR
from qwen_vl_prefill import progressive_correct as PC
from qwen_vl_prefill.cider import compute_cider

SETS = {
    "coco": dict(hf="lmms-lab/COCO-Caption2017", cfg=None, split="val", refs_key="answer"),
    "flickr30k": dict(hf="lmms-lab/flickr30k", cfg=None, split="test", refs_key="caption"),
}
PROMPT = "Provide a one-sentence caption for the image."


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--dataset", required=True, choices=list(SETS))
    ap.add_argument("--num-samples", type=int, default=0, help="0 = full")
    ap.add_argument("--num-groups", type=int, default=4)
    ap.add_argument("--overlap", type=int, default=0)
    ap.add_argument("--base-factor", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-new-tokens", type=int, default=40)
    ap.add_argument("--out-jsonl", default=None)
    args = ap.parse_args()
    dev = args.device
    spec = SETS[args.dataset]

    print(f"[caption] loading {args.model_id} ...")
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

    ds = load_dataset(spec["hf"], spec["cfg"], split=spec["split"]) if spec["cfg"] else \
        load_dataset(spec["hf"], split=spec["split"])
    n_total = len(ds)
    indices = list(range(n_total)) if args.num_samples <= 0 else \
        list(range(0, n_total, max(n_total // args.num_samples, 1)))[:args.num_samples]

    out_f = open(args.out_jsonl, "a", encoding="utf-8") if args.out_jsonl else None
    caps = {"full": {}, "base": {}, "progressive": {}}
    refs = {}

    for c_i, di in enumerate(indices):
        ex = ds[di]
        image = ex["image"].convert("RGB")
        th, tw = smart_resize(image.height, image.width, factor=factor, min_pixels=minp, max_pixels=maxp)
        image_r = image.resize((tw, th), __import__("PIL").Image.BILINEAR)
        full_np = np.array(image_r, dtype=np.uint8)
        base_np = PR.make_base(full_np, args.base_factor)
        prepared = I.prepare_inputs(model, proc, image_r, PROMPT, device=dev)
        pid = I.compute_position_ids(model, prepared)
        bands = PR.band_layout(prepared.image_grid_thw, merge_size, ip.patch_size, args.num_groups)
        prog_e, full_e, base_e = PC.progressive_corrected_embeds(
            model, tower, proc, full_np, base_np, bands, dev, overlap=args.overlap)

        row = {"idx": di, "refs": list(ex[spec["refs_key"]])}
        refs[di] = row["refs"]
        for name, e in (("full", full_e), ("base", base_e), ("progressive", prog_e)):
            ie = I.build_inputs_embeds(model, prepared, e)
            ids_ = PR.greedy_generate_from_embeds(model, ie, pid, args.max_new_tokens, eos_ids, dev)
            cap = proc.tokenizer.decode(ids_, skip_special_tokens=True).strip()
            caps[name][di] = cap
            row[name] = cap
        if out_f:
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n"); out_f.flush()
        if (c_i + 1) % 50 == 0:
            print(f"  [{c_i+1}/{len(indices)}]", flush=True)

    print(f"\n===== CAPTIONING CIDEr ({args.dataset}, N={len(indices)}, G={args.num_groups}, overlap={args.overlap}) =====")
    cider = {name: compute_cider(caps[name], refs)[0] for name in ("full", "base", "progressive")}
    print(f"  baseline (full)     CIDEr: {cider['full']*100:.2f}")
    print(f"  approx-only (base)  CIDEr: {cider['base']*100:.2f}   (delta {100*(cider['base']-cider['full']):+.2f})")
    print(f"  appcorr (progressive) CIDEr: {cider['progressive']*100:.2f}   (delta {100*(cider['progressive']-cider['full']):+.2f})")


if __name__ == "__main__":
    main()
