"""
progressive_accuracy.py -- accuracy of PROGRESSIVE VISUAL-TOKEN FINALIZATION on RefCOCO grounding.

Measures where progressive finalization (early visual-token groups frozen from a partial, coarser
image; bidirectional staleness) lands relative to the full-image baseline and the base-only worst
case. All three modes generate through the IDENTICAL LLM path (greedy decode from custom
inputs_embeds) so only the visual tokens differ:

  full        : visual tokens from the full image                       (baseline / upper bound)
  base_only   : visual tokens from the coarse base only                 (worst case)
  progressive : band g's tokens re-encoded from "full in bands 1..g,    (the modification)
                base below", monotonically frozen (no correction yet -- re-encoding = upper bound
                of what a cheap first-order correction could achieve)

Run:
    python qwen_vl_prefill/progressive_accuracy.py --model-id Qwen/Qwen2.5-VL-3B-Instruct \
        --num-samples 64 --num-groups 4 --base-factor 4 --device cuda:0
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from qwen_vl_prefill import introspect as I
from qwen_vl_prefill import progressive as PR

GROUNDING_PROMPT = (
    'Locate the region described by: "{expr}". Output ONLY the bounding box as four numbers '
    "x1,y1,x2,y2 (top-left and bottom-right pixel coordinates in this image), with no other text."
)
NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_bbox(text):
    nums = NUM_RE.findall(text)
    if len(nums) < 4:
        return None
    x1, y1, x2, y2 = (float(v) for v in nums[:4])
    return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def iou(a, b):
    if a is None:
        return 0.0
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--num-samples", type=int, default=64)
    ap.add_argument("--full", action="store_true", help="use all RefCOCO val samples (ignores --num-samples)")
    ap.add_argument("--shard", default=None, help="'k/N' -- process only shard k of N contiguous slices (for multi-GPU)")
    ap.add_argument("--num-groups", type=int, default=4)
    ap.add_argument("--base-factor", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--out-jsonl", default=None, help="append per-sample (idx, mode ious/correct) rows for aggregation")
    args = ap.parse_args()
    device = args.device

    print(f"[prog-acc] loading {args.model_id} ...")
    model, processor = I.load_model(args.model_id, device=device)
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    from datasets import load_dataset
    from PIL import Image

    ip = processor.image_processor
    merge_size, patch_size = ip.merge_size, ip.patch_size
    factor = patch_size * merge_size * 4
    min_px, max_px = ip.size["shortest_edge"], ip.size["longest_edge"]
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos) if isinstance(eos, (list, tuple)) else {eos}

    ds = load_dataset("lmms-lab/RefCOCO", split="val")
    n_total = len(ds)
    if args.full:
        indices = list(range(n_total))
    else:
        stride = max(n_total // args.num_samples, 1)
        indices = list(range(0, n_total, stride))[:args.num_samples]
    if args.shard is not None:
        k, N = (int(x) for x in args.shard.split("/"))
        per = (len(indices) + N - 1) // N
        indices = indices[k * per:(k + 1) * per]
        print(f"[prog-acc] shard {k}/{N}: {len(indices)} samples")

    import json as _json
    out_f = open(args.out_jsonl, "a", encoding="utf-8") if args.out_jsonl else None

    modes = ["full", "base_only", "progressive"]
    correct = {m: 0 for m in modes}
    iou_sum = {m: 0.0 for m in modes}

    for c, idx in enumerate(indices):
        ex = ds[idx]
        image = ex["image"].convert("RGB")
        expr = ex["answer"][0] if isinstance(ex["answer"], list) and ex["answer"] else str(ex["answer"])
        bx, by, bw, bh = ex["bbox"]
        th, tw = smart_resize(image.height, image.width, factor=factor, min_pixels=min_px, max_pixels=max_px)
        sx, sy = tw / image.width, th / image.height
        gt = (bx * sx, by * sy, (bx + bw) * sx, (by + bh) * sy)
        image_r = image.resize((tw, th), Image.BILINEAR)
        full_np = np.array(image_r, dtype=np.uint8)
        base_np = PR.make_base(full_np, args.base_factor)
        prompt = GROUNDING_PROMPT.format(expr=expr)

        # prompt scaffolding (input_ids / position_ids / image layout) -- same for all 3 modes
        prepared = I.prepare_inputs(model, processor, image_r, prompt, device=device)
        position_ids = I.compute_position_ids(model, prepared)
        bands = PR.band_layout(prepared.image_grid_thw, merge_size, patch_size, args.num_groups)

        prog_e, full_e, base_e = PR.progressive_finalized_embeds(model, processor, full_np, base_np, bands, device)
        # guard: token counts must match the prompt's image slots
        assert full_e.shape[0] == prepared.n_visual_tokens, (full_e.shape, prepared.n_visual_tokens)

        row = {"idx": idx}
        for m, embeds in (("full", full_e), ("base_only", base_e), ("progressive", prog_e)):
            inputs_embeds = I.build_inputs_embeds(model, prepared, embeds)
            ids = PR.greedy_generate_from_embeds(model, inputs_embeds, position_ids,
                                                 args.max_new_tokens, eos_ids, device)
            text = processor.tokenizer.decode(ids, skip_special_tokens=True)
            i = iou(parse_bbox(text), gt)
            iou_sum[m] += i
            correct[m] += int(i > 0.5)
            row[m + "_iou"] = i
            row[m + "_correct"] = int(i > 0.5)
        if out_f is not None:
            out_f.write(_json.dumps(row) + "\n"); out_f.flush()

        if (c + 1) % 16 == 0:
            msg = "  ".join(f"{m}={100*correct[m]/(c+1):.1f}%" for m in modes)
            print(f"  [{c+1}/{len(indices)}] {msg}", flush=True)

    n = len(indices)
    print(f"\n===== PROGRESSIVE FINALIZATION ACCURACY (RefCOCO, N={n}, G={args.num_groups}, base_factor={args.base_factor}) =====")
    for m in modes:
        print(f"  {m:12s}: acc@0.5 = {100*correct[m]/n:6.2f}%  ({correct[m]}/{n})  mean_iou={iou_sum[m]/n:.4f}")
    df = 100 * (correct["progressive"] - correct["full"]) / n
    db = 100 * (correct["progressive"] - correct["base_only"]) / n
    print(f"\n  progressive vs full:      {df:+.2f}pp   (staleness cost of freezing early groups)")
    print(f"  progressive vs base_only: {db:+.2f}pp   (recovery from re-encoding residual bands)")
    print(f"\n  Note: progressive uses actual re-encoding (upper bound); a cheap first-order correction")
    print(f"  (Phase 5) would land at or below this. nr<=400 is a sanity check -- confirm on full data.")


if __name__ == "__main__":
    main()
