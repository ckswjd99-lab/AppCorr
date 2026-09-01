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

for _p in Path(__file__).resolve().parents[1:3]:  # analysis/ (qwen_vl_prefill) + repo root (appcorr)
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from qwen_vl_prefill import introspect as I
from qwen_vl_prefill import progressive as PR
from qwen_vl_prefill import progressive_correct as PC
from qwen_vl_prefill import datasets_eval as DE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--dataset", default="realworldqa", choices=list(DE.SPECS),
                    help="realworldqa (765, VQA, default) or refcoco (8811, grounding IoU)")
    ap.add_argument("--num-samples", type=int, default=64)
    ap.add_argument("--full", action="store_true", help="use all samples (ignores --num-samples)")
    ap.add_argument("--shard", default=None, help="'k/N' -- process only shard k of N contiguous slices (for multi-GPU)")
    ap.add_argument("--method", default="correct", choices=["correct", "reencoding"],
                    help="correct = cheap first-order fork (Phase 5, real); reencoding = re-encode upper bound (Phase 4+6)")
    ap.add_argument("--num-groups", type=int, default=4)
    ap.add_argument("--overlap", type=int, default=0,
                    help="method=correct only: trailing bands re-refreshed per round (0=plain cheap)")
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

    tower = PC.build_tower(model) if args.method == "correct" else None

    spec = DE.get_spec(args.dataset)
    ds = spec.load(load_dataset)
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
        image_r, prompt, gold = spec.prepare(ds[idx], smart_resize, factor, min_px, max_px)
        full_np = np.array(image_r, dtype=np.uint8)
        base_np = PR.make_base(full_np, args.base_factor)

        # prompt scaffolding (input_ids / position_ids / image layout) -- same for all 3 modes
        prepared = I.prepare_inputs(model, processor, image_r, prompt, device=device)
        position_ids = I.compute_position_ids(model, prepared)
        bands = PR.band_layout(prepared.image_grid_thw, merge_size, patch_size, args.num_groups)

        if args.method == "correct":
            prog_e, full_e, base_e = PC.progressive_corrected_embeds(model, tower, processor, full_np, base_np, bands, device, overlap=args.overlap)
        else:
            prog_e, full_e, base_e = PR.progressive_finalized_embeds(model, processor, full_np, base_np, bands, device)
        # guard: token counts must match the prompt's image slots
        assert full_e.shape[0] == prepared.n_visual_tokens, (full_e.shape, prepared.n_visual_tokens)

        row = {"idx": idx}
        for m, embeds in (("full", full_e), ("base_only", base_e), ("progressive", prog_e)):
            inputs_embeds = I.build_inputs_embeds(model, prepared, embeds)
            ids = PR.greedy_generate_from_embeds(model, inputs_embeds, position_ids,
                                                 args.max_new_tokens, eos_ids, device)
            text = processor.tokenizer.decode(ids, skip_special_tokens=True)
            ok, sc = spec.score(text, gold)
            iou_sum[m] += sc
            correct[m] += ok
            row[m + "_iou"] = sc
            row[m + "_correct"] = ok
        if out_f is not None:
            out_f.write(_json.dumps(row) + "\n"); out_f.flush()

        if (c + 1) % 16 == 0:
            msg = "  ".join(f"{m}={100*correct[m]/(c+1):.1f}%" for m in modes)
            print(f"  [{c+1}/{len(indices)}] {msg}", flush=True)

    n = len(indices)
    metric = "mean_iou" if args.dataset == "refcoco" else "mean_score"
    print(f"\n===== PROGRESSIVE FINALIZATION ACCURACY ({args.dataset}, method={args.method}, "
          f"overlap={args.overlap}, N={n}, G={args.num_groups}, base_factor={args.base_factor}) =====")
    for m in modes:
        print(f"  {m:12s}: acc = {100*correct[m]/n:6.2f}%  ({correct[m]}/{n})  {metric}={iou_sum[m]/n:.4f}")
    df = 100 * (correct["progressive"] - correct["full"]) / n
    db = 100 * (correct["progressive"] - correct["base_only"]) / n
    print(f"\n  progressive vs full:      {df:+.2f}pp   (staleness cost of freezing early groups)")
    print(f"  progressive vs base_only: {db:+.2f}pp   (recovery from correcting residual bands)")
    if args.method == "correct":
        print(f"\n  method=correct: cheap first-order fork correction (Phase 5, the real system).")
        print(f"  Compare against method=reencoding (the accuracy upper bound) to see the correction gap.")
    else:
        print(f"\n  method=reencoding: actual re-encoding = UPPER BOUND; the cheap correction lands at/below this.")


if __name__ == "__main__":
    main()
