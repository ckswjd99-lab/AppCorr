"""
reverse_order_sanity_sweep.py -- 4-arm floor/ceiling sanity sweep for the reverse-order (bottom-up)
depth-staggered LLM prefill correction (`reverse_order_correct.py`), vs. the existing top-down
(causal, free-parallelism) scheme.

Every arm uses the SAME `reverse_order_correct.native_pyramid`-built (full_np, base_np) pair (fixes
the AGENTS.md pyramid-construction bug the OLD `progressive.make_base`-on-canvas-resolution driver
had -- see that module's docstring), so all 4 numbers are directly comparable to each other, though
NOT bit-comparable to the older README.md figures (which used the canvas-resolution base).

Arms (floor and ceiling bracket the two techniques being compared, per standing methodology: always
report recovered % of the floor->ceiling gap, not an isolated number):

  l2_approx_only : L2-level (quarter-res native Laplacian base, upsampled back) image, encoded once,
                   NO correction anywhere (vision or LLM). Floor -- the worst case if no residual
                   detail ever arrived.
  l2l0_causal    : today's scheme. Vision tower progressively corrected (existing per-band cheap
                   correction, UNCHANGED), then EXACT causal LLM prefill in raster order (g1 top ..
                   g4 bottom) -- exploits causality for free, zero extra LLM compute, bit-exact vs
                   monolithic prefill given the vision tokens it's handed.
  l2l0_reverse   : NEW. Same vision-side correction, but LLM prefill corrected in REVERSE spatial
                   order (g4 bottom .. g1 top) against depth checkpoints growing L/G at a time (see
                   `reverse_order_correct.py`). NOT exactness-preserving by design.
  l0_full        : stock full-resolution encoding, no approximation anywhere. Ceiling -- the
                   unmodified model's own performance; not what our technique is compared against in
                   isolation, but the top of the recoverable gap.

Run (nr=64 sanity first, per this repo's established discipline -- confirm the effect direction and
magnitude are sane before ever queuing a full-dataset run):
    conda activate appcorr
    python analysis/qwen_vl_prefill/reverse_order_sanity_sweep.py \
        --model-id Qwen/Qwen2.5-VL-3B-Instruct --dataset refcoco --num-samples 64 --device cuda:0
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

for _p in Path(__file__).resolve().parents[1:3]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from qwen_vl_prefill import introspect as I
from qwen_vl_prefill import progressive as PR
from qwen_vl_prefill import progressive_correct as PC
from qwen_vl_prefill import datasets_eval as DE
from qwen_vl_prefill import reverse_order_correct as RC

ARMS = ["l2_approx_only", "l2l0_causal", "l2l0_reverse", "l0_full"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--dataset", default="refcoco", choices=list(DE.SPECS))
    ap.add_argument("--num-samples", type=int, default=64)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--shard", default=None, help="'k/N' -- contiguous shard k of N (multi-GPU)")
    ap.add_argument("--num-groups", type=int, default=4)
    ap.add_argument("--base-factor", type=int, default=4, help="L2 = downsample-by-4 (H/4 x W/4 native)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--out-jsonl", default=None)
    args = ap.parse_args()
    device = args.device

    print(f"[sweep] loading {args.model_id} ...")
    model, processor = I.load_model(args.model_id, device=device)
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    ip = processor.image_processor
    merge_size, patch_size = ip.merge_size, ip.patch_size
    factor = patch_size * merge_size * 4
    min_px, max_px = ip.size["shortest_edge"], ip.size["longest_edge"]
    eos = model.generation_config.eos_token_id
    eos_ids = set(eos) if isinstance(eos, (list, tuple)) else {eos}

    tower = PC.build_tower(model)
    llm_layers = RC.build_llm_layers(model)
    L = len(llm_layers)
    assert L % args.num_groups == 0, f"L={L} not divisible by num_groups={args.num_groups}"

    spec = DE.get_spec(args.dataset)
    ds = spec.load(__import__("datasets").load_dataset)
    n_total = len(ds)
    if args.full:
        indices = list(range(n_total))
    else:
        stride = max(n_total // args.num_samples, 1)
        indices = list(range(0, n_total, stride))[: args.num_samples]
    if args.shard is not None:
        k, N = (int(x) for x in args.shard.split("/"))
        per = (len(indices) + N - 1) // N
        indices = indices[k * per : (k + 1) * per]
        print(f"[sweep] shard {k}/{N}: {len(indices)} samples")

    out_f = open(args.out_jsonl, "a", encoding="utf-8") if args.out_jsonl else None
    correct = {m: 0 for m in ARMS}
    score_sum = {m: 0.0 for m in ARMS}

    for c, idx in enumerate(indices):
        ex = ds[idx]
        # native-resolution image + prompt/gold. Each spec's prepare() resizes internally to compute
        # `gold` in the resized frame; we need the SAME (th,tw) target, so call it once to get gold
        # and the prompt, but build full_np/base_np ourselves via native_pyramid (native-first). Most
        # specs expose the native image directly at ex["image"]; VSR loads it from a local cache path.
        image_r_ref, prompt, gold = spec.prepare(ex, smart_resize, factor, min_px, max_px)
        if args.dataset == "vsr":
            from qwen_vl_prefill.datasets_eval import _VSR_IMG_DIR
            from PIL import Image as _PILImage
            import os as _os
            image_native = _PILImage.open(_os.path.join(_VSR_IMG_DIR, ex["image"])).convert("RGB")
        elif "image" in ex:
            image_native = ex["image"].convert("RGB")
        else:
            image_native = image_r_ref
        full_np, base_np = RC.native_pyramid(image_native, args.base_factor, smart_resize, factor, min_px, max_px)
        assert full_np.shape[:2] == np.array(image_r_ref).shape[:2], "native_pyramid canvas size mismatch vs spec.prepare"
        image_r = __import__("PIL.Image", fromlist=["Image"]).fromarray(full_np)

        prepared = I.prepare_inputs(model, processor, image_r, prompt, device=device)
        position_ids = I.compute_position_ids(model, prepared)
        bands = PR.band_layout(prepared.image_grid_thw, merge_size, patch_size, args.num_groups)

        with torch.inference_mode():
            pv_full, grid = PC.encode_pixels(processor, full_np, device, model.dtype)
            pv_base, grid_b = PC.encode_pixels(processor, base_np, device, model.dtype)
            assert torch.equal(grid, grid_b)

            base_e = PC.stock_embeds(model, pv_base, grid)
            full_e = PC.stock_embeds(model, pv_full, grid)
            prog_e, _full_ref, _base_ref = PC.progressive_corrected_embeds(
                model, tower, processor, full_np, base_np, bands, device, overlap=0
            )

            row = {"idx": idx}
            texts = {}

            # l2_approx_only (floor)
            ie = I.build_inputs_embeds(model, prepared, base_e)
            ids = PR.greedy_generate_from_embeds(model, ie, position_ids, args.max_new_tokens, eos_ids, device)
            texts["l2_approx_only"] = processor.tokenizer.decode(ids, skip_special_tokens=True)

            # l2l0_causal (existing top-down scheme)
            ie = I.build_inputs_embeds(model, prepared, prog_e)
            ids = PR.greedy_generate_from_embeds(model, ie, position_ids, args.max_new_tokens, eos_ids, device)
            texts["l2l0_causal"] = processor.tokenizer.decode(ids, skip_special_tokens=True)

            # l2l0_reverse (new)
            last_hidden, cache_feature, layout = RC.reverse_order_prefill(
                model, tower, llm_layers, processor, prepared, position_ids, full_np, base_np, bands, device
            )
            ids = RC.greedy_decode_from_cache(
                model, last_hidden, cache_feature, L, layout["N"], position_ids, args.max_new_tokens, eos_ids, device
            )
            texts["l2l0_reverse"] = processor.tokenizer.decode(ids, skip_special_tokens=True)

            # l0_full (ceiling)
            ie = I.build_inputs_embeds(model, prepared, full_e)
            ids = PR.greedy_generate_from_embeds(model, ie, position_ids, args.max_new_tokens, eos_ids, device)
            texts["l0_full"] = processor.tokenizer.decode(ids, skip_special_tokens=True)

        for m in ARMS:
            ok, sc = spec.score(texts[m], gold)
            correct[m] += ok
            score_sum[m] += sc
            row[m + "_text"] = texts[m]
            row[m + "_correct"] = ok
            row[m + "_score"] = sc
        if out_f is not None:
            out_f.write(json.dumps(row) + "\n")
            out_f.flush()

        if (c + 1) % 16 == 0:
            msg = "  ".join(f"{m}={100*correct[m]/(c+1):.1f}%" for m in ARMS)
            print(f"  [{c+1}/{len(indices)}] {msg}", flush=True)

    n = len(indices)
    metric = "mean_iou" if args.dataset == "refcoco" else "mean_score"
    print(f"\n===== REVERSE-ORDER SANITY SWEEP ({args.dataset}, N={n}, "
          f"G={args.num_groups}, base_factor(L2)={args.base_factor}) =====")
    for m in ARMS:
        print(f"  {m:15s}: acc = {100*correct[m]/n:6.2f}%  ({correct[m]}/{n})  {metric}={score_sum[m]/n:.4f}")

    floor, ceil_ = correct["l2_approx_only"], correct["l0_full"]
    gap = ceil_ - floor
    print(f"\n  floor->ceiling gap: {100*gap/n:+.2f}pp  ({floor}/{n} -> {ceil_}/{n})")
    for m in ("l2l0_causal", "l2l0_reverse"):
        vs_full = 100 * (correct[m] - ceil_) / n
        vs_floor = 100 * (correct[m] - floor) / n
        recovered = 100 * (correct[m] - floor) / gap if gap != 0 else float("nan")
        print(f"  {m:15s}: vs full = {vs_full:+.2f}pp   vs floor = {vs_floor:+.2f}pp   "
              f"recovers {recovered:.1f}% of the floor->ceiling gap")
    print("\n  (nr sanity check only -- confirm direction/magnitude before queuing a full-dataset run.)")


if __name__ == "__main__":
    main()
