"""Accuracy for the Qwen3.5-35B streaming arm: floor / streaming(g) / ceiling, one driver.

Every arm decodes through the SAME mechanism -- prefill produces (last-position logits, cache),
then one shared greedy loop steps with explicit positions. This is not pedantry: on Qwen2.5-VL a
baseline that decoded via one continuous `generate()` while the corrected arms decoded via a
two-stage path disagreed with the corrected mechanism on 30-35% of RefCOCO samples with ZERO
correction involved (+2.25pp on 32B) -- the decode mechanism is a real confound and the fix is to
never vary it across arms.

Positions are explicit everywhere. The model silently falls back to cached `rope_deltas` from
whatever ran last on the module when position_ids are omitted in decode -- correct by coincidence,
wrong on reorder (see the axis gate's history).

Scoring is `spec.score()` from the shared registry -- the same normalization every other model's
numbers went through.
"""
import argparse, json, os, sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from qwen_vl_prefill.datasets_eval import get_spec
from datasets import load_dataset
from transformers import AutoProcessor, AutoModelForImageTextToText
from appcorr.models.qwen35.unified import Qwen35Axis, MODEL_ID_35B
from PIL import Image


def degrade(img: Image.Image, level: int = 2, filt: str = "bicubic") -> Image.Image:
    """The transmission's level-`level` base: 2^level x down, back up. Content degrades, geometry
    does not -- the token grid must match the full-res image or the band mixing is meaningless.

    `filt` selects the DOWNSAMPLING filter: 'bicubic' is what every qwen35 number in the table was
    measured with; 'box' (area average) matches the gemma3/ov2 oracles and approximates the
    canonical cv2.pyrDown pyramid more closely. The 2026-08-28 convention audit flagged the
    divergence; the BOX-vs-BICUBIC sensitivity probe decides whether the table needs re-measuring."""
    f = 2 ** level
    w, h = img.size
    if filt == "pyr":
        # The protocol archetype itself: cv2.pyrDown chain, cv2.pyrUp back with
        # per-step dstsize (odd dims round up on pyrDown; the stored size chain
        # restores them exactly, mirroring laplacian.py's _iterative_upsample_native).
        import cv2
        import numpy as np
        arr = np.asarray(img)
        sizes = [(arr.shape[1], arr.shape[0])]
        for _ in range(level):
            arr = cv2.pyrDown(arr)
            sizes.append((arr.shape[1], arr.shape[0]))
        for i in range(level - 1, -1, -1):
            arr = cv2.pyrUp(arr, dstsize=sizes[i])
        return Image.fromarray(arr)
    down = Image.BOX if filt == "box" else Image.BICUBIC
    return img.resize((max(1, w // f), max(1, h // f)), down).resize((w, h), Image.BICUBIC)


@torch.no_grad()
def greedy(axis, logits, cache, start_pos, n=24):
    """The one decode mechanism, shared by every arm."""
    toks, cur, pos = [], logits.argmax(-1, keepdim=True), start_pos
    eos = axis.processor.tokenizer.eos_token_id
    for _ in range(n):
        t = int(cur)
        if t == eos:
            break
        toks.append(t)
        pid = torch.full((3, 1, 1), pos, device=cur.device, dtype=torch.long)
        out = axis.model(input_ids=cur, past_key_values=cache, position_ids=pid, use_cache=True)
        cache = out.past_key_values
        cur = out.logits[:, -1].argmax(-1, keepdim=True)
        pos += 1
    return axis.processor.tokenizer.decode(toks, skip_special_tokens=True)


@torch.no_grad()
def prefill_stock(axis, inputs):
    """Stock prefill (floor and ceiling), returning the same (logits, cache, decode_pos) the
    streaming arm returns, so `greedy` cannot tell the arms apart."""
    ids = inputs["input_ids"]
    mm = inputs["mm_token_type_ids"]
    pos3, _ = axis.model.model.get_rope_index(ids, mm, image_grid_thw=inputs["image_grid_thw"])
    out = axis.model(input_ids=ids, pixel_values=inputs["pixel_values"].to(axis.model.dtype),
                     image_grid_thw=inputs["image_grid_thw"], mm_token_type_ids=mm,
                     position_ids=pos3, use_cache=True)
    return out.logits[:, -1], out.past_key_values, int(pos3.max().item()) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default=MODEL_ID_35B,
                    help="checkpoint id; e.g. Qwen/Qwen3.5-122B-A10B-FP8")
    ap.add_argument("--level", type=int, default=2, help="pyramid level of the degraded base")
    # Default flipped to box 2026-08-28 after the paired probe (4/50 flips, 3:1 toward box):
    # box matches the convention's reference (area average ~ pyramid level). Every degraded-arm
    # number measured before this date used bicubic and lives in analysis/results/qwen35_accuracy/;
    # box re-measurements go to qwen35_accuracy_box/ -- NEVER append across the boundary, the
    # jsonl resume would silently mix filters.
    ap.add_argument("--degrade-filter", choices=["bicubic", "box", "pyr"], default="box")
    ap.add_argument("--arms", nargs="+", default=["floor", "streaming", "ceiling"])
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--keep", type=float, default=1.0,
                    help="fraction of image tokens corrected (streaming arm); 1.0 = correct all")
    ap.add_argument("--samples", type=int, default=0, help="0 = full split")
    ap.add_argument("--out", default="analysis/results/qwen35_accuracy")
    args = ap.parse_args()

    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype="auto", device_map="cuda:0").eval()
    axis = Qwen35Axis(model, proc)
    spec = get_spec(args.dataset)
    ds = spec.load(load_dataset)
    n = len(ds) if args.samples == 0 else min(args.samples, len(ds))
    idxs = list(range(n)) if args.samples == 0 else \
        list(range(0, len(ds), max(1, len(ds) // n)))[:n]
    os.makedirs(args.out, exist_ok=True)

    for arm in args.arms:
        slug = "" if args.model == MODEL_ID_35B else "_" + args.model.split("/")[-1].lower()
        arm_suffix = ""
        if arm == "streaming":
            arm_suffix = f"_g{args.groups}"
            if args.keep < 1.0:
                arm_suffix += f"_k{args.keep:.2f}"
        path = os.path.join(args.out, f"{args.dataset}{slug}_{arm}{arm_suffix}.jsonl")
        done = set()
        if os.path.exists(path):
            with open(path) as f:
                done = {json.loads(l)["i"] for l in f if l.strip()}
        correct, scored = 0, 0
        f = open(path, "a")
        for i in idxs:
            if i in done:
                continue
            img, q, gold = spec.prepare(ds[int(i)], lambda h, w, **kw: (h, w), 1, 1, 1 << 30)
            if img.mode != "RGB":
                img = img.convert("RGB")
            try:
                inputs = axis.build_inputs(img, q).to("cuda:0")
                if arm == "ceiling":
                    lg, kv, dp = prefill_stock(axis, inputs)
                elif arm == "floor":
                    base_inputs = axis.build_inputs(degrade(img, args.level, args.degrade_filter), q).to("cuda:0")
                    lg, kv, dp = prefill_stock(axis, base_inputs)
                else:
                    base_px = axis.build_inputs(degrade(img, args.level, args.degrade_filter), q)["pixel_values"].to("cuda:0")
                    lg, kv, st = axis.streaming_forward(inputs, base_px, args.groups,
                                                        keep=args.keep)
                    dp = st["decode_start_pos"]
                pred = greedy(axis, lg, kv, dp)
                try:
                    ok, val = spec.score(pred, gold)
                except NotImplementedError:  # wildvision: judge-only prediction dump
                    ok, val = 0, None
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                f.write(json.dumps({"i": int(i), "skip": "oom"}) + "\n")
                f.flush()
                continue
            correct += ok
            scored += 1
            f.write(json.dumps({"i": int(i), "pred": pred, "gold": gold, "ok": int(ok),
                                "val": (float(val) if val is not None else None)}) + "\n")
            if scored % 50 == 0:
                f.flush()
                print(f"[{arm}] {scored} scored, running {correct / scored * 100:.2f}%", flush=True)
        f.close()
        if scored:
            print(f"Final Summary: {{\"dataset\": \"{args.dataset}\", \"arm\": \"{arm}\", "
                  f"\"scored\": {scored}, \"acc\": {correct / scored * 100:.4f}}}", flush=True)
    print("QWEN35_ACCURACY_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
