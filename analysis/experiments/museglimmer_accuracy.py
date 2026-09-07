"""Accuracy for the Muse Glimmer streaming arm: floor / streaming(g, keep) / ceiling, one driver.

Adapted from qwen35_accuracy.py; the same non-negotiables carry over:
  * Every arm decodes through ONE mechanism (prefill -> shared greedy with explicit positions).
  * jsonl append/resume per arm; a row is {"i", "pred", "gold", "ok", "val"} (+"pred_raw" when a
    box rescale ran, so any coordinate-convention correction can be re-scored offline).
  * pyr L2 degradation (Option B), degraded relative to what the model SAMPLES: MG's processor
    caps at ~3.15M px (measured 2026-08-31: 4032x3024 and 2688x2016 both land on grid 110x146 =
    1540x2044 sampled), so MG_MAX_PX = 3_147_760 plays the role qwen35's 12.8M cap plays there.

MG specifics:
  * Run with the MG-capable transformers checkout: --transformers-path /NHNHOME/share/cjpark/tf515
    (inserted on sys.path BEFORE transformers imports).
  * ATEM channel protocol lives in MuseGlimmerAxis.build_inputs (reasoning_strength=low + forced
    ` to=user<|message|>` tail) -- identical for every arm, so it cancels in comparisons.
  * Grounding coordinate convention is UNPROBED as of writing: --box-scale defaults to "none"
    (raw pred scored as-is). Run the ceiling coord probe first (--samples 20 --arms ceiling,
    inspect preds vs image sizes), then set --box-scale thousand|frac for the campaign. The raw
    string is stored either way.
  * No batching in v1 (MG's vision is pure softmax so the qwen35 bounds-batching port would be
    legal, but it needs its own equivalence gate first).
"""
import argparse, json, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", required=True)
ap.add_argument("--model", default="meta-models/Muse-Glimmer-30B")
ap.add_argument("--transformers-path", default="/NHNHOME/share/cjpark/tf515")
ap.add_argument("--level", type=int, default=2)
ap.add_argument("--degrade-filter", choices=["bicubic", "box", "pyr"], default="pyr")
ap.add_argument("--arms", nargs="+", default=["floor", "streaming", "ceiling"])
ap.add_argument("--groups", type=int, default=4)
ap.add_argument("--keep", type=float, default=1.0)
ap.add_argument("--samples", type=int, default=0, help="0 = full split")
ap.add_argument("--contiguous", action="store_true")
ap.add_argument("--box-scale", choices=["none", "thousand", "frac"], default="none",
                help="rescale for refcoco/visdrone_det preds: 'thousand' = 0-1000 relative, "
                     "'frac' = 0-1 fractions; decide via the ceiling coord probe first")
ap.add_argument("--max-new-tokens", type=int, default=24)
ap.add_argument("--out", default="analysis/results/museglimmer_accuracy_pyr")
args = ap.parse_args()

sys.path.insert(0, args.transformers_path)                # BEFORE transformers imports
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import AutoProcessor, AutoModelForImageTextToText  # noqa: E402
from qwen_vl_prefill.datasets_eval import get_spec  # noqa: E402
from appcorr.models.museglimmer.unified import MuseGlimmerAxis  # noqa: E402

MG_MAX_PX = 3_147_760


def degrade(img: Image.Image, level: int, filt: str) -> Image.Image:
    """qwen35_accuracy.degrade with MG's sampling cap (pyramid-direction rule)."""
    f = 2 ** level
    w, h = img.size
    s = min(1.0, (MG_MAX_PX / (w * h)) ** 0.5)
    w2, h2 = (max(1, int(w * s)), max(1, int(h * s))) if s < 1.0 else (w, h)
    if filt == "pyr":
        import cv2
        import numpy as np
        arr = np.asarray(img if s == 1.0 else img.resize((w2, h2), Image.BILINEAR))
        sizes = [(arr.shape[1], arr.shape[0])]
        for _ in range(level):
            arr = cv2.pyrDown(arr)
            sizes.append((arr.shape[1], arr.shape[0]))
        for i in range(level - 1, -1, -1):
            arr = cv2.pyrUp(arr, dstsize=sizes[i])
        out = Image.fromarray(arr)
        return out if s == 1.0 else out.resize((w, h), Image.BICUBIC)
    down = Image.BOX if filt == "box" else Image.BICUBIC
    return img.resize((max(1, w2 // f), max(1, h2 // f)), down).resize((w, h), Image.BICUBIC)


@torch.no_grad()
def greedy(axis, logits, cache, start_pos, n):
    """The one decode mechanism, shared by every arm. 1D positions.

    Stop set includes the ATEM channel-protocol message/turn terminators <|eom|>/<|eot|> --
    stopping only on <|end_of_text|> let the decode run past the answer into the to=self
    channel, which poisoned whole-string scorers (textvqa read 0.00% while the answers were
    visibly correct). MCQ first-letter and first-number scorers were never affected."""
    tok = axis.processor.tokenizer
    stop = {tok.eos_token_id}
    for st in ("<|eom|>", "<|eot|>"):
        i = tok.convert_tokens_to_ids(st)
        if i is not None:
            stop.add(int(i))
    toks, cur, pos = [], logits.argmax(-1, keepdim=True), start_pos
    for _ in range(n):
        t = int(cur)
        if t in stop:
            break
        toks.append(t)
        pid = torch.full((1, 1), pos, device=cur.device, dtype=torch.long)
        out = axis.model(input_ids=cur, past_key_values=cache, position_ids=pid, use_cache=True)
        cache = out.past_key_values
        cur = out.logits[:, -1].argmax(-1, keepdim=True)
        pos += 1
    return axis.processor.tokenizer.decode(toks, skip_special_tokens=True)


@torch.no_grad()
def prefill_stock(axis, inputs):
    """Stock prefill (floor and ceiling) with explicit 1D positions, matching the streaming arm's
    position convention so `greedy` cannot tell the arms apart."""
    ids = inputs["input_ids"]
    seq = ids.shape[1]
    pos = torch.arange(seq, device=ids.device).unsqueeze(0)
    out = axis.model(input_ids=ids, pixel_values=inputs["pixel_values"].to(axis.model.dtype),
                     image_grid_thw=inputs["image_grid_thw"], position_ids=pos, use_cache=True)
    return out.logits[:, -1], out.past_key_values, seq


def main():
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype="auto", device_map="cuda:0").eval()
    axis = MuseGlimmerAxis(model, proc)
    spec = get_spec(args.dataset)
    ds = spec.load(load_dataset)
    n = len(ds) if args.samples == 0 else min(args.samples, len(ds))
    idxs = list(range(n)) if (args.samples == 0 or args.contiguous) else \
        list(range(0, len(ds), max(1, len(ds) // n)))[:n]
    os.makedirs(args.out, exist_ok=True)

    for arm in args.arms:
        arm_suffix = ""
        if arm == "streaming":
            arm_suffix = f"_g{args.groups}"
            if args.keep < 1.0:
                arm_suffix += f"_k{args.keep:.2f}"
        path = os.path.join(args.out, f"{args.dataset}_{arm}{arm_suffix}.jsonl")
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
                if arm == "ceiling":
                    inputs = axis.build_inputs(img, q).to("cuda:0")
                    lg, kv, dp = prefill_stock(axis, inputs)
                elif arm == "floor":
                    base_inputs = axis.build_inputs(
                        degrade(img, args.level, args.degrade_filter), q).to("cuda:0")
                    lg, kv, dp = prefill_stock(axis, base_inputs)
                else:
                    inputs = axis.build_inputs(img, q).to("cuda:0")
                    base_px = axis.build_inputs(
                        degrade(img, args.level, args.degrade_filter), q)["pixel_values"].to("cuda:0")
                    lg, kv, st = axis.streaming_forward(inputs, base_px, args.groups,
                                                        keep=args.keep)
                    dp = st["decode_start_pos"]
                pred = greedy(axis, lg, kv, dp, args.max_new_tokens)
                pred_raw = None
                if args.box_scale != "none" and args.dataset in ("refcoco", "visdrone_det"):
                    import re as _re
                    # MG emits JSON-style boxes ({"x1": 185, ...}) and may leak into the to=self
                    # channel after the answer; strip key tokens (their digits poison a bare
                    # number scan -- the coord probe read 0.00% from exactly that) and cut at the
                    # channel switch before extracting. Convention: 0-1000 relative (probe
                    # 2026-09-01: /1000*(w,h) maximizes IoU across hypotheses).
                    cleaned = _re.sub(r'"?[xy][12]"?\s*[:=]', " ", pred).split("assistant")[0]
                    nums = _re.findall(r"-?\d+\.?\d*", cleaned)[:4]
                    if len(nums) == 4:
                        pred_raw = pred
                        w_, h_ = img.size
                        sc = (1000.0, 1000.0) if args.box_scale == "thousand" else (1.0, 1.0)
                        x1, y1, x2, y2 = (float(v) for v in nums)
                        pred = (f"{x1 * w_ / sc[0]:.1f},{y1 * h_ / sc[1]:.1f},"
                                f"{x2 * w_ / sc[0]:.1f},{y2 * h_ / sc[1]:.1f}")
                try:
                    ok, val = spec.score(pred, gold)
                except NotImplementedError:
                    ok, val = 0, None
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                f.write(json.dumps({"i": int(i), "skip": "oom"}) + "\n")
                f.flush()
                continue
            correct += ok
            scored += 1
            row = {"i": int(i), "pred": pred, "gold": gold, "ok": int(ok),
                   "val": (float(val) if val is not None else None)}
            if pred_raw is not None:
                row["pred_raw"] = pred_raw
            f.write(json.dumps(row) + "\n")
            if scored % 50 == 0:
                f.flush()
                print(f"[{arm}] {scored} scored, running {correct / scored * 100:.2f}%", flush=True)
        f.close()
        if scored:
            print(f"Final Summary: {{\"dataset\": \"{args.dataset}\", \"arm\": \"{arm}\", "
                  f"\"scored\": {scored}, \"acc\": {correct / scored * 100:.4f}}}", flush=True)
    print("MUSEGLIMMER_ACCURACY_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
