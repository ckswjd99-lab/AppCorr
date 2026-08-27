"""Model-agnostic VLM bounds oracle: ceiling / floor via stock generate.

For any HF image-text-to-text model (AutoProcessor + AutoModelForImageTextToText
+ chat template): ceiling = stock generate on the full image, floor = stock
generate on the level-2 degraded image at the SAME size (grid preserved by
construction; pixel shapes asserted equal). This is exactly the bounds recipe
every per-model oracle uses -- factored out so a NEW model (Muse Glimmer,
Mistral Small) gets its table bounds the day it loads, before any axis exists.

WildVision has no reference answers (spec.score raises NotImplementedError):
predictions are dumped with score=null and the summary carries accuracy=null --
a judge pass consumes the dump later.

--transformers-path prepends an alternate transformers build (e.g. the 5.15.dev
tree Muse Glimmer needs) WITHOUT touching the env the running campaigns use.

Run: CUDA_VISIBLE_DEVICES=0 python analysis/experiments/vlm_bounds_oracle.py \
    --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 --dataset mmvp --arm ceiling --full
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "qwen_vl_prefill"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--arm", choices=["ceiling", "floor"], required=True)
    ap.add_argument("--level", type=int, default=2, help="degradation factor 2^level")
    ap.add_argument("--num-samples", type=int, default=12)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--transformers-path", default=None)
    ap.add_argument("--degrade-max-px", type=int, default=None,
                    help="model-sampled pixel area cap for the pyramid-direction rule "
                         "(e.g. Mistral/Pixtral 1540*1540; Gemma4 2520*256). None = native.")
    ap.add_argument("--reasoning-strength", default=None,
                    help="template kwarg for channel-protocol models (Muse Glimmer ATEM: "
                         "low/medium/high/xhigh; default template value is high, which fills "
                         "short generations with the to=self reasoning channel)")
    a = ap.parse_args()

    if a.transformers_path:
        sys.path.insert(0, a.transformers_path)
    import torch
    import transformers
    from PIL import Image
    from datasets import load_dataset
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from datasets_eval import get_spec
    print(f"[bounds] transformers {transformers.__version__} | {a.model} | "
          f"{a.dataset}/{a.arm}", flush=True)

    model = AutoModelForImageTextToText.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="cuda:0").eval()
    proc = AutoProcessor.from_pretrained(a.model)

    def degrade(img):
        # Pyramid convention (AGENTS.md): BOX down (area average = pyramid level),
        # BICUBIC up, and degrade relative to what the model SAMPLES when the
        # native image is larger (--degrade-max-px caps the area; long-side caps
        # go through the same area formula: pass side*side for square canvases).
        w, h = img.size
        f = 2 ** a.level
        s = 1.0
        if a.degrade_max_px:
            s = min(1.0, (a.degrade_max_px / (w * h)) ** 0.5)
        tw = max(1, int(w * s) // f)
        th = max(1, int(h * s) // f)
        return img.resize((tw, th), Image.BOX).resize((w, h), Image.BICUBIC)

    spec = get_spec(a.dataset)
    ds = spec.load(load_dataset)
    if len(ds) == 0:
        raise RuntimeError("VACUOUS: 0 samples")
    n = len(ds) if a.full else min(a.num_samples, len(ds))
    idxs = (list(range(len(ds))) if a.full
            else list(range(0, len(ds), max(1, len(ds) // n)))[:n])

    correct, scored, total, per = 0, 0, 0, []
    t0 = time.time()
    for idx in idxs:
        img, prompt, gold = spec.prepare(ds[idx], lambda h, w, **kw: (h, w), 1, 1, 1 << 30)
        use = degrade(img) if a.arm == "floor" else img
        msgs = [{"role": "user", "content": [{"type": "image", "image": use},
                                             {"type": "text", "text": prompt}]}]
        tmpl_kw = {}
        if a.reasoning_strength:
            tmpl_kw["reasoning_strength"] = a.reasoning_strength
        enc = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors="pt",
                                       **tmpl_kw).to("cuda:0")
        if "pixel_values" in enc and enc["pixel_values"].is_floating_point():
            enc["pixel_values"] = enc["pixel_values"].to(torch.bfloat16)
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False)
        text = proc.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        # Channel-protocol models (Muse Glimmer ATEM) emit a `to=self` reasoning
        # channel before the user-facing answer; score only the to=user channel.
        # The MMVP 49%-vs-chance incident: with the default high reasoning
        # strength and short generations, the WHOLE output was reasoning echo and
        # the letter regex harvested "(a)" from the echoed question.
        if "to=user" in text:
            text = text.split("to=user")[-1].strip()
        elif text.startswith("to=self"):
            text = text.split("\n")[-1].strip()
        try:
            ok, sc = spec.score(text, gold)
            correct += ok
            scored += 1
        except NotImplementedError:
            ok, sc = None, None
        total += 1
        per.append({"idx": idx, "pred": text, "gold": str(gold)[:120], "score": sc})
        if total % 25 == 0 or total == len(idxs):
            dt = time.time() - t0
            acc = f"acc={correct / scored:.2%}" if scored else "dump-only"
            print(f"  [{total}/{len(idxs)}] {dt:.0f}s {dt / total:.2f}s/ex  {acc}", flush=True)

    summary = {"model": a.model, "dataset": a.dataset, "arm": a.arm,
               "level": a.level, "num_samples": total,
               "accuracy": (correct / scored) if scored else None, "correct": correct}
    print(f"\n=== Final Summary: {json.dumps(summary)}", flush=True)
    if a.out_json:
        os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
        json.dump({"summary": summary, "per_sample": per}, open(a.out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
