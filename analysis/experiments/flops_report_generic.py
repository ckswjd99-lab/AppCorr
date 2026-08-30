"""Full-forward FLOPs for any HF image-text-to-text model (bounds-only rows).

Counts ONE stock prefill (no decode) per sample via the standard hooks -- fills
the Full-res Comp. cell for models that have bounds but no axis yet (Mistral
Small, Muse Glimmer). Floor compute equals full compute for these models by
construction (same grid, same token count -- the level-2 base changes values,
not shapes), so only `full` is emitted.

Run: python analysis/experiments/flops_report_generic.py \
    --model mistralai/Mistral-Small-3.1-24B-Instruct-2503 --tag mistral24b --datasets mmvp cvbench
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "qwen_vl_prefill"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True, help="inprocess_flops.json model key")
    ap.add_argument("--datasets", nargs="+", default=["mmvp", "cvbench"])
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--transformers-path", default=None)
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()
    if a.transformers_path:
        sys.path.insert(0, a.transformers_path)

    import torch
    from datasets import load_dataset
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from appcorr.flops.counter import FlopCounter
    from appcorr.flops import hooks
    from datasets_eval import get_spec

    model = AutoModelForImageTextToText.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="cuda:0").eval()
    proc = AutoProcessor.from_pretrained(a.model)
    # Everything but the LM head, matching every other report's accounting.
    roots = [model.model] if hasattr(model, "model") else [model]

    result = {"_model": a.model, "_samples": a.samples}
    for ds_name in a.datasets:
        spec = get_spec(ds_name)
        ds = spec.load(load_dataset)
        idxs = list(range(0, len(ds), max(1, len(ds) // a.samples)))[:a.samples]
        c = FlopCounter()
        for si, idx in enumerate(idxs):
            img, prompt, _ = spec.prepare(ds[idx], lambda h, w, **kw: (h, w), 1, 1, 1 << 30)
            msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                                 {"type": "text", "text": prompt}]}]
            enc = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors="pt").to("cuda:0")
            if "pixel_values" in enc and enc["pixel_values"].is_floating_point():
                enc["pixel_values"] = enc["pixel_values"].to(torch.bfloat16)
            handles = hooks.install(c, roots)
            # patch_attention required -- install() alone drops the SDPA term (2026-08-31).
            with torch.no_grad(), hooks.patch_attention(c), c.request(f"{ds_name}/{si}"), \
                    c.arrival(0):
                model.model(**{k: v for k, v in enc.items()}) if hasattr(model, "model") \
                    else model(**enc)
            hooks.remove(handles)
        agg = c.aggregate()
        result[ds_name] = {"full": round(agg["mean_total_gflops"], 1)}
        print(f"{a.tag} {ds_name:<10} full {agg['mean_total_gflops']:9.1f} GF", flush=True)

    out = a.out_json or f"analysis/results/flops/{a.tag}_flops.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(result, open(out, "w"), indent=1)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
