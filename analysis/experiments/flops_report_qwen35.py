"""Critical/total FLOPs for the Qwen3.5-35B streaming arm, via the unified axis's scopes.

Three arms per sample, matching every other model's report:
    ceiling    stock forward on the full image          -> 100% critical
    floor      stock forward on the degraded base       -> 100% critical
    streaming  vision approx (arrival 0) + per-band correct+prefill (arrivals 1..g)
               -> critical = the final band's vision correct + its prefill chunk

MoE note: expert FLOPs are COUNT-based, not data-dependent -- the handler charges
`top_k_index.numel() = n_tok * top_k` whichever experts the router hits, so two prompts of equal
length cost identically. (An earlier draft of this docstring claimed the opposite; the measured
floor == full to 0.1 GF is what corrected it -- same grid, same token count, same cost.) Routing
DOES change which weights are touched, which matters for memory traffic, not for FLOPs.

Base degradation is the transmission's own level-2 pyramid base: downsample 4x, upsample back.
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from appcorr.flops.counter import FlopCounter
from appcorr.flops import hooks
from appcorr.models.qwen35.unified import Qwen35Axis, MODEL_ID_35B

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from qwen_vl_prefill.datasets_eval import get_spec  # the registry every other report uses


def load_samples(name, n):
    """Via the shared spec registry -- it loads from the local HF cache, so it works with
    HF_HUB_OFFLINE=1. The first version of this file re-invented loading with `streaming=True`,
    which needs the network by design and died at the first sample; use the machinery that the
    Gemma 3 and Qwen 2.5 reports already run on."""
    from datasets import load_dataset
    spec = get_spec(name)
    ds = spec.load(load_dataset)
    idxs = list(range(0, len(ds), max(1, len(ds) // n)))[:n]
    out = []
    for i in idxs:
        # Identity smart_resize (the gemma3 report's convention here): the processor applies this
        # model's own resolution policy anyway, so pre-resizing to Qwen2.5's would double-resize.
        img, q, _gold = spec.prepare(ds[int(i)], lambda h, w, **kw: (h, w), 1, 1, 1 << 30)
        out.append((img, q))
    return out


def degrade(img: Image.Image) -> Image.Image:
    w, h = img.size
    return img.resize((max(1, w // 4), max(1, h // 4)), Image.BICUBIC).resize((w, h), Image.BICUBIC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID_35B)
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--datasets", nargs="+", default=["chartqa"])
    ap.add_argument("--keeps", type=float, nargs="+", default=[1.0],
                    help="streaming keep ratios to measure; 1.0 reproduces the original arm")
    ap.add_argument("--out-json", default="analysis/results/flops/qwen35_flops.json")
    args = ap.parse_args()

    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype="auto", device_map="cuda:0").eval()

    result = {"_model": args.model, "_samples": args.samples, "_groups": args.groups}
    for ds_name in args.datasets:
        samples = load_samples(ds_name, args.samples)
        arms = ["ceiling", "floor"] + [f"streaming_k{k:.2f}" for k in args.keeps]
        counters = {k: FlopCounter() for k in arms}
        axis_by = {k: Qwen35Axis(model, proc, flop_counter=c) for k, c in counters.items()}
        # Hooks installed per arm inside the loop (visual + language model); lm_head excluded
        # as everywhere.
        for si, (img, q) in enumerate(samples):
            base = degrade(img)
            inputs = axis_by["ceiling"].build_inputs(img, q).to("cuda:0")
            inputs_base = axis_by["ceiling"].build_inputs(base, q).to("cuda:0")
            with torch.no_grad():
                for arm in arms:
                    c = counters[arm]
                    axis = axis_by[arm]
                    handles = hooks.install(c, [model.model.visual, model.model.language_model])
                    with c.request(f"{ds_name}/{si}"):
                        if arm == "ceiling":
                            axis.full_forward(inputs)
                        elif arm == "floor":
                            axis.approx_only_forward(inputs, inputs_base["pixel_values"])
                        else:
                            axis.streaming_forward(inputs, inputs_base["pixel_values"],
                                                   args.groups, keep=float(arm.split("_k")[1]))
                    hooks.remove(handles)
        agg = {k: c.aggregate() for k, c in counters.items()}
        full = agg["ceiling"]["mean_total_gflops"]
        row = {"full": round(full, 1),
               "floor": round(agg["floor"]["mean_total_gflops"], 1)}
        for k in args.keeps:
            st = agg[f"streaming_k{k:.2f}"]
            # k=1.00 keeps the original key names so existing consumers keep reading them.
            suffix = f"_g{args.groups}" if k == 1.0 else f"_g{args.groups}_k{k:.2f}"
            row[f"crit{suffix}"] = round(st["mean_critical_gflops"], 1)
            row[f"total{suffix}"] = round(st["mean_total_gflops"], 1)
            print(f"{ds_name:<14} k={k:.2f} full {full:9.1f}  floor {row['floor']:9.1f}  "
                  f"streaming crit {st['mean_critical_gflops']:8.1f} "
                  f"total {st['mean_total_gflops']:9.1f}  crit/full = "
                  f"{st['mean_critical_gflops'] / full * 100:5.1f}%")
        result[ds_name] = row

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(result, open(args.out_json, "w"), indent=1)
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
