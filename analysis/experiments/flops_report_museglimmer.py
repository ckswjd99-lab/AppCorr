"""Critical/total FLOPs for the Muse Glimmer streaming arm, via the axis's scopes.

Mirrors flops_report_qwen35.py: three-plus arms per sample (ceiling / floor / streaming at each
--keeps), counted with flops.session over the stock modules the fork calls into
(vision_tower + language_model + vision_adapter + vision_projection) -- session wraps
install + patch_attention, so the quadratic term is included by construction.

Run under the MG-capable checkout: --transformers-path /NHNHOME/share/cjpark/tf515.
FLOPs-first rule: reconcile crit/full (~1/g + last-band overhead) and total/full
(~ full + one vision pass) against the streaming closed form before any accuracy campaign.
"""
import argparse, json, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--transformers-path", default="/NHNHOME/share/cjpark/tf515")
ap.add_argument("--model", default="meta-models/Muse-Glimmer-30B")
ap.add_argument("--samples", type=int, default=12)
ap.add_argument("--groups", type=int, default=4)
ap.add_argument("--level", type=int, default=2)
ap.add_argument("--datasets", nargs="+", default=["refcoco", "textvqa"])
ap.add_argument("--keeps", type=float, nargs="+", default=[0.25, 0.50, 1.0])
ap.add_argument("--out-json", default="analysis/results/flops/museglimmer_arms_flops.json")
args = ap.parse_args()

sys.path.insert(0, args.transformers_path)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import AutoProcessor, AutoModelForImageTextToText  # noqa: E402
from qwen_vl_prefill.datasets_eval import get_spec  # noqa: E402
from appcorr import flops  # noqa: E402
from appcorr.models.museglimmer.unified import MuseGlimmerAxis  # noqa: E402

MG_MAX_PX = 3_147_760  # measured 2026-08-31: grid saturates at ~1540x2044


def degrade(img: Image.Image, level: int) -> Image.Image:
    """pyr L2 with MG's sampling cap -- same function as museglimmer_accuracy.py."""
    import cv2
    import numpy as np
    w, h = img.size
    s = min(1.0, (MG_MAX_PX / (w * h)) ** 0.5)
    w2, h2 = (max(1, int(w * s)), max(1, int(h * s))) if s < 1.0 else (w, h)
    arr = np.asarray(img if s == 1.0 else img.resize((w2, h2), Image.BILINEAR))
    sizes = [(arr.shape[1], arr.shape[0])]
    for _ in range(level):
        arr = cv2.pyrDown(arr)
        sizes.append((arr.shape[1], arr.shape[0]))
    for i in range(level - 1, -1, -1):
        arr = cv2.pyrUp(arr, dstsize=sizes[i])
    out = Image.fromarray(arr)
    return out if s == 1.0 else out.resize((w, h), Image.BICUBIC)


def main():
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype="auto", device_map="cuda:0").eval()
    roots = [model.model.vision_tower, model.model.language_model,
             model.model.vision_adapter, model.model.vision_projection]

    result = {"_model": args.model, "_samples": args.samples, "_groups": args.groups}
    for ds_name in args.datasets:
        spec = get_spec(ds_name)
        ds = spec.load(load_dataset)
        idxs = list(range(0, len(ds), max(1, len(ds) // args.samples)))[:args.samples]
        arms = ["ceiling", "floor"] + [f"streaming_k{k:.2f}" for k in args.keeps]
        aggs = {}
        for arm in arms:
            with flops.session(*roots, enabled=True) as fl:
                axis = MuseGlimmerAxis(model, proc, flop_counter=fl)
                for si, i in enumerate(idxs):
                    img, q, _ = spec.prepare(ds[int(i)], lambda h, w, **kw: (h, w), 1, 1, 1 << 30)
                    img = img.convert("RGB")
                    inputs = axis.build_inputs(img, q).to("cuda:0")
                    base_px = axis.build_inputs(degrade(img, args.level), q)["pixel_values"].to("cuda:0")
                    with torch.no_grad(), fl.request(f"{ds_name}/{si}"):
                        if arm == "ceiling":
                            axis.full_forward(inputs)
                        elif arm == "floor":
                            axis.approx_only_forward(inputs, base_px)
                        else:
                            axis.streaming_forward(inputs, base_px, args.groups,
                                                   keep=float(arm.split("_k")[1]))
            aggs[arm] = fl.aggregate()
        full = aggs["ceiling"]["mean_total_gflops"]
        row = {"full": round(full, 1), "floor": round(aggs["floor"]["mean_total_gflops"], 1)}
        for k in args.keeps:
            st = aggs[f"streaming_k{k:.2f}"]
            suffix = f"_g{args.groups}" if k == 1.0 else f"_g{args.groups}_k{k:.2f}"
            row[f"crit{suffix}"] = round(st["mean_critical_gflops"], 1)
            row[f"total{suffix}"] = round(st["mean_total_gflops"], 1)
            print(f"{ds_name:<14} k={k:.2f} full {full:9.1f}  "
                  f"streaming crit {st['mean_critical_gflops']:8.1f} "
                  f"total {st['mean_total_gflops']:9.1f}  crit/full = "
                  f"{st['mean_critical_gflops'] / full * 100:5.1f}%  total/full = "
                  f"{st['mean_total_gflops'] / full * 100:5.1f}%", flush=True)
        result[ds_name] = row

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(result, open(args.out_json, "w"), indent=1)
    print(f"wrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
