"""Vision/LLM FLOPs split for the qwen35 streaming closed-form reconciliation.

The streaming arm's expected shape, in terms the report already measures:

    total_k1.00 - full  ~=  V         (the extra approx pass over the base; same grid, so the
                                       correction re-walk is the full vision cost once more)
    crit_k1.00          in  [(V + L_lin)/g,  V/g + L_lin/g + L_attn/2]
                                       (last band: 1/g of vision + 1/g of the LLM's linear work;
                                       causal attention for the last chunk lies between L_attn/g
                                       and L_attn/2 depending on where its keys start)

V is measured by running ONLY the vision tower under the hooks (patch_attention is a global SDPA
patch, so the module must be run in isolation for clean attribution -- running full_forward with
hooks on `visual` would still pour the LLM's attention into the same counter). L is then
`full - V` against the report's measured full.
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from transformers import AutoProcessor, AutoModelForImageTextToText
from appcorr.flops.counter import FlopCounter
from appcorr.flops import hooks
from appcorr.models.qwen35.unified import Qwen35Axis, MODEL_ID_35B

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from flops_report_qwen35 import load_samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID_35B)
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--datasets", nargs="+", default=["refcoco", "textvqa", "chartqa"])
    args = ap.parse_args()

    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype="auto", device_map="cuda:0").eval()
    axis = Qwen35Axis(model, proc)

    out = {"_model": args.model, "_samples": args.samples}
    for ds_name in args.datasets:
        samples = load_samples(ds_name, args.samples)
        cv = FlopCounter()
        for si, (img, q) in enumerate(samples):
            inputs = axis.build_inputs(img, q).to("cuda:0")
            with torch.no_grad():
                h = hooks.install(cv, [model.model.visual])
                with hooks.patch_attention(cv), cv.request(f"{ds_name}/{si}"):
                    model.model.visual(inputs["pixel_values"].to(model.dtype),
                                       grid_thw=inputs["image_grid_thw"])
                hooks.remove(h)
        v = cv.aggregate()["mean_total_gflops"]
        out[ds_name] = {"V": round(v, 1)}
        print(f"{ds_name:<12} V {v:9.1f}", flush=True)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
