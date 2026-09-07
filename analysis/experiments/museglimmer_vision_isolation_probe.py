"""Isolate the MG identity-gate drift: vision fork vs LLM chunked prefill.

Compares, per sample:
  A. stock vision:   model.model.get_image_features(px, grid).pooler_output  (cat'd)
  B. fork vision:    prepare_full_tokens -> approx_forward(full depth) -> merge_groups(0, n)
If A~B to bf16 noise, the gate's TV drift lives in the LLM chunked-prefill path (the known
sliding-layer boundary-numerics class); if A!=B, the vision fork has a real discrepancy.
"""
import argparse, os, sys

ap = argparse.ArgumentParser()
ap.add_argument("--transformers-path", default="/NHNHOME/share/cjpark/tf515")
ap.add_argument("--model", default="meta-models/Muse-Glimmer-30B")
ap.add_argument("--samples", type=int, default=3)
ap.add_argument("--dataset", default="realworldqa")
args = ap.parse_args()

sys.path.insert(0, args.transformers_path)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))

import torch  # noqa: E402
from transformers import AutoProcessor, AutoModelForImageTextToText  # noqa: E402
from datasets import load_dataset  # noqa: E402
from qwen_vl_prefill.datasets_eval import get_spec  # noqa: E402
from appcorr.models.museglimmer.unified import MuseGlimmerAxis  # noqa: E402

proc = AutoProcessor.from_pretrained(args.model)
model = AutoModelForImageTextToText.from_pretrained(
    args.model, dtype="auto", device_map="cuda:1" if os.environ.get("CUDA_VISIBLE_DEVICES") is None else "cuda:0").eval()
axis = MuseGlimmerAxis(model, proc)

spec = get_spec(args.dataset)
ds = spec.load(load_dataset)
idxs = list(range(0, len(ds), max(1, len(ds) // args.samples)))[:args.samples]

for i in idxs:
    img, q, _ = spec.prepare(ds[int(i)], lambda h, w, **kw: (h, w), 1, 1, 1 << 30)
    inputs = axis.build_inputs(img.convert("RGB"), q).to(model.device)
    px = inputs["pixel_values"].to(model.dtype)
    grid = inputs["image_grid_thw"]
    with torch.no_grad():
        stock = model.model.get_image_features(px, grid).pooler_output
        stock = torch.cat(stock, dim=0).float()
        ctx = axis.tower.prepare_full_tokens(px, grid)
        cache = {}
        x, cache = axis.tower.approx_forward(ctx["hidden_states"], 0, len(axis.tower.blocks),
                                             ctx, cache, "v")
        n_groups = ctx["seq_len"] // axis.tower.spatial_merge_unit
        fork = axis.tower.merge_groups(x, ctx, 0, n_groups).float()
    d = (stock - fork).abs()
    rel = float(d.norm() / stock.norm())
    print(f"[{i}] rows={tuple(stock.shape)} max|d|={float(d.max()):.4e} rel-L2={rel:.4e} "
          f"cos={float(torch.nn.functional.cosine_similarity(stock.flatten(), fork.flatten(), dim=0)):.6f}",
          flush=True)
print("MG_VISION_ISOLATION_DONE", flush=True)
