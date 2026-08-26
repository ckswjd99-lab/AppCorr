"""Gate for OV2's progressive-selection walk: keep=1.0, g=1 must reproduce the exact forward.

Same construction identity as the Gemma 3 gate, same llm_finish contract (walks return pre-finish
state). Direction check at g=4, k=0.25.
"""
import os, sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from experiments.ov2_oracle import encode, load, patch_energy
from experiments.flops_report_ov2 import l2_from_native, hw_from_grid
from qwen_vl_prefill.datasets_eval import get_spec
from datasets import load_dataset
from appcorr.models.ov2.unified import OV2UnifiedAxis


def main() -> int:
    dev, dt = "cuda:0", torch.bfloat16
    model, proc = load(dev, dt)
    axis = OV2UnifiedAxis(model.model).eval()
    spec = get_spec("chartqa")
    ds = spec.load(load_dataset)
    ok = True
    for i in (0, 700):
        img, prompt, _ = spec.prepare(ds[int(i)], lambda h, w, **kw: (h, w), 1, 1, 1 << 30)
        enc = encode(proc, img, prompt, dev)
        ids, pp = enc["input_ids"], enc["patch_positions"]
        px = enc["pixel_values"].to(dt)
        deg = l2_from_native(img, 2, proc, hw_from_grid(enc["image_grid_thw"], proc))
        px2 = encode(proc, deg, prompt, dev)["pixel_values"].to(dt)
        energy = patch_energy(px, px2)
        with torch.no_grad():
            ref = axis.full_forward(px, pp, ids)
            fl_ = axis.full_forward(px2, pp, ids)
            e1, _, _ = axis.interleaved_forward_progressive(px, px2, pp, ids, energy, 1.0, 1)
            e4, _, _ = axis.interleaved_forward_progressive(px, px2, pp, ids, energy, 0.25, 4)
            e1, e4 = axis.llm_finish(e1), axis.llm_finish(e4)
        def rel(x):
            return ((x - ref).norm() / ref.norm()).item()
        r_fl, r1, r4 = rel(fl_), rel(e1), rel(e4)
        g1 = r1 < 5e-3
        g4 = r1 < r4 < r_fl
        ok &= g1 and g4
        print(f"  sample {i}: rel-L2  floor {r_fl:.4f}  g4k.25 {r4:.4f}  g1k1.0 {r1:.6f}"
              f"  [{'PASS' if g1 else 'FAIL'} identity] [{'PASS' if g4 else 'FAIL'} ordering]")
    print()
    print("ALL GATES PASS" if ok else "GATE FAILURE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
