"""Verify the floor/ceiling setup on real samples BEFORE committing GPU hours to it.

Today a RealWorldQA sweep ran four arms to completion before the degradation was found to be too
mild, and a Gemma 3 gate reported rel 0.00e+00 twice while measuring the wrong thing. Everything
here is a check that would have caught one of those, run on a handful of real dataset items.
"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qwen_vl_prefill.datasets_eval import get_spec
from experiments.ov2_oracle import load, encode, run_one
from experiments.ov2_degradation import l2_from_native, sampled_hw
from datasets import load_dataset

N, LEVEL, DEV = 6, 2, "cuda:0"
ok = True
model, proc = load(DEV, torch.bfloat16)

for name in ("chartqa", "textvqa"):
    spec = get_spec(name); ds = spec.load(load_dataset)
    idxs = list(range(0, len(ds), max(1, len(ds) // N)))[:N]
    print(f"\n=== {name} ===")
    for j, i in enumerate(idxs):
        img, prompt, gold = spec.prepare(ds[i], lambda h, w, **kw: (h, w), 1, 1, 1 << 30)
        e1 = encode(proc, img, prompt, DEV)
        deg = l2_from_native(img, LEVEL, proc)
        e2 = encode(proc, deg, prompt, DEV)

        g1, g2 = tuple(e1["image_grid_thw"][0].tolist()), tuple(e2["image_grid_thw"][0].tolist())
        ntok = int((e1["input_ids"][0] == model.config.image_token_id).sum())
        a, b = e1["pixel_values"].float(), e2["pixel_values"].float()
        rel = float((a - b).norm() / a.norm().clamp_min(1e-9))

        c1 = "OK" if g1 == g2 else "GRID MISMATCH"
        c2 = "OK" if rel > 0.02 else "DEGRADATION IS A NO-OP"
        c3 = "OK" if ntok > 0 else "NO IMAGE TOKENS"
        good = (c1 == c2 == c3 == "OK"); ok &= good

        tc = run_one(model, proc, img, prompt, "ceiling", LEVEL, DEV, 24)
        tf = run_one(model, proc, img, prompt, "floor", LEVEL, DEV, 24)
        sc, _ = spec.score(tc, gold); sf, _ = spec.score(tf, gold)
        c4 = "OK" if tc.strip() else "EMPTY CEILING OUTPUT"; ok &= (c4 == "OK")

        print(f"  [{j}] {img.size} grid{g1} tok={ntok} pix_rel={rel:.3f} "
              f"{c1}/{c2}/{c3}/{c4}")
        print(f"      gold={gold!s:.30}  ceil={tc!r:.42} ({sc})  floor={tf!r:.42} ({sf})")

print("\n" + ("PREFLIGHT PASS" if ok else "PREFLIGHT FAILED"))
raise SystemExit(0 if ok else 1)
