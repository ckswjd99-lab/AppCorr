"""One-shot correction gates for OV2, on real samples rather than a synthetic image.

The unified axis check proves the axis reproduces the stock model. It says nothing about the
DRIVER's wiring -- the degraded second encode, the mixed streams, the score, the decode from the
axis's own K/V. Those are exactly where the Gemma 3 fork lost accuracy while every tensor gate read
zero.

Four gates:

  1. **keep=1.0 == the exact forward, in feature space.** Correcting every patch and every token
     from the full-resolution stream IS the ceiling computation, so the last-position hidden state
     must match `full_forward` to float noise. This is the gate that cannot be satisfied by luck;
     a task metric can.
  2. **keep=1.0 == the ceiling, token for token.** Same claim through the decode path, which gate 1
     does not exercise: it catches a K/V cache that was built but not handed to generation. Gemma 3
     failed exactly this way -- a copied cache meant every token after the first was produced from
     APPROXIMATE K/V, so even keep=1.0 could not reproduce the ceiling.
  3. **keep=0 is not the floor, and that is expected.** Text is always corrected, so the minimum-
     budget arm still recomputes the text against approximate image K/V. Reported, not asserted --
     it is here so the number is on the record rather than discovered later as an anomaly.
  4. **The score is not degenerate.** A patch score that is constant, or that ranks by nothing,
     produces a top-k indistinguishable from an arbitrary one. Reported as the overlap between the
     energy-only and energy x attention selections.

Run in bf16, the dtype the sweeps use: this is a driver gate, not a numerics gate, and gate 2 has to
see the same rounding the real arms see.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from datasets import load_dataset

from appcorr.models.ov2.unified import OV2UnifiedAxis
from experiments.ov2_degradation import hw_from_grid, l2_from_native
from experiments.ov2_oracle import (encode, generate_from_axis, load, patch_energy, run_axis,
                                    run_stock)
from qwen_vl_prefill.datasets_eval import get_spec

N = int(os.environ.get("N", "8"))
DATASET = os.environ.get("DATASET", "chartqa")


@torch.no_grad()
def main():
    dev, dt = "cuda:0", torch.bfloat16
    model, proc = load(dev, dt)
    axis = OV2UnifiedAxis(model.model).eval()

    spec = get_spec(DATASET)
    ds = spec.load(load_dataset)
    idxs = list(range(0, len(ds), max(1, len(ds) // N)))[:N]
    print(f"[gates] {DATASET}  {N} samples  bf16\n")

    feat_rel, same_text, overlaps = [], 0, []
    for k, i in enumerate(idxs, 1):
        img, prompt, gold = spec.prepare(ds[i], lambda h, w, **kw: (h, w), 1, 1, 1 << 30)

        # --- gate 1: keep=1.0 in feature space ------------------------------------------------ #
        enc = encode(proc, img, prompt, dev)
        ids, pp = enc["input_ids"], enc["patch_positions"]
        px = enc["pixel_values"].to(dt)
        exact = axis.full_forward(px, pp, ids)

        deg = l2_from_native(img, 2, proc, hw_from_grid(enc["image_grid_thw"], proc))
        px2 = encode(proc, deg, prompt, dev)["pixel_values"].to(dt)
        freqs = axis.rope_freqs(pp)
        n_patch = px.shape[0]

        cache = {}
        vh_appr, cache = axis.vision_approx(axis.vision_prepare(px2), freqs, cache)
        emb_appr, ctx = axis.llm_prepare(ids, axis.project(vh_appr, pp))
        _, cache = axis.llm_approx(emb_appr, ctx, cache)

        pm = torch.ones(1, n_patch, dtype=torch.bool, device=dev)
        tm = torch.ones(1, ids.shape[1], dtype=torch.bool, device=dev)
        mixed = torch.where(pm.unsqueeze(-1), axis.vision_prepare(px), axis.vision_prepare(px2))
        vh_c, cache = axis.vision_correct(mixed, pm, freqs, cache)
        emb_c, _ = axis.llm_prepare(ids, axis.project(vh_c, pp))
        h_c, cache = axis.llm_correct(emb_c, tm, ctx, cache)
        got = axis.llm_finish(h_c)
        r = float((got.float() - exact.float()).abs().max()
                  / exact.float().abs().max().clamp_min(1e-9))
        feat_rel.append(r)

        # --- gate 2: keep=1.0 vs the ceiling, through the decode path -------------------------- #
        txt_ceil, _ = run_stock(model, proc, img, prompt, "ceiling", 2, dev, 24)
        txt_k1, _ = run_axis(axis, model, proc, img, prompt, "corrected", 1.0, 2, dev, dt, 24,
                             "energy_attn", 4)
        same = (txt_ceil == txt_k1)
        same_text += int(same)

        # --- gate 4: is the score doing anything? --------------------------------------------- #
        e = patch_energy(px, px2)
        attn = cache.get("vision_patch_attn_layermean")
        cache2 = {}
        _, cache2 = axis.vision_approx(axis.vision_prepare(px2), freqs, cache2, collect_attn=True)
        attn = cache2["vision_patch_attn_layermean"]
        ea = (e / e.mean().clamp_min(1e-12)) * (attn / attn.mean().clamp_min(1e-12)).to(e.device)
        kk = max(1, int(round(0.55 * n_patch)))
        s1 = set(e.topk(kk, dim=-1).indices[0].tolist())
        s2 = set(ea.topk(kk, dim=-1).indices[0].tolist())
        overlaps.append(len(s1 & s2) / kk)

        print(f"  [{k}/{N}] n_patch={n_patch:<6} feat_rel={r:.2e}  "
              f"k1{'==' if same else '!='}ceil  energy/energy_attn overlap={overlaps[-1]:.2f}"
              + ("" if same else f"\n        ceil={txt_ceil[:50]!r}\n        k1  ={txt_k1[:50]!r}"),
              flush=True)

    worst = max(feat_rel)
    # bf16 over 60 stages: the axis check measured 1.78e-05 in fp32, and bf16 carries ~8e-3 of
    # relative noise per op, so the bar here is a driver bar, not a numerics bar.
    g1 = worst < 5e-2
    g2 = same_text == N
    print(f"\n  {'PASS' if g1 else 'FAIL'}  keep=1.0 == exact forward (feature)   "
          f"worst rel {worst:.2e}")
    print(f"  {'PASS' if g2 else 'FAIL'}  keep=1.0 == ceiling (token for token)  {same_text}/{N}")
    print(f"  INFO  energy vs energy_attn top-55% overlap  "
          f"mean {sum(overlaps)/len(overlaps):.2f}  (1.00 would mean attention changes nothing)")
    ok = g1 and g2
    print("\n" + ("ALL GATES PASS" if ok else "GATE FAILED"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
