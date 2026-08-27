"""Gemma 4 vision-fork gates (level 2): the correction machinery is trustworthy.

  V1  fork approx(full) -> finish -> embed == stock end-to-end vision features,
      torch.equal (the fork's layer walk reproduces stock numerics bitwise).
  V2  approx(degraded) then correct(ALL rows, full-res layer-0 input) -> finish
      == stock(full) end-to-end, torch.equal. THE identity: with every row
      recomputed no stale K/V remains, so any difference is a fork bug.
  V3  partial correction (top-50% by patch energy): untouched pre-pool rows stay
      bitwise at their approx values (no leakage), and the corrected result is
      strictly closer to the full features than approx was (rel improves).
      Raises on vacuity (zero selected rows / zero energy).

Run: CUDA_VISIBLE_DEVICES=0 python analysis/experiments/gemma4_vision_fork_gate.py
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from appcorr.models.gemma4.unified import Gemma4Axis, MODEL_ID_31B  # noqa: E402
from appcorr.models.gemma4.vision_fork import Gemma4VisionFork      # noqa: E402
from gemma4_axis_gate import l2_degrade                              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID_31B)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--dataset", default="realworldqa")
    a = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoProcessor, Gemma4ForConditionalGeneration
    from gemma3_oracle import get_spec

    print(f"[gemma4-fork] loading {a.model}", flush=True)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="cuda:0").eval()
    proc = AutoProcessor.from_pretrained(a.model)
    axis = Gemma4Axis(model, proc)
    fork = Gemma4VisionFork(axis.vision)

    spec = get_spec(a.dataset)
    ds = spec.load(load_dataset)
    if len(ds) == 0:
        raise RuntimeError("VACUOUS: dataset loaded 0 samples")
    idxs = list(range(0, len(ds), max(1, len(ds) // a.samples)))[:a.samples]

    for i, idx in enumerate(idxs):
        img, prompt, _ = spec.prepare(ds[idx], lambda h, w, **kw: (h, w), 1, 1, 1 << 30)
        enc = axis.build_inputs(img, prompt).to("cuda:0")
        enc2 = axis.build_inputs(l2_degrade(img), prompt).to("cuda:0")
        axis.assert_same_grid(enc, enc2)
        px_f = enc["pixel_values"].to(torch.bfloat16)
        px_a = enc2["pixel_values"].to(torch.bfloat16)
        pos = enc.get("image_position_ids")
        n_patches = px_f.shape[-2] if px_f.dim() == 2 else px_f.shape[1]

        with torch.no_grad():
            stock = axis.vision_features(px_f, pos).float()

            def run(last, padding):
                h = fork.finish(last, pos_b, padding, n_pool)
                return axis.embed_vision(inputs_embeds=h).float()

            # The tower treats pixel_values as [B?, P, C*ps^2]; normalize shapes.
            px_fb = px_f if px_f.dim() == 3 else px_f.unsqueeze(0)
            px_ab = px_a if px_a.dim() == 3 else px_a.unsqueeze(0)
            pos_b = pos if pos.dim() == 3 else pos.unsqueeze(0)
            n_pool = px_fb.shape[1]

            # V1: fork full == stock
            emb_f, pad = fork.prepare(px_fb, pos_b)
            cache = {}
            fork.approx(emb_f, pos_b, pad, cache)
            v1 = run(cache["v_last_hidden"], pad)
            if not torch.equal(v1, stock):
                d = (v1 - stock).abs().max().item()
                raise RuntimeError(f"V1 FAIL idx={idx}: fork-full vs stock max {d:.3e}")

            # V2: approx(degraded) + correct(ALL) == stock(full)
            emb_a, pad_a = fork.prepare(px_ab, pos_b)
            cache = {}
            approx_last = fork.approx(emb_a, pos_b, pad_a, cache).clone()
            all_rows = torch.ones(emb_f.shape[:2], dtype=torch.bool, device=emb_f.device)
            last2 = fork.correct(emb_f, all_rows, pos_b, pad_a, cache)
            v2 = run(last2, pad_a)
            if not torch.equal(v2, stock):
                d = (v2 - stock).abs().max().item()
                raise RuntimeError(f"V2 FAIL idx={idx}: correct-all vs stock max {d:.3e}")

            # V3: partial correction improves, untouched rows bitwise-stable
            energy = (px_fb.float() - px_ab.float()).pow(2).sum(-1)
            if energy.max() <= 0:
                raise RuntimeError("VACUOUS: zero degradation energy")
            kq = max(1, int(round(0.5 * n_pool)))
            sel = torch.zeros_like(energy, dtype=torch.bool).scatter_(
                1, energy.topk(kq, dim=-1).indices, True)
            cache = {}
            approx_last = fork.approx(emb_a, pos_b, pad_a, cache).clone()
            mixed = torch.where(sel.unsqueeze(-1), emb_f, emb_a)
            last3 = fork.correct(mixed, sel, pos_b, pad_a, cache)
            untouched = ~sel[0]
            if untouched.any() and not torch.equal(
                    last3[:, untouched], approx_last[:, untouched]):
                raise RuntimeError(f"V3 FAIL idx={idx}: untouched rows moved")
            cache_full = {}
            full_last = fork.approx(emb_f, pos_b, pad, cache_full)
            r_a = ((approx_last.float() - full_last.float()).norm()
                   / full_last.float().norm().clamp_min(1e-9)).item()
            r_c = ((last3.float() - full_last.float()).norm()
                   / full_last.float().norm().clamp_min(1e-9)).item()
            if not (r_c < r_a):
                raise RuntimeError(
                    f"V3 FAIL idx={idx}: correction did not improve ({r_c:.4f} !< {r_a:.4f})")

        print(f"  [{i + 1}/{len(idxs)}] idx={idx} V1 equal | V2 equal | "
              f"V3 approx_rel={r_a:.4f} -> corrected_rel={r_c:.4f}", flush=True)

    print(f"GEMMA4_VISION_FORK_GATE_PASS samples={len(idxs)}", flush=True)


if __name__ == "__main__":
    main()
