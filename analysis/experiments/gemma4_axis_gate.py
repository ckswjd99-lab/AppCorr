"""Gemma 4 31B axis gate — level 1 (harness == stock, degradation sane).

Four gates, in dependency order; each later gate is meaningless if an earlier one
fails, and every gate raises on vacuity (no silent PASS on empty/degenerate data):

  G1  stock generate produces sane text on real images (the model + env work;
      the 122B-FP8 lesson: gate the STOCK path before gating anything of ours).
  G2  manual full_forward first-token logits == stock forward logits, torch.equal
      in bf16 (the harness computes exactly what stock computes).
  G3  level-2 degradation keeps the native-resolution patch grid EXACTLY
      (shapes + positions), and the degraded image embeds differ from the full
      ones by rel-L2 > 0.02 (degradation must be informative — the vacuity
      guard that caught the 32px no-op on qwen35).
  G4  floor arm (stock generate on the degraded image) produces sane text.

Run: CUDA_VISIBLE_DEVICES=0 python analysis/experiments/gemma4_axis_gate.py [--samples 4]
"""
import argparse
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from appcorr.models.gemma4.unified import Gemma4Axis, MODEL_ID_31B  # noqa: E402


# Gemma 4's processor resizes aspect-preserving to at most 2520 patches of 16px
# (max_soft_tokens 280 x 3x3 pool) -- the model never sees more than this many
# pixels, so the pyramid-direction rule (AGENTS.md; pyramid_degradation_
# native_vs_canvas.md) caps degradation at the SAMPLED resolution, not native.
_GEMMA4_MAX_PX = 2520 * 16 * 16


def l2_degrade(img: Image.Image, factor: int = 4, max_px: int = _GEMMA4_MAX_PX) -> Image.Image:
    """Level-2 pyramid base at the model-sampled resolution, restored to size.

    BOX for the down (area averaging = the pyramid level, per the convention's
    reference implementation), BICUBIC for the up. The cap applies AREA-wise
    because the resize is aspect-preserving: an image larger than what the
    processor samples degrades relative to the sampled dims -- degrading native
    pixels the model would discard anyway makes the floor sit too close to the
    ceiling (the documented canvas-direction failure).
    The output size equals the input size, so the patch grid is preserved by
    construction; the gates assert it anyway.
    """
    w, h = img.size
    s = min(1.0, (max_px / (w * h)) ** 0.5)
    tw = max(1, int(w * s) // factor)
    th = max(1, int(h * s) // factor)
    return img.resize((tw, th), Image.BOX).resize((w, h), Image.BICUBIC)


def sane(txt: str) -> bool:
    """Garbage detector, not a correctness check. A bare 'A' is a legitimate MCQ
    answer; the broken-model signature (the 122B lesson) is multilingual token
    soup — non-printable or alnum-free output."""
    t = txt.strip()
    return len(t) >= 1 and any(c.isalnum() for c in t) and t.isprintable()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID_31B)
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--dataset", default="realworldqa")
    a = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoProcessor, Gemma4ForConditionalGeneration
    from gemma3_oracle import get_spec  # dataset registry is model-agnostic

    print(f"[gemma4] loading {a.model} (bf16)", flush=True)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="cuda:0").eval()
    proc = AutoProcessor.from_pretrained(a.model)
    axis = Gemma4Axis(model, proc)

    spec = get_spec(a.dataset)
    ds = spec.load(load_dataset)
    if len(ds) == 0:
        raise RuntimeError("VACUOUS: dataset loaded 0 samples")
    idxs = list(range(0, len(ds), max(1, len(ds) // a.samples)))[:a.samples]
    if not idxs:
        raise RuntimeError("VACUOUS: no sample indices")

    g2_pass = g3_rels = 0
    g3_min_rel = float("inf")
    for i, idx in enumerate(idxs):
        # Identity smart_resize: gemma4's processor handles native-resolution sizing itself.
        img, prompt, _gold = spec.prepare(ds[idx], lambda h, w, **kw: (h, w), 1, 1, 1 << 30)

        enc = axis.build_inputs(img, prompt).to("cuda:0")
        enc["pixel_values"] = enc["pixel_values"].to(torch.bfloat16)
        with torch.no_grad():
            # G1: stock generate
            out = model.generate(**enc, max_new_tokens=16, do_sample=False)
            txt = proc.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            if not sane(txt):
                raise RuntimeError(f"G1 FAIL sample {idx}: stock generate insane: {txt!r}")

            # G2: harness == stock on first-token logits
            stock = model(**{k: v for k, v in enc.items()})
            hidden, _ = axis.full_forward(enc)
            ours = axis.logits(hidden[:, -1:, :])
            theirs = stock.logits[:, -1:, :]
            if torch.equal(ours, theirs):
                g2_pass += 1
            else:
                d = (ours.float() - theirs.float()).abs().max().item()
                raise RuntimeError(f"G2 FAIL sample {idx}: logits differ, max abs {d:.3e}")

            # G3: degradation grid + informativeness (feature space, not CoT text)
            deg = l2_degrade(img)
            enc2 = axis.build_inputs(deg, prompt).to("cuda:0")
            enc2["pixel_values"] = enc2["pixel_values"].to(torch.bfloat16)
            axis.assert_same_grid(enc, enc2)
            f_full = axis.vision_features(enc["pixel_values"], enc.get("image_position_ids")).float()
            f_deg = axis.vision_features(enc2["pixel_values"], enc2.get("image_position_ids")).float()
            rel = ((f_full - f_deg).norm() / f_full.norm().clamp_min(1e-9)).item()
            g3_min_rel = min(g3_min_rel, rel)
            g3_rels += 1

            # G4: floor arm sanity
            out2 = model.generate(**enc2, max_new_tokens=16, do_sample=False)
            txt2 = proc.decode(out2[0, enc2["input_ids"].shape[1]:], skip_special_tokens=True)
            if not sane(txt2):
                raise RuntimeError(f"G4 FAIL sample {idx}: floor generate insane: {txt2!r}")

        print(f"  [{i + 1}/{len(idxs)}] idx={idx} G1 ok | G2 equal | "
              f"G3 rel={rel:.4f} | G4 ok  full={txt!r:.40} floor={txt2!r:.40}", flush=True)

    if g2_pass != len(idxs):
        raise RuntimeError(f"G2 incomplete: {g2_pass}/{len(idxs)}")
    if g3_rels == 0:
        raise RuntimeError("VACUOUS: G3 never measured")
    if g3_min_rel < 0.02:
        raise RuntimeError(f"G3 FAIL: degradation not informative, min rel {g3_min_rel:.4f}")
    print(f"GEMMA4_AXIS_GATE_PASS samples={len(idxs)} g2_equal={g2_pass} "
          f"g3_min_rel={g3_min_rel:.4f}", flush=True)


if __name__ == "__main__":
    main()
