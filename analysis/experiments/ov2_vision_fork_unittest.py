"""The OV2 vision fork must reproduce the stock encoder, and correct-at-100% must equal approx.

Two properties, in this order, because the second is meaningless if the first fails:

  1. `approx` over all 24 layers == the stock vision tower, to float noise. This is what makes the
     cached increment and K/V a faithful record of the real forward.
  2. `correct` with every token selected == `approx`. Partial correction is only a cheaper forward
     if the full-selection case is the identity; if it is not, every keep<1 number measures a
     different model rather than an approximation of the same one.

Gate on tensors, not on a metric. A downstream accuracy that happens to match proves nothing -- that
was measured directly on Gemma 3, where two arms tied at 341/765 while disagreeing on 108 samples.
"""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from PIL import Image
from appcorr.models.ov2.vision.block import ApproxCorrectOneVisionLayer

MODEL = "lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct"


def main():
    from transformers import AutoModelForImageTextToText, AutoProcessor
    tok = os.environ.get("HF_TOKEN")
    dev, dt = "cuda:0", torch.float32          # fp32: bf16 noise would hide a real divergence
    proc = AutoProcessor.from_pretrained(MODEL, token=tok, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=dt, token=tok, trust_remote_code=True).eval().to(dev)
    vt = model.model.visual

    img = Image.effect_mandelbrot((448, 448), (-2, -1.5, 1, 1.5), 40).convert("RGB")
    enc = proc(text=["<|vision_start|><|image_pad|><|vision_end|>"], images=[img],
               return_tensors="pt")
    px = enc["pixel_values"].to(dev, dt)
    grid = enc["image_grid_thw"].to(dev)
    pos = enc.get("patch_positions")
    if pos is not None:
        pos = pos.to(dev)

    with torch.no_grad():
        # `skip_merger=True` exists in the checkpoint for exactly this check: it returns the
        # encoder stack's output before the 2x2 patch merger, which is the tensor our layers produce.
        ref = vt(px, grid_thw=grid, patch_positions=pos, skip_merger=True).last_hidden_state

        # Rebuild the encoder's entry state the way the tower does: embeddings, then layernorm_pre.
        # Missing layernorm_pre would put every layer on a differently-scaled input and still run.
        h = vt.embeddings(px)
        if h.dim() == 2:
            h = h.unsqueeze(0)
        h = vt.layernorm_pre(h)

        # RoPE is D/2 wide and duplicated to D before use; feeding the un-duplicated half would
        # rotate only the first half of every head.
        pp = pos.squeeze(0) if (pos is not None and pos.dim() == 3) else pos
        freqs = vt.video_rope.forward_from_positions(pp)
        freqs = torch.cat([freqs, freqs], dim=-1)
        if freqs.dim() == 2:
            freqs = freqs.unsqueeze(0)
        print(f"  entry {tuple(h.shape)}  freqs {tuple(freqs.shape)}  ref {tuple(ref.shape)}")

        layers = [ApproxCorrectOneVisionLayer.from_stock(l) for l in vt.encoder.layers]
        cache, x = {}, h
        for i, l in enumerate(layers):
            x, cache = l.approx(x, cache, f"v{i}", freqs)
        x_post = vt.layernorm_post(x) if vt.layernorm_post is not None else x

        full = torch.ones(x.shape[0], x.shape[1], dtype=torch.bool, device=dev)
        c2, y = dict(cache), h
        for i, l in enumerate(layers):
            y, c2 = l.correct(y, full, c2, f"v{i}", freqs)
        y_post = vt.layernorm_post(y) if vt.layernorm_post is not None else y

    def rel(a, b):
        return float((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-9))

    r1 = rel(x_post, ref)
    r2 = rel(y_post, x_post)
    ok1, ok2 = r1 < 1e-5, r2 < 1e-5
    print(f"\n  {'PASS' if ok1 else 'FAIL'}  approx(24 layers) == stock encoder   rel {r1:.2e}")
    print(f"  {'PASS' if ok2 else 'FAIL'}  correct@100%      == approx           rel {r2:.2e}")
    ok = ok1 and ok2
    print("\n" + ("ALL GATES PASS" if ok else "GATE FAILED"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
