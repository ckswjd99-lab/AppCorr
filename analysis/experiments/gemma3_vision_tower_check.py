"""Check the SigLIP fork against Gemma 3's real vision tower, with real weights.

The layer test uses one random-weight layer. This runs all 27 trained layers on a real image and
asks the questions that only appear at depth:

1. `approx` over the whole tower reproduces stock.
2. `correct` over all tokens reproduces stock.
3. `correct` over a subset leaves untouched tokens where approx left them **after 27 layers** --
   a per-layer error of 1e-7 that compounds would show up here and nowhere else.
4. The projector path (`get_image_features`: tower -> multi_modal_projector -> 4096 patches pooled
   to 256 tokens) still matches, because that pooling is where a subtle indexing error would hide.

    python analysis/experiments/gemma3_vision_tower_check.py [--dtype float32]
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, "/NHNHOME/share/cjpark/AppCorr-gemma3")

from appcorr.models.gemma3.vision.block import ApproxCorrectSiglipLayer


def rep(name, got, ref, rtol):
    scale = max(ref.float().abs().max().item(), 1e-9)
    err = (got.float() - ref.float()).abs().max().item()
    ok = err / scale <= rtol
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<50} rel {err/scale:.2e}  (abs {err:.3e}, scale {scale:.1f})")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    a = ap.parse_args()

    import numpy as np
    from PIL import Image
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    dtype = torch.float32 if a.dtype == "float32" else torch.bfloat16
    rtol = 1e-4 if dtype is torch.float32 else 5e-2
    tok = os.environ.get("HF_TOKEN")
    m = Gemma3ForConditionalGeneration.from_pretrained(a.model, dtype=dtype, token=tok).eval()
    m = m.to(a.device)
    proc = AutoProcessor.from_pretrained(a.model, token=tok)

    vt = m.model.vision_tower          # SiglipVisionModel holds embeddings/encoder/post_layernorm directly
    layers = torch.nn.ModuleList(ApproxCorrectSiglipLayer.from_stock(l) for l in vt.encoder.layers)
    print(f"{a.model}: SigLIP {len(layers)} layers, hidden {vt.config.hidden_size}, "
          f"img {vt.config.image_size}, patch {vt.config.patch_size}, {a.dtype}")

    torch.manual_seed(0)
    img = Image.fromarray((np.random.rand(896, 896, 3) * 255).astype("uint8"))
    msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                         {"type": "text", "text": "Describe it."}]}]
    enc = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                   return_dict=True, return_tensors="pt").to(a.device)
    px = enc["pixel_values"].to(dtype)

    ok = True
    with torch.no_grad():
        x = vt.embeddings(px)
        print(f"  tokens: {tuple(x.shape)}  (no CLS -- patch index == token index)")

        ref = x
        for l in vt.encoder.layers:
            ref = l(ref, attention_mask=None)

        hidden, cache = x, {}
        for i, l in enumerate(layers):
            hidden, cache = l.approx(hidden, cache, f"v{i}")
        ok &= rep("approx over 27 layers == stock", hidden, ref, rtol)

        every = torch.ones(x.shape[0], x.shape[1], dtype=torch.bool, device=a.device)
        cur, c2 = x, dict(cache)
        for i, l in enumerate(layers):
            cur, c2 = l.correct(cur, every, c2, f"v{i}")
        ok &= rep("correct(all) over 27 layers == stock", cur, ref, rtol)

        mask = torch.zeros(x.shape[0], x.shape[1], dtype=torch.bool, device=a.device)
        mask[:, ::2] = True                      # half the patches
        cur, c3 = x, dict(cache)
        for i, l in enumerate(layers):
            cur, c3 = l.correct(cur, mask, c3, f"v{i}")
        ok &= rep("correct(half) untouched == approx (27 layers)",
                  cur[~mask], hidden[~mask], rtol)

        # Post-tower: post_layernorm, projector, pooling to 256 tokens.
        ref_feats = m.model.get_image_features(pixel_values=px)
        if not torch.is_tensor(ref_feats):
            po = getattr(ref_feats, "pooler_output", None)
            ref_feats = po if torch.is_tensor(po) else ref_feats[0]
        got = m.model.multi_modal_projector(vt.post_layernorm(hidden))
        ok &= rep("approx -> projector == get_image_features", got, ref_feats, rtol)
        print(f"  image features: {tuple(ref_feats.shape)}  (4096 patches pooled)")

    print("\n" + ("ALL CHECKS PASS" if ok else "SOME CHECKS FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
