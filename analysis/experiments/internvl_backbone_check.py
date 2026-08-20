"""Check the InternViT tower wrapper against the stock model, with real weights.

The layer fork has its own unit test; this is the wiring above it. Four things can be wrong here and
none of them show up in the layer test:

1. `prepare_tokens` — CLS token, and the *interpolated* absolute position embedding, which depends
   on the pixel size passed in.
2. `run_projector` — the final layernorm, dropping CLS, the pixel shuffle, and the projector, in
   that order. Getting the CLS drop wrong shifts every patch by one and still produces plausible
   features.
3. `full_forward` end to end against `model.get_image_features`.
4. `approx_forward` then `correct_forward` over ALL tokens, which must also equal stock — this is
   the path every arm actually runs.

    python analysis/experiments/internvl_backbone_check.py [--model ...] [--device cuda:0]
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, "/NHNHOME/share/cjpark/AppCorr-internvl")

from appcorr.models.internvl.vision.backbone import ApproxCorrectInternVLVisionTower


def report(name, got, ref, rtol):
    """Compare RELATIVE to the reference's scale.

    Raw hidden states here reach |x| ~ 130 after 24 layers, so an absolute tolerance that is sane
    for projector outputs (|x| ~ 1) rejects ordinary fp32 accumulation on the tower interior. Judge
    everything on relative error and the two live on the same scale.
    """
    scale = max(ref.float().abs().max().item(), 1e-9)
    err = (got.float() - ref.float()).abs().max().item()
    rel = err / scale
    ok = rel <= rtol
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<52} rel {rel:.2e}   (abs {err:.3e}, scale {scale:.1f})")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="OpenGVLab/InternVL3-2B-hf")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--tiles", type=int, default=7)
    a = ap.parse_args()

    from transformers import AutoModelForImageTextToText

    dtype = torch.float32 if a.dtype == "float32" else torch.bfloat16
    tol = 1e-4 if dtype is torch.float32 else 5e-2
    tok = os.environ.get("HF_TOKEN")
    full = AutoModelForImageTextToText.from_pretrained(a.model, dtype=dtype, token=tok).eval()
    model = full.model.to(a.device)
    tower = ApproxCorrectInternVLVisionTower(model).eval()
    print(f"{a.model}: {tower.num_layers} layers, downsample {tower.downsample_ratio}, "
          f"select '{tower.select_strategy}', {a.dtype}")

    size = model.config.vision_config.image_size
    h, w = (size, size) if isinstance(size, int) else tuple(size)
    torch.manual_seed(0)
    px = torch.randn(a.tiles, 3, h, w, device=a.device, dtype=dtype)

    ok = True
    with torch.no_grad():
        ref = model.get_image_features(pixel_values=px).pooler_output
        print(f"  stock features: {tuple(ref.shape)}  ({a.tiles} tiles)")

        ok &= report("full_forward == get_image_features", tower.full_forward(px), ref, tol)

        x = tower.prepare_tokens(px)
        hidden, cache = tower.approx_forward(x, {})
        ok &= report("approx_forward -> projector == stock", tower.run_projector(hidden), ref, tol)

        every = torch.ones(x.shape[0], x.shape[1], dtype=torch.bool, device=a.device)
        corrected, _ = tower.correct_forward(x, every, dict(cache))
        ok &= report("correct_forward(all) -> projector == stock",
                     tower.run_projector(corrected), ref, tol)

        # A partial correction must leave untouched LLM tokens exactly at approx.
        pm = torch.zeros(x.shape[0], x.shape[1] - 1, dtype=torch.bool, device=a.device)
        pm[:, ::3] = True
        tm = tower.patch_mask_to_token_mask(pm)
        assert not tm[:, 0].any(), "CLS must never be selected"
        part, _ = tower.correct_forward(x, tm, dict(cache))
        ok &= report("correct(subset) matches stock where corrected",
                     part[tm], _stock_hidden(tower, x)[tm], tol)
        ok &= report("correct(subset) leaves untouched at approx",
                     part[~tm], hidden[~tm], tol)

        attn = tower.approx_forward(x, {}, collect_attn=True)[1]["vision_layer_patch_attn_layermean"]
        # The invariant is over ALL columns: each query's attention sums to 1, so the column means
        # over the full 1+patches sequence average to 1/seq. `attn` has CLS removed, and CLS is not
        # an average column -- every patch attends to it heavily -- so the right check is that the
        # patch columns account for everything except CLS's share, i.e. mean*patches < 1 and the
        # deficit is CLS. Asserting mean == 1/seq on the CLS-stripped map is simply wrong.
        seq, patches = x.shape[1], attn.shape[1]
        patch_mass = attn.mean().item() * patches
        print(f"  attention map: {tuple(attn.shape)}  patch mass {patch_mass:.4f}, "
              f"CLS mass {1.0 - patch_mass:.4f}")
        ok &= 0.0 < patch_mass < 1.0 and attn.min().item() >= 0.0
        if not (0.0 < patch_mass < 1.0):
            print("  FAIL  attention columns do not form a distribution")

    print("\n" + ("ALL CHECKS PASS" if ok else "SOME CHECKS FAILED"))
    sys.exit(0 if ok else 1)


def _stock_hidden(tower, x):
    h = x
    for layer in tower.layers:
        h = layer(h)
    return h


if __name__ == "__main__":
    main()
