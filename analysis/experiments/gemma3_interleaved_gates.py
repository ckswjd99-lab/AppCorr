"""Contract gates for interleaved correction on the Gemma 3 unified axis.

From docs/memo/interleaved_correction_contract.md, adapted to an axis with a projector in the middle:

    1. Coverage      -- union of per-round groups == the one-shot selection, on BOTH sides.
    2. g=1 identity  -- the interleaved walk with one group reproduces one-shot. This is the real
                        gate: with the rules held it is the same computation, so expect equality,
                        not "close enough".
    3. Cost          -- rounds cost less than one-shot, weighted by stage cost (a vision layer and
                        an LLM layer are not interchangeable units here).
    4. Text          -- text tokens are never selected in any round.

    python analysis/experiments/gemma3_interleaved_gates.py [--dtype float32]
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, "/NHNHOME/share/cjpark/AppCorr-gemma3")

from appcorr.models.gemma3.unified import Gemma3UnifiedAxis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--keep", type=float, default=0.55)
    a = ap.parse_args()

    import numpy as np
    from PIL import Image
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    dtype = torch.float32 if a.dtype == "float32" else torch.bfloat16
    rtol = 3e-4 if dtype is torch.float32 else 8e-2
    tok = os.environ.get("HF_TOKEN")
    m = Gemma3ForConditionalGeneration.from_pretrained(a.model, dtype=dtype, token=tok).eval()
    m = m.to(a.device)
    proc = AutoProcessor.from_pretrained(a.model, token=tok)
    axis = Gemma3UnifiedAxis(m.model).eval()

    torch.manual_seed(0)
    img = Image.fromarray((np.random.rand(896, 896, 3) * 255).astype("uint8"))
    deg = img.resize((224, 224), Image.BOX).resize((896, 896), Image.BICUBIC)
    def enc_of(im):
        msgs = [{"role": "user", "content": [{"type": "image", "image": im},
                                             {"type": "text", "text": "What is here?"}]}]
        return proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                        return_dict=True, return_tensors="pt").to(a.device)
    e_full, e_appr = enc_of(img), enc_of(deg)
    px, px2 = e_full["pixel_values"].to(dtype), e_appr["pixel_values"].to(dtype)
    ids, tti = e_full["input_ids"], e_full.get("token_type_ids")

    ok = True
    with torch.no_grad():
        # A score with real structure, and the one-shot selections derived from it.
        n_patch = axis.patch_grid()[0] ** 2
        g = torch.linspace(0, 1, n_patch, device=a.device).unsqueeze(0)
        score = torch.sin(g * 29) ** 2 + 0.15 * torch.rand_like(g)
        pk = max(1, int(round(a.keep * n_patch)))
        pm = torch.zeros_like(score, dtype=torch.bool).scatter_(
            1, score.topk(pk, dim=-1).indices, True)

        # The approximate pass over the WHOLE axis, on the approximate image. This is the cache
        # every correction reads from -- both halves of it. An earlier version of this gate built
        # only the vision half and then asked llm_correct for `l0_k`.
        vh, cache = axis.vision_approx(axis.vision_prepare(px2), {})
        feats_appr = axis.project(vh)
        emb, ctx = axis.llm_prepare(ids, feats_appr, tti)
        _, cache = axis.llm_approx(emb, ctx, cache)
        llm_oneshot = axis.llm_mask_from_score(score, a.keep, ids.shape[1], ctx["image_positions"])
        print(f"  {axis.n_vision}+{axis.n_llm}={axis.n_stages} stages; "
              f"patches {int(pm.sum())}/{n_patch}, llm tokens {int(llm_oneshot.sum())}")

        # --- 1 & 4: coverage and text ---
        for gN in (1, 2, 4, 8):
            gp = axis.spatial_groups(pm, gN)
            union = torch.zeros_like(pm)
            for x in gp:
                union |= x
            cov = bool((union == pm).all())
            overlap = sum(int((gp[i] & gp[j]).sum()) for i in range(gN) for j in range(i + 1, gN))
            print(f"  g={gN:<2} bounds={str(axis.layer_bounds(gN)):<28} "
                  f"coverage={'OK' if cov else 'MISMATCH'} overlap={overlap}")
            ok &= cov and overlap == 0
        txt = int(llm_oneshot.sum()) - int(llm_oneshot[:, ctx["image_positions"]].sum())
        print(f"  text tokens ever selected: {txt}")
        ok &= txt == 0

        # --- 2: g=1 identity against the one-shot path ---
        # One-shot reference, built the way the driver builds it -- and the way the entry guard
        # insists on: `correct` gets the layer-0 input, the LLM stream is MIXED against the purely
        # approximate projection, and text joins the correction. An earlier version of this gate
        # made the same mistake the driver did (feeding the approximate pass's output back into
        # layer 0), which is what the `g=1` failure at rel 6.6e-01 was actually measuring.
        mixed = torch.where(pm.unsqueeze(-1), axis.vision_prepare(px), axis.vision_prepare(px2))
        vh1, c1 = axis.vision_correct(mixed, pm, dict(cache))   # copy: keeps `cache` pristine
        sel_tok = llm_oneshot[:, ctx["image_positions"]]
        feats_mixed = torch.where(sel_tok.unsqueeze(-1), axis.project(vh1), feats_appr)
        emb1, _ = axis.llm_prepare(ids, feats_mixed, tti)
        is_text = torch.ones_like(llm_oneshot)
        is_text[:, ctx["image_positions"]] = False
        h1, _ = axis.llm_correct(emb1, llm_oneshot | is_text, ctx, c1)
        ref = h1        # compare PRE-finish, the contract both paths must meet

        hI, _ = axis.interleaved_forward(px, px2, ids, tti, pm, llm_oneshot, 1)
        scale = max(ref.float().abs().max().item(), 1e-9)
        err = (hI.float() - ref.float()).abs().max().item() / scale
        print(f"  {'PASS' if err <= rtol else 'FAIL'}  g=1 interleaved == one-shot"
              f"{'':<20} rel {err:.2e}")
        ok &= err <= rtol

        # --- 3: cost signature ---
        costs = axis.stage_costs()
        print(f"  stage cost share: vision {float(costs[:axis.n_vision].sum()):.2f}, "
              f"llm {float(costs[axis.n_vision:].sum()):.2f}")
        for gN in (2, 4, 8):
            b = axis.layer_bounds(gN)
            gp = axis.spatial_groups(pm, gN)
            tot = sum(float(costs[:b[r]].sum()) * float(gp[r].sum()) for r in range(gN))
            one = 1.0 * float(pm.sum())
            print(f"  g={gN:<2} correction cost {tot/one:.2f}x one-shot")
            ok &= tot / one < 1.0

    print("\n" + ("ALL GATES PASS" if ok else "SOME GATES FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
