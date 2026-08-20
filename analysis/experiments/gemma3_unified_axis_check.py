"""Gates for the Gemma 3 unified 27+34 = 61-stage axis, with real weights.

The two forks each have their own unit test. This checks the thing neither can: that walking them as
ONE axis reproduces the stock model, and that a selection made on vision patches survives the change
of token identity at the projector.

    1. full_forward == the stock model's final hidden state.
    2. approx over all 61 stages == stock.
    3. correct over ALL tokens, both halves, == stock.
    4. a PARTIAL vision selection, mapped across the boundary, leaves untouched LLM positions
       exactly where the approximate pass left them.
    5. text positions are never selected -- they were never approximated, so correcting them would
       be spending budget on tokens that are already exact.
    6. the pooled mapping is right: 4096 patches -> 256 image tokens, 16 patches each.

    python analysis/experiments/gemma3_unified_axis_check.py [--dtype float32]
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, "/NHNHOME/share/cjpark/AppCorr-gemma3")

from appcorr.models.gemma3.unified import Gemma3UnifiedAxis


def rep(name, got, ref, rtol):
    if got.numel() == 0:
        print(f"  SKIP  {name:<52} (empty)")
        return True
    scale = max(ref.float().abs().max().item(), 1e-9)
    err = (got.float() - ref.float()).abs().max().item()
    ok = err / scale <= rtol
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<52} rel {err/scale:.2e}  (abs {err:.3e})")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--keep", type=float, default=0.5)
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
    print(f"{a.model}: {axis.n_vision} vision + {axis.n_llm} llm = {axis.n_stages} stages, {a.dtype}")

    torch.manual_seed(0)
    img = Image.fromarray((np.random.rand(896, 896, 3) * 255).astype("uint8"))
    msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                         {"type": "text", "text": "What is here?"}]}]
    enc = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                   return_dict=True, return_tensors="pt").to(a.device)
    px, ids = enc["pixel_values"].to(dtype), enc["input_ids"]
    # token_type_ids is what selects the bidirectional-image mask. Omitting it makes the model fall
    # back to a plain causal mask -- a different forward, not a subtle one.
    tti = enc.get("token_type_ids")

    ok = True
    with torch.no_grad():
        ref = m.model(input_ids=ids, pixel_values=px, token_type_ids=tti,
                      use_cache=False).last_hidden_state
        ok &= rep("full_forward == stock model", axis.full_forward(px, ids, tti), ref, rtol)

        # --- approx over the whole axis ---
        vh, cache = axis.vision_approx(axis.vision_prepare(px), {})
        emb, ctx = axis.llm_prepare(ids, axis.project(vh), tti)
        n_img = ctx["image_positions"].numel()
        print(f"  sequence {ids.shape[1]} = {n_img} image + {ids.shape[1]-n_img} text; "
              f"patches {vh.shape[1]} -> {n_img} image tokens "
              f"({vh.shape[1]//n_img} patches each)")
        ok &= (vh.shape[1] // n_img) * n_img == vh.shape[1]

        approx_h, cache = axis.llm_approx(emb, ctx, cache)
        ok &= rep("approx over 61 stages == stock", axis.llm_finish(approx_h), ref, rtol)

        # --- correct everything, both halves ---
        every_p = torch.ones(px.shape[0], vh.shape[1], dtype=torch.bool, device=a.device)
        c2 = dict(cache)
        vh2, c2 = axis.vision_correct(axis.vision_prepare(px), every_p, c2)
        emb2, _ = axis.llm_prepare(ids, axis.project(vh2), tti)
        every_t = torch.ones_like(ids, dtype=torch.bool)
        h2, c2 = axis.llm_correct(emb2, every_t, ctx, c2)
        ok &= rep("correct(all) over 61 stages == stock", axis.llm_finish(h2), ref, rtol)

        # --- partial selection, mapped across the boundary ---
        pm = torch.zeros(px.shape[0], vh.shape[1], dtype=torch.bool, device=a.device)
        k = int(a.keep * vh.shape[1])
        pm[:, torch.randperm(vh.shape[1], device=a.device)[:k]] = True
        tm = axis.patch_mask_to_llm_mask(pm, ids.shape[1], ctx["image_positions"])
        text_selected = int(tm.sum().item()) - int(tm[:, ctx["image_positions"]].sum().item())
        print(f"  patch keep {a.keep:.0%} -> {int(pm.sum())} patches -> "
              f"{int(tm.sum())} LLM tokens selected; text tokens selected: {text_selected}")
        ok &= text_selected == 0
        if text_selected:
            print("  FAIL  text positions were selected; they are never approximated")

        c3 = dict(cache)
        vh3, c3 = axis.vision_correct(axis.vision_prepare(px), pm, c3)
        emb3, _ = axis.llm_prepare(ids, axis.project(vh3), tti)
        h3, c3 = axis.llm_correct(emb3, tm, ctx, c3)
        ok &= rep("correct(partial) untouched == approx", h3[~tm], approx_h[~tm], rtol)

    print("\n" + ("ALL GATES PASS" if ok else "SOME GATES FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
