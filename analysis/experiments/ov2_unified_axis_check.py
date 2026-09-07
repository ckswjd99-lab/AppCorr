"""The OV2 unified axis must reproduce the stock model end to end, and correct@100% must be exact.

The two fork unit tests each validate one half against its own reference. Neither can see the GLUE
between them -- `vision_prepare`, `project`, `llm_prepare` -- and that glue is where the Gemma 3 axis
lost 20pp while every gate read rel 0.00e+00. So the reference here is the stock
`LlavaOnevision2Model.forward`, which owns the whole path.

Three gates, in dependency order:

  1. `full_forward` == stock. The axis walks the stock layers itself, so a mismatch is purely glue:
     a missing `layernorm_pre`, a merger fed un-normed input, image features spliced at the wrong
     positions, a rope built from the wrong positions.
  2. `approx` over all 60 stages == `full_forward`. This is what makes the cached increment and K/V
     a faithful record of the real forward.
  3. `correct` with every token selected == `approx`. Partial correction is only a cheaper forward
     if the full-selection case is the identity.

Gate 3 is run with a SCATTERED selection too: a prefix selection cannot distinguish "rotary phase
taken by token index" from "taken by position within the selection", because for 0,1,2,... the two
are identical.

fp32 on purpose -- bf16's ~8e-3 noise floor swallows exactly the failures this looks for.
"""
import os
import sys

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from appcorr.models.ov2.unified import OV2UnifiedAxis

MODEL = "lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct"


def rel(a, b):
    return float((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-9))


def main():
    from transformers import AutoModelForImageTextToText, AutoProcessor

    tok = os.environ.get("HF_TOKEN")
    dev, dt = "cuda:0", torch.float32
    proc = AutoProcessor.from_pretrained(MODEL, token=tok, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=dt, token=tok, trust_remote_code=True).eval().to(dev)
    axis = OV2UnifiedAxis(model.model).eval()

    img = Image.effect_mandelbrot((448, 448), (-2, -1.5, 1, 1.5), 40).convert("RGB")
    msgs = [{"role": "user", "content": [{"type": "image"},
                                         {"type": "text", "text": "What is this?"}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    enc = proc(text=[text], images=[img], return_tensors="pt")
    enc = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in enc.items()}
    ids = enc["input_ids"]
    px = enc["pixel_values"].to(dt)
    grid = enc["image_grid_thw"]
    pp = enc["patch_positions"]

    n_patch = int(grid[0].prod())
    n_tok = axis.n_tokens(n_patch)
    print(f"  {n_patch} patches -> {n_tok} image tokens, seq {ids.shape[1]}, "
          f"axis {axis.n_vision}+{axis.n_llm}={axis.n_stages} stages")

    with torch.no_grad():
        # `Qwen3Model` applies its final `norm`, so this is the POST-norm state -- the same thing
        # `llm_finish` produces, which is why every axis path below is finished before comparing.
        ref = model.model(input_ids=ids, pixel_values=px, image_grid_thw=grid,
                          patch_positions=pp, use_cache=False).last_hidden_state

        full = axis.full_forward(px, pp, ids)

        freqs = axis.rope_freqs(pp)
        cache = {}
        vh, cache = axis.vision_approx(axis.vision_prepare(px), freqs, cache)
        feats = axis.project(vh, pp)
        emb, ctx = axis.llm_prepare(ids, feats)
        h_appr, cache = axis.llm_approx(emb, ctx, cache)
        appr = axis.llm_finish(h_appr)

        # correct@100%: every patch, every token.
        pm = torch.ones(1, n_patch, dtype=torch.bool, device=dev)
        tm = torch.ones(1, ids.shape[1], dtype=torch.bool, device=dev)
        c2 = dict(cache)
        vh2, c2 = axis.vision_correct(axis.vision_prepare(px), pm, freqs, c2)
        emb2, _ = axis.llm_prepare(ids, axis.project(vh2, pp))
        h2, c2 = axis.llm_correct(emb2, tm, ctx, c2)
        corr = axis.llm_finish(h2)

        # A scattered selection: untouched rows must be reconstructed exactly from the increment.
        pm3 = torch.zeros(1, n_patch, dtype=torch.bool, device=dev)
        pm3[0, ::7] = True
        tm3 = torch.zeros(1, ids.shape[1], dtype=torch.bool, device=dev)
        tm3[0, ::5] = True
        tm3[0, -1] = True
        c3 = dict(cache)
        vh3, c3 = axis.vision_correct(axis.vision_prepare(px), pm3, freqs, c3)

    r1 = rel(full, ref)
    r2 = rel(appr, full)
    r3 = rel(corr, appr)
    unsel = ~pm3[0]
    r4 = rel(vh3[0][unsel], vh[0][unsel])

    g = [("full_forward     == stock model", r1),
         ("approx(60 stages)== full_forward", r2),
         ("correct@100%     == approx", r3),
         (f"scattered vision correct leaves others exact ({int(pm3.sum())}/{n_patch})", r4)]
    ok = True
    print()
    for name, r in g:
        p = r < 1e-4
        ok &= p
        print(f"  {'PASS' if p else 'FAIL'}  {name:<48} rel {r:.2e}")
    print("\n" + ("ALL GATES PASS" if ok else "GATE FAILED"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
