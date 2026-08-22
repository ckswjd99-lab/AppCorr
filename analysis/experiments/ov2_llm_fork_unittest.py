"""The OV2 LLM fork must reproduce the stock Qwen3 decoder stack, and correct@100% must equal approx.

Same two properties and the same ordering as the vision gate. The reference is the stock
`Qwen3Model` run on the SAME embeddings, so a mismatch can only come from the fork.

fp32 on purpose: bf16's ~7.8e-3 noise floor would swallow exactly the failures this is looking for
(a reordered QK-norm, a mis-sliced rotary phase, a mask row taken from the wrong query).
"""
import os, sys
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from appcorr.models.ov2.llm.decoder_layer import ApproxCorrectQwen3DecoderLayer

MODEL = "lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct"


def main():
    from transformers import AutoModelForImageTextToText, AutoTokenizer
    tok = os.environ.get("HF_TOKEN")
    dev, dt = "cuda:0", torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=dt, token=tok, trust_remote_code=True).eval().to(dev)
    lm = model.model.language_model
    tk = AutoTokenizer.from_pretrained(MODEL, token=tok, trust_remote_code=True)

    ids = tk("The quick brown fox jumps over the lazy dog. What animal is described here?",
             return_tensors="pt").input_ids.to(dev)
    n = ids.shape[1]
    print(f"  seq_len {n}  layers {len(lm.layers)}")

    with torch.no_grad():
        emb = lm.embed_tokens(ids)
        ref = lm(inputs_embeds=emb, use_cache=False).last_hidden_state

        # Rebuild what the stock stack feeds its layers: rotary embeddings and the causal mask.
        pos_ids = torch.arange(n, device=dev).unsqueeze(0)
        pe = lm.rotary_emb(emb, pos_ids)
        causal = torch.full((n, n), torch.finfo(dt).min, device=dev, dtype=dt).triu(1)
        mask = causal.view(1, 1, n, n)

        layers = [ApproxCorrectQwen3DecoderLayer.from_stock(l) for l in lm.layers]
        cache, x = {}, emb
        for i, l in enumerate(layers):
            x, cache = l.approx(x, pe, mask, cache, f"l{i}")
        x_post = lm.norm(x)

        full = torch.ones(1, n, dtype=torch.bool, device=dev)
        c2, y = dict(cache), emb
        for i, l in enumerate(layers):
            y, c2 = l.correct(y, full, pe, mask, c2, f"l{i}")
        y_post = lm.norm(y)

        # A non-prefix selection: if the rotary phase or the mask row were taken by position in the
        # selection rather than by token index, this is where it shows and a prefix test would not.
        sel = torch.zeros(1, n, dtype=torch.bool, device=dev)
        sel[0, ::3] = True
        sel[0, -1] = True
        c3, z = dict(cache), emb
        for i, l in enumerate(layers):
            z, c3 = l.correct(z, sel, pe, mask, c3, f"l{i}")

    def rel(a, b):
        return float((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-9))

    r1, r2 = rel(x_post, ref), rel(y_post, x_post)
    unsel = ~sel[0]
    r3 = rel(z[0][unsel], x[0][unsel])           # untouched rows must be reconstructed exactly
    ok1, ok2, ok3 = r1 < 1e-5, r2 < 1e-5, r3 < 1e-5
    print(f"\n  {'PASS' if ok1 else 'FAIL'}  approx(36 layers) == stock Qwen3        rel {r1:.2e}")
    print(f"  {'PASS' if ok2 else 'FAIL'}  correct@100%      == approx             rel {r2:.2e}")
    print(f"  {'PASS' if ok3 else 'FAIL'}  scattered correct leaves others exact   rel {r3:.2e}"
          f"   ({int(sel.sum())}/{n} selected)")
    ok = ok1 and ok2 and ok3
    print("\n" + ("ALL GATES PASS" if ok else "GATE FAILED"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
