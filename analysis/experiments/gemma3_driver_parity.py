"""Is the interleaved arm's g=1 wiring identical to corrected_t, at the DRIVER level?

The axis gate answers this for a synthetic case and passed at rel 0.00e+00, yet the driver's 40-
sample accuracies differed by one sample. Accuracy cannot tell a structural fault from bf16 noise,
so this runs BOTH driver paths in ONE process on the SAME real images and compares the masks and
the tensors. Masks must be exactly equal; the hidden state must agree to bf16 noise.
"""
import argparse, os, sys, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiments.gemma3_oracle import l2_from_native, patch_energy, get_spec
from appcorr.models.gemma3.unified import Gemma3UnifiedAxis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--dataset", default="chartqa")
    ap.add_argument("--keep", type=float, default=0.55)
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration
    dtype, tok = torch.bfloat16, os.environ.get("HF_TOKEN")
    model = Gemma3ForConditionalGeneration.from_pretrained(a.model, dtype=dtype, token=tok).eval().to(a.device)
    proc = AutoProcessor.from_pretrained(a.model, token=tok)
    axis = Gemma3UnifiedAxis(model.model).eval()

    spec = get_spec(a.dataset)
    ds = spec.load(load_dataset)
    idxs = list(range(0, len(ds), max(1, len(ds) // a.num_samples)))[:a.num_samples]
    size = proc.image_processor.size
    cap = int(size["height"] if isinstance(size, dict) else size.height)
    patch = int(axis.cfg.vision_config.patch_size) if hasattr(axis.cfg, "vision_config") else 14

    worst = 0.0
    for j, i in enumerate(idxs):
        img, prompt, _ = spec.prepare(ds[i], lambda h, w, **kw: (h, w), 1, 1, 1 << 30)
        msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": prompt}]}]
        enc = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors="pt").to(a.device)
        px, ids, tti = enc["pixel_values"].to(dtype), enc["input_ids"], enc.get("token_type_ids")
        deg = l2_from_native(img, a.level, cap)
        m2 = [{"role": "user", "content": [{"type": "image", "image": deg},
                                           {"type": "text", "text": prompt}]}]
        px2 = proc.apply_chat_template(m2, add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors="pt")["pixel_values"].to(a.device, dtype)

        with torch.no_grad():
            vh_appr, cache = axis.vision_approx(axis.vision_prepare(px2), {}, collect_attn=True)
            feats_appr = axis.project(vh_appr)
            emb_appr, ctx = axis.llm_prepare(ids, feats_appr, tti)
            n_img = ctx["image_positions"].numel()
            _, cache = axis.llm_approx(emb_appr, ctx, cache)

            score = patch_energy(px, px2, patch)
            attn = cache["vision_patch_attn_layermean"]
            score = (score / score.mean().clamp_min(1e-12)) * (attn / attn.mean().clamp_min(1e-12)).to(score.device)
            pooled = axis.pool_patch_score(score)
            tk = max(1, int(round(a.keep * n_img)))
            sel_tok = torch.zeros_like(pooled, dtype=torch.bool).scatter_(1, pooled.topk(tk, -1).indices, True)
            pm = axis.token_mask_to_patch_mask(sel_tok, score.shape[1])
            tm = torch.zeros(ids.shape[0], ids.shape[1], dtype=torch.bool, device=a.device)
            tm[:, ctx["image_positions"]] = sel_tok
            is_text = torch.ones_like(tm); is_text[:, ctx["image_positions"]] = False
            tm = tm | is_text

            # --- path A: corrected_t ---
            mixed = torch.where(pm.unsqueeze(-1), axis.vision_prepare(px), axis.vision_prepare(px2))
            vh_c, cacheA = axis.vision_correct(mixed, pm, dict(cache))
            fm = torch.where(sel_tok.unsqueeze(-1), axis.project(vh_c), feats_appr)
            embA, _ = axis.llm_prepare(ids, fm, tti)
            hA, cacheA = axis.llm_correct(embA, tm, ctx, cacheA)

            # --- path B: interleaved g=1 ---
            hB, cacheB = axis.interleaved_forward(px, px2, ids, tti, pm, tm, 1)

            # --- mask parity, the structural question ---
            gp = axis.spatial_groups(pm, 1)[0]
            arrived = axis.patch_mask_any_to_token(pm) & tm[:, ctx["image_positions"]]
            mask_ok = bool((gp == pm).all()) and bool((arrived == sel_tok).all())

            # The K/V cache matters as much as the hidden state: generation decodes on top of it,
            # so identical hidden with a divergent cache still yields a different answer from the
            # second token on. Comparing only `hidden` would have missed that entirely.
            cache_rel = 0.0
            for li in range(axis.n_llm):
                for w in ("k", "v"):
                    ta, tb = cacheA[f"l{li}_{w}"].float(), cacheB[f"l{li}_{w}"].float()
                    cache_rel = max(cache_rel, (ta - tb).abs().max().item()
                                    / max(ta.abs().max().item(), 1e-9))

            sc = max(hA.float().abs().max().item(), 1e-9)
            rel = (hA.float() - hB.float()).abs().max().item() / sc
            tA = int(model.lm_head(axis.llm_finish(hA)[:, -1:])[:, -1].argmax(-1))
            tB = int(model.lm_head(axis.llm_finish(hB)[:, -1:])[:, -1].argmax(-1))
            worst = max(worst, rel, cache_rel)
            print(f"  [{j}] masks={'OK' if mask_ok else 'MISMATCH'} "
                  f"patches={int(pm.sum())} sel_tok={int(sel_tok.sum())} "
                  f"rel={rel:.2e} kv={cache_rel:.2e} "
                  f"first_token {'same' if tA == tB else f'DIFFER {tA} vs {tB}'}", flush=True)

    print(f"\nworst rel over {len(idxs)} images: {worst:.2e}")
    print("bf16 eps is ~7.8e-3; a structural fault would be orders above that, not near it.")


if __name__ == "__main__":
    main()
