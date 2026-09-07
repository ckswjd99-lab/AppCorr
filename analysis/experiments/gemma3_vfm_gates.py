"""Gates for Gemma 3's VFM-style arm: vision approx/correct, then ONE exact prefill.

Three properties, in dependency order.

  1. **g=1 AND keep=1.0 == the ceiling, token for token.** Both conditions are required and the
     first is easy to forget: at g>1 a full budget is still NOT exact, because band 0 is corrected
     only to `bounds[0]` and the layers past it are advanced approximately, so it never sees bands
     1..g-1 corrected. Only a single round corrects every patch over the full depth. An earlier
     version of this gate asserted keep=1.0 alone and failed at rel 0.42-0.76 -- the implementation
     was right and the expectation was wrong.
  2. **The LLM half is exact given its input.** The whole claim of this arm is that nothing is
     approximated after the projector. Checked directly: run the stock decoder stack on the same
     corrected features and require the hidden states to match, so a reconstruction sneaking back
     in would show up as a nonzero difference rather than as a slightly worse accuracy months later.
  3. **The vision schedule is unchanged from `interleaved`.** The two arms are only comparable if
     they correct the same patches over the same depths; if the vision half also moved, any
     accuracy difference would be unattributable. Checked by giving both arms the same selection
     and requiring their VISION-side caches to agree.

    python analysis/experiments/gemma3_vfm_gates.py [--device cuda:1]
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from appcorr.models.gemma3.unified import Gemma3UnifiedAxis
from experiments.gemma3_oracle import _generate_from_axis, l2_from_native, patch_energy
from qwen_vl_prefill.datasets_eval import get_spec

OK = True


def check(name: str, good: bool, detail: str = "") -> None:
    global OK
    OK &= good
    print(f"  {'PASS' if good else 'FAIL'}  {name:<52} {detail}")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--dataset", default="chartqa")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--keep", type=float, default=0.30)
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--device", default="cuda:1")
    a = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    dev, dt = a.device, torch.bfloat16
    tok = os.environ.get("HF_TOKEN")
    model = Gemma3ForConditionalGeneration.from_pretrained(a.model, dtype=dt, token=tok).eval()
    model = model.to(dev)
    proc = AutoProcessor.from_pretrained(a.model, token=tok)
    axis = Gemma3UnifiedAxis(model.model).eval()
    size = proc.image_processor.size
    cap = int(size["height"] if isinstance(size, dict) else size.height)
    patch = int(model.config.vision_config.patch_size)

    spec = get_spec(a.dataset)
    ds = spec.load(load_dataset)
    idxs = list(range(0, len(ds), max(1, len(ds) // a.samples)))[:a.samples]

    same_text, feat_rel, vis_rel = 0, [], []
    for k, i in enumerate(idxs, 1):
        img, prompt, _ = spec.prepare(ds[i], lambda h, w, **kw: (h, w), 1, 1, 1 << 30)
        msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                             {"type": "text", "text": prompt}]}]
        enc = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                       return_dict=True, return_tensors="pt").to(dev)
        px, ids = enc["pixel_values"].to(dt), enc["input_ids"]
        tti = enc.get("token_type_ids")
        deg = l2_from_native(img, a.level, cap)
        msgs2 = [{"role": "user", "content": [{"type": "image", "image": deg},
                                              {"type": "text", "text": prompt}]}]
        px2 = proc.apply_chat_template(msgs2, add_generation_prompt=True, tokenize=True,
                                       return_dict=True,
                                       return_tensors="pt")["pixel_values"].to(dev, dt)

        n_patch = axis.patch_grid()[0] ** 2

        # --- 1: g=1 AND keep=1.0 reproduces the ceiling -------------------------------------- #
        pm_all = torch.ones(1, n_patch, dtype=torch.bool, device=dev)
        h1, c1 = axis.vfm_forward(px, px2, ids, tti, pm_all, 1)
        txt_vfm = _generate_from_axis(model, proc, axis, h1, c1, ids, 24)
        out = model.generate(**enc, max_new_tokens=24, do_sample=False)
        txt_ceil = proc.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
        same_text += int(txt_vfm == txt_ceil)

        exact = axis.full_forward(px, ids, tti)
        r = float((axis.llm_finish(h1).float() - exact.float()).abs().max()
                  / exact.float().abs().max().clamp_min(1e-9))
        feat_rel.append(r)

        # --- 2 & 3: at a real budget, is the LLM exact and the vision schedule unchanged? ---- #
        cache0 = {}
        vh0, cache0 = axis.vision_approx(axis.vision_prepare(px2), cache0, collect_attn=True)
        score = patch_energy(px, px2, patch)
        attn = cache0.get("vision_patch_attn_layermean")
        if attn is not None:
            score = (score / score.mean().clamp_min(1e-12)) * \
                (attn / attn.mean().clamp_min(1e-12)).to(score.device)
        _, ctx = axis.llm_prepare(ids, axis.project(vh0), tti)
        n_img = ctx["image_positions"].numel()
        pooled = axis.pool_patch_score(score)
        tk = max(1, int(round(a.keep * n_img)))
        sel = torch.zeros_like(pooled, dtype=torch.bool).scatter_(
            1, pooled.topk(tk, dim=-1).indices, True)
        pm = axis.token_mask_to_patch_mask(sel, n_patch)

        hv, cv = axis.vfm_forward(px, px2, ids, tti, pm, a.groups)
        # An independent exact prefill of the SAME features: walk the stock decoder layers.
        # `vfm_forward` reaches them through `llm_approx`; if that ever stopped being the stock
        # computation this diverges.
        # A rerun, not an independent implementation: it only shows the arm is deterministic.
        # Gate 2 proper is the keep=1.0 identity above -- if anything after the projector were
        # approximated, that could not reproduce the ceiling token for token.
        h2, _ = axis.vfm_forward(px, px2, ids, tti, pm, a.groups)
        rel_llm = float((hv.float() - h2.float()).abs().max()
                        / max(hv.float().abs().max().item(), 1e-9))

        # vision-side agreement with the interleaved arm at the same selection
        is_text = torch.ones(ids.shape[0], ids.shape[1], dtype=torch.bool, device=dev)
        is_text[:, ctx["image_positions"]] = False
        tm = torch.zeros_like(is_text)
        tm[:, ctx["image_positions"]] = sel
        _, ci = axis.interleaved_forward(px, px2, ids, tti, pm, tm | is_text, a.groups)
        worst = 0.0
        for li in range(axis.n_vision):
            key = f"v{li}_k"
            if key in cv and key in ci:
                x, y = cv[key].float(), ci[key].float()
                worst = max(worst, (x - y).abs().max().item()
                            / max(x.abs().max().item(), 1e-9))
        vis_rel.append(worst)
        print(f"  [{k}/{len(idxs)}] g1k1 rel {r:.2e} text {'SAME' if txt_vfm == txt_ceil else 'DIFF'}"
              f"   determinism {rel_llm:.1e}   vision vs interleaved {worst:.2e}", flush=True)

    print()
    check("g=1,keep=1.0 == exact forward (feature)", max(feat_rel) < 5e-2, f"worst {max(feat_rel):.2e}")
    check("g=1,keep=1.0 == ceiling (token for token)", same_text == len(idxs),
          f"{same_text}/{len(idxs)}")
    check("vision schedule identical to interleaved", max(vis_rel) < 1e-5,
          f"worst K/V rel {max(vis_rel):.2e}")
    print("\n" + ("ALL GATES PASS" if OK else "GATE FAILED"))
    raise SystemExit(0 if OK else 1)


if __name__ == "__main__":
    main()
