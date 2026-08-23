"""Critical backbone FLOPs for Gemma 3, interleaved g=4 at two recompute rates.

Same question and the same denominator as the OV2 report: what fraction of a normal forward can
only begin once the whole image is in hand. Two things differ and both are properties of the model,
not of the measurement.

Gemma 3 resizes every image to a fixed 896x896 canvas and emits exactly 256 image tokens, so unlike
OV2 the per-instruction FLOPs barely move across datasets -- only the prompt length varies. And its
vision tower is 27 SigLIP layers over 4096 patches against 34 decoder layers over ~272 positions,
a 1.8x LLM/vision ratio where OV2's is 6.0x, so the two halves contribute very differently to what
lands after the last byte.

    python analysis/experiments/flops_report_gemma3.py [--samples 24] [--groups 4]
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from appcorr import flops
from appcorr.models.gemma3.unified import Gemma3UnifiedAxis
from experiments.gemma3_oracle import l2_from_native, patch_energy
from qwen_vl_prefill.datasets_eval import get_spec

DATASETS = ("chartqa", "textvqa", "infovqa")



def _save(path, model, dataset, full_gf, arms, samples, groups):
    """Append one (model, dataset) block to a JSON record.

    Written on every run because the alternative is scraping stdout: total FLOPs existed only in a
    log until it was needed for the overhead column, and had to be recovered by grep.
    """
    import json as _json, os as _os
    rec = {}
    if _os.path.exists(path):
        try:
            rec = _json.load(open(path))
        except Exception:
            rec = {}
    rec.setdefault("_meta", {}).update({"groups": groups, "unit": "GFLOPs/instruction",
                                        "note": "backbone prefill only; decode excluded"})
    m = rec.setdefault(model, {})
    m["_samples"] = samples
    d = m.setdefault(dataset, {})
    d["full"] = round(full_gf, 1)
    for keep, crit, tot in arms:
        d[f"k{keep:.2f}"] = round(crit, 1)
        d[f"total_k{keep:.2f}"] = round(tot, 1)
    _os.makedirs(_os.path.dirname(_os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        _json.dump(rec, f, indent=2)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--keeps", type=float, nargs="+", default=[0.30, 0.50])
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--out-json",
                    default="analysis/results/flops/inprocess_flops.json")
    a = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    dev, dt = a.device, torch.bfloat16
    tok = os.environ.get("HF_TOKEN")
    model = Gemma3ForConditionalGeneration.from_pretrained(a.model, dtype=dt, token=tok).eval()
    model = model.to(dev)
    proc = AutoProcessor.from_pretrained(a.model, token=tok)
    size = proc.image_processor.size
    cap = int(size["height"] if isinstance(size, dict) else size.height)
    patch = int(model.config.vision_config.patch_size)
    rows = []

    for ds_name in a.datasets:
        spec = get_spec(ds_name)
        ds = spec.load(load_dataset)
        idxs = list(range(0, len(ds), max(1, len(ds) // a.samples)))[:a.samples]

        def run(fn):
            axis = Gemma3UnifiedAxis(model.model).eval()
            # Backbone = vision tower + language model. `lm_head` and the multimodal projector's
            # consumers outside these two are not counted.
            with flops.session(model.model.vision_tower, model.model.language_model,
                               enabled=True) as fl:
                axis.flops = fl
                for i in idxs:
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
                    px2 = proc.apply_chat_template(
                        msgs2, add_generation_prompt=True, tokenize=True, return_dict=True,
                        return_tensors="pt")["pixel_values"].to(dev, dt)
                    with fl.request(i, seq=int(ids.shape[1])):
                        fn(axis, fl, ids, tti, px, px2)
            return fl.aggregate()

        def ceiling(axis, fl, ids, tti, px, px2):
            axis.full_forward(px, ids, tti)

        full_g = run(ceiling)["mean_total_gflops"]
        print(f"\n══ {ds_name}  (n={len(idxs)}) ══")
        print(f"  full inference (ceiling prefill)   {full_g:10.1f} GFLOPs/instruction")

        for keep in a.keeps:
            def interleaved(axis, fl, ids, tti, px, px2, keep=keep):
                cache = {}
                with fl.arrival(0), fl.stage("approx"):
                    vh, cache = axis.vision_approx(axis.vision_prepare(px2), cache,
                                                   collect_attn=True)
                score = patch_energy(px, px2, patch)
                attn = cache.get("vision_patch_attn_layermean")
                if attn is not None:
                    score = (score / score.mean().clamp_min(1e-12)) * \
                        (attn / attn.mean().clamp_min(1e-12)).to(score.device)
                feats_appr = axis.project(vh)
                _, ctx = axis.llm_prepare(ids, feats_appr, tti)
                n_img = ctx["image_positions"].numel()
                # The composing selection: tokens lead, patches derived, so the per-round patch
                # groups map onto token groups (the identity interleaving depends on).
                pooled = axis.pool_patch_score(score)
                tk = max(1, int(round(keep * n_img)))
                sel = torch.zeros_like(pooled, dtype=torch.bool).scatter_(
                    1, pooled.topk(tk, dim=-1).indices, True)
                pm = axis.token_mask_to_patch_mask(sel, score.shape[1])
                tm = torch.zeros(ids.shape[0], ids.shape[1], dtype=torch.bool, device=dev)
                tm[:, ctx["image_positions"]] = sel
                is_text = torch.ones_like(tm)
                is_text[:, ctx["image_positions"]] = False
                axis.interleaved_forward(px, px2, ids, tti, pm, tm | is_text, a.groups)

            agg = run(interleaved)
            crit, tot = agg["mean_critical_gflops"], agg["mean_total_gflops"]
            rows.append((ds_name, keep, crit, tot, full_g))
            print(f"  interleaved g={a.groups} keep={keep:.0%}  "
                  f"critical {crit:9.1f}  total {tot:9.1f} GFLOPs   "
                  f"critical/full = {100*crit/full_g:5.1f}%")
        _save(a.out_json, "gemma3", ds_name, full_g,
              [(k, c, t) for d, k, c, t, _ in rows if d == ds_name],
              len(idxs), a.groups)


    print(f"\n\n═══ Gemma 3 4B-IT  ·  interleaved g={a.groups}  ·  critical vs full inference ═══")
    print(f"{'dataset':<12}{'keep':>7}{'critical GF':>14}{'full GF':>12}{'% of full':>12}")
    for ds_name, keep, crit, tot, full_g in rows:
        print(f"{ds_name:<12}{keep:>6.0%}{crit:>14.1f}{full_g:>12.1f}{100*crit/full_g:>11.1f}%")


if __name__ == "__main__":
    main()
