"""Critical backbone FLOPs for LLaVA-OneVision-2, interleaved g=4 at two recompute rates.

Reports, per dataset, the compute that can only begin once the server holds the whole image, and
what fraction that is of running the model normally. The denominator is the CEILING's prefill --
"what you would have to do with none of this" -- not the arm's own total, because an arm that does
more work overall but defers less of it past the last byte is the one that wins on latency.

FLOPs are a function of shapes and schedule, not of the answer, so this needs no generation and no
full evaluation set: a strided sample per dataset pins the mean image size, and the numbers are
deterministic given that. Decode is excluded throughout -- it always follows the whole image, so
counting it would raise every arm's critical share by an amount unrelated to the approximation.

    python analysis/experiments/flops_report_ov2.py [--samples 40] [--groups 4]
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from appcorr import flops
from appcorr.models.ov2.unified import OV2UnifiedAxis
from experiments.ov2_degradation import hw_from_grid, l2_from_native
from experiments.ov2_oracle import encode, load, patch_energy
from qwen_vl_prefill.datasets_eval import get_spec

DATASETS = ("chartqa", "textvqa", "infovqa", "docvqa")


def selection(axis, px, px2, keep, cache):
    """The default arm's selection: tokens lead by pooled score, patches derived from them."""
    e = patch_energy(px, px2)
    attn = cache.get("vision_patch_attn_layermean")
    score = e if attn is None else (e / e.mean().clamp_min(1e-12)) * \
        (attn / attn.mean().clamp_min(1e-12)).to(e.device)
    pooled = axis.pool_patch_score(score)
    n_tok = pooled.shape[1]
    k = max(1, int(round(keep * n_tok)))
    sel = torch.zeros_like(pooled, dtype=torch.bool).scatter_(
        1, pooled.topk(k, dim=-1).indices, True)
    return sel, axis.token_mask_to_patch_mask(sel)



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
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--keeps", type=float, nargs="+", default=[0.30, 0.50])
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
    ap.add_argument("--out-json",
                    default="analysis/results/flops/inprocess_flops.json")
    a = ap.parse_args()

    from datasets import load_dataset

    dev, dt = a.device, torch.bfloat16
    model, proc = load(dev, dt)
    rows = []

    for ds_name in a.datasets:
        spec = get_spec(ds_name)
        ds = spec.load(load_dataset)
        idxs = list(range(0, len(ds), max(1, len(ds) // a.samples)))[:a.samples]

        # One session per (dataset, arm) so `aggregate()` is that arm's per-instruction mean.
        # The backbone subtree is the vision tower plus the language model; the lm_head and every
        # task head sit outside it and are therefore never counted.
        def run(fn):
            axis = OV2UnifiedAxis(model.model).eval()
            with flops.session(model.model.visual, model.model.language_model,
                               enabled=True) as fl:
                axis.flops = fl
                for i in idxs:
                    img, prompt, _ = spec.prepare(ds[i], lambda h, w, **kw: (h, w), 1, 1, 1 << 30)
                    enc = encode(proc, img, prompt, dev)
                    ids, pp = enc["input_ids"], enc["patch_positions"]
                    px = enc["pixel_values"].to(dt)
                    deg = l2_from_native(img, a.level, proc,
                                         hw_from_grid(enc["image_grid_thw"], proc))
                    px2 = encode(proc, deg, prompt, dev)["pixel_values"].to(dt)
                    with fl.request(i, n_patch=int(px.shape[0]), seq=int(ids.shape[1])):
                        fn(axis, fl, ids, pp, px, px2)
            return fl.aggregate(), fl

        # --- ceiling: the stock prefill, no arrivals, so it is 100% critical by the rule --------- #
        def ceiling(axis, fl, ids, pp, px, px2):
            axis.full_forward(px, pp, ids)

        ceil_agg, fl_ceiling = run(ceiling)
        full_g = ceil_agg["mean_total_gflops"]
        print(f"\n══ {ds_name}  (n={len(idxs)}) ══")
        print(f"  full inference (ceiling prefill)   {full_g:10.1f} GFLOPs/instruction")

        # --- did the hooks see everything? ------------------------------------------------------ #
        # A path that reaches attention through a fused or varlen kernel never calls
        # `scaled_dot_product_attention`, and the counter then loses the entire quadratic term while
        # every other number still looks plausible. Cross-check the ceiling against a closed form
        # built from the config and the measured token counts, and say so out loud when it drifts.
        vc, tc = model.config.vision_config, model.config.text_config
        n_patch = sum(int(r.meta.get("n_patch", 0)) for r in fl_ceiling.requests) / len(idxs)
        seq = sum(int(r.meta.get("seq", 0)) for r in fl_ceiling.requests) / len(idxs)
        n_tok = n_patch / 4
        vh, th = vc.hidden_size, tc.hidden_size
        vheads = vc.num_attention_heads
        v_layer = (2 * n_patch * vh * (3 * vh + vh) + 2 * 2 * n_patch * n_patch * vh
                   + 2 * 2 * n_patch * vh * vc.intermediate_size)
        kv_dim = tc.num_key_value_heads * (th // tc.num_attention_heads)
        l_layer = (2 * seq * th * (th + 2 * kv_dim + th) + 2 * 2 * seq * seq * th
                   + 3 * 2 * seq * th * tc.intermediate_size)
        analytic = (vc.num_hidden_layers * v_layer + tc.num_hidden_layers * l_layer) / 1e9
        ratio = full_g / analytic if analytic else 0.0
        flag = "OK" if 0.9 <= ratio <= 1.15 else "!! CHECK -- hooks may be missing a path"
        print(f"  independent analytic estimate      {analytic:10.1f} GFLOPs   "
              f"measured/analytic = {ratio:4.2f}  {flag}")

        for keep in a.keeps:
            def interleaved(axis, fl, ids, pp, px, px2, keep=keep):
                freqs = axis.rope_freqs(pp)
                # The score's attention term is collected on the approximate pass, which is arrival
                # 0 work -- it reads only the base image.
                cache = {}
                with fl.arrival(0), fl.stage("approx"):
                    vh, cache = axis.vision_approx(axis.vision_prepare(px2), freqs, cache,
                                                   collect_attn=True)
                sel, pm = selection(axis, px, px2, keep, cache)
                emb, ctx = axis.llm_prepare(ids, axis.project(vh, pp))
                tm = torch.zeros(1, ids.shape[1], dtype=torch.bool, device=ids.device)
                tm[:, ctx["image_positions"]] = sel
                is_text = torch.ones_like(tm)
                is_text[:, ctx["image_positions"]] = False
                axis.interleaved_forward(px, px2, pp, ids, pm, tm | is_text, a.groups)

            agg, _ = run(interleaved)
            crit = agg["mean_critical_gflops"]
            tot = agg["mean_total_gflops"]
            rows.append((ds_name, keep, crit, tot, full_g))
            print(f"  interleaved g={a.groups} keep={keep:.0%}  "
                  f"critical {crit:9.1f}  total {tot:9.1f} GFLOPs   "
                  f"critical/full = {100*crit/full_g:5.1f}%   "
                  f"(critical/own total {100*crit/tot:4.1f}%)")
        _save(a.out_json, "ov2", ds_name, full_g,
              [(k, c, t) for d, k, c, t, _ in rows if d == ds_name],
              len(idxs), a.groups)


    print("\n\n═══ LLaVA-OneVision-2-8B  ·  interleaved g=%d  ·  critical vs full inference ═══"
          % a.groups)
    print(f"{'dataset':<12}{'keep':>7}{'critical GF':>14}{'full GF':>12}{'% of full':>12}")
    for ds_name, keep, crit, tot, full_g in rows:
        print(f"{ds_name:<12}{keep:>6.0%}{crit:>14.1f}{full_g:>12.1f}{100*crit/full_g:>11.1f}%")


if __name__ == "__main__":
    main()
