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


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--keeps", type=float, nargs="+", default=[0.30, 0.50])
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS))
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
            return fl.aggregate()

        # --- ceiling: the stock prefill, no arrivals, so it is 100% critical by the rule --------- #
        def ceiling(axis, fl, ids, pp, px, px2):
            axis.full_forward(px, pp, ids)

        ceil_agg = run(ceiling)
        full_g = ceil_agg["mean_total_gflops"]
        print(f"\n══ {ds_name}  (n={len(idxs)}) ══")
        print(f"  full inference (ceiling prefill)   {full_g:10.1f} GFLOPs/instruction")

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

            agg = run(interleaved)
            crit = agg["mean_critical_gflops"]
            tot = agg["mean_total_gflops"]
            rows.append((ds_name, keep, crit, tot, full_g))
            print(f"  interleaved g={a.groups} keep={keep:.0%}  "
                  f"critical {crit:9.1f}  total {tot:9.1f} GFLOPs   "
                  f"critical/full = {100*crit/full_g:5.1f}%   "
                  f"(critical/own total {100*crit/tot:4.1f}%)")

    print("\n\n═══ LLaVA-OneVision-2-8B  ·  interleaved g=%d  ·  critical vs full inference ═══"
          % a.groups)
    print(f"{'dataset':<12}{'keep':>7}{'critical GF':>14}{'full GF':>12}{'% of full':>12}")
    for ds_name, keep, crit, tot, full_g in rows:
        print(f"{ds_name:<12}{keep:>6.0%}{crit:>14.1f}{full_g:>12.1f}{100*crit/full_g:>11.1f}%")


if __name__ == "__main__":
    main()
