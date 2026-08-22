"""Contract gates for interleaved correction on the OV2 unified axis.

From docs/memo/interleaved_correction_contract.md, on a real sample rather than noise -- the score
is the real residual-energy x attention one, so the selection has the clustering the schedule is
supposed to exploit.

    1. Coverage       union of per-round groups == the one-shot selection, with no overlap, on BOTH
                      the patch side and the token side. The token side is the one this layout can
                      break by itself: a band edge inside a 2x2 block splits a token across rounds.
    2. Text           text tokens are never selected in any round's image budget.
    3. g=1 identity   the interleaved walk with one group reproduces one-shot. With the rules held
                      this is the SAME computation, so the bar is equality, not "close enough".
    4. Cost           rounds cost less than one-shot, weighted by stage cost -- an encoder layer and
                      a decoder layer are not interchangeable units at 1024 vs 4096 width. Checked
                      against the walk's own counter, not only the formula.
    5. Fidelity       relative L2 against the exact forward is zero only for the exact computation,
                      so nothing can beat it. An interleaved arm that appears to is a stream leak
                      (rule 2), not a discovery.

    python analysis/experiments/ov2_interleaved_gates.py [--dtype float32] [--keep 0.55]
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from appcorr.models.ov2.unified import OV2UnifiedAxis
from experiments.ov2_degradation import hw_from_grid, l2_from_native
from experiments.ov2_oracle import encode, load, patch_energy
from qwen_vl_prefill.datasets_eval import get_spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="chartqa")
    ap.add_argument("--sample", type=int, default=0)
    ap.add_argument("--keep", type=float, default=0.55)
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    a = ap.parse_args()

    from datasets import load_dataset

    dtype = torch.float32 if a.dtype == "float32" else torch.bfloat16
    # g=1 is the same computation, so the only slack is float reassociation from a different loop
    # order. fp32 leaves no room to hide a real defect; bf16 is offered for a fast re-check only.
    rtol = 3e-4 if dtype is torch.float32 else 8e-2
    model, proc = load(a.device, dtype)
    axis = OV2UnifiedAxis(model.model).eval()

    spec = get_spec(a.dataset)
    ds = spec.load(load_dataset)
    img, prompt, _ = spec.prepare(ds[a.sample], lambda h, w, **kw: (h, w), 1, 1, 1 << 30)

    ok = True
    with torch.no_grad():
        enc = encode(proc, img, prompt, a.device)
        ids, pp = enc["input_ids"], enc["patch_positions"]
        px = enc["pixel_values"].to(dtype)
        deg = l2_from_native(img, a.level, proc, hw_from_grid(enc["image_grid_thw"], proc))
        px2 = encode(proc, deg, prompt, a.device)["pixel_values"].to(dtype)
        freqs = axis.rope_freqs(pp)
        n_patch = px.shape[0]
        n_tok = axis.n_tokens(n_patch)
        seq = int(ids.shape[1])

        # The approximate pass over the WHOLE axis -- this is the cache every correction reads
        # from, both halves of it.
        vh, cache = axis.vision_approx(axis.vision_prepare(px2), freqs, {}, collect_attn=True)
        feats_appr = axis.project(vh, pp)
        emb, ctx = axis.llm_prepare(ids, feats_appr)
        _, cache = axis.llm_approx(emb, ctx, cache)

        # The DEFAULT arm's selection: tokens lead, patches derived. Interleaved is a claim about
        # scheduling, so it must share this selection exactly (contract rule 5).
        e = patch_energy(px, px2)
        attn = cache["vision_patch_attn_layermean"]
        score = (e / e.mean().clamp_min(1e-12)) * (attn / attn.mean().clamp_min(1e-12)).to(e.device)
        pooled = axis.pool_patch_score(score)
        tk = max(1, int(round(a.keep * n_tok)))
        sel_tok = torch.zeros_like(pooled, dtype=torch.bool).scatter_(
            1, pooled.topk(tk, dim=-1).indices, True)
        pm = axis.token_mask_to_patch_mask(sel_tok)
        is_text = torch.ones(1, seq, dtype=torch.bool, device=a.device)
        is_text[:, ctx["image_positions"]] = False
        llm_oneshot = torch.zeros(1, seq, dtype=torch.bool, device=a.device)
        llm_oneshot[:, ctx["image_positions"]] = sel_tok

        print(f"  {axis.n_vision}+{axis.n_llm}={axis.n_stages} stages; {n_patch} patches -> "
              f"{n_tok} tokens, seq {seq}")
        print(f"  selection: {int(pm.sum())}/{n_patch} patches, {int(sel_tok.sum())}/{n_tok} "
              f"image tokens, {int(is_text.sum())} text\n")

        # --- 1: coverage, patch side AND token side --------------------------------------------- #
        for gN in (1, 2, 4, 8):
            gp = axis.spatial_groups(pm, gN, pp)
            union = torch.zeros_like(pm)
            for x in gp:
                union |= x
            cov = bool((union == pm).all())
            ov = sum(int((gp[i] & gp[j]).sum()) for i in range(gN) for j in range(i + 1, gN))
            # The token side: each round's touched-and-budgeted tokens must partition sel_tok.
            tg = [axis.patch_mask_any_to_token(x) & sel_tok for x in gp]
            tu = torch.zeros_like(sel_tok)
            for x in tg:
                tu |= x
            tcov = bool((tu == sel_tok).all())
            tov = sum(int((tg[i] & tg[j]).sum()) for i in range(gN) for j in range(i + 1, gN))
            good = cov and ov == 0 and tcov and tov == 0
            ok &= good
            print(f"  {'PASS' if good else 'FAIL'}  g={gN:<2} bounds="
                  f"{str(axis.layer_bounds(gN, n_patch, seq)):<26} patch cov={'OK' if cov else 'X'}"
                  f"/ov={ov}  token cov={'OK' if tcov else 'X'}/ov={tov}")

        # --- 2: text is never in the image budget ----------------------------------------------- #
        txt = int((llm_oneshot & is_text).sum())
        ok &= txt == 0
        print(f"  {'PASS' if txt == 0 else 'FAIL'}  text tokens in the image budget: {txt}\n")

        # --- 3: g=1 identity against one-shot, built exactly as the driver builds it ------------ #
        mixed = torch.where(pm.unsqueeze(-1), axis.vision_prepare(px), axis.vision_prepare(px2))
        vh1, c1 = axis.vision_correct(mixed, pm, freqs, dict(cache))   # copy: keep `cache` pristine
        feats_mixed = torch.where(sel_tok.unsqueeze(-1), axis.project(vh1, pp), feats_appr)
        emb1, _ = axis.llm_prepare(ids, feats_mixed)
        ref, c1 = axis.llm_correct(emb1, llm_oneshot | is_text, ctx, c1)

        hI, cI, iso1 = axis.interleaved_forward(px, px2, pp, ids, pm, llm_oneshot, 1)
        scale = max(ref.float().abs().max().item(), 1e-9)
        err = (hI.float() - ref.float()).abs().max().item() / scale
        ok &= err <= rtol
        print(f"  {'PASS' if err <= rtol else 'FAIL'}  g=1 interleaved == one-shot"
              f"{'':<21} rel {err:.2e}")

        # --- 4: cost signature ------------------------------------------------------------------ #
        costs = axis.stage_costs(n_patch, seq)
        print(f"\n  stage cost share: vision {float(costs[:axis.n_vision].sum()):.3f}, "
              f"llm {float(costs[axis.n_vision:].sum()):.3f}")
        print(f"  one-shot layer-corrections: {axis.n_stages}")
        for gN in (2, 4, 8):
            b = axis.layer_bounds(gN, n_patch, seq)
            gp = axis.spatial_groups(pm, gN, pp)
            tot = sum(float(costs[:b[r]].sum()) * float(gp[r].sum()) for r in range(gN))
            one = 1.0 * float(pm.sum())
            _, _, iso = axis.interleaved_forward(px, px2, pp, ids, pm, llm_oneshot, gN)
            lc = iso["layer_corrections"]
            good = tot / one < 1.0 and lc < axis.n_stages * gN
            ok &= good
            print(f"  {'PASS' if good else 'FAIL'}  g={gN:<2} correction cost "
                  f"{tot/one:.3f}x one-shot   walked layer-corrections {lc}")

        # --- 5: fidelity -- nothing beats the exact forward -------------------------------------- #
        # Reported at TWO scopes, because the whole-sequence number alone is misleading here.
        #
        # A rel L2 over every position is dominated by the image block, and 45% of those tokens are
        # left approximate BY DESIGN -- no arm is trying to improve them, so the metric is mostly
        # measuring the budget rather than the correction. Measured on three ChartQA samples it
        # moves 15-20% while accuracy recovers 96.6% of the floor-ceiling gap, which reads as a
        # contradiction until the scope is fixed.
        #
        # The last position is where the answer is actually read, it is text, and it is corrected in
        # every arm. There the same runs remove 33-71%. Both are reported: the global one is the
        # honest "did anything leak" check (rule 4 -- zero is unreachable except for the exact
        # computation), the last-position one is the informative one.
        exact = axis.full_forward(px, pp, ids)

        def frel(h, sl=slice(None)):
            f = axis.llm_finish(h).float()[:, sl]
            e_ = exact.float()[:, sl]
            return float((f - e_).norm() / e_.norm().clamp_min(1e-9))

        appr_only, _ = axis.llm_approx(emb, ctx, dict(cache))
        rows = [("approx only (floor)", appr_only), ("one-shot", ref)]
        for gN in (2, 4, 8):
            hG, _, _ = axis.interleaved_forward(px, px2, pp, ids, pm, llm_oneshot, gN)
            rows.append((f"interleaved g={gN}", hG))
        print()
        f_all, f_last = frel(rows[0][1]), frel(rows[0][1], slice(-1, None))
        for name, h in rows:
            r_all, r_last = frel(h), frel(h, slice(-1, None))
            beats = r_all <= 0.0 or r_last <= 0.0
            ok &= not beats
            print(f"  {'FAIL' if beats else 'PASS'}  rel L2 vs exact  {name:<20} "
                  f"all {r_all:.4f} ({100*(1-r_all/f_all):4.1f}% removed)   "
                  f"last-pos {r_last:.4f} ({100*(1-r_last/f_last):5.1f}% removed)")

    print("\n" + ("ALL GATES PASS" if ok else "SOME GATES FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
