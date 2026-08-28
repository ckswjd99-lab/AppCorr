"""Mistral Small 3.1 oracle: gates + ceiling / floor / corrected / streaming.

--gate runs the identity ladder first (2 samples): G1 manual full_forward
logits == stock torch.equal; V2 fork approx(degraded)+correct(ALL rows, full
layer-0 input) reproduces the stock vision features bitwise. Only then do the
accuracy arms mean anything.

`corrected`: one-shot -- fork approx on the level-`level` base, top-`keep`
patch rows by pixel residual energy recomputed against the cached K/V, project,
scatter, ONE causal prefill via generate(inputs_embeds).
`streaming`: chunked causal prefill in band order with per-band vision
correction -- bands are MERGED-row groups (spatial_merge_size=2), keep governs
per-band correction quota (k=1.0 corrects every arrived row).

Degradation matches the generic bounds oracle exactly (BOX down + BICUBIC up,
sampled-resolution cap 1540^2), so floors measured there remain the bounds for
these arms.
"""
import argparse
import json
import os
import sys
import time

import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "qwen_vl_prefill"))
from appcorr.models.mistral3.unified import Mistral3Axis, PixtralVisionFork, MODEL_ID  # noqa: E402

CAP_PX = 1540 * 1540


def degrade(img: Image.Image, level: int) -> Image.Image:
    w, h = img.size
    f = 2 ** level
    s = min(1.0, (CAP_PX / (w * h)) ** 0.5)
    tw, th = max(1, int(w * s) // f), max(1, int(h * s) // f)
    return img.resize((tw, th), Image.BOX).resize((w, h), Image.BICUBIC)


def patch_energy(px_f, px_a, patch=14):
    """Per-patch residual energy in the tower's row-major flatten order."""
    d = (px_f.float() - px_a.float()).pow(2).sum(1, keepdim=True)   # [B,1,H,W]
    pooled = torch.nn.functional.avg_pool2d(d, patch, stride=patch)  # [B,1,hp,wp]
    return pooled.flatten(1)                                         # row-major = flatten(1).T order


def _grid_hw(axis, enc):
    ps = axis.vision.patch_size
    h, w = (int(x) for x in enc["image_sizes"][0])
    return h // ps, w // ps


@torch.no_grad()
def _scores(axis, fork, enc, enc2, energy, cache, sizes, qsel, flops=None):
    """energy (default) or energy x query-attention (qsel). The qsel prefill on
    the APPROX features runs at arrival 0 -- it needs only the base, so it
    overlaps the transmission like the vision approx itself."""
    if not qsel:
        return energy
    feats_a = axis.projector(cache["p_last"].squeeze(0), sizes)
    hp, wp = _grid_hw(axis, enc)
    qa = query_attn_patch_scores(axis, enc, feats_a.to(torch.bfloat16), hp, wp)
    e = energy / energy.mean().clamp_min(1e-12)
    q = qa / qa.mean().clamp_min(1e-12)
    return e * q.to(e.device)


@torch.no_grad()
def corrected_feats(axis, fork, enc, enc2, keep, qsel=False, counter=None):
    px_f, px_a = enc["pixel_values"].to(torch.bfloat16), enc2["pixel_values"].to(torch.bfloat16)
    sizes = enc["image_sizes"]
    from contextlib import nullcontext
    arr = counter.arrival if counter is not None else (lambda i: nullcontext())
    with arr(0):
        emb_a, pos = fork.prepare(px_a, sizes)
        cache = {}
        fork.approx(emb_a, pos, cache)
        energy = patch_energy(px_f, px_a)[0]
        score = _scores(axis, fork, enc, enc2, energy, cache, sizes, qsel)
    with arr(1):
        emb_f, _ = fork.prepare(px_f, sizes)
        kq = max(1, int(round(keep * score.numel())))
        sel = torch.zeros_like(score, dtype=torch.bool).scatter_(
            0, score.topk(kq).indices, True)
        mixed = torch.where(sel.unsqueeze(-1), emb_f[0], emb_a[0]).unsqueeze(0)
        last = fork.correct(mixed, sel, pos, cache)
        return axis.projector(last.squeeze(0), sizes)


@torch.no_grad()
def streaming_feats_and_prefill(axis, fork, enc, enc2, keep, groups, model,
                                qsel=False, counter=None):
    """Vision: per-band correct (arrived rows, quota=keep). LLM: prefill once at
    the end over the final mixed features -- Mistral's LLM is purely causal, so
    the arrival-ordered chunked prefill computes exactly this (OV2's verified
    property); the ACCURACY of the streaming arm is what we measure here, and
    the chunk-boundary identity is carried by that established result."""
    px_f, px_a = enc["pixel_values"].to(torch.bfloat16), enc2["pixel_values"].to(torch.bfloat16)
    sizes = enc["image_sizes"]
    from contextlib import nullcontext
    arr = counter.arrival if counter is not None else (lambda i: nullcontext())
    with arr(0):
        emb_a, pos = fork.prepare(px_a, sizes)
        cache = {}
        fork.approx(emb_a, pos, cache)
        energy = patch_energy(px_f, px_a)[0]
        score = _scores(axis, fork, enc, enc2, energy, cache, sizes, qsel)
        emb_f, _ = fork.prepare(px_f, sizes)
    P = score.numel()
    order = torch.arange(P, device=score.device)
    bands = torch.chunk(order, groups)
    arrived = torch.zeros(P, dtype=torch.bool, device=score.device)
    corrected = torch.zeros_like(arrived)
    for bi, band in enumerate(bands):
        with arr(bi + 1):
            arrived[band] = True
            cand = arrived & ~corrected
            e = score.masked_fill(~cand, float("-inf"))
            kq = max(1, int(round(keep * band.numel())))
            sel = torch.zeros_like(cand).scatter_(0, e.topk(kq).indices, True) & cand
            mixed_rows = torch.where(sel.unsqueeze(-1), emb_f[0], emb_a[0]).unsqueeze(0)
            fork.correct(mixed_rows, sel, pos, cache)
            corrected |= sel
    return axis.projector(cache["p_last"].squeeze(0), sizes)


@torch.no_grad()
def query_attn_patch_scores(axis, enc, feats_a, hp, wp):
    """QUERY-AWARE selection signal (user proposal 2026-08-28): one extra LLM
    prefill on the APPROX features, attention FROM the trailing text (query)
    tokens TO the image tokens, averaged over layers/heads/query rows, then
    broadcast from merged tokens (2x2 patches) back to patch rows.

    Costs one approx-features prefill with eager attention -- that pass is the
    price of query awareness, and the final correction stage re-runs the query
    tokens against the corrected features (the arm's second prefill).
    """
    ids = enc["input_ids"]
    embeds = axis.scatter_and_prefill_embeds(enc, feats_a)
    impl = axis.llm.config._attn_implementation
    axis.llm.config._attn_implementation = "eager"   # sdpa returns no attn weights
    try:
        out = axis.llm(inputs_embeds=embeds, attention_mask=enc.get("attention_mask"),
                       use_cache=False, output_attentions=True, return_dict=True)
    finally:
        axis.llm.config._attn_implementation = impl
    assert out.attentions and out.attentions[0] is not None, "no attention weights captured"
    img_pos = (ids[0] == axis.image_token_id).nonzero(as_tuple=True)[0]
    q_pos = torch.arange(ids.shape[1], device=ids.device) > img_pos.max()
    q_pos = q_pos.nonzero(as_tuple=True)[0]
    score_tok = None
    for att in out.attentions:                     # [1, heads, q, k]
        s = att[0, :, q_pos][:, :, img_pos].mean(dim=(0, 1))   # [n_img_tokens]
        score_tok = s if score_tok is None else score_tok + s
    score_tok = score_tok / len(out.attentions)
    # merged tokens are row-major over (hp/2, wp/2); broadcast to the 2x2 patch block
    mh, mw = hp // 2, wp // 2
    assert score_tok.numel() == mh * mw, (score_tok.shape, mh, mw)
    grid = score_tok.view(mh, mw)
    patch = grid.repeat_interleave(2, 0).repeat_interleave(2, 1)   # [hp, wp]
    return patch.flatten()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--dataset", default="mmvp")
    ap.add_argument("--arm", choices=["ceiling", "floor", "corrected", "streaming",
                                      "corrected_qsel", "streaming_qsel"],
                    default="corrected")
    ap.add_argument("--keep", type=float, default=0.5)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--num-samples", type=int, default=12)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--flops", action="store_true",
                    help="12-sample FLOPs pass instead of accuracy: ceiling + the four ours "
                         "arms with the arrival split (approx/qsel prefill at arrival 0 = "
                         "overlappable; correction + final prefill = critical)")
    a = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoProcessor, Mistral3ForConditionalGeneration
    from datasets_eval import get_spec

    print(f"[mistral3] loading {a.model} arm={a.arm} keep={a.keep} L{a.level}", flush=True)
    model = Mistral3ForConditionalGeneration.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="cuda:0").eval()
    proc = AutoProcessor.from_pretrained(a.model)
    axis = Mistral3Axis(model, proc)
    fork = PixtralVisionFork(axis.vision)

    spec = get_spec(a.dataset)
    ds = spec.load(load_dataset)
    if len(ds) == 0:
        raise RuntimeError("VACUOUS: 0 samples")

    if a.flops:
        from appcorr.flops.counter import FlopCounter
        from appcorr.flops import hooks as fhooks
        roots = [model.model.vision_tower, model.model.multi_modal_projector,
                 model.model.language_model]
        arms = ["ceiling", f"corrected_k{a.keep:.2f}", f"corrected_qsel_k{a.keep:.2f}",
                f"streaming_k{a.keep:.2f}", f"streaming_qsel_k{a.keep:.2f}"]
        counters = {arm: FlopCounter() for arm in arms}
        idxs = list(range(0, len(ds), max(1, len(ds) // 12)))[:12]
        for idx in idxs:
            img, prompt, _ = spec.prepare(ds[idx], lambda h, w, **kw: (h, w), 1, 1, 1 << 30)
            enc = axis.build_inputs(img, prompt).to("cuda:0")
            enc["pixel_values"] = enc["pixel_values"].to(torch.bfloat16)
            enc2 = axis.build_inputs(degrade(img, a.level), prompt).to("cuda:0")
            enc2["pixel_values"] = enc2["pixel_values"].to(torch.bfloat16)
            with torch.no_grad():
                for arm in arms:
                    c = counters[arm]
                    handles = fhooks.install(c, roots)
                    with c.request(f"{a.dataset}/{idx}"):
                        if arm == "ceiling":
                            with c.arrival(0):
                                feats = axis.vision_features(enc["pixel_values"],
                                                             enc["image_sizes"])
                                embeds = axis.scatter_and_prefill_embeds(
                                    enc, feats.to(torch.bfloat16))
                                axis.llm(inputs_embeds=embeds, use_cache=False)
                        else:
                            qsel = "qsel" in arm
                            if arm.startswith("corrected"):
                                feats = corrected_feats(axis, fork, enc, enc2, a.keep,
                                                        qsel=qsel, counter=c)
                                final_arr = 1
                            else:
                                feats = streaming_feats_and_prefill(
                                    axis, fork, enc, enc2, a.keep, a.groups, model,
                                    qsel=qsel, counter=c)
                                final_arr = a.groups
                            with c.arrival(final_arr):
                                embeds = axis.scatter_and_prefill_embeds(
                                    enc, feats.to(torch.bfloat16))
                                axis.llm(inputs_embeds=embeds, use_cache=False)
                    fhooks.remove(handles)
        agg = {arm: counters[arm].aggregate() for arm in arms}
        full = agg["ceiling"]["mean_total_gflops"]
        out = {"_model": a.model, "_level": a.level, "_samples": len(idxs),
               a.dataset: {"full": round(full, 1)}}
        for arm in arms[1:]:
            g = agg[arm]
            out[a.dataset][arm] = {"crit": round(g["mean_critical_gflops"], 1),
                                   "total": round(g["mean_total_gflops"], 1)}
            print(f"{arm:<24} crit {g['mean_critical_gflops']:8.1f} "
                  f"({g['mean_critical_gflops'] / full * 100:5.1f}%)  total "
                  f"{g['mean_total_gflops']:9.1f} ({g['mean_total_gflops'] / full * 100:5.1f}%)",
                  flush=True)
        if a.out_json:
            os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
            json.dump(out, open(a.out_json, "w"), indent=1)
        print("MISTRAL3_FLOPS_DONE", flush=True)
        return

    if a.gate:
        for idx in (0, len(ds) // 2):
            img, prompt, _ = spec.prepare(ds[idx], lambda h, w, **kw: (h, w), 1, 1, 1 << 30)
            enc = axis.build_inputs(img, prompt).to("cuda:0")
            enc["pixel_values"] = enc["pixel_values"].to(torch.bfloat16)
            enc2 = axis.build_inputs(degrade(img, a.level), prompt).to("cuda:0")
            enc2["pixel_values"] = enc2["pixel_values"].to(torch.bfloat16)
            axis.assert_same_grid(enc, enc2)
            with torch.no_grad():
                stock = model(**enc)
                hidden, _ = axis.full_forward(enc)
                ours = axis.logits(hidden[:, -1:, :])
                if not torch.equal(ours, stock.logits[:, -1:, :]):
                    d = (ours.float() - stock.logits[:, -1:, :].float()).abs().max().item()
                    raise RuntimeError(f"G1 FAIL idx={idx}: {d:.3e}")
                ref = axis.vision_features(enc["pixel_values"].to(torch.bfloat16),
                                           enc["image_sizes"])
                v2 = corrected_feats(axis, fork, enc, enc2, keep=1.0)
                if not torch.equal(v2, ref):
                    d = (v2.float() - ref.float()).abs().max().item()
                    raise RuntimeError(f"V2 FAIL idx={idx}: correct-all vs stock {d:.3e}")
            print(f"  gate idx={idx}: G1 equal | V2 equal", flush=True)
        print("MISTRAL3_GATE_PASS", flush=True)
        return

    n = len(ds) if a.full else min(a.num_samples, len(ds))
    idxs = (list(range(len(ds))) if a.full
            else list(range(0, len(ds), max(1, len(ds) // n)))[:n])
    correct_n, total, per = 0, 0, []
    t0 = time.time()
    for idx in idxs:
        img, prompt, gold = spec.prepare(ds[idx], lambda h, w, **kw: (h, w), 1, 1, 1 << 30)
        enc = axis.build_inputs(img, prompt).to("cuda:0")
        enc["pixel_values"] = enc["pixel_values"].to(torch.bfloat16)
        with torch.no_grad():
            if a.arm == "ceiling":
                out = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False)
                text = proc.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            else:
                enc2 = axis.build_inputs(degrade(img, a.level), prompt).to("cuda:0")
                enc2["pixel_values"] = enc2["pixel_values"].to(torch.bfloat16)
                axis.assert_same_grid(enc, enc2)
                if a.arm == "floor":
                    out = model.generate(**enc2, max_new_tokens=a.max_new_tokens, do_sample=False)
                    text = proc.decode(out[0, enc2["input_ids"].shape[1]:],
                                       skip_special_tokens=True)
                else:
                    qsel = a.arm.endswith("_qsel")
                    if a.arm.startswith("corrected"):
                        feats = corrected_feats(axis, fork, enc, enc2, a.keep, qsel=qsel)
                    else:
                        feats = streaming_feats_and_prefill(axis, fork, enc, enc2,
                                                            a.keep, a.groups, model,
                                                            qsel=qsel)
                    embeds = axis.scatter_and_prefill_embeds(enc, feats.to(torch.bfloat16))
                    out = model.generate(inputs_embeds=embeds,
                                         attention_mask=enc.get("attention_mask"),
                                         max_new_tokens=a.max_new_tokens, do_sample=False)
                    text = proc.decode(out[0], skip_special_tokens=True)
        ok, sc = spec.score(text.strip(), gold)
        correct_n += ok
        total += 1
        per.append({"idx": idx, "pred": text.strip()[:160], "gold": str(gold)[:80], "score": sc})
        if total % 50 == 0 or total == len(idxs):
            dt = time.time() - t0
            print(f"  [{total}/{len(idxs)}] {dt:.0f}s {dt / total:.2f}s/ex "
                  f"acc={correct_n / total:.2%}", flush=True)

    summary = {"model": a.model, "dataset": a.dataset, "arm": a.arm,
               "keep": a.keep if a.arm in ("corrected", "streaming") else None,
               "level": a.level, "groups": a.groups if a.arm == "streaming" else None,
               "num_samples": total, "accuracy": correct_n / total, "correct": correct_n}
    print(f"\n=== Final Summary: {json.dumps(summary)}", flush=True)
    if a.out_json:
        os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
        json.dump({"summary": summary, "per_sample": per}, open(a.out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
