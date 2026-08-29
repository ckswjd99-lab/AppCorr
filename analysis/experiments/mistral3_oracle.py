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
import re
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


def degrade(img: Image.Image, level: int, filt: str = "pyr") -> Image.Image:
    """filt="pyr" (default since the 2026-08-28 filter decision: Option B, pyr going forward)
    walks the protocol archetype itself -- cv2.pyrDown chain, cv2.pyrUp back with per-step
    dstsize, mirroring laplacian.py's _iterative_upsample_native (ported from
    qwen35_accuracy.py's pyr branch). "box"/"bicubic" retained for reproducing older numbers
    only; box was the probe chain's outlier (+4pp floor)."""
    w, h = img.size
    f = 2 ** level
    s = min(1.0, (CAP_PX / (w * h)) ** 0.5)
    if filt == "pyr":
        import cv2
        import numpy as np
        if s < 1.0:
            w2, h2 = max(1, int(w * s)), max(1, int(h * s))
            arr = np.asarray(img.resize((w2, h2), Image.BILINEAR))
        else:
            arr = np.asarray(img)
        sizes = [(arr.shape[1], arr.shape[0])]
        for _ in range(level):
            arr = cv2.pyrDown(arr)
            sizes.append((arr.shape[1], arr.shape[0]))
        for i in range(level - 1, -1, -1):
            arr = cv2.pyrUp(arr, dstsize=sizes[i])
        out = Image.fromarray(arr)
        return out if s == 1.0 else out.resize((w, h), Image.BICUBIC)
    tw, th = max(1, int(w * s) // f), max(1, int(h * s) // f)
    down = Image.BOX if filt == "box" else Image.BICUBIC
    return img.resize((tw, th), down).resize((w, h), Image.BICUBIC)


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
def chunked_prefill_logits(model, embeds, attn, ranges, counter=None, arrival_offset=0):
    """TRUE arrival-ordered chunked prefill for a purely causal LLM: feed each contiguous chunk
    with the growing past_key_values -- no fork needed, the HF cache IS the mechanism. Returns the
    final position's logits. Exact by the causal property (each chunk's queries attend exactly the
    prefix, identical to one whole-sequence prefill); the --gate-chunked mode measures that claim
    per model/dtype instead of assuming it. `ranges`: [(s, e), ...] covering [0, T) in order."""
    from contextlib import nullcontext
    arr = counter.arrival if counter is not None else (lambda i: nullcontext())
    past = None
    out = None
    for ci, (s, e) in enumerate(ranges):
        with arr(ci + arrival_offset):
            out = model(inputs_embeds=embeds[:, s:e],
                        attention_mask=attn[:, :e] if attn is not None else None,
                        past_key_values=past, use_cache=True)
            past = out.past_key_values
    return out.logits[:, -1:, :]


def llm_chunk_ranges(enc, axis, groups):
    """Contiguous LLM chunk ranges aligned to vision bands: band r's chunk ends right after its
    last image-feature position (interleaved [IMG_BREAK] rows ride in whichever chunk covers
    them -- causal, so placement is harmless); chunk 0 swallows the leading text, the final chunk
    runs to the end of the sequence (trailing text). Mirrors the Qwen2.5 streaming executor's
    frontier construction."""
    ids = enc["input_ids"]
    img_pos = (ids[0] == axis.image_token_id).nonzero(as_tuple=True)[0]
    P = img_pos.numel()
    T = ids.shape[1]
    bounds = [img_pos[min(P - 1, ((bi + 1) * P) // groups - 1)].item() + 1 for bi in range(groups)]
    ranges, start = [], 0
    for bi, b in enumerate(bounds):
        end = T if bi == groups - 1 else b
        if end > start:
            ranges.append((start, end))
        start = end
    return ranges


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
    ap.add_argument("--decode", choices=["ids", "embeds"], default="ids",
                    help="bound-arm decode path. 'embeds' = vision_features -> scatter -> "
                         "generate(inputs_embeds), the SAME mechanism the fork arms use -- "
                         "Option C (2026-08-29) for the TextVQA decode-mechanism confound: "
                         "unify the mechanism across arms instead of changing the scorer.")
    ap.add_argument("--filt", choices=["pyr", "box", "bicubic"], default="pyr",
                    help="degradation filter; pyr is the standard since the 2026-08-28 Option-B decision, box/bicubic only to reproduce older rows")
    ap.add_argument("--num-samples", type=int, default=12)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--bs", type=int, default=1,
                    help="batch size for the STOCK bound arms (ceiling/floor) only -- they are plain generate() calls and batch cleanly with left padding; the fork arms stay bs=1 by construction. Validated bs1-vs-bs8 equivalent on 50 samples before first use.")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--gate", action="store_true")
    ap.add_argument("--gate-chunked", action="store_true",
                    help="chunked-vs-single prefill logits equivalence on strided samples "
                         "(the causal-equivalence property, measured rather than assumed)")
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
            enc2 = axis.build_inputs(degrade(img, a.level, a.filt), prompt).to("cuda:0")
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
                                with c.arrival(final_arr):
                                    embeds = axis.scatter_and_prefill_embeds(
                                        enc, feats.to(torch.bfloat16))
                                    axis.llm(inputs_embeds=embeds, use_cache=False)
                            else:
                                feats = streaming_feats_and_prefill(
                                    axis, fork, enc, enc2, a.keep, a.groups, model,
                                    qsel=qsel, counter=c)
                                # TRUE chunked prefill (2026-08-29): chunk r rides arrival r
                                # alongside its vision band, so only the FINAL chunk (trailing
                                # text + last band) is critical -- the single-prefill shortcut
                                # measured the accuracy correctly (causal equivalence, see
                                # --gate-chunked) but charged the WHOLE prefill as critical
                                # (~96% of full). FLOP counts per chunk depend only on shapes,
                                # so prefilling the final embeds' slices is count-identical to
                                # prefilling each round's own values.
                                embeds = axis.scatter_and_prefill_embeds(
                                    enc, feats.to(torch.bfloat16))
                                ranges = llm_chunk_ranges(enc, axis, a.groups)
                                chunked_prefill_logits(model, embeds,
                                                       enc.get("attention_mask"),
                                                       ranges, counter=c, arrival_offset=1)
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

    if a.gate_chunked:
        idxs = list(range(0, len(ds), max(1, len(ds) // 8)))[:8]
        worst = 0.0
        for idx in idxs:
            img, prompt, _ = spec.prepare(ds[idx], lambda h, w, **kw: (h, w), 1, 1, 1 << 30)
            enc = axis.build_inputs(img, prompt).to("cuda:0")
            enc["pixel_values"] = enc["pixel_values"].to(torch.bfloat16)
            with torch.no_grad():
                feats = axis.vision_features(enc["pixel_values"], enc["image_sizes"])
                embeds = axis.scatter_and_prefill_embeds(enc, feats.to(torch.bfloat16))
                single = model(inputs_embeds=embeds, use_cache=False).logits[:, -1:, :]
                ranges = llm_chunk_ranges(enc, axis, a.groups)
                chunked = chunked_prefill_logits(model, embeds, enc.get("attention_mask"), ranges)
            d = (single.float() - chunked.float()).abs().max().item()
            worst = max(worst, d)
            tok_eq = bool(single.argmax(-1).item() == chunked.argmax(-1).item())
            bit = bool(torch.equal(single, chunked))
            print(f"  idx={idx}: ranges={len(ranges)} max_abs_diff={d:.3e} "
                  f"argmax_equal={tok_eq} bitwise={bit}", flush=True)
        print(f"MISTRAL3_CHUNKED_GATE worst={worst:.3e}", flush=True)
        return

    if a.gate:
        for idx in (0, len(ds) // 2):
            img, prompt, _ = spec.prepare(ds[idx], lambda h, w, **kw: (h, w), 1, 1, 1 << 30)
            enc = axis.build_inputs(img, prompt).to("cuda:0")
            enc["pixel_values"] = enc["pixel_values"].to(torch.bfloat16)
            enc2 = axis.build_inputs(degrade(img, a.level, a.filt), prompt).to("cuda:0")
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

    # Incremental per-sample dump: the out-json is written only at the END, so a crash at
    # sample 8000/8811 used to lose everything. Append-only jsonl beside it, flushed per sample.
    inc_f = None
    if a.out_json:
        os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
        inc_f = open(a.out_json + ".persample.jsonl", "a", encoding="utf-8")

    def _score_and_log(idx, img, text):
        nonlocal correct_n, total
        pred_for_score = text.strip()
        if a.dataset in ("refcoco", "visdrone_det"):
            m_nums = re.findall(r"-?\d+\.?\d*", pred_for_score)
            if len(m_nums) >= 4 and all(abs(float(v)) <= 1.5 for v in m_nums[:4]):
                W, H = img.size
                vals = (float(m_nums[0]) * W, float(m_nums[1]) * H,
                        float(m_nums[2]) * W, float(m_nums[3]) * H)
                pred_for_score = " ".join(f"{v:.1f}" for v in vals)
        img_gold = _gold_by_idx[idx]
        ok, sc = spec.score(pred_for_score, img_gold)
        correct_n += ok
        total += 1
        row = {"idx": idx, "pred": text.strip()[:160], "gold": str(img_gold)[:80], "score": sc}
        per.append(row)
        if inc_f is not None:
            inc_f.write(json.dumps(row) + "\n"); inc_f.flush()
        if total % 50 == 0 or total == len(idxs):
            dt = time.time() - t0
            print(f"  [{total}/{len(idxs)}] {dt:.0f}s {dt / total:.2f}s/ex "
                  f"acc={correct_n / total:.2%}", flush=True)

    _gold_by_idx = {}

    _ours_queue = []
    _feats_cache = {}

    def _flush_ours():
        if not _ours_queue:
            return
        D = _ours_queue[0][2].shape[-1]
        Tm = max(e.shape[0] for _, _, e, _ in _ours_queue)
        B = len(_ours_queue)
        dev = _ours_queue[0][2].device
        be = torch.zeros(B, Tm, D, dtype=torch.bfloat16, device=dev)
        bm = torch.zeros(B, Tm, dtype=torch.long, device=dev)
        for j, (_, _, e, m) in enumerate(_ours_queue):
            T = e.shape[0]
            be[j, Tm - T:] = e          # LEFT padding: content right-aligned
            bm[j, Tm - T:] = m
        with torch.no_grad():
            outg = model.generate(inputs_embeds=be, attention_mask=bm,
                                  max_new_tokens=a.max_new_tokens, do_sample=False)
        for j, (qidx, qimg, _, _) in enumerate(_ours_queue):
            _score_and_log(qidx, qimg, proc.decode(outg[j], skip_special_tokens=True))
        _ours_queue.clear()

    if a.arm in ("ceiling", "floor") and a.bs > 1 and a.decode == "ids":
        # Batched bound path: plain stock generate over left-padded batches. The processor's
        # chat template is applied per sample (identical to build_inputs), then tokenizer-level
        # left padding assembles the batch; pixel_values ride as the per-sample list the
        # Pixtral processor emits for batched multi-image inputs.
        proc.tokenizer.padding_side = "left"
        for b0 in range(0, len(idxs), a.bs):
            chunk = idxs[b0:b0 + a.bs]
            msgs_list, imgs_list = [], []
            for idx in chunk:
                img, prompt, gold = spec.prepare(ds[idx], lambda h, w, **kw: (h, w), 1, 1, 1 << 30)
                if a.arm == "floor":
                    img = degrade(img, a.level, a.filt)
                _gold_by_idx[idx] = gold
                imgs_list.append(img)
                msgs_list.append([{"role": "user", "content": [{"type": "image", "image": img},
                                                               {"type": "text", "text": prompt}]}])
            enc = proc.apply_chat_template(
                msgs_list, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt", padding=True).to("cuda:0")
            enc["pixel_values"] = enc["pixel_values"].to(torch.bfloat16)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False)
            T = enc["input_ids"].shape[1]
            for j, idx in enumerate(chunk):
                text = proc.decode(out[j, T:], skip_special_tokens=True)
                _score_and_log(idx, imgs_list[j], text)
        if inc_f is not None:
            inc_f.close()
        summary = {"model": a.model, "dataset": a.dataset, "arm": a.arm, "keep": None,
                   "level": a.level, "groups": None, "num_samples": total,
                   "accuracy": correct_n / max(total, 1), "correct": correct_n,
                   "mean_score": sum(r["score"] for r in per) / max(total, 1)}
        out = {"summary": summary, "per_sample": per}
        if a.out_json:
            json.dump(out, open(a.out_json, "w"), indent=1)
        print(f"=== Final Summary: {json.dumps(summary)}", flush=True)
        return

    for idx in idxs:
        img, prompt, gold = spec.prepare(ds[idx], lambda h, w, **kw: (h, w), 1, 1, 1 << 30)
        _gold_by_idx[idx] = gold
        enc = axis.build_inputs(img, prompt).to("cuda:0")
        enc["pixel_values"] = enc["pixel_values"].to(torch.bfloat16)
        with torch.no_grad():
            if a.arm in ("ceiling", "floor") and a.decode == "embeds":
                enc_use = enc
                if a.arm == "floor":
                    enc_use = axis.build_inputs(degrade(img, a.level, a.filt), prompt).to("cuda:0")
                    enc_use["pixel_values"] = enc_use["pixel_values"].to(torch.bfloat16)
                feats = axis.vision_features(enc_use["pixel_values"].to(torch.bfloat16),
                                             enc_use["image_sizes"])
                embeds = axis.scatter_and_prefill_embeds(enc_use, feats.to(torch.bfloat16))
                if a.bs > 1:
                    _ours_queue.append((idx, img, embeds[0],
                                        enc_use["attention_mask"][0] if enc_use.get("attention_mask") is not None
                                        else torch.ones(embeds.shape[1], dtype=torch.long,
                                                        device=embeds.device)))
                    if len(_ours_queue) >= a.bs:
                        _flush_ours()
                    continue
                out = model.generate(inputs_embeds=embeds,
                                     attention_mask=enc_use.get("attention_mask"),
                                     max_new_tokens=a.max_new_tokens, do_sample=False)
                text = proc.decode(out[0], skip_special_tokens=True)
            elif a.arm == "ceiling":
                out = model.generate(**enc, max_new_tokens=a.max_new_tokens, do_sample=False)
                text = proc.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            else:
                enc2 = axis.build_inputs(degrade(img, a.level, a.filt), prompt).to("cuda:0")
                enc2["pixel_values"] = enc2["pixel_values"].to(torch.bfloat16)
                axis.assert_same_grid(enc, enc2)
                if a.arm == "floor":
                    out = model.generate(**enc2, max_new_tokens=a.max_new_tokens, do_sample=False)
                    text = proc.decode(out[0, enc2["input_ids"].shape[1]:],
                                       skip_special_tokens=True)
                else:
                    qsel = a.arm.endswith("_qsel")
                    # Per-image vision-work reuse: non-qsel feats depend only on
                    # pixel_values/image_sizes, so for specs that ask several questions about
                    # one image (visdrone_*: rows are image-major sorted) the fork correction --
                    # the expensive serial half of an ours sample -- runs once per IMAGE and its
                    # features are reused across that image's questions. LRU-1 (adjacent rows
                    # only); qsel is query-aware and excluded by construction.
                    _row = ds[idx]
                    _img_key = _row.get("path") if isinstance(_row, dict) else None
                    if (not qsel and _img_key is not None
                            and _feats_cache.get("key") == (_img_key, a.arm)):
                        feats = _feats_cache["feats"]
                    else:
                        if a.arm.startswith("corrected"):
                            feats = corrected_feats(axis, fork, enc, enc2, a.keep, qsel=qsel)
                        else:
                            feats = streaming_feats_and_prefill(axis, fork, enc, enc2,
                                                                a.keep, a.groups, model,
                                                                qsel=qsel)
                        if not qsel and _img_key is not None:
                            _feats_cache["key"] = (_img_key, a.arm)
                            _feats_cache["feats"] = feats
                    embeds = axis.scatter_and_prefill_embeds(enc, feats.to(torch.bfloat16))
                    if a.bs > 1:
                        # Batched-generate path for the fork arms: the CORRECTION stays bs=1 by
                        # construction (the vision fork carries no batch dim), but the generate()
                        # that follows is plain stock decoding over inputs_embeds and batches
                        # with left padding exactly like the bound arms. Embeds are queued here
                        # and flushed in the batch loop below; padding rows are zero-embeds
                        # masked out by attention_mask (never attended, never decoded).
                        _ours_queue.append((idx, img, embeds[0],
                                            enc["attention_mask"][0] if enc.get("attention_mask") is not None
                                            else torch.ones(embeds.shape[1], dtype=torch.long,
                                                            device=embeds.device)))
                        if len(_ours_queue) >= a.bs:
                            _flush_ours()
                        continue
                    out = model.generate(inputs_embeds=embeds,
                                         attention_mask=enc.get("attention_mask"),
                                         max_new_tokens=a.max_new_tokens, do_sample=False)
                    text = proc.decode(out[0], skip_special_tokens=True)
        pred_for_score = text.strip()
        if a.dataset in ("refcoco", "visdrone_det"):
            # Mistral emits 0-1 FRACTION coordinates (three-conventions warning, handover
            # 2026-08-28: match convention before judging capability). gold is native-pixel
            # (identity resize above), so rescale fraction-looking boxes by the native size.
            m_nums = re.findall(r"-?\d+\.?\d*", pred_for_score)
            if len(m_nums) >= 4 and all(abs(float(v)) <= 1.5 for v in m_nums[:4]):
                W, H = img.size
                vals = (float(m_nums[0]) * W, float(m_nums[1]) * H,
                        float(m_nums[2]) * W, float(m_nums[3]) * H)
                pred_for_score = " ".join(f"{v:.1f}" for v in vals)
        ok, sc = spec.score(pred_for_score, gold)
        correct_n += ok
        total += 1
        row = {"idx": idx, "pred": text.strip()[:160], "gold": str(gold)[:80], "score": sc}
        per.append(row)
        if inc_f is not None:
            inc_f.write(json.dumps(row) + "\n"); inc_f.flush()
        if total % 50 == 0 or total == len(idxs):
            dt = time.time() - t0
            print(f"  [{total}/{len(idxs)}] {dt:.0f}s {dt / total:.2f}s/ex "
                  f"acc={correct_n / total:.2%}", flush=True)

    if a.bs > 1 and a.arm not in ("ceiling", "floor"):
        _flush_ours()
    summary = {"model": a.model, "dataset": a.dataset, "arm": a.arm,
               "keep": a.keep if a.arm in ("corrected", "streaming") else None,
               "level": a.level, "groups": a.groups if a.arm == "streaming" else None,
               "num_samples": total, "accuracy": correct_n / total, "correct": correct_n,
               "mean_score": sum(r["score"] for r in per) / max(total, 1)}
    if inc_f is not None:
        inc_f.close()
    print(f"\n=== Final Summary: {json.dumps(summary)}", flush=True)
    if a.out_json:
        os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
        json.dump({"summary": summary, "per_sample": per}, open(a.out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
