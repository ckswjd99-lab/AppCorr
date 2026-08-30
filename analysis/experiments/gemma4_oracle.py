"""Gemma 4 31B accuracy oracle — ceiling / floor / corrected (one-shot).

Level-3 driver assembled from the gated pieces (axis level 1, vision fork level
2). `corrected` is the one-shot arm: vision approximates on the degraded image,
the top-`keep` patch rows by residual energy are recomputed against the cached
K/V with full-resolution inputs, and the LLM prefills ONCE on the resulting soft
tokens -- no LLM-side correction, so the interleaved contract is not yet in play.
The interleaved/progressive walk is port-plan step 4.

Generation for the axis arms goes through `model.generate(inputs_embeds=...,
mm_token_type_ids=...)`: with no pixel_values the placeholder scatter is
skipped, and the dual masks (full layers causal, sliding layers block-bidir)
are built from mm_token_type_ids exactly as in the stock forward.

Run (smoke): CUDA_VISIBLE_DEVICES=0 python analysis/experiments/gemma4_oracle.py \
    --dataset realworldqa --arm corrected --keep 0.5 --num-samples 12
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "qwen_vl_prefill"))
from appcorr.models.gemma4.unified import Gemma4Axis, MODEL_ID_31B  # noqa: E402
from appcorr.models.gemma4.vision_fork import Gemma4VisionFork      # noqa: E402
from gemma4_axis_gate import l2_degrade, _GEMMA4_MAX_PX             # noqa: E402
import re
from PIL import Image


def degrade_g4(img: Image.Image, filt: str = "pyr", level: int = 2) -> Image.Image:
    """pyr (default; 2026-08-28 Option-B decision) = cv2.pyrDown/pyrUp archetype at the
    model-sampled cap (port of qwen35_accuracy/mistral3_oracle's branch); "box" delegates to
    the original l2_degrade to reproduce older rows."""
    if filt == "box":
        return l2_degrade(img)
    import cv2
    import numpy as np
    w, h = img.size
    s = min(1.0, (_GEMMA4_MAX_PX / (w * h)) ** 0.5)
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


@torch.no_grad()
def _generate_from_kv(model, proc, axis, hidden, lcache, n_seq, max_new_tokens):
    """Greedy decode on top of the fork's K/V (gemma3_oracle's `_generate_from_axis` ported).

    `model.generate(pixel_values=...)` would recompute exact features and invert the experiment;
    `generate(inputs_embeds=...)` would re-prefill the LLM, throwing away the interleaved walk.
    The fork caches ARE the corrected prefix state, so decoding must ride them. DynamicCache
    stores full-length K/V even for sliding layers (checked: lazy layers ignore the window) --
    a memory cost only, since decode masks are built from config.layer_types, not the cache."""
    from transformers import DynamicCache
    kv = DynamicCache(config=model.config.text_config)
    for i in range(len(axis.llm_fork_layers)):
        kv.update(lcache[f"l{i}_k"], lcache[f"l{i}_v"], i)

    gcfg = getattr(model, "generation_config", None)
    eos = getattr(gcfg, "eos_token_id", None) if gcfg is not None else None
    if eos is None:
        eos = model.config.text_config.eos_token_id
    eos = list(eos) if isinstance(eos, (list, tuple)) else [eos]

    nxt = axis.logits(axis.llm_finish(hidden)[:, -1:])[:, -1].argmax(-1, keepdim=True)
    produced = [nxt]
    for step in range(max_new_tokens - 1):
        if int(nxt) in eos:
            break
        pos = torch.tensor([[n_seq + step]], device=nxt.device)
        emb = model.get_input_embeddings()(nxt)
        out = axis.llm(inputs_embeds=emb, past_key_values=kv, position_ids=pos, use_cache=True)
        kv = out.past_key_values
        nxt = axis.logits(out.last_hidden_state[:, -1:])[:, -1].argmax(-1, keepdim=True)
        produced.append(nxt)
        if int(nxt) in eos:
            break
    return proc.decode(torch.cat(produced, dim=1)[0], skip_special_tokens=True)


@torch.no_grad()
def _interleaved(axis, fork, model, proc, enc, enc2, keep, groups, max_new_tokens):
    """Port-plan step 4: full approx at arrival 0, per-round vision-band + LLM corrections.

    Schedule (the gemma3 contract, docs/memo/interleaved_correction_contract.md):
      arrival 0: vision approx (degraded) full depth, LLM approx full depth on the approx feats
                 -- overlaps the detail transmission entirely.
      round r:   vision-correct THIS round's band of the selected patches; LLM-correct exactly
                 the soft tokens those patches pool into. Text joins the FINAL round only.
    Bands are horizontal strips of POOLED rows (edges snapped to the pooling kernel so no soft
    token straddles two rounds -- a straddled token would re-enter a later round with a changed
    x while its delta was computed against the old one, breaking rule-3 reconstruction).
    The LLM entry is rebuilt from the fork's `v_last_hidden` each round: untouched patch rows are
    still the approx values there, so the entry is automatically MIXED at the token level
    (bitwise -- pooling untouched rows reproduces the approx feature exactly)."""
    px_f, px_a = enc["pixel_values"], enc2["pixel_values"]
    pos = enc.get("image_position_ids")
    px_fb = px_f if px_f.dim() == 3 else px_f.unsqueeze(0)
    px_ab = px_a if px_a.dim() == 3 else px_a.unsqueeze(0)
    pos_b = pos if pos.dim() == 3 else pos.unsqueeze(0)
    n_pool = px_fb.shape[1]

    emb_f, _ = fork.prepare(px_fb, pos_b)
    emb_a, pad = fork.prepare(px_ab, pos_b)
    vcache = {}
    fork.approx(emb_a, pos_b, pad, vcache)

    # Selection: same pscore as `corrected` (residual energy, top-keep).
    energy = (px_fb.float() - px_ab.float()).pow(2).sum(-1)
    energy = energy.masked_fill(pad, float("-inf"))
    kq = max(1, int(round(keep * n_pool)))
    sel = torch.zeros_like(energy, dtype=torch.bool).scatter_(
        1, energy.topk(kq, dim=-1).indices, True)

    # patch -> soft-token map: the pooler's own kernel_idxs formula (_avg_pool_by_positions).
    pk = fork.cfg.pooling_kernel_size
    cp = pos_b.clamp(min=0)
    max_x = cp[..., 0].max(dim=-1, keepdim=True)[0] + 1
    kidx = (cp[..., 0] // pk) + (max_x // pk) * (cp[..., 1] // pk)          # [1, P]

    pooled_row = cp[..., 1] // pk
    n_rows = int(pooled_row.max().item()) + 1
    edges = torch.linspace(0, n_rows, groups + 1).round().long()
    bands = [sel & (pooled_row >= edges[r]) & (pooled_row < edges[r + 1]) & ~pad
             for r in range(groups)]

    # LLM approx on the (all-approximate) features -- arrival 0.
    ids = enc["input_ids"]
    image_mask = ids == axis.cg.config.image_token_id
    llm_ids = torch.where(image_mask, axis.cg.config.text_config.pad_token_id, ids)
    base_embeds = axis.model.get_input_embeddings()(llm_ids)

    def entry_embeds():
        soft = fork.finish(vcache["v_last_hidden"], pos_b, pad, n_pool, work_dtype=emb_a.dtype)
        feats = axis.embed_vision(inputs_embeds=soft).to(base_embeds.device, base_embeds.dtype)
        return base_embeds.masked_scatter(
            image_mask.unsqueeze(-1).expand_as(base_embeds), feats)

    embeds = entry_embeds()
    ctx = axis.build_llm_ctx(enc, embeds)
    lcache = {}
    hidden, lcache = axis.llm_approx(embeds, ctx, lcache)

    img_pos = image_mask[0].nonzero(as_tuple=True)[0]      # soft-token j sits at img_pos[j]
    arrived = torch.zeros_like(sel)
    for r in range(groups):
        last = (r == groups - 1)
        band = bands[r]
        tok_rows = torch.empty(0, dtype=torch.long, device=ids.device)
        if bool(band.any()):
            arrived = arrived | band
            # Contract rule 2: the stream carries full-res rows for everything ARRIVED;
            # this round recomputes only its OWN band (never the accumulated set).
            stream = torch.where(arrived.unsqueeze(-1), emb_f, emb_a)
            fork.correct(stream, band, pos_b, pad, vcache)
            tok_rows = img_pos[torch.unique(kidx[0][band[0]])]
        if last:
            text_rows = (~image_mask[0]).nonzero(as_tuple=True)[0]
            tok_rows = torch.cat([tok_rows, text_rows]).sort().values
        if tok_rows.numel():
            hidden, lcache = axis.llm_correct(entry_embeds(), tok_rows, ctx, lcache)

    return _generate_from_kv(model, proc, axis, hidden, lcache, ids.shape[1], max_new_tokens)


@torch.no_grad()
def run_one(axis, fork, model, proc, img, prompt, arm, keep, max_new_tokens, filt="pyr",
            groups=4):
    enc = axis.build_inputs(img, prompt).to("cuda:0")
    enc["pixel_values"] = enc["pixel_values"].to(torch.bfloat16)
    if arm == "ceiling":
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        return proc.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    enc2 = axis.build_inputs(degrade_g4(img, filt), prompt).to("cuda:0")
    enc2["pixel_values"] = enc2["pixel_values"].to(torch.bfloat16)
    axis.assert_same_grid(enc, enc2)
    if arm == "floor":
        out = model.generate(**enc2, max_new_tokens=max_new_tokens, do_sample=False)
        return proc.decode(out[0, enc2["input_ids"].shape[1]:], skip_special_tokens=True)

    if arm == "interleaved":
        return _interleaved(axis, fork, model, proc, enc, enc2, keep, groups, max_new_tokens)

    # corrected: vision approx on degraded, one-shot partial correct, LLM once
    px_f, px_a = enc["pixel_values"], enc2["pixel_values"]
    pos = enc.get("image_position_ids")
    px_fb = px_f if px_f.dim() == 3 else px_f.unsqueeze(0)
    px_ab = px_a if px_a.dim() == 3 else px_a.unsqueeze(0)
    pos_b = pos if pos.dim() == 3 else pos.unsqueeze(0)
    n_pool = px_fb.shape[1]

    emb_f, _pad_f = fork.prepare(px_fb, pos_b)
    emb_a, pad = fork.prepare(px_ab, pos_b)
    cache = {}
    fork.approx(emb_a, pos_b, pad, cache)
    energy = (px_fb.float() - px_ab.float()).pow(2).sum(-1)
    energy = energy.masked_fill(pad, float("-inf"))
    kq = max(1, int(round(keep * n_pool)))
    sel = torch.zeros_like(energy, dtype=torch.bool).scatter_(
        1, energy.topk(kq, dim=-1).indices, True)
    mixed = torch.where(sel.unsqueeze(-1), emb_f, emb_a)
    last = fork.correct(mixed, sel, pos_b, pad, cache)
    soft = fork.finish(last, pos_b, pad, n_pool, work_dtype=emb_a.dtype)
    feats = axis.embed_vision(inputs_embeds=soft)

    ids = enc["input_ids"]
    image_mask = ids == axis.cg.config.image_token_id
    llm_ids = torch.where(image_mask, axis.cg.config.text_config.pad_token_id, ids)
    embeds = axis.model.get_input_embeddings()(llm_ids)
    feats = feats.to(embeds.device, embeds.dtype)
    assert int(image_mask.sum()) * embeds.shape[-1] == feats.numel()
    embeds = embeds.masked_scatter(image_mask.unsqueeze(-1).expand_as(embeds), feats)

    out = model.generate(
        inputs_embeds=embeds, attention_mask=enc.get("attention_mask"),
        mm_token_type_ids=enc["mm_token_type_ids"],
        max_new_tokens=max_new_tokens, do_sample=False)
    # With inputs_embeds, generate returns only the NEW token ids.
    return proc.decode(out[0], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID_31B)
    ap.add_argument("--dataset", default="realworldqa")
    ap.add_argument("--arm", choices=["ceiling", "floor", "corrected", "interleaved"],
                    default="ceiling")
    ap.add_argument("--keep", type=float, default=0.5)
    ap.add_argument("--groups", type=int, default=4,
                    help="interleaved rounds; corrects the SAME selection as `corrected`, split "
                         "into horizontal pooled-row bands (identity: keep=1.0 g=1 == ceiling "
                         "modulo the documented SDPA kernel band)")
    ap.add_argument("--num-samples", type=int, default=12)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--filt", choices=["pyr", "box"], default="pyr")
    ap.add_argument("--bs", type=int, default=1,
                    help="bound-arm batching -- VALIDATE PER OUTPUT-FORMAT CLASS before first "
                         "full use (TextVQA EOS lesson, handover gotcha)")
    a = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoProcessor, Gemma4ForConditionalGeneration
    from gemma3_oracle import get_spec

    print(f"[gemma4] loading {a.model} arm={a.arm} keep={a.keep}", flush=True)
    model = Gemma4ForConditionalGeneration.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="cuda:0").eval()
    proc = AutoProcessor.from_pretrained(a.model)
    axis = Gemma4Axis(model, proc)
    fork = Gemma4VisionFork(axis.vision)

    spec = get_spec(a.dataset)
    ds = spec.load(load_dataset)
    if len(ds) == 0:
        raise RuntimeError("VACUOUS: 0 samples")
    n = len(ds) if a.full else min(a.num_samples, len(ds))
    idxs = (list(range(len(ds))) if a.full
            else list(range(0, len(ds), max(1, len(ds) // n)))[:n])

    correct, total, per = 0, 0, []
    import time
    t0 = time.time()
    inc_f = None
    if a.out_json:
        os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
        inc_f = open(a.out_json + ".persample.jsonl", "a", encoding="utf-8")
    gold_by_idx = {}

    def _score_and_log(idx, img, text):
        nonlocal correct, total
        pred_for_score = text.strip()
        if a.dataset in ("refcoco", "visdrone_det"):
            # Convention ladder (three-conventions warning). Gemma4 measured emitting 0-1000
            # NORMALIZED boxes (smoke: '207,207,908,996' on 640x480 -- y=996 cannot be a pixel;
            # rescoring that dump under /1000 took 0% -> 58%, matching the B200 probe's ~48%).
            # Ladder: all four <=1.5 -> fraction; <=1050 -> 0-1000; larger -> pixel passthrough.
            m_nums = re.findall(r"-?\d+\.?\d*", pred_for_score)
            if len(m_nums) >= 4:
                W, H = img.size
                v4 = [float(x) for x in m_nums[:4]]
                if all(abs(v) <= 1.5 for v in v4):
                    vals = (v4[0] * W, v4[1] * H, v4[2] * W, v4[3] * H)
                elif all(abs(v) <= 1050 for v in v4):
                    vals = (v4[0] * W / 1000, v4[1] * H / 1000,
                            v4[2] * W / 1000, v4[3] * H / 1000)
                else:
                    vals = None
                if vals is not None:
                    pred_for_score = " ".join(f"{v:.1f}" for v in vals)
        try:
            ok, sc = spec.score(pred_for_score, gold_by_idx[idx])
            correct += ok
        except NotImplementedError:
            ok, sc = None, None
        total += 1
        row = {"idx": idx, "pred": text, "gold": str(gold_by_idx[idx])[:120], "score": sc}
        per.append(row)
        if inc_f is not None:
            inc_f.write(json.dumps(row) + "\n"); inc_f.flush()
        if total % 25 == 0 or total == len(idxs):
            dt = time.time() - t0
            print(f"  [{total}/{len(idxs)}] {dt:.0f}s {dt / total:.2f}s/ex  "
                  f"acc={correct / total:.2%}", flush=True)

    if a.arm in ("ceiling", "floor") and a.bs > 1:
        proc.tokenizer.padding_side = "left"
        for b0 in range(0, len(idxs), a.bs):
            chunk = idxs[b0:b0 + a.bs]
            msgs_list, imgs_list = [], []
            for idx in chunk:
                img, prompt, gold = spec.prepare(ds[idx], lambda h, w, **kw: (h, w), 1, 1, 1 << 30)
                if a.arm == "floor":
                    img = degrade_g4(img, a.filt)
                gold_by_idx[idx] = gold
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
                _score_and_log(idx, imgs_list[j], proc.decode(out[j, T:], skip_special_tokens=True))
    else:
        for idx in idxs:
            img, prompt, gold = spec.prepare(ds[idx], lambda h, w, **kw: (h, w), 1, 1, 1 << 30)
            gold_by_idx[idx] = gold
            text = run_one(axis, fork, model, proc, img, prompt, a.arm, a.keep,
                           a.max_new_tokens, a.filt, a.groups)
            _score_and_log(idx, img, text)
    if inc_f is not None:
        inc_f.close()

    summary = {"model": a.model, "dataset": a.dataset, "arm": a.arm,
               "keep": a.keep if a.arm in ("corrected", "interleaved") else None,
               "groups": a.groups if a.arm == "interleaved" else None,
               "num_samples": total, "accuracy": correct / total, "correct": correct,
               "mean_score": (sum(r["score"] for r in per) / total
                              if per and per[0]["score"] is not None else None)}
    print(f"\n=== Final Summary: {json.dumps(summary)}", flush=True)
    if a.out_json:
        os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
        json.dump({"summary": summary, "per_sample": per}, open(a.out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
