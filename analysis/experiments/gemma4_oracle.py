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
from gemma4_axis_gate import l2_degrade                              # noqa: E402


@torch.no_grad()
def run_one(axis, fork, model, proc, img, prompt, arm, keep, max_new_tokens):
    enc = axis.build_inputs(img, prompt).to("cuda:0")
    enc["pixel_values"] = enc["pixel_values"].to(torch.bfloat16)
    if arm == "ceiling":
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        return proc.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    enc2 = axis.build_inputs(l2_degrade(img), prompt).to("cuda:0")
    enc2["pixel_values"] = enc2["pixel_values"].to(torch.bfloat16)
    axis.assert_same_grid(enc, enc2)
    if arm == "floor":
        out = model.generate(**enc2, max_new_tokens=max_new_tokens, do_sample=False)
        return proc.decode(out[0, enc2["input_ids"].shape[1]:], skip_special_tokens=True)

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
    ap.add_argument("--arm", choices=["ceiling", "floor", "corrected"], default="ceiling")
    ap.add_argument("--keep", type=float, default=0.5)
    ap.add_argument("--num-samples", type=int, default=12)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--out-json", default=None)
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
    for idx in idxs:
        img, prompt, gold = spec.prepare(ds[idx], lambda h, w, **kw: (h, w), 1, 1, 1 << 30)
        text = run_one(axis, fork, model, proc, img, prompt, a.arm, a.keep,
                       a.max_new_tokens)
        ok, sc = spec.score(text, gold)
        correct += ok
        total += 1
        per.append({"idx": idx, "pred": text, "gold": str(gold)[:120], "score": sc})
        if total % 25 == 0 or total == len(idxs):
            dt = time.time() - t0
            print(f"  [{total}/{len(idxs)}] {dt:.0f}s {dt / total:.2f}s/ex  "
                  f"acc={correct / total:.2%}", flush=True)

    summary = {"model": a.model, "dataset": a.dataset, "arm": a.arm,
               "keep": a.keep if a.arm == "corrected" else None,
               "num_samples": total, "accuracy": correct / total, "correct": correct}
    print(f"\n=== Final Summary: {json.dumps(summary)}", flush=True)
    if a.out_json:
        os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
        json.dump({"summary": summary, "per_sample": per}, open(a.out_json, "w"), indent=1)


if __name__ == "__main__":
    main()
