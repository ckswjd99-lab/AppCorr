"""Gemma 3 approx-then-correct arms on a VQA benchmark: floor / corrected / ceiling.

Four arms, because one design choice is genuinely open:

    floor        approximate pass only (L2 image), no correction
    corrected    SEPARATE budgets -- vision picks its patches, the LLM half picks its own tokens
                 from the 16:1-pooled score
    corrected_j  JOINT, vision-driven -- the patch selection is translated up across the projector
                 by "corrected if any of my 16 patches was"
    corrected_t  JOINT, token-driven -- tokens are picked first from the pooled score and then
                 exactly their 16 patches are corrected, so every selected token is FULLY refreshed
    ceiling      exact forward

`corrected_t` exists to test why `corrected_j` might lose. Under `any()` a selected token is usually
only partly refreshed -- one or two of its sixteen patches are fresh and the rest are still
approximate -- so it gets recomputed from a mixture. Token-driven selection removes that: same
budget, coherent inputs.

`corrected_j` is kept as an arm rather than deleted because the argument against it is indirect. It
saturates -- a 55% patch keep marks 86% of image tokens, measured -- so it recomputes far more of the
LLM half for the same vision budget. Whether that extra recompute *buys* accuracy is a different
question from whether it is expensive, and the two arms answer it directly at matched vision budget.

Scoring reuses `analysis/qwen_vl_prefill/datasets_eval.py`, the same specs and scorers the Qwen2.5-VL
work used, so the two model families land on the same axis. Reimplementing relaxed accuracy would
diverge quietly and make that comparison meaningless.

    python analysis/experiments/gemma3_oracle.py --arm ceiling --dataset chartqa --num-samples 50
    python analysis/experiments/gemma3_oracle.py --arm corrected --keep 0.55 --full
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import torch
from PIL import Image

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "analysis" / "qwen_vl_prefill"))

from appcorr.models.gemma3.unified import Gemma3UnifiedAxis
from datasets_eval import get_spec


def l2_from_native(img: Image.Image, level: int, cap: int) -> Image.Image:
    """Degrade content at native resolution, then restore the size.

    `cap` is the pyramid-direction rule: when the native image is larger than what the model will
    sample, degrade relative to the model's resolution, not the file's, or the approximation comes
    out milder than the pipeline's own.
    """
    w, h = img.size
    short = min(min(w, h), cap)
    t = max(1, short // 2 ** level)
    if h <= w:
        th, tw = t, max(1, round(w / h * t))
    else:
        tw, th = t, max(1, round(h / w * t))
    return img.resize((tw, th), Image.BOX).resize((w, h), Image.BICUBIC)


def patch_energy(px_full: torch.Tensor, px_l2: torch.Tensor, patch: int) -> torch.Tensor:
    d = ((px_full.float() - px_l2.float()) ** 2).mean(dim=1)
    b, h, w = d.shape
    return d.reshape(b, h // patch, patch, w // patch, patch).mean(dim=(2, 4)).reshape(b, -1)


@torch.no_grad()
def run_one(axis, model, proc, img, prompt, arm, keep, level, cap, patch, dtype, device,
            max_new_tokens, pscore="energy_attn", groups=4):
    msgs = [{"role": "user", "content": [{"type": "image", "image": img},
                                         {"type": "text", "text": prompt}]}]
    enc = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                   return_dict=True, return_tensors="pt").to(device)
    px, ids = enc["pixel_values"].to(dtype), enc["input_ids"]
    tti = enc.get("token_type_ids")

    if arm == "ceiling":
        out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        return proc.decode(out[0, ids.shape[1]:], skip_special_tokens=True), None

    deg = l2_from_native(img, level, cap)
    msgs2 = [{"role": "user", "content": [{"type": "image", "image": deg},
                                          {"type": "text", "text": prompt}]}]
    px2 = proc.apply_chat_template(msgs2, add_generation_prompt=True, tokenize=True,
                                   return_dict=True, return_tensors="pt")["pixel_values"].to(device, dtype)
    assert px2.shape == px.shape, f"degradation changed the input shape: {px.shape} vs {px2.shape}"

    # --- approximate pass over the WHOLE axis, on the approximate image ----------------------- #
    # The cache every correction reads from has to be built here, on the approximate input. Rebuilding
    # it on corrected features (as an earlier version did) makes the "approximate" state depend on
    # the correction, and then partial correction has no baseline to reconstruct untouched positions
    # from.
    vh_appr, cache = axis.vision_approx(axis.vision_prepare(px2), {},
                                        collect_attn=(arm != "floor" and pscore == "energy_attn"))
    feats_appr = axis.project(vh_appr)
    emb_appr, ctx = axis.llm_prepare(ids, feats_appr, tti)
    n_img = ctx["image_positions"].numel()
    _, cache = axis.llm_approx(emb_appr, ctx, cache)
    stats = {"image_tokens": int(n_img)}

    if arm == "floor":
        hidden, _ = axis.llm_approx(emb_appr, ctx, dict(cache))
        return _generate_from_axis(model, proc, axis, hidden, cache, ids, max_new_tokens), stats

    score = patch_energy(px, px2, patch)
    if pscore == "energy_attn":
        attn = cache.get("vision_patch_attn_layermean")
        if attn is None:
            raise RuntimeError("energy_attn requested but no attention was collected")
        score = (score / score.mean().clamp_min(1e-12)) * \
                (attn / attn.mean().clamp_min(1e-12)).to(score.device)

    n_patch = score.shape[1]
    pk = max(1, int(round(keep * n_patch)))
    pm = torch.zeros_like(score, dtype=torch.bool).scatter_(
        1, score.topk(pk, dim=-1).indices, True)

    if arm == "corrected_j":
        sel_tok = axis.patch_mask_any_to_token(pm)
    elif arm in ("corrected_t", "interleaved", "parity"):
        # interleaved is the INTERLEAVED FORM OF corrected_t, so it must share corrected_t's
        # selection: the patch mask is DERIVED from the chosen tokens. Leaving it in the `else`
        # branch gave it the `corrected` arm's independent patch top-k instead -- the same 141
        # tokens but a different patch set -- and the two arms disagreed on 16/40 samples while
        # each stayed perfectly reproducible. `parity` joins them so it compares like with like.
        pooled = axis.pool_patch_score(score)
        tk = max(1, int(round(keep * n_img)))
        sel_tok = torch.zeros_like(pooled, dtype=torch.bool).scatter_(
            1, pooled.topk(tk, dim=-1).indices, True)
        pm = axis.token_mask_to_patch_mask(sel_tok, n_patch)
    else:
        pooled = axis.pool_patch_score(score)
        tk = max(1, int(round(keep * n_img)))
        sel_tok = torch.zeros_like(pooled, dtype=torch.bool).scatter_(
            1, pooled.topk(tk, dim=-1).indices, True)

    tm = torch.zeros(ids.shape[0], ids.shape[1], dtype=torch.bool, device=device)
    tm[:, ctx["image_positions"]] = sel_tok
    # Text positions join the FINAL correction. They were never approximated, but they ATTEND to the
    # image tokens, so their values change when those are corrected -- and the answer is read off
    # the last position, which is text. Leaving them out gave correction no path to the output at
    # all: every corrected arm scored exactly the floor, and the last-position hidden state differed
    # from the approximate one by 0.0098.
    #
    # Only the final round needs them: text is causal and sits after the image block, so one
    # correction against the fully-corrected K/V is enough -- correcting text every round would cost
    # g times as much for an intermediate state nothing reads.
    is_text = torch.ones(ids.shape[0], ids.shape[1], dtype=torch.bool, device=device)
    is_text[:, ctx["image_positions"]] = False
    tm = tm | is_text
    stats["text_tokens_corrected"] = int(is_text.sum())
    stats["patches_corrected"] = int(pm.sum())
    stats["llm_tokens_corrected"] = int(tm.sum())
    stats["tokens_whose_block_changed"] = int(axis.patch_mask_any_to_token(pm).sum())

    if arm == "parity":
        # Both paths, in the DRIVER, off the SAME pm/sel_tok/tm. A standalone replica of these two
        # paths reported bit-identical output while the driver's two arms disagreed on 16/40
        # samples -- so the replica was wrong, and the only trustworthy comparison is this one.
        # interleaved runs first: it builds its own cache from {}, whereas the corrected_t path
        # below mutates `cache` in place via vision_correct.
        hI, cI = axis.interleaved_forward(px, px2, ids, tti, pm, tm, groups)
        txt_I = _generate_from_axis(model, proc, axis, hI, cI, ids, max_new_tokens)

        mixed = torch.where(pm.unsqueeze(-1), axis.vision_prepare(px), axis.vision_prepare(px2))
        vh_corr, cache = axis.vision_correct(mixed, pm, cache)
        feats_mixed = torch.where(sel_tok.unsqueeze(-1), axis.project(vh_corr), feats_appr)
        emb_mixed, _ = axis.llm_prepare(ids, feats_mixed, tti)
        hA, cache = axis.llm_correct(emb_mixed, tm, ctx, cache)
        txt_A = _generate_from_axis(model, proc, axis, hA, cache, ids, max_new_tokens)

        h_rel = (hA.float() - hI.float()).abs().max().item() / max(hA.float().abs().max().item(), 1e-9)
        kv_rel, kv_where = 0.0, -1
        for li in range(axis.n_llm):
            for w in ("k", "v"):
                ta, tb = cache[f"l{li}_{w}"].float(), cI[f"l{li}_{w}"].float()
                r = (ta - tb).abs().max().item() / max(ta.abs().max().item(), 1e-9)
                if r > kv_rel:
                    kv_rel, kv_where = r, li
        print(f"    parity: hidden={h_rel:.2e} kv={kv_rel:.2e}@l{kv_where} "
              f"text {'SAME' if txt_A == txt_I else 'DIFFER'}"
              + ("" if txt_A == txt_I else f"  A={txt_A[:40]!r} I={txt_I[:40]!r}"), flush=True)
        return txt_A, stats

    if arm == "interleaved":
        # Same selection as corrected_t, split into `groups` arrival rounds. The walk owns the
        # whole axis, so it rebuilds its own approximate pass -- the cache above is only used for
        # the score's attention term.
        hidden, cache = axis.interleaved_forward(px, px2, ids, tti, pm, tm, groups)
        stats["groups"] = groups
        return _generate_from_axis(model, proc, axis, hidden, cache, ids, max_new_tokens), stats

    # --- vision correction: selected patches recomputed from the full-resolution stream --------- #
    mixed = torch.where(pm.unsqueeze(-1), axis.vision_prepare(px), axis.vision_prepare(px2))
    vh_corr, cache = axis.vision_correct(mixed, pm, cache)

    # --- the LLM input carries corrected features ONLY at the selected tokens ------------------- #
    # This is the step that was missing. Pooling is 4x4, so vision correction changes essentially
    # every image token's feature; feeding all of them to the LLM means the LLM half was fully
    # recomputed no matter what its budget said. Mixing per token is what makes the LLM budget mean
    # anything -- and it is the same rule the vision half follows: the stream carries corrected
    # values exactly where correction happened.
    feats_corr = axis.project(vh_corr)
    feats_mixed = torch.where(sel_tok.unsqueeze(-1), feats_corr, feats_appr)
    emb_mixed, _ = axis.llm_prepare(ids, feats_mixed, tti)
    # Keep the corrected cache: generation decodes on top of it. Passing a copy and discarding the
    # result left every token after the first to be produced from the APPROXIMATE K/V, so even a
    # keep=1.0 run -- which must reproduce the exact forward by definition -- could not.
    hidden, cache = axis.llm_correct(emb_mixed, tm, ctx, cache)

    text = _generate_from_axis(model, proc, axis, hidden, cache, ids, max_new_tokens)
    return text, stats


def _generate_from_axis(model, proc, axis, hidden, cache, ids, max_new_tokens):
    """Decode from the axis's own KV cache, so generation actually uses the corrected prefix.

    Calling `model.generate(pixel_values=...)` here would be wrong in a way that quietly inverts the
    experiment: it recomputes image features from the FULL-resolution pixels, so every arm would
    answer from an exact vision pass and the floor would score like the ceiling. Instead the fork's
    per-layer K/V — which is what correction actually modified — is loaded into a `DynamicCache` and
    the stock model decodes on top of it.
    """
    from transformers import DynamicCache
    kv = DynamicCache(config=model.config.text_config)
    for i in range(axis.n_llm):
        kv.update(cache[f"l{i}_k"], cache[f"l{i}_v"], i)

    n = ids.shape[1]
    logits = model.lm_head(axis.llm_finish(hidden)[:, -1:])
    nxt = logits[:, -1].argmax(-1, keepdim=True)
    produced = [nxt]
    for step in range(max_new_tokens - 1):
        pos = torch.tensor([[n + step]], device=ids.device)
        emb = model.model.language_model.get_input_embeddings()(nxt)
        out = model.model.language_model(inputs_embeds=emb, past_key_values=kv,
                                         position_ids=pos, use_cache=True)
        kv = out.past_key_values
        nxt = model.lm_head(out.last_hidden_state[:, -1:])[:, -1].argmax(-1, keepdim=True)
        produced.append(nxt)
        if int(nxt) in (model.config.text_config.eos_token_id
                        if isinstance(model.config.text_config.eos_token_id, (list, tuple))
                        else [model.config.text_config.eos_token_id]):
            break
    return proc.decode(torch.cat(produced, dim=1)[0], skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/gemma-3-4b-it")
    ap.add_argument("--dataset", default="chartqa")
    ap.add_argument("--arm",
                    choices=["ceiling", "floor", "corrected", "corrected_j", "corrected_t",
                             "interleaved", "parity"],
                    default="ceiling")
    ap.add_argument("--groups", type=int, default=4,
                    help="interleaved rounds; the arm corrects the SAME tokens as corrected_t, "
                         "only split across rounds (verified: g=1 reproduces it bit-exactly)")
    ap.add_argument("--keep", type=float, default=0.55)
    ap.add_argument("--pscore", choices=["energy", "energy_attn"], default="energy_attn",
                    help="standard is residual energy x average attention")
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--num-samples", type=int, default=50)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    dtype = torch.bfloat16 if a.dtype == "bfloat16" else torch.float32
    tok = os.environ.get("HF_TOKEN")
    print(f"[gemma3] loading {a.model} ({a.dtype})", flush=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(a.model, dtype=dtype, token=tok).eval()
    model = model.to(a.device)
    proc = AutoProcessor.from_pretrained(a.model, token=tok)
    axis = Gemma3UnifiedAxis(model.model).eval()

    spec = get_spec(a.dataset)
    ds = spec.load(load_dataset)
    n = len(ds) if a.full else min(a.num_samples, len(ds))
    idxs = list(range(len(ds)))[:n] if a.full else list(range(0, len(ds), max(1, len(ds) // n)))[:n]

    size = proc.image_processor.size
    cap = int(size["height"] if isinstance(size, dict) else size.height)
    patch = int(model.config.vision_config.patch_size)
    print(f"[gemma3] {a.dataset} arm={a.arm} keep={a.keep} pscore={a.pscore} level=L{a.level} "
          f"cap={cap} patch={patch}  {n} of {len(ds)}", flush=True)

    total, correct, t0 = 0.0, 0, time.time()
    per_sample, stat_acc = [], {}
    for k, i in enumerate(idxs, 1):
        ex = ds[i]
        img, prompt, gold = spec.prepare(ex, lambda h, w, **kw: (h, w), 1, 1, 1 << 30)
        text, stats = run_one(axis, model, proc, img, prompt, a.arm, a.keep, a.level, cap,
                              patch, dtype, a.device, a.max_new_tokens, a.pscore, a.groups)
        ok, sc = spec.score(text, gold)
        correct += int(ok); total += sc
        per_sample.append({"idx": i, "gold": gold, "pred": text, "score": sc})
        if stats:
            for kk, vv in stats.items():
                stat_acc.setdefault(kk, []).append(vv)
        if k % 25 == 0 or k == n:
            el = time.time() - t0
            print(f"  [{k}/{n}] {el:.0f}s {el/k:.2f}s/ex  acc={total/k*100:.2f}%", flush=True)

    summary = {"model": a.model, "dataset": a.dataset, "arm": a.arm,
               "keep": a.keep if a.arm.startswith("corrected") else None,
               "level": a.level if a.arm != "ceiling" else None,
               "pscore": a.pscore if a.arm.startswith("corrected") else None,
               "num_samples": n, "accuracy": total / n, "correct": correct}
    for kk, vv in stat_acc.items():
        summary[f"mean_{kk}"] = sum(vv) / len(vv)
    print("\n=== Final Summary: " + json.dumps(summary))
    if a.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(a.out_json)), exist_ok=True)
        with open(a.out_json, "w") as f:
            json.dump({"summary": summary, "per_sample": per_sample}, f)


if __name__ == "__main__":
    main()
