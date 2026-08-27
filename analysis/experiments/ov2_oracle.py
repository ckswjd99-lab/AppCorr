"""LLaVA-OneVision-2 approx-then-correct arms on a VQA benchmark.

    floor        approximate pass only (L2 image), no correction
    corrected    one-shot correction. Tokens lead: the pooled score picks image tokens, and the
                 patch mask is DERIVED from them, so `patch_mask_any_to_token(pm) == sel_tok`
    corrected_split    independent budgets -- vision takes a patch top-k, the LLM its own token
                 top-k. The two masks do not nest, which is why interleaved cannot use this
    corrected_patchled patches lead -- the patch top-k, then every token any of it touches
    interleaved  the SAME selection as `corrected`, split across `--groups` arrival rounds
    parity       both paths in this driver off the same masks, reported as a tensor diff
    ceiling      exact forward

floor and ceiling keep the stock `generate` path they were measured with, so the numbers already in
analysis/results/ov2_* stay the reference and nothing about them can be confounded by correction
wiring. Every other arm walks the unified axis and decodes from the axis's own K/V.

Report floor, ours and ceiling together, and lead with the preservation rate ours/ceiling -- a
recovery percentage computed against a narrow floor-ceiling gap says almost nothing.

    python analysis/experiments/ov2_oracle.py --arm ceiling --dataset chartqa --num-samples 50
    python analysis/experiments/ov2_oracle.py --arm corrected --keep 0.55 --full
"""
import argparse, json, os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from qwen_vl_prefill.datasets_eval import get_spec
from experiments.ov2_degradation import hw_from_grid, l2_from_native
from appcorr.models.ov2.unified import OV2UnifiedAxis

MODEL = "lmms-lab-encoder/LLaVA-OneVision-2-8B-Instruct"
AXIS_ARMS = ("corrected", "corrected_split", "corrected_patchled", "interleaved", "progressive",
             "parity", "streaming")


def encode(proc, img, prompt, device):
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    enc = proc(text=[text], images=[img], return_tensors="pt", padding=True)
    assert "pixel_values" in enc, "the image did not reach the processor"
    return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in enc.items()}


def patch_energy(px_full: torch.Tensor, px_appr: torch.Tensor) -> torch.Tensor:
    """Per-patch residual energy. [1, n_patch].

    `pixel_values` here is already ONE ROW PER PATCH -- [n_patch, 3*14*14] -- because the Qwen2VL
    style processor flattens each patch. Gemma 3 needed a spatial reshape out of a [B, C, H, W]
    canvas; there is no canvas to reshape here and introducing one would only reintroduce the
    2x2-block-order question the processor has already answered.
    """
    return ((px_full.float() - px_appr.float()) ** 2).mean(dim=-1).unsqueeze(0)


@torch.no_grad()
def run_stock(model, proc, img, prompt, arm, level, device, max_new_tokens):
    """floor / ceiling: stock generate, the path both were originally measured with."""
    enc = encode(proc, img, prompt, device)
    if arm == "floor":
        # Reuse this encoding's grid instead of letting `l2_from_native` rediscover it: that call
        # costs a whole extra preprocessing pass (~265 ms) against ~35 ms of GPU inference.
        deg = l2_from_native(img, level, proc, hw_from_grid(enc["image_grid_thw"], proc))
        enc2 = encode(proc, deg, prompt, device)
        assert enc2["pixel_values"].shape == enc["pixel_values"].shape, (
            f"degradation changed the sampled grid: {enc['image_grid_thw'].tolist()} vs "
            f"{enc2['image_grid_thw'].tolist()} -- floor and ceiling would not be comparable")
        enc = enc2
    out = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
    return proc.tokenizer.decode(out[0, enc["input_ids"].shape[1]:],
                                 skip_special_tokens=True).strip(), {}


@torch.no_grad()
def run_axis(axis, model, proc, img, prompt, arm, keep, level, device, dtype,
             max_new_tokens, pscore="energy_attn", groups=4):
    enc = encode(proc, img, prompt, device)
    deg = l2_from_native(img, level, proc, hw_from_grid(enc["image_grid_thw"], proc))
    enc2 = encode(proc, deg, prompt, device)
    assert enc2["pixel_values"].shape == enc["pixel_values"].shape, (
        f"degradation changed the sampled grid: {enc['image_grid_thw'].tolist()} vs "
        f"{enc2['image_grid_thw'].tolist()}")

    ids, pp = enc["input_ids"], enc["patch_positions"]
    px, px2 = enc["pixel_values"].to(dtype), enc2["pixel_values"].to(dtype)
    freqs = axis.rope_freqs(pp)
    n_patch = px.shape[0]
    n_img = axis.n_tokens(n_patch)

    if arm == "streaming":
        # No score, no budget, no LLM approximation. Every token is prefilled exactly once, in
        # causal order, when its band arrives -- so this arm has no `keep` and the whole image is
        # eventually recomputed. What it gives up is vision freshness, not vision coverage.
        hidden, kv, sst = axis.streaming_forward(px2, px, pp, ids, groups)
        stats = {"n_patch": int(n_patch), "image_tokens": int(n_img), "groups": groups,
                 "prefill_tokens": sst["prefill_tokens"],
                 "vision_layer_passes": sst["vision_layer_passes"]}
        return generate_from_kv(model, proc, hidden, kv, ids, max_new_tokens), stats

    if arm == "progressive":
        # Canonical per-round selection (2026-08-26): the walk scores round r with
        # energy x running attn layermean itself -- no driver-side scoring pass, no upfront
        # selection. The driver supplies only the raw energy term.
        energy = patch_energy(px, px2)
        hidden, cache, pst = axis.interleaved_forward_progressive(px, px2, pp, ids, energy,
                                                                  keep, groups)
        stats = {"n_patch": int(n_patch), "image_tokens": int(n_img), "groups": groups,
                 "keep": keep, **{k: v for k, v in pst.items() if isinstance(v, (int, float))}}
        return generate_from_axis(model, proc, axis, hidden, cache, ids, max_new_tokens), stats

    # --- approximate pass over the WHOLE axis, on the approximate image ------------------------ #
    # The cache every correction reads from has to be built here, on the approximate input.
    vh_appr, cache = axis.vision_approx(
        axis.vision_prepare(px2), freqs, {},
        collect_attn=(pscore == "energy_attn"))
    feats_appr = axis.project(vh_appr, pp)
    emb_appr, ctx = axis.llm_prepare(ids, feats_appr)
    _, cache = axis.llm_approx(emb_appr, ctx, cache)
    assert ctx["image_positions"].numel() == n_img, (
        f"{ctx['image_positions'].numel()} image tokens for {n_patch} patches")
    stats = {"n_patch": int(n_patch), "image_tokens": int(n_img)}

    # --- patch score ---------------------------------------------------------------------------- #
    score = patch_energy(px, px2)
    if pscore == "energy_attn":
        attn = cache.get("vision_patch_attn_layermean")
        if attn is None:
            raise RuntimeError("energy_attn requested but no attention was collected")
        score = (score / score.mean().clamp_min(1e-12)) * \
                (attn / attn.mean().clamp_min(1e-12)).to(score.device)

    if arm == "corrected_patchled":
        # Patches lead: the patch top-k, then every token any of it touches.
        pk = max(1, int(round(keep * n_patch)))
        pm = torch.zeros_like(score, dtype=torch.bool).scatter_(
            1, score.topk(pk, dim=-1).indices, True)
        sel_tok = axis.patch_mask_any_to_token(pm)
    elif arm == "corrected_split":
        # Independent budgets. The masks do not nest, so interleaved cannot be built on this.
        pk = max(1, int(round(keep * n_patch)))
        pm = torch.zeros_like(score, dtype=torch.bool).scatter_(
            1, score.topk(pk, dim=-1).indices, True)
        pooled = axis.pool_patch_score(score)
        tk = max(1, int(round(keep * n_img)))
        sel_tok = torch.zeros_like(pooled, dtype=torch.bool).scatter_(
            1, pooled.topk(tk, dim=-1).indices, True)
    else:
        # DEFAULT (`corrected`, and everything built on it: interleaved, parity).
        # Tokens lead and the patch mask is DERIVED, so `patch_mask_any_to_token(pm) == sel_tok`
        # holds exactly -- the identity interleaved needs to split patches into groups that map
        # onto token groups. On Gemma 3 this arm briefly fell through to the independent-budget
        # branch and disagreed with one-shot on 16/40 samples at the same token count. The default
        # lives in `else` on purpose: an arm added later inherits the selection that composes.
        pooled = axis.pool_patch_score(score)
        tk = max(1, int(round(keep * n_img)))
        sel_tok = torch.zeros_like(pooled, dtype=torch.bool).scatter_(
            1, pooled.topk(tk, dim=-1).indices, True)
        pm = axis.token_mask_to_patch_mask(sel_tok)

    tm = torch.zeros(ids.shape[0], ids.shape[1], dtype=torch.bool, device=device)
    tm[:, ctx["image_positions"]] = sel_tok
    # Text joins the FINAL correction. It was never approximated, but it ATTENDS to the image
    # tokens, so its values change when those are corrected -- and the answer is read off the last
    # position, which is text. Leaving it out gives correction no path to the output at all.
    is_text = torch.ones(ids.shape[0], ids.shape[1], dtype=torch.bool, device=device)
    is_text[:, ctx["image_positions"]] = False
    tm = tm | is_text
    stats["patches_corrected"] = int(pm.sum())
    stats["image_tokens_corrected"] = int(sel_tok.sum())
    stats["text_tokens_corrected"] = int(is_text.sum())
    stats["llm_tokens_corrected"] = int(tm.sum())

    if arm == "parity":
        # Both paths in the DRIVER off the SAME pm/sel_tok/tm. A standalone replica of these two
        # paths once reported bit-identical output while the driver's arms disagreed, so the only
        # trustworthy comparison is this one. interleaved runs first: it builds its own cache from
        # {}, whereas the corrected path below mutates `cache` in place.
        hI, cI, iso = axis.interleaved_forward(px, px2, pp, ids, pm, tm, groups)
        txt_I = generate_from_axis(model, proc, axis, hI, cI, ids, max_new_tokens)

        mixed = torch.where(pm.unsqueeze(-1), axis.vision_prepare(px), axis.vision_prepare(px2))
        vh_corr, cache = axis.vision_correct(mixed, pm, freqs, cache)
        feats_mixed = torch.where(sel_tok.unsqueeze(-1), axis.project(vh_corr, pp), feats_appr)
        emb_mixed, _ = axis.llm_prepare(ids, feats_mixed)
        hA, cache = axis.llm_correct(emb_mixed, tm, ctx, cache)
        txt_A = generate_from_axis(model, proc, axis, hA, cache, ids, max_new_tokens)

        h_rel = ((hA.float() - hI.float()).abs().max().item()
                 / max(hA.float().abs().max().item(), 1e-9))
        kv_rel, kv_where = 0.0, -1
        for li in range(axis.n_llm):
            for w in ("k", "v"):
                ta, tb = cache[f"l{li}_{w}"].float(), cI[f"l{li}_{w}"].float()
                r = (ta - tb).abs().max().item() / max(ta.abs().max().item(), 1e-9)
                if r > kv_rel:
                    kv_rel, kv_where = r, li
        print(f"    parity(g={groups}): hidden={h_rel:.2e} kv={kv_rel:.2e}@l{kv_where} "
              f"lc={iso['layer_corrections']} vs oneshot={axis.n_stages} "
              f"text {'SAME' if txt_A == txt_I else 'DIFFER'}"
              + ("" if txt_A == txt_I else f"  A={txt_A[:40]!r} I={txt_I[:40]!r}"), flush=True)
        stats["parity_hidden_rel"] = h_rel
        stats["parity_kv_rel"] = kv_rel
        stats["parity_text_same"] = int(txt_A == txt_I)
        return txt_A, stats

    if arm == "interleaved":
        # Same selection as `corrected`, split into `groups` arrival rounds. The walk owns the whole
        # axis and rebuilds its own approximate pass; the cache above is used only for the score.
        hidden, icache, iso = axis.interleaved_forward(px, px2, pp, ids, pm, tm, groups)
        stats["groups"] = groups
        stats["layer_corrections"] = iso["layer_corrections"]
        return generate_from_axis(model, proc, axis, hidden, icache, ids, max_new_tokens), stats

    # --- one-shot: selected patches recomputed from the full-resolution stream ------------------ #
    mixed = torch.where(pm.unsqueeze(-1), axis.vision_prepare(px), axis.vision_prepare(px2))
    vh_corr, cache = axis.vision_correct(mixed, pm, freqs, cache)

    # --- the LLM input carries corrected features ONLY at the selected tokens ------------------- #
    # Correction changes essentially every touched token's feature, so feeding all of them to the
    # LLM would mean the LLM half was fully recomputed no matter what its budget said. Mixing per
    # token is what makes the LLM budget mean anything.
    feats_mixed = torch.where(sel_tok.unsqueeze(-1), axis.project(vh_corr, pp), feats_appr)
    emb_mixed, _ = axis.llm_prepare(ids, feats_mixed)
    # Keep the corrected cache: generation decodes on top of it.
    hidden, cache = axis.llm_correct(emb_mixed, tm, ctx, cache)
    stats["layer_corrections"] = axis.n_stages
    return generate_from_axis(model, proc, axis, hidden, cache, ids, max_new_tokens), stats


def generate_from_kv(model, proc, hidden, kv, ids, max_new_tokens):
    """Greedy decode from a DynamicCache the streaming prefill already filled.

    `hidden` comes from the stock `Qwen3Model`, which applies its final norm itself -- applying
    `llm_finish` again here would norm twice, the Gemma 3 bug that cost 20pp while every tensor
    gate read zero.
    """
    eos = model.config.text_config.eos_token_id
    eos = eos if isinstance(eos, (list, tuple)) else [eos]
    n = ids.shape[1]
    nxt = model.lm_head(hidden[:, -1:])[:, -1].argmax(-1, keepdim=True)
    produced = [nxt]
    for step in range(max_new_tokens - 1):
        if int(nxt) in eos:
            break
        pos = torch.tensor([[n + step]], device=ids.device)
        emb = model.model.language_model.get_input_embeddings()(nxt)
        out = model.model.language_model(inputs_embeds=emb, past_key_values=kv,
                                         position_ids=pos, cache_position=pos[0], use_cache=True)
        kv = out.past_key_values
        nxt = model.lm_head(out.last_hidden_state[:, -1:])[:, -1].argmax(-1, keepdim=True)
        produced.append(nxt)
    return proc.tokenizer.decode(torch.cat(produced, dim=1)[0], skip_special_tokens=True).strip()


def generate_from_axis(model, proc, axis, hidden, cache, ids, max_new_tokens):
    """Decode from the axis's own K/V, so generation actually uses the corrected prefix.

    Calling `model.generate(pixel_values=...)` here would quietly invert the experiment: it
    recomputes image features from the FULL-resolution pixels, so every arm would answer from an
    exact vision pass and the floor would score like the ceiling.
    """
    from transformers import DynamicCache
    kv = DynamicCache(config=model.config.text_config)
    for i in range(axis.n_llm):
        kv.update(cache[f"l{i}_k"], cache[f"l{i}_v"], i)

    eos = model.config.text_config.eos_token_id
    eos = eos if isinstance(eos, (list, tuple)) else [eos]
    n = ids.shape[1]
    nxt = model.lm_head(axis.llm_finish(hidden)[:, -1:])[:, -1].argmax(-1, keepdim=True)
    produced = [nxt]
    for step in range(max_new_tokens - 1):
        if int(nxt) in eos:
            break
        pos = torch.tensor([[n + step]], device=ids.device)
        emb = model.model.language_model.get_input_embeddings()(nxt)
        out = model.model.language_model(inputs_embeds=emb, past_key_values=kv,
                                         position_ids=pos, cache_position=pos[0], use_cache=True)
        kv = out.past_key_values
        nxt = model.lm_head(out.last_hidden_state[:, -1:])[:, -1].argmax(-1, keepdim=True)
        produced.append(nxt)
    return proc.tokenizer.decode(torch.cat(produced, dim=1)[0], skip_special_tokens=True).strip()


def load(device, dtype):
    from transformers import AutoModelForImageTextToText, AutoProcessor
    tok = os.environ.get("HF_TOKEN")
    proc = AutoProcessor.from_pretrained(MODEL, token=tok, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=dtype, token=tok, trust_remote_code=True).eval().to(device)
    return model, proc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="chartqa")
    ap.add_argument("--arm", choices=("ceiling", "floor") + AXIS_ARMS, default="ceiling")
    ap.add_argument("--keep", type=float, default=0.55)
    ap.add_argument("--groups", type=int, default=4,
                    help="interleaved rounds; the arm corrects the SAME tokens as `corrected`, "
                         "only split across rounds (g=1 must reproduce it)")
    ap.add_argument("--pscore", choices=["energy", "energy_attn"], default="energy_attn")
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--num-samples", type=int, default=50)
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=24)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out-json", default=None)
    a = ap.parse_args()

    from datasets import load_dataset
    dtype = torch.bfloat16 if a.dtype == "bfloat16" else torch.float32
    model, proc = load(a.device, dtype)
    axis = OV2UnifiedAxis(model.model).eval() if a.arm in AXIS_ARMS else None

    spec = get_spec(a.dataset)
    ds = spec.load(load_dataset)
    n = len(ds) if a.full else min(a.num_samples, len(ds))
    idxs = list(range(len(ds)))[:n] if a.full else list(range(0, len(ds), max(1, len(ds) // n)))[:n]
    print(f"[ov2] {a.dataset} arm={a.arm} keep={a.keep if axis else None} "
          f"pscore={a.pscore if axis else None} level=L{a.level} "
          f"groups={a.groups if a.arm in ('interleaved', 'parity') else None}  "
          f"{n} of {len(ds)}", flush=True)

    total, correct, t0 = 0.0, 0, time.time()
    per_sample, stat_acc = [], {}
    for k, i in enumerate(idxs, 1):
        img, prompt, gold = spec.prepare(ds[i], lambda h, w, **kw: (h, w), 1, 1, 1 << 30)
        if axis is None:
            text, stats = run_stock(model, proc, img, prompt, a.arm, a.level, a.device,
                                    a.max_new_tokens)
        else:
            text, stats = run_axis(axis, model, proc, img, prompt, a.arm, a.keep, a.level,
                                   a.device, dtype, a.max_new_tokens, a.pscore, a.groups)
        ok, sc = spec.score(text, gold)
        correct += int(ok); total += sc
        per_sample.append({"idx": i, "gold": gold, "pred": text, "score": sc})
        for kk, vv in stats.items():
            stat_acc.setdefault(kk, []).append(vv)
        if k % 25 == 0 or k == n:
            el = time.time() - t0
            print(f"  [{k}/{n}] {el:.0f}s {el/k:.2f}s/ex  acc={total/k*100:.2f}%", flush=True)

    # `streaming` has no budget and no score: it eventually recomputes every patch, and its only
    # knob is the number of arrival bands. Reporting a keep for it would invite a comparison at
    # "equal keep" that does not exist.
    budgeted = a.arm in AXIS_ARMS and a.arm != "streaming"
    summary = {"model": MODEL, "dataset": a.dataset, "arm": a.arm,
               "keep": a.keep if budgeted else None,
               "level": a.level if a.arm != "ceiling" else None,
               "pscore": a.pscore if budgeted else None,
               "groups": a.groups if a.arm in ("interleaved", "parity", "streaming") else None,
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
