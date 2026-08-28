"""Critical/total FLOPs for Gemma 4 31B: ceiling / floor / one-shot corrected.

Arrival semantics (the counter's one rule -- critical = work after the FINAL
arrival of the request):
    ceiling / floor   single arrival -> 100% critical
    corrected         arrival 0: vision approx on the degraded base (overlaps
                      the transmission); arrival 1: full-res prepare + partial
                      correct + pooler/embed + the ONE LLM prefill -> critical.

Decode is excluded everywhere, matching every other model's report. The LLM
pass is a single manual prefill through the dual masks (full layers causal,
sliding layers block-bidir), identical to Gemma4Axis.full_forward's path, so
hooks see exactly the modules the accuracy oracle exercises.

Run: CUDA_VISIBLE_DEVICES=0 python analysis/experiments/flops_report_gemma4.py \
    --datasets mmvp cvbench --keeps 0.25 0.50 --samples 12
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "qwen_vl_prefill"))
from appcorr.flops.counter import FlopCounter                     # noqa: E402
from appcorr.flops import hooks                                    # noqa: E402
from appcorr.models.gemma4.unified import Gemma4Axis, MODEL_ID_31B  # noqa: E402
from appcorr.models.gemma4.vision_fork import Gemma4VisionFork      # noqa: E402
from gemma4_axis_gate import l2_degrade                             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID_31B)
    ap.add_argument("--datasets", nargs="+", default=["mmvp", "cvbench"])
    ap.add_argument("--keeps", type=float, nargs="+", default=[0.25, 0.50])
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--out-json", default="analysis/results/flops/gemma4_flops.json")
    a = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoProcessor, Gemma4ForConditionalGeneration
    from datasets_eval import get_spec

    model = Gemma4ForConditionalGeneration.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map="cuda:0").eval()
    proc = AutoProcessor.from_pretrained(a.model)
    axis = Gemma4Axis(model, proc)
    fork = Gemma4VisionFork(axis.vision)
    roots = [model.model.vision_tower, model.model.embed_vision, model.model.language_model]

    @torch.no_grad()
    def llm_prefill(enc, feats):
        ids = enc["input_ids"]
        image_mask = ids == axis.cg.config.image_token_id
        llm_ids = torch.where(image_mask, axis.cg.config.text_config.pad_token_id, ids)
        embeds = axis.model.get_input_embeddings()(llm_ids)
        feats = feats.to(embeds.device, embeds.dtype)
        embeds = embeds.masked_scatter(image_mask.unsqueeze(-1).expand_as(embeds), feats)
        from transformers.models.gemma4.modeling_gemma4 import (
            create_masks_for_vision_model, get_block_sequence_ids_for_mask)
        position_ids = torch.arange(embeds.shape[1], device=embeds.device).unsqueeze(0)
        block_ids = get_block_sequence_ids_for_mask(enc["mm_token_type_ids"], embeds.device)
        masks = create_masks_for_vision_model(
            config=axis.cg.config.get_text_config(), inputs_embeds=embeds,
            attention_mask=enc.get("attention_mask"), past_key_values=None,
            position_ids=position_ids, block_sequence_ids=block_ids)
        axis.llm(attention_mask=masks, position_ids=position_ids,
                 inputs_embeds=embeds, use_cache=False, return_dict=True)

    result = {"_model": a.model, "_samples": a.samples}
    for ds_name in a.datasets:
        spec = get_spec(ds_name)
        ds = spec.load(load_dataset)
        idxs = list(range(0, len(ds), max(1, len(ds) // a.samples)))[:a.samples]
        arms = ["ceiling", "floor"] + [f"corrected_k{k:.2f}" for k in a.keeps]
        counters = {arm: FlopCounter() for arm in arms}
        for si, idx in enumerate(idxs):
            img, prompt, _ = spec.prepare(ds[idx], lambda h, w, **kw: (h, w), 1, 1, 1 << 30)
            enc = axis.build_inputs(img, prompt).to("cuda:0")
            enc["pixel_values"] = enc["pixel_values"].to(torch.bfloat16)
            enc2 = axis.build_inputs(l2_degrade(img), prompt).to("cuda:0")
            enc2["pixel_values"] = enc2["pixel_values"].to(torch.bfloat16)
            axis.assert_same_grid(enc, enc2)
            px_f, px_a = enc["pixel_values"], enc2["pixel_values"]
            pos = enc.get("image_position_ids")
            px_fb = px_f if px_f.dim() == 3 else px_f.unsqueeze(0)
            px_ab = px_a if px_a.dim() == 3 else px_a.unsqueeze(0)
            pos_b = pos if pos.dim() == 3 else pos.unsqueeze(0)
            n_pool = px_fb.shape[1]

            with torch.no_grad():
                for arm in arms:
                    c = counters[arm]
                    handles = hooks.install(c, roots)
                    with c.request(f"{ds_name}/{si}"):
                        if arm in ("ceiling", "floor"):
                            with c.arrival(0):
                                feats = axis.vision_features(
                                    px_f if arm == "ceiling" else px_a, pos)
                                llm_prefill(enc if arm == "ceiling" else enc2, feats)
                        else:
                            k = float(arm.split("_k")[1])
                            with c.arrival(0):
                                emb_a, pad = fork.prepare(px_ab, pos_b)
                                cache = {}
                                fork.approx(emb_a, pos_b, pad, cache)
                            with c.arrival(1):
                                emb_f, _ = fork.prepare(px_fb, pos_b)
                                energy = (px_fb.float() - px_ab.float()).pow(2).sum(-1)
                                energy = energy.masked_fill(pad, float("-inf"))
                                kq = max(1, int(round(k * n_pool)))
                                sel = torch.zeros_like(energy, dtype=torch.bool).scatter_(
                                    1, energy.topk(kq, dim=-1).indices, True)
                                mixed = torch.where(sel.unsqueeze(-1), emb_f, emb_a)
                                last = fork.correct(mixed, sel, pos_b, pad, cache)
                                soft = fork.finish(last, pos_b, pad, n_pool,
                                                   work_dtype=emb_a.dtype)
                                feats = axis.embed_vision(inputs_embeds=soft)
                                llm_prefill(enc, feats)
                    hooks.remove(handles)
        agg = {arm: counters[arm].aggregate() for arm in arms}
        full = agg["ceiling"]["mean_total_gflops"]
        row = {"full": round(full, 1), "floor": round(agg["floor"]["mean_total_gflops"], 1)}
        for k in a.keeps:
            g = agg[f"corrected_k{k:.2f}"]
            row[f"k{k:.2f}"] = round(g["mean_critical_gflops"], 1)
            row[f"total_k{k:.2f}"] = round(g["mean_total_gflops"], 1)
            print(f"{ds_name:<10} k={k:.2f} full {full:9.1f}  corrected crit "
                  f"{g['mean_critical_gflops']:8.1f} total {g['mean_total_gflops']:9.1f} "
                  f"crit/full = {g['mean_critical_gflops'] / full * 100:5.1f}%", flush=True)
        result[ds_name] = row

    os.makedirs(os.path.dirname(a.out_json), exist_ok=True)
    json.dump(result, open(a.out_json, "w"), indent=1)
    print(f"wrote {a.out_json}")


if __name__ == "__main__":
    main()
