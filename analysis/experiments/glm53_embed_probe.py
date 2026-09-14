"""Leg-1 follow-up: WHERE does the pushed-embeds arm diverge from the stock request on GLM-5.3?

Leg 1 (B200-8, 2026-09-13 02:2x KST, TP=2) gave arm B exact 1/8 with first-token |dlogprob|
0.04-0.18 on EVERY image, while the same gate on GLM-4.6V-FP8 at TP=1 was exact 8/8 with
dlogprob 0.0. So something the composer hands the engine is not what the engine computes for
itself. Three candidates, separated here INSIDE one TP=2 engine (no TP=1 exists for 328 GB):

  E1  text embeds: `resolve_embed_fn(model)(ids)` vs the model's own `embed_input_ids(ids)`.
  E2  image embeds: `model.visual(pv, grid_thw)` (composer) vs `model.embed_multimodal(
      pixel_values=pv, image_grid_thw=thw)` (the call `_execute_mm_encoder` makes), same pv.
  E4  repeatability: A, A' and B each run TWICE in the same process (A2/A_prime2/B2); the
      repeat pairs say whether any of the E3 differences is above the engine's own noise.
  E3  plumbing: arm A' = vLLM's OWN prompt-embeds request (`{"prompt_embeds": emb}` through
      `LLM.generate`) with the composer's embeds, vs arm A (stock image request) and arm B (our
      stream `open(final=True)`).  A' == B != A  -> the embeds differ (E1/E2 or pixel_values
      provenance); A' == A != B -> our stream/scheduler path; all three differ -> both.

Reads: rel-L2 / max-abs / bitwise per image for E1, E2; token ids + logprobs for A, A', B.
No verdict word is printed: the judgement is the reader's (docs/memo/glm53flash_port_plan.md).

Run (B200-8, GPUs 2-3, the main-nightly env; ~10 min incl. load):

  CUDA_VISIBLE_DEVICES=2,3 HF_HUB_OFFLINE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \\
  PYTHONPATH=/NHNHOME/share/cjpark/AppCorr-glm53 <main-nightly python> \\
      analysis/experiments/glm53_embed_probe.py --model zai-org/GLM-5.3-Flash \\
      --tensor-parallel-size 2 --gpu-mem 0.96 --enforce-eager --max-model-len 8192 \\
      --out /NHNHOME/share/cjpark/b200-8_logs/glm53_gate/results/glm53_embed_probe_tp2.json
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from analysis.experiments.vllm_stream_gate import (COCO, IMAGES, QUESTION, compare,  # noqa: E402
                                                   composer_for, tokens_and_lp)


def _stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    a, b = a.float(), b.float()
    return {"rel_l2": float((a - b).norm() / max(b.norm().item(), 1e-30)),
            "max_abs": float((a - b).abs().max()), "bitwise": bool(torch.equal(a, b)),
            "shape": list(a.shape)}


def _probe_embeds(model, ids, pv, thw, vllm_config):
    """Module-level (picklable for collective_rpc at TP>1). Runs on EVERY rank; the caller
    keeps rank 0's answer. Same calls, same order, as `client._embed_prompt_body`."""
    from vllm.forward_context import set_forward_context
    from appcorr.vllm_stream.embed_lookup import resolve_embed_fn
    dev = next(model.parameters()).device
    out = {}
    with torch.no_grad():
        ids_d = ids.to(dev)
        t_ours = resolve_embed_fn(model)(ids_d)
        t_stock = model.language_model.embed_input_ids(ids_d)
        out["E1_text"] = _stats(t_ours, t_stock)
        with set_forward_context(None, vllm_config):
            v_ours = model.visual(pv.to(dev, model.visual.dtype), grid_thw=thw.tolist())
            mm = model.embed_multimodal(pixel_values=pv.to(dev), image_grid_thw=thw.to(dev))
        v_stock = torch.cat([m for m in mm]) if isinstance(mm, (list, tuple)) else mm
        out["E2_vision"] = _stats(v_ours, v_stock)
        out["E2_vision_dtype"] = [str(v_ours.dtype), str(v_stock.dtype)]
        # determinism of the composer's own call: two runs of visual() on the same pv
        with set_forward_context(None, vllm_config):
            v_again = model.visual(pv.to(dev, model.visual.dtype), grid_thw=thw.tolist())
        out["E2_vision_rerun"] = _stats(v_ours, v_again)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tensor-parallel-size", type=int, default=2)
    ap.add_argument("--gpu-mem", type=float, default=0.96)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--n-images", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    from PIL import Image
    from vllm import SamplingParams
    from appcorr.vllm_stream import StreamingLLM

    llm = StreamingLLM(a.model, gpu_memory_utilization=a.gpu_mem, enforce_eager=a.enforce_eager,
                       max_model_len=a.max_model_len, limit_mm_per_prompt={"image": 1},
                       tensor_parallel_size=a.tensor_parallel_size)
    comp = composer_for(a.model)(llm)
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens, logprobs=1)
    vllm_config = llm.engine.vllm_config

    import os as _os
    res = {"_meta": {"model": a.model, "tp": a.tensor_parallel_size, "enforce_eager": a.enforce_eager,
                     "env": {k: _os.environ.get(k) for k in ("VLLM_BATCH_INVARIANT", "VLLM_ALLREDUCE_USE_FLASHINFER",
                                                              "VLLM_ALLREDUCE_USE_SYMM_MEM", "VLLM_USE_V2_MODEL_RUNNER")},
                     "max_tokens": a.max_tokens, "question": QUESTION, "images": IMAGES[:a.n_images],
                     "vllm": __import__("vllm").__version__}}
    for ii, name in enumerate(IMAGES[:a.n_images]):
        img = Image.open(os.path.join(COCO, name)).convert("RGB")
        parts = comp.parts(img, QUESTION)
        row = {"num_prompt_tokens": parts.num_tokens,
               "image_span": [parts.image_start, parts.image_start + parts.image_len]}
        fn = functools.partial(_probe_embeds, ids=parts.input_ids, pv=parts.pixel_values,
                               thw=parts.image_grid_thw, vllm_config=vllm_config)
        row.update(llm.apply_model(fn))
        emb = comp.embed(parts)

        t0 = time.perf_counter()
        o = llm.generate([{"prompt": parts.text, "multi_modal_data": {"image": img}}], sp)[0]
        row["A"] = tokens_and_lp(o) | {"wall_s": time.perf_counter() - t0,
                                       "prompt_ids_match_hf": list(o.prompt_token_ids) == parts.input_ids.tolist()}
        # A' is only meaningful on a model WITHOUT M-RoPE: vLLM treats prompt_embeds positions
        # as text positions, so on GLM-4.6V / Qwen the image span would get the wrong rotary by
        # design and the arm would measure that, not the embeds path.  The repeat arms (E4)
        # and A/B still run, which is what the cross-model control needs.
        has_aprime = not llm.engine.model_config.uses_mrope
        if has_aprime:
            t0 = time.perf_counter()
            o = llm.generate([{"prompt_embeds": emb.embeds}], sp)[0]
            row["A_prime"] = tokens_and_lp(o) | {"wall_s": time.perf_counter() - t0}
        rid = f"B-{ii}"
        t0 = time.perf_counter()
        llm.open(rid, emb.chunk(0, parts.num_tokens, final=True), sp)
        o = llm.run_until_done(rid)[rid]
        row["B"] = tokens_and_lp(o) | {"wall_s": time.perf_counter() - t0}

        # E4 (added after the first run, 02:37 KST: A' != A on 5/8 with the embeds proven
        # bitwise): is generation itself REPEATABLE in this engine?  Every arm again, same
        # process, same single-request batch.  If A2 != A the E3 table is noise and the
        # "exact n/8" criterion does not apply to this model at TP=2; if A2 == A the
        # prompt-embeds request path genuinely prefills differently from an image request.
        o = llm.generate([{"prompt": parts.text, "multi_modal_data": {"image": img}}], sp)[0]
        row["A2"] = tokens_and_lp(o)
        if has_aprime:
            o = llm.generate([{"prompt_embeds": emb.embeds}], sp)[0]
            row["A_prime2"] = tokens_and_lp(o)
        rid = f"B2-{ii}"
        llm.open(rid, emb.chunk(0, parts.num_tokens, final=True), sp)
        row["B2"] = tokens_and_lp(llm.run_until_done(rid)[rid])
        for k, ref in (("A_prime", "A"), ("B", "A"), ("B", "A_prime"),
                       ("A2", "A"), ("A_prime2", "A_prime"), ("B2", "B")):
            if k in row and ref in row:
                row[f"{k}_vs_{ref}"] = compare(row[ref], row[k])
        res[name] = row
        e1, e2 = row["E1_text"], row["E2_vision"]
        line = (f"[{ii}] {name} N={parts.num_tokens} | E1 text rel={e1['rel_l2']:.1e} bit={e1['bitwise']}"
                f" | E2 vis rel={e2['rel_l2']:.1e} max={e2['max_abs']:.1e} bit={e2['bitwise']}"
                f" rerun_bit={row['E2_vision_rerun']['bitwise']}")
        for k in ("A_prime_vs_A", "B_vs_A", "B_vs_A_prime", "A2_vs_A", "A_prime2_vs_A_prime",
                  "B2_vs_B"):
            if k not in row:
                continue
            c = row[k]
            tag = "exact" if c["exact"] else f"div@{c['first_divergence']}"
            line += f" | {k}: {tag} dlp0={c['dlogprob_first']:.2e}"
        print(line, flush=True)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
