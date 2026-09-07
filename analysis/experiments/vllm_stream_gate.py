"""Gate for the vLLM streaming-prefill integration (appcorr.vllm_stream) on Qwen2.5-VL / Qwen3.5.

Arms (same images, same question, greedy decoding, logprob of every generated token):

  A  stock         vLLM's own image request (prompt text + PIL image)                 -- anchor
  B  oneshot       the same prompt as client-composed prompt_embeds (vLLM's vision tower +
                   embedding table, client-computed M-RoPE), one message               -- the
                   embeds path itself, before any streaming
  C  stream        B's embeds in `--chunks` chunks, one appended per engine step
                   (hold-back-one: no sampling until the final chunk)                 -- ours
  C2 stream-burst  all chunks appended before the first step (scheduler merge path)
  D  stock-chunked stock A under `--max-num-batched-tokens N` (a separate engine; run as
                   `--arms D`), the numerical-noise band of vLLM's own chunked prefill

Per arm: generated token ids, per-token logprobs, and for streaming arms the scheduler state
after every step (num_computed == num_prompt-1 while open). Compared against A: exact-match
rate of the generated sequence, first-divergence index, max |dlogprob| on the common prefix.
Results merge into `--out` (json) so D from a second invocation lands next to A/B/C.

Run (GPU0, appcorr-vllm env = vllm 0.28.0; openrlhf_base = 0.11.2 also works, offline HF cache):
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  PYTHONPATH=/NHNHOME/share/cjpark/AppCorr-vllm \
  /NHNHOME/storage/users/cjpark/shk/conda_envs/appcorr-vllm/bin/python \
      analysis/experiments/vllm_stream_gate.py --arms A,B,C,C2
  ... --arms D --max-num-batched-tokens 96
  Qwen3.5 (hybrid GDN/attention MoE; the composer is the same, no deepstack in these checkpoints):
  ... --model Qwen/Qwen3.5-35B-A3B --gpu-mem 0.6 --out analysis/results/vllm_stream/gate_qwen35_35b_vllm0280.json
  ... --model Qwen/Qwen3.5-122B-A10B-FP8 --gpu-mem 0.85 --out analysis/results/vllm_stream/gate_qwen35_122b_fp8_vllm0280.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

COCO = "/NHNHOME/share/cjpark/data/coco_train2017/train2017"
IMAGES = ["000000000009.jpg", "000000000025.jpg", "000000000030.jpg", "000000000034.jpg",
          "000000000036.jpg", "000000000049.jpg", "000000000061.jpg", "000000000064.jpg"]
QUESTION = "Describe this image in two sentences, then name the most salient object."


def tokens_and_lp(out):
    c = out.outputs[0]
    lps = [d[t].logprob for t, d in zip(c.token_ids, c.logprobs)] if c.logprobs else []
    return {"text": c.text, "token_ids": list(c.token_ids), "logprobs": lps}


def compare(ref, arm):
    n = min(len(ref["token_ids"]), len(arm["token_ids"]))
    div = next((i for i in range(n) if ref["token_ids"][i] != arm["token_ids"][i]), None)
    if div is None and len(ref["token_ids"]) != len(arm["token_ids"]):
        div = n
    common = n if div is None else div
    dlp = max((abs(a - b) for a, b in zip(ref["logprobs"][:common], arm["logprobs"][:common])), default=0.0)
    d0 = abs(ref["logprobs"][0] - arm["logprobs"][0]) if ref["logprobs"] and arm["logprobs"] else float("nan")
    return {"exact": div is None, "first_divergence": div, "max_dlogprob_common": dlp, "dlogprob_first": d0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--arms", default="A,B,C,C2")
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--n-images", type=int, default=len(IMAGES))
    ap.add_argument("--gpu-mem", type=float, default=0.35)
    ap.add_argument("--max-num-batched-tokens", type=int, default=None)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--vit-backend", default=None, help="mm_encoder_attn_backend override (see compat.py)")
    ap.add_argument("--out", default=os.path.join(ROOT, "analysis/results/vllm_stream/gate_qwen25vl7b.json"))
    a = ap.parse_args()
    arms = a.arms.split(",")

    from PIL import Image
    from vllm import SamplingParams
    from appcorr.vllm_stream import StreamingLLM
    from appcorr.vllm_stream.client import Qwen25VLComposer

    from appcorr.vllm_stream.compat import fix_qwen2_5_vit_upstream_fa
    fix_qwen2_5_vit_upstream_fa()
    kw = {"mm_encoder_attn_backend": a.vit_backend} if a.vit_backend else {}
    if a.max_num_batched_tokens:
        kw["max_num_batched_tokens"] = a.max_num_batched_tokens
    llm = StreamingLLM(a.model, gpu_memory_utilization=a.gpu_mem, enforce_eager=a.enforce_eager,
                       limit_mm_per_prompt={"image": 1}, **kw)
    comp = Qwen25VLComposer(llm)
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens, logprobs=1)

    res = json.load(open(a.out)) if os.path.exists(a.out) else {"_meta": {}}
    res["_meta"].update({"model": a.model, "chunks": a.chunks, "max_tokens": a.max_tokens,
                         "question": QUESTION, "images": IMAGES[:a.n_images]})
    if a.max_num_batched_tokens:
        res["_meta"]["D_max_num_batched_tokens"] = a.max_num_batched_tokens

    for ii, name in enumerate(IMAGES[:a.n_images]):
        img = Image.open(os.path.join(COCO, name)).convert("RGB")
        parts = comp.parts(img, QUESTION)
        emb = comp.embed(parts) if any(x in arms for x in ("B", "C", "C2")) else None
        row = res.setdefault(name, {})
        row["num_prompt_tokens"] = parts.num_tokens
        row["image_span"] = [parts.image_start, parts.image_start + parts.image_len]

        if "A" in arms or "D" in arms:
            key = "D" if "D" in arms else "A"
            t0 = time.perf_counter()
            o = llm.generate([{"prompt": parts.text, "multi_modal_data": {"image": img}}], sp)[0]
            r = tokens_and_lp(o)
            r["wall_s"] = time.perf_counter() - t0
            r["prompt_ids_match_hf"] = list(o.prompt_token_ids) == parts.input_ids.tolist()
            row[key] = r
        if "B" in arms:
            rid = f"B-{ii}"
            t0 = time.perf_counter()
            llm.open(rid, emb.chunk(0, parts.num_tokens, final=True), sp)
            o = llm.run_until_done(rid)[rid]
            row["B"] = tokens_and_lp(o) | {"wall_s": time.perf_counter() - t0}
        bounds = comp.image_bounds(parts, a.chunks)
        if "C" in arms:
            rid = f"C-{ii}"
            chunks = emb.chunks(bounds)
            states = []
            t0 = time.perf_counter()
            llm.open(rid, chunks[0], sp)
            for ch in chunks[1:]:
                llm.step()
                states.append(llm.stream_state(rid))
                llm.append(rid, ch)
            o = llm.run_until_done(rid)[rid]
            row["C"] = tokens_and_lp(o) | {"wall_s": time.perf_counter() - t0, "bounds": bounds, "states": states}
        if "C2" in arms:
            rid = f"C2-{ii}"
            chunks = emb.chunks(bounds)
            t0 = time.perf_counter()
            llm.open(rid, chunks[0], sp)
            for ch in chunks[1:]:
                llm.append(rid, ch)
            o = llm.run_until_done(rid)[rid]
            row["C2"] = tokens_and_lp(o) | {"wall_s": time.perf_counter() - t0, "bounds": bounds}

        ref = row.get("A")
        line = f"[{ii}] {name} N={parts.num_tokens} img={row['image_span']}"
        for k in ("B", "C", "C2", "D"):
            if k in row and ref is not None:
                c = compare(ref, row[k])
                row[k]["vs_A"] = c
                tag = "exact" if c["exact"] else f"div@{c['first_divergence']}"
                line += f" | {k}: {tag} dlp={c['max_dlogprob_common']:.2e}"
        if "C" in row:
            st = row["C"]["states"]
            ok = all(s and s["open"] and s["num_computed_tokens"] == s["num_prompt_tokens"] - 1 and s["num_output_tokens"] == 0 for s in st)
            line += f" | hold-back {'OK' if ok else 'VIOLATED ' + json.dumps(st)}"
        print(line, flush=True)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)

    # summary
    names = IMAGES[:a.n_images]
    print("\nGATE SUMMARY vs A (stock image request):")
    for k in ("B", "C", "C2", "D"):
        rows = [res[n][k]["vs_A"] for n in names if k in res.get(n, {}) and "vs_A" in res[n][k]]
        if not rows:
            continue
        ex = sum(r["exact"] for r in rows)
        print(f"  {k:3s} exact {ex}/{len(rows)}  max dlogprob(common prefix) {max(r['max_dlogprob_common'] for r in rows):.3e}"
              f"  max |dlogprob| first token {max(r['dlogprob_first'] for r in rows):.3e}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
