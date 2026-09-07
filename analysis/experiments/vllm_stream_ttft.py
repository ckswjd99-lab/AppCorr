"""TTFT under progressive arrival: streaming prefill vs stock submit-after-last-chunk.

Emulates the AppCorr transmission/correction timeline on the serving side: the prompt's
`--chunks` chunks become available at t = k * `--gap-ms` (k = 0..chunks-1), the text being in
the last chunk. Two arms, same engine, one request at a time (the TTFT-vs-arrival question, not
throughput):

  stock      the request is submitted when the last chunk is available (t_last), one message
             (`oneshot`); TTFT is measured from t_last.
  stream     chunk k is appended at its arrival time; the engine steps in between (prefill of
             what has arrived overlaps the wait); TTFT is measured from t_last as well.

Both TTFTs are wall-clock from the last chunk's availability to the first sampled token, so
`stock - stream` is the prefill time hidden under the arrival gaps. `--gap-ms 0` is the sanity
case (stream must not be slower than stock beyond per-step overhead). Repeats `--reps` times per
image after one warm-up; reports median/mean per arm and the paired delta.

Run (GPU0, appcorr-vllm env = vllm 0.28.0; openrlhf_base = 0.11.2 also works):
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  PYTHONPATH=/NHNHOME/share/cjpark/AppCorr-vllm <python> analysis/experiments/vllm_stream_ttft.py \
      --chunks 4 --gap-ms 30 --reps 5
"""
from __future__ import annotations

import argparse
import json
import os
import statistics as st
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from analysis.experiments.vllm_stream_gate import COCO, IMAGES, QUESTION  # noqa: E402


def first_token_step(llm, rid, deadline_s=120.0):
    """Step until the request produced its first token; return the perf_counter of that step."""
    t_end = time.perf_counter() + deadline_s
    while time.perf_counter() < t_end:
        for o in llm.step():
            if o.request_id == rid and len(o.outputs[0].token_ids) >= 1:
                return time.perf_counter(), o
    raise RuntimeError(f"{rid}: no first token")


def run_stock(llm, emb, sp, rid):
    t_last = time.perf_counter()
    llm.open(rid, emb.chunk(0, emb.embeds.shape[0], final=True), sp)
    t_first, o = first_token_step(llm, rid)
    llm.run_until_done(rid)
    return t_first - t_last


def run_stream(llm, emb, bounds, sp, rid, gap_s):
    chunks = emb.chunks(bounds)
    t0 = time.perf_counter()
    arrivals = [t0 + k * gap_s for k in range(len(chunks))]
    llm.open(rid, chunks[0], sp)
    k = 1
    while k < len(chunks):
        if time.perf_counter() >= arrivals[k]:
            llm.append(rid, chunks[k])
            k += 1
        else:
            llm.step()  # prefill what has arrived while waiting
    t_last = max(arrivals[-1], t0)  # the last chunk's availability
    t_first, o = first_token_step(llm, rid)
    llm.run_until_done(rid)
    return t_first - t_last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--gap-ms", type=float, nargs="+", default=[0.0, 30.0, 100.0])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--n-images", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=8)
    ap.add_argument("--upscale", type=float, default=1.0,
                    help="resize images by this factor before the processor (larger prompts: "
                         "COCO 640x480 -> ~430 tokens at 1.0, ~3500 at 3.0)")
    ap.add_argument("--gpu-mem", type=float, default=0.35)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--out", default=None,
                    help="default analysis/results/vllm_stream/ttft_qwen25vl7b[_x<upscale>].json")
    a = ap.parse_args()
    if a.out is None:
        suffix = "" if a.upscale == 1.0 else f"_x{a.upscale:g}"
        a.out = os.path.join(ROOT, f"analysis/results/vllm_stream/ttft_qwen25vl7b{suffix}.json")

    from PIL import Image
    from vllm import SamplingParams
    from appcorr.vllm_stream import StreamingLLM
    from appcorr.vllm_stream.client import Qwen25VLComposer
    from appcorr.vllm_stream.compat import fix_qwen2_5_vit_upstream_fa
    fix_qwen2_5_vit_upstream_fa()

    llm = StreamingLLM(a.model, gpu_memory_utilization=a.gpu_mem, enforce_eager=a.enforce_eager,
                       limit_mm_per_prompt={"image": 1})
    comp = Qwen25VLComposer(llm)
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens)

    embs = []
    for name in IMAGES[:a.n_images]:
        img = Image.open(os.path.join(COCO, name)).convert("RGB")
        if a.upscale != 1.0:
            img = img.resize((round(img.width * a.upscale), round(img.height * a.upscale)), Image.BICUBIC)
        parts = comp.parts(img, QUESTION)
        embs.append((name, comp.embed(parts), comp.image_bounds(parts, a.chunks)))
    # warm-up
    run_stock(llm, embs[0][1], sp, "warm-stock")
    run_stream(llm, embs[0][1], embs[0][2], sp, "warm-stream", 0.0)

    res = {"_meta": vars(a) | {"question": QUESTION, "num_prompt_tokens": {n: int(e.embeds.shape[0]) for n, e, _ in embs}},
           "gaps": {}}
    for gap in a.gap_ms:
        g = gap / 1000.0
        rows = []
        for ii, (name, emb, bounds) in enumerate(embs):
            for r in range(a.reps):
                ts = run_stock(llm, emb, sp, f"stock-{gap}-{ii}-{r}")
                tr = run_stream(llm, emb, bounds, sp, f"stream-{gap}-{ii}-{r}", g)
                rows.append({"image": name, "rep": r, "stock_ms": ts * 1e3, "stream_ms": tr * 1e3})
        stock = [x["stock_ms"] for x in rows]
        stream = [x["stream_ms"] for x in rows]
        delta = [s - t for s, t in zip(stock, stream)]
        summ = {"stock_median_ms": st.median(stock), "stream_median_ms": st.median(stream),
                "stock_mean_ms": st.mean(stock), "stream_mean_ms": st.mean(stream),
                "delta_median_ms": st.median(delta), "delta_min_ms": min(delta), "delta_max_ms": max(delta),
                "n": len(rows)}
        res["gaps"][str(gap)] = {"summary": summ, "rows": rows}
        print(f"gap={gap:6.1f} ms  chunks={a.chunks}  TTFT from last chunk: stock median {summ['stock_median_ms']:7.1f} ms | "
              f"stream median {summ['stream_median_ms']:7.1f} ms | paired delta median {summ['delta_median_ms']:+7.1f} "
              f"[{summ['delta_min_ms']:+.1f}, {summ['delta_max_ms']:+.1f}] (n={summ['n']})", flush=True)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
