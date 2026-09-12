"""Where the interleaved correct step's wall time goes (2026-09-10).

Served probes put the last-round correct step at 26-37 ms on 35B regardless of |P| (RealWorldQA
k=0.25, ~60 rows: 27 ms; V*Bench k=1, 835 rows: 35 ms), against ~18 ms for a stock chunked
prefill of the same rows -- a fixed floor, not a per-row cost. This harness opens one request
in-process (same engine config as the served 35B), runs the g=4 keep-half schedule and profiles
the FINAL round (image rows + text suffix) with torch.profiler: GPU kernel time by family
(GDN re-scan, softmax attention, MoE, other), H2D/D2H copies, and the CPU time of the metadata
build vs the layer loop. Numbers land in analysis/results/vllm_stream/correct_profile_<tag>.json.

  APPCORR_CORRECT_CUDAGRAPH=1 python analysis/experiments/vllm_correct_profile.py \
      --model Qwen/Qwen3.5-35B-A3B --gpu-mem 0.60 --max-model-len 16384 --max-num-seqs 1024 \
      --long-side 1792 --keep 0.5 --repeat 3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from vllm_stream_gate import COCO, IMAGES, QUESTION  # noqa: E402
from vllm_correct_gate import bands_of, run_to_first_token  # noqa: E402


def family(name: str) -> str:
    n = name.lower()
    if "memcpy" in n or "memset" in n:
        return "copy"
    if any(t in n for t in ("chunk_", "fwd_recompute", "fwd_h", "fwd_o", "conv1d", "causal_conv",
                            "fused_post_conv", "l2norm", "gated_delta", "solve_tril", "fwd_prepare",
                            "fla", "recurrent")):
        return "gdn"
    if any(t in n for t in ("attn", "attention", "flash", "paged", "batchdecode", "batchprefill",
                            "reshape_and_cache", "rotary", "mrope")):
        return "attention"
    if any(t in n for t in ("moe", "grouped", "topk", "fused_experts", "silu_and_mul",
                            "invoke_fused", "cutlass", "group_gemm", "expert")):
        return "moe"
    if any(t in n for t in ("gemm", "cublas", "sm90", "sm100", "nvjet", "matmul")):
        return "gemm"
    if any(t in n for t in ("norm", "elementwise", "vectorized", "reduce", "index", "gather",
                            "scatter", "cat", "fill", "copy_", "sigmoid", "silu", "softmax",
                            "add", "mul")):
        return "elementwise"
    return "other"


def summarize(prof, wall_ms: float) -> dict:
    ka = prof.key_averages()
    fam = {}
    n_copy = {"h2d": 0, "d2h": 0, "d2d": 0}
    top = []
    for ev in ka:
        dev_us = getattr(ev, "self_device_time_total", None)
        if dev_us is None:
            dev_us = getattr(ev, "self_cuda_time_total", 0.0)
        if dev_us <= 0:
            continue
        f = family(ev.key)
        d = fam.setdefault(f, {"ms": 0.0, "launches": 0})
        d["ms"] += dev_us / 1e3
        d["launches"] += ev.count
        top.append((dev_us / 1e3, ev.count, ev.key[:90]))
        k = ev.key.lower()
        if "memcpy htod" in k:
            n_copy["h2d"] += ev.count
        elif "memcpy dtoh" in k:
            n_copy["d2h"] += ev.count
        elif "memcpy dtod" in k:
            n_copy["d2d"] += ev.count
    top.sort(reverse=True)
    gpu_ms = sum(d["ms"] for d in fam.values())
    scopes = {}
    for ev in ka:
        if ev.key.startswith("appcorr."):
            scopes[ev.key] = {"cpu_ms": ev.cpu_time_total / 1e3, "count": ev.count}
    return {"wall_ms": wall_ms, "gpu_kernel_ms": gpu_ms, "gpu_idle_ms": wall_ms - gpu_ms,
            "families": {k: {"ms": round(v["ms"], 2), "launches": v["launches"]}
                         for k, v in sorted(fam.items(), key=lambda kv: -kv[1]["ms"])},
            "copies": n_copy, "scopes": scopes,
            "top_kernels": [{"ms": round(m, 3), "n": c, "name": k} for m, c, k in top[:25]],
            "n_launches": sum(d["launches"] for d in fam.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    ap.add_argument("--gpu-mem", type=float, default=0.60)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--max-num-seqs", type=int, default=1024)
    ap.add_argument("--g", type=int, default=4)
    ap.add_argument("--keep", type=float, default=0.5, help="every 1/keep-th row of a band")
    ap.add_argument("--long-side", type=int, default=1792,
                    help="resize the gate image so its long side is this (V*-like token count)")
    ap.add_argument("--downscale", type=int, default=4)
    ap.add_argument("--repeat", type=int, default=3, help="requests; the last one is profiled")
    ap.add_argument("--staged", action="store_true")
    ap.add_argument("--image", default=IMAGES[0])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    os.environ.setdefault("VLLM_GDN_DECODE_KERNEL", "triton")
    tag = a.model.split("/")[-1].replace(".", "").lower()
    out = a.out or os.path.join(ROOT, f"analysis/results/vllm_stream/correct_profile_{tag}.json")

    from PIL import Image
    from vllm import SamplingParams
    from appcorr.vllm_stream import StreamingLLM
    from appcorr.vllm_stream import correct as _correct
    from appcorr.vllm_stream.client import Qwen25VLComposer

    llm = StreamingLLM(a.model, gpu_memory_utilization=a.gpu_mem, max_model_len=a.max_model_len,
                       limit_mm_per_prompt={"image": 1}, max_num_seqs=a.max_num_seqs)
    comp = Qwen25VLComposer(llm)
    sp = SamplingParams(temperature=0.0, max_tokens=4, logprobs=1)

    img = Image.open(os.path.join(COCO, a.image)).convert("RGB")
    w, h = img.size
    sc = a.long_side / max(w, h)
    img = img.resize((int(w * sc), int(h * sc)), Image.BICUBIC)
    w, h = img.size
    parts = comp.parts(img, QUESTION)
    N, lo, G = parts.num_tokens, parts.image_start, parts.image_len
    emb = comp.embed(parts)
    low = img.resize((max(w // a.downscale, 1), max(h // a.downscale, 1))).resize((w, h))
    emb_a = comp.embed(comp.parts(low, QUESTION))
    stride = max(1, int(round(1.0 / a.keep)))
    print(f"N={N} lo={lo} G={G} bands={bands_of(G, a.g)} stride={stride} "
          f"cudagraph={_correct.CUDAGRAPH}", flush=True)

    res = {"_meta": {"model": a.model, "N": N, "lo": lo, "G": G, "g": a.g, "keep": a.keep,
                     "staged": a.staged, "cudagraph": _correct.CUDAGRAPH,
                     "fuse_holdback": _correct.FUSE_HOLDBACK, "defer_final": llm.defer_final,
                     "max_num_seqs": a.max_num_seqs}, "rounds": []}
    prof_summary = None
    for it in range(a.repeat):
        rid = f"prof-{it}"
        llm.open(rid, emb_a.chunk(0, N, final=False), sp, correct=True, image_start=lo,
                 image_end=lo + G)
        bs = bands_of(G, a.g)
        rounds = []
        for r, (g0, g1) in enumerate(bs):
            final = r == len(bs) - 1
            pos = torch.arange(lo + g0, lo + g1, dtype=torch.int64)[::stride]
            rows = emb.embeds[pos]
            win = (lo + g0, lo + g1)
            if final:
                pos = torch.cat([pos, torch.arange(lo + G, N - 1, dtype=torch.int64)])
                rows = torch.cat([rows, emb.embeds[lo + G:N - 1]], dim=0)
                win = (lo + g0, N - 1)
            stage = (r, a.g) if a.staged else None
            if final and it == a.repeat - 1:
                llm._drain_until_prefilled(rid)      # keep the approx prefill out of the profile
                torch.cuda.synchronize()
                with torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU,
                                    torch.profiler.ProfilerActivity.CUDA]) as prof:
                    t0 = time.perf_counter()
                    info = llm.correct(rid, pos, rows, win, final, stage=stage)
                    torch.cuda.synchronize()
                    wall = (time.perf_counter() - t0) * 1e3
                prof_summary = summarize(prof, wall)
                prof_summary["P"] = int(pos.numel())
                prof_summary["window"] = list(win)
                prof_summary["t_step_ms"] = info["t_step_ms"]
                prof.export_chrome_trace(out.replace(".json", "_trace.json"))
            else:
                t0 = time.perf_counter()
                info = llm.correct(rid, pos, rows, win, final, stage=stage)
                wall = (time.perf_counter() - t0) * 1e3
            rounds.append({"r": r, "P": int(pos.numel()), "window": list(win),
                           "t_step_ms": round(info["t_step_ms"], 2), "wall_ms": round(wall, 2),
                           "cudagraph": info.get("cudagraph"), "fused": info.get("fused")})
            print(f"  it{it} round {r} P={pos.numel()} win={win} step {info['t_step_ms']:.1f} ms "
                  f"wall {wall:.1f} ms cg={info.get('cudagraph')}", flush=True)
        t0 = time.perf_counter()
        run_to_first_token(llm, rid)
        t_ft = (time.perf_counter() - t0) * 1e3
        llm.run_until_done(rid)
        res["rounds"].append({"it": it, "rounds": rounds, "first_token_ms": round(t_ft, 2),
                              "final_to_first_token_ms": round(rounds[-1]["wall_ms"] + t_ft, 2)})
        print(f"  it{it} hold-back step -> first token {t_ft:.1f} ms "
              f"(final correct wall + hold-back = {rounds[-1]['wall_ms'] + t_ft:.1f} ms, "
              f"fused={rounds[-1]['fused']})", flush=True)
    res["profile"] = prof_summary
    os.makedirs(os.path.dirname(out), exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    p = prof_summary
    print(f"\nFINAL ROUND P={p['P']} window={p['window']}: wall {p['wall_ms']:.1f} ms, "
          f"GPU kernels {p['gpu_kernel_ms']:.1f} ms ({p['n_launches']} launches), "
          f"idle {p['gpu_idle_ms']:.1f} ms; copies {p['copies']}")
    for k, v in p["families"].items():
        print(f"  {k:12s} {v['ms']:7.2f} ms  {v['launches']:5d} launches")
    for k, v in p["scopes"].items():
        print(f"  scope {k:20s} cpu {v['cpu_ms']:7.2f} ms  x{v['count']}")
    for t in p["top_kernels"][:15]:
        print(f"  {t['ms']:7.3f} ms x{t['n']:4d}  {t['name']}")
    print("wrote", out)


if __name__ == "__main__":
    main()
