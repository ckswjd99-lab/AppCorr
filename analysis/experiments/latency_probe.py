"""Quick TTFT probe for the eval table's Lat. / Crit. Lat. columns (FLOPs-style, minutes not hours).

Runs `qwen_vllm_accuracy.py --concurrency 1` (no queueing, one request in flight) for the ceiling
and the streaming arms at each keep on a few evenly spaced images per dataset and pins the medians
into analysis/results/latency/inprocess_latency.json with the SAME key names as
analysis/results/flops/inprocess_flops.json, so make_eval_table.py reads both the same way:

  full          ceiling TTFT from t0 (one-shot tower + full prefill; 100% critical, the denominator)
  total_k{keep} streaming TTFT from t0 -- every pass serialized as if the whole image were present
                at t0 (the analogue of Comp.: total work, transmission overlap NOT credited)
  k{keep}       streaming TTFT measured from the moment the LAST image band's pixels are consumed,
                i.e. the start of its vision correction (the push of band g-2 precedes it; the push
                syncs the GPU) -- what a user waits after the final byte: last-band correction +
                its push + its prefill + the trailing text + the first decode step (the analogue
                of Crit. Comp., which charges the last band's correction the same way)

t0 = the driver clock at "inputs on cuda:0, before vision()" (`t_start` in the driver); TTFT ends
at the engine's first-token timestamp relayed through the bridge. Both include the driver-side
vision pass, the socket hop and the engine's chunked prefill; decode is excluded. The first
`--warmup` rows of every arm are dropped (cold kernels/allocator), medians over the rest.

Usage (122B on the live server, all seven table datasets):
  python analysis/experiments/latency_probe.py --family qwen35 --model Qwen/Qwen3.5-122B-A10B-FP8 \
      --port 5591 --key qwen35_122b --samples 40 --warmup 4 --keeps 1.0 0.50 0.25 \
      --datasets visdrone_det:pyr visdrone_count:pyr vstar:pyr textvqa:pyr refcoco:pyr \
                 realworldqa:box chartqa:box
The dataset:filter pairs follow the campaign's degrade-filter per dataset.
"""
import argparse, json, os, statistics as st, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_JSON = os.path.join(ROOT, "analysis", "results", "latency", "inprocess_latency.json")
README = [
    "Single-request (concurrency 1) time-to-first-token medians, ms, for the eval table's Lat. /",
    "Crit. Lat. columns. Keys mirror inprocess_flops.json: 'full' = ceiling TTFT from t0 (one-shot",
    "tower + full prefill, the denominator); 'total_k*' = streaming TTFT from t0 with every pass",
    "serialized (no transmission credit; analogue of Comp.); 'k*' = streaming TTFT from the start",
    "of the LAST band's vision correction (= the push of band g-2, which syncs the GPU): last-band",
    "correction + push + prefill + trailing text + first decode -- what waits on the final byte",
    "(analogue of Crit. Comp.). 'detail' keeps ttft_last_chunk_ms (from the last push alone) too.",
    "t0 = driver inputs on the",
    "GPU before the vision pass; first token = engine timestamp via the bridge. Decode excluded.",
    "Each entry's 'detail' keeps the per-arm medians of the driver's timing fields.",
    "Reproduce: python analysis/experiments/latency_probe.py (see its docstring).",
]
FIELDS = ["t_vision_ms", "t_open_ms", "t_last_push_ms", "ttft_open_ms", "ttft_last_chunk_ms",
          "ttft_last_band_ms", "ttft_start_ms", "t_client_done_ms", "prompt_tokens", "gen_tokens"]


def rows_of(path, warmup, groups):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    rows = [r for r in rows if "skip" not in r and r.get("ttft_start_ms") is not None]
    for r in rows:
        tp = r.get("t_pushes_ms")
        if tp and len(tp) >= groups:
            # first token (driver clock) minus the push of band g-2 = the start of band g-1's
            # correction; a one-chunk arm (ceiling) has no band structure -> from t0
            r["ttft_last_band_ms"] = r["ttft_start_ms"] - tp[groups - 2]
        else:
            r["ttft_last_band_ms"] = r["ttft_start_ms"]
    return rows[warmup:]


def med(rows, k):
    v = [r[k] for r in rows if r.get(k) is not None]
    return round(st.median(v), 1) if v else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=5591)
    ap.add_argument("--key", required=True, help="model key in inprocess_latency.json, e.g. qwen35_122b")
    ap.add_argument("--datasets", nargs="+", required=True, help="dataset[:degrade-filter] ...")
    ap.add_argument("--keeps", type=float, nargs="+", default=[1.0, 0.50, 0.25])
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--out", default=None, help="driver output dir (default results/latency/probe_<key>)")
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--aggregate-only", action="store_true")
    a = ap.parse_args()
    out = a.out or os.path.join(ROOT, "analysis", "results", "latency", f"probe_{a.key}")
    os.makedirs(out, exist_ok=True)
    slug = a.model.split("/")[-1].lower()
    env = dict(os.environ, PYTHONPATH=ROOT, HF_HUB_OFFLINE=os.environ.get("HF_HUB_OFFLINE", "1"))
    drv = [a.python, os.path.join(ROOT, "analysis", "experiments", "qwen_vllm_accuracy.py"),
           "--family", a.family, "--model", a.model, "--port", str(a.port), "--groups", str(a.groups),
           "--level", str(a.level), "--workers", str(a.workers), "--load", "vision",
           "--samples", str(a.samples), "--concurrency", "1", "--out", out]

    def arm_path(ds, arm, keep=None):
        suf = "" if arm != "streaming" else f"_g{a.groups}" + (f"_k{keep:.2f}" if keep < 1.0 else "")
        return os.path.join(out, f"{ds}_{slug}_{arm}{suf}.jsonl")

    def run(ds, filt, arm, keep=1.0):
        p = arm_path(ds, arm, keep)
        if os.path.exists(p):
            os.remove(p)          # the driver resumes from existing rows; a probe must be fresh
        cmd = drv + ["--dataset", ds, "--degrade-filter", filt, "--arms", arm, "--keep", f"{keep}"]
        log = os.path.join(out, f"log_{ds}_{arm}_k{keep:.2f}.log")
        t0 = time.time()
        with open(log, "w") as lf:
            rc = subprocess.call(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=ROOT,
                                 timeout=a.timeout)
        print(f"  {ds:15s} {arm:10s} k={keep:.2f} rc={rc} {time.time() - t0:5.0f}s", flush=True)

    specs = [(d.split(":")[0], d.split(":")[1] if ":" in d else "box") for d in a.datasets]
    if not a.aggregate_only:
        for ds, filt in specs:
            run(ds, filt, "ceiling")
            for k in a.keeps:
                run(ds, filt, "streaming", k)

    J = json.load(open(OUT_JSON)) if os.path.exists(OUT_JSON) else {"_README": README}
    J["_README"] = README
    M = J.setdefault(a.key, {})
    M["_note"] = (f"{a.model} via the vLLM stream server on port {a.port}, --samples {a.samples} "
                  f"evenly spaced, --warmup {a.warmup} dropped, concurrency 1, groups {a.groups}, "
                  f"L{a.level}; per-dataset degrade filter as listed in each entry. "
                  f"Measured {time.strftime('%Y-%m-%d')}.")
    for ds, filt in specs:
        e = M.setdefault(ds, {})
        e["filter"], e["detail"] = filt, {}
        p = arm_path(ds, "ceiling")
        if os.path.exists(p):
            r = rows_of(p, a.warmup, a.groups)
            e["full"], e["n"] = med(r, "ttft_start_ms"), len(r)
            e["detail"]["ceiling"] = {k: med(r, k) for k in FIELDS}
        for k in a.keeps:
            p = arm_path(ds, "streaming", k)
            if not os.path.exists(p):
                continue
            r = rows_of(p, a.warmup, a.groups)
            e[f"k{k:.2f}"] = med(r, "ttft_last_band_ms")
            e[f"total_k{k:.2f}"] = med(r, "ttft_start_ms")
            e["detail"][f"streaming_k{k:.2f}"] = {kk: med(r, kk) for kk in FIELDS}
        full = e.get("full")
        print(f"{ds:15s} full {full} ms | " + " | ".join(
            f"k{k:.2f}: total {e.get(f'total_k{k:.2f}')} crit {e.get(f'k{k:.2f}')}" for k in a.keeps)
              + (f" | n={e.get('n')}" if full else ""), flush=True)
    json.dump(J, open(OUT_JSON, "w"), indent=1)
    print(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
