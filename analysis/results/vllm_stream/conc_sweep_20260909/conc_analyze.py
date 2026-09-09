"""Concurrency sweep analysis: per (ds, c, arm) TTFT medians/p90, throughput; per run the engine
step-time vs tokens-per-step from the server trace (bounded by the run's perf_counter window)."""
import json, os, re, statistics as st, sys
OUT = "/NHNHOME/share/cjpark/AppCorr-vllm/analysis/results/vllm_stream/conc_sweep_20260909"
WARM, G = 8, 4
def pct(v, p):
    v = sorted(v); return v[min(len(v)-1, int(round(p*(len(v)-1))))]
bounds = [json.loads(l) for l in open(f"{OUT}/bounds.jsonl") if l.strip()]
steps = []
tp = f"{OUT}/server_trace_35b.jsonl"
if os.path.exists(tp):
    for l in open(tp):
        try: e = json.loads(l)
        except Exception: continue
        if e.get("ev") == "step":
            e["tok"] = sum(v[1]-v[0] for v in e["computed"].values()); e["nreq"] = len(e["computed"]); steps.append(e)
res = {}
for b in bounds:
    ds, c, arm = b["ds"], b["c"], b["arm"]
    d = f"{OUT}/{ds}/c{c}"
    fs = [f for f in os.listdir(d) if f.endswith(".jsonl") and f"_{arm}" in f]
    if not fs: print("missing", ds, c, arm); continue
    rows = [json.loads(l) for l in open(os.path.join(d, fs[0])) if l.strip()]
    rows = [r for r in rows if "skip" not in r and r.get("ttft_start_ms") is not None][WARM:]
    tot = [r["ttft_start_ms"] for r in rows]
    if arm == "streaming":
        crit = [r["ttft_start_ms"] - r["t_sent_ms"][G-2] for r in rows if r.get("t_sent_ms") and r["t_sent_ms"][G-2] is not None]
    else:
        crit = tot
    log = open(f"{d}/log_{arm}.log").read()
    m = re.search(r'samples_per_s": ([0-9.]+)', log)
    sps = float(m.group(1)) if m else None
    ss = [s for s in steps if b["t0"] <= s["t0"] <= b["t1"]]
    pre = [s for s in ss if s["tok"] >= 64]          # prefill-carrying steps
    res[(ds, c, arm)] = dict(n=len(rows), tot_med=st.median(tot), tot_p90=pct(tot, .9),
        crit_med=st.median(crit), crit_p90=pct(crit, .9), sps=sps, rc=b["rc"],
        steps=len(ss), pre_steps=len(pre),
        tok_med=st.median([s["tok"] for s in pre]) if pre else None,
        ms_med=st.median([s["ms"] for s in pre]) if pre else None,
        nreq_med=st.median([s["nreq"] for s in pre]) if pre else None,
        ms_per_ktok=st.median([s["ms"]/s["tok"]*1e3 for s in pre]) if pre else None)
print(f"{'ds':13s} {'c':>3s} {'arm':9s} {'n':>3s} {'tot_med':>8s} {'p90':>7s} {'crit_med':>8s} {'p90':>7s} {'smp/s':>6s} | {'presteps':>8s} {'tok/step':>8s} {'ms/step':>7s} {'req/step':>8s} {'ms/ktok':>7s}")
for k in sorted(res, key=lambda k: (k[0], k[1], k[2])):
    r = res[k]
    f = lambda v, w=7: f"{v:{w}.1f}" if isinstance(v, (int, float)) and v is not None else f"{'-':>{w}s}"
    print(f"{k[0]:13s} {k[1]:3d} {k[2]:9s} {r['n']:3d} {f(r['tot_med'],8)} {f(r['tot_p90'])} {f(r['crit_med'],8)} {f(r['crit_p90'])} {f(r['sps'],6)} | {r['pre_steps']:8d} {f(r['tok_med'],8)} {f(r['ms_med'])} {f(r['nreq_med'],8)} {f(r['ms_per_ktok'])}")
# step time vs tokens, all prefill steps pooled, by token bin
print("\nengine step time by tokens/step (all runs pooled, prefill-carrying steps):")
bins = [(64,512),(512,1024),(1024,2048),(2048,4096),(4096,8192),(8192,16385)]
pre = [s for s in steps if s["tok"] >= 64]
for lo, hi in bins:
    v = [s for s in pre if lo <= s["tok"] < hi]
    if v: print(f"  [{lo:5d},{hi:5d}) n={len(v):5d} ms med {st.median([s['ms'] for s in v]):6.1f} p90 {pct([s['ms'] for s in v],.9):6.1f}  req/step med {st.median([s['nreq'] for s in v]):.0f}  ms/ktok {st.median([s['ms']/s['tok']*1e3 for s in v]):5.1f}")
json.dump({f"{k[0]}|c{k[1]}|{k[2]}": v for k, v in res.items()}, open(f"{OUT}/summary.json", "w"), indent=1)
