import json, os, re, statistics as st, glob
OUT = "/NHNHOME/share/cjpark/AppCorr-vllm/analysis/results/vllm_stream/conc_sweep_20260909/shard"
WARM, G = 4, 4
def pct(v, p):
    v = sorted(v); return v[min(len(v)-1, int(round(p*(len(v)-1))))]
bounds = [json.loads(l) for l in open(f"{OUT}/bounds.jsonl") if l.strip()]
steps = []
for l in open(f"{OUT}/server_trace_35b.jsonl"):
    try: e = json.loads(l)
    except Exception: continue
    if e.get("ev") == "step":
        e["tok"] = sum(v[1]-v[0] for v in e["computed"].values()); e["nreq"] = len(e["computed"])
        e["npre"] = sum(1 for v in e["computed"].values() if v[1]-v[0] > 1); steps.append(e)
res = {}
print(f"{'ds':13s} {'N':>3s} {'c':>2s} {'arm':9s} {'n':>3s} {'tot_med':>8s} {'p90':>7s} {'crit_med':>8s} {'p90':>7s} {'smp/s':>6s} | {'presteps':>8s} {'tok/step':>8s} {'ms/step':>7s} {'pre/step':>8s} {'req/step':>8s} {'ms/ktok':>7s} {'busy%':>5s}")
for b in bounds:
    ds, N, c, arm = b["ds"], b["N"], b["c"], b["arm"]
    d = f"{OUT}/{ds}/n{N}c{c}"
    rows = []; sps = 0.0
    for f in sorted(glob.glob(f"{d}/*_{arm}*_s*of{N}.jsonl")):
        rr = [json.loads(l) for l in open(f) if l.strip()]
        rr = [r for r in rr if "skip" not in r and r.get("ttft_start_ms") is not None]
        rows += rr[WARM:]
    for f in glob.glob(f"{d}/log_{arm}_s*.log"):
        m = re.search(r'samples_per_s": ([0-9.]+)', open(f).read())
        if m: sps += float(m.group(1))
    tot = [r["ttft_start_ms"] for r in rows]
    crit = [r["ttft_start_ms"] - r["t_sent_ms"][G-2] for r in rows if r.get("t_sent_ms") and r["t_sent_ms"][G-2] is not None] if arm == "streaming" else tot
    ss = [s for s in steps if b["t0"] <= s["t0"] <= b["t1"]]
    pre = [s for s in ss if s["tok"] >= 64]
    busy = sum(s["ms"] for s in ss) / ((b["t1"]-b["t0"])*1e3) * 100 if ss else 0
    r = dict(n=len(rows), tot_med=st.median(tot) if tot else None, tot_p90=pct(tot,.9) if tot else None,
             crit_med=st.median(crit) if crit else None, crit_p90=pct(crit,.9) if crit else None, sps=sps, rc=b["rc"],
             pre_steps=len(pre), tok_med=st.median([s["tok"] for s in pre]) if pre else None,
             ms_med=st.median([s["ms"] for s in pre]) if pre else None,
             npre_med=st.median([s["npre"] for s in pre]) if pre else None,
             nreq_med=st.median([s["nreq"] for s in pre]) if pre else None,
             ms_per_ktok=st.median([s["ms"]/s["tok"]*1e3 for s in pre]) if pre else None, busy=busy)
    res[f"{ds}|n{N}c{c}|{arm}"] = r
    f = lambda v, w=7: f"{v:{w}.1f}" if isinstance(v, (int, float)) and v is not None else f"{'-':>{w}s}"
    print(f"{ds:13s} {N:3d} {c:2d} {arm:9s} {r['n']:3d} {f(r['tot_med'],8)} {f(r['tot_p90'])} {f(r['crit_med'],8)} {f(r['crit_p90'])} {f(r['sps'],6)} | {r['pre_steps']:8d} {f(r['tok_med'],8)} {f(r['ms_med'])} {f(r['npre_med'],8)} {f(r['nreq_med'],8)} {f(r['ms_per_ktok'])} {f(r['busy'],5)}")
print("\nengine step time by tokens/step (pooled, prefill-carrying steps):")
pre = [s for s in steps if s["tok"] >= 64]
for lo, hi in [(64,512),(512,1024),(1024,2048),(2048,4096),(4096,8192),(8192,16385)]:
    v = [s for s in pre if lo <= s["tok"] < hi]
    if v: print(f"  [{lo:5d},{hi:5d}) n={len(v):5d} ms med {st.median([s['ms'] for s in v]):6.1f} p90 {pct([s['ms'] for s in v],.9):6.1f}  prefills/step med {st.median([s['npre'] for s in v]):.0f}  ms/ktok {st.median([s['ms']/s['tok']*1e3 for s in v]):5.1f}")
json.dump(res, open(f"{OUT}/summary.json", "w"), indent=1)
