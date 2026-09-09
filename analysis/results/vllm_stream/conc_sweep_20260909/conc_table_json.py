"""Fold the two concurrency sweeps into one JSON for make_eval_table.py's latency table.

Medians / p90 / n come from each sweep's summary.json (conc_analyze*.py); req/s and engine
busy% are recomputed over the ACTIVE phase of each run (first `open` .. last step end in the
server trace), the memo's definition (docs/memo/vllm_stream_design.md, 2026-09-09 section).
N=16 sharded cells are partial (OOM) and are dropped.
"""
import json, os, statistics as st
HERE = os.path.dirname(os.path.abspath(__file__))
out = {"_note": "Qwen3.5-35B-A3B on one B200 (GPU0), vLLM 0.28, port 5591, g=4, keep=1.0. "
                "single: one driver, --concurrency c, 160 images (8 warmup dropped). shard: N driver "
                "processes (own vision tower each) x c in flight, 400 images (4 warmup/shard dropped). "
                "req/s and busy over the active phase (first open .. last step end).",
       "rows": []}
for tag, tp, bp, sp in [("single", "server_trace_35b.jsonl", "bounds.jsonl", "summary.json"),
                        ("shard", "shard/server_trace_35b.jsonl", "shard/bounds.jsonl", "shard/summary.json")]:
    ev = []
    for l in open(os.path.join(HERE, tp)):
        try: ev.append(json.loads(l))
        except Exception: pass
    summ = json.load(open(os.path.join(HERE, sp)))
    for b in (json.loads(l) for l in open(os.path.join(HERE, bp)) if l.strip()):
        if b.get("N") == 16:
            continue
        w = [e for e in ev if b["t0"] <= (e.get("t0") or e.get("t")) <= b["t1"]]
        opens = [e["t"] for e in w if e["ev"] == "handle" and e["op"] == "open"]
        steps = [e for e in w if e["ev"] == "step"]
        if not opens or not steps:
            continue
        a0 = min(opens); a1 = max(e["t0"] + e["ms"] / 1e3 for e in steps)
        A = a1 - a0
        cfg = f"c{b['c']}" if "N" not in b else f"n{b['N']}c{b['c']}"
        key = f"{b['ds']}|{cfg}|{b['arm']}"
        s = summ.get(key) or {}
        pre = [e for e in steps if sum(v[1]-v[0] for v in e["computed"].values()) >= 64]
        toks = [sum(v[1]-v[0] for v in e["computed"].values()) for e in pre]
        out["rows"].append({
            "sweep": tag, "ds": b["ds"], "N": b.get("N", 1), "c": b["c"], "arm": b["arm"],
            "n": s.get("n"), "tot_med": s.get("tot_med"), "tot_p90": s.get("tot_p90"),
            "crit_med": s.get("crit_med"), "crit_p90": s.get("crit_p90"),
            "req_s": len(opens) / A, "busy": sum(e["ms"] for e in steps) / (A * 1e3) * 100,
            "active_s": A,
            "pre_tok_med": st.median(toks) if toks else None,
            "pre_ms_med": st.median([e["ms"] for e in pre]) if pre else None,
        })
p = os.path.join(HERE, "..", "..", "latency", "conc_sweep_20260909.json")
json.dump(out, open(p, "w"), indent=1)
print("wrote", os.path.abspath(p), len(out["rows"]), "rows")
for r in out["rows"]:
    print(f"{r['ds']:13s} N{r['N']:<2d}c{r['c']} {r['arm']:9s} n={r['n']} tot {r['tot_med']:.0f}/{r['tot_p90']:.0f} crit {r['crit_med']:.0f}/{r['crit_p90']:.0f} req/s {r['req_s']:.1f} busy {r['busy']:.0f}% tok {r['pre_tok_med']} ms {r['pre_ms_med']:.0f}")
