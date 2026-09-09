"""Spaced-arrival probe analysis: per (delay, dataset) medians of the chunk timeline.
push()-sleep semantics: t_send/t_sent recorded BEFORE the sleep, band r+1's correction starts after
it, so gap[r] = sent[r]-sent[r-1]-delay is band r's correction (+push) on the GPU-completion clock.
"""
import json, glob, os, statistics as st, sys
OUT = "/NHNHOME/share/cjpark/AppCorr-vllm/analysis/results/latency/probe_qwen35_moe_delay"
WARM = 4
def med(xs): xs=[x for x in xs if x is not None]; return round(st.median(xs),1) if xs else None
def rows(p):
    R=[json.loads(l) for l in open(p)]; R=[r for r in R if r.get("t_sent_ms")]; return R[WARM:]
print(f"{'d':>4} {'ds':13} {'n':>3} {'tok':>5} | {'vis':>5} {'vis-4d':>6} | L1 L2 L3 (sent[r]-push[r-1]-d) | cpu1 cpu2 cpu3 (push gaps-d) | q0 q1 q2 q3 (recv-sent) | last(recv3->FT) | Lat FT-t0 | crit(FT-sent2-d) = c3+q3+last")
for D in (0,60,150):
    for ds in ("realworldqa","visdrone_det","vstar"):
        p=f"{OUT}/d{D}/{ds}_qwen3.5-35b-a3b_streaming_g4.jsonl"
        if not os.path.exists(p): continue
        R=rows(p)
        if not R: continue
        g=lambda f: [f(r) for r in R]
        # band r correction latency: from its start (push r-1 issued + sleep) to the GPU completion of
        # its D2H (sent r); at d=0 this also contains the GPU backlog the sync-free tower runs ahead of
        corr=[med(g(lambda r: r["t_sent_ms"][k]-r["t_pushes_ms"][k-1]-D)) for k in (1,2,3)]
        cpu=[med(g(lambda r: r["t_pushes_ms"][k]-r["t_pushes_ms"][k-1]-D)) for k in (1,2,3)]
        q=[med(g(lambda r: r["t_recv_ms"][k]-r["t_sent_ms"][k])) for k in range(4)]
        last=med(g(lambda r: r["ttft_start_ms"]-r["t_recv_ms"][3]))
        crit=med(g(lambda r: r["ttft_start_ms"]-r["t_sent_ms"][2]-D))
        print(f"{D:>4} {ds:13} {len(R):>3} {med(g(lambda r:r['prompt_tokens'])):>5} | {med(g(lambda r:r['t_vision_ms'])):>5} {med(g(lambda r:r['t_vision_ms']-4*D)):>6} | "
              f"{corr[0]:>5} {corr[1]:>5} {corr[2]:>5} | {cpu[0]:>5} {cpu[1]:>5} {cpu[2]:>5} | {q[0]:>4} {q[1]:>4} {q[2]:>4} {q[3]:>4} | {last:>5} | {med(g(lambda r:r['ttft_start_ms'])):>6} | {crit:>5}")
# engine step times from the server trace, split into runs by >5 s idle gaps
tr=f"{OUT}/server_trace.jsonl"
if os.path.exists(tr):
    ev=[json.loads(l) for l in open(tr)]
    steps=[e for e in ev if e["ev"]=="step" and any(c[1]-c[0]>0 for c in e["computed"].values())]
    runs=[]; cur=[]
    for s in steps:
        if cur and s["t0"]-cur[-1]["t0"]>5: runs.append(cur); cur=[]
        cur.append(s)
    if cur: runs.append(cur)
    print("\nengine prefill steps per run (split by >5 s gaps): n, median ms, p90 ms, median tokens/step")
    for i,r in enumerate(runs):
        ms=sorted(s["ms"] for s in r); tok=[sum(c[1]-c[0] for c in s["computed"].values()) for s in r]
        print(f"  run{i:2d} n={len(r):4d} med={st.median(ms):5.1f} p90={ms[int(.9*len(ms))]:5.1f} tok/step={st.median(tok):5.0f}")
