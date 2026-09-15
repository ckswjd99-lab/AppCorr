#!/usr/bin/env python3
r"""Resolution-scaling table. Every image is resized (aspect preserved) to T merged vision tokens
before any other step, so a rung differs from its neighbours in sequence length only; the native
rung is the campaign row at the dataset's own resolution.

Layout: model block (heavy rule) -> dataset spanning its rungs with \multirow (light rule between
datasets) -> one row per T in {native, 2048, 4096, 6144}. Under each adaptive arm (k~.50, k~.25):
Acc. | Pres. (vs the ceiling on the cell's common rows) | k-bar (realised) | theta (the threshold
the arm ran at, recalibrated per (dataset, T)) | Comp. (total FLOPs, share of the full-res pass) |
Crit. Comp. (last-round share) | Crit. Lat. (TTFT share of the full-res TTFT, from the latency
probe; only the native rung has been probed so far). Unmeasured cells print as --.
Needs \usepackage{multirow,booktabs}."""
import glob, json, os, re, statistics, sys
sys.path.insert(0, os.path.dirname(__file__))
from flops_analytic import DECODERS, VISIONS

R = os.path.join(os.path.dirname(__file__), "..", "results")
MODELS = [  # (row-file slug, flops_analytic key, latency adaptive key, latency streaming key for `full`, display)
    ("qwen3.5-35b-a3b", "qwen35_35b", "qwen35_35b_ilu_adaptive", "qwen35_moe_cg1024", "Qwen3.5-35B"),
    ("qwen3.5-122b-a10b-fp8", "qwen35_122b", "qwen35_122b_ilu_adaptive", "qwen35_122b_cg1024", "Qwen3.5-122B"),
    ("glm-4.6v-fp8", "glm46v", "glm46v_ilu_adaptive", None, "GLM-4.6V"),
    ("glm-5.3-flash", "glm53", "glm53_ilu_adaptive_tp2graph_fixed", None, "GLM-5.3")]
DS = [("vstar", "V*Bench", "pyr"), ("infovqa", "InfoVQA", "pyr"), ("realworldqa", "RealWorldQA", "box"),
      ("textvqa", "TextVQA", "pyr")]
LADDER = [None, 2048, 4096, 6144]
NSUB = 7                       # sub-columns per Ours arm
NCOL = 6 + 2 * NSUB            # 20
AUTO_RX = re.compile(r"_auto([0-9.eE+-]+?)(b\d+)?(?:_t\d+)?(?:_c\d+)?\.jsonl$")
LAT = json.load(open(os.path.join(R, "latency", "inprocess_latency.json")))

def rows_of(p):
    out = {}
    for l in open(p):
        if l.strip():
            r = json.loads(l)
            if r.get("val") is not None: out[r["i"]] = r
    return out

def find(ds, slug, arm, T, filt):
    """Row files of one arm of one cell. `auto` returns [(path, theta)] sorted by theta (2 files)."""
    if T is None:
        d = {"pyr": "qwen_vllm_accuracy_il_pyr", "box": "qwen_vllm_accuracy_il"}[filt] + ("_adaptive" if arm == "auto" else "")
        pat = f"{ds}_{slug}_interleaved_unified_g4_auto*.jsonl" if arm == "auto" else f"{ds}_{slug}_{arm}*.jsonl"
        fs = [f for f in glob.glob(os.path.join(R, d, pat)) if "_lat" not in os.path.basename(f) and "_t" not in os.path.basename(f).split("_g4")[-1]]
    else:
        d = "qwen_vllm_accuracy_scale" + ("_adaptive" if arm == "auto" else "")
        # the concurrency suffix is only written when the driver runs c>1, and the two boxes run
        # different concurrencies per model (35B c4 here, GLM-5.3 c1 on the peer), so `_cN` is
        # optional -- but nothing else may follow `_tN`, or a `_t20480` would answer for `_t2048`.
        pat = f"{ds}_{slug}_interleaved_unified_g4_auto*_t{T}*.jsonl" if arm == "auto" else f"{ds}_{slug}_{arm}_t{T}*.jsonl"
        tail = re.compile(rf"_t{T}(_c\d+)?\.jsonl$")
        fs = [f for f in glob.glob(os.path.join(R, d, pat)) if tail.search(os.path.basename(f))]
    if arm != "auto":
        return fs[:1]
    fs = [f for f in fs if AUTO_RX.search(os.path.basename(f)) and not AUTO_RX.search(os.path.basename(f)).group(2)]
    fs = sorted(fs, key=lambda f: float(AUTO_RX.search(os.path.basename(f)).group(1)))
    # One arm is enough to print half a rung: a running cell shows the target that finished
    # rather than a row of dashes. The caller assigns each file to its k slot by realised keep,
    # not by position, so a lone file lands in the right half.
    return [(f, float(AUTO_RX.search(os.path.basename(f)).group(1))) for f in fs[:2]]

def comp_shares(rows, ks, dec, vision):
    """(total, crit) FLOPs share of the full-resolution pass, closed form replayed per row."""
    tot, crit = [], []
    for i in ks:
        r = rows[i]
        if not isinstance(r.get("chunks"), list): continue
        lo, n_img = r["image_run"]; ch = [tuple(x) for x in r["chunks"]]; pt = int(r["prompt_tokens"])
        try:
            c = dec.interleaved_cost(pt, int(lo), int(n_img), ch); u = vision.unified_cost(ch)
            full = vision.tower_flops(u["n_rows"]) + dec.prefill_flops(pt - 1)
        except Exception: continue
        if full > 0:
            tot.append(100 * (u["total"] + c["total"]) / full); crit.append(100 * (u["crit"] + c["crit"]) / full)
    return (statistics.mean(tot) if tot else None, statistics.mean(crit) if crit else None)

# Sequence cap of the ladder's latency probe. The correct step submits one pseudo-sequence per
# recomputed row, so max-num-seqs bounds |P_r|; a band above it splits into two passes and the
# cell measures the cap rather than the schedule. Measured on the ladder's own `band_selected`,
# |P_r| cannot exceed the band, i.e. ~T/4 (1,564 at T=6144), so 2048 clears every rung. The key
# carries the cap so a cap-limited run can never be read as a clean one.
LADDER_MNS = 2048

def crit_lat_share(ad_key, st_key, ds, theta, T=None):
    """This arm's Crit. Lat. as a share of the full-res TTFT, from the probe.

    Native reads the campaign key; a ladder rung reads `<key>_mns<cap>_t<T>`, which is written by
    scale_critlat.sh and never mixed with the native key -- the two run different engines."""
    key = ad_key if T is None else f"{ad_key}_mns{LADDER_MNS}_t{T}"
    x = (LAT.get(key) or {}).get(ds) or {}
    if T is None:
        full = x.get("full") or ((LAT.get(st_key) or {}).get(ds) or {}).get("full") if st_key else x.get("full")
    else:
        full = x.get("full")          # the rung's own denominator, same engine, same session
    v = x.get(f"auto{theta:g}")
    return 100 * v / full if (v and full) else None

pct = lambda x: f"{x:.1f}\\%" if x is not None else "--"

def arm_cells(A, ks, theta, c, dec, vision, ad_key, st_key, ds, T):
    a = 100 * sum(float(A[i]["val"]) for i in ks) / len(ks)
    kb = statistics.mean(A[i]["keep_realised"] for i in ks if isinstance(A[i].get("keep_realised"), (int, float)))
    tot, crit = comp_shares(A, ks, dec, vision) if dec else (None, None)
    lat = crit_lat_share(ad_key, st_key, ds, theta, T)
    return f"{a:.2f} & {100 * a / c:.1f}\\% & {kb:.2f} & {theta:.4f} & {pct(tot)} & {pct(crit)} & {pct(lat)}"

def main():
    L = [r"\begin{table*}[t]", r"\centering",
         r"\caption{Resolution scaling: every image is resized (aspect preserved) to $T$ merged vision tokens before "
         r"any other step, so a rung differs from its neighbours in sequence length only; \emph{native} is the "
         r"campaign row at the dataset's own resolution. Under each adaptive arm: Acc.\ (\%); Pres.: preservation "
         r"vs.\ the ceiling on the cell's common rows; $\bar k$: realised mean keep; $\theta$: the per-band pscore "
         r"threshold the arm ran at, recalibrated per (dataset, $T$); Comp.\ / Crit.\ Comp.: total / last-round FLOPs "
         r"as a share of the full-resolution pass (closed form); Crit.\ Lat.: TTFT from the last band's arrival as a "
         r"share of the full-resolution TTFT, each rung probed against its own full-resolution pass in the same session). $n$ = common rows; -- = not measured.}",
         r"\label{tab:scaling}", r"\resizebox{\textwidth}{!}{%", r"\setlength{\tabcolsep}{3pt}",
         r"\begin{tabular}{ll" + "r" * (NCOL - 2) + "}", r"\toprule",
         r"\multirow{2}{*}{Dataset} & \multirow{2}{*}{$T$} & \multirow{2}{*}{tok} & \multirow{2}{*}{$n$} & "
         r"\multirow{2}{*}{Low-res.} & \multirow{2}{*}{Full-res.} & "
         rf"\multicolumn{{{NSUB}}}{{c}}{{Ours $k{{=}}.50$}} & \multicolumn{{{NSUB}}}{{c}}{{Ours $k{{=}}.25$}} \\",
         rf"\cmidrule(lr){{7-{6 + NSUB}}} \cmidrule(lr){{{7 + NSUB}-{NCOL}}}",
         r" & & & & & & " + " & ".join([r"Acc. & Pres. & $\bar k$ & $\theta$ & Comp. & Crit.\ Comp. & Crit.\ Lat."] * 2) + r" \\"]
    for slug, mk, ad_key, st_key, mname in MODELS:
        fam = next((f for f, reg in DECODERS.items() if mk in reg), None)
        dec, vision = (DECODERS[fam][mk], VISIONS[fam]) if fam else (None, None)
        L += [r"\specialrule{1.2pt}{2pt}{2pt}", rf"\multicolumn{{{NCOL}}}{{l}}{{\textbf{{{mname}}}}} \\"]
        for di, (ds, dname, filt) in enumerate(DS):
            if di: L.append(r"\specialrule{0.4pt}{0pt}{0pt}")
            for ri, T in enumerate(LADDER):
                lead = rf"\multirow{{{len(LADDER)}}}{{*}}{{{dname}}} & " if ri == 0 else " & "
                Tlab = "native" if T is None else str(T)
                blank = NSUB * ["--"]
                fc, ff, fa = find(ds, slug, "ceiling", T, filt), find(ds, slug, "floor", T, filt), find(ds, slug, "auto", T, filt)
                if not (fc and ff):
                    L.append(lead + f"{Tlab} & " + " & ".join(["--"] * (NCOL - 2)) + r" \\"); continue
                C, F = rows_of(fc[0]), rows_of(ff[0])
                # A STILL-RUNNING arm must not join the common subset: it would shrink the rung to
                # its own prefix and silently rebase Low-res./Full-res. on a biased slice (the
                # TextVQA T=2048 ceiling sat at 2,750 of 5,000 for a day and read as complete).
                full_n = max(len(C), len(F))
                slot = {}
                for path, th in fa:
                    A = rows_of(path)
                    if len(A) < 0.95 * full_n:
                        continue
                    kb = statistics.mean(A[i]["keep_realised"] for i in A
                                         if isinstance(A[i].get("keep_realised"), (int, float)))
                    slot["k25" if kb < 0.375 else "k50"] = (A, th)
                ks = sorted(set(C) & set(F) & set.intersection(
                    *[set(v[0]) for v in slot.values()]) if slot else set(C) & set(F))
                if len(ks) < 30:
                    L.append(lead + f"{Tlab} & " + " & ".join(["--"] * (NCOL - 2)) + r" \\"); continue
                acc = lambda d: 100 * sum(float(d[i]["val"]) for i in ks) / len(ks)
                tok = statistics.median(C[i]["prompt_tokens"] for i in ks if C[i].get("prompt_tokens"))
                c, f = acc(C), acc(F)
                cells = []
                for name in ("k50", "k25"):
                    if name in slot:
                        A, th = slot[name]
                        cells.append(arm_cells(A, ks, th, c, dec, vision, ad_key, st_key, ds, T))
                    else:
                        cells.append(" & ".join(blank))
                L.append(lead + f"{Tlab} & {tok:,.0f} & {len(ks)} & {f:.2f} & {c:.2f} & "
                         + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    print("\n".join(L))

if __name__ == "__main__":
    main()
