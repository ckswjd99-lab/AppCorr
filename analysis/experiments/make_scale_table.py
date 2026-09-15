#!/usr/bin/env python3
"""Resolution-scaling table: every image forced onto one token budget T (--target-tokens), so the
only variable across rungs is sequence length. Rows = (dataset, T) with T=native as the reference
rung; columns = the four arms of the cell (floor / ceiling / adaptive k~.50 / k~.25), preservation
vs the ceiling on the COMMON rows of the cell, realised k, and the closed-form Crit. Comp share.
Reads qwen_vllm_accuracy_scale{,_adaptive}/ for the ladder and the campaign dirs for native."""
import glob, json, os, re, statistics, sys
sys.path.insert(0, os.path.dirname(__file__))
from flops_analytic import DECODERS, VISIONS
R = os.path.join(os.path.dirname(__file__), "..", "results")
MODELS = [("qwen3.5-35b-a3b", "qwen35_35b", "Qwen3.5-35B"), ("qwen3.5-122b-a10b-fp8", "qwen35_122b", "Qwen3.5-122B"),
          ("glm-4.6v-fp8", "glm46v", "GLM-4.6V"), ("glm-5.3-flash", "glm53", "GLM-5.3")]
DS = [("vstar", "V*Bench", "pyr"), ("infovqa", "InfoVQA", "pyr"), ("realworldqa", "RealWorldQA", "box"),
      ("textvqa", "TextVQA", "pyr")]
LADDER = [None, 2048, 4096, 6144]

def rows_of(p):
    out = {}
    for l in open(p):
        if l.strip():
            r = json.loads(l)
            if r.get("val") is not None: out[r["i"]] = r
    return out

def find(ds, slug, arm, T, filt):
    if T is None:
        d = {"pyr": "qwen_vllm_accuracy_il_pyr", "box": "qwen_vllm_accuracy_il"}[filt]
        d = d + ("_adaptive" if arm == "auto" else "")
        pat = f"{ds}_{slug}_interleaved_unified_g4_auto*.jsonl" if arm == "auto" else f"{ds}_{slug}_{arm}*.jsonl"
        fs = [f for f in glob.glob(os.path.join(R, d, pat)) if "_lat" not in f and "_t" not in os.path.basename(f).split("_g4")[-1]]
    else:
        d = "qwen_vllm_accuracy_scale" + ("_adaptive" if arm == "auto" else "")
        pat = f"{ds}_{slug}_interleaved_unified_g4_auto*_t{T}_c*.jsonl" if arm == "auto" else f"{ds}_{slug}_{arm}_t{T}_c*.jsonl"
        fs = glob.glob(os.path.join(R, d, pat))
    if arm == "auto":
        rx = re.compile(r"_auto([0-9.eE+-]+?)(b\d+)?(?:_t\d+)?(?:_c\d+)?\.jsonl$")
        fs = [f for f in fs if rx.search(os.path.basename(f)) and not rx.search(os.path.basename(f)).group(2)]  # drop bucket-4 variants
        fs = sorted(fs, key=lambda f: float(rx.search(os.path.basename(f)).group(1)))
        return [(f, float(rx.search(os.path.basename(f)).group(1))) for f in fs[:2]] if len(fs) >= 2 else []
    return fs[:1]

def critcomp(rows, ks, dec, vision):
    v = []
    for i in ks:
        r = rows[i]
        if not isinstance(r.get("chunks"), list): continue
        lo, n_img = r["image_run"]; ch = [tuple(x) for x in r["chunks"]]; pt = int(r["prompt_tokens"])
        try:
            c = dec.interleaved_cost(pt, int(lo), int(n_img), ch); u = vision.unified_cost(ch)
            full = vision.tower_flops(u["n_rows"]) + dec.prefill_flops(pt - 1)
        except Exception: continue
        if full > 0: v.append(100 * (u["crit"] + c["crit"]) / full)
    return statistics.mean(v) if v else None

def main():
    L = [r"\begin{table*}[t]", r"\centering", r"\small",
         r"\caption{Resolution scaling: every image is resized (aspect preserved) to $T$ merged vision tokens before "
         r"any other step, so a rung differs from its neighbours in sequence length only; ``native'' is the campaign "
         r"row at the dataset's own resolution. Acc.\ in \%, Ours with (preservation vs.\ the ceiling on the cell's "
         r"common rows) and the realised $\bar k$; Crit.\ Comp.: last-round FLOPs as a share of the full-resolution "
         r"pass (closed form). $n$ = common rows.}",
         r"\label{tab:scaling}", r"\begin{tabular}{llrrrrrrrrrrrrrr}", r"\toprule",
         r"\multirow{2}{*}{Dataset} & \multirow{2}{*}{$T$} & \multirow{2}{*}{tok} & \multirow{2}{*}{$n$} & "
         r"\multirow{2}{*}{Low-res.} & \multirow{2}{*}{Full-res.} & "
         r"\multicolumn{4}{c}{Ours $k{=}.50$} & \multicolumn{4}{c}{Ours $k{=}.25$} & "
         r"\multirow{2}{*}{CC$_{.50}$} & \multirow{2}{*}{CC$_{.25}$} \\",
         r"\cmidrule(lr){7-10} \cmidrule(lr){11-14}",
         r" & & & & & & Acc. & Pres. & $\bar k$ & $\theta$ & Acc. & Pres. & $\bar k$ & $\theta$ & & \\"]
    for slug, mk, mname in MODELS:
        fam = next((f for f, reg in DECODERS.items() if mk in reg), None)
        dec, vision = (DECODERS[fam][mk], VISIONS[fam]) if fam else (None, None)
        block = []
        for ds, dname, filt in DS:
            for T in LADDER:
                fc, ff, fa = find(ds, slug, "ceiling", T, filt), find(ds, slug, "floor", T, filt), find(ds, slug, "auto", T, filt)
                Tlab = 'native' if T is None else T
                if not (fc and ff and fa):
                    # the rung is not measured yet: keep the row so the ladder's shape is visible
                    block.append(f"{dname} & {Tlab} & " + " & ".join(["--"] * 14) + r" \\")
                    continue
                C, F = rows_of(fc[0]), rows_of(ff[0]); A50, A25 = rows_of(fa[0][0]), rows_of(fa[1][0])
                th50, th25 = fa[0][1], fa[1][1]
                ks = sorted(set(C) & set(F) & set(A50) & set(A25))
                if len(ks) < 30:
                    block.append(f"{dname} & {Tlab} & " + " & ".join(["--"] * 14) + r" \\")
                    continue
                acc = lambda d: 100 * sum(float(d[i]["val"]) for i in ks) / len(ks)
                kbar = lambda d: statistics.mean(d[i]["keep_realised"] for i in ks if isinstance(d[i].get("keep_realised"), (int, float)))
                tok = statistics.median(C[i]["prompt_tokens"] for i in ks if C[i].get("prompt_tokens"))
                c, f, a5, a2 = acc(C), acc(F), acc(A50), acc(A25)
                cc5 = critcomp(A50, ks, dec, vision) if dec else None; cc2 = critcomp(A25, ks, dec, vision) if dec else None
                fmt = lambda x: f"{x:.1f}\\%" if x is not None else "--"
                block.append(f"{dname} & {Tlab} & {tok:,.0f} & {len(ks)} & {f:.2f} & {c:.2f} & "
                             f"{a5:.2f} & {100*a5/c:.1f}\\% & {kbar(A50):.2f} & {th50:.4f} & "
                             f"{a2:.2f} & {100*a2/c:.1f}\\% & {kbar(A25):.2f} & {th25:.4f} & {fmt(cc5)} & {fmt(cc2)} \\\\")
        if block:
            L += [r"\midrule", rf"\multicolumn{{10}}{{l}}{{\textbf{{{mname}}}}} \\"] + block
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    print("\n".join(L))

if __name__ == "__main__":
    main()
