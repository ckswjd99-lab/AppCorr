"""Status table for the Gemma 3 sweeps: rows = datasets, columns = arms.

A finished arm shows its final accuracy, a running arm shows its interim accuracy with progress,
and an arm that has not started is left blank. Reading the interim number off the log matters:
without it a running arm is indistinguishable from a dead one.
"""
import json, os, re, sys

ROOT = "/NHNHOME/share/cjpark/AppCorr-gemma3/analysis/results"
DATASETS = ["chartqa", "textvqa", "infovqa", "pope", "realworldqa"]
ARMS = ["ceiling", "floor", "corrected", "interleaved_g4",
        "corrected_split", "corrected_patchled", "interleaved_g2", "interleaved_g8"]
HEAD = {"ceiling": "ceiling", "floor": "floor", "corrected": "corrected",
        "interleaved_g4": "intlv_g4", "corrected_split": "var_split",
        "corrected_patchled": "var_patch",
        "interleaved_g2": "intlv_g2", "interleaved_g8": "intlv_g8"}
PROG = re.compile(r"\[(\d+)/(\d+)\].*?acc=([\d.]+)%")


def cell(ds, arm):
    d = f"{ROOT}/gemma3_{ds}"
    j, l = f"{d}/{arm}.json", f"{d}/{arm}.log"
    if os.path.exists(j) and os.path.getsize(j) > 0:
        try:
            return f"{json.load(open(j))['summary']['accuracy']:.4f}"
        except Exception:
            pass
    if os.path.exists(l):
        m = None
        for line in open(l, errors="ignore"):
            g = PROG.search(line)
            if g:
                m = g
        if m:
            return f"~{float(m.group(3))/100:.4f} {int(m.group(1))*100//int(m.group(2))}%"
        return "started"
    return ""


rows = [(ds, [cell(ds, a) for a in ARMS]) for ds in DATASETS]
w = [max(len(HEAD[a]), max((len(r[1][i]) for r in rows), default=0), 8) for i, a in enumerate(ARMS)]
print("  " + "dataset".ljust(13) + "".join(HEAD[a].rjust(w[i] + 2) for i, a in enumerate(ARMS)))
for ds, vals in rows:
    print("  " + ds.ljust(13) + "".join((v if v else "-").rjust(w[i] + 2) for i, v in enumerate(vals)))
print("\n  ~ = in progress (interim accuracy, % of the set seen);  - = not run")
