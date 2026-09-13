"""Re-score stored row files from their `pred` with the CURRENT clean_text + dataset scorer.

The rows keep the raw generation (`pred`, already sentinel-stripped for GLM), so a scorer or
normalisation fix (2026-09-13: the "The answer is X." prefix that broke ANLS / exact-match on
GLM-4.6V) is applied here without a GPU. Originals are copied to `<dir>/_prescore_<date>/`
before the file is rewritten in place; every other field is preserved. Box datasets (refcoco,
visdrone_det) store the RESCALED box in `pred`, so no rescale is redone here.

  python analysis/experiments/rescore_rows.py --family glm46v --dataset infovqa <files...>
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))
sys.path.insert(0, os.path.join(ROOT, "analysis", "experiments"))

from qwen_vl_prefill.datasets_eval import get_spec          # noqa: E402
from qwen_vllm_accuracy import clean_text                   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("files", nargs="+")
    a = ap.parse_args()
    spec = get_spec(a.dataset)
    stamp = time.strftime("%Y%m%d")
    for f in a.files:
        rows = [json.loads(l) for l in open(f) if l.strip()]
        old_ok = sum(int(r.get("ok", 0)) for r in rows if "skip" not in r)
        old_val = [float(r["val"]) for r in rows if "skip" not in r and r.get("val") is not None]
        changed = 0
        for r in rows:
            if "skip" in r:
                continue
            pred = clean_text(a.family, str(r["pred"]))
            try:
                ok, val = spec.score(pred, r["gold"])
            except NotImplementedError:
                ok, val = 0, None
            new = (int(ok), float(val) if val is not None else None)
            if (int(r.get("ok", 0)), r.get("val")) != new:
                changed += 1
            r["ok"], r["val"] = new
        new_ok = sum(int(r["ok"]) for r in rows if "skip" not in r)
        new_val = [float(r["val"]) for r in rows if "skip" not in r and r.get("val") is not None]
        n = max(1, len([r for r in rows if "skip" not in r]))
        mv = lambda v: 100 * sum(v) / max(1, len(v))
        print(f"{os.path.basename(f):64s} n={n} ok {100*old_ok/n:.2f} -> {100*new_ok/n:.2f}  "
              f"val {mv(old_val):.2f} -> {mv(new_val):.2f}  rows changed {changed}", flush=True)
        if a.dry_run or changed == 0:
            continue
        park = os.path.join(os.path.dirname(f), f"_prescore_{stamp}")
        os.makedirs(park, exist_ok=True)
        if not os.path.exists(os.path.join(park, os.path.basename(f))):
            shutil.copy2(f, os.path.join(park, os.path.basename(f)))
        tmp = f + ".rescore.tmp"
        with open(tmp, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        os.replace(tmp, f)


if __name__ == "__main__":
    main()
