"""
aggregate_gpu_shards.py

Merges the --results-json outputs from N parallel GPU-sharded runs of
progressive_vla_libero_eval.py (see run_gpu_shard.sh) into one per-suite,
per-mode success-rate + patch-usage table, and an overall average across suites.

Run:
    python analysis/experiments/aggregate_gpu_shards.py \
        /path/to/results_gpu0 /path/to/results_gpu1 [...]
"""

import glob
import json
import sys
from pathlib import Path

import numpy as np


def main():
    shard_dirs = sys.argv[1:]
    if not shard_dirs:
        print("usage: aggregate_gpu_shards.py <shard_dir> [<shard_dir> ...]")
        sys.exit(1)

    per_suite = {}  # suite -> mode -> {"results": [...], "patch_counts": [...]}
    for shard_dir in shard_dirs:
        for path in sorted(glob.glob(str(Path(shard_dir) / "*.json"))):
            payload = json.load(open(path))
            suite = payload["suite"]
            per_suite.setdefault(suite, {})
            for mode, data in payload["modes"].items():
                bucket = per_suite[suite].setdefault(mode, {"results": [], "patch_counts": []})
                bucket["results"].extend(data["results"])
                bucket["patch_counts"].extend(data["patch_counts"])

    suites = sorted(per_suite.keys())
    modes = sorted({m for s in per_suite.values() for m in s})

    print(f"{'suite':16s} " + " ".join(f"{m:>22s}" for m in modes))
    overall = {m: [] for m in modes}
    for suite in suites:
        row = [f"{suite:16s}"]
        for mode in modes:
            bucket = per_suite[suite].get(mode)
            if not bucket:
                row.append(f"{'--':>22s}")
                continue
            results = bucket["results"]  # [(task_id, trial_idx, success), ...]
            successes = [s for _, _, s in results]
            overall[mode].extend(successes)
            rate = np.mean(successes) if successes else float("nan")
            patches = bucket["patch_counts"]
            patch_str = f" @{np.mean(patches) / 256:.0%}p" if patches else ""
            row.append(f"{sum(successes):3d}/{len(successes):<3d}({rate:.0%}){patch_str}".rjust(22))
        print(" ".join(row))

    print("\n=== Overall (averaged across suites) ===")
    for mode in modes:
        vals = overall[mode]
        if not vals:
            continue
        print(f"    {mode:16s}  {sum(vals)}/{len(vals)}  ({np.mean(vals):.1%})")

    print("\n=== Per-task breakdown (per suite/mode) ===")
    for suite in suites:
        for mode in modes:
            bucket = per_suite[suite].get(mode)
            if not bucket:
                continue
            per_task = {}
            for task_id, _, success in bucket["results"]:
                per_task.setdefault(task_id, []).append(success)
            line = "  ".join(f"t{tid}={sum(v)}/{len(v)}" for tid, v in sorted(per_task.items()))
            print(f"    {suite:16s} {mode:16s}  {line}")


if __name__ == "__main__":
    main()
