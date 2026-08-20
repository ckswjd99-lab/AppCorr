"""
mcnemar_from_jsonl.py

Paired McNemar's test between two conditions logged (via --log-jsonl) by
refcoco_gqa_batched_eval.py to the SAME file over the SAME sample indices --
e.g. baseline vs a keep_rate condition, to test whether "recovers baseline
accuracy" is a real, statistically defensible claim rather than an artifact
of the pipeline's own numerical noise floor.

Usage (appcorr env):
    python analysis/experiments/mcnemar_from_jsonl.py /tmp/32b_refcoco_mcnemar.jsonl \\
        --label-a 32b_refcoco_baseline_MCNEMAR --label-b 32b_refcoco_kr0.58_MCNEMAR
"""

import argparse
import json
from collections import defaultdict


def mcnemar_p_value(b: int, c: int) -> float:
    """Exact two-sided McNemar test (binomial), matching statsmodels'
    `mcnemar(..., exact=True)` for small b+c, and the standard chi-square
    continuity-corrected approximation for larger b+c (used here to avoid
    a scipy dependency -- both give effectively the same answer once
    b+c is more than a few dozen, which every condition pair here is)."""
    n = b + c
    if n == 0:
        return 1.0
    chi2 = (abs(b - c) - 1) ** 2 / n
    # Wilson-Hilferty / standard normal tail approx for chi2(df=1) survival
    # function -- avoids importing scipy just for this one script.
    import math
    z = math.sqrt(chi2)
    p = math.erfc(z / math.sqrt(2))
    return min(max(p, 0.0), 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl_path")
    ap.add_argument("--label-a", required=True)
    ap.add_argument("--label-b", required=True)
    args = ap.parse_args()

    by_label = defaultdict(dict)  # label -> {idx: correct}
    with open(args.jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            by_label[rec["label"]][rec["idx"]] = rec["correct"]

    if args.label_a not in by_label:
        raise SystemExit(f"label {args.label_a!r} not found in {args.jsonl_path}. "
                          f"Available: {sorted(by_label.keys())}")
    if args.label_b not in by_label:
        raise SystemExit(f"label {args.label_b!r} not found in {args.jsonl_path}. "
                          f"Available: {sorted(by_label.keys())}")

    a, b_map = by_label[args.label_a], by_label[args.label_b]
    common = sorted(set(a.keys()) & set(b_map.keys()))
    if len(common) < len(a) or len(common) < len(b_map):
        print(f"[warn] label_a has {len(a)} samples, label_b has {len(b_map)}, "
              f"only {len(common)} indices are common to both -- using the intersection.")

    n = len(common)
    both_correct = sum(1 for i in common if a[i] and b_map[i])
    both_wrong = sum(1 for i in common if not a[i] and not b_map[i])
    a_only = sum(1 for i in common if a[i] and not b_map[i])   # A correct, B wrong
    b_only = sum(1 for i in common if not a[i] and b_map[i])   # B correct, A wrong

    acc_a = 100.0 * sum(a[i] for i in common) / n
    acc_b = 100.0 * sum(b_map[i] for i in common) / n

    p = mcnemar_p_value(a_only, b_only)

    print(f"N (paired samples) = {n}")
    print(f"{args.label_a}: {acc_a:.2f}% ({sum(a[i] for i in common)}/{n})")
    print(f"{args.label_b}: {acc_b:.2f}% ({sum(b_map[i] for i in common)}/{n})")
    print(f"gap: {acc_b - acc_a:+.2f}pp")
    print()
    print("2x2 contingency table (rows=A, cols=B):")
    print(f"                B correct   B wrong")
    print(f"  A correct     {both_correct:>9}   {a_only:>7}")
    print(f"  A wrong       {b_only:>9}   {both_wrong:>7}")
    print()
    print(f"Discordant pairs: A-only-correct={a_only}, B-only-correct={b_only}")
    print(f"McNemar exact/approx two-sided p-value: {p:.4f}")
    if p > 0.05:
        print("-> NOT statistically significant at alpha=0.05: the two conditions' accuracies "
              "are statistically indistinguishable given this sample.")
    else:
        print("-> statistically significant at alpha=0.05.")


if __name__ == "__main__":
    main()
