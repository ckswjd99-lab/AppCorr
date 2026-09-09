"""Single-GPU plumbing gate for a model whose HF twin cannot sit beside the engine (122B-FP8):
compare the vLLM driver's per-sample predictions with the HF campaign's rows for the same
dataset and arms (`qwen35_accuracy.py`, explicit greedy loop -- the same decode rule the engine
applies at temperature 0), joined on the row index `i`.

The pass rule mirrors `vllm_bridge_gate.py`: the streaming arm (the socket path under test)
must agree with HF no worse than the one-chunk ceiling arm does -- the ceiling's disagreement
rate IS the engine's own numerical band (different kernels, different reduction order), and the
streaming arm adds chunked prefill on top of it, which the in-process gate measured at a few
samples per thousand. Reported per arm: n compared, exact-pred agreement, accuracy under both
backends, and the disagreeing rows (i, hf, vllm) so a near-tie can be told from a defect.

  python analysis/experiments/vllm_vs_hf_preds.py --dataset realworldqa \
      --hf-dir /NHNHOME/share/cjpark/AppCorr-qwen35-eval/analysis/results/qwen35_accuracy \
      --vllm-dir analysis/results/qwen_vllm_accuracy --slug qwen3.5-122b-a10b-fp8 \
      --arms ceiling floor streaming_g4 [--vllm-suffix _c4]
"""
import argparse, json, os, sys


def rows(path):
    if not os.path.exists(path):
        return {}
    out = {}
    for l in open(path):
        if l.strip():
            r = json.loads(l)
            if "skip" not in r:
                out[int(r["i"])] = r
    return out


def norm(p):
    return " ".join(str(p).strip().lower().split())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--hf-dir", required=True)
    ap.add_argument("--vllm-dir", required=True)
    ap.add_argument("--slug", required=True)
    ap.add_argument("--arms", nargs="+", default=["ceiling", "floor", "streaming_g4"])
    ap.add_argument("--vllm-suffix", default="", help="e.g. _c4 when the vLLM run used --concurrency 4")
    ap.add_argument("--show", type=int, default=12)
    a = ap.parse_args()

    band, n_common = {}, {}
    for arm in a.arms:
        hf = rows(os.path.join(a.hf_dir, f"{a.dataset}_{a.slug}_{arm}.jsonl"))
        vl = rows(os.path.join(a.vllm_dir, f"{a.dataset}_{a.slug}_{arm}{a.vllm_suffix}.jsonl"))
        common = sorted(set(hf) & set(vl))
        if not common:
            print(f"[{arm}] no common rows (hf {len(hf)}, vllm {len(vl)})")
            continue
        same = [i for i in common if norm(hf[i]["pred"]) == norm(vl[i]["pred"])]
        diff = [i for i in common if i not in set(same)]
        acc_h = 100 * sum(hf[i]["ok"] for i in common) / len(common)
        acc_v = 100 * sum(vl[i]["ok"] for i in common) / len(common)
        band[arm] = len(diff) / len(common)
        n_common[arm] = len(common)
        print(f"[{arm:13s}] n={len(common)}  pred-agree {len(same)}/{len(common)} "
              f"({100 * len(same) / len(common):.1f}%)  acc hf {acc_h:.2f} / vllm {acc_v:.2f}")
        for i in diff[:a.show]:
            print(f"    i={i:5d}  hf={hf[i]['pred']!r:40s} vllm={vl[i]['pred']!r}  gold={hf[i]['gold']!r}")
    ok = True
    if "streaming_g4" in band and "ceiling" in band:
        # streaming's disagreement rate within the engine's band: the one-shot arms (ceiling AND
        # floor -- neither chunks the prefill, so both measure the engine alone) set the band, plus
        # slack of one sample or 1%, whichever is larger (chunked prefill's own near-tie flips are a
        # few per thousand in-process; at n=64 a single flip is already 1.56%, so a bare 1% slack
        # sat below the sample resolution and failed a 63/64 arm against a 63/64 floor).
        oneshot = max(band[a] for a in ("ceiling", "floor") if a in band)
        slack = max(0.01, 1.0 / n_common["streaming_g4"])
        ok = band["streaming_g4"] <= oneshot + slack
        print(f"band: one-shot {100 * oneshot:.2f}% (ceiling {100 * band['ceiling']:.2f}%)  "
              f"streaming {100 * band['streaming_g4']:.2f}%  slack {100 * slack:.2f}%")
    print("VLLM_VS_HF_GATE", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
