"""Gate: the unified axis's post-crossing `correct_rows` shortcut is bitwise the full-stream form.

`QwenVLStreamingAxis.unified_rows_after_crossing` switches the rounds after the projector
crossing from `tower.correct_forward` (rebuild [T, D] + rule-3 write-back) to
`tower.correct_rows` (corrected rows only). The claim: every message the sink receives -- the
opening push and every `correct` (positions, embeds, window, stage) -- and every `chunks`
record is identical with the flag on and off, at keep=1 and keep<1 (progressive selection).

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD python analysis/experiments/vllm_unified_hybrid_gate.py \\
        --model Qwen/Qwen3.5-35B-A3B --datasets vstar realworldqa --keeps 1.0 0.5 0.25
"""
import argparse, json, os, sys, time
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from analysis.experiments.qwen35_accuracy import degrade
from analysis.experiments.qwen_vllm_accuracy import make_axis
from analysis.experiments.vllm_interleaved_axis_gate import run


def msgs_equal(a, b):
    if len(a) != len(b):
        return False, f"message count {len(a)} vs {len(b)}"
    for i, (x, y) in enumerate(zip(a, b)):
        for k in ("kind", "final", "window", "stage", "correct_from", "correct_to"):
            if x.get(k) != y.get(k):
                return False, f"msg {i} {k}: {x.get(k)} vs {y.get(k)}"
        if not torch.equal(x["positions"], y["positions"]):
            return False, f"msg {i} positions differ"
        if not torch.equal(x["embeds"].view(torch.int16), y["embeds"].view(torch.int16)):
            d = (x["embeds"].float() - y["embeds"].float()).abs()
            return False, f"msg {i} embeds differ: max|d|={d.max():.3e} rows={int((d.max(1).values>0).sum())}"
    return True, ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--keeps", type=float, nargs="+", default=[1.0, 0.5, 0.25])
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--degrade-filter", default="pyr")
    ap.add_argument("--datasets", nargs="+", default=["vstar", "realworldqa"])
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--out", default="analysis/results/vllm_stream/unified_hybrid_gate.json")
    a = ap.parse_args()

    from transformers import AutoProcessor
    from appcorr.models.vision_only import load_vision_only
    from qwen_vl_prefill.datasets_eval import get_spec
    from datasets import load_dataset
    proc = AutoProcessor.from_pretrained(a.model)
    model = load_vision_only(a.model, device="cuda:0")
    axis = make_axis("qwen35", model, proc)

    rows, ok_all = [], True
    for ds_name in a.datasets:
        spec = get_spec(ds_name)
        ds = spec.load(load_dataset)
        img, q, _ = spec.prepare(ds[a.index], lambda h, w, **kw: (h, w), 1, 1, 1 << 30)
        img = img.convert("RGB")
        base = degrade(img, a.level, a.degrade_filter)
        inputs = axis.build_inputs(img, q).to("cuda:0")
        px_base = axis.build_inputs(base, q)["pixel_values"].to("cuda:0")
        for keep in a.keeps:
            out = {}
            for flag in (False, True):
                axis.unified_rows_after_crossing = flag
                torch.cuda.synchronize(); t0 = time.perf_counter()
                sink, st = run(axis, inputs, px_base, a.groups, keep, "unified_staged")
                torch.cuda.synchronize()
                out[flag] = (sink.msgs, st["chunks"], time.perf_counter() - t0)
            ok, why = msgs_equal(out[False][0], out[True][0])
            ch_ok = [tuple(c) for c in out[False][1]] == [tuple(c) for c in out[True][1]]
            n_rows_calls = sum(1 for c in out[True][1] if c[0] == "vcorrect")
            row = {"dataset": ds_name, "keep": keep, "prompt_tokens": int(inputs["input_ids"].shape[1]),
                   "n_msgs": len(out[True][0]), "msgs_bitwise": ok, "why": why,
                   "chunks_equal": ch_ok, "n_vcorrect": n_rows_calls,
                   "bounds": st.get("unified_bounds"),
                   "t_full_stream_s": round(out[False][2], 3), "t_rows_s": round(out[True][2], 3)}
            rows.append(row); ok_all &= ok and ch_ok
            print(f"{ds_name:<12} keep={keep:.2f} N={row['prompt_tokens']} msgs={row['n_msgs']} "
                  f"bitwise={ok} chunks={ch_ok} bounds={row['bounds']} "
                  f"t {row['t_full_stream_s']}s -> {row['t_rows_s']}s {why}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"_model": a.model, "rows": rows, "PASS": ok_all}, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}: {'HYBRID_GATE_PASS' if ok_all else 'HYBRID_GATE_FAIL'}")


if __name__ == "__main__":
    main()
