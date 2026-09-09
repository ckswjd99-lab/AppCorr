"""Bitwise snapshot of everything the vLLM server consumes from the HF-side vision path, per
sample, for a refactor-identity gate: sha256 of every chunk the streaming arm pushes (bf16
embeds bit pattern, int64 M-RoPE positions, rope delta, final flag), of the one-shot ceiling
/ floor embeddings, plus the chunk boundaries and corrected-group counts. Run once on the code
before a lever (`--out before.json`) and once after (`--out after.json --compare before.json`):
any differing sample fails the gate. The vision-only loader keeps this at ~3 GB for the 122B
tower (~15 GB for a full 7B), so it fits beside a running server.

Also records the synchronised wall time of each arm's vision pass (`t_ms`), so the same run
doubles as the per-lever timing column of docs/memo/qwen_correct_forward_profile.md.

  PYTHONPATH=<tree> python analysis/experiments/qwen_axis_snapshot.py --family qwen35 \
      --model Qwen/Qwen3.5-122B-A10B-FP8 --dataset realworldqa --samples 32 \
      --out analysis/results/vllm_stream/snap_122b_before.json
  PYTHONPATH=<other tree> ... --out ..._after.json --compare ..._before.json
"""
import argparse, hashlib, json, os, statistics, sys, time

import torch

# No sys.path edits: `appcorr` and `analysis` come from PYTHONPATH, which is how the same
# script is pointed at the before-tree and the after-tree.
from analysis.experiments.qwen35_accuracy import degrade          # noqa: E402


def make_axis(family, model, proc):
    if family == "qwen25vl":
        from appcorr.models.qwen25vl.unified import Qwen25VLAxis
        return Qwen25VLAxis(model, proc)
    from appcorr.models.qwen35.unified import Qwen35Axis
    return Qwen35Axis(model, proc)


def sha(t: torch.Tensor) -> str:
    t = t.detach().contiguous()
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    return hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest()[:24]


class RecordingSink:
    """Stands in for `StreamSink`: hashes each chunk instead of sending it."""

    def __init__(self):
        self.chunks = []

    def push(self, embeds, mrope, mrope_delta, final):
        self.chunks.append({"n": int(embeds.shape[0]), "embeds": sha(embeds),
                            "mrope": (None if mrope is None else sha(mrope)),
                            "delta": (None if mrope_delta is None else int(mrope_delta)),
                            "final": bool(final)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["qwen25vl", "qwen35"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--load", choices=["full", "vision"], default="vision")
    ap.add_argument("--dataset", default="realworldqa")
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--keeps", type=float, nargs="+", default=[1.0, 0.5])
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--degrade-filter", choices=["bicubic", "box", "pyr"], default="pyr")
    ap.add_argument("--positions-mode", default=None,
                    help="set the axis' positions_mode if the tree has it (e.g. check)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--compare", default=None, help="a previous --out; fail on any difference")
    a = ap.parse_args()

    from transformers import AutoProcessor, AutoModelForImageTextToText
    from qwen_vl_prefill.datasets_eval import get_spec
    from datasets import load_dataset

    proc = AutoProcessor.from_pretrained(a.model)
    if a.load == "vision":
        from appcorr.models.vision_only import load_vision_only
        model = load_vision_only(a.model, device="cuda:0")
    else:
        model = AutoModelForImageTextToText.from_pretrained(a.model, dtype="auto",
                                                            device_map="cuda:0").eval()
    axis = make_axis(a.family, model, proc)
    if a.positions_mode is not None:
        if not hasattr(axis, "positions_mode"):
            raise SystemExit("this tree's axis has no positions_mode")
        axis.positions_mode = a.positions_mode
    spec = get_spec(a.dataset)
    ds = spec.load(load_dataset)
    n = min(a.samples, len(ds))
    idxs = list(range(0, len(ds), max(1, len(ds) // n)))[:n]

    def timed(fn):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = fn()
        torch.cuda.synchronize()
        return r, (time.perf_counter() - t0) * 1e3

    res = {}
    with torch.no_grad():
        for i in idxs:
            img, q, gold = spec.prepare(ds[i], lambda h, w, **kw: (h, w), 1, 1, 1 << 30)
            if img.mode != "RGB":
                img = img.convert("RGB")
            base = degrade(img, a.level, a.degrade_filter)
            inputs = axis.build_inputs(img, q).to("cuda:0")
            px_base = axis.build_inputs(base, q)["pixel_values"].to("cuda:0")
            row = {"prompt_tokens": int(inputs["input_ids"].shape[1])}
            for arm, px in (("ceiling", inputs["pixel_values"]), ("floor", px_base)):
                (emb, pos, delta), t_ms = timed(lambda: axis.oneshot_embeds(inputs, px))
                row[arm] = {"embeds": sha(emb), "mrope": sha(pos), "delta": int(delta), "t_ms": t_ms}
            for keep in a.keeps:
                sink = RecordingSink()
                (_, _, st), t_ms = timed(
                    lambda: axis.streaming_forward(inputs, px_base, a.groups, keep=keep, sink=sink))
                row[f"streaming_g{a.groups}_k{keep:.2f}"] = {
                    "chunks": sink.chunks, "bounds": [list(c) for c in st["chunks"]],
                    "corrected_groups": int(st["corrected_groups"]),
                    "decode_start_pos": int(st["decode_start_pos"]),
                    "rope_delta": int(st["rope_delta"]), "t_ms": t_ms}
            res[str(i)] = row
            print(f"i={i} tokens={row['prompt_tokens']} "
                  + " ".join(f"{k}={v['t_ms']:.0f}ms" for k, v in row.items() if isinstance(v, dict)),
                  flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump({"args": vars(a), "rows": res}, open(a.out, "w"), indent=1)
    arms = [k for k in next(iter(res.values())) if isinstance(next(iter(res.values()))[k], dict)]
    print("median t_ms: " + "  ".join(
        f"{arm} {statistics.median(r[arm]['t_ms'] for r in res.values()):.0f}" for arm in arms))

    if a.compare:
        ref = json.load(open(a.compare))["rows"]
        strip = lambda d: {k: (strip(v) if isinstance(v, dict) else v)  # noqa: E731
                           for k, v in d.items() if k != "t_ms"}
        bad = [i for i in res if i not in ref or strip(res[i]) != strip(ref[i])]
        for i in bad[:10]:
            for arm in arms:
                if i in ref and strip(res[i].get(arm, {})) != strip(ref[i].get(arm, {})):
                    print(f"  i={i} {arm} differs")
        print(f"QWEN_AXIS_SNAPSHOT_GATE {'FAIL' if bad else 'PASS'} "
              f"({len(res) - len(bad)}/{len(res)} identical vs {a.compare})")
        return 1 if bad else 0
    print("QWEN_AXIS_SNAPSHOT_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
