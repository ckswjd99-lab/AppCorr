"""Gate for the deferred patch score (`QwenVLStreamingAxis.pscore_defer`).

Vision-only, in-process, no engine: `streaming_forward` runs against a recording sink for a
few samples and writes one JSON record per (sample, keep, mode) with

  * sha256 of `image_embeds` (the rows the LLM would consume),
  * the per-band selected merge groups,
  * bitwise equality of the eager vs deferred attention term (same vectors, computed from the
    stashed q instead of during the base pass),
  * `t_open_ms` (first push, i.e. the vision work before the engine can start) and the total
    vision wall.

Two uses:
  --mode eager against the pre-change code (PYTHONPATH swap) must give identical sha + groups:
  the eager path is untouched.  --mode both on the new code: bands >= 1 must select the same
  groups eager vs deferred and the attention vectors must be bitwise equal; band 0 differs by
  design (energy only) and t_open must drop to the keep=1.0 level.
"""
import argparse, hashlib, json, os, sys, time
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))


class RecSink:
    """Records push times; the driver's bridge sink minus the socket."""
    def __init__(self):
        self.t = []
        self.t0 = time.perf_counter()

    def push(self, emb, pos, delta, final=False):
        torch.cuda.synchronize()
        self.t.append((time.perf_counter() - self.t0) * 1e3)


def sha(t):
    return hashlib.sha256(t.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()[:16]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    ap.add_argument("--dataset", default="realworldqa")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--keeps", default="1.0,0.5,0.25")
    ap.add_argument("--mode", choices=["eager", "deferred", "both"], default="both")
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--degrade-filter", default="box")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoProcessor
    from datasets import load_dataset
    from qwen_vl_prefill.datasets_eval import get_spec
    from analysis.experiments.qwen35_accuracy import degrade
    from analysis.experiments.qwen_vllm_accuracy import make_axis
    from appcorr.models.vision_only import load_vision_only

    proc = AutoProcessor.from_pretrained(args.model)
    model = load_vision_only(args.model, device="cuda:0")
    axis = make_axis("qwen35", model, proc)
    axis.image_embeds_with_sink = True
    spec = get_spec(args.dataset)
    ds = spec.load(load_dataset)
    idxs = list(range(0, len(ds), max(1, len(ds) // args.samples)))[:args.samples]
    modes = ["eager", "deferred"] if args.mode == "both" else [args.mode]
    keeps = [float(k) for k in args.keeps.split(",")]
    has_defer = hasattr(axis, "supports_deferred_pscore")

    fh = open(args.out, "w")
    for i in idxs:
        img, q, gold = spec.prepare(ds[int(i)], lambda h, w, **kw: (h, w), 1, 1, 1 << 30)
        img = img.convert("RGB")
        base = degrade(img, args.level, args.degrade_filter)
        inputs = axis.build_inputs(img, q)
        px_base = axis.build_inputs(base, q)["pixel_values"]
        ids = inputs["input_ids"][0]
        pos = (ids == axis.image_token_id).nonzero(as_tuple=True)[0]
        layout = {"image_run": (int(pos[0]), int(pos.numel())),
                  "grid_thw": tuple(int(v) for v in inputs["image_grid_thw"][0].tolist())}
        inputs = inputs.to("cuda:0")
        px_base = px_base.to("cuda:0")
        for keep in keeps:
            for mode in modes:
                if mode == "deferred" and (keep >= 1.0 or not has_defer):
                    continue
                axis.pscore_defer = mode == "deferred"
                for rep in range(2):  # rep 0 = warmup for the timing; record rep 1
                    sink = RecSink()
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    _, _, st = axis.streaming_forward(inputs, px_base, args.groups, keep=keep,
                                                      sink=sink, **layout)
                    torch.cuda.synchronize()
                    wall = (time.perf_counter() - t0) * 1e3
                rec = {"i": int(i), "keep": keep, "mode": mode, "n_tok": layout["image_run"][1],
                       "sha": sha(st["image_embeds"]), "t_open_ms": round(sink.t[0], 2),
                       "t_vision_ms": round(wall, 2), "pscore": st.get("pscore"),
                       "groups": [g.tolist() for g in st.get("group_idx", [])]}
                # Attention term: eager (during the base pass) vs deferred (from stashed q).
                if keep < 1.0 and has_defer and mode == "deferred":
                    gctx = axis.tower.prepare_grid(inputs["image_grid_thw"], px_base.device)
                    ctx_b = axis.tower.prepare_full_tokens(px_base, inputs["image_grid_thw"], gctx)
                    _, c1 = axis._approx_base(ctx_b, {}, collect_attn=True)
                    v_e = axis._attn_layermean(c1)
                    ctx_b = axis.tower.prepare_full_tokens(px_base, inputs["image_grid_thw"], gctx)
                    _, c2 = axis._approx_base(ctx_b, {}, collect_attn="defer")
                    v_d = axis._attn_layermean_deferred(c2, ctx_b)
                    rec["attn_bitwise"] = bool(torch.equal(v_e, v_d))
                    rec["attn_maxabs"] = float((v_e.float() - v_d.float()).abs().max())
                fh.write(json.dumps(rec) + "\n")
                fh.flush()
                print(f"i={i} keep={keep} {mode:8s} tok={rec['n_tok']} sha={rec['sha']} "
                      f"t_open={rec['t_open_ms']:.1f} vision={rec['t_vision_ms']:.1f} "
                      f"attn_bitwise={rec.get('attn_bitwise')}", flush=True)
    fh.close()


if __name__ == "__main__":
    main()
