"""Where the HF-side streaming vision pass spends its time (Qwen2.5-VL / Qwen3.5 towers), the
port of `dinov3_correct_profile.py` to `QwenVLStreamingAxis.streaming_forward`. No engine: a
recording sink stub swallows the chunks, so the vision-only tower (~3 GB for the 122B) fits
beside a running server.

Three measurements per (image size, keep):
  (a) per-stage wall via the axis' own `_stage` scopes (prepare / vision_base / vision_correct
      x rounds / merge), each bracketed by a cuda synchronize -- plus the unbracketed total, so
      the bracketing's own cost is visible;
  (b) torch.profiler over one pass: CUDA kernels by self time, coarse buckets (attention before
      GEMM -- see dinov3_correct_profile.py for why the order matters), the launch gap (wall
      minus summed kernel time = GPU idle), and the host-sync inventory
      (`aten::item` / `_local_scalar_dense` / `nonzero` / `cudaStreamSynchronize` counts);
  (c) the keep<1 delta attributed: `_approx_base` with and without the received-attention
      collection, timed alone.

Sizes are requested in image TOKENS (merged grid cells) and realised by resizing one dataset
image to the matching pixel size, so the same script gives the same shapes on every tree.

  PYTHONPATH=<tree> python analysis/experiments/qwen_correct_profile.py --family qwen35 \
      --model Qwen/Qwen3.5-122B-A10B-FP8 --tokens 2100 4800 --keeps 1.0 0.5 \
      --out analysis/results/vllm_stream/profile_122b_<tag>.json
"""
import argparse, json, math, os, statistics, sys, time
from collections import defaultdict
from contextlib import contextmanager

import torch

from analysis.experiments.qwen35_accuracy import degrade          # noqa: E402


def make_axis(family, model, proc):
    if family == "qwen25vl":
        from appcorr.models.qwen25vl.unified import Qwen25VLAxis
        return Qwen25VLAxis(model, proc)
    from appcorr.models.qwen35.unified import Qwen35Axis
    return Qwen35Axis(model, proc)


class NullSink:
    """Swallows the chunks; keeps the axis on its `sink is not None` code path."""

    def __init__(self):
        self.n = 0

    def push(self, embeds, mrope, mrope_delta, final):
        self.n += 1


class StageTimer:
    """Replacement for `axis._stage`: synchronised wall per stage name, in call order."""

    def __init__(self):
        self.rows = []

    @contextmanager
    def __call__(self, name):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            torch.cuda.synchronize()
            self.rows.append((name, (time.perf_counter() - t0) * 1e3))


def image_at_tokens(img, n_tokens, side_px):
    """Resize `img` so the merged token grid has ~n_tokens cells (aspect kept, sides multiples
    of the merged-cell pixel size)."""
    w, h = img.size
    cells = n_tokens
    ch = max(1, round(math.sqrt(cells * h / w)))
    cw = max(1, round(cells / ch))
    return img.resize((cw * side_px, ch * side_px))


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    r = fn()
    torch.cuda.synchronize()
    return r, (time.perf_counter() - t0) * 1e3


BUCKETS = {
    "attention / SDPA": ("attention", "sdpa", "flash", "fmha", "softmax"),
    "GEMM (Linear)": ("gemm", "sm90", "sm100", "cutlass", "ampere", "nvjet", "scaled_mm",
                      "tensorop", "s16816", "gett"),
    "elementwise / residual / norm": ("elementwise", "layer_norm", "rms_norm", "silu", "gelu",
                                      "reduce_kernel", "vectorized", "unrolled"),
    "gather / scatter / index": ("index", "gather", "scatter", "take", "copy", "clone",
                                 "masked", "sort", "cumsum", "unique"),
    "triton (appcorr kernels)": ("triton",),
}
SYNC_KEYS = ("aten::item", "aten::_local_scalar_dense", "aten::nonzero", "cudastreamsynchronize",
             "cudadevicesynchronize", "aten::masked_select", "memcpy dtoh", "cudamemcpyasync")


def profile_pass(fn, wall_ms):
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=False) as prof:
        fn()
        torch.cuda.synchronize()
    ka = prof.key_averages()
    evts = [e for e in ka if e.self_device_time_total > 0]
    evts.sort(key=lambda e: e.self_device_time_total, reverse=True)
    agg, launches, stall_ms = defaultdict(float), 0, 0.0
    top = []
    for e in evts:
        k = e.key.lower()
        if k.startswith("aten::") or k.startswith("cuda") or k.startswith("profiler"):
            continue
        # A profiler pseudo-event, not a kernel: the host blocked because the GPU's command
        # buffer was full (i.e. the GPU was saturated). Counting it made "kernels" exceed wall by
        # ~50% on the keep<1 pass; report it separately instead.
        if "command buffer full" in k:
            stall_ms += e.self_device_time_total / 1e3
            continue
        launches += e.count
        ms = e.self_device_time_total / 1e3
        if len(top) < 25:
            top.append((e.key[:70], ms, e.count))
        for label, pats in BUCKETS.items():
            if any(p in k for p in pats):
                agg[label] += ms
                break
        else:
            agg["other"] += ms
    kernel_ms = sum(agg.values())
    syncs = {}
    for e in ka:
        k = e.key.lower()
        if any(s in k for s in SYNC_KEYS):
            syncs[e.key] = {"calls": int(e.count), "cpu_ms": e.cpu_time_total / 1e3}
    return {"wall_ms": wall_ms, "kernel_ms": kernel_ms, "idle_ms": wall_ms - kernel_ms,
            "launches": launches, "cmdbuf_full_ms": stall_ms, "buckets": dict(agg), "top": top,
            "syncs": syncs}


def print_profile(tag, p):
    print(f"\n===== {tag}: launch gap ===== wall {p['wall_ms']:.1f} ms | kernels {p['kernel_ms']:.1f} ms "
          f"| idle {p['idle_ms']:.1f} ms ({100 * p['idle_ms'] / max(p['wall_ms'], 1e-9):.0f}% of wall) "
          f"| {p['launches']} kernel launches | command-buffer-full stalls {p.get('cmdbuf_full_ms', 0):.1f} ms")
    print(f"----- buckets (raw CUDA kernels, total {p['kernel_ms']:.1f} ms) -----")
    for label, ms in sorted(p["buckets"].items(), key=lambda kv: -kv[1]):
        print(f"  {label:<34}{ms:>9.2f} ms{100 * ms / max(p['kernel_ms'], 1e-9):>8.1f}%")
    print(f"----- top kernels -----")
    for k, ms, n in p["top"][:15]:
        print(f"  {k:<72}{ms:>8.2f} ms{n:>7}")
    print(f"----- host syncs / D2H -----")
    for k, v in sorted(p["syncs"].items(), key=lambda kv: -kv[1]["calls"]):
        print(f"  {k[:52]:<54}{v['calls']:>7} calls{v['cpu_ms']:>9.2f} ms cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["qwen25vl", "qwen35"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--load", choices=["full", "vision"], default="vision")
    ap.add_argument("--dataset", default="realworldqa")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--tokens", type=int, nargs="+", default=[2100, 4800])
    ap.add_argument("--keeps", type=float, nargs="+", default=[1.0, 0.5])
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--degrade-filter", choices=["bicubic", "box", "pyr"], default="pyr")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default=None)
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
    tower = axis.tower
    ip = proc.image_processor
    side_px = int(ip.patch_size) * int(ip.merge_size)     # pixels per merged token side

    spec = get_spec(a.dataset)
    ds = spec.load(load_dataset)
    img, q, _ = spec.prepare(ds[a.index], lambda h, w, **kw: (h, w), 1, 1, 1 << 30)
    if img.mode != "RGB":
        img = img.convert("RGB")

    info = {"args": vars(a), "tree": os.environ.get("PYTHONPATH", ""),
            "correct_rows_only": getattr(axis, "correct_rows_only", None),
            "positions_mode": getattr(axis, "positions_mode", None), "cases": {}}
    print(f"tree={info['tree']} correct_rows_only={info['correct_rows_only']} "
          f"positions_mode={info['positions_mode']}", flush=True)

    with torch.no_grad():
        for n_tok_req in a.tokens:
            im = image_at_tokens(img, n_tok_req, side_px)
            base = degrade(im, a.level, a.degrade_filter)
            inputs = axis.build_inputs(im, q).to("cuda:0")
            px_full = inputs["pixel_values"]
            px_base = axis.build_inputs(base, q)["pixel_values"].to("cuda:0")
            grid = inputs["image_grid_thw"]
            n_tok = int(inputs["mm_token_type_ids"].sum())
            print(f"\n######## requested {n_tok_req} -> {n_tok} image tokens, grid {grid[0].tolist()}, "
                  f"image {im.size} ########", flush=True)
            case = {"image_tokens": n_tok, "grid": grid[0].tolist(), "arms": {}}

            # one-shot reference (ceiling): the tower once, no correction
            for _ in range(2):
                axis.oneshot_embeds(inputs, px_full)
            t = [timed(lambda: axis.oneshot_embeds(inputs, px_full))[1] for _ in range(a.repeats)]
            case["arms"]["oneshot"] = {"wall_ms": statistics.median(t)}
            print(f"[oneshot tower] {statistics.median(t):.1f} ms (median of {a.repeats})")

            # (c) received-attention collection alone (the keep<1 base pass vs the keep=1 one)
            try:
                gctx = tower.prepare_grid(grid, "cuda:0")
                ctx_base = tower.prepare_full_tokens(px_base, grid, gctx)
            except (AttributeError, TypeError):
                ctx_base = tower.prepare_full_tokens(px_base, grid)
            for flag in (False, True):
                axis._approx_base(ctx_base, {}, collect_attn=flag)
            t_off = statistics.median(timed(lambda: axis._approx_base(ctx_base, {}, collect_attn=False))[1]
                                      for _ in range(a.repeats))
            t_on = statistics.median(timed(lambda: axis._approx_base(ctx_base, {}, collect_attn=True))[1]
                                     for _ in range(a.repeats))
            case["approx_base_ms"] = {"collect_attn_off": t_off, "collect_attn_on": t_on}
            print(f"[approx_base] collect_attn off {t_off:.1f} ms | on {t_on:.1f} ms | "
                  f"received-attention cost {t_on - t_off:.1f} ms")

            for keep in a.keeps:
                arm = f"streaming_g{a.groups}_k{keep:.2f}"
                run = lambda: axis.streaming_forward(inputs, px_base, a.groups, keep=keep, sink=NullSink())  # noqa: E731
                for _ in range(2):
                    run()
                # unbracketed wall
                walls = [timed(run)[1] for _ in range(a.repeats)]
                wall = statistics.median(walls)
                # (a) per-stage wall, bracketed
                st = StageTimer()
                orig = axis._stage
                axis._stage = st
                try:
                    _, wall_br = timed(run)
                finally:
                    axis._stage = orig
                stages = defaultdict(list)
                for name, ms in st.rows:
                    stages[name].append(ms)
                stage_ms = {k: sum(v) for k, v in stages.items()}
                print(f"\n[{arm}] wall {wall:.1f} ms (median of {a.repeats}; min {min(walls):.1f}) | "
                      f"bracketed {wall_br:.1f} ms | tower x{wall / case['arms']['oneshot']['wall_ms']:.1f}")
                for k, v in stages.items():
                    print(f"    {k:<16}{sum(v):>8.1f} ms  ({len(v)}x: "
                          + ", ".join(f"{x:.1f}" for x in v) + ")")
                # (b) profiler
                p = profile_pass(run, wall)
                print_profile(arm, p)
                case["arms"][arm] = {"wall_ms": wall, "wall_min_ms": min(walls),
                                     "wall_bracketed_ms": wall_br, "stages_ms": stage_ms,
                                     "stages_seq": st.rows, "profile": p}
            info["cases"][str(n_tok)] = case

    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(info, open(a.out, "w"), indent=1)
        print(f"\nwrote {a.out}")
    print("\n===== summary (wall ms) =====")
    hdr = ["tokens", "oneshot"] + [f"streaming_g{a.groups}_k{k:.2f}" for k in a.keeps] + ["recv-attn"]
    print("  ".join(f"{h:>20}" for h in hdr))
    for n, c in info["cases"].items():
        row = [n, f"{c['arms']['oneshot']['wall_ms']:.1f}"]
        for k in a.keeps:
            arm = c["arms"].get(f"streaming_g{a.groups}_k{k:.2f}")
            row.append(f"{arm['wall_ms']:.1f} (idle {100 * arm['profile']['idle_ms'] / arm['wall_ms']:.0f}%)" if arm else "-")
        row.append(f"{c['approx_base_ms']['collect_attn_on'] - c['approx_base_ms']['collect_attn_off']:.1f}")
        print("  ".join(f"{x:>20}" for x in row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
