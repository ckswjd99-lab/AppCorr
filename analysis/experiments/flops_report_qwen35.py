"""Critical/total FLOPs for the Qwen3.5-35B streaming arm, via the unified axis's scopes.

Three arms per sample, matching every other model's report:
    ceiling    stock forward on the full image          -> 100% critical
    floor      stock forward on the degraded base       -> 100% critical
    streaming  vision approx (arrival 0) + per-band correct+prefill (arrivals 1..g)
               -> critical = the final band's vision correct + its prefill chunk

MoE note: expert FLOPs are COUNT-based, not data-dependent -- the handler charges
`top_k_index.numel() = n_tok * top_k` whichever experts the router hits, so two prompts of equal
length cost identically. (An earlier draft of this docstring claimed the opposite; the measured
floor == full to 0.1 GF is what corrected it -- same grid, same token count, same cost.) Routing
DOES change which weights are touched, which matters for memory traffic, not for FLOPs.

Base degradation is the transmission's own level-2 pyramid base: downsample 4x, upsample back.
"""
import argparse, glob, json, os, re, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from appcorr.flops.counter import FlopCounter
from appcorr.flops import hooks
from appcorr.models.qwen35.unified import Qwen35Axis, MODEL_ID_35B

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from qwen_vl_prefill.datasets_eval import get_spec  # the registry every other report uses


def load_samples(name, n):
    """Via the shared spec registry -- it loads from the local HF cache, so it works with
    HF_HUB_OFFLINE=1. The first version of this file re-invented loading with `streaming=True`,
    which needs the network by design and died at the first sample; use the machinery that the
    Gemma 3 and Qwen 2.5 reports already run on."""
    from datasets import load_dataset
    spec = get_spec(name)
    ds = spec.load(load_dataset)
    idxs = list(range(0, len(ds), max(1, len(ds) // n)))[:n]
    out = []
    for i in idxs:
        # Identity smart_resize (the gemma3 report's convention here): the processor applies this
        # model's own resolution policy anyway, so pre-resizing to Qwen2.5's would double-resize.
        img, q, _gold = spec.prepare(ds[int(i)], lambda h, w, **kw: (h, w), 1, 1, 1 << 30)
        out.append((img, q))
    return out


def stage_split(counter):
    """(vision_total, vision_crit, llm_total, llm_crit) GFLOPs per request, from the counter's
    per-(arrival, stage) buckets. `aggregate()` splits critical from overlappable but not vision
    from decoder, and the interleaved keys below need both splits at once: the interleaved arm
    shares this arm's VISION work (same tower, same bands, plus one extra merge of the base at
    t=0) and replaces only its decoder half."""
    from appcorr.flops.counter import RequestFlops
    n = len(counter.requests)
    acc = [0, 0, 0, 0]
    for r in counter.requests:
        live = [(a, st) for (a, st) in r.buckets if st not in RequestFlops.EXCLUDED_STAGES]
        amax = max(a for a, _ in live) if live else 0
        for (a, st), b in r.buckets.items():
            if st in RequestFlops.EXCLUDED_STAGES:
                continue
            i = 2 if st == "llm_prefill" else 0
            acc[i] += b.total
            if a == amax:
                acc[i + 1] += b.total
    return [v / max(n, 1) / 1e9 for v in acc]


def interleaved_from_rows(rows_dir, ds_name, slug, groups, model_key):
    """{keep: (total, crit, n_rows, path)} decoder-side GFLOPs of the interleaved schedule,
    replayed from the accuracy driver's rows.

    The interleaved arm is sink-only (the engine holds the KV cache it rewrites), so the hooks
    cannot measure it the way the streaming arm is measured here -- the cost comes from the
    closed form instead, driven by the per-sample `chunks` records the driver stores
    (`("approx", 0, N-1)` then `("correct", s, e, |P_r|)` per round). The closed form is
    reconciled against these very hooks by `flops_analytic.validate_qwen35` (0.004% on four
    datasets, 2026-09-10); run that before trusting a number from here."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from flops_analytic import MODELS35
    dec = MODELS35[model_key]
    out = {}
    for path in sorted(glob.glob(os.path.join(
            rows_dir, f"{ds_name}_{slug}_interleaved_g{groups}*.jsonl"))):
        m = re.search(r"_interleaved_g(\d+)(?:_k(\d+\.\d+))?", os.path.basename(path))
        if m is None or int(m.group(1)) != groups:
            continue
        keep = float(m.group(2)) if m.group(2) else 1.0
        tot = crit = 0.0
        n = 0
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            if "skip" in r or not isinstance(r.get("chunks"), list):
                continue
            lo, n_img = r["image_run"]
            c = dec.interleaved_cost(int(r["prompt_tokens"]), int(lo), int(n_img),
                                     [tuple(x) for x in r["chunks"]])
            tot += c["total"] / 1e9
            crit += c["crit"] / 1e9
            n += 1
        if n:
            out[keep] = (tot / n, crit / n, n, path)
    return out


def degrade(img: Image.Image) -> Image.Image:
    w, h = img.size
    return img.resize((max(1, w // 4), max(1, h // 4)), Image.BICUBIC).resize((w, h), Image.BICUBIC)


def add_interleaved(row, args, ds_name, slug, vis, keeps=None):
    """Fold the interleaved keys for one dataset into `row`. `vis` = (vision_total, vision_crit)
    GFLOPs of the same schedule's vision half, 0 when this run measured nothing (--il-only)."""
    il = interleaved_from_rows(args.il_rows, ds_name, slug, args.groups, args.il_model)
    for keep, (tot, crit, n, path) in sorted(il.items()):
        if keeps is not None and not any(abs(keep - k) < 1e-9 for k in keeps):
            continue
        suffix = f"_g{args.groups}" if keep == 1.0 else f"_g{args.groups}_k{keep:.2f}"
        row[f"total{suffix}_il"] = round(vis[0] + tot, 1)
        row[f"crit{suffix}_il"] = round(vis[1] + crit, 1)
        row[f"_il{suffix}"] = {"decoder_total": round(tot, 1), "decoder_crit": round(crit, 1),
                               "vision_total": round(vis[0], 1), "vision_crit": round(vis[1], 1),
                               "rows": n, "source": os.path.basename(path),
                               "note": "decoder half is the closed form of "
                                       "flops_analytic.Qwen35Decoder replayed from the rows' "
                                       "`chunks`; vision half is this run's hooked streaming arm"}
        print(f"{ds_name:<14} k={keep:.2f} interleaved crit {row[f'crit{suffix}_il']:8.1f} "
              f"total {row[f'total{suffix}_il']:9.1f}  (decoder {crit:.1f}/{tot:.1f} over "
              f"{n} rows)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID_35B)
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--datasets", nargs="+", default=["chartqa"])
    ap.add_argument("--keeps", type=float, nargs="+", default=[1.0],
                    help="streaming keep ratios to measure; 1.0 reproduces the original arm")
    ap.add_argument("--out-json", default="analysis/results/flops/qwen35_flops.json")
    ap.add_argument("--il-rows", default=None,
                    help="accuracy-driver output dir holding `{ds}_{slug}_interleaved_g{g}*.jsonl`"
                         " rows; adds total_g{g}[_k{k}]_il / crit_g{g}[_k{k}]_il, the interleaved "
                         "schedule's cost replayed from each row's `chunks` (closed form, "
                         "flops_analytic.Qwen35Decoder) on top of THIS run's measured vision half")
    ap.add_argument("--il-model", default="qwen35_35b",
                    help="flops_analytic.MODELS35 key for the --il-rows replay")
    ap.add_argument("--il-only", action="store_true",
                    help="no GPU: only fold the --il-rows keys into an existing --out-json "
                         "(the interleaved cost is decoder-side then, with no vision half to add)")
    args = ap.parse_args()

    if args.il_only:
        if not args.il_rows:
            raise SystemExit("--il-only needs --il-rows")
        result = json.load(open(args.out_json)) if os.path.exists(args.out_json) else {}
        slug = args.model.split("/")[-1].lower()
        for ds_name in args.datasets:
            row = result.setdefault(ds_name, {})
            add_interleaved(row, args, ds_name, slug, vis=(0.0, 0.0))
        json.dump(result, open(args.out_json, "w"), indent=1)
        print(f"wrote {args.out_json}")
        return

    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype="auto", device_map="cuda:0").eval()

    result = {"_model": args.model, "_samples": args.samples, "_groups": args.groups}
    for ds_name in args.datasets:
        samples = load_samples(ds_name, args.samples)
        arms = ["ceiling", "floor"] + [f"streaming_k{k:.2f}" for k in args.keeps]
        counters = {k: FlopCounter() for k in arms}
        axis_by = {k: Qwen35Axis(model, proc, flop_counter=c) for k, c in counters.items()}
        # Hooks installed per arm inside the loop (visual + language model); lm_head excluded
        # as everywhere.
        for si, (img, q) in enumerate(samples):
            base = degrade(img)
            inputs = axis_by["ceiling"].build_inputs(img, q).to("cuda:0")
            inputs_base = axis_by["ceiling"].build_inputs(base, q).to("cuda:0")
            with torch.no_grad():
                for arm in arms:
                    c = counters[arm]
                    axis = axis_by[arm]
                    handles = hooks.install(c, [model.model.visual, model.model.language_model])
                    # patch_attention required -- install() alone drops the SDPA term (2026-08-31).
                    with hooks.patch_attention(c), c.request(f"{ds_name}/{si}"):
                        if arm == "ceiling":
                            axis.full_forward(inputs)
                        elif arm == "floor":
                            axis.approx_only_forward(inputs, inputs_base["pixel_values"])
                        else:
                            axis.streaming_forward(inputs, inputs_base["pixel_values"],
                                                   args.groups, keep=float(arm.split("_k")[1]))
                    hooks.remove(handles)
        agg = {k: c.aggregate() for k, c in counters.items()}
        full = agg["ceiling"]["mean_total_gflops"]
        row = {"full": round(full, 1),
               "floor": round(agg["floor"]["mean_total_gflops"], 1)}
        for k in args.keeps:
            st = agg[f"streaming_k{k:.2f}"]
            # k=1.00 keeps the original key names so existing consumers keep reading them.
            suffix = f"_g{args.groups}" if k == 1.0 else f"_g{args.groups}_k{k:.2f}"
            row[f"crit{suffix}"] = round(st["mean_critical_gflops"], 1)
            row[f"total{suffix}"] = round(st["mean_total_gflops"], 1)
            print(f"{ds_name:<14} k={k:.2f} full {full:9.1f}  floor {row['floor']:9.1f}  "
                  f"streaming crit {st['mean_critical_gflops']:8.1f} "
                  f"total {st['mean_total_gflops']:9.1f}  crit/full = "
                  f"{st['mean_critical_gflops'] / full * 100:5.1f}%")
        if args.il_rows:
            # vision half from the streaming arm measured just above (the schedules share it);
            # keep=1.00's counter is representative of the vision work at any keep only for
            # keep=1.00, so each keep uses its own arm's split.
            slug = args.model.split("/")[-1].lower()
            for k in args.keeps:
                vis = stage_split(counters[f"streaming_k{k:.2f}"])[:2]
                add_interleaved(row, args, ds_name, slug, vis, keeps=[k])
        result[ds_name] = row

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(result, open(args.out_json, "w"), indent=1)
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
