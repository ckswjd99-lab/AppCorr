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


def interleaved_from_rows(rows_dir, ds_name, slug, groups, model_key, staged=False,
                          unified=False):
    """{keep: (total, crit, n_rows, path, prefill, vis)} decoder-side GFLOPs of the interleaved
    schedule, replayed from the accuracy driver's rows. `staged`: the depth-staged arm's rows
    (`interleaved_staged_g*`), whose `chunks` carry the round index and are priced at the
    round's depth (ideal schedule: approx pass + frontier walks = one full prefill).
    `unified`: the unified vision+decoder axis's rows (`interleaved_unified_g*`), which price their
    decoder rows at an EXPLICIT per-round depth AND carry the vision half's records -- `vis` is
    then {"ratio_total", "ratio_crit"}, the closed-form unified vision half over the closed-form
    FULL-DEPTH one (the vision work every other arm does), so the caller can rescale a hooked
    vision half instead of mixing a closed-form absolute into a measured column. `vis` is None
    for the other arms, whose vision half is common with the streaming arm's.

    The interleaved arm is sink-only (the engine holds the KV cache it rewrites), so the hooks
    cannot measure it the way the streaming arm is measured here -- the cost comes from the
    closed form instead, driven by the per-sample `chunks` records the driver stores
    (`("approx", 0, N-1)` then `("correct", s, e, |P_r|)` per round). The closed form is
    reconciled against these very hooks by `flops_analytic.validate_qwen35` (0.004% on four
    datasets, 2026-09-10); run that before trusting a number from here. `prefill` is the
    closed-form stock prefill of the same rows (the "1" of the schedule), so a caller can turn
    total/crit into ratios and apply them to a hooked prefill of a different sample."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from flops_analytic import MODELS35, QWEN35_VISION
    dec = MODELS35[model_key]
    out = {}
    arm = "interleaved_unified" if unified else ("interleaved_staged" if staged else "interleaved")
    for path in sorted(glob.glob(os.path.join(
            rows_dir, f"{ds_name}_{slug}_{arm}_g{groups}*.jsonl"))):
        m = re.search(rf"_{arm}_g(\d+)(?:_k(\d+\.\d+))?", os.path.basename(path))
        if m is None or int(m.group(1)) != groups:
            continue
        keep = float(m.group(2)) if m.group(2) else 1.0
        tot = crit = pre = 0.0
        v_tot = v_crit = v_ref_tot = v_ref_crit = 0.0
        n = 0
        for line in open(path):
            if not line.strip():
                continue
            r = json.loads(line)
            if "skip" in r or not isinstance(r.get("chunks"), list):
                continue
            lo, n_img = r["image_run"]
            chunks = [tuple(x) for x in r["chunks"]]
            c = dec.interleaved_cost(int(r["prompt_tokens"]), int(lo), int(n_img), chunks)
            assert c["staged"] == (staged or unified), (path, c["staged"])
            tot += c["total"] / 1e9
            crit += c["crit"] / 1e9
            pre += dec.prefill_flops(int(r["prompt_tokens"]) - 1) / 1e9
            if unified:
                v = QWEN35_VISION.unified_cost(chunks)
                nr = v["n_rows"]
                rowl = QWEN35_VISION.row_layer_flops(nr)
                # Reference = the SAME corrections at full tower depth on top of one full
                # approximate pass: the vision half every other arm runs, so the ratio is what
                # the staging changed and nothing else.
                corr = [int(x[3]) for x in chunks if x[0] == "vcorrect"]
                v_tot += v["total"] / 1e9
                v_crit += v["crit"] / 1e9
                v_ref_tot += (QWEN35_VISION.tower_flops(nr)
                              + sum(corr) * QWEN35_VISION.layers * rowl) / 1e9
                v_ref_crit += ((corr[-1] if corr else 0) * QWEN35_VISION.layers * rowl) / 1e9
            n += 1
        if n:
            vis = None
            if unified:
                vis = {"ratio_total": v_tot / max(v_ref_tot, 1e-12),
                       "ratio_crit": v_crit / max(v_ref_crit, 1e-12),
                       "closed_vision_total": v_tot / n, "closed_vision_crit": v_crit / n,
                       "closed_vision_ref_total": v_ref_tot / n,
                       "closed_vision_ref_crit": v_ref_crit / n}
            out[keep] = (tot / n, crit / n, n, path, pre / n, vis)
    return out


def degrade(img: Image.Image) -> Image.Image:
    w, h = img.size
    return img.resize((max(1, w // 4), max(1, h // 4)), Image.BICUBIC).resize((w, h), Image.BICUBIC)


def add_interleaved(row, args, ds_name, slug, vis, keeps=None):
    """Fold the interleaved keys for one dataset into `row`. `vis` = (vision_total, vision_crit)
    GFLOPs of the same schedule's vision half, 0 when this run measured nothing (--il-only)."""
    vis_arg = tuple(vis)
    for staged, unified, tag in ((False, False, "il"), (True, False, "ils"),
                                 (False, True, "ilu")):
        il = interleaved_from_rows(args.il_rows, ds_name, slug, args.groups, args.il_model,
                                   staged=staged, unified=unified)
        for keep, (tot, crit, n, path, pre, vscale) in sorted(il.items()):
            vis = vis_arg
            if keeps is not None and not any(abs(keep - k) < 1e-9 for k in keeps):
                continue
            suffix = f"_g{args.groups}" if keep == 1.0 else f"_g{args.groups}_k{keep:.2f}"
            # --il-only re-fold: keep the vision half an earlier hooked run stored for this
            # (or the sibling) arm rather than zeroing it
            prev = row.get(f"_il{suffix}") or row.get(f"_ils{suffix}") or {}
            if vis == (0.0, 0.0) and prev.get("vision_total"):
                vis = (prev["vision_total"], prev["vision_crit"])
            # The closed form is a mean over ALL accuracy rows (hundreds), the vision half a mean
            # over this run's --samples hooked rows (12, different prompts, different mean N).
            # Adding the two mixes samples: on V* it put the 35B unstaged k=0.50 total 5 pp of
            # the full pass too high (2026-09-10). So the decoder half is folded as a RATIO to
            # the stock prefill -- total/prefill and crit/prefill of the closed form, both
            # means over the same rows -- applied to this run's hooked LLM prefill of the
            # streaming arm (`total_g4* - vision_total`, the same 12 samples as the vision
            # half). The ratio is what the schedule fixes (1 + k f_img + f_text); it moved by
            # 0.002 between the 12 and the 191 V* rows. A decoder-only fold (--il-only with no
            # streaming arm in the json) has no hooked prefill and keeps the absolute closed form.
            llm = row.get(f"total{suffix}", 0.0) - vis[0] if vis[0] else 0.0
            if vscale is not None:
                # The unified arm's vision half is NOT the streaming arm's: its per-band
                # corrections run only over the tower layers the frontier had reached. The
                # closed form of both halves gives the ratio; applying it to the hooked vision
                # column keeps the measured basis (as the decoder half does) instead of mixing a
                # closed-form absolute into it. Critical comes out at ratio 1 by construction --
                # the last round always corrects at full tower depth -- which is the check that
                # the ratio is measuring what it claims. Not covered: the merger, ~0.25% of the
                # tower, which the unified arm calls once more than the streaming arm (the whole
                # image at the crossing) and once less than the interleaved arm.
                vis = (vis[0] * vscale["ratio_total"], vis[1] * vscale["ratio_crit"])
            ratio_t, ratio_c = tot / pre, crit / pre
            if llm > 0:
                dec_t, dec_c, basis = llm * ratio_t, llm * ratio_c, "hooked_prefill_x_ratio"
            else:
                dec_t, dec_c, basis = tot, crit, "closed_form_absolute"
            row[f"total{suffix}_{tag}"] = round(vis[0] + dec_t, 1)
            row[f"crit{suffix}_{tag}"] = round(vis[1] + dec_c, 1)
            row[f"_{tag}{suffix}"] = {
                "decoder_total": round(dec_t, 1), "decoder_crit": round(dec_c, 1),
                "vision_total": round(vis[0], 1), "vision_crit": round(vis[1], 1),
                "decoder_basis": basis, "hooked_llm_prefill": round(llm, 1),
                "closed_prefill": round(pre, 1), "closed_total": round(tot, 1),
                "closed_crit": round(crit, 1),
                "ratio_total": round(ratio_t, 4), "ratio_crit": round(ratio_c, 4),
                "rows": n, "source": os.path.basename(path),
                "note": "decoder half is the closed form of flops_analytic.Qwen35Decoder "
                        "replayed from the rows' `chunks`"
                        + (" (depth-staged: rows priced at their round's depth, approx pass + "
                           "frontier walks = one full prefill)" if staged else "")
                        + (" (unified axis: rows priced at their round's EXPLICIT depth over the "
                           "joint vision+decoder cost split)" if unified else "")
                        + ", as a ratio to the stock prefill applied to the hooked LLM prefill of"
                          " this run's streaming arm; vision half is this run's hooked streaming"
                          " arm (same samples)"
                        + (", rescaled by the closed-form unified/full-depth vision ratio"
                           if unified else "")}
            if vscale is not None:
                row[f"_{tag}{suffix}"]["vision_ratio"] = [round(vscale["ratio_total"], 4),
                                                          round(vscale["ratio_crit"], 4)]
            name = "unified" if unified else ("interleaved-staged" if staged else "interleaved")
            print(f"{ds_name:<14} k={keep:.2f} {name:<18} crit "
                  f"{row[f'crit{suffix}_{tag}']:8.1f} total {row[f'total{suffix}_{tag}']:9.1f}  "
                  f"(decoder {dec_c:.1f}/{dec_t:.1f} = x{ratio_c:.3f}/x{ratio_t:.3f} of the "
                  f"{basis} {llm if llm > 0 else pre:.1f}; closed form over {n} rows"
                  + (f"; vision x{vscale['ratio_total']:.3f}/x{vscale['ratio_crit']:.3f}"
                     if vscale is not None else "") + ")", flush=True)


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
                         "(vision half from the stored _split when the json has one, else decoder-side only)")
    args = ap.parse_args()

    if args.il_only:
        if not args.il_rows:
            raise SystemExit("--il-only needs --il-rows")
        result = json.load(open(args.out_json)) if os.path.exists(args.out_json) else {}
        slug = args.model.split("/")[-1].lower()
        for ds_name in args.datasets:
            row = result.setdefault(ds_name, {})
            # a hooked run that stored its per-arm stage split lets the fold use the same
            # vision half (and the hooked-prefill x ratio basis) as the hooked path above
            split = row.get("_split", {})
            for k in args.keeps:
                sp = split.get(f"streaming_k{k:.2f}")
                vis = tuple(sp[:2]) if sp else (0.0, 0.0)
                add_interleaved(row, args, ds_name, slug, vis, keeps=[k])
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
        # per-arm (vision_total, vision_crit, llm_total, llm_crit) so the decoder half can be
        # reconciled against the closed form (flops_analytic) without re-running the model.
        row["_split"] = {arm: [round(v, 1) for v in stage_split(c)] for arm, c in counters.items()}
        result[ds_name] = row
        # write after every dataset so a timeout keeps the finished ones
        os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
        json.dump(result, open(args.out_json, "w"), indent=1)
        print(f"wrote {args.out_json} ({len(result)} datasets)", flush=True)


if __name__ == "__main__":
    main()
