"""GPU gates for the unified (vision + decoder) depth-staged schedule -- memo §7.12.

Two modes, because the schedule changes two things that are gated differently.

`--mode driver` (one GPU, `--load vision`, NO engine, no server): the tower is real and the LLM
side is a `RecordingSink`, so what is gated is exactly what the driver decides -- where the
bounds fall on real prompts, which rounds reach the LLM, what the messages carry, and the
feature-space fidelity of the image rows the LLM would consume.

  U1  g=1 identity: `unified_staged` reproduces `interleaved` and `interleaved_staged` BITWISE on
      every pushed prompt position. At g=1 the tower is walked to full depth before the opening
      push and one round corrects everything, so the two schedules are the same computation --
      expect equality, not "close" (contract gate 2). Also checked at keep=0.5 against the EAGER
      pscore arm: the unified score is the full-tower layer mean at g=1, which is the eager
      arm's score, while the campaign default defers it past band 0.
  U2  structure at g=4: one opening push, one `correct` per LLM round (fewer than `groups`), all
      positions inside their window and below the hold-back, the text suffix only in the last
      message, every merge group corrected at most once, the keep budget met.
  U3  G5-analogue, feature space. At g>1 the unified arm is NOT bitwise-equal to the interleaved
      one and MUST NOT be gated as if it were: band r is corrected over the tower prefix walked
      so far and then carried through the remaining layers, whose K/V the approximate walk
      recomputes from the partly corrected stream -- a different (and strictly better-informed)
      computation, not a re-addressing of the same one. What is reported instead is each arm's
      relative L2 against the CEILING image rows (the stock full-resolution tower), computed
      here from the tower rather than read out of any arm under test. The floor is printed as
      the scale. HYPOTHESIS to be judged from these numbers, not asserted by them: unified <=
      interleaved_staged <= streaming at equal keep.

`--mode served` (needs a running `appcorr.vllm_stream.server --interleaved`): runs the accuracy
driver's arms through the engine on a few images and compares predictions row by row --
`ceiling` (stock full-resolution one-shot prefill), `interleaved_staged` and `unified_staged` at
keep=1, plus `floor` for the scale. keep=1 g=4 is expected to agree with the ceiling on the first
token; disagreements are read against the staged form's own band (memo §7.11's g6: 8/8 argmax,
first-token |dlogprob| 3.9e-5 on 35B), not against an absolute constant.

Commands are in the module docstring of `docs/memo/vllm_interleaved_design.md` §7.12.
"""
import argparse, json, os, subprocess, sys, time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image
from analysis.experiments.qwen35_accuracy import degrade
from analysis.experiments.qwen_vllm_accuracy import make_axis
from analysis.experiments.vllm_interleaved_axis_gate import RecordingSink


def run(axis, inputs, px_base, groups, keep, schedule, eager_pscore=False):
    sink = RecordingSink(f"{schedule}-g{groups}-k{keep}")
    ids = inputs["input_ids"][0]
    pos = (ids == axis.image_token_id).nonzero(as_tuple=True)[0]
    layout = {"image_run": (int(pos[0]), int(pos.numel())),
              "grid_thw": tuple(int(v) for v in inputs["image_grid_thw"][0].tolist())}
    old = axis.pscore_defer
    if eager_pscore:
        axis.pscore_defer = False
    try:
        _, _, st = axis.streaming_forward(inputs, px_base, groups, keep=keep, sink=sink,
                                          llm_schedule=schedule, **layout)
    finally:
        axis.pscore_defer = old
    return sink, st


@torch.no_grad()
def ceiling_rows(axis, inputs):
    """The stock full-resolution image rows -- recomputed here, never read out of an arm: a
    harness that shares the quantity under test deletes that axis (contract, `how the gates
    missed it`)."""
    feats = axis.model.model.get_image_features(
        inputs["pixel_values"].to(axis.model.dtype), inputs["image_grid_thw"])
    feats = feats.pooler_output if hasattr(feats, "pooler_output") else feats
    if isinstance(feats, (list, tuple)):
        feats = torch.cat(list(feats), dim=0)
    return feats.float().cpu()


def rel_l2(a, b):
    return float((a - b).pow(2).sum().sqrt() / b.pow(2).sum().sqrt().clamp_min(1e-12))


def gate_driver(a):
    from transformers import AutoProcessor, AutoModelForImageTextToText
    proc = AutoProcessor.from_pretrained(a.model)
    if a.load == "vision":
        from appcorr.models.vision_only import load_vision_only
        model = load_vision_only(a.model, device="cuda:0")
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            a.model, dtype="auto", device_map="cuda:0").eval()
    axis = make_axis(a.family, model, proc)

    if a.dataset:
        from qwen_vl_prefill.datasets_eval import get_spec
        from datasets import load_dataset
        spec = get_spec(a.dataset)
        ds = spec.load(load_dataset)
        samples = [spec.prepare(ds[int(i)], lambda h, w, **kw: (h, w), 1, 1, 1 << 30)[:2]
                   for i in range(0, len(ds), max(1, len(ds) // a.samples))[:a.samples]]
    else:
        g = torch.Generator().manual_seed(0)
        arr = (torch.rand(a.size[1], a.size[0], 3, generator=g) * 255).to(torch.uint8).numpy()
        samples = [(Image.fromarray(arr, "RGB"), "What is in this image?")] * a.samples

    report, fails = [], []
    for si, (img, q) in enumerate(samples):
        img = img.convert("RGB")
        base = degrade(img, a.level, a.degrade_filter)
        inputs = axis.build_inputs(img, q).to("cuda:0")
        px_base = axis.build_inputs(base, q)["pixel_values"].to("cuda:0")
        seq = int(inputs["input_ids"].shape[1])
        ids = inputs["input_ids"][0]
        pos = (ids == axis.image_token_id).nonzero(as_tuple=True)[0]
        lo, n_tok = int(pos[0]), int(pos.numel())
        row = {"i": si, "seq": seq, "image_run": [lo, n_tok]}

        # --- U1: g=1 identity ------------------------------------------------------------- #
        row["u1"] = {}
        for keep, eager in ((1.0, False), (0.5, True)):
            u, _ = run(axis, inputs, px_base, 1, keep, "unified_staged")
            for other in ("interleaved", "interleaved_staged"):
                o, _ = run(axis, inputs, px_base, 1, keep, other, eager_pscore=eager)
                x, y = u.final_embeds(seq), o.final_embeds(seq)
                ok = bool(torch.equal(x.view(torch.int16), y.view(torch.int16))) \
                    if x.dtype == torch.bfloat16 else bool(torch.equal(x, y))
                row["u1"][f"k{keep}_{other}"] = {
                    "bitwise": ok, "max_abs": float((x.float() - y.float()).abs().max())}
                if not ok:
                    fails.append(f"U1 sample {si} keep={keep} vs {other}")

        # --- U2: structure at g=4 --------------------------------------------------------- #
        row["u2"] = {}
        for keep in a.keeps:
            sink, st = run(axis, inputs, px_base, a.groups, keep, "unified_staged")
            bounds = st["unified_bounds"]
            n_vis = len(axis.tower.blocks)
            llm_rounds = [r for r in range(a.groups) if bounds[r] > n_vis]
            corr = [m for m in sink.msgs if m["kind"] == "correct"]
            sel = torch.cat([g_.to("cpu") for g_ in st["group_idx"]])
            d = {
                "bounds": bounds, "llm_bounds": st["unified_llm_bounds"],
                "n_llm_rounds": len(llm_rounds), "n_correct": len(corr),
                "one_push": sink.msgs[0]["kind"] == "push" and not sink.msgs[0]["final"]
                            and int(sink.msgs[0]["embeds"].shape[0]) == seq,
                "rounds_match": len(corr) == len(llm_rounds),
                "last_final": bool(corr and corr[-1]["final"])
                              and not any(m["final"] for m in corr[:-1]),
                "in_window": all(int(m["positions"][0]) >= m["window"][0]
                                 and int(m["positions"][-1]) < m["window"][1] <= seq - 1
                                 for m in corr),
                "text_only_last": all(int(m["positions"][-1]) < lo + n_tok for m in corr[:-1]),
                "each_group_once": int(sel.unique().numel()) == int(sel.numel()),
                "budget": [int(sel.numel()),
                           n_tok if keep >= 1.0 else max(1, round(keep * n_tok))],
                "stages": [m["stage"] for m in corr],
                "chunks": [list(c) for c in st["chunks"]],
            }
            d["PASS"] = all(v for k_, v in d.items() if isinstance(v, bool)) \
                and d["rounds_match"] and d["budget"][0] == d["budget"][1]
            row["u2"][f"k{keep}"] = d
            if not d["PASS"]:
                fails.append(f"U2 sample {si} keep={keep}")

        # --- U3: feature-space distance to the ceiling ------------------------------------ #
        ref = ceiling_rows(axis, inputs)
        row["u3"] = {}
        for schedule in ("streaming", "interleaved_staged", "unified_staged"):
            for keep in a.keeps:
                sink, _ = run(axis, inputs, px_base, a.groups, keep, schedule)
                row["u3"][f"{schedule}_k{keep}"] = rel_l2(
                    sink.final_embeds(seq)[lo:lo + n_tok].float(), ref)
        # the floor's own distance, as the scale the three arms are read against
        base_rows = axis.model.model.get_image_features(
            px_base.to(axis.model.dtype), inputs["image_grid_thw"])
        base_rows = base_rows.pooler_output if hasattr(base_rows, "pooler_output") else base_rows
        if isinstance(base_rows, (list, tuple)):
            base_rows = torch.cat(list(base_rows), dim=0)
        row["u3"]["floor"] = rel_l2(base_rows.float().cpu(), ref)
        report.append(row)
        print(f"[{si}] seq {seq} img {n_tok} bounds "
              f"{row['u2'][f'k{a.keeps[0]}']['bounds']} u1 "
              f"{ {k: v['bitwise'] for k, v in row['u1'].items()} } u3 "
              f"{ {k: round(v, 5) for k, v in row['u3'].items()} }", flush=True)

    out = {"_model": a.model, "_mode": "driver", "_groups": a.groups, "_keeps": a.keeps,
           "rows": report, "fails": fails, "PASS": not fails}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1, default=str)
    print(f"wrote {a.out}: {'UNIFIED_DRIVER_PASS' if out['PASS'] else 'UNIFIED_DRIVER_FAIL'}")
    if fails:
        for f in fails:
            print("  " + f)
        raise SystemExit(1)


def gate_served(a):
    """Run the accuracy driver's arms through a live server and compare predictions per row."""
    out_dir = a.out.replace(".json", "_rows")
    os.makedirs(out_dir, exist_ok=True)
    slug = a.model.split("/")[-1].lower()
    env = dict(os.environ, PYTHONPATH=ROOT, HF_HUB_OFFLINE=os.environ.get("HF_HUB_OFFLINE", "1"))
    base = [sys.executable, os.path.join(ROOT, "analysis", "experiments", "qwen_vllm_accuracy.py"),
            "--family", a.family, "--model", a.model, "--port", str(a.port),
            "--dataset", a.dataset or "vstar", "--degrade-filter", a.degrade_filter,
            "--level", str(a.level), "--groups", str(a.groups), "--samples", str(a.samples),
            "--concurrency", "1", "--load", "vision", "--out", out_dir]
    arms = [("ceiling", "streaming", 1.0, ""), ("floor", "streaming", 1.0, "")]
    for k in a.keeps:
        arms += [("streaming", "interleaved_staged", k, ""), ("streaming", "unified_staged", k, ""),
                 # the unified form with the engine's stock prefill instead of the open walk:
                 # the two must agree row for row (same state by construction)
                 ("streaming", "unified_staged", k, "noow")]
    paths = {}
    for arm, sched, keep, variant in arms:
        odir = out_dir + (f"_{variant}" if variant else "")
        os.makedirs(odir, exist_ok=True)
        cmd = [*base[:-1], odir, "--arms", arm, "--llm-schedule", sched, "--keep", f"{keep}"]
        if variant == "noow":
            cmd.append("--no-open-walk")
        tag = arm if arm != "streaming" else (
            "interleaved_unified" if sched == "unified_staged" else sched)
        suf = "" if arm != "streaming" else f"_g{a.groups}" + (f"_k{keep:.2f}" if keep < 1 else "")
        p = os.path.join(odir, f"{a.dataset or 'vstar'}_{slug}_{tag}{suf}.jsonl")
        if variant:
            tag = f"{tag}_{variant}"
        if os.path.exists(p):
            os.remove(p)
        t0 = time.time()
        rc = subprocess.call(cmd, env=env, cwd=ROOT)
        print(f"  {tag}{suf:16s} rc={rc} {time.time() - t0:5.0f}s", flush=True)
        paths[f"{tag}{suf}"] = p

    rows = {}
    for key, p in paths.items():
        if not os.path.exists(p):
            continue
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            if "skip" in r:
                continue
            rows.setdefault(int(r["i"]), {})[key] = r.get("pred")
    ref = "ceiling"
    summary = {}
    for key in paths:
        if key == ref:
            continue
        agree = sum(1 for v in rows.values() if ref in v and key in v and v[ref] == v[key])
        n = sum(1 for v in rows.values() if ref in v and key in v)
        summary[key] = [agree, n]
        print(f"{key:34s} agrees with ceiling on {agree}/{n}")
    for key in paths:
        if "_noow" not in key:
            continue
        base_key = key.replace("_noow", "")
        agree = sum(1 for v in rows.values() if base_key in v and key in v and v[base_key] == v[key])
        n = sum(1 for v in rows.values() if base_key in v and key in v)
        summary[f"{base_key}==open_walk_off"] = [agree, n]
        print(f"{base_key:34s} open walk on == off on {agree}/{n}"
              f"{'' if agree == n else '  <-- MISMATCH'}")
    json.dump({"_mode": "served", "_model": a.model, "summary": summary,
               "rows": {str(k): v for k, v in rows.items()}},
              open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["driver", "served"], default="driver")
    ap.add_argument("--family", choices=["qwen25vl", "qwen35"], default="qwen35")
    ap.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    ap.add_argument("--port", type=int, default=5591)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--keeps", type=float, nargs="+", default=[1.0, 0.5])
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--degrade-filter", choices=["bicubic", "box", "pyr"], default="pyr")
    ap.add_argument("--dataset", default=None, help="a real dataset instead of the synthetic image")
    ap.add_argument("--size", type=int, nargs=2, default=[896, 896])
    ap.add_argument("--load", choices=["full", "vision"], default="vision")
    ap.add_argument("--out", default="analysis/results/vllm_stream/unified_gate.json")
    a = ap.parse_args()
    (gate_driver if a.mode == "driver" else gate_served)(a)


if __name__ == "__main__":
    main()
