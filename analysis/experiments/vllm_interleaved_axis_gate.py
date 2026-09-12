"""G5: the driver half of the interleaved schedule, gated without an engine.

`streaming_forward(llm_schedule="interleaved")` sends the same vision output as the streaming
schedule, only addressed differently: one push of the whole approximate prompt at t=0 and then a
`correct` per band that REWRITES that band's rows. So the quantity to gate is what the LLM ends
up holding at every prompt position -- reconstruct it from the messages of each schedule and
demand they are BITWISE equal at keep=1 (docs/memo/vllm_interleaved_design.md §3.5).

Bitwise, not close: a corrected row IS the LLM's input, and the two schedules run the identical
merger call for a band, so anything but equality means the interleaved branch addressed a row
wrongly -- which is the failure the interleaved contract says every fork makes.

What is checked (all on a `RecordingSink`, no vLLM import, no server):

  G5a  keep=1: per-position final embeddings, interleaved vs streaming -> bitwise equal.
  G5b  the approx push's image rows == `merger(x_base_out)` computed INDEPENDENTLY here (a second
       base pass through the tower), not read out of the axis: the harness must not share the
       quantity under test.
  G5c  message structure: one push + one correct per non-empty band, the last one final; every
       correct's positions lie inside its window and below the held-back row seq-1; the text
       suffix appears only in the last message.
  G5d  keep=0.5: P_r is a subset of band r of size `corrected_groups`/round, coverage over the
       rounds equals the streaming arm's corrected set, and the reconstructed embeddings still
       match the streaming schedule position by position (reported, asserted at rel 0).

Two ways to run it. `--tiny` builds the randomly initialised CPU models of
`qwen_axis_cpu_unittest.py` (real HF classes, real forks, fp32) and needs no GPU and no weights --
the addressing logic this branch adds is shape logic, so that is where it is cheapest to gate:

  PYTHONPATH=$PWD python analysis/experiments/vllm_interleaved_axis_gate.py --tiny \\
      --family qwen35 --groups 4 --keeps 1.0 0.5

Without `--tiny` it runs the real tower (bf16, a real image, `--load vision` loads only the tower
+ embed_tokens), which is GPU work:

  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTHONPATH=$PWD \\
  python analysis/experiments/vllm_interleaved_axis_gate.py --family qwen35 \\
      --model Qwen/Qwen3.5-35B-A3B --groups 4 --keeps 1.0 0.5
"""
import argparse, json, os, sys
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from PIL import Image
from analysis.experiments.qwen35_accuracy import degrade
from analysis.experiments.qwen_vllm_accuracy import make_axis


class RecordingSink:
    """Quacks like `StreamSink` for the axis and keeps every message's tensors.

    `push` carries no positions (the streaming schedule's chunks are consecutive by construction,
    and the interleaved schedule's single push is the whole prompt), so they are reconstructed
    from a running offset -- the same rule the server applies when it appends a chunk.
    """

    def __init__(self, rid="gate"):
        self.rid, self.opened, self.closed, self.error = rid, False, False, None
        self.msgs = []
        self._off = 0

    def push(self, embeds, mrope, mrope_delta, final, *, correct_from=None, correct_to=None,
             open_walk=None):
        assert not self.closed, "push after final"
        assert correct_from is None or not self.opened, "correct_from only on the opening push"
        n = int(embeds.shape[0])
        self.msgs.append({"kind": "push", "final": bool(final), "window": None,
                          "correct_from": correct_from, "correct_to": correct_to,
                          "open_walk": open_walk,
                          "positions": torch.arange(self._off, self._off + n),
                          "embeds": embeds.detach().to("cpu").clone(),
                          "mrope": None if mrope is None else mrope.detach().to("cpu").clone()})
        self._off += n
        self.opened = True
        self.closed = bool(final)

    def correct(self, positions, embeds, window, final, *, stage=None):
        assert self.opened, "correct before the prompt was pushed"
        assert not self.closed, "correct after final"
        assert positions.shape[0] == embeds.shape[0], (positions.shape, embeds.shape)
        self.msgs.append({"kind": "correct", "final": bool(final), "stage": stage,
                          "window": (int(window[0]), int(window[1])),
                          "positions": positions.detach().to("cpu").clone(),
                          "embeds": embeds.detach().to("cpu").clone(), "mrope": None})
        self.closed = bool(final)

    def result(self):
        raise RuntimeError("RecordingSink has no engine behind it")

    # -- reconstruction -----------------------------------------------------------------------
    def final_embeds(self, seq: int) -> torch.Tensor:
        """What the LLM holds at every prompt position after the last message: the pushes lay the
        prompt down, each correct overwrites its own rows (later message wins, which is the
        schedule's own semantics)."""
        out = None
        for m in self.msgs:
            if out is None:
                out = torch.zeros(seq, m["embeds"].shape[1], dtype=m["embeds"].dtype)
                seen = torch.zeros(seq, dtype=torch.bool)
            out[m["positions"]] = m["embeds"]
            seen[m["positions"]] = True
        assert bool(seen.all()), f"{int((~seen).sum())} prompt positions were never sent"
        return out


def run(axis, inputs, px_base, groups, keep, schedule):
    sink = RecordingSink(f"{schedule}-k{keep}")
    ids = inputs["input_ids"][0]
    pos = (ids == axis.image_token_id).nonzero(as_tuple=True)[0]
    layout = {"image_run": (int(pos[0]), int(pos.numel())),
              "grid_thw": tuple(int(v) for v in inputs["image_grid_thw"][0].tolist())}
    _, _, st = axis.streaming_forward(inputs, px_base, groups, keep=keep, sink=sink,
                                      llm_schedule=schedule, **layout)
    return sink, st


@torch.no_grad()
def merger_of_base(axis, inputs, px_base, bands):
    """`merger(x_base_out)` for every group, recomputed here from the tower rather than read out
    of the axis run under test (a harness that shares the quantity deletes that axis)."""
    dev = inputs["pixel_values"].device
    grid = inputs["image_grid_thw"]
    px_full = inputs["pixel_values"].to(axis.model.dtype)
    gctx = axis.tower.prepare_grid(grid, dev)
    ctx_full = axis.tower.prepare_full_tokens(px_full, grid, gctx)
    ctx_base = axis.tower.prepare_full_tokens(px_base.to(dev, px_full.dtype), grid, gctx)
    x_base_out, _ = axis._approx_base(ctx_base, {}, collect_attn=False)
    out = []
    for g0, g1 in bands:                      # band-sized merger calls, as the axis makes them
        if g1 <= g0:
            continue
        rows = axis._rows_of_groups(ctx_full, torch.arange(g0, g1, device=dev))
        out.append(axis.tower.merger(x_base_out[rows]))
    return torch.cat(out).to("cpu")


def check(axis, inputs, px_base, groups, keep, report):
    seq = int(inputs["input_ids"].shape[1])
    ids = inputs["input_ids"][0]
    pos = (ids == axis.image_token_id).nonzero(as_tuple=True)[0]
    lo, n_tok = int(pos[0]), int(pos.numel())
    bands = axis._bands(groups, n_tok)

    s_sink, s_st = run(axis, inputs, px_base, groups, keep, "streaming")
    i_sink, i_st = run(axis, inputs, px_base, groups, keep, "interleaved")
    row = {"keep": keep, "groups": groups, "seq": seq, "image_run": [lo, n_tok],
           "streaming_chunks": [list(c) for c in s_st["chunks"]],
           "interleaved_chunks": [list(c) for c in i_st["chunks"]],
           "prefill_tokens": {"streaming": s_st["prefill_tokens"],
                              "interleaved": i_st["prefill_tokens"]},
           "corrected_groups": {"streaming": int(s_st["corrected_groups"]),
                                "interleaved": int(i_st["corrected_groups"])}}

    # G5a / G5d -- what the LLM holds, position by position
    a, b = s_sink.final_embeds(seq), i_sink.final_embeds(seq)
    diff = (a.float() - b.float()).abs()
    n_bad = int((a.view(torch.int16) != b.view(torch.int16)).any(dim=1).sum())
    row["bitwise_equal"] = n_bad == 0
    row["n_positions_differing"] = n_bad
    row["max_abs_diff"] = float(diff.max())
    row["rel_l2"] = float(diff.pow(2).sum().sqrt() / a.float().pow(2).sum().sqrt().clamp_min(1e-12))

    # G5b -- the approx push's image rows
    approx = i_sink.msgs[0]
    row["approx_push"] = {"kind": approx["kind"], "n": int(approx["embeds"].shape[0]),
                          "final": approx["final"]}
    ref = merger_of_base(axis, inputs, px_base, bands).to(approx["embeds"].dtype)
    got = approx["embeds"][lo:lo + n_tok]
    row["approx_image_rows_bitwise"] = bool(torch.equal(got.view(torch.int16),
                                                        ref.view(torch.int16)))
    row["approx_image_rows_max_abs"] = float((got.float() - ref.float()).abs().max())

    # G5c -- message structure
    corr = [m for m in i_sink.msgs if m["kind"] == "correct"]
    nonempty = [(g0, g1) for g0, g1 in bands if g1 > g0]
    row["n_messages"] = len(i_sink.msgs)
    row["n_correct"] = len(corr)
    structure = []
    for k, m in enumerate(corr):
        p = m["positions"]
        s, e = m["window"]
        last = k == len(corr) - 1
        img = p[p < lo + n_tok]
        txt = p[p >= lo + n_tok]
        g0, g1 = nonempty[k] if k < len(nonempty) else nonempty[-1]
        structure.append({
            "window": [s, e], "n": int(p.numel()), "n_img": int(img.numel()),
            "n_text": int(txt.numel()), "final": m["final"],
            "in_window": bool(p.numel() == 0 or (int(p[0]) >= s and int(p[-1]) < e)),
            "below_holdback": bool(p.numel() == 0 or int(p[-1]) < seq - 1),
            "sorted": bool(p.numel() < 2 or bool((p[1:] > p[:-1]).all())),
            "img_in_band": bool(img.numel() == 0 or (int(img[0]) >= lo + g0
                                                     and int(img[-1]) < lo + g1)),
            "text_only_last": bool(txt.numel() == 0 or last)})
    row["correct_messages"] = structure
    row["ok_structure"] = all(all(v for k_, v in d.items() if isinstance(v, bool)
                                  and k_ != "final") for d in structure) \
        and bool(corr and corr[-1]["final"]) and not any(m["final"] for m in corr[:-1]) \
        and i_sink.msgs[0]["kind"] == "push" and not i_sink.msgs[0]["final"]

    # coverage: the two schedules must correct the SAME groups (contract rule 5)
    s_sel = torch.cat([g.to("cpu") for g in s_st["group_idx"]]).sort().values
    i_sel = torch.cat([g.to("cpu") for g in i_st["group_idx"]]).sort().values
    row["same_selection"] = bool(torch.equal(s_sel, i_sel))
    # ...and must have selected them with the SAME score. The deferred keep<1 score completes
    # after the first band's message; the interleaved branch has no `pos_done` to key that off,
    # and an arm that ranks band 1.. on energy alone while the streaming arm ranks on
    # energy x attention is a rule-5 (shared selection) break that `same_selection` can miss by
    # luck on a small or random-weight model.
    row["pscore"] = {"streaming": s_st.get("pscore"), "interleaved": i_st.get("pscore")}
    row["same_pscore"] = s_st.get("pscore") == i_st.get("pscore")
    row["n_selected"] = int(i_sel.numel())
    corrected_pos = torch.cat([m["positions"] for m in corr])
    row["coverage_ok"] = bool(torch.equal(
        corrected_pos[corrected_pos < lo + n_tok].sort().values, (lo + i_sel).sort().values))

    row["PASS"] = bool(row["ok_structure"] and row["same_selection"] and row["same_pscore"]
                       and row["coverage_ok"]
                       and row["approx_image_rows_bitwise"]
                       and (row["bitwise_equal"] if keep >= 1.0 else True))
    report.append(row)
    print(f"keep={keep:.2f} g={groups}  bitwise={row['bitwise_equal']} "
          f"(bad {n_bad}/{seq}, max|d| {row['max_abs_diff']:.3g}, rel {row['rel_l2']:.3g})  "
          f"approx_rows={row['approx_image_rows_bitwise']}  structure={row['ok_structure']}  "
          f"selection={row['same_selection']} pscore={row['pscore']['interleaved']} "
          f"coverage={row['coverage_ok']}  "
          f"prefill {row['prefill_tokens']}  -> {'PASS' if row['PASS'] else 'FAIL'}", flush=True)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["qwen25vl", "qwen35", "glm46v"], default="qwen35")
    ap.add_argument("--model", default="Qwen/Qwen3.5-35B-A3B")
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--keeps", type=float, nargs="+", default=[1.0, 0.5])
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--degrade-filter", choices=["bicubic", "box", "pyr"], default="box")
    ap.add_argument("--dataset", default=None,
                    help="pull one real sample instead of the synthetic image")
    ap.add_argument("--size", type=int, nargs=2, default=[672, 672], help="synthetic image size")
    ap.add_argument("--load", choices=["full", "vision"], default="vision")
    ap.add_argument("--tiny", action="store_true",
                    help="randomly initialised CPU model + synthetic patches "
                         "(qwen_axis_cpu_unittest.tiny_models): no GPU, no weights, ~10 s")
    ap.add_argument("--grid", type=int, nargs=2, default=[8, 12], help="--tiny patch grid (h, w)")
    ap.add_argument("--out", default="analysis/results/vllm_stream/interleaved_axis_gate.json")
    a = ap.parse_args()

    if a.tiny:
        from analysis.experiments.qwen_axis_cpu_unittest import make_inputs, tiny_models
        model = tiny_models()[a.family]
        axis = make_axis(a.family, model, None)
        inputs, px_base = make_inputs(model, a.grid[0], a.grid[1], 0)
        report = []
        for keep in a.keeps:
            check(axis, inputs, px_base, a.groups, keep, report)
        out = {"_model": f"tiny-{a.family}", "_family": a.family, "_groups": a.groups,
               "_source": f"synthetic patches, grid {a.grid[0]}x{a.grid[1]}", "rows": report,
               "PASS": all(r["PASS"] for r in report)}
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(out, open(a.out.replace(".json", "_tiny.json"), "w"), indent=1)
        print(f"wrote {a.out.replace('.json', '_tiny.json')}: "
              f"{'G5_PASS' if out['PASS'] else 'G5_FAIL'}")
        return

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
        img, q, _ = spec.prepare(ds[0], lambda h, w, **kw: (h, w), 1, 1, 1 << 30)
        img = img.convert("RGB")
    else:
        g = torch.Generator().manual_seed(0)
        arr = (torch.rand(a.size[1], a.size[0], 3, generator=g) * 255).to(torch.uint8).numpy()
        img, q = Image.fromarray(arr, "RGB"), "What is in this image?"

    base = degrade(img, a.level, a.degrade_filter)
    inputs = axis.build_inputs(img, q).to("cuda:0")
    px_base = axis.build_inputs(base, q)["pixel_values"].to("cuda:0")

    report = []
    for keep in a.keeps:
        check(axis, inputs, px_base, a.groups, keep, report)
    out = {"_model": a.model, "_family": a.family, "_groups": a.groups,
           "_source": a.dataset or f"synthetic {a.size[0]}x{a.size[1]}", "rows": report,
           "PASS": all(r["PASS"] for r in report)}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}: {'G5_PASS' if out['PASS'] else 'G5_FAIL'}")


if __name__ == "__main__":
    main()
