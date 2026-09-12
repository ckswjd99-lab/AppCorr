"""CPU gate for the unified (vision + decoder) depth-staged schedule -- memo §7.12.

No GPU, no weights, no engine: everything this schedule adds is shape logic (where the bounds
fall, which rounds reach the LLM, what goes on the wire, what the cost script is handed), and
shape logic is cheapest to gate on a randomly initialised tiny model.

    PYTHONPATH=$PWD /home/nxclab/anaconda3/envs/appcorr/bin/python \\
        analysis/experiments/test_unified_axis.py
    # optional: also gate the three EXISTING schedules bitwise against another tree
    ... analysis/experiments/test_unified_axis.py --ref-root /NHNHOME/share/cjpark/AppCorr-il-engine

What is checked:

  B1  bounds are strictly increasing, inside [1, n_stages], `groups` of them, last == n_stages,
      at g in {1,2,4,8} on the real 35B / 122B / 4B configs and two prompt shapes each.
  B2  the axis's per-stage cost model reconciles with the table's closed form: the decoder half
      summed over its layers IS `Qwen35Decoder.prefill_flops(N)` (i.e. the axis and
      `flops_analytic` price the same stage), and one vision stage IS
      `Qwen35Vision.layer_flops(n_rows)`.
  B3  an ALL-DECODER axis (a tower of zero cost) reproduces `stage_bounds(L, g)` up to the
      hybrid layer mix -- the check that the cost split degenerates to the equal-layer form
      when the layers really are equal.
  W1  wire round-trip of `stage`: `[r, g]` survives unchanged (byte-for-byte the pre-existing
      form) and `[r, g, [b...]]` survives through a real `Frame` encode/decode.
  W2  the engine's `stage_spec` derives the same bounds from `(r, g)` as before and takes
      explicit ones when given, rejecting non-monotone / short / wrong-last bounds.
  A1  the driver's walk on a tiny real Qwen3.5: message structure, per-round depths, the
      `chunks` records, coverage, and the closed-form replay of both halves.
  A2  g=1 identity: `unified_staged` == `interleaved` and `interleaved_staged` bitwise on the
      pushed embeddings, at keep=1 and (with the eager pscore) at keep=0.5.
  R1  (--ref-root) streaming / interleaved / interleaved_staged unchanged against a reference
      tree -- the "every new behaviour is behind the new schedule name" claim.
"""
import argparse, importlib, os, sys, tempfile, types

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

torch.manual_seed(0)
torch.set_num_threads(8)

MODEL_IDS = {"qwen35_35b": "Qwen/Qwen3.5-35B-A3B",
             "qwen35_122b": "Qwen/Qwen3.5-122B-A10B-FP8",
             "qwen35_4b": "Qwen/Qwen3.5-4B"}

FAILS = []


def check(name, cond, detail=""):
    ok = bool(cond)
    if not ok:
        FAILS.append(f"{name}: {detail}")
    print(f"  [{'ok ' if ok else 'FAIL'}] {name}{(' -- ' + detail) if detail and not ok else ''}")
    return ok


# --------------------------------------------------------------------------------------------- #
# shims: the bounds/cost hooks need only the config and the tower's layer COUNT, so the real
# 35B/122B entries can be gated on CPU without a checkpoint.
# --------------------------------------------------------------------------------------------- #

def axis_shim(model_id):
    from transformers import AutoConfig
    from appcorr.models.qwen35.unified import Qwen35Axis
    cfg = AutoConfig.from_pretrained(model_id)
    ax = Qwen35Axis.__new__(Qwen35Axis)
    ax.cfg = cfg
    ax.tower = types.SimpleNamespace(blocks=[None] * int(cfg.vision_config.depth))
    return ax


def import_engine_correct():
    """`appcorr.vllm_stream.correct` without vLLM: the module's only vllm import is the runner
    class it type-annotates with, so a stub is enough to reach `stage_spec` / `stage_bounds` on
    CPU. The rest of the file is gated on the GPU (`vllm_unified_gate.py`)."""
    for name in ("vllm", "vllm.v1", "vllm.v1.worker", "vllm.v1.worker.gpu_model_runner"):
        if name not in sys.modules:
            sys.modules[name] = types.ModuleType(name)
    sys.modules["vllm.v1.worker.gpu_model_runner"].GPUModelRunner = type("GPUModelRunner", (), {})
    return importlib.import_module("appcorr.vllm_stream.correct")


def import_ref(ref_root):
    d = tempfile.mkdtemp(prefix="appcorr_ref_")
    os.symlink(os.path.join(ref_root, "appcorr"), os.path.join(d, "appcorr_ref"))
    sys.path.insert(0, d)
    return importlib.import_module("appcorr_ref.models.qwen35.unified").Qwen35Axis


# --------------------------------------------------------------------------------------------- #
# B -- bounds and the cost model
# --------------------------------------------------------------------------------------------- #

SHAPES = [("V*-like", 1000, 835), ("TextVQA-like", 500, 445)]


def test_bounds():
    print("B1/B2/B3  bounds + cost model (real configs, CPU)")
    from flops_analytic import MODELS35, QWEN35_VISION, Qwen35Vision
    for key, mid in MODEL_IDS.items():
        ax = axis_shim(mid)
        dec = MODELS35[key]
        n_vis = len(ax.tower.blocks)
        n_llm = int(ax.cfg.text_config.num_hidden_layers)
        for label, N, n_img in SHAPES:
            n_rows = 4 * n_img
            # B2 -- the axis and the table price the same stage
            costs = ax._llm_stage_costs(N)
            check(f"B2 {key} {label} decoder layers sum == prefill_flops(N)",
                  abs(sum(costs) / dec.prefill_flops(N) - 1) < 1e-12,
                  f"{sum(costs):.6e} vs {dec.prefill_flops(N):.6e}")
            v = ax.cfg.vision_config
            vis = Qwen35Vision(layers=int(v.depth), hidden=int(v.hidden_size),
                               heads=int(v.num_heads), ffn=int(v.intermediate_size))
            check(f"B2 {key} {label} vision stage == Qwen35Vision.layer_flops",
                  ax._vision_stage_cost(n_rows) == vis.layer_flops(n_rows),
                  f"{ax._vision_stage_cost(n_rows)} vs {vis.layer_flops(n_rows)}")
            # the table's frozen entry is the tower the 35B and the 122B SHARE; the 4B's is its
            # own (24 layers, h1024), which is exactly why the axis reads the config
            check(f"B2 {key} frozen QWEN35_VISION matches the config",
                  (vis == QWEN35_VISION) == (key != "qwen35_4b"), f"{vis} vs {QWEN35_VISION}")
            check(f"B2 {key} {label} layer count",
                  len(costs) == n_llm and len(ax.unified_stage_costs(n_rows, N)) == n_vis + n_llm)
            # B1 -- the bounds themselves
            for g in (1, 2, 4, 8):
                b = ax.unified_bounds(g, n_rows, N)
                check(f"B1 {key} {label} g={g}",
                      len(b) == g and b[-1] == n_vis + n_llm
                      and all(1 <= x <= n_vis + n_llm for x in b)
                      and all(x < y for x, y in zip(b, b[1:])), str(b))
    # B3 -- a zero-cost tower degenerates to the equal-COST decoder split; with a uniform
    # decoder that is the equal-LAYER split `stage_bounds` uses.
    ax = axis_shim(MODEL_IDS["qwen35_35b"])
    n_llm = int(ax.cfg.text_config.num_hidden_layers)
    ax.tower = types.SimpleNamespace(blocks=[])
    uniform = types.MethodType(lambda self, n: [1.0] * n_llm, ax)
    ax._llm_stage_costs = uniform
    from flops_analytic import stage_bounds
    for g in (1, 2, 4, 5, 8):
        check(f"B3 all-decoder uniform g={g} == stage_bounds",
              ax.unified_bounds(g, 1, 1) == stage_bounds(n_llm, g),
              f"{ax.unified_bounds(g, 1, 1)} vs {stage_bounds(n_llm, g)}")


# --------------------------------------------------------------------------------------------- #
# W -- the wire field and the engine's reader
# --------------------------------------------------------------------------------------------- #

def test_wire():
    print("W1/W2  wire `stage` field + engine stage_spec")
    from appcorr.vllm_stream.wire import Frame, stage_from_header, stage_to_header
    check("W1 legacy (r, g) header unchanged", stage_to_header((2, 4)) == [2, 4])
    check("W1 legacy round-trip", stage_from_header(stage_to_header((2, 4))) == (2, 4))
    exp = (1, 3, (13, 26, 40))
    check("W1 explicit header", stage_to_header(exp) == [1, 3, [13, 26, 40]])
    check("W1 explicit round-trip", stage_from_header(stage_to_header(exp)) == exp)
    check("W1 None passes through", stage_from_header(None) is None)
    for stage in ((2, 4), exp):
        f = Frame({"op": "correct", "rid": "t", "final": True, "window": [3, 9],
                   "stage": stage_to_header(stage)})
        f.put_tensor("positions", torch.arange(3, 9, dtype=torch.int64))
        f.put_tensor("embeds", torch.randn(6, 8, dtype=torch.bfloat16))
        from appcorr.vllm_stream.wire import FrameParser
        got = FrameParser().feed(f.encode())[0]
        check(f"W1 Frame round-trip {stage}",
              stage_from_header(got.header["stage"]) == stage
              and torch.equal(got.get_tensor("positions"), torch.arange(3, 9, dtype=torch.int64)),
              str(got.header))

    correct = import_engine_correct()
    check("W2 (r, g) derives stage_bounds",
          correct.stage_spec((1, 4), 40) == (1, 4, [10, 20, 30, 40]),
          str(correct.stage_spec((1, 4), 40)))
    check("W2 explicit bounds taken as given",
          correct.stage_spec((0, 2, (13, 40)), 40) == (0, 2, [13, 40]),
          str(correct.stage_spec((0, 2, (13, 40)), 40)))
    for bad, why in (((0, 2, (13, 39)), "last != L"), ((0, 2, (26, 13)), "not increasing"),
                     ((0, 3, (13, 40)), "g != len(bounds)"), ((2, 2, (13, 40)), "r out of range"),
                     ((0, 2, (0, 40)), "zero bound")):
        try:
            correct.stage_spec(bad, 40)
            check(f"W2 rejects {why}", False, f"{bad} accepted")
        except AssertionError:
            check(f"W2 rejects {why}", True)


# --------------------------------------------------------------------------------------------- #
# A -- the driver's walk on a tiny real model
# --------------------------------------------------------------------------------------------- #

def tiny_axis():
    from qwen_axis_cpu_unittest import make_inputs, tiny_models
    from appcorr.models.qwen35.unified import Qwen35Axis
    model = tiny_models()["qwen35"]
    axis = Qwen35Axis(model, None)
    inputs, px_base = make_inputs(model, 8, 12, 0)          # 96 rows -> 24 merge groups
    return model, axis, inputs, px_base


def run(axis, inputs, px_base, groups, keep, schedule, eager_pscore=False):
    from vllm_interleaved_axis_gate import RecordingSink
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


def test_axis_walk():
    print("A1  unified walk on a tiny Qwen3.5 (real HF classes, random weights, fp32)")
    from flops_analytic import MODELS35, Qwen35Decoder, Qwen35Vision, qwen35_from_config  # noqa: F401
    model, axis, inputs, px_base = tiny_axis()
    seq = int(inputs["input_ids"].shape[1])
    lo = int((inputs["input_ids"][0] == axis.image_token_id).nonzero()[0])
    n_vis = len(axis.tower.blocks)
    n_llm = int(axis.cfg.text_config.num_hidden_layers)
    n_rows = int(inputs["pixel_values"].shape[0])
    n_groups = n_rows // axis.tower.spatial_merge_unit

    for groups in (2, 4):
        for keep in (1.0, 0.5):
            sink, st = run(axis, inputs, px_base, groups, keep, "unified_staged")
            bounds = st["unified_bounds"]
            llm_rounds = [r for r in range(groups) if bounds[r] > n_vis]
            tag = f"g={groups} k={keep}"

            # message structure
            msgs = sink.msgs
            corr = [m for m in msgs if m["kind"] == "correct"]
            check(f"A1 {tag} one opening push",
                  msgs[0]["kind"] == "push" and not msgs[0]["final"]
                  and int(msgs[0]["embeds"].shape[0]) == seq,
                  str([(m["kind"], m["final"]) for m in msgs]))
            check(f"A1 {tag} one correct per LLM round",
                  len(corr) == len(llm_rounds), f"{len(corr)} vs {len(llm_rounds)} ({bounds})")
            check(f"A1 {tag} only the last is final",
                  corr[-1]["final"] and not any(m["final"] for m in corr[:-1]))
            check(f"A1 {tag} positions inside window and below hold-back",
                  all(int(m["positions"][0]) >= m["window"][0]
                      and int(m["positions"][-1]) < m["window"][1] <= seq - 1 for m in corr))
            check(f"A1 {tag} text suffix only in the last message",
                  all(int(m["positions"][-1]) < lo + n_groups for m in corr[:-1])
                  and int(corr[-1]["positions"][-1]) >= lo + n_groups)

            # the depths the schedule ran at
            recs = [tuple(c) for c in st["chunks"]]
            vap = [c for c in recs if c[0] == "vapprox"]
            vco = [c for c in recs if c[0] == "vcorrect"]
            cor = [c for c in recs if c[0] == "correct"]
            check(f"A1 {tag} vapprox ranges tile [0, n_vis) once, in order",
                  [c[1] for c in vap] == [0] + [c[2] for c in vap[:-1]]
                  and vap[-1][2] == n_vis and all(c[3] == n_rows for c in vap), str(vap))
            check(f"A1 {tag} vcorrect depth == min(bounds[r], n_vis)",
                  [c[2] for c in vco] == [min(bounds[r], n_vis) for r in range(len(vco))],
                  f"{[c[2] for c in vco]} vs {[min(bounds[r], n_vis) for r in range(groups)]}")
            check(f"A1 {tag} correct depth == bounds[r] - n_vis, last is full",
                  [c[6] for c in cor] == [bounds[r] - n_vis for r in llm_rounds]
                  and cor[-1][6] == n_llm, str([c[6] for c in cor]))
            check(f"A1 {tag} round index / count on the wire",
                  [(c[4], c[5]) for c in cor] == [(j, len(llm_rounds))
                                                  for j in range(len(llm_rounds))],
                  str([(c[4], c[5]) for c in cor]))

            # coverage (contract rule 4): every corrected group corrected exactly once
            sel = torch.cat([g.to("cpu") for g in st["group_idx"]])
            check(f"A1 {tag} groups corrected at most once",
                  int(sel.unique().numel()) == int(sel.numel()))
            check(f"A1 {tag} keep budget",
                  int(sel.numel()) == (n_groups if keep >= 1.0
                                       else max(1, round(keep * n_groups))),
                  f"{int(sel.numel())} of {n_groups}")
            if keep >= 1.0:
                # every image position was sent, either in the opening push or a correct
                emb = sink.final_embeds(seq)
                check(f"A1 {tag} every prompt position sent", emb.shape[0] == seq)
            check(f"A1 {tag} pscore label",
                  st.get("pscore") == ("progressive" if keep < 1.0 else None))

            # the closed-form replay, on this tiny model's own dims
            vis = Qwen35Vision(layers=n_vis, hidden=int(model.config.vision_config.hidden_size),
                               heads=int(model.config.vision_config.num_heads),
                               ffn=int(model.config.vision_config.intermediate_size))
            vc = vis.unified_cost(recs)
            full_tower = vis.tower_flops(n_rows)
            n_corr_rows = sum(c[3] for c in vco)
            ref = full_tower + n_corr_rows * n_vis * vis.row_layer_flops(n_rows)
            check(f"A1 {tag} vision total <= the full-depth arm's, > the bare tower",
                  full_tower < vc["total"] <= ref + 1e-6,
                  f"{vc['total']:.4e} vs tower {full_tower:.4e} / full-depth {ref:.4e}")
            check(f"A1 {tag} vision crit == the last band at FULL tower depth",
                  abs(vc["crit"] - vco[-1][3] * n_vis * vis.row_layer_flops(n_rows)) < 1e-3,
                  f"{vc['crit']:.6e}")
            dec = tiny_decoder(model)
            dc = dec.interleaved_cost(seq, lo, n_groups, recs)
            check(f"A1 {tag} decoder replay is staged and ordered",
                  dc["staged"] and 0 < dc["crit"] < dc["total"], str(dc))


def tiny_decoder(model):
    """`Qwen35Decoder` for the tiny test model -- the same fields `qwen35_from_config` reads."""
    from flops_analytic import Qwen35Decoder
    t = model.config.text_config
    lt = list(t.layer_types)
    return Qwen35Decoder(
        layers=len(lt), hidden=int(t.hidden_size), heads=int(t.num_attention_heads),
        kv_heads=int(t.num_key_value_heads),
        head_dim=int(getattr(t, "head_dim", t.hidden_size // t.num_attention_heads)),
        n_full=sum(1 for x in lt if x == "full_attention"),
        n_linear=sum(1 for x in lt if x == "linear_attention"),
        lin_k_heads=int(t.linear_num_key_heads), lin_v_heads=int(t.linear_num_value_heads),
        lin_k_dim=int(t.linear_key_head_dim), lin_v_dim=int(t.linear_value_head_dim),
        conv_kernel=int(t.linear_conv_kernel_dim), dense_inter=int(t.intermediate_size),
        vocab=int(t.vocab_size))


def test_g1_identity():
    print("A2  g=1 identity against the interleaved schedules")
    _, axis, inputs, px_base = tiny_axis()
    seq = int(inputs["input_ids"].shape[1])
    for keep, eager in ((1.0, False), (0.5, True)):
        u, ust = run(axis, inputs, px_base, 1, keep, "unified_staged")
        for other in ("interleaved", "interleaved_staged"):
            o, ost = run(axis, inputs, px_base, 1, keep, other, eager_pscore=eager)
            a, b = u.final_embeds(seq), o.final_embeds(seq)
            same = torch.equal(a, b)
            check(f"A2 g=1 keep={keep} unified == {other} (bitwise embeds)", same,
                  f"max|d| {float((a - b).abs().max()):.3e}")
            usel = torch.cat([g.to('cpu') for g in ust["group_idx"]]).sort().values
            osel = torch.cat([g.to('cpu') for g in ost["group_idx"]]).sort().values
            check(f"A2 g=1 keep={keep} same selection as {other}", torch.equal(usel, osel),
                  f"{usel.tolist()} vs {osel.tolist()}")


# --------------------------------------------------------------------------------------------- #
# R -- the existing schedules are untouched
# --------------------------------------------------------------------------------------------- #

def test_reference(ref_root):
    print(f"R1  existing schedules vs {ref_root}")
    from qwen_axis_cpu_unittest import make_inputs, tiny_models
    from appcorr.models.qwen35.unified import Qwen35Axis
    RefAxis = import_ref(ref_root)
    model = tiny_models()["qwen35"]
    new, ref = Qwen35Axis(model, None), RefAxis(model, None)
    for grid in ((8, 12), (6, 10)):
        inputs, px_base = make_inputs(model, grid[0], grid[1], 1)
        for schedule in ("interleaved", "interleaved_staged"):
            for groups in (1, 4):
                for keep in (1.0, 0.5):
                    seq = int(inputs["input_ids"].shape[1])
                    a, ast = run(new, inputs, px_base, groups, keep, schedule)
                    b, bst = run(ref, inputs, px_base, groups, keep, schedule)
                    check(f"R1 {schedule} g={groups} k={keep} grid={grid}",
                          torch.equal(a.final_embeds(seq), b.final_embeds(seq))
                          and [list(c) for c in ast["chunks"]] == [list(c) for c in bst["chunks"]]
                          and int(ast["corrected_groups"]) == int(bst["corrected_groups"]))
        for groups in (1, 4):
            for keep in (1.0, 0.5):
                ra = new.streaming_forward(inputs, px_base, groups, keep=keep)
                rb = ref.streaming_forward(inputs, px_base, groups, keep=keep)
                check(f"R1 streaming g={groups} k={keep} grid={grid}",
                      torch.equal(ra[2]["image_embeds"], rb[2]["image_embeds"])
                      and torch.equal(ra[0], rb[0])
                      and ra[2]["chunks"] == rb[2]["chunks"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref-root", default=None,
                    help="another worktree to gate the pre-existing schedules against")
    ap.add_argument("--only", default=None, help="substring filter on the test function names")
    a = ap.parse_args()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_CACHE", "/NHNHOME/huggingface/hub")

    tests = [test_bounds, test_wire, test_axis_walk, test_g1_identity]
    for t in tests:
        if a.only and a.only not in t.__name__:
            continue
        t()
    if a.ref_root:
        test_reference(a.ref_root)

    print()
    if FAILS:
        print(f"FAIL ({len(FAILS)}):")
        for f in FAILS:
            print("  " + f)
        raise SystemExit(1)
    print("PASS")


if __name__ == "__main__":
    main()
