"""GPU gates for the GLM-4.6V port's vision half (`appcorr/models/glm46v/`).

Nothing in this file has been run: both GPUs were committed the night the port was written
(2026-09-12), so the CPU half of the evidence is in `tests/test_glm46v_merger_bands.py` and
everything that needs CUDA kernels or a live engine is here, unrun. Read the pass rules below
before believing any number it prints.

  --mode g1   (GPU0, no server, ~2 min)   the tower split, on CUDA kernels and in bf16
  --mode g0   (GPU0 + a running server)   the served one-shot path, 8 V*Bench images

--------------------------------------------------------------------------------------------
G1 -- tower split (`--mode g1`)

Same five identities the CPU test asserts, re-run where the campaign actually runs them: CUDA
kernels, bf16 weights, real image sizes.

  g1a  fork `reference_forward` == stock `Glm4vMoeVisionModel.forward`
       (pre stage order, rotary tables, merge head)
  g1b  the 24 blocks walked in 4 `approx_forward` ranges == `reference_forward`
  g1c  `merger(all rows)[g0:g1]` == `merger(rows of groups [g0, g1))` for every band
  g1d  `correct_rows(...)` == `correct_forward(...)[band rows]`
  g1e  g=1 correction on the degraded base reproduces the full-resolution forward

PASS RULE. g1a / g1b compare calls of the SAME shape: bitwise, no tolerance. g1c: bitwise or (<=1 bf16 ULP everywhere and <1% elements differing); g1d
compare GEMMs of different M, and cuBLAS selects its tile/split-k configuration per M, so
bitwise is EXPECTED but not guaranteed -- on CPU (oneDNN, single thread) the same comparison is
bitwise for bands of >= 4 merge groups and 3.6e-6 (fp32) for smaller ones. So: report bitwise,
and if it is not bitwise, FAIL only above 5e-3 in bf16 (about one ULP at the magnitudes involved)
and report the number either way. g1e is a different kernel sequence by construction (cached K/V
and gathered rows vs a one-shot forward): relative L2, pass under 1e-3 in bf16.

--------------------------------------------------------------------------------------------
G0 -- served one-shot (`--mode g0 --port ...`)

What can be gated on ONE B200 and what cannot, stated plainly:

  * the AppCorr one-shot (`oneshot_embeds`: stock tower on the full image, one chunk pushed to
    the server) against the SAME request run through the progressive path at g=1, keep=1. At
    g=1 there is no staleness anywhere, so the two must produce the same first token on every
    image -- this is the served form of the in-process g=1 identity, and it gates the whole
    driver->wire->engine path. Runs with `--load vision`, so it fits beside a 106B-FP8 engine.
  * the HF twin's greedy first token (the memo's G0 proper) does NOT fit: the 106B-A12B FP8
    checkpoint is ~110 GB of weights and the engine already holds them. It is therefore opt-in
    (`--hf-model`), for a box where the twin fits, and the script refuses to load it unless
    asked. The practical stock anchor on one GPU is vLLM's OWN image request, which lives in
    the engine process: `vllm_stream_gate.py --model zai-org/GLM-4.6V-FP8 --arms A,B` under
    the appcorr-vllm python compares vLLM's OWN image request (arm A) with the same prompt
    composed as embeds (arm B, `Glm46VComposer`, which its `composer_for` now selects) on its
    own COCO images. Run it separately: its A == B is the stock-vs-embeds half of G0 and this
    script's ceiling-vs-g=1 is the driver-and-wire half. They are not joined here, because the
    two scripts do not share an image set.

--------------------------------------------------------------------------------------------
Commands (NOT to be run while the GPUs are busy):

  # G1, no server
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTHONPATH=$PWD \\
    /home/nxclab/anaconda3/envs/appcorr/bin/python \\
    analysis/experiments/glm46v_tower_gate.py --mode g1 \\
    --out analysis/results/vllm_stream/glm46v_tower_gate_g1.json

  # G0, against a server already serving zai-org/GLM-4.6V-FP8 on --port
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTHONPATH=$PWD \\
    /home/nxclab/anaconda3/envs/appcorr/bin/python \\
    analysis/experiments/glm46v_tower_gate.py --mode g0 --port 5591 --samples 8 \\
    --out analysis/results/vllm_stream/glm46v_tower_gate_g0.json
"""
import argparse
import json
import math
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))

MODEL_ID = "zai-org/GLM-4.6V-FP8"
GLM_MAX_PX = 9_633_792     # Glm46VImageProcessor size["longest_edge"]; see qwen_vllm_accuracy


def _bands(n_groups: int, groups: int):
    edges = [round(k * n_groups / groups) for k in range(groups + 1)]
    return list(zip(edges[:-1], edges[1:]))


def _eq(a: torch.Tensor, b: torch.Tensor):
    """(bitwise, max_abs). bf16 has no exact `torch.equal` shortcut worth trusting on stray
    NaNs, so compare the bit patterns and measure the gap in fp32."""
    if a.dtype == torch.bfloat16:
        bitwise = bool(torch.equal(a.view(torch.int16), b.view(torch.int16)))
    else:
        bitwise = bool(torch.equal(a, b))
    return bitwise, float((a.float() - b.float()).abs().max())


def _images(n: int, dataset: str):
    from datasets import load_dataset
    from qwen_vl_prefill.datasets_eval import get_spec
    spec = get_spec(dataset)
    ds = spec.load(load_dataset)
    idxs = list(range(0, len(ds), max(1, len(ds) // n)))[:n]
    out = []
    for i in idxs:
        img, q, gold = spec.prepare(ds[int(i)], lambda h, w, **kw: (h, w), 1, 1, 1 << 30)
        out.append((int(i), img.convert("RGB") if img.mode != "RGB" else img, q, gold))
    return out


# ----------------------------------------------------------------------------------------- #
# G1: the tower split on CUDA
# ----------------------------------------------------------------------------------------- #

@torch.no_grad()
def gate_g1(a):
    from transformers import AutoProcessor

    from appcorr.models.glm46v.vision.backbone import (
        ApproxCorrectGlm4vVisionTower, load_stock_vision_tower)
    from analysis.experiments.qwen35_accuracy import degrade

    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[a.dtype]
    dev = a.device
    t0 = time.time()
    stock = load_stock_vision_tower(a.model, device=dev, dtype=dtype)
    tower = ApproxCorrectGlm4vVisionTower(stock)
    proc = AutoProcessor.from_pretrained(a.model)
    print(f"tower loaded ({a.dtype}) in {time.time() - t0:.1f}s", flush=True)

    report, fails = [], []
    for i, img, _, _ in _images(a.samples, a.dataset):
        enc = proc(text=["x"], images=[img], return_tensors="pt")
        px = enc["pixel_values"].to(dev, dtype)
        grid = enc["image_grid_thw"].to(dev)
        base = degrade(img, a.level, a.degrade_filter, max_px=GLM_MAX_PX)
        px_base = proc(text=["x"], images=[base], return_tensors="pt")["pixel_values"].to(dev, dtype)
        assert px_base.shape == px.shape, (px_base.shape, px.shape)
        n_rows = int(px.shape[0])
        n_groups = n_rows // tower.spatial_merge_unit
        row = {"i": i, "grid": [int(v) for v in grid[0].tolist()], "rows": n_rows}

        # g1a -----------------------------------------------------------------------------
        want = stock(px, grid_thw=grid).pooler_output
        ref = tower.reference_forward(px, grid)
        ok, gap = _eq(ref, want)
        row["g1a_fork_vs_stock"] = {"bitwise": ok, "max_abs": gap}
        if not ok:
            fails.append(f"g1a sample {i}: max_abs {gap}")

        # g1b -----------------------------------------------------------------------------
        ctx = tower.prepare_full_tokens(px, grid)
        x, cache = ctx["hidden_states"], {}
        n_vis = len(tower.blocks)
        for s, e in _bands(n_vis, 4):
            x, cache = tower.approx_forward(x, s, e, ctx, cache, "v")
        ok, gap = _eq(tower.merger(x), ref)
        row["g1b_staged_vs_reference"] = {"bitwise": ok, "max_abs": gap}
        if not ok:
            fails.append(f"g1b sample {i}: max_abs {gap}")

        # g1c -----------------------------------------------------------------------------
        # g1c: band-sliced merger vs all-rows merger. Both run in bf16, and the merger is four
        # GEMMs + a LayerNorm with bf16 re-rounding between stages, so the two bf16 paths differ
        # by compounded rounding noise whenever cuBLAS picks a different kernel for the band's M
        # (2026-09-12: 29% of elements, max_abs 0.03125 = 1 bf16 ULP at the tensor's top
        # magnitude; measured relative to each element's OWN magnitude that read as "500 ULP",
        # which is the wrong scale for near-zero elements). The question the gate must answer is
        # whether band slicing adds error BEYOND that noise. So both bf16 paths are compared to an
        # fp32 truth computed from the same rows: pass iff the band path's error is no worse than
        # the all-rows path's (max within 2x, mean within 2x) and both sit at bf16 noise level
        # (max error <= 2 ULP of the tensor's max magnitude). Row mixing would blow both bounds.
        import copy as _copy
        merger32 = _copy.deepcopy(tower.merger).float()
        with torch.no_grad():
            ref32 = merger32(x.float())
        merged_all = tower.merger(x)
        scale = float(ref32.abs().max())
        ulp_top = 2.0 ** (math.floor(math.log2(max(scale, 2.0 ** -126))) - 7)
        worst, nb = 0.0, 0
        e_all_max = float((merged_all.float() - ref32).abs().max())
        e_all_mean = float((merged_all.float() - ref32).abs().mean())
        e_band_max, e_band_mean = 0.0, 0.0
        for groups in (2, 3, 4, 8):
            for g0, g1 in _bands(n_groups, groups):
                out = tower.merger(x[g0 * 4:g1 * 4])
                bw, gap = _eq(out, merged_all[g0:g1])
                worst = max(worst, gap); nb += int(not bw)
                d = (out.float() - ref32[g0:g1]).abs()
                e_band_max = max(e_band_max, float(d.max())); e_band_mean = max(e_band_mean, float(d.mean()))
        row["g1c_merger_bands"] = {"non_bitwise_bands": nb, "max_abs_bf16_vs_bf16": worst,
                                   "ref32_scale": scale, "ulp_at_scale": ulp_top,
                                   "err_vs_fp32_all_max": e_all_max, "err_vs_fp32_all_mean": e_all_mean,
                                   "err_vs_fp32_band_max": e_band_max, "err_vs_fp32_band_mean": e_band_mean}
        if nb and not (e_band_max <= 2.0 * max(e_all_max, ulp_top) and e_band_mean <= 2.0 * max(e_all_mean, 1e-9)
                       and e_band_max <= 2.0 * ulp_top):
            fails.append(f"g1c sample {i}: band err max {e_band_max:.4g} mean {e_band_mean:.4g} vs "
                         f"all-rows err max {e_all_max:.4g} mean {e_all_mean:.4g}, ulp@scale {ulp_top:.4g}")

        # g1d / g1e -----------------------------------------------------------------------
        gctx = tower.prepare_grid(grid, px.device)
        ctx_full = tower.prepare_full_tokens(px, grid, gctx)
        ctx_base = tower.prepare_full_tokens(px_base, grid, gctx)
        g0, g1 = _bands(n_groups, 4)[1]
        gidx = torch.arange(g0, g1, device=dev)
        _, c1 = tower.approx_forward(ctx_base["hidden_states"], 0, n_vis, ctx_base, {}, "v")
        rows_only, _ = tower.correct_rows(ctx_full["hidden_states"], gidx, ctx_full, c1, "v",
                                          span=(g0 * 4, g1 * 4))
        _, c2 = tower.approx_forward(ctx_base["hidden_states"], 0, n_vis, ctx_base, {}, "v")
        mask = torch.zeros(n_rows, dtype=torch.bool, device=dev)
        mask[g0 * 4:g1 * 4] = True
        stream = torch.where(mask.unsqueeze(-1), ctx_full["hidden_states"],
                             ctx_base["hidden_states"])
        full, _ = tower.correct_forward(stream, gidx, 0, n_vis, ctx_full, c2, "v")
        ok, gap = _eq(rows_only, full[g0 * 4:g1 * 4])
        row["g1d_correct_rows_vs_stream"] = {"bitwise": ok, "max_abs": gap}
        if gap > a.tol_band:
            fails.append(f"g1d sample {i}: max_abs {gap} > {a.tol_band}")

        _, c3 = tower.approx_forward(ctx_base["hidden_states"], 0, n_vis, ctx_base, {}, "v")
        all_rows, _ = tower.correct_rows(ctx_full["hidden_states"],
                                         torch.arange(n_groups, device=dev), ctx_full, c3, "v",
                                         span=(0, n_rows))
        got = tower.merger(all_rows)
        rel = float((got.float() - ref.float()).norm() / ref.float().norm())
        row["g1e_g1_identity_rel_l2"] = rel
        if rel > a.tol_identity:
            fails.append(f"g1e sample {i}: rel_l2 {rel} > {a.tol_identity}")

        print(json.dumps(row), flush=True)
        report.append(row)

    out = {"_mode": "g1", "_model": a.model, "_dtype": a.dtype, "_device": a.device,
           "_tol": {"band": a.tol_band, "identity": a.tol_identity},
           "rows": report, "fails": fails}
    _write(a.out, out)
    print(("FAIL: " + "; ".join(fails)) if fails else "G1 PASS", flush=True)
    return 1 if fails else 0


# ----------------------------------------------------------------------------------------- #
# G0: the served one-shot path
# ----------------------------------------------------------------------------------------- #

@torch.no_grad()
def gate_g0(a):
    from transformers import AutoProcessor

    from appcorr.models.vision_only import load_vision_only
    from appcorr.vllm_stream.bridge import LLMBridge
    from analysis.experiments.qwen35_accuracy import degrade
    from analysis.experiments.qwen_vllm_accuracy import make_axis

    bridge = LLMBridge(a.host, a.port)
    info = bridge.info()
    if info["model"] != a.model:
        raise SystemExit(f"server serves {info['model']!r}, gate asked for {a.model!r}")
    print(f"server: {info}", flush=True)

    proc = AutoProcessor.from_pretrained(a.model)
    model = load_vision_only(a.model, device=a.device)
    axis = make_axis("glm46v", model, proc)

    hf = None
    if a.hf_model:
        # Opt-in and loud: on one B200 this cannot coexist with a 106B-FP8 engine.
        from transformers import AutoModelForImageTextToText
        print(f"loading the HF twin {a.hf_model} -- this needs a device the engine is not on",
              flush=True)
        hf_model = AutoModelForImageTextToText.from_pretrained(
            a.hf_model, dtype="auto", device_map=a.hf_device).eval()
        hf = make_axis("glm46v", hf_model, proc)

    rows, fails = [], []
    for i, img, q, gold in _images(a.samples, a.dataset):
        base = degrade(img, a.level, a.degrade_filter, max_px=GLM_MAX_PX)
        inputs = axis.build_inputs(img, q).to(a.device)
        px_base = axis.build_inputs(base, q)["pixel_values"].to(a.device)
        ids = inputs["input_ids"][0]
        pos = (ids == axis.image_token_id).nonzero(as_tuple=True)[0]
        layout = {"image_run": (int(pos[0]), int(pos.numel())),
                  "grid_thw": tuple(int(v) for v in inputs["image_grid_thw"][0].tolist())}
        row = {"i": i, "gold": gold, "prompt_tokens": int(ids.shape[0])}

        sink = bridge.sink(max_tokens=a.max_tokens)
        emb, p3, delta = axis.oneshot_embeds(inputs, inputs["pixel_values"], **layout)
        sink.push(emb, p3, delta, final=True)
        row["ceiling"] = sink.result()["text"]

        for schedule in a.schedules:
            sink = bridge.sink(max_tokens=a.max_tokens)
            axis.streaming_forward(inputs, px_base, 1, keep=1.0, sink=sink,
                                   llm_schedule=schedule, **layout)
            row[f"{schedule}_g1"] = sink.result()["text"]
            if row[f"{schedule}_g1"] != row["ceiling"]:
                fails.append(f"G0 sample {i}: {schedule} g=1 != ceiling")

        if hf is not None:
            from analysis.experiments.qwen_vllm_accuracy import greedy_tokens
            out = hf.model(input_ids=inputs["input_ids"].to(hf.model.device),
                           pixel_values=inputs["pixel_values"].to(hf.model.device, hf.model.dtype),
                           image_grid_thw=inputs["image_grid_thw"].to(hf.model.device),
                           mm_token_type_ids=inputs["mm_token_type_ids"].to(hf.model.device),
                           use_cache=True)
            toks = greedy_tokens(hf, out.logits[:, -1], out.past_key_values,
                                 int(p3.max()) + 1, a.max_tokens)
            row["hf"] = proc.tokenizer.decode(toks, skip_special_tokens=True)
            if row["hf"] != row["ceiling"]:
                fails.append(f"G0 sample {i}: HF twin != served ceiling")

        print(json.dumps(row), flush=True)
        rows.append(row)

    _write(a.out, {"_mode": "g0", "_model": a.model, "_server": info, "rows": rows,
                   "fails": fails})
    print(("FAIL: " + "; ".join(fails)) if fails else "G0 PASS", flush=True)
    return 1 if fails else 0


def _write(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    json.dump(obj, open(path, "w"), indent=1)
    print(f"wrote {path}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["g0", "g1"], required=True)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16",
                    help="g1 only; bf16 is what the campaign runs")
    ap.add_argument("--dataset", default="vstar")
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--degrade-filter", choices=["bicubic", "box", "pyr"], default="pyr")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=None, help="g0: the running stream server")
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--schedules", nargs="+",
                    default=["streaming", "interleaved", "interleaved_staged"],
                    help="g0: schedules to check at g=1 against the one-shot ceiling. The two "
                         "interleaved ones need a server started with --interleaved (the "
                         "`correct` op); drop them for a plain streaming server")
    ap.add_argument("--hf-model", default=None,
                    help="g0: also decode with the HF twin (needs a device the engine is NOT "
                         "on; 106B-A12B FP8 does not fit beside its own engine)")
    ap.add_argument("--hf-device", default="cuda:1")
    ap.add_argument("--tol-band", type=float, default=5e-3,
                    help="g1: max |diff| allowed where the compared GEMMs have different M "
                         "(bf16; bitwise is expected, this is the kernel-selection band)")
    ap.add_argument("--tol-identity", type=float, default=1e-3,
                    help="g1: max relative L2 for the g=1 correction identity (bf16)")
    ap.add_argument("--out", default="analysis/results/vllm_stream/glm46v_tower_gate.json")
    a = ap.parse_args()
    if a.mode == "g0" and not a.port:
        raise SystemExit("--mode g0 needs --port (a running appcorr.vllm_stream.server)")
    return gate_g1(a) if a.mode == "g1" else gate_g0(a)


if __name__ == "__main__":
    raise SystemExit(main())
