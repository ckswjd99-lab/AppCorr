"""GPU gates for the GLM-4.6V-FP8 interleaved correct step (+ the CPU FLOPs reconcile).

GLM-4.6V's decoder is **pure softmax GQA** (46 x `Glm4MoeDecoderLayer`), so the Qwen3.5 gate's
DeltaNet sub-gates do not exist here: there is no side buffer to fill, no window to re-scan and
no recurrent block to write back (`correct.check_gdn_path` returns [] and `appcorr_rows_step`
runs the pseudo-sequence softmax path alone -- `tests/test_correct_gdn_gating.py`).  What is left
to gate is exactly the softmax half, so this file is `vllm_correct_gate.py` minus g0/g1 and minus
every ssm/conv column, plus the FLOPs reconcile the port plan asks for.

  chunk  CONTROL, and the only thing the others are judged against: the SAME corrected prompt
         prefilled by the stock engine in `--chunks` streaming chunks instead of one.  Its
         KV rel-L2 / dlogprob vs the one-shot arm is the engine's own numerical band on this
         checkpoint; an absolute pass mark would be meaningless (memo §6.6, and FP8 chunking is
         lossy on the Qwen 122B -- assume nothing about the FP8 band here either).
  g2     IDENTITY, keep=1, g=`--g` (4): open with the approximate prompt (the base-resolution
         image, same grid, same N), push the corrected image rows band by band, the last band
         carrying the text suffix, `final=True`.  At keep=1 every image row is corrected exactly
         once and the text suffix once, so the final KV must equal a stock prefill of the
         corrected prompt.  Reported vs `one`: first-token argmax (expect 8/8), generated-sequence
         exact match, |dlogprob| of the first token, and the max over layers of the KV rel-L2 at
         the IMAGE slots and at the TEXT slots separately (the text suffix is corrected in the
         last round only, so a text-slot regression is a different bug from an image-slot one).
  g3     g=1 vs g=4 at keep=1, both vs `one`: one band and four bands are the same computation
         at keep=1, so the two must land in the same band as each other and as `chunk`.
  flops  CPU, no engine: gate F.  The closed form (`flops_analytic.Glm46VDecoder.prefill_flops`,
         per sample because the softmax term is quadratic in N) against the hooked decoder half
         of `flops_report_qwen35.py --family glm46v` (`_split[arm][2]`, i.e. llm_total) at BOTH
         the ceiling and the floor arm.  Pass: |calc/meas - 1| < 1%.

Every accuracy claim here needs a GPU and NONE of it has been run: this file was written on
2026-09-12 while both devices were committed.  The CPU-checkable parts (the GDN gating, the
closed form, the experts hook point) are under `tests/`.

Commands
--------
G2 (identity) and G3, in-process, one engine, GPU0.  `--interleaved`-style env is not needed:
there is no GDN kernel to force.  The 122B recipe is the starting point for memory
(`--gpu-mem 0.85 --max-model-len 8192`); with no side buffers the margin is larger, so try
`--max-model-len 16384` and MEASURE rather than assume.  `--max-num-seqs` must exceed the largest
round's |P| or the step sub-batches (still correct, just slower):

  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \\
  PYTHONPATH=/NHNHOME/share/cjpark/AppCorr-glm46v \\
  /NHNHOME/storage/users/cjpark/shk/conda_envs/appcorr-vllm/bin/python \\
      analysis/experiments/glm46v_correct_gate.py --gate g2 --g 4 \\
      --gpu-mem 0.85 --max-model-len 8192 --max-num-seqs 1024

  ... --gate g3 --gpu-mem 0.85 --max-model-len 8192 --max-num-seqs 1024

FLOPs, two steps.  First the hooked run (GPU, HF load; needs `compressed-tensors>=0.15.0` in the
env -- NOT installed in `appcorr` as of 2026-09-12, install it into a cjpark-owned env, never
into `openrlhf_base`):

  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/NHNHOME/share/cjpark/AppCorr-glm46v \\
  /home/nxclab/anaconda3/envs/appcorr/bin/python analysis/experiments/flops_report_qwen35.py \\
      --family glm46v --datasets realworldqa vstar --samples 12 \\
      --out-json analysis/results/flops/glm46v_flops.json

then the reconcile (CPU, no GPU):

  PYTHONPATH=/NHNHOME/share/cjpark/AppCorr-glm46v \\
  /home/nxclab/anaconda3/envs/appcorr/bin/python analysis/experiments/glm46v_correct_gate.py \\
      --gate flops --flops-json analysis/results/flops/glm46v_flops.json \\
      --datasets realworldqa vstar --samples 12
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from vllm_stream_gate import COCO, IMAGES, QUESTION, compare, tokens_and_lp  # noqa: E402
from vllm_correct_gate import bands_of, rel_l2, run_to_first_token           # noqa: E402

MODEL = "zai-org/GLM-4.6V-FP8"
IMAGE_TOKEN = "<|image|>"        # id 151363 = config.image_token_id (the placeholder run)


# --- prompt composition ------------------------------------------------------------------------ #

def make_composer(llm):
    """The GLM composer lives in `appcorr.vllm_stream.client` (added with the vision port):
    `IMAGE_TOKEN = "<|image|>"` as a CLASS attribute so the base validation sees it, and
    `embed` resolving the embedding table through `resolve_embed_fn` (GLM has no
    `model.embed_input_ids`; it is `model.language_model.embed_input_ids`). The earlier local
    subclass set `image_pad_id` only AFTER `super().__init__`, which validates the class-level
    default `<|image_pad|>` first and raised on GLM's tokenizer (2026-09-12 22:11, run 3)."""
    from appcorr.vllm_stream.client import Glm46VComposer
    return Glm46VComposer(llm)

def kv_split(snap_a: dict, snap_b: dict, n_img: int) -> dict:
    """Max-over-layers KV rel-L2 of `snap_a` vs `snap_b`, image slots and text slots apart.

    The snapshots are taken at `positions = [lo, N-1)`, so rows `[0, n_img)` are the image and
    `[n_img, ...)` the post-image text suffix.  Separating them is not cosmetic: the suffix is
    rewritten in the LAST round only, so a suffix-only regression means the final round's
    positions/window are wrong, while an image-only one means a band is.
    """
    img, txt = {}, {}
    for ln, b in snap_b["kv"].items():
        a = snap_a["kv"][ln]
        img[ln] = rel_l2(a[:n_img], b[:n_img])
        if a.shape[0] > n_img:
            txt[ln] = rel_l2(a[n_img:], b[n_img:])
    worst_i = max(img, key=img.get)
    out = {"kv_img_rel_l2_max": img[worst_i], "kv_img_rel_l2_max_layer": worst_i,
           "kv_img_rel_l2": img}
    if txt:
        worst_t = max(txt, key=txt.get)
        out.update({"kv_txt_rel_l2_max": txt[worst_t], "kv_txt_rel_l2_max_layer": worst_t,
                    "kv_txt_rel_l2": txt})
    assert not snap_b["mamba"], (
        "GLM-4.6V has no recurrent state; a non-empty mamba snapshot means the gate is running "
        "against the wrong model or `_mamba_group_ids` regressed")
    return out


# --- the engine gates (GPU) ---------------------------------------------------------------------- #

def gate_engine(a):
    from PIL import Image
    from vllm import SamplingParams
    from appcorr.vllm_stream import StreamingLLM
    from appcorr.vllm_stream import correct as _correct

    kw = {}
    if a.max_num_seqs:
        kw["max_num_seqs"] = a.max_num_seqs
    llm = StreamingLLM(a.model, gpu_memory_utilization=a.gpu_mem, max_model_len=a.max_model_len,
                       enforce_eager=a.enforce_eager, limit_mm_per_prompt={"image": 1}, **kw)
    comp = make_composer(llm)
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens, logprobs=1)
    runner = llm.runner

    # the decoder must be the pure-softmax one this gate is about
    assert _correct.check_gdn_path(llm.engine.vllm_config) == [], "GDN layers on GLM-4.6V?"
    assert not _correct.has_gdn(runner.model)
    n_layers = _correct.num_layers(runner)
    assert n_layers == 46, n_layers
    assert _correct._mamba_group_ids(runner) == [], "unexpected mamba kv-cache group"

    res = json.load(open(a.out)) if os.path.exists(a.out) else {"_meta": {}}
    backends = {str(gid): [ag.backend.__name__ for ag in runner.attn_groups[gid]]
                for gid in range(len(runner.kv_cache_config.kv_cache_groups))}
    res["_meta"].update({
        "model": a.model, "gate": a.gate, "g": a.g, "max_tokens": a.max_tokens,
        "question": QUESTION, "images": IMAGES[:a.n_images], "downscale": a.downscale,
        "attn_backends": backends, "decoder_layers": n_layers, "gdn_layers": 0,
        "max_num_seqs": llm.engine.vllm_config.scheduler_config.max_num_seqs,
        "max_num_batched_tokens": llm.engine.vllm_config.scheduler_config.max_num_batched_tokens,
    })
    print("attn backends:", backends, flush=True)
    arms = {"g2": ["one", "chunk", "g2"], "g3": ["one", "chunk", "g3_1", "g3_4"]}[a.gate]

    for ii, name in enumerate(IMAGES[:a.n_images]):
        img = Image.open(os.path.join(COCO, name)).convert("RGB")
        w, h = img.size
        parts = comp.parts(img, QUESTION)
        N, lo, G = parts.num_tokens, parts.image_start, parts.image_len
        emb = comp.embed(parts)
        row = res.setdefault(name, {})
        row.update({"num_prompt_tokens": N, "image_span": [lo, lo + G]})
        cmp_pos = torch.arange(lo, N - 1, dtype=torch.int64)

        low = img.resize((max(w // a.downscale, 1), max(h // a.downscale, 1))).resize((w, h))
        parts_a = comp.parts(low, QUESTION)
        assert (parts_a.num_tokens, parts_a.image_start, parts_a.image_len) == (N, lo, G), (
            "the approx prompt must have the same grid/length as the corrected one "
            "(degrade content, never geometry)")
        emb_a = comp.embed(parts_a)
        d = emb_a.embeds[lo:lo + G].float() - emb.embeds[lo:lo + G].float()
        row["approx_img_rel_l2"] = float(d.norm() / emb.embeds[lo:lo + G].float().norm())
        assert torch.equal(emb_a.embeds[:lo], emb.embeds[:lo]), "pre-image text rows differ"
        assert torch.equal(emb_a.embeds[lo + G:], emb.embeds[lo + G:]), "post-image text rows differ"

        snaps = {}

        def one_shot(key):
            rid = f"{key}-{ii}"
            t0 = time.perf_counter()
            llm.open(rid, emb.chunk(0, N, final=True), sp)
            run_to_first_token(llm, rid)
            snaps[key] = runner.appcorr_snapshot(llm._core_id(rid), cmp_pos)
            o = llm.run_until_done(rid)[rid]
            row[key] = tokens_and_lp(o) | {"wall_s": time.perf_counter() - t0}

        def chunked(key):
            rid = f"{key}-{ii}"
            t0 = time.perf_counter()
            bounds = comp.image_bounds(parts, a.chunks)
            chunks = emb.chunks(bounds)
            llm.open(rid, chunks[0], sp)
            for ch in chunks[1:]:
                llm.step()
                llm.append(rid, ch)
            run_to_first_token(llm, rid)
            snaps[key] = runner.appcorr_snapshot(llm._core_id(rid), cmp_pos)
            o = llm.run_until_done(rid)[rid]
            row[key] = tokens_and_lp(o) | {"wall_s": time.perf_counter() - t0, "bounds": bounds}

        def interleaved(key, g):
            rid = f"{key}-{ii}"
            t0 = time.perf_counter()
            llm.open(rid, emb_a.chunk(0, N, final=False), sp, correct=True,
                     image_start=lo, image_end=lo + G)
            infos, bs = [], bands_of(G, g)
            for r, (g0, g1) in enumerate(bs):
                final = r == len(bs) - 1
                pos = torch.arange(lo + g0, lo + g1, dtype=torch.int64)
                rows = emb.embeds[pos]
                win = (lo + g0, lo + g1)
                if final:                       # the text suffix joins the last round
                    pos = torch.cat([pos, torch.arange(lo + G, N - 1, dtype=torch.int64)])
                    rows = torch.cat([rows, emb.embeds[lo + G:N - 1]], dim=0)
                    win = (lo + g0, N - 1)
                infos.append(llm.correct(rid, pos, rows, win, final))
            run_to_first_token(llm, rid)
            snaps[key] = runner.appcorr_snapshot(llm._core_id(rid), cmp_pos)
            o = llm.run_until_done(rid)[rid]
            row[key] = tokens_and_lp(o) | {"wall_s": time.perf_counter() - t0, "bands": bs,
                                           "info": infos}
            # the softmax-only path must report no side-buffer bytes and no mamba blocks
            for inf in infos:
                assert inf["mamba_blocks"] == {}, inf["mamba_blocks"]
            row[key]["side_buffer_mb"] = infos[-1]["side_buffer_mb"]

        if "one" in arms:
            one_shot("one")
        if "chunk" in arms:
            chunked("chunk")
        if "g2" in arms:
            interleaved("g2", a.g)
        if "g3_1" in arms:
            interleaved("g3_1", 1)
            interleaved("g3_4", 4)

        ref = row.get("one")
        line = f"[{ii}] {name} N={N} img=[{lo},{lo + G})"
        for k in arms:
            if k == "one" or k not in row:
                continue
            c = compare(ref, row[k])
            row[k]["vs_one"] = c
            st = kv_split(snaps[k], snaps["one"], G)
            row[k]["vs_one_state"] = st
            line += (f" | {k}: {'exact' if c['exact'] else 'div@' + str(c['first_divergence'])}"
                     f" dlp0={c['dlogprob_first']:.2e} kv_img={st['kv_img_rel_l2_max']:.2e}"
                     f" kv_txt={st.get('kv_txt_rel_l2_max', float('nan')):.2e}")
        if "g3_4" in row and "g3_1" in row:
            c = compare(row["g3_1"], row["g3_4"])
            row["g3_4"]["vs_g1"] = c
            row["g3_4"]["vs_g1_state"] = kv_split(snaps["g3_4"], snaps["g3_1"], G)
            line += (f" | g4-vs-g1 dlp0={c['dlogprob_first']:.2e}"
                     f" kv_img={row['g3_4']['vs_g1_state']['kv_img_rel_l2_max']:.2e}")
        print(line, flush=True)
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)

    names = IMAGES[:a.n_images]
    print(f"\nGATE {a.gate.upper()} SUMMARY vs the one-shot arm (`one`); "
          f"judge every row against the `chunk` control, not against a constant:")
    for k in ("chunk", "g2", "g3_1", "g3_4"):
        rows = [res[n][k] for n in names if k in res.get(n, {})]
        cmps = [r["vs_one"] for r in rows if "vs_one" in r]
        if not cmps:
            continue
        first_eq = sum(1 for n in names
                       if k in res.get(n, {})
                       and res[n][k]["token_ids"][:1] == res[n]["one"]["token_ids"][:1])
        sts = [r["vs_one_state"] for r in rows if "vs_one_state" in r]
        print(f"  {k:5s} first-token argmax {first_eq}/{len(cmps)}  seq exact "
              f"{sum(c['exact'] for c in cmps)}/{len(cmps)}  max |dlogprob| first "
              f"{max(c['dlogprob_first'] for c in cmps):.3e}\n"
              f"        KV rel-L2 max: image {max(s['kv_img_rel_l2_max'] for s in sts):.3e}"
              f"  text {max(s.get('kv_txt_rel_l2_max', 0.0) for s in sts):.3e}")
    print(f"wrote {a.out}")


# --- gate F: the FLOPs reconcile (CPU) ----------------------------------------------------------- #

def prompt_lengths(model: str, datasets, n: int) -> dict:
    """Per-sample prompt token counts, from the HF processor alone (no model, no GPU).

    Per sample, never a mean: the softmax term is quadratic in N, so the mean of the shapes is
    not the shape of the mean (`flops_analytic.QWEN35_SHAPES` carries the same warning)."""
    from transformers import AutoProcessor
    from datasets import load_dataset
    from qwen_vl_prefill.datasets_eval import get_spec
    proc = AutoProcessor.from_pretrained(model)
    tmpl = dict(tokenize=False, add_generation_prompt=True, enable_thinking=False)
    out = {}
    for name in datasets:
        spec = get_spec(name)
        ds = spec.load(load_dataset)
        idxs = list(range(0, len(ds), max(1, len(ds) // n)))[:n]
        lens = []
        for i in idxs:
            img, q, _ = spec.prepare(ds[int(i)], lambda h, w, **kw: (h, w), 1, 1, 1 << 30)
            msgs = [{"role": "user",
                     "content": [{"type": "image"}, {"type": "text", "text": q}]}]
            text = proc.apply_chat_template(msgs, **tmpl)
            enc = proc(text=[text], images=[img.convert("RGB")], return_tensors="pt")
            lens.append(int(enc["input_ids"].shape[1]))
        out[name] = lens
    return out


def gate_flops(a):
    from flops_analytic import GLM46V_VISION, MODELS46
    dec = MODELS46[a.il_model]
    meas = json.load(open(a.flops_json))
    shapes = json.load(open(a.shapes_json)) if a.shapes_json else \
        prompt_lengths(a.model, a.datasets, a.samples)

    print(f"{'dataset':<14}{'arm':<10}{'N (mean)':>10}{'meas full':>11}{'meas V':>10}"
          f"{'meas L':>10}{'calc L':>10}{'ratio':>9}")
    worst, rows = 0.0, []
    for ds in a.datasets:
        ns = shapes[ds]
        split = meas[ds]["_split"]
        for arm in ("ceiling", "floor"):
            if arm not in split:
                continue
            v_tot, _, l_tot, _ = split[arm]
            calc = sum(dec.prefill_flops(n) for n in ns) / len(ns) / 1e9
            r = calc / max(l_tot, 1e-30)
            worst = max(worst, abs(r - 1))
            rows.append({"dataset": ds, "arm": arm, "n_mean": sum(ns) / len(ns),
                         "meas_vision": v_tot, "meas_llm": l_tot, "calc_llm": calc, "ratio": r})
            print(f"{ds:<14}{arm:<10}{sum(ns) / len(ns):>10.1f}{meas[ds]['full']:>11.1f}"
                  f"{v_tot:>10.1f}{l_tot:>10.1f}{calc:>10.1f}{r:>9.5f}")
        # the tower half, as a separate line: closed form vs the hooked vision column
        nrows = None
        if a.image_rows and ds in a.image_rows:
            nrows = a.image_rows[ds]
        if nrows:
            vcalc = GLM46V_VISION.tower_flops(nrows) / 1e9
            print(f"{ds:<14}{'tower':<10}{'-':>10}{'-':>11}"
                  f"{split['ceiling'][0]:>10.1f}{'-':>10}{vcalc:>10.1f}"
                  f"{vcalc / max(split['ceiling'][0], 1e-30):>9.5f}")
    ok = worst < 0.01
    print(f"\n  worst |calc/meas - 1| on the decoder prefill: {100 * worst:.3f}%  "
          f"(gate F: < 1%)  -> {'PASS' if ok else 'FAIL'}")
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump({"_model": a.model, "shapes": shapes, "rows": rows,
                   "worst_abs_rel_err": worst, "PASS": ok}, open(a.out, "w"), indent=1)
        print(f"wrote {a.out}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", default="g2", choices=["g2", "g3", "flops"])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--g", type=int, default=4, help="number of correction bands")
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--n-images", type=int, default=len(IMAGES))
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-num-seqs", type=int, default=1024)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--downscale", type=int, default=4)
    ap.add_argument("--chunks", type=int, default=4, help="chunks for the `chunk` control arm")
    ap.add_argument("--out", default=None)
    # gate flops
    ap.add_argument("--flops-json", default="analysis/results/flops/glm46v_flops.json")
    ap.add_argument("--shapes-json", default=None,
                    help="precomputed {dataset: [N, ...]}; recomputed with the HF processor if absent")
    ap.add_argument("--datasets", nargs="+", default=["realworldqa"])
    ap.add_argument("--samples", type=int, default=12)
    ap.add_argument("--il-model", default="glm46v")
    a = ap.parse_args()
    a.image_rows = None
    if a.gate == "flops":
        a.out = a.out or os.path.join(ROOT, "analysis/results/flops/glm46v_flops_gateF.json")
        raise SystemExit(0 if gate_flops(a) else 1)
    a.out = a.out or os.path.join(
        ROOT, f"analysis/results/vllm_stream/correct_gate_{a.model.split('/')[-1].lower()}.json")
    gate_engine(a)


if __name__ == "__main__":
    main()
