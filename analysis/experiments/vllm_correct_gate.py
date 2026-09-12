"""Gate for interleaved k<1 LLM correction inside vLLM (appcorr.vllm_stream.correct) on Qwen3.5.

Arms (same images, same question, greedy decoding, logprob of every generated token):

  one    one-shot reference: the CORRECTED prompt as client-composed prompt_embeds in one message
  g1     softmax-only rewrite: open with the corrected prompt (approx == corrected), let the stock
         engine prefill it, then `correct(P=[lo, N-1), final=True)` with the GDN layers in *replay*
         mode (the approx pass' captured GDN output is replayed for the corrected rows, so the
         residual stream is exact and only the softmax K/V rewrite is under test)
  g2     identity: open with the APPROX prompt (4x down-then-up-sampled image, same grid, same N),
         push the corrected image rows band by band (`--g` bands), the last band carrying the text
         suffix, then compare against `one`
  g3     g=1 vs g=4, both against `one`
  g6     depth-staged identity: `g2`'s messages tagged stage=(r, g) (correct.appcorr_staged_correct);
         at keep=1 the staged final state equals the unstaged one, so both must match `one`
  g7     CUDA-graph correct step: `g2cg` (PIECEWISE replay, |P| padded to a capture size) vs
         `g2` (eager) on the same images -- identity within the bf16 tiling band, plus step times.
  g6k    keep<1 sanity of the staged form (every other group corrected, band 1 sends NOTHING so the
         skipped-round catch-up walk runs): staged vs unstaged vs `one` -- no identity expected,
         the staged first-token logprob is expected CLOSER to `one` (hypothesis, reported)
  chunk  the engine's OWN noise band, in-process: the same corrected prompt prefilled in
         `--chunks` streaming chunks instead of one (arm C of vllm_stream_gate.py). Any KV /
         ssm / logprob difference of `g1`/`g2` must sit inside this band.
  band   the same band across engines: the `one` arm under `--max-num-batched-tokens 96`
         (a SECOND invocation / second engine, like arm D of vllm_stream_gate.py)

Reported per image: generated-token exact match vs `one`, |dlogprob| of the first token, max over
layers of the KV rel-L2 at the prompt slots, and (g2/g3) the mamba block's ssm rel-L2 / conv
max-abs-diff against `one`'s block.  Results merge into `--out`.

Run (GPU0, appcorr-vllm env = vllm 0.28.0, offline HF cache):
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  PYTHONPATH=/NHNHOME/share/cjpark/AppCorr-il-engine \
  /NHNHOME/storage/users/cjpark/shk/conda_envs/appcorr-vllm/bin/python \
      analysis/experiments/vllm_correct_gate.py --gate g1
  ... --gate g2 --g 4
  ... --gate band --max-num-batched-tokens 96
  ... --model Qwen/Qwen3.5-35B-A3B --gpu-mem 0.6 --out .../correct_gate_qwen35_35b.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from vllm_stream_gate import COCO, IMAGES, QUESTION, compare, tokens_and_lp  # noqa: E402


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    den = b.float().norm().item()
    return float((a.float() - b.float()).norm().item() / max(den, 1e-30))


def cmp_snapshots(s_a: dict, s_b: dict) -> dict:
    kv = {ln: rel_l2(s_a["kv"][ln], s_b["kv"][ln]) for ln in s_b["kv"]}
    ssm, conv = {}, {}
    for ln, (c_b, m_b) in s_b["mamba"].items():
        c_a, m_a = s_a["mamba"][ln]
        ssm[ln] = rel_l2(m_a, m_b)
        conv[ln] = float((c_a - c_b).abs().max().item())
    worst_kv = max(kv, key=kv.get)
    worst_ssm = max(ssm, key=ssm.get)
    return {"kv_rel_l2_max": kv[worst_kv], "kv_rel_l2_max_layer": worst_kv,
            "ssm_rel_l2_max": ssm[worst_ssm], "ssm_rel_l2_max_layer": worst_ssm,
            "conv_absdiff_max": max(conv.values()),
            "kv_rel_l2": kv, "ssm_rel_l2": ssm, "conv_absdiff": conv}


def bands_of(n_groups: int, g: int) -> list[tuple[int, int]]:
    cuts = [n_groups * k // g for k in range(g + 1)]
    return list(zip(cuts[:-1], cuts[1:]))


def run_to_first_token(llm, rid: str, max_steps: int = 512):
    """Step until the request has sampled its first token (prompt KV is then complete)."""
    for _ in range(max_steps):
        st = llm.stream_state(rid)
        if st is not None and st["num_output_tokens"] >= 1:
            return st
        llm.step()
    raise RuntimeError(f"{rid}: no token sampled in {max_steps} steps")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--gate", default="g2", choices=["g0", "g1", "g2", "g3", "g6", "g6k", "g7", "g8", "band"])
    ap.add_argument("--g", type=int, default=4, help="number of correction bands")
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--n-images", type=int, default=len(IMAGES))
    ap.add_argument("--gpu-mem", type=float, default=0.3)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--max-num-seqs", type=int, default=None)
    ap.add_argument("--max-num-batched-tokens", type=int, default=None)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--attn-backend", default=None, help="VLLM_ATTENTION_BACKEND override")
    ap.add_argument("--downscale", type=int, default=4)
    ap.add_argument("--chunks", type=int, default=4, help="chunks for the `chunk` control arm")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tag = a.model.split("/")[-1].replace(".", "").lower()
    out = a.out or os.path.join(ROOT, f"analysis/results/vllm_stream/correct_gate_{tag}.json")
    if a.attn_backend:
        os.environ["VLLM_ATTENTION_BACKEND"] = a.attn_backend
    # vllm 0.28.0 defaults VLLM_GDN_DECODE_KERNEL to "cuda", which routes every GDN layer through
    # `_forward_core_fused_norm_packed` and past the correction hooks (see correct.check_gdn_path)
    os.environ.setdefault("VLLM_GDN_DECODE_KERNEL", "triton")

    from PIL import Image
    from vllm import SamplingParams
    from appcorr.vllm_stream import StreamingLLM
    from appcorr.vllm_stream.client import Qwen25VLComposer

    kw = {}
    if a.max_num_seqs:
        kw["max_num_seqs"] = a.max_num_seqs
    if a.max_num_batched_tokens:
        kw["max_num_batched_tokens"] = a.max_num_batched_tokens
    llm = StreamingLLM(a.model, gpu_memory_utilization=a.gpu_mem, max_model_len=a.max_model_len,
                       enforce_eager=a.enforce_eager, limit_mm_per_prompt={"image": 1}, **kw)
    comp = Qwen25VLComposer(llm)
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens, logprobs=1)
    runner = llm.runner

    res = json.load(open(out)) if os.path.exists(out) else {"_meta": {}}
    backends = {}
    for gid, grp in enumerate(runner.kv_cache_config.kv_cache_groups):
        backends[str(gid)] = [ag.backend.__name__ for ag in runner.attn_groups[gid]]
    res["_meta"].update({
        "model": a.model, "max_tokens": a.max_tokens, "question": QUESTION,
        "images": IMAGES[:a.n_images], "downscale": a.downscale,
        "attn_backends": backends,
        "max_num_seqs": llm.engine.vllm_config.scheduler_config.max_num_seqs,
        "max_num_batched_tokens": llm.engine.vllm_config.scheduler_config.max_num_batched_tokens,
    })
    if a.gate == "band":
        res["_meta"]["band_max_num_batched_tokens"] = a.max_num_batched_tokens
    print("attn backends:", backends, flush=True)

    gates = {"g0": ["one", "chunk", "g0"], "g1": ["one", "chunk", "g1"], "g2": ["one", "chunk", "g2"],
             "g3": ["one", "chunk", "g3_1", "g3_4"], "g6": ["one", "g2", "g6"],
             "g6k": ["one", "g2k", "g6k"], "g7": ["one", "g2", "g2b", "g2pe", "g2cg"],
             "g8": ["one", "g2", "g2d", "g2f", "g6f"],
             "band": ["band"]}[a.gate]
    if a.gate == "g7":
        from appcorr.vllm_stream import correct as _correct
        _correct.CUDAGRAPH = False   # `g2` = eager reference, `g2cg` = PIECEWISE graph replay
    if a.gate == "g8":
        # fused hold-back: `g2` = synchronous final step (reference), `g2d` = deferred into the
        # release step but unfused (must equal g2 bitwise), `g2f` = fused (N-1 computed in the
        # correct batch, state committed past it, release step samples from the stashed row);
        # `g6f` = the staged form fused. Graphs at their default (the served path).
        from appcorr.vllm_stream import correct as _correct
        llm.defer_final = False

    for ii, name in enumerate(IMAGES[:a.n_images]):
        img = Image.open(os.path.join(COCO, name)).convert("RGB")
        w, h = img.size
        parts = comp.parts(img, QUESTION)
        N, lo, G = parts.num_tokens, parts.image_start, parts.image_len
        emb = comp.embed(parts)
        row = res.setdefault(name, {})
        row.update({"num_prompt_tokens": N, "image_span": [lo, lo + G]})
        cmp_pos = torch.arange(lo, N - 1, dtype=torch.int64)

        emb_a = None
        if any(k.startswith(("g2", "g3")) for k in gates):
            low = img.resize((max(w // a.downscale, 1), max(h // a.downscale, 1))).resize((w, h))
            parts_a = comp.parts(low, QUESTION)
            assert (parts_a.num_tokens, parts_a.image_start, parts_a.image_len) == (N, lo, G), (
                "the approx prompt must have the same grid/length as the corrected one")
            emb_a = comp.embed(parts_a)
            d = (emb_a.embeds[lo:lo + G].float() - emb.embeds[lo:lo + G].float())
            row["approx_img_rel_l2"] = float(d.norm() / emb.embeds[lo:lo + G].float().norm())
            assert torch.equal(emb_a.embeds[:lo], emb.embeds[:lo]), "pre-image text rows differ"
            assert torch.equal(emb_a.embeds[lo + G:], emb.embeds[lo + G:]), "post-image text rows differ"

        snaps = {}

        def one_shot(key: str):
            rid = f"{key}-{ii}"
            t0 = time.perf_counter()
            llm.open(rid, emb.chunk(0, N, final=True), sp)
            run_to_first_token(llm, rid)
            snaps[key] = runner.appcorr_snapshot(llm._core_id(rid), cmp_pos)
            o = llm.run_until_done(rid)[rid]
            row[key] = tokens_and_lp(o) | {"wall_s": time.perf_counter() - t0}

        if "one" in gates or "band" in gates:
            one_shot("one" if "one" in gates else "band")

        if "chunk" in gates:
            rid = f"chunk-{ii}"
            t0 = time.perf_counter()
            bounds = comp.image_bounds(parts, a.chunks)
            chunks = emb.chunks(bounds)
            llm.open(rid, chunks[0], sp)
            for ch in chunks[1:]:
                llm.step()
                llm.append(rid, ch)
            run_to_first_token(llm, rid)
            snaps["chunk"] = runner.appcorr_snapshot(llm._core_id(rid), cmp_pos)
            o = llm.run_until_done(rid)[rid]
            row["chunk"] = tokens_and_lp(o) | {"wall_s": time.perf_counter() - t0, "bounds": bounds}

        if "g0" in gates:
            from appcorr.vllm_stream import correct as _correct
            _correct.set_debug_state(True)
            # isolate the GDN re-scan: corrected prompt (so the approx pass already holds the
            # right state), rewrite ONE row with its own embedding, window = the whole prompt.
            # `_rescan` then recomputes [0, N-1) from the captured pre-conv inputs and its result
            # must reproduce what the stock prefill left in the request's mamba block.
            rid = f"g0-{ii}"
            t0 = time.perf_counter()
            llm.open(rid, emb.chunk(0, N, final=False), sp, correct=True, image_start=lo)
            llm._drain_until_prefilled(rid)
            before = runner.appcorr_snapshot(llm._core_id(rid), cmp_pos)
            pos = torch.arange(N - 2, N - 1, dtype=torch.int64)
            info = llm.correct(rid, pos, emb.embeds[N - 2:N - 1], (0, N - 1), True)
            after = runner.appcorr_snapshot(llm._core_id(rid), cmp_pos)
            run_to_first_token(llm, rid)
            o = llm.run_until_done(rid)[rid]
            row["g0"] = tokens_and_lp(o) | {"wall_s": time.perf_counter() - t0, "info": info,
                                            "state_before_after": cmp_snapshots(after, before)}

        if "g1" in gates:
            rid = f"g1-{ii}"
            t0 = time.perf_counter()
            llm.open(rid, emb.chunk(0, N, final=False), sp, correct=True, capture_out=True,
                     image_start=lo)
            llm._drain_until_prefilled(rid)
            before = runner.appcorr_snapshot(llm._core_id(rid), cmp_pos)
            pos = torch.arange(lo, N - 1, dtype=torch.int64)
            info = llm.correct(rid, pos, emb.embeds[lo:N - 1], (lo, N - 1), True, replay=True)
            after = runner.appcorr_snapshot(llm._core_id(rid), cmp_pos)
            run_to_first_token(llm, rid)
            o = llm.run_until_done(rid)[rid]
            row["g1"] = tokens_and_lp(o) | {"wall_s": time.perf_counter() - t0, "info": info,
                                            "kv_before_after": cmp_snapshots(after, before)}

        def interleaved(key: str, g: int, staged: bool = False, keep_half: bool = False):
            rid = f"{key}-{ii}"
            t0 = time.perf_counter()
            llm.open(rid, emb_a.chunk(0, N, final=False), sp, correct=True, image_start=lo,
                     image_end=lo + G)
            infos = []
            bs = bands_of(G, g)
            for r, (g0, g1) in enumerate(bs):
                final = r == len(bs) - 1
                pos = torch.arange(lo + g0, lo + g1, dtype=torch.int64)
                if keep_half:
                    # every other row of the band; band 1 sends nothing at all (skipped round)
                    pos = pos[::2] if r != 1 else pos[:0]
                rows = emb.embeds[pos]
                win = (lo + g0, lo + g1)
                if final:
                    pos = torch.cat([pos, torch.arange(lo + G, N - 1, dtype=torch.int64)])
                    rows = torch.cat([rows, emb.embeds[lo + G:N - 1]], dim=0)
                    win = (lo + g0, N - 1)
                if pos.numel() == 0:
                    continue
                infos.append(llm.correct(rid, pos, rows, win, final,
                                         stage=((r, g) if staged else None)))
            run_to_first_token(llm, rid)
            snaps[key] = runner.appcorr_snapshot(llm._core_id(rid), cmp_pos)
            o = llm.run_until_done(rid)[rid]
            row[key] = tokens_and_lp(o) | {"wall_s": time.perf_counter() - t0, "bands": bs,
                                           "info": infos}

        if "g2" in gates:
            interleaved("g2", a.g)
        if "g2b" in gates:      # eager repeat: run-to-run determinism of the eager step
            interleaved("g2b", a.g)
        if "g2pe" in gates:     # padded to the capture size, eager: isolates the padded-M kernels
            from appcorr.vllm_stream import correct as _correct
            _correct.CUDAGRAPH, _correct.PAD_EAGER = True, True
            interleaved("g2pe", a.g)
            _correct.CUDAGRAPH, _correct.PAD_EAGER = False, False
        if "g2cg" in gates:
            from appcorr.vllm_stream import correct as _correct
            _correct.CUDAGRAPH = True
            interleaved("g2cg", a.g)
            _correct.CUDAGRAPH = False
        if "g2d" in gates:
            from appcorr.vllm_stream import correct as _correct
            llm.defer_final, _correct.FUSE_HOLDBACK = True, False
            interleaved("g2d", a.g)
            llm.defer_final = False
        if "g2f" in gates:
            from appcorr.vllm_stream import correct as _correct
            llm.defer_final, _correct.FUSE_HOLDBACK = True, True
            interleaved("g2f", a.g)
            llm.defer_final = False
        if "g6f" in gates:
            from appcorr.vllm_stream import correct as _correct
            llm.defer_final, _correct.FUSE_HOLDBACK = True, True
            interleaved("g6f", a.g, staged=True)
            llm.defer_final = False
        if "g6" in gates:
            interleaved("g6", a.g, staged=True)
        if "g2k" in gates:
            interleaved("g2k", a.g, keep_half=True)
        if "g6k" in gates:
            interleaved("g6k", a.g, staged=True, keep_half=True)
        if "g3_1" in gates:
            interleaved("g3_1", 1)
            interleaved("g3_4", 4)

        ref = row.get("one")
        line = f"[{ii}] {name} N={N} img=[{lo},{lo + G})"
        for k in gates:
            if k == "one" or k not in row:
                continue
            if ref is not None:
                c = compare(ref, row[k])
                row[k]["vs_one"] = c
                line += (f" | {k}: {'exact' if c['exact'] else 'div@' + str(c['first_divergence'])}"
                         f" dlp0={c['dlogprob_first']:.2e}")
            if k in snaps and "one" in snaps:
                st = cmp_snapshots(snaps[k], snaps["one"])
                row[k]["vs_one_state"] = st
                line += f" kv={st['kv_rel_l2_max']:.2e} ssm={st['ssm_rel_l2_max']:.2e}"
            for k2, rf in (("g2b", "g2"), ("g2pe", "g2"), ("g2cg", "g2pe"), ("g2d", "g2"),
                           ("g2f", "g2"), ("g6f", "g2f")):
                if k == k2 and rf in row:
                    c2 = compare(row[rf], row[k])
                    st2 = cmp_snapshots(snaps[k], snaps[rf])
                    row[k]["vs_" + rf] = c2
                    row[k]["vs_" + rf + "_state"] = st2
                    line += (f" | {k}-vs-{rf} dlp0={c2['dlogprob_first']:.2e}"
                             f" kv={st2['kv_rel_l2_max']:.2e} ssm={st2['ssm_rel_l2_max']:.2e}")
            if k in ("g2d", "g2f", "g6f") and "g2" in row:
                fi = row[k]["info"]
                line += (f" | {k} fused={[bool(i.get('fused')) for i in fi][-1]}"
                         f" step ms={[round(i['t_step_ms']) for i in fi]}"
                         f" ref={[round(i['t_step_ms']) for i in row['g2']['info']]}")
            if k == "g2cg" and "g2" in row:
                c = compare(row["g2"], row["g2cg"])
                row[k]["vs_eager"] = c
                st = cmp_snapshots(snaps["g2cg"], snaps["g2"])
                row[k]["vs_eager_state"] = st
                t_e = [i["t_step_ms"] for i in row["g2"]["info"]]
                t_g = [i["t_step_ms"] for i in row["g2cg"]["info"]]
                sizes = [i.get("cudagraph") for i in row["g2cg"]["info"]]
                line += (f" | graph-vs-eager dlp0={c['dlogprob_first']:.2e}"
                         f" kv={st['kv_rel_l2_max']:.2e} ssm={st['ssm_rel_l2_max']:.2e}"
                         f" step ms eager={[round(t) for t in t_e]} graph={[round(t) for t in t_g]}"
                         f" pad={sizes}")
            if k == "g6k" and "g2k" in row:
                c = compare(row["g2k"], row["g6k"])
                row[k]["vs_unstaged"] = c
                st = cmp_snapshots(snaps["g6k"], snaps["g2k"])
                row[k]["vs_unstaged_state"] = st
                line += (f" | staged-vs-unstaged dlp0={c['dlogprob_first']:.2e}"
                         f" kv={st['kv_rel_l2_max']:.2e} ssm={st['ssm_rel_l2_max']:.2e}")
            if k == "g1":
                line += f" kv(before/after)={row['g1']['kv_before_after']['kv_rel_l2_max']:.2e}"
            if k == "g0":
                b = row["g0"]["state_before_after"]
                line += (f" rescan-vs-stock: ssm={b['ssm_rel_l2_max']:.2e}"
                         f" conv={b['conv_absdiff_max']:.2e} kv={b['kv_rel_l2_max']:.2e}")
        print(line, flush=True)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        json.dump(res, open(out, "w"), indent=1)

    names = IMAGES[:a.n_images]
    print(f"\nGATE {a.gate.upper()} SUMMARY vs one-shot (`one`):")
    for k in ("chunk", "g0", "g1", "g2", "g2b", "g2pe", "g2cg", "g2d", "g2f", "g6f", "g3_1", "g3_4",
              "g6", "g2k", "g6k", "band"):
        rows = [res[n][k] for n in names if k in res.get(n, {})]
        cmps = [r["vs_one"] for r in rows if "vs_one" in r]
        if not cmps:
            continue
        ex = sum(c["exact"] for c in cmps)
        first_eq = sum(1 for n, c in zip(names, cmps)
                       if res[n][k]["token_ids"][:1] == res[n]["one"]["token_ids"][:1])
        msg = (f"  {k:5s} first-token argmax {first_eq}/{len(cmps)}  seq exact {ex}/{len(cmps)}"
               f"  max |dlogprob| first {max(c['dlogprob_first'] for c in cmps):.3e}"
               f"  max |dlogprob| common {max(c['max_dlogprob_common'] for c in cmps):.3e}")
        sts = [r["vs_one_state"] for r in rows if "vs_one_state" in r]
        if sts:
            msg += (f"\n        KV rel-L2 max {max(s['kv_rel_l2_max'] for s in sts):.3e}"
                    f"  ssm rel-L2 max {max(s['ssm_rel_l2_max'] for s in sts):.3e}"
                    f"  conv |d| max {max(s['conv_absdiff_max'] for s in sts):.3e}")
        print(msg)
    if a.gate == "g1":
        b = [res[n]["g1"]["kv_before_after"]["kv_rel_l2_max"] for n in names if "g1" in res.get(n, {})]
        if b:
            print(f"  g1 KV rel-L2 (rewrite vs approx pass, same embeds) max {max(b):.3e}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
