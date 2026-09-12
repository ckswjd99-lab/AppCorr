"""GPU gate for the GLM-5.3-Flash sparse-MLA half of the interleaved correct step (NOT RUN).

Item 3's gate (`docs/memo/glm53_correct_design.md`, "G-MLA rows step vs chunk control ... plus
indexer caches compared bytewise"), mirroring `analysis/experiments/glm46v_correct_gate.py`.
Written 2026-09-13 on CPU; every number below has to be produced on a GPU before it means
anything.

What it compares
----------------
Three arms over the same corrected prompt, on the 11 sparse layers
(`layer_types == "deepseek_sparse_attention"`: 3, 7, 11, ..., 43):

  one    the stock engine prefilling the corrected prompt in ONE shot.  Reference.
  chunk  CONTROL: the same corrected prompt prefilled by the stock engine in `--chunks`
         streaming chunks.  Its spread against `one` is the engine's own numerical band for
         this checkpoint -- an absolute pass mark would be meaningless (memo §6.6).  Chunk
         bounds MUST be multiples of `index_kpool` (=4): `Glm5NextIndexerCache.__init__`
         asserts `block_size % index_kpool == 0` precisely so that chunked-prefill boundaries
         stay pool-aligned, and `_kpool_compress_insert` "assumes pool-aligned chunk starts".
  rows   the correct step: open with the approximate prompt, then `llm.correct(...)` the image
         rows band by band.  At keep=1 every row is corrected exactly once, so the final state
         must equal `one`.

and for each arm snapshots, per sparse layer:

  latent   the MLA latent rows at the compared positions (512 per token, `MLAAttentionSpec(
           num_kv_heads=1, head_size=kv_lora_rank + qk_rope_head_dim = 512 + 0)`).  Compared by
           relative L2, because the arms differ by kernel arithmetic even in `chunk`.
  k_cache  the fp8 pooled indexer entries + their fp32 scales, BYTEWISE.  The pooled entry is a
           quantised value: a bytewise diff is the only diff that means anything at 8 bits
           (a 1-ulp fp8 step is 12.5% relative), so this is reported as "pools differing" and
           "max |dequantised| over the differing pools", never as a rel-L2 alone.
  tail     the request's tail block (raw bf16 K + gate of the last `index_kpool` rows), bytewise.
  logits   first-token argmax + |dlogprob|, the end-to-end number.

Why this gate and not the glm46v one
------------------------------------
GLM-4.6V is pure-softmax GQA: one cache per layer, and `appcorr_snapshot` reads it.  GLM-5.3's
sparse layers own THREE caches and only the latent is reachable through `appcorr_snapshot`; the
other two are addressed at pool granularity and per request, so this file adds `snapshot_indexer`
(below) rather than extending the shared helper.

Commands (B200-8, GPUs 2-3 for this window; one task per GPU)
--------------------------------------------------------------
The gate needs the vLLM env, not `appcorr` (which has no vllm):

  CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  APPCORR_CORRECT_CUDAGRAPH=0 \
  PYTHONPATH=/NHNHOME/share/cjpark/AppCorr-glm53 \
  /NHNHOME/storage/users/cjpark/shk/conda_envs/appcorr-vllm/bin/python \
      analysis/experiments/glm53_mla_gate.py --gate mla --g 4 \
      --gpu-mem 0.85 --max-model-len 8192 --max-num-seqs 1024 --chunks 4 \
      --out analysis/results/glm53/mla_gate.json

Both regimes are implemented.  `index_topk` is 2048, so a prompt under 2048 tokens exercises the
DENSE path (exact causal indices, no scoring) and anything above it the SPARSE path (stock
scoring over the rewritten pooled cache).  Run the dense case first as the simpler control:

  ... --gate mla --regime dense --max-model-len 4096

then the sparse case, which is the one that matters (V*Bench ~4k, RefCOCO/InfoVQA up to 8192):

  ... --gate mla --regime sparse --min-prompt-tokens 3000 --capture-topk

`--regime` asserts the observed regime per round instead of silently taking whichever fires.
`--capture-topk` snapshots the last sparse layer's `topk_indices_buffer` for the corrected rows
and checks the visibility invariant ON REAL DATA -- for every corrected row p, every returned
key <= p, and the in-progress tail [((p+1)//4)*4, p+1) is present.  That is the GPU half of
`tests/test_glm53_indexer.py::test_d_*`, which can only check the index arithmetic on CPU.

To isolate a regression to the rewrite rather than the scoring, re-run with
`APPCORR_GLM53_SPARSE=0` (a >2048 round then raises instead of running) and with
`APPCORR_GLM53_CHECK=0` (drops ~33 device syncs per round; the numbers must not move).
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

from vllm_stream_gate import COCO, IMAGES, QUESTION, compare, tokens_and_lp   # noqa: E402
from vllm_correct_gate import bands_of, rel_l2, run_to_first_token            # noqa: E402

MODEL = "zai-org/GLM-5.3-Flash"


# --- snapshots ---------------------------------------------------------------------------- #

def snapshot_indexer(runner, req_id: str, n_valid: int) -> dict:
    """The three sparse-layer caches for one request, as CPU tensors.

    `appcorr_snapshot` covers the MLA latent (it is an ordinary attention kv-cache group); the
    pooled indexer cache and the tail cache are addressed at pool granularity and one block per
    request respectively, so they are read here through `glm53_indexer`'s own accessors -- the
    same code path the rewrite writes with, which is what makes a bytewise comparison a test of
    the rewrite and not of a second, independent reader."""
    from appcorr.vllm_stream import glm53_indexer as gi

    caches = gi.layer_caches(runner, req_id)
    out = {}
    for key, lc in caches.items():
        n_pools = n_valid // lc.kpool
        pools = torch.arange(n_pools, device=lc.k_cache.device)
        slots = lc.slots_of(pools)
        entry, scale = gi.k_cache_read(lc.k_cache, slots, lc.head_dim)
        tk, tg = gi.tail_read(lc.tail_cache, lc.tail_block, lc.kpool, lc.head_dim)
        out[key] = {
            "entry_u8": entry.view(torch.uint8).cpu().clone(),
            "scale": scale.float().cpu().clone(),
            "tail_k": tk.float().cpu().clone(),
            "tail_gate": tg.float().cpu().clone(),
            "n_pools": n_pools,
        }
    return out


def cmp_indexer(a: dict, b: dict) -> dict:
    """Bytewise pooled-entry / tail comparison, per layer, worst layer summarised."""
    rows = {}
    for key in sorted(a):
        ea, eb = a[key]["entry_u8"], b[key]["entry_u8"]
        diff = (ea != eb).any(dim=-1)
        n_diff = int(diff.sum())
        # dequantised magnitude of the disagreement, over the differing pools only
        da = ea.view(torch.float8_e4m3fn).float() * a[key]["scale"][:, None]
        db = eb.view(torch.float8_e4m3fn).float() * b[key]["scale"][:, None]
        dmax = float((da - db).abs().max()) if ea.numel() else 0.0
        rows[key] = {
            "pools": int(ea.shape[0]),
            "pools_differing": n_diff,
            "pools_differing_frac": n_diff / max(1, int(ea.shape[0])),
            "scale_bitwise": bool(torch.equal(a[key]["scale"], b[key]["scale"])),
            "entry_dequant_absmax": dmax,
            "tail_k_bitwise": bool(torch.equal(a[key]["tail_k"], b[key]["tail_k"])),
            "tail_gate_bitwise": bool(torch.equal(a[key]["tail_gate"], b[key]["tail_gate"])),
            "tail_k_absmax": float((a[key]["tail_k"] - b[key]["tail_k"]).abs().max()),
        }
    worst = max(rows, key=lambda k: rows[k]["pools_differing_frac"])
    return {"per_layer": rows, "worst_layer": worst,
            "pools_differing_frac_max": rows[worst]["pools_differing_frac"],
            "all_bitwise": all(r["pools_differing"] == 0 and r["tail_k_bitwise"]
                               and r["tail_gate_bitwise"] for r in rows.values())}


def cmp_latent(a: dict, b: dict, sparse_only: bool = True) -> dict:
    """Max-over-layers rel-L2 of the MLA latent rows (`appcorr_snapshot`'s `kv` dict)."""
    d = {}
    for ln, bt in b["kv"].items():
        if sparse_only and ".attn" not in ln:
            continue
        d[ln] = rel_l2(a["kv"][ln], bt)
    worst = max(d, key=d.get)
    return {"latent_rel_l2_max": d[worst], "latent_rel_l2_max_layer": worst,
            "latent_rel_l2": d}


def install_topk_capture(runner):
    """Stash the last sparse layer's returned `topk_indices_buffer` rows for each correct round.

    The buffer is shared by all 11 layers and rewritten in place per layer, so only a hook can
    see a round's value.  Wraps the op the same way `glm53_indexer` does (per-instance
    `_forward_method` on the `CustomOp`), outside its own hook so the two compose."""
    from appcorr.vllm_stream import correct as _correct
    from appcorr.vllm_stream import glm53_indexer as gi

    sparse = gi.sparse_layers(_correct._decoder(runner))
    _, _mod, ix = sparse[-1]
    op = ix.indexer_op
    stash = {"rows": None}
    inner = op._forward_method

    def wrapper(hidden_states, *a, **kw):
        out = inner(hidden_states, *a, **kw)
        n = int(hidden_states.shape[0])
        stash["rows"] = out[:n].detach().clone()
        return out

    op._forward_method = wrapper
    return stash, (lambda: setattr(op, "_forward_method", inner))


def check_visibility_invariant(topk_rows: torch.Tensor, positions: torch.Tensor,
                               kpool: int) -> dict:
    """For every corrected row p: max returned key <= p, and the tail pool is present.

    This is the on-device form of the invariant `glm53_indexer`'s sparse path relies on
    (`index_kpool_always_select_tail`): scoring may only return COMPLETE pools, all of whose
    tokens are <= p, and the in-progress pool [((p+1)//kpool)*kpool, p+1) is appended
    unconditionally."""
    rows = topk_rows.cpu()
    pos = positions.cpu().to(torch.int64)
    worst, missing, n_keys = -(10 ** 9), 0, []
    for i, p in enumerate(pos.tolist()):
        keys = rows[i]
        keys = keys[keys >= 0].to(torch.int64)
        n_keys.append(int(keys.numel()))
        if keys.numel():
            worst = max(worst, int(keys.max()) - p)
        start = ((p + 1) // kpool) * kpool
        tail = torch.arange(start, p + 1, dtype=torch.int64)
        if tail.numel() and not bool(torch.isin(tail, keys).all()):
            missing += 1
    return {"max_key_minus_p": worst, "rows_missing_tail": missing,
            "keys_per_row_min": min(n_keys), "keys_per_row_max": max(n_keys),
            "causal": worst <= 0, "tail_complete": missing == 0}


# --- the gate ------------------------------------------------------------------------------ #

# ---------------------------------------------------------------------------------------------
# TP wrapper.  GLM-5.3-Flash needs two B200s, so `llm.runner` (the in-process runner) does not
# exist for this model; the four places this gate touched it are shipped to EVERY rank through
# `StreamingLLM.run_on_ranks` (`appcorr/vllm_stream/tp_worker.py`).  The comparison functions
# above -- `snapshot_indexer`, `cmp_indexer`, `check_visibility_invariant` -- are UNCHANGED; only
# where `snapshot_indexer` runs moved.
#
# Unlike the KDA state, everything this gate compares is REPLICATED across TP ranks: the MLA
# latent cache is MQA (one kv head) and the whole sparse indexer is built with `ReplicatedLinear`
# / `disable_tp=True` (`vllm/models/glm5next/nvidia/attention.py:250-266`).  So every rank must
# rewrite them identically, and `_digest` + `agree_across_ranks` assert exactly that -- the check
# that TP cannot silently half-apply a correction.
# ---------------------------------------------------------------------------------------------

def _digest(obj) -> dict:
    """Order-independent float fingerprint of a snapshot, for cross-rank comparison.

    Shipping two ranks' full snapshots back and diffing them would move hundreds of MB through
    the reply queue; a per-layer (sum, abs-sum, count) triple over the same bytes catches any
    difference that matters and costs nothing. Exact equality is the bar -- these tensors are
    replicated, so the ranks are not allowed to differ by rounding either."""
    out = {}

    def walk(prefix, v):
        if isinstance(v, torch.Tensor):
            f = v.reshape(-1).to(torch.float64)
            out[prefix] = [float(f.sum()), float(f.abs().sum()), int(f.numel())]
        elif isinstance(v, dict):
            for k2, v2 in v.items():
                walk(f"{prefix}/{k2}", v2)
        elif isinstance(v, (list, tuple)):
            for i2, v2 in enumerate(v):
                walk(f"{prefix}[{i2}]", v2)
        elif isinstance(v, (int, float)):
            out[prefix] = [float(v), abs(float(v)), 1]

    walk("", obj)
    return out


def probe_setup(runner) -> dict:
    """The structural asserts + backend metadata this gate opened with, now per rank."""
    from appcorr.vllm_stream import correct as _correct
    from appcorr.vllm_stream import glm53_indexer as gi

    dec = _correct._decoder(runner)
    sparse = gi.sparse_layers(dec)
    mla = gi.mla_layers(dec)
    assert len(dec.layers) == 45, len(dec.layers)
    assert len(mla) == len(sparse) == 11, (len(mla), len(sparse))
    topk = gi.topk_tokens_of(runner.model)
    assert topk == 2048, topk          # the checkpoint's index_topk; the dense predicate uses it
    kp = {int(ix.index_kpool) for _, _m, ix in sparse}
    assert kp == {4}, kp
    return {
        "device": str(runner.device),
        "sparse_layers": [i for i, _m, _x in sparse],
        "topk_tokens": topk,
        "index_kpool": 4,
        "attn_backends": {str(g): [ag.backend.__name__ for ag in runner.attn_groups[g]]
                          for g in range(len(runner.kv_cache_config.kv_cache_groups))},
    }


def snapshot_body(runner, cid: str, cmp_pos, n_valid: int) -> dict:
    """`appcorr_snapshot` + `snapshot_indexer` on this rank, plus their digests."""
    snap = runner.appcorr_snapshot(cid, cmp_pos)
    ix = snapshot_indexer(runner, cid, n_valid)
    return {"snap": snap, "ix": ix, "device": str(runner.device),
            "digest": {"snap": _digest(snap), "ix": _digest(ix)}}


def arm_topk(runner) -> str:
    stash, undo = install_topk_capture(runner)
    runner._appcorr_mla_topk = (stash, undo)
    return str(runner.device)


def read_topk(runner):
    stash, _undo = runner._appcorr_mla_topk
    r = stash["rows"]
    return None if r is None else r.detach().cpu()


def disarm_topk(runner) -> bool:
    st = getattr(runner, "_appcorr_mla_topk", None)
    if st is None:
        return False
    st[1]()
    del runner._appcorr_mla_topk
    return True


def gate_mla(a):
    from PIL import Image
    from vllm import SamplingParams
    from appcorr.vllm_stream import StreamingLLM
    from appcorr.vllm_stream import correct as _correct
    from appcorr.vllm_stream import glm53_indexer as gi

    kw = {"max_num_seqs": a.max_num_seqs} if a.max_num_seqs else {}
    llm = StreamingLLM(a.model, gpu_memory_utilization=a.gpu_mem, max_model_len=a.max_model_len,
                       enforce_eager=a.enforce_eager, limit_mm_per_prompt={"image": 1},
                       tensor_parallel_size=a.tensor_parallel_size, **kw)
    from appcorr.vllm_stream.client import Glm53Composer          # agent V's composer
    from appcorr.vllm_stream.tp_worker import agree_across_ranks
    comp = Glm53Composer(llm)
    sp = SamplingParams(temperature=0.0, max_tokens=a.max_tokens, logprobs=1)

    # (a) the runner patch reached every rank; SP is off (it would put collectives inside the
    # layer that a per-rank row rewrite is not aligned to).
    winfo = llm.worker_info()
    assert all(w["correct_patched"] and w["has_correct_step"] for w in winfo), winfo
    assert not any(w["sequence_parallel_moe"] for w in winfo), winfo
    setup = llm.run_on_ranks(probe_setup)
    topk = setup[0]["topk_tokens"]
    _ag = agree_across_ranks(setup, ["topk_tokens", "index_kpool"])
    assert _ag["agree"], _ag

    res = json.load(open(a.out)) if os.path.exists(a.out) else {"_meta": {}}
    res["_meta"].update({
        "model": a.model, "gate": a.gate, "g": a.g, "chunks": a.chunks, "regime": a.regime,
        "env_sparse": os.environ.get("APPCORR_GLM53_SPARSE", "1"),
        "env_check": os.environ.get("APPCORR_GLM53_CHECK", "1"),
        "tensor_parallel_size": a.tensor_parallel_size, "worker_info": winfo,
        "sparse_layers": setup[0]["sparse_layers"], "topk_tokens": topk, "index_kpool": 4,
        "block_size": llm.engine.vllm_config.cache_config.block_size,
        "max_num_seqs": llm.engine.vllm_config.scheduler_config.max_num_seqs,
        "attn_backends": setup[0]["attn_backends"],
        "side_buffer_B_per_token_per_layer": gi.IndexerSideBuffer.bytes_per_token_per_layer(),
    })
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)

    for ii, name in enumerate(IMAGES[:a.n_images]):
        img = Image.open(os.path.join(COCO, name)).convert("RGB")
        w, h = img.size
        parts = comp.parts(img, QUESTION)
        N, lo, G = parts.num_tokens, parts.image_start, parts.image_len
        if a.min_prompt_tokens and N < a.min_prompt_tokens:
            continue
        emb = comp.embed(parts)
        row = res.setdefault(name, {})
        row.update({"num_prompt_tokens": N, "image_span": [lo, lo + G],
                    "dense_regime": gi.dense_regime(N - 1, topk)})
        cmp_pos = torch.arange(lo, N - 1, dtype=torch.int64)

        low = img.resize((max(w // a.downscale, 1), max(h // a.downscale, 1))).resize((w, h))
        parts_a = comp.parts(low, QUESTION)
        assert (parts_a.num_tokens, parts_a.image_start, parts_a.image_len) == (N, lo, G), (
            "the approx prompt must have the same grid/length as the corrected one")
        emb_a = comp.embed(parts_a)

        snaps, ix_snaps = {}, {}

        def take(key, rid):
            cid = llm._core_id(rid)
            per_rank = llm.run_on_ranks(snapshot_body, cid, cmp_pos, N - 1)
            snaps[key] = per_rank[0]["snap"]
            ix_snaps[key] = per_rank[0]["ix"]
            # The MLA latent and BOTH indexer caches are REPLICATED across TP ranks
            # (`attention.py:250-266`), so a correction that reached only some ranks shows up
            # here and nowhere else. Digests, not the tensors: two full snapshots per rank would
            # be hundreds of MB through the reply queue for a comparison that is exact anyway.
            digests = [{f"{w}:{k}": tuple(v) for w, d in r["digest"].items() for k, v in d.items()}
                       for r in per_rank]
            agree = agree_across_ranks(digests, sorted(digests[0]))
            row.setdefault("rank_agreement", {})[key] = {
                "n_ranks": agree["n_ranks"], "agree": agree["agree"],
                "n_disagreements": len(agree["disagreements"]),
                "first": sorted(agree["disagreements"])[:3]}
            assert agree["agree"], (
                f"ranks disagree on the REPLICATED {key} snapshot: "
                f"{sorted(agree['disagreements'])[:5]}")

        def one_shot(key):
            rid = f"{key}-{ii}"
            t0 = time.perf_counter()
            llm.open(rid, emb.chunk(0, N, final=True), sp)
            run_to_first_token(llm, rid)
            take(key, rid)
            o = llm.run_until_done(rid)[rid]
            row[key] = tokens_and_lp(o) | {"wall_s": time.perf_counter() - t0}

        def chunked(key):
            rid = f"{key}-{ii}"
            t0 = time.perf_counter()
            bounds = comp.image_bounds(parts, a.chunks)
            # pool alignment: `Glm5NextIndexerCache` requires pool-aligned chunk starts
            assert all(b % 4 == 0 for b in bounds), (
                f"chunk bounds must be multiples of index_kpool=4, got {bounds}")
            chunks = emb.chunks(bounds)
            llm.open(rid, chunks[0], sp)
            for ch in chunks[1:]:
                llm.step()
                llm.append(rid, ch)
            run_to_first_token(llm, rid)
            take(key, rid)
            o = llm.run_until_done(rid)[rid]
            row[key] = tokens_and_lp(o) | {"wall_s": time.perf_counter() - t0, "bounds": bounds}

        def rows_step(key, g):
            rid = f"{key}-{ii}"
            t0 = time.perf_counter()
            llm.open(rid, emb_a.chunk(0, N, final=False), sp, correct=True,
                     image_start=lo, image_end=lo + G)
            infos, bs, inv = [], bands_of(G, g), []
            armed = bool(a.capture_topk) and bool(llm.run_on_ranks(arm_topk))
            try:
                for r, (g0, g1) in enumerate(bs):
                    final = r == len(bs) - 1
                    pos = torch.arange(lo + g0, lo + g1, dtype=torch.int64)
                    rowsv = emb.embeds[pos]
                    win = (lo + g0, lo + g1)
                    if final:
                        pos = torch.cat([pos, torch.arange(lo + G, N - 1, dtype=torch.int64)])
                        rowsv = torch.cat([rowsv, emb.embeds[lo + G:N - 1]], dim=0)
                        win = (lo + g0, N - 1)
                    infos.append(llm.correct(rid, pos, rowsv, win, final))
                    if armed:
                        rows0 = llm.run_on_ranks(read_topk)[0]
                        if rows0 is not None:
                            inv.append(check_visibility_invariant(rows0, pos, 4))
            finally:
                if armed:
                    llm.run_on_ranks(disarm_topk)
            # the regime must be the one the run asked for, per round
            dense = [inf.get("glm53_indexer_dense") for inf in infos]
            if a.regime == "dense":
                assert all(dense), f"expected the dense regime, got {dense} (N={N})"
            elif a.regime == "sparse":
                assert not any(dense), f"expected the sparse regime, got {dense} (N={N})"
            run_to_first_token(llm, rid)
            take(key, rid)
            o = llm.run_until_done(rid)[rid]
            row[key] = tokens_and_lp(o) | {"wall_s": time.perf_counter() - t0, "bands": bs,
                                           "info": infos}
            row[key]["indexer_pools_rewritten"] = sum(
                sum(d.get("pools_rewritten", 0) for d in inf.get("glm53_indexer", {}).values())
                for inf in infos)
            row[key]["indexer_dense"] = [inf.get("glm53_indexer_dense") for inf in infos]
            row[key]["indexer_side_buffer_mb"] = infos[-1].get("glm53_indexer_mb")
            if inv:
                row[key]["visibility"] = inv
                row[key]["visibility_ok"] = all(d["causal"] and d["tail_complete"] for d in inv)

        one_shot("one")
        chunked("chunk")
        rows_step("rows", a.g)

        for key in ("chunk", "rows"):
            if key not in snaps:
                continue
            row[f"{key}_vs_one"] = (cmp_latent(snaps["one"], snaps[key])
                                    | cmp_indexer(ix_snaps["one"], ix_snaps[key])
                                    | compare(row["one"], row[key]))
        print(f"[{ii}] {name} N={N} dense={row['dense_regime']}", flush=True)
        if row.get("rows", {}).get("visibility"):
            v = row["rows"]["visibility"]
            print(f"    visibility: ok={row['rows']['visibility_ok']} "
                  f"max(key-p)={max(d['max_key_minus_p'] for d in v)} "
                  f"rows_missing_tail={sum(d['rows_missing_tail'] for d in v)} "
                  f"keys/row=[{min(d['keys_per_row_min'] for d in v)}, "
                  f"{max(d['keys_per_row_max'] for d in v)}]", flush=True)
        for key in ("chunk", "rows"):
            c = row.get(f"{key}_vs_one")
            if c:
                print(f"    {key:6s} latent_rel_l2={c['latent_rel_l2_max']:.3e} "
                      f"pools_diff={c['pools_differing_frac_max']:.4f} "
                      f"all_bitwise={c['all_bitwise']} exact={c.get('exact')} "
                      f"first_div={c.get('first_divergence')} "
                      f"|dlp0|={c.get('dlogprob_first'):.3e}", flush=True)
        json.dump(res, open(a.out, "w"), indent=1, default=float)
    print(f"wrote {a.out}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gate", default="mla", choices=["mla"])
    p.add_argument("--model", default=MODEL)
    p.add_argument("--g", type=int, default=4)
    p.add_argument("--chunks", type=int, default=4)
    p.add_argument("--n-images", type=int, default=8)
    p.add_argument("--downscale", type=int, default=4)
    p.add_argument("--min-prompt-tokens", type=int, default=0)
    p.add_argument("--regime", default="auto", choices=["auto", "dense", "sparse"],
                   help="assert the observed indexer regime per round")
    p.add_argument("--capture-topk", action="store_true",
                   help="snapshot the last sparse layer's topk rows and check causality + tail")
    p.add_argument("--max-tokens", type=int, default=16)
    p.add_argument("--gpu-mem", type=float, default=0.85)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--max-num-seqs", type=int, default=0)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--tensor-parallel-size", type=int, default=1,
                   help="GLM-5.3-Flash does not fit on one B200, so this is 2 in practice. Every "
                        "snapshot is taken on EVERY rank and the replicated ones (MLA latent, "
                        "both indexer caches) must agree bytewise -- that assert is the check "
                        "that a correction cannot half-apply across ranks")
    p.add_argument("--out", default="analysis/results/glm53/mla_gate.json")
    gate_mla(p.parse_args())


if __name__ == "__main__":
    main()
