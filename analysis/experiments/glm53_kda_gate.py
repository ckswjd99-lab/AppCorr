"""G-KDA: the GLM-5.3-Flash KDA re-scan vs the stock layer, on a real checkpoint (GPU).

NOT RUN as of 2026-09-13 (CPU-only night; agent K).  Everything CPU-checkable about this path is
in `tests/test_correct_glm53_kda.py` -- the mHC layer walk, the side-buffer capture at the
`_forward` seam, the beta form, the conv-window arithmetic and re-scan invariance on an fp32
reference.  What only a GPU can answer is whether the TRITON kernels agree with that design:

  * does `chunk_kda_with_fused_gate(..., initial_state=S_start, output_final_state=True)` over a
    window reproduce the stock `Glm5NextLinearAttention.forward`'s `core_attn_out` for that
    window, and its recurrent state?
  * does the merged causal conv over `SB[start-3:end]` (zero initial state) reproduce the stock
    prefill's conv output on `[start, end)` and leave the right conv tail?
  * does the checkpoint chain compose -- scan `[0, s)`, then `[s, N)` from the checkpoint == one
    scan of `[0, N)`?

This file gates exactly that and NOTHING else: no composer, no images, no correct step, no
pseudo-sequence path.  It prefills a plain TEXT prompt with the stock engine, captures every KDA
layer's seam inputs and outputs with a temporary patch, and then re-runs `correct._rescan` over
the captured rows.  So it does not wait on the GLM-5.3 composer / 1-D positions (agents V and M)
and can run the moment a checkpoint is on the box.

Reported per KDA layer: rel-L2 and max-abs of (our window output vs the stock `core_attn_out`),
of (our final recurrent state vs the block's), and of (our conv tail vs the block's conv state);
plus the split-vs-single re-scan comparison.  Pass marks are not absolutes -- bf16 kernels on a
2k-row prompt have their own band -- but the layer-to-layer SPREAD is the signal: one bad layer
means a wiring bug, a uniform 1e-3 means bf16.

Command (B200-8, GPU 2 or 3; one task per GPU, and GPU1 is off limits on B200-6):

  CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \\
  PYTHONPATH=/NHNHOME/share/cjpark/AppCorr-glm53 \\
  /NHNHOME/storage/users/cjpark/shk/conda_envs/appcorr-vllm/bin/python \\
      analysis/experiments/glm53_kda_gate.py \\
      --model zai-org/GLM-5.3-Flash --gpu-mem 0.85 --max-model-len 8192 \\
      --prompt-tokens 2048 --split 900 \\
      --out analysis/results/vllm_stream/glm53_kda_gate.json

(The env must be the one that has vLLM main 658c813 -- 0.28 has no `glm5next` module at all.
`--enforce-eager` is not required: the seam is an explicit eager break.)
"""
import argparse
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float(), b.float()
    return float((a - b).norm() / max(b.norm().item(), 1e-30))


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.float() - b.float()).abs().max())


def resolve_runner(llm):
    """The in-process `GPUModelRunner` behind a plain `vllm.LLM` (V1, no multiprocessing)."""
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    core = llm.llm_engine.engine_core
    core = getattr(core, "engine_core", core)
    r = core.model_executor.driver_worker.worker.model_runner
    assert isinstance(r, GPUModelRunner), type(r)
    return r


# ---------------------------------------------------------------------------------------------
# Rank-side bodies.  GLM-5.3-Flash does not fit on one B200, so this gate has to run under TP>1,
# where `resolve_runner` (the in-process `driver_worker`) does not exist.  The two functions below
# are the SAME code that used to sit inline in `main()`, moved verbatim so it can be shipped to
# every rank as a cloudpickled callable through `LLMRanks.run_on_ranks`
# (`appcorr/vllm_stream/tp_worker.py`).  Nothing about the comparison changed -- only where it
# runs.  Two calls, because the capture has to be armed BEFORE the driver's `generate()` and
# harvested after; the state between them is stashed on the RUNNER object, because each callable
# is pickled by value and two calls do not share a module namespace in the worker.
#
# What the ranks do NOT have to agree on: the KDA conv and recurrent state are sharded by head
# (`vllm/model_executor/layers/mamba/abstract.py:59-61`), so each rank owns a disjoint head slice
# and the per-layer numbers legitimately differ between ranks.  The gate reports every rank and
# takes rank 0 for the summary; it does NOT assert cross-rank equality here (that assert belongs
# to the replicated MLA/indexer quantities in `glm53_mla_gate.py`).
# ---------------------------------------------------------------------------------------------

def arm_capture(runner):
    """Temporarily patch the KDA `_forward` seam on this rank and record the whole prefill."""
    from appcorr.vllm_stream import correct as C

    layers = C.gdn_modules(runner.model)
    assert layers, "no recurrent layers found -- wrong checkpoint?"
    flavor = {C._gdn_flavor(m) for m in layers}
    assert flavor == {"kda"}, f"expected the KDA seam, found {flavor}"

    cap: dict = {}
    cls = type(layers[0])
    stock_forward = cls._forward

    def _capture_forward(self, qkv_proj_states, g1, beta, core_attn_out):
        T = g1.shape[1]
        d = cap.setdefault(self.prefix, {})
        d["qkv"] = qkv_proj_states[:T].detach().clone()
        d["b"] = beta[0, :T].detach().clone()
        d["a"] = g1[0, :T].reshape(T, -1).detach().clone()
        ret = stock_forward(self, qkv_proj_states=qkv_proj_states, g1=g1, beta=beta,
                            core_attn_out=core_attn_out)
        d["out"] = core_attn_out[0, :T].detach().clone()
        return ret

    cls._forward = _capture_forward
    runner._appcorr_kda_gate = {"cap": cap, "cls": cls, "stock": stock_forward,
                                "keys": [ly.prefix for ly in layers]}
    return {"n_layers": len(layers), "device": str(runner.device)}


def harvest_and_compare(runner, split: int, n_ids: int):
    """Restore the seam, then K's per-layer comparison, verbatim, on this rank's own slice."""
    import torch as _torch
    from appcorr.vllm_stream import correct as C

    st = runner._appcorr_kda_gate
    st["cls"]._forward = st["stock"]                       # always restore, even on a failure
    cap = st["cap"]
    layers = C.gdn_modules(runner.model)

    N = int(next(iter(cap.values()))["qkv"].shape[0])
    assert N == n_ids, (f"captured {N} of {n_ids} rows: the prefill was chunked, "
                        "raise --max-num-batched-tokens")

    mamba_gids = C._mamba_group_ids(runner)
    req_id = next(iter(runner.requests))
    blocks = C._mamba_blocks(runner, req_id, mamba_gids)

    rows = {}
    for layer in layers:
        key = layer.prefix
        d = cap[key]
        sb = C.SideBuffer(n=N, device=runner.device)
        sb.store(key, d["qkv"], d["b"], d["a"], 0)
        sb.conv_scratch[key] = _torch.zeros_like(layer.kv_cache[0][:2])

        # (1) one re-scan of the whole prompt from a zero state == the stock prefill
        o_full = C._rescan(layer, sb, key, 0, N, commit=True)
        state_full, conv_full = sb.ckpt[key].clone(), sb.conv_scratch[key][1].clone()
        blk = blocks[key]
        row = {
            "out_rel_l2": rel_l2(o_full, d["out"]),
            "out_max_abs": max_abs(o_full, d["out"]),
            "state_rel_l2": rel_l2(state_full[0], layer.kv_cache[1][blk]),
            "state_max_abs": max_abs(state_full[0], layer.kv_cache[1][blk]),
            "conv_rel_l2": rel_l2(conv_full, layer.kv_cache[0][blk]),
            "conv_max_abs": max_abs(conv_full, layer.kv_cache[0][blk]),
        }

        # (2) the checkpoint chain: [0, s) then [s, N) from the checkpoint
        s = min(max(split, 1), N - 1)
        sb2 = C.SideBuffer(n=N, device=runner.device)
        sb2.store(key, d["qkv"], d["b"], d["a"], 0)
        sb2.conv_scratch[key] = _torch.zeros_like(layer.kv_cache[0][:2])
        C._rescan(layer, sb2, key, 0, s, commit=True)
        o_tail = C._rescan(layer, sb2, key, s, N, commit=True)
        row.update({
            "split_out_rel_l2": rel_l2(o_tail, o_full[s:]),
            "split_out_max_abs": max_abs(o_tail, o_full[s:]),
            "split_state_max_abs": max_abs(sb2.ckpt[key], state_full),
            "split_conv_max_abs": max_abs(sb2.conv_scratch[key][1], conv_full),
        })
        rows[key] = row
    del runner._appcorr_kda_gate
    return {"N": N, "layers": rows, "device": str(runner.device)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="zai-org/GLM-5.3-Flash")
    ap.add_argument("--gpu-mem", type=float, default=0.85)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--prompt-tokens", type=int, default=2048)
    ap.add_argument("--split", type=int, default=900,
                    help="checkpoint position for the split-vs-single re-scan comparison")
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--tensor-parallel-size", type=int, default=1,
                    help="GLM-5.3-Flash does not fit on one B200, so this is 2 in practice. At "
                         "TP>1 the per-layer comparison is shipped to every rank with "
                         "`LLMRanks.run_on_ranks`; each rank owns a disjoint KDA head slice, so "
                         "the ranks legitimately DISAGREE and all of them are reported")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(ROOT, "analysis/results/vllm_stream/glm53_kda_gate.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    from vllm import LLM, SamplingParams
    from appcorr.vllm_stream import correct as C

    from appcorr.vllm_stream.tp_worker import LLMRanks

    kw = {}
    if a.tensor_parallel_size > 1:
        from appcorr.vllm_stream.tp_worker import WORKER_EXTENSION_CLS
        kw["tensor_parallel_size"] = a.tensor_parallel_size
        kw["worker_extension_cls"] = WORKER_EXTENSION_CLS
    llm = LLM(model=a.model, gpu_memory_utilization=a.gpu_mem, max_model_len=a.max_model_len,
              enforce_eager=a.enforce_eager, **kw)
    ranks = LLMRanks(llm, a.tensor_parallel_size)

    # ---- arm the seam capture ON EVERY RANK, prefill, harvest ------------------------------
    # Same three steps as before; only the capture and the comparison moved into the ranks
    # (`arm_capture` / `harvest_and_compare` above are the old inline code, verbatim).
    armed = ranks.run_on_ranks(arm_capture)
    print(f"armed on {len(armed)} rank(s): {armed}", flush=True)
    try:
        tok = llm.get_tokenizer()
        text = ("The quick brown fox jumps over the lazy dog. " * 4096)
        ids = tok(text).input_ids[:a.prompt_tokens]
        t0 = time.perf_counter()
        llm.generate({"prompt_token_ids": ids},
                     SamplingParams(temperature=0.0, max_tokens=1))
        t_prefill = time.perf_counter() - t0
        per_rank = ranks.run_on_ranks(harvest_and_compare, int(a.split), len(ids))
    finally:
        # `harvest_and_compare` restores the seam itself; this is the path where it never ran.
        pass

    N = per_rank[0]["N"]
    print(f"captured N={N} rows x {len(per_rank[0]['layers'])} layers in {t_prefill:.2f}s",
          flush=True)

    res = {"_meta": {"model": a.model, "N": N, "n_layers": len(per_rank[0]["layers"]),
                     "split": a.split, "prefill_s": t_prefill,
                     "tensor_parallel_size": a.tensor_parallel_size,
                     "n_ranks": len(per_rank),
                     "rank_devices": [r["device"] for r in per_rank]},
           # Rank 0 is the summary; every rank is kept. The KDA state is sharded by head, so the
           # ranks are EXPECTED to differ -- a cross-rank equality assert here would be a false
           # alarm. What must hold on each rank separately is that its own re-scan reproduces its
           # own slice of the stock prefill, which is what these numbers say.
           "layers": per_rank[0]["layers"],
           "per_rank": [r["layers"] for r in per_rank]}

    for key, row in res["layers"].items():
        print(f"{key:52s} out {row['out_rel_l2']:.3e}  state {row['state_rel_l2']:.3e}  "
              f"conv {row['conv_rel_l2']:.3e}  split {row['split_out_rel_l2']:.3e}", flush=True)

    for k in ("out_rel_l2", "state_rel_l2", "conv_rel_l2", "split_out_rel_l2"):
        v = [r[k] for r in res["layers"].values()]
        res["_meta"][f"max_{k}"] = max(v)
        res["_meta"][f"median_{k}"] = sorted(v)[len(v) // 2]
        # the same two numbers per rank: one bad RANK reads differently from one bad LAYER
        res["_meta"][f"max_{k}_per_rank"] = [max(r[k] for r in lay.values())
                                             for lay in res["per_rank"]]
    json.dump(res, open(out, "w"), indent=2)
    print("\nsummary:", {k: v for k, v in res["_meta"].items() if k.startswith(("max_", "median_"))})
    print("wrote", out)


if __name__ == "__main__":
    main()
