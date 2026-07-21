"""
oracle_progressive.py -- Phase 1 (baseline stage timing) + Phase 3 (oracle progressive streaming).

Isolates the SYSTEM-LEVEL benefit of overlapping visual-token prefill with image transmission,
using ORACLE full-image visual embeddings (no correction yet -- that's Phase 5). It answers:

    If corrected visual-token groups could be finalized progressively as residuals arrive, how much
    of the LLM prefill hides under image transmission, and how much sooner does the answer start?

It measures real GPU compute (CUDA-event timed, warmed up) for each stage, then runs a discrete-event
timeline simulation combining that compute with a SIMULATED transmission schedule (CPU/network time
kept separate from GPU time, per the spec). Three modes are compared:

  (1) baseline monolithic : [full image transmission] -> visual encoder -> monolithic LLM prefill -> decode
  (2) chunked-after-full  : [full image transmission] -> visual encoder -> chunked prefill (no overlap) -> decode
                            (measures chunking overhead only)
  (3) oracle progressive  : visual groups become available on the transmission schedule; each group's
                            LLM prefill runs as soon as it arrives (pipeline), overlapping transmission.

Transmission schedule (simulated): total image transmission time `--total-tx-ms`, of which `--base-frac`
is the low-frequency base (arrives first); the remaining residual time is split evenly across G groups
(even split is FLOPs/latency-optimal per the ProgVFM paper). Group g becomes available at
t_g = t_base + g * (residual_time / G).

NOTE (oracle optimism): the oracle folds the per-group visual-encoder + correction compute into "group
available at t_g" -- i.e. it assumes finalized corrected embeds are ready the moment the residual
arrives. This is the deliberate optimistic upper bound; Phase 5 adds the real correction compute.

Run:
    python qwen_vl_prefill/oracle_progressive.py --model-id Qwen/Qwen2.5-VL-3B-Instruct \
        --num-groups 4 --total-tx-ms 200 --base-frac 0.15 --device cuda:0 --trace /tmp/qwenvl_timeline.json
"""

import argparse
import json
import sys
from pathlib import Path

import torch

for _p in Path(__file__).resolve().parents[1:3]:  # analysis/ (qwen_vl_prefill) + repo root (appcorr)
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from qwen_vl_prefill import introspect as I
from qwen_vl_prefill import prefill as P


def _sync_time(device, fn, iters=3):
    """Warm up once, then median of `iters` CUDA-event-timed runs (ms)."""
    fn()
    torch.cuda.synchronize(device)
    ts = []
    for _ in range(iters):
        ev0 = torch.cuda.Event(enable_timing=True); ev1 = torch.cuda.Event(enable_timing=True)
        ev0.record(); out = fn(); ev1.record()
        torch.cuda.synchronize(device)
        ts.append(ev0.elapsed_time(ev1))
    ts.sort()
    return ts[len(ts) // 2], out


@torch.inference_mode()
def decode_one_step(model, cache, last_hidden, position_ids, seq_len, device):
    """One decode step: lm_head(last_hidden) -> next token -> one LLM forward appending to cache.
    Returns the step's forward time is handled by caller; here we just do the compute."""
    lm = model.model.language_model
    next_id = model.lm_head(last_hidden)[:, -1].argmax(-1, keepdim=True)  # [1,1]
    emb = lm.embed_tokens(next_id)
    # M-RoPE position for the new token = previous max + 1 on all 3 axes (text token)
    next_pos = position_ids[:, :, -1:] + 1
    out = lm(inputs_embeds=emb, position_ids=next_pos, past_key_values=cache,
             cache_position=torch.arange(seq_len, seq_len + 1, device=device), use_cache=True)
    return out


def simulate_timelines(t_base, t_groups, stage_ms, boundaries):
    """Discrete-event timeline for the three modes. stage_ms has: visual_encoder, monolithic_prefill,
    per_group_ms (list aligned to visual groups), query_prefill, decode1, total_tx.
    Returns dict of {mode: {events:[...], ttft_ms: float}}. Times are ms from t=0 (base starts arriving)."""
    total_tx = stage_ms["total_tx"]
    ve = stage_ms["visual_encoder"]
    mono = stage_ms["monolithic_prefill"]
    per_group = stage_ms["per_group_ms"]      # list, one per visual group
    q = stage_ms["query_prefill"]
    d1 = stage_ms["decode1"]

    out = {}

    # (1) baseline monolithic: everything serial after full image arrives
    ev = []
    t = 0.0
    ev.append(("image_transmission", t, t + total_tx, None)); t += total_tx
    ev.append(("visual_encoder", t, t + ve, None)); t += ve
    ev.append(("monolithic_prefill", t, t + mono, None)); t += mono
    ev.append(("decode_first_token", t, t + d1, None)); t += d1
    out["baseline_monolithic"] = {"events": ev, "ttft_ms": t}

    # (2) chunked-after-full: full image -> visual encoder -> chunked prefill (back-to-back) -> decode
    ev = []
    t = 0.0
    ev.append(("image_transmission", t, t + total_tx, None)); t += total_tx
    ev.append(("visual_encoder", t, t + ve, None)); t += ve
    gi = 0
    for (a, b, label), _ in zip(boundaries, boundaries):
        if label.startswith("visual_group_"):
            dur = per_group[gi]; gi += 1
        elif label == "post_text":
            dur = q
        else:  # pre_text
            dur = stage_ms.get("pre_text", 0.0)
        ev.append((f"prefill_{label}", t, t + dur, gi - 1 if label.startswith("visual_group_") else None)); t += dur
    ev.append(("decode_first_token", t, t + d1, None)); t += d1
    out["chunked_after_full"] = {"events": ev, "ttft_ms": t}

    # (3) oracle progressive: LLM is a serial resource; group g prefill ready at t_groups[g],
    #     runs when (arrived AND LLM free). pre_text prefills at t_base (base carries the prompt prefix).
    ev = []
    llm_free = t_base + stage_ms.get("pre_text", 0.0)  # pre-image text prefilled right after base
    ev.append(("base_transmission", 0.0, t_base, None))
    if stage_ms.get("pre_text", 0.0) > 0:
        ev.append(("prefill_pre_text", t_base, llm_free, None))
    for g in range(len(per_group)):
        # residual group g transmission window
        arrive = t_groups[g]
        ev.append((f"residual_group_{g}_transmission",
                   t_base if g == 0 else t_groups[g - 1], arrive, g))
        start = max(arrive, llm_free)
        end = start + per_group[g]
        ev.append((f"prefill_visual_group_{g}", start, end, g))
        llm_free = end
    # query prefill after last visual group (query text known from t=0, so only gated by LLM)
    qs = llm_free; qe = qs + q
    ev.append(("prefill_query_text", qs, qe, None)); llm_free = qe
    ev.append(("decode_first_token", llm_free, llm_free + d1, None)); llm_free += d1
    out["oracle_progressive"] = {"events": ev, "ttft_ms": llm_free}

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--image", default=None)
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--num-groups", type=int, default=4)
    ap.add_argument("--total-tx-ms", type=float, default=200.0, help="simulated total image transmission time")
    ap.add_argument("--base-frac", type=float, default=0.15, help="fraction of transmission time for the base")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--trace", default=None, help="write timeline JSON here")
    args = ap.parse_args()
    device = args.device

    print(f"[oracle] loading {args.model_id} ...")
    model, processor = I.load_model(args.model_id, device=device)

    if args.image is not None:
        from PIL import Image
        image = Image.open(args.image).convert("RGB"); prompt = args.prompt or "Describe this image."
    else:
        from qwen_vl_prefill.equivalence_test import load_default_image
        image, dp = load_default_image(); prompt = args.prompt or dp

    prepared = I.prepare_inputs(model, processor, image, prompt, device=device)
    layout = I.token_layout(prepared)
    boundaries = I.streaming_chunk_boundaries(layout, args.num_groups)
    print(f"[oracle] seq_len={prepared.seq_len}, n_visual_tokens={prepared.n_visual_tokens}, G={args.num_groups}")

    # ---- measure GPU compute (warmed up) ----
    ve_ms, visual_embeds = _sync_time(device, lambda: I.extract_visual_embeds(model, prepared))
    inputs_embeds = I.build_inputs_embeds(model, prepared, visual_embeds)
    position_ids = I.compute_position_ids(model, prepared)
    mono_ms, (mono_logits, _) = _sync_time(device, lambda: P.monolithic_prefill(model, inputs_embeds, position_ids))

    # per-chunk prefill times (warm up once, then measure)
    P.chunked_prefill_timed(model, inputs_embeds, position_ids, boundaries, device)  # warmup
    last_hidden, cache, per_chunk_ms, labels = P.chunked_prefill_timed(model, inputs_embeds, position_ids, boundaries, device)
    pre_text_ms = sum(m for m, l in zip(per_chunk_ms, labels) if l == "pre_text")
    per_group_ms = [m for m, l in zip(per_chunk_ms, labels) if l.startswith("visual_group_")]
    query_ms = sum(m for m, l in zip(per_chunk_ms, labels) if l == "post_text")

    d1_ms, _ = _sync_time(device, lambda: decode_one_step(model, cache, last_hidden, position_ids, prepared.seq_len, device))

    t_base = args.base_frac * args.total_tx_ms
    resid = args.total_tx_ms - t_base
    t_groups = [t_base + (g + 1) * resid / args.num_groups for g in range(args.num_groups)]

    stage_ms = {
        "visual_encoder": ve_ms, "monolithic_prefill": mono_ms, "per_group_ms": per_group_ms,
        "pre_text": pre_text_ms, "query_prefill": query_ms, "decode1": d1_ms, "total_tx": args.total_tx_ms,
    }

    print("\n---------- MEASURED GPU COMPUTE (ms, median) ----------")
    print(f"  visual_encoder:      {ve_ms:8.3f}")
    print(f"  monolithic_prefill:  {mono_ms:8.3f}")
    print(f"  pre_text prefill:    {pre_text_ms:8.3f}")
    print(f"  per-group prefill:   {[round(x,2) for x in per_group_ms]}  (sum {sum(per_group_ms):.2f})")
    print(f"  query prefill:       {query_ms:8.3f}")
    print(f"  decode first token:  {d1_ms:8.3f}")
    print(f"\n---------- SIMULATED TRANSMISSION (ms) ----------")
    print(f"  total_tx={args.total_tx_ms}, base_frac={args.base_frac} -> t_base={t_base:.1f}, "
          f"residual group arrivals={[round(x,1) for x in t_groups]}")

    tl = simulate_timelines(t_base, t_groups, stage_ms, boundaries)

    print(f"\n========== TIME-TO-FIRST-TOKEN (ms from base start) ==========")
    base_ttft = tl["baseline_monolithic"]["ttft_ms"]
    for mode in ["baseline_monolithic", "chunked_after_full", "oracle_progressive"]:
        ttft = tl[mode]["ttft_ms"]
        print(f"  {mode:22s}: {ttft:8.2f} ms   ({ttft - base_ttft:+.2f} vs baseline, "
              f"{100*(base_ttft-ttft)/base_ttft:+.1f}% speedup)")

    # amount of LLM prefill hidden under transmission (oracle)
    hidden = sum(per_group_ms[:-1])  # groups 0..G-2 prefill fully overlaps transmission in the ideal case
    print(f"\n  LLM prefill (visual groups) potentially hidden under transmission: "
          f"~{hidden:.2f} ms of {sum(per_group_ms):.2f} ms total visual prefill")

    if args.trace:
        with open(args.trace, "w") as f:
            json.dump({"stage_ms": {k: (v if not isinstance(v, list) else v) for k, v in stage_ms.items()},
                       "t_base": t_base, "t_groups": t_groups, "timelines": tl}, f, indent=2)
        print(f"\n[oracle] timeline trace -> {args.trace}")


if __name__ == "__main__":
    main()
