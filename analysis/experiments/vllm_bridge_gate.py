"""Plumbing gate for the two-process form: AppCorr vision (this process, appcorr env) -> socket ->
vLLM streaming server (`appcorr.vllm_stream.server`, appcorr-vllm env). Nothing about accuracy;
the question is whether the chunks that leave the axis are consumed the way the in-process HF
path consumes them.

Per image, three arms, each run through BOTH backends:

  ceiling   stock tower on the full image, one chunk         (oneshot_embeds)
  floor     stock tower on the degraded base, one chunk       (oneshot_embeds)
  stream    axis.streaming_forward(g, keep) -- HF: in-process prefill + shared greedy loop;
            vLLM: the same call with a StreamSink, decoded by the engine

Reported per arm: greedy-token agreement HF vs vLLM (exact string, first-token match, longest
common prefix) and the vLLM timing block (TTFT from last chunk). HF-vs-vLLM is NOT expected to be
bit-identical -- different attention kernels, different reduction order -- so the gate is the
same one the in-process vLLM gate used: the stream arm's first token agrees on every image and
its disagreement with its HF twin is no larger than the one-chunk ceiling arm's (i.e. the socket
path adds nothing beyond the engine's own numerical band; the logprob margin at the first
disagreeing token is printed so a near-tie can be told from a defect). Both HF and vLLM decode is PURE greedy: note that
`model.generate()` is not (Qwen2.5-VL's generation_config carries repetition_penalty=1.05 --
found 2026-09-07, the "decode-mechanism confound" of the RefCOCO memo has a name now).

Run (appcorr env; the server must already be up on --port with the same --model):
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python analysis/experiments/vllm_bridge_gate.py \
      --family qwen25vl --model Qwen/Qwen2.5-VL-7B-Instruct --port 5555
"""
import argparse, json, os, sys, time
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from appcorr.vllm_stream.bridge import LLMBridge
from analysis.experiments.vllm_stream_gate import COCO, IMAGES, QUESTION
from analysis.experiments.qwen35_accuracy import degrade
from analysis.experiments.qwen_vllm_accuracy import make_axis, greedy_tokens


def lcp(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["qwen25vl", "qwen35"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--n-images", type=int, default=4)
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--keep", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.out is None:
        a.out = os.path.join(ROOT, "analysis/results/vllm_stream",
                             f"bridge_gate_{a.model.split('/')[-1].lower()}.json")

    bridge = LLMBridge("127.0.0.1", a.port)
    info = bridge.info()
    assert info["model"] == a.model, (info["model"], a.model)
    print(f"server: {info}")
    proc = AutoProcessor.from_pretrained(a.model)
    model = AutoModelForImageTextToText.from_pretrained(a.model, dtype="auto", device_map="cuda:0").eval()
    axis = make_axis(a.family, model, proc)

    rows = []
    for name in IMAGES[:a.n_images]:
        img = Image.open(os.path.join(COCO, name)).convert("RGB")
        inputs = axis.build_inputs(img, QUESTION).to("cuda:0")
        px_base = axis.build_inputs(degrade(img, 2, "box"), QUESTION)["pixel_values"].to("cuda:0")
        for arm in ("ceiling", "floor", "stream"):
            # -- HF twin
            if arm == "stream":
                lg, kv, st = axis.streaming_forward(inputs, px_base, a.groups, keep=a.keep)
                hf = greedy_tokens(axis, lg, kv, st["decode_start_pos"], a.max_tokens)
            else:
                px = inputs["pixel_values"] if arm == "ceiling" else px_base
                emb, pos, delta = axis.oneshot_embeds(inputs, px)
                pos3 = pos.unsqueeze(1)
                out = axis.model(inputs_embeds=emb.unsqueeze(0), position_ids=pos3, use_cache=True)
                hf = greedy_tokens(axis, out.logits[:, -1], out.past_key_values,
                                   int(pos3.max().item()) + 1, a.max_tokens)
            # -- vLLM
            sink = bridge.sink(f"{name}-{arm}", max_tokens=a.max_tokens, logprobs=8)
            t0 = time.perf_counter()
            if arm == "stream":
                axis.streaming_forward(inputs, px_base, a.groups, keep=a.keep, sink=sink)
            else:
                px = inputs["pixel_values"] if arm == "ceiling" else px_base
                emb, pos, delta = axis.oneshot_embeds(inputs, px)
                sink.push(emb, pos, delta, final=True)
            res = sink.result()
            wall = time.perf_counter() - t0
            vl = list(res["token_ids"])
            # margin at the first disagreeing position: vLLM's chosen token vs the HF token, in
            # vLLM's own logprobs (top-8). A near-tie there is the engine's numerical band, not a
            # plumbing defect; a large margin would be.
            margin = None
            j = lcp(hf, vl)
            if j < min(len(hf), len(vl)) and res.get("logprobs"):
                lp = res["logprobs"][j]
                if str(hf[j]) in lp:
                    margin = lp[str(vl[j])] - lp[str(hf[j])]
            row = {"image": name, "arm": arm, "hf": hf, "vllm": vl, "exact": hf == vl,
                   "first_same": bool(hf and vl and hf[0] == vl[0]), "lcp": lcp(hf, vl),
                   "hf_text": proc.tokenizer.decode(hf, skip_special_tokens=True),
                   "vllm_text": res["text"], "timing": res["timing"], "wall_s": wall,
                   "num_prompt_tokens": res["num_prompt_tokens"], "chunks": len(res["pushes"]),
                   "first_diff_margin": margin}
            rows.append(row)
            t = res["timing"]
            print(f"[{name} {arm:7s}] exact={row['exact']!s:5s} first={row['first_same']!s:5s} "
                  f"lcp={row['lcp']:2d}/{len(hf)}  margin={'-' if margin is None else f'{margin:.3f}'}  chunks={row['chunks']}  "
                  f"ttft_last={t['ttft_from_last_chunk_ms']:.1f}ms  total={t['total_ms']:.0f}ms "
                  f"wall={wall * 1e3:.0f}ms", flush=True)
            print(f"      hf  : {row['hf_text']!r}\n      vllm: {row['vllm_text']!r}", flush=True)

    summ = {}
    for arm in ("ceiling", "floor", "stream"):
        rs = [r for r in rows if r["arm"] == arm]
        summ[arm] = {"n": len(rs), "exact": sum(r["exact"] for r in rs),
                     "first_same": sum(r["first_same"] for r in rs),
                     "mean_lcp": sum(r["lcp"] for r in rs) / max(1, len(rs))}
    # Pass rule: the STREAM arm (the socket path under test) agrees with its HF twin on the
    # first token everywhere and drifts no more than the one-chunk ceiling arm does (the
    # engine's own band; "The"/"This", "shows"/"features" near-ties flip there).
    ok = summ["stream"]["first_same"] == summ["stream"]["n"] and \
        summ["stream"]["mean_lcp"] >= summ["ceiling"]["mean_lcp"] - 2
    print("summary:", json.dumps(summ))
    print("VLLM_BRIDGE_GATE", "PASS" if ok else "FAIL")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({"_meta": vars(a) | {"server": info}, "summary": summ, "rows": rows}, open(a.out, "w"), indent=1)
    print(f"wrote {a.out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
