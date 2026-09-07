"""Accuracy + serving-time campaign driver for the two-process form: AppCorr vision in THIS
process (appcorr env), the LLM in a vLLM streaming server (`appcorr.vllm_stream.server`,
appcorr-vllm env), chunks over a socket. One driver for both Qwen families:

  --family qwen25vl   Qwen2.5-VL 7B/32B/72B   (windowed tower; `Qwen25VLAxis`)
  --family qwen35     Qwen3.5-35B-A3B / 122B-A10B-FP8 (`Qwen35Axis`)

Arms (all decoded by the SAME mechanism -- vLLM temperature-0 greedy -- so the decode path is
never a confound between arms, the rule qwen35_accuracy.py's docstring explains):

  floor      stock tower on the degraded base, one chunk       (`oneshot_embeds`)
  ceiling    stock tower on the full image, one chunk          (`oneshot_embeds`)
  streaming  approx-then-correct per band, `--groups` chunks, `--keep` fraction corrected
             (`streaming_forward(..., sink=)`); the server prefills band r while the tower is
             still correcting band r+1 -- that overlap is what the timing columns measure.

Per row the jsonl carries the score AND the server's clock: TTFT measured from the LAST chunk
(what the user waits after the image finished arriving), TTFT from the first chunk, total time,
prompt tokens, chunk count, and the client's vision wall time. `--concurrency N` keeps N requests
in flight (push N, then wait for the oldest) -- the engine batches them, which is the throughput
lever; N=1 is the latency form. `--backend hf` runs the identical loop in-process with the shared
explicit greedy loop (a consistency reference, not a campaign arm).

Same degrade()/get_spec()/record() conventions as qwen35_accuracy.py; output naming
  {dataset}_{model-slug}_{arm}[_g{groups}[_k{keep}]][_c{concurrency}].jsonl
under --out, resumable by row index.

Run (appcorr env; server first, in the appcorr-vllm env, same --model):
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  PYTHONPATH=$PWD <appcorr-vllm python> -m appcorr.vllm_stream.server \
      --model Qwen/Qwen2.5-VL-7B-Instruct --port 5591 --gpu-mem 0.35
  CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 python analysis/experiments/qwen_vllm_accuracy.py \
      --family qwen25vl --model Qwen/Qwen2.5-VL-7B-Instruct --port 5591 --dataset gqa \
      --arms floor streaming ceiling --groups 4 --samples 240
For the 122B-FP8 the HF side (vision + embed_tokens, bf16 weights of the whole model) and the
server do not fit one GPU together: server on GPU1 (--gpu-mem 0.9), this driver on GPU0.
"""
import argparse, json, os, re, sys, time
from collections import deque
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "analysis"))
from PIL import Image
from analysis.experiments.qwen35_accuracy import degrade


def make_axis(family: str, model, proc):
    if family == "qwen25vl":
        from appcorr.models.qwen25vl.unified import Qwen25VLAxis
        return Qwen25VLAxis(model, proc)
    if family == "qwen35":
        from appcorr.models.qwen35.unified import Qwen35Axis
        return Qwen35Axis(model, proc)
    raise ValueError(family)


@torch.no_grad()
def greedy_tokens(axis, logits, cache, start_pos, n=24):
    """qwen35_accuracy.greedy, returning token ids (the HF twin of vLLM temperature-0)."""
    toks, cur, pos = [], logits.argmax(-1, keepdim=True), start_pos
    eos = axis.processor.tokenizer.eos_token_id
    for _ in range(n):
        t = int(cur)
        if t == eos:
            break
        toks.append(t)
        pid = torch.full((3, 1, 1), pos, device=cur.device, dtype=torch.long)
        out = axis.model(input_ids=cur, past_key_values=cache, position_ids=pid, use_cache=True)
        cache = out.past_key_values
        cur = out.logits[:, -1].argmax(-1, keepdim=True)
        pos += 1
    return toks


def rescale_box(pred: str, size) -> str:
    """Qwen3-generation grounding emits 0-1000 relative coords; gold is pixels."""
    nums = re.findall(r"-?\d+\.?\d*", pred)[:4]
    if len(nums) != 4:
        return pred
    w_, h_ = size
    x1, y1, x2, y2 = (float(v) for v in nums)
    return f"{x1 * w_ / 1000:.1f},{y1 * h_ / 1000:.1f},{x2 * w_ / 1000:.1f},{y2 * h_ / 1000:.1f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["qwen25vl", "qwen35"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5591)
    ap.add_argument("--arms", nargs="+", default=["floor", "streaming", "ceiling"])
    ap.add_argument("--groups", type=int, default=4)
    ap.add_argument("--keep", type=float, default=1.0)
    ap.add_argument("--level", type=int, default=2)
    ap.add_argument("--degrade-filter", choices=["bicubic", "box", "pyr"], default="box")
    ap.add_argument("--samples", type=int, default=0, help="0 = full split")
    ap.add_argument("--contiguous", action="store_true")
    ap.add_argument("--max-tokens", type=int, default=24)
    ap.add_argument("--concurrency", type=int, default=1,
                    help="requests kept in flight on the server (vllm backend only)")
    ap.add_argument("--think", action="store_true", help="qwen35: enable_thinking in the template")
    ap.add_argument("--out", default="analysis/results/qwen_vllm_accuracy")
    args = ap.parse_args()
    if args.backend == "hf":
        args.concurrency = 1

    from transformers import AutoProcessor, AutoModelForImageTextToText
    from qwen_vl_prefill.datasets_eval import get_spec
    from datasets import load_dataset

    bridge, info = None, {}
    if args.backend == "vllm":
        from appcorr.vllm_stream.bridge import LLMBridge
        bridge = LLMBridge(args.host, args.port)
        info = bridge.info()
        if info["model"] != args.model:
            raise SystemExit(f"server serves {info['model']!r}, driver asked for {args.model!r}")
        print(f"server: {info}", flush=True)

    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype="auto", device_map="cuda:0").eval()
    axis = make_axis(args.family, model, proc)
    tmpl_kw = {"think": True} if (args.think and args.family == "qwen35") else {}
    spec = get_spec(args.dataset)
    ds = spec.load(load_dataset)
    n = len(ds) if args.samples == 0 else min(args.samples, len(ds))
    idxs = list(range(n)) if args.samples == 0 else \
        (list(range(n)) if args.contiguous else list(range(0, len(ds), max(1, len(ds) // n)))[:n])
    os.makedirs(args.out, exist_ok=True)
    slug = args.model.split("/")[-1].lower()

    for arm in args.arms:
        suffix = ""
        if arm == "streaming":
            suffix = f"_g{args.groups}" + (f"_k{args.keep:.2f}" if args.keep < 1.0 else "")
        if args.concurrency > 1:
            suffix += f"_c{args.concurrency}"
        if args.backend == "hf":
            suffix += "_hf"
        path = os.path.join(args.out, f"{args.dataset}_{slug}_{arm}{suffix}.jsonl")
        done = set()
        if os.path.exists(path):
            with open(path) as fh:
                done = {json.loads(l)["i"] for l in fh if l.strip()}
        pending = [i for i in idxs if i not in done]
        correct, scored = 0, 0
        fh = open(path, "a")
        t_arm0 = time.perf_counter()

        def record(i, pred, gold, size, extra):
            nonlocal correct, scored
            if args.dataset in ("refcoco", "visdrone_det"):
                pred = rescale_box(pred, size)
            try:
                ok, val = spec.score(pred, gold)
            except NotImplementedError:
                ok, val = 0, None
            correct += ok
            scored += 1
            row = {"i": int(i), "pred": pred, "gold": gold, "ok": int(ok),
                   "val": (float(val) if val is not None else None)}
            row.update(extra)
            fh.write(json.dumps(row) + "\n")
            if scored % 50 == 0:
                fh.flush()
                el = time.perf_counter() - t_arm0
                print(f"[{arm}] {scored} scored, running {correct / scored * 100:.2f}%  "
                      f"({scored / el:.2f} samples/s)", flush=True)

        def build(i):
            img, q, gold = spec.prepare(ds[int(i)], lambda h, w, **kw: (h, w), 1, 1, 1 << 30)
            if img.mode != "RGB":
                img = img.convert("RGB")
            return img, q, gold

        def vision(i, sink):
            """Run this arm's vision + chunk pushes for sample i; returns (extra-fields, HF twin
            (logits, cache, start_pos) when backend=hf else None)."""
            img, q, gold = build(i)
            t0 = time.perf_counter()
            base = degrade(img, args.level, args.degrade_filter)
            hf = None
            if arm == "streaming":
                inputs = axis.build_inputs(img, q, **tmpl_kw).to("cuda:0")
                px_base = axis.build_inputs(base, q, **tmpl_kw)["pixel_values"].to("cuda:0")
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                lg, kv, st = axis.streaming_forward(inputs, px_base, args.groups, keep=args.keep,
                                                    sink=sink)
                extra = {"corrected_groups": int(st["corrected_groups"]),
                         "chunks": len(st["chunks"])}
                if sink is None:
                    hf = (lg, kv, st["decode_start_pos"])
            else:
                use = img if arm == "ceiling" else base
                inputs = axis.build_inputs(use, q, **tmpl_kw).to("cuda:0")
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                emb, pos, delta = axis.oneshot_embeds(inputs, inputs["pixel_values"])
                extra = {"chunks": 1}
                if sink is not None:
                    sink.push(emb, pos, delta, final=True)
                else:
                    pos3 = pos.unsqueeze(1)
                    out = axis.model(inputs_embeds=emb.unsqueeze(0), position_ids=pos3, use_cache=True)
                    hf = (out.logits[:, -1], out.past_key_values, int(pos3.max().item()) + 1)
            torch.cuda.synchronize()
            # t_prep: CPU work that every arm pays and no serving path would put on the critical
            # path (degrade + HF image processor -- two images for the streaming arm);
            # t_vision: the GPU vision pass incl. the chunk pushes -- the number to set beside
            # the server's ttft_open_ms (which for the streaming arm already contains it).
            extra["t_prep_ms"] = (t1 - t0) * 1e3
            extra["t_vision_ms"] = (time.perf_counter() - t1) * 1e3
            extra["prompt_tokens"] = int(inputs["input_ids"].shape[1])
            return gold, img.size, extra, hf

        if args.backend == "hf":
            for i in pending:
                try:
                    gold, size, extra, (lg, kv, dp) = vision(i, None)
                    toks = greedy_tokens(axis, lg, kv, dp, args.max_tokens)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    fh.write(json.dumps({"i": int(i), "skip": "oom"}) + "\n")
                    continue
                record(i, proc.tokenizer.decode(toks, skip_special_tokens=True), gold, size, extra)
        else:
            inflight = deque()

            def drain_one():
                i, sink, gold, size, extra = inflight.popleft()
                res = sink.result()
                t = res["timing"]
                extra.update({"ttft_last_chunk_ms": t["ttft_from_last_chunk_ms"],
                              "ttft_open_ms": t["ttft_from_open_ms"], "total_ms": t["total_ms"],
                              "gen_tokens": len(res["token_ids"]),
                              "finish_reason": res["finish_reason"],
                              "t_client_done_ms": (time.perf_counter() - extra.pop("_t_start")) * 1e3})
                record(i, res["text"], gold, size, extra)

            for i in pending:
                sink = bridge.sink(f"{arm}-{i}", max_tokens=args.max_tokens)
                t_start = time.perf_counter()
                try:
                    gold, size, extra, _ = vision(i, sink)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if sink.opened and not sink.closed:
                        bridge.abort(sink.rid)
                    fh.write(json.dumps({"i": int(i), "skip": "oom"}) + "\n")
                    continue
                extra["_t_start"] = t_start
                inflight.append((i, sink, gold, size, extra))
                while len(inflight) >= args.concurrency:
                    drain_one()
            while inflight:
                drain_one()
        fh.close()
        el = time.perf_counter() - t_arm0
        if scored:
            print(f"Final Summary: {{\"dataset\": \"{args.dataset}\", \"model\": \"{slug}\", "
                  f"\"arm\": \"{arm}{suffix}\", \"scored\": {scored}, "
                  f"\"acc\": {correct / scored * 100:.4f}, \"elapsed_s\": {el:.1f}, "
                  f"\"samples_per_s\": {scored / el:.3f}}}", flush=True)
    print("QWEN_VLLM_ACCURACY_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
