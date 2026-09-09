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
Single-GPU form (2026-09-08, GPU1 off limits): with the vllm backend the HF side loads only the
vision tower + embed_tokens (`--load vision`, default; `appcorr.models.vision_only`, ~3 GB for
122B, bitwise the full model's tower), so the 122B-FP8 server (--gpu-mem ~0.85) and this driver
share GPU0. Throughput form: `--workers 4-8` (forked CPU workers for degrade + image processor;
the 200-420 ms/sample of CPU prep otherwise starves the engine) plus `--concurrency 2-4`.
"""
import argparse, json, os, re, sys, time
from collections import deque
import torch

from appcorr.vllm_stream.bridge import BridgeError  # raised per sink; caught per sample

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


@torch.no_grad()
def merge_shards(args) -> None:
    """`--merge N`: for every arm, fold `<row file>_sKofN.jsonl` (K = 0..N-1) and whatever the
    canonical row file already holds into the canonical file, one row per i (a later shard
    row never overrides an existing canonical row), sorted by i. The shard files are left in
    place; make_eval_table / vllm_vs_hf_preds read only the canonical file."""
    slug = args.model.split("/")[-1].lower()
    n_sh = int(args.merge)
    for arm in args.arms:
        suffix = ""
        if arm == "streaming":
            suffix = f"_g{args.groups}" + (f"_k{args.keep:.2f}" if args.keep < 1.0 else "")
        if args.concurrency > 1:
            suffix += f"_c{args.concurrency}"
        if args.backend == "hf":
            suffix += "_hf"
        base = os.path.join(args.out, f"{args.dataset}_{slug}_{arm}{suffix}")
        rows = {}
        srcs = [f"{base}.jsonl"] + [f"{base}_s{k}of{n_sh}.jsonl" for k in range(n_sh)]
        counts = {}
        for src in srcs:
            if not os.path.exists(src):
                counts[os.path.basename(src)] = None
                continue
            c = 0
            with open(src) as fh:
                for l in fh:
                    if l.strip():
                        r = json.loads(l)
                        c += 1
                        rows.setdefault(int(r["i"]), r)
            counts[os.path.basename(src)] = c
        if not rows:
            print(f"[{arm}] nothing to merge ({counts})")
            continue
        tmp = f"{base}.jsonl.merging"
        with open(tmp, "w") as fh:
            for i in sorted(rows):
                fh.write(json.dumps(rows[i]) + "\n")
        os.replace(tmp, f"{base}.jsonl")
        sc = [r for r in rows.values() if "skip" not in r]
        acc = 100 * sum(r["ok"] for r in sc) / max(1, len(sc))
        print(f"[{arm}] merged {len(rows)} rows ({len(sc)} scored, acc {acc:.2f}, "
              f"{len(rows) - len(sc)} skipped) from {counts} -> {base}.jsonl", flush=True)


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
    ap.add_argument("--prefetch", type=int, default=0,
                    help="samples to preprocess ahead on a CPU thread (0 = inline); use for "
                         "throughput runs so the HF image processor is not the bottleneck")
    ap.add_argument("--workers", type=int, default=0,
                    help="CPU worker PROCESSES for degrade + image processing (DataLoader, "
                         "forked); unlike --prefetch they do not share the GIL with the vision "
                         "loop. 0 = inline. Recommended 4-8 for throughput runs")
    ap.add_argument("--load", choices=["full", "vision"], default=None,
                    help="HF-side model: 'vision' loads only the tower + embed_tokens "
                         "(appcorr.models.vision_only; ~3 GB for 122B) -- the default for the "
                         "vllm backend, where the decoder never runs; 'full' is forced for hf")
    ap.add_argument("--max-prompt-tokens", type=int, default=None,
                    help="skip (record a `skip` row for) any sample whose prompt + --max-tokens "
                         "exceeds this; default = the server's max_model_len. A 15.5k-token "
                         "TextVQA image crashed the ceiling arm and hung the streaming arm of the "
                         "122B campaign against an 8192 engine (2026-09-08); the user's call is "
                         "to keep the proven 8192 engine and skip past such rows, counted in the "
                         "row file (`skip`, `prompt_tokens`) and in the Final Summary (`skipped`)")
    ap.add_argument("--shard", default=None, metavar="K/N",
                    help="run only every N-th sample starting at K (0-based) and write to "
                         "`<row file>_sKofN.jsonl`: N driver processes, each with its own vision "
                         "tower, overlap their vision passes against the one server (the "
                         "streaming arm is driver-bound: ~3 samples/s vs the engine's 6-10). "
                         "Interleaved, not contiguous, so every shard sees the same image-size "
                         "mix. `--merge N` folds the shard files back into the canonical row file")
    ap.add_argument("--merge", type=int, default=None, metavar="N",
                    help="no runs: merge the N shard files of each arm (plus any rows already in "
                         "the canonical file) into the canonical row file, sorted by i, and exit")
    ap.add_argument("--out", default="analysis/results/qwen_vllm_accuracy")
    args = ap.parse_args()
    shard = None
    if args.shard is not None:
        k, n_sh = (int(v) for v in args.shard.split("/"))
        if not (0 <= k < n_sh):
            raise SystemExit(f"--shard {args.shard}: need 0 <= K < N")
        shard = (k, n_sh)
    if args.merge is not None:
        return merge_shards(args)
    if args.backend == "hf":
        args.concurrency = 1
        args.load = "full"
    elif args.load is None:
        args.load = "vision"
    if args.workers and args.prefetch:
        raise SystemExit("--workers and --prefetch are alternatives; pick one")

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
    max_prompt = args.max_prompt_tokens
    if max_prompt is None and info.get("max_model_len"):
        max_prompt = int(info["max_model_len"]) - args.max_tokens
    elif max_prompt is not None:
        max_prompt -= args.max_tokens
    print(f"prompt cap: {max_prompt} tokens (+{args.max_tokens} generated)", flush=True)

    proc = AutoProcessor.from_pretrained(args.model)
    t_load = time.perf_counter()
    if args.load == "vision":
        from appcorr.models.vision_only import load_vision_only
        model = load_vision_only(args.model, device="cuda:0")
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            args.model, dtype="auto", device_map="cuda:0").eval()
    print(f"model ({args.load}) loaded in {time.perf_counter() - t_load:.1f}s, "
          f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)
    axis = make_axis(args.family, model, proc)
    tmpl_kw = {"think": True} if (args.think and args.family == "qwen35") else {}
    spec = get_spec(args.dataset)
    ds = spec.load(load_dataset)
    n = len(ds) if args.samples == 0 else min(args.samples, len(ds))
    idxs = list(range(n)) if args.samples == 0 else \
        (list(range(n)) if args.contiguous else list(range(0, len(ds), max(1, len(ds) // n)))[:n])
    if shard is not None:
        idxs = idxs[shard[0]::shard[1]]
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
        if shard is not None:
            suffix += f"_s{shard[0]}of{shard[1]}"
        path = os.path.join(args.out, f"{args.dataset}_{slug}_{arm}{suffix}.jsonl")
        done = set()
        if os.path.exists(path):
            with open(path) as fh:
                done = {json.loads(l)["i"] for l in fh if l.strip()}
        pending = [i for i in idxs if i not in done]
        correct, scored, skipped = 0, 0, {}
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

        def prep(i):
            """CPU side of one sample: degrade + HF image processor (two images for the
            streaming arm). Runs on the prefetch thread when --prefetch > 0."""
            img, q, gold = build(i)
            t0 = time.perf_counter()
            base = degrade(img, args.level, args.degrade_filter)
            if arm == "streaming":
                inputs = axis.build_inputs(img, q, **tmpl_kw)
                px_base = axis.build_inputs(base, q, **tmpl_kw)["pixel_values"]
            else:
                inputs = axis.build_inputs(img if arm == "ceiling" else base, q, **tmpl_kw)
                px_base = None
            # Prompt layout read here, on the CPU tensors, so the vision pass never reads it
            # back from the GPU (`image_run`, `grid_thw` kwargs of the axis).
            ids = inputs["input_ids"][0]
            pos = (ids == axis.image_token_id).nonzero(as_tuple=True)[0]
            if pos.numel() == 0 or int(pos[-1] - pos[0]) + 1 != pos.numel():
                raise ValueError(f"sample {i}: image tokens are not one contiguous run")
            return {"size": img.size, "gold": gold, "inputs": inputs, "px_base": px_base,
                    "image_run": (int(pos[0]), int(pos.numel())),
                    "grid_thw": tuple(int(v) for v in inputs["image_grid_thw"][0].tolist()),
                    "t_prep_ms": (time.perf_counter() - t0) * 1e3}

        if args.prefetch > 0:
            from concurrent.futures import ThreadPoolExecutor
            pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prep")
            futs = {}

            def prepared(k):
                """prep(pending[k]) with the next --prefetch samples already submitted."""
                for j in range(k, min(k + 1 + args.prefetch, len(pending))):
                    if j not in futs:
                        futs[j] = pool.submit(prep, pending[j])
                return futs.pop(k).result()
        elif args.workers > 0:
            # Forked worker processes run `prep` (PIL degrade + HF image processor, CPU only;
            # the workers never touch CUDA, so inheriting the parent's model handle is fine);
            # the main process only pops ready samples in order. Batch size 1, no collation.
            from torch.utils.data import DataLoader, Dataset

            class _Prep(Dataset):
                def __len__(self):
                    return len(pending)

                def __getitem__(self, k):
                    return prep(pending[k])

            def _worker_init(_):
                # A killed driver used to leave its forked workers alive holding the parent's
                # CUDA context (27.8 GB on GPU0 until killed by PID, 2026-09-08): ask the kernel
                # to SIGTERM them when the parent goes.
                import ctypes, signal
                try:
                    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)  # PR_SET_PDEATHSIG
                except OSError:
                    pass

            loader_it = iter(DataLoader(_Prep(), batch_size=None, shuffle=False,
                                        num_workers=args.workers, prefetch_factor=4,
                                        multiprocessing_context="fork", persistent_workers=False,
                                        worker_init_fn=_worker_init))
            next_k = 0

            def prepared(k):
                nonlocal next_k
                if k != next_k:
                    raise RuntimeError(f"--workers prep is sequential: asked {k}, next is {next_k}")
                next_k += 1
                return next(loader_it)
        else:
            def prepared(k):
                return prep(pending[k])

        def skip(i, reason, **fields):
            skipped[reason] = skipped.get(reason, 0) + 1
            fh.write(json.dumps({"i": int(i), "skip": reason, **fields}) + "\n")
            fh.flush()
            print(f"[{arm}] skip i={i}: {reason} {fields}", flush=True)

        def too_long(i, p):
            """Prompt cap (see --max-prompt-tokens): checked on the CPU-side input_ids BEFORE
            any GPU work or chunk push, so the engine never sees a prompt it cannot hold."""
            n_prompt = int(p["inputs"]["input_ids"].shape[1])
            if max_prompt is not None and n_prompt > max_prompt:
                skip(i, "prompt_too_long", prompt_tokens=n_prompt, cap=max_prompt)
                return True
            return False

        def vision(p, sink):
            """Run this arm's vision + chunk pushes for a prepared sample; returns (extra-fields,
            HF twin (logits, cache, start_pos) when backend=hf else None)."""
            size, gold = p["size"], p["gold"]
            layout = {"image_run": p["image_run"], "grid_thw": p["grid_thw"]}
            t_h = time.perf_counter()
            inputs = p["inputs"].to("cuda:0")
            hf = None
            if arm == "streaming":
                px_base = p["px_base"].to("cuda:0")
                torch.cuda.synchronize()
                t_loop_h2d = (time.perf_counter() - t_h) * 1e3
                t1 = time.perf_counter()
                lg, kv, st = axis.streaming_forward(inputs, px_base, args.groups, keep=args.keep,
                                                    sink=sink, **layout)
                extra = {"corrected_groups": int(st["corrected_groups"]),
                         "chunks": len(st["chunks"])}
                if sink is None:
                    hf = (lg, kv, st["decode_start_pos"])
            else:
                torch.cuda.synchronize()
                t_loop_h2d = (time.perf_counter() - t_h) * 1e3
                t1 = time.perf_counter()
                emb, pos, delta = axis.oneshot_embeds(inputs, inputs["pixel_values"], **layout)
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
            extra["t_prep_ms"] = p["t_prep_ms"]
            extra["t_loop_h2d_ms"] = t_loop_h2d       # pageable host -> GPU copies of the inputs
            extra["t_vision_ms"] = (time.perf_counter() - t1) * 1e3
            extra["prompt_tokens"] = int(inputs["input_ids"].shape[1])
            return gold, size, extra, hf

        if args.backend == "hf":
            for k, i in enumerate(pending):
                p = prepared(k)
                if too_long(i, p):
                    continue
                try:
                    gold, size, extra, (lg, kv, dp) = vision(p, None)
                    toks = greedy_tokens(axis, lg, kv, dp, args.max_tokens)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    skip(i, "oom")
                    continue
                record(i, proc.tokenizer.decode(toks, skip_special_tokens=True), gold, size, extra)
        else:
            inflight = deque()

            def drain_one():
                i, sink, gold, size, extra = inflight.popleft()
                t_w = time.perf_counter()
                try:
                    res = sink.result()
                except BridgeError as e:
                    # The server rejected one of this request's chunks (its error rides the ack
                    # of that push and is raised by this sink alone); the request is already
                    # gone on the server. Record and move on -- one row must not end an arm.
                    skip(i, "bridge_error", error=str(e)[:300])
                    return
                t = res["timing"]
                # Main-loop accounting (the driver is serial per sample): t_loop_prep = blocked
                # on the prep workers, t_loop_wait = blocked in result() for THIS row (its
                # request was already the oldest in flight), t_loop_push = bridge encode+queue.
                extra["t_loop_wait_ms"] = (time.perf_counter() - t_w) * 1e3
                extra["t_loop_push_ms"] = sum(
                    (r["t_queued"] - r["t_send"]) * 1e3 for r in res["pushes"])
                t_start = extra.pop("_t_start")
                # Latency view from the driver's clock (t_start = inputs on the GPU, vision pass
                # about to begin): when the first / last chunk left, and first token from there.
                # ttft_from_open is a server-side duration, so adding it to t_open needs no
                # clock alignment.
                t_open_ms = (res["pushes"][0]["t_send"] - t_start) * 1e3
                extra.update({"ttft_last_chunk_ms": t["ttft_from_last_chunk_ms"],
                              "ttft_open_ms": t["ttft_from_open_ms"], "total_ms": t["total_ms"],
                              "t_open_ms": t_open_ms,
                              "t_last_push_ms": (res["pushes"][-1]["t_send"] - t_start) * 1e3,
                              # every push's send time: push r is band r (band 0 with the leading
                              # text), the last one the trailing text; band r's correction starts
                              # right after push r-1 (push() syncs the GPU for its D2H copy)
                              "t_pushes_ms": [round((q["t_send"] - t_start) * 1e3, 2)
                                              for q in res["pushes"]],
                              "ttft_start_ms": t_open_ms + t["ttft_from_open_ms"],
                              "gen_tokens": len(res["token_ids"]),
                              "finish_reason": res["finish_reason"],
                              "t_client_done_ms": (time.perf_counter() - t_start) * 1e3})
                record(i, res["text"], gold, size, extra)

            t_iter = None
            for k, i in enumerate(pending):
                t_p = time.perf_counter()
                t_loop_iter = None if t_iter is None else (t_p - t_iter) * 1e3   # whole previous iteration
                t_iter = t_p
                p = prepared(k)
                t_loop_prep = (time.perf_counter() - t_p) * 1e3
                if too_long(i, p):
                    continue
                sink = bridge.sink(f"{arm}-{i}", max_tokens=args.max_tokens)
                t_start = time.perf_counter()
                try:
                    gold, size, extra, _ = vision(p, sink)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if sink.opened and not sink.closed and sink.error is None:
                        bridge.abort(sink.rid)
                    skip(i, "oom")
                    continue
                except BridgeError as e:
                    skip(i, "bridge_error", error=str(e)[:300])
                    continue
                extra["_t_start"] = t_start
                extra["t_loop_prep_ms"] = t_loop_prep
                extra["t_loop_iter_ms"] = t_loop_iter     # of the iteration before this sample's
                inflight.append((i, sink, gold, size, extra))
                while len(inflight) >= args.concurrency:
                    drain_one()
            while inflight:
                drain_one()
        fh.close()
        if args.prefetch > 0:
            pool.shutdown(wait=True)
        el = time.perf_counter() - t_arm0
        if scored:
            print(f"Final Summary: {{\"dataset\": \"{args.dataset}\", \"model\": \"{slug}\", "
                  f"\"arm\": \"{arm}{suffix}\", \"scored\": {scored}, "
                  f"\"acc\": {correct / scored * 100:.4f}, \"elapsed_s\": {el:.1f}, "
                  f"\"samples_per_s\": {scored / el:.3f}, "
                  f"\"skipped\": {json.dumps(skipped)}}}", flush=True)
    print("QWEN_VLLM_ACCURACY_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
