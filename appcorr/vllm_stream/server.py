"""vLLM-side streaming server: prompt embeddings in over a socket, generated text out.

Runs in the `appcorr-vllm` env (vllm 0.28.0) and owns the LLM; the AppCorr process (vision fork,
`appcorr` env) talks to it through `bridge.LLMBridge` using the frames in `wire.py`. Single
event loop: poll the sockets, apply every message that arrived (open / append / final / correct /
result), then run one engine step if any request is live -- the engine prefills whatever has
arrived while the vision side is still correcting the next band, which is the whole point
(§ docs/memo/vllm_stream_design.md). Several clients / requests may be live at once; the engine
batches them (that is the throughput measurement's lever).

    CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
    PYTHONPATH=/NHNHOME/share/cjpark/AppCorr-vllm \
    <appcorr-vllm python> -m appcorr.vllm_stream.server --model Qwen/Qwen2.5-VL-7B-Instruct \
        --port 5555 --gpu-mem 0.5 [--max-model-len 16384]

Timing per request (server perf_counter): t_open, t_final, t_first_token, t_done; the client gets
them back in `result` and pairs them with its own send times.
"""
from __future__ import annotations

import argparse
import os
import selectors
import socket
import sys
import time
from typing import Any, Optional

import torch

from .wire import Frame, FrameParser, error_frame, stage_from_header, stage_to_header


class _Req:
    __slots__ = ("rid", "conn", "t_open", "t_final", "t_first", "t_done", "n_prompt", "out",
                 "waiter", "finished", "logprobs", "max_tokens", "t_correct")

    def __init__(self, rid, conn, logprobs, max_tokens=0):
        self.rid, self.conn, self.logprobs = rid, conn, logprobs
        self.max_tokens = int(max_tokens)
        self.t_open = time.perf_counter()
        self.t_final = None
        self.t_correct = []     # interleaved schedule: one arrival per `correct` round
        self.t_first = None
        self.t_done = None
        self.n_prompt = 0
        self.out = None
        self.waiter = None      # connection waiting on `result`
        self.finished = False


class _Conn:
    __slots__ = ("sock", "parser", "rids")

    def __init__(self, sock):
        self.sock, self.parser, self.rids = sock, FrameParser(), set()


class StreamServer:
    def __init__(self, llm, host: str, port: int):
        self.llm = llm
        self.sel = selectors.DefaultSelector()
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind((host, port))
        self.srv.listen(16)
        self.srv.setblocking(False)
        self.sel.register(self.srv, selectors.EVENT_READ, data=None)
        self.conns: dict[socket.socket, _Conn] = {}
        self.reqs: dict[str, _Req] = {}
        self.n_done = 0
        # Diagnostics (off unless the env vars are set; the 2026-09-09 critical-latency analysis).
        # APPCORR_SERVER_TRACE=<file>: one JSON line per handled chunk frame and per engine step
        # (server perf_counter, comparable with the driver's -- same host, CLOCK_MONOTONIC).
        # APPCORR_STEP_PROFILE=<dir>:<first_step>:<count>[,<first>:<count>...]: torch.profiler per
        # engine step over those windows -> <dir>/steps.jsonl (wall vs summed CUDA kernel time,
        # top kernels). Profiled steps are slower than unprofiled ones: read the wall from the
        # trace outside the windows, the GPU/CPU split from inside them.
        self._trace = None
        tp = os.environ.get("APPCORR_SERVER_TRACE")
        if tp:
            self._trace = open(tp, "a", buffering=1)
        self._n_steps = 0
        self._prof = None
        self._prof_window = None
        pw = os.environ.get("APPCORR_STEP_PROFILE")
        if pw:
            d, spec = pw.split(":", 1)
            os.makedirs(d, exist_ok=True)
            self._prof_window = (d, [tuple(int(v) for v in w.split(":")) for w in spec.split(",")])
        self._last_step_info = None
        # APPCORR_CORRECT_PROFILE=<dir>:<first>:<count>: same profiler window over `correct` calls
        # (drain + correct step), written to <dir>/corrects.jsonl with the wall/CUDA split.
        self._cprof_window = None
        self._n_corrects = 0
        pw = os.environ.get("APPCORR_CORRECT_PROFILE")
        if pw:
            d, f, n = pw.split(":")
            os.makedirs(d, exist_ok=True)
            self._cprof_window = (d, int(f), int(n))

    def _tr(self, ev: str, **kw):
        if self._trace is not None:
            import json
            kw["ev"] = ev
            self._trace.write(json.dumps(kw) + "\n")

    # -- socket plumbing ----------------------------------------------------------------------
    def _accept(self):
        s, _ = self.srv.accept()
        s.setblocking(False)
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.conns[s] = _Conn(s)
        self.sel.register(s, selectors.EVENT_READ, data=self.conns[s])

    def _drop(self, c: _Conn):
        self.sel.unregister(c.sock)
        self.conns.pop(c.sock, None)
        c.sock.close()
        live = [r for r in c.rids if r in self.reqs and not self.reqs[r].finished]
        if live:
            try:
                self.llm.engine.abort_request(live)
            except Exception as e:  # noqa: BLE001
                print(f"[server] abort on disconnect failed: {e}", file=sys.stderr)
            for r in live:
                self.reqs.pop(r, None)

    def _reply(self, c: _Conn, frame: Frame):
        try:
            c.sock.setblocking(True)
            c.sock.sendall(frame.encode())
        finally:
            c.sock.setblocking(False)

    # -- protocol -----------------------------------------------------------------------------
    def _handle(self, c: _Conn, f: Frame):
        op = f.header.get("op")
        try:
            if op == "info":
                self._reply(c, Frame({"ok": True, "model": self.llm.model_name,
                                      "vllm": self.llm.vllm_version, "pid": os.getpid(),
                                      "max_model_len": self.llm.max_model_len,
                                      "live_requests": sum(1 for r in self.reqs.values() if not r.finished),
                                      "done_requests": self.n_done}))
            elif op in ("open", "append"):
                self._chunk(c, f, op)
            elif op == "correct":
                self._correct(c, f)
            elif op == "result":
                self._result(c, f.header["rid"])
            elif op == "abort":
                rid = f.header["rid"]
                r = self.reqs.pop(rid, None)
                if r is not None and not r.finished:
                    self.llm.engine.abort_request([rid])
                self._reply(c, Frame({"ok": True}))
            else:
                self._reply(c, error_frame(f"unknown op {op!r}"))
        except Exception as e:  # noqa: BLE001 -- report to the client, keep serving
            import traceback
            traceback.print_exc()
            self._reply(c, error_frame(f"{type(e).__name__}: {e}"))

    def _chunk(self, c: _Conn, f: Frame, op: str):
        from vllm import SamplingParams
        from .request import StreamChunk
        rid = f.header["rid"]
        emb = f.get_tensor("embeds")
        mrope = f.get_tensor("mrope")
        delta = f.header.get("mrope_delta")
        final = bool(f.header.get("final", False))
        chunk = StreamChunk(embeds=emb, final=final, mrope_positions=mrope,
                            mrope_delta=(None if delta is None else int(delta)))
        if op == "open":
            if rid in self.reqs:
                raise KeyError(f"request id {rid!r} already open")
            lp = f.header.get("logprobs")
            max_tokens = int(f.header["max_tokens"])
            self._check_len(rid, chunk.num_tokens, max_tokens)
            sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, logprobs=lp)
            r = _Req(rid, c, lp, max_tokens)
            self.reqs[rid] = r
            c.rids.add(rid)
            try:
                self.llm.open(rid, chunk, sp, correct=bool(f.header.get("correct", False)),
                              image_start=int(f.header.get("image_start", 0)),
                              image_end=int(f.header.get("image_end", 0)),
                              open_walk=int(f.header.get("open_walk", 0)))
            except Exception:
                self.reqs.pop(rid, None)     # a rejected open must not leave a dead entry
                c.rids.discard(rid)
                raise
        else:
            r = self.reqs.get(rid)
            if r is None:
                raise KeyError(f"append to unknown request {rid!r}")
            if r.t_final is not None:
                raise ValueError(f"{rid}: append after final")
            try:
                self._check_len(rid, r.n_prompt + chunk.num_tokens, r.max_tokens)
                self.llm.append(rid, chunk)
            except Exception:
                # The engine validates prompt length only at `open`; a request grown past
                # max_model_len by appends is never scheduled and its `result` would block
                # forever (the 2026-09-08 hang). Abort it so the error frame is the end of it.
                self.reqs.pop(rid, None)
                c.rids.discard(rid)
                try:
                    self.llm.engine.abort_request([rid])
                except Exception as e:  # noqa: BLE001
                    print(f"[server] abort of over-length {rid} failed: {e}", file=sys.stderr)
                raise
        r.n_prompt += chunk.num_tokens
        t = time.perf_counter()
        if final:
            r.t_final = t
        self._reply(c, Frame({"ok": True, "t_recv": t, "num_prompt_tokens": r.n_prompt}))
        self._tr("chunk", t=t, rid=rid, op=op, n=chunk.num_tokens, final=final, n_prompt=r.n_prompt)

    def _correct(self, c: _Conn, f: Frame):
        """Interleaved schedule (docs/memo/vllm_interleaved_design.md §3.2): rewrite prompt rows
        of an OPEN request and re-run the decoder on them. Adds no prompt rows, so there is no
        length check against max_model_len; what is checked instead is that every rewritten row
        and the whole re-scan window sit below the held-back last row (`request.py`)."""
        rid = f.header["rid"]
        r = self.reqs.get(rid)
        if r is None:
            raise KeyError(f"correct on unknown request {rid!r}")
        if r.t_final is not None:
            raise ValueError(f"{rid}: correct after final")
        positions = f.get_tensor("positions")
        embeds = f.get_tensor("embeds")
        final = bool(f.header.get("final", False))
        w = f.header["window"]
        window = (int(w[0]), int(w[1]))
        # [r, g] (equal-layer bounds, derived engine-side) or [r, g, [b_0..b_{g-1}]] (explicit
        # -- the unified axis's cost split; wire.py §correct)
        stage = stage_from_header(f.header.get("stage"))
        t0 = time.perf_counter()
        prof = None
        if self._cprof_window is not None:
            d, f0, n = self._cprof_window
            if f0 <= self._n_corrects < f0 + n:
                from torch.profiler import ProfilerActivity, profile
                prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
                prof.__enter__()
            self._n_corrects += 1
        try:
            self._check_rows(rid, positions, embeds, window, r.n_prompt)
            n_steps0 = self._n_steps
            out = self.llm.correct(rid, positions, embeds, window, final, step_fn=self._step,
                                   stage=stage) or {}
            if prof is not None:
                prof.__exit__(None, None, None)
                torch.cuda.synchronize()
                self._write_correct_profile(prof, rid, positions.shape[0], window, final,
                                            self._n_steps - n_steps0, out, time.perf_counter() - t0)
                prof = None
        except Exception:
            # Same abort-on-error behaviour as an over-length append: the request can no longer
            # be completed correctly, and left alone it would never be scheduled again (it is
            # open with 0 schedulable tokens), so its `result` would block forever.
            self.reqs.pop(rid, None)
            c.rids.discard(rid)
            try:
                self.llm.engine.abort_request([rid])
            except Exception as e:  # noqa: BLE001
                print(f"[server] abort of failed correct {rid} failed: {e}", file=sys.stderr)
            raise
        t = time.perf_counter()
        r.t_correct.append(t)
        if final:
            r.t_final = t
        n_rows = int(positions.shape[0])
        # t_recv = arrival (same meaning as a push's t_recv); the drain + correct step run
        # between it and t_done, so the latency probe can split transport from the step.
        rep = {"ok": True, "t_recv": t0, "t_done": t, "num_rows": n_rows,
               "n_sub": int(out.get("n_sub", 1)),
               "t_step_ms": float(out.get("t_step_ms", (t - t0) * 1e3))}
        if out.get("stage") is not None:
            rep["stage"] = out["stage"]      # per-step walk/correct timings of the staged form
        self._reply(c, Frame(rep))
        self._tr("correct", t=t, rid=rid, n=n_rows, window=list(window), final=final,
                 stage=(None if stage is None else stage_to_header(stage)))

    def _check_rows(self, rid: str, positions, embeds, window: tuple, n_prompt: int) -> None:
        n = int(n_prompt)
        if positions is None or positions.ndim != 1 or positions.numel() == 0:
            raise ValueError(f"{rid}: correct needs a non-empty 1-D positions tensor")
        if embeds is None or embeds.ndim != 2 or embeds.shape[0] != positions.shape[0]:
            shape = None if embeds is None else tuple(embeds.shape)
            raise ValueError(f"{rid}: correct embeds {shape} do not match "
                             f"{int(positions.numel())} positions")
        p0, p1 = int(positions[0]), int(positions[-1])
        if positions.numel() > 1 and not bool((positions[1:] > positions[:-1]).all()):
            raise ValueError(f"{rid}: correct positions must be strictly increasing")
        s, e = window
        # Row n-1 is held back while the request is open and is computed only at final; nothing
        # may rewrite it or re-scan across it.
        if p0 < 0 or p1 >= n - 1:
            raise ValueError(f"{rid}: correct positions [{p0}, {p1}] outside [0, {n - 1}) "
                             f"(prompt {n} rows, last one held back)")
        if not (0 <= s < e <= n - 1):
            raise ValueError(f"{rid}: correct window [{s}, {e}) outside [0, {n - 1})")
        if p0 < s or p1 >= e:
            raise ValueError(f"{rid}: correct positions [{p0}, {p1}] outside window [{s}, {e})")

    def _check_len(self, rid: str, n_prompt: int, max_tokens: int) -> None:
        cap = self.llm.max_model_len
        if n_prompt + max_tokens > cap:
            raise ValueError(f"{rid}: prompt {n_prompt} + max_tokens {max_tokens} tokens exceeds "
                             f"max_model_len {cap}")

    def _result(self, c: _Conn, rid: str):
        r = self.reqs.get(rid)
        if r is None:
            raise KeyError(f"result for unknown request {rid!r}")
        if r.finished:
            self._send_result(c, r)
        else:
            r.waiter = c   # answered from the step loop

    def _send_result(self, c: _Conn, r: _Req):
        o = r.out.outputs[0]
        h = {"ok": True, "rid": r.rid, "text": o.text, "token_ids": list(o.token_ids),
             "finish_reason": o.finish_reason, "num_prompt_tokens": r.n_prompt,
             "timing": {"t_open": r.t_open, "t_final": r.t_final, "t_first_token": r.t_first,
                        "t_done": r.t_done, "t_correct": list(r.t_correct),
                        "ttft_from_last_chunk_ms": (None if r.t_first is None or r.t_final is None
                                                    else (r.t_first - r.t_final) * 1e3),
                        "ttft_from_open_ms": (None if r.t_first is None
                                              else (r.t_first - r.t_open) * 1e3),
                        "total_ms": (r.t_done - r.t_open) * 1e3}}
        if r.logprobs and o.logprobs:
            # per generated position: {token_id: logprob} for the top-k + the sampled token
            h["logprobs"] = [{str(k): float(v.logprob) for k, v in lp.items()} for lp in o.logprobs]
        self._reply(c, Frame(h))
        self.reqs.pop(r.rid, None)
        c.rids.discard(r.rid)

    # -- loop ---------------------------------------------------------------------------------
    def _step(self):
        live = [rid for rid, r in self.reqs.items() if not r.finished]
        before = {rid: (self.llm.stream_state(rid) or {}).get("num_computed_tokens", 0) for rid in live} \
            if (self._trace is not None or self._prof_window is not None) else None
        if self._prof_window is not None:
            self._prof_step_begin()
        t0 = time.perf_counter()
        outs = self.llm.step()
        t1 = time.perf_counter()
        firsts, dones = [], []
        for o in outs:
            r = self.reqs.get(o.request_id)
            if r is None:
                continue
            if r.t_first is None and o.outputs and len(o.outputs[0].token_ids) >= 1:
                r.t_first = time.perf_counter()
                firsts.append(o.request_id)
            if o.finished:
                dones.append(o.request_id)
        if before is not None:
            comp = {}
            for rid in live:
                st = self.llm.stream_state(rid) or {}
                comp[rid] = (before[rid], st.get("num_computed_tokens", before[rid]), st.get("num_prompt_tokens"))
            info = {"step": self._n_steps, "t0": t0, "ms": (t1 - t0) * 1e3, "computed": comp,
                    "first": firsts, "done": dones}
            self._last_step_info = info
            self._tr("step", **info)
        self._n_steps += 1
        if self._prof_window is not None:
            self._prof_step_end()
        for o in outs:
            r = self.reqs.get(o.request_id)
            if r is None:
                continue
            if o.finished:
                r.finished = True
                r.t_done = time.perf_counter()
                r.out = o
                self.n_done += 1
                if r.waiter is not None:
                    w, r.waiter = r.waiter, None
                    if w.sock in self.conns:
                        self._send_result(w, r)

    # -- per-step profiler window (diagnostic) --------------------------------------------------
    def _prof_step_begin(self):
        d, windows = self._prof_window
        if self._prof is None and any(f <= self._n_steps < f + n for f, n in windows):
            from torch.profiler import ProfilerActivity, profile
            self._prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])
            self._prof.__enter__()

    def _prof_step_end(self):
        if self._prof is None:
            return
        import json
        d, _ = self._prof_window
        self._prof.__exit__(None, None, None)
        torch.cuda.synchronize()
        ka = self._prof.key_averages()
        cuda_ms = sum(getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0.0))
                      for e in ka) / 1e3
        top = sorted(ka, key=lambda e: -getattr(e, "self_device_time_total",
                                                 getattr(e, "self_cuda_time_total", 0.0)))[:12]
        info = dict(self._last_step_info or {})
        info.update({"cuda_ms": cuda_ms,
                     "n_kernels": sum(e.count for e in ka
                                      if getattr(e, "self_device_time_total",
                                                 getattr(e, "self_cuda_time_total", 0.0)) > 0),
                     "top": [(e.key[:60], e.count,
                              round(getattr(e, "self_device_time_total",
                                            getattr(e, "self_cuda_time_total", 0.0)) / 1e3, 3))
                             for e in top]})
        with open(os.path.join(d, "steps.jsonl"), "a") as fh:
            fh.write(json.dumps(info) + "\n")
        self._prof = None

    def _write_correct_profile(self, prof, rid, n_rows, window, final, drain_steps, out, wall_s):
        import json
        d, _, _ = self._cprof_window
        ka = prof.key_averages()

        def dev(e):
            return getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0.0))
        top = sorted(ka, key=lambda e: -dev(e))[:15]
        info = {"rid": rid, "n_rows": int(n_rows), "window": list(window), "final": bool(final),
                "drain_steps": int(drain_steps), "wall_ms": wall_s * 1e3,
                "t_step_ms": float(out.get("t_step_ms", 0.0)), "n_sub": int(out.get("n_sub", 1)),
                "cuda_ms": sum(dev(e) for e in ka) / 1e3,
                "n_kernels": sum(e.count for e in ka if dev(e) > 0),
                "top": [(e.key[:60], e.count, round(dev(e) / 1e3, 3)) for e in top]}
        with open(os.path.join(d, "corrects.jsonl"), "a") as fh:
            fh.write(json.dumps(info) + "\n")

    def serve_forever(self):
        print(f"[server] {self.llm.model_name} (vllm {self.llm.vllm_version}) listening on "
              f"{self.srv.getsockname()}", flush=True)
        while True:
            live = any(not r.finished for r in self.reqs.values())
            for key, _ in self.sel.select(timeout=0.0 if live else 0.05):
                if key.data is None:
                    self._accept()
                    continue
                c: _Conn = key.data
                try:
                    data = c.sock.recv(1 << 22)
                except (BlockingIOError, InterruptedError):
                    continue
                except OSError:
                    data = b""
                if not data:
                    self._drop(c)
                    continue
                for f in c.parser.feed(data):
                    th = time.perf_counter()
                    self._handle(c, f)
                    if self._trace is not None:
                        self._tr("handle", t=th, op=f.header.get("op"), rid=f.header.get("rid"),
                                 ms=(time.perf_counter() - th) * 1e3)
            if any(not r.finished for r in self.reqs.values()):
                self._step()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--gpu-mem", type=float, default=0.5)
    ap.add_argument("--max-model-len", type=int, default=16384)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--tensor-parallel-size", type=int, default=1,
                    help="vLLM TP degree. 1 keeps the model runner in THIS process (the path the "
                         "Qwen3.5 / GLM-4.6V campaigns measured, byte-identical). >1 makes vLLM "
                         "use MultiProcExecutor, one worker process per rank: the runner patch "
                         "is installed in each worker through --worker-extension-cls and every "
                         "correction op is dispatched with collective_rpc "
                         "(appcorr/vllm_stream/tp_worker.py, docs/memo/glm53_tp_plan.md). "
                         "GLM-5.3-Flash on B200-8 is the first model that needs this")
    ap.add_argument("--max-num-seqs", type=int, default=None)
    ap.add_argument("--max-num-batched-tokens", type=int, default=None,
                    help="engine prefill batch cap. vLLM defaults it to max_model_len; at 16384 "
                         "the 122B-FP8 trtllm MoE workspace for one 15.5k-token prefill exceeded "
                         "the ~31 GiB left beside a 48.9 GiB KV cache (OOM, 2026-09-08). 8192 "
                         "keeps the per-step activation at the size the 8192-max-len runs proved")
    ap.add_argument("--moe-backend", default=None,
                    help="vLLM MoE kernel backend (auto | triton | flashinfer_trtllm | "
                         "flashinfer_cutlass | batched_triton). The auto pick on B200 for "
                         "Qwen3.5 bf16 is FlashInfer TRTLLM, whose prefill step costs ~35 ms "
                         "flat from ~130 to ~1400 tokens (2026-09-09 critical-latency probe)")
    ap.add_argument("--cudagraph-capture-sizes", type=int, nargs="+", default=None,
                    help="explicit CUDA-graph capture sizes (token counts). The correct step pads "
                         "|P| UP to the nearest captured size, so a coarse ladder such as "
                         "8 16 32 64 128 256 384 512 serves it at a fraction of the capture memory "
                         "of vLLM's default ~83-size table (GLM-5.3 TP=2: capture OOMs at the "
                         "default, 2026-09-14)")
    ap.add_argument("--max-cudagraph-capture-size", type=int, default=None,
                    help="largest token count a captured CUDA graph covers. vLLM defaults it to "
                         "2 x max_num_seqs (128 at --max-num-seqs 64), so every chunk prefill step "
                         "above that runs eager: 35B steps of 128..850 tokens cost 35-39 ms flat "
                         "vs 15-16 ms for graph replay at <=128 tokens (2026-09-09 trace). 1024 "
                         "(the Blackwell default cap) covers every g=4 band of the table's datasets")
    ap.add_argument("--interleaved", action="store_true",
                    help="serve the interleaved schedule: forces VLLM_GDN_DECODE_KERNEL=triton (the "
                         "default cuda path bypasses the DeltaNet capture/correct hook, memo §6.1) and "
                         "requires --max-num-seqs >= the largest correct batch (band rows + text "
                         "suffix; 1024 covers the table's datasets at g=4)")
    a = ap.parse_args()
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    if a.interleaved:
        os.environ["VLLM_GDN_DECODE_KERNEL"] = "triton"
        # a correct round is sub-batched at max_num_seqs rows (correct.py); a small value only
        # costs repeated DeltaNet window re-scans. 1024 on 35B; the 122B-FP8 server at gpu-mem
        # 0.85 / 8k has ~360 mamba blocks, which bounds it to ~256.
        assert a.max_num_seqs and a.max_num_seqs >= 64, \
            "--interleaved needs --max-num-seqs >= 64 (rounds are sub-batched at that size)"
    from .client import StreamingLLM
    kw = {"limit_mm_per_prompt": {"image": 1}}
    if a.max_num_seqs:
        kw["max_num_seqs"] = a.max_num_seqs
    if a.max_num_batched_tokens:
        kw["max_num_batched_tokens"] = a.max_num_batched_tokens
    if a.moe_backend:
        kw["moe_backend"] = a.moe_backend
    if a.max_cudagraph_capture_size or a.cudagraph_capture_sizes:
        cc = {}
        if a.max_cudagraph_capture_size:
            cc["max_cudagraph_capture_size"] = a.max_cudagraph_capture_size
        if a.cudagraph_capture_sizes:
            cc["cudagraph_capture_sizes"] = sorted(set(a.cudagraph_capture_sizes))
        kw["compilation_config"] = cc
    llm = StreamingLLM(a.model, gpu_memory_utilization=a.gpu_mem, max_model_len=a.max_model_len,
                       enforce_eager=a.enforce_eager,
                       tensor_parallel_size=a.tensor_parallel_size, **kw)
    StreamServer(llm, a.host, a.port).serve_forever()


if __name__ == "__main__":
    main()
