"""vLLM-side streaming server: prompt embeddings in over a socket, generated text out.

Runs in the `appcorr-vllm` env (vllm 0.28.0) and owns the LLM; the AppCorr process (vision fork,
`appcorr` env) talks to it through `bridge.LLMBridge` using the frames in `wire.py`. Single
event loop: poll the sockets, apply every message that arrived (open / append / final / result),
then run one engine step if any request is live -- the engine prefills whatever has arrived while
the vision side is still correcting the next band, which is the whole point (§ docs/memo/
vllm_stream_design.md). Several clients / requests may be live at once; the engine batches them
(that is the throughput measurement's lever).

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

from .wire import Frame, FrameParser, error_frame


class _Req:
    __slots__ = ("rid", "conn", "t_open", "t_final", "t_first", "t_done", "n_prompt", "out",
                 "waiter", "finished", "logprobs")

    def __init__(self, rid, conn, logprobs):
        self.rid, self.conn, self.logprobs = rid, conn, logprobs
        self.t_open = time.perf_counter()
        self.t_final = None
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
                                      "live_requests": sum(1 for r in self.reqs.values() if not r.finished),
                                      "done_requests": self.n_done}))
            elif op in ("open", "append"):
                self._chunk(c, f, op)
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
            sp = SamplingParams(temperature=0.0, max_tokens=int(f.header["max_tokens"]),
                                logprobs=lp)
            r = _Req(rid, c, lp)
            self.reqs[rid] = r
            c.rids.add(rid)
            self.llm.open(rid, chunk, sp)
        else:
            r = self.reqs.get(rid)
            if r is None:
                raise KeyError(f"append to unknown request {rid!r}")
            if r.t_final is not None:
                raise ValueError(f"{rid}: append after final")
            self.llm.append(rid, chunk)
        r.n_prompt += chunk.num_tokens
        t = time.perf_counter()
        if final:
            r.t_final = t
        self._reply(c, Frame({"ok": True, "t_recv": t, "num_prompt_tokens": r.n_prompt}))

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
                        "t_done": r.t_done,
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
        for o in self.llm.step():
            r = self.reqs.get(o.request_id)
            if r is None:
                continue
            if r.t_first is None and o.outputs and len(o.outputs[0].token_ids) >= 1:
                r.t_first = time.perf_counter()
            if o.finished:
                r.finished = True
                r.t_done = time.perf_counter()
                r.out = o
                self.n_done += 1
                if r.waiter is not None:
                    w, r.waiter = r.waiter, None
                    if w.sock in self.conns:
                        self._send_result(w, r)

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
                    self._handle(c, f)
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
    ap.add_argument("--max-num-seqs", type=int, default=None)
    a = ap.parse_args()
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from .client import StreamingLLM
    kw = {"limit_mm_per_prompt": {"image": 1}}
    if a.max_num_seqs:
        kw["max_num_seqs"] = a.max_num_seqs
    llm = StreamingLLM(a.model, gpu_memory_utilization=a.gpu_mem, max_model_len=a.max_model_len,
                       enforce_eager=a.enforce_eager, **kw)
    StreamServer(llm, a.host, a.port).serve_forever()


if __name__ == "__main__":
    main()
