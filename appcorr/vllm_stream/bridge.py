"""AppCorr-side client of the vLLM streaming server (`server.py`). NO vllm import -- this module
is what the `appcorr` env imports; the vLLM process runs in `appcorr-vllm`. Import it as
`appcorr.vllm_stream.bridge` (the package __init__ is lazy about its vllm-dependent modules).

    bridge = LLMBridge("127.0.0.1", 5555)
    sink = bridge.sink("req-0", max_tokens=24)          # one request
    sink.push(embeds_chunk, mrope_chunk, mrope_delta, final=False)   # as each band is ready
    ...
    sink.push(last_chunk, mrope_last, mrope_delta, final=True)
    out = sink.result()     # {"text", "token_ids", "timing": {...}}

`StreamSink` is the object the model axes accept (`streaming_forward(..., sink=...)`): the first
push opens the request, later pushes append, the push marked `final` closes the prompt. Chunks are
sent in the order the axis produces them, which is sequence order -- the server never reorders.
One-shot arms (floor/ceiling) are a single `push(..., final=True)` -- the same request path with
one chunk, so every arm decodes through the identical engine.

Pushes do not wait for the server's acknowledgement (added 2026-09-07): the server answers a
chunk only between engine steps, so a blocking push parked the vision side for 5-15 ms per band
-- 26-50 ms of a ~150 ms streaming pass on Qwen2.5-VL-7B -- which is exactly the overlap the
design exists to buy. Acks are read back, in order, before the next reply-bearing call
(`result`, `info`, `abort`); the per-push `t_ack` therefore means "ack observed", not "ack
arrived", and `t_recv_server` (server clock) is the one to use for arrival. The bytes themselves
go out on a sender thread: a band is ~3.7 MB and the loopback send buffer autotunes to 4 MB, so
`sendall` on the caller would still park it for the length of a server step.
"""
from __future__ import annotations

import queue
import socket
import threading
import time
from typing import Any, Optional

import torch

from .wire import Frame, recv_frame, send_frame


class BridgeError(RuntimeError):
    pass


class LLMBridge:
    def __init__(self, host: str = "127.0.0.1", port: int = 5555, timeout_s: float = 3600.0):
        self.host, self.port = host, port
        self.sock = socket.create_connection((host, port))
        self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.sock.settimeout(timeout_s)
        self._n = 0
        self._pending: list = []      # (sink, push-record) per un-acked push, in send order
        self._tx: "queue.Queue[Optional[tuple]]" = queue.Queue()
        self._tx_err: Optional[BaseException] = None
        self._tx_thread = threading.Thread(target=self._tx_loop, name="bridge-tx", daemon=True)
        self._tx_thread.start()

    def close(self) -> None:
        self._tx.put(None)
        self._tx_thread.join(timeout=5.0)
        try:
            self.sock.close()
        except OSError:
            pass

    def _tx_loop(self) -> None:
        while True:
            item = self._tx.get()
            if item is None:
                self._tx.task_done()
                return
            data, rec = item
            try:
                if self._tx_err is None:
                    self.sock.sendall(data)
                    if rec is not None:
                        rec["t_sent"] = time.perf_counter()
            except BaseException as e:  # noqa: BLE001 -- surfaced on the caller's next call
                self._tx_err = e
            finally:
                self._tx.task_done()

    def _flush(self) -> None:
        self._tx.join()
        if self._tx_err is not None:
            raise BridgeError(f"send failed: {self._tx_err!r}")

    def _recv_ok(self) -> Frame:
        rep = recv_frame(self.sock)
        if not rep.header.get("ok", False):
            raise BridgeError(rep.header.get("error", "unknown server error"))
        return rep

    def drain_acks(self) -> None:
        """Read every outstanding push acknowledgement (replies arrive in send order)."""
        self._flush()
        while self._pending:
            sink, rec = self._pending.pop(0)
            rep = self._recv_ok()
            rec["t_ack"] = time.perf_counter()
            rec["t_recv_server"] = float(rep.header["t_recv"])

    def _call(self, frame: Frame) -> Frame:
        self._flush()
        self.drain_acks()
        send_frame(self.sock, frame)
        return self._recv_ok()

    def _send_async(self, frame: Frame, sink: "StreamSink", rec: dict) -> None:
        self._tx.put((frame.encode(), rec))
        self._pending.append((sink, rec))

    # -- protocol ----------------------------------------------------------------------------
    def info(self) -> dict:
        return self._call(Frame({"op": "info"})).header

    def open(self, rid: str, embeds: torch.Tensor, mrope: Optional[torch.Tensor],
             mrope_delta: Optional[int], final: bool, max_tokens: int,
             logprobs: Optional[int] = None, sink: Optional["StreamSink"] = None,
             rec: Optional[dict] = None) -> None:
        f = Frame({"op": "open", "rid": rid, "final": bool(final), "max_tokens": int(max_tokens),
                   "mrope_delta": (None if mrope_delta is None else int(mrope_delta)),
                   "logprobs": logprobs, "t_client": time.perf_counter()})
        f.put_tensor("embeds", embeds).put_tensor("mrope", mrope)
        self._send_async(f, sink, rec if rec is not None else {})

    def append(self, rid: str, embeds: torch.Tensor, mrope: Optional[torch.Tensor],
               mrope_delta: Optional[int], final: bool, sink: Optional["StreamSink"] = None,
               rec: Optional[dict] = None) -> None:
        f = Frame({"op": "append", "rid": rid, "final": bool(final),
                   "mrope_delta": (None if mrope_delta is None else int(mrope_delta)),
                   "t_client": time.perf_counter()})
        f.put_tensor("embeds", embeds).put_tensor("mrope", mrope)
        self._send_async(f, sink, rec if rec is not None else {})

    def result(self, rid: str) -> dict:
        """Blocks until the request finished; returns text / token_ids / timing (+ logprobs)."""
        rep = self._call(Frame({"op": "result", "rid": rid}))
        return rep.header

    def abort(self, rid: str) -> None:
        self._call(Frame({"op": "abort", "rid": rid}))

    def sink(self, rid: Optional[str] = None, max_tokens: int = 24,
             logprobs: Optional[int] = None) -> "StreamSink":
        if rid is None:
            rid = f"r{self._n}-{time.time_ns() & 0xffffff:06x}"
            self._n += 1
        return StreamSink(self, rid, max_tokens, logprobs)


class StreamSink:
    """Chunk consumer handed to an axis' `streaming_forward`. Records client-side send times."""

    def __init__(self, bridge: LLMBridge, rid: str, max_tokens: int, logprobs: Optional[int]):
        self.bridge, self.rid, self.max_tokens, self.logprobs = bridge, rid, max_tokens, logprobs
        self.opened = False
        self.closed = False
        self.pushes: list[dict] = []
        self.num_tokens = 0

    def push(self, embeds: torch.Tensor, mrope: Optional[torch.Tensor],
             mrope_delta: Optional[int], final: bool) -> None:
        """embeds [T, D] (any device, bf16/fp16/fp32), mrope [3, T] int64 or None."""
        if self.closed:
            raise BridgeError(f"{self.rid}: push after final")
        assert embeds.ndim == 2, embeds.shape
        if mrope is not None:
            assert mrope.shape == (3, embeds.shape[0]), (mrope.shape, embeds.shape)
        rec = {"n": int(embeds.shape[0]), "final": bool(final), "t_send": time.perf_counter(),
               "t_ack": None, "t_recv_server": None}
        if not self.opened:
            self.bridge.open(self.rid, embeds, mrope, mrope_delta, final,
                             self.max_tokens, self.logprobs, sink=self, rec=rec)
            self.opened = True
        else:
            self.bridge.append(self.rid, embeds, mrope, mrope_delta, final, sink=self, rec=rec)
        rec["t_queued"] = time.perf_counter()   # encoded and handed to the sender thread
        self.num_tokens += int(embeds.shape[0])
        self.pushes.append(rec)
        if final:
            self.closed = True

    def result(self) -> dict:
        if not self.closed:
            raise BridgeError(f"{self.rid}: result() before the final chunk was pushed")
        out = self.bridge.result(self.rid)
        out["pushes"] = self.pushes
        out["num_prompt_tokens_sent"] = self.num_tokens
        return out
