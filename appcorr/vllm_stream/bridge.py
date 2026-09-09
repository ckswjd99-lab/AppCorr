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
design exists to buy. Acks are read back, in order, by the reader thread; the per-push `t_ack`
is when the reader saw it, and `t_recv_server` (server clock) is the arrival. The bytes themselves
go out on a sender thread: a band is ~3.7 MB and the loopback send buffer autotunes to 4 MB, so
`sendall` on the caller would still park it for the length of a server step.

Replies are read by a reader thread (added 2026-09-09) and `result` travels on a second socket.
Before, `result()` first drained every outstanding ack on the one socket -- including those of
the requests pushed AFTER the one being collected -- and the server answers each chunk only
between engine steps, so with 4 requests in flight the driver's main loop sat ~100 ms per sample
in `result()` waiting for the newest sample's five acks (t_loop_wait, 122B VisDrone c4:
vision 95 ms + wait 100 ms per iteration, 4.4 samples/s against a 7.8/s ceiling arm). Now the
data socket carries pushes/acks/info/abort (replies in send order, matched FIFO by the reader),
the result socket carries only `result` (one at a time, synchronous); a sink waits for its own
final ack before asking, so the server has seen the whole prompt.
"""
from __future__ import annotations

import os
import queue
import socket
import threading
import time
from typing import Any, Optional

import torch

from .wire import Frame, recv_frame, send_frame

# Diagnostic knob (2026-09-09): sleep after every push so a fast driver reproduces a slow one's
# chunk-arrival spacing -- used to tell engine batch-timing nondeterminism (which chunks the
# scheduler prefills together) from a real wire-path difference. Off unless set.
_PUSH_DELAY_S = float(os.environ.get("APPCORR_PUSH_DELAY_MS", "0")) / 1e3


class BridgeError(RuntimeError):
    pass


class LLMBridge:
    def __init__(self, host: str = "127.0.0.1", port: int = 5555, timeout_s: float = 3600.0):
        self.host, self.port = host, port
        self.timeout_s = timeout_s
        self.sock = self._connect()          # pushes, acks, info, abort
        self.rsock = self._connect()         # result requests only (never queued behind chunks)
        self._n = 0
        self._lock = threading.Lock()
        self._fifo: list = []          # reply targets on `sock`, in send order: ("push", sink, rec) | ("call", entry)
        self._tx: "queue.Queue[Optional[tuple]]" = queue.Queue()
        self._tx_err: Optional[BaseException] = None
        self._rx_err: Optional[BaseException] = None
        self._tx_thread = threading.Thread(target=self._tx_loop, name="bridge-tx", daemon=True)
        self._tx_thread.start()
        self._rx_thread = threading.Thread(target=self._rx_loop, name="bridge-rx", daemon=True)
        self._rx_thread.start()
        # Result fetches run on their own thread: the reader enqueues a sink the moment its
        # final chunk is acked, the worker asks the server (one outstanding request on `rsock`)
        # and parks the reply on the sink. The caller's `sink.result()` then finds it ready
        # instead of paying the server's between-steps reply latency in the main loop
        # (measured 37-47 ms per sample at c4/c8, 2026-09-09).
        self._rq: "queue.Queue[Optional[StreamSink]]" = queue.Queue()
        self._rs_err: Optional[BaseException] = None
        self._rs_thread = threading.Thread(target=self._result_loop, name="bridge-result", daemon=True)
        self._rs_thread.start()

    def _connect(self) -> socket.socket:
        s = socket.create_connection((self.host, self.port))
        s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        s.settimeout(self.timeout_s)
        return s

    def close(self) -> None:
        self._tx.put(None)
        self._tx_thread.join(timeout=5.0)
        for so in (self.sock, self.rsock):
            try:
                so.close()
            except OSError:
                pass
        self._rx_thread.join(timeout=5.0)
        self._rq.put(None)
        self._rs_thread.join(timeout=5.0)

    def _tx_loop(self) -> None:
        while True:
            item = self._tx.get()
            if item is None:
                self._tx.task_done()
                return
            frame, rec = item
            try:
                if self._tx_err is None:
                    # encode() here, not on the caller's thread: a frame built with
                    # async_d2h waits for its device->host copies at this point.
                    self.sock.sendall(frame.encode())
                    if rec is not None:
                        rec["t_sent"] = time.perf_counter()
            except BaseException as e:  # noqa: BLE001 -- surfaced on the caller's next call
                self._tx_err = e
            finally:
                self._tx.task_done()

    def _rx_loop(self) -> None:
        """Reads every reply on the data socket and hands it to the FIFO head: a push ack fills
        its record (an error goes onto that push's sink, raised by the sink's next call, never
        by whichever caller happened to be waiting -- 2026-09-08); a synchronous call gets its
        reply and its event set."""
        try:
            while True:
                rep = recv_frame(self.sock)
                with self._lock:
                    if not self._fifo:
                        raise BridgeError(f"unsolicited reply {rep.header}")
                    target = self._fifo.pop(0)
                if target[0] == "push":
                    _, sink, rec = target
                    if not rep.header.get("ok", False):
                        msg = rep.header.get("error", "unknown server error")
                        rec["error"] = msg
                        if sink is not None and sink.error is None:
                            sink.error = msg
                    else:
                        rec["t_ack"] = time.perf_counter()
                        rec["t_recv_server"] = float(rep.header["t_recv"])
                    if sink is not None and rec.get("final"):
                        sink.final_acked.set()
                        if sink.error is None:
                            self._rq.put(sink)
                        else:
                            sink.done.set()
                else:
                    entry = target[1]
                    entry["rep"] = rep
                    entry["ev"].set()
        except BaseException as e:  # noqa: BLE001 -- socket closed or protocol error
            self._rx_err = e
            with self._lock:
                pending, self._fifo = self._fifo, []
            for target in pending:
                if target[0] == "push":
                    _, sink, rec = target
                    rec["error"] = f"reader failed: {e!r}"
                    if sink is not None:
                        if sink.error is None:
                            sink.error = rec["error"]
                        sink.final_acked.set()
                        sink.done.set()
                else:
                    target[1]["ev"].set()

    def _result_loop(self) -> None:
        """Serves the result queue: one `result` request at a time on `rsock`, reply (or the
        server's error) parked on the sink, then `sink.done`."""
        try:
            while True:
                sink = self._rq.get()
                if sink is None:
                    return
                try:
                    sink._result = self.result(sink.rid)
                except BridgeError as e:
                    sink.error = str(e)
                sink.done.set()
        except BaseException as e:  # noqa: BLE001 -- socket closed or protocol error
            self._rs_err = e
            while True:
                try:
                    sink = self._rq.get_nowait()
                except queue.Empty:
                    break
                if sink is not None:
                    sink.error = f"result thread failed: {e!r}"
                    sink.done.set()

    def _flush(self) -> None:
        self._tx.join()
        if self._tx_err is not None:
            raise BridgeError(f"send failed: {self._tx_err!r}")

    def _check_rx(self) -> None:
        if self._rx_err is not None:
            raise BridgeError(f"reader failed: {self._rx_err!r}")

    def drain_acks(self) -> None:
        """Wait until every push sent so far has been answered (all pushes, all sinks). The
        per-request path no longer needs this (`StreamSink.result` waits for its own final
        ack); kept for callers that want a quiescent bridge, e.g. before `info` in a gate."""
        self._flush()
        while True:
            with self._lock:
                busy = any(t[0] == "push" for t in self._fifo)
            self._check_rx()
            if not busy:
                return
            time.sleep(0.0005)

    def _call(self, frame: Frame) -> Frame:
        """Synchronous request on the data socket (info, abort): queued behind the pushes
        already sent, answered in order by the reader thread."""
        self._flush()
        self._check_rx()
        entry = {"ev": threading.Event(), "rep": None}
        with self._lock:
            self._fifo.append(("call", entry))
            send_frame(self.sock, frame)
        if not entry["ev"].wait(self.timeout_s):
            raise BridgeError("server reply timed out")
        self._check_rx()
        rep = entry["rep"]
        if not rep.header.get("ok", False):
            raise BridgeError(rep.header.get("error", "unknown server error"))
        return rep

    def _send_async(self, frame: Frame, sink: "StreamSink", rec: dict) -> None:
        with self._lock:
            self._fifo.append(("push", sink, rec))
            self._tx.put((frame, rec))

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
        f.put_tensor("embeds", embeds, async_d2h=True).put_tensor("mrope", mrope, async_d2h=True)
        self._send_async(f, sink, rec if rec is not None else {})

    def append(self, rid: str, embeds: torch.Tensor, mrope: Optional[torch.Tensor],
               mrope_delta: Optional[int], final: bool, sink: Optional["StreamSink"] = None,
               rec: Optional[dict] = None) -> None:
        f = Frame({"op": "append", "rid": rid, "final": bool(final),
                   "mrope_delta": (None if mrope_delta is None else int(mrope_delta)),
                   "t_client": time.perf_counter()})
        f.put_tensor("embeds", embeds, async_d2h=True).put_tensor("mrope", mrope, async_d2h=True)
        self._send_async(f, sink, rec if rec is not None else {})

    def result(self, rid: str) -> dict:
        """Blocks until the request finished; returns text / token_ids / timing (+ logprobs).
        Goes over the result socket, so it is answered as soon as the request finishes instead
        of after every chunk queued behind it on the data socket. The caller must know the
        server has seen the request (`StreamSink.result` waits for its final ack)."""
        send_frame(self.rsock, Frame({"op": "result", "rid": rid}))
        rep = recv_frame(self.rsock)
        if not rep.header.get("ok", False):
            raise BridgeError(rep.header.get("error", "unknown server error"))
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
        self.error: Optional[str] = None   # first server error answering one of this sink's pushes
        self.final_acked = threading.Event()   # set by the reader when the final chunk is answered
        self.done = threading.Event()          # set by the result thread: `_result` or `error` filled
        self._result: Optional[dict] = None
        self.pushes: list[dict] = []
        self.num_tokens = 0

    def push(self, embeds: torch.Tensor, mrope: Optional[torch.Tensor],
             mrope_delta: Optional[int], final: bool) -> None:
        """embeds [T, D] (any device, bf16/fp16/fp32), mrope [3, T] int64 or None. The
        device->host copy is deferred to the sender thread (`wire.tensor_to_wire_async`), so
        this returns without stalling the caller's CUDA stream."""
        if self.error is not None:
            raise BridgeError(f"{self.rid}: {self.error}")
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
        if _PUSH_DELAY_S > 0:
            time.sleep(_PUSH_DELAY_S)          # diagnostic only: mimic a slower producer
        self.num_tokens += int(embeds.shape[0])
        self.pushes.append(rec)
        if final:
            self.closed = True

    def result(self) -> dict:
        if not self.closed:
            raise BridgeError(f"{self.rid}: result() before the final chunk was pushed")
        # Only THIS request's chunks must be on the server before asking (the result request
        # takes the other socket); a rejected chunk of ours surfaces here, on this sink. No
        # `_flush()`: that joins the sender queue, i.e. waits for the NEWEST sample's chunks to
        # leave -- the server reads the socket only between steps, so it is the same coupling
        # the reader thread was added to remove (measured 47 ms per sample at c4/c8).
        if not self.final_acked.wait(self.bridge.timeout_s):
            if self.bridge._tx_err is not None:
                raise BridgeError(f"{self.rid}: send failed: {self.bridge._tx_err!r}")
            raise BridgeError(f"{self.rid}: final ack timed out")
        self.bridge._check_rx()
        if self.error is not None:
            raise BridgeError(f"{self.rid}: {self.error}")
        # The result thread asked the server as soon as the final ack landed (see `_result_loop`).
        if not self.done.wait(self.bridge.timeout_s):
            if self.bridge._rs_err is not None:
                raise BridgeError(f"{self.rid}: result thread failed: {self.bridge._rs_err!r}")
            raise BridgeError(f"{self.rid}: result timed out")
        if self.error is not None:
            raise BridgeError(f"{self.rid}: {self.error}")
        out = dict(self._result)
        out["pushes"] = self.pushes
        out["num_prompt_tokens_sent"] = self.num_tokens
        return out
