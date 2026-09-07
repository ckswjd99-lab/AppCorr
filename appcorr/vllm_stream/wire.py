"""Wire format between the AppCorr process (vision, `appcorr` env) and the vLLM process
(`appcorr-vllm` env). Two conda envs because vLLM 0.28.0 pins a torch the AppCorr fork does not
run on; two processes is also the deployment shape (the vision side is the "server-side
progressive decoder", the LLM side is the serving engine).

Framing, one message = one frame:

    u32 big-endian  header length H
    H bytes         JSON header (utf-8)
    N bytes         binary payload, concatenated in the order of header["bin"] (list of sizes)

Tensors travel as raw bytes + a {"shape", "dtype", "bin": k} descriptor in the header (`bin` is the
index into header["bin"]). bf16 has no numpy dtype, so it is carried as its int16 bit pattern and
reinterpreted on the other side (`tensor_to_wire` / `wire_to_tensor`); nothing is rounded.

Request flow (client -> server; every message gets one reply frame):

    {"op": "info"}                                     -> {"ok", "model", "vllm", ...}
    {"op": "open",   "rid", "final", "max_tokens", "mrope_delta", "embeds": T, "mrope": T|null}
                                                       -> {"ok", "t_recv"}          (final=True is one-shot)
    {"op": "append", "rid", "final", "mrope_delta", "embeds": T, "mrope": T|null}
                                                       -> {"ok", "t_recv"}
    {"op": "result", "rid"}                            -> {"ok", "text", "token_ids", "timing": {...}}
                                                          (blocks until the request finishes)
    {"op": "abort",  "rid"}                            -> {"ok"}

Timing (server perf_counter, seconds): t_open, t_final (last chunk received), t_first_token
(first sampled token observed), t_done. TTFT-from-last-chunk = t_first_token - t_final, the
same quantity vllm_stream_ttft.py measures in-process.

No vllm import here -- this module is imported on BOTH sides.
"""
from __future__ import annotations

import json
import socket
import struct
from typing import Any, Optional

import numpy as np
import torch

_LEN = struct.Struct(">I")
MAX_HEADER = 64 << 20

_TORCH_TO_NP = {
    torch.float32: ("float32", np.float32),
    torch.float16: ("float16", np.float16),
    torch.bfloat16: ("bfloat16", np.int16),   # bit pattern
    torch.int64: ("int64", np.int64),
    torch.int32: ("int32", np.int32),
}


def tensor_to_wire(t: torch.Tensor) -> tuple[dict, bytes]:
    """(descriptor, bytes). Moves to CPU; bf16 goes out as its int16 bit pattern."""
    t = t.detach()
    if t.dtype not in _TORCH_TO_NP:
        raise TypeError(f"unsupported dtype {t.dtype}")
    name, npdt = _TORCH_TO_NP[t.dtype]
    if t.dtype == torch.bfloat16:
        t = t.view(torch.int16)
    arr = t.contiguous().cpu().numpy()
    assert arr.dtype == npdt, (arr.dtype, npdt)
    return {"shape": list(t.shape), "dtype": name}, arr.tobytes()


def wire_to_tensor(desc: dict, buf: bytes) -> torch.Tensor:
    name = desc["dtype"]
    shape = tuple(desc["shape"])
    if name == "bfloat16":
        arr = np.frombuffer(buf, dtype=np.int16).reshape(shape)
        return torch.from_numpy(arr.copy()).view(torch.bfloat16)
    arr = np.frombuffer(buf, dtype=np.dtype(name)).reshape(shape)
    return torch.from_numpy(arr.copy())


class Frame:
    """A message being assembled: header dict + list of binary blobs."""

    def __init__(self, header: Optional[dict] = None):
        self.header: dict = dict(header or {})
        self.blobs: list[bytes] = []

    def put_tensor(self, key: str, t: Optional[torch.Tensor]) -> "Frame":
        if t is None:
            self.header[key] = None
            return self
        desc, b = tensor_to_wire(t)
        desc["bin"] = len(self.blobs)
        self.blobs.append(b)
        self.header[key] = desc
        return self

    def get_tensor(self, key: str) -> Optional[torch.Tensor]:
        desc = self.header.get(key)
        if desc is None:
            return None
        return wire_to_tensor(desc, self.blobs[desc["bin"]])

    def encode(self) -> bytes:
        h = dict(self.header)
        h["bin"] = [len(b) for b in self.blobs]
        hb = json.dumps(h).encode("utf-8")
        return b"".join([_LEN.pack(len(hb)), hb] + self.blobs)


def send_frame(sock: socket.socket, frame: Frame) -> None:
    sock.sendall(frame.encode())


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    parts = []
    got = 0
    while got < n:
        chunk = sock.recv(min(n - got, 1 << 22))
        if not chunk:
            raise ConnectionError("peer closed")
        parts.append(chunk)
        got += len(chunk)
    return b"".join(parts)


def recv_frame(sock: socket.socket) -> Frame:
    (hlen,) = _LEN.unpack(_recv_exact(sock, _LEN.size))
    if hlen > MAX_HEADER:
        raise ValueError(f"header too large: {hlen}")
    header = json.loads(_recv_exact(sock, hlen).decode("utf-8"))
    sizes = header.pop("bin", [])
    f = Frame(header)
    for n in sizes:
        f.blobs.append(_recv_exact(sock, n))
    return f


class FrameParser:
    """Incremental parser for a non-blocking socket: feed bytes, pop complete frames."""

    def __init__(self):
        self.buf = bytearray()

    def feed(self, data: bytes) -> list[Frame]:
        self.buf += data
        out = []
        while True:
            if len(self.buf) < _LEN.size:
                break
            (hlen,) = _LEN.unpack_from(self.buf, 0)
            if len(self.buf) < _LEN.size + hlen:
                break
            header = json.loads(bytes(self.buf[_LEN.size:_LEN.size + hlen]).decode("utf-8"))
            sizes = header.get("bin", [])
            total = _LEN.size + hlen + sum(sizes)
            if len(self.buf) < total:
                break
            header.pop("bin", None)
            f = Frame(header)
            off = _LEN.size + hlen
            for n in sizes:
                f.blobs.append(bytes(self.buf[off:off + n]))
                off += n
            del self.buf[:total]
            out.append(f)
        return out


def error_frame(msg: str) -> Frame:
    return Frame({"ok": False, "error": msg})
