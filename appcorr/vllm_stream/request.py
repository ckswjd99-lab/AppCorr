"""Streaming-prompt request for vLLM v1: a request whose prompt embeddings ARRIVE IN CHUNKS.

AppCorr's streaming arm (chunked causal prefill over progressively corrected vision embeddings,
`offload/server/model/qwen25vl_executor.py` `llm_schedule == "streaming"`) maps onto vLLM as a
request that is *opened* with its first chunk of `prompt_embeds` and *appended to* until the
client marks the last chunk `final`; only then is the request allowed to sample. Every prompt
position is prefilled exactly once (append-only: chunked/new-only never rewrites a row the LLM
has already consumed, so Stream2LLM's LCP/invalidate "update" mode is not needed).

Wire format (works over the in-process client today and is msgpack-clean for a future
multi-process engine core): an ordinary `EngineCoreRequest` whose `trace_headers` carry
    x-appcorr-stream = "open" | "oneshot" | "append" | "final"
plus, for M-RoPE models (Qwen2-VL family), the chunk's 3xT positions as JSON under
`x-appcorr-mrope` and the request's mrope delta under `x-appcorr-mrope-delta`. `open` is a
stock add_request with the first chunk in `prompt_embeds`; `append`/`final` re-use the same
`request_id` with the next chunk in `prompt_embeds` (sampling_params are ignored there);
`oneshot` = open+final in one message (a complete embeds prompt with client-supplied M-RoPE
positions -- the one-shot reference arm, and what a stock `prompt_embeds` request cannot do for
an M-RoPE model because the runner has no token ids to derive positions from).

The hold-back-one rule. While a request is open the scheduler sees ONE FEWER token than it
holds (`num_tokens` below). vLLM samples whenever a request's last known token gets computed;
holding the last row back until the next chunk lands means the model never samples on a
non-final chunk, no sampled token has to be discarded or rolled back, and the held row's
compute is simply deferred (it is prefilled together with the next chunk). The model runner is
told the full length, so from its side an open request is just a chunked prefill that is one
token short -- its stock `seq_len < num_tokens` discard path handles the logits.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
import torch
from vllm.v1.request import Request

STREAM_HEADER = "x-appcorr-stream"
MROPE_HEADER = "x-appcorr-mrope"
MROPE_DELTA_HEADER = "x-appcorr-mrope-delta"
MODES = ("open", "oneshot", "append", "final")


def cpu_cat(a: torch.Tensor, b: torch.Tensor, dim: int) -> torch.Tensor:
    """`torch.cat` for small CPU tensors that does not go through the intra-op thread pool.

    With the default thread count (72 here) `torch.cat` on a 1 MB CPU tensor costs ~90 ms of
    OpenMP fork/join, single-threaded it is 0.06 ms; the engine process should not have its
    thread count changed by us, so the copy is done by numpy (a plain memcpy). bf16 has no numpy
    dtype -- the bytes are moved as int16."""
    assert a.device.type == "cpu" and b.device.type == "cpu", (a.device, b.device)
    b = b.to(a.dtype)
    view = torch.int16 if a.dtype == torch.bfloat16 else a.dtype
    out = np.concatenate([a.contiguous().view(view).numpy(), b.contiguous().view(view).numpy()], axis=dim)
    return torch.from_numpy(out).view(a.dtype)


@dataclass
class StreamChunk:
    """One arriving slice of the prompt: embeddings [T, D] (CPU) and, for M-RoPE models, the
    3xT positions of exactly those T rows plus the request-level mrope delta."""
    embeds: torch.Tensor
    final: bool
    mrope_positions: Optional[torch.Tensor] = None  # [3, T] int64
    mrope_delta: Optional[int] = None

    def __post_init__(self):
        assert self.embeds.ndim == 2, self.embeds.shape
        if self.mrope_positions is not None:
            assert self.mrope_positions.shape == (3, self.embeds.shape[0]), (
                self.mrope_positions.shape, self.embeds.shape)

    @property
    def num_tokens(self) -> int:
        return int(self.embeds.shape[0])


def make_stream_headers(mode: str, chunk: StreamChunk) -> dict[str, str]:
    assert mode in MODES, mode
    h = {STREAM_HEADER: mode}
    if chunk.mrope_positions is not None:
        h[MROPE_HEADER] = json.dumps(chunk.mrope_positions.tolist())
        h[MROPE_DELTA_HEADER] = str(int(chunk.mrope_delta))
    return h


def parse_stream_headers(headers: Optional[Mapping[str, str]]):
    """-> (mode, mrope_positions [3,T] | None, mrope_delta | None), or None for a stock request."""
    if not headers or STREAM_HEADER not in headers:
        return None
    mode = headers[STREAM_HEADER]
    assert mode in MODES, mode
    pos = delta = None
    if MROPE_HEADER in headers:
        pos = torch.tensor(json.loads(headers[MROPE_HEADER]), dtype=torch.int64)
        delta = int(headers[MROPE_DELTA_HEADER])
    return mode, pos, delta


@dataclass
class StreamAppend:
    """What `EngineCore.add_request` receives for an append/final message (not a Request)."""
    request_id: str
    chunk: StreamChunk


class StreamingRequest(Request):
    """`Request` whose prompt grows by `stream_append` and which holds one token back while open.

    `prompt_token_ids` stays None (embeds-only request; `_all_token_ids` is the stock `[0] * n`
    placeholder list and grows with it). `mrope_positions`/`mrope_delta` accumulate the full
    prompt's positions so the runner can be handed them at first schedule (`NewRequestData`) or
    per appended chunk (`SchedulerOutput.appcorr_stream_updates`).
    """

    def __init__(self, *args, stream_open: bool = True, mrope_positions=None, mrope_delta=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        assert self.prompt_embeds is not None and self.prompt_token_ids is None, (
            "streaming requests are prompt_embeds-only")
        self.stream_open = stream_open
        self.mrope_positions = mrope_positions
        self.mrope_delta = mrope_delta

    @property
    def num_tokens(self) -> int:
        # hold-back-one while open (see module docstring)
        return len(self._all_token_ids) - (1 if self.stream_open else 0)

    @property
    def num_tokens_with_spec(self) -> int:
        return self.num_tokens + len(self.spec_token_ids)

    def stream_append(self, chunk: StreamChunk) -> None:
        assert self.stream_open, f"{self.request_id}: append after final"
        assert self.num_output_tokens == 0, f"{self.request_id}: append after sampling started"
        t = chunk.num_tokens
        self.prompt_embeds = cpu_cat(self.prompt_embeds, chunk.embeds, dim=0)
        self._all_token_ids.extend([0] * t)
        self.num_prompt_tokens += t
        if chunk.mrope_positions is not None:
            assert self.mrope_positions is not None, "mrope positions on append but none at open"
            self.mrope_positions = cpu_cat(self.mrope_positions, chunk.mrope_positions, dim=1)
            self.mrope_delta = chunk.mrope_delta
        self.stream_open = not chunk.final
