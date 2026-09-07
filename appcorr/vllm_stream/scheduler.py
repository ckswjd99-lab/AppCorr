"""`StreamingScheduler`: the stock v1 scheduler plus `stream_append`.

Nothing in the stock scheduling loop changes -- an open `StreamingRequest` presents itself one
token short (`request.py`), so `Scheduler.schedule()` chunk-prefills whatever has arrived and
naturally schedules 0 tokens (and does not sample) while the request waits for its next chunk.
This class only (a) grows the request when a chunk lands and (b) tells the model runner about
the growth, by decorating the `SchedulerOutput` it would have produced anyway:

  * `NewRequestData.appcorr_stream`      -- (mrope_positions, mrope_delta) for a streaming
                                            request at its first schedule (the runner cannot
                                            derive M-RoPE positions from an embeds-only prompt).
  * `SchedulerOutput.appcorr_stream_updates` -- {req_id: merged StreamChunk} for requests the
                                            runner already holds; attached on the first step the
                                            request is scheduled after the append, i.e. together
                                            with the tokens that need those rows.

Both are plain attributes on the stock dataclasses: fine for the in-process engine core this
prototype targets; a multi-process core would need them as real (msgpack) fields.
Select with `scheduler_cls="appcorr.vllm_stream.scheduler.StreamingScheduler"`.
"""
from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Optional

import torch
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.request import RequestStatus

from .request import StreamChunk, StreamingRequest, cpu_cat


@dataclass
class StreamNewInfo:
    mrope_positions: Optional[torch.Tensor]
    mrope_delta: Optional[int]


def _merge_chunks(chunks: list[StreamChunk]) -> StreamChunk:
    if len(chunks) == 1:
        return chunks[0]
    pos = None
    if chunks[0].mrope_positions is not None:
        pos = functools.reduce(lambda a, b: cpu_cat(a, b, dim=1), [c.mrope_positions for c in chunks])
    emb = functools.reduce(lambda a, b: cpu_cat(a, b, dim=0), [c.embeds for c in chunks])
    return StreamChunk(embeds=emb, final=chunks[-1].final, mrope_positions=pos,
                       mrope_delta=chunks[-1].mrope_delta)


class StreamingScheduler(Scheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # req_id -> chunks appended since the runner last saw the request's prompt
        self._runner_updates: dict[str, list[StreamChunk]] = {}
        # requests the runner has been handed as NewRequestData at least once
        self._runner_knows: set[str] = set()

    # -- append path (called from the patched EngineCore.add_request) ------------------------
    def stream_append(self, request_id: str, chunk: StreamChunk) -> None:
        req = self.requests.get(request_id)
        if req is None:
            raise KeyError(f"stream append for unknown/finished request {request_id}")
        if not isinstance(req, StreamingRequest):
            raise TypeError(f"{request_id} was not opened as a streaming request")
        req.stream_append(chunk)
        if request_id in self._runner_knows:
            self._runner_updates.setdefault(request_id, []).append(chunk)
        # else: the runner will get the grown prompt in NewRequestData at first schedule

    # -- decorate the stock output ---------------------------------------------------------
    def schedule(self, *args, **kwargs) -> SchedulerOutput:  # 0.28.0 passes `throttle_prefills`
        out = super().schedule(*args, **kwargs)
        for nr in out.scheduled_new_reqs:
            req = self.requests[nr.req_id]
            if isinstance(req, StreamingRequest):
                nr.appcorr_stream = StreamNewInfo(req.mrope_positions, req.mrope_delta)
                self._runner_knows.add(nr.req_id)
                self._runner_updates.pop(nr.req_id, None)
        updates: dict[str, StreamChunk] = {}
        for rid in out.scheduled_cached_reqs.req_ids:
            pending = self._runner_updates.pop(rid, None)
            if pending:
                updates[rid] = _merge_chunks(pending)
        out.appcorr_stream_updates = updates
        return out

    def _free_request(self, request):
        self._runner_updates.pop(request.request_id, None)
        self._runner_knows.discard(request.request_id)
        return super()._free_request(request)

    # -- introspection for the client ---------------------------------------------------------
    def stream_state(self, request_id: str) -> Optional[dict]:
        req = self.requests.get(request_id)
        if req is None or not isinstance(req, StreamingRequest):
            return None
        return {"open": req.stream_open, "num_prompt_tokens": req.num_prompt_tokens,
                "num_computed_tokens": req.num_computed_tokens,
                "status": RequestStatus(req.status).name, "num_output_tokens": req.num_output_tokens}
