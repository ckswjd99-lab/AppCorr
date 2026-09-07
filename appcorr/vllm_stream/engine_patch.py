"""EngineCore hooks: route `x-appcorr-stream` requests (see `request.py`).

`preprocess_add_request` builds a `StreamingRequest` for "open"/"oneshot" and a `StreamAppend` for
"append"/"final"; `add_request` hands a `StreamAppend` to `StreamingScheduler.stream_append`
instead of the scheduler's add path. Stock requests are untouched. Installed once per process by
`appcorr.vllm_stream.install()`; the engine core loads vLLM general plugins in its own process
(`vllm/v1/engine/core.py`, "plugins need to be loaded at the engine/scheduler level too"), so
the same hooks would apply under a multi-process core once the SchedulerOutput side is
serialisable (scheduler.py).
"""
from __future__ import annotations

from vllm.v1.engine import EngineCoreRequest
from vllm.v1.engine.core import EngineCore
from vllm.v1.structured_output.request import StructuredOutputRequest  # noqa: F401  (parity)

from .request import StreamAppend, StreamChunk, StreamingRequest, parse_stream_headers

_ORIG_PREPROCESS = EngineCore.preprocess_add_request
_ORIG_ADD = EngineCore.add_request


def _preprocess_add_request(self: EngineCore, request: EngineCoreRequest):
    parsed = parse_stream_headers(request.trace_headers)
    if parsed is None:
        return _ORIG_PREPROCESS(self, request)
    mode, pos, delta = parsed
    assert request.prompt_embeds is not None and request.prompt_token_ids is None, (
        "streaming messages carry the chunk in prompt_embeds (no token ids)")
    if mode in ("open", "oneshot"):
        assert request.mm_features is None or len(request.mm_features) == 0, (
            "streaming requests carry no multimodal inputs -- vision embeds arrive as prompt_embeds")
        req = StreamingRequest.from_engine_core_request(request, self.request_block_hasher)
        req.mrope_positions, req.mrope_delta = pos, delta
        req.stream_open = mode == "open"
        if req.use_structured_output:
            self.structured_output_manager.grammar_init(req)
        return req, request.current_wave
    chunk = StreamChunk(embeds=request.prompt_embeds, final=(mode == "final"),
                        mrope_positions=pos, mrope_delta=delta)
    return StreamAppend(request.request_id, chunk), request.current_wave


def _add_request(self: EngineCore, request, request_wave: int = 0):
    if isinstance(request, StreamAppend):
        sched = self.scheduler
        if not hasattr(sched, "stream_append"):
            raise TypeError("streaming append needs scheduler_cls=appcorr.vllm_stream.scheduler.StreamingScheduler")
        sched.stream_append(request.request_id, request.chunk)
        return
    return _ORIG_ADD(self, request, request_wave)


def install() -> None:
    if getattr(EngineCore, "_appcorr_stream_patched", False):
        return
    EngineCore.preprocess_add_request = _preprocess_add_request
    EngineCore.add_request = _add_request
    EngineCore._appcorr_stream_patched = True
