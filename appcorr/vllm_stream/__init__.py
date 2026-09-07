"""AppCorr streaming prefill inside vLLM v1 (0.11.2 / 0.28.0), as a plugin -- no vLLM fork.

    import appcorr.vllm_stream as vs
    vs.install()                                   # patch EngineCore + GPUModelRunner (idempotent)
    llm = vs.StreamingLLM("Qwen/Qwen2.5-VL-7B-Instruct", ...)   # in-process engine
    llm.open(rid, chunk0, sampling_params); llm.append(rid, chunk1); ...; llm.append(rid, last, final=True)
    out = llm.run_until_done(rid)

Pieces: request.py (StreamingRequest, wire format, hold-back-one), scheduler.py
(StreamingScheduler), engine_patch.py / runner_patch.py (the four hooks), client.py
(StreamingLLM + Qwen2.5-VL prompt composition). Design memo: docs/memo/vllm_stream_design.md.

Pinned to the vllm releases in SUPPORTED_VLLM -- `install()` refuses any other version, because
the hooks wrap private methods whose contracts were read from those releases. Differences between
the two supported releases are version-gated in place (`vllm_version()`), see the memo.
"""
from __future__ import annotations

SUPPORTED_VLLM = ("0.11.2", "0.28.0")


def vllm_version() -> str:
    import vllm
    return vllm.__version__


def install() -> None:
    import vllm
    if vllm.__version__ not in SUPPORTED_VLLM:
        raise RuntimeError(f"appcorr.vllm_stream is pinned to vllm {SUPPORTED_VLLM}, found {vllm.__version__}")
    from . import engine_patch, runner_patch
    engine_patch.install()
    runner_patch.install()


def register() -> None:
    """vLLM general-plugin entry point (`vllm.general_plugins`); same as install()."""
    install()


from .request import StreamChunk, StreamingRequest, make_stream_headers  # noqa: E402
from .scheduler import StreamingScheduler  # noqa: E402
from .client import StreamingLLM  # noqa: E402

__all__ = ["install", "register", "StreamChunk", "StreamingRequest", "StreamingScheduler",
           "StreamingLLM", "make_stream_headers"]
