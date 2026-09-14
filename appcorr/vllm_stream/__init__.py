"""AppCorr streaming prefill inside vLLM v1 (0.11.2 / 0.28.0 / main 658c8131c), as a plugin -- no fork.

    import appcorr.vllm_stream as vs
    vs.install()                                   # patch EngineCore + GPUModelRunner (idempotent)
    llm = vs.StreamingLLM("Qwen/Qwen2.5-VL-7B-Instruct", ...)   # in-process engine
    llm.open(rid, chunk0, sampling_params); llm.append(rid, chunk1); ...; llm.append(rid, last, final=True)
    out = llm.run_until_done(rid)

Pieces: request.py (StreamingRequest, wire format, hold-back-one), scheduler.py
(StreamingScheduler), engine_patch.py / runner_patch.py (the four hooks), client.py
(StreamingLLM + Qwen2.5-VL prompt composition); server.py / bridge.py / wire.py (the two-process
form: vLLM in one env, the AppCorr vision fork in another, prompt embeds over a socket).
Design memo: docs/memo/vllm_stream_design.md.

Pinned to the vllm releases in SUPPORTED_VLLM -- `install()` refuses any other version, because
the hooks wrap private methods whose contracts were read from those releases. Differences between
the two supported releases are version-gated in place (`vllm_version()`), see the memo.
"""
from __future__ import annotations

# The third entry is vLLM MAIN at commit 658c8131c ("0.1.1.dev65+g658c8131c" is what
# `vllm.__version__` reports for that build) -- the nightly B200-8 serves for GLM-5.3-Flash,
# which is the only build that has `vllm/models/glm5next/` at all.  Admitted on the hook-point
# diff against 0.28.0 (docs/memo/glm53_vllm_survey.md): all six monkeypatch targets, both
# scheduler signatures and both EngineCore signatures are identical ON THE V1 RUNNER CLASS --
# which main no longer instantiates by default; `_pin_v1_model_runner()` below forces it back
# (the diff checked signatures, not which class the engine runs).  EXACT string on purpose --
# a prefix or regex would silently admit any other main nightly, whose private contracts nobody
# has read.  Every version gate below is `== "0.11.2"` / `!= "0.11.2"`, so this build takes the
# 0.28.0 branch everywhere, which is what the diff says it should.
SUPPORTED_VLLM = ("0.11.2", "0.28.0", "0.1.1.dev65+g658c8131c")


def vllm_version() -> str:
    import vllm
    return vllm.__version__


def _pin_v1_model_runner() -> None:
    """vLLM main ships TWO GPUModelRunner classes and instantiates the V2 one by default
    (`vllm/v1/worker/gpu/model_runner.py`, `gpu_worker.py` "Using V2 Model Runner").  Our hooks
    patch the V1 class (`vllm/v1/worker/gpu_model_runner.py`); three of the targets
    (`_update_states`, `_init_mrope_positions`, `_preprocess`) and the `self.requests` table the
    hook bodies read do not exist on V2, so on the default path the patches attach to a class the
    engine never runs and the first stream request dies with
    "'GPUModelRunner' object has no attribute 'requests'" (B200-8, leg 2, 2026-09-13).
    `VllmConfig.use_v2_model_runner` honours an explicit VLLM_USE_V2_MODEL_RUNNER, read lazily
    from the environment, so pinning it here -- before any VllmConfig is built -- puts V1 back.
    Two features (HiSparse, watermarking) force V2 regardless; neither is on our path.  This is a
    deprecation clock, not a fix: porting to V2 means re-basing those three hooks."""
    import os
    v = os.environ.get("VLLM_USE_V2_MODEL_RUNNER")
    if v is not None and v.strip().lower() in ("1", "true", "yes"):
        raise RuntimeError("VLLM_USE_V2_MODEL_RUNNER=1 is set, but appcorr.vllm_stream patches the "
                           "V1 GPUModelRunner only (see _pin_v1_model_runner); unset it.")
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"


def install() -> None:
    import vllm
    if vllm.__version__ not in SUPPORTED_VLLM:
        raise RuntimeError(f"appcorr.vllm_stream is pinned to vllm {SUPPORTED_VLLM}, found {vllm.__version__}")
    if vllm.__version__ == "0.1.1.dev65+g658c8131c":
        _pin_v1_model_runner()
    from . import engine_patch, runner_patch
    engine_patch.install()
    runner_patch.install()


def register() -> None:
    """vLLM general-plugin entry point (`vllm.general_plugins`); same as install()."""
    install()


# The vllm-dependent modules are imported lazily: `bridge.py` / `wire.py` are imported from the
# AppCorr process (appcorr env, no vllm), and `import appcorr.vllm_stream.bridge` must not drag
# `client.py` (which imports vllm) in through this __init__.
_LAZY = {"StreamChunk": ".request", "StreamingRequest": ".request", "make_stream_headers": ".request",
         "StreamingScheduler": ".scheduler", "StreamingLLM": ".client"}


def __getattr__(name):
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(name)
    import importlib
    return getattr(importlib.import_module(mod, __name__), name)


__all__ = ["install", "register", "StreamChunk", "StreamingRequest", "StreamingScheduler",
           "StreamingLLM", "make_stream_headers"]
