"""vLLM main defaults to the V2 GPUModelRunner, which our hooks do not patch (see
`appcorr.vllm_stream._pin_v1_model_runner`).  On that build `install()` must pin V1 through the
environment before any VllmConfig exists, and must refuse an explicit V2 request."""
import os
import pytest

import appcorr.vllm_stream as vs


def test_pin_sets_env(monkeypatch):
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    vs._pin_v1_model_runner()
    assert os.environ["VLLM_USE_V2_MODEL_RUNNER"] == "0"


def test_pin_refuses_explicit_v2(monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    with pytest.raises(RuntimeError, match="V1 GPUModelRunner only"):
        vs._pin_v1_model_runner()


def test_config_honours_pin(monkeypatch):
    vllm = pytest.importorskip("vllm")
    if vllm.__version__ != "0.1.1.dev65+g658c8131c":
        pytest.skip("V1/V2 runner split exists on the main nightly only")
    monkeypatch.delenv("VLLM_USE_V2_MODEL_RUNNER", raising=False)
    vs._pin_v1_model_runner()
    from vllm.config import VllmConfig
    assert VllmConfig().use_v2_model_runner is False
