"""The composer's embed callable crosses `collective_rpc` at TP>1 through the shm_broadcast
pickler (standard pickle). A nested closure does not survive that (GLM-5.3 TP gate, run 3); a
`functools.partial` of a module-level function must."""
import functools, pickle, types, torch
import pytest
pytest.importorskip("vllm")  # client.py imports vllm; run this file under the appcorr-vllm env
from appcorr.vllm_stream import client as C


def _partial(fn):
    cfg = types.SimpleNamespace(tag="vllm_config_stand_in")
    return functools.partial(fn, ids=torch.arange(6), pv=torch.zeros(4, 8), thw=torch.tensor([[1, 2, 2]]),
                             i0=1, L=4, image_embeds=None, vllm_config=cfg)


def test_mrope_body_pickles():
    p = _partial(C._embed_prompt_mrope)
    q = pickle.loads(pickle.dumps(p, protocol=5))
    assert q.func is C._embed_prompt_mrope and q.keywords["L"] == 4


def test_nomrope_body_pickles():
    p = _partial(C._embed_prompt_nomrope)
    q = pickle.loads(pickle.dumps(p, protocol=5))
    assert q.func is C._embed_prompt_nomrope and torch.equal(q.keywords["ids"], torch.arange(6))


def test_no_closure_left_in_embed():
    import inspect
    for cls in (C.Qwen25VLComposer, C.Glm53Composer):
        src = inspect.getsource(cls.embed)
        assert "def fn(" not in src, cls
        assert "functools.partial(_embed_prompt_" in src, cls
