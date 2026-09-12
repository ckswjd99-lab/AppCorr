"""`correct._leaf_spec` must see through vLLM main's `UniformTypeKVCacheSpecs` container and be
the identity on a leaf spec (and on builds without the container class)."""
import pytest

pytest.importorskip("vllm")          # correct.py imports the GPUModelRunner at module level
from appcorr.vllm_stream.correct import _leaf_spec  # noqa: E402


class _Leaf:
    pass


def test_identity_on_leaf():
    leaf = _Leaf()
    assert _leaf_spec(leaf) is leaf
    assert _leaf_spec(leaf, "any.layer") is leaf


def test_descends_into_container():
    kvi = pytest.importorskip("vllm.v1.kv_cache_interface")
    if not hasattr(kvi, "UniformTypeKVCacheSpecs"):
        pytest.skip("no container spec on this vLLM")
    a, b = _Leaf(), _Leaf()
    cont = kvi.UniformTypeKVCacheSpecs.__new__(kvi.UniformTypeKVCacheSpecs)
    object.__setattr__(cont, "kv_cache_specs", {"l.a": a, "l.b": b})
    assert _leaf_spec(cont, "l.b") is b
    assert _leaf_spec(cont) is a
