"""The interleaved correct step's Gated-DeltaNet half must be conditional on the served model.

Qwen3.5's decoder is hybrid (30 of 40 layers recurrent), GLM-4.6V's is not (46 x
`Glm4MoeDecoderLayer`, all softmax GQA).  `appcorr/vllm_stream/correct.py` therefore decides by
walking `_decoder_module(model).layers` and looking at layer CLASSES -- never at a model name --
and only then imports/patches `QwenGatedDeltaNetAttention._forward_core`.

These tests run on CPU with no GPU and no vllm install: the two vllm symbols `correct.py` needs
are stubbed into `sys.modules` before the import (`GPUModelRunner` is only used as a patch
target and a type annotation, `QwenGatedDeltaNetAttention` only as the class to patch), and the
"model" is a plain `torch.nn` tree of the two shapes.  What is being tested is the dispatch, so a
fake tree is the right fixture; the real-engine behaviour is gate G2 in
`analysis/experiments/glm46v_correct_gate.py`.

    python -m pytest tests/test_correct_gdn_gating.py -q
"""
import sys
import types
import unittest

import torch
import torch.nn as nn


# --- stub the two vllm symbols correct.py needs, if vllm is not installed --------------------- #

def _install_vllm_stubs():
    try:                                            # real vllm: use it
        import vllm.v1.worker.gpu_model_runner            # noqa: F401
        import vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn  # noqa: F401
        return False
    except Exception:
        pass
    for name in ("vllm", "vllm.v1", "vllm.v1.worker", "vllm.model_executor",
                 "vllm.model_executor.layers", "vllm.model_executor.layers.mamba",
                 "vllm.model_executor.layers.mamba.gdn"):
        sys.modules.setdefault(name, types.ModuleType(name))
    runner_mod = types.ModuleType("vllm.v1.worker.gpu_model_runner")

    class GPUModelRunner:                            # patch target only
        pass

    runner_mod.GPUModelRunner = GPUModelRunner
    sys.modules["vllm.v1.worker.gpu_model_runner"] = runner_mod

    gdn_mod = types.ModuleType(
        "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn")

    class QwenGatedDeltaNetAttention(nn.Module):
        def _forward_core(self, mixed_qkv, b, a, core_attn_out):
            return "stock"

    gdn_mod.QwenGatedDeltaNetAttention = QwenGatedDeltaNetAttention
    sys.modules[gdn_mod.__name__] = gdn_mod
    return True


_STUBBED = _install_vllm_stubs()

from appcorr.vllm_stream import correct  # noqa: E402

_GDN_MOD_NAME = "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
_GDN_CLASS = sys.modules[_GDN_MOD_NAME].QwenGatedDeltaNetAttention


# --- fake module trees ------------------------------------------------------------------------ #

class _Attn(nn.Module):
    """Stand-in for `Glm4MoeAttention`: an ordinary softmax GQA block."""


class _Gdn(_GDN_CLASS):
    """Stand-in for a Qwen3.5 recurrent layer: a real subclass of the patched class."""

    def __init__(self, prefix):
        super().__init__()
        self.prefix = prefix
        self.enable_fused_gdn_decode = False


class _SoftmaxLayer(nn.Module):
    def __init__(self, i):
        super().__init__()
        self.self_attn = _Attn()


class _HybridLayer(nn.Module):
    def __init__(self, i, recurrent):
        super().__init__()
        if recurrent:
            self.linear_attn = _Gdn(f"model.layers.{i}.linear_attn")
        else:
            self.self_attn = _Attn()


class _Decoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)


class _CausalLM(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.model = decoder


class _VLM(nn.Module):
    """`Glm4vMoeForConditionalGeneration` / `Qwen3_5VLForConditionalGeneration` shape:
    `.language_model` -> `.model` -> `.layers`."""

    def __init__(self, decoder):
        super().__init__()
        self.language_model = _CausalLM(decoder)


def glm_like():
    """46 softmax layers, no recurrent layer anywhere (GLM-4.6V)."""
    return _VLM(_Decoder([_SoftmaxLayer(i) for i in range(46)]))


def qwen35_like():
    """40 layers, `[lin, lin, lin, full]` -- 30 recurrent, 10 softmax (Qwen3.5-35B-A3B)."""
    return _VLM(_Decoder([_HybridLayer(i, (i + 1) % 4 != 0) for i in range(40)]))


class _FakeSpec:
    pass


class _FakeGroup:
    def __init__(self, spec, layer_names):
        self.kv_cache_spec = spec
        self.layer_names = layer_names


class _FakeRunner:
    """The two attributes `_mamba_group_ids` / `_ensure_gdn` touch."""

    def __init__(self, model, groups):
        self.model = model
        self.kv_cache_config = types.SimpleNamespace(kv_cache_groups=groups)


def _fake_vllm_config(layer_modules):
    return types.SimpleNamespace(compilation_config=types.SimpleNamespace(
        static_forward_context=layer_modules))


class GdnDetectionTest(unittest.TestCase):
    def setUp(self):
        correct.reset_gdn_cache()
        self._orig = correct._ORIG_FORWARD_CORE
        self._stock = _GDN_CLASS._forward_core

    def tearDown(self):
        correct.reset_gdn_cache()
        correct._ORIG_FORWARD_CORE = self._orig
        _GDN_CLASS._forward_core = self._stock

    # -- decoder walk ------------------------------------------------------------------------
    def test_decoder_module_resolves_both_trees(self):
        self.assertEqual(len(correct._decoder_module(glm_like()).layers), 46)
        self.assertEqual(len(correct._decoder_module(qwen35_like()).layers), 40)
        # bare causal LM (`.model.layers`) and a bare decoder both resolve too
        dec = _Decoder([_SoftmaxLayer(0)])
        self.assertIs(correct._decoder_module(_CausalLM(dec)).layers, dec.layers)
        self.assertIs(correct._decoder_module(dec).layers, dec.layers)

    # -- detection ---------------------------------------------------------------------------
    def test_qwen35_like_tree_takes_the_gdn_path(self):
        model = qwen35_like()
        self.assertTrue(correct.has_gdn(model))
        self.assertEqual(len(correct.gdn_modules(model)), 30)
        self.assertTrue(correct.install_gdn_patch(model))
        self.assertIs(_GDN_CLASS._forward_core, correct._forward_core_patch)
        self.assertIs(correct._ORIG_FORWARD_CORE, self._stock)

    def test_glm_like_tree_takes_the_softmax_only_path(self):
        model = glm_like()
        self.assertFalse(correct.has_gdn(model))
        self.assertEqual(correct.gdn_modules(model), [])
        self.assertFalse(correct.install_gdn_patch(model))
        # the class was NOT patched and `_ORIG_FORWARD_CORE` was not captured
        self.assertIs(_GDN_CLASS._forward_core, self._stock)
        self.assertIsNone(correct._ORIG_FORWARD_CORE)

    def test_install_does_not_import_the_gdn_module(self):
        """`install()` runs as a vllm plugin, before a model exists: it must not pull the Qwen
        GDN class in (deliverable 4 of docs/memo/glm46v_port_plan.md).  Asserted on the CODE of
        the function (its docstring names the class on purpose)."""
        import ast
        import inspect
        import textwrap
        fn = ast.parse(textwrap.dedent(inspect.getsource(correct.install))).body[0]
        body = fn.body[1:] if isinstance(getattr(fn.body[0], "value", None), ast.Constant) \
            else fn.body
        code = "\n".join(ast.dump(st) for st in body)
        self.assertNotIn("QwenGatedDeltaNetAttention", code)
        self.assertNotIn("qwen_gdn_linear_attn", code)
        self.assertNotIn("mamba", code)

    # -- check_gdn_path ----------------------------------------------------------------------
    def test_check_gdn_path_is_empty_and_silent_without_gdn(self):
        ctx = {f"language_model.model.layers.{i}.self_attn.attn": _Attn() for i in range(46)}
        self.assertEqual(correct.check_gdn_path(_fake_vllm_config(ctx)), [])

    def test_check_gdn_path_lists_and_patches_with_gdn(self):
        ctx = {f"model.layers.{i}.linear_attn": _Gdn(f"model.layers.{i}.linear_attn")
               for i in range(30)}
        names = correct.check_gdn_path(_fake_vllm_config(ctx))
        self.assertEqual(len(names), 30)
        self.assertIs(_GDN_CLASS._forward_core, correct._forward_core_patch)

    def test_check_gdn_path_still_rejects_the_fused_decode_kernel(self):
        layer = _Gdn("model.layers.0.linear_attn")
        layer.enable_fused_gdn_decode = True
        with self.assertRaises(AssertionError) as cm:
            correct.check_gdn_path(_fake_vllm_config({"model.layers.0.linear_attn": layer}))
        self.assertIn("VLLM_GDN_DECODE_KERNEL", str(cm.exception))

    # -- mamba groups ------------------------------------------------------------------------
    def test_mamba_group_ids_and_blocks_without_mamba(self):
        # `_mamba_group_ids` filters by `isinstance(spec, MambaSpec)`; stub the interface module
        # so the import inside it resolves on a box with no vllm.
        mod = types.ModuleType("vllm.v1.kv_cache_interface")

        class MambaSpec:
            pass

        mod.MambaSpec = MambaSpec
        sys.modules.setdefault("vllm.v1.kv_cache_interface", mod)
        groups = [_FakeGroup(_FakeSpec(), ["a"]), _FakeGroup(_FakeSpec(), ["b"])]
        runner = _FakeRunner(glm_like(), groups)
        gids = correct._mamba_group_ids(runner)
        self.assertEqual(gids, [])
        self.assertEqual(correct._mamba_blocks(runner, "r0", gids), {})
        # and the same runner reports "no recurrent layers", which is what gates the assertions
        # in `_correct_sub` / `appcorr_rows_step`
        self.assertFalse(correct._ensure_gdn(runner))

    def test_detection_is_cached_per_process(self):
        model = glm_like()
        self.assertFalse(correct.install_gdn_patch(model))
        # a second call must not re-walk into a different answer, and must not import anything
        self.assertFalse(correct.install_gdn_patch(qwen35_like()))
        correct.reset_gdn_cache()
        self.assertTrue(correct.install_gdn_patch(qwen35_like()))


if __name__ == "__main__":
    unittest.main()
