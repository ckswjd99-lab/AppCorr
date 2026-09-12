"""GLM-5.3-Flash: the mHC layer walk and the KDA side buffer (CPU, fp32).

Two engine-side generalisations of `appcorr/vllm_stream/correct.py`, both tested here on fakes
because the real pieces are CUDA/Triton-only:

1. **The layer walk** now carries GLM-5.3's four mHC streams.  `Glm5NextDecoderLayer` is called
   `layer(positions, hidden, residual, post, comb)` and returns `(hidden, residual, post, comb)`
   (`$V/vllm/models/glm5next/nvidia/model.py:401-509`): layer 0 expands, the last layer
   materialises its deferred `hc_post` and contracts, and every layer in between DEFERS its
   `hc_post` into the next layer's fused pre.  A partial-depth walk therefore cannot carry two
   tensors.  The fake layer below mimics exactly that 5-in/4-out convention with the same
   deferral, and the test is the property the staged schedule needs: walking `[0, L)` in two
   stages through the frontier buffers is BITWISE one straight pass.

2. **The KDA side buffer** at `Glm5NextLinearAttention._forward(qkv_proj_states, g1, beta,
   core_attn_out)` (`kda.py:333`).  Captured per token per layer: merged q|k|v (pre-conv), the
   RAW pre-sigmoid b, and g1.  The re-scan's data prep (`_kda_split_qkv` / `_kda_raw_g` /
   `_kda_beta`) is pure torch and is exercised here against a pure-PyTorch KDA reference that
   follows `$V/tests/models/glm5next/test_kda_recurrent.py:25-51`.

   What is NOT claimed here: that `chunk_kda_with_fused_gate` agrees with that reference, or that
   `causal_conv1d_fn` agrees with `F.conv1d`.  Both are GPU gates --
   `analysis/experiments/glm53_kda_gate.py`.

    python -m pytest tests/test_correct_glm53_kda.py -q
"""
import inspect
import math
import sys
import types
import unittest

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- stub the vllm symbols correct.py needs at import time ------------------------------------ #

def _install_vllm_stubs():
    try:
        import vllm.v1.worker.gpu_model_runner            # noqa: F401
        return False
    except Exception:
        pass
    for name in ("vllm", "vllm.v1", "vllm.v1.worker", "vllm.model_executor",
                 "vllm.model_executor.layers", "vllm.model_executor.layers.mamba",
                 "vllm.model_executor.layers.mamba.gdn"):
        sys.modules.setdefault(name, types.ModuleType(name))
    runner_mod = types.ModuleType("vllm.v1.worker.gpu_model_runner")

    class GPUModelRunner:
        pass

    runner_mod.GPUModelRunner = GPUModelRunner
    sys.modules["vllm.v1.worker.gpu_model_runner"] = runner_mod

    gdn_mod = types.ModuleType("vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn")

    class QwenGatedDeltaNetAttention(nn.Module):
        def _forward_core(self, mixed_qkv, b, a, core_attn_out):
            return "stock"

    gdn_mod.QwenGatedDeltaNetAttention = QwenGatedDeltaNetAttention
    sys.modules[gdn_mod.__name__] = gdn_mod
    return True


_STUBBED = _install_vllm_stubs()

from appcorr.vllm_stream import correct  # noqa: E402

_QWEN_CLASS = sys.modules[
    "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"].QwenGatedDeltaNetAttention

H, D, TP = 4, 8, 1                 # tiny stand-ins for 64 heads x 128 (the real config)
P = H * D                          # per-projection width
WIDTH = 4                          # linear_conv_kernel_dim
LOWER_BOUND = -5.0


# =============================================================================================
# 1. mHC layer walk
# =============================================================================================

class _MhcLayer(nn.Module):
    """A stand-in for `Glm5NextDecoderLayer`: 5 in, 4 out, deferred `hc_post`.

    Not GLM's arithmetic (that is the served model's job) but GLM's DATA FLOW: four residual
    streams, an fp32 `post` [T, n] and an fp32 `comb` [T, n, n] carried between layers, layer 0
    expanding, the last layer materialising the deferred post and contracting.  Any walk that
    drops or reorders a stream produces different numbers.
    """

    def __init__(self, idx: int, n_layers: int, n: int = 4, d: int = 6):
        super().__init__()
        self.layer_idx, self.num_hidden_layers, self.n, self.d = idx, n_layers, n, d
        g = torch.Generator().manual_seed(1000 + idx)
        self.wa = nn.Parameter(torch.randn(d, d, generator=g) / math.sqrt(d))
        self.wm = nn.Parameter(torch.randn(d, d, generator=g) / math.sqrt(d))
        self.wp = nn.Parameter(torch.randn(d, n, generator=g) / math.sqrt(d))

    # --- the three mHC primitives -------------------------------------------------------- #
    def _pre(self, res):
        """[T, n, d] -> (post [T, n] fp32, comb [T, n, n] fp32, x [T, d])."""
        f = res.float()
        post = torch.tanh(f @ self.wp.float()).mean(1)                       # [T, n]
        comb = torch.softmax(f @ f.transpose(1, 2) / self.d, dim=-1)         # [T, n, n]
        x = (comb @ f).mean(1).to(res.dtype)                                 # [T, d]
        return post, comb, x

    def _post(self, x, res, post, comb):
        """The deferred `hc_post`: fold x back into the four streams."""
        return (comb @ res.float()).to(res.dtype) + post.unsqueeze(-1).to(res.dtype) \
            * x.unsqueeze(1)

    def _fused_post_pre(self, x, res, post, comb):
        res = self._post(x, res, post, comb)
        post, comb, y = self._pre(res)
        return res, post, comb, y

    # --- the layer ----------------------------------------------------------------------- #
    def forward(self, positions, hidden_states, residual=None, post=None, comb=None):
        x = hidden_states
        if post is None:
            if self.layer_idx == 0:
                x = x.unsqueeze(1).expand(-1, self.n, -1).contiguous()       # hc_expand
            residual = x
            post, comb, x = self._pre(x)
        else:
            residual, post, comb, x = self._fused_post_pre(x, residual, post, comb)
        x = torch.tanh(x @ self.wa) + 0.01 * positions.reshape(-1, 1).to(x.dtype)  # "attention"
        residual, post, comb, x = self._fused_post_pre(x, residual, post, comb)
        x = torch.tanh(x @ self.wm)                                          # "mlp"
        if self.layer_idx == self.num_hidden_layers - 1:
            x = self._post(x, residual, post, comb)
            x = x.mean(1)                                                    # hc_contract
            _MhcLayer.SINK.append(x)
            return x, None, None, None
        return x, residual, post, comb


_MhcLayer.SINK = []


class _Res2Layer(nn.Module):
    """A stand-in for a stock 2-tuple decoder layer (Qwen3.5 / GLM-4.6V)."""

    def __init__(self, idx: int, d: int = 6):
        super().__init__()
        g = torch.Generator().manual_seed(2000 + idx)
        self.w = nn.Parameter(torch.randn(d, d, generator=g) / math.sqrt(d))

    def forward(self, positions, hidden_states, residual=None):
        if residual is None:
            residual = hidden_states
        h = torch.tanh(hidden_states @ self.w) + residual
        return h, residual


class _Decoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)


class _CausalLM(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.model = decoder


class _VLM(nn.Module):
    """`.language_model` -> `.model` -> `.layers`, plus the full-depth forward `_run_layers`
    delegates to when `[a, b) == [0, L)`."""

    def __init__(self, decoder):
        super().__init__()
        self.language_model = _CausalLM(decoder)

    def forward(self, input_ids=None, positions=None, intermediate_tensors=None,
                inputs_embeds=None):
        layers = self.language_model.model.layers
        four = "post" in inspect.signature(type(layers[0]).forward).parameters
        if four:
            st = (inputs_embeds, None, None, None)
            for layer in layers:
                st = layer(positions, *st)
            return st[0]
        hs, res = inputs_embeds, None
        for layer in layers:
            hs, res = layer(positions, hs, res)
        return hs


class _FakeRunner:
    def __init__(self, model):
        self.model = model


def _mhc_runner(n_layers=4):
    return _FakeRunner(_VLM(_Decoder([_MhcLayer(i, n_layers) for i in range(n_layers)])))


class MhcWalkTest(unittest.TestCase):
    def setUp(self):
        _MhcLayer.SINK.clear()
        torch.manual_seed(0)
        self.T, self.d, self.L = 5, 6, 4
        self.pos = torch.arange(self.T)
        self.emb = torch.randn(self.T, self.d)
        self.runner = _mhc_runner(self.L)
        self.layers = self.runner.model.language_model.model.layers

    def _straight(self):
        """Reference: one pass over every layer, written out here rather than reused from
        `_run_layers` (so the test is not comparing the walk to itself)."""
        st = (self.emb, None, None, None)
        mid = []
        for layer in self.layers:
            st = layer(self.pos, *st)
            mid.append(st)
        return mid

    def test_walk_kind_detects_the_convention_by_signature(self):
        self.assertEqual(correct._walk_kind(self.layers[0]), "mhc4")
        self.assertEqual(correct._walk_kind(_Res2Layer(0)), "res2")

    def test_two_stage_walk_is_bitwise_one_straight_pass(self):
        mid = self._straight()
        ref_out = _MhcLayer.SINK[-1].clone()      # the contracted last-layer output
        _MhcLayer.SINK.clear()

        for split in (1, 2, 3):
            with self.subTest(split=split):
                _MhcLayer.SINK.clear()
                st = correct._run_layers(self.runner, (0, split), self.pos, self.emb, None)
                self.assertEqual(len(st), 4)
                for a, b in zip(st, mid[split - 1]):
                    self.assertTrue(torch.equal(a, b))
                out = correct._run_layers(self.runner, (split, self.L), self.pos, None, st)
                self.assertIsNone(out)            # a walk that ends at L returns no value
                self.assertEqual(len(_MhcLayer.SINK), 1)
                self.assertTrue(torch.equal(_MhcLayer.SINK[-1], ref_out))

    def test_three_stage_walk_is_bitwise_one_straight_pass(self):
        self._straight()
        ref_out = _MhcLayer.SINK[-1].clone()
        _MhcLayer.SINK.clear()
        st = correct._run_layers(self.runner, (0, 1), self.pos, self.emb, None)
        st = correct._run_layers(self.runner, (1, 3), self.pos, None, st)
        correct._run_layers(self.runner, (3, self.L), self.pos, None, st)
        self.assertTrue(torch.equal(_MhcLayer.SINK[-1], ref_out))

    def test_full_depth_goes_through_the_compiled_model(self):
        """`[0, L)` must still take the `self.model(...)` branch (the MVP path)."""
        out = correct._run_layers(self.runner, (0, self.L), self.pos, self.emb, None)
        self.assertTrue(torch.is_tensor(out))
        ref = self._straight()[-1][0]
        self.assertTrue(torch.equal(out, ref))

    def test_res2_walk_is_unchanged(self):
        runner = _FakeRunner(_VLM(_Decoder([_Res2Layer(i) for i in range(4)])))
        layers = runner.model.language_model.model.layers
        hs, res = self.emb, None
        for layer in layers[:3]:
            hs, res = layer(self.pos, hs, res)
        st = correct._run_layers(runner, (0, 2), self.pos, self.emb, None)
        self.assertEqual(len(st), 2)
        st = correct._run_layers(runner, (2, 3), self.pos, None, st)
        self.assertTrue(torch.equal(st[0], hs) and torch.equal(st[1], res))

    def test_frontier_buffers_round_trip_four_streams(self):
        sb = correct.SideBuffer(n=self.T, device=torch.device("cpu"))
        st = correct._run_layers(self.runner, (0, 2), self.pos, self.emb, None)
        rows = torch.tensor([1, 3])
        sb.store_frontier(rows, *(t[rows] for t in st))
        self.assertEqual(tuple(sb.fr_r.shape), (self.T, 4, self.d))
        self.assertEqual(tuple(sb.fr_post.shape), (self.T, 4))
        self.assertEqual(tuple(sb.fr_comb.shape), (self.T, 4, 4))
        back = sb.frontier_rows(rows)
        self.assertEqual(len(back), 4)
        for a, b in zip(back, st):
            self.assertTrue(torch.equal(a, b[rows]))
        # and the walk resumed from the buffers is the walk resumed from the tensors
        one = correct._run_layers(self.runner, (2, 3), self.pos[rows], None,
                                  tuple(t[rows] for t in st))
        two = correct._run_layers(self.runner, (2, 3), self.pos[rows], None, back)
        for a, b in zip(one, two):
            self.assertTrue(torch.equal(a, b))

    def test_frontier_buffers_still_round_trip_two_streams(self):
        sb = correct.SideBuffer(n=self.T, device=torch.device("cpu"))
        sb.store_frontier(torch.tensor([0, 2]), torch.ones(2, 3), torch.zeros(2, 3))
        self.assertIsNone(sb.fr_post)
        h, r = sb.frontier_rows(torch.tensor([0, 2]))
        self.assertTrue(torch.equal(h, torch.ones(2, 3)))
        self.assertTrue(torch.equal(r, torch.zeros(2, 3)))


# =============================================================================================
# 2. KDA: the fake layer, the reference, and the side buffer at the `_forward` seam
# =============================================================================================

class _FakeKda(_QWEN_CLASS.__mro__[1] if False else nn.Module):
    """`Glm5NextLinearAttention`'s surface, as `_kda_forward_patch` / `_rescan_kda_impl` use it.

    Deliberately NOT a `GatedDeltaNetAttention` subclass in name only: the flavour test below
    builds one that is, to check `_gdn_flavor`.
    """

    def __init__(self, prefix="model.layers.0.self_attn", n_blocks=2, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.prefix = prefix
        self.local_num_heads, self.head_dim = H, D
        self.local_projection_size = P
        self.conv_size = WIDTH
        self.kda_safe_gate, self.kda_lower_bound = True, LOWER_BOUND
        for name in ("q_conv1d", "k_conv1d", "v_conv1d"):
            m = nn.Module()
            m.weight = nn.Parameter(torch.randn(P, 1, WIDTH, generator=g))
            m.bias = None
            setattr(self, name, m)
        self._merged_conv_weight = None
        self.A_log = nn.Parameter(0.5 * torch.randn(1, 1, H, 1, generator=g))
        self.dt_bias = nn.Parameter(0.1 * torch.randn(H * D, generator=g))
        self.kv_cache = (torch.zeros(n_blocks, 3 * P, WIDTH - 1),
                         torch.zeros(n_blocks, H, D, D))
        self.calls = []

    def _forward(self, qkv_proj_states, g1, beta, core_attn_out):
        """Stand-in for the stock core: records the call and writes something deterministic."""
        self.calls.append((qkv_proj_states.clone(), g1.clone(), beta.clone()))
        core_attn_out[0, :qkv_proj_states.shape[0]] = qkv_proj_states[:, :P].reshape(
            -1, H, D).to(core_attn_out.dtype)
        return None


def naive_kda(q, k, v, raw_g, raw_beta, a_log, g_bias, state, lower_bound=LOWER_BOUND):
    """fp32 KDA reference for one sequence; `[T, H, D]` in, `[H, D, D]` (v-major) state.

    Mirrors `$V/tests/models/glm5next/test_kda_recurrent.py:25-51` exactly: the in-kernel gate
    `lower_bound * sigmoid(exp(A_log) * (g1 + dt_bias))`, `beta = sigmoid(b)`, q/k l2-normed and
    q scaled by `D**-0.5`, `S <- S*exp(gate) + beta*(v - S k) k^T`, `o = S q`.
    """
    q, k, v, raw_g, raw_beta = (x.float() for x in (q, k, v, raw_g, raw_beta))
    d = q.shape[-1]
    q = q / torch.sqrt(q.square().sum(-1, keepdim=True) + 1e-6) * d ** -0.5
    k = k / torch.sqrt(k.square().sum(-1, keepdim=True) + 1e-6)
    gate = lower_bound * torch.sigmoid(a_log.exp()[:, None] * (raw_g + g_bias))
    beta = torch.sigmoid(raw_beta)
    s = state.clone()
    out = torch.empty_like(v)
    for t in range(q.shape[0]):
        s = s * gate[t].exp()[:, None, :]
        u = beta[t][:, None] * (v[t] - torch.einsum("hvk,hk->hv", s, k[t]))
        s = s + u[:, :, None] * k[t][:, None, :]
        out[t] = torch.einsum("hvk,hk->hv", s, q[t])
    return out, s


def ref_causal_conv(x, weight, width=WIDTH):
    """Depthwise causal conv over [L, C] with a ZERO initial state, + silu -- the pure-torch
    stand-in for `causal_conv1d_fn` (CUDA-only).  Used to check the WINDOW arithmetic only."""
    xt = x.t().unsqueeze(0)                                    # [1, C, L]
    xt = F.pad(xt, (width - 1, 0))
    out = F.conv1d(xt, weight.unsqueeze(1), groups=x.shape[1])
    return F.silu(out.squeeze(0).t())


def _fill_side_buffer(sb, layer, n, seed=3):
    g = torch.Generator().manual_seed(seed)
    qkv = torch.randn(n, 3 * P, generator=g)
    b = torch.randn(n, H, generator=g)
    a = torch.randn(n, H * D, generator=g)
    sb.store(layer.prefix, qkv, b, a, 0)
    return qkv, b, a


class KdaSideBufferTest(unittest.TestCase):
    def setUp(self):
        correct.reset_gdn_cache()
        correct._ST.mode, correct._ST.ctx, correct._ST.captures = correct.MODE_NONE, None, []
        self._orig_kda = correct._ORIG_KDA_FORWARD

    def tearDown(self):
        correct._ST.mode, correct._ST.ctx, correct._ST.captures = correct.MODE_NONE, None, []
        for cls in list(correct._KDA_PATCHED):
            cls._forward = correct._ORIG_KDA_FORWARD
        correct._KDA_PATCHED.clear()
        correct._ORIG_KDA_FORWARD = self._orig_kda
        correct.reset_gdn_cache()

    # -- flavour detection ----------------------------------------------------------------- #
    def test_gdn_flavor_separates_the_two_seams(self):
        class _GdnBase(nn.Module):                       # the shared base defines neither
            pass

        class _Kda(_GdnBase):
            def _forward(self, qkv_proj_states, g1, beta, core_attn_out):
                return None

        class _Qwen(_GdnBase):
            def _forward_core(self, mixed_qkv, b, a, core_attn_out):
                return None

        self.assertEqual(correct._gdn_flavor(_Kda()), "kda")
        self.assertEqual(correct._gdn_flavor(_Qwen()), "qwen")
        with self.assertRaises(RuntimeError):
            correct._gdn_flavor(_GdnBase())

    def test_patch_replaces_the_kda_forward_method_on_the_class(self):
        class _Kda(nn.Module):
            def _forward(self, qkv_proj_states, g1, beta, core_attn_out):
                return "stock"

        stock = _Kda._forward
        mods = [_Kda(), _Kda()]
        self.assertTrue(correct._patch_gdn_class(mods))
        self.assertEqual(correct.gdn_flavor(), "kda")
        self.assertIs(_Kda._forward, correct._kda_forward_patch)
        self.assertIs(correct._ORIG_KDA_FORWARD, stock)
        # MODE_NONE is a pure pass-through
        self.assertEqual(mods[0]._forward(qkv_proj_states=None, g1=None, beta=None,
                                          core_attn_out=None), "stock")
        self.assertIs(_QWEN_CLASS._forward_core, _QWEN_CLASS._forward_core)

    def test_mixed_flavours_are_rejected(self):
        class _Kda(nn.Module):
            def _forward(self, *a):
                return None

        class _Qwen(nn.Module):
            def _forward_core(self, *a):
                return None

        with self.assertRaises(AssertionError):
            correct._patch_gdn_class([_Kda(), _Qwen()])

    # -- capture / correct at the seam ------------------------------------------------------ #
    def test_capture_stores_merged_qkv_raw_beta_and_g1(self):
        layer = _FakeKda()
        sb = correct.SideBuffer(n=6, device=torch.device("cpu"))
        correct._ORIG_KDA_FORWARD = type(layer)._forward
        T, pos0 = 4, 2
        qkv = torch.randn(T, 3 * P)
        g1 = torch.randn(1, T, H, D)
        beta = torch.randn(1, T, H)
        out = torch.zeros(1, T, H, D)
        correct._ST.mode = correct.MODE_CAPTURE
        correct._ST.captures = [correct._Capture(sb=sb, tok0=0, ntok=T, pos0=pos0)]
        correct._kda_forward_patch(layer, qkv, g1, beta, out)
        key = layer.prefix
        self.assertTrue(torch.equal(sb.qkv[key][pos0:pos0 + T], qkv))
        self.assertTrue(torch.equal(sb.b[key][pos0:pos0 + T], beta[0]))       # RAW, pre-sigmoid
        self.assertTrue(torch.equal(sb.a[key][pos0:pos0 + T], g1[0].reshape(T, -1)))
        self.assertEqual(len(layer.calls), 1)                                 # stock core ran
        self.assertEqual(sb.qkv[key].shape[1], 3 * P)

    def test_correct_mode_writes_the_rows_before_reading_and_replays(self):
        layer = _FakeKda()
        sb = correct.SideBuffer(n=8, device=torch.device("cpu"), capture_out=True)
        correct._ORIG_KDA_FORWARD = type(layer)._forward
        _fill_side_buffer(sb, layer, 8)
        sb.out[layer.prefix] = torch.randn(8, H, D)
        pos = torch.tensor([3, 5])
        new_qkv, new_g1, new_beta = (torch.randn(2, 3 * P), torch.randn(1, 2, H, D),
                                     torch.randn(1, 2, H))
        out = torch.zeros(1, 2, H, D)
        correct._ST.mode = correct.MODE_CORRECT
        correct._ST.ctx = correct._CorrectCtx(
            sb=sb, positions=pos, window=(0, 8), final=False, replay=True, mamba_blocks={})
        correct._kda_forward_patch(layer, new_qkv, new_g1, new_beta, out)
        key = layer.prefix
        self.assertTrue(torch.equal(sb.qkv[key][pos], new_qkv))
        self.assertTrue(torch.equal(sb.b[key][pos], new_beta[0]))
        self.assertTrue(torch.equal(sb.a[key][pos], new_g1[0].reshape(2, -1)))
        self.assertTrue(torch.equal(out[0], sb.out[key][pos]))

    # -- beta form -------------------------------------------------------------------------- #
    def test_beta_for_the_chunk_kernel_is_sigmoided_fp32(self):
        """`chunk_kda_with_fused_gate` takes beta ALREADY sigmoided, in fp32 (kda.py:546 does
        `_cast_sigmoid` at the stock call site; the kernel never sigmoids).  The recurrent kernel
        is the one that takes the raw bf16 value (`sigmoid_beta=True`).  The design memo has this
        inverted; the code wins."""
        layer = _FakeKda()
        sb = correct.SideBuffer(n=6, device=torch.device("cpu"))
        _, raw_b, _ = _fill_side_buffer(sb, layer, 6)
        beta = correct._kda_beta(sb, layer.prefix, 1, 5)
        self.assertIs(beta.dtype, torch.float32)
        self.assertEqual(tuple(beta.shape), (1, 4, H))
        self.assertTrue(torch.equal(beta[0], torch.sigmoid(raw_b[1:5].float())))
        self.assertTrue(bool((beta > 0).all() and (beta < 1).all()))
        self.assertFalse(torch.allclose(beta[0], raw_b[1:5]))   # not the raw value

    def test_side_buffer_slot_shapes_match_the_seam(self):
        layer = _FakeKda()
        sb = correct.SideBuffer(n=6, device=torch.device("cpu"))
        _fill_side_buffer(sb, layer, 6)
        q, k, v = correct._kda_split_qkv(layer, torch.randn(4, 3 * P))
        for t in (q, k, v):
            self.assertEqual(tuple(t.shape), (1, 4, H, D))
        self.assertEqual(tuple(correct._kda_raw_g(layer, sb, layer.prefix, 1, 5).shape),
                         (1, 4, H, D))

    def test_merged_conv_weight_is_qkv_concatenated(self):
        layer = _FakeKda()
        w = correct._kda_conv_weight(layer)
        self.assertEqual(tuple(w.shape), (3 * P, WIDTH))
        self.assertTrue(torch.equal(w[:P], layer.q_conv1d.weight.view(P, WIDTH)))
        self.assertTrue(torch.equal(w[P:2 * P], layer.k_conv1d.weight.view(P, WIDTH)))
        self.assertTrue(torch.equal(w[2 * P:], layer.v_conv1d.weight.view(P, WIDTH)))
        self.assertIs(correct._kda_conv_weight(layer), w)      # cached on the layer


# =============================================================================================
# 3. re-scan invariance and conv-window equivalence
# =============================================================================================

class KdaRescanMathTest(unittest.TestCase):
    """The two properties the re-scan design rests on, on the fp32 reference.

    They are what makes "round r re-scans [ckpt_end, window_end) from the checkpoint" equal to a
    single pass over the prompt, and what makes a windowed conv equal to the full conv.
    """

    def setUp(self):
        torch.manual_seed(7)
        self.layer = _FakeKda(seed=11)
        self.n = 24
        self.sb = correct.SideBuffer(n=self.n, device=torch.device("cpu"))
        self.qkv, self.b, self.a = _fill_side_buffer(self.sb, self.layer, self.n, seed=5)

    def _scan(self, start, end, state):
        """The reference re-scan of [start, end) from `state`, using the SAME data prep the
        engine's `_rescan_kda_impl` uses (`_kda_conv_weight`, the window arithmetic,
        `_kda_split_qkv`, `_kda_raw_g`, `_kda_beta`) with `ref_causal_conv` in place of the
        CUDA conv and `naive_kda` in place of the Triton chunk kernel."""
        layer, sb, key = self.layer, self.sb, self.layer.prefix
        width = layer.conv_size
        c0 = max(0, start - (width - 1))
        conv = ref_causal_conv(sb.qkv[key][c0:end], correct._kda_conv_weight(layer), width)
        conv = conv[start - c0:]
        q, k, v = correct._kda_split_qkv(layer, conv)
        raw_g = correct._kda_raw_g(layer, sb, key, start, end)
        beta = correct._kda_beta(sb, key, start, end)
        # naive_kda takes the RAW b (it sigmoids internally, as the recurrent kernel does), so
        # feed it the logit of the chunk kernel's beta -- i.e. exercise the same tensor.
        raw_beta = torch.log(beta[0] / (1 - beta[0]))
        return naive_kda(q[0], k[0], v[0], raw_g[0], raw_beta,
                         layer.A_log.reshape(H), layer.dt_bias.reshape(H, D), state)

    def test_rescan_from_a_checkpoint_equals_the_full_scan(self):
        s, e = 9, 19
        zero = torch.zeros(H, D, D)
        full_out, full_state = self._scan(0, self.n, zero)
        part_out, ckpt = self._scan(0, s, zero)
        win_out, win_state = self._scan(s, e, ckpt)
        err = (win_out - full_out[s:e]).abs().max().item()
        self.assertLess(err, 1e-4, f"window output max-abs {err}")
        serr = (win_state - self._scan(0, e, zero)[1]).abs().max().item()
        self.assertLess(serr, 1e-4, f"state max-abs {serr}")
        print(f"\n[rescan invariance] window [{s},{e}) out max-abs {err:.3e}, "
              f"state max-abs {serr:.3e}")
        # the prefix rows agree too (the conv is run over a shorter slice there, so this is
        # fp32-close, not bitwise: `torch.conv1d` picks a different blocking per length)
        perr = (part_out - full_out[:s]).abs().max().item()
        self.assertLess(perr, 1e-5, f"prefix max-abs {perr}")
        print(f"[rescan prefix] max-abs {perr:.3e}")

    def test_rescan_chain_over_three_rounds(self):
        zero = torch.zeros(H, D, D)
        full_out, _ = self._scan(0, self.n, zero)
        state, outs = zero, []
        for a, b in ((0, 7), (7, 15), (15, self.n)):
            o, state = self._scan(a, b, state)
            outs.append(o)
        err = (torch.cat(outs, 0) - full_out).abs().max().item()
        self.assertLess(err, 1e-4, f"3-round chain max-abs {err}")
        print(f"[rescan chain x3] max-abs {err:.3e}")

    def test_conv_window_with_a_three_row_left_pad_is_the_full_conv_slice(self):
        """Bitwise: `conv(SB[start-3:end])[3:]` == `conv(SB[0:N])[start:end]` for start >= 3,
        and the `start < 3` case is covered by the zero initial state."""
        w = correct._kda_conv_weight(self.layer)
        full = ref_causal_conv(self.sb.qkv[self.layer.prefix], w)
        for start, end in ((3, 9), (9, 19), (19, self.n), (0, 5), (1, 4)):
            c0 = max(0, start - (self.layer.conv_size - 1))
            win = ref_causal_conv(self.sb.qkv[self.layer.prefix][c0:end], w)[start - c0:]
            self.assertTrue(torch.equal(win, full[start:end]),
                            f"window [{start},{end}) differs from the full conv slice")

    def test_the_impl_uses_that_window_arithmetic(self):
        """Ties the property above to the engine code (the real conv is CUDA-only, so the
        kernel comparison itself is the GPU gate)."""
        src = inspect.getsource(correct._rescan_kda_impl)
        self.assertIn("c0 = max(0, start - (width - 1))", src)
        self.assertIn("conv_out[start - c0:]", src)
        self.assertIn("width = int(layer.conv_size)", src)
        self.assertIn("use_qk_l2norm_in_kernel=True", src)
        self.assertIn("safe_gate=layer.kda_safe_gate", src)
        self.assertIn("lower_bound=layer.kda_lower_bound", src)


if __name__ == "__main__":
    unittest.main()
