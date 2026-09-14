"""Interleaved k<1 LLM correction inside vLLM (engine side).

The streaming form (``request.py`` / ``scheduler.py`` / ``runner_patch.py``) pushes the prompt in
bands and prefills each band once.  The *interleaved* form pushes the WHOLE prompt at t=0 with the
image rows at their approximate value, lets the stock engine prefill it (the "approx pass"), and
then rewrites individual prompt rows in place as their corrected embeddings arrive:

  * softmax layers -- the corrected rows are re-run as |P| pseudo-sequences of length 1 whose
    ``seq_len`` is ``pos+1``; the stock kernels write their K/V into the request's own slots
    (write-before-read inside the round is what ``unified_kv_cache_update`` before the attention
    call gives us) and read every key at ``key_pos <= query_pos``;
  * recurrent layers (Qwen3.5 Gated DeltaNet, GLM-5.3-Flash KDA) cannot be rewritten in place
    (the recurrent state is a running product),
    so the layer's *pre-conv* inputs (``mixed_qkv``, ``b``, ``a``) are captured for every prompt row
    during the approx pass into a per-request side buffer, the corrected rows overwrite their side
    buffer entries, and the round re-scans the window ``[ckpt_end, window_end)`` from a checkpoint
    of the recurrent state.  Total re-scan work over all rounds is one pass over the prompt.

Semantics follow ``docs/memo/interleaved_correction_contract.md`` (round r corrects P_r only; rows
outside P_r keep their captured value; the checkpoint chain is what persists earlier rounds).

The whole second half -- side buffer, capture, re-scan, recurrent write-back -- is CONDITIONAL on
the served decoder actually having Gated DeltaNet layers (``has_gdn`` / ``install_gdn_patch``,
2026-09-12).  A pure-softmax decoder (GLM-4.6V: 46 x ``Glm4MoeDecoderLayer``, all GQA) leaves
``_SIDE``'s qkv/b/a dicts empty, has no mamba KV-cache group, never reaches
``_forward_core_patch`` (so the re-scan window is a no-op) and runs ``appcorr_rows_step`` as the
pseudo-sequence softmax path alone.  Detection walks ``_decoder_module(model).layers`` and looks
at layer classes -- never at the model name.

Depth staging (``appcorr_staged_correct``, 2026-09-10): round r corrects its rows over layers
``[0, b_r)`` only and a *frontier walk* then carries every image row -- corrected input or not --
through ``[b_r, b_{r+1})`` with the corrected rows' K/V and side-buffer rows visible as context
(ProgVFM §3.3, the HF axes' ``interleaved_forward``).  The served form keeps the stock full-depth
approx prefill (block allocation, hold-back mechanics) and re-walks on top of it, so its final
state is the staged schedule's while its GPU work is NOT (the closed form prices the ideal
schedule; ``flops_analytic.interleaved_cost`` with depth records).

The bounds normally come from ``stage_bounds(L, g)`` (equal layer counts).  ``stage_spec`` also
accepts them EXPLICITLY -- the unified vision+decoder axis (§7.12) splits one joint axis by cost,
so only its later rounds reach the decoder at all and their depths are not ``L(r+1)/g``.  Nothing
below distinguishes the two cases.

Design memo: ``docs/memo/vllm_interleaved_design.md`` (§2, §7.11, §7.12).  Deviations from it are
listed in ``install``'s docstring.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from types import SimpleNamespace
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

from vllm.v1.worker.gpu_model_runner import GPUModelRunner

# ---------------------------------------------------------------------------------------------
# module state
# ---------------------------------------------------------------------------------------------

_SIDE: dict[str, "SideBuffer"] = {}          # core request id -> side buffer

MODE_NONE, MODE_CAPTURE, MODE_CORRECT = None, "capture", "correct"


@dataclass
class _Capture:
    sb: "SideBuffer"
    tok0: int          # first row of this request inside the step's flat token batch
    ntok: int
    pos0: int          # prompt position of that first row


@dataclass
class _CorrectCtx:
    sb: "SideBuffer"
    positions: torch.Tensor       # int64 [P] device, sorted ascending
    window: tuple[int, int]
    final: bool
    replay: bool                  # G1: replay the captured GDN output instead of re-scanning
    mamba_blocks: dict            # GDN layer prefix -> its request's block id in that layer's group
    commit: bool = True           # advance the DeltaNet checkpoint (False for non-last sub-batches)
    commit_end: int = -1          # commit the checkpoint at this position (<0: the window end);
                                  # a frontier walk scans past it uncommitted (outputs only)
    debug: dict = field(default_factory=dict)   # per-layer diagnostics (filled at `final`)


# PIECEWISE CUDA-graph replay of the full-depth correct step (the launch-bound part: the
# projections / MoE / norms of |P| pseudo-sequences are replayed from the runner's captured
# graphs; attention, the KV write and the GDN core are splitting ops and run eagerly as before,
# so the side-buffer patch sees exactly what it sees in eager mode). |P| is padded to the
# nearest capture size with zero rows; the attention backends slice by `num_actual_tokens`,
# the KV write by the slot mapping's length and `core_attn_out` is zero-initialised, so the
# padded rows never touch the request's state. Env APPCORR_CORRECT_CUDAGRAPH=0 forces eager.
import os as _os
CUDAGRAPH = _os.environ.get("APPCORR_CORRECT_CUDAGRAPH", "1") == "1"   # gate g7 passed 2026-09-10 (graph == padded-eager bitwise on 4B + 35B)
# Diagnostic: pad |P| to the capture size exactly as the graph path does, but run eager
# (isolates the padded-M kernel selection from the graph replay itself). Gate g7 only.
PAD_EAGER = _os.environ.get("APPCORR_CORRECT_PAD_EAGER", "0") == "1"
# GLM-5.3 (sparse-MLA + KDA): let the full-depth correct step dispatch a PIECEWISE graph too.
# OFF by default -- the family has never been gated on the graph path.  Under this nightly's
# breakable cudagraphs (VLLM_USE_BREAKABLE_CUDAGRAPH defaults ON, compilation mode NONE) the
# `@eager_break_during_capture` ops -- KDA `_forward`, the sparse indexer, MLA attention -- are
# re-invoked eagerly on every replay, so the side-buffer / indexer hooks patched onto them run
# with the live `_ST` / indexer context exactly as in eager mode; capture itself happens only at
# warmup (MODE_NONE), never under a correcting request.  What differs from the Qwen graph path
# is only the positions buffer (1-D `positions`, no M-RoPE).  Gate before trusting: graph vs
# padded-eager (APPCORR_CORRECT_PAD_EAGER=1) judged against this model's own A-vs-A band --
# it is not repeatable run to run, so a bitwise gate is void here (2026-09-14).
GLM53_CUDAGRAPH = _os.environ.get("APPCORR_GLM53_CUDAGRAPH", "0") == "1"
# Fused hold-back (memo §7 item 14). The final round's correct step also computes the held-back
# row N-1 -- one more pseudo-sequence at the end of the batch, fed the request's stored
# embedding -- and commits the DeltaNet state past it, so the engine step that releases the
# hold-back has nothing left to compute: its forward is replaced by the stashed final hidden
# row of N-1 (`_model_forward` stub) and the step only samples. The correct step is deferred
# to that engine step's `_prepare_inputs` (the scheduler has allocated N-1's KV block by then;
# `client.correct` arms it, releases the hold-back and runs the step). Fusion is skipped -- the
# plain final step runs there and the stock decode computes N-1 -- when the engine step carries
# any other token (another request's prefill/decode), so the result never depends on the
# concurrency. Env APPCORR_FUSE_HOLDBACK=0 disables it (the correct step still runs deferred).
FUSE_HOLDBACK = _os.environ.get("APPCORR_FUSE_HOLDBACK", "1") == "1"
# A step whose rows are one contiguous position range [p0, p0+P) -- every frontier walk, and a
# keep=1 band's correction -- is submitted as ONE prefill sequence of P query tokens over the
# request's KV blocks (query_start_loc [0, P], seq_len p0+P, num_computed p0) instead of P
# one-token pseudo-sequences: the same causal semantics (row p reads rows <= p: its own batch
# through the kernel's causal mask, the earlier rows from the paged cache), on the prefill
# kernel instead of P decode queries. Same row values up to kernel arithmetic (gated served);
# the 3294-row walk of a V*Bench prompt took 132 ms as pseudo-sequences (2026-09-11).
# Env APPCORR_CONTIG_ROWS=0 keeps the pseudo-sequence form for every step.
CONTIG_ROWS = _os.environ.get("APPCORR_CONTIG_ROWS", "1") == "1"


class _State:
    mode = MODE_NONE
    captures: list[_Capture] = []
    ctx: Optional[_CorrectCtx] = None
    stub: Optional[dict] = None   # fused hold-back: batch token index -> final hidden row
    skip_forward = False          # this step's batch is skip-prefill prompt rows only: no forward


@dataclass
class _Pending:
    """A final correct step armed by `appcorr_arm_final`, run by `_prepare_inputs`."""
    positions: torch.Tensor
    inputs_embeds: torch.Tensor
    window: tuple
    stage: Optional[tuple]
    replay: bool
    t_arm: float


_PENDING: dict[str, _Pending] = {}
_FINAL_INFO: dict[str, dict] = {}


_ST = _State()

# Per-layer state diagnostics at `final` (norms + rel-L2 vs whatever the prefill left in the
# block). Costs one GPU sync per GDN layer, so it is off unless a gate asks for it (`g0`).
_DEBUG_STATE = False


def set_debug_state(on: bool) -> None:
    global _DEBUG_STATE
    _DEBUG_STATE = bool(on)


# ---------------------------------------------------------------------------------------------
# side buffer
# ---------------------------------------------------------------------------------------------


@dataclass
class SideBuffer:
    """Per-request store of every recurrent layer's pre-conv inputs for every prompt row.

    Keyed by the layer's ``prefix`` (its name in ``static_forward_context``).  Rows are prompt
    positions ``[0, n)``; row ``n-1`` is only filled when the hold-back is released.

    The three slots are the seam's inputs, and their meaning is per flavour (``_gdn_flavor``):

    ==========  ==================================  ==========================================
    slot        qwen (`_forward_core`)              kda (`_forward`, GLM-5.3)
    ==========  ==================================  ==========================================
    ``qkv``     ``mixed_qkv``  [n, 2*Dk + Dv]       ``qkv_proj_states`` [n, 3*P] (merged q|k|v)
    ``b``       ``b``          [n, Hv]              ``beta[0]``         [n, H]  (RAW, presigmoid)
    ``a``       ``a``          [n, Hv]              ``g1[0]``           [n, H*128]
    ==========  ==================================  ==========================================

    On a pure-softmax decoder the ``qkv``/``b``/``a``/``ckpt`` dicts stay empty (nothing captures
    into them) and the object degenerates to the request's bookkeeping: ``n``/``lo``/``hi``, the
    frontier buffers of the depth-staged walks, and ``committed_end``/``n_corrected``.
    ``nbytes()`` is then the frontier buffers alone.
    """
    n: int
    device: torch.device
    lo: int = 0                                     # first image row
    hi: int = 0                                     # one past the last image row (0: unknown)
    capture_out: bool = False                       # also store the GDN layer output (G1 replay)
    # depth staging: residual stream of every image row at the walk frontier, and where it is.
    # ``fr_h``/``fr_r`` are the 2-tuple contract (Qwen3.5 / GLM-4.6V: hidden + residual);
    # ``fr_post``/``fr_comb`` are the two extra mHC streams a GLM-5.3 walk carries (they stay
    # None on a 2-tuple decoder, and ``frontier_rows`` then returns a 2-tuple as it always did).
    fr_h: Optional[torch.Tensor] = None
    fr_r: Optional[torch.Tensor] = None
    fr_post: Optional[torch.Tensor] = None
    fr_comb: Optional[torch.Tensor] = None
    frontier: int = 0                               # layers [0, frontier) walked for [lo, hi)
    committed_end: int = 0                          # window end of the last correction
    n_corrected: int = 0
    stage_log: list = field(default_factory=list)
    qkv: dict[str, torch.Tensor] = field(default_factory=dict)
    b: dict[str, torch.Tensor] = field(default_factory=dict)
    a: dict[str, torch.Tensor] = field(default_factory=dict)
    out: dict[str, torch.Tensor] = field(default_factory=dict)
    ckpt: dict[str, torch.Tensor] = field(default_factory=dict)
    ckpt_end: dict[str, int] = field(default_factory=dict)
    conv_scratch: dict[str, torch.Tensor] = field(default_factory=dict)
    n_capture_steps: int = 0
    hold_hidden: Optional[torch.Tensor] = None      # fused hold-back: final hidden row of N-1
    # open walk (unified axis): the stock prefill is skipped (its scheduled steps run no model
    # forward for this request) and rows [walk_lo, hi) are walked through [0, open_walk) at open
    open_walk: int = 0
    skip_prefill: bool = False
    walk_lo: int = 0                                # first walked row: 0 when the prefill is skipped
    open_walk_info: Optional[dict] = None

    def _alloc(self, key: str, qkv: torch.Tensor, b: torch.Tensor, a: torch.Tensor) -> None:
        if key in self.qkv:
            return
        self.qkv[key] = torch.zeros((self.n, qkv.shape[1]), dtype=qkv.dtype, device=self.device)
        self.b[key] = torch.zeros((self.n, b.shape[1]), dtype=b.dtype, device=self.device)
        self.a[key] = torch.zeros((self.n, a.shape[1]), dtype=a.dtype, device=self.device)
        self.ckpt_end[key] = 0

    def store(self, key, qkv, b, a, pos0: int) -> None:
        self._alloc(key, qkv, b, a)
        t = qkv.shape[0]
        assert pos0 + t <= self.n, (pos0, t, self.n)
        self.qkv[key][pos0:pos0 + t] = qkv
        self.b[key][pos0:pos0 + t] = b
        self.a[key][pos0:pos0 + t] = a

    def store_out(self, key, out, pos0: int) -> None:
        if key not in self.out:
            self.out[key] = torch.zeros((self.n, *out.shape[1:]), dtype=out.dtype, device=self.device)
        t = out.shape[0]
        self.out[key][pos0:pos0 + t] = out

    def scatter(self, key, positions, qkv, b, a) -> None:
        self._alloc(key, qkv, b, a)
        self.qkv[key][positions] = qkv.to(self.qkv[key].dtype)
        self.b[key][positions] = b.to(self.b[key].dtype)
        self.a[key][positions] = a.to(self.a[key].dtype)

    def nbytes(self) -> int:
        tot = 0
        for d in (self.qkv, self.b, self.a, self.out):
            tot += sum(t.numel() * t.element_size() for t in d.values())
        for t in (self.fr_h, self.fr_r, self.fr_post, self.fr_comb):
            if t is not None:
                tot += t.numel() * t.element_size()
        return tot

    def frontier_rows(self, positions: torch.Tensor):
        """The walk state of ``positions`` at the frontier: 2 tensors, or 4 on an mHC decoder."""
        assert self.fr_h is not None, "frontier buffers not initialised"
        if self.fr_post is None:
            return self.fr_h[positions], self.fr_r[positions]
        return (self.fr_h[positions], self.fr_r[positions],
                self.fr_post[positions], self.fr_comb[positions])

    def store_frontier(self, positions: torch.Tensor, *state: Optional[torch.Tensor]) -> None:
        """Store the walk state of ``positions``: ``(hidden, residual)`` on a 2-tuple decoder,
        ``(hidden, residual, post, comb)`` on an mHC one.  A trailing ``None`` (the last mHC
        layer contracts and returns None for the three stream tensors) is rejected -- the
        frontier is only ever stored below full depth, where all four are live."""
        assert len(state) in (2, 4), len(state)
        assert all(t is not None for t in state), "frontier state carries a None"
        hs, res = state[0], state[1]
        if self.fr_h is None:
            self.fr_h = torch.zeros((self.n, *hs.shape[1:]), dtype=hs.dtype, device=self.device)
            self.fr_r = torch.zeros((self.n, *res.shape[1:]), dtype=res.dtype, device=self.device)
            if len(state) == 4:
                self.fr_post = torch.zeros((self.n, *state[2].shape[1:]),
                                           dtype=state[2].dtype, device=self.device)
                self.fr_comb = torch.zeros((self.n, *state[3].shape[1:]),
                                           dtype=state[3].dtype, device=self.device)
        assert (self.fr_post is not None) == (len(state) == 4), "walk arity changed mid-request"
        self.fr_h[positions] = hs.to(self.fr_h.dtype)
        self.fr_r[positions] = res.to(self.fr_r.dtype)
        if len(state) == 4:
            self.fr_post[positions] = state[2].to(self.fr_post.dtype)
            self.fr_comb[positions] = state[3].to(self.fr_comb.dtype)


# --------------------------------------------------------------------------------------------
# Gated-DeltaNet detection.  Everything above (the side buffer) and the ``_rescan`` machinery
# below exists for models whose decoder carries recurrent layers.  A pure-softmax decoder
# (GLM-4.6V: 46 x ``Glm4MoeDecoderLayer``, all GQA) has nothing to capture, nothing to re-scan
# and no recurrent block to write back, so the correct step is the pseudo-sequence softmax path
# alone -- and ``install()`` must not even import the Qwen GDN class.  Detection walks the served
# model's decoder layers (NOT the model name): a layer is recurrent iff any module under it has
# ``GatedDeltaNet`` in its class MRO.
# --------------------------------------------------------------------------------------------

_GDN_MRO_MARKER = "GatedDeltaNet"
_GDN: Optional[bool] = None                    # None = not determined yet for this process
_GDN_FLAVOR: Optional[str] = None              # "qwen" (GDN) | "kda" (GLM-5.3), with _GDN True


def _is_gdn_module(mod) -> bool:
    return any(_GDN_MRO_MARKER in c.__name__ for c in type(mod).__mro__)


# >>> AppCorr/K (GLM-5.3 KDA): which recurrent seam a layer exposes >>>
def _gdn_flavor(mod) -> str:
    """Which recurrent seam this module exposes -- by STRUCTURE, never by model name.

    Both flavours are ``GatedDeltaNetAttention`` subclasses (so ``_is_gdn_module`` finds both),
    and they are told apart by the method the core runs behind:

      * ``qwen``: ``QwenGatedDeltaNetAttention._forward_core(mixed_qkv, b, a, core_attn_out)``
        (qwen_gdn_linear_attn.py:1268) -- conv + ``fused_post_conv_prep`` + chunk_gated_delta_rule;
      * ``kda``:  ``Glm5NextLinearAttention._forward(qkv_proj_states, g1, beta, core_attn_out)``
        (glm5next/nvidia/kda.py:333) -- merged q|k|v conv + ``chunk_kda_with_fused_gate``.

    The shared base (``mamba/gdn/base.py:22``) defines NEITHER, so the two names are disjoint
    discriminators on the concrete class.
    """
    cls = type(mod)
    if hasattr(cls, "_forward_core"):
        return "qwen"
    if hasattr(cls, "_forward"):
        return "kda"
    raise RuntimeError(
        f"{cls.__name__}: a Gated-DeltaNet layer with neither a `_forward_core` (Qwen) nor a "
        "`_forward` (GLM-5.3 KDA) seam -- the side buffer has nothing to patch")
# <<< AppCorr/K <<<


def gdn_modules(model) -> list:
    """Every Gated-DeltaNet module under the served model's decoder layers, in layer order."""
    out = []
    for layer in _decoder_module(model).layers:
        for _, sub in layer.named_modules():
            if _is_gdn_module(sub):
                out.append(sub)
    return out


def has_gdn(model) -> bool:
    """True iff the served model's decoder has recurrent (Gated DeltaNet) layers."""
    return bool(gdn_modules(model))


def _patch_gdn_class(mods: list) -> bool:
    """Install the capture/correct patch for the (non-empty) recurrent modules ``mods``.

    Idempotent, and the answer is cached in ``_GDN`` -- one engine process serves one model.
    Deferred from ``install()`` (which runs as a vLLM plugin, before any model exists) to the
    first moment a model is in hand: ``check_gdn_path`` at ``open(correct=True)`` and, as a
    backstop, the first ``execute_model`` that carries a side buffer.  Both are before any
    capture, and the patch is a pure pass-through while ``_ST.mode is MODE_NONE``, so the
    deferral is invisible: a Qwen3.5 run behaves exactly as it did when the patch went on at
    import time.
    """
    global _GDN, _GDN_FLAVOR, _ORIG_FORWARD_CORE, _ORIG_KDA_FORWARD
    if not mods:
        if _GDN is None:
            _GDN = False
        return bool(_GDN)
    flavors = sorted({_gdn_flavor(m) for m in mods})
    assert len(flavors) == 1, f"mixed recurrent seams in one decoder: {flavors}"
    flavor = flavors[0]
    if flavor == "qwen":
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
            QwenGatedDeltaNetAttention)
        bad = [type(m).__name__ for m in mods if not isinstance(m, QwenGatedDeltaNetAttention)]
        assert not bad, (
            f"recurrent decoder layers of an unknown class {sorted(set(bad))}: the side-buffer "
            "capture patches QwenGatedDeltaNetAttention._forward_core only")
        if _ORIG_FORWARD_CORE is None:
            _ORIG_FORWARD_CORE = QwenGatedDeltaNetAttention._forward_core
            QwenGatedDeltaNetAttention._forward_core = _forward_core_patch
    else:
        cls = type(mods[0])
        bad = [type(m).__name__ for m in mods if type(m) is not cls]
        assert not bad, (
            f"recurrent decoder layers of several KDA classes {sorted(set(bad) | {cls.__name__})}")
        # patch the class METHOD by attribute: `forward` calls `self._forward(...)`, so the
        # bound lookup goes through the class and every instance follows.  The stock method is
        # `@eager_break_during_capture`-decorated; keeping the decorated original as the
        # pass-through target preserves the eager break in MODE_NONE.
        if _ORIG_KDA_FORWARD is None:
            _ORIG_KDA_FORWARD = cls._forward
            cls._forward = _kda_forward_patch
            _KDA_PATCHED.append(cls)
    _GDN, _GDN_FLAVOR = True, flavor
    return True


def install_gdn_patch(model) -> bool:
    """Patch ``_forward_core`` iff the served ``model`` has GDN layers; returns whether it does."""
    if _GDN is not None:
        return _GDN
    return _patch_gdn_class(gdn_modules(model))


def _ensure_gdn(self: GPUModelRunner) -> bool:
    """Does this runner's decoder have recurrent layers?  Resolved once, then cached."""
    return install_gdn_patch(self.model)


def reset_gdn_cache() -> None:
    """Forget the detection (tests only: one process serves one model in production)."""
    global _GDN, _GDN_FLAVOR
    _GDN, _GDN_FLAVOR = None, None


def gdn_flavor() -> Optional[str]:
    """"qwen" / "kda" / None -- which recurrent seam this process patched (diagnostics)."""
    return _GDN_FLAVOR


def check_gdn_path(vllm_config) -> list[str]:
    """The GDN layers, if any, must route through ``_forward_core`` (the method this patches).

    Qwen3.5 only: the GLM-5.3 KDA layer has no ``enable_fused_gdn_decode`` flag (its decode path
    is always ``fused_recurrent_kda``), so ``getattr(..., False)`` passes it through untouched.

    ``VLLM_GDN_DECODE_KERNEL`` defaults to ``cuda`` in vllm 0.28.0, which makes ``forward_cuda``
    call ``qwen_gdn_attention_core_fused_norm_packed`` -> ``_forward_core_fused_norm_packed``
    instead -- the capture and correct hooks would silently never fire.  (The design memo says
    this path "is not patched -- assert it is off"; it is ON by default, so every run of the
    interleaved path must set ``VLLM_GDN_DECODE_KERNEL=triton``.)

    Returns [] for a pure-softmax decoder -- which is NOT an error: GLM-4.6V's 46 layers are all
    softmax GQA, so there is no side-buffer machinery to check and `appcorr_rows_step` runs the
    pseudo-sequence path alone.  Detection here is by class MRO on the forward context's layer
    modules, so nothing GDN-specific is imported when the model has none.
    """
    ctx = vllm_config.compilation_config.static_forward_context
    gdn = [ln for ln, m in ctx.items() if _is_gdn_module(m)]
    if not gdn:
        return []
    for ln in gdn:
        assert not getattr(ctx[ln], "enable_fused_gdn_decode", False), (
            f"{ln}: VLLM_GDN_DECODE_KERNEL=cuda bypasses _forward_core "
            "(qwen_gdn_linear_attn.py:1781); run with VLLM_GDN_DECODE_KERNEL=triton")
    _patch_gdn_class([ctx[ln] for ln in gdn])
    return gdn


def open_buffer(req_id: str, n: int, device, *, lo: int = 0, hi: int = 0,
                capture_out: bool = False, open_walk: int = 0) -> SideBuffer:
    sb = SideBuffer(n=n, device=torch.device(device), lo=lo, hi=hi, capture_out=capture_out)
    if open_walk:
        assert not capture_out, "open_walk skips the prefill: nothing to capture for a replay"
        assert 0 < hi <= n - 1 and 0 <= lo < hi, (lo, hi, n)
        sb.open_walk, sb.skip_prefill, sb.walk_lo = int(open_walk), True, 0
    else:
        sb.walk_lo = lo
    _SIDE[req_id] = sb
    return sb


def get_buffer(req_id: str) -> Optional[SideBuffer]:
    return _SIDE.get(req_id)


def free(req_id: str) -> None:
    _SIDE.pop(req_id, None)
    _PENDING.pop(req_id, None)
    _FINAL_INFO.pop(req_id, None)


def active() -> bool:
    return bool(_SIDE)


# ---------------------------------------------------------------------------------------------
# GDN patch
# ---------------------------------------------------------------------------------------------

_ORIG_FORWARD_CORE = None       # QwenGatedDeltaNetAttention._forward_core
_ORIG_KDA_FORWARD = None        # Glm5NextLinearAttention._forward
_KDA_PATCHED: list = []         # the classes whose `_forward` we replaced (tests restore them)


_WIN_CONSTS: "OrderedDict[tuple, dict]" = OrderedDict()
_WIN_CONSTS_MAX = 4096


def _win_consts(dev: torch.device, conv_len: int, scan_len: int) -> dict:
    """Per-window device constants for ``_conv_window`` / ``_rescan_impl``, built once per
    (device, conv window length, scan window length) and reused by every layer and every later
    step with the same window.

    Why: the first profile of the correct step (35B, N=2379, P=316, 30 GDN layers) showed the
    layer loop CPU-bound -- ~0.94 ms of host time per rescan, 7 ``cudaStreamSynchronize`` per
    layer, all from tensors rebuilt on every call: pageable ``.to(dev)`` of ``cu``/chunk index
    tensors, ``torch.tensor(..., device=dev)`` for the conv metadata, and the stock
    ``causal_conv1d_fn`` ``metadata=None`` branch doing ``query_start_loc.diff().to("cpu")`` plus
    two 1024-element ``torch.full`` allocations.  With a prebuilt conv metadata object (the same
    ``compute_causal_conv1d_metadata`` the stock prefill builder uses) and cached index tensors the
    rescan issues no host sync at all.
    """
    key = (str(dev), int(conv_len), int(scan_len))
    c = _WIN_CONSTS.get(key)
    if c is not None:
        _WIN_CONSTS.move_to_end(key)
        return c
    from vllm.third_party.flash_linear_attention.ops.index import (
        prepare_chunk_indices, prepare_chunk_offsets)
    from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE
    from vllm.v1.attention.backends.utils import compute_causal_conv1d_metadata

    qsl_cpu = torch.tensor([0, conv_len], dtype=torch.int32)
    nums_dict, batch_ptr, tco_ptr = compute_causal_conv1d_metadata(qsl_cpu, device=dev)
    cu_cpu = torch.tensor([0, scan_len], dtype=torch.int32)
    c = dict(
        has_initial_state=torch.zeros(1, dtype=torch.bool, device=dev),
        cache_indices=torch.ones(1, dtype=torch.int32, device=dev),   # block 0 is the null block
        query_start_loc=qsl_cpu.to(dev),
        conv_meta=SimpleNamespace(nums_dict=nums_dict, batch_ptr=batch_ptr,
                                  token_chunk_offset_ptr=tco_ptr),
        cu=cu_cpu.to(dev),
        chunk_indices=prepare_chunk_indices(cu_cpu, FLA_CHUNK_SIZE).to(dev),
        chunk_offsets=prepare_chunk_offsets(cu_cpu, FLA_CHUNK_SIZE).to(dev),
    )
    # the pinned->device copies inside compute_causal_conv1d_metadata are non_blocking; make the
    # constants safe to reuse from any later stream before they are cached
    torch.cuda.synchronize(dev)
    _WIN_CONSTS[key] = c
    while len(_WIN_CONSTS) > _WIN_CONSTS_MAX:
        _WIN_CONSTS.popitem(last=False)
    return c


def _conv_window(layer, x: torch.Tensor, scratch: torch.Tensor, *,
                 weight: Optional[torch.Tensor] = None, bias=None,
                 activation=None) -> torch.Tensor:
    """Stock ``causal_conv1d_fn`` over a [L, C] slice with a zero initial state.

    ``weight``/``bias``/``activation`` default to the Qwen3.5 layer's single ``conv1d`` (the
    path this function has always run); the GLM-5.3 KDA layer passes its MERGED q|k|v weight
    (``_kda_conv_weight``), ``q_conv1d.bias`` and ``"silu"`` instead -- same kernel, same
    scratch layout, one conv over the concatenated channels (kda.py:388-399).

    ``scratch`` is a (2, *conv_state_shape) buffer shaped like two blocks of the layer's real conv
    cache; the kernel leaves the trailing ``width-1`` inputs in row 1, in exactly the layout the
    decode path (``causal_conv1d_update``) expects, which is what the ``final`` write-back copies.

    Row 1, not row 0: ``NULL_BLOCK_ID == 0`` (v1/attention/backends/utils.py:46) and the kernel
    treats a null cache index as "no state" -- it skips the state write AND leaves the conv output
    for that sequence in the uninitialised ``torch.empty_like`` buffer
    (causal_conv1d.py:144 / :490-491).

    All index/metadata tensors come from ``_win_consts`` (no per-call host syncs, see there).
    """
    from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
    from vllm.model_executor.layers.mamba.ops.causal_conv1d import causal_conv1d_fn

    L = x.shape[0]
    wc = _win_consts(x.device, L, L)
    scratch.zero_()
    cs = scratch if is_conv_state_dim_first() else scratch.transpose(-1, -2)
    if weight is None:
        weight = layer.conv1d.weight.view(
            layer.conv1d.weight.size(0), layer.conv1d.weight.size(2)
        )
        bias, activation = layer.conv1d.bias, layer.activation
    out = causal_conv1d_fn(
        x.transpose(0, 1),
        weight,
        bias,
        activation=activation,
        conv_states=cs,
        has_initial_state=wc["has_initial_state"],
        cache_indices=wc["cache_indices"],
        query_start_loc=wc["query_start_loc"],
        metadata=wc["conv_meta"],
        validate_data=False,
    )
    return out.transpose(0, 1)


def _rescan(layer, sb: SideBuffer, key: str, start: int, end: int, *,
            commit: bool = True) -> torch.Tensor:
    """Re-run conv + chunked delta rule over prompt rows ``[start, end)`` from ``sb.ckpt[key]``.
    ``commit=False`` leaves the checkpoint where it was (a round split into sub-batches re-scans
    the same window once per sub-batch; only the last one advances it).

    Mirrors the stock prefill branch of the layer's own core:

      * ``qwen``: ``_forward_core`` (qwen_gdn_linear_attn.py:1345-1520) -- causal conv with the
        layer's activation -> ``fused_post_conv_prep`` -> ``chunk_gated_delta_rule`` with
        ``use_qk_l2norm_in_kernel=False``.  Returns [end-start, HV, V].
      * ``kda``: ``_forward`` (kda.py:433-546) -- merged q|k|v causal conv (silu) ->
        ``chunk_kda_with_fused_gate`` with ``use_qk_l2norm_in_kernel=True`` and the gate computed
        in-kernel from ``A_log``/``dt_bias``.  Returns [end-start, H, D].
    """
    with torch.profiler.record_function("appcorr.rescan"):
        if _gdn_flavor(layer) == "kda":
            return _rescan_kda_impl(layer, sb, key, start, end, commit=commit)
        return _rescan_impl(layer, sb, key, start, end, commit=commit)


def _rescan_impl(layer, sb: SideBuffer, key: str, start: int, end: int, *, commit: bool):
    from vllm.third_party.flash_linear_attention.ops import fused_post_conv_prep

    dev = sb.device
    ssm_state = layer.kv_cache[1]
    width = layer.conv1d.weight.size(2)
    c0 = max(0, start - (width - 1))
    conv_out = _conv_window(layer, sb.qkv[key][c0:end].contiguous(), sb.conv_scratch[key])
    conv_out = conv_out[start - c0:].contiguous()

    q, k, v, g, beta = fused_post_conv_prep(
        conv_output=conv_out,
        a=sb.a[key][start:end].contiguous(),
        b=sb.b[key][start:end].contiguous(),
        A_log=layer.A_log,
        dt_bias=layer.dt_bias,
        num_k_heads=layer.num_k_heads // layer.tp_size,
        head_k_dim=layer.head_k_dim,
        head_v_dim=layer.head_v_dim,
        apply_l2norm=True,
        output_g_exp=False,
    )
    q, k, v = q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0)
    g, beta = g.unsqueeze(0), beta.unsqueeze(0)

    L = end - start
    wc = _win_consts(dev, end - c0, L)
    cu, chunk_indices, chunk_offsets = wc["cu"], wc["chunk_indices"], wc["chunk_offsets"]

    init = sb.ckpt.get(key)
    if init is None:
        init = torch.zeros_like(ssm_state[:1])
    out, last = layer.chunk_gated_delta_rule(
        q=q, k=k, v=v, g=g, beta=beta,
        initial_state=init,
        output_final_state=True,
        cu_seqlens=cu,
        chunk_indices=chunk_indices,
        chunk_offsets=chunk_offsets,
        use_qk_l2norm_in_kernel=False,
    )
    if commit:
        sb.ckpt[key] = last.to(ssm_state.dtype)
        sb.ckpt_end[key] = end
    return out.squeeze(0)


# >>> AppCorr/K (GLM-5.3 KDA side buffer: the `_forward` seam, the re-scan, the write-back) >>>
# ---------------------------------------------------------------------------------------------
# GLM-5.3-Flash KDA (`Glm5NextLinearAttention`): the same side buffer at a different seam
#
# Seam: `_forward(qkv_proj_states, g1, beta, core_attn_out)` (kda.py:333), the eager break the
# stock `forward` calls after its merged qkvbfg_a GEMM and the two gate projections.  What the
# seam hands us, per token:
#
#   qkv_proj_states  [T, 3*P]      merged q|k|v, PRE-conv, bf16   (P = 64*128/tp)
#   g1               [1, T, H, 128] f_b_proj(f_a), bf16           (H = 64/tp)
#   beta             [1, T, H]      RAW b, pre-sigmoid, bf16
#
# so the SideBuffer's three slots carry `qkv <- qkv_proj_states`, `b <- beta[0]` and
# `a <- g1[0]` flattened to [T, H*128].
#
# g1, not f_a: the design memo's cheaper option (store the replicated 128-wide `f_a` and
# recompute `g1 = f_b_proj(f_a)` in the re-scan) is NOT available at this seam -- `_forward`
# never sees f_a/g_a, they die inside `forward`.  Reaching them would mean patching `forward`
# itself (reimplementing the merged GEMM + o_norm + o_proj, which is inside the piecewise
# compiled region) or hanging a module hook on `f_b_proj` (inside that region too).  Both trade
# a ~24% side-buffer saving for the one property that makes this seam safe: it is an explicit
# eager break, so what we see is what the kernels get.  `g_a`/`g2` are not stored at all and are
# not needed: `o_norm(core_attn_out, g2)` is applied by the stock `forward` AFTER `_forward`
# returns, for exactly the rows in the correct batch, from their corrected hidden states.
#
# Per token per KDA layer (bf16 side buffer, 64 heads x 128, conv dim 3*8192 = 24576):
#   TP=1: qkv 24576*2 = 49152 B + b 64*2 = 128 B + g1 8192*2 = 16384 B  =  65664 B (64.1 KiB)
#   TP=2: qkv 12288*2 = 24576 B + b 32*2 =  64 B + g1 4096*2 =  8192 B  =  32832 B (32.1 KiB)
# x 34 KDA layers: 2.13 MiB/token at TP=1, 1.06 MiB/token per rank at TP=2 (a 2048-row prompt
# is 4.3 GiB / 2.2 GiB per rank).  Storing f_a+g_a instead of g1 would be 48.6 / 24.6 KiB.
#
# Under TP everything at this seam is ALREADY rank-local (in_proj q/k/v/b are column-sharded,
# f_b_proj is ColumnParallel so g1 carries the rank's head slice), the recurrent state is
# head-sharded (`is_kv_cache_tp_replicated=False`, abstract.py:59) and `o_proj`'s all-reduce
# happens after the seam.  So the re-scan is rank-local and needs no index remapping: each rank
# captures and re-scans its own heads.  f_a/g_a are the replicated parts and we store neither.
# ---------------------------------------------------------------------------------------------


def _kda_conv_weight(layer) -> torch.Tensor:
    """The layer's merged q|k|v conv weight [3*P, width], built exactly as the stock
    `_forward` builds it (kda.py:388-399) and cached on the layer for both to use."""
    w = getattr(layer, "_merged_conv_weight", None)
    if w is None:
        def _w(m):
            return m.weight.view(m.weight.size(0), m.weight.size(2))
        w = torch.cat([_w(layer.q_conv1d), _w(layer.k_conv1d), _w(layer.v_conv1d)],
                      dim=0).contiguous()
        layer._merged_conv_weight = w
    return w


def _kda_split_qkv(layer, conv_out: torch.Tensor):
    """Post-conv [L, 3*P] -> q, k, v as [1, L, H, D] (kda.py:`_rearr`)."""
    q, k, v = conv_out.split(layer.local_projection_size, dim=-1)
    h, d = layer.local_num_heads, layer.head_dim
    return (q.reshape(1, -1, h, d), k.reshape(1, -1, h, d), v.reshape(1, -1, h, d))


def _kda_raw_g(layer, sb: SideBuffer, key: str, start: int, end: int) -> torch.Tensor:
    """The captured g1 rows as the kernel's ``raw_g`` [1, L, H, D]."""
    return sb.a[key][start:end].reshape(1, end - start, layer.local_num_heads, layer.head_dim)


def _kda_beta(sb: SideBuffer, key: str, start: int, end: int) -> torch.Tensor:
    """The captured RAW b rows as the chunk kernel's ``beta`` [1, L, H] fp32.

    WHICH FORM DOES WHICH KERNEL TAKE (read off the code, not the memo):

      * ``chunk_kda_with_fused_gate`` takes beta ALREADY SIGMOIDED, in fp32.  Its own
        docstring-free body passes ``beta`` straight through ``chunk_kda_with_fused_gate_fwd`` ->
        ``_chunk_kda_fwd_with_cumulative_g`` (kernels.py:1199-1228, 1119-1162), which never
        sigmoids; and the stock prefill call site does the sigmoid itself:
        ``beta=_cast_sigmoid(beta_ns.squeeze(0)).unsqueeze(0)`` with
        ``_cast_sigmoid(x) = x.float().sigmoid()`` (kda.py:546, :121).
      * ``fused_recurrent_kda`` (decode/spec) takes the RAW bf16 b and sigmoids it in-kernel
        (``sigmoid_beta=True``, kda.py:504/571).

    The design memo (`glm53_correct_design.md`, item 2) says the opposite -- "pass the PRE-sigmoid
    fp32 value the chunk kernel expects; never the sigmoided bf16 the recurrent kernel takes".
    The code wins: the chunk kernel gets sigmoid(b) as fp32, the recurrent kernel gets raw b.
    """
    beta = sb.b[key][start:end].float().sigmoid().unsqueeze(0)
    assert beta.dtype is torch.float32 and beta.dim() == 3, (beta.dtype, beta.shape)
    return beta


def _rescan_kda_impl(layer, sb: SideBuffer, key: str, start: int, end: int, *, commit: bool):
    from vllm.models.glm5next.nvidia.ops.third_party.kda import chunk_kda_with_fused_gate

    dev = sb.device
    recurrent_state = layer.kv_cache[1]
    width = int(layer.conv_size)
    c0 = max(0, start - (width - 1))
    conv_out = _conv_window(layer, sb.qkv[key][c0:end].contiguous(), sb.conv_scratch[key],
                            weight=_kda_conv_weight(layer), bias=layer.q_conv1d.bias,
                            activation="silu")
    conv_out = conv_out[start - c0:].contiguous()

    q, k, v = _kda_split_qkv(layer, conv_out)
    raw_g = _kda_raw_g(layer, sb, key, start, end)
    beta = _kda_beta(sb, key, start, end)

    L = end - start
    wc = _win_consts(dev, end - c0, L)
    init = sb.ckpt.get(key)
    if init is None:
        init = torch.zeros_like(recurrent_state[:1])
    # NB `chunk_kda_with_fused_gate` derives chunk_indices from cu_seqlens itself
    # (kernels.py:1135) -- unlike the Qwen chunk kernel it takes no precomputed index tensors,
    # so `_win_consts`' chunk_indices/chunk_offsets are unused on this path.
    out, last = chunk_kda_with_fused_gate(
        q=q, k=k, v=v,
        raw_g=raw_g,
        beta=beta,
        A_log=layer.A_log,
        g_bias=layer.dt_bias,
        initial_state=init,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=wc["cu"],
        safe_gate=layer.kda_safe_gate,
        lower_bound=layer.kda_lower_bound,
    )
    if commit:
        sb.ckpt[key] = last.to(recurrent_state.dtype)
        sb.ckpt_end[key] = end
    return out.squeeze(0)


def _kda_write_back(layer, sb: SideBuffer, key: str, blk: int) -> None:
    """Final round: publish the re-scan's recurrent state and the merged conv tail into the
    request's own state block, so the decode that follows continues from CORRECTED state.

    * recurrent: ``scatter_states(recurrent_state, ckpt, [blk])`` -- the write-side counterpart
      of the ``gather_initial_states`` the stock prefill reads with (kda.py:533/557).
    * conv: the kernel already left the trailing ``width-1`` pre-conv inputs of the window --
      i.e. ``SB[end-3:end]``, rows N-4..N-2 for a window ending at the hold-back row -- in row 1
      of the conv scratch, in the layout ``causal_conv1d_update`` expects (DS: [C, width-1];
      SD: transposed, which is how the scratch was allocated, so the copy is layout-agnostic).
    """
    conv_state, state = layer.kv_cache[0], layer.kv_cache[1]
    ckpt = sb.ckpt[key].to(state.dtype)
    tail = sb.conv_scratch[key][1:2]
    if state.is_cuda:
        from vllm.model_executor.layers.mamba.ops.scatter_states import scatter_states
        idx = torch.tensor([blk], dtype=torch.int32, device=state.device)
        scatter_states(state, ckpt, idx)
        scatter_states(conv_state, tail.to(conv_state.dtype), idx)
    else:                                   # CPU tests: the triton kernel is CUDA-only
        state[blk] = ckpt[0]
        conv_state[blk] = tail[0].to(conv_state.dtype)


def _kda_forward_patch(self, qkv_proj_states, g1, beta, core_attn_out):
    """``Glm5NextLinearAttention._forward`` with the side buffer spliced in.

    MODE_NONE is a pure pass-through to the stock (decorated) method, so a served model that
    never opens a correcting request behaves exactly as it did unpatched -- including the
    `@eager_break_during_capture` behaviour, which lives in the ORIGINAL callable we delegate to
    (breakable_cudagraph.py:96-119: outside a capture it just calls through; inside one it
    DEFERS the call via `add_eager`).  That deferral is why capture/correct must never run under
    an active breakable capture: `store_out` reads `core_attn_out` right after the stock call,
    which a deferred call would not have written yet.  They cannot -- a capture only runs during
    the runner's warmup, before any request exists -- but if `VLLM_USE_BREAKABLE_CUDAGRAPH`
    ever starts capturing mid-serving, this is the line that breaks.
    """
    mode = _ST.mode
    if mode is MODE_NONE:
        return _ORIG_KDA_FORWARD(self, qkv_proj_states=qkv_proj_states, g1=g1, beta=beta,
                                 core_attn_out=core_attn_out)

    key = self.prefix
    T = g1.shape[1]
    qkv2, b2, a2 = qkv_proj_states, beta[0], g1[0].reshape(T, -1)

    if mode == MODE_CAPTURE:
        for cap in _ST.captures:
            s = slice(cap.tok0, cap.tok0 + cap.ntok)
            cap.sb.store(key, qkv2[s], b2[s], a2[s], cap.pos0)
        ret = _ORIG_KDA_FORWARD(self, qkv_proj_states=qkv_proj_states, g1=g1, beta=beta,
                                core_attn_out=core_attn_out)
        for cap in _ST.captures:
            if cap.sb.capture_out:
                s = slice(cap.tok0, cap.tok0 + cap.ntok)
                cap.sb.store_out(key, core_attn_out[0][s], cap.pos0)
        return ret

    assert mode == MODE_CORRECT
    ctx = _ST.ctx
    sb, pos = ctx.sb, ctx.positions
    P = pos.numel()
    # (1) write first: the corrected rows replace their captured pre-conv / pre-gate inputs
    sb.scatter(key, pos, qkv2[:P], b2[:P], a2[:P])

    if ctx.replay:
        core_attn_out[0, :P] = sb.out[key][pos].to(core_attn_out.dtype)
        return None

    if key not in sb.conv_scratch:
        sb.conv_scratch[key] = torch.zeros_like(self.kv_cache[0][:2])

    s, e = ctx.window
    start = sb.ckpt_end.get(key, 0)
    assert start <= s, f"{key}: checkpoint {start} is past the window start {s}"
    ce = e if ctx.commit_end < 0 else int(ctx.commit_end)
    if not ctx.commit or ce >= e:
        out_w = _rescan(self, sb, key, start, e, commit=ctx.commit)
    else:
        assert start <= ce, (key, start, ce)
        parts = []
        if ce > start:
            parts.append(_rescan(self, sb, key, start, ce, commit=True))
        else:
            sb.ckpt_end[key] = start
        parts.append(_rescan(self, sb, key, ce, e, commit=False))
        out_w = torch.cat(parts, 0) if len(parts) > 1 else parts[0]
    core_attn_out[0, :P] = out_w[pos - start].to(core_attn_out.dtype)

    if ctx.final:
        blk = ctx.mamba_blocks[key]
        if _DEBUG_STATE:
            ssm_stock, conv_stock = self.kv_cache[1][blk], self.kv_cache[0][blk]
            qkv = sb.qkv[key]
            ctx.debug[key] = {
                "ssm_rel": float((sb.ckpt[key][0].float() - ssm_stock.float()).norm()
                                 / max(ssm_stock.float().norm().item(), 1e-30)),
                "conv_rel": float((sb.conv_scratch[key][1].float() - conv_stock.float()).norm()
                                  / max(conv_stock.float().norm().item(), 1e-30)),
                "ssm_stock_norm": float(ssm_stock.float().norm()),
                "ssm_ours_norm": float(sb.ckpt[key].float().norm()),
                "qkv_zero_rows": int((qkv.float().abs().sum(1) == 0).sum()),
                "qkv_norm": float(qkv.float().norm()),
            }
        _kda_write_back(self, sb, key, blk)
    return None


# <<< AppCorr/K <<<


def _forward_core_patch(self, mixed_qkv, b, a, core_attn_out):
    mode = _ST.mode
    if mode is MODE_NONE:
        return _ORIG_FORWARD_CORE(self, mixed_qkv, b, a, core_attn_out)

    key = self.prefix

    if mode == MODE_CAPTURE:
        for cap in _ST.captures:
            s = slice(cap.tok0, cap.tok0 + cap.ntok)
            cap.sb.store(key, mixed_qkv[s], b[s], a[s], cap.pos0)
        ret = _ORIG_FORWARD_CORE(self, mixed_qkv, b, a, core_attn_out)
        for cap in _ST.captures:
            if cap.sb.capture_out:
                s = slice(cap.tok0, cap.tok0 + cap.ntok)
                cap.sb.store_out(key, core_attn_out[s], cap.pos0)
        return ret

    assert mode == MODE_CORRECT
    ctx = _ST.ctx
    sb, pos = ctx.sb, ctx.positions
    P = pos.numel()
    # (1) write first: the corrected rows replace their captured pre-conv inputs
    sb.scatter(key, pos, mixed_qkv[:P], b[:P], a[:P])

    if ctx.replay:
        core_attn_out[:P] = sb.out[key][pos].to(core_attn_out.dtype)
        return None

    if key not in sb.conv_scratch:
        sb.conv_scratch[key] = torch.zeros_like(self.kv_cache[0][:2])

    s, e = ctx.window
    start = sb.ckpt_end.get(key, 0)
    assert start <= s, f"{key}: checkpoint {start} is past the window start {s}"
    ce = e if ctx.commit_end < 0 else int(ctx.commit_end)
    if not ctx.commit or ce >= e:
        out_w = _rescan(self, sb, key, start, e, commit=ctx.commit)
    else:
        # frontier walk: the checkpoint moves to `ce` (the last corrected window's end); the rows
        # beyond it are scanned for their outputs only, so the next correction's window
        # `[ce, e')` re-scans them from a checkpoint that already carries every correction
        assert start <= ce, (key, start, ce)
        parts = []
        if ce > start:
            parts.append(_rescan(self, sb, key, start, ce, commit=True))
        else:
            sb.ckpt_end[key] = start      # nothing to commit; keep the chain consistent
        parts.append(_rescan(self, sb, key, ce, e, commit=False))
        out_w = torch.cat(parts, 0) if len(parts) > 1 else parts[0]
    core_attn_out[:P] = out_w[pos - start].to(core_attn_out.dtype)

    if ctx.final:
        blk = ctx.mamba_blocks[key]
        if _DEBUG_STATE:
            ssm_stock, conv_stock = self.kv_cache[1][blk], self.kv_cache[0][blk]
            qkv = sb.qkv[key]
            ctx.debug[key] = {
                # our re-scan vs whatever the (approx) prefill left in the block -- meaningful
                # only when the approx prompt == the corrected prompt (gate g0)
                "ssm_rel": float((sb.ckpt[key][0].float() - ssm_stock.float()).norm()
                                 / max(ssm_stock.float().norm().item(), 1e-30)),
                "conv_rel": float((sb.conv_scratch[key][1].float() - conv_stock.float()).norm()
                                  / max(conv_stock.float().norm().item(), 1e-30)),
                "ssm_stock_norm": float(ssm_stock.float().norm()),
                "ssm_ours_norm": float(sb.ckpt[key].float().norm()),
                "qkv_zero_rows": int((qkv.float().abs().sum(1) == 0).sum()),
                "qkv_norm": float(qkv.float().norm()),
            }
        self.kv_cache[1][blk] = sb.ckpt[key]
        self.kv_cache[0][blk] = sb.conv_scratch[key][1]
    return None


# ---------------------------------------------------------------------------------------------
# runner hooks
# ---------------------------------------------------------------------------------------------

_ORIG_PREPARE_INPUTS = None
_ORIG_EXECUTE_MODEL = None
_ORIG_MODEL_FORWARD = None


def appcorr_arm_final(self: GPUModelRunner, req_id: str, positions: torch.Tensor,
                      inputs_embeds: torch.Tensor, window, *, stage=None,
                      replay: bool = False) -> None:
    """Defer the final correct step to the engine step that computes the held-back row N-1
    (`_run_pending`, from `_prepare_inputs`), where it can be fused with it. The caller releases
    the hold-back next and steps the engine; `appcorr_take_final_info` returns the step's info
    once it has run (None while the scheduler has not yet picked the request up)."""
    assert req_id in _SIDE, req_id
    assert req_id not in _PENDING, f"{req_id}: final step armed twice"
    _PENDING[req_id] = _Pending(positions=positions, inputs_embeds=inputs_embeds,
                                window=(int(window[0]), int(window[1])), stage=stage,
                                replay=bool(replay), t_arm=time.perf_counter())


def appcorr_take_final_info(self: GPUModelRunner, req_id: str) -> Optional[dict]:
    return _FINAL_INFO.pop(req_id, None)


def _run_pending(self: GPUModelRunner, num_scheduled_tokens, starts) -> None:
    """Run the armed final steps of the requests this engine step schedules (after
    `_update_states`: their block tables hold N-1's block). One decision for the whole step:
    fused iff every scheduled token is an armed request's released hold-back row (and no arm
    replays captured outputs, which do not exist for N-1); otherwise the plain final step."""
    ib = self.input_batch
    due = []
    for rid, pend in _PENDING.items():
        idx = ib.req_id_to_index.get(rid)
        if idx is None or int(num_scheduled_tokens[idx]) == 0:
            continue
        sb = _SIDE[rid]
        pos0, ntok = int(ib.num_computed_tokens_cpu[idx]), int(num_scheduled_tokens[idx])
        assert pos0 == sb.n - 1 and ntok == 1, (rid, pos0, ntok, sb.n)
        due.append((rid, pend, idx))
    if not due:
        return
    total = int(np.sum(num_scheduled_tokens))
    fused = FUSE_HOLDBACK and total == len(due) and not any(p.replay for _, p, _ in due)
    stub: dict[int, torch.Tensor] = {}
    for rid, pend, idx in due:
        del _PENDING[rid]
        t0 = time.perf_counter()
        if pend.stage is None:
            info = appcorr_rows_step(self, rid, pend.positions, (0, num_layers(self)),
                                     inputs_embeds=pend.inputs_embeds, window=pend.window,
                                     final=True, replay=pend.replay, fuse=fused)
        else:
            info = appcorr_staged_correct(self, rid, pend.positions, pend.inputs_embeds,
                                          pend.window, True, pend.stage, replay=pend.replay,
                                          fuse=fused)
        if fused:
            sb = _SIDE[rid]
            assert sb.hold_hidden is not None, rid
            stub[int(starts[idx])] = sb.hold_hidden
            sb.hold_hidden = None
        info["fused"] = bool(fused)
        info["t_armed_ms"] = (t0 - pend.t_arm) * 1e3
        _FINAL_INFO[rid] = info
    _ST.stub = stub if fused else None


def _model_forward(self: GPUModelRunner, input_ids=None, positions=None,
                   intermediate_tensors=None, inputs_embeds=None, **model_kwargs):
    """Fused hold-back: the released row's forward is replaced by the hidden rows the fused
    correct step stashed (`execute_model` then samples from them as usual)."""
    if _ST.skip_forward:
        # every token of this step is a prompt row of a request opened with `open_walk`: its
        # K/V, side-buffer rows and recurrent state come from the walks, and the request is one
        # token short so nothing is sampled from these rows -- no forward at all
        _ST.skip_forward = False
        n = (inputs_embeds if inputs_embeds is not None else input_ids).shape[0]
        return torch.zeros((n, self.model_config.get_hidden_size()),
                           dtype=self.model_config.dtype, device=self.device)
    stub = _ST.stub
    if not stub:
        return _ORIG_MODEL_FORWARD(self, input_ids=input_ids, positions=positions,
                                   intermediate_tensors=intermediate_tensors,
                                   inputs_embeds=inputs_embeds, **model_kwargs)
    _ST.stub = None
    assert intermediate_tensors is None
    n = (inputs_embeds if inputs_embeds is not None else input_ids).shape[0]
    row0 = next(iter(stub.values()))
    hs = torch.zeros((n, row0.shape[-1]), dtype=row0.dtype, device=row0.device)
    for tok, row in stub.items():
        hs[tok] = row
    return hs


def _prepare_inputs(self: GPUModelRunner, scheduler_output, num_scheduled_tokens):
    starts = None
    if _SIDE:
        starts = np.zeros(len(num_scheduled_tokens) + 1, dtype=np.int64)
        np.cumsum(num_scheduled_tokens, out=starts[1:])
        if _PENDING:
            # before the stock call: the correct step's graph path uses the runner's persistent
            # input buffers, which the stock call then refills for this step
            _run_pending(self, num_scheduled_tokens, starts)
    out = _ORIG_PREPARE_INPUTS(self, scheduler_output, num_scheduled_tokens)
    caps: list[_Capture] = []
    if _SIDE:
        for req_id, sb in _SIDE.items():
            idx = self.input_batch.req_id_to_index.get(req_id)
            if idx is None:
                continue
            ntok = int(num_scheduled_tokens[idx])
            if ntok <= 0:
                continue
            pos0 = int(self.input_batch.num_computed_tokens_cpu[idx])
            if pos0 + ntok > sb.n:      # decode steps: nothing left to capture
                continue
            # on a softmax-only decoder nothing consumes these (`_forward_core_patch` is never
            # reached); they are kept because `skip_forward` / `n_capture_steps` below are about
            # the request's scheduling, not about recurrent state
            caps.append(_Capture(sb=sb, tok0=int(starts[idx]), ntok=ntok, pos0=pos0))
            sb.n_capture_steps += 1
        total = int(starts[-1])
        # rows of the skipped approximate prefill = prompt rows below the hold-back (the step
        # that computes row N-1 at `final` -- fused stub or stock -- and every decode step run)
        skipped = sum(c.ntok for c in caps
                      if c.sb.skip_prefill and c.pos0 + c.ntok <= c.sb.n - 1)
        # the whole batch is skip-prefill prompt rows: no model forward this step (a batch that
        # also carries other tokens runs the stock forward; the skipped request's rows are then
        # computed for nothing, which is harmless -- the walks overwrite everything they read)
        _ST.skip_forward = total > 0 and skipped == total
        _ST.captures = [c for c in caps if not c.sb.skip_prefill]
        return out
    _ST.captures = caps
    return out


def _execute_model(self: GPUModelRunner, scheduler_output, intermediate_tensors=None):
    if not _SIDE:
        return _ORIG_EXECUTE_MODEL(self, scheduler_output, intermediate_tensors)
    # backstop for the deferred GDN patch (`check_gdn_path` at open is the normal entry): this
    # runs before the stock call, so before any capture. One walk of the decoder, then cached.
    _ensure_gdn(self)
    _glm53_sparse(self)     # AppCorr/M: install the kpool-indexer hook before the first capture
    prev, _ST.mode = _ST.mode, MODE_CAPTURE
    try:
        return _ORIG_EXECUTE_MODEL(self, scheduler_output, intermediate_tensors)
    finally:
        _ST.mode = prev
        _ST.captures = []
        _ST.stub = None
        _ST.skip_forward = False


def _leaf_spec(spec, layer_name: str | None = None):
    """The per-layer KV-cache spec behind a group's spec.

    vLLM main wraps same-type layers in `UniformTypeKVCacheSpecs` -- a CONTAINER whose
    `.kv_cache_specs` maps layer name -> leaf spec -- so `isinstance(group.kv_cache_spec, X)`
    is False for every leaf class X and a dispatch on the group's spec sees only the wrapper
    (leg 3 of the TP gate, third failure: `AssertionError: <class '...UniformTypeKVCacheSpecs'>`
    in `glm53_indexer.layer_caches`, B200-8 2026-09-13 04:10 KST).  Descend to the named
    layer's spec, or to the first one when the question is about the group as a whole (all
    members are uniform by construction)."""
    try:
        from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
    except ImportError:                      # 0.28 and earlier: groups carry leaf specs
        return spec
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return spec.kv_cache_specs[layer_name] if layer_name is not None else spec.first_spec
    return spec


def _mamba_group_ids(self: GPUModelRunner) -> list[int]:
    """KV-cache groups holding recurrent state.  EMPTY for a pure-softmax decoder (GLM-4.6V),
    where every group is an attention group and the correct step touches nothing else."""
    from vllm.v1.kv_cache_interface import MambaSpec
    return [gid for gid, g in enumerate(self.kv_cache_config.kv_cache_groups)
            if isinstance(_leaf_spec(g.kv_cache_spec), MambaSpec)]


def _block_row(self: GPUModelRunner, req_id: str, gid: int) -> torch.Tensor:
    """The request's block-table row for kv group `gid`, built from the runner's request table
    rather than the persistent batch: any engine step that does not schedule the request (idle
    steps between the approx prefill and the first `correct`, other requests' steps at c>1)
    evicts it from `input_batch` (`_update_states`: unscheduled_req_ids -> remove_request) while
    `self.requests` keeps it. Same contents as `block_table.gpu[idx]` when it is in the batch
    (kernel-block mapping included), zero-padded to the batch's row width."""
    import numpy as np
    bt = self.input_batch.block_table[gid]
    ids = np.asarray(self.requests[req_id].block_ids[gid], dtype=np.int32)
    if bt.blocks_per_kv_block > 1:
        ids = bt.map_to_kernel_blocks(ids, bt.blocks_per_kv_block, bt._kernel_block_arange)
    row = torch.zeros(bt.max_num_blocks_per_req, dtype=torch.int32, device=self.device)
    row[:len(ids)] = torch.from_numpy(np.ascontiguousarray(ids)).to(self.device)
    return row


def _slot_mapping(self: GPUModelRunner, gid: int, req_id: str, positions: torch.Tensor):
    """`_compute_slot_mapping_kernel` (block_table.py:397-442) for one request, DCP/PCP world 1.

    With TOTAL_CP_WORLD_SIZE == 1 the kernel reduces to
    ``block_table[pos // kernel_block_size] * kernel_block_size + pos % kernel_block_size``
    (the hybrid-block split cancels: KV_CACHE_BLOCK_SIZE == block_size * BLOCKS_PER_KV_BLOCK)."""
    bt = self.input_batch.block_table[gid]
    assert bt.dcp_world_size == 1 and bt.pcp_world_size == 1, "DCP/PCP not handled"
    row = _block_row(self, req_id, gid)
    bs = bt.block_size
    slots = row[positions // bs].to(torch.int64) * bs + (positions % bs)
    return row, slots


# >>> AppCorr/M (GLM-5.3 sparse MLA): the kpool indexer's paged tail cache >>>

def _is_kpool_tail_group(self: GPUModelRunner, gid: int) -> bool:
    """Is kv-cache group ``gid`` a ``KpoolTailSpec`` group (the sparse indexer's tail buffer)?

    Structural, not model-keyed: the spec class exists only for the kpool indexer.  Imported
    lazily because vLLM builds without GLM-5.3 do not define it."""
    try:
        from vllm.v1.kv_cache_interface import KpoolTailSpec
    except ImportError:
        return False
    return isinstance(_leaf_spec(self.kv_cache_config.kv_cache_groups[gid].kv_cache_spec),
                      KpoolTailSpec)


def _kpool_tail_slot_mapping(self: GPUModelRunner, gid: int, req_id: str,
                             positions: torch.Tensor):
    """``compute_kpool_tail_slot_mapping`` (mla/indexer.py:520-545) for one request.

    ``KpoolTailSpec`` allocates exactly ONE block of ``index_kpool`` slots per request
    (`kv_cache_interface.py:974-980`), used as a circular buffer keyed on ``pos % kpool``.  The
    generic ``_slot_mapping`` cannot build this: it evaluates ``block_row[pos // block_size]``
    into a row that is ONE entry wide, so every position >= kpool reads out of bounds (garbage
    slot on CPU, an illegal access or a silent wrong write on GPU).  The tail builder recomputes
    the mapping itself when ``positions`` is set, so this value only has to be in-bounds and
    correct for `set_forward_context`'s per-layer slot map."""
    bt = self.input_batch.block_table[gid]
    row = _block_row(self, req_id, gid)
    kp = int(bt.block_size)
    slots = row[0].to(torch.int64) * kp + (positions % kp)
    return row, slots

# <<< AppCorr/M <<<


def _mamba_blocks(self: GPUModelRunner, req_id: str, mamba_gids: list[int]) -> dict[str, int]:
    """GDN layer prefix -> the request's recurrent-state block, per mamba KV-cache group.

    Qwen3.5 splits its GDN layers over several mamba groups (4B: 3 groups + 1 attention group),
    so a single block id is not enough.  Returns {} when `mamba_gids` is empty (GLM-4.6V)."""
    out: dict[str, int] = {}
    for gid in mamba_gids:
        blk = int(_block_row(self, req_id, gid)[0])
        for ln in self.kv_cache_config.kv_cache_groups[gid].layer_names:
            out[ln] = blk
    return out


def _decoder_module(model):
    """The text decoder of a served model (`Qwen3_5Model`, `Glm4MoeModel`, ...): its `layers`
    are what a partial-depth step walks and what the GDN detection inspects.  Structural walk
    (`.unwrap()` -> `.language_model` -> `.model`), no model-name or class-name keying: GLM-4.6V's
    `Glm4vMoeForConditionalGeneration.language_model` is a `Glm4MoeForCausalLM` whose `.model`
    holds the 46 `Glm4MoeDecoderLayer`s, so the same walk resolves it (checked on the vLLM
    sources, glm4_1v.py:1803 + glm4_moe.py:402/524)."""
    m = model
    m = m.unwrap() if hasattr(m, "unwrap") else m
    lm = getattr(m, "language_model", m)
    lm = getattr(lm, "model", lm)
    assert hasattr(lm, "layers"), type(lm)
    return lm


def _decoder(self: GPUModelRunner):
    return _decoder_module(self.model)


def num_layers(self: GPUModelRunner) -> int:
    return len(_decoder(self).layers)


def stage_bounds(n_layers: int, n_rounds: int) -> list[int]:
    """Depth bound of round r, equal layer counts: round r corrects over `[0, bounds[r])` and the
    frontier walk after it covers `[bounds[r], bounds[r+1])`; the last round is full depth.
    Same function as `flops_analytic.stage_bounds` (the closed form prices what this runs)."""
    assert n_rounds >= 1
    return [int(round(n_layers * (r + 1) / n_rounds)) for r in range(n_rounds)]


def stage_spec(stage, n_layers: int) -> tuple[int, int, list[int]]:
    """`stage` -> (round, n_rounds, bounds). `(r, g)` derives the equal-layer bounds here, as it
    always did; `(r, g, bounds)` takes them from the caller -- the unified vision+decoder axis
    (memo §7.12), whose LLM rounds are fewer than the schedule's `groups` and whose depths come
    from a cost split across BOTH halves, so neither the count nor the spacing is `L(r+1)/g`.
    The contract the rest of this file relies on is unchanged: strictly increasing, last == L."""
    r, g = int(stage[0]), int(stage[1])
    bounds = stage_bounds(n_layers, g) if len(stage) == 2 else [int(b) for b in stage[2]]
    assert 0 <= r < g == len(bounds), (r, g, bounds)
    assert bounds[-1] == n_layers, (bounds, n_layers)
    assert all(0 < a < b for a, b in zip(bounds, bounds[1:])) and bounds[0] > 0, bounds
    return r, g, bounds


def appcorr_correct_step(self: GPUModelRunner, req_id: str, positions: torch.Tensor,
                         inputs_embeds: torch.Tensor, window, final: bool,
                         replay: bool = False, fuse: bool = False) -> dict:
    """Full-depth correction of prompt rows ``positions`` (the MVP schedule): see
    `appcorr_rows_step`."""
    return appcorr_rows_step(self, req_id, positions, (0, num_layers(self)),
                             inputs_embeds=inputs_embeds, window=window, final=final,
                             replay=replay, fuse=fuse)


def appcorr_rows_step(self: GPUModelRunner, req_id: str, positions: torch.Tensor, layers,
                      *, inputs_embeds: Optional[torch.Tensor] = None, from_frontier: bool = False,
                      window, commit_end: int = -1, final: bool = False,
                      store_frontier: bool = False, replay: bool = False,
                      fuse: bool = False) -> dict:
    """Eager forward(s) of |P| pseudo-sequences that rewrite prompt rows ``positions`` over the
    decoder layers ``[layers[0], layers[1])``.

    Inputs: ``inputs_embeds`` [P, D] (layer 0), or ``from_frontier`` (the request's frontier
    buffers at ``layers[0]``), or neither (layer 0 with the request's stored prompt embeddings --
    the approximate rows). ``store_frontier`` writes the rows' residual stream at ``layers[1]``
    back to the frontier buffers. ``window`` = (start, end) prompt positions of this step's
    DeltaNet re-scan (every position lies in it); ``commit_end`` is where the re-scan commits its
    checkpoint (default: the window end). ``fuse`` (final only): also compute the held-back
    row N-1 from the request's stored embedding -- window and checkpoint extend to N, the
    recurrent state written back is the one after N-1, and its final hidden row is left in
    ``sb.hold_hidden`` for the fused hold-back step.

    A step with more rows than ``max_num_seqs`` (the attention builders size their persistent
    buffers by it) is run as consecutive sub-batches in position order. That is equivalent to
    one batch: a rewritten row p reads only rows <= p at every layer, and the earlier sub-batches
    have written theirs (softmax K/V and side-buffer rows) before the later ones run; the
    DeltaNet checkpoint is advanced by the last sub-batch only (the earlier re-scans are repeated
    work, reported as ``n_sub``).
    """
    t0 = time.perf_counter()
    sb = _SIDE[req_id]
    assert req_id in self.requests, f"{req_id} not in the runner's request table"

    P = int(positions.numel())
    assert P > 0, P
    if inputs_embeds is not None:
        assert inputs_embeds.shape[0] == P, (P, inputs_embeds.shape)
        assert not from_frontier
    sched = self.vllm_config.scheduler_config
    s, e = int(window[0]), int(window[1])
    a, b = int(layers[0]), int(layers[1])
    L = num_layers(self)
    assert 0 <= a < b <= L, (a, b, L)
    assert (a == 0) == (not from_frontier), "from_frontier iff the step starts above layer 0"
    if final:
        assert b == L, "the final round (state write-back) runs at full depth"
    assert not replay or _ensure_gdn(self), (
        "replay mode (gate g1) isolates the softmax rewrite from the DeltaNet re-scan; on a "
        "pure-softmax decoder there is nothing to replay and the correct step IS the softmax path")
    pos_cpu = positions.detach().cpu()
    assert bool((pos_cpu[1:] > pos_cpu[:-1]).all()) if P > 1 else True, "positions must be sorted"
    assert int(pos_cpu[0]) >= s and int(pos_cpu[-1]) < e, (int(pos_cpu[0]), int(pos_cpu[-1]), s, e)
    assert int(pos_cpu[-1]) < sb.n - 1, "row N-1 is never rewritten (hold-back)"

    if fuse:
        # the released hold-back row joins the batch as its last pseudo-sequence
        assert final and inputs_embeds is not None and not from_frontier
        assert e == sb.n - 1, (e, sb.n)
        n1 = sb.n - 1
        pe = self.requests[req_id].prompt_embeds
        assert pe is not None and pe.shape[0] >= sb.n, (None if pe is None else pe.shape)
        pos_cpu = torch.cat([pos_cpu, torch.tensor([n1], dtype=pos_cpu.dtype)])
        positions = pos_cpu
        inputs_embeds = torch.cat(
            [inputs_embeds, pe[n1:n1 + 1].to(inputs_embeds.device, inputs_embeds.dtype)], 0)
        P += 1
        e, commit_end = sb.n, -1

    if inputs_embeds is None and not from_frontier:
        pe = self.requests[req_id].prompt_embeds
        assert pe is not None and pe.shape[0] >= sb.n, (None if pe is None else pe.shape)
        inputs_embeds = pe[pos_cpu].to(self.device)

    contig = CONTIG_ROWS and P > 1 and int(pos_cpu[-1]) - int(pos_cpu[0]) + 1 == P
    # contiguous rows go in as one sequence: the token budget bounds the sub-batch, not the
    # sequence count (the sub-batches of a contiguous range are contiguous themselves)
    cap = int(sched.max_num_batched_tokens) if contig else \
        min(int(sched.max_num_seqs), int(sched.max_num_batched_tokens))
    bounds = list(range(0, P, cap)) + [P]
    info, cg = None, []
    for j, (a0, a1) in enumerate(zip(bounds[:-1], bounds[1:])):
        last = j == len(bounds) - 2
        info = _correct_sub(self, req_id, sb, positions[a0:a1],
                            None if inputs_embeds is None else inputs_embeds[a0:a1], (s, e),
                            layers=(a, b), from_frontier=from_frontier, commit_end=commit_end,
                            final=final and last, commit=last, replay=replay,
                            store_frontier=store_frontier, fuse=fuse and last)
        cg.append(info["cudagraph"])
    torch.cuda.synchronize()
    return {"num_rows": P, "window": [s, e], "layers": [a, b], "final": bool(final),
            "fused": bool(fuse), "contig": bool(contig),
            "n_sub": len(bounds) - 1, "t_step_ms": (time.perf_counter() - t0) * 1e3,
            "cudagraph": cg,
            "mamba_blocks": info["mamba_blocks"], "side_buffer_mb": sb.nbytes() / 2**20,
            "capture_steps": sb.n_capture_steps, "debug": info["debug"]}


def appcorr_staged_correct(self: GPUModelRunner, req_id: str, positions: torch.Tensor,
                           inputs_embeds: torch.Tensor, window, final: bool, stage,
                           replay: bool = False, fuse: bool = False) -> dict:
    """Depth-staged round ``stage = (r, g)`` (or ``(r, g, bounds)``): correct ``positions`` over
    ``[0, b_r)``, then walk every image row ``[lo, hi)`` through ``[b_r, b_{r+1})`` from the
    frontier buffers (the corrected rows carry their corrected input, the others their
    approximate one; all of them see the corrected K/V and side-buffer rows as context -- the
    staged schedule's semantics).

    The first round walks the image rows through ``[0, b_0)`` from the stored approximate prompt
    first (identical values to the stock prefill's; it only materialises the frontier), and a
    round whose band sent no rows leaves a gap the next round's catch-up walk closes. The last
    round is the MVP's full-depth step (``final`` writes the recurrent state back).

    With explicit ``bounds`` (`stage_spec`) nothing above changes; only where the bounds come
    from does. That is what the unified axis needs: its LLM rounds are the schedule's rounds that
    reached past the vision tower, so there are fewer of them and they start deeper.
    """
    sb = _SIDE[req_id]
    assert sb.hi > sb.lo, f"image rows unknown ({sb.lo}, {sb.hi}): open with image_end"
    L = num_layers(self)
    r, g, bounds = stage_spec(stage, L)
    b = bounds[r]
    nxt = bounds[r + 1] if r + 1 < g else None
    assert (b == L) == bool(final), (r, g, b, L, final)
    rows = torch.arange(sb.walk_lo, sb.hi, dtype=torch.int64)
    n1 = sb.n - 1
    log = []
    if sb.open_walk_info is not None:            # report the open walk with this round's steps
        log.append(sb.open_walk_info)
        sb.open_walk_info = None

    def walk(a, bb, from_prompt: bool):
        inf = appcorr_rows_step(self, req_id, rows, (a, bb), from_frontier=not from_prompt,
                                window=(0, n1), commit_end=sb.committed_end,
                                store_frontier=(bb < L))
        log.append({"kind": "walk", "layers": [a, bb], "rows": int(rows.numel()),
                    "t_ms": inf["t_step_ms"], "n_sub": inf["n_sub"]})
        sb.frontier = bb

    if sb.frontier < b and (sb.n_corrected > 0 or b < L):
        walk(sb.frontier, b, from_prompt=(sb.frontier == 0))
    assert not fuse or final, "fuse is the final round's option"
    info = appcorr_rows_step(self, req_id, positions, (0, b), inputs_embeds=inputs_embeds,
                             window=window, commit_end=int(window[1]), final=final,
                             store_frontier=(b < L), replay=replay, fuse=fuse)
    log.append({"kind": "correct", "layers": [0, b], "rows": int(positions.numel()),
                "t_ms": info["t_step_ms"], "n_sub": info["n_sub"]})
    sb.committed_end = int(window[1])
    sb.n_corrected += 1
    if nxt is not None:
        walk(b, nxt, from_prompt=False)
    sb.stage_log.extend(log)
    info["stage"] = {"round": r, "rounds": g, "bounds": bounds, "steps": log,
                     # the open walk ran at open, before this round: reported, not charged
                     "t_total_ms": sum(x["t_ms"] for x in log if x["kind"] != "open_walk")}
    info["t_step_ms"] = info["stage"]["t_total_ms"]
    return info


def appcorr_open_walk(self: GPUModelRunner, req_id: str) -> dict:
    """The unified axis's opening step (`open_walk` = b_0): walk rows [0, hi) -- the text
    prefix and every image row -- through decoder layers [0, b_0) from the prompt embeddings,
    in place of the stock full-depth prefill this request skipped. Same values the first staged
    round's catch-up walk produced from the stock prefill's rows (it walked from the prompt
    too); the text prefix joins the walk because no prefill computed its K/V. The post-image
    text is computed once, by the final round (it is in that round's positions)."""
    sb = _SIDE[req_id]
    assert sb.skip_prefill and sb.open_walk > 0 and sb.frontier == 0, (
        sb.skip_prefill, sb.open_walk, sb.frontier)
    L = num_layers(self)
    b0 = min(sb.open_walk, L)
    rows = torch.arange(sb.walk_lo, sb.hi, dtype=torch.int64)
    inf = appcorr_rows_step(self, req_id, rows, (0, b0), from_frontier=False,
                            window=(0, sb.n - 1), commit_end=sb.committed_end,
                            store_frontier=(b0 < L))
    sb.frontier = b0
    sb.open_walk_info = {"kind": "open_walk", "layers": [0, b0], "rows": int(rows.numel()),
                         "t_ms": inf["t_step_ms"], "n_sub": inf["n_sub"],
                         "t_done": time.perf_counter()}
    sb.stage_log.append(sb.open_walk_info)
    return inf


# >>> AppCorr/K (GLM-5.3 mHC): the layer walk's state arity >>>
_WALK_KIND: dict = {}            # decoder-layer class -> "res2" | "mhc4"


def _walk_kind(layer) -> str:
    """The layer's inter-layer state contract, read off its SIGNATURE (never a model name).

      * ``res2``  ``layer(positions, hidden_states, residual) -> (hidden, residual)``
        -- Qwen3.5 / GLM-4.6V and every other stock vLLM decoder layer.
      * ``mhc4``  ``layer(positions, hidden_states, residual, post, comb)
        -> (hidden, residual, post, comb)`` -- GLM-5.3-Flash's ``Glm5NextDecoderLayer``
        (model.py:401-509): four mHC residual streams, ``residual`` [T, 4, H] bf16, ``post``
        [T, 4] fp32, ``comb`` [T, 4, 4] fp32.  Layer 0 expands (``hc_expand``), the LAST layer
        materialises its ``hc_post`` and contracts, and every layer in between DEFERS its
        ``hc_post`` into the next layer's ``hc_fused_post_pre`` -- which is exactly why the
        walk has to carry ``post``/``comb`` and cannot call a layer in isolation.

    Cached per class: one process serves one model, and the signature cannot change under us.
    """
    cls = type(layer)
    kind = _WALK_KIND.get(cls)
    if kind is None:
        import inspect
        params = inspect.signature(cls.forward).parameters
        kind = "mhc4" if ("post" in params and "comb" in params) else "res2"
        _WALK_KIND[cls] = kind
    return kind
# <<< AppCorr/K <<<


def _run_layers(self: GPUModelRunner, layers, positions_gpu, inputs_embeds, hidden_in):
    """Decoder layers ``[a, b)`` on the batch: the compiled model for the full depth (the MVP's
    path, bit-for-bit what it ran before), an eager loop over the layer modules otherwise.
    Returns the walk state at layer ``b`` -- ``(hidden, residual)``, or the mHC 4-tuple
    ``(hidden, residual, post, comb)`` on a ``mhc4`` decoder; at full depth the model's output
    (the normed final hidden states, read by the fused hold-back only).

    Entry at ``a == 0`` starts from ``inputs_embeds`` with every carried stream None, so the
    layer does its own expand; entry at ``a > 0`` resumes from ``hidden_in``, which must be the
    same arity (the frontier buffers store what this returned).  A partial walk that ends at
    ``b == L`` runs the last layer's contract and returns None: its purpose is the K/V and
    recurrent state it leaves behind, not a value."""
    a, b = layers
    lm = _decoder(self)
    L = len(lm.layers)
    if a == 0 and b == L:
        return self.model(input_ids=None, positions=positions_gpu,
                          intermediate_tensors=None, inputs_embeds=inputs_embeds)
    if _walk_kind(lm.layers[a]) == "mhc4":          # >>> AppCorr/K (GLM-5.3 mHC 4-tuple walk)
        if a == 0:
            state = (inputs_embeds, None, None, None)
        else:
            state = tuple(hidden_in)
            assert len(state) == 4, f"an mHC walk resumed from {len(state)} tensors"
        for layer in lm.layers[a:b]:
            state = layer(positions=positions_gpu, hidden_states=state[0], residual=state[1],
                          post=state[2], comb=state[3])
        return None if b == L else tuple(state)     # <<< AppCorr/K
    if a == 0:
        hs, res = inputs_embeds, None
    else:
        hs, res = hidden_in
    for layer in lm.layers[a:b]:
        hs, res = layer(positions=positions_gpu, hidden_states=hs, residual=res)
    return None if b == L else (hs, res)


# >>> AppCorr/M (GLM-5.3 sparse MLA) >>>

_GLM53_SPARSE: Optional[bool] = None


def _glm53_sparse(self: GPUModelRunner) -> bool:
    """Does the served decoder have kpool-sparse MLA layers?  Resolved once, then cached.

    Detection is by module structure (`glm53_indexer.sparse_layers`: a
    ``MultiHeadLatentAttentionWrapper`` with a non-None indexer whose ``index_kpool > 1``), never
    by model name -- a dense-MLA DeepSeek and a GQA decoder both answer False."""
    global _GLM53_SPARSE
    if _GLM53_SPARSE is None:
        from appcorr.vllm_stream import glm53_indexer as _gi
        # `sparse_layers` only touches attributes, so it is safe on any decoder and needs no
        # try/except: a vLLM without the GLM-5.3 tree simply has no module with an `.indexer`.
        # An install failure on a decoder that DOES have them must raise, not silently disable
        # the rewrite -- that would corrupt the pooled cache with no signal.
        _GLM53_SPARSE = bool(_gi.sparse_layers(_decoder(self)))
        if _GLM53_SPARSE:
            n = _gi.install(self.model)
            assert n == len(_gi.sparse_layers(_decoder(self))), (n,)
    return _GLM53_SPARSE


def _glm53_correct_context(self: GPUModelRunner, req_id: str, sb: SideBuffer,
                           positions: torch.Tensor, max_seq_len: int):
    """The round's indexer rewrite context, or None on every non-sparse-MLA decoder."""
    if not _glm53_sparse(self):
        return None
    from appcorr.vllm_stream import glm53_indexer as _gi
    return _gi.CorrectContext(
        buf=_gi.buffer_for(sb),
        caches=_gi.layer_caches(self, req_id),
        positions=positions,
        max_seq_len=int(max_seq_len),
        topk_tokens=_gi.topk_tokens_of(self.model),
    )


def _glm53_set_context(ctx):
    if ctx is None and _GLM53_SPARSE is not True:
        return None
    from appcorr.vllm_stream import glm53_indexer as _gi
    return _gi.set_context(ctx)

# <<< AppCorr/M <<<


def _correct_sub(self: GPUModelRunner, req_id: str, sb: SideBuffer, positions: torch.Tensor,
                 inputs_embeds: Optional[torch.Tensor], window, *, layers, from_frontier: bool,
                 commit_end: int, final: bool, commit: bool, replay: bool,
                 store_frontier: bool, fuse: bool = False) -> dict:
    """One sub-batch of `appcorr_rows_step` (all of it when |P| <= max_num_seqs)."""
    from vllm.config import CUDAGraphMode
    from vllm.forward_context import set_forward_context
    from vllm.v1.attention.backend import CommonAttentionMetadata

    req_state = self.requests[req_id]   # the caller (client.correct) drained the approx prefill
    P = int(positions.numel())
    sched = self.vllm_config.scheduler_config
    s, e = int(window[0]), int(window[1])
    pos_cpu = positions.detach().cpu()
    p0, p1 = int(pos_cpu[0]), int(pos_cpu[-1])
    contig = CONTIG_ROWS and P > 1 and p1 - p0 + 1 == P
    # the round is submitted as |P| pseudo-sequences of one token (builders -- FlashInfer's
    # paged_kv_indptr, flashinfer.py:921 -- size their persistent buffers by max_num_seqs), or
    # as ONE prefill sequence of P tokens when the rows are contiguous (CONTIG_ROWS)
    if not contig:
        assert P <= sched.max_num_seqs, (P, sched.max_num_seqs)
    assert P <= sched.max_num_batched_tokens, (P, sched.max_num_batched_tokens)

    dev = self.device
    positions = positions.to(dev, torch.int64)
    hidden_in = None
    if from_frontier:
        hidden_in = sb.frontier_rows(positions)
        inputs_embeds = None
    else:
        inputs_embeds = inputs_embeds.to(dev, self.model_config.dtype)

    # >>> AppCorr/M (GLM-5.3): positions of exactly these rows.  GLM-5.3-Flash has NO M-RoPE
    # (`Glm5NextTextConfig` ships no `mrope_section`, so `uses_mrope` is False and the runner
    # passes flat `[num_tokens]` positions -- survey §A).  The decoder has no rotary at all
    # (KDA ignores `positions`, MLA's `rotary_emb` is None under `mla_nope`), but `positions` is
    # NOT dead: the sparse indexer consumes it for the tail slot (`pos % kpool`) and the
    # short-prefill causal fill, so the 1-D form must be the row's true prompt position.
    if self.uses_mrope:
        mrope = req_state.mrope_positions
        assert mrope is not None and mrope.shape[1] >= sb.n, (
            None if mrope is None else mrope.shape)
        positions_gpu = mrope[:, pos_cpu].to(dev, torch.int64)
    else:
        positions_gpu = positions
    # <<< AppCorr/M <<<

    # A pure-softmax decoder has no mamba group and no side-buffer state: `mamba_gids` is empty,
    # `mamba_blocks` is {}, the `gid in mamba_gids` skip below never fires (every group is an
    # attention group) and `_forward_core_patch` is never reached, so the re-scan window is a
    # no-op and this step is the pseudo-sequence softmax path alone.
    mamba_gids = _mamba_group_ids(self)
    assert mamba_gids or not _ensure_gdn(self), (
        "the decoder has Gated DeltaNet layers but no mamba kv-cache group")
    mamba_blocks = _mamba_blocks(self, req_id, mamba_gids)

    _rf_meta = torch.profiler.record_function("appcorr.metadata")
    _rf_meta.__enter__()
    if contig:
        qsl_cpu = torch.tensor([0, P], dtype=torch.int32)
        seq_cpu = torch.tensor([p1 + 1], dtype=torch.int32)
        ncomp_cpu = torch.tensor([p0], dtype=torch.int32)
        n_reqs, max_q = 1, P
        is_pref = torch.ones(1, dtype=torch.bool)
    else:
        qsl_cpu = torch.arange(P + 1, dtype=torch.int32)
        seq_cpu = (pos_cpu + 1).to(torch.int32)
        ncomp_cpu = pos_cpu.to(torch.int32)
        n_reqs, max_q = P, 1
        is_pref = torch.zeros(P, dtype=torch.bool)
    attn_metadata: dict = {}
    slot_by_layer: dict[str, torch.Tensor] = {}
    kv_group_shapes = {}
    # >>> AppCorr/M (version shim, NOT MLA-specific -- flagged to K/V) >>>
    # `_seq_lens_cpu` / `_num_computed_tokens_cpu` were fields of `CommonAttentionMetadata` in
    # vLLM 0.28.0 and are GONE in main @658c813 (the build B200-8 serves GLM-5.3 with): grepping
    # the whole tree finds neither name.  Passing them unconditionally is a TypeError at the
    # first correct step on that build, for every model, so they are passed only when the
    # dataclass still declares them.  Nothing is lost: the builders read `seq_lens` /
    # `seq_lens_cpu_upper_bound` and derive num_computed from `query_start_loc`.
    import dataclasses as _dc
    _cm_fields = {f.name for f in _dc.fields(CommonAttentionMetadata)}
    _cm_compat = {k: v for k, v in (("_seq_lens_cpu", seq_cpu),
                                    ("_num_computed_tokens_cpu", ncomp_cpu))
                  if k in _cm_fields}
    # <<< AppCorr/M <<<
    for gid, group in enumerate(self.kv_cache_config.kv_cache_groups):
        if gid in mamba_gids:
            continue
        # >>> AppCorr/M (GLM-5.3): the kpool indexer's tail group is 1 block/request >>>
        if _is_kpool_tail_group(self, gid):
            row, slots = _kpool_tail_slot_mapping(self, gid, req_id, positions)
        else:
            row, slots = _slot_mapping(self, gid, req_id, positions)
        # <<< AppCorr/M <<<
        blk = row.unsqueeze(0).expand(n_reqs, -1).contiguous()
        cm = CommonAttentionMetadata(
            query_start_loc=qsl_cpu.to(dev),
            query_start_loc_cpu=qsl_cpu,
            seq_lens=seq_cpu.to(dev),
            **_cm_compat,                                        # AppCorr/M version shim
            seq_lens_cpu_upper_bound=seq_cpu,
            num_reqs=n_reqs,
            num_actual_tokens=P,
            max_query_len=max_q,
            max_seq_len=p1 + 1,
            block_table_tensor=blk,
            slot_mapping=slots,
            causal=True,
            is_prefilling=is_pref,
            positions=positions,
        )
        for ag in self.attn_groups[gid]:
            md = ag.get_metadata_builder(0).build(common_prefix_len=0, common_attn_metadata=cm)
            for ln in ag.layer_names:
                attn_metadata[ln] = md
                slot_by_layer[ln] = slots
        kv_group_shapes[gid] = tuple(blk.shape)

    ctx = _CorrectCtx(sb=sb, positions=positions, window=(s, e), final=final,
                      replay=replay, mamba_blocks=mamba_blocks, commit=commit,
                      commit_end=int(commit_end))

    # >>> AppCorr/M (GLM-5.3 sparse MLA): rewrite the indexer's pooled K + tail caches >>>
    # The 11 `Glm5NextMLAAttention` layers own two more caches than the MLA latent, and the stock
    # write path is wrong for a pseudo-sequence batch (see `glm53_indexer._hook`).  `_glm53_ctx`
    # is a no-op context on every other model.
    glm53_ctx = _glm53_correct_context(self, req_id, sb, positions, p1 + 1)
    # <<< AppCorr/M <<<

    # CUDA-graph dispatch (full depth only: the partial-depth walks run the eager layer loop)
    cg_mode, cg_desc, n_pad = CUDAGraphMode.NONE, None, P
    if glm53_ctx is not None and not GLM53_CUDAGRAPH:
        # AppCorr/M: no CUDA graph on a sparse-MLA decoder by default.  The two reasons first
        # recorded here were (i) the per-round indexer rewrite writes two caches from Python and
        # (ii) the graph path pads through `self.mrope_positions.gpu`, which this model does not
        # read.  Re-read 2026-09-14: (ii) is a buffer choice (handled below by `uses_mrope`), and
        # (i) does not bind under breakable-cudagraph replay, where the decorated ops -- and the
        # hooks patched onto them -- run eagerly every replay (see GLM53_CUDAGRAPH).  Kept OFF
        # until the graph path is gated on this family; APPCORR_GLM53_CUDAGRAPH=1 opts in.
        pass
    elif CUDAGRAPH and not from_frontier and tuple(layers) == (0, num_layers(self)):
        cg_mode, cg_desc = self.cudagraph_dispatcher.dispatch(
            num_tokens=P, uniform_decode=False, invalid_modes={CUDAGraphMode.FULL})
        if cg_mode == CUDAGraphMode.PIECEWISE:
            n_pad = int(cg_desc.num_tokens)
            # the captured graphs read the runner's persistent input buffers (the same ones
            # `execute_model` refreshes for every scheduled step, so nothing leaks)
            emb_buf = self.inputs_embeds.gpu
            emb_buf[:P].copy_(inputs_embeds)
            emb_buf[P:n_pad].zero_()
            if self.uses_mrope:                      # Qwen: [3, N] M-RoPE positions
                pos_buf = self.mrope_positions.gpu
                pos_buf[:, :P].copy_(positions_gpu)
                pos_buf[:, P:n_pad].zero_()
                positions_gpu = pos_buf[:, :n_pad]
            else:                                    # GLM-5.3: flat [N] positions
                pos_buf = self.positions.gpu
                pos_buf[:P].copy_(positions_gpu)
                pos_buf[P:n_pad].zero_()
                positions_gpu = pos_buf[:n_pad]
            inputs_embeds = emb_buf[:n_pad]
            if PAD_EAGER:   # diagnostic: same padding, eager execution
                cg_mode, cg_desc = CUDAGraphMode.NONE, None
        else:
            cg_mode, cg_desc = CUDAGraphMode.NONE, None
    _rf_meta.__exit__(None, None, None)
    prev_mode, prev_ctx = _ST.mode, _ST.ctx
    _ST.mode, _ST.ctx = MODE_CORRECT, ctx
    glm53_prev = _glm53_set_context(glm53_ctx)          # AppCorr/M
    try:
        with torch.inference_mode(), torch.profiler.record_function("appcorr.layers"), \
                set_forward_context(
            attn_metadata, self.vllm_config, num_tokens=n_pad,
            cudagraph_runtime_mode=cg_mode, batch_descriptor=cg_desc,
            slot_mapping=slot_by_layer,
        ):
            out = _run_layers(self, layers, positions_gpu, inputs_embeds, hidden_in)
    finally:
        _ST.mode, _ST.ctx = prev_mode, prev_ctx
        _glm53_set_context(glm53_prev)                  # AppCorr/M
    if store_frontier:
        assert isinstance(out, tuple), "store_frontier at full depth"
        sb.store_frontier(positions, *out)
    if fuse:
        assert torch.is_tensor(out) and final, (type(out), final)
        sb.hold_hidden = out[P - 1].clone()      # row N-1 (last of the batch, before padding)
    info = {"mamba_blocks": mamba_blocks, "debug": ctx.debug,
            "cudagraph": n_pad if (cg_mode == CUDAGraphMode.PIECEWISE or PAD_EAGER) else 0}
    if glm53_ctx is not None:                          # AppCorr/M
        info["glm53_indexer"] = dict(glm53_ctx.debug)
        info["glm53_indexer_dense"] = bool(glm53_ctx.dense)
        info["glm53_indexer_mb"] = glm53_ctx.buf.nbytes() / 2 ** 20
    return info


# ---------------------------------------------------------------------------------------------
# introspection helper (gates)
# ---------------------------------------------------------------------------------------------


def appcorr_snapshot(self: GPUModelRunner, req_id: str, positions: torch.Tensor) -> dict:
    """KV rows at the given prompt positions + the request's mamba block, as CPU tensors."""
    positions = positions.to(self.device, torch.int64)
    mamba_gids = _mamba_group_ids(self)
    ctx = self.vllm_config.compilation_config.static_forward_context
    kv: dict[str, torch.Tensor] = {}
    layouts: dict[str, str] = {}
    for gid, group in enumerate(self.kv_cache_config.kv_cache_groups):
        if gid in mamba_gids or _is_kpool_tail_group(self, gid):
            continue
        slots = None
        # One kv-cache group can carry several attention groups with DIFFERENT backends, so the
        # backend is read per attention group, not from `attn_groups[gid][0]`.
        for ag in self.attn_groups[gid]:
            backend = ag.backend.__name__
            if "Indexer" in backend or "KpoolTail" in backend:
                # GLM-5.3 sparse layers own three caches: the MLA latent (an ordinary attention
                # group, read below) plus the kpool indexer's fp8 k_cache (`[num_blocks,
                # num_states, head_dim + 4]` uint8, ONE entry per kpool positions) and its tail
                # buffer (one block per request).  Neither is addressable per token position, so
                # they are read through `glm53_indexer`'s own accessors (the MLA gate's
                # `snapshot_indexer`), never here.  Leg 3 of the TP gate died on exactly this
                # entry (B200-8, 2026-09-13 02:31 KST).  `_slot_mapping` is NOT evaluated for
                # them: their block tables are pool-granular and the token formula reads past
                # the row.
                for ln in ag.layer_names:
                    layouts[ln] = f"skipped:{backend}"
                continue
            if slots is None:
                _, slots = _slot_mapping(self, gid, req_id, positions)
            for ln in ag.layer_names:
                cache = ctx[ln].kv_cache
                if isinstance(cache, (list, tuple)):
                    cache = cache[0]
                if cache.dim() == 3 and "MLA" in backend:
                    # (num_kernel_blocks, kernel_block_size, kv_lora_rank): the MLA latent has
                    # no head axis at all -- GLM-5.3 sparse layers show (4930, 64, 512) under
                    # FlashInferMLASparseTRTLLMBackend (B200-8, 2026-09-13 02:50 KST), paged
                    # at 64 while the same layer's indexer k_cache is paged at 32.  `slots`
                    # are ABSOLUTE token slots (block * block_size + offset, `_slot_mapping`),
                    # so they are re-split at THIS view's own page width, not the block
                    # table's: a kernel-block split is a uniform unflatten of the manager
                    # block (`kv_cache_interface.py` "grouping is a pure view").
                    kb = cache.shape[1]
                    rows = cache[slots // kb, slots % kb]
                    layouts[ln] = f"mla_bnc@{kb}"
                elif cache.dim() == 4 and ("FlashInfer" in backend or "MLA" in backend):
                    # [B, H, N, C] = (num_blocks, num_kv_heads, block_size, 2*head_size) for
                    # FlashInfer (flashinfer.py:2528); on main every attention layer is this
                    # logical view (`create_kv_cache_views`), the MLA latent as H=1, C=head.
                    # Same absolute-slot re-split at the view's own page width (== bs today).
                    kb = cache.shape[2]
                    rows = cache[slots // kb, :, slots % kb, :]
                    layouts[ln] = f"bhnc@{kb}"
                elif cache.dim() == 5 and cache.shape[0] == 2:
                    # (2, num_blocks, block_size, num_kv_heads, head_size)  -- FlashAttention
                    kb = cache.shape[2]
                    rows = cache[:, slots // kb, slots % kb]
                    layouts[ln] = f"fa_2bnhd@{kb}"
                else:
                    raise RuntimeError(
                        f"unrecognised kv cache layout {tuple(cache.shape)} ({backend}) for {ln}")
                # int8/uint8 views (fp8 caches on main) are kept as bytes: the gate compares
                # them bitwise; a float cast of raw bytes would be neither a value nor a byte.
                kv[ln] = (rows.cpu().clone() if rows.dtype in (torch.int8, torch.uint8)
                          else rows.float().cpu())
                layouts[ln] += f":{str(rows.dtype).replace('torch.', '')}"
    mamba: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    blocks = _mamba_blocks(self, req_id, mamba_gids)
    for ln, blk in blocks.items():
        c, ssm = ctx[ln].kv_cache
        mamba[ln] = (c[blk].float().cpu().clone(), ssm[blk].float().cpu().clone())
    return {"kv": kv, "mamba": mamba, "layouts": layouts, "mamba_blocks": blocks}


# ---------------------------------------------------------------------------------------------


def install() -> None:
    """Patch the GDN core and add the runner's correct step.

    Deviations from `docs/memo/vllm_interleaved_design.md` §2:
      * the SideBuffer is created by ``StreamingLLM.open(..., correct=True)`` directly on the
        runner (in-process engine core) instead of riding on ``NewRequestData.appcorr_stream`` --
        no wire/scheduler change is needed and the buffer exists before the first schedule either
        way;
      * a round re-scans ``[ckpt_end, window_end)`` in ONE ``chunk_gated_delta_rule`` call rather
        than "advance the checkpoint to s, then scan [s,e)": the two differ by an extra cast of the
        intermediate state, and the single call is what a stock chunked prefill does;
      * the G1 "passthrough" mode of the memo is a *replay* mode: the approx pass also captures the
        GDN layer output, and the correct step replays it for the corrected rows.  Writing nothing
        (as the memo says) would leave ``core_attn_out`` zero and corrupt the residual stream, so
        the softmax path could not be isolated at all;
      * the recurrent half has TWO seams, chosen by `_gdn_flavor` (structure, not model name):
        `QwenGatedDeltaNetAttention._forward_core` and, for GLM-5.3-Flash,
        `Glm5NextLinearAttention._forward` (`_kda_forward_patch`).  Same side buffer, same
        checkpoint chain, same write-back; the KDA re-scan runs one MERGED q|k|v conv and
        `chunk_kda_with_fused_gate` (gate in-kernel from A_log/dt_bias, beta SIGMOIDED fp32);
      * the GDN side-buffer half is installed LAZILY (`install_gdn_patch`, 2026-09-12).  This
        function is a vLLM general plugin: it runs before a model exists, so it cannot know
        whether the served decoder is hybrid.  It therefore patches the runner only, and the
        `QwenGatedDeltaNetAttention` import + `_forward_core` patch happen at the first
        `check_gdn_path` / `execute_model` that sees a model WITH recurrent layers.  A
        pure-softmax decoder (GLM-4.6V: 46 softmax GQA layers) never imports it.
    """
    global _ORIG_PREPARE_INPUTS, _ORIG_EXECUTE_MODEL, _ORIG_MODEL_FORWARD
    if getattr(GPUModelRunner, "_appcorr_correct_patched", False):
        return

    _ORIG_PREPARE_INPUTS = GPUModelRunner._prepare_inputs
    _ORIG_EXECUTE_MODEL = GPUModelRunner.execute_model
    _ORIG_MODEL_FORWARD = GPUModelRunner._model_forward
    GPUModelRunner._prepare_inputs = _prepare_inputs
    GPUModelRunner.execute_model = _execute_model
    GPUModelRunner._model_forward = _model_forward
    GPUModelRunner.appcorr_arm_final = appcorr_arm_final
    GPUModelRunner.appcorr_take_final_info = appcorr_take_final_info
    GPUModelRunner.appcorr_correct_step = appcorr_correct_step
    GPUModelRunner.appcorr_rows_step = appcorr_rows_step
    GPUModelRunner.appcorr_staged_correct = appcorr_staged_correct
    GPUModelRunner.appcorr_open_walk = appcorr_open_walk
    GPUModelRunner.appcorr_snapshot = appcorr_snapshot
    GPUModelRunner._appcorr_correct_patched = True
