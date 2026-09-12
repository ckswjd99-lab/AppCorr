"""Interleaved k<1 LLM correction inside vLLM (engine side).

The streaming form (``request.py`` / ``scheduler.py`` / ``runner_patch.py``) pushes the prompt in
bands and prefills each band once.  The *interleaved* form pushes the WHOLE prompt at t=0 with the
image rows at their approximate value, lets the stock engine prefill it (the "approx pass"), and
then rewrites individual prompt rows in place as their corrected embeddings arrive:

  * softmax layers -- the corrected rows are re-run as |P| pseudo-sequences of length 1 whose
    ``seq_len`` is ``pos+1``; the stock kernels write their K/V into the request's own slots
    (write-before-read inside the round is what ``unified_kv_cache_update`` before the attention
    call gives us) and read every key at ``key_pos <= query_pos``;
  * Gated DeltaNet layers cannot be rewritten in place (the recurrent state is a running product),
    so the layer's *pre-conv* inputs (``mixed_qkv``, ``b``, ``a``) are captured for every prompt row
    during the approx pass into a per-request side buffer, the corrected rows overwrite their side
    buffer entries, and the round re-scans the window ``[ckpt_end, window_end)`` from a checkpoint
    of the recurrent state.  Total re-scan work over all rounds is one pass over the prompt.

Semantics follow ``docs/memo/interleaved_correction_contract.md`` (round r corrects P_r only; rows
outside P_r keep their captured value; the checkpoint chain is what persists earlier rounds).

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
    """Per-request store of every GDN layer's pre-conv inputs for every prompt row.

    Keyed by the layer's ``prefix`` (its name in ``static_forward_context``).  Rows are prompt
    positions ``[0, n)``; row ``n-1`` is only filled when the hold-back is released.
    """
    n: int
    device: torch.device
    lo: int = 0                                     # first image row
    hi: int = 0                                     # one past the last image row (0: unknown)
    capture_out: bool = False                       # also store the GDN layer output (G1 replay)
    # depth staging: residual stream of every image row at the walk frontier, and where it is
    fr_h: Optional[torch.Tensor] = None
    fr_r: Optional[torch.Tensor] = None
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
        for t in (self.fr_h, self.fr_r):
            if t is not None:
                tot += t.numel() * t.element_size()
        return tot

    def frontier_rows(self, positions: torch.Tensor):
        assert self.fr_h is not None, "frontier buffers not initialised"
        return self.fr_h[positions], self.fr_r[positions]

    def store_frontier(self, positions: torch.Tensor, hs: torch.Tensor, res: torch.Tensor) -> None:
        if self.fr_h is None:
            self.fr_h = torch.zeros((self.n, hs.shape[1]), dtype=hs.dtype, device=self.device)
            self.fr_r = torch.zeros((self.n, res.shape[1]), dtype=res.dtype, device=self.device)
        self.fr_h[positions] = hs.to(self.fr_h.dtype)
        self.fr_r[positions] = res.to(self.fr_r.dtype)


def check_gdn_path(vllm_config) -> list[str]:
    """The GDN layers must route through ``_forward_core`` (the method this module patches).

    ``VLLM_GDN_DECODE_KERNEL`` defaults to ``cuda`` in vllm 0.28.0, which makes ``forward_cuda``
    call ``qwen_gdn_attention_core_fused_norm_packed`` -> ``_forward_core_fused_norm_packed``
    instead -- the capture and correct hooks would silently never fire.  (The design memo says
    this path "is not patched -- assert it is off"; it is ON by default, so every run of the
    interleaved path must set ``VLLM_GDN_DECODE_KERNEL=triton``.)"""
    from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
    ctx = vllm_config.compilation_config.static_forward_context
    gdn = [ln for ln, m in ctx.items() if isinstance(m, GatedDeltaNetAttention)]
    assert gdn, "no Gated DeltaNet layers in this model"
    for ln in gdn:
        assert not getattr(ctx[ln], "enable_fused_gdn_decode", False), (
            f"{ln}: VLLM_GDN_DECODE_KERNEL=cuda bypasses _forward_core "
            "(qwen_gdn_linear_attn.py:1781); run with VLLM_GDN_DECODE_KERNEL=triton")
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

_ORIG_FORWARD_CORE = None


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


def _conv_window(layer, x: torch.Tensor, scratch: torch.Tensor) -> torch.Tensor:
    """Stock ``causal_conv1d_fn`` over a [L, C] slice with a zero initial state.

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
    conv_weights = layer.conv1d.weight.view(
        layer.conv1d.weight.size(0), layer.conv1d.weight.size(2)
    )
    out = causal_conv1d_fn(
        x.transpose(0, 1),
        conv_weights,
        layer.conv1d.bias,
        activation=layer.activation,
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

    Mirrors the stock prefill branch of ``_forward_core`` (qwen_gdn_linear_attn.py:1345-1520):
    causal conv with the layer's activation -> ``fused_post_conv_prep`` -> ``chunk_gated_delta_rule``
    with ``use_qk_l2norm_in_kernel=False``.  Returns the window output [end-start, HV, V].
    """
    with torch.profiler.record_function("appcorr.rescan"):
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
    prev, _ST.mode = _ST.mode, MODE_CAPTURE
    try:
        return _ORIG_EXECUTE_MODEL(self, scheduler_output, intermediate_tensors)
    finally:
        _ST.mode = prev
        _ST.captures = []
        _ST.stub = None
        _ST.skip_forward = False


def _mamba_group_ids(self: GPUModelRunner) -> list[int]:
    from vllm.v1.kv_cache_interface import MambaSpec
    return [gid for gid, g in enumerate(self.kv_cache_config.kv_cache_groups)
            if isinstance(g.kv_cache_spec, MambaSpec)]


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


def _mamba_blocks(self: GPUModelRunner, req_id: str, mamba_gids: list[int]) -> dict[str, int]:
    """GDN layer prefix -> the request's recurrent-state block, per mamba KV-cache group.

    Qwen3.5 splits its GDN layers over several mamba groups (4B: 3 groups + 1 attention group),
    so a single block id is not enough."""
    out: dict[str, int] = {}
    for gid in mamba_gids:
        blk = int(_block_row(self, req_id, gid)[0])
        for ln in self.kv_cache_config.kv_cache_groups[gid].layer_names:
            out[ln] = blk
    return out


def _decoder(self: GPUModelRunner):
    """The text decoder (`Qwen3_5Model`): its `layers` are what a partial-depth step walks."""
    m = self.model
    m = m.unwrap() if hasattr(m, "unwrap") else m
    lm = getattr(m, "language_model", m)
    lm = getattr(lm, "model", lm)
    assert hasattr(lm, "layers"), type(lm)
    return lm


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


def _run_layers(self: GPUModelRunner, layers, positions_gpu, inputs_embeds, hidden_in):
    """Decoder layers ``[a, b)`` on the batch: the compiled model for the full depth (the MVP's
    path, bit-for-bit what it ran before), an eager loop over the layer modules otherwise.
    Returns the residual stream at layer ``b``; at full depth the model's output (the normed
    final hidden states, read by the fused hold-back only)."""
    a, b = layers
    lm = _decoder(self)
    L = len(lm.layers)
    if a == 0 and b == L:
        return self.model(input_ids=None, positions=positions_gpu,
                          intermediate_tensors=None, inputs_embeds=inputs_embeds)
    if a == 0:
        hs, res = inputs_embeds, None
    else:
        hs, res = hidden_in
    for layer in lm.layers[a:b]:
        hs, res = layer(positions=positions_gpu, hidden_states=hs, residual=res)
    return None if b == L else (hs, res)


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

    # M-RoPE positions of exactly these rows (mirrors _prepare_inputs' mrope path)
    assert self.uses_mrope, "non-M-RoPE models not handled"
    mrope = req_state.mrope_positions
    assert mrope is not None and mrope.shape[1] >= sb.n, (None if mrope is None else mrope.shape)
    positions_gpu = mrope[:, pos_cpu].to(dev, torch.int64)

    mamba_gids = _mamba_group_ids(self)
    assert mamba_gids, "no mamba/GDN kv cache group found"
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
    for gid, group in enumerate(self.kv_cache_config.kv_cache_groups):
        if gid in mamba_gids:
            continue
        row, slots = _slot_mapping(self, gid, req_id, positions)
        blk = row.unsqueeze(0).expand(n_reqs, -1).contiguous()
        cm = CommonAttentionMetadata(
            query_start_loc=qsl_cpu.to(dev),
            query_start_loc_cpu=qsl_cpu,
            seq_lens=seq_cpu.to(dev),
            _seq_lens_cpu=seq_cpu,
            _num_computed_tokens_cpu=ncomp_cpu,
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

    # CUDA-graph dispatch (full depth only: the partial-depth walks run the eager layer loop)
    cg_mode, cg_desc, n_pad = CUDAGraphMode.NONE, None, P
    if CUDAGRAPH and not from_frontier and tuple(layers) == (0, num_layers(self)):
        cg_mode, cg_desc = self.cudagraph_dispatcher.dispatch(
            num_tokens=P, uniform_decode=False, invalid_modes={CUDAGraphMode.FULL})
        if cg_mode == CUDAGraphMode.PIECEWISE:
            n_pad = int(cg_desc.num_tokens)
            # the captured graphs read the runner's persistent input buffers (the same ones
            # `execute_model` refreshes for every scheduled step, so nothing leaks)
            emb_buf, pos_buf = self.inputs_embeds.gpu, self.mrope_positions.gpu
            emb_buf[:P].copy_(inputs_embeds)
            emb_buf[P:n_pad].zero_()
            pos_buf[:, :P].copy_(positions_gpu)
            pos_buf[:, P:n_pad].zero_()
            inputs_embeds, positions_gpu = emb_buf[:n_pad], pos_buf[:, :n_pad]
            if PAD_EAGER:   # diagnostic: same padding, eager execution
                cg_mode, cg_desc = CUDAGraphMode.NONE, None
        else:
            cg_mode, cg_desc = CUDAGraphMode.NONE, None
    _rf_meta.__exit__(None, None, None)
    prev_mode, prev_ctx = _ST.mode, _ST.ctx
    _ST.mode, _ST.ctx = MODE_CORRECT, ctx
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
    if store_frontier:
        assert isinstance(out, tuple), "store_frontier at full depth"
        sb.store_frontier(positions, out[0], out[1])
    if fuse:
        assert torch.is_tensor(out) and final, (type(out), final)
        sb.hold_hidden = out[P - 1].clone()      # row N-1 (last of the batch, before padding)
    return {"mamba_blocks": mamba_blocks, "debug": ctx.debug,
            "cudagraph": n_pad if (cg_mode == CUDAGraphMode.PIECEWISE or PAD_EAGER) else 0}


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
        if gid in mamba_gids:
            continue
        _, slots = _slot_mapping(self, gid, req_id, positions)
        bs = self.input_batch.block_table[gid].block_size
        blk_i, off_i = slots // bs, slots % bs
        backend = self.attn_groups[gid][0].backend.__name__
        for ln in group.layer_names:
            cache = ctx[ln].kv_cache
            if isinstance(cache, (list, tuple)):
                cache = cache[0]
            if cache.dim() == 4 and "FlashInfer" in backend:
                # (num_blocks, num_kv_heads, block_size, 2*head_size)  -- flashinfer.py:2528
                rows = cache[blk_i, :, off_i, :]
                layouts[ln] = "flashinfer_bhn2d"
            elif cache.dim() == 5 and cache.shape[0] == 2:
                # (2, num_blocks, block_size, num_kv_heads, head_size)  -- FlashAttention
                rows = cache[:, blk_i, off_i]
                layouts[ln] = "fa_2bnhd"
            else:
                raise RuntimeError(
                    f"unrecognised kv cache layout {tuple(cache.shape)} ({backend}) for {ln}")
            kv[ln] = rows.float().cpu()
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
        the softmax path could not be isolated at all.
    """
    global _ORIG_FORWARD_CORE, _ORIG_PREPARE_INPUTS, _ORIG_EXECUTE_MODEL, _ORIG_MODEL_FORWARD
    if getattr(GPUModelRunner, "_appcorr_correct_patched", False):
        return
    from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
        QwenGatedDeltaNetAttention)

    _ORIG_FORWARD_CORE = QwenGatedDeltaNetAttention._forward_core
    QwenGatedDeltaNetAttention._forward_core = _forward_core_patch

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
