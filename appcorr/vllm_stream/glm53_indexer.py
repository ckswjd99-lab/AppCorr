"""GLM-5.3-Flash sparse-MLA indexer cache rewrite for the interleaved correct step.

Item 3 of ``docs/memo/glm53_correct_design.md``.  The 11 ``Glm5NextMLAAttention`` layers
(``layer_types == "deepseek_sparse_attention"``: 3, 7, 11, ..., 43) own THREE caches, not one:

  1. the MLA latent, 512 B/token, written from ``forward_context.slot_mapping`` by
     ``MLAAttention.forward`` (`vllm/model_executor/layers/attention/mla_attention.py:797-830`)
     -- the pseudo-sequence path in ``correct._correct_sub`` already drives this;
  2. ``indexer.k_cache`` (``Glm5NextIndexerCache``, `models/glm5next/nvidia/attention.py:78`):
     ONE fp8 entry per ``index_kpool``(=4) consecutive tokens, 132 B per pooled entry
     (128 fp8 + one fp32 scale), addressed at POOL granularity;
  3. ``indexer.tail_cache`` (``Glm5NextTailCache``, `attention.py:159`): one block of ``kpool``
     slots per request holding the raw bf16 K and gate score of the request's last ``kpool``
     tokens, indexed by ``pos % kpool``.

Rewriting a prompt row ``p`` therefore has to rewrite (2) for the pool ``p // kpool`` -- which
needs the CURRENT indexer inputs of p's three pool siblings, not just p's -- and (3) when p is
among the request's last ``kpool`` rows.  That is what this module does.

Where the indexer inputs come from
----------------------------------
``Indexer.forward`` (`attention.py:316-401`) computes, per token, from the layer's hidden state
``h``:

    kw          = wk_weights_proj(h)                      # fused [head_dim + n_head] GEMM
    k           = layernorm(kw[:, :head_dim])             # k_norm, eps 1e-6, fp32 then cast back
    weights     = h.float() @ wk_weights_proj.weight[head_dim:].T.float()
    gate_score  = F.linear(h, index_kpool_compress_gate)  # [T, head_dim]
    q           = fwht128_quant_fp8(wq_b(q_c))            # query side; NOT cached

and hands ``(k, gate_score)`` to ``indexer_op``.  Only ``k`` and ``gate_score`` reach the caches,
so the side buffer stores exactly those two, NOT the hidden state:

    512 B per token per sparse layer  (k 128 x bf16 = 256 B, gate 128 x bf16 = 256 B)
      + 1 B                           (a per-row "captured" flag, so a missing pool sibling
                                       asserts instead of pooling zeros)         = 513 B
    x 11 sparse layers                = 5.51 KiB per token
    x 8192 tokens                     = 44.1 MiB per request

Storing the hidden state instead would be 4096 x 2 = 8 KiB per token per layer (16x more) and
would still have to re-run the two GEMMs at rewrite time, so k+gate wins on both axes.
``indexer_inputs_from_hidden`` below is the explicit recompute the design memo describes; the
engine hook does not use it (it intercepts ``indexer_op``'s own arguments, which are the same
tensors bit-for-bit and cost no extra GEMM), but the gate and the tests do, to prove the two agree.

Pool math (ported from ``models/glm5next/nvidia/ops/kpool_compress.py:137-258``)
-------------------------------------------------------------------------------
Per pool, over its ``kpool`` slots, ALL IN FP32 and PER CHANNEL (the softmax is over the pool
axis, separately for each of the 128 dims -- not over the head dim):

    score[s, d] = gate[s, d] + ape[s, d]
    prob        = exp(score - max_s score)
    x[d]        = sum_s k[s, d] * prob[s, d] / sum_s prob[s, d]
    x           = bf16(x)                       # kernel rounds to bf16 and back
    x           = bf16(hadamard128(x))          # Sylvester H_128 / sqrt(128)
    absmax      = max(|x|, 1e-4)
    scale       = 2 ** ceil(log2(absmax / 448))  (round_scale, i.e. scale_fmt="ue8m0")
    entry       = fp8_e4m3(clamp(x / scale, -448, 448))

``ape`` is ``indexer.index_kpool_compress_ape`` ``[kpool, head_dim]`` fp32 (the per-slot position
bias); it is what makes the pool order-sensitive, so the four siblings must be presented in
position order.

Cache layouts
-------------
``k_cache.kv_cache`` is ``[num_blocks, num_states, head_dim + 4]`` uint8 with the fp8 values and
the fp32 scales in SEPARATE regions of each page (`kpool_compress.py:229-248`):

    page_bytes  = kv_cache.stride(0)            # = num_states * (head_dim + 4)
    value bytes = page * page_bytes + slot * head_dim              .. + head_dim
    scale bytes = page * page_bytes + num_states * head_dim + slot * 4

``tail_cache.kv_cache`` is flat ``[num_blocks, 2, kpool, head_dim]`` bf16 with ``[blk, 0, j]`` the
raw K and ``[blk, 1, j]`` the gate score (`kpool_compress.py:371-410`).

Pool -> physical slot is ``get_compressed_slot_mapping``
(`vllm/v1/attention/backends/mla/compressor_utils.py:60-84`) reduced to one request:

    pool  = pos // kpool                        (valid only when (pos + 1) % kpool == 0)
    slot  = block_row[pool // num_states] * num_states + pool % num_states

with ``num_states = k_cache.kv_cache.shape[1]`` -- the KERNEL page width (32 on main, 128
tokens), NOT ``spec.block_size // kpool`` (the manager block, 2176 tokens = 544 pools, is split
uniformly into kernel pages and ``block_row`` is already at kernel granularity).

Short-prefill / dense regime
----------------------------
``index_topk`` is **2048** for this checkpoint and ``index_kpool`` is 4, so ``topk_tokens = 2048``
and the top-k buffer is ``[max_num_batched_tokens, 2176]`` int32 (`model.py:588-606`:
``ceil((2048 + 3) / 128) * 128``).  The indexer skips sparse scoring entirely and fills EXACT
causal indices when ``max_seq_len <= topk_tokens``
(`sparse_attn_indexer_kpool.py:200-216` for the decode-shaped batch our correct step builds,
`:421-449` for the prefill-shaped one).  Our correct step's ``max_seq_len`` is ``p1 + 1`` where
``p1`` is the LAST corrected row, so:

    prompt (or last corrected row) < 2048 tokens  -> dense, exact, no scoring
    V*Bench ~4k, our 8192 cap                     -> SPARSE; the dense fill does NOT apply

so the dense path is taken only where the predicate holds, and it is checked per round, never
assumed.  BOTH regimes are implemented: below 2048 the round fills exact causal indices and skips
scoring entirely; above it the stock scoring runs with ``skip_k_cache_insert=True`` over the
pooled cache this module has already rewritten.  See the block comment above ``pool_len_of`` for
the visibility invariant the sparse round relies on, and
``pseudo_sequence_indexer_metadata`` for why P one-token pseudo-sequences sharing one block-table
row are just P decode requests as far as every builder and DeepGEMM are concerned.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F

import os as _os

# Kill-switch for the sparse regime (APPCORR_GLM53_SPARSE=0 makes a >2048 round raise instead of
# running, so a gate can isolate the dense path).  APPCORR_GLM53_CHECK gates the validation
# syncs: `IndexerSideBuffer.rows`' coverage assert and the sparse path's scoreable-pool assert
# cost ~3 device syncs per sparse layer per round (33 per round over the 11 layers).  Keep it on
# until `glm53_mla_gate.py --gate mla --expect-sparse` passes, then measure and decide.
SPARSE_ENABLED = _os.environ.get("APPCORR_GLM53_SPARSE", "1") == "1"
CHECK = _os.environ.get("APPCORR_GLM53_CHECK", "1") == "1"

INDEX_HEAD_DIM = 128
FP8_MAX = 448.0
FP8_DTYPE = torch.float8_e4m3fn
HADAMARD_SCALE = 0.08838834764831845        # 1 / sqrt(128), exact in fp32


# ---------------------------------------------------------------------------------------------
# module detection -- by class MRO / attributes, never by model name
# ---------------------------------------------------------------------------------------------

def is_mla_layer(mod) -> bool:
    """``mod`` is a multi-head-latent attention module (``Glm5NextMLAAttention`` and friends).

    Keyed on the wrapper vLLM actually installs (``MultiHeadLatentAttentionWrapper``, which every
    MLA model routes through, `vllm/model_executor/layers/mla.py:44`) plus the latent rank, so a
    GQA layer with an ``.mla_attn`` attribute cannot match by accident."""
    w = getattr(mod, "mla_attn", None)
    return w is not None and hasattr(w, "kv_lora_rank") and hasattr(w, "mla_attn")


def sparse_indexer_of(mod):
    """The layer's kpool sparse indexer, or None (a dense-MLA layer has ``indexer is None``)."""
    if not is_mla_layer(mod):
        return None
    ix = getattr(mod.mla_attn, "indexer", None)
    if ix is None:
        return None
    if not hasattr(ix, "index_kpool") or int(getattr(ix, "index_kpool", 1)) <= 1:
        return None            # a non-pooled DSA indexer (DeepSeek V3.2): different cache layout
    return ix


def mla_layers(decoder) -> list[tuple[int, object]]:
    """``[(layer_idx, mla_module)]`` over ``decoder.layers``; empty on a non-MLA decoder."""
    out = []
    for i, layer in enumerate(getattr(decoder, "layers", [])):
        sa = getattr(layer, "self_attn", None)
        if sa is not None and is_mla_layer(sa):
            out.append((i, sa))
    return out


def sparse_layers(decoder) -> list[tuple[int, object, object]]:
    """``[(layer_idx, mla_module, indexer)]`` for the kpool-sparse MLA layers only."""
    out = []
    for i, mod in mla_layers(decoder):
        ix = sparse_indexer_of(mod)
        if ix is not None:
            out.append((i, mod, ix))
    return out


# ---------------------------------------------------------------------------------------------
# reference math (pure torch; runs on CPU)
# ---------------------------------------------------------------------------------------------

def hadamard128(x: torch.Tensor) -> torch.Tensor:
    """Sylvester Hadamard-128 rotation / sqrt(128), butterfly-for-butterfly as ``_hadamard128``.

    ``kpool_compress.py:26-45`` runs seven stages ``(GROUPS, STRIDE)`` =
    (64,1) (32,2) (16,4) (8,8) (4,16) (2,32) (1,64); each views the 128 lanes as
    ``(GROUPS, 2, STRIDE)`` and replaces ``(a, b)`` by ``(a + b, a - b)``.  Keeping the same stage
    order (and fp32 throughout) keeps the floating-point op order identical to the kernel's, which
    is what a bitwise GPU comparison needs.  ``x`` is ``[..., 128]``; returns fp32."""
    assert x.shape[-1] == 128, x.shape
    lead = x.shape[:-1]
    y = x.float().reshape(-1, 128)
    for groups, stride in ((64, 1), (32, 2), (16, 4), (8, 8), (4, 16), (2, 32), (1, 64)):
        v = y.reshape(-1, groups, 2, stride)
        a, b = v[:, :, 0, :], v[:, :, 1, :]
        y = torch.stack((a + b, a - b), dim=2).reshape(-1, 128)
    return (y * HADAMARD_SCALE).reshape(*lead, 128)


def pool_compress(slot_k: torch.Tensor, slot_gate: torch.Tensor, ape: torch.Tensor,
                  *, round_scale: bool = True) -> tuple[torch.Tensor, torch.Tensor]:
    """``_kpool_softmax_rotate_write_cache_kernel`` (kpool_compress.py:137-250) in torch.

    Args:
        slot_k:    ``[P, kpool, head_dim]`` bf16 -- raw per-token indexer K, in position order.
        slot_gate: ``[P, kpool, head_dim]`` bf16 -- the per-token gate score, same order.
        ape:       ``[kpool, head_dim]`` fp32 -- ``indexer.index_kpool_compress_ape``.
    Returns:
        ``(entry [P, head_dim] float8_e4m3fn, scale [P] fp32)``.

    The pool softmax is PER CHANNEL over the ``kpool`` axis.  The accumulation is written as an
    explicit slot loop (not ``.sum(dim=1)``) so the fp32 add order matches the kernel's
    ``tl.static_range`` loop."""
    assert slot_k.shape == slot_gate.shape and slot_k.ndim == 3, (slot_k.shape, slot_gate.shape)
    assert ape.shape == slot_k.shape[1:], (ape.shape, slot_k.shape)
    assert ape.dtype == torch.float32, ape.dtype
    kpool = slot_k.shape[1]

    score = [slot_gate[:, s, :].float() + ape[s].float() for s in range(kpool)]
    mx = score[0]
    for s in range(1, kpool):
        mx = torch.maximum(mx, score[s])
    acc = torch.zeros_like(mx)
    den = torch.zeros_like(mx)
    for s in range(kpool):
        prob = torch.exp(score[s] - mx)
        den = den + prob
        acc = acc + slot_k[:, s, :].float() * prob
    x = (acc / den).to(torch.bfloat16).float()
    x = hadamard128(x).to(torch.bfloat16).float()

    absmax = x.abs().amax(dim=-1).clamp_min(1e-4)
    if round_scale:
        scale = torch.exp2(torch.ceil(torch.log2(absmax * (1.0 / FP8_MAX))))
    else:
        scale = absmax * (1.0 / FP8_MAX)
    q = (x / scale[:, None]).clamp(-FP8_MAX, FP8_MAX)
    return q.to(FP8_DTYPE), scale.float()


def indexer_inputs_from_hidden(indexer, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(k, gate_score)`` recomputed from a layer's hidden state, exactly as ``Indexer.forward``.

    ``attention.py:316-401``: one fused ``wk_weights_proj`` GEMM whose first ``head_dim`` columns
    are K, then ``k_norm`` (a ``LayerNorm``, eps 1e-6) in fp32 cast back to the input dtype, then
    ``F.linear(h, index_kpool_compress_gate)``.  ``qk_rope_head_dim == 0`` for GLM-5.3 (config
    ``qk_rope_head_dim: 0``, ``mla_use_nope: true``) so the rope split/rotary/cat is skipped --
    the ``rope_dim > 0`` branch is asserted away rather than silently ignored."""
    assert int(getattr(indexer, "rope_dim", 0)) == 0, (
        "indexer_inputs_from_hidden does not implement the rope branch "
        f"(rope_dim={indexer.rope_dim}); GLM-5.3 has qk_rope_head_dim=0")
    hd = int(indexer.head_dim)
    kw = indexer.wk_weights_proj(hidden)[0]
    k = kw[:, :hd]
    kn = indexer.k_norm
    k = F.layer_norm(k.float(), (hd,), kn.weight, kn.bias, kn.eps).type_as(k)
    gate = F.linear(hidden, indexer.index_kpool_compress_gate)
    return k, gate


def fill_causal_topk(rows: torch.Tensor, positions: torch.Tensor) -> None:
    """``_fill_causal_indices`` (sparse_attn_indexer_kpool.py:193-198), in place.

    ``rows`` is ``topk_indices_buffer[:P]`` ``[P, width]`` int32; row i becomes
    ``[0, 1, ..., pos_i, -1, -1, ...]``.  This IS the ``key_pos <= query_pos`` contract of
    ``docs/memo/vllm_interleaved_design.md`` expressed in the sparse layer's index space."""
    rng = torch.arange(rows.shape[1], device=rows.device, dtype=torch.int32)
    pos = positions.to(rows.device, torch.int32)
    rows[:] = rng[None, :]
    rows[rng[None, :] > pos[:, None]] = -1


# ---------------------------------------------------------------------------------------------
# cache addressing + raw access
# ---------------------------------------------------------------------------------------------

def num_states_of(k_cache: torch.Tensor) -> int:
    """Pooled entries per cache PAGE (the kernel block: ``k_cache.shape[1]``, 32 on main)."""
    assert k_cache.ndim == 3 and k_cache.dtype == torch.uint8, (k_cache.shape, k_cache.dtype)
    return int(k_cache.shape[1])


def pool_slots(block_row: torch.Tensor, pools: torch.Tensor, num_states: int,
               kpool: int | None = None, bt_block_size: int | None = None) -> torch.Tensor:
    """Physical pooled-cache slot (page * num_states + state) of each pool index, one request.

    ``block_row`` is the request's indexer-group block-table row (kernel-block mapped), int32.
    Two geometries are live on vLLM main for the same layer (leg 3 attempts 4-5, B200-8
    2026-09-13): the BLOCK TABLE is paged at ``bt_block_size`` TOKENS (64: 34 entries per
    2176-token manager block), while the pooled cache VIEW is paged at ``num_states`` POOLS
    (32 = 128 tokens: 17 pages per manager block).  Neither width is derivable from the other
    or from ``kpool``; both are read.  The address that is right under both is the absolute
    token slot, i.e. `_slot_mapping`'s formula on the block table's own width, divided by
    ``kpool`` -- a manager block is a uniform unflatten into kernel blocks AND into cache pages
    (`create_kv_cache_views`), so absolute token offsets agree between the two.  The 3-argument
    form (block table paged exactly at the cache page, ``get_compressed_slot_mapping``
    compressor_utils.py:76-79) is kept for the CPU tests and for builds where the two coincide."""
    pools = pools.to(block_row.device, torch.int64)
    row = block_row.to(torch.int64)
    if bt_block_size is None or kpool is None:
        blk = row[pools // num_states]
        return blk * num_states + (pools % num_states)
    tok = pools * kpool
    abs_tok = row[tok // bt_block_size] * bt_block_size + (tok % bt_block_size)
    return abs_tok // kpool


def k_cache_write(k_cache: torch.Tensor, slots: torch.Tensor,
                  entry: torch.Tensor, scale: torch.Tensor, head_dim: int = INDEX_HEAD_DIM) -> None:
    """Write ``entry``/``scale`` at the given pooled slots (the kernel's byte layout)."""
    assert entry.dtype == FP8_DTYPE and scale.dtype == torch.float32
    ns = num_states_of(k_cache)
    page, off = (slots // ns).to(torch.int64), (slots % ns).to(torch.int64)
    buf8 = k_cache.view(FP8_DTYPE).reshape(k_cache.shape[0], -1)     # [blocks, ns*(hd+4)]
    buf32 = k_cache.view(torch.float32).reshape(k_cache.shape[0], -1)
    idx = off[:, None] * head_dim + torch.arange(head_dim, device=k_cache.device)
    buf8[page[:, None], idx] = entry
    buf32[page, (ns * head_dim) // 4 + off] = scale


def k_cache_read(k_cache: torch.Tensor, slots: torch.Tensor,
                 head_dim: int = INDEX_HEAD_DIM) -> tuple[torch.Tensor, torch.Tensor]:
    """Inverse of :func:`k_cache_write`."""
    ns = num_states_of(k_cache)
    page, off = (slots // ns).to(torch.int64), (slots % ns).to(torch.int64)
    buf8 = k_cache.view(FP8_DTYPE).reshape(k_cache.shape[0], -1)
    buf32 = k_cache.view(torch.float32).reshape(k_cache.shape[0], -1)
    idx = off[:, None] * head_dim + torch.arange(head_dim, device=k_cache.device)
    return buf8[page[:, None], idx], buf32[page, (ns * head_dim) // 4 + off]


def tail_view(tail_cache: torch.Tensor, kpool: int, head_dim: int = INDEX_HEAD_DIM):
    """``[num_blocks, 2, kpool, head_dim]`` bf16 view of the paged tail cache.

    ``_kpool_tail_seed_kernel`` addresses it as ``(blk * 2 * KPOOL + t % KPOOL) * HEAD_DIM`` for
    the K half and ``+ KPOOL * HEAD_DIM`` for the gate half, i.e. exactly this shape."""
    assert tail_cache.dtype == torch.bfloat16, tail_cache.dtype
    return tail_cache.reshape(-1, 2, kpool, head_dim)


def tail_write(tail_cache: torch.Tensor, tail_block: int, positions: torch.Tensor,
               k: torch.Tensor, gate: torch.Tensor, kpool: int,
               head_dim: int = INDEX_HEAD_DIM) -> None:
    """Seed the request's tail block from ``positions`` (slot ``pos % kpool``)."""
    t = tail_view(tail_cache, kpool, head_dim)
    j = (positions.to(t.device, torch.int64) % kpool)
    t[tail_block, 0, j] = k.to(torch.bfloat16)
    t[tail_block, 1, j] = gate.to(torch.bfloat16)


def tail_read(tail_cache: torch.Tensor, tail_block: int, kpool: int,
              head_dim: int = INDEX_HEAD_DIM) -> tuple[torch.Tensor, torch.Tensor]:
    t = tail_view(tail_cache, kpool, head_dim)
    return t[tail_block, 0].clone(), t[tail_block, 1].clone()


def stock_tail_positions(n_valid: int, kpool: int) -> torch.Tensor:
    """The rows a stock prefill seeds into the tail block.

    ``_kpool_tail_seed_kernel`` keeps token i iff no token ``i + kpool`` of the same request is in
    the batch, i.e. the request's LAST ``kpool`` tokens -- which may straddle the last complete
    pool and the in-progress one.  Slots the in-progress pool does not own are never read back
    (the decode kernel reads slots ``< pos % kpool``), but reproducing them keeps a bytewise gate
    against a stock prefill meaningful."""
    lo = max(0, n_valid - kpool)
    return torch.arange(lo, n_valid, dtype=torch.int64)


# ---------------------------------------------------------------------------------------------
# per-request side buffer
# ---------------------------------------------------------------------------------------------

@dataclass
class IndexerSideBuffer:
    """Per-request, per-sparse-layer store of the indexer's cache inputs for every prompt row.

    Keyed by the indexer's ``prefix`` (its name in ``static_forward_context``), so it is stable
    across TP ranks -- and under TP the indexer is fully replicated (``wq_b`` is
    ``ReplicatedLinear``, ``wk_weights_proj`` has ``disable_tp=True``, `attention.py:250-266`),
    so every rank stores and rewrites the SAME bytes.  No collective, no rank-local slicing.

    Memory: ``2 * head_dim * 2 B = 512 B`` of payload per token per sparse layer, plus a 1 B
    per-row "captured" flag = 513 B (5.51 KiB/token over the 11 GLM-5.3 layers; 44.1 MiB for an
    8192-token prompt).  ``nbytes`` reports it exactly."""
    n: int
    device: torch.device
    head_dim: int = INDEX_HEAD_DIM
    dtype: torch.dtype = torch.bfloat16
    k: dict[str, torch.Tensor] = field(default_factory=dict)
    gate: dict[str, torch.Tensor] = field(default_factory=dict)
    filled: dict[str, torch.Tensor] = field(default_factory=dict)   # bool [n], per row
    n_valid: int = 0                     # rows [0, n_valid) hold live values

    def _alloc(self, key: str) -> None:
        if key not in self.k:
            self.k[key] = torch.zeros((self.n, self.head_dim), dtype=self.dtype, device=self.device)
            self.gate[key] = torch.zeros((self.n, self.head_dim), dtype=self.dtype,
                                         device=self.device)
            self.filled[key] = torch.zeros((self.n,), dtype=torch.bool, device=self.device)

    def store(self, key: str, positions: torch.Tensor,
              k: torch.Tensor, gate: torch.Tensor) -> None:
        """Scatter rows ``positions`` (capture during the approx pass, overwrite at correct)."""
        self._alloc(key)
        pos = positions.to(self.k[key].device, torch.int64)
        self.k[key][pos] = k.to(self.dtype)
        self.gate[key][pos] = gate.to(self.dtype)
        self.filled[key][pos] = True
        hi = int(pos.max().item()) + 1 if pos.numel() else 0
        self.n_valid = max(self.n_valid, hi)

    def rows(self, key: str, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pos = positions.to(self.k[key].device, torch.int64)
        assert not CHECK or bool(self.filled[key][pos].all()), (
            "the indexer side buffer was asked for rows it never captured: a pool sibling of a "
            "corrected row is missing, so the pooled entry would be built from zeros.  The "
            "approx pass must run with the hook installed (correct._execute_model calls "
            "_glm53_sparse before MODE_CAPTURE); `skip_prefill` open-walk requests do not "
            "capture and are not supported by the indexer rewrite yet.  Missing rows: "
            f"{pos[~self.filled[key][pos]][:8].tolist()}")
        return self.k[key][pos], self.gate[key][pos]

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size()
                   for d in (self.k, self.gate, self.filled) for t in d.values())

    @staticmethod
    def bytes_per_token_per_layer(head_dim: int = INDEX_HEAD_DIM,
                                  dtype: torch.dtype = torch.bfloat16) -> int:
        """512 B of payload (k + gate, both ``head_dim`` wide) + 1 B of per-row "captured" flag."""
        return 2 * head_dim * torch.empty((), dtype=dtype).element_size() + 1


# ---------------------------------------------------------------------------------------------
# the rewrite
# ---------------------------------------------------------------------------------------------

@dataclass
class LayerCaches:
    """The three tensors + addressing constants a rewrite needs for one sparse layer."""
    k_cache: torch.Tensor                 # [blocks, num_states, head_dim+4] uint8
    tail_cache: torch.Tensor              # bf16, viewed [blocks, 2, kpool, head_dim]
    block_row: torch.Tensor               # indexer-group block-table row of the request (int32)
    tail_block: int                       # the request's single tail block
    ape: torch.Tensor                     # index_kpool_compress_ape [kpool, head_dim] fp32
    kpool: int = 4
    head_dim: int = INDEX_HEAD_DIM
    round_scale: bool = True              # scale_fmt == "ue8m0"
    bt_block_size: int | None = None      # block-table page width in TOKENS (64 on main); None = paged at the cache page

    @property
    def num_states(self) -> int:
        return num_states_of(self.k_cache)

    def slots_of(self, pools: torch.Tensor) -> torch.Tensor:
        """`pool_slots` with this layer's two page widths."""
        return pool_slots(self.block_row, pools, self.num_states, self.kpool, self.bt_block_size)


def affected_pools(positions: torch.Tensor, kpool: int, n_valid: int) -> torch.Tensor:
    """Pools touched by ``positions`` that are COMPLETE (all ``kpool`` rows exist).

    An incomplete trailing pool has no ``k_cache`` entry at all -- it lives in the tail cache
    until decode completes it -- so writing one would corrupt the next request's slot."""
    pools = torch.unique(positions.to(torch.int64) // kpool)
    return pools[(pools + 1) * kpool <= n_valid]


def rewrite_rows(buf: IndexerSideBuffer, key: str, positions: torch.Tensor,
                 caches: LayerCaches, *, k: Optional[torch.Tensor] = None,
                 gate: Optional[torch.Tensor] = None) -> dict:
    """Rewrite the pooled ``k_cache`` and the ``tail_cache`` after correcting ``positions``.

    ``k``/``gate`` are the corrected rows' indexer inputs; when given they are stored into the
    side buffer first (the normal engine call), when omitted the buffer is assumed current (the
    test / gate path).  Every complete pool that any corrected row belongs to is recomputed from
    the buffer's CURRENT four siblings -- corrected or not -- which is the whole reason the buffer
    exists.  The tail block is re-seeded from the request's last ``kpool`` rows unconditionally:
    it is four rows, it is idempotent when nothing in them changed, and it removes a whole class
    of "the tail was stale because p happened to be the 3rd of its pool" bugs.

    Returns a dict of counters for the engine's ``info``/gate output."""
    if k is not None:
        assert gate is not None, "k without gate"
        buf.store(key, positions, k, gate)
    kp = caches.kpool
    pools = affected_pools(positions, kp, buf.n_valid)
    n_pools = int(pools.numel())
    if n_pools:
        rows = (pools[:, None] * kp + torch.arange(kp, device=pools.device)).reshape(-1)
        sk, sg = buf.rows(key, rows)
        sk = sk.reshape(n_pools, kp, caches.head_dim)
        sg = sg.reshape(n_pools, kp, caches.head_dim)
        entry, scale = pool_compress(sk, sg, caches.ape, round_scale=caches.round_scale)
        slots = caches.slots_of(pools)
        k_cache_write(caches.k_cache, slots, entry, scale, caches.head_dim)
    tpos = stock_tail_positions(buf.n_valid, kp).to(positions.device)
    tk, tg = buf.rows(key, tpos)
    tail_write(caches.tail_cache, caches.tail_block, tpos, tk, tg, kp, caches.head_dim)
    return {"pools_rewritten": n_pools, "tail_rows": int(tpos.numel()),
            "n_valid": int(buf.n_valid)}


# ---------------------------------------------------------------------------------------------
# sparse regime: which keys a query may see
# ---------------------------------------------------------------------------------------------
#
# `index_kpool_always_select_tail = True` for this checkpoint, and it is the property that makes
# pooling compatible with causality.  For a query whose token-granular context length is
# ``seq_len = pos + 1``:
#
#   pool_len   = seq_len // kpool          COMPLETE pools; only these are SCORED
#   tail_start = pool_len * kpool
#   tail_count = seq_len - tail_start      in [0, kpool); appended UNCONDITIONALLY, never scored
#
# The scored region is the pooled ``k_cache`` (`expand_pools_and_append_tail`,
# `ops/kpool_compress.py:820-892`, and its two-step torch equivalent `:715-816`); the tail region
# is emitted as raw token ids.  The identical bound appears in the prefill branch, where
# `BuildPrefillChunkMetadataKernel.kernel` sets
# ``len_per_token = (start_pos + 1 + offset) // COMPRESS_RATIO`` (`mla/indexer.py:355-357`) --
# i.e. ``pool_len(pos)``.  So decode-shaped and prefill-shaped batches enforce the same rule.
#
# THE INVARIANT the correct step enforces (and `tests/test_glm53_indexer.py` checks):
#
#   For every corrected row p, the set of candidate keys its scoring may return is exactly the
#   set a stock decode at position p sees:
#       candidates(p) subset-of [0, p + 1)
#       every SCORED pool j < (p+1)//kpool covers tokens [j*kpool, (j+1)*kpool) -- all <= p
#       the tail [ ((p+1)//kpool)*kpool , p+1 ) is always present
#   and the pooled entries those pools resolve to are the CORRECTED ones, because
#   `rewrite_rows` has already rewritten every pool this round touches BEFORE the layer's
#   indexer_op runs.
#
# Two consequences worth stating, because they are what makes the sparse round safe:
#   * a scored pool can never contain a row LATER than the query.  Pool ``p // kpool`` is scored
#     only when ``(p+1)//kpool > p//kpool``, i.e. ``p % kpool == kpool - 1``, in which case the
#     pool is exactly ``{p-3, p-2, p-1, p}``.  Otherwise p's own pool is the tail.  Pooling
#     therefore never leaks a future row into a query's context.
#   * SUB-BATCHING is safe without extra ordering.  When |P| exceeds ``max_num_seqs``,
#     ``appcorr_rows_step`` splits the round into contiguous slices of the ASCENDING row list,
#     and sub-batch 1 rewrites the pool straddling the boundary while sub-batch 2's rows are
#     still at their approximate value.  That intermediate entry is never read by sub-batch 1:
#     a straddling pool contains a row >= b (the first row of sub-batch 2), and a pool is
#     scored by query p only when all its rows are <= p, while every sub-batch-1 query p
#     satisfies p <= a < b.  Sub-batch 2 then rewrites the pool from four corrected rows
#     before anything else reads it.  The same argument makes the ORDER of rewrites within a
#     round irrelevant.
#   * the ``tail_cache`` is NOT read by the scoring path at all.  Grepping the op, `tail_kv_cache`
#     is touched only by `kpool_seed_tail_cache` (prefill write, line 386) and by
#     `kpool_decode_update_and_maybe_write_cache_batched` (decode write, line 706) -- both under
#     ``not skip_k_cache_insert``.  So disabling the stock write does not blind the round to its
#     boundary pool; the boundary pool was never scored, only appended.  The tail cache still has
#     to be correct for the DECODE steps that follow the correct step, which is why
#     `rewrite_rows` re-seeds it every round.


def pool_len_of(seq_len, kpool: int):
    """Number of COMPLETE (hence scoreable) pools for a query with context ``seq_len``."""
    if torch.is_tensor(seq_len):
        return seq_len.to(torch.int64) // kpool
    return int(seq_len) // kpool


def tail_span(seq_len: int, kpool: int) -> tuple[int, int]:
    """``(tail_start, tail_count)`` of the unconditionally-appended in-progress pool."""
    start = pool_len_of(seq_len, kpool) * kpool
    return start, int(seq_len) - start


def expand_pools_and_append_tail_ref(pool_ids: torch.Tensor, seq_lens: torch.Tensor,
                                     kpool: int) -> torch.Tensor:
    """Pure-torch port of `expand_pools_and_append_tail` (kpool_compress.py:820-892).

    Identical output to the two-step torch path (`expand_pools_to_tokens` +
    `append_tail_to_topk` with neither ``page_table`` nor ``topk_offsets``), which is the only
    path the GLM-5.3 indexer uses.  ``pool_ids`` is ``[rows, select_k]`` (-1 = unselected),
    ``seq_lens`` is TOKEN-granular ``[rows]``.  Returns ``[rows, select_k*kpool + kpool - 1]``
    int32 with -1 in every unused column."""
    rows, n_groups = pool_ids.shape
    topk = n_groups * kpool
    out_cols = topk + kpool - 1
    dev = pool_ids.device
    cols = torch.arange(out_cols, device=dev)[None, :]
    seq = seq_lens.to(dev, torch.int64)[:, None]
    pool_len = seq // kpool
    tail_start = pool_len * kpool
    tail_count = seq - tail_start

    is_hist = cols < topk
    g = torch.clamp(cols, max=topk - 1) // kpool
    o = cols % kpool
    pid = torch.gather(pool_ids.to(torch.int64), 1, g.expand(rows, out_cols))
    hist = torch.where(pid >= 0, pid * kpool + o, torch.full_like(pid, -1))

    tail_off = cols - topk
    is_tail = (tail_off >= 0) & (tail_off < tail_count)
    tail = tail_start + tail_off
    return torch.where(is_hist, hist, torch.where(is_tail, tail, torch.full_like(tail, -1))
                       ).to(torch.int32)


def candidate_keys(pool_ids_row: torch.Tensor, seq_len: int, kpool: int) -> torch.Tensor:
    """The sorted set of key positions one query may attend to, given its selected pools."""
    exp = expand_pools_and_append_tail_ref(pool_ids_row.reshape(1, -1),
                                           torch.tensor([seq_len]), kpool)[0]
    return torch.unique(exp[exp >= 0])


def scoreable_pool_slots(caches: "LayerCaches", seq_len: int) -> torch.Tensor:
    """Physical pooled-cache slots a query with context ``seq_len`` may be scored against."""
    pools = torch.arange(pool_len_of(seq_len, caches.kpool), device=caches.block_row.device)
    return caches.slots_of(pools)


# ---------------------------------------------------------------------------------------------
# what the indexer metadata builder makes of a pseudo-sequence batch
# ---------------------------------------------------------------------------------------------
#
# QUESTION (item b): does `decode_metadata.schedule_metadata` -- DeepGEMM's paged-MQA scheduling
# tensor -- carry per-request state that breaks when P one-token "requests" share ONE block-table
# row?  ANSWER: no.  Traced through vLLM main @658c813:
#
#   `get_paged_mqa_logits_metadata(context_lens, block_size, num_sms, indices=None)`
#   (`vllm/utils/deep_gemm.py:583-610`) takes ONLY the [B, next_n] context-length tensor, the
#   page width (`kv_cache_spec.num_states`) and the SM count, and returns a [slots+1, 2] work
#   schedule.  It never sees the block table's CONTENTS, request ids, or slot mappings -- it
#   partitions total KV work across SMs.  Two pseudo-sequences pointing at the same physical
#   blocks are simply two rows with context lengths p_i+1; that is what the softmax path already
#   relies on.
#
# Everything upstream of it is per-row arithmetic with no cross-request coupling:
#   * `split_decodes_and_prefills` early-returns "all decodes" at max_query_len == 1
#     (`backends/utils.py:802-807`), so num_decodes = P, decode_lens = 1, max_decode_len = 1.
#   * `PrepareUniformDecodeKernel` at max_decode_len == 1 computes
#     ``per_token_seq_len = max(seq_len - 1 + 0 + 1, 0) == seq_len`` and COPIES the request's
#     block-table row per token (`mla/indexer.py:104-118`) -- an identity on a batch that is
#     already one token per row.
#   * `compressed_seq_lens = seq_lens // compress_ratio` (`:1400-1409`) then `.unsqueeze(-1)`.
#   * the SPARSE MLA side (`SparseMLACommonMetadataBuilder.build`,
#     `model_executor/layers/attention/sparse_mla_attention.py:338-439`) likewise reads only
#     `query_start_loc`, `seq_lens`, `block_table_tensor`, `slot_mapping` and
#     `seq_lens_cpu_upper_bound`; its only derived per-token tensor is `req_id_per_token`, built
#     from `query_start_loc`.  On sm_100 with a bf16 latent cache and 32 local heads (TP=2) the
#     selected backend is FLASHMLA_SPARSE, then FLASHINFER_MLA_SPARSE
#     (`platforms/cuda.py:95-129`); both subclass that builder, so neither adds per-request state.
#
# So `_correct_sub`'s existing `CommonAttentionMetadata` is sufficient -- no new field.  The
# function below writes out what the builder WILL produce, so a CPU test can check the values
# against a per-request computation without importing vLLM (which is not installed in the
# `appcorr` env, and whose indexer backend imports DeepGEMM regardless).


def pseudo_sequence_indexer_metadata(positions: torch.Tensor, block_row: torch.Tensor,
                                     kpool: int, num_states: int,
                                     block_table_width: int | None = None,
                                     bt_block_size: int | None = None) -> dict:
    """The indexer decode metadata a batch of P one-token pseudo-sequences produces.

    ``positions`` are the corrected rows (ascending); ``block_row`` is the request's indexer
    block-table row, SHARED by all P rows.  Returns the tensors the op consumes:

        seq_lens_pool   [P, 1] int32   pool-granular context length ((p+1) // kpool)
        block_table     [P, W] int32   the request's row, repeated per token
        decode_lens     [P]    int32   all ones
        slot_mapping    [P]    int64   pool slot, or -1 when p does not complete a pool
        scoreable_pools [P]    int64   (p+1) // kpool -- how many pools each row may score

    ``schedule_metadata`` is deliberately NOT built here: it is a DeepGEMM CUDA call whose only
    inputs are ``seq_lens_pool``, ``num_states`` and the SM count, so checking those three is the
    whole of what a CPU test can establish."""
    pos = positions.to(torch.int64)
    P = int(pos.numel())
    W = int(block_table_width if block_table_width is not None else block_row.numel())
    seq = pos + 1
    pools = pos // kpool
    completes = ((pos + 1) % kpool) == 0
    slots = pool_slots(block_row, pools, num_states, kpool, bt_block_size)
    return {
        "seq_lens_pool": (seq // kpool).to(torch.int32).reshape(P, 1),
        "block_table": block_row[:W].to(torch.int32).unsqueeze(0).expand(P, W).contiguous(),
        "decode_lens": torch.ones(P, dtype=torch.int32),
        "slot_mapping": torch.where(completes, slots, torch.full_like(slots, -1)),
        "scoreable_pools": seq // kpool,
    }


def dense_regime(max_seq_len: int, topk_tokens: int) -> bool:
    """``max_prefill_seq_len <= topk_tokens`` -- the exact-causal, no-scoring regime.

    Mirrors BOTH short-prefill predicates (`sparse_attn_indexer_kpool.py:200-216` decode-shaped,
    `:421-449` prefill-shaped): the correct step's ``max_seq_len`` is ``last_corrected_pos + 1``.
    For GLM-5.3-Flash ``topk_tokens == index_topk == 2048``."""
    return int(max_seq_len) <= int(topk_tokens)


# ---------------------------------------------------------------------------------------------
# engine hook: intercept `Indexer.indexer_op`
# ---------------------------------------------------------------------------------------------
#
# Seam choice.  ``Indexer.forward`` hands ``indexer_op`` exactly the two tensors that reach the
# caches -- ``k`` (post-``k_norm``) and ``gate_score`` -- plus ``positions``.  Wrapping the op is
# therefore both the cheapest capture point (no duplicated ``wk_weights_proj`` /
# ``index_kpool_compress_gate`` GEMM, and the captured values are the stock ones bit-for-bit) and
# the only point that sits BETWEEN "the corrected row's k/gate exist" and "the caches are read"
# -- which is what a correct step needs, because the layer's own indexer scoring must see the
# rewritten pool.  ``SparseAttnIndexerKpool`` is a ``CustomOp``: ``forward`` dispatches through
# the per-instance ``_forward_method`` (``vllm/model_executor/custom_op.py:130-136``), so the
# hook is a per-instance attribute swap -- no class patch, nothing leaks to another model.

def buffer_for(sb) -> IndexerSideBuffer:
    """The indexer side buffer attached to a request's ``correct.SideBuffer`` (created on first
    use).  Hanging it off the existing per-request object instead of a second registry keeps the
    lifetime rules identical -- ``correct.free(req_id)`` drops the SideBuffer and this with it --
    and keeps this module out of ``correct.SideBuffer``'s field list, which agent K owns."""
    b = getattr(sb, "_glm53_ix", None)
    if b is None:
        b = IndexerSideBuffer(n=int(sb.n), device=sb.device)
        sb._glm53_ix = b
    return b


_HOOKED: list[tuple[object, object]] = []      # (indexer_op, original _forward_method)
_CTX: Optional["CorrectContext"] = None


@dataclass
class CorrectContext:
    """What the hook needs to know for the round currently running."""
    buf: IndexerSideBuffer
    caches: dict[str, LayerCaches]        # indexer prefix -> its caches for THIS request
    positions: torch.Tensor               # int64 [P], the corrected rows, ascending
    max_seq_len: int                      # last corrected position + 1
    topk_tokens: int
    capture_only: bool = False            # approx pass: store k/gate, touch no cache
    debug: dict = field(default_factory=dict)

    @property
    def dense(self) -> bool:
        return dense_regime(self.max_seq_len, self.topk_tokens)


def set_context(ctx: Optional["CorrectContext"]) -> Optional["CorrectContext"]:
    global _CTX
    prev, _CTX = _CTX, ctx
    return prev


def _op_prefix(op) -> str:
    """The indexer's stable key: the pooled K cache's ``static_forward_context`` name."""
    return str(op.k_cache.prefix)


def _hook(op, orig):
    def wrapper(hidden_states, q_quant, k, weights, *, gate_score=None, compress_ape=None,
                index_kpool=1, positions=None):
        from appcorr.vllm_stream import correct as _c

        ctx = _CTX
        if ctx is None:
            if _c._ST.mode == _c.MODE_CAPTURE and _c._ST.captures:
                # Approx pass.  `_Capture` gives each streamed request's slice of the step's flat
                # token batch (tok0/ntok) and the prompt position of its first row, exactly as
                # the GDN side-buffer capture uses it.  Store the stock k/gate for those rows;
                # the stock op still writes the caches, so the approx pass is untouched.
                key = _op_prefix(op)
                for cap in _c._ST.captures:
                    buf = buffer_for(cap.sb)
                    sl = slice(cap.tok0, cap.tok0 + cap.ntok)
                    rows = torch.arange(cap.pos0, cap.pos0 + cap.ntok, device=k.device)
                    buf.store(key, rows, k[sl], gate_score[sl])
            return orig(hidden_states, q_quant, k, weights, gate_score=gate_score,
                        compress_ape=compress_ape, index_kpool=index_kpool, positions=positions)
        key = _op_prefix(op)
        assert positions is not None, "the kpool indexer needs positions; got None"
        n = int(hidden_states.shape[0])
        pos = positions[:n]
        if ctx.capture_only:
            # approx pass: record the stock inputs, let the stock op write the caches.
            ctx.buf.store(key, pos, k[:n], gate_score[:n])
            return orig(hidden_states, q_quant, k, weights, gate_score=gate_score,
                        compress_ape=compress_ape, index_kpool=index_kpool, positions=positions)

        # correct step.  The stock write path is WRONG for a pseudo-sequence batch: each
        # corrected row arrives as its own one-token "request" over the same block table, so
        # `kpool_decode_update_and_maybe_write_cache_batched` would stash every row into the
        # request's single tail block at `pos % kpool` (clobbering the real tail) and would
        # complete a pool from three siblings that are not in the batch.  Disable it and do the
        # pool arithmetic ourselves from the side buffer.
        caches = ctx.caches[key]
        prev_skip = op.skip_k_cache_insert
        op.skip_k_cache_insert = True
        try:
            stats = rewrite_rows(ctx.buf, key, pos, caches, k=k[:n], gate=gate_score[:n])
            ctx.debug.setdefault(key, {}).update(stats)
            if ctx.dense:
                # Exactly what `_fill_short_decode_causal_indices` would have produced, without
                # reading the pooled cache at all.  Returning here also skips the paged-MQA
                # logits + top-k, which is the whole cost of the indexer.
                buf = op.topk_indices_buffer
                buf[:n] = -1
                fill_causal_topk(buf[:n], pos)
                return buf
            # ---- sparse regime (max_seq_len > topk_tokens; every real prompt) ----------
            # Nothing more has to be computed here: the round's scoring reads ONLY the pooled
            # `k_cache`, which `rewrite_rows` has already brought up to date for every pool this
            # round touches, and it appends the boundary pool by token id without consulting the
            # tail cache (see the block comment above `pool_len_of`).  So the stock op runs
            # unchanged apart from `skip_k_cache_insert=True`, which suppresses only the WRITE
            # path -- the one thing a pseudo-sequence batch cannot be allowed to do.
            #
            # What is asserted before handing over: every row a scoreable pool can cover must be
            # in the side buffer.  A missing row would mean some pool < pool_len was never
            # captured, so the query would be scored against an entry built from zeros.  The
            # check is one device sync per sparse layer per round, so it is gated on
            # APPCORR_GLM53_CHECK (default on while the path is being brought up).
            if not SPARSE_ENABLED:
                raise NotImplementedError(
                    f"sparse-regime correct step disabled by APPCORR_GLM53_SPARSE=0 "
                    f"(max_seq_len={ctx.max_seq_len} > topk_tokens={ctx.topk_tokens})")
            n_scoreable = pool_len_of(ctx.max_seq_len, caches.kpool)
            if CHECK:
                covered = n_scoreable * caches.kpool
                assert bool(ctx.buf.filled[key][:covered].all()), (
                    "a scoreable pool is missing rows from the indexer side buffer: the query "
                    "would be scored against an entry built from zeros.  Rows "
                    f"{(~ctx.buf.filled[key][:covered]).nonzero()[:8].flatten().tolist()} of "
                    f"[0, {covered}) were never captured (max_seq_len={ctx.max_seq_len}, "
                    f"kpool={caches.kpool}).")
            ctx.debug.setdefault(key, {}).update(
                scoreable_pools=int(n_scoreable),
                tail_span=tail_span(ctx.max_seq_len, caches.kpool),
                sparse=True)
            return orig(hidden_states, q_quant, k, weights, gate_score=gate_score,
                        compress_ape=compress_ape, index_kpool=index_kpool,
                        positions=positions)
        finally:
            op.skip_k_cache_insert = prev_skip
    return wrapper


def install(model) -> int:
    """Wrap every kpool indexer op under ``model``'s decoder.  Idempotent; returns the count."""
    from appcorr.vllm_stream.correct import _decoder_module
    if _HOOKED:
        return len(_HOOKED)
    for _, _mod, ix in sparse_layers(_decoder_module(model)):
        op = ix.indexer_op
        orig = op._forward_method
        op._forward_method = _hook(op, orig)
        _HOOKED.append((op, orig))
    return len(_HOOKED)


def uninstall() -> None:
    while _HOOKED:
        op, orig = _HOOKED.pop()
        op._forward_method = orig


def layer_caches(runner, req_id: str, model=None) -> dict[str, LayerCaches]:
    """Build the per-layer :class:`LayerCaches` for one request, from the runner's tables.

    The indexer's pooled cache and the tail cache are ordinary vLLM KV-cache groups (they are
    ``torch.nn.Module``s with ``get_kv_cache_spec``), so the request's block ids come from the
    same ``correct._block_row`` the MLA latent uses.  The tail group is
    ``KpoolTailSpec(block_size=kpool)`` with exactly ONE block per request
    (`kv_cache_interface.py:971-990`), which is why its slot mapping cannot be built by the
    generic ``correct._slot_mapping`` (that would index ``block_row[pos // 4]`` into a row of
    width 1)."""
    from appcorr.vllm_stream.correct import _block_row, _decoder_module, _leaf_spec
    from vllm.v1.kv_cache_interface import KpoolTailSpec

    model = model if model is not None else runner.model
    gid_of: dict[str, int] = {}
    for gid, group in enumerate(runner.kv_cache_config.kv_cache_groups):
        for ln in group.layer_names:
            gid_of[ln] = gid

    out: dict[str, LayerCaches] = {}
    for _, _mod, ix in sparse_layers(_decoder_module(model)):
        kpre, tpre = str(ix.k_cache.prefix), str(ix.tail_cache.prefix)
        kgid, tgid = gid_of[kpre], gid_of[tpre]
        # `_leaf_spec`: on main a group's spec may be a `UniformTypeKVCacheSpecs` container.
        tspec = _leaf_spec(runner.kv_cache_config.kv_cache_groups[tgid].kv_cache_spec, tpre)
        assert isinstance(tspec, KpoolTailSpec), type(tspec)
        kspec = _leaf_spec(runner.kv_cache_config.kv_cache_groups[kgid].kv_cache_spec, kpre)
        kp = int(ix.index_kpool)
        assert int(kspec.tokens_per_state) == kp, (kspec.tokens_per_state, kp)
        kc, tc = ix.k_cache.kv_cache, ix.tail_cache.kv_cache
        if isinstance(kc, (list, tuple)):
            kc = kc[0]
        if isinstance(tc, (list, tuple)):
            tc = tc[0]
        # The pooled cache is paged at the KERNEL block, not the manager block: on main the
        # manager block is 2176 tokens (544 pools) while the k_cache view is [pages, 32, 132],
        # i.e. 32 states = 128 tokens per page, and `_block_row` already expands the request's
        # block ids to kernel granularity (`map_to_kernel_blocks`), so `pool_slots(row, pools,
        # num_states)` with the VIEW's own width is the right address.  The old guard compared
        # the view width with block_size // kpool (= 544) and refused a correct geometry (leg 3
        # attempt 4, B200-8 2026-09-13 04:18 KST).  The invariant that actually protects
        # `pool_slots` is: the block table's page width in tokens == states per page x kpool.
        kbt = runner.input_batch.block_table[kgid]
        bt_bs, mgr = int(kbt.block_size), int(kspec.block_size)
        page_tok = num_states_of(kc) * kp
        # Both the block table (bt_bs tokens/entry) and the cache view (page_tok tokens/page)
        # must be uniform splits of the manager block; that is what makes `pool_slots`'
        # absolute-token address valid.  On main: 2176 = 34 x 64 = 17 x 128.
        assert mgr % bt_bs == 0 and mgr % page_tok == 0, (
            f"indexer group geometry is not a uniform split: manager block_size={mgr}, "
            f"block_table.block_size={bt_bs}, cache page={num_states_of(kc)} states x kpool={kp}"
            f" = {page_tok} tokens (blocks_per_kv_block={getattr(kbt, 'blocks_per_kv_block', None)})")
        out[kpre] = LayerCaches(
            k_cache=kc, tail_cache=tc,
            block_row=_block_row(runner, req_id, kgid),
            tail_block=int(_block_row(runner, req_id, tgid)[0]),
            ape=ix.index_kpool_compress_ape.detach().float(),
            kpool=kp, head_dim=int(ix.head_dim),
            round_scale=ix.scale_fmt is not None,
            bt_block_size=bt_bs,
        )
    return out


def topk_tokens_of(model) -> int:
    """``index_topk`` as the served model actually holds it (2048 for GLM-5.3-Flash)."""
    from appcorr.vllm_stream.correct import _decoder_module
    vals = {int(ix.topk_tokens) for _, _m, ix in sparse_layers(_decoder_module(model))}
    assert len(vals) == 1, f"sparse layers disagree on topk_tokens: {vals}"
    return vals.pop()
