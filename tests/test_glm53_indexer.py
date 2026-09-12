"""CPU tests for the GLM-5.3-Flash sparse-indexer cache rewrite (`appcorr.vllm_stream.glm53_indexer`).

Item 3 of `docs/memo/glm53_correct_design.md`, gate part.  No GPU, no vLLM import: every shape
below is the REAL one for the served `zai-org/GLM-5.3-Flash` checkpoint, read off its config and
off the vLLM sources (`$V = vllm-main @ 658c813`):

    index_kpool      4            (text_config.index_kpool)
    index_head_dim   128          (text_config.index_head_dim)
    index_topk       2048         (text_config.index_topk)  -> topk_tokens
    index_n_heads    32           (text_config.index_n_heads; the survey's "16" is wrong)
    topk buffer      [max_num_batched_tokens, 2176]   = ceil((2048 + 3) / 128) * 128
    k_cache          [num_blocks, block_size // 4, 132] uint8   (128 fp8 + one fp32 scale)
    tail_cache       [num_blocks, 2, 4, 128] bf16                (K half, gate half)

Three checks, in the order the task asks for them:

  (a) `pool_compress` (the port of `_kpool_softmax_rotate_write_cache_kernel`) against an
      INDEPENDENT reference written straight from the kernel's algebra -- explicit Sylvester
      Hadamard matrix instead of the seven butterfly stages, vectorised softmax instead of the
      slot loop.  The bitwise claim against the Triton kernel itself is GPU-only (the kernel
      needs Triton); this is the arithmetic port check.
  (b) after rewriting row p, `k_cache[p // 4]` and the tail block equal a from-scratch fill of
      the same rows -- bytewise, on the raw uint8 / bf16 storage.
  (c) the causal `topk_indices` fill for a pseudo-sequence at position p equals the stock
      short-prefill fill (`_fill_causal_indices`, sparse_attn_indexer_kpool.py:193-198) for
      query p, ported here because vLLM is not importable in the `appcorr` env.

Run:
    PYTHONPATH=/NHNHOME/share/cjpark/AppCorr-glm53 \\
    /home/nxclab/anaconda3/envs/appcorr/bin/python -m pytest tests/test_glm53_indexer.py -q
"""
from __future__ import annotations

import math

import torch

from appcorr.vllm_stream.glm53_indexer import (
    FP8_DTYPE,
    INDEX_HEAD_DIM,
    IndexerSideBuffer,
    LayerCaches,
    affected_pools,
    dense_regime,
    fill_causal_topk,
    hadamard128,
    k_cache_read,
    k_cache_write,
    pool_compress,
    pool_slots,
    rewrite_rows,
    stock_tail_positions,
    tail_read,
    tail_write,
)

KPOOL = 4
HEAD_DIM = INDEX_HEAD_DIM
BLOCK_SIZE = 512                 # cache_config.block_size; must be a multiple of kpool*32 = 128
NUM_STATES = BLOCK_SIZE // KPOOL         # 128 pooled entries per page
TOPK_TOKENS = 2048
TOPK_WIDTH = ((TOPK_TOKENS + KPOOL - 1 + 127) // 128) * 128      # 2176


# ------------------------------------------------------------------------------------------- #
# independent reference for (a)
# ------------------------------------------------------------------------------------------- #

def _sylvester(n: int) -> torch.Tensor:
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h.float()


_H128 = _sylvester(128)


def ref_pool_entry(slot_k: torch.Tensor, slot_gate: torch.Tensor,
                   ape: torch.Tensor, round_scale: bool = True):
    """`_kpool_softmax_rotate_write_cache_kernel` rewritten from its algebra, not its loops.

    Per-channel softmax over the pool axis (`kpool_compress.py:174-208`), bf16 round trip, the
    Hadamard-128 rotation as an explicit matmul by the symmetric Sylvester matrix / sqrt(128)
    (`_hadamard128`, :36-45), a second bf16 round trip, then per-vector absmax fp8 quant with a
    power-of-two (ue8m0) scale and the 1e-4 absmax floor (:218-231)."""
    score = slot_gate.float() + ape.float()[None]          # [P, kpool, D]
    prob = torch.exp(score - score.amax(dim=1, keepdim=True))
    x = (slot_k.float() * prob).sum(1) / prob.sum(1)
    x = x.to(torch.bfloat16).float()
    x = (x @ _H128) / math.sqrt(128.0)
    x = x.to(torch.bfloat16).float()
    absmax = torch.clamp(x.abs().amax(-1), min=1e-4)
    scale = torch.exp2(torch.ceil(torch.log2(absmax / 448.0))) if round_scale else absmax / 448.0
    q = torch.clamp(x / scale[:, None], -448.0, 448.0)
    return q.to(FP8_DTYPE), scale


def _stock_fill_causal_indices(rows: torch.Tensor, positions: torch.Tensor) -> None:
    """`_fill_causal_indices` verbatim (sparse_attn_indexer_kpool.py:193-198)."""
    causal_range = torch.arange(rows.shape[1], device=rows.device, dtype=torch.int32)
    positions = positions.to(torch.int32)
    rows[:] = causal_range[None, :]
    rows[causal_range[None, :] > positions[:, None]] = -1


# ------------------------------------------------------------------------------------------- #
# fixtures
# ------------------------------------------------------------------------------------------- #

def _fake_layer(n_tokens: int, seed: int = 0):
    """A side buffer filled with plausible indexer inputs + the caches at real shapes."""
    g = torch.Generator().manual_seed(seed)
    k = (torch.randn(n_tokens, HEAD_DIM, generator=g) * 0.7).to(torch.bfloat16)
    gate = (torch.randn(n_tokens, HEAD_DIM, generator=g) * 1.3).to(torch.bfloat16)
    ape = torch.randn(KPOOL, HEAD_DIM, generator=g, dtype=torch.float32) * 0.5

    buf = IndexerSideBuffer(n=n_tokens, device=torch.device("cpu"))
    buf.store("L", torch.arange(n_tokens), k, gate)

    n_blocks = 8
    k_cache = torch.zeros(n_blocks, NUM_STATES, HEAD_DIM + 4, dtype=torch.uint8)
    tail = torch.zeros(n_blocks * 2 * KPOOL * HEAD_DIM, dtype=torch.bfloat16)
    # a deliberately non-identity block row: pool j must follow the block table, not j itself
    block_row = torch.tensor([5, 2, 7, 1, 6, 0, 4, 3], dtype=torch.int32)
    caches = LayerCaches(k_cache=k_cache, tail_cache=tail, block_row=block_row,
                         tail_block=3, ape=ape, kpool=KPOOL, head_dim=HEAD_DIM)
    return buf, caches, k, gate, ape


def _fill_from_scratch(buf, caches, n_valid: int) -> None:
    """What a stock prefill of rows [0, n_valid) leaves in both caches.

    `_kpool_compress_insert` writes one entry per COMPLETE pool (`slot_mapping` is pool-granular
    and only the pool's last token carries a valid slot), and `kpool_seed_tail_cache` seeds the
    request's last `kpool` tokens at `pos % kpool`."""
    n_pools = n_valid // caches.kpool
    if n_pools:
        pools = torch.arange(n_pools)
        rows = (pools[:, None] * caches.kpool + torch.arange(caches.kpool)).reshape(-1)
        sk, sg = buf.rows("L", rows)
        entry, scale = pool_compress(sk.reshape(n_pools, caches.kpool, caches.head_dim),
                                     sg.reshape(n_pools, caches.kpool, caches.head_dim),
                                     caches.ape, round_scale=caches.round_scale)
        k_cache_write(caches.k_cache, pool_slots(caches.block_row, pools, caches.num_states),
                      entry, scale, caches.head_dim)
    tpos = stock_tail_positions(n_valid, caches.kpool)
    tk, tg = buf.rows("L", tpos)
    tail_write(caches.tail_cache, caches.tail_block, tpos, tk, tg, caches.kpool, caches.head_dim)


# ------------------------------------------------------------------------------------------- #
# (a) the pooling port
# ------------------------------------------------------------------------------------------- #

def test_a_pool_compress_matches_reference():
    buf, caches, k, gate, ape = _fake_layer(64, seed=1)
    n_pools = 16
    rows = torch.arange(n_pools * KPOOL)
    sk, sg = buf.rows("L", rows)
    sk = sk.reshape(n_pools, KPOOL, HEAD_DIM)
    sg = sg.reshape(n_pools, KPOOL, HEAD_DIM)

    got_q, got_s = pool_compress(sk, sg, ape)
    ref_q, ref_s = ref_pool_entry(sk, sg, ape)

    assert got_q.dtype == FP8_DTYPE and got_s.dtype == torch.float32
    byte_eq = torch.equal(got_q.view(torch.uint8), ref_q.view(torch.uint8))
    dq = (got_q.float() - ref_q.float()).abs().max().item()
    ds = (got_s - ref_s).abs().max().item()
    print(f"\n(a) pool entry: bytewise={byte_eq} max|dq|={dq:.3e} max|dscale|={ds:.3e} "
          f"(n_pools={n_pools})")
    assert ds == 0.0, ds
    assert dq == 0.0, dq
    assert byte_eq


def test_a_hadamard_is_orthogonal_and_matches_matmul():
    """The butterfly port vs the explicit Sylvester matmul, and H H^T / 128 == I."""
    g = torch.Generator().manual_seed(7)
    x = torch.randn(37, 128, generator=g)
    got = hadamard128(x)
    ref = (x @ _H128) / math.sqrt(128.0)
    d = (got - ref).abs().max().item()
    rel = d / ref.abs().max().item()
    ident = ((_H128 @ _H128.t()) / 128.0 - torch.eye(128)).abs().max().item()
    print(f"\n(a') hadamard128: max|butterfly - matmul|={d:.3e} (rel {rel:.3e}), "
          f"max|H H^T/128 - I|={ident:.3e}")
    assert d < 1e-4, d
    assert ident == 0.0


# ------------------------------------------------------------------------------------------- #
# (b) the rewrite
# ------------------------------------------------------------------------------------------- #

def test_b_rewrite_equals_from_scratch_fill():
    n = 40                     # 10 complete pools, no tail remainder
    buf0, caches0, k0, gate0, ape = _fake_layer(n, seed=2)
    _fill_from_scratch(buf0, caches0, n)          # the "approx prefill" state

    # corrected values for a scattered row set, including the first/last of a pool and a row in
    # the request's last pool (which the tail cache also holds)
    P = torch.tensor([0, 5, 6, 17, 36, 39], dtype=torch.int64)
    g = torch.Generator().manual_seed(99)
    new_k = (torch.randn(len(P), HEAD_DIM, generator=g) * 0.7).to(torch.bfloat16)
    new_gate = (torch.randn(len(P), HEAD_DIM, generator=g) * 1.3).to(torch.bfloat16)

    stats = rewrite_rows(buf0, "L", P, caches0, k=new_k, gate=new_gate)

    # ground truth: a from-scratch fill of the SAME final row values into pristine caches
    buf1, caches1, _, _, _ = _fake_layer(n, seed=2)
    buf1.store("L", P, new_k, new_gate)
    caches1.ape = ape
    _fill_from_scratch(buf1, caches1, n)

    same_k = torch.equal(caches0.k_cache, caches1.k_cache)
    same_t = torch.equal(caches0.tail_cache, caches1.tail_cache)
    dk = (caches0.k_cache.int() - caches1.k_cache.int()).abs().max().item()
    dt = (caches0.tail_cache.float() - caches1.tail_cache.float()).abs().max().item()
    print(f"\n(b) rewrite vs from-scratch: k_cache bytewise={same_k} (max|dbyte|={dk}), "
          f"tail bytewise={same_t} (max|d|={dt:.3e}), stats={stats}")
    assert stats["pools_rewritten"] == len(torch.unique(P // KPOOL))
    assert same_k and same_t

    # and the untouched pools must be byte-identical to the pre-rewrite state, i.e. the rewrite
    # touched exactly the pools it claimed
    buf2, caches2, _, _, _ = _fake_layer(n, seed=2)
    _fill_from_scratch(buf2, caches2, n)
    touched = set(int(p) for p in torch.unique(P // KPOOL))
    for j in range(n // KPOOL):
        s = pool_slots(caches0.block_row, torch.tensor([j]), NUM_STATES)
        a = k_cache_read(caches0.k_cache, s)[0].view(torch.uint8)
        b = k_cache_read(caches2.k_cache, s)[0].view(torch.uint8)
        eq = torch.equal(a, b)
        assert eq == (j not in touched), (j, eq, touched)


def test_b_incomplete_tail_pool_is_not_written():
    """A trailing partial pool has no `k_cache` entry; writing one would clobber a neighbour."""
    n = 38                     # 9 complete pools + rows 36,37 in the in-progress pool 9
    buf, caches, _, _, _ = _fake_layer(n, seed=3)
    _fill_from_scratch(buf, caches, n)
    before = caches.k_cache.clone()

    P = torch.tensor([36, 37], dtype=torch.int64)
    g = torch.Generator().manual_seed(5)
    stats = rewrite_rows(buf, "L", P, caches,
                         k=(torch.randn(2, HEAD_DIM, generator=g)).to(torch.bfloat16),
                         gate=(torch.randn(2, HEAD_DIM, generator=g)).to(torch.bfloat16))
    tk, tg = tail_read(caches.tail_cache, caches.tail_block, KPOOL, HEAD_DIM)
    exp_k, exp_g = buf.rows("L", torch.arange(n - KPOOL, n))
    j = torch.arange(n - KPOOL, n) % KPOOL
    print(f"\n(b') incomplete pool: pools_rewritten={stats['pools_rewritten']} "
          f"(expect 0), k_cache untouched={torch.equal(before, caches.k_cache)}, "
          f"tail rows={stats['tail_rows']}")
    assert stats["pools_rewritten"] == 0
    assert torch.equal(before, caches.k_cache)
    assert torch.equal(tk[j], exp_k.to(torch.bfloat16))
    assert torch.equal(tg[j], exp_g.to(torch.bfloat16))
    assert affected_pools(P, KPOOL, n).numel() == 0


def test_b_pool_slot_addressing():
    """`pool_slots` == `get_compressed_slot_mapping`'s kernel for one request."""
    block_row = torch.tensor([5, 2, 7, 1, 6, 0, 4, 3], dtype=torch.int32)
    pools = torch.arange(0, 3 * NUM_STATES, 37, dtype=torch.int64)
    got = pool_slots(block_row, pools, NUM_STATES)
    ref = torch.tensor([int(block_row[int(p) // NUM_STATES]) * NUM_STATES + int(p) % NUM_STATES
                        for p in pools], dtype=torch.int64)
    print(f"\n(b'') pool slots: n={len(pools)} max|d|={int((got - ref).abs().max())}")
    assert torch.equal(got, ref)


def test_b_cache_layout_roundtrip():
    """`k_cache_write` / `k_cache_read` use the kernel's two separate byte regions per page."""
    k_cache = torch.zeros(4, NUM_STATES, HEAD_DIM + 4, dtype=torch.uint8)
    g = torch.Generator().manual_seed(11)
    slots = torch.tensor([0, 1, NUM_STATES + 7, 3 * NUM_STATES - 1], dtype=torch.int64)
    entry = (torch.randn(4, HEAD_DIM, generator=g)).to(FP8_DTYPE)
    scale = torch.rand(4, generator=g) + 0.5
    k_cache_write(k_cache, slots, entry, scale)
    got_e, got_s = k_cache_read(k_cache, slots)
    # the fp8 values live in the first num_states*head_dim bytes of the page, the fp32 scales
    # after them -- a value write must never land in the scale region
    page0 = k_cache[0].reshape(-1)
    print(f"\n(b''') layout: entry bytewise={torch.equal(got_e.view(torch.uint8), entry.view(torch.uint8))} "
          f"scale max|d|={(got_s - scale).abs().max().item():.3e} "
          f"scale bytes set on page 3 (never written)={int((k_cache[3].reshape(-1)[NUM_STATES * HEAD_DIM:] != 0).sum())}")
    assert torch.equal(got_e.view(torch.uint8), entry.view(torch.uint8))
    assert torch.equal(got_s, scale)
    assert page0[NUM_STATES * HEAD_DIM:].view(torch.float32).numel() == NUM_STATES


# ------------------------------------------------------------------------------------------- #
# (c) the causal top-k fill
# ------------------------------------------------------------------------------------------- #

def test_c_causal_topk_matches_stock_short_prefill():
    positions = torch.tensor([0, 1, 3, 4, 127, 128, 1000, 2047], dtype=torch.int64)
    P = positions.numel()
    got = torch.empty(P, TOPK_WIDTH, dtype=torch.int32)
    ref = torch.empty(P, TOPK_WIDTH, dtype=torch.int32)
    fill_causal_topk(got, positions)
    _stock_fill_causal_indices(ref, positions)
    eq = torch.equal(got, ref)
    print(f"\n(c) causal top-k fill: bytewise={eq} width={TOPK_WIDTH} "
          f"max|d|={int((got - ref).abs().max())}")
    assert eq
    # the contract: key_pos <= query_pos, and every key at or below is present exactly once
    for i, p in enumerate(positions.tolist()):
        row = got[i]
        assert torch.equal(row[:p + 1], torch.arange(p + 1, dtype=torch.int32))
        assert bool((row[p + 1:] == -1).all())


def test_c_dense_regime_predicate():
    """`index_topk = 2048`: our 4k V*Bench prompts are NOT dense.  This is the finding the
    design memo's item 3 ("prefer the short-prefill dense regime") has to live with."""
    assert dense_regime(2048, TOPK_TOKENS)
    assert dense_regime(1, TOPK_TOKENS)
    assert not dense_regime(2049, TOPK_TOKENS)
    assert not dense_regime(4096, TOPK_TOKENS)       # V*Bench ~4k
    assert not dense_regime(8192, TOPK_TOKENS)       # our max_model_len cap
    print(f"\n(c') dense regime with topk_tokens={TOPK_TOKENS}: <=2048 dense, "
          f"4096/8192 sparse (V*Bench and the 8192 cap both fall in the SPARSE regime)")


# ------------------------------------------------------------------------------------------- #
# side-buffer memory
# ------------------------------------------------------------------------------------------- #

def test_side_buffer_bytes():
    b = IndexerSideBuffer.bytes_per_token_per_layer()
    buf, _, _, _, _ = _fake_layer(256, seed=4)
    per_layer = buf.nbytes()
    print(f"\nside buffer: {b} B/token/layer, {b * 11} B/token over 11 sparse layers, "
          f"{b * 11 * 8192 / 2**20:.1f} MiB for an 8192-token prompt; "
          f"one fake layer of 256 rows = {per_layer} B")
    assert b == 513                     # 512 B payload + 1 B per-row captured flag
    assert per_layer == 256 * 513


# ------------------------------------------------------------------------------------------- #
# (d) the sparse-regime visibility invariant
#
#   For a corrected row p, the set of candidate keys visible to p's scoring == the set a stock
#   decode at p sees.  Both are determined entirely by seq_len = p + 1:
#       pool_len   = seq_len // kpool   complete pools are SCORED (from the pooled k_cache)
#       tail       = [pool_len*kpool, seq_len)   appended unconditionally, never scored
#   so the check is (i) my expansion == the stock torch expansion, (ii) causality, (iii) the
#   pooled slots the two paths address are the same.
# ------------------------------------------------------------------------------------------- #

from appcorr.vllm_stream.glm53_indexer import (                     # noqa: E402
    SPARSE_ENABLED,
    candidate_keys,
    expand_pools_and_append_tail_ref,
    pool_len_of,
    pseudo_sequence_indexer_metadata,
    scoreable_pool_slots,
    tail_span,
)


def _stock_expand_pools_to_tokens(group_ids, group_valid, topk, pool_size):
    """`expand_pools_to_tokens` verbatim (kpool_compress.py:715-752), identity path."""
    offsets = torch.arange(pool_size, dtype=torch.int64)
    token_ids = group_ids.to(torch.int64).unsqueeze(-1) * pool_size + offsets
    token_ids = token_ids.reshape(group_ids.shape[0], topk)
    valid = group_valid.unsqueeze(-1).expand(-1, -1, pool_size).reshape(
        group_ids.shape[0], topk)
    output = token_ids.to(torch.int32)
    return torch.where(valid, output, torch.full_like(output, -1))


def _stock_append_tail_to_topk(topk_result, seq_lens, pool_lens, pool_size):
    """`append_tail_to_topk` verbatim (kpool_compress.py:754-816), identity path."""
    tail_pool = pool_size - 1
    if tail_pool == 0:
        return topk_result
    rows, n_cols = topk_result.shape
    out_cols = n_cols + tail_pool
    pool_len = pool_lens.to(torch.int32)
    tail_start = pool_len * pool_size
    seq_len = seq_lens.to(torch.int32)
    tail_count = seq_len - tail_start
    cols = torch.arange(out_cols)[None, :]
    is_history = cols < n_cols
    tail_off = cols - n_cols
    is_tail = (tail_off >= 0) & (tail_off < tail_count[:, None])
    safe_hist = torch.minimum(cols, torch.full_like(cols, n_cols - 1)).expand(rows, out_cols)
    history_val = torch.gather(topk_result, 1, safe_hist)
    tail_val = (tail_start[:, None] + tail_off).to(torch.int32)
    out = torch.where(is_history, history_val, torch.full_like(tail_val, -1))
    return torch.where(is_tail, tail_val, out)


def _stock_two_step(pool_ids, seq_lens, kpool):
    topk = pool_ids.shape[1] * kpool
    exp = _stock_expand_pools_to_tokens(pool_ids, pool_ids >= 0, topk, kpool)
    return _stock_append_tail_to_topk(exp, seq_lens, seq_lens.to(torch.int64) // kpool, kpool)


def test_d_expansion_matches_stock_two_step():
    """The fused reference == `expand_pools_to_tokens` + `append_tail_to_topk`."""
    g = torch.Generator().manual_seed(21)
    positions = torch.tensor([3, 4, 7, 1000, 2047, 2048, 4095, 8191], dtype=torch.int64)
    seq = positions + 1
    select_k = 8                                   # small stand-in for 2048 // 4 = 512
    pool_ids = torch.stack([
        torch.cat([torch.randperm(max(1, int(s) // KPOOL), generator=g)[:select_k],
                   torch.full((select_k,), -1)])[:select_k].to(torch.int32)
        for s in seq])
    got = expand_pools_and_append_tail_ref(pool_ids, seq, KPOOL)
    ref = _stock_two_step(pool_ids, seq, KPOOL)
    eq = torch.equal(got, ref)
    print(f"\n(d) expansion vs stock two-step: bytewise={eq} shape={tuple(got.shape)} "
          f"max|d|={int((got.int() - ref.int()).abs().max())}")
    assert eq
    assert got.shape == (len(positions), select_k * KPOOL + KPOOL - 1)


def test_d_causality_and_tail_always_present():
    """candidates(p) subset [0, p+1); the tail is always there; no scored pool holds a row > p."""
    g = torch.Generator().manual_seed(22)
    worst_hi, n_checked = -1, 0
    for p in [0, 1, 2, 3, 4, 7, 8, 15, 2047, 2048, 2051, 4095, 8191]:
        seq = p + 1
        n_pools = pool_len_of(seq, KPOOL)
        select_k = 8
        if n_pools:
            sel = torch.randperm(n_pools, generator=g)[:select_k].to(torch.int32)
            sel = torch.cat([sel, torch.full((select_k - sel.numel(),), -1, dtype=torch.int32)])
        else:
            sel = torch.full((select_k,), -1, dtype=torch.int32)
        keys = candidate_keys(sel, seq, KPOOL)
        start, count = tail_span(seq, KPOOL)
        tail = torch.arange(start, start + count, dtype=torch.int64)
        hi = int(keys.max()) if keys.numel() else -1
        worst_hi = max(worst_hi, hi - p)
        n_checked += 1
        assert keys.numel() == 0 or hi <= p, (p, hi)          # causality
        assert bool(torch.isin(tail, keys).all()), (p, tail, keys)   # tail always present
        # every scored pool lies entirely at or below p
        for j in sel[sel >= 0].tolist():
            assert (j + 1) * KPOOL <= seq, (p, j)
    print(f"\n(d') causality: {n_checked} positions checked, max(key - p) = {worst_hi} "
          f"(must be <= 0); tail present at every position")
    assert worst_hi <= 0


def test_d_correct_step_sees_what_a_stock_decode_sees():
    """The round's candidate set for p == a stock decode at p, given the same selected pools.

    Both paths derive the candidate set from `seq_len = p + 1` alone, so this compares the
    pseudo-sequence batch's per-row expansion against a one-row-at-a-time (stock decode)
    expansion, AND checks that the two address the same pooled-cache slots."""
    _, caches, _, _, _ = _fake_layer(64, seed=6)
    g = torch.Generator().manual_seed(23)
    P = torch.tensor([7, 11, 19, 34, 39, 40, 47], dtype=torch.int64)
    seq = P + 1
    select_k = 4
    pool_ids = torch.stack([
        torch.cat([torch.randperm(max(1, int(s) // KPOOL), generator=g)[:select_k].to(torch.int32),
                   torch.full((select_k,), -1, dtype=torch.int32)])[:select_k]
        for s in seq])

    batched = expand_pools_and_append_tail_ref(pool_ids, seq, KPOOL)
    n_slot_mismatch = 0
    for i, p in enumerate(P.tolist()):
        single = expand_pools_and_append_tail_ref(pool_ids[i:i + 1], seq[i:i + 1], KPOOL)
        assert torch.equal(batched[i:i + 1], single), p
        # and the pooled-cache slots each path may score against
        slots_round = scoreable_pool_slots(caches, p + 1)
        slots_stock = pool_slots(caches.block_row,
                                 torch.arange((p + 1) // KPOOL), NUM_STATES)
        n_slot_mismatch += int(not torch.equal(slots_round, slots_stock))
    print(f"\n(d'') round vs stock decode: {len(P)} rows, per-row expansion identical, "
          f"pooled-slot mismatches={n_slot_mismatch}")
    assert n_slot_mismatch == 0


def test_d_pool_never_leaks_a_future_row():
    """A SCORED pool can never contain a row later than the query -- the whole reason pooling is
    causally safe.  Pool p//kpool is scored only when p % kpool == kpool - 1."""
    bad = []
    for p in range(0, 64):
        seq = p + 1
        own_pool_scored = (p // KPOOL) < pool_len_of(seq, KPOOL)
        expect = (p % KPOOL) == KPOOL - 1
        if own_pool_scored != expect:
            bad.append(p)
        if own_pool_scored:
            hi = (p // KPOOL + 1) * KPOOL - 1
            if hi > p:
                bad.append(("leak", p, hi))
    print(f"\n(d''') own-pool scored iff p%kpool==kpool-1: violations={bad}")
    assert not bad


# ------------------------------------------------------------------------------------------- #
# (e) the pseudo-sequence indexer metadata (item b)
# ------------------------------------------------------------------------------------------- #

def test_e_pseudo_sequence_metadata_equals_per_request():
    """P one-token pseudo-sequences sharing one block row == P independent decode requests.

    Reproduces what the builder computes: `PrepareUniformDecodeKernel` at max_decode_len==1 is
    ``per_token_seq_len = max(seq_len - 1 + 0 + 1, 0) == seq_len`` plus a row copy, then
    ``seq_lens //= compress_ratio`` and ``unsqueeze(-1)``.  `schedule_metadata` consumes only
    that tensor, `num_states` and the SM count, so it cannot depend on the sharing."""
    block_row = torch.tensor([5, 2, 7, 1, 6, 0, 4, 3], dtype=torch.int32)
    P = torch.tensor([3, 7, 8, 130, 511, 512, 2051], dtype=torch.int64)
    md = pseudo_sequence_indexer_metadata(P, block_row, KPOOL, NUM_STATES)

    # per-request ground truth, one row at a time
    for i, p in enumerate(P.tolist()):
        one = pseudo_sequence_indexer_metadata(torch.tensor([p]), block_row, KPOOL, NUM_STATES)
        assert int(md["seq_lens_pool"][i, 0]) == int(one["seq_lens_pool"][0, 0])
        assert torch.equal(md["block_table"][i], one["block_table"][0])
        assert int(md["slot_mapping"][i]) == int(one["slot_mapping"][0])
        assert int(md["seq_lens_pool"][i, 0]) == (p + 1) // KPOOL
        assert int(md["slot_mapping"][i]) == (
            pool_slots(block_row, torch.tensor([p // KPOOL]), NUM_STATES).item()
            if (p + 1) % KPOOL == 0 else -1)
    assert torch.equal(md["decode_lens"], torch.ones(len(P), dtype=torch.int32))
    assert md["block_table"].shape == (len(P), block_row.numel())
    assert bool((md["block_table"] == block_row.to(torch.int32)[None, :]).all())
    print(f"\n(e) pseudo-sequence metadata: P={len(P)}, seq_lens_pool="
          f"{md['seq_lens_pool'].flatten().tolist()}, "
          f"slot_mapping={md['slot_mapping'].tolist()} (-1 = does not complete a pool), "
          f"decode_lens all 1, block table shared across all {len(P)} rows")


def test_e_slot_mapping_only_on_pool_completion():
    """The builder emits a valid pool slot only for tokens with (pos+1) % kpool == 0
    (`compressor_utils.py:72-83`); every other row is -1 and writes nothing."""
    block_row = torch.arange(8, dtype=torch.int32)
    P = torch.arange(0, 16, dtype=torch.int64)
    md = pseudo_sequence_indexer_metadata(P, block_row, KPOOL, NUM_STATES)
    valid = (md["slot_mapping"] >= 0)
    expect = ((P + 1) % KPOOL) == 0
    print(f"\n(e') pool-completion mask: valid={valid.nonzero().flatten().tolist()} "
          f"expect={expect.nonzero().flatten().tolist()}")
    assert torch.equal(valid, expect)


def test_e_sparse_enabled_default():
    print(f"\n(e'') SPARSE_ENABLED={SPARSE_ENABLED} (APPCORR_GLM53_SPARSE=0 disables)")
    assert SPARSE_ENABLED


def test_d_subbatching_never_reads_a_straddling_pool():
    """Splitting an ascending row set into sub-batches cannot expose a half-rewritten pool.

    When |P| exceeds max_num_seqs, `appcorr_rows_step` splits the round into contiguous slices
    of the ascending row list.  Sub-batch 1 rewrites the pool straddling the split while
    sub-batch 2's rows are still approximate.  That entry is never read by sub-batch 1: a
    straddling pool holds a row >= b (sub-batch 2's first row), a pool is scored by query p only
    when all its rows are <= p, and every sub-batch-1 query satisfies p <= a < b.  Checked
    exhaustively over every split of 40 random ascending row sets."""
    g = torch.Generator().manual_seed(31)
    n, violations, checked = 96, [], 0
    for trial in range(40):
        m = int(torch.randint(4, 40, (1,), generator=g))
        rows = torch.sort(torch.randperm(n, generator=g)[:m]).values
        for split in range(1, m):
            first, second = rows[:split], rows[split:]
            straddling = ({int(j) for j in torch.unique(first // KPOOL)}
                          & {int(j) for j in torch.unique(second // KPOOL)})
            for p in first.tolist():
                scored = set(range(pool_len_of(p + 1, KPOOL)))
                bad = scored & straddling
                checked += 1
                if bad:
                    violations.append((trial, split, p, sorted(bad)))
    label = "(d" + "'" * 4 + ")"
    print(f"\n{label} sub-batching: {checked} (split, query) pairs checked, "
          f"straddling-pool reads = {len(violations)}")
    assert not violations
