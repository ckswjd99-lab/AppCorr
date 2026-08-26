"""
streaming.py

Chunked (streaming) prefill for the Qwen3.5-MoE decoder -- the LLM-side counterpart to the vision
tower's approx/correct fork, and deliberately NOT an approx/correct fork itself.

**Why this decoder gets streaming instead of correction.** AppCorr's correction recomputes a chosen
subset of token positions and splices their fresh K/V into a cache. That is well defined for a
bidirectional vision tower, where a token's contribution is a K/V row that can be overwritten in
place. It is not well defined here: 30 of Qwen3.5-35B's 40 decoder layers (36 of 48 on the 122B) are
`linear_attention`, implemented as `Qwen3_5MoeGatedDeltaNet` -- a RECURRENT layer whose entire
history is compressed into a running state. There is no row for token i to overwrite; correcting
token i changes the state every later token sees, so "correct a subset" degenerates into "replay
everything after the earliest corrected position".

The good news is that the same property makes the alternative free. A causal decoder needs no
approximation at all: feed the sequence in contiguous chunks as they arrive, carrying the cache
forward, and the result is what a single prefill would have produced -- an identity in exact
arithmetic, not an approximation whose loss must be measured. That matches what this project already
found on LLaVA-OneVision2, where streaming prefill beat interleaved approx-then-correct outright
(ChartQA 85.0 vs 81.0) and did so regardless of the round count. Linear attention only strengthens
the case: a recurrent state is exactly a streaming primitive.

**The one real constraint: chunks must be contiguous in position space.** A chunk is appended to the
cache, so its positions must continue where the last chunk stopped. For AppCorr's interleaved vision
groups this makes the grouping strategy load-bearing rather than cosmetic -- SEQUENTIAL grouping
gives each round a contiguous run of image-token positions and works directly; GRID grouping
scatters a round's tokens across the sequence and cannot be streamed without reordering the tokens
the LLM sees. `assert_contiguous` enforces this instead of letting a grid-grouped config silently
produce a wrong prefill.
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch


def assert_contiguous(boundaries: Sequence[int], total: int) -> None:
    """Chunk boundaries must partition [0, total) into contiguous ascending runs.

    Raises rather than repairing. A boundary list that does not partition the sequence means the
    caller is streaming something this decoder cannot stream -- most likely a grid-grouped
    transmission schedule -- and the resulting prefill would be wrong while looking ordinary.
    """
    if not boundaries:
        raise ValueError("streaming prefill: empty boundary list")
    if list(boundaries) != sorted(boundaries) or len(set(boundaries)) != len(boundaries):
        raise ValueError(f"streaming prefill: boundaries must be strictly ascending, got {list(boundaries)}")
    if boundaries[0] != 0 or boundaries[-1] != total:
        raise ValueError(
            f"streaming prefill: boundaries must span [0, {total}], got "
            f"[{boundaries[0]}, {boundaries[-1]}]. Chunks are APPENDED to the cache, so a gap or an "
            "overlap silently shifts every later position."
        )


def stream_prefill(
    model: torch.nn.Module,
    input_ids: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.Tensor] = None,
    boundaries: Optional[Sequence[int]] = None,
    past_key_values: Optional[Any] = None,
    flop_counter: Optional[Any] = None,
    position_ids: Optional[torch.Tensor] = None,
    **model_kwargs: Any,
) -> Tuple[torch.Tensor, Any]:
    """Prefill `[1, T, ...]` in contiguous chunks, returning the FINAL chunk's logits and the cache.

    Only the last chunk's logits are returned because that is all a prefill is for: the next token is
    sampled from the final position. Earlier chunks' logits are computed (they are on the critical
    path for nothing) and dropped.

    Args:
        boundaries: chunk edges including 0 and T, e.g. `[0, 256, 512, 900]` for three chunks. A
            single chunk `[0, T]` reproduces an ordinary one-shot prefill and is the identity case
            the gate checks.
        position_ids: FULL-SEQUENCE positions, last dim T, sliced per chunk. REQUIRED for any
            multimodal prompt on this family: Qwen3.5 gives image tokens interleaved 3D M-RoPE
            positions (shape (4, B, T) via `get_rope_index`), and the arange fallback below
            replicates a 1D counter across all four axes -- plausible logits, wrong image geometry,
            and invisible to any A/B whose two arms share the fallback (the trap Qwen2.5-VL's
            M-RoPE bug already sprang once). The fallback exists for TEXT-ONLY sequences, where
            1D-replicated is what get_rope_index itself produces.
        flop_counter: optional `FlopCounter`. Each chunk is recorded under its own arrival index, so
            the critical/overlappable split falls out of the schedule with no special casing: only
            the final chunk carries the highest index and is therefore critical.

    Returns:
        (logits_of_last_chunk, cache)
    """
    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("streaming prefill: pass exactly one of input_ids / inputs_embeds")
    seq = input_ids if input_ids is not None else inputs_embeds
    if seq.shape[0] != 1:
        # Not a fundamental limit, but every eval driver here runs one request at a time and a
        # padded batch would make the position bookkeeping below silently wrong for short rows.
        raise ValueError(f"streaming prefill: batch size must be 1, got {seq.shape[0]}")
    total = seq.shape[1]
    boundaries = list(boundaries) if boundaries is not None else [0, total]
    assert_contiguous(boundaries, total)

    if past_key_values is None:
        from transformers.cache_utils import DynamicCache
        past_key_values = DynamicCache(config=model.config)

    logits = None
    for r, (lo, hi) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        # Positions continue where the previous chunk stopped -- the whole point of the contiguity
        # check above. Passed explicitly rather than left to the model's cache-length inference,
        # which is what breaks first when a caller streams something non-contiguous.
        if position_ids is not None:
            chunk_pos = position_ids[..., lo:hi]
        else:
            chunk_pos = torch.arange(lo, hi, device=seq.device).unsqueeze(0)
        chunk_kwargs = dict(model_kwargs)
        if input_ids is not None:
            chunk_kwargs["input_ids"] = input_ids[:, lo:hi]
        else:
            chunk_kwargs["inputs_embeds"] = inputs_embeds[:, lo:hi]

        ctx = flop_counter.arrival(r) if flop_counter is not None else _null()
        with ctx:
            out = model(
                position_ids=chunk_pos,
                past_key_values=past_key_values,
                use_cache=True,
                **chunk_kwargs,
            )
        past_key_values = out.past_key_values
        logits = out.logits

    return logits, past_key_values


class _null:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False
