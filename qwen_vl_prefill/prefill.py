"""
prefill.py -- monolithic vs chunked LLM prefill over the stock Qwen2.5-VL text model, using the
standard HuggingFace KV-cache API (DynamicCache + past_key_values). No model introspection and no
benchmark logic here (kept separate).

The whole point is that CHUNKED prefill -- feeding the sequence to the LLM in contiguous chunks and
appending each to the same KV cache -- must reproduce MONOLITHIC prefill (feeding the whole sequence
at once) up to numerical tolerance, GIVEN identical embeddings and identical M-RoPE position_ids.
That equivalence is what makes progressive visual-token streaming into the LLM correct.

Key correctness details (verified against transformers 5.13.0):
  - We drive `model.model.language_model` (Qwen2_5_VLTextModel) directly with `inputs_embeds`, so the
    vision encoder is NOT re-run and no image re-splicing happens inside the LLM forward.
  - position_ids are the FULL-sequence M-RoPE map [3,1,T] computed once (see introspect.compute_position_ids),
    sliced per chunk. Never recomputed per chunk.
  - Each chunk passes `cache_position = arange(a, b)` and the SAME DynamicCache, so the causal mask
    over the growing cache is built correctly by the text model; attention_mask is left None (pure
    causal prefill, no padding).
  - logits come from `model.lm_head` applied to the text model's last_hidden_state.
"""

from typing import List, Tuple

import torch


@torch.inference_mode()
def monolithic_prefill(model, inputs_embeds: torch.Tensor, position_ids: torch.Tensor):
    """Standard single-shot prefill over the whole [1, T, H] sequence.
    Returns (logits [1, T, vocab], past_key_values)."""
    lm = model.model.language_model
    out = lm(
        inputs_embeds=inputs_embeds,
        position_ids=position_ids,
        use_cache=True,
    )
    logits = model.lm_head(out.last_hidden_state)
    return logits, out.past_key_values


@torch.inference_mode()
def chunked_prefill(
    model,
    inputs_embeds: torch.Tensor,
    position_ids: torch.Tensor,
    chunk_boundaries: List[Tuple[int, int, str]],
):
    """Prefill the sequence chunk-by-chunk into ONE growing KV cache, exactly as a progressive
    visual-token stream would. Each chunk_boundaries entry is (start, end, label) covering a
    contiguous absolute span; they must partition [0, T) in order.

    Returns (logits [1, T, vocab] reassembled from per-chunk outputs, past_key_values, per_chunk_hidden).
    The reassembled logits let us compare against monolithic prefill position-by-position."""
    from transformers import DynamicCache

    lm = model.model.language_model
    cache = DynamicCache()
    device = inputs_embeds.device

    # validate the boundaries partition [0, T) contiguously and in order
    expected = 0
    T = inputs_embeds.shape[1]
    for a, b, _ in chunk_boundaries:
        assert a == expected, f"chunk gap/overlap: expected start {expected}, got {a}"
        expected = b
    assert expected == T, f"chunks cover [0,{expected}) but sequence is length {T}"

    per_chunk_logits = []
    for a, b, _label in chunk_boundaries:
        chunk_embeds = inputs_embeds[:, a:b]
        chunk_pos = position_ids[:, :, a:b]
        cache_position = torch.arange(a, b, device=device)
        out = lm(
            inputs_embeds=chunk_embeds,
            position_ids=chunk_pos,
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
        )
        cache = out.past_key_values
        per_chunk_logits.append(model.lm_head(out.last_hidden_state))

    logits = torch.cat(per_chunk_logits, dim=1)  # [1, T, vocab]
    return logits, cache, per_chunk_logits


def compare_logits(logits_a: torch.Tensor, logits_b: torch.Tensor) -> dict:
    """Max/mean absolute + relative difference between two [1, T, vocab] logit tensors, plus whether
    the argmax next-token (last position) agrees."""
    a = logits_a.float()
    b = logits_b.float()
    abs_diff = (a - b).abs()
    denom = a.abs().clamp_min(1e-6)
    rel_diff = abs_diff / denom
    last_a = a[:, -1].argmax(dim=-1)
    last_b = b[:, -1].argmax(dim=-1)
    # per-position argmax agreement across the whole sequence
    argmax_agree = (a.argmax(-1) == b.argmax(-1)).float().mean().item()
    return {
        "max_abs_diff": abs_diff.max().item(),
        "mean_abs_diff": abs_diff.mean().item(),
        "max_rel_diff": rel_diff.max().item(),
        "last_token_argmax_match": bool((last_a == last_b).all().item()),
        "argmax_agreement_frac": argmax_agree,
        "shape": tuple(a.shape),
    }
