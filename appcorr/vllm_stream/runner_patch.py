"""GPUModelRunner hooks: grow a request's prompt in the worker when a chunk lands.

The worker keeps its own copy of every request's prompt (`CachedRequestState.prompt_embeds`,
`num_prompt_tokens`, M-RoPE positions) and a persistent batch (`InputBatch`) built from it.
Stock cached-request updates carry no prompt data, so on a `SchedulerOutput.appcorr_stream_updates`
entry we (1) extend the cached state and (2) evict the request from the persistent batch so
the stock `_update_states` re-adds it from the grown state -- the same path a request takes
after a step it was not scheduled in, so no InputBatch internals are touched here. M-RoPE
positions for an embeds-only prompt cannot be derived by the runner (0.11.2 crashes, 0.28.0
assigns plain text positions), so `_init_mrope_positions` takes them from
`NewRequestData.appcorr_stream` instead. On 0.11.2 only, `_preprocess` fixes a stock bug that
discards prompt_embeds on multimodal models (fixed upstream by 0.28.0, see below).
"""
from __future__ import annotations

import torch
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from .request import cpu_cat

_ORIG_UPDATE = GPUModelRunner._update_states
_ORIG_INIT_MROPE = GPUModelRunner._init_mrope_positions
_ORIG_PREPROCESS = GPUModelRunner._preprocess


def _update_states(self: GPUModelRunner, scheduler_output) -> None:
    # (a) mrope positions for streaming requests at first schedule
    stash = {}
    for nr in scheduler_output.scheduled_new_reqs:
        info = getattr(nr, "appcorr_stream", None)
        if info is not None:
            stash[nr.req_id] = info
    self._appcorr_new_info = stash

    # (b) appended chunks for requests the worker already holds
    updates = getattr(scheduler_output, "appcorr_stream_updates", None)
    if updates:
        for req_id, chunk in updates.items():
            st = self.requests.get(req_id)
            if st is None:  # finished/aborted meanwhile
                continue
            assert st.prompt_token_ids is None and st.prompt_embeds is not None, req_id
            st.prompt_embeds = cpu_cat(st.prompt_embeds, chunk.embeds, dim=0)
            st.num_prompt_tokens = int(st.prompt_embeds.shape[0])
            if chunk.mrope_positions is not None:
                assert st.mrope_positions is not None, req_id
                st.mrope_positions = cpu_cat(st.mrope_positions, chunk.mrope_positions, dim=1)
                st.mrope_position_delta = chunk.mrope_delta
            if req_id in self.input_batch.req_id_to_index:
                # force the stock re-add path (persistent batch rebuilt from `st`)
                self.input_batch.remove_request(req_id)
    return _ORIG_UPDATE(self, scheduler_output)


def _init_mrope_positions(self: GPUModelRunner, req_state) -> None:
    info = getattr(self, "_appcorr_new_info", {}).pop(req_state.req_id, None)
    if info is None:
        return _ORIG_INIT_MROPE(self, req_state)
    assert info.mrope_positions is not None, (
        f"{req_state.req_id}: M-RoPE model but the streaming request carried no positions")
    assert info.mrope_positions.shape[1] == req_state.num_prompt_tokens, (
        info.mrope_positions.shape, req_state.num_prompt_tokens)
    req_state.mrope_positions = info.mrope_positions
    req_state.mrope_position_delta = info.mrope_delta


def _preprocess(self: GPUModelRunner, scheduler_output, num_input_tokens, intermediate_tensors=None):
    """(0.11.2 only; 0.28.0 embeds the token-id rows through `torch.where(is_token_ids, ...)` and
    leaves the staged rows alone.) vllm 0.11.2 drops `prompt_embeds` on multimodal models: `_prepare_inputs` stages them in
    `self.inputs_embeds` (rows where `is_token_ids` is False), but the multimodal branch of
    `_preprocess` then overwrites the whole buffer with `embed_input_ids(input_ids)` -- the
    placeholder ids of an embeds request -- so the model sees garbage (the text-only
    `elif enable_prompt_embeds` branch that honours the staged rows is never reached). Snapshot
    the staged rows and restore them after the stock call. Row order is the persistent batch's,
    both before and after (`_prepare_inputs` already ran)."""
    n = scheduler_output.total_num_scheduled_tokens
    keep = None
    if self.enable_prompt_embeds and self.supports_mm_inputs and n > 0:
        is_tok = self.is_token_ids.gpu[:n]
        idx = (~is_tok).nonzero(as_tuple=False).squeeze(1)
        if idx.numel() > 0:
            keep = (idx, self.inputs_embeds.gpu[idx].clone())
    out = _ORIG_PREPROCESS(self, scheduler_output, num_input_tokens, intermediate_tensors)
    if keep is not None:
        idx, rows = keep
        self.inputs_embeds.gpu[idx] = rows
    return out


def install() -> None:
    if getattr(GPUModelRunner, "_appcorr_stream_patched", False):
        return
    from . import vllm_version
    GPUModelRunner._update_states = _update_states
    GPUModelRunner._init_mrope_positions = _init_mrope_positions
    if vllm_version() == "0.11.2":
        GPUModelRunner._preprocess = _preprocess
    GPUModelRunner._appcorr_stream_patched = True
