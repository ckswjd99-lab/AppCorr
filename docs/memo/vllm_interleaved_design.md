# Interleaved k<1 LLM correction inside vLLM (Qwen3.5) -- design + interfaces

Date: 2026-09-10. Status: DESIGN (implementation in progress on `develop/vllm-interleaved`).
Companion: `docs/memo/interleaved_correction_contract.md` (semantics), `docs/memo/vllm_stream_qwen35.md`
(the streaming form this extends), plan file `moonlit-foraging-frost.md` (why / what was cut).

## 0. One paragraph

The streaming form pushes the prompt to the engine in bands and prefills each band once. The
interleaved form pushes the WHOLE prompt once at t=0 with the image rows at their approximate
(base-resolution) value, lets the stock engine prefill it (the "approx pass"), and then, as each
band of corrected image rows arrives, re-runs the decoder on ONLY those rows -- rewriting their
K/V (softmax layers) and their recurrent contribution (Gated DeltaNet layers) in place -- so the
critical path after the last arrival is `k/g` of the image rows + the text suffix, not the whole
prompt. Total decoder compute = 1 (approx) + k·f_img + f_text. No depth staging in this MVP.

Notation: prompt = `[0,lo)` pre-image text · `[lo, lo+G)` image rows (one per merge group; G groups
in `g` bands, band r = groups `[g0_r, g1_r)`) · `[lo+G, N)` post-image text. Hold-back-one: row
`N-1` is never prefilled before `final` (existing streaming invariant, `request.py`). P_r = prompt
positions corrected in round r (`lo + group_idx` for the groups selected in band r; the last round
also carries the text suffix `[lo+G, N-1)`).

## 1. Semantics (ported from the HF contract; all four rules)

1. Round r corrects only P_r. Pre-image text is never corrected; post-image text only in the last
   round (it must see the fully corrected image).
2. Within a round the corrected rows first WRITE (K/V or side-buffer row), then READ: same-round
   peers at lower positions are visible; `key_pos <= query_pos` (causal). This is exactly what a
   decode-style batch of 1-token pseudo-sequences gives on the softmax layers, and what a re-scan
   of the round's window gives on the DeltaNet layers.
3. Non-corrected rows are never touched (their K/V and side-buffer rows stay). There is no
   "increment" to persist: each corrected row is recomputed from its embedding through all layers.
4. Coverage: at k=1 every image row is corrected exactly once and the text suffix once, so the
   final KV/state must equal a stock prefill of the corrected prompt (this is the gate).

## 2. Engine-side interface (agent A)

### 2.1 Entry point (in-process, `appcorr/vllm_stream/client.py`)

```python
class StreamingLLM:
    def open(self, request_id, chunk: StreamChunk, sampling_params) -> None   # existing
    def correct(self, request_id: str, positions: torch.Tensor, embeds: torch.Tensor,
                final: bool) -> dict:
        """Rewrite prompt rows `positions` (int64 [P], strictly increasing, all < N-1, none < lo
        for image rounds) with `embeds` (bf16 [P, D]) and re-run the decoder on them.

        Pre: the request was opened with final=False and has NOT been closed. Drives
        `self.step()` until the request's approx prefill is complete
        (scheduler.stream_state(...)['num_computed_tokens'] == num_prompt_tokens - 1), then
        calls the runner's `appcorr_correct_step` synchronously (no scheduler involvement).
        Post: if `final`, the request is closed exactly like `append(final=True)` with an EMPTY
        chunk (hold-back released; the next engine step computes row N-1 and samples). Row N-1
        itself is never in `positions`.
        Returns {"t_recv", "t_step_ms", "num_rows": P, "round": r}.
        """
```

Runner handle: `self.core.model_executor.driver_worker.worker.model_runner` (UniProcExecutor,
`VLLM_ENABLE_V1_MULTIPROCESSING=0`, which the server/gate already require). Assert the class is
`GPUModelRunner`.

Closing with an empty chunk: `StreamAppend(core_id, StreamChunk(embeds=empty[0,D], final=True,
mrope_positions=empty[3,0], mrope_delta=<unchanged>))` through the existing `engine_core.add_request`
path -- `stream_append` handles t=0 (cpu_cat of empty tensors must be a no-op; check `cpu_cat`).
Also scatter `embeds` into `CachedRequestState.prompt_embeds[positions]` and
`StreamingRequest.prompt_embeds[positions]` (CPU copies) so a preempted request re-prefills the
corrected rows.

### 2.2 Runner step (`appcorr/vllm_stream/correct.py`, installed by `runner_patch.install()`)

```python
def appcorr_correct_step(self: GPUModelRunner, req_id: str, positions: torch.Tensor,
                         inputs_embeds: torch.Tensor, window: tuple[int, int],
                         final: bool) -> None:
    """One eager forward of |P| pseudo-sequences.
    req_id      internal (core) request id
    positions   int64 [P] on device, sorted asc
    inputs_embeds bf16 [P, D] on device (the corrected rows)
    window      (start, end) prompt positions of this round for the DeltaNet re-scan:
                image round r -> (lo+g0_r, lo+g1_r); final text round -> (lo+G, N-1).
                Every element of `positions` lies in [start, end).
    final       True on the last round: also commit the DeltaNet state to the request's
                mamba block (2.4 e).
    """
```

Softmax layers (stock kernels, stock builders):
- `req_state = self.requests[req_id]`; `idx = self.input_batch.req_id_to_index[req_id]` (the request
  IS in the persistent batch: it was scheduled with 0 tokens last step; assert).
- mrope: `req_state.mrope_positions[:, positions]` -> `positions_gpu` [3, P] (the model's `positions`
  arg; `_prepare_inputs` feeds `self.mrope_positions.gpu[:, :n]` for M-RoPE models -- mirror that).
- Block table: `self.input_batch.block_table[kv_group].block_table.gpu[idx]` (one row); expand to
  [P, max_blocks]. Slot mapping per KV group: `block_ids[p // block_size] * block_size + p % block_size`
  computed from that row (the stock `_get_slot_mappings` does exactly this vectorised; reuse or copy
  its 3-line formula). For the mamba group (`mamba_cache_mode="none"`, one block per request, block
  size = max_model_len) the slot is irrelevant: the GDN path is bypassed (2.4).
- `CommonAttentionMetadata(query_start_loc=arange(P+1), query_start_loc_cpu=..., seq_lens=positions+1,
  seq_lens_cpu=..., num_computed_tokens_cpu=positions (cpu), num_reqs=P, num_actual_tokens=P,
  max_query_len=1, max_seq_len=int(positions[-1])+1, block_table_tensor=row.expand(P,-1),
  slot_mapping=slot_gid, causal=True, ...)` -- copy the exact field list from `_dummy_run`
  (`gpu_model_runner.py:~6178-6193`) and `_build_attention_metadata` (:2496-2509); then
  `builder.build(common_prefix_len=0, common_attn_metadata=cm)` per KV group -> `attn_metadata`
  dict {layer_name: md} for the softmax layers only. Do NOT put GDN layer names in the dict.
- `with set_forward_context(attn_metadata, self.vllm_config, num_tokens=P,
  cudagraph_runtime_mode=CUDAGraphMode.NONE, batch_descriptor=None, slot_mapping={layer: slot_gid ...}):
  out = self.model(input_ids=None, positions=positions_gpu, inputs_embeds=inputs_embeds, ...)`
  -- the KV write reads `forward_context.slot_mapping[layer_name]` (`attention.py:679-724`).
- The whole batch classifies as decode (`max_query_len == 1`): row p attends keys `0..p` on FA and
  FlashInfer alike (check which backend the engine picked from the startup log line and note it in
  the gate output). Hidden states out of the forward are discarded (no sampling).
- Pseudo-sequence count: P can be ~500 (k=1, band of a 2k-row image). If a builder needs
  `num_reqs <= max_num_seqs`, chunk `positions` into slices of `max_num_seqs` -- the pseudo-
  sequences are independent given write-before-read across slices is preserved (each slice writes
  its K/V before the next slice reads; rows within a slice see each other through the causal read
  of freshly written K/V ONLY if the kernel reads after the write -- it does: write is
  `unified_kv_cache_update` before the attention call). Slice boundaries are then invisible. The
  GDN re-scan (2.4) must run ONCE per round, not per slice: do it in the first slice and cache
  the window outputs for the later slices.

### 2.3 Side buffer (per request, engine process, outside vLLM's KV budget)

```python
@dataclass
class SideBuffer:            # one per correcting request, keyed by core req_id
    n: int                   # prompt length N (rows 0..N-1; row N-1 stays empty until final)
    lo: int                  # first image row
    mixed_qkv: list[Tensor]  # per GDN layer: bf16 [N, key_dim*2 + value_dim]  (pre-conv)
    b: list[Tensor]          # per GDN layer: [N, num_v_heads]
    a: list[Tensor]          # per GDN layer: [N, num_v_heads]
    ckpt: list[Tensor|None]  # per GDN layer: recurrent state after the last completed window,
                             #   fp32 [1, H, Dv, Dk] (the kernel's layout; check chunk.py)
    ckpt_end: int            # prompt position the ckpt covers ([0, ckpt_end) scanned)
```
35B: key_dim 2048 (x2 for q,k) + value_dim 4096 = 8192 dims bf16 = 16 KB/row/layer (it is the
`in_proj_qkvz` output minus z; use the layer's own `key_dim`/`value_dim` attrs), 30 layers ->
~0.5 MB/row; 2k rows ~1 GB/request. 122B: ~2x. Allocate lazily at the first capture, free in
`StreamingScheduler._free_request` (call a `correct.free(req_id)` hook).

### 2.4 GDN patch (`QwenGatedDeltaNetAttention._forward_core`, class-level; the custom op body
calls `self._forward_core(mixed_qkv, b, a, core_attn_out)` in Python, `qwen_gdn_linear_attn.py:1924`)

Module-level state `_MODE in {None, "capture", "correct"}` + `_CTX` (set by the runner wrapper /
`appcorr_correct_step` around the forward; reset in `finally`).

a) capture (during the stock approx prefill): before calling the stock `_forward_core`, copy
   `mixed_qkv[s], b[s], a[s]` for every `(req_id, token_slice s, pos0)` in `_CTX.captures` into
   `SB[req_id].mixed_qkv[layer_idx][pos0:pos0+len(s)]` etc. The runner wrapper (`execute_model`
   pre-hook after `_prepare_inputs`, or a wrapper around `_prepare_inputs`) fills `_CTX.captures`
   from `scheduler_output.num_scheduled_tokens` + `req_state.num_computed_tokens` for every request
   with a SideBuffer (positions are prompt positions; token slices follow the persistent batch's
   `query_start_loc`). `layer_idx` = index of this layer among the GDN layers (from `self.prefix`).
b) correct: `_CTX = {req_id, positions P (in batch order), window (s,e), layer outputs cache}`.
   1. `SB.mixed_qkv[l][P] = mixed_qkv; SB.b[l][P] = b; SB.a[l][P] = a` (write first).
   2. If `SB.ckpt_end[l] < s`: bring the checkpoint forward by scanning `[ckpt_end, s)` (this is
      the pre-image text on the first round: `ckpt_end==0`, `s==lo`). Then re-scan the window
      `[s, e)`: conv input = `SB.mixed_qkv[l][s-3:e]` (rows before 0 are zeros -- the stock conv
      state at prompt start is zero), depthwise causal conv with `self.conv1d.weight` (+bias if any)
      + the same activation the stock path applies (read `_forward_core`'s prefill branch and call
      the same helper; `causal_conv1d_fn` with `conv_states=None`/no cache is acceptable if it
      accepts that, else `F.conv1d(groups=C)` on a [1, C, L] view and drop the first 3 outputs).
      `fused_post_conv_prep(...)` -> q, k, v, g, beta for the window (same call as the stock
      prefill branch, :1416). `self.chunk_gated_delta_rule(q, k, v, g, beta,
      initial_state=SB.ckpt[l], output_final_state=True, cu_seqlens=tensor([0, e-s]), ...)`
      (same kwargs as :1506; `chunk_indices/chunk_offsets` as the stock code derives them for a
      single sequence) -> `out_w [e-s, H, Dv]`, `last_state`. `SB.ckpt[l] = last_state;
      SB.ckpt_end[l] = e`.
   3. `core_attn_out[:] = out_w[P - s]` (rows of the window that are in P, batch order). Return
      without calling the stock `_forward_core`.
   4. If `final` (window is the text suffix `[lo+G, N-1)`): after step 2, write the request's mamba
      block: `ssm_state[block] = SB.ckpt[l]` and `conv_state[block] = SB.mixed_qkv[l][N-4:N-1]`
      in the layout `MambaBase.bind_kv_cache` / the decode path expect (`causal_conv1d_update`
      reads `conv_state[block]` as [C, K-1] = the last K-1 pre-conv inputs, oldest first -- verify
      against the stock prefill's write-back and gate it: G2 compares the block bytes).
      Block id: `self.input_batch.block_table[mamba_group].block_table.gpu[idx, 0]`.
      The stock hold-back step then runs the decode path from the corrected state.
   Correctness note: the stock approx pass already wrote ssm/conv state for `[0, N-1)`; we
   overwrite both at final. Between rounds the block content is stale but unused (the request is
   scheduled with 0 tokens).

Non-goals: the fused-decode kernel path (`enable_fused_gdn_decode`, only with
`VLLM_GDN_DECODE_KERNEL=cuda`) is not patched -- assert it is off. Spec decode off.

### 2.5 Gate script `analysis/experiments/vllm_correct_gate.py` (in-process, one engine)

Model default `Qwen/Qwen3.5-4B` (dense, 32 layers, same `[lin,lin,lin,full]` pattern; weights
in `/NHNHOME/huggingface/hub` once the download finishes), `--gpu-mem 0.3`; `--model
Qwen/Qwen3.5-35B-A3B --gpu-mem 0.6` for the final run. Reuse `vllm_stream_gate.py`'s composer
(`Qwen25VLComposer` from `client.py`, works for Qwen3.5), images and comparison helpers.
"Approx" prompt for an image = the composer on `img.resize((w//4, h//4)).resize((w, h))` (same
grid, same N, low-frequency content); "corrected" = the composer on the original image.

- G1 softmax-only (`--gate g1`): approx == corrected (same embeds); open(final=False) -> run
  until computed == N-1 -> snapshot KV bytes at the request's slots for `[lo, N-1)` and the
  mamba block -> `correct(P=[lo, N-1), final=True)` with the GDN patch in a "passthrough" mode
  that writes NOTHING (stock `_forward_core` is skipped, `core_attn_out` left as the pre-correct
  output is not available -> instead compare only KV: rel-L2 of the KV bytes at the corrected
  slots before/after <= 1e-3, argmax of the first token equal, max |dlogit| reported) -> run to
  done. Report per layer rel-L2 (max over layers) + first-token dlogprob vs the same request
  run as a stock one-shot (`open(final=True)`).
- G2 identity (`--gate g2 --g 4`): approx = low-res embeds, corrected rows pushed band by band
  (bands = 4 contiguous group ranges), `final` on the last with the text suffix; compare against
  a one-shot open of the corrected prompt: first-token argmax equal on all 8 images; first-token
  dlogprob within the engine's own chunked-prefill band (35B gate table: ~3e-5..5e-5; for 4B
  measure arm D of `vllm_stream_gate.py` once); KV rel-L2 at all image + text slots; mamba
  block rel-L2 (ssm) and conv state. Generated sequence exact-match rate as in
  `vllm_stream_gate.py`.
- G3 (`--gate g3`): g=1 vs g=4, both vs one-shot -> same band.
Output json in `analysis/results/vllm_stream/correct_gate_<model>.json`.

## 3. Wire / server / driver interface (agent B)

### 3.1 `wire.py` -- new op
```
{"op": "correct", "rid", "final": bool, "positions": T(int64 [P]), "embeds": T(bf16 [P, D])}
        -> {"ok", "t_recv", "t_step_ms", "num_rows"}
```
M-RoPE is NOT sent (the runner keeps the request's positions). `window` is derived server-side
from `positions` when contiguous; for k<1 rounds (non-contiguous P) the client sends
`"window": [start, end]` explicitly -- make it always explicit: `"window": [s, e]` required.

### 3.2 `server._chunk` -- dispatch `correct` to `self.llm.correct(rid, positions, embeds,
window, final)`; `_check_len` as for append (positions < N-1). Errors abort the request like an
over-length append does. Timing: record `t_correct[r]` next to `t_final`; `final=True` sets
`t_final` (the anchor the latency probe uses = last band's arrival).

### 3.3 `bridge.py` -- `LLMBridge.correct(...)` (same tx thread, same ack path as `append`) and
`StreamSink.correct(positions, embeds, window, final)`: records a push rec like `push()` with
`"kind": "correct"`; `closed=True` on final; `_PUSH_DELAY_S` sleep applies (band spacing).

### 3.4 Axis (`appcorr/models/qwen_vl_axis.py`, `streaming_forward`) -- `llm_schedule` arg
("streaming" default | "interleaved"), plumbed from `qwen_vllm_accuracy.py --llm-schedule`.
Interleaved branch, sink path only (HF `sink is None` raises NotImplementedError):
- after `_approx_base`: `emb_all[:, lo:lo+G] = merger(x_base_out)` for ALL groups (merge in
  band-sized slices to bound memory), then `sink.push(emb_all[0, :seq], pos_3d[:, 0, :seq],
  rope_delta, final=False)` -- the approx prompt, whole, at t=0. NOTE the hold-back: the server
  prefills N-1 rows; row N-1 (the last text token) is never rewritten and is computed at final.
- per band r (same vision correct + `merged` computation as today): P_r = `lo + group_idx`
  (sorted), rows = `merged` rows of those groups; `window = (lo+g0, lo+g1)`.
  Last band: append the text suffix `emb_all[0, lo+G:seq-1]` with positions `[lo+G, seq-1)`
  to P_r/rows, `window = (lo+g0, seq-1)`?? -- NO: two windows (image band + text) are two
  separate DeltaNet windows only if non-adjacent; here `[lo+g0, lo+G)` and `[lo+G, seq-1)` ARE
  adjacent, so one `correct` with `window=(lo+g0, seq-1)` and P = image positions ∪ text positions
  is exact. Send ONE message: `sink.correct(P, rows, window=(lo+g0_last, seq-1), final=True)`.
  For k<1 the non-selected rows of the band are simply absent from P (they keep their approx
  value; the re-scan window still spans the band).
- `stats["chunks"]` gets `("correct", s, e, len(P))` records; `prefill_tokens` counts the approx
  pass N-1 + Σ|P_r|; `stats["llm_schedule"] = "interleaved"`.
- `prefill()` closure unused in this branch (assert `pos_done` semantics are not mixed).
Output tag: `interleaved_g{g}_k{keep}` (+`_c{c}`), files next to the streaming ones.

### 3.5 G5 driver gate -- `analysis/experiments/vllm_interleaved_axis_gate.py` (CPU or GPU, no
engine): run `streaming_forward` twice with a recording fake sink (streaming vs interleaved,
k=1, g=4, same image); assert per-position final embeddings bitwise equal (the union of the
interleaved pushes must equal the streaming pushes' rows position-by-position; the approx push
rows for image positions must equal `merger(x_base_out)`). Also k=0.5: positions in P_r ⊆ band r,
|P_r| = selected count, text suffix only in the last message.

### 3.6 Cost (`appcorr/flops/flops_analytic.py` Qwen3.5 entry + `analysis/experiments/flops_report_qwen35.py`)
Closed form per corrected row at prompt position p (per layer type; dims from the HF config):
- full-attention layer: q/k/v/o (+ gate) projections `2·D·(Hq·dh·2 + Hkv·dh·2)` + scores/values
  `2·2·Hq·dh·(p+1)`.
- GDN layer: `in_proj_qkvz` + `in_proj_ba` + `out_proj` GEMMs + conv `2·C·K` + scan per token
  (`_qwen35_deltanet_core_flops` in `appcorr/flops/hooks.py` -- same formula, per token).
- MoE per token: top-8 experts + shared expert (`_qwen35_experts_flops` formula per token);
  4B dense: the dense MLP.
- Re-scan overhead per round: `(e-s) × scan-per-token × n_gdn_layers` (counted; it is real).
- lm_head once (final step), embeddings zero.
Reconcile: closed-form prefill of N tokens vs the hooked floor/ceiling FLOPs already in
`analysis/results/qwen35_flops*` (or a fresh `flops_report_qwen35.py` run on 8 images) within 1%
BEFORE reporting. New keys: `total_g{g}_k{k}_il`, `crit_g{g}_k{k}_il` (interleaved), computed from
the driver's `stats["chunks"]` records (per-sample P_r sizes and windows) -- the report script
reads the accuracy jsonl's per-row stats, so make the driver store `chunks` in the jsonl row.

## 4. Gates summary

| gate | what | pass |
|---|---|---|
| G1 | softmax rewrite, identical embeds | KV rel-L2 <= 1e-3 at corrected slots; first-token argmax equal |
| G2 | k=1 g=4 low-res -> full-res vs one-shot | argmax equal 8/8; dlogprob in engine band; KV + ssm/conv block rel-L2 reported |
| G3 | g=1 vs g=4 | same band |
| G4 | k in {0.25, 0.5, 1} on 40 V* rows | floor <= k.25 <= k.5 <= ceiling (sanity, not identity) |
| G5 | axis pushes streaming vs interleaved, k=1 | bitwise per position |
| F | closed form vs hooked prefill | < 1% |

## 5. Not in scope (see plan): depth-staged approx pass, concurrency-integrated correct steps,
CUDA-graph capture of the correct step, HF hybrid reference, 122B beyond G2 + campaign.
