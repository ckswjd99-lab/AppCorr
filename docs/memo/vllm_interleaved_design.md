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
prompt. Total decoder compute = 1 (approx) + k·f_img + f_text. No depth staging in the MVP;
the depth-staged form (§7.11, `interleaved_staged`) was added on top of it on 2026-09-10.

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

## 5. Not in scope (see plan): concurrency-integrated correct steps, CUDA-graph capture of
the correct step, HF hybrid reference, 122B beyond G2 + campaign. (The depth-staged approx
pass, originally listed here, exists since 2026-09-10 in its "simulation" form -- §7.11.)

## 6. Engine-side implementation notes (agent A, 2026-09-10)

The engine side is implemented in `appcorr/vllm_stream/correct.py` (+ `client.py` `open(correct=True)`
/ `correct()`, the `runner_patch.install()` and `StreamingScheduler._free_request` hooks) and gated by
`analysis/experiments/vllm_correct_gate.py`. Corrections to §2 found while building it:

1. **`VLLM_GDN_DECODE_KERNEL` defaults to `cuda` in 0.28.0**, not off. `forward_cuda` then calls
   `qwen_gdn_attention_core_fused_norm_packed` -> `_forward_core_fused_norm_packed`
   (`qwen_gdn_linear_attn.py:1781`) for *every* batch, prefill included, so the `_forward_core`
   patch never fires and both capture and correct silently no-op. Every interleaved run must set
   `VLLM_GDN_DECODE_KERNEL=triton`; `correct.check_gdn_path()` asserts it at `open(correct=True)`.
2. **Block 0 is the null block** (`NULL_BLOCK_ID = 0`, `v1/attention/backends/utils.py:46`).
   `causal_conv1d_fn` with `cache_indices=[0]` skips the conv-state write *and leaves that
   sequence's conv output in the uninitialised `torch.empty_like` buffer*
   (`causal_conv1d.py:144`). The re-scan's scratch conv cache must therefore be 2 blocks with
   `cache_indices=[1]`. This was the whole of the first G2 failure (ssm rel-L2 ~20).
3. **Qwen3.5 has several mamba KV-cache groups** (4B: groups 0-2 GDN + group 3 attention), so §2.4e's
   single `mamba_group` block id is wrong -- the block must be looked up per GDN layer.
4. §2.2's `positions`/`seq_lens` metadata works as written, but note the hybrid-block case: with
   DCP/PCP world 1 the stock slot-mapping kernel (`block_table.py:397-442`) still reduces to
   `block_table[pos // kernel_block_size] * kernel_block_size + pos % kernel_block_size`.
5. §2.5's G1 "passthrough that writes NOTHING" cannot work (a zero `core_attn_out` corrupts the
   residual stream). G1 is implemented as a *replay* mode: the approx pass optionally also captures
   `core_attn_out`, and the correct step replays it for the corrected rows -- the softmax rewrite is
   then genuinely isolated.
6. §4's `KV rel-L2 <= 1e-3` pass mark is below the engine's own floor. Added an in-engine control
   arm `chunk` (the same corrected prompt prefilled in 4 streaming chunks): its KV rel-L2 vs
   one-shot is 1.2e-2..2.9e-2 and its ssm rel-L2 2.1e-2..3.7e-2 on the 4B. Every gate is judged
   against that band, not against an absolute constant. A new gate `g0` isolates the DeltaNet
   re-scan alone (corrected prompt, one row rewritten, window = the whole prompt) and compares
   `_rescan`'s state against the block the stock prefill left: layer 0 is bit-identical.
7. A round re-scans `[ckpt_end, window_end)` in ONE `chunk_gated_delta_rule` call (not "advance the
   checkpoint to s, then scan [s,e)"): the two differ by an extra cast of the intermediate state and
   the single call is what a stock chunked prefill does.
8. The side buffer is created by `StreamingLLM.open(..., correct=True)` directly on the runner
   rather than through `NewRequestData.appcorr_stream`; no wire/scheduler field was needed.
9. `P` pseudo-sequences means `max_num_seqs >= P` (FlashInfer sizes `paged_kv_indptr` by
   `max_num_seqs`, `flashinfer.py:921`); the gate runs with `--max-num-seqs 2048`.

## 7. Integration notes (main session, 2026-09-10)

Driver side (agent B) merged into the engine worktree; fixes found on the first served run (4B):

1. **Open-time flag.** The engine only keeps the DeltaNet side buffer for requests opened with
   `StreamingLLM.open(..., correct=True, image_start=lo)`, so the `open` wire op carries
   `"correct": true, "image_start": lo` (optional keys, `wire.py`). The axis passes
   `correct_from=lo` on the interleaved schedule's t=0 `sink.push(...)`; `StreamSink.push` refuses
   it on any push but the opening non-final one; `LLMBridge.open` puts it in the header; the server
   forwards it to `llm.open`. Streaming-schedule sinks never see the kwarg.
2. **`--interleaved` server flag** sets `VLLM_GDN_DECODE_KERNEL=triton` (§6.1) before vLLM is
   imported and requires `--max-num-seqs` (>= 512; 1024 for the table's datasets at g=4: the
   largest correct batch is the last band's rows + the text suffix, 379 rows on RWQA/4B, ~330 on
   V*). `check_gdn_path()` still asserts at open time.
3. **The persistent batch is not a stable handle.** `_update_states` removes every request the
   step did not schedule from `input_batch` (`unscheduled_req_ids -> remove_request`) while
   keeping it in `runner.requests`; between the approx prefill's last step and the first
   `correct` the server keeps stepping (idle steps at c=1, other requests' steps at c>1), so the
   correct step must not index `input_batch` by request. `correct.py` now builds the block-table
   row from `runner.requests[rid].block_ids[gid]` (`_block_row`, kernel-block mapping applied,
   zero-padded to the batch's row width) for both the slot mapping and the mamba blocks. The
   in-process gate never hit this because its drain loop's last step scheduled the request.
4. **Drain steps go through the server's `_step`.** `client.correct(..., step_fn=server._step)`:
   the drain to "approx prefill complete" advances every other live request too, and their
   first-token times / finished outputs are booked in `_step`; draining with the bare
   `client.step` would silently drop them (their `result` would block forever at c>1).
5. `StreamingScheduler._free_request` forwards vLLM 0.28's `delay_free_blocks=` kwarg (the
   abort path passes it; the old signature raised inside the abort-on-error handler).

First served run (4B, RWQA 8 rows, one server for all arms): interleaved k=1 preds == ceiling ==
streaming 8/8; k=0.5 == floor on the one row where floor differs from ceiling. Chunks of row 0:
`("approx",0,1362)`, 4 corrects of 326/326/327/379 rows (last = 326 image + 53 text suffix).
6. **Rounds larger than `max_num_seqs` are sub-batched** (`appcorr_correct_step` ->
   `_correct_sub`): V* prompts reach ~8k image rows, i.e. ~2k rows per g=4 band at k=1, above the
   1024 the 35B server can run (mamba block count caps `max_num_seqs` at gpu-mem 0.6). Sub-batches
   run in position order; each is exactly one batch's worth of the original step, and the split is
   equivalent because a corrected row p reads rows <= p only (softmax K/V written by earlier
   sub-batches at every layer; the DeltaNet re-scan output at p depends on side-buffer rows <= p).
   The checkpoint is advanced by the last sub-batch (`_rescan(commit=False)` before that), so the
   earlier sub-batches repeat the window scan -- reported as `n_sub` in the correct reply; the
   closed-form FLOPs count one scan per round (understates by (n_sub-1) x scan(window) x 30 layers
   on the few rows where it triggers). Equivalence gate (4B, 8 COCO images, g2 at
   `--max-num-seqs 32` = 2-4 sub-batches per round vs `2048` = one batch): first-token
   |dlogprob| <= 3.8e-4, max over 48 generated tokens <= 0.065, one divergence at token 39 -- the
   same band as the stock one-shot prefill's own run-to-run difference between the two engine
   configurations (`one` vs `one`: <= 2.2e-4 / <= 0.084 / one divergence at token 41), so no
   bitwise reference exists across configurations and the split is gated at the band level
   (`logs/vllm_stream/il4b_subgate_mns{2048,32}.log`). On 35B V* at k=1 the last band reaches
   |P| = 1259 > 1024, i.e. `n_sub` = 2 on those rows.
7. **35B end-to-end (2026-09-10, served, one interleaved server per chain).** G5 on the real
   tower PASS (k=1 and k=0.5, bitwise per-position embeds vs the streaming arm). 40-row V* (c4,
   pyr): ceiling 87.5, streaming 87.5, interleaved k=1 85.0 (+0/-1 paired vs ceiling), k=0.5 85.0
   (+0/-1), k=0.25 82.5 (+1/-3), floor 72.5 (+1/-7) -- G4 order holds; max |P| 1259 (n_sub=2
   exercised); 0 server tracebacks. FLOPs (`qwen35_flops_il.json`, V*, GF): full 49677; streaming
   crit/total k1 12592/80511, k.5 8663/64785, k.25 6694/56918; interleaved k1 12653/99219
   (total = 2.0x full, the no-depth-staging MVP), k.5 6505/74959, k.25 3425/62822 -- reconciled
   with total = 1 + k*f_img + f_text (decoder ~18 TF, vision ~31.7 TF per full pass); crit shrinks
   3.7x from k=1 to k=0.25 (streaming: 1.9x), which is the claim the arm exists for.
   **Latency anchor fix.** The first probe reported Crit. Lat. ~8 ms: the server stamped `t_recv`
   after `llm.correct` returned and `rows_of` anchored at that stamp, so the correct step itself
   was outside the window. Now `t_recv` = arrival, `t_done` = after drain + step, and the
   interleaved arm uses the streaming anchor (last band's pixel arrival = `t_pushes[-2] + delay`,
   g+1 messages), decomposed as last-band vision correct / transport / drain+step
   (`last_correct_step_ms` = t_done - t_recv) / hold-back step. Re-probe d150, 36 samples, c1
   (`inprocess_latency.json` key `qwen35_35b_il`; streaming reference `qwen35_moe_cg1024`):
   V* k1 99.0 / k.5 85.2 / k.25 76.9 ms (streaming 62.8), RWQA 72.1 / 69.9 / 68.4 (streaming
   33.8). Decomposition, V* k1: vision correct 34.4 (streaming 34.3) + correct step 55.2 +
   hold-back 8.6; the step is flat in |P| and in prompt length (55.2 / 53.1 / 53.9 ms on V* for
   822 / 411 / 206 rows, 52-54 ms on RWQA with 1351-token prompts) while the vision half scales
   with k (34.4 / 23.4 / 13.5).
   **Profile** (`APPCORR_CORRECT_PROFILE=<dir>:<first>:<count>` -> `corrects.jsonl`, 6 correct
   calls at k=1, |P| 822-868): CUDA time 39-44 ms per step out of 64-142 ms wall under the
   profiler, ~3800-3900 kernel launches, `drain_steps` 0 (the approx prefill is already complete
   when the first band lands under d150); top kernels moe_forward_shared 10.2 ms (40 calls),
   the two GDN-projection bmm 6.2 + 3.3 ms, aten::mm 2.9 ms (270 calls), gdn core 2.0 ms,
   attention 1.5 ms. HYPOTHESIS (evidence: flat step time vs |P|, ~3.9k launches, CUDA 40 ms <<
   wall): the eager correct step is CPU-launch-bound, ~50 ms fixed, so the interleaved Crit. Lat.
   on 35B is overhead-bound rather than FLOPs-bound; CUDA-graph capture of the step (cut from the
   MVP, plan "What was cut") is the lever, not fewer rows. Not yet tested: a captured step or
   `torch.compile` of `_correct_sub`.
8. **35B full-dataset paired results (2026-09-10 03:32-04:11, `il_next_chain.sh` step A, one
   interleaved server, c4, 0 tracebacks).** Eight arms per dataset on one engine, all rows paired.
   V* (191, pyr; `qwen_vllm_accuracy_il_pyr/`): floor 73.30 / streaming 83.77 / ceiling 84.82;
   streaming k.5 84.29, k.25 81.68; interleaved k1 84.29, k.5 84.82, k.25 81.68. Paired
   interleaved vs streaming: k1 +3/-2, k.5 +2/-1, k.25 +2/-2 (of 191). vLLM bounds match the HF
   campaign within one row (HF ceiling 84.29, floor 72.77, streaming 83.77; HF k.5 82.20 / k.25
   81.15 vs vLLM 84.29 / 81.68 -- served numbers, chunked prefill, not the same engine).
   RWQA (765, box; `qwen_vllm_accuracy_il/`): floor 74.25 / streaming 76.99 / ceiling 78.04;
   streaming k.5 76.34, k.25 77.12; interleaved k1 76.47, k.5 76.60, k.25 76.47. Paired
   interleaved vs streaming: k1 +4/-8 (-0.52 pp), k.5 +12/-10 (+0.26), k.25 +8/-13 (-0.65) --
   every delta inside +-13 rows of 765; preservation vs ceiling 98.0 / 98.2 / 98.0 %.
   The k-ordering is not monotone on RWQA for either schedule (streaming k.25 > k1 > k.5),
   as on the HF path: the floor-ceiling gap is 3.8 pp = 29 rows, so k<1 differences are noise.
   FLOPs (`qwen35_flops_il.json`, `qwen35_flops_il_rwqa.json`; closed form replayed over the
   actual windows of every row): V* interleaved crit 12.80 / 6.58 / 3.46 TF (25.8 / 13.2 / 7.0 %
   of the 49.68 TF full pass) vs streaming 12.59 / 8.66 / 6.69 (25.3 / 17.4 / 13.5 %); total
   100.3 / 75.8 / 63.6 TF (202 / 153 / 128 %) vs 80.5 / 64.8 / 56.9 (162 / 130 / 115 %). RWQA
   interleaved crit 4.03 / 2.16 / 1.22 TF (25.4 / 13.6 / 7.7 % of 15.86 TF) vs streaming 4.14 /
   3.09 / 2.57 (26.1 / 19.5 / 16.2 %); total 30.6 / 23.2 / 19.5 TF (193 / 146 / 123 %) vs 24.1 /
   19.9 / 17.8 (152 / 126 / 112 %). Table: `make_eval_table.py --table interleaved` ->
   `analysis/results/interleaved_table_20260910.tex` (`\label{tab:interleaved_results}`;
   presentation vs the main table still to be decided with the user).
9. **122B-FP8 G2 (2026-09-10 04:11-04:28, in-process, `--gpu-mem 0.85 --max-model-len 8192
   --max-num-seqs 256`, 8 gate images, `logs/vllm_stream/g2_122b.{log,json}`).** Judged against
   the stock chunked-prefill control (`chunk`) of the same prompts, not the 35B band (FP8
   chunking is lossy on 122B: memo `project_qwen35_fp8_chunking_lossy`). Summary vs one-shot:
   chunk -- first-token argmax 8/8, seq exact 6/8, max |dlogprob| first 5.10e-2, KV rel-L2 max
   0.218, ssm rel-L2 max 0.146, conv |d| max 1.25; g2 -- argmax 8/8, seq exact 4/8, |dlogprob|
   first 4.74e-2, KV 0.228, ssm 0.163, conv 1.63. Per image the two arms interleave on every
   metric (KV 0.14-0.23 for both, ssm 0.12-0.16, dlp0 up to 5e-2 on both; g2 divergence points
   at 4/4/36/43 tokens vs chunk's 11/13): PASS relative to the control band; the band is the
   engine's FP8 chunking band, not an interleaved-path residual. Engine facts: FlashInfer
   backend, KV cache 29.9 GiB = 729k tokens at mns 256; init 111 s (compile 61 s).
10. **122B-FP8 served probe + latency (2026-09-10 04:28-04:52, `--gpu-mem 0.85 --max-model-len
    8192 --max-num-seqs 256 --interleaved`, init 15 min, 0 tracebacks).** 40-row V* (pyr, c2,
    paired vs ceiling 82.50): floor 75.00 (+1/-4), streaming 87.50 (+2/-0), interleaved k1
    85.00 (+2/-1; vs streaming +0/-1), k.5 85.00 (+2/-1), k.25 77.50 (+2/-4); max |P| 1259 at
    k=1, sub-batched through the 256 cap. Streaming k<1 was not part of the 122B probe arms.
    Latency (`inprocess_latency.json` key `qwen35_122b_il`, d150, 36 samples, c1): V* 742 /
    207 / 87 ms, RWQA 143 / 80 / 75 (streaming reference `qwen35_122b_cg1024`: V* 86 / 75 /
    65). Decomposition (medians): the correct step is quantized at ~58 ms x ceil(|P|/256) --
    60 / 118 / 228 ms for last-band |P| 251 / 457 / 868 on V*, 59 / 62 / 119 for 132 / 214 /
    374 on RWQA -- i.e. the same ~55 ms fixed cost per launch batch as on 35B, multiplied by
    the sub-batch count the 256-sequence cap forces (35B ran at 1024 = one sub-batch). Per-row
    compute is ~3x 35B's and a sub-batch costs the same: consistent with the launch-bound
    hypothesis of item 7, still unproven by a captured step. At k=1 on V* the 228 ms rounds
    exceed the 150 ms band spacing, so the last band's message queues behind earlier rounds:
    398 of the 742 ms is `last_chunk_wait` (server-side queueing), 92 vision correct, 228 step,
    11 hold-back. Lever: `--max-num-seqs 1024` on the 122B server (KV cache had 29.9 GiB free at
    256; the side buffer is outside the KV budget) -- a ~25 min re-probe, not run (needs a go).
    **FLOPs finding.** The 122B decoder closed form (`MODELS35["qwen35_122b"]`, config-checked:
    48 layers, 12 full / 36 linear, H 3072, 256 experts top-8, I 1024, shared 1024) gives
    60.0 TF for a stock prefill of N=3341 (the V* median), but the campaign's hooked 122B
    reference (`flops/qwen35_122b_vstar_flops.json`, full 66.56 TF) implies 34.9 TF of decoder.
    Cause (code-confirmed): `hooks.install` matches `type(mod).__name__` against
    `_SPECIAL_HOOKS`, and on the FineGrainedFP8 checkpoint transformers 5.13 instantiates the
    experts as `FP8Experts`, not `Qwen3_5MoeExperts` -- the routed-expert + router term
    (2*8*3*1024*3072 + 2*3072*256 per layer = 7.32 GF/token) was never counted on ANY hooked
    122B run. Check: 60.0 - 7.32e-3 * 3341 = 35.5 TF vs hooked 34.9 (1.7%); with the corrected
    full (~91.1 TF) the interleaved shares become crit 25.8 / 13.4 / 7.3 %, total 200 / 151 /
    127 % -- the 35B pattern. Fixed by adding `"FP8Experts": _qwen35_experts_flops` to
    `_SPECIAL_HOOKS` (all three worktrees); every hooked 122B number in the main table (full,
    floor, streaming total/crit) needs a re-measure with the fixed hook (GPU, HF FP8 load) --
    not run. Streaming rows store only a chunk COUNT, so no closed-form replay of that arm is
    possible from the rows. The interleaved table renders the 122B FLOPs cells with a dagger
    (`IL_FLOPS_PENDING`) until then; the interleaved decoder half itself is closed-form and
    unaffected.
    **Re-measured 2026-09-10 16:47-16:55 (user go).** One HF FP8 load, all seven datasets,
    n=12, `--il-rows` for the V* interleaved keys (`flops/qwen35_122b_flops_fixed.json`, log
    `logs/vllm_stream/flops122_fixed.log`, ~1 min per dataset). Two launch gotchas: the
    `kernels` cache has no `refs/v2`/`refs/v3` tags for deep-gemm / finegrained-fp8, so
    `HF_HUB_OFFLINE=1` fails the version lookup -- run ONLINE (weights still come from the cache)
    with `TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1` (the Triton fallback, as every 122B run).
    Full pass old -> new: V* 66.56 -> 90.61 TF (x1.36), RWQA 22.60 -> 33.07 (x1.46), RefCOCO
    4.54 -> 7.15 (x1.57), TextVQA 11.05 -> 16.70 (x1.51), VisDrone Count 17.58 -> 25.82 (x1.47),
    Det 15.49 -> 23.01 (x1.49), ChartQA 6.04 -> 10.10 (x1.67, it also gains the attention term).
    Gate F on the ceiling: full - tower (35B `QWEN35_MEASURED` V, same tower) vs the closed
    form = 0.99953 / 0.99948 / 0.99947 / 0.99947 on RefCOCO / ChartQA / TextVQA / RWQA -- a
    constant -0.05 %, PASS (the streaming arm's own decoder split reads 0.3-1.3 % below the
    closed form, growing with N; hypothesis: its chunked prefill is charged at the chunks'
    actual Sq x Sk, not n^2 -- real work of that arm, not a hook gap). Folded into
    `inprocess_flops.json["qwen35_122b"]` (pre-fix entry kept as `qwen35_122b_prefix`; new
    rows carry `_split` = per-arm [vision_total, vision_crit, llm_total, llm_crit]) and
    `qwen35_122b_flops_il.json` (now with the 122B run's own vision half; the decoder-only
    fold kept as `*_il_decoderonly.json`); `IL_FLOPS_PENDING` cleared, dagger gone (table v7).
    122B interleaved shares now match the 35B pattern: total 193.8 / 146.8 / 123.2 %, crit
    25.0 / 13.0 / 7.0 % at k = 1 / 0.5 / 0.25 (35B: 195 / 147 / 123 and 24.9 / 12.8 / 6.7);
    streaming 133.0 / 115.6 / 107.0 %, crit 25.7 / 21.3 / 19.2 %. NOTE: the main table's 122B
    compute cells change with it (every dataset); `eval_table_*.tex` was NOT regenerated --
    the three trees currently render different 35B k<1 accuracy cells (chain 2b rows differ per
    tree), to be reconciled before the next main-table send.

11. **Depth staging (2026-09-10, user go: "했다 치고 정확도가 어떻게 나오는지 시뮬레이션").**
    The claim in the plan that the final state of the MVP equals the depth-staged schedule's
    was WRONG for k<1 (right at k=1 only). In the HF/OV2 `interleaved_forward` the approximate
    pass is split by depth: round r corrects group r over layers `[0, b_r)`, then the frontier
    walk carries EVERY row through `[b_r, b_{r+1})` with the corrected rows' K/V and side-buffer
    rows visible, so a row that is never corrected (keep<1) still gets its deep layers recomputed
    with the corrected context of earlier bands. The MVP never recomputes an uncorrected row, so
    its deep-layer K/V for those rows are stale w.r.t. every correction. Same critical path (the
    last round is full depth in both), lower total (`1 + b̄·k·f_img + f_text`, b̄ = 0.625 at g=4),
    different k<1 accuracy -- direction to be measured, hypothesis: staged ≥ MVP.
    *Implementation* (`correct.py`): `appcorr_rows_step(req, positions, layers=(a, b), ...)`
    generalises the correct step to a layer range -- the full range goes through the compiled
    `self.model(...)` exactly as before (bit-for-bit the MVP path), a partial range through an
    eager Python loop over `Qwen3_5Model.layers[a:b]` under the same forward context / metadata
    (the pseudo-sequence trick is per layer, so it needs nothing new). The side buffer gains
    frontier buffers `fr_h/fr_r [N, D]` (the residual stream of the image rows at the walk
    frontier) and `_CorrectCtx.commit_end`: a walk's re-scan window is the whole prompt `(0,
    N-1)` but the DeltaNet checkpoint is committed only up to the last corrected window's end,
    the rest scanned uncommitted for outputs -- so the next round's correction re-scans from a
    checkpoint that already carries every correction. `appcorr_staged_correct(stage=(r, g))`:
    catch-up walk `[frontier, b_r)` (from the stored prompt embeds at frontier 0, else from the
    frontier buffers; also closes the gap a band with no selected rows leaves), correction of
    `P_r` over `[0, b_r)` with the checkpoint committed at the window end, walk `[b_r, b_{r+1})`
    over the image rows `[lo, hi)`, store the frontier. Text rows are never walked (post-image
    text is corrected at full depth in the last round and no image row attends to it; pre-image
    text does not depend on the image) -- same final state as walking everything, less work.
    `stage_bounds(L, g) = round(L(r+1)/g)`: 35B 10/20/30/40, 122B 12/24/36/48, 4B 8/16/24/32;
    layers `< b` hold `b//4` softmax layers. Plumbing: `open(image_end=)`, `correct(stage=)`
    on client/server/bridge/sink; axis `llm_schedule="interleaved_staged"` (rounds = the
    non-empty bands, records `("correct", s, e, |P|, r, g)`); drivers accept the choice (arm
    tag `interleaved_staged_g4[_k]`, probe key `qwen35_35b_il_staged`).
    *What the served run is and is not.* The engine keeps the stock full-depth approximate
    prefill (block allocation, hold-back, prompt-embeds bookkeeping all unchanged) and re-walks
    the frontier on top of it, so its final state is the staged schedule's while its GPU work
    is roughly the MVP's plus the walks -- the "simulation form". The reported Comp. is the
    closed form of the ideal schedule (`flops_analytic.interleaved_cost` with depth records:
    approx pass + walks = exactly one `prefill_flops(N-1)`, corrections priced at their depth
    via `Qwen35Decoder.layer_split`); Crit. Comp. is unchanged by construction; Crit. Lat. is
    real (the last round does no walk -- the walk after round g-2 runs inside that round's
    `correct` call, before the last band's message, so it only shows up as queueing if it
    outruns the 150 ms band spacing; the probe's `stage` timings in the server reply say).
    *Gates* (4B, in-process, `vllm_correct_gate.py --gate g6 / g6k`): g6 (k=1, g=4, staged vs
    one-shot) 4/4 first-token argmax, 3/4 sequences exact, |dlogprob| first ≤ 1.4e-4 -- the same
    band as the unstaged g2 on the same images (3.7e-4), KV rel-L2 2.4e-2 / ssm 3.5e-2 vs g2's
    1.4e-2 / 2.5e-2; g6k (every other row corrected, band 1 sends nothing -> the skipped-round
    catch-up walk runs): both forms 4/4 argmax vs one-shot, staged-vs-unstaged KV rel-L2
    0.25-0.28 / ssm 0.16-0.23 (the two schedules DO differ at k<1, as derived), staged ssm
    closer to one-shot on 4/4 images and first-token |dlogprob| closer on 3/4 (n=4: consistent
    with the hypothesis, not evidence). The first walk of a process pays ~0.5 s of Triton
    compile; afterwards a 300-row 8-layer walk is ~11 ms on 4B. 35B g6/g6k + the served V* /
    RWQA arms at k=1/.5/.25 + the d150 probe: `$S/il35_staged_chain.sh` (done 13:51).

    **35B results (2026-09-10).** Gates: g6 8/8 argmax, seq exact 6/8, first-token |dlogprob|
    3.9e-5, KV rel-L2 0.172 / ssm 0.071 vs one-shot (g2 on the same images: 4.8e-5 / 0.177 /
    0.072 -> staged == unstaged at k=1 within the band); g6k 4/4 argmax both forms,
    staged-vs-unstaged KV 0.43-0.47 / ssm 0.13-0.15, staged closer to one-shot on 4/4 (KV 0.612
    vs 0.641 max) -- hypothesis-level, n=4. Served accuracy (c4, paired bootstrap 4000x, vs the
    unstaged interleaved arm / vs streaming): V* n=191 -- k1 83.77: -0.52 [-2.62,+1.57] (+2/-3)
    / 0.00 (+2/-2); k.5 83.25: -1.57 [-3.66,+0.52] (+1/-4) / -1.05 (+2/-4); k.25 80.63: -1.05
    [-3.14,+1.05] (+1/-3) / -1.05. RWQA n=765 -- k1 76.73: +0.26 [-0.65,+1.18] (+7/-5) / -0.26
    (+5/-7); k.5 76.47: -0.13 (+8/-9) / +0.13 (+11/-10); k.25 76.47: 0.00 (+7/-7) / -0.65
    [-1.83,+0.52] (+9/-14). Verdict: no measurable staged-vs-unstaged accuracy difference (every
    CI covers 0; V* leans negative inside the c4 nondeterminism band of ~1 pp) -- the "staged >=
    unstaged" hypothesis from the g6k state distances is NOT supported at the accuracy level. The
    paper convention (user, 2026-09-10): staged Comp. by the closed form (V* 93.81/72.51/61.86 TF
    = 188.8/146.0/124.5 % at k=1/.5/.25 vs unstaged 202.0/152.6/127.9 %; RWQA 177.7/138.5/118.9 %
    vs 193.0/146.3/122.9 %), accuracy from the measured staged arm (table column) -- both are in
    `interleaved_table_20260910.tex`. Crit. Lat. on the staged schedule (d150, medians of 36):
    V* 92/80/65 ms, RWQA 65/58/56 ms vs unstaged 99/85/77 and 72/70/68 -- 7-12 ms lower in all six
    cells although the staged last round does MORE work (final-depth walk of the previous band +
    the correction); hypothesis: server warmth / run order (the staged probe ran after six
    accuracy arms on the same server, the unstaged one right after start-up). The table keeps the
    unstaged Crit. Lat. for both; the CUDA-graph re-probe (item 12) runs both back to back.
    Closed-form fold: `flops_report_qwen35.py --il-only` -> `interleaved_staged` keys in
    `qwen35_flops_il{,_rwqa}.json`; paired script `$S/paired_staged.py`.

12. **CUDA-graph correct step (2026-09-10, user go "더 할 수 있는 최적화는 해봐라").** The eager
    full-depth correct step cost ~45-55 ms per round on 35B regardless of |P| (launch-bound), so
    `_correct_sub` now dispatches the full-depth step through vLLM's own PIECEWISE graphs:
    `cudagraph_dispatcher.dispatch(num_tokens=|P|, uniform_decode=False, invalid_modes={FULL})`
    returns the padded capture size; the inputs are copied into the runner's persistent
    `inputs_embeds.gpu` / `mrope_positions.gpu` buffers (the ones every `execute_model` step
    refreshes), padded rows zeroed, and `set_forward_context(num_tokens=n_pad,
    cudagraph_runtime_mode=PIECEWISE, batch_descriptor=...)` replays the captured graphs
    (`CUDAGraphWrapper` captures lazily on first use of a size). Attention, the KV write and
    the GDN core are splitting ops and stay eager, sliced by `num_actual_tokens` / the
    slot_mapping length / our `_forward_core` window, so the zero-padded rows never touch
    the request's state. Partial-depth walks (frontier / catch-up) stay eager. Flag
    `APPCORR_CORRECT_CUDAGRAPH` (module `CUDAGRAPH`, default "1" since the gate passed).
    Gate g7 (`vllm_correct_gate.py --gate g7`: arms one/g2/g2b/g2pe/g2cg, 4B n=4 and 35B n=8):
    4B (dense) graph == eager bitwise 4/4. 35B (MoE) graph vs eager: first-token argmax 8/8,
    |dlogprob| first <= 2.3e-5, but KV rel-L2 up to 0.11 / ssm 0.046 -- the diagnostic arms
    attribute all of it to the padded batch size, not the replay: `g2b` (eager repeated) ==
    `g2` bitwise 8/8 (the eager step is deterministic), `g2pe` (padded to the capture size,
    eager) vs `g2` = exactly the graph-vs-eager numbers, and `g2cg` (graph) == `g2pe` bitwise
    8/8. Padded-eager vs one-shot sits inside the eager band (KV 0.193 vs 0.177, ssm 0.072
    both, |dlogprob| first 3.1e-5 vs 4.8e-5, seq exact 7/8 vs 6/8). Hypothesis for the
    padding effect: the MoE grouped GEMM's tile schedule changes with the token count,
    bf16 accumulation order shifts, and a few top-8 routings flip -- the same class of
    difference concurrency already introduces between served runs (batch composition).
    Verdict PASS: the graph path adds nothing beyond a batch-size change. Step time 35B:
    45.9 -> 24.5 ms mean per full-depth step (per round [45,42,42,43] -> [25,24,24,25]);
    4B 20.2 -> 16.0 ms. Latency re-probes with graphs (35B il + ils, 122B at mns 1024):
    `$S/cg_chain2.sh` / `cg_chain3.sh`, eager values kept under `*_eager` keys in
    `inprocess_latency.json`. Results (d150, medians of 36, eager -> graph): 35B unstaged V*
    99/85/77 -> 80/64/52 ms, RWQA 72/70/68 -> 50/44/40 ms (streaming 63/52/41, 35/30/30);
    last-round step 53-55 -> 27-37 ms. 35B staged V* 92/80/65 -> 94/63/50, RWQA 65/58/56 ->
    50/44/39: equal to unstaged at k<1, +14 ms at V* k=1 (the final round also walks the
    previous band's ~835 rows through the last stage, eager -- 9 ms of queueing shows up).
    122B: `--max-num-seqs 1024` is refused by vLLM (1024 > 604 Mamba cache blocks at gpu-mem
    0.85 / mml 8192, graph capture needs one block per decode seq), so the re-probe ran at 512:
    V* 742/207/87 -> 207/86/67 ms (streaming 86/75/65), RWQA 142/80/74 -> 67/59/54; a 512-row
    sub-batch is ~48 ms, V* k=1 = 2 sub-batches + 22 ms queueing. Table v4
    (`interleaved_table_20260910.tex`) carries the graph numbers; caption updated.
13. **Host-sync hoist of the DeltaNet re-scan (2026-09-10, user go "최적화 더 할 거 있냐? 있으면
    계속 해봐라").** Why the correct step had a ~30 ms floor independent of |P| (RWQA k=.25
    P~60: 27 ms; V* k=1 P=835: 35 ms) while streaming's last chunk -> first token is 26 ms:
    `torch.profiler` on the final round of the profile harness (`vllm_correct_profile.py`,
    35B, N=2379, P=316, window 610, graphs on) -- step 29.5 ms, layer loop 39 ms of CPU span
    under the profiler for 16.4 ms of GPU kernel time; the 30 GDN re-scans alone were 21.7 ms
    of span for 2.4 ms of kernels, 0.94 ms of host time each, 7 `cudaStreamSynchronize` per
    layer (222 per step): pageable `.to(dev)` of `cu`/`chunk_indices`/`chunk_offsets`,
    `torch.tensor([0, L], device=dev)` + `torch.zeros/ones(device=dev)` for the conv call, and
    inside stock `causal_conv1d_fn`'s `metadata=None` branch `query_start_loc.diff().to("cpu")`
    plus two `torch.full((1024,))` allocations. The layer loop was CPU-bound, not GPU-bound.
    Fix (`correct.py` `_win_consts`): all of those are functions of (device, conv window
    length, scan window length) only, so they are built once per window shape and cached
    (`OrderedDict`, 4096 entries; identical across the 30 layers of a step and across requests
    with the same band sizes), the conv gets a prebuilt metadata object from the same
    `compute_causal_conv1d_metadata` the stock prefill builder uses (pinned `batch_ptr` /
    `token_chunk_offset_ptr`, `nums_dict`), and `_rescan_impl` / `_conv_window` issue no host
    sync at all. `client.correct`'s CPU bookkeeping (`prompt_embeds[pos] = rows` on both
    request copies) now goes through numpy (`_cpu_scatter_rows`): torch's CPU `index_put_` on
    316 rows costs 30 ms per call in the 72-thread appcorr env (0.03 ms via numpy; the server
    process is not that slow -- wall vs step in the profile showed <1 ms -- but it is now
    thread-count independent; `t_scatter_ms` reported in the correct info).
    Re-profile (same harness): step 29.4 -> 16.8 ms at P=294, 17.7-19.4 at P=316; rescan host
    time 0.94 -> 0.29 ms/layer, syncs 222 -> 12 (all in the metadata build, before the layers);
    layer loop 17.6 ms GPU span for 16.0 ms of kernels -> GPU-bound now. Of those 16 ms, 8.7 ms
    are the two MoE grouped GEMMs (40 layers x 0.22 ms, `bmm_Bfloat16..t128x16x128`): with
    316 tokens x top-8 over 256 experts every expert is touched, so each layer streams its full
    1.75 GB of expert weights -- 0.22 ms at 8 TB/s (hypothesis from the arithmetic, consistent
    with the measured time; the same 35B weight read is why streaming's last chunk costs 26 ms).
    That is the floor of a full-depth correct step on this model (~9-10 ms + attention/GDN/glue
    ~7 ms); only weight quantisation or a shallower final stage moves it further.
    Gates: 4B g7 (graph vs eager, one/g2/g2b/g2pe/g2cg) bit-identical to the pre-hoist json
    on every metric of every image; 35B g2 first-token argmax 8/8, |dlogprob| first <= 3.1e-5
    (chunk band 3.0e-5), KV/ssm rel-L2 in the same band as before (the pre-hoist 35B json is
    the eager-era run, so the small per-image drift is the graph/padding effect of item 12, not
    the hoist; 3/8 images bitwise equal).
    Served re-probe (d150, 36 samples/4 warmup, previous keys kept as `*_prehoist`; paired
    per-sample deltas against the graph-era probes `probe_*_cg_d150` / `probe_qwen35_122b_il_mns512_d150`,
    bootstrap 95% CI of the median delta): the last correct step drops by a constant
    -11.0 .. -12.7 ms on every 35B cell (il and staged, V* and RWQA, k=1/.5/.25; 32/32 samples
    faster each) and -12.1 .. -14.4 ms on the 122B cells (k=1 V* -25.8: the largest P), i.e. the
    hoist removes a fixed per-step host cost, not a per-token one. Crit. Lat. (medians, ms):
    35B il V* 79.6/63.7/51.5 -> 69.1/52.6/39.5, RWQA 50.4/44.0/40.0 -> 36.9/32.5/28.5;
    35B staged V* 93.6/63.0/49.9 -> 66.6/52.0/39.7, RWQA 49.6/43.5/39.1 -> 37.8/30.8/27.4;
    122B il V* 207.4/86.0/67.2 -> 117.4/71.1/55.0, RWQA 67.2/58.9/53.7 -> 56.6/45.3/41.5
    (paired dmed -10.7 .. -13.9 ms on 35B, -11.6 .. -15.2 on 122B at k<=.5; 122B k=1 V*
    -87 ms [-91, -31], the wide CI is the old run's queue tail). Against streaming d150
    (35B V* 63/52/41, RWQA 35/30/30) the interleaved cells are now level at k<=0.5 on V* and
    within 2-3 ms on RWQA; the remaining structural item is the hold-back decode step
    (`last_chunk_to_ft_ms` 6-8 ms on 35B, 9-11 on 122B) -> item 14.

14. **Fused hold-back (2026-09-10, same go).** After the hoist the final round's tail was
    correct step (~17-21 ms) + hold-back decode step (5.7-8 ms in-process / served): the
    engine step that computes row N-1 and samples the first token, a full 40-layer forward of
    one token. The fused path folds row N-1 into the final correct step and skips that forward.
    Design (`correct.py` `FUSE_HOLDBACK`, `client.py` `defer_final`, both default on): the
    client no longer runs the final correct synchronously -- `correct(final=True)` arms the
    request (`runner.appcorr_arm_final`: positions, embeds, window, stage, replay), scatters
    the CPU prompt rows, releases the hold-back and calls one engine step. The runner's
    `_prepare_inputs` wrapper drains armed requests AFTER `_update_states` (so N-1's KV block
    is allocated) and BEFORE the stock `_prepare_inputs`: if every token scheduled in this
    step is an armed request's row N-1 (single request, or several finalising together) the
    correct step runs with `fuse=True` -- row N-1 is appended to P (embedding from
    `prompt_embeds[N-1]`), the DeltaNet window end moves to N with `commit_end=-1` so the one
    rescan leaves state_{N-1} and the conv state including N-1 in the request's block, the
    softmax layers write N-1's K/V through the same pseudo-sequence path, and `_run_layers`
    (full depth now returns the normed hidden states) stashes `out[P-1]` in
    `sb.hold_hidden`. A class-level `GPUModelRunner._model_forward` stub then returns a zero
    `[n, D]` tensor with the stashed rows at the scheduled token offsets, the stock
    `compute_logits`/sampler run on it, and the stub is cleared in the `_execute_model`
    finally. Any other batch composition (another request's tokens in the same step, a
    replay arm) runs the deferred correct step plainly and lets the stock decode compute N-1,
    so `FUSE_HOLDBACK=0` / a mixed batch == the synchronous path. The info dict carries
    `fused`, `deferred`, `t_armed_ms`.
    Gates (`vllm_correct_gate.py --gate g8`: one/g2/g2d/g2f/g6f; g2 = synchronous
    reference): 4B and 35B -- g2d (deferred, unfused) bitwise equal to g2 on every metric of
    every image (the deferral itself is transparent); g2f vs g2 KV bitwise, ssm rel-L2
    5e-3..1.8e-2 (state_{N-1} from the chunked scan vs the stock recurrent decode kernel; the
    g2-vs-one band is 2e-2..7e-2), first-token |dlogprob| <= 1.9e-5 on 35B (chunk band 3.1e-5),
    argmax 8/8, 35B seq exact 8/8 (g2: 7/8); g6f (staged + fused) in the same band vs one.
    Fused final step wall == unfused (12-13 ms on 35B gate prompts, 8-9 on 4B): the extra row
    is free. Profile (35B, N=2379, P=316->317, graphs on): final round wall + hold-back
    24.6 / 28.1 ms (hoist-only, it1/it2) -> 22.6 / 26.5 ms fused; the saved decode forward
    was only ~2-3 ms in-process, the rest of the old 5.7-7 ms hold-back was the engine step's
    own scheduler/prepare/sampler/bookkeeping (~3.6 ms = wall - step of the fused step), which
    is still paid once. Served, the fuse also removes the server-loop hop between the correct
    reply and the next step (`last_chunk_to_ft_ms` 6.5-8.1 ms on 35B, 8.8-10.6 on 122B), so
    the served gain should exceed the in-process one.
    Served re-probe (`$S/fuse_reprobe.sh`, d150, hoist-only keys kept as `*_hoistonly`,
    probe dirs `probe_*_fuse_d150`; paired per sample vs `probe_*_hoist_d150`, bootstrap 95%
    CI): `last_chunk_to_ft_ms` 6.0-8.7 -> 1.2-2.4 ms on 35B (-4.7..-6.2, 32/32) and 8.8-10.7
    -> 1.3-2.2 ms on 122B (-7.5..-8.5, 32/32) -- what is left is the server's reply hop;
    `last_correct_step_ms` grows by +2.0..+4.2 ms on 35B and +2.7..+5.4 on 122B (the engine
    step that now wraps the correct step: scheduler, stock `_prepare_inputs`, stub forward,
    logits, sampler, bookkeeping; larger on V* than RWQA and on 122B than 35B -- an
    N-dependent cost, hypothesis, not profiled). Net Crit. Lat. (medians, ms): 35B il V*
    69.1/52.6/39.5 -> 65.8/50.7/38.1, RWQA 36.9/32.5/28.5 -> 35.6/27.9/25.4 (paired dmed
    -1.5..-4.5, 24-29/32 faster); 35B staged V* 66.6/52.0/39.7 -> 64.8/49.3/37.5, RWQA
    37.8/30.8/27.4 -> 33.9/28.5/24.6 (-1.9..-3.6, 28-32/32); 122B il V* 117.4/71.1/55.0 ->
    113.8/67.5/51.8, RWQA 56.6/45.3/41.5 -> 52.0/41.2/35.8 (-3.3..-5.8, 27-32/32). Predictions
    agree with the hoist-only run on 31-35/36 rows per cell (the rest are answer-format
    tails, one first-token flip at 35B V* k=.5; ok-counts unchanged on 122B, -1/36 on that
    cell) -- the chunked-vs-recurrent state_{N-1} difference sits inside the chunk band.
    Where the interleaved tail stands now (35B V* k=.5): last-band vision correction 22.6 +
    transport 0.5 + fused correct step 25.1 (~17 ms forward with a ~9 ms MoE weight-stream
    floor and ~7 ms of kernel floor, plus ~4 ms engine step) + reply 2.3 = 50.7 ms vs
    streaming 52; RWQA 27.9 vs 30. The LLM-side items left are vLLM's per-step overhead
    (~2-5 ms) and the reply hop (~1-2 ms); the forward itself is at the model's floor.

## §7.12 Unified (vision + decoder) depth-staged schedule

Date: 2026-09-10. Status: IMPLEMENTED, CPU-gated only -- no GPU run of any kind yet (GPU0 was
occupied by a 12-hour campaign). Branch `develop/vllm-unified-axis`, driver flag
`--llm-schedule unified_staged`, row files `interleaved_unified_g{g}[_k{keep}]`, fold keys
`total_g4*_ilu` / `crit_g4*_ilu` / `_ilu_g4*`.

**What it is.** §7.11 stages the DECODER by depth and leaves the vision tower where it always
was: approximated at full depth at arrival 0, each band corrected at full depth as it lands. The
unified schedule puts the tower inside the staging. The 27 tower layers and the L decoder layers
become ONE axis of `27 + L` stages, cut into `groups` rounds of equal COST; round r corrects its
band over the stages walked so far and then pushes the approximate frontier to the next bound.
This is not a new design -- it is `appcorr/models/gemma3/unified.py`'s `interleaved_forward`,
whose `stage_costs()` / `layer_bounds()` / walk this ports onto the Qwen3.5 pair. What is new
here is that the LLM half lives in another process, so "the frontier crossed the projector" has
to become a wire event.

### Cost model

Counting stages would be wrong -- a tower layer runs `4 x n_image_tokens` rows at width 1152 and
a decoder layer N tokens at width 2048/3072 with three layers in four recurrent -- so the split
is by FLOPs, computed per REQUEST from `(n_rows, N)`:

- tower layer: `2 n h (3h) + 2*2 H n^2 (h/H) + 2 n h^2 + 2*2 n h I` with `h=1152, H=16, I=4304`
  (fused qkv, full attention over the whole image, out proj, the 2-layer UNGATED MLP -- Qwen3.5's
  vision MLP is `linear_fc2(act(linear_fc1(x)))`, no gate). Identical to
  `flops_analytic.vision_layer_flops` / `Qwen35Vision.layer_flops`.
- decoder layer, per layer (the mix matters -- one layer in four is softmax): MoE block
  (`2 top_k 3 I H + 2 H E` routed+router, shared expert, gate) on every layer, plus either the
  GDN projections+conv+scan, or `q/k/v/o` and the quadratic term at `Sq = Sk = N`. Summed over
  the layers this IS `Qwen35Decoder.prefill_flops(N)` -- asserted to 1e-12 by the CPU gate, which
  is what makes "the axis and the cost table price the same stage" a check and not a claim.

`QwenVLStreamingAxis.unified_bounds(groups, n_rows, N)`: equal cumulative cost, `bounds[r]` is
round r's correction depth, last bound is always the whole axis. Degenerates to
`stage_bounds(L, g)` when the tower has zero cost and the decoder layers are uniform (gated).

### Bounds, g=4 (27 tower layers + 40 / 48 decoder layers)

`vision` / `decoder` are the two halves of one full pass; `tower depth` and `decoder depth` are
`min(b_r, 27)` and `max(0, b_r - 27)`.

| prompt | model | vision | decoder | bounds | tower depth | decoder depth |
|---|---|---:|---:|---|---|---|
| V*-like N=1000, 835 img tok | 35B | 4.13 TF (44.8%) | 5.10 TF | 16, 31, 49, 67 | 16, 27, 27, 27 | 0, 4, 22, 40 |
| V*-like N=1000, 835 img tok | 122B | 4.13 TF (19.5%) | 17.03 TF | 31, 46, 61, 75 | 27, 27, 27, 27 | 4, 19, 34, 48 |
| TextVQA-like N=500, 445 img tok | 35B | 1.86 TF (42.5%) | 2.51 TF | 16, 33, 50, 67 | 16, 27, 27, 27 | 0, 6, 23, 40 |
| TextVQA-like N=500, 445 img tok | 122B | 1.86 TF (18.1%) | 8.42 TF | 32, 46, 61, 75 | 27, 27, 27, 27 | 5, 19, 34, 48 |
| V* median as measured N=3341, 3290 img tok | 35B | 32.37 TF (63.8%) | 18.33 TF | 11, 22, 40, 67 | 11, 22, 27, 27 | 0, 0, 13, 40 |
| V* median as measured N=3341, 3290 img tok | 122B | 32.37 TF (35.0%) | 59.99 TF | 20, 39, 57, 75 | 20, 27, 27, 27 | 0, 12, 30, 48 |

Read off it: **the LLM `open` is delayed**, and by how much depends on the shape, not on a
constant. On the campaign's real V* prompts (the last two rows -- N=3341 is the measured median;
"835 image tokens" is the per-band `|P|` at g=4, i.e. 3340/4, so the first two rows are a smaller
image than the campaign ran) the 35B spends its first TWO rounds entirely inside the tower and
opens the prompt only at the end of round 1, with two LLM rounds left at depths 13 and 40. The
122B, whose decoder is 3.3x the 35B's, crosses one round earlier. On the smaller V*-like prompt
the 122B's very first bound already crosses, which is the `groups=1` case generalised.

### The walk (`qwen_vl_axis.streaming_forward`, `llm_schedule="unified_staged"`)

    arrival 0   tower approx over [0, min(b_0, 27))  on the BASE image
                if b_0 > 27: merge every group, push the whole prompt (`open`), LLM depth b_0-27
    round r     band r arrives
                vision-correct band r's selected groups over [0, min(b_r, 27)) on the MIXED
                  layer-0 stream (full-res rows for everything arrived, base rows for the rest)
                if the LLM is open: merge this band's corrected groups, `correct` its rows with
                  an explicit decoder bound b_r - 27
                advance the frontier: tower approx over [min(b_r,27), min(b_{r+1},27)) on the
                  stream the correction just produced
                if b_{r+1} > 27 and the LLM is not open yet: merge every group from the stream
                  AS IT STANDS (bands 0..r corrected and carried, the rest approximate) and push
                  the whole prompt

Three things this makes load-bearing that §7.11 did not:

1. **`correct_forward`, not `correct_rows`.** The fast path skips the rule-3 write-back and the
   `[T, D]` reconstruction because the streaming axis never reads a non-corrected row again. Here
   the approximate walk that follows a correction READS the stream, so both are needed: without
   rule 3 an earlier band's rows would be rebuilt from approximate increments on top of their
   full-resolution layer-0 value -- the self-inconsistent combination the CLIP memo measured
   below the floor. Cost: the per-round vision correct materialises the whole stream.
   Unmeasured; HYPOTHESIS: this shows up as a few ms per round on 35B V*, not as a factor.
2. **The opening push is not the base merge.** It carries the bands corrected so far. So the
   crossing round's `x_base_out` -- the merge reference for rows a later band does not select --
   is pinned to the stream at the crossing, not to the pure base output: an uncorrected row's LLM
   input must be the value the opening push actually carried.
3. **The selection score is PROGRESSIVE.** The received-attention term is the mean over the tower
   layers walked so far (`prefix_attn_layermean`), because the full-tower mean does not exist
   when band 0 must be chosen. Two consequences, both stated rather than discovered later:
   - the deferred pscore is meaningless here (it hides the column sum behind the first push, and
     there is no push yet), so `unified_staged` records `pscore = "progressive"`;
   - at keep<1 this arm selects a DIFFERENT SET from the streaming / interleaved arms. Contract
     rule 5 therefore says the k<1 cells are not a clean schedule-only comparison against them;
     k=1 is. The alternative -- a full-depth scoring pass before round 0 -- runs the tower twice,
     which is the defect `interleaved_forward_progressive` exists to avoid.

### Wire change

`stage` gains a third, optional element: `[r, g]` (unchanged) or `[r, g, [b_0..b_{g-1}]]`.
`wire.stage_to_header` / `stage_from_header` are the one place it is (de)serialised;
`correct.stage_spec(stage, L)` is the one place the engine turns it into bounds -- `(r, g)` still
derives `stage_bounds(L, g)` there, so every existing run is bit-identical. `g` in the explicit
form is the number of LLM ROUNDS, not the schedule's `groups`: the rounds before the crossing
never reach the decoder. Everything below `stage_spec` in `correct.py` is unchanged -- the
frontier walk, the checkpoint chain, the fused hold-back and the CUDA-graph path all take the
bounds as given.

### FLOPs

`Qwen35Vision.unified_cost(chunks)` is the new half. The vision half is NO LONGER common across
arms -- the approximate pass is still exactly one pass over every row and layer, but each band's
CORRECTION runs only over the layers walked so far -- so the accuracy rows carry
`("vapprox", a, b, n_rows)` and `("vcorrect", 0, depth, rows)` records alongside the decoder's,
and the decoder's `correct` records carry an explicit depth `("correct", s, e, |P|, r, g, b_r)`.
`flops_report_qwen35.interleaved_from_rows(..., unified=True)` returns the closed-form ratio of
the unified vision half to the full-depth one, which `add_interleaved` applies to the hooked
vision column (the same hooked-basis convention the decoder half already uses). Critical comes
out at ratio 1 by construction: the last round always corrects at full tower depth.

Closed form on the measured V* 35B shape (N=3341, 3290 image tokens, lo=5; full pass = 32.37 TF
tower + 18.33 TF prefill = 50.69 TF), the SAME formula applied to both schedules so the delta is
the schedule and nothing else:

| k | ils total | ils crit | ilu total | ilu crit | delta total | delta crit |
|---:|---:|---:|---:|---:|---:|---:|
| 1.00 | 185.7 % | 25.2 % | 163.4 % | 25.2 % | -22.2 pp | 0.0 pp |
| 0.50 | 143.2 % | 12.9 % | 132.0 % | 12.9 % | -11.2 pp | 0.0 pp |
| 0.25 | 122.0 % |  6.7 % | 116.3 % |  6.7 % |  -5.7 pp | 0.0 pp |

122B on the same shape (full 92.36 TF): total 174.8 / 138.0 / 119.6 -> 163.0 / 132.0 / 116.6 %,
crit 25.5 / 13.2 / 7.1 % unchanged.

Two halves move. The vision half falls because bands corrected before the crossing pay 11 or 22
layers instead of 27 (64.72 -> 58.43 TF at k=1 on 35B V*). The decoder half falls MORE, and for a
reason worth naming: a band that lands before the LLM opens gets no decoder correction at all --
its corrected rows are simply in the prompt when the prompt is first prefilled. At g=4 on V*/35B
that removes two of the four correct rounds (29.41 -> 24.43 TF at k=1).

**Calibration.** The campaign's published `ils` V* cells are total 195/147/123 %, crit
24.9/12.8/6.7 % -- means over 191 real rows on the hooked basis. This table's `ils` column is the
closed form on ONE median shape, so it lands 5-9 pp low on total and within 0.3 pp on critical.
The `ilu - ils` delta is the number to read, not the absolute.

**Crit. Lat. is a separate question.** Crit. Comp. is unchanged BY CONSTRUCTION (the last round is
full depth on both halves in both schedules), and that is arithmetic, not a measurement. What the
served form actually does in the last round's shadow -- the engine keeps its stock full-depth
approximate prefill and re-walks on top of it, exactly as §7.11's simulation form does, and the
frontier walk after the second-to-last round now covers a deeper layer band -- is unmeasured.

### Gates

CPU, runnable with no GPU (`analysis/experiments/test_unified_axis.py`, PASS 2026-09-10):

| id | what | result |
|---|---|---|
| B1 | bounds strictly increasing, in `[1, n_stages]`, `groups` of them, last == whole axis; 35B/122B/4B x 2 prompt shapes x g in {1,2,4,8} | PASS |
| B2 | the axis's decoder stage costs sum to `Qwen35Decoder.prefill_flops(N)` (rel < 1e-12) and its vision stage IS `Qwen35Vision.layer_flops` | PASS |
| B3 | a zero-cost tower + uniform decoder degenerates to `stage_bounds(L, g)` | PASS |
| W1 | `stage` round-trip through a real `Frame`: `[r,g]` byte-for-byte unchanged, `[r,g,[b..]]` survives | PASS |
| W2 | `correct.stage_spec` derives the old bounds from `(r,g)`, takes explicit ones, rejects non-monotone / short / wrong-last / zero | PASS |
| A1 | the walk on a tiny real Qwen3.5 (g in {2,4}, k in {1, 0.5}): message structure, `vapprox` ranges tile `[0, 27)` once in order, `vcorrect` depth == `min(b_r, 27)`, `correct` depth == `b_r - 27` with the last full, round index/count on the wire, coverage, keep budget, both closed-form replays | PASS |
| A2 | `groups=1` identity: bitwise equal pushed embeddings and identical selection vs `interleaved` and `interleaved_staged`, at k=1 and (eager pscore) k=0.5 | PASS |
| R1 | `streaming` / `interleaved` / `interleaved_staged` bit-identical to the campaign worktree's code (2 grids x g in {1,4} x k in {1, 0.5}) | PASS |

GPU, written and NOT run (`analysis/experiments/vllm_unified_gate.py`):

- `--mode driver` (real tower, `--load vision`, no engine): U1 = A2 on the real tower and real
  images; U2 = A1's structure checks; U3 = the G5 ANALOGUE. Note what U3 is not: at `g > 1` the
  unified arm is **not** bitwise equal to `interleaved_staged`, and gating it as if it were would
  be wrong. Band r is corrected over the tower prefix walked so far and then carried through the
  remaining layers, whose K/V the approximate walk recomputes from the partly corrected stream --
  a different computation, not a re-addressing of the same one. So U3 reports each arm's relative
  L2 against the ceiling image rows (recomputed inside the gate, never read out of an arm) with
  the floor as the scale. HYPOTHESIS, to be judged from those numbers: unified <=
  interleaved_staged <= streaming at equal keep, because a row that is never corrected still gets
  its deep layers recomputed against corrected context.
- `--mode served`: ceiling / floor / `interleaved_staged` / `unified_staged` through one engine on
  a few images, per-row prediction agreement with the ceiling. Judge against §7.11's g6 band
  (35B: 8/8 argmax, first-token |dlogprob| 3.9e-5), not an absolute constant.

Commands (35B; the 4B `Qwen/Qwen3.5-4B` is cached for a faster first pass):

```
# server (one terminal)
VLLM_ENABLE_V1_MULTIPROCESSING=0 CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 \
HF_HUB_CACHE=/NHNHOME/huggingface/hub PYTHONPATH=/NHNHOME/share/cjpark/AppCorr-il-unified \
VLLM_GDN_DECODE_KERNEL=triton \
/NHNHOME/storage/users/cjpark/shk/conda_envs/appcorr-vllm/bin/python \
  -m appcorr.vllm_stream.server --model Qwen/Qwen3.5-35B-A3B --port 5591 \
  --gpu-mem 0.60 --max-model-len 16384 --max-num-seqs 1024 --interleaved

# driver-side gate (no engine; can run on its own GPU slot before the server exists)
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 HF_HUB_CACHE=/NHNHOME/huggingface/hub \
PYTHONPATH=/NHNHOME/share/cjpark/AppCorr-il-unified \
/home/nxclab/anaconda3/envs/appcorr/bin/python \
  analysis/experiments/vllm_unified_gate.py --mode driver --model Qwen/Qwen3.5-35B-A3B \
  --dataset vstar --degrade-filter pyr --groups 4 --keeps 1.0 0.5 --samples 8 \
  --out analysis/results/vllm_stream/unified_gate_35b.json

# served gate (against the server above)
HF_HUB_OFFLINE=1 HF_HUB_CACHE=/NHNHOME/huggingface/hub \
PYTHONPATH=/NHNHOME/share/cjpark/AppCorr-il-unified \
/home/nxclab/anaconda3/envs/appcorr/bin/python \
  analysis/experiments/vllm_unified_gate.py --mode served --model Qwen/Qwen3.5-35B-A3B \
  --port 5591 --dataset vstar --degrade-filter pyr --groups 4 --keeps 1.0 --samples 8 \
  --out analysis/results/vllm_stream/unified_gate_served_35b.json

# one accuracy arm (the campaign form)
HF_HUB_OFFLINE=1 HF_HUB_CACHE=/NHNHOME/huggingface/hub \
PYTHONPATH=/NHNHOME/share/cjpark/AppCorr-il-unified \
/home/nxclab/anaconda3/envs/appcorr/bin/python analysis/experiments/qwen_vllm_accuracy.py \
  --family qwen35 --model Qwen/Qwen3.5-35B-A3B --port 5591 --dataset vstar \
  --degrade-filter pyr --arms streaming --llm-schedule unified_staged --groups 4 --keep 1.0 \
  --concurrency 4 --load vision --out analysis/results/qwen_vllm_accuracy_il_pyr

# latency probe (spaced arrival, the 2026-09-09 rule)
... analysis/experiments/latency_probe.py --family qwen35 --model Qwen/Qwen3.5-35B-A3B \
  --port 5591 --key qwen35_35b_il_unified --llm-schedule unified_staged --push-delay-ms 150 \
  --samples 40 --warmup 4 --keeps 1.0 0.50 0.25 --datasets vstar:pyr realworldqa:box \
  --skip-ceiling

# FLOPs fold (CPU, after the accuracy rows exist)
... analysis/experiments/flops_report_qwen35.py --il-only --il-rows <rows dir> \
  --datasets vstar realworldqa --keeps 1.0 0.5 0.25 --out-json <existing qwen35_flops_il.json>
```

### Open questions

1. **Accuracy.** RESOLVED (§7.13): tie with `ils` on 6 cells. Original text: The staging saves total compute at an unchanged critical
   path, so the interesting question is whether it costs accuracy. Two effects pull opposite
   ways and neither is measured: bands corrected early are corrected SHALLOWLY (worse), and every
   row's deep layers are recomputed against corrected context (better -- the same argument §7.11
   made for staged >= unstaged, which the accuracy campaign did NOT support).
2. **keep<1 is not a schedule-only comparison** against `ils` (progressive selection, above).
   Either report k=1 as the clean cell, or add an `ils` arm run with progressive selection so the
   two share a signal. Not built.
3. RESOLVED (§7.13: hybrid built, bitwise, no Crit. Lat. effect because the path is engine-bound). **`correct_forward` per round** costs the full-stream reconstruction the streaming arm avoids.
   Latency impact unmeasured. If it bites, a hybrid is available: rounds after the crossing could
   use `correct_rows` (they correct at full tower depth and nothing reads their stream again).
4. **Where the crossing lands is prompt-dependent**, so `g` no longer means the same thing across
   datasets: on 122B/TextVQA all four rounds are LLM rounds, on 35B/V* only two are. Any
   cross-dataset table must say so.
5. **The merger is uncounted** (~0.25% of the tower): the unified arm calls it once more than the
   streaming arm and once less than the interleaved arm.
6. RESOLVED (§7.13: `open_walk`). **The engine still runs a full-depth approximate prefill** at `open` (§7.11's simulation form),
   so the served GPU work is not the closed form's. Unchanged from §7.11 -- but the gap is larger
   here, because the ideal schedule's opening prefill is only `b_0 - 27` layers deep.

## §7.13 Unified schedule: what was measured and what was fixed (2026-09-11)

All on 35B bf16, served (one B200, `--gpu-mem 0.60 --max-num-seqs 1024`), V*Bench pyr n=191 /
RealWorldQA box n=765 at c=4, latency probe 36 images d=150 (`analysis/results/latency/
inprocess_latency.json` keys `qwen35_35b_il_unified{,_rows,_ow,_ow2}`; the unified keys live in the
il-unified tree's json, `qwen35_35b_il_unified` is also merged into il-engine's).

### Accuracy (open question 1): tie

Paired against `interleaved_staged` (same server, same rows, 95% bootstrap CI, up/down counts):

| | k=1 | k=0.5 | k=0.25 |
|---|---|---|---|
| V* unified / staged | 83.25 / 83.77, -0.52 [-2.1,+1.1] 1/2 | 82.72 / 83.25, -0.52 [-2.6,+1.6] 2/3 | 81.15 / 80.63, +0.52 [-1.6,+2.6] 3/2 |
| RWQA unified / staged | 77.39 / 76.73, +0.65 [-0.3,+1.7] 10/5 | 76.21 / 76.47, -0.26 [-1.6,+0.9] 11/13 | 77.52 / 76.47, +1.05 [-0.3,+2.4] 18/10 |

Two independent served runs of the six unified arms agreed per sample (0 diffs). keep<1 uses the
progressive prefix-layer-mean selection (open question 2 stands: not a schedule-only comparison).
The rows are `qwen_vllm_accuracy_il{,_pyr}/*_interleaved_unified_g4*` in il-engine; the runs before
the engine changes below are kept in `ilu_run1/` (first) and `ilu_run2/` (identical re-run).

### Compute (closed form, `_ilu` keys in `qwen35_flops_il{,_rwqa}.json`)

Total, share of the full-res pass, unified / staged / streaming: V* 161 / 183 / 162 % (k=1),
130 / 141 / 130 (k=.5), 115 / 120 / 115 (k=.25); RWQA 163 / 181 / 152, 132 / 141 / 126,
117 / 121 / 113. Crit. Comp. equals staged by construction (the last round walks the full
remaining depth of both halves). The fold needs `--keeps 1.0 0.5 0.25` (the first fold ran k=1 only).

### Crit. Lat.: engine-bound, fixed by moving the opening prefill off the path

Decomposition (V*, ms, median; crit = wait for the engine + last band's vision correct + LLM step):

| form | k=1 | k=0.5 | k=0.25 | RWQA k=1/.5/.25 |
|---|---|---|---|---|
| staged (reference) | 65 = 0.6+34+28 | 49 = 0.4+23+25 | 37.5 = 0.3+13+22 | 34 / 28.5 / 25 |
| unified, `correct_forward` after crossing | 157 = 65+62+28 | 110 = 36+48+24 | 95 = 36+36+21 | 36 / 30 / 26 |
| + `correct_rows` after crossing (hybrid, open q. 3) | 153 = 66+58+28 | 123 = 65+33+24 | 105 = 64+23+21 | 35 / 29 / 26 |
| + `open_walk` (open q. 6) | **80** = 0.6+49+28 | **56** = 0.4+30+24 | **36.5** = 0.2+13+21 | 34 / 27.5 / 25 |

- The hybrid is bitwise on every message (`vllm_unified_hybrid_gate.py`, 6 cells) and shortens the
  vision correct as expected, but the wait grew by the same amount: the last band's message waits
  for the engine's first LLM round whatever the tower does. Wait and vision correct are a max, not
  a sum -- the 2026-09-10 decomposition that called them independent was wrong.
- The engine's first LLM round (V*: correct 17 ms + catch-up walk [0,13) ~66 ms + walk [13,40)
  132 ms = 244 ms, after a stock full-depth prefill of 3341 tokens at the crossing t~300 ms) ran
  past the last band's arrival. `open_walk` (wire: `open` header `open_walk = b_0`; engine:
  `correct.appcorr_open_walk`, run from `StreamingLLM.step` / `_drain_until_prefilled` once the
  request's prefill steps have been scheduled) makes the scheduled prefill steps of that request
  no-ops (`_ST.skip_forward`: only when the whole batch is such rows and all below the hold-back
  -- the first cut also skipped the hold-back step and sampled the first token from zeros, 0/8)
  and walks rows [0, hi) -- text prefix included, nothing else computes it -- through [0, b_0)
  at open. The first LLM round then only corrects + walks [b_0, b_1). Served work now equals the
  closed form (the post-image text is computed once, in the final round). Served gate
  (`unified_gate_ow.json`, 8 V* images): open walk on 7/8 vs off 6/8 agree with the ceiling; the
  one row differs as 'B' vs '(B) white' (answer equal, format flip -- the text prefix rows are
  pseudo-sequence walks instead of the stock prefill, kernel-level difference). Accuracy re-run
  with the open walk: see the table update below.
- Residual (k=1 +15 ms, k=.5 +7): the last band's vision correct overlaps the engine's walk
  [13,40) (132 ms for 3294 rows as one-token pseudo-sequences) on the same GPU (V* k=1: round-2
  done at 573 ms, band 4 at 546). Next lever = that walk (`CONTIG_ROWS`: contiguous rows as one
  prefill sequence -- measured below when it lands).
- e2e (`t_client_done`) unified is shorter than staged by 94/212/213 ms on V* (tower approx
  overlapped with the arrival gaps) and 140/157/159 ms on RWQA.

### Contiguous rows as one prefill sequence (`CONTIG_ROWS`, engine-wide)

The walk of 3294 rows through [13,40) cost 132 ms as one-token pseudo-sequences (decode-style
attention, P queries). A step whose rows are one contiguous range -- every frontier walk, the
open walk, a keep=1 band's correction (+ text suffix) -- is now submitted as ONE prefill
sequence (`query_start_loc [0,P]`, `seq_len p1+1`, `num_computed p0`, `is_prefilling`), so the
prefill kernel runs it; keep<1 corrections stay pseudo-sequences. V* k=1 engine steps: open walk
66 -> 25 ms, walk [13,40) 132 -> 42.5 ms, final correct 24 -> 22 ms; round 2 now ends (482 ms)
before band 4 arrives (544), so the last band's vision correct no longer shares the GPU with it.

| form (V*, ms) | k=1 | k=0.5 | k=0.25 | RWQA |
|---|---|---|---|---|
| + open walk (pseudo-sequence steps) | 80 = 0.6+49+28 | 56 = 0.4+30+24 | 36.5 | 34 / 27.5 / 25 |
| + contiguous rows (`qwen35_35b_il_unified_ow2`) | **61.6** = 0.5+34+26 | **48.6** = 0.3+23+24 | **36.3** | 34 / 27.6 / 24.3 |
| staged, same engine (`_staged_v2`) | (re-probed 2026-09-11, see the table) | | | |

Served gate on the contig engine (`unified_gate_contig.json`): every prediction difference
against the pseudo-sequence engine, and between open walk on/off, is on the same two borderline
images (rows 23, 92) and is format only ('B' / '(B)' / '(B) white'; '(D)' / '(D) black') -- the
answer token is the same in every form. The flag is engine-wide, so the reference arms' served
form changed too: their Crit. Lat. was re-probed on the same engine before the table was
regenerated (`_il_v2`, `_il_staged_v2`; the pre-contig entries stay under the old keys' `_v1`).

### Verdict

Promising: no accuracy cost, total compute -22/-11/-6 pp vs staged (V*), Crit. Lat. at (or 1-3 ms
below) the staged level after the three engine changes, e2e shorter. It cannot beat staged on
Crit. Lat. by more than the served-form overheads (equal critical compute by construction); its
claim is the total-compute + e2e one at equal Crit. Lat.

### Full-table campaign (2026-09-11 07:16-11:37, `ilu_full_chain.sh`, unified arms only)

35B (c4, full splits) + 122B (c2, strided 40 / 240 rows, over-length skipped on InfoVQA: 213/240):
floor + ceiling where missing, unified k=1/.5/.25, latency probes (d150), FLOPs (35B hooked
`qwen35_flops_il_ext.json`; 122B `--il-only` folds into `_il.json` / `_fixed.json`, InfoVQA hooked
`qwen35_122b_flops_infovqa.json`). 122B served gate on V* (8 images): unified 7/8 agree with the
ceiling (floor 6/8, staged 6/8); open walk on/off differ on the two format-flip rows only.
Cells are in `interleaved_table_20260910.tex`; notes in `interleaved_table_notes.md`.

Reading: (1) accuracy preservation of the unified arm tracks the streaming arm's on every
dataset (grounding / counting fall with k on both); TextVQA / InfoVQA k<1 unified >= staged / il
by +0.3..+1.3 pp (progressive selection, confounded with the selection set). (2) Comp.: equal to
streaming on V* / RWQA (35B), above it where the decoder share is large (TextVQA / VisDrone 35B:
163 vs 150 %; 122B V*: 225 vs 145 %) because the unified decoder half is 1.3-1.5x the stock
prefill; below it on long-prompt InfoVQA 35B (160 vs 170 %). (3) Crit. Lat.: at the interleaved /
staged level everywhere; on 122B V* 89/66/49 ms vs interleaved 114/68/52 and streaming 86/75/65.
