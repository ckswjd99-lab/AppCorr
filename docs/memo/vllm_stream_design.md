# vLLM streaming prefill for AppCorr (`appcorr/vllm_stream/`)

Status: prototype on `develop/vllm-stream`, vLLM 0.11.2 (env `openrlhf_base`, someone else's --
read-only) and vLLM 0.28.0 (env `appcorr-vllm`, ours: torch 2.13.0+cu130, transformers 5.16.1;
the `appcorr` env keeps torch 2.12.1 and must not get vLLM), in-process engine core, TP=1,
Qwen2.5-VL as the first model. No vLLM fork: four monkey-patched hooks + a scheduler subclass,
installable as a `vllm.general_plugins` entry point. 0.28.0 is the target from here on (it has
Qwen3.5 / Gemma 4 / LLaVA-OneVision-2; 0.11.2 has none of them).

## What it is

AppCorr's streaming arm (`offload/server/model/qwen25vl_executor.py`, `llm_schedule="streaming"`)
prefills the causal LLM over vision embeddings *as they are corrected*, round by round, with the
text once at the end. In a serving engine that is a request whose prompt embeddings **arrive in
chunks**: open with chunk 0, append chunks, mark the last one final, sample only then. Every
prompt row is prefilled exactly once (append-only) -- the chunked/new-only vision correction
never rewrites a row the LLM already consumed, so Stream2LLM's LCP + invalidate "update mode" is
not needed and not implemented.

## Mapping onto vLLM v1

| concern | stock vLLM 0.11.2 | here |
|---|---|---|
| prompt data | `prompt_token_ids` xor `prompt_embeds` (`--enable-prompt-embeds`) | embeds only, per chunk |
| message | `EngineCoreRequest` | same struct; `trace_headers["x-appcorr-stream"]` = `open`/`oneshot`/`append`/`final`, M-RoPE positions of the chunk as JSON in `x-appcorr-mrope` (+ `-delta`) |
| engine entry | `EngineCore.preprocess_add_request` -> `Request`; `add_request` -> scheduler | patched: `open` -> `StreamingRequest`; `append`/`final` -> `StreamAppend` -> `StreamingScheduler.stream_append` |
| growth | -- | `StreamingRequest.stream_append`: cat `prompt_embeds`, extend `_all_token_ids` by `[0]*t`, `num_prompt_tokens += t`, cat mrope |
| "don't sample yet" | sampling happens when the request's last token is computed | **hold-back-one**: `num_tokens = len(all) - 1` while open; the scheduler never schedules the last held row, so it never samples; the runner sees the full length and its stock `seq_len < num_tokens` discard handles logits |
| runner copy | `CachedRequestState.prompt_embeds/num_prompt_tokens/mrope_positions` fixed at `NewRequestData` | `SchedulerOutput.appcorr_stream_updates[req_id] = merged chunk`; patched `_update_states` cats them and evicts the request from `InputBatch` so the stock re-add path rebuilds its row |
| M-RoPE | `_init_mrope_positions` calls `model.get_mrope_input_positions(prompt_token_ids, mm_features)` -- crashes on an embeds-only prompt | client computes positions from the token skeleton (same model function, duck-typed features) and ships them per chunk; `NewRequestData.appcorr_stream` carries them at first schedule |
| KV cache | block table grows per scheduled tokens | unchanged (append-only ⇒ blocks only grow) |
| prefix caching | block hashing on token ids | disabled for streaming engines (embeds have no ids) |

Idle behaviour: an open request with everything-but-one computed presents `num_new_tokens == 0`
and the stock running loop `continue`s over it (no busy work, no sampling, blocks kept).

## Client (`client.py`)

`StreamingLLM` wraps `vllm.LLM(enable_prompt_embeds=True, enable_prefix_caching=False,
scheduler_cls=StreamingScheduler)` with `VLLM_ENABLE_V1_MULTIPROCESSING=0`. Chunks are turned
into `EngineCoreRequest`s by vLLM's own `Processor` (stock max_tokens/eos handling); `open` goes
through `LLMEngine.add_request` (registers the output-processor state), `append` straight to the
in-process engine core (`OutputProcessor.add_request` would reject the duplicate id).
`step()` = one engine step; the caller interleaves arrivals and steps.

`Qwen25VLComposer` builds the exact prompt vLLM would run (HF processor, same token skeleton --
checked against the stock request's `prompt_token_ids` in the gate), embeds text rows with the
model's embedding table and image rows with either the model's own vision tower or a supplied
`[image_len, D]` tensor (milestone 2: AppCorr's progressive tower), and gets M-RoPE positions
from the model's `get_mrope_input_positions`.

## Gate (`analysis/experiments/vllm_stream_gate.py`)

A stock image request vs B one-shot embeds vs C streamed (one chunk per step) vs C2 burst
(all chunks before the first step) vs D stock chunked prefill (`max_num_batched_tokens`,
separate engine): greedy tokens + per-token logprobs. Results:
`analysis/results/vllm_stream/gate_qwen25vl7b.json`.

Result (2026-09-07, Qwen2.5-VL-7B, 8 COCO images, 4 chunks, max_tokens 48, eager, GPU0 shared with a
running OpenVLA job -- timing not measured here):

| arm | greedy tokens == A | max dlogprob (common prefix) | first-token dlogprob |
|---|---|---|---|
| B one-shot embeds | 8/8 | 0 | 0 |
| C2 burst append (4 chunks, then step) | 8/8 | 0 | 0 |
| C one chunk per step | 5/8 (div@26, @5, @31) | 0.102 | 0.043 |
| D stock chunked prefill, 128 tok | 5/8 (same images, same positions) | 0.130 | 0.043 |

B and C2 are bit-identical to the stock image request: the embeds path, m-rope headers and the
hold-back-one bookkeeping introduce nothing. C matches D image by image (identical divergence
positions and per-image dlogprob on 7/8; D's 0.130 is its own 128-token boundary on image 0),
i.e. per-step appends sit in the numeric band of stock chunked prefill -- the only difference
between the two arms is *which* boundaries the prefill is cut at. `stream_state` confirmed the
hold-back on every image (num_computed == num_prompt - 1 and no output while open).

Two stock vLLM 0.11.2 bugs surfaced and are worked around in this package (not upstreamed):

1. `GPUModelRunner._preprocess` overwrites `inputs_embeds[:n]` with `embed_input_ids(input_ids)`
   on multimodal models, so `prompt_embeds` are silently replaced by the placeholder embedding
   (garbage output, mechanics otherwise fine). `runner_patch._preprocess` snapshots the
   embeds rows (`~is_token_ids`) and restores them after the stock call.
2. `Qwen2_5_VisionAttention.__init__` discards the `use_upstream_fa=True` returned by
   `maybe_get_vit_flash_attn_backend`, so the bundled FA2 kernel is used and rejects the ViT's
   head dim 80 ("headdim not being a multiple of 32"); `mm_encoder_attn_backend=TORCH_SDPA`
   does not help (converted back to FLASH_ATTN on CUDA). `compat.fix_qwen2_5_vit_upstream_fa()`.

### Port to vLLM 0.28.0 (2026-09-07, env `appcorr-vllm`)

Same four hooks, same scheduler subclass; what changed (all version-gated in place, 0.11.2 kept):

| 0.11.2 | 0.28.0 | in this package |
|---|---|---|
| `LLMEngine.processor.process_inputs(...)` | `LLMEngine.input_processor.process_inputs(..., supported_tasks=)` | `client._ecr` |
| request ids used verbatim | `add_request` randomises the id (`<id>-<8 hex>`); outputs carry the external id, the engine core the internal one | `StreamingLLM._internal` map; appends are re-addressed to the core id |
| `Scheduler.schedule()` | `schedule(throttle_prefills)` | pass-through |
| `_init_mrope_positions` crashes on embeds-only | assigns *text* positions to an embeds-only prompt | hook unchanged (client positions win) |
| `_preprocess` clobbers prompt_embeds (bug 1) | fixed upstream (`torch.where(is_token_ids, ...)`) | patch installed on 0.11.2 only |
| `Qwen2_5_VisionAttention` drops `use_upstream_fa` (bug 2) | ViT attention is `MMEncoderAttention`, backend chosen with the head size | `compat` is a no-op on 0.28.0 |
| `get_mrope_input_positions` scans token ids | walks `mm_features` by `mm_position.offset` + `modality` | `_DuckFeature` grew `modality`, `mm_position`, `identifier`, `data.get` |
| pixel normalisation in the HF processor | on the device by default (`mm_device_do_normalize=True`) | `StreamingLLM` passes `False` so the stock arm and the composer share the CPU path |
| sync scheduler, one batch in flight | async scheduling on by default (AsyncScheduler + 2-deep batch queue) | `StreamingLLM` passes `async_scheduling=False`; the hooks were gated on the sync engine |
| -- | Model Runner V2 is the default for common architectures, but "does not yet support prompt embeds" -> falls back to V1 | our hooks are on V1's `GPUModelRunner`; the day V2 takes prompt_embeds they need a second home |
| -- | upstream "streaming input sessions" (`resumable=True`, `StreamingUpdate`): token-ids only, the request must *finish* between chunks (one sampled token per chunk, discarded on resume) | not usable for embeds; our hold-back-one is the no-sample-in-between analogue. `_update_streaming_request` in the runner is the stock twin of our `_update_states` hook |

Gate on 0.28.0 (same protocol, `gate_qwen25vl7b_vllm0280.json`; flashinfer/trtllm-gen + cutlass FA
sm100 kernels, eager, GPU0 shared with the OpenVLA job):

| arm | greedy tokens == A | max dlogprob (common prefix) | first-token dlogprob |
|---|---|---|---|
| B one-shot embeds | 8/8 | 0 | 0 |
| C2 burst append | 8/8 | 0 | 0 |
| C one chunk per step | 5/8 (div@0, @32, @31) | 0.124 | 0.060 |
| D stock chunked prefill, 128 tok | 5/8 (same images, same positions) | 0.124 | 0.060 |

B/C2 bit-identical again; C == D on all 8 images to three digits (0.11.2: 7/8), including image 0
where both flip the *first* token -- the chunk-boundary band of the 0.28.0 kernels is wider than
0.11.2's (first-token dlogprob 0.060 vs 0.043) and it is stock vLLM's band, not ours. Observation,
not a mechanism: C and D cut at different boundaries yet land on identical dlogprobs, so what
separates them from A looks like "chunked vs one-shot prefill", not the boundary position.

## TTFT under progressive arrival (`analysis/experiments/vllm_stream_ttft.py`)

Chunks become available at t = k * gap (text in the last chunk); `stock` submits one message
at t_last, `stream` appends each chunk on arrival and steps in between. TTFT is wall-clock from
t_last to the first sampled token for both, so the delta is the prefill hidden under the gaps.
Qwen2.5-VL-7B, cudagraphs on, 4 images x 5 reps, paired; GPU0 shared with a running OpenVLA
LIBERO job (its contention spikes are the wide ranges -- medians only below).
`analysis/results/vllm_stream/ttft_qwen25vl7b{,_x3}.json`.

| prompt | gap | stock median | stream median | paired delta median [min, max] |
|---|---|---|---|---|
| 288-427 tok (COCO as-is) | 0 ms | 10.3 ms | 11.1 ms | -0.9 [-30, +18] |
| | 30 ms | 16.1 | 28.8 | -7.6 [-43, +15] |
| | 100 ms | 10.8 | 9.2 | +1.3 [-4, +11] |
| 3210-3555 tok (`--upscale 3`) | 0 ms | 59.7 | 65.4 | -0.3 [-46, +30] |
| | 30 ms | 68.8 | 35.1 | +30.0 [-13, +59] |
| | 100 ms | 43.9 | 23.9 | +22.2 [+10, +51] |

Reading: at ~400 tokens a 7B prefill on B200 is ~10 ms and launch-bound, there is nothing to
hide and the arms tie (gap=0 delta is the per-append overhead, ~1 ms). At ~3.3k tokens the
last-chunk prefill (~900 tokens + text) is what remains after the last arrival: 24 ms vs
44-69 ms, every pair positive at gap=100. The gap=30 rows are the noisy ones: a ~10-60 ms step
in flight when a chunk arrives delays that append, which the shared GPU's spikes amplify.

Perf bug found on the way (fixed, `request.cpu_cat`): the per-append `torch.cat` of the CPU
prompt-embeds/M-RoPE tensors cost 10-96 ms *each* (stream TTFT 160 ms vs stock 14 ms in the
first run). Not our logic -- CPU `torch.cat` on a 1 MB tensor with the engine's default 72
intra-op threads is 90 ms on this box (0.06 ms single-threaded, OpenMP fork/join). The engine
process's thread count is not ours to change, so the concatenations go through numpy (memcpy);
appends are now 0.2-0.7 ms. Stock's own prompt-embeds staging (`copy_` into the pinned buffer,
~14 calls, 6 ms per step) has the same disease and is left alone (it is paid by both arms).

### Qwen3.5 on 0.28.0 (2026-09-07): same composer, same hooks, gated

The composer is model-agnostic within the Qwen2-VL family (`<|image_pad|>` span, LM embedding
table + `model.visual(pixel_values, grid_thw)` -> `[T, D]`, `get_mrope_input_positions` over a
duck-typed placeholder feature), so Qwen3.5 needed no code beyond taking `vllm_config` from the
engine (the Qwen3.5 model classes do not keep `.vllm_config`) and the deepstack assert above.
Qwen3.5 is a hybrid (GDN linear attention + full attention, MoE); the hold-back-one chunked
prefill therefore also exercises vLLM's mamba-state chunking (attention block size raised to
1056 tokens to match the mamba page). Same protocol (8 COCO images, 4 chunks, 48 greedy tokens,
eager, GPU0 shared with the OpenVLA job); `gate_qwen35_35b_vllm0280.json`,
`gate_qwen35_122b_fp8_vllm0280.json`:

| model | arm | greedy tokens == A | max dlogprob (common prefix) | first-token dlogprob |
|---|---|---|---|---|
| Qwen3.5-35B-A3B bf16 | B one-shot embeds | 8/8 | 0 | 0 |
| | C2 burst append | 8/8 | 0 | 0 |
| | C one chunk per step | 6/8 (div@12, @12) | 0.163 | 2.9e-5 |
| | D stock chunked prefill, 128 tok | 6/8 (div@39, @12) | 0.209 | 5.1e-5 |
| Qwen3.5-122B-A10B-FP8 | B one-shot embeds | 8/8 | 0 | 0 |
| | C2 burst append | 8/8 | 0 | 0 |
| | C one chunk per step | 6/8 (div@13, @43) | 0.417 | 0.027 |
| | D stock chunked prefill, 128 tok | 7/8 (div@46) | 0.433 | 0.065 |

Reading: B/C2 bit-identical on both, so the embeds/positions/append plumbing is exact for the
hybrid stack too. C sits inside D's band on 35B (its max dlogprob is below D's, first token
~1e-5 for both, so the GDN chunked-state path is nearly exact and the band is the attention
kernels'). On 122B-FP8 the band is larger for C and D alike (max dlogprob 0.42 / 0.43, first token
0.027 / 0.065; C 6/8 vs D 7/8 is one image at 8 samples) -- the FP8 MoE GEMMs add
chunk-shape-dependent rounding; this is the vLLM counterpart of the HF-side finding
that 122B-FP8 chunked prefill is lossy while 35B bf16 is not, and again stock vLLM's own chunked
prefill shows it (D), not our hooks. Model load: 35B 23 s, 122B-FP8 50 s; the first 122B run
spent ~12 min in FP8 MoE kernel JIT/cubin fetch (cached afterwards).

## Known limits / next

* In-process only. `NewRequestData.appcorr_stream` and `SchedulerOutput.appcorr_stream_updates`
  are plain attributes; a multi-process core needs them as msgpack fields (or a side channel).
* `open` must set `max_tokens` (the Processor derives the default from the first chunk's length).
* Structured output / spec decode / async scheduling untested with streaming requests (0.28.0:
  async scheduling is explicitly turned off by `StreamingLLM`).
* Qwen3-VL proper (deepstack): its vision tower also emits per-level features that the LM adds
  to its hidden states at `deepstack_visual_indexes` layers, through a model side buffer
  (`_set_deepstack_input_embeds`, filled by `embed_input_ids` from mm inputs) that a
  `prompt_embeds`-only request never fills. Streaming them would need `[levels, T, D]` per chunk
  on the wire and a runner hook scattering the scheduled rows in persistent-batch order. NOT
  needed for Qwen3.5: every Qwen3.5 checkpoint here (4B, 35B-A3B, 122B-A10B, 122B-A10B-FP8) ships
  `deepstack_visual_indexes: []` (`deepstack_num_level == 0`, tower output is plain `[T, D]`,
  and vLLM's `Qwen3_5*ForConditionalGeneration.embed_input_ids` does not compute deepstack at
  all). `Qwen25VLComposer.embed` asserts `deepstack_num_level == 0` so a real Qwen3-VL fails
  loudly instead of silently dropping the levels.
* Milestone 2: AppCorr vision (appcorr env, transformers 5) in a separate process, per-round
  merged embeds to the vLLM process over a socket. The TTFT script already has the
  chunk-arrival timeline; the milestone replaces its synthetic `gap` with AppCorr's real
  per-round correction times.
