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
* Milestone 2 (below, 2026-09-07): AppCorr vision (appcorr env, transformers 5) in a separate
  process, per-round merged embeds to the vLLM process over a socket. Done for the Qwen2.5-VL
  and Qwen3.5 axes; the campaign driver exists; accuracy / TTFT / throughput runs are NOT
  started (waiting for the OpenVLA runs to finish and an explicit go).

## Milestone 2: two processes, one socket (2026-09-07)

### Shape

```
appcorr env (transformers 5, torch 2.12)        appcorr-vllm env (vllm 0.28.0, torch 2.13)
  QwenVLStreamingAxis.streaming_forward  --TCP-->  StreamServer (server.py)
     band r corrected+merged -> sink.push()          open/append -> StreamingLLM.open/append
     ...                                             engine.step() between messages
     sink.result()  <------------------------------  text, token_ids, timing (server clock)
```

* `appcorr/vllm_stream/wire.py`: u32 length + JSON header + raw blobs (bf16 as int16 bytes);
  `Frame`, `FrameParser` (server side, incremental), `send_frame`/`recv_frame` (client side).
* `appcorr/vllm_stream/server.py`: one selector loop over all client sockets; applies every
  arrived `open`/`append`/`result`/`abort`, then one `llm.step()` if any request is live, so
  the engine prefills band r while the vision side corrects band r+1. Several requests may
  be live (that is the throughput lever). Timing per request on the server's `perf_counter`:
  `t_open`, `t_final`, `t_first_token`, `t_done` -> `ttft_from_last_chunk_ms`,
  `ttft_from_open_ms`, `total_ms`; `logprobs=k` returns top-k per generated position.
  Launch: `CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0
  PYTHONPATH=<repo> <appcorr-vllm python> -m appcorr.vllm_stream.server --model <id>
  --port 5591 --gpu-mem 0.3` (port 5555 on this box is held by a foreign process).
* `appcorr/vllm_stream/bridge.py` (no vllm import): `LLMBridge` + `StreamSink`; the sink is
  what `streaming_forward(..., sink=)` takes, one-shot arms are one `push(final=True)`.
  Pushes are asynchronous: encoded on the caller, sent by a `bridge-tx` thread, acks read back
  in order before the next reply-bearing call. Measured on 7B: a blocking push parked the
  vision side 5-15 ms per band (26-50 ms of a ~100-150 ms streaming pass) because the server
  only answers between engine steps and a band (~3.7 MB) exceeds the 4 MB loopback send
  buffer; after the change the caller spends 2-4 ms per push (the chunk's own `.cpu()` sync).
* `appcorr/models/qwen_vl_axis.py`: the streaming loop shared by `Qwen35Axis`
  (`qwen35/unified.py`) and `Qwen25VLAxis` (`qwen25vl/unified.py`, new: the only port-specific
  piece is `_rows_of_groups` through the tower's window permutation). `streaming_forward`
  returns `(logits, kv, stats)` in-process or `(None, None, stats)` with a sink;
  `oneshot_embeds` is the floor/ceiling arm. Refactor gated bitwise on 35B; Qwen2.5-VL axis
  gated by `analysis/experiments/qwen25vl_axis_gate.py` (row mapping fp64-exact, g=1 embeds
  == stock tower rel-L2 0.00000, first token == stock, keep<1 lands between floor/ceiling).

### Plumbing gate (`analysis/experiments/vllm_bridge_gate.py`)

Per COCO image, ceiling / floor / stream, each through HF in-process AND through the socket.
Pass rule: the stream arm's first greedy token agrees with its HF twin on every image, and its
drift is no larger than the one-chunk ceiling arm's (the engine's own numerical band; a
`first_diff_margin` -- vLLM logprob of its token minus the HF token's -- tells a near-tie from
a defect). Qwen2.5-VL-7B, 4 images, 16 tokens: floor 4/4 exact, stream 4/4 exact (16/16
tokens each), ceiling 2/4 with the two misses at margin 0.125 nats ("The"/"This",
"shows"/"features") -> PASS (`bridge_gate_qwen2.5-vl-7b-instruct.json`). Stream
`ttft_from_last_chunk` 5-12 ms vs 8-17 ms one-shot; `ttft_from_open` 60-80 ms for the whole
streaming pass on a shared GPU.

### Two defects found on the way (both fixed, both worth knowing about)

**1. cuDNN SDPA returns a wrong head on real decode activations.** PyTorch's cuDNN attention
backend (on by default in torch 2.12.1+cu130 on B200) gives a wrong output for ONE head of
Qwen2.5-VL-7B's decode step: q `[1,28,1,128]`, kv `[1,4,382,128]`, GQA, no mask -> max |diff|
1.56 vs the math backend (flash: 3.9e-3), head 3 only. Independent of `enable_gqa`, scale,
mask, contiguity; random tensors of the same shape (even with logit magnitudes to +-18000) do
NOT trigger it -- data-dependent. Symptom: HF decodes "ribbeded"/"dotteded" where vLLM /
eager / flash / math give "ribbed"/"dotted"; every HF loop and `model.generate` show it;
prefill and vision logits are within bf16 noise for all backends (q_len=1 only). Fix:
`torch.backends.cuda.enable_cudnn_sdp(False)` at import of `qwen_vl_axis.py` (every HF-side
consumer imports it); repro tensors in `analysis/results/vllm_stream/sdpa_cudnn_repro.pt`.
Eligibility is head_dim <= 128 (cuDNN refuses 256): Qwen2.5-VL 7B/32B/72B, Mistral-Small-3.1,
LLaVA-OneVision-2-8B, InternVL3.5-8B are eligible; Qwen3.5-35B/122B, Gemma 4, Gemma 3
(head_dim 256) are not. Any historical number measured through an HF decode loop on an
eligible model with this torch build (installed 2026-08-31 after the instance reset) is
worth a second look; the "7B RefCOCO boxes degenerate" note is a candidate explanation --
a hypothesis, not checked.

**2. `attention.correct` was sync-bound.** The corrected band's window loop did
`(token_idx in window).any()` + two boolean gathers per window x 28 windowed layers x 4
bands (~14k GPU->CPU syncs per 2k-token image): `vision_correct` 2.08 s against 140 ms for
the whole base pass. `backbone.correct_plan` now sorts the query rows once per call (one
sync) and derives every layer's `(window, a:b)` slices on the CPU; `attention.correct(plan=)`
runs the same sdpa on the same rows. A/B on 2 images x keep {1.0, 0.5}: image embeds and
logits bitwise equal, `streaming_forward` 0.55-1.4 s -> 0.15-0.25 s. The executor path
(`offload/server/model/qwen25vl_executor.py`) keeps `plan=None` and is untouched. Remaining
floor: both base and correct passes launch one sdpa per window (~3.6k / ~14.5k launches);
`vision_correct` ~150-230 ms vs `vision_base` ~100-140 ms on the shared GPU. A varlen kernel
would remove it (no flash-attn in the appcorr env); not needed for the campaign.

### Campaign driver (`analysis/experiments/qwen_vllm_accuracy.py`)

`--family qwen25vl|qwen35 --model <id> --dataset <name> --backend vllm|hf --port 5591
--arms floor streaming ceiling --groups 4 --keep 1.0 --level 2 --degrade-filter box
--samples 0 --max-tokens 24 --concurrency 1 [--think] --out analysis/results/qwen_vllm_accuracy`.
Same `degrade()/get_spec()/record()` as `qwen35_accuracy.py`; resumable by `i`; output
`{dataset}_{model-slug}_{arm}[_g{groups}[_k{keep}]][_c{concurrency}][_hf].jsonl`. Per row:
`t_prep_ms` (degrade + HF image processor, CPU, two images for the streaming arm),
`t_vision_ms` (GPU vision pass incl. pushes), and from the server clock
`ttft_last_chunk_ms`, `ttft_open_ms` (for the streaming arm this already contains the
vision pass), `total_ms`, `gen_tokens`, `finish_reason`. `--concurrency k` keeps k requests
in flight (deque of sinks; the engine batches them) -- the throughput measurement. All arms
decode via vLLM temperature-0 greedy; HF `model.generate` on Qwen2.5-VL is NOT pure greedy
(generation_config `repetition_penalty=1.05`), which is why the driver never uses it.
Smoke (8 RealWorldQA samples, 7B, shared GPU0; NOT campaign data): ceiling `t_vision` 144 ms
+ `ttft_open` 32 ms; streaming `t_vision` 327 ms with `ttft_last_chunk` 9.5 ms,
`ttft_open` 169 ms. The per-sample wall time is dominated by `t_prep` (200-420 ms of CPU
image processing); a throughput run should precompute pixel tensors or use loader workers,
otherwise the client, not the engine, is the bottleneck.

`--prefetch N` preprocesses N samples ahead on a CPU thread: per-sample wall 1591 -> 1291 ms
at concurrency 2 on 7B, but the thread contends with the launch-bound correct loop for the
GIL and inflates `t_vision_ms` (360 -> 680 ms) -- use it for throughput runs, not for the
latency columns.

**The streaming arm is timing-dependent on near-tie samples (expected, not a defect).**
RealWorldQA #504 flipped Yes/No between identical runs. Isolated with the same five vision
chunks pushed three ways (vision output bitwise identical across runs): all chunks queued
before the engine steps -> one prefill -> "No" (logprobs No -0.73 / Yes -0.73, a dead heat);
chunks 30 ms apart -> five prefill steps -> "Yes" (-0.669 / -0.794); each way reproducible
4/4. The prefill-chunk boundaries follow arrival timing, and vLLM's chunked prefill has the
~0.2 nat band measured in the D arm above, so a sample within that band can land either way
from run to run. Report streaming accuracy with a paired count against the ceiling, and expect
run-to-run jitter of a few samples per thousand; the one-shot arms are deterministic.

### How to run each model (nothing started yet)

| model | server | driver | note |
|---|---|---|---|
| Qwen2.5-VL-7B | GPU0 `--gpu-mem 0.3` | GPU0 `--family qwen25vl` | gated end-to-end (above) |
| Qwen2.5-VL-32B | GPU0 `--gpu-mem 0.42` | GPU0 `--family qwen25vl` | gated end-to-end: 12/12 arms exact, 16/16 tokens (`bridge_gate_qwen2.5-vl-32b-instruct.json`); stream `ttft_from_last_chunk` 19-34 ms on the shared GPU; HF 32B bf16 ~67 GB |
| Qwen3.5-35B-A3B | GPU0 `--gpu-mem 0.44 --max-num-seqs 64` | GPU0 `--family qwen35` | gated end-to-end: stream 4/4 exact (16/16 tokens), floor/ceiling near-ties at margin 0-0.375 (`bridge_gate_qwen3.5-35b-a3b.json`); stream `ttft_from_last_chunk` 13-19 ms. `--max-num-seqs` must be <= the mamba cache blocks (431 at 0.45) or the server refuses to start; the HF twin needs no_grad (an autograd graph on the 1.8k-token prefill pushed it to 82 GB and OOMed the shared GPU) |
| Qwen3.5-122B-A10B-FP8 | GPU0 `--gpu-mem 0.85 --max-model-len 4096 --max-num-seqs 32` | GPU0 `--family qwen35 --workers 6 --concurrency 4` (vision-only HF side) | single-GPU form (GPU1 off limits from 2026-09-08): the driver loads only the tower + embed_tokens, so the engine and the AppCorr side share GPU0; gate = `vllm_vs_hf_preds.py` against the HF-chain rows (below) |

### Single-GPU form (2026-09-08)

With GPU1 off limits the two-process split no longer buys a second GPU, but it still buys the
engine: the HF side of the driver needs only `model.model.visual`, `embed_tokens`, the config-only
`get_rope_index`/`get_vision_position_ids` and `get_image_features`, never a decoder layer, so
`appcorr/models/vision_only.py::load_vision_only` builds just those two submodules and streams
their tensors out of the safetensors shards (7B: 1.8 s / 2.3 GiB; 122B: ~3 GB). Checked bitwise
against the full model's tower on 7B (max|diff| 0.0) after one fix: the tower must be built in
fp32 and its PARAMETERS cast afterwards -- initialising under a bf16 default dtype computes the
non-persistent rotary `inv_freq` buffer in bf16 (features off by 3.8% rel-L2), whereas
`from_pretrained` keeps such buffers at their fp32 init. `--load vision` is the vllm-backend
default (`full` forced for `--backend hf`); output naming is unchanged because the arm is the same.

CPU prep moves off the main thread with `--workers N` (a forked `DataLoader` over the sample
indices; each worker runs degrade + the HF image processor, the main process does the GPU vision
pass and the socket) -- unlike `--prefetch` the workers do not share the GIL with the
launch-bound correct loop. 7B smoke on a shared GPU: ceiling 2.86, floor 3.21, streaming 1.95
samples/s (t_prep 14-139 ms in-worker vs 200-420 ms inline).

The 122B-FP8 twin cannot sit beside the engine for an in-process embeds gate, so the single-GPU
gate compares predictions instead: `analysis/experiments/vllm_vs_hf_preds.py` joins the driver's
rows with the HF campaign rows (`qwen35_accuracy.py`, explicit greedy loop = the engine's
temperature-0 rule) on `i`, and passes when the streaming arm's disagreement rate is within the
ceiling arm's + 1% -- the ceiling's disagreement IS the engine's numerical band (kernels, reduction
order, and the CPU-torchvision vs. GPU-torchvision image processor: the HF chain's
`--fast-processor` routes `apply_chat_template` through cuda, the forked workers cannot), and it
enters both arms equally, so the rule isolates chunked prefill. Chain:
`scratchpad/chain_gpu0_122b_vllm.sh` (server -> 64-sample RWQA gate -> full RWQA + VisDrone
Count re-measurement, `--workers 6 --concurrency 4`, outputs `qwen_vllm_accuracy[_pyr]/`).

Bridge/driver code is family-agnostic (the sink only sees `[T, D]` embeds + `[3, T]` mrope +
delta); the 35B/122B checks are the `vllm_bridge_gate.py --family qwen35` run, same pass rule.

### Driver throughput work (2026-09-08/09, branch `develop/vision-correct-perf`)

The vision-side levers (L1-L5 + the keep=1.0 span slice; docs/memo/qwen_correct_forward_profile.md)
took the streaming pass on 122B VisDrone Det from 200/281 ms (c1/c4) to 73/99 -- the one-shot
tower is 74 -- and throughput at `--concurrency 4` from 2.69 to 4.46 samples/s. c8 gave the same
4.38/s, two shards x c2 gave 5.71/s: the engine was not the bound. Main-loop accounting fields
(`t_loop_prep_ms` blocked on the prep workers, `t_loop_push_ms` bridge encode+queue,
`t_loop_wait_ms` blocked in `result()`) put it in the bridge protocol:

| arm (150 samples) | t_vision | t_loop_wait | t_loop_prep | t_loop_push |
|---|---:|---:|---:|---:|
| streaming c1 | 72 | 210 | 3 | 0.8 |
| streaming c4 | 95 | 101 | 3 | 0.8 |
| ceiling c4 | 65 | 70 | 1 | 0.2 |

`result()` first drained every ack on the one socket, including the five of the sample pushed
just before it; the server (one thread: select -> handle frames -> engine step, 13-19 ms) answers
a chunk only between steps, so the loop paid ~one step per newest-sample chunk, ~100 ms per
iteration, while the engine idled. Fix (bridge only, no server change): a reader thread consumes
acks FIFO (errors land on the push's own sink), `result` goes over a second socket and a sink
waits only for its OWN final ack before asking. Two follow-ups the new accounting fields exposed
(`t_loop_iter_ms` = whole previous iteration, `t_loop_h2d_ms` = the pageable H2D copies of the
inputs): `result()` still joined the sender queue (`_flush`) -- the newest sample's chunks leaving,
again one engine step each -- removed; and the remaining wait was the `result` round trip itself
(the server answers only between steps, ~one step of latency), so a result thread now asks the
server the moment a sink's final ack lands and the main loop collects a parked reply.

VisDrone Det 448, keep 1.0, medians (means in brackets where the distribution is bimodal):

| bridge | conc | samples/s | t_vision | t_loop_wait | t_loop_iter | server total_ms |
|---|---|---:|---:|---:|---:|---:|
| ack-draining `result()` (ab_new) | 4 | 4.46 | 99 | 101 | - | 529 |
| reader thread + result socket (ab_rx) | 4 | 5.00 | 104 | 59 | - | 560 |
| | 8 | 5.88 | 109 | 37 (52) | 154 (170) | 954 |
| + no `_flush` in `result()` (ab_rx2) | 4 | 5.16 | 105 | 41 | 154 | 542 |
| + result thread (ab_rx3) | 4 | **5.44** | 103 | **0 (69)** | 128 (183) | 506 |
| | 8 | **6.28** | 110 | 0 (41) | 127 (159) | 861 |
| ceiling arm, for scale (ab_loop2) | 4 | 7.07 | 76 | 17 | - | 384 |

The protocol wait is gone (median 0; an iteration is prep 3 + H2D 11 + vision 103 + push 1). What
is left is bimodal: p90 wait 247 ms at c4 = the oldest request really is not finished, i.e. the
engine is now the co-bottleneck. Server total_ms / concurrency is ~110-127 ms of engine occupancy
per streaming sample at c4/c8 (5 chunk prefills + 21 decode steps; the ceiling's single prefill
costs 96), and the driver's vision pass is ~100 ms on the SAME GPU: two processes time-slice, so
the two do not fully overlap and the rate lands at 5.4-6.3/s rather than min(8, 10)/s. Levers
past this point are GPU-side (fewer / larger chunks on the engine, MPS, or a second GPU for the
vision pass), not driver code; the driver-side plan (L1-L6 + bridge) is closed at 2.69 -> 6.28/s.

**Chunked prefill is not run-to-run deterministic at the pred level** (122B-FP8, 448 VisDrone Det
boxes, c1): ceiling old/new/re-run 448/448 identical (single chunk: the engine and the whole wire
path are exact), streaming old vs its own re-run 316/448, old vs the fast driver 166/448, and old
vs the fast driver with a 40 ms sleep after every push (`APPCORR_PUSH_DELAY_MS`, diagnostic knob)
287/448 -- back at the self-rate. Chunk arrival spacing decides which chunks the scheduler
prefills in one step, which changes the FP8/MoE GEMM shapes; the disagreements are 1-px box
shifts and accuracy moved within +-1 pp. Consequence: driver identity is gated on the pushed bytes
(`qwen_axis_snapshot.py`), never on streaming preds; streaming rows compare at the accuracy level.

### Latency view, concurrency 1 (2026-09-09, live tree after the port, server on the new code)

VisDrone Det 448, medians, driver clock from "inputs on the GPU" (t=0). Ceiling = one-shot tower
then one prefill; streaming = base pass, 4 correction rounds, 5 chunks (`latency_c1_20260909/`).

| arm | t_open | t_last_push | TTFT from t=0 | TTFT after last chunk | end-to-end |
|---|---:|---:|---:|---:|---:|
| ceiling | 26 | 26 | **95** | 95 (= tower 23 + prefill 68) | 211 |
| streaming k1.0 | 21 | 54 | 178 | **48** | 295 |
| streaming k0.25 | 62 | 86 | 211 | 46 | 307 |

Reading: with the whole image on the box at t=0 there is nothing to hide, and chunking costs --
five scheduler steps in series instead of one prefill (open -> first token 156 vs 68 ms), so
streaming is +83 ms to first token and +84 ms end-to-end. What streaming buys is the part after
the FULL-resolution data lands: 48 ms to first token vs the ceiling's 95 (tower + full prefill),
-50%. Break-even: the full image must arrive >= ~83 ms after the base (178 - 95); beyond that the
saving approaches 47 ms per request. keep 0.25 pays +41 ms before the first chunk (the received-
attention pass runs inside the base tower pass) and gains nothing on the last chunk (46 vs 48).

### Lat. / Crit. Lat. table columns (2026-09-09)

`analysis/experiments/latency_probe.py` is the FLOPs-style quick probe behind the eval table's
two new latency columns: per dataset, ceiling + streaming keep {1.0, 0.50, 0.25} at concurrency 1,
40 evenly spaced images (first 4 dropped as warm-up), medians pinned into
`analysis/results/latency/inprocess_latency.json` with the FLOPs file's key names (`full`,
`total_k*` = Lat., `k*` = Crit. Lat.). The whole 122B sweep (7 datasets x 4 arms) takes ~10 min on
the live server.

Definitions (all are TTFT = the engine's first token; decode excluded):
* **Lat.** = first token - driver t0 (inputs on cuda:0, before the vision pass): vision and prefill
  serialized, no transmission credit. The ceiling's Lat. is the Full-res TTFT.
* **Crit. Lat.** = first token - the moment the LAST band's vision correction starts
  (`ttft_start_ms - t_pushes_ms[groups-2]`, the previous band's push having synced the GPU). It
  contains the last band's correction, its push, the last image chunk's prefill, the trailing
  text and the first token -- the same accounting as Crit. Comp. (the last arrival's work). The
  first version of this column measured from the LAST push (`ttft_last_chunk_ms`: only the
  trailing-text chunk + first token, ~47 ms flat); it is kept in `detail.*.ttft_last_chunk_ms`.

122B-FP8 result (ms; ratio to the ceiling TTFT):

| dataset | Full TTFT | Stream k1.0 Lat. / Crit. | k0.50 Lat. / Crit. | k0.25 Lat. / Crit. | v1 Crit. (from last push) |
|---|--:|--:|--:|--:|--:|
| ChartQA | 60 | 116 (193%) / 86 (144%) | 125 / 83 | 128 / 85 | 47 |
| RealWorldQA | 115 | 213 (185%) / 153 (133%) | 262 / 150 | 254 / 147 | 50 |
| RefCOCO | 61 | 116 (191%) / 88 (145%) | 123 / 88 | 121 / 86 | 47 |
| TextVQA | 77 | 132 (170%) / 96 (124%) | 161 / 99 | 154 / 93 | 49 |
| VisDrone Count | 78 | 178 (227%) / 135 (172%) | 216 / 138 | 213 / 134 | 48 |
| VisDrone Det | 96 | 177 (184%) / 135 (140%) | 214 / 134 | 216 / 137 | 46 |
| V*Bench | 293 | 456 (156%) / 254 (87%) | 736 / 272 | 699 / 265 | 62 |

Reading: on the single-GPU served form the last band's window is 85-150 ms, i.e. 120-175% of the
Full-res TTFT everywhere except V* (87%), although its FLOPs share is 20-34%. Hypothesis (not
isolated yet): GPU sharing -- band r's vision correction runs on the same GPU as the engine's
prefill of chunk r-1, and the last chunk's prefill queues behind whatever of chunk 2's prefill is
still running, so the wall window is closer to (engine occupancy + tower) serialized than to the
FLOPs share. A per-push timeline (t_pushes_ms vs the engine's per-chunk done times) would settle
it; a second GPU / MPS for the tower is the lever if so. The engine's own floor (v1 column, trailing-text chunk + first token) is ~47 ms flat. The
keep<1 arms are slower on Lat. by the received-attention pass (+30-50 ms; V* +250 ms) and
identical on Crit. -- exactly the Comp.-vs-Crit. asymmetry the FLOPs columns show. A second GPU
(or MPS) for the tower is the lever; it is a deployment choice, not driver code.

The HF in-process VLMs use `appcorr/latency.py` (`LatencyCounter`: FlopCounter's scopes, wall
time, critical = from the first entry into the highest arrival index) through each FLOPs report
script's `--latency` flag (ov2, gemma3, gemma4, museglimmer, qwen25vl_arms, mistral3_oracle
--flops), so a latency cell times exactly the arm its Comp. cell counts. Chain:
`scripts/latency_fill_hf_20260909.sh`.
