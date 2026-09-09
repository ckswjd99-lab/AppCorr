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

### Trailing text rides the last band's chunk (2026-09-09)

`QwenVLStreamingAxis.streaming_forward` used to push the last band's image rows and then the
trailing text (question + generation prompt) as a separate chunk -- two engine prefill steps for
one arrival. Now the last band's chunk carries its image rows AND the trailing text (`prefill(seq)`
on the last non-empty band; `groups` pushes per request instead of `groups + 1`; the same holds
on the in-process HF path). Accounting is unchanged: the trailing text was already charged to the
last arrival. What moves:
- `t_pushes_ms` has `groups` entries; `ttft_last_chunk_ms` (the v1 Crit. Lat.) now spans image
  tail + text + first token, so it is no longer comparable with the entries measured before
  this change (the v2 `ttft_start - t_pushes[groups-2]` is unaffected).
- CPU gate (`qwen_axis_cpu_unittest.py` vs a pre-change copy): image_embeds, decode_start_pos,
  corrected_groups bitwise; logits differ by <=4e-7 fp32 (chunked-vs-merged prefill reduction
  order); the FLOPs counter's `llm_prefill` grows by 0.1-1.4% on the toy models because it counts
  each chunk's attention dense (q x all keys of the chunk) -- the same convention the one-chunk
  ceiling has always been counted under, so the served rows' Comp. cells would shift by well under
  1% if re-run. No GPU gate yet: the served preds (`vllm_vs_hf_preds.py`) and the table's
  latency entries for Qwen3.5 were all measured with the separate trailing push.
- Served A/B (Qwen3.5-35B-A3B bf16, port 5591, concurrency 1, RWQA 192 evenly spaced, old push
  structure vs merged; rows in `analysis/results/vllm_stream/tail_merge_ab_20260909/{old,new,new_rep}`):
  preds are NOT bitwise on the served path either, but the old-vs-new disagreement is the same size
  as a same-code repeat -- k1.0: old==new 187/192, new==new_rep 189/192, acc 78.65 in all three runs;
  k0.5: old==new 184/192, new==new_rep 189/192, acc 80.73 -> 78.65 / 79.17 (three answer flips that
  reproduce across the two new runs, all three against the merge, 3-0 paired, p=0.125 -- noise-level
  on 192, not claimed either way; the merged runs agree MORE with the ceiling preds, 145 -> 150/149).
  The same-code 3/192 spread is the chunk-coalescing nondeterminism already seen on 122B, at 35B scale.
  Latency probe (chartqa / realworldqa / visdrone_det, n=36, v2 Crit.): k1.0 53.2->52.2 / 93.6->89.9 /
  79.8->71.6 ms (one engine step saved, ~2.5 ms on the RWQA TTFT median); k0.5 and k0.25 move
  both ways within the probe's own +-20 ms run-to-run spread. The table's served entries were kept
  (old structure; the difference is inside probe noise).

### Why the streaming Crit. Lat. exceeds the one-shot prefill (2026-09-09, measured)

Question: streaming's total TTFT being above the one-shot ceiling is expected (it adds work), but
its critical window -- the last band's correction + last chunk's prefill -- came out at 91 ms on
35B RWQA, above the ceiling's whole prefill (~40 ms). Server-side trace (`APPCORR_SERVER_TRACE`:
every chunk arrival, every `_step` with its per-request computed-token delta) + per-step
`torch.profiler` (`APPCORR_STEP_PROFILE`) on Qwen3.5-35B-A3B bf16, RWQA 24 samples, port 5591,
concurrency 1. Rows + traces: `analysis/results/vllm_stream/critlat_20260909/{,triton/}`.

**1. The engine's prefill step is flat in tokens.** Unprofiled step wall (ms, median): 329 tok
43 / 326 tok 34-39 / 655-690 tok 39-41 / 1343 tok ~40 (ceiling's single step). Profiled CUDA sum:
133 tok 26, 329 tok 35, 456 tok 32, 1000 tok 39. Per step: `vllm::moe_forward_shared` 9.5-13.5 ms
(40 layers; 128 experts x top-8 -> every expert is hit from a few dozen tokens on, so each step
reads the whole expert weight set: ~66 GB bf16 / 8 TB/s = ~8 ms -- a bandwidth floor, hypothesis
consistent with the measured 9.5-11 ms at 329 tokens, not isolated further), `aten::mm` 2-3 ms,
GDN 2-4 ms, and ~2,470 kernel launches per step whose CPU side alone is 13-17 ms (wall - CUDA).
`--moe-backend triton` does not change this (fused_moe_kernel 9.6 ms at 329 tok, 13.5 at 1000;
steps 43-47 ms, i.e. slightly slower than the TRTLLM default; first-shape autotune costs 460 ms
once). So on this model, below ~1.5k prompt tokens, a chunk's prefill costs the same as the whole
prompt's: chunking the prompt cannot shorten the last prefill, it only multiplies the floor.

**2. The single-threaded server cannot read a chunk during a step.** Pushes leave the driver every
12-20 ms (band corrections are launch-bound, ~20 ms each), steps take ~40: chunk 1 waits for
step 0, chunks 2 and 3 coalesce behind step 1 (3 steps for 4 pushes is the norm), and the LAST
chunk sits in the socket buffer ~30 ms before its step starts. Timeline of streaming-62 (driver
clock, ms; t_recv = server arrival on the same host clock): base pass done + chunk 0 arrives 51 |
step(329) 51-94 | chunk 1 arrives 96 (queued) | step(326) 96-130 | chunks 2+3 arrive 131-132 (both
queued; the GPU had finished band 3 at ~98) | step(688) 132-173 | decode step -> first token 177.
Reading the socket during the step (a reader thread) would only remove the ~1 ms handle latency:
the step itself is the queue.

**3. Accounting bug in the driver's latency view (fixed in the same change).** `ttft_start_ms` was
`t_open + ttft_from_open`, with `t_open` the CPU issue time of push 0 -- but the tower is sync-free
and the CPU runs 20+ ms ahead of the GPU, so the server-side span was anchored before the chunk
had left: the streaming TTFT was under-reported by ~20-23 ms and the ceiling's by ~6 ms (its
one 5.5 MB chunk's D2H + transport). Crit. Lat. started from `t_pushes[g-2]`, the CPU issue of
push g-2, which is ~20 ms before the GPU actually reached band g-1 -- the two errors roughly
cancelled (93.6 reported vs ~99 true for streaming-62). Now: `ttft_start_ms = t_recv_ms[0] +
ttft_open_ms` (server arrival, same monotonic host clock; the old value is kept as
`ttft_start_issue_ms`), rows carry `t_sent_ms` (sender thread, after the D2H that waits on the
band's GPU work = when the band's correction really finished), and `latency_probe.py` starts the
critical window at `t_sent_ms[g-2]`. Every served latency entry in the table (122B and 35B) was
measured with the old anchors and needs a re-measure: the ceiling TTFTs move up ~6 ms, the
streaming ones ~20 ms, so the streaming/ceiling ratios grow.

Decomposition (35B RWQA, true anchors): Crit ~ 20 (last band correction, launch-bound) + ~30
(socket wait behind the previous chunk's step) + ~41 (last chunk's step, flat floor) + 4 (first
decode step) + transport ~= 99 ms vs one-shot 31 (tower) + 6 (chunk) + 40 (step) + 4 ~= 86 ms.
The window is not "too long" for what it contains; the premise that the last chunk's prefill is
cheap is what fails at this scale. Where the step does scale with tokens (V* ~5k-token prompts:
Crit 87% of Full TTFT in the 122B table; dense LLMs), the window drops below the ceiling.

Levers, honest: (a) fewer chunks (g=2) removes one queue stage, floor stays: Crit >= 20 + 41 + 4;
(b) the flat floor is vLLM's eager prefill path (2.5k launches) + the MoE weight read -- CUDA-graph
capture of prefill shapes would cut the launch half for the ceiling and streaming alike, the weight
read only shrinks when a chunk is small enough to leave experts untouched (<~16 tokens), which no
band is; (c) a second GPU / MPS for the tower removes the GPU-sharing part of the 30 ms wait but not
the step serialization. None of these make the last chunk's prefill scale with its size on an MoE
model at these prompt lengths.

### Deferred patch score for keep<1 (2026-09-09; `pscore_defer`, gated, not yet in any table row)

The keep<1 arms' first push was ~44 ms later than keep=1.0's in-process (87 vs 43 ms, 35B RWQA
vision-only gate) because `score = energy x received attention` needs the O(T^2) column-sum pass
over all 27 base-pass layers before band 0 can be selected. User's call: take the score off the
first band's critical path. Literal "band 0 fully corrected, score the rest" degenerates at
keep=0.25 (band 0 IS a quarter of the groups, so the whole budget would go to the spatially
top quarter); the shipped form keeps the budget split and ranks band 0 on energy alone, then
computes the attention term right after push 0 (`PSCORE` stage, excluded from FLOPs as before)
from the base pass's stashed post-RoPE q AND base k (`{tag}_q`, `{tag}_k0`; the k must be
stashed because `correct` overwrites the cached kv rows in place -- reading k from the cache
after band 0's correction gave a different score and flipped later bands' selections in the
first gate). Bands 1..g-1 then rank on the identical score. `--pscore eager` keeps the old path.
Gate (`analysis/experiments/qwen_pscore_defer_gate.py`, 35B tower, RWQA 8-16 images, keep
1.0/0.5/0.25, `analysis/results/vllm_stream/pscore_defer_gate_20260909/`): eager image_embeds
sha256 == pre-change code on all 24 cases; deferred attention vectors bitwise == eager; bands
1..3 select identical groups; band 0 overlaps eager's 67-96% (energy-only ranking, by design);
first push 87 -> 39 ms (keep 0.5) / 86 -> 37 ms (keep 0.25), below keep=1.0's 43 (band 0
corrects fewer rows); total vision wall unchanged (the pass moved, it did not shrink). Accuracy
of the keep<1 arms under the deferred score is NOT measured yet (band 0's selection changed):
the served keep<1 rows in the table are eager.

### keep=1.0 latency re-measured under the fixed anchors (2026-09-09, n=36 each, table cells)

Same probe settings as before (40 evenly spaced images, 4 warmup dropped, concurrency 1, g=4, L2).
Old = CPU-issue anchors (biased low: ceiling −6 ms, streaming crit ≈ −20 ms); new = server open
receipt for TTFT, chunk g−2 departure (= its GPU completion) for the critical window. The keep<1
entries were NOT re-measured (they also predate the deferred pscore) and are stashed under
`stale_issue_anchor` in `inprocess_latency.json`; their table cells print `--` until re-run.

| dataset | 35B full old→new | 35B k1 total old→new | 35B k1 crit old→new | 122B full old→new | 122B k1 total old→new | 122B k1 crit old→new |
|---|---|---|---|---|---|---|
| VisDrone Det | 65.1→67.8 | 117.2→121.5 | 76.4→63.5 | 96→101.0 | 177→192.2 | 135→128.3 |
| VisDrone Count | 66.4→69.3 | 116.4→125.4 | 76.4→68.9 | 78→80.2 | 178→192.6 | 135→125.7 |
| V*Bench | 174.0→186.5 | 315.0→388.9 | 113.3→101.0 | 293→304.9 | 456→531.9 | 254→215.9 |
| TextVQA | 60.8→62.7 | 104.7→113.8 | 68.2→64.9 | 77→78.9 | 132→135.0 | 96→84.4 |
| RefCOCO | 47.6→47.8 | 79.8→81.1 | 53.2→48.3 | 61→61.7 | 116→119.0 | 88→87.3 |
| RealWorldQA | 75.3→79.3 | 149.2→173.0 | 91.3→83.3 | 115→121.7 | 213→233.3 | 153→144.0 |
| ChartQA | 50.4→52.0 | 83.8→89.5 | 52.7→53.7 | 60→64.9 | 116→120.5 | 86→82.5 |
| MMVP | 22.9→24.9 | 59.9→61.1 | 36.1→31.7 | – | – | – |
| CV-Bench | 56.3→59.5 | 102.7→91.7 | 61.0→52.2 | – | – | – |
| VSR | (rc=1) →50.3 | →80.9 | →48.8 | – | – | – |

Reading: with the window anchored at the true GPU completion of band g−2, streaming Crit. Lat.
sits at 0.9–1.2× the one-shot TTFT on 35B (V*: 0.54×, CV-Bench 0.88×) and 1.1–1.6× on 122B
(V*: 0.71×). The residual over 1× is the three-cause stack from the previous section (last-band
correction + socket wait behind the in-flight step + the flat ~40 ms MoE step floor); the totals
moved up 2–24% because the old total anchor also started too early. Note the two runs are not
paired (fresh server, different GPU thermal state), so the old→new deltas mix the anchor fix with
run-to-run noise of a few ms; the anchor fix alone is the −6/−20 ms figure from the traced sample.

### Does concurrency make streaming's latency advantage show? (2026-09-09, 35B, measured: no)

Question: the single-request Crit. Lat. of streaming is >= the one-shot TTFT because the MoE
prefill step is flat in tokens (~40 ms floor for <=1.5k tokens), so the last chunk's step costs as
much as a whole prompt's. Batching should amortize that floor across requests and let streaming's
1/g critical work show up as latency. Two sweeps on GPU0 (Qwen3.5-35B-A3B, port 5591, g=4, keep=1.0,
RealWorldQA + VisDrone Det, `APPCORR_SERVER_TRACE` on):
- `conc_sweep_20260909/`: one driver, `--concurrency` 1/4/8/16, 160 evenly spaced images (8 warmup dropped).
- `conc_sweep_20260909/shard/`: N driver processes (`--shard k/N`, each its own vision tower on the
  same GPU) x c in flight each, 400 images. N=16 OOM'd (16 towers x 4.4 GB + server 107 GB > 178 GB):
  those cells are partial (RWQA streaming 12/16 drivers; VisDrone ceiling 3/16) and are not used.
Scripts + `summary.json` are in the result dirs.

**Single driver saturates the driver, not the engine.** Streaming throughput 5.4 -> 6.5/s (RWQA)
from c=1 to c=4 and flat after; ceiling 10.5 -> 13.5/s. Prefill-carrying steps still hold one
prompt (tok/step = one prompt, prefills/step median 1) at every c: the serial vision pass never
lets two prefills queue. This sweep cannot test the batching hypothesis.

**Sharded drivers saturate the engine (86-95% step-busy over the active phase).** Medians, ms
(TTFT for the ceiling; total / crit for streaming), and achieved throughput:

| ds | cfg | ceiling TTFT | ceil req/s | stream total | stream crit | stream req/s | crit / ceil TTFT (same cfg) |
|---|---|--:|--:|--:|--:|--:|--:|
| RWQA | 1 driver c=1 | 80 | 11.0 | 174 | 83 | 5.5 | 1.04 |
| RWQA | N=4 c=1 | 208 | 15.6 | 404 | 154 | 8.1 | 0.74 |
| RWQA | N=8 c=1 | 287 | 16.5 | 632 | 262 | 9.0 | 0.91 |
| RWQA | N=8 c=2 | 433 | 18.9 | 938 | 537 | 9.6 | 1.24 |
| VisDrone | 1 driver c=1 | 69 | 7.7 | 120 | 63 | 5.3 | 0.92 |
| VisDrone | N=4 c=1 | 172 | 14.1 | 327 | 110 | 9.6 | 0.64 |
| VisDrone | N=8 c=1 | 268 | 16.7 | 550 | 172 | 10.8 | 0.64 |
| VisDrone | N=8 c=2 | 382 | 21.5 | 675 | 307 | 12.0 | 0.80 |

Engine step cost by tokens per step (sharded trace, prefill-carrying steps pooled): <512 tok
67 ms (226 ms/ktok), 512-1k 58 (82), 1-2k 54 (40), 2-4k 74 (26), 4-8k 126 (24), 8-16k 74 (8.7).
So the floor IS amortizable -- ms/ktok falls ~10x from chunk-sized to 4k+ steps -- but streaming's
own steps never got there: its prefill steps carried a median 1-3 chunks (~0.7-1.3k tokens) at
83-95 ms/ktok in every configuration, vs 35-38 ms/ktok for the ceiling's whole-prompt steps.
Chunks arrive one at a time, gated by each stream's vision correction, and the scheduler takes
whatever is waiting at each step boundary; nothing accumulates a batch of chunks.

Reading:
1. At the SAME in-flight count the crit/ceiling ratio does move in streaming's favour (1.0 ->
   0.64-0.74 at N=4-8, c=1), but that comparison is at unequal load: streaming's capacity is about
   half the ceiling's (RWQA ~10 vs ~19 req/s, VisDrone ~12 vs ~22), so the ceiling cell is the
   more saturated one. At EQUAL throughput the picture reverses: the ceiling serves 11 req/s at
   80 ms TTFT unsaturated, while streaming at 8-9 req/s is at 154-262 ms crit / 404-632 ms total.
2. Why capacity is half, not the 1/1.5 the FLOPs column suggests: each request costs the engine
   4 chunk steps near the flat floor (4 x ~40-55 ms of step time vs 1 x ~40) plus 4 vision passes
   that share the GPU with the engine (prefill steps slow from 38 to 54-107 ms as N grows). FLOPs
   undercount small MoE steps by ~2.5x (88 vs 35 ms/ktok).
3. p90s widen faster for streaming (RWQA N=8 c=2: 1250 total / 789 crit vs 593 ceiling).

Verdict: in this serving form (vision tower in the driver, single-threaded server, one GPU),
concurrency does not make streaming's advantage more prominent; it exposes the throughput cost.
What would change it (not measured): batching chunks across streams inside the server before
stepping (trades the batching delay against the floor), or moving the correction into the engine
process so its steps and the tower stop contending -- both design changes, not knobs. Caveats:
closed-loop arrivals; towers and engine on one GPU (a separate vision GPU would slow the engine
less); N=16 unusable.

### Spaced band arrival: does the queueing -- and the tower/engine contention -- go away? (2026-09-09, 35B, measured)

Setup: one 35B server on GPU0 (`--gpu-mem 0.60`, `APPCORR_SERVER_TRACE` on), streaming keep=1.0
g=4, concurrency 1, 36 images per cell (4 warm-up dropped), the driver launched with
`APPCORR_PUSH_DELAY_MS` = 0 / 60 / 150 (a sleep inside `StreamSink.push` after each push, i.e.
band r+1's correction starts d ms after band r's chunk was issued -- a producer whose bands
arrive d ms apart). The 0 ms arm is the in-session control and reproduces the table
(RWQA crit 84 / VisDrone Det 65 / V* 101). Files: `analysis/results/latency/probe_qwen35_moe_delay/d{0,60,150}/`,
`run.sh`, `analyze.py`.

Anchors. `L_r` = band r's correction from its start (push r-1 issued + sleep) to the GPU
completion of its D2H (`t_sent[r]`); `q_r` = server receipt minus send of chunk r (the wait
behind an engine step); `last` = last chunk received -> first token (the final prefill step +
first decode); `crit` = first token minus the START of band g-1's correction (= the moment its
pixels arrived), i.e. L_3 + q_3 + last. Medians, ms.

| d | dataset | tok | L_1 L_2 L_3 | q_0 q_1 q_2 q_3 | last | crit (x full-res TTFT) | engine prefill step |
|--:|---|--:|---|---|--:|--:|--:|
| 0 | RWQA | 1351 | 37 43 49 (GPU backlog included) | 1 31 13 31 | 37 | 84 (1.06x of 79) | 36 |
| 0 | VisDrone Det | 1066 | 24 27 33 | 1 34 21 20 | 37 | 65 (0.96x of 68) | -- (decode-dominated trace) |
| 0 | V* | 3341 | 133 129 148 | 3 3 3 3 | 40 | 101 (0.54x of 187) | 52 |
| 60 | RWQA | | 12 12 12 | 1 1 1 1 | 36 | 49 (0.62x) | 34 |
| 60 | VisDrone Det | | 8.5 8.2 8.3 | 1 1 1 1 | 37 | 46 (0.68x) | -- |
| 60 | V* | | 74 65 59 | 3 3 3 3 | 40 | 101 (0.54x) | 48 |
| 150 | RWQA | | 12 12 12 | 1 1 1 1 | 38 | 52 (0.65x) | 36 |
| 150 | VisDrone Det | | 8.6 8.4 8.5 | 1 1 1 1 | 37 | 47 (0.69x) | -- |
| 150 | V* | | 34 34 34 | 3 2 2 3 | 41 | 78 (0.42x) | 38 |

(The d=0 crit uses the table's anchor, the departure of chunk g-2; for d>0 the arrival anchor
above. The approx+band-0 phase, during which the engine is idle in every arm, is identical
across d: 51 / 37 / 175 ms -- the control that the arms differ only in the spacing.)

1. Queueing disappears at any spacing >= a step: q_1..q_3 fall from 13-34 ms to ~1 ms at
   d=60 and d=150 (V* was already spaced by its 60 ms bands). Crit drops to 46-52 ms on the
   two ~1.1-1.4k-token datasets: band correction 8-12 + transfer 1 + the last step 36-38.
   That last step is the MoE step floor (36 ms at 327-358 tokens/step, the same at d=0), so
   spaced arrival gets streaming to ~0.6-0.7x of the one-shot TTFT, not to 1/4: with a flat
   per-step cost the last chunk's prefill costs what the whole image's prefill costs.
2. The tower/engine contention is real and is measured on V*, where the per-band GPU work is
   long enough to overlap an engine step. At d=150 (sleep outlasts the band's GPU work and
   the step it triggers) a band costs 34 ms and the engine step 38 ms; at d=60 the sleep
   ends exactly when the chunk lands on the engine (the sync-free tower issues the push
   ~60 ms before the GPU finishes the band), so band r+1's correction runs on top of chunk
   r's step: 59-74 ms per band and 48 ms per step -- both sides pay, roughly a step's worth
   (+31 ms) on the tower and +10-14 ms on the engine. At d=0 V*'s bands 1-3 take 172 ms of
   GPU-serial time against 3 x 34 = 102 alone (+70), and RWQA's 55 vs 36 (+19), VisDrone's
   37 vs 26 (+11). The RWQA/VisDrone engine step itself does not slow (36 ms at every d):
   their 8-12 ms bands lose to the step, the step does not lose to them.
3. So: spacing removes the queueing wholesale, and removes the contention only when the
   spacing exceeds band GPU time + step time (V* needs >= ~80-100 ms between bands; the
   ~1k-token datasets are contention-free from 60 ms). Below that the two GPU clients
   overlap and the band correction stretches by about one engine step.
4. What remains fixed regardless of spacing: the step floor (36-40 ms) plus one band
   correction (8-34 ms) plus ~1-3 ms transfer. For a slow-transmission scenario the honest
   Crit. Lat. of this design is ~46-52 ms on 35B at 1-1.4k tokens (0.62-0.69x of one-shot),
   ~78 ms on V* (0.42x); the table's d=0 numbers (queue behind the previous step) are the
   fast-producer worst case.

### Crit. Lat. re-measured with 150 ms band spacing (2026-09-09, table cells; rule from now on)

Rule (user, 2026-09-09): streaming latency is measured with the bands spaced far enough apart
that neither the chunk queue nor the tower/engine overlap enters the critical window --
`latency_probe.py --push-delay-ms 150 --skip-ceiling` (env `APPCORR_PUSH_DELAY_MS=150` on the
streaming arms). `k1.00` in `inprocess_latency.json` is now that number, anchored at the last
band's pixel arrival (push g-2 issue + delay); `full`, `total_k1.00` and the other streaming
columns keep their back-to-back (fast-producer) values -- `total_k*` with the spacing inside
would be the link, not the pipeline. The fast-producer crit stays in
`detail.streaming_k1.00.ttft_last_band_ms`; the spaced run's decomposition in
`detail.streaming_k1.00_d150` (`last_band_correct_ms`, `last_chunk_wait_ms`,
`last_chunk_to_ft_ms`, `max_chunk_wait_ms`). Rows: `probe_qwen35_{moe,122b}_d150/`.

35B (n=36, medians ms; crit = last band correction + chunk wait + last prefill step):

| dataset | tok | full-res TTFT | crit back-to-back | crit spaced | ratio | L_3 | wait | last step |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| VisDrone Det | 1066 | 67.8 | 63.5 | 46.0 | 0.68 | 8.6 | 1.0 | 36.3 |
| VisDrone Count | 1036 | 69.3 | 68.9 | 46.3 | 0.67 | 8.8 | 1.1 | 36.3 |
| V* | 3341 | 186.5 | 101.0 | 75.9 | 0.41 | 34.5 | 2.3 | 39.0 |
| TextVQA | 799 | 62.7 | 64.9 | 45.9 | 0.73 | 7.1 | 0.9 | 37.9 |
| RefCOCO | 320 | 47.8 | 48.3 | 22.8 | 0.48 | 6.1 | 0.6 | 15.7 |
| RealWorldQA | 1351 | 79.3 | 83.3 | 50.8 | 0.64 | 12.5 | 1.2 | 37.2 |
| ChartQA | 466 | 52.0 | 53.7 | 47.3 | 0.91 | 7.0 | 0.9 | 39.3 |
| MMVP | 110 | 24.9 | 31.7 | 21.2 | 0.85 | 6.4 | 0.6 | 14.1 |
| CV-Bench | 366 | 59.5 | 52.2 | 45.8 | 0.77 | 7.6 | 1.3 | 36.9 |
| VSR | 346 | 50.3 | 48.8 | 23.1 | 0.46 | 6.7 | 0.8 | 15.5 |

Every chunk wait is ~1 ms (max 0.8-3.1), so the spaced numbers are the pipeline's own cost.

**The last prefill step is bimodal, and the threshold is the CUDA-graph capture limit.** The
server trace of the spaced RWQA / VisDrone / V* runs, engine step time by tokens computed in
the step: 50-128 tokens 15.0-15.8 ms; 128-850 tokens 35.3-39.2 ms, flat. The 35B server runs
with `--max-num-seqs 64`, and vLLM 0.28 derives `max_cudagraph_capture_size = 128` from it
(`cudagraph_capture_sizes` 1..128 in the server log): a step of <= 128 tokens replays a
piecewise CUDA graph, a larger one runs the eager Python/launch path, and the ~37 ms "MoE step
floor" the earlier sections attributed to the model is therefore mostly launch overhead of the
eager path (Qwen3.5's 40 hybrid layers x MoE routing + gated-delta-net ops), not GPU math --
the graph path does the same math in 15 ms. RefCOCO / MMVP / VSR land in the 15 ms mode
because their last chunk (band + short trailing text) is under 128 tokens; ChartQA's long
question pushes a 117-token band over it (39 ms, 0.91x). The 122B server (`--max-num-seqs
32`) captures only up to 64 tokens, so every chunk step there is eager (47-63 ms).
Hypothesis to test, not yet run: `--max-num-seqs 512` (or `compilation_config`
`max_cudagraph_capture_size`) so chunk-sized steps replay graphs -- if the 15 ms mode holds up
to ~400 tokens the 35B Crit. Lat. would drop to ~25 ms (0.35-0.45x) on the 1k-token datasets,
and the one-shot TTFT would fall less (its ~1k-token step stays eager unless capture goes past
it). Needs a go; it changes the engine config of every served number.

122B-FP8 (same run, `--max-num-seqs 32` -> capture <= 64 tokens, so every chunk step is eager):

| dataset | tok | full-res TTFT | crit back-to-back | crit spaced | ratio | L_3 | wait | last step |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| VisDrone Det | 1066 | 101.0 | 128.3 | 57.1 | 0.57 | 8.6 | 1.2 | 47.9 |
| VisDrone Count | 1036 | 80.2 | 125.7 | 58.3 | 0.73 | 8.6 | 1.2 | 48.0 |
| V* | 3341 | 304.9 | 215.9 | 102.0 | 0.33 | 34.6 | 3.3 | 64.1 |
| TextVQA | 799 | 78.9 | 84.4 | 60.1 | 0.76 | 7.4 | 1.3 | 51.0 |
| RefCOCO | 320 | 61.7 | 87.3 | 58.5 | 0.95 | 6.8 | 0.8 | 50.3 |
| RealWorldQA | 1351 | 121.7 | 144.0 | 64.7 | 0.53 | 12.9 | 1.7 | 50.1 |
| ChartQA | 466 | 64.9 | 82.5 | 60.3 | 0.93 | 6.9 | 1.2 | 51.8 |

The 122B step floor (48-52 ms eager, 64 at 830 tokens) sets the crit almost alone: the
tower's last band is 7-13 ms on every dataset but V*. On the small-image datasets (RefCOCO,
ChartQA) the one-shot prefill is itself one such step, so streaming cannot beat 0.93-0.95x
there; it wins where the one-shot tower + prefill are large (RWQA 0.53, VisDrone Det 0.57,
V* 0.33). Table regenerated (`latency_table_20260909.tex`, both trees) with these `k1.00`
cells; the (%) is the ratio to the fast-producer one-shot TTFT.

### CUDA-graph capture limit: the "MoE step floor" is half eager launch overhead (2026-09-09, go)

vLLM sizes the piecewise CUDA graphs it captures as `min(2 x max_num_seqs, 1024 on Blackwell)`:
128 at the 35B server's `--max-num-seqs 64`, 64 at the 122B's `--max-num-seqs 32`. Every chunk
prefill step above the largest captured size runs eager, and the earlier trace showed those steps
flat at 35-40 ms from 130 to 850 tokens while <=128-token steps replayed in 15 ms. New server flag
`--max-cudagraph-capture-size N` (-> `compilation_config.max_cudagraph_capture_size`; default
unchanged). 35B at 1024: 83 piecewise + 11 full-decode graphs, 1.24 GiB, captured in 9 s, server
listening 2.5 min after launch. Probe (`analysis/results/latency/probe_qwen35_moe_cg1024/`, json key
`qwen35_moe_cg1024`, n=32, d=150, ceiling + k=1.0):

| step tokens | cg128 median ms | cg1024 median ms |
|---|--:|--:|
| <=128 | 14.7 | 14.7 |
| 129-256 | 38.3 | 16.8 |
| 257-512 | 35.8 | 18.7 |
| 513-1024 | 40.3 | 23.3 |
| 1025-2048 | 47.5 | 37.0 (still eager) |

| dataset | Crit. Lat. cg128 -> cg1024 | last-band correction | chunk wait | last chunk -> FT | Full-res TTFT |
|---|--:|--:|--:|--:|--:|
| RealWorldQA | 50.8 -> 33.8 (0.64x -> 0.43x) | 12.5 / 12.7 | 1.2 / 1.3 | 37.2 -> 19.9 | 79.3 / 79.2 |
| VisDrone Det | 46.0 -> 29.3 (0.68x -> 0.43x) | 8.6 / 8.5 | 1.0 / 1.1 | 36.3 -> 19.3 | 67.8 / 67.9 |
| V*Bench | 75.9 -> 62.8 (0.41x -> 0.33x) | 34.5 / 34.3 | 2.3 / 2.6 | 39.0 -> 25.5 | 186.5 / 187.8 |

Only the engine span moved; correction, transfer and the one-shot ceiling (prompt > 1024 tokens,
still eager) are unchanged, so the whole 17 ms is the eager-launch overhead of a 27-layer MoE
step and the remaining ~20 ms is the prefill step (17-23) plus the first decode step (3.7). The
table keeps the cg128 numbers; re-measuring the table under 1024 (and 122B, where the 64 limit
leaves every chunk step eager) needs a go -- costs 1.2 GiB of graph memory and startup time only.

### keep<1 (Ours 50/25%) latency cells under the deferred pscore + 150 ms rule (2026-09-09, go)

Same probe, keep 0.50/0.25, two passes per model: d=0 (`total_k*`, Lat.) and d=150 (`k*`, Crit.
Lat.), `analysis/results/latency/probe_qwen35_{moe,122b}_klt1_d{0,150}/`. 35B (n=36; k=1.0 for
reference):

| dataset | Lat. k1 / k0.5 / k0.25 | Crit. k1 / k0.5 / k0.25 | last-band corr. k1 / k0.5 / k0.25 | last chunk->FT k1 / k0.5 / k0.25 |
|---|---|---|---|---|
| VisDrone Det | 121.5 / 166.5 / 168.4 | 46.0 / 45.7 / 44.9 | 8.6 / 7.3 / 7.3 | 36.3 / 36.8 / 36.3 |
| VisDrone Count | 125.4 / 163.3 / 169.8 | 46.3 / 46.8 / 46.4 | 8.8 / 7.7 / 7.1 | 36.3 / 37.7 / 37.5 |
| V*Bench | 388.9 / 607.7 / 580.9 | 75.9 / 67.0 / 56.9 | 34.5 / 23.1 / 13.7 | 39.0 / 41.1 / 40.4 |
| TextVQA | 113.8 / 158.4 / 149.7 | 45.9 / 48.8 / 51.3 | 7.1 / 7.6 / 7.8 | 37.9 / 39.9 / 42.3 |
| RefCOCO | 81.1 / 103.1 / 100.5 | 22.8 / 27.5 / 27.8 | 6.1 / 7.4 / 7.9 | 15.7 / 17.6 / 16.5 |
| RealWorldQA | 173.0 / 200.1 / 201.8 | 50.8 / 53.8 / 49.1 | 12.5 / 10.3 / 7.4 | 37.2 / 39.8 / 39.1 |
| ChartQA | 89.5 / 107.4 / 125.6 | 47.3 / 50.0 / 50.7 | 7.0 / 7.3 / 7.7 | 39.3 / 40.4 / 41.0 |
| MMVP | 61.1 / 61.3 / 63.1 | 21.2 / 24.1 / 24.9 | 6.4 / 7.1 / 6.9 | 14.1 / 15.4 / 15.3 |
| CV-Bench | 91.7 / 103.8 / 121.5 | 45.8 / 45.6 / 45.2 | 7.6 / 7.6 / 7.5 | 36.9 / 36.7 / 36.7 |
| VSR | 80.9 / 97.0 / 96.8 | 23.1 / 25.1 / 23.2 | 6.7 / 7.7 / 6.9 | 15.5 / 15.6 / 15.1 |

Reading: Crit. Lat. at keep<1 equals keep=1.0 within +-3 ms (the last band corrects fewer rows,
but its correction is already only 6-9 ms at ~250 tokens and the prefill step is the same size);
only V* (830-token bands) drops with keep (34 -> 23 -> 14 ms correction). Lat. is 20-45 ms
higher than keep=1.0 (V*: +190/+220) because the deferred received-attention score pass runs
after push 0 on the serialized path (`t_vision` 68 -> 100 on VisDrone) -- keep<1 buys FLOPs, not
first-token time, on this axis. The stale `stale_issue_anchor` entries stay in the json for the
record.

122B-FP8 (n=36, same probe; server listened 15.4 min after launch):

| dataset | Lat. k1 / k0.5 / k0.25 | Crit. k1 / k0.5 / k0.25 | last-band corr. k1 / k0.5 / k0.25 | last chunk->FT k1 / k0.5 / k0.25 |
|---|---|---|---|---|
| VisDrone Det | 192.2 / 213.9 / 213.0 | 57.1 / 60.3 / 58.3 | 8.6 / 7.3 / 7.6 | 47.9 / 50.0 / 48.5 |
| VisDrone Count | 192.6 / 215.5 / 214.6 | 58.3 / 58.7 / 59.0 | 8.6 / 8.3 / 7.8 | 48.0 / 48.8 / 49.2 |
| V*Bench | 531.9 / 760.2 / 717.7 | 102.0 / 92.7 / 80.4 | 34.6 / 23.8 / 14.4 | 64.1 / 64.6 / 63.2 |
| TextVQA | 135.0 / 196.5 / 191.8 | 60.1 / 57.9 / 58.9 | 7.4 / 7.5 / 7.0 | 51.0 / 48.9 / 49.7 |
| RefCOCO | 119.0 / 119.3 / 125.4 | 58.5 / 59.5 / 58.9 | 6.8 / 7.1 / 7.5 | 50.3 / 50.9 / 49.7 |
| RealWorldQA | 233.3 / 306.6 / 296.7 | 64.7 / 61.0 / 61.0 | 12.9 / 8.9 / 8.4 | 50.1 / 50.1 / 49.9 |
| ChartQA | 120.5 / 130.2 / 134.0 | 60.3 / 58.4 / 58.7 | 6.9 / 7.5 / 7.7 | 51.8 / 50.2 / 50.1 |

Same picture as 35B: Crit. Lat. is keep-independent (+-3 ms) except V* (102 -> 93 -> 80 from the
830-token band's correction), and on 122B the window is dominated by the ~50 ms eager chunk step
(capture limit 64 at `--max-num-seqs 32`), so the capture-limit lever above is worth more here.
Lat. carries the deferred score pass (+20-70 ms; V* +190-230). Table cells regenerated
(`analysis/results/latency_table_20260909.tex`), mirrored to AppCorr-qwen35-eval.
