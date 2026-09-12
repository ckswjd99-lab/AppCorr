# GLM-5.3-Flash interleaved correct step -- design (2026-09-13 01:05)

User go (2026-09-13 00:40): "네가 말한 정밀한 KDA 관련 부분 재연산 설계를 실제 구현하라. 이후 평가도 쭉
돌리겠다." Six hours of B200-8 GPUs 2-3 for gates; idle GPUs go back to sionna.

READ FIRST: `docs/memo/glm53_vllm_survey.md` (the code map, with file:line into the vLLM main
clone at `$S/vllm-main`, HEAD 658c813 == the nightly wheel B200-8 serves) and
`docs/memo/glm53flash_port_plan.md` (HF-side survey; items its §H corrects are listed in the code
survey). Engine we extend: `appcorr/vllm_stream/correct.py` + `runner_patch.py` (Qwen3.5 GDN side
buffer + pseudo-sequence softmax path + CONTIG_ROWS + open_walk), `docs/memo/vllm_interleaved_design.md`.

## Facts that decide the design (from the survey)
- Decoder: 45 x `Glm5NextDecoderLayer`, called `layer(positions, hidden, residual, post, comb)`
  -> `(hidden, residual, post, comb)`; mHC with 4 residual streams (`residual [T,4,4096]`,
  `post [T,4]` fp32, `comb [T,4,4]` fp32), layer 0 expands, last layer contracts; every layer's
  `hc_post` is DEFERRED into the next layer. Our `_run_layers` 2-tuple contract breaks.
- 34 KDA layers `Glm5NextLinearAttention(GatedDeltaNetAttention)`: our MRO detector finds them.
  Seam: `_forward(qkv_proj_states, g1, beta, core_attn_out)` (no `_forward_core`). Per-token
  pre-conv projections to cache: merged q|k|v (24576/tp), b (64/tp), f_a (128, replicated),
  g_a (128, replicated). Conv kernel 4, MERGED q|k|v state `(24576/tp, 3)`. Recurrent state
  `(64/tp,128,128)` fp32, sharded by head -> a re-scan is rank-local. Prefill kernel
  `chunk_kda_with_fused_gate(q,k,v,raw_g,beta,A_log,g_bias, initial_state=, output_final_state=,
  use_qk_l2norm_in_kernel=, cu_seqlens=, safe_gate=, lower_bound=-5.0) -> (o, final_state)`;
  `beta` PRE-SIGMOIDED fp32 `[1,T,H]`; `raw_g` = g1 `[1,T,H,128]`; gate done in-kernel from
  A_log/g_bias(dt_bias). State addressing via `gather_initial_states` / `scatter_states`.
  Output gate `g2 = g_b_proj(g_a)` applied by `o_norm` (FusedRMSNormGated sigmoid) AFTER the core.
- 11 sparse-MLA layers `Glm5NextMLAAttention`: latent cache 512/token (no rope part), written
  through `forward_context.slot_mapping` like any attention layer -> the pseudo-sequence path's
  KV write applies. PLUS per layer an fp8 pooled indexer `k_cache` (kpool=4 granularity, 132 B per
  pooled entry) and a circular `tail_cache` (1 block/request, keyed `pos % kpool`). Rewriting rows
  must rewrite these too. Short prefill (`max_prefill_seq_len <= topk_tokens`) skips sparse
  scoring and fills exact causal indices -> effectively dense at short prompts.
- `positions` is 1-D, NOT used by KDA or MLA rotary (there is none), but IS used by the indexer.
  No M-RoPE. Image rows get plain sequential positions.
- Vision `Glm5NextVisionTransformer`: 24 blocks, hidden 1024, 16 heads, patch 14 (temporal 2),
  full varlen attention per frame, 2-D RoPE (head 64, partial 0.5, neox), per-head q/k RMSNorm
  eps 1e-5 (vLLM forces rms_norm_eps 1e-6 for the tower), NO abs posemb, NO post-conv norm;
  adapter `post_layernorm -> [N,2,2,C] -> Conv2d 1024->4096 k2s2 -> Glm5NextPatchMerger`
  (projection_intermediate 10240). Tower bf16 (`quant_config=None`). Token ids: image 154854,
  begin/end 154830/154831; count `grid_t*h*w // 4`; prompt expansion owned by vLLM.
- TP=2: KDA proj/state sharded by head (rank-local re-scan, `o_proj` all-reduce after); MLA
  latent replicated; indexer fully replicated (every rank rewrites identically); MoE EP; SP must
  be OFF for us (it inserts collectives inside the layer).
- MoE: read `mlp_layer_types` (not `first_k_dense_replace`); verify `num_experts_per_tok` (vLLM
  default 7, checkpoint says 8); `index_n_heads` 16.

## Design
Same contract as Qwen3.5 (`vllm_interleaved_design.md` §semantics): round r corrects rows P_r;
corrected rows write their state then read; non-corrected rows keep theirs.

1. **Layer walk with mHC state.** Generalise `_run_layers` to carry `(hidden, residual, post,
   comb)`; a per-row walk over layers [a,b) for a row set P needs those four tensors captured
   per row at the walk's entry layer -> the side buffer gains an mHC slot per layer boundary the
   engine can start a walk from (stage bounds only), not every layer. Layer 0 expand / last
   contract handled by the walk.
2. **KDA layers = the Qwen3.5 side-buffer design at a new seam.** Capture in `_forward` (per
   token, per layer): merged q|k|v post-projection PRE-conv, b, f_a, g_a. Re-scan of a window W
   from checkpoint S_start: conv over `SB[W]` left-padded with `SB[start-3:start]`, then
   `chunk_kda_with_fused_gate(..., initial_state=S_start, output_final_state=True)` ->
   outputs for W and S_end; copy rows in P into `core_attn_out`; the caller applies
   `o_norm(g2)` and `o_proj` as the stock path does. Checkpoints S_r per band from the re-scans
   themselves (as before). Final round: `scatter_states` the final S and conv tail into the
   request's state block so decode continues from corrected state. Beta (CORRECTED by agent K,
   code wins): `chunk_kda_with_fused_gate` takes beta ALREADY SIGMOIDED in fp32 -- the stock call
   site does `_cast_sigmoid(beta)` (`kda.py:546`) and the kernel body never sigmoids
   (`kernels.py:1119-1162`); `fused_recurrent_kda` (decode) takes RAW bf16 b with
   `sigmoid_beta=True`. Encoded as `_kda_beta()` plus an fp32/shape assertion and a test.
   Rank-local under TP (state sharded by head; f_a/g_a replicated).
3. **MLA layers = pseudo-sequence path + indexer rewrite.** KV write of the 512-latent per
   corrected row via `slot_mapping` (existing). NEW: for each corrected row p, recompute the
   indexer's k (layernorm(wk(h))) and gate score, rewrite the pooled `k_cache` entry for pool
   `p // kpool` (re-pool the 4 rows of that pool from their current hidden states -- so the side
   buffer must keep the indexer inputs for the 3 sibling rows), and the `tail_cache` slot
   `p % kpool` if p is in the tail pool. Under TP identical on every rank. Prefer the short-prefill
   dense regime for the gate (prompt <= topk_tokens); measure where `topk_tokens` lands.
4. **Vision**: `ApproxCorrectGlm5NextVisionTower` from the glm46v tower code: drop abs-posemb and
   post-conv norm, add per-head q/k RMSNorm, 2-D RoPE partial 0.5 neox, adapter as above.
5. **Axis/composer/driver**: `Glm53Axis` from `Glm46VAxis`: image id 154854, sentinels
   154830/154831, positions 1-D sequential (no mrope), `<|begin_of_box|>` stripping if the
   template uses it (check the chat template), `--family glm53`, table row stub
   (`glm53_il` keys, slug from the driver).
6. **TP=2 serving of our stream server**: our server runs the engine in-process
   (`VLLM_ENABLE_V1_MULTIPROCESSING=0`); with TP>1 vLLM spawns worker processes and the runner
   patch must be installed in EACH worker (plugin path), the `correct` op must be dispatched to
   all ranks (collective RPC), and side buffers live per rank. Determine the minimal change and
   gate it FIRST with the pushed-embeds arm (no correction) before any of 1-3.

## Gates (GPUs 2-3 on B200-8, this window)
G-TP0 stock TP=2 one-shot decode (B200-8, running). G-TP1 our server TP=2, pushed embeds vs
stock: exact 8/8 (arm B). G-V tower split bitwise vs stock tower (CPU/GPU). G-KDA re-scan from
checkpoint == full scan on the window (fp32 CPU reference from `tests/models/glm5next/
test_kda_recurrent.py` semantics; then bf16 GPU against the stock kernel). G-MLA rows step vs
chunk control (as G2 on GLM-4.6V, plus indexer caches compared bytewise). G2/G3 identity as before.

## Split (three agents, CPU only tonight; gates run by B200-8)
K: items 1-2 (+ unit tests). M: item 3 (+ tests with fake caches). V: items 4-6 (+ the TP plan
written down with exact code points). Nobody edits another agent's files; the engine-side
contract between K and M is `correct.py`'s existing function boundaries -- K owns the GDN/KDA
functions and `_run_layers`, M owns `_correct_sub`'s MLA/indexer part and adds
`appcorr/vllm_stream/glm53_indexer.py`.

## Corrections from implementation (agent M, 2026-09-13 01:35; code wins)
- **The real workload is the SPARSE regime.** Checkpoint `index_topk = 2048`, so prompts above
  2048 positions (V*Bench ~4k, RefCOCO/InfoVQA up to 8192) score sparsely; the dense exact-causal
  path is auto-selected only below 2048. Sparse path is the critical item (in progress).
- `CommonAttentionMetadata` on vLLM main has no `_seq_lens_cpu` / `_num_computed_tokens_cpu`;
  `_correct_sub` would TypeError on main for EVERY model. Shim `_cm_compat` added -- applies to any
  future move of the Qwen/GLM-4.6V ports onto main.
- `KpoolTailSpec` cache groups have `max_num_blocks_per_req == 1`; the generic slot mapping is
  out of bounds for pos >= 4; dedicated `_kpool_tail_slot_mapping`.
- Indexer seam is `indexer_op` (per-instance `_forward_method` swap on the `CustomOp`), not
  `Indexer.forward`: captures the exact k / gate tensors that reach the caches. Side buffer 513 B
  per token per sparse layer (x11 = 5.6 KB/token; 44 MiB at 8192) and must hold EVERY row (pools
  are re-formed per round). `skip_k_cache_insert=True` during the correct step; complete pools
  re-pooled from the buffer; incomplete trailing pool never written; tail re-seeded from the last
  kpool rows; replicated identically on every TP rank.
- Config facts: `index_n_heads` 32 (not 16), `num_experts_per_tok` 8, `first_k_dense_replace`
  3 with `mlp_layer_types` shipped (dense 0-2), `indexer_rope_interleave` dead (rope_dim 0),
  `rope_parameters` present so the survey's item-12 failure point does not fire.
- CPU tests 9/9: pool entry bytewise vs an independent Sylvester-matrix reference incl. fp8 bytes
  and ue8m0 scales; rewrite vs from-scratch bytewise; causal top-k fill bytewise vs the stock
  filler; slot mapping vs the kpool kernel's formula. Triton-kernel bitwise equality is GPU-only.

## Corrections from implementation (agent V -- items 4/5/6, 2026-09-13 02:10; code wins)

Checked against the checkpoint (`.../snapshots/eb9eb20.../`), transformers 5.16.1
`models/glm5_next/`, and vLLM main 658c813.

- **Two epsilons in the tower, not one.** §Facts says "per-head q/k RMSNorm eps 1e-5 (vLLM forces
  rms_norm_eps 1e-6 for the tower)", which is right but reads as if one of them wins. Both are
  live at once and they disagree: `norm1`/`norm2`/`post_layernorm` run at **1e-6** (forced,
  `vllm/transformers_utils/configs/glm5_next.py:315-319`, over the checkpoint's 1e-5) and
  `q_norm`/`k_norm` at **1e-5** (hard-coded, `multimodal.py:142-143`, i.e. NOT following the
  override). An HF `Glm5NextVisionModel` built straight from the checkpoint config is 1e-5
  everywhere and is therefore NOT what vLLM serves. `load_stock_vision_tower(..., vllm_eps=True)`
  is the default and retunes it.
- **The checkpoint ships a FUSED `attn.qkv`.** vLLM's `hf_to_vllm_mapper` remaps
  `.attn.q/.k/.v` onto a stacked `qkv` (`multimodal.py:338-346`), which reads as if the
  checkpoint were split. It is not: the index has `model.visual.blocks.N.attn.qkv.{weight,bias}`
  and no `.attn.q.*`, so the mapper is a no-op here and `_qkv_heads`' Qwen2-VL layout applies
  unchanged. `attention_bias: true`, so qkv/proj/mlp all carry biases (GLM-4.6V's do not).
- **All 347 `model.visual.*` tensors are in shard 62 of 62**, bf16, none FP8-quantised. The tower
  load opens one file.
- **The vision MLP's width is `vision_config.intermediate_size` (4096), not `out_hidden_size`.**
  On GLM-4.6V those were aliased and the FLOP closed form read `out_hidden_size`; GLM-5.3 gives
  the merger its own `projection_intermediate_size` (10240), so copying that line over would
  misprice every vision stage by 4096/10240.
- **"positions 1-D sequential (no mrope)" understates what the engine needs.** The right value to
  PUSH is `mrope=None`, not a broadcast 1-D tensor: `_init_mrope_positions` is called only under
  `if self.uses_mrope` (`vllm/v1/worker/gpu_model_runner.py:1343-1345, 1676-1677`) and is the
  only filler of `CachedRequestState.mrope_positions`, which `runner_patch.py:45-48` asserts
  non-None before extending. A tensor would assert-fail on the first appended chunk.
  `QwenVLStreamingAxis` now carries `uses_mrope` (True by default) and the three push /
  `position_ids=` sites honour it. On the HF backend `position_ids=None` is also the right
  value: `Glm5NextTextModel.forward` builds `arange(T) + past_seen` (5.16.1 :1450-1453) and the
  model class has **no `get_rope_index`**, so `positions_mode="reference"`/`"check"` do not exist
  for this family.
- **"`<|begin_of_box|>` stripping if the template uses it"** -- the template does NOT use it
  (0 hits in `chat_template.jinja`), but `<|begin_of_box|>` 154852 / `<|end_of_box|>` 154853 are
  still added tokens of the tokenizer, i.e. the trained behaviour is reachable. `clean_text`
  strips them for `glm53` as for `glm46v`; stripping is a no-op when absent and the failure it
  prevents (the 'B' of `<|BEGIN_OF_BOX|>` scoring every MCQ as B) cost a whole V*Bench arm once.
- **There is no thinking switch at all.** The template has neither `enable_thinking` nor
  `/nothink` (GLM-4.6V had both); its generation prompt ends `<|assistant|><think>`
  unconditionally, and `reasoning_effort` is forced to one of low/high/max with the system line
  always emitted. Non-thinking is therefore OURS to make: append `</think>` (154842).
  `Glm53Axis.build_inputs` does it on the ids, `Glm53Composer.prompt_text` on the text, and a
  gate asserts the two agree.
- **The low-res knob is `max_image_tokens`, not `size`.** `Glm5NextImageProcessor.size` is
  `{"longest_edge": 1}` with a `# TODO` saying it is unused (5.16.1
  `image_processing_glm5_next.py:127`); the budget is `min_image_tokens`/`max_image_tokens`, which
  `smart_resize` turns into `tokens * temporal_patch_size * (patch*merge*expand)**2` pixels and
  compares against `aligned_frames(=2 for a still) * H * W`. Spatial area cap =
  `8000 * 28**2 = 6,272,000` px -> `FAMILY_MAX_PX["glm53"]`.
- **`AutoProcessor` DOES resolve this checkpoint** on transformers 5.16.1 (`Glm5NextProcessor` +
  `Glm5NextImageProcessor`), contrary to what vLLM's `Glm5NextProcessingInfo` docstring implies
  (`multimodal.py:610-625` -- that is about vLLM's own resolution path). It also EXPANDS the
  single `<|image|>` placeholder to `t*h*w/4` tokens, so the HF-side axis needs no prompt-update
  machinery of its own; only the vLLM side owns expansion
  (`_hf_processor_applies_updates -> False`, `multimodal.py:725-732`).
- **§Facts "`index_n_heads` 16"** -- the checkpoint says **32** (agent M found the same
  independently). `num_experts_per_tok` is 8 and `first_k_dense_replace` is 3 with
  `mlp_layer_types` shipped, as §H predicted.
- **TP: "SP must be OFF for us" needs no action at TP=2.** `use_sequence_parallel_moe` also
  requires `data_parallel_size > 1` (`vllm/config/parallel.py:714-730`), which the B200-8 TP=2 /
  DP=1 configuration does not have. Assert it rather than disable it.
- **TP: "the runner patch must be installed in EACH worker (plugin path)"** -- the plugin path
  needs the package INSTALLED (`bin/pip` is broken in the served env). Use
  `--worker-extension-cls` instead: it is resolved by qualified name inside `init_worker`
  (`vllm/v1/worker/worker_base.py:285-310`), so our module is imported in every worker with no
  packaging, and it is also where the `collective_rpc` targets belong. Do NOT rely on `fork`
  inheriting the patch: `_maybe_force_spawn` overrides the start method whenever CUDA is already
  initialised (`vllm/utils/system_utils.py:125-164`).
- **TP: the pushed embeddings need no serialisation work.** `scheduler.py`'s docstring guesses
  that "a multi-process core would need them as real (msgpack) fields"; the broadcast queue is
  **pickle protocol 5** with a torch CPU-tensor reducer, not msgspec
  (`vllm/distributed/device_communicators/shm_broadcast.py:823-855, 421-449`), and
  `SchedulerOutput`/`NewRequestData` are plain dataclasses -- so `appcorr_stream` and
  `appcorr_stream_updates` cross to every worker as they are. Full details in
  `docs/memo/glm53_tp_plan.md`.
- **Blocker for any gate in the served env**: `appcorr.vllm_stream.install()` refuses any vllm
  outside `SUPPORTED_VLLM = ("0.11.2", "0.28.0")` and `appcorr-vllm-main` reports
  `0.1.1.dev65+g658c8131c`. Widen it (or add an explicit override) before running
  `glm53_tp_gate.sh`; the script stops on this in its preflight.

## Sparse regime resolved (agent M, 02:05; code wins)
- `index_kpool_always_select_tail = True`: the in-progress (boundary) pool is NEVER scored, it is
  appended by raw token id (`expand_pools_and_append_tail`, `ops/kpool_compress.py:820-892`);
  only complete pools `j < (p+1)//kpool` are scored from `k_cache`. `tail_kv_cache` is not read by
  the scoring path at all (only written, and only when not `skip_k_cache_insert`). So suppressing
  the stock write during the correct step blinds nothing; the tail is re-seeded per round for the
  decode steps that follow.
- Invariant (tested): for a corrected row p, candidates(p) subset of [0, p+1); pool p//kpool is
  scored only when p % kpool == kpool-1 (then it is exactly {p-3..p}); sub-batching order within a
  round is irrelevant (a straddling pool is never read by a lower sub-batch's queries).
- `get_paged_mqa_logits_metadata(context_lens, block_size, num_sms)` sees no block-table contents
  or request state; P one-token pseudo-sequences sharing one block row are P decode requests.
  Backend on sm_100 at TP=2: FLASHMLA_SPARSE then FLASHINFER_MLA_SPARSE, both reading only
  `query_start_loc/seq_lens/block_table/slot_mapping`. No new `CommonAttentionMetadata` field.
- Sparse branch = coverage assertion + stock `indexer_op` with `skip_k_cache_insert=True`.
  17 CPU tests pass (expansion vs stock two-step bytewise; causality at 2047/2048/4095/8191;
  round vs stock decode identical; sub-batch straddling reads 0; pseudo-seq metadata by hand).
- GPU-only: Triton pooling kernel bitwise; the sparse scoring kernels actually running with P
  rows; `k_cache` vs chunked control bytewise (the gate's core comparison). Gate:
  `analysis/experiments/glm53_mla_gate.py --gate mla --regime {dense,sparse} [--capture-topk]`,
  and it must run under the MAIN env (`/NHNHOME/share/cjpark/backup/env/appcorr-vllm-main/bin/python3.11`),
  not the 0.28 env M's command line names.

## Serving + scoring facts from the stock TP=2 proof (B200-8, 01:50)
- Stock vLLM main, TP=2, eager, `gpu_memory_utilization 0.96`: load 524 s, weights 153.07 GiB
  per rank, **13.29 GiB KV per rank** -- the budget every side buffer must fit inside. 0.85 is
  BELOW the weight footprint; cudagraph profiling OOMs; eager required.
- `/tmp` is noexec on BOTH boxes; FlashInfer's JIT `dlopen` fails there. Set `TMPDIR`,
  `FLASHINFER_WORKSPACE_BASE`, `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR` to an exec fs
  (e.g. under `/home/nxclab`) for every GLM-5.3 run.
- The model answers V*Bench in thinking prose by default; the MCQ first-letter scorer awarded
  the point for the A in "ASKING". The GLM-4.6V rows already show the same bias (24 prose rows
  across V*Bench arms, mis-scored both ways). An anchored extractor with a no-answer outcome is
  being built on B200-8 with a zero-change regression gate over all Qwen MCQ rows; GLM-5.3 arms
  are not scored with the old rule, and `enable_thinking=False` + a budget re-measured for this
  model are prerequisites for any accuracy number.

## Agent K landed (04:20; code wins) -- and the gate gap it exposes
- Layer walk classified by `inspect.signature(cls.forward)` ("mhc4" iff `post`/`comb`), 4-tuple
  frontier stored at stage bounds only; res2 path byte-identical.
- KDA seam `_kda_forward_patch` on `Glm5NextLinearAttention._forward` via `_gdn_flavor`
  ("qwen" defines `_forward_core`, "kda" defines `_forward`). Stored per token: merged q|k|v
  (pre-conv), raw b, g1 -- not f_a/g_a (unreachable at the eager seam). **65664 B/token/layer at
  TP=1 (2.13 MiB/token over 34 layers); 32832 B at TP=2 (1.06 MiB/token/rank)** -> an 8192-token
  prompt needs 8.7 GiB per rank OUTSIDE vLLM's budget; a 4k V*Bench prompt 4.3 GiB. Serve the
  interleaved form at `--gpu-mem ~0.90` (KV per token is tiny: MLA 512x2Bx11 + indexer 132Bx11/4;
  KDA state is per request, ~71 MB/rank), not 0.96.
- Re-scan on `chunk_kda_with_fused_gate` (chunk indices derived from `cu_seqlens`); write-back
  via `scatter_states` for recurrent state AND merged conv tail. `SUPPORTED_VLLM` widened to the
  exact nightly string. Whole `tests/`: 135 passed, 8 skipped, 0 failed.
- **Gate gap:** K's and M's GPU gates build an in-process `StreamingLLM` and read `llm.runner`,
  which raises at TP>1; GLM-5.3 (328 GB) cannot load at TP=1 on a B200, and V's TP gate leg 1
  (TP=1) cannot either. So tonight only the TP=2 STREAMING legs (arms A/B/C across the process
  boundary) can run. The correct-step gates need plan §(a)+(b) implemented: worker-extension
  install of the patches in each rank, collective-RPC dispatch of open/append/correct (and of
  the gates' per-rank comparisons) with per-rank side buffers. Assigned to agent V next.

## Item 6 landed: TP>1 dispatch (agent V, 2026-09-13 02:50)

§Design item 6 said "determine the minimal change and gate it FIRST with the pushed-embeds arm".
The determination is `docs/memo/glm53_tp_plan.md`; the change is now written (CPU-gated, no GPU
run yet) because K's and M's gates cannot run at all without it -- both reach the in-process
`driver_worker.worker.model_runner`, and GLM-5.3-Flash does not fit on one B200.

* `appcorr/vllm_stream/tp_worker.py` (new): `AppcorrWorkerExtension` (`--worker-extension-cls`;
  importing it runs `install()` inside each worker), `OpDispatchMixin` (`_dispatch` /
  `_dispatch_worker` / `run_on_ranks` / `worker_info`, mixed into `StreamingLLM`), `LLMRanks`
  (the same dispatch around a plain `vllm.LLM`, for the two gates), `agree_across_ranks`.
* Two defects that are invisible at TP=1 and fatal above it, both fixed: the side buffers were
  freed in the engine-core process and so leaked in every worker; and the hold-back-release
  chunk carried `mrope_positions=torch.empty((3, 0))`, which is "not None" and would have
  tripped `runner_patch`'s assert on the LAST correct round of every GLM-5.3 request.
* §Design's "SP must be OFF for us" is now asserted rather than assumed --
  `worker_info()["sequence_parallel_moe"]`, checked by `glm53_mla_gate.py` before it runs.
* Which quantities the ranks must agree on is now explicit in code, because getting it backwards
  gives a false alarm either way: **replicated** (assert equality) = the MLA latent and both
  indexer caches; **sharded by head** (expect difference) = everything KDA. `agree_across_ranks`
  says so in its docstring and M's gate asserts only the first kind.
