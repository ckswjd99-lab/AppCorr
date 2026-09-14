# GLM-5.3-Flash in vLLM main 658c813 -- code survey (Explore agent, 2026-09-13 00:55)

Root for all vLLM paths below: `V = /tmp/claude-3092/-NHNHOME-share-cjpark/a4974444-17dd-4d66-a897-140c17dbc4af/scratchpad/vllm-main` (HEAD 658c813).

## A. Module tree and forward path

| level | class | file:line |
|---|---|---|
| top (MM) | `Glm5NextForConditionalGeneration(Glm4vForConditionalGeneration, HasInnerState, IsHybrid, MixtureOfExperts)` | `$V/vllm/models/glm5next/nvidia/model.py:985` |
| `.visual` | `Glm5NextVisionTransformer` | `model.py:1035` |
| `.language_model` | `Glm5NextForCausalLM` (via `init_vllm_registered_model`, arch `"Glm5NextForCausalLM"`) | `model.py:1054`, class at `model.py:889` |
| `.language_model.model` | `Glm5NextModel` | `model.py:899`, class `model.py:578` |
| `.layers` | 45 × `Glm5NextDecoderLayer` | `model.py:630` (`make_layers`), class `model.py:278` |

Your `_decoder()` walk (`model → language_model → model → layers`) resolves correctly. Registry: `$V/vllm/model_executor/models/registry.py:123,429,693`.

Forward: the MM wrapper inherits `Glm4vForConditionalGeneration.forward` (`$V/vllm/model_executor/models/glm4_1v.py:2329-2358`), which calls `self.language_model.model(input_ids, positions, intermediate_tensors, inputs_embeds)`. `Glm5NextModel.forward` at `model.py:656-704`; layer loop at `model.py:685-688`:
```python
for layer in self._active_layers:
    hidden_states, residual, post, comb = layer(positions, hidden_states, residual, post, comb)
```
Note the **4-tuple return and 5-arg call** — not the `(hs, res)` your `_run_layers` (`correct.py:1070`) assumes.

`positions`: 1-D. `uses_mrope` is driven by `mrope_section` in `rope_parameters` (`$V/vllm/transformers_utils/config.py:595-626`); `Glm5NextTextConfig` has none, so the runner passes flat `[num_tokens]` and `get_mrope_input_positions` (inherited, `glm4_1v.py:2292`) is never invoked. KDA **ignores** `positions` entirely (`kda.py:285-289`, arg unused). MLA's `rotary_emb` is `None` (`attention.py:495-520`, `skip_rope=config.mla_nope`). But `positions` is **not dead**: the sparse indexer consumes it (`attention.py:400`, `indexer_op(..., positions=positions)`) for kpool tail-slot addressing (`pos % kpool`) and the short-prefill causal fill (`$V/vllm/model_executor/layers/sparse_attn_indexer_kpool.py:195-216, 421-445`).

MM splice: `embed_multimodal` inherited (`glm4_1v.py:2221`); scatter via `embed_input_ids` → `_merge_multimodal_embeddings` (`$V/vllm/model_executor/models/interfaces.py:500-535`).

Standard `GPUModelRunner` path throughout. `vllm/models/glm5next/nvidia/` holds only model/attention/kda/mtp/multimodal/ops — **no custom runner**; the only `v1/` reference is the MTP proposer name (`$V/vllm/v1/spec_decode/llm_base_proposer.py:996`). KDA metadata is `GDNAttentionMetadata`, read from `get_forward_context().attn_metadata[self.prefix]` (`kda.py:341-352`).

## B. KDA layers (34 of 45)

**The class is NOT in `kimi_gdn_linear_attn.py`.** That file's `KimiGatedDeltaNetAttention` (`$V/vllm/model_executor/layers/mamba/gdn/kimi_gdn_linear_attn.py:156`) belongs to Kimi-K3 and is never imported by glm5next. GLM-5.3 uses `Glm5NextLinearAttention(GatedDeltaNetAttention)` at `$V/vllm/models/glm5next/nvidia/kda.py:126`.

Entry: `forward(hidden_states, positions) -> Tensor` (`kda.py:285`); core at `_forward(qkv_proj_states, g1, beta, core_attn_out)` (`kda.py:333`), decorated `@eager_break_during_capture`.

Pre-conv projections (`kda.py:181-196, 292-314`), one merged GEMM `in_proj_qkvbfg_a`, per token:

| tensor | width (full) | TP | cached in side buffer? |
|---|---|---|---|
| q,k,v | 3 × 64·128 = 24576 | sharded | yes — this is the conv input |
| b (beta) | 64 | sharded | yes |
| f_a | 128 | replicated (shard 4) | yes |
| g_a | 128 | replicated (shard 5) | yes |

Then `g1 = f_b_proj(f_a) → [1,T,H,128]` (`kda.py:309`), `g2 = g_b_proj(g_a) → [T,H,128]` (output gate, applied post-core by `o_norm = FusedRMSNormGated(128, activation="sigmoid")`, `kda.py:255,329`).

Conv: three separate `q/k/v_conv1d` (`kda.py:211-231`, kernel 4, fp32 params), lazily concatenated into one merged weight `self._merged_conv_weight` (`kda.py:390-399`). Conv state is stored **merged q|k|v**, unlike Kimi which splits it.

State (`kda.py:140-153` → `MambaStateShapeCalculator.kda_state_shape`, `$V/vllm/model_executor/layers/mamba/mamba_utils.py:298-321`):
- conv: `(24576/tp, 4-1+num_spec)` (DS layout; SD is transposed — `kda.py:381`, `mamba_utils.py:48,172`)
- recurrent: `(64/tp, 128, 128)` fp32 (`mamba_utils.py:320`; dtype `kda_state_dtype`, `mamba_utils.py:133-149`)

`MambaBase.bind_kv_cache` (`$V/vllm/model_executor/layers/mamba/abstract.py:29-43`) slices one int8 page per block into `self.kv_cache = (conv_state[B,...], recurrent_state[B,H,D,D])`. Per-request addressing is by block id in `non_spec_state_indices_tensor` / `spec_state_indices_tensor` (`kda.py:355,362`), read with `gather_initial_states` (`kda.py:533`) and written with `scatter_states` (`kda.py:557`) — i.e. `state[indices] = src`.

Dispatch: prefill → `chunk_kda_with_fused_gate` (`kda.py:539`); decode/spec → `fused_recurrent_kda` (`kda.py:504, 571`). Both from `$V/vllm/models/glm5next/nvidia/ops/third_party/kda/kernels.py` (`__init__.py:4`), ROCm mirror under `amd/`.

Gate math (`kernels.py:873-951`, `SAFE_GATE=True` here):
```
gate_t = lower_bound * sigmoid( exp(A_log)[h] * (g1_t + dt_bias) )   # lower_bound = -5.0
beta_t = sigmoid(b_t);  q,k l2-normalized in kernel; q scaled by D**-0.5
S_t = S_{t-1} * exp(gate_t)[:,None,:]
u   = beta_t * (v_t - S_t @ k_t);  S_t += u ⊗ k_t;  o_t = S_t @ q_t
```
Exact fp32 reference: `$V/tests/models/glm5next/test_kda_recurrent.py:25-51`. `A_log` is `[1,1,H_local,1]` fp32 (`kda.py:243`, loader reshapes 1-D → 4-D at `kda.py:270-275`); `dt_bias` is `[H·D/tp]` fp32 (`kda.py:205`); `linear_lower_bound = -5.0` (`$V/vllm/transformers_utils/configs/glm5_next.py:64,199`).

**Yes, there is a Python-callable chunked kernel with `initial_state`/`final_state`** — the direct analogue of `chunk_gated_delta_rule`:
```python
chunk_kda_with_fused_gate(q, k, v, raw_g, beta, A_log, g_bias, scale=None,
                          initial_state=None, output_final_state=False,
                          use_qk_l2norm_in_kernel=False, cu_seqlens=None,
                          safe_gate=False, lower_bound=-5.0) -> (o, final_state)
```
`kernels.py:1199-1228`. q/k/v `[1,T,H,128]`, `raw_g` `[1,T,H,128]`, `beta` **pre-sigmoided fp32** `[1,T,H]` (kda.py:546 does `_cast_sigmoid`), `initial_state` `[num_seq,H,128,128]`. This is your `_rescan` primitive; note it takes `raw_g` + `A_log` + `g_bias` and does the gate internally, so no `fused_post_conv_prep` equivalent is needed (there is none for KDA).

## C. MLA / sparse attention (11 of 45)

`Glm5NextMLAAttention` (`$V/vllm/models/glm5next/nvidia/attention.py:404`), `forward(hidden_states, positions) -> self.mla_attn(positions, hidden_states)` (`attention.py:586-590`) where `mla_attn` is `MultiHeadLatentAttentionWrapper` (`$V/vllm/model_executor/layers/mla.py:44`, forward `163-253`).

KV cache: one latent per token, `MLAAttentionSpec(num_kv_heads=1, head_size=kv_lora_rank + qk_rope_head_dim, head_size_v=0)` → **512 + 0 = 512** for this config (`$V/vllm/v1/kv_cache_interface.py:646-660`). The rope pad path (`model.py:752-756`) is a no-op since `qk_rope_head_dim == 0`. The write happens inside `MLAAttention` from `forward_context` `slot_mapping` (the same `set_forward_context(..., slot_mapping=...)` your `_correct_sub` already builds, `correct.py:1198`).

Each sparse layer owns **three** caches, not one: the MLA latent, plus `indexer.k_cache` = `Glm5NextIndexerCache` (`attention.py:78`, uint8 fp8+scale, `head_dim + head_dim/128*4 = 132` bytes, `tokens_per_state = index_kpool`) and `indexer.tail_cache` = `Glm5NextTailCache` (`attention.py:159`, `KpoolTailSpec`, 1 block/request, bf16 raw K + gate score, overwritten at `pos % kpool`).

Indexer per query (`attention.py:316-401`): `q = wq_b(q_c)` → `[T,16,128]` (`index_n_heads=16`, see the `n_head < 32` zero-pad at `attention.py:387-390`); `k = layernorm(wk(h))` → `[T,128]`; `weights = h @ weights_projᵀ` in fp32; FWHT-128 + fp8 quant of q (`fwht128_quant_fp8`, `ops/kpool_compress.py`); `gate_score = F.linear(h, index_kpool_compress_gate)` → `[T,128]`. `index_kpool` consecutive K's are softmax-pooled into one stored entry, so **top-k runs at pool granularity** (`select_k = index_topk // index_kpool`; `glm5_next.py:70-74`). Output is `topk_indices_buffer` `[max_num_batched_tokens, ceil((topk+kpool-1)/128)*128]` int32, allocated once in `Glm5NextModel.__init__` (`model.py:588-606`) and shared by all 11 layers.

Prefill dense-vs-sparse: `short_prefill` when `max_prefill_seq_len <= topk_tokens` → skips sparse scoring entirely and fills exact causal indices (`sparse_attn_indexer_kpool.py:421-449`). Same for short decode (`_fill_short_decode_causal_indices`, `:202-216`). So at short prompt lengths the layer is effectively dense — good for a pseudo-sequence correct step.

Decode (1-token queries) path: tail cache is read to finish the boundary pool, paged-MQA fp8 logits over the pooled K cache, top-k → `topk_indices_buffer`, then sparse MLA absorbs the 512-latent. Per-request indexer state = the 1-block tail buffer; it must be rewritten too if you rewrite rows.

## D. Residual stream (mHC)

`mhc=True`, `mhc_num_residual_streams = 4` (`glm5_next.py:76-77`). Between layers you carry **four** tensors (`model.py:406-418, 685-688`):

| name | shape | dtype |
|---|---|---|
| `hidden_states` (`x`) | `[T, 4096]` (the MLP output, pre-`hc_post`) | bf16 |
| `residual` | `[T, 4, 4096]` | bf16 |
| `post` (post_layer_mix) | `[T, 4]` | fp32 |
| `comb` (comb_res_mix) | `[T, 4, 4]` | fp32 |

Shapes from `$V/vllm/model_executor/kernels/mhc/torch.py:64-78`. Layer 0 does `hc_expand(x, 4)` (replicate, `$V/vllm/model_executor/layers/mhc.py:901`); the last layer does `hc_post` then `hc_contract` (mean over dim 1, `mhc.py:906`) and returns `(x, None, None, None)` (`model.py:506-509`). Every intermediate layer **defers** its `hc_post` into the next layer's `hc_fused_post_pre` (`model.py:459, 485`) — so you cannot call a layer in isolation without threading `post`/`comb`. Per-layer weights: `hc_attn_fn`/`hc_ffn_fn` `[(2+n)·n=24, n·hidden=16384]` fp32, `hc_*_base` `[24]`, `hc_*_scale` `[3]` (`model.py:389-400`). Sinkhorn, 20 iters, `mhc_post_mult_value=2.0`, `hc_eps=1e-6`. The "forget gate" the peer memo names is this mix, not a separate module.

## E. MoE

`Glm5NextMoE` (`model.py:151`), `forward(hidden_states, already_sequence_parallel=False)` (`model.py:250`). Routed via `FusedMoEFactory` (`model.py:225-248`): `n_routed_experts=288`, `top_k=num_experts_per_token`, `moe_intermediate_size=2048`, grouped topk (`n_group`/`topk_group`), `scoring_func="sigmoid"`, `routed_scaling_factor=2.5`, `e_score_correction_bias` when `topk_method=="noaux_tc"`. Shared expert = `Glm5NextMLP(intermediate=2048·n_shared, reduce_results=False)` (`model.py:214-223`), passed *into* the factory so it is fused. Dense MLP: `Glm5NextMLP.forward(x)` (`model.py:144`), `SiluAndMulWithClamp` when `swiglu_limit` set.

`first_k_dense_replace` defaults to **0** in vLLM (`glm5_next.py:42`) and is only used to synthesize `mlp_layer_types` when that field is absent (`glm5_next.py:184-193`); layer selection reads `mlp_layer_types[layer_idx]` (`model.py:339-345`).

## F. Vision tower

`Glm5NextVisionTransformer` (`$V/vllm/models/glm5next/nvidia/multimodal.py:336`). depth 24, hidden 1024, 16 heads, patch 14, temporal_patch 2, merge 2, out 4096, `projection_intermediate_size=10240` (`glm5_next.py:274-324`).

- Patch embed: `Glm5NextVisionPatchEmbed` (`multimodal.py:48`), **no absolute pos-embed, no post-conv norm** (`multimodal.py:380`).
- Attention: **full varlen per frame** via `MMEncoderAttention` with `cu_seqlens = cumsum(h*w)` (`multimodal.py:164, 583-587`) — not windowed. Per-head `q_norm`/`k_norm` RMSNorm eps 1e-5 (`multimodal.py:142-143`), `attention_bias=True`.
- Position encoding: **2-D RoPE** over (h, w) ids, `get_rope(head_size=64, partial_rotary_factor=0.5, neox)` (`multimodal.py:390-395`), ids built in `rot_pos_emb` (`multimodal.py:443-479`). No learned absolute embedding.
- Adapter: `post_layernorm` → reshape to `[N,2,2,C]` → `downsample` Conv2d(1024→4096, k=2,s=2) → `Glm5NextPatchMerger` (`multimodal.py:601-606`, merger at `:284`).
- `rms_norm_eps` is **forced to 1e-6** regardless of checkpoint (`glm5_next.py:319`), and the tower is loaded with `quant_config=None` (BF16) (`model.py:1049`).

Token ids (`glm5_next.py:339-344`): image 154854, video 154855, image_start 154830 / end 154831, video_start 154832 / end 154833. Token count: `num_patches // merge_size**2` where `num_patches = grid_t·grid_h·grid_w` (`multimodal.py:712`, processor `$V/vllm/transformers_utils/processors/glm5next.py:895-905`). Prompt expansion is owned by **vLLM**, not HF: `_hf_processor_applies_updates` returns `False` (`multimodal.py:725-732`).

Text side needs **no** position tensor for image tokens: 1-D `positions` only, used solely by the indexer.

## G. Tensor parallel (TP=2)

| module | sharding | notes for a per-rank correct step |
|---|---|---|
| KDA `in_proj_qkvbfg_a` | q/k/v/b col-sharded; **f_a, g_a replicated** (`kda.py:181-196`) | side buffer must store the rank-local q/k/v/b but the full f_a/g_a |
| KDA `f_b_proj`, `g_b_proj`, `q/k/v_conv1d` | ColumnParallel | |
| KDA `dt_bias` | `sharded_weight_loader(0)` (`kda.py:209`) | |
| KDA `A_log` | `sharded_weight_loader(2)` (heads) (`kda.py:246`) | |
| KDA state | conv `(24576/2, 3)`, recurrent `(32,128,128)` — **sharded by head**, `is_kv_cache_tp_replicated=False` (`abstract.py:59-61`) | each rank owns a disjoint head slice; a re-scan is rank-local and needs no communication |
| KDA `o_proj` | RowParallel → **all-reduce** (`kda.py:256`) | unless SP, where `reduce_results` is cleared (`model.py:368-369`) |
| MLA `q_b_proj`/`kv_b_proj`/`o_proj` | head-sharded + all-reduce | |
| MLA latent cache | replicated (MQA latent, 1 kv head) | every rank writes the same 512-vec |
| Indexer | **fully replicated**: `wq_b` `ReplicatedLinear`, `wk_weights_proj` `disable_tp=True` (`attention.py:250-266`) | `k_cache` + `tail_cache` replicated per rank; a correct step must rewrite them on every rank identically |
| MoE | EP group (`model.py:166-206`) + optional SP | |
| mHC params | replicated fp32; SP shards by token (`model.py:473-482`) | in SP the layer gathers before attention and reduce-scatters after — an extra collective mid-layer |

Non-trivial for a correct step: (i) the deferred `post`/`comb` mHC state makes layer-by-layer walking stateful in 4 tensors; (ii) SP inserts `sp_all_gather`/`sp_reduce_scatter` *inside* the layer around attention (`model.py:473-482`), so per-rank row rewrites must be token-aligned to the SP shard; (iii) indexer tail cache is a circular 1-block buffer keyed on `pos % kpool` — rewriting rows p changes which pool they land in.

## H. Corrections to the peer memo

1. **"Engine support: BLOCKED in vLLM / no `Glm5*` entry"** — stale. vLLM main has a full native implementation (`$V/vllm/models/glm5next/`, registry `:123/:429/:693`), incl. MTP, AMD mirror, and tests.
2. **"`Glm5NextTextLinearAttention` … our side-buffer precedent is Qwen3.5"** — in vLLM the class is `Glm5NextLinearAttention` (`kda.py:126`), a `GatedDeltaNetAttention` subclass. `kimi_gdn_linear_attn.py` is **Kimi-K3's**, not GLM's, and its kernel signatures differ (`raw_beta`/`dt_bias`/`fused_recurrent_kda_packed_decode` vs GLM's `beta`/`g_bias`/`compute_gate`). Do not port against it.
3. **"the recurrent-state write … I would not assume it's expressible in the existing side-buffer interface"** — it largely is. `chunk_kda_with_fused_gate(initial_state=…, output_final_state=True)` (`kernels.py:1199`) is a drop-in for `chunk_gated_delta_rule`, and `gather_initial_states`/`scatter_states` give the same block addressing. The conv state is **merged q|k|v in one buffer** (`kda.py:388`), unlike Kimi's split — your `_conv_window` scratch must be shaped `(2, 24576/tp, 3)`.
4. **"no rotary in the decoder at all → positions carry nothing"** — imprecise. `positions` is still consumed by the sparse indexer (`attention.py:400`; `sparse_attn_indexer_kpool.py:195-216, 421-445`) for tail-pool slots and short-prefill causal fill. It is live, just not rotary.
5. **"The 11 that cache, cache latents (one 512-dim vector)"** — true for MLA, but incomplete: each sparse layer also owns an fp8 pooled indexer K cache and a paged tail cache (`attention.py:78, 159`). Three caches, not one.
6. **kpool omitted entirely.** `index_kpool` (default 4) means indexer storage and top-k are at **pool granularity**, with hard block-size constraints (`block_size % (index_kpool*32) == 0`, `attention.py:134-144`). This constrains `cache_config.block_size` and therefore any hand-built `CommonAttentionMetadata`.
7. **"first_k_dense_replace 3 → layers 0-2 dense"** — vLLM's default is 0 (`glm5_next.py:42`) and `mlp_layer_types` from the checkpoint wins (`glm5_next.py:184-193`, `model.py:339-345`). Read `mlp_layer_types`, not `first_k_dense_replace`.
8. **"288 routed / top-8"** — vLLM's default `num_experts_per_token` is **7** (`glm5_next.py:38`, aliased from `num_experts_per_tok`). Verify against the checkpoint.
9. **Indexer head count**: `index_n_heads` is **16** for this checkpoint (`attention.py:387` comment + zero-pad to 32; test uses `H=16`), not 64.
10. **"hyper-connections change the residual stream shape"** — right, but the concrete cost is that `layer(...)` is a 5-in/4-out call with deferred `post`/`comb`; your `_run_layers` 2-tuple contract (`correct.py:1070`) breaks outright.
11. **Vision `rms_norm_eps`**: checkpoint ships 1e-5, vLLM force-overrides to 1e-6 (`glm5_next.py:315-319`) and reads it from `vision_config`, not the mirrored top-level (`model.py:1038-1042`). Also `swiglu_limit` falls back to `text_config.swiglu_limit` (`multimodal.py:373-378`).
12. **Minor risk not in the memo**: `Glm5NextMLAAttention` builds `indexer_rope_emb = get_rope(qk_rope_head_dim=0, rope_parameters=config.rope_parameters, …)` unconditionally when `is_v32` (`attention.py:525-530`), even though `Indexer.forward` skips rope when `rope_dim == 0`. If the checkpoint ships `rope_parameters=None` this construction is a live failure point.

## Correction (2026-09-13 02:15 KST): the hook-point diff checked the wrong class
vLLM main has two runner classes: `vllm/v1/worker/gpu_model_runner.py` (V1, `self.requests`) and
`vllm/v1/worker/gpu/model_runner.py` (V2, `self.req_states`), and `gpu_worker.py` instantiates V2
by default ("Using V2 Model Runner"). The diff above compared signatures on V1 only. On V2,
`_update_states` / `_init_mrope_positions` / `_preprocess` and `self.requests` do not exist, so
the patches attach to a class the engine never runs; first symptom (B200-8, leg 2):
`'GPUModelRunner' object has no attribute 'requests'`. `VllmConfig.use_v2_model_runner` honours
an explicit `VLLM_USE_V2_MODEL_RUNNER` (HiSparse and watermarking force V2; not our path), so
`install()` now pins `VLLM_USE_V2_MODEL_RUNNER=0` on this build (`_pin_v1_model_runner`,
`tests/test_v1_runner_pin.py`; verified `VllmConfig().use_v2_model_runner` True -> False in the
main env). This is a deprecation clock, not a port: V2 is where main is going, and the three
absent methods are the real re-base surface. Lesson: "signatures match" is not "the patch attaches
to the running object" -- check the instantiated class, not the module.
