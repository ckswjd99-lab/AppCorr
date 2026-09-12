# GLM-4.6V (106B-A12B, FP8) port for the interleaved table -- plan (2026-09-12)

## Why this model
User asked for GLM-4.7-Flash (30B-A3B). It is text-only (`Glm4MoeLiteForCausalLM`, no
vision_config) -- no image rows to stream or correct -- so it cannot sit in a table whose axis is
vision-token correction. User chose the GLM vision MoE sibling: `zai-org/GLM-4.6V-FP8`, 106B-A12B,
downloaded 2026-09-12 20:40 (110.0 GB, 41 shards, verified against the index). One B200, like the
Qwen3.5-122B-FP8 row. Interleaved-table row stub already in `make_eval_table.py` (`IL_MODELS`,
slug `_glm-4.6v-fp8`, suffix `_c2`, keys `glm46v_il` / `glm46v_cg1024` / `glm46v`).

## Tree
This directory (`AppCorr-glm46v`) is a PLAIN COPY of `AppCorr-il-unified` taken 20:50 (rsync,
no results, no logs, `.git` pointer removed). Reason: the il-unified tree is live -- the RefCOCO
shard chain re-imports its driver per arm until ~22:30 -- so nothing there may be edited tonight.
Merge back later by `diff -r`. Results written by this tree go to
`/NHNHOME/share/cjpark/AppCorr-il-engine/analysis/results/` like every other tree.

## Architecture facts (survey 2026-09-12, vLLM 0.28 `glm4_1v.py` + HF `glm4v_moe`)
Paths: `$V` = appcorr-vllm site-packages `vllm/model_executor/models`, `$T` = appcorr
transformers/models.

**Vision tower** `Glm4vVisionTransformer` (`$V/glm4_1v.py:608`): 24 identical blocks, **full
bidirectional attention within each image** (varlen `cu_seqlens` per frame; no windows, no
causal). hidden 1536, 12 heads x 128, MLP = **gated SwiGLU** `down(silu(gate(x))*up(x))` width 4096 (= out_hidden_size; `blocks.N.mlp.gate_proj [4096,1536]`). RMSNorm pre-norm,
fused-residual `norm2(x, residual=)` form (`:414-433`). Patch embed **Conv3d (2,14,14)**,
temporal_patch_size 2 (`:435-462`). Positions: learned abs `Embedding(576,1536)` on a 24x24 grid
**bicubic grid_sample-interpolated to the actual grid** (`:514-606`) PLUS 2D RoPE partial 0.5
(`:652-658`, `:708-745`). `post_conv_layernorm` after patch-embed, before pos-add; `post_layernorm`
after block 24 (`:682,692,930-947`). Downsample = **2x2 Conv2d stride 2, 1536->4096** on
`view(-1,2,2,C)` (`:686-691`, `:950-953`). Merger `Glm4vPatchMerger`: proj 4096->4096 -> LayerNorm
-> GELU -> SwiGLU(4096->10944->4096) (`:464-512`; context_dim 10944 in THIS checkpoint, HF default
13696). Row order after `rot_pos_emb` is merge-group-major (`reshape(h/2,2,w/2,2).permute`,
`:717-733`) = Qwen2-VL convention, so `_bands` row math carries over. `image_size: 336` only
sizes the abs-posemb table; real limits come from `preprocessor_config.json`
`size={shortest_edge:12544, longest_edge:9633792}` (pixel AREAS), `smart_resize(factor=28)`.
**Tower is un-quantized bf16** (FP8 `ignore` list covers all `visual.*`).

**Prompt / positions.** ids: `<|begin_of_image|>` 151339, `<|end_of_image|>` 151340,
`<|image|>` (placeholder) **151363** = `config.image_token_id`. Placeholders =
`grid_t*grid_h*grid_w // 4`; the two sentinels stay as ordinary text rows flanking the run
(`:1764-1768`). **M-RoPE 3D**, `mrope_section=[8,12,12]`, partial 0.5, theta 5e5; single-image
formula bit-identical to Qwen2.5-VL (`get_mrope_input_positions` `:2247-2280`; HF
`get_rope_index`), so `qwen_vl_axis._positions_fast` (`:281-316`) transfers verbatim with the
token id swapped. Mandatory prefix `[gMASK]<sop>` (151331, 151333) at position 0. Thinking is ON
by default; `enable_thinking=False` renders `/nothink` (151360) AND `<think></think>\n` after `<|assistant|>` -- verify against the rendered template.

**Text decoder** 46 x `Glm4MoeDecoderLayer` (`$V/glm4_moe.py:320`): **pure softmax GQA, no
DeltaNet / Mamba anywhere.** 96 q / 8 kv heads, head_dim 128 (explicit), `attention_bias: true`
(q/k/v biases, none on o), no qk-norm, partial rotary 0.5. Layer 0 dense (SwiGLU 10944), layers
1-45 MoE: 128 routed top-8 + 1 shared (width 1408), fp32 sigmoid router with
`e_score_correction_bias`, `norm_topk_prob`, `routed_scaling_factor 1.0`. No MTP layers.

**FP8** = compressed-tensors, weights per-CHANNEL static, activations per-token dynamic ->
`CompressedTensorsW8A8Fp8` (cutlass / `_scaled_mm`). **Not** the DeepGEMM block path; the
`TRANSFORMERS_DISABLE_DEEPGEMM_LINEAR=1` / Triton-finegrained workaround of the 122B row does not
apply. Vision QKV prefix is `attn.qkv_proj` under this quant config (`:300-307`).

**vLLM input path.** No `get_input_embeddings(ids, mm)`; model implements `embed_multimodal`
(`:2196-2215`) and `forward(input_ids, positions, intermediate_tensors, inputs_embeds)`. The
AppCorr `prompt_embeds` push bypasses both -> unaffected. `get_mrope_input_positions` has the Qwen
signature; `client._DuckFeature/_DuckItem` work unchanged. Module tree `model.visual` /
`model.language_model.model.layers`; `correct._decoder` (`correct.py:714-722`) walks it as-is.
Processor family is `Glm46VProcessor` / `Glm46VImageProcessor` (`$T/glm46v/`), NOT `Glm4vProcessor`
-- do not isinstance-check the latter. No `min_pixels/max_pixels` kwargs: low-res bases are made
by setting `size`.

## What is generic (reuse as-is)
wire/server/bridge; `runner_patch._update_states/_init_mrope_positions`; `correct.appcorr_rows_step`
(pseudo-sequence softmax path), `_block_row`, `_slot_mapping`, `stage_bounds/stage_spec`,
`_decoder`, `CONTIG_ROWS`; `_positions_fast`; band / merge-group row math; the `correct` op
semantics (`vllm_interleaved_design.md` 37-121); the accuracy driver, latency probe, table.

## What needs a GLM counterpart (the work)
1. **Tower stage split** -- new `appcorr/models/glm46v/vision/backbone.py`:
   `ApproxCorrectGlm4vVisionTower` mirroring `appcorr/models/qwen35/vision/backbone.py`: pre-stage
   (Conv3d patch-embed -> post_conv_layernorm -> bicubic abs-posemb add -> 2D-RoPE tables), 24 block
   stages (varlen full attention over the ONE image, fused-residual RMSNorm), post stage
   (`post_layernorm`). Unified axis: 24 (+pre, +merger) tower stages instead of 27.
2. **Merger** -- `post_layernorm -> view(-1,2,2,C) -> Conv2d downsample -> Glm4vPatchMerger`,
   applied per band. The Conv2d is per merge-group, so band-slicing is exact; RE-ASSERT that
   invariant with a bitwise test (merger(all rows)[band] == merger(band rows)).
3. **Axis class** -- `Glm46VAxis` in `appcorr/models/qwen_vl_axis.py` (or a sibling module)
   parameterizing: `image_token_id=151363`, sentinels 151339/151340 flanking the run (`lo-1`,
   `lo+n` are special tokens now), `_positions_fast` unchanged, prompt prefix `[gMASK]<sop>`,
   `enable_thinking=False`, low-res base via processor `size`, tower/merger from (1)-(2).
   Composer: `image_pad_id` parameterized (`client.py:342`).
4. **Engine gating** -- `runner_patch.py` tail: `correct.install()` currently imports
   `QwenGatedDeltaNetAttention` unconditionally (`correct.py:1153-1157`); make the GDN side-buffer
   machinery (`SideBuffer`, `_forward_core_patch`, `_rescan`, `_mamba_group_ids`, `_mamba_blocks`,
   `check_gdn_path` and its `assert gdn` at `:250`, `VLLM_GDN_DECODE_KERNEL`) conditional on the
   model having GDN layers. For GLM the layer-type table is a constant "all 46 softmax"; the
   DeltaNet re-scan window is a no-op.
5. **FLOPs closed form** -- `appcorr/models/qwen35/unified.py::_llm_stage_costs` sibling for GLM:
   drop the GDN term; layer 0 dense (10944), layers 1-45 MoE (top-8 x 1408 + shared 1408 + 128-way
   router); attention with q/k/v biases, head_dim 128, 96/8 heads, partial rotary; lm_head
   **151552** (`lm_head.weight [151552, 4096]`; the 154880 in the text-only GLM-4.7-Flash config does not apply). `_vision_stage_cost`: h 1536, heads 12, ffn 4096, + downsample conv + merger terms.
   `flops_report_qwen35.py`: a `--family glm46v` path (hooked run must count routed experts --
   check whether the FP8Experts hook fix generalizes to `CompressedTensors` MoE).
6. **Driver / probe / table plumbing** -- `qwen_vllm_accuracy.py` and `latency_probe.py`
   `--family glm46v`; `make_eval_table` row already stubbed; FLOPs json name
   `glm46v_flops_il.json`; latency keys `glm46v_il{,_staged,_unified}` and `glm46v_cg1024`.

## Gates (GPU needed; the box is busy until ~2026-09-13 00:30)
- **G0 stock sanity**: served GLM-4.6V-FP8 one-shot on 8 V*Bench images == HF greedy first token.
- **G1 tower split**: staged tower forward (24 stages) bitwise == stock tower forward; merger
  band-slicing bitwise.
- **G2 identity**: keep=1, g=4, approx=base-res, corrected=full-res -> first-token argmax equal to
  a stock one-shot request on 8 images, KV rel-L2 at image+text slots within the 122B-FP8 band.
  Softmax-only, so no GDN sub-gate.
- **G3 g=1 vs g=4** at k=1 within band. **G5 driver**: interleaved arm's pushed embeds == streaming
  arm's, bitwise.
- **FLOPs gate**: closed form vs hooked stock prefill within 1% at floor and ceiling N.
- **Server config** for the campaign: start from the 122B recipe (`--gpu-mem 0.85 --max-model-len
  8192 --max-num-seqs 512 --interleaved`); with no side buffers the OOM margin is larger, so
  `--max-model-len 16384` may fit -- measure, do not assume.

## Known traps carried over from today (see docs/memo/interleaved_table_notes.md)
one session per row; probe at `--samples 36 --warmup 4` (n=32 stored); `_vsr_images` symlink
needed in any new tree; verify every arm from the file; equivalence gates compare on `pred`.

## Corrections from implementation (2026-09-12 21:20, engine agent; code wins)
- `lm_head` vocab is 151552, not 154880. Vision block MLP is gated SwiGLU (width 4096), not
  2-matmul; vision `intermediate_size` 10944 is the MERGER context dim. `enable_thinking=False`
  renders `/nothink` and `<think></think>\n`.
- `Glm4vForConditionalGeneration` has no `embed_input_ids`; the table is
  `model.language_model.embed_input_ids`. Composer must resolve per model.
- No `q_proj` output gate on GLM (`q_proj [12288, 4096]`), so the attention projection term is
  half of Qwen3.5's.
- FLOPs hook point: `Glm4vMoeTextExperts` (transformers `glm4v_moe`), NOT `FP8Experts` -- the
  compressed-tensors quantizer never replaces the experts class, so the Qwen3.5 FP8Experts hook
  fires 0 times and silently under-counts. Handler `_qwen35_experts_flops` applies verbatim
  (same forward signature and attrs). Pinned by `tests/test_glm46v_flops_hooks.py`.
- Hooked FLOPs run needs `compressed-tensors>=0.15.0` in the HF env (absent from `appcorr` as of
  2026-09-12); loader must pass `attn_implementation="sdpa"` so the patched SDPA counts the
  quadratic term.
- GDN gating lives in `correct.gdn_modules(model)` (class-MRO check for `GatedDeltaNet`), so
  `install()` no longer imports `QwenGatedDeltaNetAttention`; Qwen3.5 behaviour unchanged
  (`tests/test_correct_gdn_gating.py` 9 passed, regression 36 passed, `--validate-qwen35` worst
  0.004%). With no side buffers `--max-num-seqs` is not capped by mamba blocks; the 122B memory
  margin does not transfer -- measure `--max-model-len 16384`.

## Gate results (2026-09-12, GPU0, B200-6)
- **G1 tower split**: fork vs stock bitwise; 24-stage staged forward vs reference bitwise;
  correct_rows vs stream bitwise; g=1 identity rel-L2 0.0 on all 8 images. Merger band slicing
  is NOT bitwise in bf16 (kernel choice by M), but against an fp32 reference the band path's max
  error equals the all-rows path's to 4 digits on every sample (e.g. 0.02553 vs 0.02553), mean
  within 1.3x, both under 1 ULP at the tensor's max magnitude -> no error beyond bf16 noise. PASS.
  (Two earlier criteria were wrong: a 5e-3 absolute tolerance, then per-element ULP relative to
  the element's own magnitude, which reads near-zero elements as "500 ULP".)
- **SG stock vs pushed embeds** (in-process, `vllm_stream_gate.py`): arm B (one pushed chunk)
  **exact 8/8, dlogprob 0.0** against the stock image request; prompt ids match HF on 8/8. Arm C
  (vLLM's own chunked prefill, 4 chunks) exact 4/8, first-token dlogprob up to 0.23 -- the FP8
  chunking effect; direction agrees with Qwen3.5-122B-FP8, the rate at n=8 says nothing.
- **G2 identity** (keep=1, g=4, approx=base-res, corrected=full-res), judged against the chunk
  control: first-token argmax vs one-shot 7/8 for BOTH; max |dlogprob| first 0.259 vs 0.234;
  KV rel-L2 max image 0.222 vs 0.207, text 0.193 vs 0.172. Approx-image rel-L2 0.46-0.72 (the
  low-res pass really differs; the correction does real work) and the corrected state lands in
  the chunk band. PASS.
- Composer bug fixed on the way: the gate's local `Glm46VComposer` set the image token after
  `super().__init__` validated the Qwen default; now uses `client.Glm46VComposer`.
- **FL hooked FLOPs**: the FP8 checkpoint cannot be loaded by transformers 5.13 -- its
  compressed-tensors path leaves the per-expert `{gate,up,down}_proj.weight(+_scale)` tensors
  UNEXPECTED and the fused `experts.gate_up_proj` / `down_proj` MISSING (not an OOM; the GPU
  attempt OOMed first only because the partial bf16 decompression already exceeded 178 GB).
  Measured instead from the bf16 sibling `zai-org/GLM-4.6V` (215 GB, 41 shards) on CPU with
  `--device cpu` (2 TB host RAM): same architecture and shapes, identical counts. Provenance of
  the FLOPs column is therefore "hooked on the bf16 twin"; the accuracy/latency rows are FP8.
- **Scorer trap found on the first campaign arms (22:37).** GLM wraps its final answer in
  `<|begin_of_box|>...<|end_of_box|>`. Every MCQ scorer takes the FIRST A-D letter of the
  upper-cased text, so the `B` of `<|BEGIN_OF_BOX|>` scored every V*Bench answer as "B": floor ==
  ceiling == streaming == interleaved == 36.13% (= the share of gold B). Caught by reading the
  jsonl (`pred '<|begin_of_box|>A<|end_of_box|>', gold 'A', ok 0`). Fix: `clean_text(family, text)`
  in the driver strips the two sentinels before `record` at both call sites; the seven finished
  V*Bench files were re-scored in place from their raw preds (backup in
  `_glm_vstar_prescore_20260912/`): ceiling 36.13 -> 81.68, floor -> 74.35, streaming k=1 -> 81.15,
  k0.50 -> 80.63, k0.25 -> 78.53, interleaved k=1 -> 80.63. Free-text scorers would have failed
  exact match on the same sentinels. The G0 gate was unaffected (it compares arms' strings to
  each other).
- **`--max-tokens 64` on every GLM arm (decided 23:30).** The driver default 24 made GLM emit a
  prose preamble then truncate mid-box on 29% of RefCOCO rows (ceiling 53.0 at 24 vs 89.0 at 64);
  a six-probe control on B200-8 (24/64/160 x2, same 200 rows) showed the format decision is a
  deterministic function of the budget (prose 28.5/29.0% at 24, 0.0% at 64 and 160; `ok`
  reproducing 200/200 within a budget; 64 vs 160 differing on 0/200) while coordinates drift ~15%
  between repeats at every budget (FP8). Set explicitly in the campaign chains, not as a driver
  default. The 24-token V*Bench files are parked in `_glm_maxtok24_invalid_20260912/`.
  Memory: `reference_glm46v_max_tokens_budget`.
