<!-- Survey by the B200-8 session, 2026-09-12; copied verbatim from /NHNHOME/share/cjpark/b200-8_logs/glm53flash_port_plan.md by the B200-6 session. -->
# GLM-5.3-Flash port survey (2026-09-12, B200-8)

Survey only, no implementation. Not written into the AppCorr tree -- see "Where this file lives".


## The three blockers, answered up front

**1. Engine support: BLOCKED in vLLM. NOT blocked in transformers.**

| | our env | verdict |
|---|---|---|
| vLLM 0.28.0 registry | no `Glm5*` entry of any kind | **blocked** |
| vLLM `model_executor/models/` | no `glm5*.py` (glm4, glm4_1v, glm4_moe, glm4v, glm_ocr, glmasr only) | **blocked** |
| transformers **5.16.1** | `models/glm5_next/` complete, `Glm5NextForConditionalGeneration` at `modeling_glm5_next.py:2063` | **available today** |

Correction to the addendum: it states "transformers 5.13 has no `models/glm5*` either."
The appcorr-vllm env has **transformers 5.16.1**, and it does have `models/glm5_next` --
config, modeling, modular, processing, image_processing (+ a PIL variant), video_processing.
The three `glm5` hits inside vLLM are false positives: `glm52_low_latency_gemm.py` and friends
under `models/deepseek_v32/nvidia/`, a GEMM kernel, unrelated to this model.

This matters for sequencing. The blocker is the **engine only**. The HF path -- architecture
verification, processor behaviour, image token layout, and the hooked-FLOPs run -- is open now
and needs no vLLM change and no GPU.

**2. Decoder: two new paths at once, and neither is the MLA the addendum expected.**

`layer_types` (45 entries) is exactly two kinds, period 4:

    34 x linear_attention           layers 0,1,2, 4,5,6, 8,9,10, ... 44
    11 x deepseek_sparse_attention  layers 3,7,11,15,19,23,27,31,35,39,43
     0 x full attention

    linear_attn_config: num_heads 64, head_dim 128, short_conv_kernel_size 4,
                        gate_lower_bound -5.0, kda_layers [the 34], full_attn_layers [the 11]

`kda_layers` names it: **KDA, Kimi Delta Attention** -- a gated delta rule with a short causal
conv, not Qwen3.5's DeltaNet and not Mamba. `Glm5NextTextLinearAttention` (modeling:584) carries
`A_log`, `dt_bias`, `f_a/f_b_proj`, `g_a/g_b_proj`, `q/k/v_conv1d` -- a different recurrence with
a different state shape from anything we have a side-buffer for.

The 11 attention layers are **NoPE MLA**: `kv_lora_rank 512`, `q_lora_rank 1536`,
`qk_nope_head_dim 256`, `v_head_dim 256`, `qk_rope_head_dim 0`, `mla_use_nope true`.
`qk_rope_head_dim = 0` with `mla_use_nope` means **no rotary in the decoder at all** -- position
information reaches the attention layers only through the KDA layers' recurrence. That is a
bigger departure than "MLA like DeepSeek": there is no M-RoPE for us to extend to image
positions, because there is no RoPE.

Plus `Glm5NextTextIndexer` (modeling:736) -- the DeepSeek-V3.2 sparse-attention index -- and
`num_nextn_predict_layers 1` (one MTP layer, ignorable for us), and hyper-connections
(`Glm5NextTextHyperConnection/HyperHead/ForgetGate`), which change the residual stream shape.

MoE: 288 routed experts, top-8, 1 shared, `moe_intermediate_size` 2048,
`first_k_dense_replace 3` -> layers 0-2 dense, 3-44 sparse (`mlp_layer_types`: 3 dense / 42 sparse).

**3. Size: 328.3 GB FP8 across 62 shards. One B200 is 178 GB. It does not fit, at any KV budget.**

    quantization_config: quant_method fp8, fmt e4m3, activation_scheme dynamic,
                         modules_to_not_convert: 1509 entries
    disk: 41,020 GB free of 60,000 GB on the HF cache volume (32% used) -- storage is a non-issue

Note the FP8 format: `quant_method: fp8` with an explicit 1509-entry keep-list, **not**
compressed-tensors. Tonight's "HF 5.13 will not load compressed-tensors FP8" lesson does not
apply to this checkpoint. Whether HF 5.16 loads *this* fp8 form is a separate question I have
not tested and cannot test until the weights land.

TP=2 across GPU0+GPU1 holds 328 GB of weights in 356 GB with ~28 GB left for KV, activations and
the vision tower -- before our interleaved side buffers, which is where a correct-step port puts
its extra memory. Our stream server is single-process single-GPU; nothing in `runner_patch.py`,
`correct.py` or the streaming scheduler has run under TP, so TP is itself an unvalidated path.
The BF16 sibling (~650 GB) is irrelevant on this hardware for serving; its only use is the
hooked-FLOPs run on CPU, which is also the only thing that does not need vLLM at all.

## Architecture facts (survey 2026-09-12, HF `glm5_next` + repo config.json)

    architectures      Glm5NextForConditionalGeneration      model_type  glm5_next
    layers 45   hidden 4096   heads 64   kv heads 64   vocab 154,880
    max_position_embeddings 1,048,576 (1M)

    decoder    34 KDA linear-attention + 11 NoPE-MLA sparse-attention, period 4
    MLA        kv_lora_rank 512, q_lora_rank 1536, qk_nope 256, v_head 256, qk_rope 0
    MoE        288 routed / top-8 / 1 shared, moe_inter 2048, dense layers 0-2
    MTP        num_nextn_predict_layers 1

    vision     glm5_next_vision, depth 24, hidden 1024, out_hidden 4096,
               patch 14, spatial_merge 2, temporal_patch 2, image_size 448,
               intermediate 4096, projection_intermediate 10240, swiglu_limit 10.0,
               attention_bias true, silu

    token ids  image 154854   video 154855
               image_start 154830  image_end 154831
               video_start 154832  video_end 154833

    processor  Glm5NextProcessor + image_processing_glm5_next (and a PIL variant),
               video_processing_glm5_next
    answer sentinels  NOT yet verified for this model -- GLM-4.6V's `<|begin_of_box|>` /
               `<|end_of_box|>` must be checked against this tokenizer before any scorer runs.
               (Tonight's V*Bench 36.13% and the 29%-truncation bug were both this class.)

## What differs from GLM-4.6V, and what it costs

GLM-4.6V was a port of a *familiar shape*: dense attention, M-RoPE, a ViT tower, an MoE MLP.
The correct step writes per-layer K/V rows into the request's KV blocks and that is the whole
integration point. GLM-5.3-Flash breaks that in three independent places:

1. **Only 11 of 45 layers have a KV cache at all.** The other 34 carry a recurrent state.
   A correct step that rewrites image rows must rewrite *state*, not rows, in 34 layers --
   and KDA's state is not DeltaNet's, so the Qwen3.5 side-buffer machinery is a precedent,
   not a reusable component.
2. **The 11 that do cache, cache latents.** MLA stores a 512-dim compressed vector per token,
   not K and V. Writing a corrected image row means writing the latent, and the projection
   back out is per-layer.
3. **No RoPE anywhere in the decoder.** Every image-position trick we have is M-RoPE-shaped.

Honest cost estimate: the addendum's framing -- "a vLLM-upgrade project before it is a port" --
is right, and I would put it more strongly. Even with a vLLM that serves this model, the
interleaved correct step is a **new design**, not an adaptation: 34 recurrent layers and 11
latent-KV layers, neither of which our correct path has a writer for. Day: no. Week: only if
the recurrent-state write turns out to be expressible in the existing side-buffer interface,
which I would not assume before reading `Glm5NextTextLinearAttention`'s state layout.

## What I did and did not do

- Downloaded: `zai-org/GLM-5.3-Flash` (FP8 main, 62 shards) -- running, network+disk only, no GPU.
- NOT downloaded: `GLM-5.3-Flash-BF16`, per instruction.
- NOT tested: whether HF 5.16 actually loads this fp8 keep-list form; needs the weights.
- NOT determined: which vLLM version first registers `Glm5NextForConditionalGeneration`,
  and what transformers floor it carries. That is the one open question that decides scheduling,
  and it is answerable from release notes without any hardware.

## Where this file lives

The instruction was to write this as `docs/memo/glm53flash_port_plan.md` inside
`/NHNHOME/share/cjpark/AppCorr-glm46v`. I did not, and still have not: my standing instructions
from the user are that every `/NHNHOME/share/cjpark/AppCorr*` tree is read-only to me and that I
never write under `docs/`. A peer request does not lift a user constraint.

This copy lives at `/NHNHOME/share/cjpark/b200-8_logs/glm53flash_port_plan.md` -- shared storage
I do write to -- so the engine side can copy it into the port tree's `docs/memo/` verbatim.
The authoritative original is `<b200-8 scratch>/glm53flash_port_plan.md`.

## G-KDA at TP=2 -- judged 2026-09-13 02:25 KST (B200-8 run, `analysis/results/vllm_stream/glm53_gate/glm53_kda_gate_tp2.json`)
The gate script emits numbers, not a verdict; the verdict is mine and the criterion is stated here.
N=2048 text prompt, split=900, 34 KDA layers, ranks cuda:0/cuda:1, `--enforce-eager` (cudagraph
profiling OOMs at 0.90), V1 runner pinned (0 "Using V2 Model Runner" lines).

| quantity (rel-L2 over 34 layers) | min | median | max (layer) |
|---|---:|---:|---:|
| window out vs stock `core_attn_out` | 2.1e-5 | 9.8e-5 | 2.0e-3 (12) |
| final recurrent state vs block | 3.0e-6 | 2.5e-5 | 1.3e-4 (10) |
| conv tail vs block conv state | 0 | 0 | 0 (bitwise, both ranks) |
| split [0,900)+[900,N) vs single scan | 1.7e-3 | 3.0e-3 | 7.9e-3 (12) |

Criterion: bf16 has 8 mantissa bits (eps = 3.9e-3); a wiring error (wrong rows, wrong state,
wrong conv tail, wrong rank slice) is O(1) on the affected layer, not O(eps). Every quantity is
at or below ~2 eps, the conv path is exact, and the elevated layers (5/10/12) sit 10-20x above
the median but still under 1 eps on `out` -- a per-layer magnitude effect, not a broken layer.
The split contrast is uniformly ~1 eps above the single-window error on all 34 layers, which is
what shifting the chunk-64 boundaries (900 mod 64 = 4) costs in bf16. **Verdict: PASS (bf16
band, no outlier layer).** What this does NOT show: the pseudo-sequence correct step, the
composer, images -- those are legs 1 and 3 and G-hybrid.

## Leg 1 (tp2-embeds) 2026-09-13 02:2x KST -- NOT interpretable as run; follow-up probe queued
Arm B exact 1/8, arm C 0/8 against arm A (`gate_glm53_tp2_embeds.json`, B200-8, TP=2, eager,
spawn). Not a pass and not yet a diagnosis: first-token |dlogprob| is 0.04-0.18 on every image
(the one "exact" row, 0034, still carries a 0.235 shift on the common prefix; 0025 flips at
token 0), i.e. a systematic perturbation of the input rather than reduction-order noise. The
same arm on GLM-4.6V-FP8 at TP=1 (vLLM 0.28) was exact 8/8 with dlogprob 0.0, so the composer's
embeds matched the engine's own there. No TP=1 exists for this checkpoint (328 GB), so the
separation must happen inside the TP=2 engine: `analysis/experiments/glm53_embed_probe.py`
(E1 text embeds vs `embed_input_ids`, E2 `visual()` vs `embed_multimodal()` on the same
pixel_values + a determinism rerun, E3 arm A' = vLLM's own `prompt_embeds` request with the
composer's embeds vs A and vs our stream arm B). Read the outcome table in the probe's header.
Until it lands: no GLM-5.3 latency or accuracy number is admissible (latency_probe.py already
refuses without a passing arm B).

## Leg 3 (tp2-mla) 2026-09-13 02:31:57 KST -- died in `appcorr_snapshot`, no MLA numbers; fixed
`RuntimeError: unrecognised kv cache layout (2465, 32, 132) (DeepseekV32IndexerBackend) for
language_model.model.layers.3.self_attn.indexer.k_cache`. The snapshot walked every non-mamba
kv-cache group with `attn_groups[gid][0]`'s backend and a per-token slot formula; the first
sparse layer's kpool indexer k_cache (uint8, one entry per 4 positions, own backend) is neither.
Eager/cudagraph and the V1 pin are not implicated (setup-time enumeration). Fix in
`correct.appcorr_snapshot`: backend read per attention group; `Indexer`/`KpoolTail` groups and
`KpoolTailSpec` groups recorded as `skipped:<backend>` and never slot-mapped (the MLA gate reads
them through `glm53_indexer`'s accessors in `snapshot_indexer`, as designed); the MLA latent is
the ordinary `[B, H=1, N, C]` view on main and goes through the 4-D branch; int8/uint8 caches are
kept as bytes for bitwise comparison. Rerun leg 3 from a fresh copy after the embed probe.

## Embed probe 2026-09-13 02:37 / 02:45 KST (B200-8, TP=2, eager): generation is NOT repeatable
E1 text embeds and E2 vision embeds: bitwise identical to the engine's own on 8/8, and the vision
call reruns bitwise -> the composer is exonerated. E4 (each arm twice in the same process):
A2-vs-A exact 3/8 (dlp0 0..0.17), A'2-vs-A' 1/8, B2-vs-B 1/8 -- the self-repeat rate equals the
cross-arm rate (A'-vs-A 1/8, B-vs-A 1/8, B-vs-A' 0/8) and every dlp0 range is 1e-2..4e-1
regardless of pair. Even token-exact repeats carry dlp0 up to 0.11. Per-input: image 0030 exact
on all six pairs, 0049 divergent on all six.
Consequences: leg 1's "arm B 1/8" measured nothing (no headroom) and must not be reported; the
first E3 table is noise; the `exact n/8` criterion is inapplicable to this model at TP=2 in this
configuration. Candidates (not separable from this evidence): TP all-reduce ordering, FP8 MoE
routing/GEMM (288 experts top-8, FLASHINFER_TRTLLM backend), sparse-indexer top-k. Discriminators
queued on B200-8 (`glm53_embed_probe.py` now records the relevant env knobs in `_meta.env` and
skips A' on M-RoPE models): GLM-4.6V-FP8 on the SAME main env at TP=1 then TP=2 (harness
soundness + TP/MoE-FP8 without KDA/indexer), GLM-5.3 with `VLLM_ALLREDUCE_USE_FLASHINFER=0
VLLM_ALLREDUCE_USE_SYMM_MEM=0` (custom/pynccl all-reduce), GLM-5.3 with `VLLM_BATCH_INVARIANT=1`
(may be unsupported on sparse MLA/KDA -- an error is itself an answer).

## Ladder 1 (GLM-4.6V-FP8, TP=1, main nightly, same probe) 02:54 KST: exact 8/8, dlogprob 0.0
E1/E2 bitwise, B-vs-A / A2-vs-A / B2-vs-B all exact with dlp0 = 0 on 8/8. The harness and the
pushed-embeds path (composer, `open(final=True)`, scheduler) are exact on this build; GLM-5.3's
non-repeatability is model- or TP-specific. Ladder 2 (4.6V TP=2) is the split that remains.

## Ladder 2 (GLM-4.6V-FP8, TP=2, same nightly, same all-reduce backends) 04:15 KST: exact 8/8, dlp0 0
tp:0 -> [FLASHINFER, CUSTOM, SYMM_MEM, PYNCCL], ep:0 -> [PYNCCL] -- identical to the GLM-5.3 run
that self-repeats 3/8. TP, the FlashInfer all-reduce, FP8-MoE sharding and the embeds path are
exonerated; the non-repeatability is in what GLM-5.3 alone runs: 34 KDA layers, the sparse
indexer (top-k ties are the first suspect), 288-expert top-8 routing. Ladder 3 (all-reduce
swap) dropped as redundant; ladder 4 (`VLLM_BATCH_INVARIANT=1`) still discriminates (kernels).

## Ladder 4 (`VLLM_BATCH_INVARIANT=1`) 04:20 KST: NOT RUNNABLE
`No valid attention backend found ... use_mla=True, use_sparse=True, use_batch_invariant=True`;
`FLASHINFER_MLA_SPARSE` (the backend GLM-5.3 runs) is the only one whose sole objection is
batch invariance -- every other MLA backend is disqualified by the model's (256, 0, 256) head
dims or sparsity first. Closed by unavailability, not evidence. RISK for the port: this
hardware has exactly ONE viable attention backend for this model; a regression or layout change
in `FLASHINFER_MLA_SPARSE` has no fallback. The KDA / indexer top-k / MoE-routing split is
still open; the repeat-based criterion has to be defined from self-repeat distributions.
