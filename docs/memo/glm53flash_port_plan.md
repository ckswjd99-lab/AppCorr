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
