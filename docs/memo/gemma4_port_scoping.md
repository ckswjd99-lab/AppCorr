# Gemma 4 31B port — scoping (2026-08-28, pre-maintenance)

User decision: extend AppCorr to `google/gemma-4-31B-it` (dense 31B, 62.5 GB bf16,
downloaded to /NHNHOME/huggingface — survives maintenance). transformers 5.13.0
supports it natively (`Gemma4ForConditionalGeneration`).

## Architecture facts (config + modeling_gemma4.py, verified 2026-08-28)

**Vision** (`Gemma4VisionModel`): NATIVE-RESOLUTION ViT, 27 layers, hidden 1152,
patch 16, 2D rotary position embeddings. Processor does aspect-preserving resize to
<= 2520 patches (= 280 soft tokens x 3x3 pool), patchifies to
[n_patches, 16*16*3] + `pixel_position_ids` (x,y int coords; (-1,-1) = padding).
Tower = patch_embedder -> encoder (bidirectional, padding mask only) -> 3x3 pooler
-> float32 standardize (std_bias/std_scale buffers) -> `embed_vision`
(RMSNorm-no-scale + Linear 1152->5376). Soft tokens per image <= 280.

**Text**: 60 layers, hidden 5376, 32 heads / 16 KV, head_dim 256. layer_types =
50 sliding (window 1024) : 10 full, pattern 5:1. Dense (no MoE), no per-layer
inputs (`hidden_size_per_layer_input=0` — the Gemma-3n PLE machinery is OFF for
31B). vocab 262144.

**THE mask finding** (`create_masks_for_vision_model`, use_bidirectional_attention
= "vision" for 31B): unlike Gemma 3 (image block bidirectional on ALL layers),
Gemma 4 image tokens are:
  - full-attention layers (10): STRICTLY CAUSAL — no bidirectional overlay;
  - sliding layers (50): AND(window-1024, OR(causal, same-image-block bidir)).
Block ids come from `mm_token_type_ids` (contiguous vision runs).

## What this means for AppCorr

Gemma 4 is a NEW, third point on the causal<->bidirectional axis:
  Qwen/OV2 (fully causal, streaming lossless) — **Gemma 4 (causal on full layers,
  block-bidir on sliding layers)** — Gemma 3 (block-bidir everywhere, streaming
  impossible).
An image block is <= 280 tokens < window 1024, so within-block bidirectionality on
sliding layers is complete. Streaming the LLM prefill in sub-block chunks violates
ONLY the 50 sliding layers' future-in-block attendance; the 10 full layers are
already exactly causal. So:
  - pure streaming: NOT lossless (unlike OV2), but the violation is confined and
    quantifiable per layer type — worth measuring as its own result;
  - approximate-then-correct (interleaved): fully applicable, and the vision tower
    is bidirectional as always, so the vision-side correction ports directly;
  - the 2D patch coordinates (`pixel_position_ids`) make band-wise transmission
    NATURAL — no grid inference needed, the positions are explicit.

## Port plan (post-maintenance backlog, priority order)

1. `Gemma4Axis.build_inputs` + `full_forward` (manual replication of
   Gemma4Model.forward: embed -> get_image_features -> masked_scatter -> dual
   masks -> language_model) gated harness==stock on first-token logits.
   [STARTED 2026-08-28 — appcorr/models/gemma4/unified.py]
2. Degradation: degrade the IMAGE before patchify at the same resolution — the
   resize/patch grid depends only on (H,W), so approx and full share shapes; the
   gemma3 `l2_from_native`-with-shape-assert pattern carries over.
3. Vision fork for approx/correct: follow the gemma3 backbone-fork pattern
   (K/V cache per layer, `_received_attention` running layermean, merge-group
   correct_forward). New wrinkles: 2D rope (recompute per arrived set is
   position-indexed — gather rows like the VGGT bucket fix), 3x3 pooler +
   standardize sit AFTER the encoder (correction granularity = pre-pool patches,
   soft tokens re-pooled per round), padding rows must stay out of top-k
   (energy = -inf on padding).
4. Interleaved progressive walk (canonical per-round selection) + identity gate
   keep=1.0 g=1 == full_forward, then keep-monotone gate.
5. Streaming arm (sub-block chunked prefill) measured AGAINST the known sliding-
   layer violation — report as the "partial-causality" point of the unified axis.
6. FLOPs: standard hooks should cover (dense Linears); pooler/embedder are
   Linear/conv-free ops — verify with the diff-vs-closed-form check as for qwen35.

## Gotchas recorded

- `do_normalize=false`, mean 0 / std 1 — degradation must NOT assume SigLIP-style
  normalize; rescale only (1/255).
- Vision runs the standardize step in float32 and casts back — keep that exact
  order in any fork or the parity gate will fail at ~1e-2.
- `get_placeholder_mask` compares against image_token_id (258880); chat template
  emits boi/eoi (255999/256000-ish) around the run — token_type_ids path uses
  mm_token_type_ids==1|2 for block ids.
- Audio/video towers exist in the class but audio_config=null for 31B; ignore.
