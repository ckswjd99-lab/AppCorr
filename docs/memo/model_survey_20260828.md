# 30B-class VLM survey for AppCorr applicability (2026-08-28)

User request: after the Gemma 4 port, assess these families at ~30B+ for our
technique. Facts below are from HF configs + (where it mattered) the actual
modeling source; nothing is guessed.

## Verdict table

| Model | Size | HF-native | Image tokens in LLM | Vision | AppCorr fit |
|---|---|---|---|---|---|
| Qwen3.5 | 35B-A3B / 122B-A10B | 5.13 OK | causal (GatedDeltaNet 30/40) | windowed ViT | DONE — in the table |
| **Muse Glimmer** | **29.6B dense** | needs >= 5.15.dev | **purely causal (verified in source)** | 50L windowed, merge, NoPE layers | **BEST new fit — lossless streaming + keep-limited vision** |
| Mistral Small 3.1 | 24B | 5.13 OK (`mistral3`) | causal (llava-style scatter) | Pixtral 24L, 1540px native | good secondary — cheap to port |
| InternVL3.5 | 38B / 30B-A3B | `-HF` ports exist | causal (qwen3) | InternViT-6B (45L!) + dynamic tiling | LOW for our technique — tile streaming already lossless there (verified on InternVL); use as the tile-streaming BASELINE |
| Aya Vision | 32B | 5.13 OK (`aya_vision`) | causal (Cohere 40L) | SigLIP 364px tiles (small tower) | streaming applies, but vision share is small -> limited correction upside; OV2-style arch at 32B (size-scaling companion). Gated license (CC-BY-NC). |
| Ovis2 / 2.6 | 34B / 30B-A3B | custom code only at these sizes | causal (qwen2 / qwen3-moe) | SigLIP2-NaViT; PROBABILISTIC visual tokens (embedding-table lookup) | novel angle (discrete visual tokens under degradation) but trust_remote_code fork = high port cost. Middle priority. |
| DeepSeek-VL2 | 27B (A4.5B) | none (custom, MLA) | causal (deepseek_v2 MoE) | tiling | LAST — oldest (2024-12), highest port cost, least momentum |

## The load-bearing finding: Muse Glimmer is purely causal

`modeling_muse_glimmer.py` (transformers main): the text model builds ONLY
`create_causal_mask` / `create_sliding_window_causal_mask` — no bidirectional
overlay, no block-sequence machinery of any kind. Image tokens are ordinary
causal tokens, exactly like Qwen/OV2. Combined with:
- 29.6B DENSE (52L, 3:1 sliding:full, window 2048, GQA 16:1, gated attention,
  NoPE on some layers — layer_rope_theta==0),
- a 50-layer/1536-hidden windowed vision tower (Qwen2.5-VL-style
  get_vision_window_index + spatial merge) — our windowed-vision fork
  experience transfers,
- a huge deployment footprint (GGUF mirrors alone ~1.9M downloads),
this is the strongest candidate for the streaming category at 30B scale, and
the vision tower is big enough for keep-limited correction to matter too.
Cost: requires transformers >= 5.15.0.dev0 (we run 5.13) — upgrade AFTER the
current campaigns, or in a scoped env.

## Recommended order (post-Gemma4)

1. **Muse Glimmer 30B** — completes the causal<->bidirectional axis at 30B:
   causal (Muse Glimmer) / hybrid (Gemma 4) / bidirectional-block (Gemma 3),
   all at comparable scale. Streaming + keep-limited vision both apply.
2. **Mistral Small 3.1 24B** — native in our transformers TODAY, Pixtral tower
   at 1540px native res (vision share meaningful), causal LLM. Cheapest port.
3. InternVL3.5-38B-HF — NOT for AppCorr; run as the tile-streaming baseline
   the paper compares against (the InternVL streaming-beats-AppCorr result at
   38B scale).
4. Aya Vision 32B / Ovis2-34B — only if the axis needs more points.
5. DeepSeek-VL2 — skip.
