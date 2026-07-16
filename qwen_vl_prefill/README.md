# Qwen-VL Progressive Prefill Prototype

Extends ProgVFM's interleaved correction into a Qwen2.5-VL VLM/VLA serving prototype: turn the
visual encoder into a *progressive visual-token producer* so that visual-token correction and most
of the LLM prefill for early visual groups **overlap with image transmission**, and the final
answer/action arrives sooner. The goal is latency (time-to-final-answer), **not** early low-quality
answers.

Built on the **stock HuggingFace** `Qwen2_5_VLForConditionalGeneration` (transformers 5.13.0) with the
standard KV-cache API (`DynamicCache` / `past_key_values`). The ProgVFM first-order correction plugs
in later at the visual-embedding interface (a correction module outputs *corrected visual-embedding
groups*; the LLM prefill path stays pure-stock).

## Modules (kept separate: introspection ≠ correction math ≠ benchmark logic)

- `introspect.py` — model loading + multimodal input plumbing for the stock model: `prepare_inputs`,
  `extract_visual_embeds` (= `model.model.get_image_features(...).pooler_output`, post-merger,
  LLM-order), `build_inputs_embeds` (masked_scatter), `compute_position_ids` (M-RoPE `[3,1,T]`,
  computed once), `token_layout` / `streaming_chunk_boundaries`.
- `prefill.py` — `monolithic_prefill`, `chunked_prefill` (one growing `DynamicCache`), `compare_logits`.
- `equivalence_test.py` — **milestone**: chunked == monolithic prefill test.

## Run

```bash
conda activate appcorr

# Monolithic-vs-chunked prefill equivalence test (first milestone)
python qwen_vl_prefill/equivalence_test.py \
    --model-id Qwen/Qwen2.5-VL-3B-Instruct --num-groups 4 --device cuda:0
# default image = RefCOCO val[0]; add --image PATH --prompt "..." to override.
# --dtype fp32 isolates bf16 accumulation noise from real bugs.
```

Model id is configurable (`Qwen/Qwen2.5-VL-3B-Instruct` for dev; 7B/32B later). Number of visual
groups `G` is configurable via `--num-groups`.

## Milestone status

- [x] **Phase 0** — inspect stock model: `Qwen2_5_VLForConditionalGeneration`; visual embeds =
  `model.model.get_image_features(pixel_values, image_grid_thw).pooler_output`; position_ids =
  `get_rope_index(...)` → `[3,1,T]` M-RoPE.
- [x] **Phase 2** — monolithic-vs-chunked prefill equivalence. **PASS**: in fp32 max abs logit diff
  `7.2e-3`, 100% per-position argmax agreement; in bf16 the residual diff is pure matmul
  accumulation-order noise (97.4% argmax agreement, next-token argmax identical). Mechanism verified:
  chunked prefill with sliced M-RoPE position_ids + per-chunk `cache_position` into one `DynamicCache`
  reproduces monolithic prefill.
- [ ] Phase 1 — monolithic baseline latency benchmark (stage-by-stage timing).
- [ ] Phase 3 — oracle progressive visual-token streaming (simulated arrival schedule; upper-bound
  latency benefit; timeline traces).
- [ ] Phase 4 — Laplacian base/residual decomposition + visual-token-aligned grouping.
- [ ] Phase 5 — actual ProgVFM first-order correction on the Qwen visual encoder.
- [ ] Phase 6 — monotonic visual-token finalization + stale-cache analysis (bidirectional vision
  means later residuals perturb earlier tokens; measure the error, optional re-prefill fallback).
- [ ] Phase 7 — benchmarks + timeline plots across all modes.

## Notes on exactness

Chunked prefill is mechanism-exact (fp32 reproduces monolithic to ~1e-3 over the full depth). bf16
introduces small logit differences from the different matmul accumulation order under chunking, but
the next-token argmax is identical — so bf16 chunked prefill is practically equivalent to monolithic.
