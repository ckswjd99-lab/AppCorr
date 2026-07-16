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
- [x] **Phase 1 + Phase 3** — baseline stage timing + oracle progressive streaming
  (`oracle_progressive.py`). Measures real GPU compute (CUDA-event, warmed up) for visual encoder /
  monolithic prefill / per-group prefill / query prefill / decode, then runs a discrete-event
  timeline simulation combining that with a simulated transmission schedule. Compares baseline
  monolithic vs chunked-after-full vs oracle progressive; emits JSON timeline traces.

  **Key findings (Qwen2.5-VL, B200):**
  1. Chunked prefill has real per-chunk overhead (GPU underutilization at small chunk sizes), worst
     on large models: `chunked_after_full` runs 17–38% SLOWER than monolithic. This overhead is a
     cost that partially eats the overlap benefit.
  2. The oracle progressive benefit scales strongly with **visual sequence length** (image resolution
     → visual-token count), because the LLM prefill must be large relative to the vision encoder +
     chunking overhead for hiding it under transmission to pay off:
     - 345 visual tokens (typical RefCOCO): +2–6% TTFT speedup (3B/7B/32B).
     - 3136 visual tokens (high-res, 32B): **+14.1% TTFT speedup** (baseline 777ms → oracle 667ms;
       ~288ms of prefill hidden under transmission). This is the realistic high-res VLM/VLA regime.
  3. Fundamental tail cost: causal attention makes later visual groups more expensive (they attend to
     a longer KV cache — per-group prefill grows 75→136ms across G=4), and the last group + query
     prefill depend on the final residual, so they cannot be hidden.
- [~] **Vision-side accuracy bracket** (`accuracy_impact.py`) — before any latency work, measured the
  accuracy impact of the vision-side degradation on RefCOCO grounding (nr=64, 3B, exact-match IoU>0.5).
  full image = 81.25% baseline; base-only worst case depends strongly on base coarseness:
  half-res base (factor 2) −3.1pp (≈noise on nr=64), quarter-res base (factor 4) **−14.1pp**
  (67.19%, mean_iou 0.60). Confirms the vision-side change is real and base-coarseness-dependent, and
  motivates residual finalization (base-only is the worst case; progressive finalization sits between
  base and full). nr=64 is a sanity check only — full-dataset confirmation pending.
- [x] **Phase 4 + Phase 6 (accuracy, via re-encoding)** — `progressive.py` (band-aligned
  base/residual decomposition; visual-token groups == horizontal image bands in raster order) +
  `progressive_accuracy.py`. Measures the ACCURACY of monotonic progressive finalization: band g's
  visual tokens are re-encoded from "full-resolution in bands 1..g, coarse base below" and frozen
  (bidirectional staleness -- band g attends to the still-coarse bottom). Re-encoding is the UPPER
  BOUND of what a cheap first-order correction (Phase 5) can achieve.

  **Result (3B, RefCOCO nr=64, G=4, coarse base factor 4):**
  | vision tokens | acc@0.5 | mean_iou |
  |---|---|---|
  | full (baseline) | 84.38% | 0.7965 |
  | base_only (worst) | 73.44% | 0.6622 |
  | **progressive** | **79.69%** | **0.7878** |

  progressive vs full: **−4.69pp acc / −0.009 mean_iou** (the bidirectional-staleness cost of
  freezing early bands); progressive vs base_only: **+6.25pp / +0.126** (recovery from re-encoding
  residual bands). It recovers ~57% of the base→full acc gap and ~93% of the mean_iou gap — the box
  is nearly right, only borderline (IoU≈0.5) cases flip. Staleness cost is real but small,
  especially on IoU. (nr=64 sanity check.)

  **Full-dataset confirmation on RealWorldQA (N=765, VQA — the primary test set going forward):**
  | vision tokens | accuracy (N=765) |
  |---|---|
  | full (baseline) | 60.13% (460/765) |
  | base_only (worst) | 57.65% (441/765) |
  | **progressive** | **60.39% (462/765)** |

  progressive vs full: **+0.26pp** (2 samples — negligible / noise). progressive vs base_only: +2.75pp
  (fully recovers the base→full gap). **The staleness cost is TASK-DEPENDENT: negligible on VQA
  (global scene understanding, robust to finalization staleness — full-dataset), but measurable on
  grounding (precise localization needs the target region sharp AND finalized with full context —
  −4.69pp on RefCOCO nr=64).** RealWorldQA's coarse base also hurts far less (−2.48pp) than RefCOCO's
  (−14pp), confirming VQA is much more robust to the coarse base than precise grounding.
- [x] **Phase 5 — actual cheap correction on the Qwen visual encoder** (`progressive_correct.py`,
  driving the validated `appcorr/models/qwen25vl/vision/ApproxCorrectQwen25VLVisionTower` fork
  standalone; `progressive_accuracy.py --method correct`). Instead of re-encoding, band g's tokens are
  recomputed exactly against ONE growing K/V cache (bands 1..g−1 corrected in prior rounds, g+1..G
  still base) — the accumulated bidirectional staleness the re-encoding upper bound does NOT have.

  **Fork validated standalone (`correct_validate.py`, vs the STOCK model, bit-exact / 0.0e+00):**
  `approx(base)` == `get_image_features(base)`; `correct-all-in-one-round` == `get_image_features(full)`
  (collapses the base→full gap, mean abs ~1.03, to exactly 0). The correction is mechanism-exact.

  **Result (3B, RealWorldQA N=765, G=4, base factor 4) vs the re-encoding upper bound:**
  | vision tokens | re-encoding upper bound | **cheap correction (real)** |
  |---|---|---|
  | full | 60.13% (460/765) | 60.13% (460/765) |
  | base_only | 57.65% (441/765) | 57.65% (441/765) |
  | **progressive** | 60.39% (462/765) | **61.05% (467/765)** |

  The cheap first-order correction **reproduces the re-encoding upper bound** (+5 samples, noise):
  progressive vs full **+0.92pp**, vs upper bound +0.66pp. **The accumulated bidirectional staleness
  of the per-band cheap correction costs nothing measurable on VQA** — Phase 5 confirms the whole
  scheme works with a realistic cheap correction, not just the re-encode upper bound. (full/base_only
  are bit-identical to the Phase 4+6 run — same deterministic encodings — so only `progressive` moved.)
- [ ] Phase 7 — benchmarks + timeline plots across all modes.

## Notes on exactness

Chunked prefill is mechanism-exact (fp32 reproduces monolithic to ~1e-3 over the full depth). bf16
introduces small logit differences from the different matmul accumulation order under chunking, but
the next-token argmax is identical — so bf16 chunked prefill is practically equivalent to monolithic.
