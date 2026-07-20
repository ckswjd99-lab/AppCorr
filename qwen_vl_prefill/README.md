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

  **Full-dataset RefCOCO grounding (N=8811, cheap correction) — the staleness-sensitive stress test:**
  | vision tokens | acc@0.5 | mean_iou |
  |---|---|---|
  | full | 85.05% (7494/8811) | 0.7859 |
  | base_only | 77.40% (6820/8811) | 0.7027 |
  | **progressive (cheap)** | **81.93% (7219/8811)** | **0.7570** |

  progressive vs full **−3.12pp acc / −0.029 iou**; vs base_only +4.53pp / +0.054 — recovers **59% of
  the base→full acc gap (65% of the iou gap)**. **So the staleness cost is TASK-DEPENDENT and REAL on
  precise grounding** (−3.12pp, full-dataset) while negligible on VQA (+0.92pp): precise localization
  is sensitive to which bands are finalized when (accumulated bidirectional staleness), global scene
  understanding is not. The cheap correction still recovers most of the base→full gap on both.
- [x] **Phase 5b — decompose the RefCOCO −3.12pp** (full-dataset re-encoding upper bound, N=8811).
  full/base_only are bit-identical to the cheap run (85.05% / 77.40%) — consistency confirmed.
  | progressive | acc@0.5 | mean_iou | vs full |
  |---|---|---|---|
  | re-encode (upper bound) | 84.50% | 0.7796 | **−0.56pp / −0.006** |
  | cheap correction (real) | 81.93% | 0.7570 | **−3.12pp / −0.029** |

  **The −3.12pp splits as: inherent progressive-finalization staleness −0.56pp (nearly free!) +
  cheap-correction overhead −2.56pp (the dominant part).** This flips the naive expectation: even on
  precise grounding, *progressive finalization itself* costs almost nothing — the cost is the CHEAP
  correction's accumulated cache staleness. Re-encoding jointly re-encodes bands 1..g (each fresh),
  whereas the cheap correction attends band g against earlier bands' K/V **frozen from their own
  earlier, staler rounds**; precise localization is sensitive to that accumulated staleness, global
  VQA is not (on RealWorldQA cheap==re-encode==full). **Implication: the −2.56pp is recoverable
  headroom** — a better correction (re-refresh earlier bands' K/V as residuals arrive, or extra
  rounds) approaches the −0.56pp upper bound, at more compute.
- [x] **Phase 5c — GQA (open-ended short-answer VQA, N=12578)** + decomposition. Cheap correction:
  full 61.18% (7695), base_only 55.98% (7041), progressive 60.34% (7590); **progressive vs full
  −0.83pp**; vs base_only +4.36pp. Re-encode upper bound: progressive 60.42% (full/base_only
  bit-identical to the cheap run). **Decomposition: −0.76pp inherent progressive-finalization staleness
  + −0.07pp cheap-correction overhead** — the OPPOSITE of RefCOCO, where cheap overhead dominated.

  **Three-task summary (3B, G=4, base factor 4, full datasets; progressive vs full, decomposed):**
  | task | inherent (re-enc vs full) | cheap overhead (cheap vs re-enc) | total (cheap vs full) |
  |---|---|---|---|
  | RealWorldQA (global MCQ) | +0.66pp | +0.26pp | **+0.92pp** |
  | GQA (compositional short-answer) | −0.76pp | −0.07pp | **−0.83pp** |
  | RefCOCO (precise grounding) | −0.56pp | **−2.56pp** | **−3.12pp** |

  **The refined conclusion:** (1) the **cheap-correction overhead is ~0 on BOTH VQA tasks** and blows
  up (−2.56pp) **only on precise pixel grounding** — so the cheap correction is production-ready and
  effectively lossless for VQA/VLA-style global+compositional understanding, matching the re-encode
  upper bound; (2) the **inherent progressive-finalization staleness is uniformly small (≤0.76pp)**
  across all tasks including grounding. Practical takeaway: deploy the cheap correction as-is for
  VQA/VLA; the only place worth improving the correction (to recover the −2.56pp headroom) is precise
  bbox grounding.
- [~] **Phase C (preliminary) — dependence-aware `overlap` recovers the grounding cheap-correction
  overhead.** Motivated by spatial_dependence.py (local dependence) + flops_windowed_vs_full.py (the
  −2.56pp overhead lives entirely in the 4 full-attention layers, the only cross-band staleness
  locus): at correction round g, re-refresh the trailing `overlap` already-arrived bands so those 4
  layers mix fresher values for band g's nearest past dependencies (`progressive_correct.overlap`).
  Token-level: mean‖prog−full‖ 27.7 (o=0) → 10.9 (o=1) → 5.1 (o=2) → 0 (o=3=G−1).

  **RefCOCO 3B FULL-DATASET (N=8811)** confirmed (full 85.05% / 0.7859, base 77.40% / 0.7027):
  | overlap | acc@0.5 | mean_iou | vs full | recovered | correction cost |
  |---|---|---|---|---|---|
  | 0 (plain cheap) | 81.93% | 0.7570 | −3.12pp | — | 1× |
  | 1 | 84.05% | 0.7756 | **−1.00pp** | 68% | ~1.75× |
  | 2 | 84.62% | 0.7821 | **−0.43pp** | 86% | ~2.25× |

  band_o0 reproduces the independent full-dataset cheap-correction number (−3.12pp) exactly. A small
  local trailing re-refresh recovers most of the −2.56pp cheap-correction overhead at ≪ the 5×
  re-encode cost: **overlap=2 reaches −0.43pp ≈ the −0.56pp re-encode upper bound (essentially
  lossless)** at ~2.25×. Confirms the dependence-aware recovery idea. (Full-dataset is MORE optimistic
  than the N=800 preliminary — o1 −1.88→−1.00, o2 −1.38→−0.43 — the nr-sanity rule again.)
- [~] **Causal-order permutation of the visual prefill** (`causal_order.py`) — with M-RoPE positions
  carried along (each token keeps its true 2D position), only the causal MASK changes. Does it hurt?
  | order | RealWorldQA (full 765) | RefCOCO (full 8811) |
  |---|---|---|
  | identity (raster) | 60.13% | 85.05% |
  | reverse | −0.13pp | **−44.19pp** |
  | colmajor (=1-3-2-4) | −0.39pp | −23.60pp |
  | random | −0.52pp | −15.99pp |
  | **block4 (2D-local)** | +0.00pp | **−0.65pp** |

  **VQA is completely invariant to causal order; grounding is catastrophically sensitive to it —
  BUT only through 2D locality.** A causal order that keeps spatially-local tokens near each other
  (block-raster) is essentially lossless (−0.65pp on full 8811) while orders that scatter locality
  collapse (reverse −44pp). So the visual prefill is NOT locked to raster: any 2D-locality-preserving
  order works, enabling arrival/dependence/block-order streaming. Both datasets full-dataset confirmed.
- [x] **2D-block vs 1D-band correction (NEGATIVE result)** (`block_correct.py`, `block_accuracy.py`).
  Hypothesis: 2D-block granularity + spatial-nearest-neighbor overlap would recover more grounding
  overhead than 1D trailing bands. **Refuted.** RefCOCO N=3000 strided (vs full):
  | scheme | overlap 0 | 1 | 2 | 3 |
  |---|---|---|---|---|
  | 1D band (P=4,Q=1) | −2.47pp | −0.43pp | **−0.07pp** | — |
  | 2D block (4×4, nearest) | −3.77pp | −2.00pp | −1.17pp | −0.63pp |

  **1D horizontal bands beat 2D blocks at matched (or lower) refresh budget** — band_o0 (−2.47, budget
  N) vs blk44_o0 (−3.77, budget N); best 1D band_o2 (−0.07, ~2.25N) beats best 2D blk44_o3 (−0.63,
  ~3.8N). Grounding coordinate reasoning needs **horizontal consistency**: a full-width band finalizes
  a whole Y-level together in one context, whereas 2D blocks fragment the width and finalize left/right
  of the same row at different times/contexts, hurting precise localization. (Not a contradiction with
  the block-causal-order result, which is about ORDER, not correction granularity.) **Takeaway: 1D
  horizontal bands + overlap is the recovery mechanism; band overlap=2 recovers the −2.56pp overhead
  to ~full (−0.07pp).** Full-dataset confirmation of band overlap pending.
- [x] **Independent tile encoding (NEGATIVE result)** (`split_encode.py`). Test: encode P×Q image
  tiles separately through the vision encoder, stitch tokens to raster, feed the LLM. Qwen2.5-VL
  tokens ARE location-agnostic (2D RoPE relative, no absolute pos embedding; M-RoPE re-assigns global
  position — verified 1×1==full bit-exact). But encoding tiles standalone destroys **cross-tile
  relative structure AND cross-tile attention** (each tile restarts at pos (0,0), per-tile windowing,
  no cross-tile mixing). Result (vs 1×1 full-image, interim ~30%):
  | split | RealWorldQA (VQA) | RefCOCO (grounding) |
  |---|---|---|
  | 2×1 (full-width tiles) | −5.0pp | −43.9pp |
  | 2×2 | −9.7pp | −76.0pp (85→9%) |
  | 4×4 | −9.3pp | −79.6pp |

  **Grounding collapses; VQA loses ~10pp.** Far worse than progressive correction (grounding −3.12pp)
  because progressive keeps the SHARED encoder (full-image windowing/positions, only non-arrived K/V
  is stale-base), whereas tile-splitting discards the global coordinate frame + all cross-tile
  attention. So independent-tile encoding is not viable (grounding), and horizontal-preserving splits
  (2×1) always hurt least — consistent with the 1D-band / horizontal-consistency findings.
- [ ] Phase 7 — benchmarks + timeline plots across all modes (deferred).

## Notes on exactness

Chunked prefill is mechanism-exact (fp32 reproduces monolithic to ~1e-3 over the full depth). bf16
introduces small logit differences from the different matmul accumulation order under chunking, but
the next-token argmax is identical — so bf16 chunked prefill is practically equivalent to monolithic.
