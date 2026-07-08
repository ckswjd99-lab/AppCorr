# AppCorr for CLIP-ViT-bigG/14: session log

Branch: `experiment/clip-appcorr` (off `main`). Full plan: see the approved plan mode session
(context/phases 0-5). This log records what has actually been run.

## Phase 0 -- Oracle (commit eeb4306)
- `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k` loaded via `transformers.CLIPModel` (native sharded
  safetensors checkpoint, ~10.16GB, no `open_clip` needed for loading). 48 vision layers, 32 text
  layers, `hidden_act="gelu"` (not quick_gelu), 224px/patch14 -> 256 patches + 1 CLS.
  `open_clip_torch` installed just for its `IMAGENET_CLASSNAMES`/`OPENAI_IMAGENET_TEMPLATES`
  metadata (1000 classnames, 80 templates) -- exact, standard, no need to hand-transcribe.
- `clip_bigg_oracle.py` dumps per-layer CLS states, final image embeddings, all 1000 zero-shot
  class embeddings, real COCO caption embeddings. Sanity: top1 predictions correctly recover
  classes 0/1/2/3 for the first 4 ImageNet val images.

## Phase 1 -- Vision tower fork (commits c237d39, 8b5ab17)
- `appcorr/models/openclip/vision/{attention,block,backbone}.py`, mirroring
  `appcorr/models/openvla/vision/` (non-RoPE ViT, SDPA, batch_size=1 => no packed variable-length
  query state needed). CLIP deltas: separate q/k/v_proj (no fused qkv, no qk-norm), 1 prefix token
  (CLS only), FULL 48-layer depth (no truncation -- only the final CLS state matters, no
  downstream LLM), `get_image_embeds()` = post_layernorm(CLS) -> visual_projection -> normalize.
- Refactored to layer-chunked `approx_forward(start_l, end_l)`/`correct_forward(..., start_l,
  end_l)` so the EXISTING `GroupTriggerPolicy` scheduling policy drives it directly (same contract
  as `DINOv3ClassifierExecutor`) -- no new scheduling code.
- Unit test (`clip_vision_fork_unittest.py`), 4-tier: (a) approx()-only vs stock: bit-exact
  (max_abs_err=0.0). (b) correct(all patches, from a blurred approx pass) vs stock: bit-exact.
  (c) correct(half patches): real, bounded approximation error (mean=0.011, max=0.081) -- expected,
  same accepted property as DINOv2/DINOv3 (bidirectional attention, multi-round staleness). (d)
  layer-chunked (4x12) approx matches the one-shot result exactly -- validates the chunking
  contract itself.

## Phase 2+3 -- Executor + ImageNet zero-shot eval driver (commits fafebee, plus cherry-pick 4cedaf2)
- `offload/server/model/openclip_executor.py`: `OpenCLIPExecutor`, all 9 `ModelExecutor` ABC
  methods. Text tower always full one-shot forward (never streamed). 1000-class zero-shot weights
  precomputed once at `load_model`. Registered in `offload/server/model/__init__.py`.
- 3 configs at CLIP's native 224px/patch14 resolution, `total_layers=48`:
  `imagenet_clip_bigg_{sequential,approx_only_l2,interleaved_g4}.json`.
- `analysis/experiments/clip_zeroshot_offload_eval.py`: near-copy of
  `dinov3_classifier_offload_eval.py` (same deterministic strided sampling, same CUDA-event
  per-op latency).
- Hit the SAME pre-existing `laplacian.py` decode bug found earlier this session for the DINOv3
  classifier work (`pyramid_levels=[2]` alone never upsampling back to native resolution) -- this
  branch forked from `main` before that fix (commit `3b375c1`, branch `experiment/energy-grouping`)
  had landed there. Cherry-picked cleanly (commit `4cedaf2`).
- **nr=10 results** (grid grouping, num_groups=4, no per-group pruning implemented yet -- every
  patch eventually gets corrected across the 4 groups):

  | config | top1 | top5 |
  |---|---|---|
  | full baseline (sequential) | 90% | 100% |
  | approx-only (heavily blurred, 1/4 res) | 70% | 100% |
  | interleaved correction (grid, g4) | **90%** | 100% |

  Interleaved correction recovers the FULL baseline's accuracy exactly (90% == 90%), while
  approx-only shows a real, expected degradation (70%) -- a clean sanity confirmation that the
  approx/correct mechanism is working correctly end-to-end for this new model family. Not
  surprising that interleaved==baseline exactly: with no per-group token pruning yet (unlike the
  DINOv3 classifier's validated `token_keep_thres` mechanism), 100% of patches get corrected by
  the last group, so this is mathematically expected to match full baseline (same "token_keep_
  ratio=1.0 -> all strategies converge" property observed in the earlier energy-grouping
  investigation). Latency-wise, interleaved isn't yet faster than full baseline either, for the
  same reason (no pruning => same total work, just chunked) -- expected until Phase 5 adds pruning.

## Next steps (not yet done)
- Scale ImageNet zero-shot nr=10 -> nr=20/50 for a more robust baseline/approx-only/interleaved
  comparison (same discipline as the DINOv3 investigation).
- Phase 4: COCO retrieval (captions_val2017.json, 5000 images) -- new captions loader,
  retrieval-mode executor path (already stubbed via `clip_task="retrieval"` in the executor, not
  yet wired to a config/driver), recall@1/5/10 i2t+t2i.
- Phase 5: port the validated `residual_energy x avg_attn` thresholded pruning (this session's
  DINOv3 classifier finding) into `appcorr/models/openclip/vision/block.py`'s `correct()`, giving
  interleaved correction an actual latency advantage over full baseline (currently it has none,
  since nothing is pruned yet) -- then sweep grouping strategy/num_groups/threshold on both tasks.
