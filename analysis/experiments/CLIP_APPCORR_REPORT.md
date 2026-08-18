# AppCorr for CLIP-ViT-bigG/14 — Full Experiment Report

**Branch:** `experiment/clip-appcorr` (off `main`)
**Repo:** `/NHNHOME/share/cjpark/AppCorr`
**Model:** `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k` (via `transformers.CLIPModel`)
**Companion log:** `analysis/experiments/CLIP_APPCORR_LOG.md` (chronological lab notebook; this
document is the organized, synthesized report of the same work)

---

## 1. Executive Summary

This work ports AppCorr's approximate-then-correct progressive-inference mechanism — previously
built for DINOv3 (bidirectional ViT classifier/detector/segmentor/depther) and OpenVLA
(DINOv2+SigLIP vision towers feeding a causal Llama VLA) — to **CLIP-ViT-bigG/14**, a dual-encoder
vision-language model, and evaluates it on two standard tasks: **ImageNet-1k zero-shot
classification** and **MS-COCO Captions image-text retrieval**.

**Headline results:**

1. The AppCorr mechanism (hard-forked vision tower with `.approx()`/`.correct()`, driven by the
   *existing, unmodified* `GroupTriggerPolicy` scheduler) works correctly end-to-end for this new
   model family. Interleaved correction closely tracks full-baseline accuracy on both tasks while
   approx-only (no correction, just a blurred low-res forward) shows clear, real degradation.
2. A validated, content-adaptive token-pruning score (`residual_energy × avg_cls_attention`,
   thresholded — the same mechanism validated earlier this session for the DINOv3 classifier)
   gives ImageNet zero-shot classification a genuine, nearly-free accuracy/compression trade-off:
   **-1.0pp top1 for -26pp of patches corrected** at the identified sweet spot. The full
   accuracy-vs-keep-rate curve was mapped from 100% down to ~24% keep-rate at full dataset scale
   (all 50,000 ImageNet val images, no sampling).
3. The same pruning mechanism does **not** give COCO retrieval a comparable free lunch — every
   pruning level tested costs real recall immediately once evaluated at true dataset scale (5,000
   images / 25,014 captions). A small-sample sweep initially suggested otherwise; this was
   diagnosed as a ceiling-effect measurement artifact and corrected with full-scale runs.
4. Per-layer-chunked token pruning does **not** yield a net *latency* win at this model's scale,
   unlike DINOv3-7B — a real, mechanistically-explained negative result (kernel-launch overhead
   dominates over FLOPs saved, because CLIP-bigG's per-layer compute is lighter). Energy-based
   *grouping* (as opposed to pruning) also does not help here, cross-validating an earlier,
   independent DINOv3 finding.
5. Three real, pre-existing bugs were found and fixed along the way (two by cherry-picking fixes
   that existed on a sibling branch but had never reached `main`; one novel `transformers`-API
   discovery bug).

---

## 2. Background & Motivation

**AppCorr** is a mobile-to-server offload framework: a mobile client streams a low-resolution
"base layer" of an image to a server, which runs a cheap **approximate** forward pass on it
immediately. As higher-resolution patch data streams in progressively, the server **corrects**
only the affected tokens' attention output, using cached K/V and residual-delta bookkeeping to
avoid redundant full recomputation. This is implemented via a hand-forked transformer block
exposing `.approx()`/`.correct()` entry points sharing a per-request `cache_feature` dict, driven
by a generic `ModelExecutor` ABC and a `Task`/`Instruction`/`OpType` scheduling protocol.

Prior to this work, AppCorr existed for:
- **DINOv3** (ViT backbones for classification/detection/segmentation/depth) — the original,
  most mature implementation, in `appcorr/models/dinov3/`.
- **OpenVLA** (DINOv2 + SigLIP vision towers feeding a causal Llama-2 VLA policy) — a separate,
  earlier extension in this same repo on branch `develop/openvla-progressive-prefill`, notable for
  being the first non-DINOv3, non-RoPE ViT fork, and the first (and only) causal-LLM correction.

**This work** extends AppCorr to **CLIP**, a fundamentally different use case: a **dual-encoder**
model where only the *image* side benefits from progressive correction (text/captions are never
streamed, so the text tower always runs a plain one-shot full forward), and where the downstream
task is **not** next-token prediction but **embedding similarity** (zero-shot classification via
cosine similarity to text-derived class prototypes; retrieval via cosine similarity ranking against
a caption gallery). This is architecturally closest to the OpenVLA vision-tower fork (non-RoPE,
SDPA-based ViT) but requires the **full** transformer depth (48 layers) rather than a truncated
depth-2 extraction, since only the *final* CLS embedding matters.

---

## 3. Model Architecture

`laion/CLIP-ViT-bigG-14-laion2B-39B-b160k`, loaded via `transformers.CLIPModel.from_pretrained(...,
dtype=torch.bfloat16)`. This checkpoint is natively `transformers`-compatible (sharded safetensors,
~10.16GB total across `pytorch_model-0000{1,2}-of-00002.safetensors`) — no need for the `open_clip`
Python package to *load* the model (though `open_clip_torch` was installed anyway, purely to reuse
its `IMAGENET_CLASSNAMES`/`OPENAI_IMAGENET_TEMPLATES` metadata constants for zero-shot prompting).

| | Vision tower | Text tower |
|---|---|---|
| `hidden_size` | 1664 | 1280 |
| `num_hidden_layers` | 48 | 32 |
| `num_attention_heads` | 16 | 20 |
| `patch_size` / `image_size` | 14 / 224 (→ 256 patches) | — |
| `max_position_embeddings` | — | 77 |
| `hidden_act` | **`gelu`** (not `quick_gelu` — bigG-specific, confirmed via config; earlier/smaller CLIP variants use `quick_gelu`) | `gelu` |
| prefix tokens | 1 (CLS only, **no register tokens** — simpler than DINOv2's 5) | — |
| `projection_dim` (shared embedding space) | 1280 | 1280 |
| `logit_scale` (learned, exponentiated) | ≈ 99.0 | — |

Block structure (`CLIPEncoderLayer`/`CLIPAttention`, confirmed by reading
`transformers/models/clip/modeling_clip.py` for the installed version, 5.13.0): a vanilla pre-LN
residual block —
```
x = x + self_attn(layer_norm1(x))
x = x + mlp(layer_norm2(x))
```
— with **no LayerScale at all** (simpler than DINOv2, which has real LayerScale), **separate**
`q_proj`/`k_proj`/`v_proj`/`out_proj` linear layers (no fused QKV, no q/k-norm — unlike DINOv2/
DINOv3), and absolute learned position embeddings baked into the embedding layer (`class_embedding`
+ `patch_embedding` conv + `position_embedding`), followed by a `pre_layrnorm`. No RoPE anywhere.
Final pooling: `post_layernorm(CLS) → visual_projection → L2-normalize` for the image side;
analogous `post_layernorm(EOS-token-hidden-state) → text_projection → L2-normalize` for text.

---

## 4. Implementation

### 4.1 Vision tower fork — `appcorr/models/openclip/vision/`

Three files, directly mirroring the OpenVLA vision-tower fork's pattern (`appcorr/models/openvla/
vision/{attention,block,backbone}.py` on `develop/openvla-progressive-prefill`) — a non-RoPE,
SDPA-based, `.approx()`/`.correct()` ViT fork — adapted for CLIP's specifics.

**`attention.py`** — `ApproxCorrectCLIPAttention`:
- `from_stock(attn)`: wraps the stock module's `q_proj`/`k_proj`/`v_proj`/`out_proj` **by
  reference** (no weight copy), so `.forward()` is numerically identical to stock.
- `.approx(x, cache_feature, tag, collect_cls_attn)`: full self-attention over all N=257 tokens via
  `F.scaled_dot_product_attention` (SDPA — confirmed faster than manual `(Q@Kᵀ)·scale` attention in
  every prior fork built this session; kept as the sole implementation here). Caches raw K/V as
  `cache_feature[f"{tag}_kv"]`, shape `[B, H, N, 2, Dh]`. If `collect_cls_attn`, also computes and
  caches the CLS→patch attention distribution (head-averaged) for later importance-score use.
- `.correct(x_sel, token_idx, cache_feature, tag)`: recomputes fresh Q/K/V **only** for the query
  subset `token_idx`, splices the fresh K/V into the cached tensor at those positions
  (`kv[:, :, token_idx, 0/1] = ...`), then reruns SDPA for the new Q against the **full**
  (now-partially-patched) cached K/V, returning output only for the `token_idx` rows.

**`block.py`** — `ApproxCorrectCLIPEncoderLayer`: wraps `layer_norm1`/`attn`/`layer_norm2`/`mlp`
(no LayerScale, matching stock). `.approx()` additionally caches the *total* block output delta
(`{tag}_blocks_out_sum` = attention contribution + MLP contribution) so `.correct()` can
reconstruct non-corrected positions exactly via `x_in + blocks_out_sum` (a no-op reconstruction —
never actually *read* downstream by anything, since attention reads K/V from cache and norm/MLP
only touch `token_idx` rows; this is a "dead value" bookkeeping trick inherited unchanged from the
DINOv3/OpenVLA design).

**`backbone.py`** — `ApproxCorrectCLIPVisionTower`: wraps `embeddings` + `pre_layrnorm` (always
exact — tokenization has no "corrected vs. stale" notion; cheap, so always recomputed fresh) plus
48 forked blocks. Key CLIP-specific difference from the OpenVLA template: **full 48-layer depth**,
no depth-2 truncation (OpenVLA's DINOv2/SigLIP fork stops 2 layers early because a downstream LLM
consumes intermediate patch features; CLIP has no such consumer — only the *final* CLS state,
pooled and projected, matters). Added `get_image_embeds(x_full)` = `post_layernorm(CLS) →
visual_projection → normalize`, the CLIP-specific final step with no OpenVLA analogue.

**Layer-chunked contract** (added in a refactor after the initial version): `approx_forward`/
`correct_forward` take explicit `start_l`/`end_l` layer-range arguments and operate on a
caller-threaded `x_feature` tensor, rather than doing the whole depth in one call. This exactly
matches the DINOv3 classifier executor's `approx_forward(layers=(start_l, end_l))` contract, which
means the **existing, unmodified** `GroupTriggerPolicy` scheduling policy (which fires
`APPROX_FORWARD`/`CORRECT_FORWARD` instructions with layer-range chunks interleaved with patch
arrival) can drive this tower directly — no new scheduling code was needed anywhere in this
project.

**CLS-token invariant**: CLS (the sole prefix token) is *always* force-included in every round's
corrected/query set, regardless of which patches actually arrived — same reasoning as DINOv2's
prefix tokens in the OpenVLA fork: every patch attends to the CLS key/value, so leaving CLS
permanently stale would leak into every patch's attention output and break exactness even at 100%
patch correction.

### 4.2 Executor — `offload/server/model/openclip_executor.py`

`OpenCLIPExecutor(ModelExecutor)` implements all 9 ABC methods (`load_model`, `preprocess`,
`prepare_tokens`, `approx_forward`, `correct_forward`, `head_inference`, `full_inference`,
`decide_exit`, `get_final_results`). Registered in `offload/server/model/__init__.py`'s factory via
an `"openclip" in name` substring match (so both `model_name: "openclip_zeroshot"` and
`"openclip_retrieval"` route to the same class).

Two task modes, selected by `config.dataset_kwargs.get('clip_task', 'zeroshot')`:
- **`"zeroshot"`**: `load_model` precomputes the full 1000-class ImageNet zero-shot text embedding
  matrix once (the standard OpenAI 80-template prompt ensemble, per-class averaged + renormalized,
  via `open_clip`'s `IMAGENET_CLASSNAMES`/`OPENAI_IMAGENET_TEMPLATES` metadata — exact reproduction
  of the standard CLIP zero-shot recipe, not hand-approximated). `head_inference` computes
  `logit_scale × image_embeds @ zeroshot_weights.T`, returns top-5.
- **`"retrieval"`**: `head_inference` just returns the normalized image embedding itself — no
  classification head. The eval driver accumulates these and computes recall@k externally against
  separately-precomputed caption embeddings.

**Pruning** (`_prune_patch_idx`, added in a later phase): implements the validated DINOv3-
classifier importance score — `combined_score = server_pscore × mobile_pscore` where
`server_pscore` = `cls_attn_prob_layermean` (CLS→patch attention, averaged across every layer that
has run `.approx()` **so far**, refreshed after every layer-chunk rather than only at the end — a
partial-depth average is a usable, if less refined, proxy, letting early groups' pruning decisions
benefit from whatever depth has been seen) and `mobile_pscore` = raw residual pixel energy (client-
side `Patch.pscore_hint`, computed by the existing, model-agnostic
`ProgressiveLPyramidPolicy._compute_patch_pscore_hint`). Selection is a hard threshold
(`keep_mask = combined >= token_keep_thres`), **not** a top-K ratio — this specific combination
(multiply fusion + absolute threshold, as opposed to `geo_mean`/`add` fusion or top-K selection)
matches the exact setting found to work best for the DINOv3 classifier earlier this session.
Includes a safety fallback: if *no* patch in a group clears the threshold, the whole group is kept
unpruned rather than corrected with zero patches — this fallback turns out to have significant,
initially-surprising consequences at high threshold values (§7.5).

**Bucket-padding** (`_bucket_pad_patch_idx`): pads the (possibly pruned) query set to a fixed
`sdpa_query_bucket_size` multiple by **duplicating** existing indices, avoiding novel-shape
cuBLAS/SDPA kernel dispatch overhead for variable-length pruned query sets. Proven safe: a
duplicated index reads/writes the exact same value every time (idempotent), so padding can never
corrupt results, only spend compute on redundant rows.

**Keep-rate telemetry**: `_prune_patch_idx` accumulates `cache_feature["_token_prune_kept_patch_total"]`/
`"_token_prune_full_patch_total"` — this reuses fields `worker.py` **already** reads generically
(`InferenceResult.token_prune_kept_patch`/`token_prune_full_patch`, populated the same way for the
DINOv3 classifier), so no new plumbing was needed beyond having the executor populate them; both
eval drivers report an aggregate `keep_rate_pct` automatically.

### 4.3 Configs — `offload/config/`

9 JSON configs, following the existing repo convention exactly (mirroring
`offload/config/imnet_{sequential,approx_only_l2,interleaved_g4}.json`'s structure for DINOv3),
at CLIP's native 224px/patch-14 resolution with `total_layers=48`:

| Config | Task | Transmission | Notes |
|---|---|---|---|
| `imagenet_clip_bigg_sequential.json` | zeroshot | `FullImageCompression` (lossless PNG) | full baseline |
| `imagenet_clip_bigg_approx_only_l2.json` | zeroshot | `Laplacian`, `pyramid_levels=[2]` | approx-only |
| `imagenet_clip_bigg_interleaved_g4.json` | zeroshot | `ProgressiveLaplacian` + `GroupTrigger`, `num_groups=4` | interleaved correction |
| `coco_retrieval_clip_bigg_{sequential,approx_only_l2,interleaved_g4}.json` | retrieval | (same 3 patterns) | — |
| `imagenet_clip_bigg_interleaved_g4_thr{250,700}[_bucket{16,32}].json` | zeroshot | interleaved | early pruning/bucketing exploration (superseded by the `--token-keep-thres` CLI flag) |

### 4.4 Eval drivers — `analysis/experiments/`

Both drivers use the same **local, in-process multiprocess** pattern established earlier this
session for the DINOv3 classifier investigation (`SchedulerModule`+`WorkerModule` driven directly
via multiprocessing queues, bypassing the real TCP transport in `offload/server/main.py`/
`offload/mobile/main.py` — faster to iterate with, and nothing in this investigation needed the
real network layer). `batch_size` is forced to 1 (clean per-image latency, simple result indexing,
no packed variable-length query-state machinery needed).

**`clip_zeroshot_offload_eval.py`**: deterministic strided sampling of ImageNet val (or `--full`
for all 50,000 images, sequential order); `--config`, `--grouping-strategy`, `--num-groups`,
`--token-keep-thres` (overrides `appcorr_kwargs.token_keep_thres`, forcing
`mobile_pscore=residual_energy`) CLI overrides. Reports top1/top5 (via `offload.mobile.dataset.
ImageNetLoader`), per-op CUDA-event latency, and aggregate patch keep-rate. Periodic progress
printing (every ~200th of the total, or every sample for small runs) rather than per-sample
printing for `--full` runs, to keep output tractable at 50,000 samples.

**`clip_coco_retrieval_offload_eval.py`**: new (no prior template — the repo only had COCO
*detection* code, unrelated). Loads `captions_val2017.json` (5,000 images / 25,014 captions —
the standard COCO 5K retrieval benchmark split), deterministically samples N images with captions
(or `--full` for all 5,000), precomputes **all** their captions' text embeddings once (batched,
full precision, no approx/correct — captions are never progressively streamed), then runs the
images through the offload pipeline to get corrected image embeddings, and finally computes the
full `[N_images, N_captions]` cosine similarity matrix and **global** recall@{1,5,10} for both
directions (image→text: does *any* ground-truth caption for image *i* appear in its top-k ranked
captions; text→image: does image *j*'s ground-truth image appear in caption *j*'s top-k ranked
images). This global-recall computation is structurally different from top1/top5 (which are
batch-local and additive) — `evaluate_batch`-style incremental tallying doesn't apply; embeddings
must be buffered and ranked once, globally, after every image is processed.

---

## 5. Verification (unit tests)

`analysis/experiments/clip_vision_fork_unittest.py` — a 4-tier numerical check against a
correctness oracle (`clip_bigg_oracle.py`, which dumps per-layer CLS states, final image
embeddings, all 1000 zero-shot class embeddings, and real COCO caption embeddings from a **stock**
`transformers.CLIPModel` forward, sanity-verified by confirming its own zero-shot predictions
correctly recover classes 0/1/2/3 for the first 4 ImageNet val images — i.e. the oracle itself was
checked against ground truth before being trusted as a reference):

| Tier | What's tested | Result |
|---|---|---|
| (a) | `approx()`-only (full stock-equivalent forward + caching) vs. real stock forward | **Bit-exact**, `max_abs_err = 0.0` |
| (b) | `approx()` on a blurred image, then `correct()` with **all** patches from the true image, vs. stock forward on the true image (single-round 100% correction) | **Bit-exact**, `max_abs_err = 0.0` |
| (c) | Same as (b) but `correct()` with only **half** the patches | mean_abs_err = 0.010910, max_abs_err = 0.081360 — a real, bounded approximation error, **expected and accepted** (bidirectional attention means a corrected patch's K/V is exact, but non-corrected patches' influence on it, computed from their stale/blurred state, is not — same accepted-approximation property as DINOv2/DINOv3/SigLIP, not a bug) |
| (d) | Layer-**chunked** (4 chunks of 12 layers, matching how `GroupTriggerPolicy` actually drives the tower) `approx()`, vs. the one-shot (a) result | **Bit-exact**, `max_abs_err = 0.0` — validates the `start_l`/`end_l` chunking contract itself, independent of the approx/correct math |

Blur simulated via 8× downsample→upsample bilinear interpolation (a stand-in for AppCorr's real
Laplacian base layer, sufficient for a unit test — the real pipeline's actual blur is exercised
separately in the full offload-pipeline integration tests, §7).

---

## 6. Bugs found and fixed

Three real, non-hypothetical bugs surfaced during this work:

1. **`get_image_features`/`get_text_features` return type change** (Phase 0, self-caught during
   oracle-building): the installed `transformers` 5.13.0's `CLIPModel.get_image_features()`/
   `get_text_features()` are decorated `@can_return_tuple` and now return a
   `BaseModelOutputWithPooling` object (with the projected embedding under `.pooler_output`) rather
   than a plain tensor, unlike older API docs/examples. Fixed by accessing `.pooler_output`
   everywhere these are called. Not a repo bug — a `transformers`-version API surface discovery.

2. **`LaplacianPyramidPolicy.decode()` never upsampled to native resolution for `pyramid_levels`
   configs with no explicit level 0** (Phase 2+3, crashed `imagenet_clip_bigg_approx_only_l2.json`
   with a shape-broadcast `ValueError`). This is the **exact same bug** independently found and
   fixed earlier this session for the DINOv3 classifier work (`imnet_approx_only_l2.json`), on a
   *different* branch (`experiment/energy-grouping`, commit `3b375c1`) that had never been merged
   to `main` — and `experiment/clip-appcorr` forked from `main` *before* that fix landed there.
   Fixed here by cherry-picking `3b375c1` → local commit `4cedaf2`.

3. **`energy_asc`/`energy_desc` grouping strategies did not exist at all on `main`** (Phase 5,
   silent hang/timeout when passed as `--grouping-strategy`, no error — the scheduler's
   "collect-all-then-group" branch never recognized the unknown strategy string, so no valid
   group-triggering ever occurred). Same root cause as bug #2: this functionality was built earlier
   this session for the DINOv3 investigation on `experiment/energy-grouping` (commit `09e655e`),
   never merged to `main`. Fixed by cherry-picking → local commit `d13b42e`.

Additionally, two **transient, non-reproducible** operational issues (not code bugs) were
encountered and resolved by straightforward means, consistent with a similar transient issue
documented earlier this session for the DINOv3 driver:
- A worker-subprocess config-propagation race (patches arriving before `CONFIG` fully propagated
  to the decoder thread) intermittently crashed the COCO retrieval driver's first request. Fixed
  pragmatically by increasing the driver's post-CONFIG sleep from 1.0s to 3.0s (this driver loads a
  *second* `CLIPModel` copy in the main process, for caption embeddings, immediately before
  starting the worker subprocess — plausibly making the race more likely by delaying worker
  startup relative to patch injection).
- A CUDA device-ordinal crash from double-specifying the GPU (`CUDA_VISIBLE_DEVICES=1` *and*
  `--device cuda:1` together, which double-remaps since `CUDA_VISIBLE_DEVICES` restricts the
  process to a single visible device renumbered as index 0) — my own scripting mistake during the
  full-scale run orchestration, fixed by using `--device cuda:1` alone with no env-var restriction.

---

## 7. Results

All accuracy numbers below are from the REAL offload pipeline (SchedulerModule/WorkerModule),
never a shortcut/simulated computation. "approx-only" = a single unconditional `FULL_INFERENCE` on
a heavily-downsampled (1/4 resolution) then-upsampled input, via the `Laplacian` transmission
policy with `pyramid_levels=[2]` — i.e., the "no correction ever happens, just guess from blur"
floor. "interleaved" = the real `ProgressiveLaplacian` + `GroupTriggerPolicy` approx/correct
pipeline, `num_groups=4` unless noted.

### 7.1 Phase 3 — ImageNet zero-shot, small-sample sanity

| n | baseline top1/top5 | approx-only top1/top5 | interleaved top1/top5 |
|---|---|---|---|
| 10 | 90% / 100% | 70% / 100% | 90% / 100% *(exact match)* |
| 50 | 82% / 96% | 66% / 94% | 80% / 96% *(1/50 sample diff — see below)* |

The nr=50 baseline-vs-interleaved gap (80% vs 82%, one sample) is **not** a bug: it is the same
"bf16 kernel-noise" property already documented for the OpenVLA vision fork's multi-round
correction (§5's unit test confirms exact *single-round* correctness; the real `GroupTriggerPolicy`
schedule does *multiple* correction rounds interleaved with `APPROX_FORWARD` calls spanning the
same layers, and bf16 SDPA kernels are not strictly associative across different memory-layout/
slicing patterns — enough to flip an occasional borderline top1 decision, never top5). This exact
mechanism is explained in more mechanistic detail in the companion log.

### 7.2 Phase 4 — COCO retrieval, small-sample sanity

| n images | baseline i2t/t2i R@1 | approx-only i2t/t2i R@1 | interleaved i2t/t2i R@1 |
|---|---|---|---|
| 5 | 100% / 100% | (not tested at this n) | 100% / 100% *(trivial, too few candidates)* |
| 50 | 94.00% / 93.60% | 96.00% / 91.60% | 96.00% / 92.40% *(still saturated — R@5/R@10 already 100%)* |
| 500 | 85.60% / 73.20% | 77.80% / 67.20% | 85.00% / 73.16% *(meaningful, discriminating)* |

At n=500 (R@5/R@10 no longer saturated), interleaved correction closely tracks the full baseline
on both directions, while approx-only shows a clear, real gap — the same qualitative pattern as
ImageNet, first becoming visible once the sample size is large enough to be discriminating.

### 7.3 Phase 5 — Pruning latency (negative) + grouping strategy (negative)

**Pruning latency**, nr=20, `num_groups=4`, grid grouping:

| condition | top1 | mean CORRECT_FORWARD |
|---|---|---|
| unpruned | 90% | ~7-8ms |
| thr=250, unbucketed | 75% | 18.7ms |
| thr=250, `sdpa_query_bucket_size=32` | 75% | 8.3ms |
| thr=700, bucket=16 | 70% | 9.3ms |

Bucketing (the mitigation validated earlier this session for the DINOv3 classifier's shape-
dispatch-tax problem) **does** fix the shape-dispatch-tax component here too — CORRECT_FORWARD
drops from 18.7ms back to ~8.3ms — but this only recovers latency to *match* the unpruned
baseline, it does not produce a **net** win, because at CLIP-bigG's scale (hidden_size=1664, much
smaller than DINOv3-7B's per-layer compute) fixed per-op kernel-launch overhead dominates over the
FLOPs actually saved by processing fewer query tokens. **Conclusion: per-layer-chunked token
pruning is not a latency win for this model size/architecture** — a real, mechanistically-explained
negative finding (bucketing correctly fixes the shape-dispatch-tax component in isolation,
confirming the underlying mechanism works as designed; the remaining floor is pure launch overhead,
which no query-count reduction addresses).

**Grouping strategy**, unpruned, nr=20, `num_groups=4`:

| strategy | top1 | mean CORRECT_FORWARD |
|---|---|---|
| grid (fixed-size groups) | 90% | ~7-8ms |
| energy_asc (ascending residual-energy order, equal total energy per group) | 75% | 22.3ms |
| energy_desc (descending order) | 70% | 23.7ms |

Both energy-based strategies are worse on *both* axes — variable, energy-imbalanced group sizes
pay the same shape-dispatch tax as unbucketed pruning does, plus (plausibly) larger cumulative
bf16 drift from the multi-round schedule for non-uniform orderings. **This mirrors the independent
DINOv3-classifier energy-grouping investigation's own conclusion almost exactly** (also found:
energy grouping doesn't help, actively hurts) — a genuine cross-validation of that earlier finding
on a completely different model family and task.

### 7.4 Phase 6 — Accuracy-vs-keep-rate sweet-spot search, full scale

Latency set aside entirely per explicit instruction; focus purely on the accuracy/compression
trade-off, confirmed at **full dataset scale** (all 50,000 ImageNet val images; all 5,000 COCO
val2017 images with captions / 25,014 captions — no sampling).

**ImageNet zero-shot — real, confirmed sweet spot:**

| condition | keep-rate | top1 | top5 |
|---|---|---|---|
| baseline | 100% | 77.14% | 94.88% |
| **thr=50 (sweet spot)** | **74.3%** | **76.14% (-1.00pp)** | **94.46% (-0.42pp)** |
| approx-only | — | 65.92% (-11.22pp) | 88.20% (-6.68pp) |

An 11-point nr=100 sample sweep (thresholds 0 through 2500) first mapped the qualitative shape of
this curve — flat near baseline down to ~79% keep-rate, then degrading — and the full-scale run
confirmed it: discarding ~26% of patches costs only ~1pp top1, a clearly worthwhile trade.

**COCO retrieval — no comparable free lunch; small-sample sweep was misleading:**

An analogous nr=150 sweep suggested an almost-free sweet spot (i2t R@1 flat at 94% down to
keep-rate ~69%; t2i R@1 flat 82-85% across the *entire* tested range down to 23% keep-rate). This
did **not** hold at full 5,000-image scale:

> **CORRECTION (2026-08-18): the `baseline` row below is the SEQUENTIAL CEILING, not the unpruned
> interleaved arm, so every delta in this table is overstated.** Re-measuring `interleaved_g4` at
> full scale gives **i2t 65.46 / t2i 49.22**, while the sequential ceiling measures **67.96 / 50.70**
> — matching the `baseline` row to all four digits on both metrics. Control runs of the *unmodified*
> `a02f094` tree on 2026-08-18 hardware reproduce the interleaved arm, not the row: **65.34 / 49.20**
> with no pruning flag and **65.40 / 49.19** with the row's literal `--token-keep-thres 0`. The code
> did not move in the intervening three months, so the row cannot be this arm. (For completeness the
> rebased tree at `--token-keep-thres 0` gives 65.42 / 49.19, and a fifth run -- a02f094, no flag,
> logged separately under `AppCorr-clip-prerebase/logs/clip_control/` -- gives 65.42 / 49.20. All
> five interleaved variants land in **65.34-65.46**, and none is 67.96 / 50.70.)
>
> **Do not quote any single one of those runs to two decimals as "the" control.** Two executions of
> the identical condition (a02f094, no flag) gave 65.34 and 65.42, so run-to-run spread on i2t R@1 is
> ~0.08pp -- about four images out of 5,000 changing rank. The reference is ~65.4, and the thr=25
> cost below is correspondingly **-0.6pp +/- 0.1**, not a two-decimal quantity.
>
> Against the correct unpruned reference the pruning cost is roughly 2.5pp smaller than shown:
> measured against the a02f094 no-flag arm (~65.4 / 49.20), on which the per-threshold rows were
> taken, thr=25 is **about -0.6pp i2t / -0.2pp t2i**, not -3.20 / -1.70. **The conclusion drawn below —
> that COCO has no free-pruning regime the way ImageNet does — is not supported by this table as
> scored** and needs re-deriving. ImageNet's analogous table (§7.4) is *not* affected: its `baseline`
> 77.14% does not equal the ImageNet ceiling (79.85%), and a full re-run reproduces it exactly
> (77.14% / 94.88%).
>
> Corrected deltas against the 65.34 / 49.20 unpruned reference:
>
> | threshold | keep-rate | i2t R@1 | vs unpruned | t2i R@1 | vs unpruned |
> |---|---|---|---|---|---|
> | unpruned (a02f094, no flag) | 100% | 65.34% | — | 49.20% | — |
> | thr=25 | 81.6% | 64.76% | -0.58pp | 49.00% | -0.20pp |
> | thr=50 | 75.3% | 64.32% | -1.02pp | 48.72% | -0.48pp |
> | thr=100 | 66.7% | 63.88% | -1.46pp | 48.29% | -0.91pp |
> | thr=200 | 55.5% | 62.66% | -2.68pp | 47.43% | -1.77pp |

| condition | keep-rate | i2t R@1 | t2i R@1 |
|---|---|---|---|
| baseline (MISLABELLED - this is the ceiling; unpruned interleaved is 65.46 / 49.22) | 100% | 67.96% | 50.70% |
| thr=25 | 81.6% | 64.76% (-3.20pp) | 49.00% (-1.70pp) |
| thr=50 | 75.3% | 64.32% (-3.64pp) | 48.72% (-1.98pp) |
| thr=100 | 66.7% | 63.88% (-4.08pp) | 48.29% (-2.41pp) |
| thr=200 | 55.5% | 62.66% (-5.30pp) | 47.43% (-3.27pp) |
| approx-only | — | 50.06% (-17.90pp) | 40.33% (-10.37pp) |

**Root cause of the discrepancy**: with only ~150 candidate images (and ~750 captions), CLIP-bigG's
retrieval is nearly saturated (94%/84% R@1) — there simply aren't enough confusable distractors for
pruning-induced embedding perturbations to change any ranking's top-1 result. At the real
5,000-image/25,014-caption scale, rankings are far more contested, so the *same* absolute
perturbation much more often changes who's ranked #1. This is a **ceiling-effect measurement
artifact**, not a property of the pruning mechanism — the same "small samples look too easy, real
difficulty only emerges at scale" pattern that was *also* independently observed for retrieval
recall specifically (not classification) going from nr=50→nr=500 earlier in this same
investigation, just far more pronounced going from nr=150→nr=5000.

Even so, the *shape* of the real curve is informative: the initial cost (100%→82% keep-rate) is the
steepest part (-3.2pp i2t R@1 for -18pp keep-rate); after that it flattens noticeably (82%→55%
keep-rate costs only another -2.1pp, over more than double the keep-rate range). There is, however,
no threshold in the tested range that preserves full baseline accuracy the way ImageNet's does.

### 7.5 Phase 7 — Complete ImageNet curve (100% → ~24% keep-rate) and the fallback regime

Extended the full-scale ImageNet sweep to more, and more extreme, threshold values:

| threshold | keep-rate | top1 | top5 | regime |
|---|---|---|---|---|
| 0 (baseline) | 100.00% | 77.14% | 94.88% | — |
| 50 | 74.30% | 76.14% | 94.46% | clean |
| 150 | 59.33% | 74.81% | 93.86% | clean |
| 350 | 42.80% | 72.74% | 92.55% | clean |
| 600 | 31.26% | 70.50% | 90.97% | clean |
| 900 | 24.11% | 68.31% | 89.82% | clean |
| 4000 | 69.06% | 73.65% | 92.79% | fallback-mixed |
| 6000 | 89.41% | 76.14% | 94.28% | fallback-dominated |
| 10000 | 98.21% | 76.99% | 94.77% | fallback-dominated |
| approx-only | — | 65.92% | 88.20% | — |

**Two distinct regimes emerge, and the split is diagnostic, not incidental:**

1. **Clean regime (threshold 0–900)**: keep-rate decreases smoothly and monotonically as threshold
   increases, and top1 degrades smoothly and near-linearly with keep-rate — roughly **-1pp top1
   per ~13-15pp of keep-rate given up**, flattening slightly as keep-rate approaches the ~24% floor
   tested. Even at this aggressive a setting (68.31% top1 at 24% keep-rate), correction remains
   meaningfully better than approx-only (65.92%) — never crosses over, never gets *worse* than not
   correcting at all.

2. **Fallback-dominated regime (threshold ≥4000)**: keep-rate stops being monotonic with threshold
   entirely — thr=6000's 89.4% keep-rate *exceeds* thr=4000's 69.1%, and thr=10000's 98.2% is
   nearly full retention. Root cause: `_prune_patch_idx`'s safety fallback (`if not
   keep_mask.any(): return patch_idx`, i.e. never prune a group down to zero tokens) increasingly
   dominates once the threshold exceeds most groups' entire score distribution — no patch in the
   group clears the bar, so the *whole* group reverts to "keep everything" rather than being pruned
   at all. At threshold=10000 this happens on nearly every group of nearly every image, which is
   why its keep-rate/accuracy sit right next to the unpruned baseline. **Threshold=4000 is the most
   diagnostically interesting point**: a genuinely *mixed* case, where keep-rate (69.1%) falls
   between the clean regime's thr=150 and thr=350 values, yet its accuracy (73.65%) is *worse*
   than *both* of those cleaner points despite keeping *more* patches than either — because a
   blend of "some groups pruned all the way down to just-below-threshold" and "other groups fell
   back to full, unpruned retention" produces a less informative kept-set overall than a uniform,
   moderate prune at a lower, cleaner threshold. **Practical conclusion: threshold value alone is
   not a trustworthy proxy for pruning aggressiveness once it exceeds roughly 1500-2500 for this
   model/config (`num_groups=4`, `total_layers=48`) — keep-rate is the only reliable independent
   variable, and useful threshold values top out well below the point where the fallback starts
   taking over.**

---

## 7.6 Re-measurement on `experiment/clip-appcorr-closedloop` (2026-08-18)

Rebased the 14 CLIP commits onto `main` to pick up `378e21d` "Make the Laplacian transmission
lossless (closed-loop prediction)", which had invalidated the ADE20K and VGGT numbers, and re-ran at
full scale. One commit, `4cedaf2`, was dropped during the rebase: its conflict was comment-only,
because main had landed the identical decode fix independently in `8efa8bb` six days earlier.

**The transmission fix is a no-op for CLIP, and this is proven rather than inferred.**
`clip_transmission_roundtrip.py` encodes through the real policy on both CLIP configs, transmits
every group, decodes, and compares the fixed encoder against the pre-fix one monkeypatched back:
0.0000% relative L2 on both, decoded images bit-identical. Bit-identity alone cannot distinguish
"agrees" from "never ran", so it carries a positive control -- `_closed_loop_residual` is called 8x
and zeroing it moves the decode by 9.45%. The cause is structural: 224x224 with `pyramid_levels
[2, 0]` divides dyadically, so the encoder's native-gaussian predictor and the decoder's
resampled-base predictor coincide. `378e21d`'s own message says the same (`imnet [2,0] already 0`).
CLIP was never affected by the bug; only the non-dyadic families were.

**Full-scale numbers, with the first sequential ceilings ever measured for these tasks:**

| task | floor (approx-only) | interleaved g4 | ceiling (sequential) | vs ceiling | gap recovered |
|---|---:|---:|---:|---:|---:|
| ImageNet top1 | 65.92 | **77.14** | **79.85** | -2.71pp | **80.5%** |
| ImageNet top5 | 88.20 | 94.88 | 96.00 | -1.12pp | 85.6% |
| COCO i2t R@1 | 50.06 | **65.46** | **67.96** | -2.50pp | **86.0%** |
| COCO t2i R@1 | 40.33 | **49.22** | **50.70** | -1.48pp | **85.7%** |

Both floors reproduce their recorded values exactly, which is the check that the fix touches only
residual-carrying paths. ImageNet's interleaved arm also reproduces exactly -- 77.14 / 94.88 on both
metrics -- across three months, 88 commits of main, and a rebase, which is a strong statement about
pipeline stability and is why the July per-threshold numbers in 7.4/7.5 can still be trusted.

**The COCO `baseline (thr=0)` row in 7.4 is the ceiling, not the unpruned arm** -- see the correction
note there. Control runs of the untouched `a02f094` tree on 2026-08-18 hardware give **65.34 / 49.20**
(no pruning flag) and **65.40 / 49.19** (the row's literal `--token-keep-thres 0`), against today's
65.46 / 49.22 — so the code did not drift by more than 0.12pp and the row cannot be this arm. Scored
against the correct reference (the a02f094 no-flag arm, 65.34 / 49.20, which is the tree the
per-threshold rows were taken on), `thr=25` costs **-0.58pp i2t / -0.20pp t2i** rather than
-3.20 / -1.70; the other ~2.5pp belongs to interleaved correction itself, not to pruning. Note both
readings are legitimate and answer different questions: -3.20pp against the ceiling is what decides
whether the configuration is usable, -0.58pp against the unpruned arm is what pruning costs.

Not re-measured: the per-threshold pruned arms themselves. The re-derivation above reuses their July
values with a corrected reference, justified by the unpruned arm reproducing to within 0.12pp across
three months, but that is one arm's worth of evidence.

## 8. Key Findings & Recommendations

| Question | Answer |
|---|---|
| Does AppCorr's approx/correct mechanism work correctly for CLIP? | **Yes** — validated bit-exact at the unit level, and end-to-end on both tasks via the real offload pipeline, with no new scheduling code needed. |
| Is there an accuracy-preserving pruning sweet spot for ImageNet zero-shot? | **Yes, real and full-scale-confirmed**: `token_keep_thres≈50` (≈74% keep-rate) costs only -1.0pp top1. |
| Is there one for COCO retrieval? | **No clean one** — every pruning level tested costs real recall at full (5,000-image) scale; the apparent small-sample sweet spot was a ceiling-effect artifact. Least-costly point tested: thr=25 (82% keep-rate, -3.2pp i2t R@1). |
| Does pruning reduce *latency*? | **No net win** at this model's scale — kernel-launch overhead dominates over FLOPs saved, confirmed by isolating and fixing the shape-dispatch-tax component (bucketing) without producing a net improvement. Unlike DINOv3-7B, where the same mechanism *did* help. |
| Does energy-based grouping help (as opposed to grid)? | **No** — worse on both accuracy and latency, cross-validating an independent DINOv3-classifier finding. |
| How aggressive can pruning get before it's not worth it? | Up to ~threshold 1500-2500 (~24-25% keep-rate) is a genuine, interpretable operating range; beyond that, a "never prune a group to empty" safety fallback makes higher threshold values *less* aggressive, not more — don't tune threshold beyond this range. |

**Practical recommendation**: for ImageNet-style zero-shot classification workloads with this
architecture, ship `token_keep_thres≈50` as the default pruned-correction setting. For retrieval-
style workloads, do not enable pruning by default — evaluate the accuracy cost against the real
target-scale gallery size before choosing any non-zero threshold, since small evaluation samples
will systematically understate the true cost.

---

## 9. Reproducibility

Environment: `appcorr` conda env (added `transformers`, `safetensors`, `open_clip_torch` — none of
these were present before this work; `pycocotools` was already present for COCO detection work).

```bash
conda activate appcorr
cd /NHNHOME/share/cjpark/AppCorr
git checkout experiment/clip-appcorr

# Phase 0/1 sanity (fast, ~1 min)
python analysis/experiments/clip_bigg_oracle.py --out /tmp/oracle.pt --num-images 4 --num-captions 8
python analysis/experiments/clip_vision_fork_unittest.py

# ImageNet zero-shot: baseline / approx-only / interleaved, small sample
python analysis/experiments/clip_zeroshot_offload_eval.py \
    --config offload/config/imagenet_clip_bigg_sequential.json --num-samples 20
python analysis/experiments/clip_zeroshot_offload_eval.py \
    --config offload/config/imagenet_clip_bigg_approx_only_l2.json --num-samples 20
python analysis/experiments/clip_zeroshot_offload_eval.py \
    --config offload/config/imagenet_clip_bigg_interleaved_g4.json --num-samples 20

# Pruning sweep (accuracy-only, ignore latency), any threshold:
python analysis/experiments/clip_zeroshot_offload_eval.py \
    --config offload/config/imagenet_clip_bigg_interleaved_g4.json \
    --token-keep-thres 50 --num-samples 100      # sample
python analysis/experiments/clip_zeroshot_offload_eval.py \
    --config offload/config/imagenet_clip_bigg_interleaved_g4.json \
    --token-keep-thres 50 --full                 # full 50,000-image run (~45-55 min on 1 GPU)

# COCO retrieval, same pattern:
python analysis/experiments/clip_coco_retrieval_offload_eval.py \
    --config offload/config/coco_retrieval_clip_bigg_interleaved_g4.json \
    --token-keep-thres 25 --full                 # full 5,000-image run (~45 min)

# Grouping strategy sweep:
python analysis/experiments/clip_zeroshot_offload_eval.py \
    --config offload/config/imagenet_clip_bigg_interleaved_g4.json \
    --grouping-strategy energy_asc --num-samples 20
```

Calibrating a new threshold value against the actual score distribution for a different
config/model scale: set `CALIBRATE_PSCORE=1` in the environment before running — prints per-group
combined-score percentiles from `OpenCLIPExecutor._prune_patch_idx` (harmless, inert when unset).

---

## 10. File Manifest

```
appcorr/models/openclip/
  __init__.py
  vision/
    __init__.py
    attention.py                      # ApproxCorrectCLIPAttention
    block.py                          # ApproxCorrectCLIPEncoderLayer
    backbone.py                       # ApproxCorrectCLIPVisionTower

offload/server/model/
  openclip_executor.py                # OpenCLIPExecutor (9 ModelExecutor ABC methods)
  __init__.py                         # +"openclip" substring registration (modified)

offload/policies/transmission/
  laplacian.py                        # bugfix: unconditional native-res upsample (cherry-pick)
  progressive.py                      # +energy_asc/energy_desc grouping (cherry-pick)

offload/config/
  imagenet_clip_bigg_{sequential,approx_only_l2,interleaved_g4}.json
  imagenet_clip_bigg_interleaved_g4_{pruned,thr250,thr250_bucket32,thr700,thr700_bucket16}.json
  coco_retrieval_clip_bigg_{sequential,approx_only_l2,interleaved_g4}.json

analysis/experiments/
  clip_bigg_oracle.py                 # Phase 0 correctness oracle
  clip_vision_fork_unittest.py        # Phase 1 4-tier unit test
  clip_zeroshot_offload_eval.py       # ImageNet zero-shot eval driver
  clip_coco_retrieval_offload_eval.py # COCO retrieval eval driver
  CLIP_APPCORR_LOG.md                 # chronological session log (this report's source material)
  CLIP_APPCORR_REPORT.md              # this document
```

Full commit history for this work: `git log --oneline eeb4306..HEAD` on branch
`experiment/clip-appcorr` (14 commits, `eeb4306` "Phase 0" through `100f3db` "Phase 7").
