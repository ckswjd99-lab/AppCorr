# Handoff: pi0-FAST progressive-vision (partial-token ViT+LLM) on LIBERO

You are continuing work in the **AppCorr** repo (`/NHNHOME/share/cjpark/AppCorr`, branch
`libero-eval-fixes`). The overarching project ports AppCorr's *progressive offload* technique
(approx on a low-res base → correct only important tokens) across VLA models. OpenVLA and OpenVLA-OFT
are already done; the current model is **pi0-FAST** (lerobot's `PI0FastPolicy` = PaliGemma[SigLIP
So400m ViT + Gemma-2B LLM] + FAST DCT action tokens). The immediate task the user set:

> Implement DINOv3-style **partial-token correct** on pi0-FAST's **ViT AND the vision-token LLM**.
> pscore = **residual × avg-attention computed inside the SigLIP vision encoder** (NOT the LLM).
> One approx + one correct. Measure LIBERO success. **The LLM must ALSO do partial correct** (only
> the selected vision tokens' K/V get updated, not the whole prefix).

This is essentially DONE and validated; you are in the **evaluation** phase. Read this whole doc.

---

## Environment (critical — get this exactly right)
- conda env **`pi0fast`**: `source ~/anaconda3/etc/profile.d/conda.sh; conda activate pi0fast`.
  transformers==4.57.1, lerobot 0.4.4, torch 2.10+cu128 (Blackwell). torchcodec/scipy present.
- LIBERO via `PYTHONPATH=/NHNHOME/share/cjpark/openvla_deps/LIBERO:$PYTHONPATH`.
- **Rendering**: B200 has NO Vulkan; LIBERO/MuJoCo only via `MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=2
  MUJOCO_EGL_ALLOW_ANY_DEVICE=1`. **≥3 concurrent EGL sims deadlock** → run 1 task/process
  (`--env.task_ids=[N]`); up to 2 processes OK (one per GPU, both render on EGL device 2).
- Always set `TORCHDYNAMO_DISABLE=1` (triton inductor cache crashes otherwise).
- The pi0-FAST LIBERO eval entrypoints now call
  `analysis/experiments/pi0fast_libero_runtime.py` before importing LeRobot. It automatically
  applies the LIBERO sibling path and the four rendering/runtime variables above. Explicit
  `LIBERO_ROOT`, `MUJOCO_GL`, `MUJOCO_EGL_DEVICE_ID`, `MUJOCO_EGL_ALLOW_ANY_DEVICE`, and
  `TORCHDYNAMO_DISABLE` values override its defaults.
- 2× B200 (GPU 0/1), 183 GB each. Model load ~90 s; each 10-episode rollout ~30-40 min (autocast).
- Scratchpad for temp files/logs: `/tmp/claude-3092/-NHNHOME-share-cjpark-openvla/b626e184-fa94-416c-8644-34697e8e7372/scratchpad`.
- Datasets: eval env is the LIBERO SIM (via lerobot). For OFFLINE numeric checks use dataset
  `HuggingFaceVLA/libero` (keys `observation.images.image` / `image2`, state 8-dim, action 7-dim).

## TWO bugs that cost enormous time — do not rediscover them
1. **Gemma double-scaling** (transformers 4.57 × lerobot 0.4.4). `GemmaModel.forward` scales
   inputs_embeds by √hidden, and lerobot's `embed_prefix_fast` ALSO pre-scales language/FAST embeds
   → they collapse (teacher-forced CE 316, all-`<bos>`/constant action). **The CORRECT fix touches
   ONLY the language path**: `install_gemma_scaling_fix()` (in
   `appcorr/models/pi0fast/progressive_model.py`) patches `PI0FastPaliGemma.embed_language_tokens`
   to pre-divide by √hidden so lerobot's `*√dim` cancels; IMAGES are untouched (they come from
   `get_image_features`, already /√hidden, so GemmaModel's single scaling is right for them).
   An earlier WRONG fix divided the whole inputs_embeds inside GemmaModel → also shrank the image
   tokens 45× → weak vision → the arm approached objects but grasped imprecisely → **0% LIBERO**
   while the model card is **82.5%**. Always import/call `install_gemma_scaling_fix()` before any
   pi0-FAST forward.
2. **Image keys.** obs must use `observation.images.image` / `image2` (this checkpoint's
   `config.image_features`), NOT `base_0_rgb`/`left_wrist_0_rgb`. Wrong keys silently pad the real
   cameras as "missing" (mask=0) → model sees no vision → constant action. (The lerobot LiberoEnv
   already maps agentview→image, wrist→image2, so in the eval harness this is handled.)

## Established baselines (libero_spatial, task 0, official lerobot-eval + fix, 10 episodes)
- **stock = 30%** (== model's real baseline; card 82.5% is the all-task average, task 0 is a small
  sample). `lerobot/pi0fast-libero` is a GOOD checkpoint — do NOT call it weak.
- **progressive vision-only @100% correct = 30%** (rollout-lossless; validated).
- **progressive vision-only @50% NAIVE top-to-bottom = 0%** (top-half correct leaves the object
  region, which is low in the image, at low res → can't grasp). This is what motivated pscore.

## Key files (all committed on `libero-eval-fixes`)
- `appcorr/models/pi0fast/siglip_vision.py` — ApproxCorrect SigLIP fork. `approx_forward(...,
  pscore=True)` computes, per patch: `avg_attn` (softmax(qk^T) mean over heads+queries+layers =
  DINOv3 `patch_attn_prob`) and `residual_mag` (L2 norm of the per-block output update, mean over
  layers). `get_pscore()` = residual_mag × avg_attn (ProgVFM contrib_i). `correct_forward(pixel,
  patch_idx, ...)` recomputes ONLY `patch_idx`.
- `appcorr/models/pi0fast/gemma_prefill_layer.py` — ApproxCorrect Gemma fork (bit-exact vs stock
  GemmaDecoderLayer). `.approx(causal=False, attn_mask=...)` and `.correct/.prefill(..., attn_mask=...)`
  take an explicit additive mask for the bidirectional-prefix + empty-cam/pad masking.
- `appcorr/models/pi0fast/progressive_model.py` — `Pi0FastProgressiveModel`.
  - `install_gemma_scaling_fix()` (module fn) — the fix above.
  - `__init__(checkpoint, device)` loads + casts float32 (standalone/offline use).
  - `from_policy(pol, device)` — wrap lerobot-eval's already-loaded policy, NO reload, keeps its
    autocast-bf16 precision (used by the eval harness).
  - `predict_action_partial_token(obs, keep, base_factor, correct_text)` → `_partial_from_batch(b, ...)`:
    THE method. Per real image: SigLIP base approx (+pscore) → top-`keep` patch correct. Then LLM:
    bidirectional **approx** on base vision features + text → **correct** only {selected vision
    tokens} ∪ {all text tokens (permanent group)} via the Gemma fork (non-selected vision keep base
    LLM K/V) → `_generate_fast_from_cache` (incremental FAST decode) → `_detok`.
  - **VERIFIED: keep=1.0 (+correct_text) == stock lerobot EXACTLY offline in float32
    (|action diff| = 0.000000)**; keep=0.5 gives an approximate action.
  - Generation gotchas already fixed: mask padding/empty-cam keys during decode; FAST-token RoPE
    positions use cumsum-based (`p_valid = pad.sum()`), NOT the padded length P.
- Eval launchers in `analysis/experiments/`:
  - `pi0fast_libero_official_eval.py` — STOCK / vision-only progressive (embed_image swap). Parity
    by construction (STOCK mode == official eval).
  - `pi0fast_libero_partial_token_eval.py` — **the ViT+LLM partial-token eval**: patches
    `PI0FastPolicy.predict_action_chunk` → `_partial_from_batch`. Env: `TASK_ID N_EP PTC_KEEP
    PTC_BASE PTC_CORRECT_TEXT`. arg1 = output dir. Uses `--policy.use_amp=true` (REQUIRED, else
    SigLIP Float-vs-BFloat16 error) and `--env.task_ids=[N]`.
  - `pi0fast_libero_rollout_eval.py` — an older custom single-env rollout harness (360 res).
    Superseded by the official-eval launchers for parity, but works.
  - `pi0fast_progressive_eval.py` — OFFLINE teacher-forced CE-vs-recompute (no sim).

## What is RUNNING / just finished (check these first)
Background evals (partial-token ViT+LLM, libero_spatial task 0), logs in scratchpad:
- `scratchpad/ptc_v1_k1.log` — **keep=1.0, 3 ep** (parity check). **FINISHED: 0/3 = 0%.** This does
  NOT match stock (30%) → the bf16 drift below is MATERIAL, not hypothetical. Treat resolving it as
  step 0 before trusting any partial-token rollout number.
- `scratchpad/ptc_v1_k50.log` — **keep=0.5, 10 ep** (measurement, still running). At last look 0/2.
  Given keep=1.0 is already 0/3, the keep=0.5 number is not yet interpretable — fix the bf16/parity
  issue first, then re-run.
Check with, e.g.:
```
tr '\r' '\n' < scratchpad/ptc_v1_k50.log | grep -oE "[0-9]+/10 \[[^]]*running_success_rate=[0-9.]+%" | tail -3
```
(look for `K1_EXIT` / `K50_EXIT` for completion). Launch pattern (GPU per process):
```
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=2 MUJOCO_EGL_ALLOW_ANY_DEVICE=1 TORCHDYNAMO_DISABLE=1 \
CUDA_VISIBLE_DEVICES=0 TASK_ID=0 N_EP=10 PTC_KEEP=0.5 \
PYTHONPATH=/NHNHOME/share/cjpark/openvla_deps/LIBERO:$PYTHONPATH \
python analysis/experiments/pi0fast_libero_partial_token_eval.py <outdir>
```

## OPEN ISSUE #0 (BLOCKER — fix before trusting numbers)
**keep=1.0 rollout = 0/3, but stock = 30% → the partial forward diverges from stock under
autocast-bf16.** Offline in FLOAT32 the two are bit-exact (|action diff| = 0.000000), so the LOGIC
is correct; this is a precision/path artifact of running approx-then-correct (extra recompute → more
bf16 rounding) through the forks under autocast, vs stock's single lerobot forward. Options to try,
in order:
  1. Run the partial forward in **float32** (like the standalone `__init__` path) instead of
     autocast-bf16 — the cleanest fair comparison. Either cast the eval policy to float32 (as
     `to_bfloat16_for_selected_params("float32")`) and drop `--policy.use_amp`, or wrap
     `_partial_from_batch` to upcast. NOTE without autocast you must avoid the SigLIP
     Float-vs-BFloat16 error (that's exactly why the standalone `__init__` casts the whole model to
     float32). Re-verify keep=1.0 rollout ≈ stock 30% before reporting keep<1.0.
  2. If float32 is too slow, reduce the approx→correct redundancy so fewer bf16-lossy ops accumulate.
  3. Sanity-check offline UNDER bf16 (not just float32): reproduce the smoke test
    (`scratchpad/ptc_llm_smoke.py`) with the model in bf16 + autocast and measure |keep=1.0 − stock|;
     if it's already >0 there, the drift is confirmed pre-rollout and you can iterate fast without
     the sim.
Do not report keep=0.5 as "the partial-token result" until keep=1.0 rollout matches stock.

## Suggested next steps
1. Collect the keep=0.5 (and ideally keep=0.25) partial-token success rates; compare to stock 30%
   and to the naive-top-half 0%. The hypothesis: pscore selection (importance = residual×avg_attn)
   should beat naive top-half at the same recompute budget.
2. If keep=1.0 rollout drifts from stock, decide on the float32-vs-bf16 question above so the
   stock-vs-partial comparison is fair.
3. Optionally sweep keep and/or run more tasks/episodes for tighter numbers (parallel across 2 GPUs,
   1 task each, EGL device 2).
4. Report: recompute-rate (ViT patch fraction + LLM {selected vision + text}/full) vs success rate.

## Memory / notes
Persistent memory files (auto-loaded) already capture the double-scaling fix + offline-eval recipe:
`~/.claude/projects/-NHNHOME-share-cjpark-openvla/memory/project_pi0fast_double_scaling_bug.md` and
`project_vla_progressive_multimodel.md`. Commit incrementally on `libero-eval-fixes`; end commit
messages with the project's Co-Authored-By / Claude-Session trailers.

Do not re-litigate the checkpoint quality (82.5%, good) or the two bugs above — they are settled.

---

## 2026-07-25 continuation: full-correction parity restored

This section supersedes **OPEN ISSUE #0** and the stale AMP-based rollout numbers above.

### Root cause and fix

- The supported comparison path is now shared **float32 weights with AMP disabled** for both stock
  and partial execution. `PTC_PRECISION=float32` is the official local parity setting;
  `amp_bf16` remains available only as a diagnostic mode.
- Float32 alone was not sufficient for every rollout input. One captured batch diverged at FAST
  token 8 even though corrected SigLIP features and direct corrected Gemma active-token outputs
  appeared exact.
- The remaining cause was operation ordering. Stock computes image embeddings as
  `(projector / sqrt(hidden)) * sqrt(hidden)` and language embeddings through the patched
  `embed_language_tokens`, LeRobot scaling, then Gemma scaling. Skipping these mathematically
  redundant round trips changes float32 rounding by about `1.5e-5` at the prefix and can cross a
  later FAST argmax boundary.
- `Pi0FastProgressiveModel._proj_raw()` and `_language_at_layer_scale()` now preserve that exact
  ordering. FAST decoding also uses LeRobot's mask builder and always emits the stock fixed token
  count, including the post-EOS tail.

### New diagnostics and evaluation controls

- `analysis/experiments/pi0fast_partial_token_parity.py` compares stock and full-correction FAST
  token IDs before detokenization. `--diagnose` captures stock prefix/layer/lm-head inputs and
  compares them against the progressive caches. It accepts `--batch-file` for replaying a rollout
  mismatch.
- `analysis/experiments/pi0fast_libero_partial_token_eval.py` now supports:
  - `PTC_MODE=stock|partial`
  - `PTC_PRECISION=float32|amp_bf16`
  - `PTC_COMPARE_STOCK=1`
  - `PTC_DUMP_MISMATCH=/path/batch.pt`
  - `PTC_STOP_ON_MISMATCH=1`
- The launcher prints actual ViT and LLM query recompute counts once per run.
- `Pi0FastProgressiveModel.last_recompute_stats` exposes the same counts programmatically.

### Validation completed

- Offline float32, dataset frames 0/7/14: exact FAST tokens and exact actions.
- Previously failing captured rollout batch `/tmp/codex_pi0fast_k1_mismatch.pt`: exact after the
  scaling-order fix.
- Official compare rollout, keep=1.0: every action chunk through success at step 92 was exactly
  equal to stock.
- Matched 3-episode float32 runs on `libero_spatial`, task 0:
  - stock: **2/3 (66.7%)**, outcomes `[True, True, False]`
  - partial keep=1.0: **2/3 (66.7%)**, outcomes `[True, True, False]`
  - partial keep=0.875: **1/3 (33.3%)**
  - partial keep=0.75: **1/3 (33.3%)**
  - partial keep=0.5: **0/3**
  - partial keep=0.25: **0/3**
- AMP/bfloat16 full-correction is intentionally not claimed lossless: the checked sample had action
  max diff `0.080351114` and mean diff `0.01865156`.

The small 3-episode success rate is a matched parity smoke, not a replacement for the established
10-episode baseline. Output directories are:

```text
/tmp/codex_pi0fast_stock_fp32_ep3
/tmp/codex_pi0fast_partial_fp32_k1_ep3_retry
/tmp/codex_pi0fast_partial_fp32_k875_ep3
/tmp/codex_pi0fast_partial_fp32_k75_ep3
/tmp/codex_pi0fast_partial_fp32_k50_ep3
/tmp/codex_pi0fast_partial_fp32_k25_ep3
```

### Current next step

The follow-up keep=0.9375 and keep=0.96875 runs both failed the first stock-success initial state
at the 280-step limit, so they were stopped rather than spending another two episodes on settings
that could no longer match keep=1.0's outcome vector. This non-monotonic small-sample behavior means
there is no promising compressed setting to promote to a 10-episode run yet; token ranking or
correction scheduling needs improvement first. Do not use the old `scratchpad/ptc_v1_*` AMP
numbers in comparisons.

---

## 2026-07-25 continuation: Gemma vision-attention pscore

The partial-token path now has an optional downstream-aware score mode:

- `score_mode="vit"` preserves the prior SigLIP-only ranking.
- `score_mode="vit_llm_vision"` runs the Gemma approximate prefill before selection and collects
  layer-averaged received-attention from all valid vision queries to all valid vision keys.
- The attention softmax uses the real full-prefix key denominator and additive mask, so language
  keys influence normalization.
- ViT pscore and LLM vision-attention are mean-normalized and fused geometrically in log space.
  `llm_vision_weight=0` explicitly returns the legacy ViT score to preserve its ranking.
- The Gemma collector accepts named query groups; both `"vision"` and `"language"` are now wired
  into the progressive selector.

Interfaces:

```text
PTC_SCORE_MODE=vit|vit_llm_vision|vit_llm_language
PTC_LLM_VISION_WEIGHT=1.0
PTC_DUMP_PSCORE=/tmp/components.pt
--score-mode vit|vit_llm_vision|vit_llm_language
--llm-vision-weight 1.0
--score-report-keep 0.5
```

Validation:

- Synthetic masked-softmax test passes exactly for both `"vision"` and future `"language"` query
  groups: `analysis/experiments/test_pi0fast_llm_attention_score.py`.
- Dataset frames 0/7/14: both `vit` and `vit_llm_vision` keep=1.0 produce exact stock FAST tokens
  and actions.
- Official keep=1.0 compare rollout: all 10 action chunks exact; success at step 92.
- At report keep=0.5, ViT-vs-LLM score Pearson correlation was approximately 0.43–0.56,
  Spearman 0.57–0.61, and fused-vs-ViT top-k overlap 80–86%.
- Single-frame timing (separate runs; treat as directional): legacy partial 2858 ms, fused partial
  3099 ms, about 8% collector overhead. Both reported the same 695.9 MiB peak allocation delta.

Matched rollout screening with `llm_vision_weight=1.0` did not improve task 0:

- keep=0.5: **0/3**, same as ViT-only.
- keep=0.75: **0/3**, versus ViT-only 1/3.
- keep=0.875: first stock-success state failed **0/1**, so no longer run was started.

The implementation and diagnostics are ready, but weight 1.0/all-layer vision-attention is not a
promising rollout policy. The language-query alternative below was implemented and screened next.

## 2026-07-25 continuation: language-query attention

`score_mode="vit_llm_language"` collects attention received by each real vision key from valid
language-prefix queries: non-padding instruction tokens plus the appended BOS. It preserves the
full bidirectional prefix key axis and real additive mask in the softmax denominator, averages
heads, valid language queries, and Gemma layers, then fuses the result with the SigLIP pscore using
the existing `llm_vision_weight`.

The available modes are:

- `vit`: SigLIP-only ranking.
- `vit_llm_vision`: vision-query -> vision-key attention.
- `vit_llm_language`: language-query -> vision-key attention.

Rollout:

```bash
PTC_SCORE_MODE=vit_llm_language PTC_LLM_VISION_WEIGHT=1.0 PTC_KEEP=0.5 \
    python analysis/experiments/pi0fast_libero_partial_token_eval.py /tmp/pi0fast_language50
```

Validation:

- Dataset frames 0/7/14 at keep=1.0: exact stock FAST tokens and actions for all three frames.
- At diagnostic keep=0.5, ViT-vs-language-attention Pearson correlation was 0.286–0.344,
  Spearman 0.438–0.498, and fused-vs-ViT top-k overlap 75.0–78.9%.
- Official task-0 keep=0.5 rollout: **1/3 (33.3%)**, outcomes `[True, False, False]`.
  This improves over both ViT-only and all-layer vision-query keep=0.5 (**0/3**), while remaining
  below the matched stock/full-correction outcome (**2/3**).
- First rollout call used 151 valid language queries against 512 real vision keys. The exact query
  count varies with instruction length.

Output: `/tmp/codex_pi0fast_llmlanguage_k50_ep3`.

## 2026-07-25 continuation: 4-group block-causal Gemma

A separate `PTC_MODE=block_causal` path intentionally gives up stock-prefix exactness and does no
token pruning or pscore selection:

1. Run the low-resolution SigLIP base.
2. Split each camera's 256 patches into four contiguous top-to-bottom groups.
3. At round `g`, cumulatively correct all arrived SigLIP patches. The final round is therefore
   bit-exact with the stock vision tower.
4. Prefill only the newly arrived positions through Gemma. Queries in group `g` attend
   bidirectionally to all keys in groups `0..g`, but never to future vision groups.
5. Prefill valid language tokens plus appended BOS once as the final bidirectional block over all
   real vision keys and the language block.
6. Reuse the resulting prefix K/V cache for causal FAST decode.

For two real cameras, each LLM vision block has 128 tokens. The first rollout call reported:

```text
vision block sizes = [128, 128, 128, 128]
SigLIP correction queries = 1280  # cumulative: 2 cameras * (64+128+192+256)
Gemma vision tokens = 512         # each exactly once
Gemma language+BOS tokens = 151   # each exactly once
Gemma valid prefix tokens = 663
```

Validation and measured result:

- The deterministic grouped-prefill test matches an explicit block-lower-triangular attention mask
  exactly.
- Final progressive SigLIP features for both real cameras are bit-exact with stock, confirming that
  downstream divergence comes from the intended Gemma mask change.
- Dataset frames 0/7/14 all differ from stock actions; mean absolute action differences were
  `0.342`, `0.463`, and `0.510`.
- Three-frame timing was effectively flat: stock `2005.8 ms`, block-causal `1997.0 ms` (about 0.4%
  faster, within run noise). Peak allocation delta increased from `264.6 MiB` to `387.6 MiB`.
- Official LIBERO spatial task 0 rollout: **0/3**, outcomes `[False, False, False]`, versus the
  matched stock/full-correction result **2/3**.

Command:

```bash
PTC_MODE=block_causal PTC_NUM_GROUPS=4 PTC_BASE=4 \
    python analysis/experiments/pi0fast_libero_partial_token_eval.py \
    /tmp/pi0fast_block_causal
```

Output: `/tmp/codex_pi0fast_blockcausal_g4_ep3`.

## 2026-07-25 continuation: L2-to-L0 visual-residual pscore

The partial-token path now also accepts `score_mode="visual_residual_attn"`:

```text
visual_energy_i = sum over RGB and the 14x14 pixel patch of (L0 - upsample(L2))^2
pscore_i = visual_energy_i * SigLIP layer-averaged received-attention_i
```

Here L0 is the exact 224x224 pi0-FAST input and L2 is the progressive model's bilinear
1/4-resolution base restored to 224x224. This mode does not use Gemma attention. It corrects the
selected SigLIP tokens and the same Gemma vision positions exactly like the other partial modes.

Validation:

- The deterministic patch-energy test matches an explicit 2x2-grid calculation.
- On the saved task-0 rollout batch, keep=1.0 produces bit-exact stock FAST tokens and actions.
- At diagnostic keep=0.5, its mask overlaps the legacy hidden-residual ViT pscore mask by 78.1% and
  79.7% for the two real cameras.
- The first official rollout call dumped two unique 128/256 selections and verified
  `combined == visual_residual_energy * siglip_attention` exactly.
- Official LIBERO spatial task-0 keep=0.5 rollout: **0/3**, outcomes
  `[False, False, False]`. This matches ViT-only keep=0.5 (0/3) and is below language-query
  keep=0.5 (1/3) and matched stock/full correction (2/3).

Run:

```bash
PTC_MODE=partial PTC_SCORE_MODE=visual_residual_attn PTC_KEEP=0.5 PTC_BASE=4 \
    python analysis/experiments/pi0fast_libero_partial_token_eval.py \
    /tmp/pi0fast_visual_residual50
```

Measured output: `/tmp/codex_pi0fast_visualres_l2l0_k50_ep3`.

## 2026-07-25 continuation: visual residual times language-query attention

`score_mode="visual_residual_llm_language"` directly combines the L2-to-L0 visual residual with
downstream language demand:

```text
visual_energy_i = sum over RGB and the 14x14 pixel patch of (L0 - upsample(L2))^2
language_attn_i = mean over Gemma layers, heads, and valid language-prefix queries of attn(q, key_i)
pscore_i = visual_energy_i * language_attn_i
```

The Gemma attention calculation keeps the full bidirectional prefix and its real additive mask in
the softmax denominator. Selection remains per real camera, and a 50% keep rate selects 128 unique
patches from each camera. A masked third image produces no score or selection.

Validation:

- The saved task-0 rollout batch at keep=1.0 produces bit-exact stock FAST tokens and actions.
- The first keep=0.5 call verified `combined == visual_residual_energy * llm_attention` with zero
  maximum error for both real cameras.
- Both cameras selected 128 unique indices. Relative to the legacy ViT pscore mask, top-50%
  overlap was 79.7% and 78.9%.
- Official LIBERO spatial task-0 keep=0.5 rollout: **0/3**, outcomes
  `[False, False, False]`. This does not improve on visual-residual × SigLIP-attention or ViT-only
  selection (both 0/3), and is below fused language-query selection (1/3) and matched stock/full
  correction (2/3).

Run:

```bash
CUDA_VISIBLE_DEVICES=0 TASK_ID=0 N_EP=3 PTC_MODE=partial PTC_PRECISION=float32 \
PTC_KEEP=0.5 PTC_BASE=4 PTC_SCORE_MODE=visual_residual_llm_language \
PTC_SELECTION_VIDEO=1 \
    /home/nxclab/anaconda3/envs/pi0fast/bin/python \
    analysis/experiments/pi0fast_libero_partial_token_eval.py \
    /tmp/pi0fast_visual_residual_llm_language50
```

`PTC_SELECTION_VIDEO=1` (the partial-mode default) records the actual two pi0 policy-input camera
tensors side by side. Recomputed patches stay bright with cyan borders; approximate patches are
dimmed. Each policy decision is repeated for the FAST action horizon so the diagnostic MP4 has the
same frame count and FPS as the official simulator video.

Measured output:
`/tmp/codex_pi0fast_visualres_llmlanguage_k50_ep3`. All three original and all three selection
videos are 280 frames at 80 FPS; selection videos are 448x262 and simulator videos are 256x256.

### Base-resolution comparison

The same `visual_residual_llm_language`, keep=0.5 task-0 rollout was repeated with
`PTC_BASE=2`: downsample the 224x224 policy input to 112x112 instead of 56x56, then bilinearly
restore it before the approximate SigLIP pass.

- Official rollout: **0/3**, outcomes `[False, False, False]`, the same as `PTC_BASE=4`.
- First-call mean visual residual energy fell from 3.8833 to 1.3647 for camera 1 and from 4.8991
  to 1.8486 for camera 2.
- Despite the smaller residual, the selected top-128 masks overlapped the factor-4 masks by 97.7%
  and 95.3%. The ranking therefore changed very little.
- The saved pscore again verified the direct energy-attention product with zero maximum error.
- All original and selection videos are 280 frames at 80 FPS.

Measured output: `/tmp/codex_pi0fast_visualres_llmlanguage_base2_k50_ep3`.

The official LeRobot harness is seeded by default: global seed 1000 and environment reset seeds
1000, 1001, and 1002 for these three episodes. The checkpoint uses FAST temperature 0.0, so action
tokens are selected by argmax rather than multinomial sampling. Runs are therefore intentionally
deterministic for matched software/hardware, although the harness does not request strict
bitwise-deterministic CUDA execution (`cudnn.benchmark` and TF32 are enabled).

### Deterministic rerun and OpenVLA comparison

The earlier best compressed pi0-FAST setting was rerun unchanged:

```text
score_mode=vit_llm_language, llm_vision_weight=1.0
keep=0.5, base_factor=4, precision=float32
LIBERO spatial task 0, seeds 1000/1001/1002
```

It reproduced the prior result exactly: **1/3**, outcomes `[True, False, False]`. The first episode
succeeded at step 152; the other two reached the 280-step limit. Output:
`/tmp/codex_pi0fast_llmlanguage_k50_ep3_rerun`. Original and selection-overlay videos were saved;
the successful episode has 152 frames and both failed episodes have 280 frames, all at 80 FPS.

For context, the previously successful OpenVLA progressive configuration was also rerun on task 0
initial states 0, 1, and 2:

```text
schedule=interleaved, num_groups=4, grouping=rank
coverage=1.0, base_factor=4, max_steps=220
```

OpenVLA succeeded **3/3**, at 115, 107, and 98 steps. This is the full-coverage progressive path:
all four spatial groups arrive and are corrected; it is not a 50% partial-token pruning setting.
The prior larger measurements for this path were 38/50 and 36/50 on two shards.
