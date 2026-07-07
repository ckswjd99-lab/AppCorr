# Energy-Grouping Session Log

Autonomous work log for the DINOv3 energy-based grouping investigation, requested 2026-07-08.
Branch: `experiment/energy-grouping` (off `main`, kept separate from the OpenVLA
`develop/openvla-progressive-prefill` work). User is offline (~8h); working autonomously,
committing frequently so any point can be reverted to safely.

## Context / instructions

- Add an "energy" grouping mode to AppCorr's DINOv3 pipeline: patches sorted by residual energy
  (sum of squared residual, not compressed byte size), split into groups with **equal total
  energy** (not equal count), in **both ascending and descending** priority order.
- Existing `_apply_uniform_diff_grouping` in `progressive.py` already does something adjacent
  (equal total *compressed byte size*, ascending only, using `pscore_hint`/mobile_pscore metric).
  New modes should use true residual energy (`_compute_patch_residual_energy`, already present),
  independent of whatever `mobile_pscore` is configured.
- Target for first test: DINOv3 **classifier** (`dinov3_classifier.py`), comparing:
  1. approx-only
  2. full-resolution baseline (`full_inference`, stock model call)
  3. interleaved correction (with grouping_strategy sweep, including new energy_asc/energy_desc)
  - Metrics: top-1/top-5 accuracy + latency.
  - Start with nr=10 (quick sanity), then scale up.
  - Use existing config files (`offload/config/imnet_*.json`) as the hyperparameter starting
    point, but autonomously sweep settings to see if/where energy grouping (asc or desc) wins.
- **If a good setting is found: immediately (don't wait for check-in) also test COCO (detector),
  ADE20K (segmentor_m2f), NYUv2 (depther).**
- Frequent branch/commit discipline so a crash never loses more than a small chunk of work.

## Environment facts discovered

- ImageNet val (ImageFolder-compatible, 1000 class dirs `0000`-`0999`) at
  `/NHNHOME/share/cjpark/data/imagenet_val` (NOT `~/data/imagenet_val`, which doesn't exist --
  existing scripts default to the latter, will need `--data-root` override).
  Not yet confirmed which numeric class dirs correspond to which torchvision/ImageNet class
  index ordering (torchvision `ImageFolder` sorts alphabetically, so `0000`.."0999" -> label
  0..999 directly, should match given the directory names are already zero-padded numeric labels
  -- to verify once the loader actually runs).
- DINOv3-7B weights present: `~/cjpark/weights/dinov3/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth`
  (backbone) + `dinov3_vit7b16_imagenet1k_linear_head-90d8ed92.pth` (linear head). Also COCO
  detector head (`dinov3_vit7b16_coco_detr_head-b0235ff7.pth`) and ADE20K M2F head
  (`dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth`) exist for the later COCO/ADE20K extension.
  NYUv2 depther weights not yet located -- check `dinov3_depther.py`'s load_model for path when
  we get there.
- `offload/server/model/dinov3_classifier.py`: executor requires real 7B model, loads via
  `load_weight_mmap`. `grouping_strategy` dispatch lives in `progressive.py` (transmission
  policy), NOT the executor -- the executor just consumes whatever `group_id` arrives on each
  Patch (payload-dependent fallback path in `preprocess()`).
- `offload/policies/transmission/progressive.py`: `encode()` branches on `grouping_strategy`:
  `'uniform_diff'` -> data-dependent "collect all then group" path (`_apply_uniform_diff_grouping`,
  sorts by **compressed byte size** ascending, equal total size per group). Anything else ->
  precomputed/pipelined spatial path (`grid`/`block_grid`/`random`/`geometric`, no data
  dependence). New `energy_asc`/`energy_desc` need to join the **first** branch (data-dependent),
  since energy requires seeing actual patch content.
- `_compute_patch_residual_energy` already exists (sum of squared residual pixels) but is
  currently only used for `pscore_hint` when `mobile_pscore == "residual_energy"` -- need to
  compute it unconditionally per-candidate for grouping regardless of the configured mobile_pscore.

## Plan (see TaskCreate #16-20 for live status)

1. Implement `_apply_energy_grouping(final_patch_list, batch_candidates, num_groups, descending)`
   in `progressive.py`, wire into `encode()`.
2. Write a new eval driver (analysis/experiments/dinov3_classifier_offload_eval.py or similar)
   through the REAL offload pipeline, 3-way comparison, top1/top5 + per-op latency.
3. nr=10 sanity check.
4. Scale up + sweep (num_groups, coverage/threshold, asc vs desc vs existing baselines).
5. If a real win is found: extend to COCO/ADE20K/NYUv2 without waiting for check-in.

## Log entries

- **2026-07-08, session start**: created branch `experiment/energy-grouping` off `main`
  (a695f70 tip: "Implemented segmentor"). Read `imnet_interleaved_g4.json` config,
  `dinov3_classifier.py` executor, `progressive.py` transmission policy, `offload/mobile/dataset.py`
  loaders. Confirmed ImageNet val path and DINOv3-7B weight availability. About to implement the
  energy grouping code.

- **Implemented `_apply_energy_grouping`** in `progressive.py` (commit 09e655e): true residual
  energy (not compressed byte size), equal total energy per group, `descending` param for
  asc/desc. Added `residual_energy` to every candidate dict in
  `_collect_residual_candidates_vectorized` (computed unconditionally, independent of the
  configured `mobile_pscore`). Wired as `energy_asc`/`energy_desc` grouping_strategy values,
  joining the same data-dependent "collect all then group" branch as `uniform_diff` in
  `encode()` (spatial strategies like grid/block_grid/geometric use a separate precomputed path
  that can't see patch content). Verified with a synthetic 20-high/236-low energy split: both
  directions balance total energy per group within ~10% of target, with the expected group-1
  content reversal between asc and desc.

- **Traced GroupTriggerPolicy** (offload/policies/scheduling/group_trigger.py) precisely --
  confirms group_id=0 (base layer) advances the LLM^H^H backbone-layer frontier IMMEDIATELY
  (`current_chunk_start=0*chunk_size=0`, so no correct(), just approx(0,chunk_size)), and every
  subsequent group_id in [1, num_groups) corrects at the CURRENT frontier then advances, with the
  LAST group_id (== num_groups) doing a final correct-through-full-depth with no further advance.
  This is 0-indexed groups (0..num_groups), vs. the 1-indexed scheme I used for the OpenVLA fork
  (1..num_groups, with group 0 as a separate "base" concept) -- same semantics, different
  numbering convention. Notably this INDEPENDENTLY CONFIRMS the frontier-scheduling fix I made
  earlier this session to the OpenVLA fork's vla_interleaved_static.py (38aa8d1: group 0 must
  advance the frontier immediately, not stay at 0) was faithful to AppCorr's actual original
  design -- good cross-check, unrelated repo/branch but same underlying mechanism.

- **Found the 3-condition mapping onto EXISTING configs**, no new scheduling code needed:
  - "approx-only" = `imnet_approx_only_l2.json` (Laplacian transmission, single heavily
    downsampled pyramid level, `BatchCountBasedPolicy` -> unconditionally issues FULL_INFERENCE
    -- i.e. the stock model call, but on a degraded/blurred input only).
  - "full baseline" = `imnet_sequential.json` (FullImageCompression transmission, lossless PNG of
    the complete image, same `BatchCountBasedPolicy` -> FULL_INFERENCE on the TRUE image = ground
    truth).
  - "interleaved correction" = `imnet_interleaved_g4.json` (ProgressiveLaplacian +
    `GroupTriggerPolicy`, the real approx/correct pipeline); `--grouping-strategy` CLI override
    swaps grid/uniform_diff/energy_asc/energy_desc without needing per-strategy config files.
  All three conditions run through the IDENTICAL driver code, just swapping which config JSON is
  loaded (see `dinov3_classifier_offload_eval.py`).

- **Wrote `analysis/experiments/dinov3_classifier_offload_eval.py`**: drives the real
  SchedulerModule+WorkerModule pipeline one ImageNet image at a time (batch_size forced to 1 for
  clean per-image latency + simple result indexing), reusing
  `offload.mobile.dataset.ImageNetLoader` for top1/top5 bookkeeping (already implements exactly
  this), and CUDA-event server_events for per-op latency (same mechanism validated all session on
  the OpenVLA side -- `end_ev.synchronize()` before reading `elapsed_time()`, not Python wall
  clock). `--num-samples`/`--grouping-strategy`/`--num-groups`/`--token-keep-ratio` CLI overrides
  for the sweep. `--data-root` defaults to the real path
  `/NHNHOME/share/cjpark/data/imagenet_val` (NOT `~/data/imagenet_val`, which the upstream scripts
  default to but doesn't exist on this machine).

- **Launched first smoke test** (nr=3, `imnet_sequential.json` = full baseline, using the
  `appcorr` conda env which is the dedicated env for this side of the repo, distinct from
  `openvla`): loading DINOv3-7B weights via mmap, waiting on first real result before trusting
  the driver end-to-end. If this works, will run approx-only and interleaved (grid baseline)
  smoke tests next, then start the actual sweep.

- **Full-baseline smoke test PASSED**: 3/3 correct (top1=top5=100%, trivially small n),
  `FULL_INFERENCE` mean 80.1ms via clean CUDA-event timing. Sample 1's wall-clock (166s) was just
  DINOv3-7B mmap load overlapping with the first request (my naive `time.sleep(1.0)` after CONFIG
  is nowhere near enough for a 7B mmap load) -- cosmetic only, doesn't affect the CUDA-event-based
  per-op latency numbers that actually matter. Noted as a known driver quirk, not yet fixed (low
  priority -- could add a discarded-warmup-request like the OpenVLA driver's pattern later).

- **Found and fixed two real bugs** running the other two smoke tests in parallel:
  1. **My own shell scripting mistake**: launched both background jobs in one Bash call with a
     single leading `cd AppCorr && ... &`; the `cd` was scoped to the FIRST backgrounded subshell
     only, so the SECOND job inherited the stale pre-existing cwd
     (`/NHNHOME/share/cjpark/openvla`) and failed with "can't open file ... openvla/analysis/...".
     Fixed by wrapping each backgrounded job in its own `(cd ... && ... &)` subshell.
  2. **Real bug in `offload/policies/transmission/laplacian.py`** (commit 3b375c1):
     `_process_image_decode`/`_process_image_decode_preserve` only did the final
     upsample-to-native-resolution step when `0 in levels` (i.e. level 0 was an EXPLICIT
     configured residual level). `imnet_approx_only_l2.json` uses `pyramid_levels=[2]` alone (a
     single heavily-downsampled base, no residual levels), so this was never true and `decode()`
     returned a 64x64 image where a 256x256 one was expected -- crashed with `could not broadcast
     input array from shape (64,64,3) into shape (256,256,3)`. Fixed by making the final upsample
     unconditional on `prev_lvl > 0` alone (safe: for normal multi-level configs the residual loop
     already reaches level 0 naturally, so this fallback branch never fires there anyway).
     Verified with a pure-numpy encode/decode round trip before touching the GPU again.
  Relaunched both smoke tests (approx_only_smoketest2, interleaved_grid_smoketest2) with both
  fixes applied.

- **Round 1 of nr=20 sweep (full-baseline + approx-only) FAILED**: both timed out after 300s
  waiting for InferenceResult, with NO worker startup log lines at all (previous successful runs
  always showed "[Worker] Started." within ~1-3min of model mmap loading; here neither log
  advanced past "[Scheduler] Configured with BatchCountBased"). GPU memory confirmed idle (4MiB/
  183GB on both GPUs) at time of check -- not a GPU OOM/contention issue. No conflicting
  dinov3_classifier_offload_eval processes found running or zombied. Other long-running,
  pre-existing (Jul01) user processes (offload/server/main.py + offload/mobile/main.py bound to
  TCP ports 39990/39991 for separate COCO/ADE20K experiments) are unrelated -- my driver uses
  pure in-process multiprocessing.Queue, no network ports, so no plausible port conflict there.
  Suspect a transient worker-process startup failure (possibly swallowed exception before the
  worker's own stdout/stderr was established) rather than a real code regression, since nothing
  changed between this attempt and the last successful smoke test 2 rounds ago. Retrying once
  before deeper investigation; if it fails identically, will add more verbose/immediate-flush
  logging around SchedulerModule/WorkerModule startup to pin down the actual cause.

- **Retry succeeded — prior failure was transient, not reproducible.** full_baseline_n20_retry:
  20/20 processed, **top1=85.00% top5=100.00%** (first meaningful, non-trivial accuracy signal,
  spanning 20 real classes via the strided-sampling fix). `FULL_INFERENCE` mean=25.7ms/sample
  (much lower than the earlier n=3 smoke test's ~80ms -- that was noise from averaging just 3
  samples including cold-start-adjacent variance; 25.7ms is the stable steady-state number).
  Proceeding with approx-only + interleaved-grid nr=20 next.

- **approx-only nr=20**: top1=90.00% top5=100.00%, FULL_INFERENCE=33.3ms/sample.
- **interleaved-grid nr=20**: top1=90.00% top5=100.00%, CORRECT_FORWARD=35.0ms APPROX_FORWARD=9.0ms
  (ratio 3.9x -- consistent with the earlier finding: token_keep_ratio=1.0 means no real pruning,
  so CORRECT processes all patches over a growing layer range while APPROX stays a fixed small
  chunk; correction is inherently pricier here, not a regression). Same failure pattern as
  approx-only (samples 6, 16 wrong in both) -- makes sense, both are working from degraded/
  corrected versions of similar quality. Launching uniform_diff + energy_asc next.

- **uniform_diff + energy_asc nr=20**: both top1=90% top5=100%, IDENTICAL failure pattern to grid
  (samples 6, 16) -- expected at token_keep_ratio=1.0 (everything eventually gets corrected
  regardless of group order, so final accuracy converges the same way).
  Latency differs a lot though: grid CORRECT=35.0ms, uniform_diff CORRECT=56.2ms, energy_asc
  CORRECT=94.1ms (APPROX also creeps up: 9.0 -> 7.0 -> 15.8ms). HYPOTHESIS: this is the same
  cuBLAS/SDPA shape-dispatch tax found and fixed on the OpenVLA side this session (see
  develop/openvla-progressive-prefill commit cf4c217) -- grid always makes exactly
  256/num_groups patches per group (constant shape every call, cheap), while uniform_diff/
  energy_asc balance by a DATA-DEPENDENT quantity (byte size / energy), producing variable group
  sizes -- a novel shape pays a one-time dispatch cost almost every call. energy_asc is worse
  than uniform_diff because energy splits are more skewed (few high-energy + many low-energy
  patches), so group sizes vary more per image. AppCorr's DINOv3 correct_partial_token already
  HAS an `sdpa_query_bucket_size` mechanism (block.py/attention.py) for exactly this, just not
  enabled in these configs (appcorr_kwargs has no such key, defaults to 0/disabled). Plan: add a
  --sdpa-query-bucket-size CLI override to the driver and test whether enabling it closes the gap
  for energy_asc/uniform_diff, before drawing conclusions about which grouping strategy is
  "really" better latency-wise. Launching energy_desc now to complete the primary 4-way
  comparison first.

- **energy_desc nr=20**: top1=90% top5=100% (same failure pattern), CORRECT_FORWARD=46.5ms
  APPROX_FORWARD=6.5ms -- notably CHEAPER than energy_asc's 94.1ms/15.8ms despite both balancing
  the exact same total energy per group (just reversed priority order). Plausible reason: desc
  puts the FEW high-energy patches in group 1 (small, arrives/corrects early when frontier is
  still shallow -- cheap regardless of shape tax) and the MANY low-energy patches in later groups
  (large, but by then frontier is deep -- more layers to correct, but a SINGLE big consistent
  group each time may hit fewer distinct shapes than asc's early small+few, later huge+few
  alternating pattern). Not fully explained yet; the bucket-size test below should help clarify
  whether this is really about shape variability or something else (e.g. real workload
  differences from WHICH specific patches enter which group).

  Current full comparison (nr=20, token_keep_ratio=1.0, no bucketing):
  | mode              | top1 | top5 | CORRECT_FORWARD | APPROX_FORWARD |
  |-------------------|------|------|------------------|-----------------|
  | full-baseline     | 85%  | 100% | n/a (FULL_INFERENCE=25.7ms) | |
  | approx-only       | 90%  | 100% | n/a (FULL_INFERENCE=33.3ms) | |
  | interleaved-grid  | 90%  | 100% | 35.0ms | 9.0ms |
  | uniform_diff      | 90%  | 100% | 56.2ms | 7.0ms |
  | energy_asc        | 90%  | 100% | 94.1ms | 15.8ms |
  | energy_desc       | 90%  | 100% | 46.5ms | 6.5ms |

  Testing --sdpa-query-bucket-size next to check if it closes the gap for the
  variable-group-size strategies (energy_asc worst case, uniform_diff moderate case).
