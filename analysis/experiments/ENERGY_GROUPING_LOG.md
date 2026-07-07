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
