# OpenCLIP's broken accuracy, and why the "redundant" residual round is not redundant

Two independent things went wrong here, one a real bug in the OpenCLIP executor and one a wrong fix
of mine that broke every `GroupTrigger`-scheduled model. Both are written down because the second
one cost more than the first, and the reasoning that produced it looked sound at every step.

## Bug 1 (real, fixed): the executor returned 1 of every 32 predictions

`openclip_executor.get_final_results()` returned `{0: output[0]}` -- the first sample of the batch
only -- while every OpenCLIP config runs at `batch_size: 32`. `worker.py` fills the gaps with
`final_map.get(i, [])`, and `ImageNetLoader.evaluate_batch` skips an empty prediction for the
correct-count while still counting it in the denominator. So the reported accuracy was capped at
exactly 1/32 = 3.125%.

Measured before the fix, 640 images: **top5 3.125%** (one scored sample per batch, 20 batches) and
**top1 2.97%** (19 of those 20 correct) -- the arithmetic matches the cap to the digit, and the model
was actually ~95% correct on what it scored. It hit ceiling, floor and every corrected arm equally,
so no ordering between arms was visible either; the whole OpenCLIP block of the table was noise.

The executor's own module docstring says `batch_size is forced to 1 by the eval drivers ... so there
is no per-batch-item variable masking to handle`. That assumption is false for these configs and is
what the code was written against. Fixed by returning one entry per batch index. After the fix,
640-image subset: **ceiling 90.47 / floor 79.22 top-1** (subset is the first 640 of a class-sorted
val set, so absolute values run high -- use `sample_stride` for a comparable number).

Still open, not chased: `group_map` and `mobile_pscore_hint_map` are `(1, num_patches)` -- one map
shared by all 32 images in a batch. Group assignment is positional so that part is fine, but
`pscore_hint` (residual energy) is per-image and gets last-writer-wins clobbered. Selection quality
is therefore driven by whichever image in the batch wrote last. Correctness of the corrected forward
is unaffected; only *which* patches get chosen is. Fix properly (per-image maps, or `batch_size: 1`)
before quoting an OpenCLIP selection-quality result.

## Bug 2 (mine, reverted): the residual round corrects the group nothing else does

Reading `ade20k_window_trigger.py` and `group_trigger.py`, their final "residual" task issues
`CORRECT_FORWARD({"layers": (0, total), "group_id": n})` where `n` is not a real correction group
(those are `0..n-1`). `dinov3_depther` and `dinov3_segmentor_m2f` resolve that miss as
`else: target_gids = sorted(cached_dindices.keys())` -- correct *everything*, over the full depth.
That is textbook [[interleaved_correction_contract]] rule 1 ("the accumulated set equals the full
selection ... the schedule collapses into one-shot"), and since `boundaries[-1] == total` every token
looked like it had already reached full depth. So: redundant, remove it.

**Wrong.** Walk the actual instruction stream for `n=4`:

| task | correction | approx |
|---|---|---|
| gid 0 | *none* (`chunk_start == 0`) | `(0, 12)` |
| gid 1 | `(0, 12)` group 1 | `(12, 24)` |
| gid 2 | `(0, 24)` group 2 | `(24, 36)` |
| gid 3 | `(0, 36)` group 3 | `(36, 48)` |
| gid 4 | `(0, 48)` **all groups** | -- |

Group 0 has no correction round of its own -- `if group_id > 0 and chunk_start > 0` skips it -- so
the residual round is the *only* thing that ever corrects it. Instrumented confirmation: at `g=4`
exactly three `_prune_patch_idx` calls fire per image, with the cumulative arrived-hint count going
64 -> 128 -> 192 of 256. The fourth quarter is never corrected without the residual round.

The identity gate the contract itself recommends settles it, on OpenCLIP at 640 images:

| arm | top-1 | top-5 |
|---|---:|---:|
| floor (`approx_only_l2`) | 79.21875 | 94.84375 |
| interleaved `g=1, keep=1.0`, **my fix applied** | 79.21875 | 94.84375 |
| interleaved `g=1, keep=1.0`, residual round restored | **90.46875** | **97.96875** |
| ceiling (`sequential`) | **90.46875** | **97.96875** |

With the residual round removed, `g=1` reproduces the floor *bit-identically* -- correction had
become a complete no-op, because at `n=1` the only correcting instruction in the whole schedule is
the residual one. With it restored, `g=1, keep=1.0` reproduces the ceiling bit-identically, which is
the contract's own pass condition and also clears the OpenCLIP fork of any deeper defect.

Reverted in both `group_trigger.py` and `ade20k_window_trigger.py`.

### What the earlier A/B should have caught and did not

The ADE20K A/B ran floor 42.67 / fixed 49.02 / buggy 52.06 and was read as "the fix removes
over-correction, still well above floor, therefore right." The reading was wrong because **the
ceiling was never measured on that subset**, so "49.02 is a sane place to land" had nothing to be
sane relative to. That is the standing rule -- floor *and* ceiling, every time -- broken in exactly
the way it exists to prevent. The 3pp drop was group 0 going uncorrected, not over-correction being
removed.

### Then the "real remaining over-correction" was measured too, and it mostly is not there

Rather than ship a third change on reasoning, the residual round was instrumented to print what it
actually corrects. Two facts that no amount of code-reading had produced:

**Groups are 1-indexed.** OpenCLIP, `num_groups=4`, one image:

```
[round] layers=0-12 group_id=1 patch_idx=64 groups_present={0: 8, 1: 64}
[round] layers=0-24 group_id=2 patch_idx=64 groups_present={1: 64, 2: 64}
[round] layers=0-36 group_id=3 patch_idx=64 groups_present={1: 64, 2: 64, 3: 64}
[round] layers=0-48 group_id=4 patch_idx=64 groups_present={1:64, 2:64, 3:64, 4:64}
```

The residual round's `group_id = n` is a **real correction group with its own 64 patches** -- the
transmission emits groups `1..n`, and `group_id 0` is the base level. So on OpenCLIP the residual
round corrects exactly its own group, over the full depth, which is precisely what the contract
asks for. There is no over-correction and nothing to fix. The proposed fix ("make it correct group
0 only") would have corrected the *base level* and skipped a real quarter of the image -- a third
wrong change, avoided only by measuring first.

**On m2f the fallback does fire, but it is load-bearing.** ADE20K, one image, crop 0:

```
[round] L0-20 gid=1 branch=own_group   targets=[1] keys=[1]
[round] L0-40 gid=2 branch=FALLBACK_ALL targets=[1] keys=[1]
```

The round count (`num_correction_groups`) exceeds the number of groups actually present for that
crop, so the residual round's `group_id` matches nothing and the fallback re-corrects group 1 --
already corrected to depth 20 -- over the full depth 0-40. That extra work is real. But it is not
free to remove: it is what gives the last real group its full depth, and dropping it cost 3pp
(52.06 -> 49.02). If the approximate pass over layers 20-40 carried corrected tokens forward the way
the contract says it should, this re-correction would be numerically a no-op; it is not, so
something in the m2f path is not carrying corrections through subsequent approx passes.

**Caveat on that second reading:** the trace filtered to `src_idx == 0`, and m2f runs multiple
sliding-window/TTA crops with a *per-crop* `cached_dindices`. Other crops may carry different group
sets, so "round count exceeds group count" is confirmed for crop 0 only, not for the request.

### What is actually left to do, and why it is not done here

The remaining defect is m2f-specific and is NOT "the residual round over-corrects". It is one of:
the round count disagreeing with the per-crop group count, or the approx pass failing to carry
corrected tokens forward (which the residual full-depth pass then masks). Distinguishing them needs
a per-crop trace across all sources, not crop 0.

Not attempted here, deliberately. Two changes this session were shipped on reasoning that read as
airtight and were wrong, and the cost each time was silent corruption of measurements already in
flight. Whatever comes next must pass `g=1, keep=1.0 == ceiling` bit-identically on at least one
model, and be A/B'd against floor AND ceiling on the model it targets, before it is believed.

## Bug 3 (real, fixed, small): the CLIP fork never persisted the corrected increment

`appcorr/models/openclip/vision/block.py`'s `correct()` read `{tag}_blocks_out_sum` and never wrote
back -- rule 3 of the contract, the pre-`ac0238f` shape the contract memo explicitly says "the CLIP
fork carried ... and SAM 3 inherited from CLIP". Fixed (write `(x_attn_active - x_active) +
mlp_out_new` back over the approximate increment).

Effect on the 640-image subset was small: `k=0.25` 78.44 -> 78.59, `k=0.50` unchanged at 81.88. Kept
anyway because it is a correctness requirement rather than a tuning knob, but note it was **not** the
cause of the below-floor anomaly that prompted the search -- Bug 2 was. Its effect on SAM 3 (same
inherited shape) has not been measured.

## Also found, not yet fixed

`coco_retrieval_clip_bigg_*.json` names `dataset_name: coco_captions`, for which
`offload/mobile/dataset.get_dataset_loader` has no entry at all -- every run dies at handshake with
`ValueError: Unknown dataset name: coco_captions` **and still exits rc=0**, so a campaign script
counts it as a success. Same shape as the Qwen2.5-VL `realworldqa` loader gap. OpenCLIP retrieval
rows cannot be produced until a loader exists.
