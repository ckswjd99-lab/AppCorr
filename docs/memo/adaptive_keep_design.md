# Bucketized threshold selection ("adaptive k") -- design (user GO 2026-09-13 05:20)

## What exists today (`appcorr/models/qwen_vl_axis.py` ~497-525, verified)
Per merge group g of the image (unit = rows per group):
  resid_r  = mean_j (px_full[r,j] - px_base[r,j])^2          # pixel_values: PROCESSOR-NORMALISED units
  E_g      = mean_{r in g} resid_r ;  E_g <- E_g / mean_g E_g  # per-image mean-1 normalisation
  a_g      = mean_{r in g} attn_layermean[r] ;  a_g <- a_g / mean_g a_g   # = N * a (received attention
                                                                          # sums to 1 over rows), mean 1
  score_g  = E_g * a_g   (band 0 under the deferred pscore: E_g alone; unified: prefix-layer mean)
  selection: fixed quota n_sel = round(keep * G) split across bands, top-quota per band.
So BOTH factors are already per-image mean-1 normalised: the score is "relative to this image's
average", scale-free but blind to the ABSOLUTE amount of degradation. `pixel_values` are already
(x/255 - mu)/sigma per channel (Qwen processor), i.e. the user's /255 is applied plus a ~3.7x
per-channel scale; one raw gray level ~ 0.014 units.

## Proposed score for thresholding (theory first)
The contribution of correcting row i to the attended output is first-order
  |delta out| ~ a_i * ||delta v_i||  (ProgVFM Eq. 6: attention x residual NORM),
so the natural score is attention x RMS residual, not attention x squared residual:
  RMS_g = sqrt( mean_{r in g} mean_j (x_full - x_base)^2 )   with x in RAW [0,1] pixel units
         (undo the processor's per-channel normalisation, or recompute from uint8/255): model-
         independent, physical -- RMS_g = 0.03 means ~8 gray levels of error on average.
  A_g   = N * mean_{r in g} attn_layermean[r]                 (relative to uniform; mean 1;
         sum-to-1 preserved; this is exactly what the current a_g/mean does)
  S_g   = RMS_g * A_g                                          ("gray-level error, weighted by how
                                                                much the image attends to it")
Absolute on the energy axis (a nearly-exact base yields tiny S everywhere -> few rows), relative
on the attention axis (which is a per-image distribution by construction). The current code's
E (MSE, mean-1) vs RMS (absolute) is a ranking difference too (a*E vs a*sqrt(E)); the offline
simulation compares both under the same bucket rule, at equal mean k.

## Selection rule
Per band r (progressive selection is canonical: candidates = groups arrived in band r):
  m_r  = |{g in band r : S_g >= theta}|
  m_r' = ceil(m_r / (G_r/8)) * (G_r/8)     # bucketize to 1/8 of the band's groups, ceiling
  clamp m_r' to [G_r/8, G_r]                # never 0 rows; never above all
  select the top-m_r' groups of band r BY SCORE (the bucket's extra slots go to the next-best
  groups, not to arbitrary ones)
theta is GLOBAL (same for every image and dataset), calibrated once so that the mean realised
k on a calibration split equals a target (e.g. 0.5); the per-image k then floats with content.
Bucket = 1/8 maps onto the engine's CUDA-graph capture sizes (the pseudo-sequence batch is
already padded to a captured size); no new compiled paths.

## Evaluation design (paired, equal mean compute)
Arms: fixed k in {0.25, 0.5} (existing) vs adaptive at theta calibrated to the same mean k.
Report per dataset: accuracy (paired McNemar on `ok`/`val` vs the fixed arm), mean and p95 of
realised k, Comp./Crit. Comp. as mean and p95, Crit. Lat. as before. Datasets with headroom only
(V*Bench, ChartQA, InfoVQA, VisDrone Det, TextVQA); VSR excluded (saturated).
Record `keep_realised` and `theta` in every jsonl row.

## Steps
1. `analysis/experiments/pscore_dump.py`: one tower pass per image (full + base), dump per-group
   RMS_g (raw units), E_g (current), A_g, band id, N -> npz per dataset (12-36 images per dataset
   first, full split later). GPU, queued behind the running campaigns.
2. `analysis/experiments/threshold_sim.py`: CPU; for theta on a grid and both scores, realised
   k per image with the bucket rule, mean/p95, and the theta that hits mean k = {0.25, 0.5}.
3. `Qwen35Axis`/`Qwen25VLAxis` selection mode `keep="auto"` + `--pscore-threshold theta
   --pscore-bucket 8 --pscore-score {rms,mse}` in the driver; `keep_realised` in rows; table
   loader accepts the new arm tag `streaming_g4_auto<theta>`.
4. Gate: at theta -> infinity selection == fixed floor; at theta -> 0 == k=1; identity of the
   per-band top-m' with the fixed-quota path when m' == quota (bitwise on the selected set).

---

## As built (2026-09-13, CPU-only session; the GPU steps are written but NOT run)

Four deltas from the design above. Where the memo and the code disagree, the code below is what
runs.

**1. The processor constants are NOT CLIP-like on Qwen3.5.** Read off the loaded processors
(`AutoProcessor`, offline cache), printed by `adaptive_keep_gate.py --processor` and by
`pscore_dump.py` at startup:

| model | processor | image_mean | image_std | rescale | one gray level, normalised units |
|---|---|---|---|---|---|
| Qwen3.5-35B-A3B / 122B-A10B-FP8 / 4B | Qwen2VLImageProcessor | 0.5, 0.5, 0.5 | 0.5, 0.5, 0.5 | 1/255 | 0.00784 |
| Qwen2.5-VL-7B-Instruct | Qwen2VLImageProcessor | 0.4815, 0.4578, 0.4082 | 0.2686, 0.2613, 0.2758 | 1/255 | 0.0146 / 0.0150 / 0.0142 |

So the memo's "~3.7x per-channel scale, one raw gray level ~ 0.014 units" holds for Qwen2.5-VL
and NOT for the Qwen3.5 campaign models, where the scale is a flat 2x. A theta calibrated on one
family is not transferable to the other in normalised units -- which is the argument for scoring
in raw units in the first place.

**2. The raw units need no uint8 and no second preprocessing pass.** The residual of two images
is mean-free, so undoing `(x/255 - mean)/std` on a DIFFERENCE is multiplication by `std` alone:
`(x_f - x_b)_raw = (x_f - x_b)_norm * std`. `pixel_values`' last axis is channel-major --
`(channel, temporal_patch, patch_h, patch_w)`, from transformers'
`Qwen2VLImageProcessor.permute(0, 2, 5, 3, 6, 1, 4, 7)` -- so the per-element scale is
`image_std.repeat_interleave(T*P*P)` (`QwenVLStreamingAxis._pixel_std`, gate D checks it against
a flat image of known per-channel levels: max error 1.8e-7).

**3. The bucket rule is stated in integers.** `ceil(m_r / (G_r/8)) * (G_r/8)` is not an integer
count when `G_r` is not a multiple of 8 (it bites: V*Bench bands are rarely multiples of 8).
`appcorr.models.qwen_vl_axis.bucket_quota` -- imported by BOTH the axis and `threshold_sim.py`,
so calibration and runtime cannot drift -- is

    q  = clamp(ceil(n_over * bucket / G_r), 1, bucket)      # buckets
    m' = clamp(ceil(q * G_r / bucket),      1, G_r)         # groups

which reproduces the memo exactly when `bucket | G_r`, and otherwise keeps both limits exact
(`q = bucket -> m' = G_r`; `n_over = 0 -> m' = ceil(G_r/bucket)`) and stays monotone.

**4. Band 0 of the deferred arm thresholds a DIFFERENT quantity.** The existing arm ranks band 0
on the energy factor alone (the attention column sum has not run yet), bands 1.. on
energy x attention. That is unchanged -- only the quota rule is new -- but it means the one global
theta is applied to two different scales inside the same image. `threshold_sim.py --pscore
{deferred,eager}` simulates both rather than averaging it away; the calibration must be run with
the arm's own setting.

**Cost note.** The count `(score[g0:g1] >= theta).sum()` reaches Python once per band (`topk`
needs an int), i.e. one host sync per band that the fixed-quota path does not pay. Same shape of
problem as the FP4 threshold path (`project_appcorr_threshold_sync_next`); measure before
treating it as free.

**Gates run (CPU, tiny randomly-initialised models, both families):**
`analysis/experiments/adaptive_keep_gate.py` (A: bitwise identity with the fixed quota;
B: the two limits; C: the lattice + monotonicity; D: the processor constants/layout) and
`threshold_sim.py --selftest` (34 assertions on synthetic scores). The fixed-k path is unchanged
bitwise: `vllm_interleaved_axis_gate.py --tiny` G5_PASS with a byte-identical report before and
after, and `qwen_axis_cpu_unittest.py --ref-root <pre-edit snapshot>` PASS over
2 families x 2 grids x 2 seeds x groups {1,4} x keep {1.0, 0.5}.
