# VGGT-Omega on AppCorr: state, findings, and what to do next

2026-08-14. Written as a handover: the session that produced this ran out of context part-way
through, so this is the record needed to continue without re-deriving anything.

## Where things stand

Done and verified:

- `appcorr/models/vggt_omega/` — vendored from facebookresearch/vggt-omega @ 39a0cb8, absolute
  imports rewritten. Copied, not depended on, because correction needs block-level hooks.
- `offload/server/model/vggt_omega.py` — `VGGTOmegaExecutor`, registered as `"vggt_omega"`. Stock
  FULL_INFERENCE only; the staged ops raise.
- `offload/server/model/vggt_preprocess.py` — fork of upstream's `load_and_preprocess_images` for
  in-memory frames.
- `offload/mobile/co3d_loader.py` — `CO3DSequenceLoader`, registered as `"co3dv2"`. One request =
  one sequence.
- `offload/policies/transmission/vggt_laplacian.py` — `VGGTLaplacian`, the pyramid anchored to
  VGGT's per-frame canvas. Configs in `offload/config/co3d/` (`co3d_full` = ceiling at level 0,
  `co3d_approx_only_l2` = floor, `co3d_approx_only_l3`).
- Weights in `/NHNHOME/share/cjpark/weights/vggt/` (`vggt_omega_1b_512.pt`, `..._256_text.pt`,
  `vggt_1b_model.safetensors`). Data in `/NHNHOME/share/cjpark/data/co3dv2/extracted/` (4
  categories, 47 GB) and `data/NYU`.

- **The aggregator now runs on appcorr's instrumented blocks.** `aggregator.py` builds
  `SelfAttentionBlock`, `Mlp` and the patch-embed `DinoVisionTransformer` from
  `appcorr/models/dinov3/` instead of the vendored copies. Verified end-to-end against the real
  checkpoint: identical metrics to the vendored build (AbsRel 0.0155 / 99.76% / 0.68 deg on the
  same 3 sequences), checkpoint still loads with zero missing or unexpected keys.
- **qk-norm ported** into `appcorr/models/dinov3/layers/{attention,block}.py` as a default-off
  option. This is the change that made the swap possible; details below.

The full client/server path runs: `run_local.sh offload/config/co3d/co3d_full.json` completes and
reports depth and pose.

Not done: threading approx/correct through `Aggregator.forward`, interleaved scheduler.

### The block swap, and why it was worth it

The vendored VGGT blocks and appcorr's instrumented ones are the *same architecture* -- verified
bit-identical output under VGGT's own weights, with state_dict keys matching exactly
(`scratchpad/block_swap_check.py`). appcorr's additionally carry approx/correct, pscore selection
and the precision controllers. So correction does not need a second implementation; it needs the
aggregator to call machinery that already exists.

The one architectural gap was **qk-norm**: VGGT's aggregator is trained with `q_norm`/`k_norm`,
DINOv3 has no such parameters. Now `SelfAttention(use_qk_norm=...)`, off by default, applied through
`apply_qk_norm()` at four sites (stock, approx, partial-token correct, partial-channel approx).
Verified inert when off -- adds no parameters, existing checkpoints load `strict=True`, output
unchanged -- and bit-identical to VGGT's stock attention when on, with and without RoPE
(`scratchpad/qknorm_check.py`). `nyu_appcorr` was rerun through the real correction path afterwards
and is unaffected.

Two ordering constraints that are easy to get wrong and impossible to see afterwards:

- qk-norm goes **before** RoPE, never after. The approx pass caches K post-RoPE, so normalizing on
  the wrong side would make corrected tokens disagree with cached ones in a way that reads as
  approximation error rather than as a bug.
- In `correct_partial_token` the normalized K must be written **back into `kv_new` in place**.
  `_apply_rope_to_active_tokens` rotates that buffer in place and the result is scattered straight
  into the cache, so a normalized copy that is not written back is silently discarded.

Call `patch_embed.forward_features(...)`, not `patch_embed(...)`: on the instrumented ViT `forward`
is the classifier entry point and returns the CLS token alone.

### Triton needs `_configure_compile_environment()`, and VGGT was not calling it

The correct path first failed with `Cannot find ptxas` out of Triton's JIT. The cause is narrow and
was already documented in CLAUDE.md: the installed triton wheel ships without
`backends/nvidia/include` (no `cuda.h`) and without `bin/ptxas`, and
`offload/server/model/dinov3_precision.py::_configure_compile_environment()` exists precisely to
paper over that -- it supplies `CPATH` from `CUDA_HOME`, finds `ptxas` on PATH, and symlinks
libcuda.

That function was only reachable through the **DINOv3 precision controller**. VGGT-Omega does not
build one, so nothing configured the toolchain and its first correction died. It is now called from
`WorkerModule._check_triton_runtime()` at CONFIG time, so every model gets it.

**A wrong conclusion was recorded here first and is worth flagging.** The default Triton cache at
`~/.triton/cache` was nearly empty, which looked like proof that no AppCorr kernel had ever
compiled -- and therefore that every correction latency ever measured here was an eager fallback.
That was false. `_configure_compile_environment()` redirects `TRITON_CACHE_DIR` to
`~/.cache/appcorr/torchinductor/triton`, which holds 494 entries including `_gather_rows_kernel`,
`_scatter_rows_kernel`, `_rope_active_inplace_kernel` and `_active_token_update_kernel`. Triton has
been working for DINOv3 all along and **existing latency numbers stand.** Check the configured cache
directory, not the default one.

### Triton fallbacks now raise by default

Separately, and prompted by the above: a kernel that declines to run used to return `False` and let
the caller take the eager path silently. That is right for a service and wrong for a repository that
measures kernel latency -- the number still looks fine.

`appcorr/models/dinov3/layers/triton_kernels/_strict.py` now routes every fallback and every
swallowed exception through `note_fallback()`, controlled by `APPCORR_TRITON_FALLBACK`:
`error` (default, raises), `warn` (prints once per distinct reason), `silent` (the old behaviour).
`verify_triton_runtime()` compiles a probe kernel so a broken toolchain is caught at startup with a
diagnosis rather than at the first correction.

### Bit-exactness is not achievable for VGGT correction, and that is not a bug

The staged **approx** path is bit-exact: `PREPARE_TOKENS -> APPROX_FORWARD -> HEAD_INFERENCE`
reproduces `FULL_INFERENCE` with `max|diff| = 0` on every head and at all four cached layers,
patch-embed stack included.

The **correct** path is not, and cannot be. It gathers the selected rows, computes, and scatters
back, so the arithmetic order differs from one dense GEMM; per block it agrees to ~2e-7 in fp32,
which is roundoff. Measured at 100% correction, zero degradation:

| cached layer | correct vs stock | stock re-run with a 1.5e-8 input perturbation |
|---|---|---|
| 4 | 5.7e-6 | 7.6e-6 |
| 11 | 5.7e-6 | 7.6e-6 |
| 17 | 3.4e-4 | 4.2e-4 |
| 23 | 3.8e-3 | 3.5e-3 |

The two columns match, so the residual *is* roundoff amplified through 48 blocks -- the model
multiplies a 1.5e-8 input difference by ~2e5 by layer 23. Confirmations along the way: both block
types reproduce their own stock forward at 100% correction to fp32 roundoff (frame 3.6e-7, inter
2.4e-7), and the residual is identical at L0 and L3 degradation and does not shrink in fp32.

**So the acceptance criterion for VGGT correction is agreement with this sensitivity floor, not bit
equality.** Any future parity test must measure the floor alongside, exactly as the table above
does; comparing against zero will always fail and will look like a bug.

### The canvas must follow the *original* shape, not the reconstructed one

Found while building the point-cloud metric, and it was a real bug rather than a test artifact.

VGGT's canvas is derived per-frame from the frame's aspect ratio. For scale-up frames
the policy reconstructs at a *patch-aligned* version of the native shape (357x637 -> 368x640), and
that slightly different aspect ratio can push `model_target_hw` onto a different token split
(384x688 vs 384x672). Measured on the four Co3D categories: **15 of 288 distinct shapes, 3943 of
62235 frames (6.3%), across 23 sequences.**

The consequence is quiet and bad: the floor and ceiling conditions feed the model *different
canvases* for those frames, so they are not comparable, the effective intrinsics shift, and that
moves pose -- the metric the whole comparison rests on.

Fixed by carrying the native shape to the server (it is already in the patch metadata as
`target_shape`) and deriving the canvas from it: `preprocess_frames(..., native_shapes=...)`, fed
from `context["input_native_shapes"]` which `VGGTOmegaExecutor.preprocess` now populates. Verified
0/288 mismatches afterwards. Scale-down frames were never affected (their reconstructed shape
*is* the canvas, and `model_target_hw` is idempotent on all 288 real shapes), which is why the
hotdog-only numbers elsewhere in this memo are unaffected.

### Traps in *testing* these modules standalone

Both cost real time. A bare block, constructed outside a ViT, has two uninitialized tensors because
the parent's `init_weights` normally sets them:

- `LinearKMaskedBias.bias_mask` is filled with **NaN** at construction (should be ones with the K
  third zeroed).
- `LayerScale.gamma` is `torch.empty` -- **uninitialized memory**. This one produces ~1e36 or NaN
  depending on what was in that memory, so the failure moves between runs and reads like a genuine
  numerical bug. It is not; call `reset_parameters()`.

Also: synthetic RoPE must satisfy `sin^2 + cos^2 = 1`. Random normals make `rope_apply` an amplifier
instead of a rotation.

### The frame axis: `batch_size` *is* `S`

Settled, and worth stating because it is load-bearing. One request is one multi-view sequence, and
its S frames occupy the batch axis the whole way down; only `VGGTOmegaExecutor._frames_to_tensor`
folds them into the model's `[1, S, 3, H, W]`. So `"batch_size": 8` in a Co3D config means eight
*views of one scene*, not eight independent items, and changing it changes S.

This needed no change to the transmission policy or the scheduler, which is the entire argument for
it. The cost is real though: batching several sequences into one request is now unreachable without
revisiting this, because the two meanings of the axis would collide. Given that a single 8-frame
512-scale sequence is already a substantial forward, that trade looks right for now.

Results come back keyed at index 0 (`get_final_results` returns `{0: {...}}`) with the frame axis
*inside* each array, and the loader emits exactly one label per request, so `evaluate_batch`'s zip
pairs them correctly. Do not "fix" the apparent length mismatch between the 8 preds and 1 label.

## Architecture facts that drive the design

Read off the vendored code, not the paper:

- **`patch_embed` is a full 24-layer DINOv3 ViT-L, not a conv stem** (`aggregator.py:220`,
  `_build_patch_embed` -> `DinoVisionTransformer(depth=24, embed_dim=1024)`). So a request runs
  24 ViT blocks per frame *before* the aggregator's 24 frame + 24 inter-frame blocks. This is the
  single most useful fact for correction: the degraded image enters at `patch_embed`, and
  `patch_embed` is the architecture AppCorr already instruments.
- **`vggt_resolution: 512` is a token budget, not a side length.** `_balanced_target_shape` holds the
  count near `(512/16)**2 = 1024` patch tokens and lays them out along each frame's own aspect ratio.
  Measured over 62235 Co3D frames: 1008-1036 tokens each, but only 9 distinct canvases, dominated by
  **688x384** (36612 frames) and 384x688 (17151). A 512x512 canvas happens only for square frames --
  2815 frames, 4.5%. So "resize to 512" and "L3 is 64x64" are both wrong in general: for the common
  1898x1067 frame the canvas is 688x384 and L3 is 96x48. Mean L3 shape across the dataset is 76x61.
  This is why the degradation rule branches on scaling *direction* rather than on a size threshold.
- 24 blocks. Each runs a frame block then an inter-frame block.
- Inter-frame attention is **global in 19 of 24** (full attention over `S x tokens`) and restricted
  to the 17 camera+register tokens in blocks **2, 6, 9, 14, 20**. Frames are *not* independent; a
  degraded view perturbs every other view's patch tokens.
- Token layout per frame: `[camera 1][register 16][patch N]`, `patch_token_start = 17`.
- `cached_layer_indices = (4, 11, 17, 23)` — the only blocks whose output the heads read. Correction
  only has to agree at those four points, which is the natural parity target.
- Omega exposes `camera_and_register_tokens` `[B, S, 17, 2048]` as an output. That is the inter-frame
  bottleneck, available without hooks -- useful both as a parity signal and for measuring how much
  error propagates through registers rather than through global attention.
- Omega has no `world_points` head (VGGT does). Point maps are reachable only by unprojecting
  depth+pose, which is not an equal-footing comparison against VGGT's direct head.

## Measurements

All at 512-scale, Co3Dv2, stride 5 (~22 deg between views), 8 frames/sequence.

### Co3D ceiling vs L2/L3, all 310 sequences

Definitive run: every sequence in the four categories, ceiling / L2 / L3, reported both as a
difference of means (with a bootstrap CI over sequences) and as a paired per-sequence sign test.
`logs/vggt/l2_full316.log`.

| metric | level | mean gap (95% CI) | paired: worse in | sign-test p |
|---|---|---|---|---|
| depth AbsRel | L2 | +12.0% **[-1.1%, +34.1%]** | 156/310 | 0.95 |
| depth AbsRel | L3 | +30.6% [+14.4%, +55.3%] | 218/310 | <0.0001 |
| rotation | L2 | +19.1% [+7.4%, +40.0%] | 202/310 | <0.0001 |
| rotation | L3 | +51.5% [+27.6%, +87.4%] | 243/310 | <0.0001 |
| delta<1.10 | L2 | -0.1% [-0.9%, +0.6%] | -- | -- |
| delta<1.10 | L3 | -1.5% [-2.5%, -0.6%] | -- | -- |

**The two methods agree in every case.** Where the CI crosses zero the paired test also fails to
detect anything; where it excludes zero the paired test detects it too.

Result: **at L2 only pose degrades; from L3 everything does.** Depth's L2 mean gap is positive
(+12%) but its CI crosses zero even with all 310 sequences, so L2 depth degradation is not
established on the data available.

Two symmetric mistakes this run corrects, both of which were made in earlier versions of this memo:

- **Quoting a bare mean.** Earlier subsamples gave +67%, +16%, -0.5% for the same quantity, which
  was read as the mean being unusable. It was not -- the number simply had a wide CI that was never
  computed. With all the data and a CI attached, a difference of means is perfectly usable here, and
  it is the right statistic when aggregate corpus quality is what matters.
- **Quoting a p-value as if it were an effect size.** `delta<1.10` at L2 reaches p < 0.0001 on the
  paired test while the actual change is -0.1% with CI [-0.9%, +0.6%] -- i.e. nothing. At n = 310 a
  tiny systematic shift becomes detectable. Significance is not magnitude.

Report both: the effect size with an interval, and the paired count when the distribution is skewed.
Co3D depth is skewed (median 0.0236, mean 0.0556, max 0.4567; 9% of sequences carry 49% of the
summed error), which is why the two can answer differently -- the mean asks whether corpus-total
error rose, the paired count asks whether a typical sequence got worse.

For absolute depth quality use NYU: Co3D's depth GT is MVS-derived and covers only ~1% of pixels.

### Headline table: ceiling vs L2 vs L3, all sequences

n = 309-310 sequences (every sequence in the four categories), means over sequences.

| | ceiling (L0) | L2 | L3 |
|---|---|---|---|
| depth AbsRel (lower better) | 0.0426 | 0.0477 | 0.0557 |
| rotation, deg (lower better) | 1.305 | 1.554 | 1.977 |
| delta<1.10 (higher better) | 92.91% | 92.81% | 91.54% |
| 3D point error (lower better) | 0.1572 | 0.1464 | 0.1772 |
| 3D inlier <10% (higher better) | 62.61% | 66.25% | 57.58% |
| transmitted, KB/frame | 336 | 38.2 | 8.8 |

**At L2 only rotation degrades** (+19.1%). Depth, delta<1.10 and the 3D metric are flat or slightly
*better* -- mild blur appears to help shape estimation. **At L3 everything degrades**: depth +30.6%,
rotation +51.5%, 3D +12.7%, inlier -5.0 points.

So AppCorr's recoverable headroom at L2 is rotation alone. Recovering depth as well requires
dropping the floor to L3.

### The 3D metric: align on points, not on cameras

The 3D number above is the median per-point distance to the GT reconstruction, in units of GT scene
radius, after a **point-based Sim(3) alignment** -- i.e. shape quality once global pose and scale are
best-fitted away. Pose error is deliberately excluded here; the rotation row measures it.

Aligning on camera centres instead is unusable, and the failure is instructive. VGGT predicts depth
and camera in only *approximately* the same scale -- the measured ratio ranges 0.84 to 1.44 across
sequences -- so a camera-fitted similarity mis-scales the whole cloud by whatever that sequence's
mismatch happens to be. Result: a ceiling error of **4.95 scene radii** on average, with sequences
whose rotation was excellent (0.98 deg) scoring 11.7 radii and one at 0.21 deg scoring 3.87. The
metric was measuring the model's internal scale consistency, not reconstruction quality. Under point
alignment the same ceiling is 0.157 radii with 62.6% of points inside 10%.

Two other things paid for while building this:

- **Do not swap depth from one condition with the camera from another.** Same shared-scale reason:
  each condition is internally consistent, so a swap injects the difference between two scales and
  scores *worse than either endpoint*, which is how the first attempt produced an impossible
  ordering.
- **Validate the GT reconstruction against `pointcloud.ply` first.** Co3D's `ndc_isotropic`
  conversion is scale = min(H, W)/2 with the principal point subtracted; that lands a median 1.2% of
  scene radius from Co3D's own cloud, whereas the plausible-looking per-axis `(W/2, H/2)` variant
  lands at 46%. `scratchpad/gt_cloud_check.py`.

Per-point *means* are unusable at any alignment: a few points per cloud sit ~100x the median, giving
3.96 scene radii. Use a per-sequence median over points, then average over sequences.

### Through the serving path (single category -- see the caveat above)

`run_local.sh`, `-nr 40`, 39 sequences scored (one had too little valid depth GT):

| config | AbsRel | d<1.25 | d<1.10 | rot | RRA@15 | KB/frame |
|---|---|---|---|---|---|---|
| `co3d_full` (L0, ceiling) | 0.0521 | 94.31% | 90.73% | 1.79° | 97.99% | 336 |
| `co3d_approx_only_l2` (floor) | 0.0871 | 92.50% | 88.68% | 2.08° | 97.80% | 38.2 |
| `co3d_approx_only_l3` | 0.0908 | 93.09% | 87.38% | 3.19° | 94.87% | 8.8 |

**These 39 sequences are all `hotdog`.** The loader iterates categories in config order and hotdog
has 115 sequences, so `-nr 40` never reaches another category. The offline sweep below took 40 from
each of four categories. The two tables are therefore *not* comparable, and an earlier version of
this memo compared them anyway.

The serving-path L2 gap here is +67% AbsRel against the offline sweep's +16%. **The reason is the
sequence subset, not the degradation chain.** Measured directly (`logs/vggt/l2_decompose.log`), on
the same 24 hotdog sequences:

| chain | AbsRel | gap vs ceiling |
|---|---|---|
| ceiling | 0.0778 | — |
| offline `pyramid_degrade` | 0.1332 | +71.1% |
| offline chain at the policy's band shape | 0.1315 | +69.0% |
| policy band, reconstructed with pyrUp | 0.1327 | +70.4% |
| the actual transmission policy | 0.1330 | +70.8% |

All four agree within 1.3%, and the two band shapes are 172x96 vs 176x96. So `pyrDown`/`pyrUp`
versus `INTER_AREA`/`INTER_LINEAR`, and the uint8+zlib round-trip, make no measurable difference at
this level -- a previous version of this memo asserted they did, from reasoning rather than
measurement. They do not.

An earlier version of this memo read the L2 -> L3 move off this single-category table and concluded
that it "costs almost nothing on depth but wrecks pose". The paired test above says otherwise: at L2
depth is unaffected *and* pose is mildly affected; at L3 **both** degrade (depth +13.2%, pose
+40.3%). Pose is simply the more sensitive metric everywhere, not a metric that turns on separately.
Do not use these single-category means for that comparison.

Verified: at level 0 the pipeline reproduces a direct executor call *exactly* (AbsRel 0.0155 vs
0.0155 on the same 3 sequences), so the transmission path itself alters nothing it should not.
`scratchpad/parity.py` -- worth rerunning after any policy change.

**Post-block-swap regression** (`logs/vggt/post_swap_regress.log`). All three digit-for-digit
identical to their pre-swap runs, which is the bar this had to clear -- not "close":

| config | AbsRel before | AbsRel after |
|---|---|---|
| `co3d_full` | 0.05213208882233653 | 0.05213208882233653 |
| `co3d_approx_only_l2` | 0.08714157500519203 | 0.08714157500519203 |
| `nyu_appcorr` (DINOv3 correct path) | 0.08895853948992444 | 0.08895853948992444 |

The third one is the one that matters for risk: `appcorr/models/dinov3/layers/` is shared with every
existing paper config, so any future edit there must rerun a real correction config, not just a unit
check.

### Where the forward actually goes

`scratchpad/profile_split.py`, S=8, 688x384 canvas (1032 patch tokens/frame), B200, bf16:

| stage | ms | share |
|---|---:|---:|
| `patch_embed` (24 ViT-L blocks) | 24.1 | 28.7% |
| frame blocks (24) | 29.5 | 35.2% |
| inter-frame global (19) | 21.5 | 25.7% |
| inter-frame register (5) | 1.0 | 1.2% |
| **full model incl. heads** | **83.8** | 100% |

This kills the shortcut worth naming, because it looks attractive until you measure it: *correct
only inside `patch_embed`, then rerun the rest of the aggregator in full*. `patch_embed` is only
28.7%, so that plan still pays 71% of a full forward and cannot beat the ceiling on latency no
matter how good the correction is. **Correction has to reach the frame blocks too** -- the two
per-frame stacks together are 63.9%, and they are precisely the architecture appcorr already
instruments, which is why the block-swap plan is worth the qk-norm work.

The 5 register-restricted inter-frame blocks are free (1.2%): they touch 17 tokens per frame instead
of 1049. The 19 global ones are 25.7% at S=8 and grow quadratically in S, so at larger S the balance
tips toward them and any per-frame-only correction scheme gets progressively less attractive.

### Offline pyramid sweep (pyrDown/pyrUp chain, not the serving path)

Degradation anchored to the model's input scale (scale-down frames are reduced to the canvas first,
scale-up frames are degraded natively and scaled up afterwards):

| level | eff. res | AbsRel | d<1.25 | d<1.10 | rot | RRA@15 |
|---|---|---|---|---|---|---|
| L0 (base) | 536x458 | 0.0542 | 96.07% | 92.22% | 1.76° | 97.91% |
| L1 | 268x229 | 0.0526 | 95.86% | 92.07% | 1.77° | 97.88% |
| L2 | 134x114 | 0.0628 | 95.79% | 91.69% | 1.99° | 97.67% |
| L3 | 67x57 | 0.0714 | 95.66% | 90.43% | 2.65° | 96.22% |
| L4 | 33x29 | 0.1210 | 93.38% | 84.70% | 3.93° | 93.70% |

n = 135 sequences x 8 frames (40/category over 4 categories); 864 of 1080 frames sit above model
scale. `logs/vggt/pyr_sweep_wide.log`. An earlier n=11 run put L3 *above* L2 on AbsRel
(0.0629 -> 0.0567); at n=135 every metric degrades monotonically, so that inversion was noise.

Superseded: this sweep's monotone-looking degradation is a difference of means on the
outlier-dominated depth metric (see the caveat at the top of Measurements). The paired test is the
one to trust, and it puts no measurable depth degradation at L2 at all.

One thing this rules out: degradation down to L1 is free or better than free -- the model discards
that detail anyway, so anything above the model scale is pure transmission waste. On this chain
depth and pose also appear to degrade *together* (both turn at L2-L3, both collapse at L4), which
would argue for one shared transmission policy -- but the serving-path table above contradicts that,
so trust the serving path.

## Traps already paid for

- **Anisotropic resize destroys pose while leaving depth fine.** Squashing frames to a square gave
  AbsRel 0.024 alongside ~100 deg rotation error, and no GT convention rescued it. Always go through
  `vggt_preprocess.preprocess_frames`.
- **Co3D camera convention**: `X_cam = X_world @ R + T` (row-vector, PyTorch3D; `data_types.py:65`),
  so column-vector rotation is `R.T`, then diag(-1,-1,1) for PyTorch3D->OpenCV axes. Handled inside
  the loader.
- **Whole-orbit frame sampling inflates pose error.** Eight frames spread over a full 360 deg orbit
  put opposing views in one batch: 31 deg median error. At stride 5 or 25 the model lands at
  0.5-1.6 deg. Sample by stride, not by even division of the sequence.
- **Co3D resolution is not uniform**: 288 distinct sizes across 62k frames, none above 1.3% share;
  71% have a long side >= 1000, and 251 of 355 sequences are uniformly high-res. Sweep by absolute
  resolution, never by a fixed 1/N of native.
- Co3D depth GT is MVS-derived and covers only ~1% of pixels (object region, masked). Fine for
  relative comparisons between conditions; use NYU when absolute depth quality matters.

## Known broken

**Translation-direction metrics.** Rotation is solid (0.5-1.6 deg) but camera-centre direction error
sits at 45-50 deg with RTA@15 around 50-60%, which cannot be right next to sub-degree rotation. The
`D @ T` handling in the centre computation `C = -R^T T` was never verified the way the rotation
convention was. The loader deliberately reports rotation only; fix before quoting any translation
number.

## Next steps, in dependency order

B2 (transmission policy + config), the frame-axis question, the qk-norm port and the block swap are
all **done**; see above. What remains is the correction path itself.

1. **Thread approx/correct through `Aggregator.forward` (B3).** The blocks already expose
   `approx(x, rope, cache_feature, tag, **kw)` and `correct(x, dindice, rope, cache_feature, tag,
   **kw)`. The work is plumbing, not algorithm: one tag per (stack, block index) across the three
   stacks (`patch_embed`, `frame_blocks`, `inter_frame_blocks`), plus the executor's staged ops
   (`PREPARE_TOKENS` / `APPROX_FORWARD` / `CORRECT_FORWARD` / `HEAD_INFERENCE`), which currently
   raise.

   **Copy the serving pattern, not the standalone one.** Two drivers exist and only one is the right
   model:
   - `DinoVisionTransformer.forward_features_list_appcorr` (`models/vision_transformer.py:341`) is
     plan-driven and complete, but it builds its own input pyramid by interpolating the tensor. That
     is the offline/standalone path. In the serving path the degraded image already arrives from the
     transmission policy, so this would degrade twice.
   - `DINOv3DeptherExecutor.approx_forward` (`dinov3_depther.py:762`) is the one to mirror: the
     executor holds prepared tokens, rope, and a per-source `cache_feature` dict in `context`, and
     `approx_forward(layers=(a, b))` loops the block range calling `blk.approx(...)`. `correct_forward`
     is the same shape with `dindice`.

   The VGGT-specific complications on top of that shape: `patch_embed` is itself a 24-block stack
   that needs the same treatment, and the frame <-> inter-frame reshape sits between every pair of
   blocks.

   Build the parity check first: 100% correction must reproduce the stock forward at the four cached
   layers plus `camera_and_register_tokens`. Three separate "runs but is secretly doing the wrong
   thing" bugs have shown up in this work already.
2. **Interleaved scheduler (B4).** Only meaningful once 1 works.

### The token-axis constraint (blocks the correct path, not the approx path)

Discovered while refactoring the block loop. The token axis is *reinterpreted* between the two
stacks, and `correct_partial_token` takes `dindice` as `[B', K]` -- a **fixed K per row**:

| stack | tensor | `dindice` indexes | one row = |
|---|---|---|---|
| frame blocks | `[B*S, N, D]` | `N` | one frame |
| inter-frame global | `[B, S*N, D]` | `S*N` | the whole sequence |

So the frame stack can only express "the same number of corrected tokens in every frame", while the
global stack can allocate freely across frames. Those are incompatible unless K is uniform per frame
-- and the architecture argues *against* uniform, since 19 of 24 inter-frame blocks attend globally
and the patch that matters most for frame 3 may live in frame 7.

Three ways out, and the choice changes what the paper measures:

1. **Uniform top-K per frame.** Simple, works unchanged in both stacks, forbids the cross-frame
   budget allocation the model's own attention motivates.
2. **Sequence-global selection, padded per frame** to the max per-frame count in the frame stack.
   Preserves the interesting capability; wastes work proportional to the spread across frames.
   Subsumes option 1 when the spread is zero.
3. **Correct only in the inter-frame stack**, where global selection is natural. But frame blocks
   are 35.2% of the forward, so this leaves a lot uncorrected.

`embed()` / `run_blocks()` are already split out and verified bit-identical, so the approx path can
be built without settling this. The correct path cannot.

### Open question B3 has to answer

Patch importance is now cross-frame. 19 of 24 inter-frame blocks attend globally over `S x tokens`,
so the patch that matters most for frame 3's depth may live in frame 7. Every existing pscore is
computed within one image. Whether per-image scores suffice here, or whether selection has to become
sequence-global, is genuinely unknown -- and it is the part of this model that is actually novel for
the paper.

The profile constrains the answer: correction must cover `patch_embed` *and* the frame blocks (63.9%
together) to beat the ceiling at all, and the 19 global inter-frame blocks (25.7% at S=8, growing
quadratically in S) cannot be corrected per-frame even in principle.
