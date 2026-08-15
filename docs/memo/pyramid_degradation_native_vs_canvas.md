# Pyramid levels must degrade against the original, not the canvas

> **Pre-fix numbers.** Every interleaved-correction figure here was measured before
> `persist_correction_residual` (2026-08-16, `96889a5`), while interleaved correction discarded
> all but the last round. On ADE20K the same configuration moved 61.042 -> 61.597 mIoU once the
> corrected increment was persisted (80.8% -> 89.7% of the floor-ceiling gap). Comparisons *within*
> this memo shared the defect and are likely to survive it; the absolute values will not. See
> [[dinov3_correct_low_precision_status]].

2026-08-11/12. Why COCO's approx-only floor used to equal its full-transmission ceiling, what was
changed, and what is still open.

## The rule

AGENTS.md ("Approx/Correct Contracts") requires pyramid levels to be built from the original image in
native coordinates, and only the *selected* level to be scaled to the model input shape. Scaling to
the canvas first and building the pyramid from that changes the approximate image content and
invalidates comparisons against configuration-based low-resolution approximation.

## What went wrong

The canvas is an **upscale** of the original on three of four datasets:

| dataset | native (median) | canvas | direction |
|---|---|---|---|
| COCO | 480x640 | 1024x1024 | upscale |
| ADE20K | ~512x683 | 896 short side | upscale |
| NYU | 480x640 | 768x768 | upscale |
| ImageNet | 375x500 | 256x256 | downscale |

Degrading an upscaled canvas barely touches the real content: the upscale added no information, so
removing it again costs nothing. COCO's group-0 base was `_base_hw` = the detector's first window
(352x352, i.e. canvas/3), which is only a 1.36x/1.82x reduction of a 480x640 original — 40% of the
pixels survive. Measured consequence at nr=100:

```
floor  coco_approx_only_windowbase  mAP 0.7033 @  266 KB
ceil   coco_sequential              mAP 0.7034 @ 3072 KB
```

Floor and ceiling agreed to 1e-4. There was nothing for correction to recover, so every COCO number
between them was uninterpretable.

ImageNet was unaffected for the same reason inverted — its canvas is a downscale, so canvas-relative
degradation is already real-relative.

## What was changed

**`COCOWindowProgressiveLaplacian._downsample_base`** now takes the original image and reduces it by
`global_base_downscale` (default 3, matching the 3x3 window grid) *before* handing it over at
`_base_hw`. Degradation is native-relative; the transmitted tensor stays the size the network and
the server are configured for. Result: 40% -> 11% of native pixels.

**`FourierLaplacianHybrid._keep_hw` / `FourierLaplacianProgressive._keep_hw`** now size the kept
coefficient count against the original. Keeping `k` coefficients leaves `k` samples spanning the
field of view, so counting `k` against an oversampled canvas under-degrades by exactly the canvas
upscale factor. The two COCO hybrid configs moved from `dct_keep: 352` to `dct_keep: "window"` so
they track the window base instead of a literal canvas-relative count.

Measured after (nr=100, COCO ceiling 0.7034):

| config | before | after |
|---|---|---|
| `coco_approx_only_windowbase` (floor) | 0.7033 | 0.6386 |
| `coco_approx_only_windowbase_fourier` (floor) | 0.7037 | 0.6350 |
| `coco_interleaved_static` | 0.7055 | 0.6730 |
| `coco_interleaved_static_fourier_hybrid` | 0.7061 | 0.6786 |

The floor–ceiling gap went from 1e-4 to ~0.065, so COCO comparisons mean something again. Run-to-run
noise on untouched configs was <= 0.0003, for scale.

`Laplacian` and `ProgressiveLaplacian` needed no change: they already `pyrDown` from the original
(`_build_native_gaussians`) and project the chosen band afterwards.

## Open

**The Fourier/DCT policies are experimental** — whether they ship at all is undecided, so the change
above is recorded rather than settled.

- **`FourierProgressive` is not fixed.** It runs a *per-patch* DCT (a low-frequency block inside each
  16x16 patch) rather than a global low-pass, so the same defect needs a different correction:
  scaling the per-patch keep by the native/canvas ratio. `coco_interleaved_static_fourier` is
  therefore still canvas-relative.
- **ADE20K and NYU Fourier configs shifted** (`ade20k_m2f_fourier_hybrid_appcorr` 52.44 -> 51.07 mIoU,
  `nyu_fourier_hybrid_appcorr` 0.1943 -> 0.1980 rmse). Not a regression: `_target_hw_for_level` takes
  only the *aspect ratio* from the native image under `preserve_input_shape`, while the magnitude
  still comes from the config short side (896 for ADE20K vs a 512-tall original). Those bases were
  under-degraded too, just less severely. Their pre-2026-08-12 Fourier numbers are not comparable to
  later ones.
- **Correction itself is sound.** With the floor genuinely lowered, `coco_interleaved_static` at 100%
  correction reaches mAP 0.7055 vs the 0.7034 ceiling — the parity tier holds. At the default
  `token_keep_thres=0.002` it recovers only 53% of the gap while selecting 20.4% of tokens, and that
  keep rate is identical before and after the base change, i.e. the threshold does not react to how
  bad the approximation is. A hyper-parameter question for the full-dataset sweep, not a defect.
