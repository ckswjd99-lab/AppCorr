# Exact approx/correct decomposition — FP4 belongs on the correction path

**Status date:** 2026-08-04
**Branch:** `develop/dinov3-approx-fp4`
**Script:** `analysis/experiments/dinov3_fp4_feature_fidelity.py`

## Bottom line

Rebuilt the approx/correct split so its arithmetic is **exact**, then measured DINOv3 ViT-7B feature
fidelity with NVFP4 assigned independently to each path. Result: **FP4 on the correction path is
nearly free, FP4 on the base path costs the same as quantizing everything.**

| | ImageNet-1k (N=5000, 256px) | | COCO2017 (N=500, 1024px) | |
|---|---:|---:|---:|---:|
| condition | rel L2 | cosine | rel L2 | cosine |
| `fp4_full` — plain forward, all 5 Linears FP4 | 0.087965 | 0.996777 | 0.053603 | 0.998733 |
| `exact_bf16_bf16` — control | 0.012258 | 0.999927 | 0.005420 | 0.999988 |
| **`exact_bf16_fp4` — FP4 on the delta only** | **0.062740** | **0.998328** | **0.014312** | **0.999904** |
| `exact_fp4_bf16` — FP4 on the base only | 0.086372 | 0.996784 | 0.053538 | 0.998734 |
| `exact_fp4_fp4` — control | 0.093825 | 0.996312 | 0.054428 | 0.998701 |

Reference is the plain BF16 forward; metrics are over patch tokens (cls/storage excluded), averaged
per image. First 10% of each dataset.

- Putting FP4 **only on the correction delta** cuts feature error by **29% on ImageNet** (0.0880 →
  0.0627) and **73% on COCO** (0.0536 → 0.0143) versus quantizing the whole forward.
- Putting FP4 **only on the base** is indistinguishable from quantizing everything (0.0864 vs
  0.0880; 0.0535 vs 0.0536). **All of FP4's damage originates in the base path.**
- On COCO the delta-only variant (0.0143) sits close to the method's own floor (0.0054), i.e. FP4
  on the correction is very nearly free at that resolution.

## The decomposition

Every activation is carried as a pair `(a, d)` whose true value is `a + d` (`a` = base/approx path,
`d` = correction path), pushed through the block op by op:

```
linear  W x + b :  a' = W a + b          d' = W d              # bias cancels in the difference
nonlinear   g   :  a' = g(a)             d' = g(a + d) - g(a)  # recomputed exactly
```

Both lines are identities, so the pair telescopes — `a' + d' = f(a + d)` exactly, for *any* `a`.
There is **no Taylor truncation**: unlike the 1st-order expansions in *Theories for Approx-correct*
§2, the non-linearities here are re-evaluated rather than linearized, so the only error left in the
system is arithmetic precision. This is what makes the FP4 comparison clean.

The five weight-GEMMs (`attn.qkv`, `attn.proj`, `mlp.w1/w2/w3`) are the expensive ops and the only
ones quantized. In the correction path they consume `d` rather than `x`, so their absolute
quantization error scales with `|d|`. Everything non-linear (LayerNorm, the softmax/attention core,
SiLU⊙product) is recomputed and contributes nothing.

SwiGLU maps to the paper's notation as `w1 = W_g`, `w2 = W_u`, `w3 = W_d`
(`hidden = SiLU(w1 x) ⊙ w2 x`, `out = w3 hidden`).

### The two controls that validate the implementation

The telescoping identity only holds when both paths use the same weights, which gives two free
correctness checks — both pass:

- **`exact_bf16_bf16` reproduces the reference** (0.0123 / 0.0054 rel L2, cosine ≥ 0.99993).
- **`exact_fp4_fp4` reproduces `fp4_full`** (0.0938 vs 0.0880; 0.0544 vs 0.0536).

Any bug in the delta bookkeeping would break both.

### Method floor

`exact_bf16_bf16` is not exactly zero because splitting one GEMM into two (`W a + W d` instead of
`W (a+d)`) rounds twice in BF16, and both operands are full-magnitude. Carrying the `(a, d)` pair in
float32 (`--bookkeep fp32`, the default) barely moves it (0.0113 → 0.0096 on a 25-image probe), which
confirms the floor comes from the **BF16 GEMMs themselves**, not the bookkeeping. At ~1/5 of the
signal it does not affect the conclusions.

## Why COCO benefits so much more

Measured relative delta magnitude at the patch-embedding input (20 images each):

| dataset | image size | `‖d‖ / ‖x‖` |
|---|---|---:|
| ImageNet-1k | 256 | 0.2932 |
| COCO2017 | 1024 | 0.1592 |

The L2 base is a 4× downsample-and-restore, so at 1024px it still retains 256px of real detail while
at 256px it retains only 64px — the correction signal is relatively **half the size** on COCO, and
FP4's absolute error on the delta shrinks with it.

The benefit is *not* simply proportional to that ratio, though. Removing the method floor in
quadrature, the FP4-on-delta error contribution is 0.0615 (ImageNet) and 0.0132 (COCO); against
`fp4_full` those are 0.70× and 0.25× respectively, versus input delta ratios of 0.29 and 0.16. So
ImageNet gains distinctly less than its input delta ratio would predict — consistent with the delta
growing relative to the base as it propagates through 40 blocks, which erodes the "delta is small"
premise faster when it starts larger. Worth confirming with a per-layer `‖d‖/‖a‖` trace before
leaning on it.

## Follow-up A — the delta really is narrower along the channel axis

NVFP4 quantizes along the reduction (channel) dimension, so what sets its relative error is the
dynamic range the input carries on that axis. Measured at the input of every quantized Linear, all
40 blocks pooled, 50 images each (`dinov3_exact_decomposition_stats.py`):

| linear | var d/a (ImageNet) | var d/a (COCO) | amax d/a (ImageNet) | amax d/a (COCO) |
|---|---:|---:|---:|---:|
| attn.qkv | 0.4505 | 0.1005 | 0.4340 | 0.3451 |
| attn.proj | 0.6262 | 0.1357 | 0.9254 | 0.7294 |
| mlp.w1 | 0.4167 | 0.0914 | 0.3913 | 0.3012 |
| mlp.w2 | 0.4167 | 0.0914 | 0.3913 | 0.3012 |
| mlp.w3 | 0.4091 | 0.0640 | 0.4421 | 0.2926 |
| **all** | **0.4334** | **0.0796** | | |

The correction delta carries ~2.3x less channel-axis variance than the base on ImageNet and ~12x
less on COCO — the same ordering as the FP4-on-delta win (29% vs 73% error reduction), which is the
mechanism behind it.

**`attn.proj` is the weak spot.** Its delta shrinks least on both datasets (amax ratio 0.93 / 0.73),
because its input is the attention output — a convex combination of V rows — so differences there
are not compressed the way they are at the other four. If a hybrid is ever wanted, that is the one
Linear to consider leaving in BF16.

## Follow-up B — the control error is **not** linearly subtractable

The control's residual error invites subtracting it from the other arms. Measuring the FP4
contribution directly instead (using the control itself as the reference, so no model is assumed):

| | ImageNet | COCO |
|---|---:|---:|
| `\|e_ctrl\|` control vs bf16_full | 0.009391 | 0.005417 |
| `\|e_fp4\|` FP4-on-delta vs bf16_full | 0.052969 | 0.014060 |
| **`\|e_iso\|` FP4-on-delta vs control (measured)** | **0.052374** | **0.013201** |
| linear subtraction predicts | 0.043578 (−16.8%) | 0.008643 (−34.5%) |
| quadrature subtraction predicts | 0.052130 (−0.5%) | 0.012975 (−1.7%) |
| cos(e_ctrl, e_fp4) | 0.150 | 0.353 |

**Quadrature subtraction `sqrt(total^2 - ctrl^2)` is correct** (within 0.5–1.7% of the measured
value); the two perturbations are close to orthogonal (cos 0.15–0.35). Linear subtraction
underestimates by up to 35%.

Concretely: the isolated FP4-on-delta error on COCO is **0.0132, not <0.01** — the sub-0.01 figure
is an artifact of linear subtraction. The headline conclusion strengthens anyway, since against
`fp4_full` measured on the same footing the isolated FP4-on-delta error is **4.1x lower on COCO**
and **1.7x lower on ImageNet**.

## Why this matters for the acceleration work

The accuracy sweeps could not resolve this question at all — at full-2000 ADE20K, FP4 on correction
and FP4 on the whole forward both landed inside a ~0.2pp mIoU noise floor
([dinov3_correct_low_precision_status.md](dinov3_correct_low_precision_status.md)). Feature-level
metrics separate them cleanly, and they say the placement is not arbitrary: **quantize the
correction, keep the base in BF16.** That is also the cheaper half to quantize in the AppCorr
setting, since the correction is what runs repeatedly per round.

Caveat: this measures a decomposition that does **not** yet exist in the serving path. The shipped
`correct_precision=fp4` quantizes the *absolute* activations of selected tokens
(`_correct_forward_partial_token_batched`), which is the `fp4_full`-like regime, not
`exact_bf16_fp4`. Realizing the gain requires implementing the delta-carrying decomposition in
`appcorr/models/dinov3/layers/block.py` — see the NVFP4 acceleration notes in
[dinov3_correct_low_precision_status.md](dinov3_correct_low_precision_status.md) (query bucketing,
zero padding, `M % 128`, dynamo cache limit) for the rest of that work.

## Reproduce

```bash
PYTHONPATH="$PWD/appcorr/models:$PWD" python analysis/experiments/dinov3_fp4_feature_fidelity.py \
    --dataset imagenet-1k --fraction 0.1     # ~40 min on one B200
PYTHONPATH="$PWD/appcorr/models:$PWD" python analysis/experiments/dinov3_fp4_feature_fidelity.py \
    --dataset coco2017 --fraction 0.1        # ~18 min
```
