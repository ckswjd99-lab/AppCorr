# DINOv3 Jacobian-support ImageNet-1K evaluation and FLOP audit

## Experiment

- Date: 2026-07-28
- Branch: `experiment/jacobian-support-oracle`
- Evaluation source commit: `8d9e4df`
- Result commit before this report: `7ef6822`
- Hardware: 2× NVIDIA B200, one 25,000-image shard per GPU
- Dataset: complete ImageNet-1K validation set, 50,000 images
- Input: resize and center crop to 256×256
- Draft: L2 canvas made by two `pyrDown` and two `pyrUp` operations
- Model: DINOv3 ViT-7B, 40 layers, hidden size 4096, 32 heads
- Tokens: 1 CLS + 4 register/storage + 256 patch tokens = 261
- SwiGLU hidden size: 8192

The experiment evaluates one L2-to-L0 correction. CLS and register tokens are
always corrected. At every layer, residual energy selects the top 50% of the
256 patch tokens, so 128 patch tokens and 5 special tokens are corrected.

Attention and FFN use exact nonlinear output differences:

- Attention: exact full-denominator softmax product delta
  `S1 V1 - S0 V0`, restricted to structured support.
- FFN: exact SwiGLU hidden delta
  `silu(g1) u1 - silu(g0) u0`, restricted to structured channel support.
- Attention support: query blocks of 8, key blocks of 16, groups of 4 heads.
- FFN support: token blocks of 8 and channel blocks of 128.

The layerwise policy was fit from a 200-class stratified ImageNet sensitivity
sweep. Each layer/component curve uses `mean + 0.25 × sample STD`, decreasing
isotonic regression, and a marginal greedy allocator. This is a fitted
sensitivity model, not a learned neural predictor.

## Policies

The target budget applies only to attention and FFN. Patch-token pruning is
fixed at 50% in both policies.

| Policy | Patch-token pruning | Mean attention pruning | Mean FFN pruning | Requested equal-work pruning over all three components |
|---|---:|---:|---:|---:|
| Attn+FFN 25% | 50.0% | 50.0% | 0.0% | 33.3% |
| Attn+FFN 50% | 50.0% | 75.5% | 24.5% | 50.0% |

Stage averages:

| Policy | Stage | Attention pruning | FFN pruning |
|---|---|---:|---:|
| 25% | Layers 0–12 | 52.3% | 0.0% |
| 25% | Layers 13–26 | 46.4% | 0.0% |
| 25% | Layers 27–39 | 51.5% | 0.0% |
| 50% | Layers 0–12 | 82.3% | 7.7% |
| 50% | Layers 13–26 | 72.1% | 37.9% |
| 50% | Layers 27–39 | 72.3% | 26.9% |

## ImageNet-1K results

| Method | Top-1 | Δ Top-1 vs L0 | Top-5 | Δ Top-5 vs L0 | L0 top-1 prediction match |
|---|---:|---:|---:|---:|---:|
| L0 full | 88.104% | — | 98.438% | — | 100.000% |
| L2 approximate only | 84.496% | -3.608 pp | 97.412% | -1.026 pp | 90.278% |
| Token 50% + Attn/FFN 25% | **87.500%** | **-0.604 pp** | 98.346% | -0.092 pp | 95.728% |
| Token 50% + Attn/FFN 50% | 86.550% | -1.554 pp | 98.074% | -0.364 pp | 93.552% |

The 25% policy recovers 3.004 of the 3.608 percentage points lost by L2,
or 83.3% of the top-1 gap. It satisfies a 1.0 pp accuracy budget. The 50%
policy recovers 56.9% of the gap and does not satisfy that budget.

Feature and logit fidelity:

| Method | Token rel. L2 | Token cosine | Pooled rel. L2 | Pooled cosine | Logit rel. L2 | Logit cosine |
|---|---:|---:|---:|---:|---:|---:|
| L2 approximate only | 0.588 | 0.824 | 0.551 | 0.841 | 0.416 | 0.913 |
| Token 50% + Attn/FFN 25% | **0.473** | **0.886** | **0.330** | **0.944** | **0.232** | **0.973** |
| Token 50% + Attn/FFN 50% | 0.524 | 0.860 | 0.446 | 0.896 | 0.314 | 0.950 |

## FLOP model

The FLOP audit models the intended sparse correction kernels. It does **not**
describe FLOPs actually executed by the dense PyTorch accuracy oracle.
One multiply-accumulate counts as two FLOPs. LayerNorm, softmax, SiLU, RoPE,
residual additions, selection, packing, scatter, and memory traffic are
excluded.

For `N=261`, `D=4096`, and SwiGLU width `M=8192`, one approximate transformer
layer is decomposed as:

```text
token-associated projections = 2 × 4 × N × D²
attention products           = 2 × 2 × N² × D
FFN projections              = 2 × 3 × N × D × M
```

The four token-associated projections are Q, K, V, and attention output.
The two attention products are QK and probability×V. The three FFN
projections are the gate, up, and down projections.

This cost model assumes that all four token-associated projections scale with
the effective token support, that product delta reuses cached base `SV`, and
that all three FFN projections scale with selected channel blocks. A kernel
that must project additional output queries will execute more FLOPs, so the
sparse numbers are target-cost estimates rather than measured instruction
counts.

| Approximate component | FLOPs over 40 layers | Share |
|---|---:|---:|
| Token-associated projections | 1.401 TFLOPs | 39.50% |
| Attention products | 0.0446 TFLOPs | 1.26% |
| FFN projections | 2.102 TFLOPs | 59.25% |
| **Total L2 approximate backbone** | **3.548 TFLOPs** | **100.00%** |

Structured block rounding is included in the correction estimate:

- Patch keep 50% plus 5 special tokens gives 133/261 token-projection work.
- Attention has 17 key blocks; requested keep is rounded up to whole blocks.
- FFN has 64 channel blocks; requested keep is rounded up to whole blocks.

| Policy | Token-projection correction | Attention correction | FFN correction | Total correction | Correction / approx | Approx + correction / approx |
|---|---:|---:|---:|---:|---:|---:|
| Attn+FFN 25% | 0.714 TFLOPs | 0.0240 TFLOPs | 2.102 TFLOPs | **2.840 TFLOPs** | **80.05%** | **180.05%** |
| Attn+FFN 50% | 0.714 TFLOPs | 0.0127 TFLOPs | 1.598 TFLOPs | **2.324 TFLOPs** | **65.52%** | **165.52%** |

Without block rounding the correction ratios are 80.00% and 65.16%,
respectively. The rounding penalty is therefore small.

Although the 25% policy has 33.3% requested equal-component pruning, it saves
only about 20% of approximate FLOPs. Its allocator spends the entire
attention/FFN budget on attention, but attention products account for only
1.26% of this ViT's approximate FLOPs. Full FFN correction alone accounts for
59.25% of approximate FLOPs.

For the 25% policy, FFN work is 74.0% of correction FLOPs. For the 50% policy,
FFN work is 68.7% of correction FLOPs. A FLOP-aware policy should therefore
use the repository's `vit7b_flops` cost mode or a measured-latency cost model
instead of the equal-component allocator used here.

### FLOPs actually executed by the accuracy oracle

The current evaluator does not realize the sparse figures above. At an
intermediate attention support ratio it computes dense QKV, then invokes the
dense `attention_delta` implementation twice. Each invocation evaluates four
QK-like and six probability-value products. Intermediate FFN support computes
dense gate/up projections and sends the masked result through a dense down
projection.

Counting those leading GEMMs gives:

| Dense evaluator work | FLOPs | Relative to one approx |
|---|---:|---:|
| 25% policy branch | 3.919 TFLOPs | 110.48% |
| 50% policy branch | 3.758 TFLOPs | 105.94% |
| Shared support producer | 2.452 TFLOPs | 69.12% |
| Complete two-policy evaluator, including approx and L0 reference | 17.225 TFLOPs | 485.53% |

These values explain why the ImageNet evaluation runtime is only an accuracy
measurement. The target Triton path must avoid the duplicate dense products
and materialize only selected blocks before a wall-clock acceleration claim.

L2 and L0 use the same 261-token transformer shape. Consequently, a sequential
approximate pass followed by correction costs about 1.80× or 1.66× one
backbone inference before overlap. This experiment establishes accuracy under
structured support, not compute-only end-to-end acceleration.

## Layerwise pruning schedule

Patch-token pruning is 50% in every layer; all five special tokens are always
corrected.

| Layer | 25% attention | 25% FFN | 50% attention | 50% FFN |
|---:|---:|---:|---:|---:|
| 0 | 0% | 0% | 10% | 100% |
| 1 | 0% | 0% | 100% | 0% |
| 2 | 0% | 0% | 100% | 0% |
| 3 | 80% | 0% | 100% | 0% |
| 4 | 80% | 0% | 90% | 0% |
| 5 | 80% | 0% | 90% | 0% |
| 6 | 70% | 0% | 90% | 0% |
| 7 | 70% | 0% | 80% | 0% |
| 8 | 70% | 0% | 90% | 0% |
| 9 | 70% | 0% | 80% | 0% |
| 10 | 70% | 0% | 80% | 0% |
| 11 | 50% | 0% | 80% | 0% |
| 12 | 40% | 0% | 80% | 0% |
| 13 | 40% | 0% | 80% | 0% |
| 14 | 40% | 0% | 70% | 0% |
| 15 | 40% | 0% | 70% | 20% |
| 16 | 50% | 0% | 70% | 20% |
| 17 | 40% | 0% | 70% | 20% |
| 18 | 40% | 0% | 70% | 40% |
| 19 | 50% | 0% | 70% | 40% |
| 20 | 40% | 0% | 70% | 50% |
| 21 | 40% | 0% | 70% | 60% |
| 22 | 50% | 0% | 70% | 60% |
| 23 | 50% | 0% | 70% | 60% |
| 24 | 50% | 0% | 70% | 60% |
| 25 | 70% | 0% | 80% | 60% |
| 26 | 50% | 0% | 80% | 40% |
| 27 | 70% | 0% | 80% | 50% |
| 28 | 50% | 0% | 80% | 40% |
| 29 | 70% | 0% | 80% | 40% |
| 30 | 70% | 0% | 80% | 40% |
| 31 | 50% | 0% | 80% | 30% |
| 32 | 50% | 0% | 70% | 20% |
| 33 | 50% | 0% | 70% | 20% |
| 34 | 50% | 0% | 70% | 20% |
| 35 | 50% | 0% | 70% | 20% |
| 36 | 40% | 0% | 70% | 20% |
| 37 | 50% | 0% | 70% | 30% |
| 38 | 50% | 0% | 70% | 20% |
| 39 | 20% | 0% | 50% | 0% |

## Interpretation and limitations

1. The 25% policy is the accuracy-valid operating point, but its correction
   still costs about 80% of the approximate backbone under the sparse FLOP
   model.
2. The 50% policy reduces correction to about 66% of approximate FLOPs, but
   loses 1.554 top-1 percentage points versus L0.
3. Equal-component pruning is poorly aligned with actual compute. Attention
   can be pruned heavily without saving much because FFN and projections
   dominate this model.
4. The ImageNet evaluator is a dense reference: it computes dense QKV,
   probabilities, and FFN intermediates before masking. Its measured runtime
   cannot be used as sparse-kernel latency.
5. The FLOP model is necessary but insufficient for a speed claim. Selector,
   packing, cache traffic, kernel launch overhead, and tensor-core utilization
   must be included in B200 kernel benchmarks.
6. The next policy search should optimize measured B200 latency or at least
   use `cost_mode=vit7b_flops`, then rerun the accuracy gate.

## Reproduction

```bash
# Generate the constrained policy
python analysis/experiments/fit_jacobian_pruning_policy.py \
  --target-pruning 0.25,0.5 \
  --fixed-component-keeps input_token=0.5 \
  --budget-components attention_edge,ffn_channel \
  --output analysis/experiments/results/jacobian_pruning_policy_token50_attnffn.json

# Recompute the FLOP audit
python analysis/experiments/jacobian_policy_flops.py

# Merge the two completed ImageNet shards
python analysis/experiments/merge_jacobian_policy_imagenet.py \
  logs/analysis/jacobian_policy_token50_special_attnffn_shard0.json \
  logs/analysis/jacobian_policy_token50_special_attnffn_shard1.json \
  --output analysis/experiments/results/jacobian_policy_token50_special_attnffn_imagenet1k_summary.json
```

Committed artifacts:

- `jacobian_pruning_policy_token50_attnffn.json`: complete layerwise schedule.
- `jacobian_policy_token50_special_attnffn_imagenet1k_summary.json`: merged
  50,000-image accuracy and fidelity results.
- `jacobian_policy_token50_special_attnffn_flops.json`: reproducible nominal
  and block-rounded FLOP accounting.
