# Runtime-valid strict-token correction: small-set result

## Implemented semantics

This follow-up removes oracle-only support from the correction workload.

- Input token support is selected once from the prepared L2-to-L0 embedding
  residual. The 5 CLS/register tokens are always active and 128/256 patch
  tokens are retained, giving 133/261 active tokens.
- Inactive tokens execute no correction LayerNorm, QKV, attention query,
  output projection, or FFN. Each inactive layer output is the cached L2
  output.
- Active attention queries use cached draft K/V for inactive keys and
  corrected K/V for active keys. The exact corrected softmax denominator still
  covers every key.
- Attention key blocks are selected only from draft probabilities. A new
  Triton consumer reads query-block/head-group-specific descriptors and
  computes selected `P1 V1 - P0 V0`; it does not materialize an expanded mask.
- FFN gate/up weights share a model-static rank-16 right subspace. Approx
  caches `z0 = x0 B`. Correction predicts channel-block energy from `z0` and
  `z1`, then computes exact finite differences only for the selected
  `W_gate`, `W_up`, and `W_down` blocks. Exact base gate/up outputs are retained
  from the approx FFN.

The low-rank factor is a randomized subspace iteration followed by a small
Rayleigh-Ritz solve of the joint
`W_gate.T W_gate + W_up.T W_up` operator. No hidden-by-hidden covariance is
materialized.

## Validation

The 100% strict-token path was checked on three real images:

| Metric | Result |
|---|---:|
| L0 top-1 match | 3/3 |
| Token relative L2 | 0.0101 |
| Pooled relative L2 | 0.00752 |
| Logit relative L2 | 0.00598 |
| Logit cosine | 0.999982 |

The remaining error is BF16/attention arithmetic-order drift. CUDA tests check
the new block descriptor kernel against an explicit FP32 selected-edge
reference.

## 50-image trade-off

The subset uses ImageNet validation indices `0,1000,...,49000`, covering
classes across the sorted validation tree. This is a small paired comparison,
not a replacement for full ImageNet evaluation.

| Method | Top-1 | L0 match | Logit rel. L2 | Logit cosine | Correction / approx | Support prep / approx | Added cache |
|---|---:|---:|---:|---:|---:|---:|---:|
| L0 full | 82% | 100% | 0 | 1 | 100% | 0 | — |
| L2 approx | 80% | 92% | 0.400 | 0.919 | 0 | 0 | — |
| Token 50%, Attn/FFN full | **82%** | **96%** | **0.179** | **0.984** | **50.96%** | 0 | baseline caches only |
| Token 50%, policy-25 | **82%** | 94% | 0.221 | 0.975 | **50.81%** | 0.153% | 38.4 MiB |
| Token 50%, policy-50 | 80% | 92% | 0.320 | 0.948 | **42.06%** | 0.117% | 234.4 MiB |

Realized structured support:

| Method | Active tokens | Attention edges | FFN channels |
|---|---:|---:|---:|
| Token only | 50.96% | 100% | 100% |
| Policy-25 | 50.96% | 55.15% | 100% |
| Policy-50 | 50.96% | 31.70% | 77.96% |

Rank 16 was retained over rank 32. On the same 50-image numerical reference,
rank 16 gave logit relative L2 0.3203 versus 0.3218 for rank 32 while requiring
less predictor work. The final rank-16 Triton-integrated result is 0.3201.

## Interpretation

Strict token pruning is the useful first-stage reduction: it halves all large
projection and FFN workloads and preserves full-set top-1 on this subset.

The policy-25 allocation is not useful. It prunes attention edges, but
attention products are only about 1.26% of the ViT-7B approximate FLOPs. It
saves only 0.14 percentage points beyond token-only while increasing error and
cache.

Policy-50 saves a further 8.90 percentage points of approximate FLOPs, mainly
through FFN channel pruning. On this small subset it loses the token-only
accuracy benefit, so it is not yet an acceptable operating point. The next
allocator should spend budget according to measured FLOPs/latency and search
between full FFN and the current 77.96% realized FFN support.

The reported evaluator runtime (14.86 seconds for all 50 images) executes L0,
L2, and three correction branches together. It is not a per-policy
end-to-end latency measurement. Logical FLOPs include the low-rank predictor
and exact selected GEMMs; LayerNorm, softmax, selector reductions, scatter, and
memory traffic remain excluded.

## Reproduction

```bash
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 \
python analysis/experiments/jacobian_policy_imagenet_eval.py \
  --data-root "$HOME/data/imagenet_val" \
  --policy-json \
    analysis/experiments/results/jacobian_pruning_policy_token50_attnffn.json \
  --targets 0.25,0.5 --base-level 2 --keep-special-tokens \
  --runtime-valid-support --include-token-only \
  --ffn-low-rank 16 --ffn-low-rank-oversample 8 \
  --ffn-low-rank-power-iterations 1 \
  --shard-index 0 --num-shards 1000 --max-samples 50 \
  --batch-size 1 --device cuda:0 \
  --output /tmp/jacobian_runtime_tradeoff_kernel_rank16.json
```

Tests:

```bash
PYTHONPATH="$PWD" python -m pytest -q \
  analysis/experiments/test_jacobian_policy_imagenet_eval.py \
  analysis/experiments/test_jacobian_support_math.py \
  analysis/experiments/test_jacobian_policy_flops.py

PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 \
  python analysis/experiments/test_jacobian_attention_kernel.py
```
