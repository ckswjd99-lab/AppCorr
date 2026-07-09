# Qwen2.5-VL RealWorldQA keep-rate sweep (raw results, in progress)

Methodology: `grouping_strategy=top_energy`, `num_groups=1` (static single-shot correction --
merge-groups ranked by residual energy, top `keep_rate` fraction transmitted/corrected at real
resolution, rest stay approx-only/blurred for the whole request). This sidesteps the layer-chunking
depth confound that `num_groups>1` progressive scheduling has (see main log for details). All runs
at nr=50 (strided sample of RealWorldQA's 765-example test split), post SDPA-kernel-dispatch fix
(commit `38bae33`).

Baseline (full resolution, stock sequential inference) at nr=50: **32B = 74% (37/50)**, **72B = 76% (38/50)**.

## Round 1 (25%-85%)

| keep_rate | 32B accuracy | 72B accuracy |
|-----------|-------------|-------------|
| 25% | 74.00% (37/50) | 76.00% (38/50) |
| 40% | 80.00% (40/50) | 74.00% (37/50) |
| 55% | 80.00% (40/50) | 74.00% (37/50) |
| 70% | 82.00% (41/50) | 74.00% (37/50) |
| 85% | 78.00% (39/50) | 76.00% (38/50) |

Observation: flat across the whole 25-85% range, within noise of baseline (all points within
+/-8pp, mostly +/-2pp) for BOTH models. No visible degradation even at the lowest tested point
(25%) -- the real "elbow" must be below 25%. Round 2 extends downward.

## Round 2 (2%-15%, + 100% reference)

(filled in once /tmp/32b_keeprate_sweep2.log and /tmp/72b_keeprate_sweep2.log complete)
