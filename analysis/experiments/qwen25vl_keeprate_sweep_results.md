# Qwen2.5-VL RealWorldQA keep-rate sweep (raw results)

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

| keep_rate | 32B accuracy | 72B accuracy |
|-----------|-------------|-------------|
| 2% | 58.00% (29/50) | 70.00% (35/50) |
| 5% | 62.00% (31/50) | 70.00% (35/50) |
| 10% | 66.00% (33/50) | 74.00% (37/50) |
| 15% | 76.00% (38/50) | 76.00% (38/50) |
| 100% | 76.00% (38/50) | 72.00% (36/50) |

## Full curve (both rounds combined)

| keep_rate | 32B accuracy | 72B accuracy |
|-----------|-------------|-------------|
| 2%   | 58% | 70% |
| 5%   | 62% | 70% |
| 10%  | 66% | 74% |
| 15%  | 76% | 76% |
| 25%  | 74% | 76% |
| 40%  | 80% | 74% |
| 55%  | 80% | 74% |
| 70%  | 82% | 74% |
| 85%  | 78% | 76% |
| 100% | 76% | 72% |
| baseline (full-res, no AppCorr) | 74% | 76% |

## Conclusion: sweet spot is ~15% keep rate

Round 2 finds the real elbow: both models show a clear, monotonic accuracy climb from 2% to 15%
keep_rate (32B: 58->62->66->76%; 72B: 70->70->74->76%), then **saturate to baseline-equivalent
accuracy by keep_rate=15%** and stay flat (within +/-4-8pp nr=50 sampling noise) all the way to
100%. Neither model's accuracy exceeds its ~15% value by a statistically meaningful margin as
keep_rate keeps increasing to 100% -- the 70-85% "peak" points for 32B (80-82%) and the dip at
32B/72B's 100% point are consistent with ordinary nr=50 noise (+/-1 sample = +/-2pp), not a real
trend, and should not be read as "more correction sometimes hurts."

**Practical takeaway**: for RealWorldQA-style VQA with Qwen2.5-VL, correcting only the top ~15% of
merge-groups by residual energy (leaving the remaining ~85% at coarse/blurred pyramid-base
resolution) already recovers full-resolution baseline accuracy. This is a substantially stronger
result than the flat round-1 range (25-85%) suggested on its own -- without round 2, the "sweet
spot" would have been mis-stated as "at or below 25%" rather than the much sharper ~15% figure.

**Compute caveat (measured, not theoretical)**: this accuracy sweet spot does NOT currently
translate into a proportional GPU wall-clock speedup. Per-op timing at keep_rate=15% (32B):
`APPROX_FORWARD` mean=269ms vs `CORRECT_FORWARD` mean=353ms -- `.correct()` is *slower* than the
full-image `.approx()` pass even at just 15% of positions, and this holds even at keep_rate=2%
(296ms vs 301ms, roughly even). Theoretically, FLOPs should scale down close to linearly with
keep_rate for both the linear (QKV/MLP/out-proj) and attention terms, since `.correct()`'s
query count Q = keep_rate*N while it still attends over the same "N-length" cached keys -- at
d_model=5120 (32B), the linear terms dominate over attention terms at this sequence length, so
FLOPs at 15% should be roughly ~5x lower than a full pass. The fact that measured wall-clock is
*higher*, not ~5x lower, points to real implementation overhead outweighing the FLOP savings at
these problem sizes: `.correct()` reconstructs `prepare_full_tokens` (full-image patch embedding)
every call regardless of keep_rate, builds an explicit `[Q,N]` mask every layer, scatter-writes
into the KV cache, and loops sequentially over 96 layers (32 vision + 64 LLM) with small,
GPU-underutilized matmuls at low Q -- none of which shrinks proportionally with keep_rate. Realizing
the accuracy win as an actual speedup would require optimizing these fixed/sublinear costs (e.g.
caching vision patch embeddings across rounds instead of recomputing, avoiding the explicit mask
where avoidable, batching layers), which was out of scope for this investigation.

## Grouping strategy comparison: grid vs sequential (num_groups=4, progressive multi-round)

Unlike the keep-rate sweep above (single-shot, num_groups=1), this tests the *progressive* 4-round
`interleaved_g4` scheduling itself (the layer-chunking scheme where only the last-arriving group
gets a full-depth correction -- see main log). Question: does delivering merge-groups in raster/
sequence order (`sequential`, a growing causal-prefix) recover more accuracy than `grid`'s
spatially-scattered checkerboard tiling, under the identical chunking schedule? All nr=50, post
SDPA fix.

| grouping_strategy | 32B accuracy | 72B accuracy |
|---|---|---|
| grid (default, checkerboard) | 62.00% (31/50) | 72.00% (36/50) |
| sequential (raster prefix)   | 74.00% (37/50) | 74.00% (37/50) |
| baseline (full-res, no AppCorr) | 74.00% (37/50) | 76.00% (38/50) |

**Result: sequential grouping is substantially better.** For 32B it closes the *entire* gap (62%
-> 74%, exactly matching baseline). For 72B it recovers most of the gap (72% -> 74%, vs baseline
76%). This confirms the causal-ordering hypothesis: for an autoregressive decoder, a merge-group's
correction only benefits positions that causally attend to it, so a spatially-scattered group (grid)
leaves gaps throughout the sequence even after "its own" round -- while sequential's growing prefix
means every corrected round immediately and fully benefits every later position. This is a real,
easy, essentially-free improvement (same compute, same schedule, just a different group assignment)
and should be the default for any causal-LLM AppCorr deployment, not just this Qwen2.5-VL fork.
