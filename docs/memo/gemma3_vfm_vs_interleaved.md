# Gemma 3: splitting the LLM half is worth it after all

**Decision (2026-08-25): interleaved. The `vfm` arm is kept as a measured negative result, not
as the shipping configuration.**

## What was being tested

Gemma 3's vision tower is 5.45 TF against 1.73 TF for the LLM's share of the image tokens -- vision
is 73.5% of the backbone and 3.2x the image-token prefill. That raised an obvious question: if the
LLM half is a minority of the compute, why interleave it at all? Do what the VFMs do -- correct the
vision half progressively, then run the prefill **once**, exactly, at the end.

`vfm_forward()` in `appcorr/models/gemma3/unified.py` implements that. Its vision schedule is
deliberately identical to `interleaved_forward`'s -- same bounds, same bands, same per-round group --
so a comparison isolates the LLM-half change and nothing else. Gated by
`analysis/experiments/gemma3_vfm_gates.py`: `g=1, keep=1.0` reproduces the ceiling token for token,
and the vision K/V cache matches `interleaved_forward`'s at the same selection (rel 0.00e+00).

## Result: accuracy is a wash, and critical computation is 2.6-3.5x worse

Accuracy (ceiling in parentheses):

| | interleaved k25 / k50 | vfm k25 / k50 |
|---|---|---|
| ChartQA (58.76) | 53.84 / 57.20 | **55.08 / 58.88** |
| TextVQA (60.87) | 59.15 / 59.88 | 59.45 / 59.77 |
| InfoVQA (39.56) | **29.02** / 32.92 | 28.84 / 32.92 |

vfm wins ChartQA by 1.2-1.7pp (at keep 50% it even passes the ceiling), and the other two are
indistinguishable.

Critical computation, GFLOPs per instruction, and as a share of the ceiling prefill:

| | interleaved | vfm |
|---|---|---|
| ChartQA k25 | 924 (12.5%) | 2431 (33.0%) |
| TextVQA k25 | 614 (8.4%) | 2157 (29.5%) |
| InfoVQA k25 | 750 (10.2%) | 2274 (30.9%) |

Total compute favours vfm by 2-3% (14,759 vs 15,080 GF on InfoVQA k25), which is noise next to the
critical difference.

## Why interleaved wins on the metric that matters

The whole claim is about what can only start once the last byte has arrived. `vfm_forward` cannot
begin the prefill until the vision features are final, so **all** of it lands after the last arrival.
That is the 2.6-3.5x, and it is structural rather than an implementation cost -- no amount of tuning
removes it.

Trading 1.2-1.7pp on one dataset for a 3x worse critical share is the wrong trade for this project.

## The negative result is the point, and worth stating in the paper

The hypothesis was reasonable and specific: **when the vision tower dominates the FLOPs, the LLM half
is not worth interleaving.** It is false, and measurably so. Interleaving the LLM half costs almost
nothing in accuracy and buys a 3x reduction in deferred computation, *even on a model where the LLM
is a minority of the total work*. The FLOPs share turns out to be the wrong thing to reason from --
what matters is that the prefill is a single indivisible block at the end unless you split it.

One caution if this is quoted: the wall-clock and FLOPs rankings for Gemma 3 disagree. Vision
dominates in FLOPs (73.5%) while the LLM prefill dominates in wall clock (16.96 ms against 9.39 ms),
because a ~290-token prefill runs small, poorly-utilised matrices. Quote whichever matches the claim
being made and do not mix them.
