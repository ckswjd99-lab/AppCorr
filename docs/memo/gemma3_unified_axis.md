# Gemma 3: vision and prefill as one approx/correct axis

Gemma 3 4B is the model where this technique is actually needed, and the first where the LLM half is
worth correcting rather than just the vision tower. This records why, what was built, and the four
things measurement changed.

## Why Gemma 3

**Its image tokens are bidirectional**, so lossless tile-style streaming — which beats
approx-then-correct outright on InternVL — is unavailable. Measured, not read off a comment: the 4D
mask has 28 forward-attending entries, all inside the image block, while text rows are causal.
(PaliGemma differs: image *and* prefix text form one bidirectional block. Either way, streaming is
blocked.) The cost of forcing causality is already known from pi0-FAST: **+12% CE, −4pp**.

**Its vision input is large.** 896x896 at patch 14 = 4096 patches pooled to 256 LLM tokens. A 224px
model would leave almost nothing to approximate.

**It is dense.** Gemma 4 (26B-A4B, 128 experts) is deferred: correcting a token can change which
experts fire, making recompute cost unpredictable and the correctness gates a separate question.

## Why unify, rather than fork the vision tower alone

    vision tower (4096 patches x 27L, h1152)     9.39 ms   27% of forward
    LLM prefill  (277 tokens  x 34L, h2560)     16.96 ms   49%
    full forward                                34.38 ms

**Prefill is 1.8x the vision tower.** Vision-only correction overlaps 27% of the forward; the unified
axis reaches ~76%. Same lesson as VGGT, where treating 24 patch-embed blocks as preprocessing
repeated them on every interleaved round — 28.7% of the forward, wasted.

## What was built

| piece | file | gate |
|---|---|---|
| SigLIP layer fork | `appcorr/models/gemma3/vision/block.py` | 15 assertions; real 27-layer tower **rel 0.00** |
| Gemma decoder fork | `appcorr/models/gemma3/llm/decoder_layer.py` | 12 assertions incl. a wrong-phase RoPE guard |
| 61-stage unified axis | `appcorr/models/gemma3/unified.py` | `full_forward`/`approx`/`correct(all)` all **rel 0.00** vs stock |

Cache: **849.6 MB per image** (vision 764.4, LLM 85.2) — an eighth of the ~10 GB that stalled
DINOv3's Phase 1, and 117 images fit in 100 GB.

## Four things measurement changed

**1. `pre_global` does not transfer.** Gemma 3's 29 sliding + 5 full layers look like SAM 3's 28
windowed + 4 global, so the trick that bought 0.60x compute there should apply. It does not: the
sliding window is 1024 and one image plus a short prompt is ~272 tokens, so sliding never bites and
all 34 layers behave as full attention. Reading the layer list without checking sequence length gave
the wrong answer first.

**2. Vision is not the dominant half.** Prefill is 1.8x it — the reverse of the intuition that 272
tokens must be negligible.

**3. Sliding and full layers need different rope AND different masks.** `Gemma3RotaryEmbedding` takes
a `layer_type`; masks arrive as a `{full_attention, sliding_attention}` dict. Reusing one type across
all layers corrupts the 5 full-attention layers at rel 9e-2 — large, and invisible in an argmax. The
LLM fork's unit test measures exactly this by rolling the phase, which is what makes its *passing*
result meaningful rather than vacuous.

**4. A patch selection cannot be translated across the projector.** Each LLM image token pools 16
patches, so "corrected if any of mine was" saturates. On 20 RealWorldQA images with the real
residual-energy score:

| keep | ideal tokens | real score | random | real/random |
|---:|---:|---:|---:|---:|
| 10% | 26 | 109.6/256 | 210.2/256 | 0.52x |
| 25% | 64 | 162.2/256 | 253.7/256 | 0.64x |
| 55% | 141 | 219.8/256 | 256.0/256 | 0.86x |
| 70% | 180 | 237.9/256 | 256.0/256 | 0.93x |

Real scores cluster — at 10% they touch half the tokens a random selection would — but not nearly
enough: an ideal selection needs 26 tokens, the real one takes 109.6. At 55%, saving 45% of patches
saved 14% of tokens. So the halves get **separate budgets**: the patch score is pooled 16:1 into
per-token scores and the LLM half runs its own top-k. That is also a knob worth having — the tower
and the prefill need not be equally approximation-sensitive.

## The gate that mattered most

The reference call omitted `token_type_ids`, so the stock model built a plain causal mask while the
axis built the bidirectional one; they differed by **rel 0.886**. Read as "the fork is wrong", the
fix would have been to match the reference — producing a Gemma 3 *without bidirectional image
attention*, which is the entire reason this model was chosen over InternVL. The whole experiment
would have measured the wrong thing while passing every check.

## Not done

- No accuracy arms yet: floor / corrected / ceiling on a benchmark.
- Interleaved rounds over the 61 stages, and the `(1/g)·Σ bounds` cost signature.
- Whether the vision and LLM budgets should differ, now that they can.
- Latency: as everywhere in this repo, the transmission-to-compute ratio is unmeasured, so the 76%
  overlap figure is a compute-share claim, not a latency claim.
