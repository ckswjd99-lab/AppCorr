# InternVL: where approx-then-correct is not needed

InternVL3 was picked as a fourth model family because its tiling is a different high-resolution
strategy from Qwen's dynamic resolution. Measuring it produced a more useful result than "the method
also works here": **this is a model where the method is unnecessary, and it can be shown to be.**

## The claim

InternViT encodes each 448x448 tile independently — tiles are the batch dimension and attention never
crosses them — and the LLM is causal over a sequence in which each tile's tokens form one contiguous
block. So a serving system can transmit tile 0, encode and prefill it while tile 1 is in flight, and
so on, overlapping compute with transmission and arriving at **exactly** the monolithic result.

If that holds there is no staleness to correct, and approximation only costs accuracy for nothing.

## It holds

`analysis/experiments/internvl_tile_pipeline_check.py`, InternVL3-2B, fp32:

| check | result |
|---|---|
| encoder, batched vs one tile at a time (same input) | **rel 0.000e+00**, all 24 layers |
| patch-embedding conv on **CPU**, batch 7 vs 1 | **rel 0.000e+00** |
| image tokens form one contiguous span | 1792 = 7 x 256, contiguous |
| chunked prefill (9 chunks) vs monolithic | rel 1.24e-05 |

The one check that "failed" on GPU — encoding tiles one at a time vs together, rel 1.28e-02 — is
**cuDNN, not structure**. The divergence starts at the patch-embedding conv (rel 2.1e-04) and is
identical with `cudnn.deterministic=True`, because the algorithm *choice* depends on batch size; the
same conv on CPU is bit-exact. It then amplifies through 24 layers. This is the difference any change
of inference batch size produces, and nobody calls that lossy.

## What this bounds

Tile pipelining requires two properties, and most models here have neither:

| model | vision | LLM attention over image | lossless streaming |
|---|---|---|---|
| **InternVL3** | tiles independent | causal | **yes, per tile — verified** |
| LLaVA-OneVision | tiles independent | causal | yes, but per tile *row*: `pack_image_features` reassembles crops into one grid and flattens raster-order, interleaving them |
| Qwen2.5-VL | full attention across the whole image at layers 7/15/23/31 | causal | no — blocked in the vision tower |
| PaliGemma / Gemma 3 / pi0-FAST | SigLIP, global | **bidirectional** (`use_bidirectional_attention`; "images … attend to all prev images and to itself bidirectionally") | no — blocked on both sides |
| DINOv3 / CLIP / SAM 3 | global or windowed | — | no |

The Gemma row is the sharpest counter-example, and it is already measured (pi0-FAST, 5 frames,
teacher-forced CE / argmax accuracy): full-bidirectional 4.70 / 0.238, **forcing strict token-causal
5.27 / 0.198 — +12% CE, −4pp**. "Just stream it in order" is not free there.

Note what the same measurement also shows: **block-causal** — each chunk internally bidirectional,
attending to prior chunks, text always visible — is essentially lossless (4.70 / 4.65 / 4.71 at
G=2/4/8). The damage came from breaking intra-chunk bidirectionality and text visibility, not from
chunk ordering. That is the shape approx-then-correct already has.

## The measurement, for completeness

InternVL3-8B, full RealWorldQA test split (765), L2 approximation, 55% of patches recomputed,
score = residual energy x average attention. Same `score_answer` as the Qwen2.5-VL work, so the
numbers are comparable across model families.

| arm | accuracy | preservation | recovery |
|---|---:|---:|---:|
| floor (approximate pass only) | 60.78% (465/765) | 89.3% | 0% |
| corrected, 55% | **66.54%** (509/765) | **97.7%** | 78.7% |
| ceiling (exact) | 68.10% (521/765) | 100% | — |

97.7% preservation is in line with every other benchmark measured (SAM 3 COCO/LVIS/SA-Co: 97–100%),
so the technique transfers to a tiled VLM. **It is still the wrong tool for this model**: lossless
tile pipelining gets 100% for the same overlap. The right reading of this table is as the *cost of
approximating when you did not have to* — 2.3pp — not as a success.

Reference point for the ceiling: Qwen2.5-VL on the same 765 examples scores 68.89% at 32B and 72.29%
at 72B. InternVL3-8B at 68.10% is a much smaller model in the same range, which is a sanity check on
the harness rather than a comparison.

## What to do with InternVL

Keep it as the boundary case. The paper's claim is stronger stated as "approx-then-correct is needed
where the encoder is not tile-independent, which is most of them" than as "it works everywhere" —
and this is the measurement that draws the line. Extending the model suite for coverage should
prefer families that *cannot* stream: Gemma-based VLMs are the clearest, and pi0-FAST work already
exists.

## Not measured

- The actual latency of tile pipelining vs approx-then-correct. Both overlap compute with
  transmission and neither reduces bytes; which wins depends on the transmission-to-compute ratio,
  and **that ratio has never been measured in this repo**. Every latency statement so far, here and
  in the SAM 3 memos, is offline-driver time with no link in it.
- LLaVA-OneVision end to end. Its per-tile-row granularity and the `unpad_image` / `interpolate`
  steps after reassembly make partial-arrival handling harder than InternVL's; whether it is
  practical is unknown.
