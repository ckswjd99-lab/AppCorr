# SAM 3 on COCO: approx-then-correct, one-shot and interleaved

Full COCO val2017, **4952 images / 34,104 annotations** (every image with at least one usable box;
`--max-boxes 20`). `Sam3TrackerModel` prompted with the ground-truth box, mask AP via COCOeval segm.
1008x1008 canvas, 72x72 patch grid = 5184 tokens, bf16, single B200.

Approximate input is pyramid level 2 built **from the native image** and then resized to the canvas,
per `pyramid_degradation_native_vs_canvas.md`. Patch score is residual energy x average attention.
All corrected arms recompute the **same 55%** of tokens (2851 / 5184) — verified, not assumed.

Produced by `scripts/sam3_full_coco.sh` and `scripts/sam3_full_coco_inter.sh` on
`develop/sam3-implement`.

## Results

| arm | mask AP | AP50 | small | medium | large | recovery | correction cost | wall clock |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| floor (approximate pass only) | 0.5374 | 0.8323 | 0.3321 | 0.6178 | 0.7397 | 0% | — | 7.9 min |
| one-shot, 55% | 0.5961 | 0.8767 | 0.4450 | 0.6544 | 0.7431 | 92.4% | 1.00x | 22.5 min |
| interleaved g=4, `aligned` | 0.5958 | 0.8752 | 0.4448 | 0.6536 | 0.7445 | 91.9% | 0.63x | 22.1 min |
| **interleaved g=4, `pre_global`** | **0.5961** | 0.8759 | 0.4453 | 0.6540 | 0.7447 | **92.3%** | **0.60x** | **15.3 min** |
| ceiling (exact forward) | 0.6010 | 0.8772 | 0.4529 | 0.6587 | 0.7465 | 100% | — | 7.0 min |

Recovery is `(arm - floor) / (ceiling - floor)`; the gap is 0.0635 AP.

**The headline: `pre_global` matches one-shot to four decimals (0.5961 vs 0.5961) at 0.60x the
correction compute and 0.69x the wall clock.** Interleaving is not more accurate — it is the same
accuracy, cheaper. Feature-space relative L2 against the exact forward agrees and is the cleaner
statement, since it cannot be beaten by luck: one-shot 0.4664, `aligned` 0.4694, `pre_global` 0.4642
(20 images, FPN outputs).

## Where the loss is, and where it is recovered

| | floor | pre_global | ceiling | loss | recovered |
|---|---:|---:|---:|---:|---:|
| small | 0.3321 | 0.4453 | 0.4529 | −0.1207 | 93.7% |
| medium | 0.6178 | 0.6540 | 0.6587 | −0.0410 | 88.5% |
| large | 0.7397 | 0.7447 | 0.7465 | −0.0068 | 73.5% |

Almost all of the approximation's damage is on small objects — an L2 pyramid discards high-frequency
detail, and large-object masks barely notice (0.68pp). So correction is repairing the thing that is
actually broken. The 73.5% on large objects is a ratio over a 0.68pp gap and means nothing.

## The `pre_global` trick

SAM 3's ViT puts global attention at layers **7, 15, 23, 31**; the other 28 layers use 24x24 windowed
attention (9 windows of 576 tokens). Round `r` corrects its group over layers `[0, bounds[r])`, so an
even split (`aligned` = 8/16/24/32) ends each round *just past* a global layer and re-corrects it in
every later round. Stopping one layer earlier (`pre_global` = 7/15/23/32) defers it.

Counting global layers inside each round's correction range:

| round | `aligned` [8,16,24,32] | `pre_global` [7,15,23,32] |
|---|---|---|
| 0 | 0..7 → {7} | 0..6 → {} |
| 1 | 0..15 → {7,15} | 0..14 → {7} |
| 2 | 0..23 → {7,15,23} | 0..22 → {7,15} |
| 3 | 0..31 → {7,15,23,31} | 0..31 → {7,15,23,31} |
| **total** | **10** | **7** |

**7/10 = 0.70 predicted, 15.3/22.1 = 0.69 measured.** A corrected query on a global layer attends
over all 5184 K/V instead of its window's 576, so global layers dominate correction time; a
layer-count model that weights them equally predicts 0.95x and is wrong. The last round pays all four
regardless — the saving is entirely in the earlier rounds.

## Cost model

Round `r` corrects `|group r|` tokens over `bounds[r]` layers, so total correction work is
`(1/g)·Σ bounds[r]` token-layers against one-shot's `n_keep · num_layers`:

| g | bounds | coverage | cost |
|---|---|---|---:|
| 1 | [32] | == one-shot | 1.00x |
| 4 `aligned` | [8,16,24,32] | == one-shot | 0.63x |
| 4 `pre_global` | [7,15,23,32] | == one-shot | 0.60x |
| 9 `aligned` | [4,7,11,…,32] | == one-shot | 0.56x |
| 16 `aligned` | [2,4,…,32] | == one-shot | 0.53x |

`g=1` is exactly 1.00x because it *is* one-shot — the identity gate below.

## Caveats

**The wall clock understates the compute saving.** `aligned` costs 0.63x the correction FLOPs and
runs in 0.98x the time. Correction is launch-bound in this driver, not FLOP-bound: the schedule needs
`Σ bounds[r] = 80` layer-corrections against one-shot's 32 (inherent — group `r` has never been
corrected in layers `0..bounds[r]`), and each runs a serial 9-iteration Python loop over windows, so
720 SDPA launches against one-shot's 288. Only `pre_global` gets ahead, and it does so by cutting the
*expensive* launches rather than the count. Batching the window loop with padding would collapse 720
launches to 80 and is a pure optimisation — no accuracy effect.

**These are offline-driver timings, not serving latency.** They exclude transmission entirely, which
is what interleaving exists to overlap. A latency claim has to come from the offload pipeline.

## Gates

Per `interleaved_correction_contract.md`, before any of the above was reported:

1. **Coverage** — union of per-round groups == one-shot selection (2851/5184) at g ∈ {1,4,9,16}, both
   bound modes.
2. **`g=1` identity** — the interleaved path forced to one group returns `0.6988702752481474`,
   bit-identical to the one-shot path. (An earlier version differed by 0.0006 and that difference was
   wrongly attributed to float reassociation; it was the leak in rule 2.)
3. **Cost signature** — matches `(1/g)·Σ bounds[r]`, and interleaved comes out *cheaper* than
   one-shot, which is the direction that catches an accumulated corrected set.
4. **Feature-space rel-L2** — no arm beats the exact forward.

Three defects were fixed on the way here and each is written up in the contract memo: the corrected
increment was never persisted; rounds corrected the accumulated arrived set instead of their own
group (which made interleaved 2.5x *more* expensive and inverted the claim); and the input stream
carried full resolution at tokens that had not arrived yet, which is what produced an earlier
"interleaved beats the ceiling at 103.2%" result and a since-retracted claim that interleaving is
8.3% more faithful than one-shot.

## Not measured

- The detector path (`Sam3Model`) — oracle only (0.5496 at 500 images); no floor/corrected arms. Its
  `--det-score-thresh 0.3` is arbitrary and emits 2.2x more predictions than there are GT boxes.
- Keep ratios other than 55%.
- `g` other than 1 and 4 on the full set.
