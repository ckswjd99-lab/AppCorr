# SAM 3 on SA-Co/Gold: cgF1, and what it separates that AP cannot

SA-Co is SAM 3's own benchmark and the only one here that asks about concepts that are **not
present**. A datapoint is an (image, noun-phrase) pair; ~80% of pairs have no masks, and answering
"none" is part of the score:

```
cgF1 = positive_micro_F1 x IL_MCC
        (mask quality)     (does the concept occur at all)
```

Scored with Meta's own evaluator, vendored under `analysis/experiments/saco_eval/` — Hungarian
matching, not COCOeval's greedy. The three files per subset are three independent annotators and
`CGF1Evaluator` scores against each, keeping the most favourable ("oracle setting"), and drops any
pair not instance-exhaustive in all three.

Produced by `scripts/sam3_saco_gold.sh`. Presence token **on** here (see `--presence auto`).

## Subsets are chosen by object size

Size is the only variable shown to predict approximation damage — rarity, vocabulary size and task
type were all orthogonal across COCO and LVIS (`sam3_lvis_results.md`). Median annotation area as a
fraction of image area, and the share below COCO's "small" threshold:

| subset | median area | % small | images | prompts |
|---|---:|---:|---:|---:|
| crowded | 0.0028 | 54.1% | 846 | 20,687 |
| sa1b | 0.0029 | 51.9% | 997 | 13,258 |
| metaclip | 0.0058 | 41.3% | 606 | 33,393 |
| *COCO val2017* | *0.0063* | *39.5%* | — | — |
| fg_food | 0.0077 | 30.1% | 1,506 | 13,951 |
| wiki_common | 0.0155 | 25.6% | 7,610 | 65,502 |
| fg_sports_equipment | 0.0209 | 23.1% | 1,346 | 12,166 |
| **attributes** | **0.0498** | **16.5%** | 2,890 | 9,245 |

`crowded` and `sa1b` are the only two with smaller objects than COCO, so they are where correction
can be read at all. Reporting a single subset as "SA-Co" would be misleading — the domains differ by
18x in median object area.

## `attributes`: approximation is nearly free, and recognition is untouched

Full subset, 2,890 images / 9,245 prompts (1,812 positive).

| term | floor | one-shot | `pre_global` | ceiling | relative loss |
|---|---:|---:|---:|---:|---:|
| **cgF1** | 0.5396 | — | 0.5436 | 0.5421 | **0.5%** |
| positive_micro_F1 | 0.7216 | — | 0.7271 | 0.7254 | 0.5% |
| **IL_MCC** | **0.7478** | — | 0.7477 | **0.7474** | **−0.1%** |
| IL_FPR | 0.0727 | — | — | 0.0776 | — |

**Two results, neither of them a recovery number.**

**The gap is 0.5%, so correction cannot be measured here** — the corrected arm lands *above* the
ceiling, which at this width is noise, not a finding. That is itself the point: this subset's objects
are 8x COCO's median area, an L2 pyramid barely touches them, and **the method has nothing to do on
large-object domains**. Worth knowing where a technique is unnecessary.

**Recognition is immune to the approximation.** IL_MCC moves −0.1% — deciding whether "black tire"
occurs in an image survives an L2 downsample, even where outlining it would not. AP cannot show this
because it fuses finding and outlining into one number; cgF1's factorisation separates them. It also
suggests why the whole approach works: the approximate pass has already settled recognition, and
correction is only refining boundaries.

The ceiling, cgF1 **0.5421**, sits on the paper's **54.1** for SA-Co/Gold. Different scope (one
subset against seven) so not a reproduction, but the sanity the first attempt's 0.0147 plainly
lacked.

## Faults this benchmark exposed that COCO and LVIS could not

Four, in the order they surfaced:

1. **The harness ignored SAM 3's presence token.** The documented score is
   `pred_logits.sigmoid() * presence_logits.sigmoid()`; we used the query logit alone, discarding the
   recognition half and answering "present" for 98% of negative prompts. cgF1 0.0147 → 0.5655.
   Invisible on COCO/LVIS, which only ever prompt categories the GT says are present — and *harmful*
   there (−1.03pp), so `--presence auto` keeps it off for them.
2. **Scoring ran over the whole subset, not the images evaluated.** `CGF1Evaluator` walks every
   exhaustive pair it loaded and counts a prediction-less positive as IL_FN; a 200-image slice was
   graded against 9,222 prompts instead of its 757.
3. **Meta's evaluator aliases predictions across the three annotators**, mutating them in `loadRes`
   so a and b/c take different branches and derive `area` differently — and on numpy ≥ 1.25 it
   crashes first. Needs the oracle setting to appear at all.
4. **One 44-token prompt aborts an 846-image run.** SAM 3's text encoder is CLIP with a 32-token
   limit, `Sam3Processor` silently drops `truncation`/`max_length`, and exactly one prompt in all of
   SA-Co/Gold exceeds it.

None of these could fire on COCO or LVIS. Plausible numbers from an easier configuration are not
validation of the harness.

## `crowded` and `sa1b`: the two subsets with a measurable gap

Full subsets. Preservation is `arm / ceiling`; recovery is `(arm - floor) / (ceiling - floor)`.

**`crowded`** — 846 images, 20,687 prompts (24% positive), object side 10.2px at L2.

| term | floor | one-shot | `pre_global` | ceiling | one-shot prsv | `pre_global` prsv |
|---|---:|---:|---:|---:|---:|---:|
| **cgF1** | 0.5315 | 0.5793 | 0.5762 | 0.5895 | **98.3%** | **97.8%** |
| positive_micro_F1 | 0.5989 | 0.6480 | 0.6474 | 0.6570 | 98.6% | 98.5% |
| IL_MCC | 0.8874 | 0.8940 | 0.8901 | 0.8972 | 99.6% | 99.2% |

**`sa1b`** — 997 images, 13,258 prompts (58% positive), object side 16.3px at L2. Measured *after*
the pyramid-direction fix; see the note below.

| term | floor | one-shot | `pre_global` | ceiling | one-shot prsv | `pre_global` prsv |
|---|---:|---:|---:|---:|---:|---:|
| **cgF1** | 0.5278 | 0.5378 | 0.5376 | 0.5394 | **99.7%** | **99.7%** |
| positive_micro_F1 | 0.6130 | 0.6192 | 0.6188 | 0.6213 | 99.7% | 99.6% |
| IL_MCC | 0.8610 | 0.8686 | 0.8688 | 0.8682 | 100.0% | 100.1% |

Damage splits the same way it did on `attributes`: `crowded` loses 8.8% of mask quality and 1.1% of
IL_MCC. Correction repairs the mask term (98.5% preserved) and has almost nothing to do to
recognition. The approximate pass settles *what is there*; correction refines *where its edges are*.

### Recovery rate exaggerated a 0.5pp difference into 5.3pp

On `crowded`, one-shot recovers 82.5% of the gap and `pre_global` 77.2% — a 5.3pp spread that looked
like the first real evidence that interleaving costs accuracy. In preservation the same two numbers
are 98.3% and 97.8%, and the absolute difference is 0.0031 cgF1. Dividing by a 0.0579 gap magnified
it 19x. `sa1b` then shows the two arms identical (0.5378 vs 0.5376, both 99.7%).

Across all six measurements — COCO tracker/detector, LVIS, and these three subsets — one-shot and
`pre_global` sit within 0.5pp of preservation of each other, with `pre_global` at **0.60x the
correction compute**. That is the claim; "more accurate" was withdrawn earlier and "less accurate"
does not survive either.

### The pyramid rule has two halves and only one was implemented

`sa1b`'s first run showed floor *above* ceiling (cgF1 0.5418 vs 0.5394), which was explained here as
high native resolution leaving objects legible at L2 (24.4px per object side against `crowded`'s
10.2). That explanation was wrong. `l2_from_native` built the pyramid from the original
unconditionally, which is correct only when the native image is *smaller* than the model input; when
it is larger the image must be fitted to the input first. `sa1b` is 2089x1500 against a 1008 canvas —
100% of its images over the limit, where COCO and LVIS medians are 428 and the other Gold subsets
~600 — so its L2 short side was 375 where the rule gives 252, **1.49x too mild**.

Corrected, the object side is 16.3px, the floor drops to 0.5278, and the gap opens to 2.2%. `sa1b`
was never a counterexample to the size law; it was a third point on it:

| subset | object side at L2 | approximation cost |
|---|---:|---:|
| crowded | 10.2 px | 9.8% |
| sa1b | 16.3 px | 2.2% |
| attributes | 38.6 px | 0.5% |

Only `sa1b` was affected — COCO and LVIS have no images over the canvas at the median, and the
ceiling arm never touches this path (it reproduced 0.5394316677 exactly across the fix).

## Discussion: content-adaptive selection

Every arm here recomputes a **fixed 55% of tokens on every image**, and the subsets show how badly
that mismatches what each one needs. With zero recompute:

| run | floor / ceiling | object side in L2 |
|---|---:|---:|
| SA-Co `attributes` | **99.5%** | 38.6 px |
| SA-Co `crowded` | 90.2% | 10.2 px |
| COCO tracker | 89.4% | ~30 px |
| COCO detector | 85.1% | ~30 px |
| LVIS detector | 73.1% | ~30 px |

`attributes` is already at 99.5% of ceiling before any correction runs, so its 55% is spent buying
0.5pp. LVIS starts at 73.1% and needs every token it gets. A fixed ratio pays the same on both.

**So a threshold on the patch score, rather than a top-k, should cut total recompute at equal
accuracy** — the score is residual energy x average attention, and on large-object content there is
little residual energy anywhere, so few patches would clear a threshold. The saving would come from
adapting the *rate* per image, not from choosing better patches: `pyramid_degradation_native_vs_canvas.md`
measured top-k and threshold as equivalent at matched recompute (n=100: 50.00% -> 0.6969 vs 50.09%
-> 0.6987, inside noise). Rate, not ranking, is what a threshold buys.

**The one existing measurement is not encouraging, and says what to fix.** In DINOv3 the shipped
`token_keep_thres=0.002` selected 20.4% of tokens *identically before and after* the base image was
made 4x more degraded — the threshold did not react to how bad the approximation was, which is the
entire premise. Whatever quantity the threshold is applied to has to move with approximation damage;
that one evidently did not, so it behaved as an awkward way of setting a fixed rate.

Two things to check before treating this as a result: whether the per-image score distribution
actually separates `attributes` from `crowded` (measurable offline from the pscore, no arms needed),
and whether a threshold tuned on one dataset transfers, since the useful claim is a single operating
point that adapts, not a per-dataset knob. Note also that any variable-rate scheme complicates the
interleaved cost model, which assumes a known token count per round.
