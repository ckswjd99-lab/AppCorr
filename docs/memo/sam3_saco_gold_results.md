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

## `crowded` and `sa1b`

Running.
