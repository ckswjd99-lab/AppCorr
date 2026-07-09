# Qwen2.5-VL RefCOCO (val) keep-rate sweep

Methodology identical to the RealWorldQA sweep: `grouping_strategy=top_energy`, `num_groups=1`
(static single-shot correction). All runs nr=50 (strided sample of RefCOCO val's 8811 examples).
Task: referring-expression comprehension (text -> bounding box), scored Acc@0.5 (IoU >= 0.5) plus
mean IoU as a more continuous signal. Driver: `analysis/experiments/refcoco_offload_eval.py`. See
that file's docstring for the input-direction note (dataset's own template is captioning-direction;
this driver uses it in the standard grounding direction, `answer[0]` as the referring expression).

## Full curve

| keep_rate | 32B Acc@0.5 | 32B mean IoU | 72B Acc@0.5 | 72B mean IoU |
|-----------|------------|-------------|------------|-------------|
| 2%   | 72% (36/50) | 0.621 | 90% (45/50) | 0.800 |
| 5%   | 72% (36/50) | 0.631 | 88% (44/50) | 0.785 |
| 10%  | 78% (39/50) | 0.640 | 88% (44/50) | 0.769 |
| 15%  | 74% (37/50) | 0.654 | 92% (46/50) | 0.783 |
| 25%  | 76% (38/50) | 0.655 | 90% (45/50) | 0.781 |
| 50%  | 88% (44/50) | 0.762 | 94% (47/50) | 0.796 |
| 100% | 90% (45/50) | 0.791 | 90% (45/50) | 0.815 |
| baseline (full-res, no AppCorr) | 84% (42/50) | 0.743 | 90% (45/50) | 0.813 |

## Interpretation

**This is the clearest task-dependent signal across all three datasets tested this session.**
Unlike RealWorldQA and GQA, RefCOCO's grounding task shows a **real, visible cost at low keep_rate
for 32B**: both Acc@0.5 and mean IoU climb steadily from keep_rate=2% (72%, IoU 0.621) up through
50% (88%, IoU 0.762) before leveling off near keep_rate=100%/baseline (90%/84%, IoU
0.79/0.74) -- a genuine ~15-16pp accuracy gap and ~0.12-0.14 IoU gap between the low and high ends
of the curve, tracking together (accuracy and the more continuous IoU signal agree, which is a good
consistency check that this isn't a metric-thresholding artifact). **32B's elbow for RefCOCO sits
around keep_rate=50%, substantially higher than RealWorldQA's ~15%** -- i.e. the hypothesis that
precise spatial localization needs more of the image corrected than semantic VQA does is
*supported* by this data, at least for the smaller model.

**72B is flat and near-baseline across the entire curve** (88-94% Acc@0.5, IoU 0.77-0.82 throughout,
including keep_rate=2%) -- the same robustness pattern seen on RealWorldQA and GQA. For this model
size, even 2% keep rate is enough to localize the referred object at baseline accuracy on this
sample. Notably 72B's accuracy is already very high even at the coarsest setting -- referring
expressions in RefCOCO's val split are frequently the *only* prominent object of that description in
the scene, so even a blurred image may carry enough coarse spatial/color/category signal for a
72B-scale model to localize it approximately; a harder or more cluttered grounding benchmark might
show a lower floor.

**Cross-dataset conclusion**: the "how much correction is needed" question does **not** have a
single universal answer -- it is genuinely task-dependent (RefCOCO's spatial grounding needs
substantially more than RealWorldQA's/GQA's semantic VQA, at least for the 32B model) and
model-size-dependent (72B is robust across all three tasks tested at even the lowest keep_rate
tried, 2%, while 32B's required keep_rate varies from ~15% to ~50% depending on the task). See
`QWEN25VL_APPCORR_LOG.md`'s cross-dataset comparison section for the full picture.
