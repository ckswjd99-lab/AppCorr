#!/usr/bin/env bash
# SAM 3 on SA-Co/Gold -- promptable concept segmentation, scored by cgF1.
#
# Subsets are chosen by OBJECT SIZE, the only variable the approximation has been shown to care
# about -- rarity, vocabulary size and task type all turned out orthogonal across COCO and LVIS;
# size did not. Median annotation area as a fraction of the image, and the share below COCO's
# "small" threshold:
#
#     crowded      0.0028   54.1% small     <- most exposed
#     sa1b         0.0029   51.9%
#     metaclip     0.0058   41.3%
#     COCO val     0.0063   39.5%           (reference)
#     fg_food      0.0077   30.1%
#     wiki_common  0.0155   25.6%
#     fg_sports    0.0209   23.1%
#     attributes   0.0498   16.5%           <- least exposed
#
# `attributes` was run first and gave a floor-ceiling gap of 0.5% (cgF1 0.5396 vs 0.5421) -- too
# narrow to measure recovery in, and its corrected arm landed above the ceiling on noise. Its
# objects are 8x COCO's median area, so the L2 pyramid had almost nothing to destroy. Kept as a
# result in its own right (approximation is nearly free on large-object domains), not as a
# measurement of correction. `crowded` and `sa1b` are the only subsets with smaller objects than
# COCO, so they are where the technique can be read at all.
#
# cgF1 = positive_micro_F1 x IL_MCC also splits mask quality from recognition, and `attributes`
# already answered one open question: IL_MCC was unaffected by approximation (-0.1%), so deciding
# whether a concept is present survives an L2 downsample even when outlining it does not.
#
# Presence token ON here (auto): SA-Co asks about absent concepts and scores the answer. It is off
# for COCO/LVIS, where every prompt is present by construction and presence costs 1.03pp of ceiling.
#
# GT is the 3-annotator oracle setting -- CGF1Evaluator scores against a, b and c and keeps the most
# favourable, and drops any prompt not instance-exhaustive in all three.
set -u
REPO=/NHNHOME/share/cjpark/AppCorr-sam3
PY=/home/nxclab/anaconda3/envs/appcorr/bin/python
LOG=$REPO/logs/sam3_saco_gold
mkdir -p "$LOG"; cd "$REPO" || exit 1

run() {  # subset  tag  extra...
  local sub=$1 tag=$2; shift 2
  echo "##### START $sub/$tag @ $(date +%F' '%H:%M:%S) #####"
  for p in $(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader); do kill -9 "$p" 2>/dev/null; done
  sleep 4
  HF_TOKEN="${HF_TOKEN:?}" CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 timeout 86400 \
    "$PY" analysis/experiments/sam3_coco_oracle.py --dataset "saco_gold:$sub" --path detector \
      --full --max-boxes 100000 --det-per-cat 30 --det-max-dets 100 \
      --out-json "$LOG/${sub}_${tag}.summary.json" "$@" > "$LOG/${sub}_${tag}.log" 2>&1
  echo "##### END $sub/$tag rc=$? @ $(date +%F' '%H:%M:%S) #####"
  grep -aoE '=== Final Summary: .*' "$LOG/${sub}_${tag}.log" | tail -1
}

for sub in crowded sa1b; do
  run "$sub" ceiling         --arm ceiling
  run "$sub" floor           --arm floor
  run "$sub" inter_preglobal --arm corrected --keep-ratio 0.55 --groups 4 --bounds pre_global
  run "$sub" oneshot         --arm corrected --keep-ratio 0.55 --groups 1
done
echo "SAM3_SACO_GOLD_DONE"
