#!/usr/bin/env bash
# SAM 3 oracle baselines on COCO val2017, 500 images, both paths.
#
# 500 not 5000: this is the run that turns "the wiring works" into a usable reference and measures
# throughput, so the full-set decision is made on a measured rate rather than a guess. Both arms use
# the SAME slice and the SAME --max-boxes 20, because the 20-image smoke had max_boxes=5 capping the
# tracker's prompts while the detector had no such limit -- that comparison was not fair.
#
# Smoke results (20 images, max_boxes=5, NOT comparable to these):
#   tracker  AP 0.530  AP50 0.708  small 0.313  preds 77
#   detector AP 0.517  AP50 0.711  small 0.229  preds 240
#
# The detector's --det-score-thresh 0.3 is still an arbitrary pick; it emitted 3x more predictions
# than there are ground-truth objects. Recorded in the summary so a later sweep has a reference.
set -u
REPO=/NHNHOME/share/cjpark/AppCorr-sam3
PY=/home/nxclab/anaconda3/envs/appcorr/bin/python
LOG=$REPO/logs/sam3_oracle
mkdir -p "$LOG"; cd "$REPO" || exit 1

for path in tracker detector; do
  echo "##### START $path @ $(date +%H:%M:%S) #####"
  # GPU 0 only. GPU 1 belongs to another user's job -- never sweep compute-apps without -i.
  HF_TOKEN="${HF_TOKEN:?HF_TOKEN must be set by the caller}" CUDA_VISIBLE_DEVICES=0 \
  PYTHONUNBUFFERED=1 timeout 21600 \
    "$PY" analysis/experiments/sam3_coco_oracle.py \
      --path "$path" --num-images 500 --max-boxes 20 \
      --out-json "$LOG/${path}_500.json" \
    > "$LOG/${path}_500.log" 2>&1
  echo "##### END $path rc=$? @ $(date +%H:%M:%S) #####"
  grep -aoE '=== Final Summary: .*' "$LOG/${path}_500.log" | tail -1
  # rc alone is not proof: a run that dies at startup can exit 0 with no summary at all.
  echo "summary_present=$(grep -ac 'Final Summary' "$LOG/${path}_500.log")"
  sleep 10
done
echo "SAM3_ORACLE_DONE"
