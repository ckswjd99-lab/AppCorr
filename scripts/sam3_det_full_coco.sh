#!/usr/bin/env bash
# SAM 3 DETECTOR path (Sam3Model, category name as text prompt), full COCO val2017, five arms.
#
# Readout protocol fixed at --det-per-cat 30 --det-max-dets 100, chosen by sweeping the CEILING arm
# to convergence (logs/sam3_det_protocol): AP 0.4044 / 0.5418 / 0.5570 / 0.5631 / 0.5635 at
# per_cat 1/5/10/30/100 on 200 images; 100 saturates maxDets and buys +0.04pp over 30. The former
# hand-set --det-score-thresh 0.3 scores 0.5535 on the same slice. Tuning on the ceiling is the
# point: the exact forward gets its best score, so approximations are read against a reference that
# is not handicapped by the protocol.
#
# Unlike the tracker path this folds FINDING the objects into the same number as outlining them, so
# a drop here does not say which of the two got worse. The tracker table is the one that isolates
# segmentation quality.
set -u
REPO=/NHNHOME/share/cjpark/AppCorr-sam3
PY=/home/nxclab/anaconda3/envs/appcorr/bin/python
LOG=$REPO/logs/sam3_det_full_coco
mkdir -p "$LOG"; cd "$REPO" || exit 1

run() {
  local tag=$1; shift
  echo "##### START $tag @ $(date +%F' '%H:%M:%S) #####"
  for p in $(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader); do kill -9 "$p" 2>/dev/null; done
  sleep 4
  HF_TOKEN="${HF_TOKEN:?}" CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 timeout 86400 \
    "$PY" analysis/experiments/sam3_coco_oracle.py --dataset coco --path detector \
      --full --max-boxes 20 --det-per-cat 30 --det-max-dets 100 "$@" > "$LOG/${tag}.log" 2>&1
  echo "##### END $tag rc=$? @ $(date +%F' '%H:%M:%S) #####"
  grep -aoE '=== Final Summary: .*' "$LOG/${tag}.log" | tail -1
}

run ceiling         --arm ceiling
run floor           --arm floor
run oneshot         --arm corrected --keep-ratio 0.55 --groups 1
run inter_preglobal --arm corrected --keep-ratio 0.55 --groups 4 --bounds pre_global
run inter_aligned   --arm corrected --keep-ratio 0.55 --groups 4 --bounds aligned
echo "SAM3_DET_FULL_COCO_DONE"
