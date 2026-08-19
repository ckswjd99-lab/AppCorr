#!/usr/bin/env bash
# SAM 3 floor / one-shot / interleaved(aligned) / interleaved(pre_global) / ceiling.
#
# Small set first: this is a wiring and direction check, not the paper number. 50 images cannot
# separate arms that land within a point of each other.
#
# The two interleaved arms differ only in where each round stops. SAM 3's global-attention layers
# sit at 7/15/23/31, so `aligned` (8/16/24/32) ends each round just AFTER a global layer and
# re-corrects it every subsequent round; `pre_global` (7/15/23/32) ends just BEFORE one, deferring
# the expensive full-5184-token attention to the next round. Same total work by the last round --
# the question is whether moving it costs accuracy.
#
# 500-image reference, one-shot, same slice family: floor 0.5731 | corrected 0.6278 | ceiling 0.6331
set -u
REPO=/NHNHOME/share/cjpark/AppCorr-sam3
PY=/home/nxclab/anaconda3/envs/appcorr/bin/python
LOG=$REPO/logs/sam3_interleaved
N="${N:-50}"
mkdir -p "$LOG"; cd "$REPO" || exit 1

run() {  # tag  extra-args...
  local tag=$1; shift
  echo "##### START $tag @ $(date +%H:%M:%S) #####"
  for p in $(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader); do kill -9 "$p" 2>/dev/null; done
  sleep 4
  HF_TOKEN="${HF_TOKEN:?}" CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 timeout 10800 \
    "$PY" analysis/experiments/sam3_coco_oracle.py --path tracker \
      --num-images "$N" --max-boxes 20 "$@" > "$LOG/${tag}.log" 2>&1
  echo "##### END $tag rc=$? @ $(date +%H:%M:%S) #####"
  grep -aoE '=== Final Summary: .*' "$LOG/${tag}.log" | tail -1
  echo "summary_present=$(grep -ac 'Final Summary' "$LOG/${tag}.log")"
}

run ceiling            --arm ceiling
run floor              --arm floor
run oneshot            --arm corrected --keep-ratio 0.55 --groups 1
run inter_aligned      --arm corrected --keep-ratio 0.55 --groups 4 --bounds aligned
run inter_preglobal    --arm corrected --keep-ratio 0.55 --groups 4 --bounds pre_global
echo "SAM3_INTERLEAVED_DONE"
