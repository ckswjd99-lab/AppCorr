#!/usr/bin/env bash
# SAM 3's other five table rows: COCO detector, LVIS detector, and three SA-Co gold subsets.
#
# Only the COCO *tracker* row was ever run. The other five share one vision encoder -- which is why
# their Crit. Comp. column repeats the same FLOPs by construction, not because it was measured six
# times -- but they are different TASKS, so accuracy has to be measured per task.
#
# No new driver: `sam3_coco_oracle.py` already takes `--dataset coco|lvis|saco_gold:<subset>` and
# `--path tracker|detector`, and every dataset is present locally (COCO and LVIS annotations,
# saco_gold under /NHNHOME/share/cjpark/data).
#
# `--bounds pre_global` throughout, matching the tracker row: SAM 3 puts global attention at layers
# 7/15/23/31 and the other 28 are windowed, so stopping each round one layer BEFORE a global layer
# defers it instead of re-correcting it every round. Measured at 0.60x one-shot's correction cost
# against `aligned`'s 0.63x, for the same mask AP -- see docs/memo/sam3_coco_interleaved_results.md.
set -u
cd /NHNHOME/share/cjpark/AppCorr-flops
PY=/home/nxclab/anaconda3/envs/appcorr/bin/python
export CUDA_VISIBLE_DEVICES=${GPU:-1}
export HF_HUB_OFFLINE=1   # facebook/sam3 is gated; a local cache already exists
OUT=analysis/results/vfm_accuracy
mkdir -p "$OUT"

arm () {   # arm <tag> <extra args...>
  local tag=$1; shift
  # Success is a `Final Summary` line, never the exit code: this repo's runners exit 0 on a missing
  # dataset loader, an invalid device ordinal and a busy port alike.
  if [ -s "$OUT/$tag.log" ] && grep -qa "Final Summary" "$OUT/$tag.log"; then
    echo "[skip ] $tag"; return
  fi
  echo "[start] $tag  $(date +%F' '%H:%M:%S)"
  $PY analysis/experiments/sam3_coco_oracle.py --full --max-boxes 20 \
      --arm corrected --groups 4 --bounds pre_global \
      --out-json "$OUT/$tag.json" "$@" > "$OUT/$tag.log" 2>&1
  echo "[done ] $tag rc=$?  $(date +%F' '%H:%M:%S)"
  grep -aoE "Final Summary: .*" "$OUT/$tag.log" | tail -1 | cut -c1-220
}

# Cheapest first, so a completed row exists early if this is stopped: COCO detector reuses data
# already proven to load, LVIS is 19,626 images, and the SA-Co subsets are scored by a
# three-annotator oracle merge that is slower per image than plain COCOeval.
for K in 0.25 0.50; do
  arm "sam3_cocodet_k${K}"  --dataset coco --path detector --keep-ratio "$K"
done
for K in 0.25 0.50; do
  arm "sam3_lvis_k${K}"     --dataset lvis --path detector --keep-ratio "$K"
done
for SUB in crowded sa1b attributes; do
  for K in 0.25 0.50; do
    arm "sam3_saco_${SUB}_k${K}" --dataset "saco_gold:${SUB}" --path detector --keep-ratio "$K"
  done
done
echo "SAM3_TASKS_COMPLETE $(date)"
