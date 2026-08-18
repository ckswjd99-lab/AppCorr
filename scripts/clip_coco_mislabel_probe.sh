#!/usr/bin/env bash
# Pass 3 -- settle where the old COCO "baseline (thr=0) = 67.96 / 50.70" row came from.
#
# Measured facts so far: the full-scale sequential CEILING is 67.96 / 50.70, matching that row to
# every digit on two independent metrics, while the rebased interleaved arm measures 65.46 / 49.22.
# Two numbers agreeing to 2dp is not chance, so either the old row was produced by the sequential
# config (a mislabel), or the interleaved arm has regressed since.
#
# Three arms separate those:
#   coco_inter_CONTROL_noflag  pre-rebase code (a02f094), interleaved config, no flag
#                              -> if this is 65.46, the old row is not reproducible from the
#                                 interleaved config at all and the mislabel is proven.
#                              -> if it is 67.96, the rebase regressed COCO.
#   coco_inter_thr0_OLD        pre-rebase code, --token-keep-thres 0: the old row's literal command.
#   coco_inter_thr0_NEW        rebased code, --token-keep-thres 0: same command on today's code.
#
# thr=0 also flips mobile_pscore to residual_energy, so it is not a priori identical to the no-flag
# run even though pruning is a no-op at that threshold. Measured, not argued.
#
# Accuracy-only arms, so these may share the GPU with another accuracy run; no latency is claimed
# from this script.
set -u

NEW=/NHNHOME/share/cjpark/AppCorr-clip-closedloop
OLD=/NHNHOME/share/cjpark/AppCorr-clip-prerebase
PY=/home/nxclab/anaconda3/envs/appcorr/bin/python
LOGDIR="$NEW/logs/clip_closedloop"
CFG=offload/config/coco_retrieval_clip_bigg_interleaved_g4.json
mkdir -p "$LOGDIR"

run() {  # tag  repo  extra-args...
  local tag=$1 repo=$2; shift 2
  local log="$LOGDIR/${tag}.log"
  echo "##### START $tag @ $(date +%F_%H:%M:%S) #####"
  ( cd "$repo" && CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
      "$PY" analysis/experiments/clip_coco_retrieval_offload_eval.py \
        --config "$CFG" --full --device cuda:0 "$@" \
        > "$log" 2>&1 )
  local rc=$?
  echo "##### END   $tag rc=$rc @ $(date +%F_%H:%M:%S) #####"
  if grep -qa "=== Summary" "$log"; then
    echo "COMPLETED $tag"
    grep -aA3 "=== Summary" "$log" | head -4
  else
    echo "INCOMPLETE $tag (no Final Summary -- rc=$rc)"
  fi
}

run coco_inter_CONTROL_noflag "$OLD"
run coco_inter_thr0_OLD       "$OLD" --token-keep-thres 0
run coco_inter_thr0_NEW       "$NEW" --token-keep-thres 0

echo "CLIP_COCO_MISLABEL_PROBE_DONE @ $(date +%F_%H:%M:%S)"
