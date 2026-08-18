#!/usr/bin/env bash
# Pass 2 of the CLIP closed-loop re-measurement.
#
#  (a) Re-runs the two arms that pass 1 lost to an external SIGKILL (rc=137, no Final Summary).
#  (b) Adds a PRE-REBASE CONTROL: the same interleaved arms run from a worktree pinned at
#      a02f094 (the original experiment/clip-appcorr tip) on TODAY's environment. Pass 1's COCO
#      interleaved arm moved -2.5pp against the recorded number while the transmission round trip
#      is provably bit-identical, so the control separates "the rebase moved it" from "the
#      environment moved it since July". A recorded value is only valid for the commit AND the
#      stack it was taken on.
set -u

NEW=/NHNHOME/share/cjpark/AppCorr-clip-closedloop
OLD=/NHNHOME/share/cjpark/AppCorr-clip-prerebase
PY=/home/nxclab/anaconda3/envs/appcorr/bin/python
LOGDIR="$NEW/logs/clip_closedloop"
mkdir -p "$LOGDIR"

run() {  # tag  repo  driver  config  scale-flag
  local tag=$1 repo=$2 driver=$3 cfg=$4 scale=$5
  local log="$LOGDIR/${tag}.log"
  echo "##### START $tag @ $(date +%F_%H:%M:%S) #####"
  ( cd "$repo" && CUDA_VISIBLE_DEVICES=1 PYTHONUNBUFFERED=1 \
      "$PY" "analysis/experiments/${driver}" \
        --config "offload/config/${cfg}" $scale --device cuda:0 \
        > "$log" 2>&1 )
  local rc=$?
  echo "##### END   $tag rc=$rc @ $(date +%F_%H:%M:%S) #####"
  if grep -qa "=== Summary" "$log"; then
    echo "COMPLETED $tag"
    grep -aA4 "=== Summary" "$log" | head -5
  else
    echo "INCOMPLETE $tag (no Final Summary -- rc=$rc)"
  fi
  echo "ARM_DONE2 $tag"
}

# --- (a) re-runs of the SIGKILLed arms, rebased code ---
run coco_ceiling_sequential_rerun "$NEW" clip_coco_retrieval_offload_eval.py coco_retrieval_clip_bigg_sequential.json      --full
run imnet_interleaved_g4_rerun    "$NEW" clip_zeroshot_offload_eval.py       imagenet_clip_bigg_interleaved_g4.json         --full

# --- (b) pre-rebase controls, same environment, same GPU ---
run coco_interleaved_g4_CONTROL   "$OLD" clip_coco_retrieval_offload_eval.py coco_retrieval_clip_bigg_interleaved_g4.json   --full
run imnet_interleaved_g4_CONTROL  "$OLD" clip_zeroshot_offload_eval.py       imagenet_clip_bigg_interleaved_g4.json         --full

echo "CLIP_CLOSEDLOOP_PASS2_DONE @ $(date +%F_%H:%M:%S)"
