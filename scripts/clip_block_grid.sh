#!/usr/bin/env bash
# CLIP with block_grid grouping instead of grid, on both tasks.
#
# `grid` tiles a 2x2 pattern across the 16x16 patch grid, so each group is a checkerboard spread over
# the whole image. `block_grid` gives each group one contiguous quadrant. Everything else is held
# fixed -- same configs, same [2,0] levels, same 4 groups, same full datasets -- so the only variable
# is the grouping.
#
# Reference, same trees, grid grouping:
#   COCO     interleaved g4  i2t 65.46 / t2i 49.22   (floor 50.06/40.33, ceiling 67.96/50.70)
#   ImageNet interleaved g4  77.14 / 94.88           (floor 65.92/88.20, ceiling 79.85/96.00)
#
# The startup race that killed two earlier arms ('NoneType' has no attribute
# transmission_policy_name -> KeyError vision_layer0_kv) is flaky and tree-independent, so each arm
# retries once before being called a failure.
set -u
REPO=/NHNHOME/share/cjpark/AppCorr-clip-closedloop
PY=/home/nxclab/anaconda3/envs/appcorr/bin/python
LOG=$REPO/logs/clip_blockgrid
mkdir -p "$LOG"; cd "$REPO" || exit 1

run() {  # tag driver config
  local tag=$1 driver=$2 cfg=$3
  for attempt in 1 2; do
    echo "##### START $tag attempt=$attempt @ $(date +%H:%M:%S) #####"
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 timeout 10800 \
      "$PY" "analysis/experiments/$driver" \
        --config "offload/config/$cfg" --full --device cuda:0 \
        --grouping-strategy block_grid \
        > "$LOG/${tag}.log" 2>&1
    echo "##### END $tag rc=$? @ $(date +%H:%M:%S) #####"
    if grep -qaE 'i2t \(image->text\)|top1_acc' "$LOG/${tag}.log"; then
      grep -aE "i2t \(image->text\)|t2i \(text->image\)|top1_acc|patch keep_rate" "$LOG/${tag}.log" | tail -3
      echo "OK $tag"; return 0
    fi
    echo "no summary on attempt $attempt: $(grep -aoE '[A-Za-z]+Error.{0,60}' "$LOG/${tag}.log" | head -1)"
    sleep 15
  done
  echo "FAILED $tag after 2 attempts"
}

run coco_blockgrid  clip_coco_retrieval_offload_eval.py coco_retrieval_clip_bigg_interleaved_g4.json
run imnet_blockgrid clip_zeroshot_offload_eval.py       imagenet_clip_bigg_interleaved_g4.json
echo "BLOCKGRID_DONE"
