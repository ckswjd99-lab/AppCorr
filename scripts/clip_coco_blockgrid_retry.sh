#!/usr/bin/env bash
# COCO block_grid, re-run on the fixed worker.
#
# The two earlier attempts both died the same way, and it was not the flake I first called it: the
# decoder thread never assigned `self.config` (fixed in 8aa122f), so whether it survived depended on
# the GPU worker draining its queue first. This run is therefore also the test of that fix -- if the
# AttributeError/KeyError pair is gone, the fix holds.
#
# Reference, same tree, grid grouping: i2t 65.46 / t2i 49.22 (floor 50.06/40.33, ceiling 67.96/50.70)
# ImageNet block_grid already came in at 77.48/95.01 against grid's 77.14/94.88.
set -u
REPO=/NHNHOME/share/cjpark/AppCorr-clip-closedloop
PY=/home/nxclab/anaconda3/envs/appcorr/bin/python
LOG=$REPO/logs/clip_blockgrid
cd "$REPO" || exit 1
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 timeout 10800 \
  "$PY" analysis/experiments/clip_coco_retrieval_offload_eval.py \
    --config offload/config/coco_retrieval_clip_bigg_interleaved_g4.json \
    --full --device cuda:0 --grouping-strategy block_grid \
    > "$LOG/coco_blockgrid_retry.log" 2>&1
echo "rc=$?"
grep -aE "i2t \(image->text\)|t2i \(text->image\)|patch keep_rate" "$LOG/coco_blockgrid_retry.log" | tail -3
echo "config-race-recurred=$(grep -ac "has no attribute 'transmission_policy_name'" "$LOG/coco_blockgrid_retry.log")"
echo "COCO_BG_DONE"
