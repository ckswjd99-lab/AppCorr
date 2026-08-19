#!/usr/bin/env bash
# Reproduces the grid / block_grid / expansion comparison on both CLIP tasks.
#
# Replaces ten one-off scripts written while investigating; their results are in
# analysis/experiments/CLIP_APPCORR_REPORT.md §7.6 and the findings below.
#
# All arms: same tree, same configs, [2,0] levels, 4 groups, FULL datasets, keep_rate 100% (no token
# pruning). Grouping is the only variable -- every token is corrected either way, so what changes is
# the ORDER in which patches arrive, and interleaved correction is order-dependent because each round
# corrects against the state the previous rounds left.
#
# Measured 2026-08-19 (floor = approx-only, ceiling = sequential):
#
#   ImageNet top1   floor 65.92 | grid 77.14 | block_grid 77.48 | expansion (pending) | ceiling 79.85
#   COCO i2t R@1    floor 50.06 | grid 65.46 | block_grid 65.90 | expansion 64.96     | ceiling 67.96
#   COCO t2i R@1    floor 40.33 | grid 49.22 | block_grid 49.06 | expansion 49.00     | ceiling 50.70
#
# block_grid wins on ImageNet top1 (+0.34) and COCO i2t (+0.44); COCO t2i is a wash across all three
# (49.00-49.22). expansion is clearly worst on COCO i2t (-0.94 vs block_grid), which is ~12x the
# ~0.08pp run-to-run spread of this eval -- centre-first ordering does not suit a retrieval task
# where the caption can describe anything in the frame.
#
# Usage: bash scripts/clip_grouping_comparison.sh [strategy ...]   (default: all three)
set -u
REPO=/NHNHOME/share/cjpark/AppCorr-clip-closedloop
PY=/home/nxclab/anaconda3/envs/appcorr/bin/python
LOG=$REPO/logs/clip_grouping
mkdir -p "$LOG"; cd "$REPO" || exit 1

for strat in "${@:-grid block_grid expansion}"; do
  for task in coco imnet; do
    if [ "$task" = coco ]; then
      driver=clip_coco_retrieval_offload_eval.py; cfg=coco_retrieval_clip_bigg_interleaved_g4.json
    else
      driver=clip_zeroshot_offload_eval.py;       cfg=imagenet_clip_bigg_interleaved_g4.json
    fi
    tag="${task}_${strat}"
    echo "##### START $tag @ $(date +%H:%M:%S) #####"
    # GPU 0 only -- GPU 1 may hold another user's job. Never sweep compute-apps without -i.
    for p in $(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader); do kill -9 "$p" 2>/dev/null; done
    sleep 5
    CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 timeout 14400 \
      "$PY" "analysis/experiments/$driver" --config "offload/config/$cfg" \
        --full --device cuda:0 --grouping-strategy "$strat" \
      > "$LOG/${tag}.log" 2>&1
    echo "##### END $tag rc=$? @ $(date +%H:%M:%S) #####"
    grep -aE "i2t \(image->text\)|t2i \(text->image\)|top1_acc|patch keep_rate" "$LOG/${tag}.log" | tail -3
    # rc alone is not enough: a run that aborts at startup can still exit 0 with an empty result.
    echo "summary_present=$(grep -acE 'i2t \(image->text\)|top1_acc' "$LOG/${tag}.log")"
  done
done
echo "GROUPING_COMPARISON_DONE"
