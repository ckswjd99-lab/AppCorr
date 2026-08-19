#!/usr/bin/env bash
# Re-measure the COCO pruned arms so the corrected table is measurement, not re-derivation.
#
# Until now the per-threshold rows were July values re-scored against a corrected reference. That
# reuse was justified only by the unpruned arm reproducing to ~0.1pp -- one arm's worth of evidence.
# These two thresholds bracket the knee, so they are the ones that decide whether the re-derived
# curve is real.
#
# All three arms on the SAME tree (rebased), so the comparison has no cross-tree confound:
#   unpruned (measured 2026-08-18) : i2t 65.46 / t2i 49.22
#   July thr=25                    : i2t 64.76 / t2i 49.00
#   July thr=100                   : i2t 63.88 / t2i 48.29
# Run-to-run spread on this eval is ~0.08pp i2t, so only differences above ~0.2pp mean anything.
set -u
REPO=/NHNHOME/share/cjpark/AppCorr-clip-closedloop
PY=/home/nxclab/anaconda3/envs/appcorr/bin/python
LOG=$REPO/logs/clip_closedloop
cd "$REPO" || exit 1

for thr in 25 100; do
  echo "##### START thr=$thr @ $(date +%H:%M:%S) #####"
  # GPU 0 only: GPU 1 belongs to another user's job (Qwen rollouts, ~166 GB). Never sweep it.
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 timeout 7200 \
    "$PY" analysis/experiments/clip_coco_retrieval_offload_eval.py \
      --config offload/config/coco_retrieval_clip_bigg_interleaved_g4.json \
      --full --device cuda:0 --token-keep-thres "$thr" \
      > "$LOG/coco_thr${thr}_remeasure.log" 2>&1
  rc=$?
  echo "##### END thr=$thr rc=$rc @ $(date +%H:%M:%S) #####"
  # Report rc AND the summary marker: rc=0 with no summary means the run aborted before starting.
  grep -aE "i2t \(image->text\)|t2i \(text->image\)|patch keep_rate" "$LOG/coco_thr${thr}_remeasure.log" | tail -3
  echo "summary_present=$(grep -ac 'i2t (image->text)' "$LOG/coco_thr${thr}_remeasure.log")"
  sleep 10
done
echo "PRUNED_REMEASURE_DONE"
