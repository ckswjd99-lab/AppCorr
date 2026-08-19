#!/usr/bin/env bash
# SAM 3 floor / corrected / ceiling on COCO val2017, 500 images, tracker path.
#
# L2 is built from the ORIGINAL image in native coordinates and only then scaled to the model
# canvas, per docs/memo/pyramid_degradation_native_vs_canvas.md. Degrading the 1008x1008 canvas
# instead -- which an earlier version of this driver did -- under-degrades by the canvas upscale
# factor and compresses the floor-ceiling gap. At 20 images it made the gap 0.0216 instead of
# 0.1039, and produced a 73.1% recovery figure that is withdrawn.
#
# 20-image reference for these three arms (native-relative, NOT comparable to 500):
#   floor 0.6001 | corrected(55%) 0.6902 | ceiling 0.7041 | recovery 86.7%
#
# Selection is residual energy alone. The standing default is residual energy x average attention;
# the SAM 3 fork does not cache attention yet, so this run is the control that the attention-weighted
# score will be read against.
set -u
REPO=/NHNHOME/share/cjpark/AppCorr-sam3
PY=/home/nxclab/anaconda3/envs/appcorr/bin/python
LOG=$REPO/logs/sam3_arms
mkdir -p "$LOG"; cd "$REPO" || exit 1

for arm in ceiling floor corrected; do
  echo "##### START $arm @ $(date +%H:%M:%S) #####"
  for p in $(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader); do kill -9 "$p" 2>/dev/null; done
  sleep 5
  HF_TOKEN="${HF_TOKEN:?}" CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 timeout 21600 \
    "$PY" analysis/experiments/sam3_coco_oracle.py \
      --path tracker --arm "$arm" --num-images 500 --max-boxes 20 \
      --out-json "$LOG/${arm}_500.json" \
    > "$LOG/${arm}_500.log" 2>&1
  echo "##### END $arm rc=$? @ $(date +%H:%M:%S) #####"
  grep -aoE '=== Final Summary: .*' "$LOG/${arm}_500.log" | tail -1
  echo "summary_present=$(grep -ac 'Final Summary' "$LOG/${arm}_500.log")"
  sleep 10
done
echo "SAM3_ARMS_DONE"
