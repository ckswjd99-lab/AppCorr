#!/usr/bin/env bash
# Pick the detector readout protocol on the CEILING arm, then freeze it for every other arm.
#
# The old `--det-score-thresh 0.3` was picked by hand and emitted 2.2x more predictions than there
# are GT boxes. AP is a ranking metric -- COCOeval sorts by score and sweeps the threshold itself,
# keeping the top maxDets per image -- so a hand-set confidence floor can only remove recall the
# metric would have used. This sweeps top-k-per-prompt instead.
#
# Tuning the readout on the ceiling is legitimate and is the point: it gives the exact forward its
# best possible score, so every approximation is then read against a reference that is not
# handicapped by a bad protocol. The chosen value is then held fixed across floor/corrected.
set -u
REPO=/NHNHOME/share/cjpark/AppCorr-sam3
PY=/home/nxclab/anaconda3/envs/appcorr/bin/python
LOG=$REPO/logs/sam3_det_protocol
N="${N:-200}"
mkdir -p "$LOG"; cd "$REPO" || exit 1

for k in 1 5 10 30 100; do
  echo "##### START per_cat=$k @ $(date +%H:%M:%S) #####"
  for p in $(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader); do kill -9 "$p" 2>/dev/null; done
  sleep 3
  HF_TOKEN="${HF_TOKEN:?}" CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 timeout 7200 \
    "$PY" analysis/experiments/sam3_coco_oracle.py --path detector --arm ceiling \
      --num-images "$N" --max-boxes 20 --det-per-cat "$k" --det-max-dets 100 \
      > "$LOG/percat_${k}.log" 2>&1
  echo "##### END per_cat=$k rc=$? @ $(date +%H:%M:%S) #####"
  grep -aoE '=== Final Summary: .*' "$LOG/percat_${k}.log" | tail -1
done
# The old setting, for reference.
echo "##### START legacy_thresh0.3 @ $(date +%H:%M:%S) #####"
for p in $(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader); do kill -9 "$p" 2>/dev/null; done
sleep 3
HF_TOKEN="${HF_TOKEN:?}" CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 timeout 7200 \
  "$PY" analysis/experiments/sam3_coco_oracle.py --path detector --arm ceiling \
    --num-images "$N" --max-boxes 20 --det-per-cat 200 --det-max-dets 100000 \
    --det-score-thresh 0.3 > "$LOG/legacy.log" 2>&1
echo "##### END legacy rc=$? @ $(date +%H:%M:%S) #####"
grep -aoE '=== Final Summary: .*' "$LOG/legacy.log" | tail -1
echo "SAM3_DET_PROTOCOL_DONE"
