#!/usr/bin/env bash
# SAM 3 tracker, full COCO -- the two interleaved arms, rerun after the correction contract fixes.
#
# The floor / one-shot / ceiling rows from scripts/sam3_full_coco.sh stand unchanged (they do not
# touch the interleaved path): floor 0.5374 | one-shot 0.5961 | ceiling 0.6010.
#
# All three corrected arms recompute the same 55% of tokens. They differ in WHEN, and therefore in
# cost: correcting group r over depth bounds[r] costs (1/g)*sum(bounds) -- 0.63x one-shot for
# aligned, 0.60x for pre_global, against 1.00x for one-shot.
set -u
REPO=/NHNHOME/share/cjpark/AppCorr-sam3
PY=/home/nxclab/anaconda3/envs/appcorr/bin/python
LOG=$REPO/logs/sam3_full_coco
mkdir -p "$LOG"; cd "$REPO" || exit 1

run() {
  local tag=$1; shift
  echo "##### START $tag @ $(date +%F' '%H:%M:%S) #####"
  for p in $(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader); do kill -9 "$p" 2>/dev/null; done
  sleep 4
  HF_TOKEN="${HF_TOKEN:?}" CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 timeout 43200 \
    "$PY" analysis/experiments/sam3_coco_oracle.py --path tracker \
      --full --max-boxes 20 "$@" > "$LOG/${tag}.log" 2>&1
  echo "##### END $tag rc=$? @ $(date +%F' '%H:%M:%S) #####"
  grep -aoE '=== Final Summary: .*' "$LOG/${tag}.log" | tail -1
}

run inter_aligned   --arm corrected --keep-ratio 0.55 --groups 4 --bounds aligned
run inter_preglobal --arm corrected --keep-ratio 0.55 --groups 4 --bounds pre_global
echo "SAM3_FULL_COCO_INTER_DONE"
