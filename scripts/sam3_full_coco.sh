#!/usr/bin/env bash
# SAM 3 tracker path, full COCO val2017 -- the paper numbers.
#
# Five arms so the table carries its own references: floor is the approximate pass alone (what you
# get by sending only the L2 pyramid level), ceiling is the exact forward. Every corrected arm is
# read as a fraction of that gap; without both rows an absolute AP means nothing.
#
# All corrected arms recompute the SAME 55% of tokens -- verified, the union over interleaved rounds
# equals the one-shot set exactly. The arms differ only in WHEN correction happens, so any gap
# between them is the interleaving, not extra compute.
#
# 50-image probe (direction only, not separable at that size):
#   floor 0.5423 | oneshot 0.6170 | aligned 0.6256 | pre_global 0.6265 | ceiling 0.6230
# Feature-space rel-L2 vs exact, 20 images (monotone in fidelity, unlike AP):
#   floor 0.6417 | oneshot 0.4664 | aligned 0.4278 | pre_global 0.4278
set -u
REPO=/NHNHOME/share/cjpark/AppCorr-sam3
PY=/home/nxclab/anaconda3/envs/appcorr/bin/python
LOG=$REPO/logs/sam3_full_coco
mkdir -p "$LOG"; cd "$REPO" || exit 1

run() {  # tag  extra-args...
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

run ceiling         --arm ceiling
run floor           --arm floor
run oneshot         --arm corrected --keep-ratio 0.55 --groups 1
run inter_aligned   --arm corrected --keep-ratio 0.55 --groups 4 --bounds aligned
run inter_preglobal --arm corrected --keep-ratio 0.55 --groups 4 --bounds pre_global
echo "SAM3_FULL_COCO_DONE"
