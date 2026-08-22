#!/usr/bin/env bash
# One-shot correction smoke test for OV2: does the `corrected` arm land between floor and ceiling,
# and is the plumbing sane end to end?
#
# All arms run on the SAME subset (the oracle's stride sampler is deterministic given --num-samples),
# because the full-dataset floor/ceiling were measured over all 2500 and a 100-sample subset of them
# is a different number. Comparing a subset arm against a full-dataset floor is the mistake this
# script exists to avoid.
#
# This is a SANITY CHECK, not a result. Nothing here decides a keep ratio or a ranking -- a
# 100-sample accuracy cannot separate arms that differ by a few points.
cd /NHNHOME/share/cjpark/AppCorr-ov2
PY=/home/nxclab/anaconda3/envs/appcorr/bin/python
export CUDA_VISIBLE_DEVICES=${GPU:-0}
N=${N:-100}
DS=${DS:-chartqa}
KEEP=${KEEP:-0.55}
OUT=analysis/results/ov2_smoke_$DS; mkdir -p $OUT

run () {   # run <tag> <extra args...>
  local tag=$1; shift
  echo "[start] $DS/$tag  $(date +%H:%M:%S)"
  $PY analysis/experiments/ov2_oracle.py --dataset $DS --num-samples $N --level 2 \
      --out-json "$OUT/$tag.json" "$@" > "$OUT/$tag.log" 2>&1
  echo "[done ] $DS/$tag  $(date +%H:%M:%S)  $(grep -aoE '"accuracy": [0-9.]+' "$OUT/$tag.log" | head -1)"
}

run ceiling   --arm ceiling
run floor     --arm floor
run corrected --arm corrected --keep $KEEP

echo "OV2 SMOKE COMPLETE $(date)"
