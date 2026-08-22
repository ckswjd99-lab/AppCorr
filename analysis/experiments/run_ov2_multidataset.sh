#!/usr/bin/env bash
# LLaVA-OneVision-2 unified axis across datasets: floor / corrected / interleaved / ceiling.
#
# The dataset set matches docs/memo/gemma3_multidataset_results.md so the two model families land on
# one axis. Three of those five carry an interpretable preservation number (ChartQA, TextVQA,
# InfoVQA); POPE and RealWorldQA are kept because a NARROW gap is itself the finding -- Gemma 3's
# 1.20pp and 0.78pp gaps make preservation arithmetic rather than evidence, and OV2 has to be shown
# to be in the same position rather than assumed into it.
#
# Every arm on a dataset must run over the SAME examples, which `--full` guarantees.
# floor/ceiling for chartqa and textvqa already exist under analysis/results/ov2_<ds>/ and are
# reused rather than recomputed -- the driver's stock path is unchanged, and the smoke run
# reproduced the ceiling to 0.3pp on a 100-sample subset.
set -u
cd /NHNHOME/share/cjpark/AppCorr-ov2
PY=/home/nxclab/anaconda3/envs/appcorr/bin/python
export CUDA_VISIBLE_DEVICES=${GPU:-0}
KEEP=${KEEP:-0.55}
GROUPS=${GROUPS:-4}
DATASETS=${DATASETS:-"chartqa textvqa infovqa realworldqa pope"}

run () {   # run <dataset> <tag> <extra args...>
  local ds=$1 tag=$2; shift 2
  local out=analysis/results/ov2_$ds; mkdir -p "$out"
  if [ -s "$out/$tag.json" ]; then echo "[skip ] $ds/$tag"; return; fi
  echo "[start] $ds/$tag  $(date +%H:%M:%S)"
  $PY analysis/experiments/ov2_oracle.py --dataset "$ds" --full --level 2 \
      --out-json "$out/$tag.json" "$@" > "$out/$tag.log" 2>&1
  echo "[done ] $ds/$tag  $(date +%H:%M:%S)  $(grep -aoE '"accuracy": [0-9.]+' "$out/$tag.log" | head -1)"
}

for DS in $DATASETS; do
  # Order matters for triage, not for correctness: the two cheap stock arms first, so a dataset
  # whose gap turns out to be narrow is known to be uninterpretable before hours go into its
  # correction arms.
  run "$DS" ceiling  --arm ceiling
  run "$DS" floor    --arm floor
  run "$DS" "corrected_k${KEEP}"        --arm corrected   --keep "$KEEP"
  run "$DS" "interleaved_g${GROUPS}_k${KEEP}" --arm interleaved --keep "$KEEP" --groups "$GROUPS"
done
echo "OV2 MULTIDATASET SWEEP COMPLETE $(date)"
