#!/usr/bin/env bash
# Full ChartQA (2500) sweep on the Gemma 3 unified axis. Arms run SEQUENTIALLY and each writes its
# own log, so one arm dying does not take the sweep with it -- an arm that fails leaves a log ending
# without an "accuracy" line, which is how the monitor tells failure from still-running.
cd /NHNHOME/share/cjpark/AppCorr-gemma3
OUT=analysis/results/gemma3_chartqa
PY=/home/nxclab/anaconda3/envs/appcorr/bin/python
export HF_TOKEN='REDACTED_SEE_HEAD'
export CUDA_VISIBLE_DEVICES=0

run () {  # name, extra args
  local name=$1; shift
  [ -s "$OUT/$name.json" ] && { echo "[skip] $name already has results"; return; }
  echo "[start] $name  $(date +%H:%M:%S)"
  $PY analysis/experiments/gemma3_oracle.py --dataset chartqa --keep 0.55 \
      --full --out-json "$OUT/$name.json" "$@" > "$OUT/$name.log" 2>&1
  echo "[done ] $name  $(date +%H:%M:%S)  $(grep -aoE '"accuracy": [0-9.]+' "$OUT/$name.log" | head -1)"
}

run ceiling       --arm ceiling
run floor         --arm floor
run corrected     --arm corrected
run interleaved_g4 --arm interleaved --keep 0.55 --groups 4
run corrected_split --arm corrected_split
run corrected_patchled --arm corrected_patchled
run interleaved_g2 --arm interleaved --keep 0.55 --groups 2
run interleaved_g8 --arm interleaved --keep 0.55 --groups 8
echo "SWEEP COMPLETE $(date)"
