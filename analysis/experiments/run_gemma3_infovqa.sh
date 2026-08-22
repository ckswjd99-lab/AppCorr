#!/usr/bin/env bash
# InfoVQA on the Gemma 3 unified axis: the third dataset with a gap wide enough to interpret.
# Qwen2.5-VL measured an 18.02pp floor-ceiling gap here, and ANLS is a CONTINUOUS metric -- which
# makes this the second continuous-scored point, the thing needed to tell whether corrected_t's
# habit of landing above the ceiling tracks binary scoring or something else.
cd /NHNHOME/share/cjpark/AppCorr-gemma3
OUT=analysis/results/gemma3_infovqa
mkdir -p $OUT
PY=/home/nxclab/anaconda3/envs/appcorr/bin/python
export HF_TOKEN='REDACTED_SEE_HEAD'
export CUDA_VISIBLE_DEVICES=0

run () {
  local name=$1; shift
  [ -s "$OUT/$name.json" ] && { echo "[skip] infovqa/$name"; return; }
  echo "[start] infovqa/$name  $(date +%H:%M:%S)"
  $PY analysis/experiments/gemma3_oracle.py --dataset infovqa --keep 0.55 --full \
      --out-json "$OUT/$name.json" "$@" > "$OUT/$name.log" 2>&1
  echo "[done ] infovqa/$name  $(date +%H:%M:%S)  $(grep -aoE '"accuracy": [0-9.]+' "$OUT/$name.log" | head -1)"
}
run ceiling        --arm ceiling
run floor          --arm floor
run corrected_t    --arm corrected_t
run interleaved_g4 --arm interleaved --keep 0.55 --groups 4
echo "INFOVQA SWEEP COMPLETE $(date)"
