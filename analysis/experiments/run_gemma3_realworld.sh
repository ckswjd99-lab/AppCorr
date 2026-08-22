#!/usr/bin/env bash
# Four-arm AppCorr sweep over real-world-image benchmarks on the Gemma 3 unified axis.
# Datasets run shortest-first so a broken loader or a degenerate arm shows up in the first hour
# rather than the twelfth. Arms are sequential and independently logged: a dying arm does not take
# the sweep with it, and an arm that already has results is skipped on re-run.
cd /NHNHOME/share/cjpark/AppCorr-gemma3
PY=/home/nxclab/anaconda3/envs/appcorr/bin/python
# Never hardcode the token: a committed credential is what GitHub push protection blocks, and it
# has to be scrubbed from every commit that carried it. Take it from the environment and fail loudly
# if it is missing, rather than running and failing later with an opaque 401 from the hub.
: "${HF_TOKEN:?set HF_TOKEN in the environment before running this sweep}"
export HF_TOKEN
export CUDA_VISIBLE_DEVICES=0

for DS in realworldqa textvqa pope; do
  OUT=analysis/results/gemma3_$DS
  mkdir -p $OUT
  run () {
    local name=$1; shift
    [ -s "$OUT/$name.json" ] && { echo "[skip] $DS/$name"; return; }
    echo "[start] $DS/$name  $(date +%H:%M:%S)"
    $PY analysis/experiments/gemma3_oracle.py --dataset $DS --keep 0.55 --full \
        --out-json "$OUT/$name.json" "$@" > "$OUT/$name.log" 2>&1
    echo "[done ] $DS/$name  $(date +%H:%M:%S)  $(grep -aoE '"accuracy": [0-9.]+' "$OUT/$name.log" | head -1)"
  }
  run ceiling        --arm ceiling
  run floor          --arm floor
  run corrected      --arm corrected
  run interleaved_g4 --arm interleaved --keep 0.55 --groups 4
  echo "[dataset done] $DS  $(date +%H:%M:%S)"
done
echo "REALWORLD SWEEP COMPLETE $(date)"
