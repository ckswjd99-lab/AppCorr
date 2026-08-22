#!/usr/bin/env bash
# Step 3 for LLaVA-OneVision-2: is the floor-ceiling gap wide enough for this model to be worth a
# unified axis at all? Only ceiling and floor -- there is no fork yet, so both arms are stock
# generate and nothing here can be confounded by correction wiring.
# ChartQA and TextVQA are the two datasets where BOTH Qwen2.5-VL and Gemma 3 showed a wide gap,
# so they are the comparable axis across all three model families.
cd /NHNHOME/share/cjpark/AppCorr-ov2
PY=/home/nxclab/anaconda3/envs/appcorr/bin/python
export HF_TOKEN='REDACTED_SEE_HEAD'
export CUDA_VISIBLE_DEVICES=0

for DS in chartqa textvqa; do
  OUT=analysis/results/ov2_$DS; mkdir -p $OUT
  for ARM in ceiling floor; do
    [ -s "$OUT/$ARM.json" ] && { echo "[skip] $DS/$ARM"; continue; }
    echo "[start] ov2 $DS/$ARM  $(date +%H:%M:%S)"
    $PY analysis/experiments/ov2_oracle.py --dataset $DS --arm $ARM --level 2 --full \
        --out-json "$OUT/$ARM.json" > "$OUT/$ARM.log" 2>&1
    echo "[done ] ov2 $DS/$ARM  $(date +%H:%M:%S)  $(grep -aoE '"accuracy": [0-9.]+' "$OUT/$ARM.log" | head -1)"
  done
done
echo "OV2 GAP SWEEP COMPLETE $(date)"
