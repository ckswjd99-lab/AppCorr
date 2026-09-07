#!/usr/bin/env bash
# Small-scale RefCOCO capability probes (user request): can Mistral / MG /
# Gemma4 do grounding at all? (OV2 precedent: ~6% = task-incapable.)
# 50 strided samples each, ceiling arm only, preds dumped for dual-convention
# scoring (pixel vs 0-1000) in the session afterward.
set -u
cd /NHNHOME/share/cjpark/AppCorr-flops
export PATH=/home/nxclab/anaconda3/envs/appcorr/bin:$PATH
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1
unset HF_TOKEN
OUT=analysis/results/refcoco_probe
mkdir -p "$OUT"

while pgrep -f "qwen35_accuracy.py --dataset refcoco" > /dev/null; do sleep 60; done

python analysis/experiments/gemma4_oracle.py --dataset refcoco --arm ceiling \
  --num-samples 50 --max-new-tokens 32 --out-json "$OUT/gemma4_ceiling50.json" \
  > "$OUT/gemma4.log" 2>&1
echo "PS: refcoco-probe gemma4 rc=$? $(grep -aoE '"accuracy": [0-9.]+' $OUT/gemma4.log | head -1) $(date)"

python analysis/experiments/mistral3_oracle.py --dataset refcoco --arm ceiling \
  --num-samples 50 --max-new-tokens 32 --out-json "$OUT/mistral_ceiling50.json" \
  > "$OUT/mistral.log" 2>&1
echo "PS: refcoco-probe mistral rc=$? $(grep -aoE '"accuracy": [0-9.]+' $OUT/mistral.log | head -1) $(date)"

python analysis/experiments/vlm_bounds_oracle.py --model meta-models/Muse-Glimmer-30B \
  --dataset refcoco --arm ceiling --num-samples 50 --max-new-tokens 48 \
  --transformers-path /NHNHOME/share/cjpark/tf515 \
  --out-json "$OUT/museglimmer_ceiling50.json" > "$OUT/museglimmer.log" 2>&1
echo "PS: refcoco-probe museglimmer rc=$? $(grep -aoE '"accuracy": [0-9.]+' $OUT/museglimmer.log | head -1) $(date)"
echo "REFCOCO_PROBE_COMPLETE $(date)"
