#!/usr/bin/env bash
# The causal-LLM streaming category (user directive): LLM prefill streams, vision corrects
# at keep=1.0. Runs after queue v2 + the 122B FLOPs chain. Order: the two missing Qwen3.5
# pieces, then OV2 streaming accuracy across its datasets (no new code -- keep=1.0 IS the
# existing streaming_forward), then OV2 streaming FLOPs.
set -u
cd /NHNHOME/share/cjpark/AppCorr-flops
export PATH=/home/nxclab/anaconda3/envs/appcorr/bin:$PATH
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1
unset HF_TOKEN
PY=python
until grep -qa "QWEN122B_FLOPS_COMPLETE" analysis/results/qwen122b_flops_chain.log 2>/dev/null; do sleep 60; done
echo "SC: start $(date)"

# Qwen3.5 ChartQA streaming k1.0 (the category's canonical arm; was skipped this morning)
$PY analysis/experiments/qwen35_accuracy.py --dataset chartqa --arms streaming --groups 4 \
  > analysis/results/qwen35_chartqa_k1.0.log 2>&1
echo "SC: qwen35 chartqa k1.0 rc=$? $(grep -ao '"acc": [0-9.]*' analysis/results/qwen35_chartqa_k1.0.log | tail -1) $(date)"

# OV2 streaming accuracy: key wide-gap datasets first, then the rest as time allows
for DS in chartqa textvqa infovqa docvqa realworldqa pope gqa vsr; do
  out=analysis/results/ov2_${DS}; mkdir -p "$out"
  [ -s "$out/streaming_g4.json" ] && { echo "SC: skip ov2/$DS"; continue; }
  $PY analysis/experiments/ov2_oracle.py --dataset "$DS" --full --level 2 \
      --arm streaming --groups 4 --out-json "$out/streaming_g4.json" \
      > "$out/streaming_g4.log" 2>&1
  echo "SC: ov2/$DS rc=$? $(grep -aoE '"accuracy": [0-9.]+' "$out/streaming_g4.log" | head -1) $(date)"
done
echo "STREAMING_CATEGORY_COMPLETE $(date)"
