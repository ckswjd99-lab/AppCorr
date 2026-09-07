#!/usr/bin/env bash
# VSR (1222, COCO real photos, true/false) across the stack, queued behind ChartQA.
# Order: Qwen3.5 3-arm first (the model under evaluation), then gemma3 and ov2 through their own
# oracle drivers so the row becomes a cross-model comparison. All cheap (~1.2k samples, one-word
# answers).
set -u
cd /NHNHOME/share/cjpark/AppCorr-flops
export PATH=/home/nxclab/anaconda3/envs/appcorr/bin:$PATH
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1
unset HF_TOKEN
until grep -qa "QWEN35_ACCURACY_COMPLETE" analysis/results/qwen35_chartqa_accuracy.log; do sleep 60; done
echo "VSRCHAIN: chartqa done $(date)"

python analysis/experiments/qwen35_accuracy.py --dataset vsr --arms ceiling floor streaming --groups 4 \
  > analysis/results/qwen35_vsr_accuracy.log 2>&1
echo "VSRCHAIN: qwen35 rc=$? $(date)"

for M in gemma3 ov2; do
  script=analysis/experiments/${M}_oracle.py
  out=analysis/results/${M}_vsr; mkdir -p "$out"
  for spec in "ceiling:--arm ceiling" "floor:--arm floor" \
              "interleaved_g4_k0.25:--arm interleaved --keep 0.25 --groups 4" \
              "interleaved_g4_k0.50:--arm interleaved --keep 0.50 --groups 4"; do
    tag="${spec%%:*}"; args="${spec#*:}"
    [ -s "$out/$tag.json" ] && { echo "VSRCHAIN: skip $M/$tag"; continue; }
    python "$script" --dataset vsr --full --level 2 --out-json "$out/$tag.json" $args \
      > "$out/$tag.log" 2>&1
    echo "VSRCHAIN: $M/$tag rc=$? $(grep -aoE '"accuracy": [0-9.]+' "$out/$tag.log" | head -1) $(date)"
  done
done
echo "VSR_CHAIN_COMPLETE"
