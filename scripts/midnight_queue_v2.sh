#!/usr/bin/env bash
# v2 (user reorder): VSR before the remaining ChartQA arm. The in-flight chartqa k0.25
# finishes untouched; everything after is VSR-first.
set -u
cd /NHNHOME/share/cjpark/AppCorr-flops
export PATH=/home/nxclab/anaconda3/envs/appcorr/bin:$PATH
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1
unset HF_TOKEN
PY=python
ACC=analysis/experiments/qwen35_accuracy.py
LOGDIR=analysis/results

until grep -qa "QWEN35_ACCURACY_COMPLETE" "$LOGDIR/qwen35_chartqa_k0.25.log" 2>/dev/null; do sleep 60; done
echo "MQ2: chartqa k0.25 done $(date)"

# --- VSR first: qwen35 bounds -> keep arms -> gemma3 -> ov2 ---------------------------- #
$PY $ACC --dataset vsr --arms ceiling floor --groups 4 > "$LOGDIR/qwen35_vsr_bounds.log" 2>&1
echo "MQ2: vsr bounds rc=$? $(date)"
for K in 0.25 0.50; do
  $PY $ACC --dataset vsr --arms streaming --groups 4 --keep "$K" > "$LOGDIR/qwen35_vsr_k${K}.log" 2>&1
  echo "MQ2: vsr qwen35 k=$K rc=$? $(grep -ao '"acc": [0-9.]*' "$LOGDIR/qwen35_vsr_k${K}.log" | tail -1) $(date)"
done
for M in gemma3 ov2; do
  script=analysis/experiments/${M}_oracle.py
  out=analysis/results/${M}_vsr; mkdir -p "$out"
  for spec in "ceiling:--arm ceiling" "floor:--arm floor" \
              "interleaved_g4_k0.25:--arm interleaved --keep 0.25 --groups 4" \
              "interleaved_g4_k0.50:--arm interleaved --keep 0.50 --groups 4"; do
    tag="${spec%%:*}"; args="${spec#*:}"
    [ -s "$out/$tag.json" ] && continue
    $PY "$script" --dataset vsr --full --level 2 --out-json "$out/$tag.json" $args > "$out/$tag.log" 2>&1
    echo "MQ2: $M/$tag rc=$? $(grep -aoE '"accuracy": [0-9.]+' "$out/$tag.log" | head -1) $(date)"
  done
done

# --- then the remaining ChartQA arm and FLOPs ------------------------------------------ #
$PY $ACC --dataset chartqa --arms streaming --groups 4 --keep 0.50 > "$LOGDIR/qwen35_chartqa_k0.50.log" 2>&1
echo "MQ2: chartqa k=0.50 rc=$? $(grep -ao '"acc": [0-9.]*' "$LOGDIR/qwen35_chartqa_k0.50.log" | tail -1) $(date)"
$PY analysis/experiments/flops_report_qwen35.py --samples 12 --groups 4 \
    --datasets chartqa realworldqa > "$LOGDIR/flops/qwen35_flops_keeps.log" 2>&1 || true
echo "MQ2: flops rc=$? $(date)"
echo "MIDNIGHT_QUEUE_V2_COMPLETE $(date)"
