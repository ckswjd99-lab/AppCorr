#!/usr/bin/env bash
# Everything that must land before the 24:00 maintenance stop, in priority order.
# Every stage is resumable: the accuracy driver skips finished jsonl indices, and each
# stage checks for its own completed output before running.
set -u
cd /NHNHOME/share/cjpark/AppCorr-flops
export PATH=/home/nxclab/anaconda3/envs/appcorr/bin:$PATH
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1
unset HF_TOKEN
PY=python
ACC=analysis/experiments/qwen35_accuracy.py
LOGDIR=analysis/results

# 1) wait for the running ChartQA arm to finish its FLOOR, then take the GPU: the k=1.0
#    streaming arm that would follow is not a table column and its ~2.5h buys the keep arms.
until grep -qa '"arm": "floor"' "$LOGDIR/qwen35_chartqa_accuracy.log"; do sleep 60; done
for p in $(ps -eo pid,cmd | grep "[q]wen35_accuracy.py --dataset chartqa" | awk '{print $1}'); do
  kill -9 "$p" 2>/dev/null
done
echo "MQ: chartqa floor done, k1.0 streaming skipped $(date)"

# 2) RealWorldQA streaming keep arms (short first)
for K in 0.25 0.50; do
  $PY $ACC --dataset realworldqa --arms streaming --groups 4 --keep "$K" > "$LOGDIR/qwen35_rwqa_k${K}.log" 2>&1
  echo "MQ: rwqa k=$K rc=$? $(grep -ao '"acc": [0-9.]*' "$LOGDIR/qwen35_rwqa_k${K}.log" | tail -1) $(date)"
done

# 3) ChartQA streaming keep arms
for K in 0.25 0.50; do
  $PY $ACC --dataset chartqa --arms streaming --groups 4 --keep "$K" > "$LOGDIR/qwen35_chartqa_k${K}.log" 2>&1
  echo "MQ: chartqa k=$K rc=$? $(grep -ao '"acc": [0-9.]*' "$LOGDIR/qwen35_chartqa_k${K}.log" | tail -1) $(date)"
done

# 4) FLOPs with the keep knob (fills the 25% compute columns properly)
$PY analysis/experiments/flops_report_qwen35.py --samples 12 --groups 4 \
    --datasets chartqa realworldqa > "$LOGDIR/flops/qwen35_flops_keeps.log" 2>&1 || true
echo "MQ: flops rc=$? $(date)"

# 5) VSR with whatever time remains (qwen35 first, then gemma3/ov2) -- resumable
$PY $ACC --dataset vsr --arms ceiling floor --groups 4 > "$LOGDIR/qwen35_vsr_bounds.log" 2>&1
echo "MQ: vsr bounds rc=$? $(date)"
for K in 0.25 0.50; do
  $PY $ACC --dataset vsr --arms streaming --groups 4 --keep "$K" > "$LOGDIR/qwen35_vsr_k${K}.log" 2>&1
  echo "MQ: vsr k=$K rc=$? $(date)"
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
    echo "MQ: $M/$tag rc=$? $(date)"
  done
done
echo "MIDNIGHT_QUEUE_COMPLETE $(date)"
