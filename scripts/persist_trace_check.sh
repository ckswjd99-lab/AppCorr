#!/usr/bin/env bash
# Positive proof that the persist write executes, for a given config.
#
# A metric that barely moves cannot tell "small real effect" apart from "still not wired" -- COCO's
# 8-image gate differed by 0.00017 mAP, which is within numerical noise. APPCORR_PERSIST_TRACE makes
# the block print once when it actually writes `blocks_out_sum`, so absence of the line is proof of
# absence rather than a weak inference from a metric.
#
# Usage: persist_trace_check.sh <config> [nr] [port]
set -u
cd /NHNHOME/share/cjpark/AppCorr
CFG=$1; NR=${2:-2}; RECV=${3:-39998}; SEND=$((RECV + 1))
TAG=$(basename "$CFG" .json)
ps -eo pid,cmd | grep "[-]-recv-port $RECV" | awk '{print $1}' | xargs -r kill -9 2>/dev/null
sleep 5
CUDA_VISIBLE_DEVICES=${GPU:-1} RECV_PORT=$RECV SEND_PORT=$SEND APPCORR_PERSIST_TRACE=1 \
PYTHONUNBUFFERED=1 APPCORR_RESULT_TIMEOUT=1800 timeout 1800 \
  ./offload/run_local.sh "$CFG" -nr "$NR" -nw 0 --set device=cuda:0 \
  --set appcorr_kwargs.persist_correction_residual=true \
  > "logs/vggt/trace_${TAG}.log" 2>&1
n=$(grep -ac "\[persist\] blocks_out_sum write executed" "logs/vggt/trace_${TAG}.log")
echo "TRACE ${TAG}: persist-write lines = ${n}"
grep -a "\[persist\]" "logs/vggt/trace_${TAG}.log" | head -1
[ "$n" -gt 0 ] && echo "TRACE_OK ${TAG}" || echo "TRACE_FAILED ${TAG} -- the write never ran"
