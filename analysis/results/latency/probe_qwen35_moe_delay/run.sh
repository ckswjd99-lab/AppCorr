#!/bin/bash
# Spaced-arrival latency probe (35B, GPU0): streaming keep=1.0 with APPCORR_PUSH_DELAY_MS in
# {0, 60, 150} on RWQA / VisDrone Det / V*, 36 samples each, concurrency 1. Same server for all
# arms; 0 ms is the in-session control. Outputs under results/latency/probe_qwen35_moe_delay/d<ms>/.
set -u
REPO=/NHNHOME/share/cjpark/AppCorr-vllm; cd $REPO
PYV=/NHNHOME/storage/users/cjpark/shk/conda_envs/appcorr-vllm/bin/python
PY=/home/nxclab/anaconda3/envs/appcorr/bin/python
L=$REPO/logs/vllm_stream; PORT=5591
OUT=$REPO/analysis/results/latency/probe_qwen35_moe_delay
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTHONPATH=$REPO
unset HF_TOKEN
export APPCORR_SERVER_TRACE=$OUT/server_trace.jsonl
VLLM_ENABLE_V1_MULTIPROCESSING=0 $PYV -m appcorr.vllm_stream.server --model Qwen/Qwen3.5-35B-A3B --port $PORT \
  --gpu-mem 0.60 --max-model-len 16384 --max-num-seqs 64 > $L/server_qwen35_35b_delay.log 2>&1 &
SERVER=$!; echo "server pid $SERVER $(date +%T)"
for i in $(seq 1 90); do
  grep -q "listening on" $L/server_qwen35_35b_delay.log && break
  kill -0 $SERVER 2>/dev/null || { echo "server died"; exit 1; }
  sleep 10
done
grep -q "listening on" $L/server_qwen35_35b_delay.log || { echo "server never listened"; kill $SERVER; exit 1; }
echo "server up $(date +%T)"
unset APPCORR_SERVER_TRACE
for D in 0 60 150; do
  mkdir -p $OUT/d$D
  for spec in realworldqa:box visdrone_det:pyr vstar:pyr; do
    ds=${spec%%:*}; filt=${spec##*:}
    rm -f $OUT/d$D/${ds}_qwen3.5-35b-a3b_streaming_g4.jsonl
    t0=$(date +%s)
    APPCORR_PUSH_DELAY_MS=$D $PY analysis/experiments/qwen_vllm_accuracy.py --family qwen35 --model Qwen/Qwen3.5-35B-A3B \
      --port $PORT --groups 4 --level 2 --workers 6 --load vision --samples 36 --concurrency 1 \
      --out $OUT/d$D --pscore deferred --dataset $ds --degrade-filter $filt --arms streaming --keep 1.0 \
      > $OUT/d$D/log_${ds}.log 2>&1
    echo "d=$D $ds rc=$? $(( $(date +%s) - t0 ))s $(date +%T)"
  done
done
kill $SERVER; for i in $(seq 1 30); do kill -0 $SERVER 2>/dev/null || break; sleep 2; done
echo "server stopped $(date +%T)"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "DELAY_DONE $(date +%T)"
