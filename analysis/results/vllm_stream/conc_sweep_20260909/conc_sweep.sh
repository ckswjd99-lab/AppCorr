#!/bin/bash
# Concurrency sweep (35B): does batching move the engine off the flat ~40 ms step floor and let
# streaming's 1/g critical work show up as latency? Closed loop, N in flight, 160 evenly spaced
# images, 2 datasets x {1,4,8,16} x {ceiling, streaming k=1.0}. Server trace on for step tokens.
set -u
REPO=/NHNHOME/share/cjpark/AppCorr-vllm; cd $REPO
PYV=/NHNHOME/storage/users/cjpark/shk/conda_envs/appcorr-vllm/bin/python
PY=/home/nxclab/anaconda3/envs/appcorr/bin/python
L=$REPO/logs/vllm_stream; PORT=5591
OUT=$REPO/analysis/results/vllm_stream/conc_sweep_20260909
mkdir -p $OUT
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTHONPATH=$REPO
unset HF_TOKEN
export APPCORR_SERVER_TRACE=$OUT/server_trace_35b.jsonl
rm -f $APPCORR_SERVER_TRACE
VLLM_ENABLE_V1_MULTIPROCESSING=0 $PYV -m appcorr.vllm_stream.server --model Qwen/Qwen3.5-35B-A3B --port $PORT \
  --gpu-mem 0.60 --max-model-len 16384 --max-num-seqs 64 > $L/server_qwen35_35b_conc.log 2>&1 &
SERVER=$!; echo "server pid $SERVER $(date +%T)"
for i in $(seq 1 90); do grep -q "listening on" $L/server_qwen35_35b_conc.log && break; kill -0 $SERVER 2>/dev/null || { echo "server died"; exit 1; }; sleep 10; done
grep -q "listening on" $L/server_qwen35_35b_conc.log || { echo "server never listened"; kill $SERVER; exit 1; }
echo "server up $(date +%T)"
unset APPCORR_SERVER_TRACE
BOUNDS=$OUT/bounds.jsonl; : > $BOUNDS
for spec in realworldqa:box visdrone_det:pyr; do
  ds=${spec%%:*}; filt=${spec##*:}
  for c in 1 4 8 16; do
    for arm in ceiling streaming; do
      d=$OUT/$ds/c$c; mkdir -p $d
      t0=$($PY -c "import time;print(time.perf_counter())")
      $PY analysis/experiments/qwen_vllm_accuracy.py --family qwen35 --model Qwen/Qwen3.5-35B-A3B --port $PORT \
        --groups 4 --level 2 --workers 6 --load vision --samples 160 --concurrency $c --out $d \
        --dataset $ds --degrade-filter $filt --arms $arm --keep 1.0 > $d/log_$arm.log 2>&1
      rc=$?
      t1=$($PY -c "import time;print(time.perf_counter())")
      echo "{\"ds\":\"$ds\",\"c\":$c,\"arm\":\"$arm\",\"t0\":$t0,\"t1\":$t1,\"rc\":$rc}" >> $BOUNDS
      echo "  $ds c=$c $arm rc=$rc $(grep -o 'samples_per_s\": [0-9.]*' $d/log_$arm.log) $(date +%T)"
    done
  done
done
kill $SERVER; for i in $(seq 1 30); do kill -0 $SERVER 2>/dev/null || break; sleep 2; done
echo "server $SERVER stopped $(date +%T)"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "CONC_DONE $(date +%T)"
