#!/bin/bash
# Sharded concurrency sweep (35B): N driver processes (each its own vision tower) x c in flight
# each, against one server -> engine sees N*c requests and can batch prefills, which the single
# serial driver never let it do (req/step stayed 1 up to c=8). Same 2 datasets, streaming k=1.0
# vs ceiling, server trace on for tokens/step.
set -u
REPO=/NHNHOME/share/cjpark/AppCorr-vllm; cd $REPO
PYV=/NHNHOME/storage/users/cjpark/shk/conda_envs/appcorr-vllm/bin/python
PY=/home/nxclab/anaconda3/envs/appcorr/bin/python
L=$REPO/logs/vllm_stream; PORT=5591
OUT=$REPO/analysis/results/vllm_stream/conc_sweep_20260909/shard
mkdir -p $OUT
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTHONPATH=$REPO
unset HF_TOKEN
export APPCORR_SERVER_TRACE=$OUT/server_trace_35b.jsonl
rm -f $APPCORR_SERVER_TRACE
VLLM_ENABLE_V1_MULTIPROCESSING=0 $PYV -m appcorr.vllm_stream.server --model Qwen/Qwen3.5-35B-A3B --port $PORT \
  --gpu-mem 0.60 --max-model-len 16384 --max-num-seqs 64 > $L/server_qwen35_35b_shard.log 2>&1 &
SERVER=$!; echo "server pid $SERVER $(date +%T)"
for i in $(seq 1 90); do grep -q "listening on" $L/server_qwen35_35b_shard.log && break; kill -0 $SERVER 2>/dev/null || { echo "server died"; exit 1; }; sleep 10; done
grep -q "listening on" $L/server_qwen35_35b_shard.log || { echo "server never listened"; kill $SERVER; exit 1; }
echo "server up $(date +%T)"
unset APPCORR_SERVER_TRACE
BOUNDS=$OUT/bounds.jsonl; : > $BOUNDS
NS=${NS:-400}
for spec in realworldqa:box visdrone_det:pyr; do
  ds=${spec%%:*}; filt=${spec##*:}
  for cfg in 4:1 8:1 8:2 16:1; do
    N=${cfg%%:*}; c=${cfg##*:}
    for arm in ceiling streaming; do
      d=$OUT/$ds/n${N}c$c; mkdir -p $d
      t0=$($PY -c "import time;print(time.perf_counter())")
      pids=()
      for k in $(seq 0 $((N-1))); do
        $PY analysis/experiments/qwen_vllm_accuracy.py --family qwen35 --model Qwen/Qwen3.5-35B-A3B --port $PORT \
          --groups 4 --level 2 --workers 2 --load vision --samples $NS --concurrency $c --shard $k/$N --out $d \
          --dataset $ds --degrade-filter $filt --arms $arm --keep 1.0 > $d/log_${arm}_s$k.log 2>&1 &
        pids+=($!)
      done
      rc=0; for p in "${pids[@]}"; do wait $p || rc=1; done
      t1=$($PY -c "import time;print(time.perf_counter())")
      echo "{\"ds\":\"$ds\",\"N\":$N,\"c\":$c,\"arm\":\"$arm\",\"t0\":$t0,\"t1\":$t1,\"rc\":$rc}" >> $BOUNDS
      echo "  $ds N=$N c=$c $arm rc=$rc $(date +%T) mem $(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i 0)"
    done
  done
done
kill $SERVER; for i in $(seq 1 30); do kill -0 $SERVER 2>/dev/null || break; sleep 2; done
echo "server $SERVER stopped $(date +%T)"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
echo "SHARD_DONE $(date +%T)"
