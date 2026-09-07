#!/bin/bash
# disjoint + eqthr full-2000 (l2l0 already done: 61.8261).
set -uo pipefail
cd "$(dirname "$0")/.."
L=analysis/results/logs
mkdir -p "$L"

run_arm() {
  local name=$1 cfg=$2 pb=$3
  pkill -9 -f "offload/server/main.py|offload/mobile/main.py" 2>/dev/null || true
  sleep 3
  echo "=== ARM $name start $(date +%H:%M:%S) ==="
  RECV_PORT=$pb SEND_PORT=$((pb+1)) CUDA_VISIBLE_DEVICES=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    offload/run_local.sh "$cfg" -nr 2000 -nw 1 --set device=cuda:0 \
    --set exp_id="refix2000_${name}" > "$L/refix2000_${name}.log" 2>&1 || true
  echo "=== ARM $name done: $(grep -ao "Final Summary: {[^}]*}" "$L/refix2000_${name}.log" | tail -1)"
}

run_arm l2l1l0_disjoint offload/config/ade20k_m2f_interleaved_l2l1l0_static.json 43300
run_arm l2l1l0_eqthr    offload/config/ade20k_m2f_interleaved_l2l1l0_threshold_only.json 43400
echo "=== BOTH ARMS COMPLETE ==="
