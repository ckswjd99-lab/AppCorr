#!/bin/bash
# Paired 100-image re-measurement of the L2-L1-L0 track on the fixed codebase
# (persist 96889a5 + closed-loop 378e21d now under every arm).
# Pre-fix reference (2026-07-28/29, same first-100 protocol):
#   full 52.9229 | L2-L0 52.0812 | disjoint 50.6071 | equal-thr 52.1872
set -uo pipefail
cd "$(dirname "$0")/.."
LOGDIR=analysis/results/logs
mkdir -p "$LOGDIR"

preflight() {
  pkill -9 -f "offload/server/main.py|offload/mobile/main.py" 2>/dev/null; sleep 3
  [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ] \
    || { echo "GPU busy before $1 -- aborting"; exit 1; }
}

run_arm() {  # name config port_base
  local name=$1 cfg=$2 pb=$3
  preflight "$name"
  echo "=== ARM $name start $(date +%H:%M:%S) ==="
  RECV_PORT=$pb SEND_PORT=$((pb+1)) CUDA_VISIBLE_DEVICES=0 \
    offload/run_local.sh "$cfg" -nr 100 -nw 1 --set device=cuda:0 \
    --set exp_id="refix100_${name}" \
    > "$LOGDIR/refix100_${name}.log" 2>&1
  echo "=== ARM $name done: $(grep -o "Final Summary: {[^}]*}" "$LOGDIR/refix100_${name}.log" | tail -1)"
}

run_arm sequential offload/config/ade20k/ade20k_m2f_sequential.json            42100
run_arm l2l0_t4e5  offload/config/ade20k/ade20k_m2f_interleaved_static.json    42200
run_arm l2l1l0_disjoint offload/config/ade20k_m2f_interleaved_l2l1l0_static.json 42300
run_arm l2l1l0_eqthr offload/config/ade20k_m2f_interleaved_l2l1l0_threshold_only.json 42400
echo "=== CAMPAIGN COMPLETE ==="
