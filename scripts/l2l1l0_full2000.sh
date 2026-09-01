#!/bin/bash
# Full-2000 promotion of the post-fix L2-L1-L0 arms (paired-100 verdict 2026-09-02:
# disjoint level with L2-L0 at equal correction FLOPs; eq-thr above full).
# Sequential full-2000 ceiling (62.236) is reused -- unaffected by both fixes,
# reuse-validated in dinov3_correct_low_precision_status.md.
set -uo pipefail
cd "$(dirname "$0")/.."
LOGDIR=analysis/results/logs
mkdir -p "$LOGDIR"

preflight() {
  pkill -9 -f "offload/server/main.py|offload/mobile/main.py" 2>/dev/null; sleep 3
  [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" = "0" ] \
    || { echo "GPU busy before $1 -- aborting"; exit 1; }
}

run_arm() {
  local name=$1 cfg=$2 pb=$3
  preflight "$name"
  echo "=== ARM $name start $(date +%H:%M:%S) ==="
  RECV_PORT=$pb SEND_PORT=$((pb+1)) CUDA_VISIBLE_DEVICES=0 \
    offload/run_local.sh "$cfg" -nr 2000 -nw 1 --set device=cuda:0 \
    --set exp_id="refix2000_${name}" \
    > "$LOGDIR/refix2000_${name}.log" 2>&1
  echo "=== ARM $name done: $(grep -o "Final Summary: {[^}]*}" "$LOGDIR/refix2000_${name}.log" | tail -1)"
}

run_arm l2l0_t4e5       offload/config/ade20k/ade20k_m2f_interleaved_static.json          43200
run_arm l2l1l0_disjoint offload/config/ade20k_m2f_interleaved_l2l1l0_static.json          43300
run_arm l2l1l0_eqthr    offload/config/ade20k_m2f_interleaved_l2l1l0_threshold_only.json  43400
echo "=== FULL-2000 COMPLETE ==="
