#!/bin/bash
# Post-fix conditional re-entry sweep, full-2000, single-knob design:
# base operating point == the strict-disjoint arm (l1 1e-5, l0 4e-5, no safety
# gate -- July's l0=1.2e-4 retune and the 0.5 gate were bug-era compensations),
# support_mode=conditional_reentry, ratio in {0.10, 0.25, 0.50}.
# Anchors (2026-09-02 full-2000): disjoint(r=0) 61.126 | eqthr(~no block) 61.902
#                                 | L2-L0 61.826 | ceiling 62.236
set -uo pipefail
cd "$(dirname "$0")/.."
L=analysis/results/logs
mkdir -p "$L"

run_arm() {
  local name=$1 ratio=$2 pb=$3
  pkill -9 -f "offload/server/main.py|offload/mobile/main.py" 2>/dev/null || true
  sleep 3
  echo "=== ARM $name (ratio=$ratio) start $(date +%H:%M:%S) ==="
  RECV_PORT=$pb SEND_PORT=$((pb+1)) CUDA_VISIBLE_DEVICES=0 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    offload/run_local.sh offload/config/ade20k_m2f_interleaved_l2l1l0_conditional.json \
    -nr 2000 -nw 1 --set device=cuda:0 \
    --set appcorr_kwargs.l0_token_keep_thres=4e-5 \
    --set appcorr_kwargs.l1_remaining_energy_ratio_max=null \
    --set appcorr_kwargs.l1_l0_reentry_ratio=$ratio \
    --set exp_id="refix2000_${name}" > "$L/refix2000_${name}.log" 2>&1 || true
  echo "=== ARM $name done: $(grep -ao "Final Summary: {[^}]*}" "$L/refix2000_${name}.log" | tail -1)"
}

run_arm reentry10 0.10 43500
run_arm reentry25 0.25 43600
run_arm reentry50 0.50 43700
echo "=== REENTRY SWEEP COMPLETE ==="
