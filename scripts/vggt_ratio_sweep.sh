#!/usr/bin/env bash
set -u
cd /NHNHOME/share/cjpark/AppCorr
NR=20
for r in 0.05 0.10 0.20 0.40 0.70 1.00; do
  ps -eo pid,cmd | grep -E "offload/(server|mobile)/main" | grep -v grep | awk '{print $1}' | xargs -r kill -9 2>/dev/null
  sleep 4
  echo "##### token_keep_ratio=$r #####"
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 APPCORR_RESULT_TIMEOUT=1800 timeout 3600 \
    ./offload/run_local.sh offload/config/co3d/co3d_appcorr.json -nr $NR -nw 0 \
    -d /NHNHOME/share/cjpark/data/co3dv2/extracted \
    --set appcorr_kwargs.token_keep_ratio=$r 2>&1 \
  | grep -aoE "Final Summary: \{[^}]*\}|Avg. Transfer: [0-9.]+ KB|keep ratio during correction: [0-9.]+%|Avg recomputed patch count per active sample: [0-9.]+|CORRECT_FORWARD +\| *[0-9.]+|!!! \[Worker\][^:]{0,80}" | tail -6
done
echo "RATIO_SWEEP_DONE"
