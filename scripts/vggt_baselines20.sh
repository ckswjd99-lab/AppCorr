#!/usr/bin/env bash
# Ceiling and floor on the SAME 20 sequences the ratio sweep used, so recovery is computable.
# Without these the sweep numbers cannot be turned into "% of the gap recovered".
set -u
cd /NHNHOME/share/cjpark/AppCorr
for c in co3d_full co3d_approx_only; do
  ps -eo pid,cmd | grep -E "offload/(server|mobile)/main" | grep -v grep | awk '{print $1}' | xargs -r kill -9 2>/dev/null
  sleep 4
  echo "##### $c #####"
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 APPCORR_RESULT_TIMEOUT=1800 timeout 3600 \
    ./offload/run_local.sh offload/config/co3d/$c.json -nr 20 -nw 0 \
    -d /NHNHOME/share/cjpark/data/co3dv2/extracted 2>&1 \
  | grep -aoE "Final Summary: \{[^}]*\}|Avg. Transfer: [0-9.]+ KB" | tail -2
done
echo "BASELINES20_DONE"
