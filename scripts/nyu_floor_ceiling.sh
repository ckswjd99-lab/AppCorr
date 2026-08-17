#!/usr/bin/env bash
# NYU floor and ceiling, so the interleaved persist A/B can be read as a fraction of the available
# gap instead of two bare abs_rel values.
#
# `nyu_interleaved_static` degrades to pyramid level 2, so its floor is `nyu_approx_only_l2`
# (levels [2], no correction) and its ceiling is `nyu_sequential` (raw transmission, no degradation).
# Both are 654 samples, a few minutes each.
#
# Runs on GPU 1 alongside the COCO queue on its own port pair -- ~163 GB was free and these are short.
set -u
cd /NHNHOME/share/cjpark/AppCorr
RECV=39998; SEND=39999
for cfg in nyu_approx_only_l2:floor nyu_sequential:ceiling; do
  name=${cfg%%:*}; tag=${cfg##*:}
  ps -eo pid,cmd | grep "[-]-recv-port $RECV" | awk '{print $1}' | xargs -r kill -9 2>/dev/null
  sleep 5
  echo "##### nyu $tag ($name) @ $(date +%H:%M) #####"
  CUDA_VISIBLE_DEVICES=1 RECV_PORT=$RECV SEND_PORT=$SEND \
  PYTHONUNBUFFERED=1 APPCORR_RESULT_TIMEOUT=3600 timeout 3600 \
    ./offload/run_local.sh "offload/config/nyu/$name.json" -nw 0 --set device=cuda:0 \
    > "logs/vggt/refresh_nyu_${tag}.log" 2>&1
  grep -aoE "Final Summary: \{[^}]*\}|!!! \[Worker\] Pipeline Error \(Req [0-9]+\): .{0,80}" \
    "logs/vggt/refresh_nyu_${tag}.log" | tail -2
done
echo "NYU_BOUNDS_DONE"
