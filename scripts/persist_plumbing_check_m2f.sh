#!/usr/bin/env bash
# Same wiring check as persist_plumbing_check.sh, but for the ADE20K m2f path.
#
# Needed separately because m2f's real correction call bypasses `run_dinov3_correct_block` and reads
# `appcorr_options["persist_correction_residual"]` directly -- the NYU check exercises the shared
# entry point and says nothing about this one. ADE20K is the headline table, so a KeyError here
# would surface an hour into the long arm.
set -u
cd /NHNHOME/share/cjpark/AppCorr
for p in false true; do
  ps -eo pid,cmd | grep '[-]-recv-port 39996' | awk '{print $1}' | xargs -r kill -9 2>/dev/null
  sleep 5
  CUDA_VISIBLE_DEVICES=1 RECV_PORT=39996 SEND_PORT=39997 \
  PYTHONUNBUFFERED=1 APPCORR_RESULT_TIMEOUT=1800 timeout 2400 \
    ./offload/run_local.sh offload/config/ade20k/ade20k_m2f_interleaved_static.json -nr 8 -nw 0 \
    --set device=cuda:0 --set "appcorr_kwargs.persist_correction_residual=${p}" \
    > "logs/vggt/plumb_ade_${p}.log" 2>&1
  echo "persist=${p}: $(grep -aoE "'mIoU': [0-9.]+|mIoU: [0-9.]+" "logs/vggt/plumb_ade_${p}.log" | tail -1) $(grep -acE "Traceback|KeyError" "logs/vggt/plumb_ade_${p}.log") errors"
done
echo "PLUMB_M2F_DONE"
