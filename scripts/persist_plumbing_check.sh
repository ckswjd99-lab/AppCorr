#!/usr/bin/env bash
# Does `persist_correction_residual` actually reach the DINOv3 blocks?
#
# It did not, at first: the flag was plumbed only through the VGGT executor, so every DINOv3 config
# ignored it and both arms of an A/B came out bit-identical -- which reads as "the fix does nothing"
# rather than "the fix was never applied". Two arms that differ is the evidence that the plumbing
# works; identical output means it is still broken and the long sweeps would be wasted.
#
# Small on purpose: this checks wiring, not accuracy.
set -u
cd /NHNHOME/share/cjpark/AppCorr
for p in false true; do
  ps -eo pid,cmd | grep '[-]-recv-port 39996' | awk '{print $1}' | xargs -r kill -9 2>/dev/null
  sleep 5
  CUDA_VISIBLE_DEVICES=1 RECV_PORT=39996 SEND_PORT=39997 \
  PYTHONUNBUFFERED=1 APPCORR_RESULT_TIMEOUT=1800 timeout 1800 \
    ./offload/run_local.sh offload/config/nyu/nyu_interleaved_static.json -nr 2 -nw 0 \
    --set device=cuda:0 --set "appcorr_kwargs.persist_correction_residual=${p}" \
    > "logs/vggt/plumb_nyu_${p}.log" 2>&1
  echo "persist=${p}: $(grep -aoE "'abs_rel': [0-9.]+" "logs/vggt/plumb_nyu_${p}.log" | tail -1)"
done
echo "PLUMB_DONE"
