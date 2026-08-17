#!/usr/bin/env bash
# floor and ceiling at the SAME n as the keep-threshold calibration sweep.
#
# COCO's mAP swings hard with the subset: `thres=0.002` gives 0.6011 over the full 5000 and 0.6727
# over the first 100. So the calibration curve cannot be read against the full-set floor (0.5583) and
# ceiling (0.6314) -- it needs its own bounds at matching n, or every recovered-fraction computed
# from it is nonsense.
set -u
cd /NHNHOME/share/cjpark/AppCorr
N=${N:-100}
until grep -qa "COCO_KEEP_SWEEP_DONE" logs/vggt/cocokeep.log 2>/dev/null; do sleep 20; done
for cfg in coco_approx_only_windowbase:floor coco_sequential:ceiling; do
  name=${cfg%%:*}; tag=${cfg##*:}
  ps -eo pid,cmd | grep '[-]-recv-port 39990' | awk '{print $1}' | xargs -r kill -9 2>/dev/null
  sleep 4
  CUDA_VISIBLE_DEVICES=0 RECV_PORT=39990 SEND_PORT=39991 \
  PYTHONUNBUFFERED=1 APPCORR_RESULT_TIMEOUT=1800 timeout 2400 \
    ./offload/run_local.sh "offload/config/coco/$name.json" -nr "$N" -nw 0 --set device=cuda:0 \
    > "logs/vggt/cocokeep_${tag}.log" 2>&1
  echo "${tag} (n=${N}) | $(grep -aoE "'mAP': [0-9.]+" "logs/vggt/cocokeep_${tag}.log" | tail -1)"
done
echo "COCO_KEEP_BOUNDS_DONE"
