#!/usr/bin/env bash
# Fill in the steep stretch of the COCO keep-threshold curve.
#
# The calibration pass jumps 20.41% -> 50.09% of tokens between thres 0.002 and 0.001, and that one
# step carries +0.026 mAP while everything above 90% is flat. All of the interesting trade-off lives
# in that gap, and two endpoints cannot say whether it bends.
#
# Same N as the first pass so the points are comparable; COCO's mAP moves too much between subsets to
# mix sample sizes on one curve.
set -u
cd /NHNHOME/share/cjpark/AppCorr
N=${N:-100}
CFG=offload/config/coco/coco_interleaved_static.json
until grep -qa "COCO_KEEP_BOUNDS_DONE" logs/vggt/cocokeep.log 2>/dev/null; do sleep 20; done
for thres in 0.0018 0.0016 0.0014 0.0012; do
  ps -eo pid,cmd | grep '[-]-recv-port 39990' | awk '{print $1}' | xargs -r kill -9 2>/dev/null
  sleep 4
  tag=${thres//./p}
  CUDA_VISIBLE_DEVICES=0 RECV_PORT=39990 SEND_PORT=39991 \
  PYTHONUNBUFFERED=1 APPCORR_RESULT_TIMEOUT=1800 timeout 2400 \
    ./offload/run_local.sh "$CFG" -nr "$N" -nw 0 --set device=cuda:0 \
    --set "appcorr_kwargs.token_keep_thres=$thres" \
    > "logs/vggt/cocokeep_${tag}.log" 2>&1
  keep=$(tr '\r' '\n' < "logs/vggt/cocokeep_${tag}.log" | grep -aoE "keep ratio during correction: [0-9.]+%" | tail -1)
  cnt=$(tr '\r' '\n' < "logs/vggt/cocokeep_${tag}.log" | grep -aoE "recomputed patch count per active sample: [0-9.]+" | tail -1)
  map=$(grep -aoE "'mAP': [0-9.]+" "logs/vggt/cocokeep_${tag}.log" | tail -1)
  err=$(grep -acE "Traceback|Pipeline Error" "logs/vggt/cocokeep_${tag}.log")
  echo "thres=${thres} | ${keep} | ${cnt} | ${map} | errors=${err}"
done
echo "COCO_KEEP_FILL_DONE"
