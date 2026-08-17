#!/usr/bin/env bash
# COCO: recompute vs accuracy as a function of `token_keep_thres`.
#
# The shipped default (0.002) recomputes only 20.6% of tokens and reaches mAP 0.6011 against a
# floor of 0.5583 and a ceiling of 0.6314 -- 58.6% of the gap. That is a hyper-parameter operating
# point, not a property of correction, so this maps the trade-off instead of arguing about one point.
#
# Calibration pass first: keep ratio is a per-token property and settles quickly, so N=100 is enough
# to map threshold -> recompute fraction and to see the rough shape. Full 5000-image runs cost ~1.5 h
# each and are worth spending only on the points that matter, chosen from this.
set -u
cd /NHNHOME/share/cjpark/AppCorr
N=${N:-100}
CFG=offload/config/coco/coco_interleaved_static.json
for thres in 0.002 0.001 0.0005 0.0002 0.0001 0.0; do
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
echo "COCO_KEEP_SWEEP_DONE"
