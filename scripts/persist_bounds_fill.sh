#!/usr/bin/env bash
# Floor and ceiling for imnet and COCO, so their persist A/B can be read as a recovered fraction.
#
# Learned from NYU: two bare metric values do not say whether an improvement is large. NYU's pair
# came back with no floor/ceiling measured, and the frame turned out to be unusable once they were --
# both correction arms sat past the ceiling on two of three metrics because L2 barely costs NYU
# anything. Measure the bounds *with* the pair, not after wondering.
#
#   imnet: floor imnet_approx_only_l2 (levels [2], the interleaved base), ceiling imnet_sequential
#   COCO : floor coco_approx_only_windowbase (window base, no correction), ceiling coco_sequential
#
# Waits on log markers rather than process names -- `pgrep -f <pattern>` matches its own command line.
set -u
cd /NHNHOME/share/cjpark/AppCorr

run() {  # cfg tag gpu recv
  ps -eo pid,cmd | grep "[-]-recv-port $4" | awk '{print $1}' | xargs -r kill -9 2>/dev/null
  sleep 5
  echo "##### $2 @ $(date +%H:%M) #####"
  CUDA_VISIBLE_DEVICES=$3 RECV_PORT=$4 SEND_PORT=$(($4 + 1)) \
  PYTHONUNBUFFERED=1 APPCORR_RESULT_TIMEOUT=3600 \
    ./offload/run_local.sh "$1" -nw 0 --set device=cuda:0 \
    > "logs/vggt/refresh_$2.log" 2>&1
  grep -aoE "Final Summary: \{[^}]*\}|!!! \[Worker\] Pipeline Error \(Req [0-9]+\): .{0,80}" \
    "logs/vggt/refresh_$2.log" | tail -2
}

until grep -qa "IMNET_DONE" logs/vggt/refresh_imnet.log 2>/dev/null; do sleep 30; done
run offload/config/imnet/imnet_approx_only_l2.json imnet_floor   0 39990
run offload/config/imnet/imnet_sequential.json     imnet_ceiling 0 39990
echo "IMNET_BOUNDS_DONE"

until grep -qa "GPU1_REFRESH_DONE" logs/vggt/refresh_gpu1.log 2>/dev/null; do sleep 30; done
run offload/config/coco/coco_approx_only_windowbase.json coco_floor   1 39996
run offload/config/coco/coco_sequential.json             coco_ceiling 1 39996
echo "COCO_BOUNDS_DONE"
echo "BOUNDS_DONE"
