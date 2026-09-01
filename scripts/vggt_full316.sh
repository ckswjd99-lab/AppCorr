#!/usr/bin/env bash
# VGGT-Omega over all 316 Co3D sequences, so its numbers stand on the same footing as the DINOv3
# families (ADE20K 2000, COCO 5000, ImageNet 50000, NYU 654 -- all full sets).
#
# Everything published for VGGT so far is n=20, i.e. 6.3% of the data. The conclusions drawn from it
# -- that with the fix the round count and the grouping stop mattering (G=4 88.6% vs G=8 88.4%,
# spatial vs per_frame 1.2pp) -- rest on differences of a few hundredths of a degree at that sample
# size, which is not enough to carry them.
#
# These are all post-fix runs: persisting the corrected increment is unconditional now, so the
# pre-fix arms recorded at n=20 cannot be reproduced and are not attempted.
#
# Usage: vggt_full316.sh <gpu> <recv-port> <tag> <config...>
set -u
cd /NHNHOME/share/cjpark/AppCorr
GPU=$1; RECV=$2; TAG=$3; shift 3
NR=${NR:-316}
for c in "$@"; do
  ps -eo pid,cmd | grep "[-]-recv-port $RECV" | awk '{print $1}' | xargs -r kill -9 2>/dev/null
  sleep 4
  echo "##### [$TAG] $c @ $(date +%H:%M) #####"
  CUDA_VISIBLE_DEVICES=$GPU RECV_PORT=$RECV SEND_PORT=$((RECV + 1)) \
  PYTHONUNBUFFERED=1 APPCORR_RESULT_TIMEOUT=3600 \
    ./offload/run_local.sh "offload/config/co3d/$c.json" -nr "$NR" -nw 0 \
    -d /NHNHOME/share/cjpark/data/co3dv2/extracted > "logs/vggt/full316_${c}.log" 2>&1
  grep -aoE "Final Summary: \{[^}]*\}|!!! \[Worker\] Pipeline Error \(Req [0-9]+\): .{0,80}" \
    "logs/vggt/full316_${c}.log" | tail -2
done
echo "FULL316_${TAG}_DONE"
