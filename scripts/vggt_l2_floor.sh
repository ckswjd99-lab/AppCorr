#!/usr/bin/env bash
# The same VGGT table with the base at pyramid level 2 instead of 3 -- a milder degradation, so a
# narrower floor-to-ceiling gap and a harder test of what correction is worth.
#
# `pyramid_levels` lives only in transmission_kwargs for these configs, so one `--set` moves it
# without tripping the appcorr/transmission consistency check. The ceiling (`co3d_full`) has no
# pyramid at all and is reused from the L3 sweep: 1.3303 rot / 0.04253 AbsRel.
#
# Usage: vggt_l2_floor.sh <gpu> <recv-port> <tag> <config...>
set -u
cd /NHNHOME/share/cjpark/AppCorr
GPU=$1; RECV=$2; TAG=$3; shift 3
NR=${NR:-316}
for c in "$@"; do
  ps -eo pid,cmd | grep "[-]-recv-port $RECV" | awk '{print $1}' | xargs -r kill -9 2>/dev/null
  sleep 4
  echo "##### [$TAG] $c @ $(date +%H:%M) #####"
  # co3d_approx_only_l2 already *is* the L2 base; everything else needs its [3,0] moved to [2,0]
  extra=(); [ "$c" = "co3d_approx_only_l2" ] || extra=(--set "transmission_kwargs.pyramid_levels=[2,0]")
  CUDA_VISIBLE_DEVICES=$GPU RECV_PORT=$RECV SEND_PORT=$((RECV + 1)) \
  PYTHONUNBUFFERED=1 APPCORR_RESULT_TIMEOUT=3600 \
    ./offload/run_local.sh "offload/config/co3d/$c.json" -nr "$NR" -nw 0 "${extra[@]}" \
    -d /NHNHOME/share/cjpark/data/co3dv2/extracted > "logs/vggt/l2_${c}.log" 2>&1
  grep -aoE "Final Summary: \{[^}]*\}|!!! \[Worker\] Pipeline Error \(Req [0-9]+\): .{0,80}" \
    "logs/vggt/l2_${c}.log" | tail -2
done
echo "L2FLOOR_${TAG}_DONE"
