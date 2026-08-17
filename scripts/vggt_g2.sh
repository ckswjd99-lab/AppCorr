#!/usr/bin/env bash
# VGGT interleaved at G=2, both arms, to extend the round-count series.
#
# Reference (n=20, floor 5.440 / ceiling 2.885, rot_deg): spatial G=4 3.177, G=8 3.181,
# one-shot 3.059. G=2 says whether the flatness across round counts extends down toward one-shot.
set -u
cd /NHNHOME/share/cjpark/AppCorr
NR=${NR:-20}
for p in run; do
  ps -eo pid,cmd | grep '[-]-recv-port 39990' | awk '{print $1}' | xargs -r kill -9 2>/dev/null
  sleep 5
  echo "##### co3d_il_spatial2 @ $(date +%H:%M) #####"
  CUDA_VISIBLE_DEVICES=0 RECV_PORT=39990 SEND_PORT=39991 \
  PYTHONUNBUFFERED=1 APPCORR_RESULT_TIMEOUT=1800 timeout 3600 \
    ./offload/run_local.sh offload/config/co3d/co3d_il_spatial2.json -nr $NR -nw 0 \
    -d /NHNHOME/share/cjpark/data/co3dv2/extracted > "logs/vggt/g2.log_run" 2>&1
  grep -aoE "Final Summary: \{[^}]*\}|!!! \[Worker\] Pipeline Error \(Req [0-9]+\): .{0,80}" \
    "logs/vggt/g2.log_run" | tail -2
done
echo "G2_DONE"
