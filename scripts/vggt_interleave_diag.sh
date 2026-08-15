#!/usr/bin/env bash
# Where does interleaved correction lose accuracy: layer truncation or token partitioning?
#
#   co3d_il_g1            interleaved scheduler, one group -- must reproduce co3d_appcorr (3.059).
#                         If it does not, the interleaved path itself is wrong and nothing below
#                         means anything.
#   *_full                every round corrects over all 48 stages, so correction compute equals
#                         one-shot and only the G-separate-passes structure remains. Matching
#                         co3d_appcorr here pins the loss on layer truncation (inherent); falling
#                         short pins it on the partitioning (a design or implementation problem).
set -u
cd /NHNHOME/share/cjpark/AppCorr
NR=${NR:-20}
for c in co3d_il_g1 co3d_il_spatial8_full co3d_il_frame8_full; do
  ps -eo pid,cmd | grep -E "offload/(server|mobile)/main" | grep -v grep | awk '{print $1}' | xargs -r kill -9 2>/dev/null
  sleep 4
  echo "##### $c #####"
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 APPCORR_RESULT_TIMEOUT=1800 timeout 3600 \
    ./offload/run_local.sh offload/config/co3d/$c.json -nr $NR -nw 0 \
    -d /NHNHOME/share/cjpark/data/co3dv2/extracted > "logs/vggt/diag_${c}.log" 2>&1
  grep -aoE "Final Summary: \{[^}]*\}|!!! \[Worker\] Pipeline Error \(Req [0-9]+\): .{0,80}" \
    "logs/vggt/diag_${c}.log" | tail -2
done
echo "INTERLEAVE_DIAG_DONE"
