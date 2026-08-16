#!/usr/bin/env bash
# Interleaved correction vs one-shot, at matched keep ratio.
#
# co3d_appcorr is the one-shot reference: the whole residual arrives at once and correction runs
# once over the full depth. The interleaved configs split the same residual into rounds, so the same
# tokens are ultimately corrected -- what changes is when, and against how much of the network.
#
# The pre-fix arm is gone: persisting the corrected increment is unconditional as of 2026-08-16, so
# there is no switch to turn it off. What that arm measured is recorded in
# docs/memo/dinov3_correct_low_precision_status.md -- rounds used to cost 12.6pp of recovery between
# G=4 and G=8, and now cost ~0.
set -u
cd /NHNHOME/share/cjpark/AppCorr
NR=${NR:-20}
TAG=${TAG:-ile}
for c in co3d_full co3d_approx_only co3d_appcorr co3d_interleaved co3d_il_frame4 co3d_il_spatial8 co3d_il_spatial4; do
  ps -eo pid,cmd | grep -E "offload/(server|mobile)/main" | grep -v grep | awk '{print $1}' | xargs -r kill -9 2>/dev/null
  sleep 4
  echo "##### $c ($TAG) #####"
  CUDA_VISIBLE_DEVICES=${GPU:-0} PYTHONUNBUFFERED=1 APPCORR_RESULT_TIMEOUT=1800 timeout 3600 \
    ./offload/run_local.sh offload/config/co3d/$c.json -nr $NR -nw 0 \
    -d /NHNHOME/share/cjpark/data/co3dv2/extracted > "logs/vggt/ile_${TAG}_${c}.log" 2>&1
  grep -aoE "Final Summary: \{[^}]*\}|!!! \[Worker\] Pipeline Error \(Req [0-9]+\): .{0,80}" \
    "logs/vggt/ile_${TAG}_${c}.log" | tail -2
done
echo "INTERLEAVE_EVAL_DONE"
