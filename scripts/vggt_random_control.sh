#!/usr/bin/env bash
# Random-selection control at the same keep ratios as the pscore sweep.
#
# Reading: random recovery should track the keep ratio roughly linearly. If the real pscore is
# *below* that line it is mis-ranking; if random shows the same flat-then-jump shape, the ranking is
# not the problem and partial correction itself is failing on this architecture.
set -u
cd /NHNHOME/share/cjpark/AppCorr
for r in 0.05 0.10 0.20 0.40 0.70; do
  ps -eo pid,cmd | grep -E "offload/(server|mobile)/main" | grep -v grep | awk '{print $1}' | xargs -r kill -9 2>/dev/null
  sleep 4
  echo "##### random ratio=$r #####"
  # Keep the full log. Piping straight into grep discarded the traceback that explained a stall
  # here once already -- the run looked like a deadlock when it was a rejected config value.
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 APPCORR_RESULT_TIMEOUT=1800 timeout 3600 \
    ./offload/run_local.sh offload/config/co3d/co3d_appcorr.json -nr 20 -nw 0 \
    -d /NHNHOME/share/cjpark/data/co3dv2/extracted \
    --set appcorr_kwargs.token_keep_ratio=$r \
    --set appcorr_kwargs.server_pscore=random > "logs/vggt/random_r${r}.log" 2>&1
  grep -aoE "Final Summary: \{[^}]*\}|keep ratio during correction: [0-9.]+%|!!! \[Worker\][^:]{0,120}|ValueError: [^,]{0,100}" \
    "logs/vggt/random_r${r}.log" | tail -3
done
echo "RANDOM_CONTROL_DONE"
