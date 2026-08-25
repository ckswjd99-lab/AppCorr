#!/usr/bin/env bash
# Wait for the VFM campaign to finish, then run the OpenCLIP arms on the same GPU.
#
# Both halves live in THIS one script on purpose. Writing the wait as
#   nohup bash -c 'while pgrep -f run_vfm_accuracy_campaign.sh; do sleep 30; done; exec next.sh' &
# puts the literal string `run_vfm_accuracy_campaign.sh` into this waiter's own argv, so `pgrep -f`
# matches the waiter forever and the second half never runs -- silently. Wait on the marker the
# first script prints instead. (CLAUDE.md, "Always arm a completion trigger".)
set -u
cd /NHNHOME/share/cjpark/AppCorr-flops
VFM_LOG=analysis/results/vfm_accuracy_campaign.log

echo "[chain] waiting for VFM campaign to finish  $(date +%F' '%H:%M:%S)"
# Cover failure as well as success: if the campaign dies, its shell exits and the marker never
# lands, so also stop waiting once no campaign process is alive.
while true; do
  grep -qa "VFM_ACCURACY_CAMPAIGN_COMPLETE" "$VFM_LOG" && { echo "[chain] VFM complete"; break; }
  if ! ps -p "${VFM_PID:-0}" > /dev/null 2>&1; then
    echo "[chain] VFM campaign process gone without its completion marker -- proceeding anyway"
    break
  fi
  sleep 60
done

# Bounds first: the campaign's "ours" arms are already on disk and uninterpretable until the floor
# and ceiling sit beside them, so those come before the new OpenCLIP arms.
bash analysis/experiments/run_vfm_bounds.sh
exec bash analysis/experiments/run_openclip_accuracy.sh
