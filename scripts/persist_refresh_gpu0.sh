#!/usr/bin/env bash
# GPU 0 half of the persist re-measurement, resumed after the plumbing was fixed.
#
# The `off` arm for ADE20K already completed under the old code, which is behaviourally identical to
# persist=false, so it is kept as the control and only the `on` arm is run here.
#
# Waits on a log marker rather than on a process name: a `pgrep -f <pattern>` waiter matches its own
# command line and never exits, which has already cost this project time more than once.
#
# imnet is gated on its own short A/B. Three separate wiring faults tonight each produced
# bit-identical arms with no error -- a config that has not demonstrated a difference on 8 requests
# has not earned an hour.
set -u
cd /NHNHOME/share/cjpark/AppCorr
CTRL=logs/vggt/refresh_ade20k_static_false.log
until grep -qa "Final Summary" "$CTRL" 2>/dev/null; do sleep 30; done
echo "##### control arm finished: $(grep -aoE "'mIoU': [0-9.]+" "$CTRL" | tail -1) #####"

kill_port() { ps -eo pid,cmd | grep "[-]-recv-port $1" | awk '{print $1}' | xargs -r kill -9 2>/dev/null; }

run() {  # cfg persist tag nr
  kill_port 39990; sleep 5
  echo "##### $3 persist=$2 nr=${4:-full} @ $(date +%H:%M) #####"
  local extra=(); [ -n "${4:-}" ] && extra=(-nr "$4")
  CUDA_VISIBLE_DEVICES=0 RECV_PORT=39990 SEND_PORT=39991 \
  PYTHONUNBUFFERED=1 APPCORR_RESULT_TIMEOUT=3600 \
    ./offload/run_local.sh "$1" -nw 0 "${extra[@]}" \
    --set device=cuda:0 --set "appcorr_kwargs.persist_correction_residual=$2" \
    > "logs/vggt/refresh_$3_$2.log" 2>&1
  grep -aoE "Final Summary: \{[^}]*\}|!!! \[Worker\] Pipeline Error \(Req [0-9]+\): .{0,80}" \
    "logs/vggt/refresh_$3_$2.log" | tail -2
}

run offload/config/ade20k/ade20k_m2f_interleaved_static.json true ade20k_static
echo "ADE20K_DONE"

# --- imnet: prove the flag moves the number before spending the full run on it ---
IM=offload/config/imnet/imnet_interleaved_g4.json
run "$IM" false plumb_imnet 4
run "$IM" true  plumb_imnet 4
a=$(grep -aoE "'acc(_top1|1|uracy)?': [0-9.]+" logs/vggt/refresh_plumb_imnet_false.log | tail -1)
b=$(grep -aoE "'acc(_top1|1|uracy)?': [0-9.]+" logs/vggt/refresh_plumb_imnet_true.log | tail -1)
echo "IMNET_PLUMB false=[$a] true=[$b]"
if [ -n "$a" ] && [ "$a" = "$b" ]; then
  echo "IMNET_PLUMB_FAILED identical arms -- not running the full sweep, wiring is still broken"
  echo "REFRESH_DONE"; exit 0
fi
run "$IM" false imnet_g4
run "$IM" true  imnet_g4
echo "REFRESH_DONE"
