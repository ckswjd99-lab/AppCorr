#!/usr/bin/env bash
# Heartbeat for scripts/clip_closedloop_remeasure.sh. Emits on progress, on failure, and on stall.
set -u
LOGDIR=/NHNHOME/share/cjpark/AppCorr-clip-closedloop/logs/clip_closedloop
D="$LOGDIR/sweep_driver.log"
STATE=$(mktemp -d)
: > "$STATE/old"
last_size=-1
stall=0

while true; do
  sleep 600

  if [ -f "$D" ]; then
    grep -aE '^##### (START|END)|^ARM_DONE|CLIP_CLOSEDLOOP_ALL_DONE' "$D" | tail -3 > "$STATE/new"
    if ! cmp -s "$STATE/new" "$STATE/old"; then
      tail -1 "$STATE/new"
      cp "$STATE/new" "$STATE/old"
    fi
  fi

  bad=$(grep -alE 'Traceback|Main Loop Error|Model Executor not loaded|CUDA out of memory|already in use' "$LOGDIR"/imnet_*.log "$LOGDIR"/coco_*.log 2>/dev/null | tr '\n' ' ')
  if [ -n "$bad" ]; then
    echo "FAILURE signature in: $bad"
    grep -ahE 'Traceback|Main Loop Error|Model Executor not loaded|CUDA out of memory|already in use' "$LOGDIR"/imnet_*.log "$LOGDIR"/coco_*.log 2>/dev/null | tail -2
  fi

  newest=$(ls -t "$LOGDIR"/imnet_*.log "$LOGDIR"/coco_*.log 2>/dev/null | head -1)
  if [ -n "$newest" ]; then
    sz=$(stat -c %s "$newest" 2>/dev/null || echo 0)
    if [ "$sz" = "$last_size" ]; then
      stall=$((stall + 1))
      if [ "$stall" -ge 5 ]; then
        gpu=$(nvidia-smi -i 1 --query-gpu=utilization.gpu,memory.used --format=csv,noheader)
        echo "STALLED: $(basename "$newest") unchanged ~10min; gpu1=$gpu; driver_alive=$(pgrep -fc clip_closedloop_remeasure || echo 0)"
        stall=0
      fi
    else
      stall=0
      last_size=$sz
      echo "PROGRESS $(basename "$newest"): $(grep -aoE '\[[0-9]+/[0-9]+\].*' "$newest" | tail -1 | cut -c1-110)"
    fi
  fi

  if grep -qa CLIP_CLOSEDLOOP_ALL_DONE "$D" 2>/dev/null; then
    echo "ALL ARMS COMPLETE"
    break
  fi
done
