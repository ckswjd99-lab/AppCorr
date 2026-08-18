#!/usr/bin/env bash
# Block until the CLIP closed-loop sweep finishes, the driver dies, or MAXWAIT seconds elapse.
set -u
LOGDIR=/NHNHOME/share/cjpark/AppCorr-clip-closedloop/logs/clip_closedloop
L="$LOGDIR/sweep_driver.log"
MAXWAIT=${MAXWAIT:-570}
end=$((SECONDS + MAXWAIT))

while true; do
  if grep -qa CLIP_CLOSEDLOOP_ALL_DONE "$L" 2>/dev/null; then echo "STATUS: ALL_DONE"; break; fi
  if ! pgrep -f clip_closedloop_remeasure >/dev/null 2>&1; then echo "STATUS: DRIVER_GONE (no ALL_DONE marker)"; break; fi
  if [ "$SECONDS" -ge "$end" ]; then echo "STATUS: STILL_RUNNING (wait window elapsed)"; break; fi
  sleep 30
done

echo "--- arm markers ---"
grep -aE '^#####|ARM_DONE' "$L" 2>/dev/null
echo "--- newest arm log tail ---"
newest=$(ls -t "$LOGDIR"/imnet_*.log "$LOGDIR"/coco_*.log 2>/dev/null | head -1)
[ -n "$newest" ] && echo "($(basename "$newest"))" && tail -2 "$newest"
