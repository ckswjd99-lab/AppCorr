#!/usr/bin/env bash
# Block until pass 2 finishes, its driver dies, or MAXWAIT seconds elapse.
set -u
LOGDIR=/NHNHOME/share/cjpark/AppCorr-clip-closedloop/logs/clip_closedloop
L="$LOGDIR/sweep_driver2.log"
MAXWAIT=${MAXWAIT:-570}
end=$((SECONDS + MAXWAIT))

while true; do
  if grep -qa CLIP_CLOSEDLOOP_PASS2_DONE "$L" 2>/dev/null; then echo "STATUS: PASS2_DONE"; break; fi
  if ! pgrep -f clip_closedloop_remeasure2 >/dev/null 2>&1; then echo "STATUS: DRIVER_GONE (no PASS2_DONE marker)"; break; fi
  if [ "$SECONDS" -ge "$end" ]; then echo "STATUS: STILL_RUNNING"; break; fi
  sleep 30
done

echo "--- markers ---"
grep -aE '^#####|^COMPLETED|^INCOMPLETE' "$L" 2>/dev/null
echo "--- newest tail ---"
newest=$(ls -t "$LOGDIR"/*_rerun.log "$LOGDIR"/*_CONTROL.log 2>/dev/null | head -1)
[ -n "$newest" ] && echo "($(basename "$newest"))" && tail -2 "$newest"
