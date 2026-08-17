#!/usr/bin/env bash
# Periodic progress + liveness report for whatever sweeps are running under logs/vggt/.
#
# Earlier versions mapped GPUs to hard-coded filename globs, and that was wrong twice in one night:
# a new log name (`plumb_ade_*`, then `trace_*`) fell outside the glob and the heartbeat reported a
# *killed* run as the live job -- worse than no heartbeat, because it looks like everything is fine.
#
# So: do not guess from names. A log is live if it was written recently, full stop. Anything modified
# within FRESH seconds is reported; nothing recent means nothing is running, which is itself the
# thing worth saying out loud.
set -u
cd /NHNHOME/share/cjpark/AppCorr
INTERVAL=${INTERVAL:-900}
FRESH=${FRESH:-180}

# tqdm position if there is one; otherwise say what stage the run is at, so "still loading the 7B
# backbone" cannot be mistaken for "idle".
pos() {
  local p
  p=$(tail -c 4000 "$1" 2>/dev/null | tr '\r' '\n' \
      | grep -aoE "[0-9]+/[0-9]+ \[[0-9:]+<[0-9:]+" | tail -1)
  if [ -n "$p" ]; then echo "$p"
  elif grep -aq "Final Summary" "$1" 2>/dev/null; then echo "(finished)"
  elif grep -aqE "Loading|mmap=True" "$1" 2>/dev/null; then echo "(loading)"
  else echo "(starting)"; fi
}

while true; do
  sleep "$INTERVAL"
  # `find -newermt` on mtime, not size: tqdm rewrites the same line with \r, so a log that is very
  # much alive can stop growing. A slow tail (1.8 s/it near the end of a run) was enough to make this
  # report IDLE while 1825/2000 images were still going. Also count live client processes, so a
  # quiet-but-running job cannot be announced as nothing.
  live=$(find logs/vggt -maxdepth 1 -name '*.log' -newermt "-${FRESH} seconds" 2>/dev/null | sort)
  procs_now=$(ps -eo cmd | grep -c '[o]ffload/mobile/main')
  if [ -z "$live" ] && [ "$procs_now" -gt 0 ]; then
    echo "HEARTBEAT QUIET: $procs_now client process(es) alive but no log touched in ${FRESH}s"
  elif [ -z "$live" ]; then
    echo "HEARTBEAT IDLE: no log in logs/vggt written in the last ${FRESH}s, no client processes"
  else
    msg=""
    for L in $live; do
      msg="${msg}$(basename "$L" .log) $(pos "$L") | "
    done
    gpu=$(nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader 2>/dev/null \
          | tr '\n' ';' | tr -d ' ')
    echo "HEARTBEAT ${msg}gpu=[${gpu}] mobile_procs=$(ps -eo cmd | grep -c '[o]ffload/mobile/main')"
  fi
done
