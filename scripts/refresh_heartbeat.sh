#!/usr/bin/env bash
# Periodic progress + stall report for the two persist re-measurement sweeps.
#
# The completion monitor alone is not enough: a wedged run emits nothing and looks exactly like a
# running one. This reports the tqdm position of whichever arm log each GPU is currently writing,
# and shouts if a log has not grown since the previous tick.
#
# Lives in scripts/ rather than /tmp because /tmp is mounted noexec here.
set -u
cd /NHNHOME/share/cjpark/AppCorr
INTERVAL=${INTERVAL:-900}
prev0=""; prev1=""
newest() { ls -t $@ 2>/dev/null | head -1; }
# Falls back to a word when there is no tqdm line yet: loading the 7B backbone takes minutes, and a
# blank position printed next to "ok" reads as if the run were idle when it is only still starting.
pos() {
  local p
  p=$(tail -c 4000 "$1" 2>/dev/null | tr '\r' '\n' \
      | grep -aoE "[0-9]+/[0-9]+ \[[0-9:]+<[0-9:]+" | tail -1)
  if [ -n "$p" ]; then echo "$p"
  elif grep -aqE "Loading|mmap=True" "$1" 2>/dev/null; then echo "(loading)"
  else echo "(no progress line)"; fi
}
while true; do
  sleep "$INTERVAL"
  # The plumbing-check logs belong in the GPU-1 set: while they were missing, the newest GPU-1 file
  # stayed a *killed* run's log, and the size check happily reported "ok" off a position that had
  # stopped moving. A heartbeat that can point at a dead run is worse than none.
  L0=$(newest logs/vggt/refresh_ade20k_*.log logs/vggt/refresh_imnet_*.log)
  # `plumb_*` rather than `plumb_nyu_*`: naming the checks individually meant the ADE20K one
  # (`plumb_ade_*`) fell outside the glob, and the heartbeat cheerfully reported a *finished* NYU
  # run as the live GPU-1 job. Every plumbing check runs on GPU 1, so match them all.
  L1=$(newest logs/vggt/refresh_nyu_il_*.log logs/vggt/refresh_coco_il_*.log \
              logs/vggt/plumb_*.log)
  msg=""
  for slot in 0 1; do
    if [ "$slot" = 0 ]; then L="$L0"; p="$prev0"; else L="$L1"; p="$prev1"; fi
    if [ -z "$L" ]; then msg="${msg}gpu${slot}: no log yet | "; continue; fi
    sz=$(wc -c < "$L")
    state="ok"
    [ "$sz" = "$p" ] && state="STALLED"
    msg="${msg}gpu${slot} ${state} $(basename "$L" .log) $(pos "$L") | "
    if [ "$slot" = 0 ]; then prev0="$sz"; else prev1="$sz"; fi
  done
  procs=$(ps -eo cmd | grep -c "[o]ffload/mobile/main")
  echo "HEARTBEAT ${msg}mobile_procs=${procs}"
done
