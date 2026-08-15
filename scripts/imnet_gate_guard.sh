#!/usr/bin/env bash
# Backstop for the imnet gate inside persist_refresh_gpu0.sh, whose metric regex was written for
# 'acc'/'top1' but imnet reports 'top1_acc'. Both captures come back empty, the equality test is
# skipped, and the gate fails *open* -- the full 50k sweep runs whether or not the flag is wired.
#
# This watches the two gate logs, compares the real key, and stops the queue if they match. Kills by
# PID from a bracketed `ps` match rather than `pkill -f`, which matches its own command line.
set -u
cd /NHNHOME/share/cjpark/AppCorr
F=logs/vggt/refresh_plumb_imnet_false.log
T=logs/vggt/refresh_plumb_imnet_true.log
until grep -qa "top1_acc" "$F" 2>/dev/null && grep -qa "top1_acc" "$T" 2>/dev/null; do sleep 15; done
a=$(grep -aoE "'top1_acc': [0-9.]+" "$F" | tail -1)
b=$(grep -aoE "'top1_acc': [0-9.]+" "$T" | tail -1)
echo "IMNET_GUARD false=[$a] true=[$b]"
if [ "$a" = "$b" ]; then
  echo "IMNET_GUARD_STOP identical arms -- killing the queue before the full sweep"
  ps -eo pid,cmd | grep '[p]ersist_refresh_gpu0' | awk '{print $1}' | xargs -r kill -9 2>/dev/null
  sleep 2
  ps -eo pid,cmd | grep '[-]-recv-port 39990' | awk '{print $1}' | xargs -r kill -9 2>/dev/null
  echo "IMNET_GUARD_DONE stopped"
else
  echo "IMNET_GUARD_OK arms differ -- letting the full sweep proceed"
fi
