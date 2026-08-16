#!/usr/bin/env bash
# Re-run spatial G=8 over all 316 sequences after the empty-round fix.
#
# The first attempt died 26 minutes in on a frame whose patch grid is 5x6: spatial grouping bands by
# row, and 5 rows cannot fill 8 rounds, so `_residual_round` raised rather than deadlock. n=20 had no
# frame that small; n=310 does. The fix bands the flattened grid when there are too few rows.
#
# Waits on the heavy batch's own completion marker -- a log string, never `pgrep -f`, which matches
# the waiting shell's command line and hangs forever.
set -u
cd /NHNHOME/share/cjpark/AppCorr
until grep -qa "FULL316_heavy_DONE" logs/vggt/full316_gpu0.log 2>/dev/null; do sleep 30; done
exec bash scripts/vggt_full316.sh 0 39990 retry co3d_il_spatial8
