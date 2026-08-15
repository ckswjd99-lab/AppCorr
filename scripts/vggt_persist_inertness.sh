#!/usr/bin/env bash
# Inertness check for the `persist_correction_residual` edit to appcorr/models/dinov3/layers/block.py.
#
# That directory is shared with every paper config, so the bar is digit-for-digit, not "close":
# nyu_appcorr must reproduce AbsRel 0.08895853948992444 (docs/memo/vggt_omega_status.md). The flag
# defaults off, so any difference means the default path was disturbed.
#
# The config pins cuda:1, where Triton dies -- see the correction-is-cuda:0-only note. Pick the GPU
# with CUDA_VISIBLE_DEVICES and always override the config to cuda:0.
#
# `-nr 5` is not optional (batch is 8, so 5 requests = 40 samples): the recorded 0.08895853948992444 is a **40-sample** run. Without it the
# client walks all 654 NYU samples and returns 0.0497, which looks like a regression and is not one.
#
# Separate port pair so this can run beside another sweep, and deliberately no `pkill` of
# server/mobile processes -- any pattern broad enough to catch this run's leftovers would take a
# concurrent sweep down with it.
set -u
cd /NHNHOME/share/cjpark/AppCorr
echo "##### nyu_appcorr inertness #####"
CUDA_VISIBLE_DEVICES=0 RECV_PORT=39994 SEND_PORT=39995 \
PYTHONUNBUFFERED=1 APPCORR_RESULT_TIMEOUT=1800 timeout 5400 \
  ./offload/run_local.sh offload/config/nyu/nyu_appcorr.json -nr 5 -nw 0 --set device=cuda:0 \
  > logs/vggt/inert_nyu_appcorr.log 2>&1
grep -aoE "Final Summary: \{[^}]*\}|!!! \[Worker\] Pipeline Error \(Req [0-9]+\): .{0,80}" \
  logs/vggt/inert_nyu_appcorr.log | tail -2
echo "expected AbsRel 0.08895853948992444"
echo "INERTNESS_DONE"
