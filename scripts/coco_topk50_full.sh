#!/usr/bin/env bash
# COCO window-based interleaved correction at a 50% recompute rate, full 5000 images.
#
# Selection is top-k (`token_keep_ratio: 0.5`) rather than the shipped threshold, so the recompute
# rate is exactly 50% by construction instead of an emergent property of `token_keep_thres`. The
# threshold has to be set to null explicitly -- leaving the config's 0.002 in place would keep the
# threshold path and silently ignore the ratio.
#
# Anchors already measured on the same 5000 images:
#   floor   `coco_approx_only_windowbase`  mAP 0.5583
#   default `token_keep_thres=0.002`, 20.6% recompute, mAP 0.6011  (58.6% of the gap)
#   ceiling `coco_sequential`              mAP 0.6314
#
# The n=100 calibration put 50% recompute at ~93% of the gap; this is the full-set confirmation.
set -u
cd /NHNHOME/share/cjpark/AppCorr
ps -eo pid,cmd | grep '[-]-recv-port 39990' | awk '{print $1}' | xargs -r kill -9 2>/dev/null
sleep 4
echo "##### coco topk50 full @ $(date +%H:%M) #####"
CUDA_VISIBLE_DEVICES=0 RECV_PORT=39990 SEND_PORT=39991 \
PYTHONUNBUFFERED=1 APPCORR_RESULT_TIMEOUT=3600 \
  ./offload/run_local.sh offload/config/coco/coco_interleaved_static.json -nw 0 --set device=cuda:0 \
  --set appcorr_kwargs.token_keep_thres=null --set appcorr_kwargs.token_keep_ratio=0.5 \
  > logs/vggt/coco_topk50_full.log 2>&1
tr '\r' '\n' < logs/vggt/coco_topk50_full.log | grep -aoE "keep ratio during correction: [0-9.]+%" | tail -1
grep -aoE "Final Summary: \{[^}]*\}|!!! \[Worker\] Pipeline Error \(Req [0-9]+\): .{0,80}" \
  logs/vggt/coco_topk50_full.log | tail -2
echo "COCO_TOPK50_DONE"
