#!/usr/bin/env bash
# GPU 1 half of the persist re-measurement: NYU then COCO, both arms.
#
# The earlier NYU pair is discarded -- it ran before the flag was plumbed into the DINOv3 executors,
# so both arms were effectively persist=off and came out identical to 17 digits.
#
# COCO is gated on a short A/B first. Tonight produced three separate wiring faults (flag plumbed
# only into VGGT; m2f bypassing the shared entry point; the m2f batch cache scattering `_kv` but not
# `_blocks_out_sum`), each of which looked exactly like "the fix does nothing". COCO's detector has
# its own machinery and gets the same treatment: no full run without a demonstrated difference.
set -u
cd /NHNHOME/share/cjpark/AppCorr
RECV=39996; SEND=39997

kill_port() { ps -eo pid,cmd | grep "[-]-recv-port $RECV" | awk '{print $1}' | xargs -r kill -9 2>/dev/null; }

run() {  # cfg persist tag nr
  kill_port; sleep 5
  echo "##### $3 persist=$2 nr=${4:-full} @ $(date +%H:%M) #####"
  local extra=(); [ -n "${4:-}" ] && extra=(-nr "$4")
  CUDA_VISIBLE_DEVICES=1 RECV_PORT=$RECV SEND_PORT=$SEND \
  PYTHONUNBUFFERED=1 APPCORR_RESULT_TIMEOUT=3600 \
    ./offload/run_local.sh "$1" -nw 0 "${extra[@]}" \
    --set device=cuda:0 --set "appcorr_kwargs.persist_correction_residual=$2" \
    > "logs/vggt/refresh_$3_$2.log" 2>&1
  grep -aoE "Final Summary: \{[^}]*\}|!!! \[Worker\] Pipeline Error \(Req [0-9]+\): .{0,80}" \
    "logs/vggt/refresh_$3_$2.log" | tail -2
}

NYU=offload/config/nyu/nyu_interleaved_static.json
run "$NYU" false nyu_il
run "$NYU" true  nyu_il
echo "NYU_ARMS_DONE"

COCO=offload/config/coco/coco_interleaved_static.json
run "$COCO" false plumb_coco 8
run "$COCO" true  plumb_coco 8
a=$(grep -aoE "'(mAP|bbox_mAP|AP)': [0-9.]+" logs/vggt/refresh_plumb_coco_false.log | tail -1)
b=$(grep -aoE "'(mAP|bbox_mAP|AP)': [0-9.]+" logs/vggt/refresh_plumb_coco_true.log | tail -1)
echo "COCO_PLUMB false=[$a] true=[$b]"
if [ -n "$a" ] && [ "$a" = "$b" ]; then
  echo "COCO_PLUMB_FAILED identical arms -- not running the full sweep, wiring is still broken"
  echo "GPU1_REFRESH_DONE"; exit 0
fi
run "$COCO" false coco_il
run "$COCO" true  coco_il
echo "GPU1_REFRESH_DONE"
