#!/usr/bin/env bash
# imnet interleaved g4, both arms, full ImageNet val.
#
# No metric gate here: top-1 on the 128-image gate was identical between arms, but that is a property
# of the metric (5 errors out of 128 -- far too coarse to move), not evidence about the wiring. The
# wiring was settled positively instead, by APPCORR_PERSIST_TRACE showing the block actually writing
# `blocks_out_sum` (tag=layer0, rows=2208). Gate on a trace, not on a blunt accuracy number.
set -u
cd /NHNHOME/share/cjpark/AppCorr
IM=offload/config/imnet/imnet_interleaved_g4.json
for p in false true; do
  ps -eo pid,cmd | grep '[-]-recv-port 39990' | awk '{print $1}' | xargs -r kill -9 2>/dev/null
  sleep 5
  echo "##### imnet_g4 persist=$p @ $(date +%H:%M) #####"
  CUDA_VISIBLE_DEVICES=0 RECV_PORT=39990 SEND_PORT=39991 \
  PYTHONUNBUFFERED=1 APPCORR_RESULT_TIMEOUT=3600 \
    ./offload/run_local.sh "$IM" -nw 0 --set device=cuda:0 \
    --set "appcorr_kwargs.persist_correction_residual=$p" \
    > "logs/vggt/refresh_imnet_g4_$p.log" 2>&1
  grep -aoE "Final Summary: \{[^}]*\}|!!! \[Worker\] Pipeline Error \(Req [0-9]+\): .{0,80}" \
    "logs/vggt/refresh_imnet_g4_$p.log" | tail -2
done
echo "IMNET_DONE"
