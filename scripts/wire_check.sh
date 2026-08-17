#!/usr/bin/env bash
# Does correct_fp4_per_tensor_scale actually flip the path? Small n: the question is whether the two
# arms differ at all and whether the announced state matches, not how much.
set -u
cd /NHNHOME/share/cjpark/AppCorr
CFG=offload/config/ade20k/ade20k_m2f_interleaved_static_correct_fp4_topk55.json
for arm in "off:" "on:--set correct_fp4_per_tensor_scale=true"; do
  tag=${arm%%:*}; extra=${arm#*:}
  ps -eo pid,cmd | grep '[-]-recv-port 39990' | awk '{print $1}' | xargs -r kill -9 2>/dev/null
  sleep 4
  CUDA_VISIBLE_DEVICES=0 RECV_PORT=39990 SEND_PORT=39991 PYTHONUNBUFFERED=1 \
  APPCORR_RESULT_TIMEOUT=1800 timeout 2400 \
    ./offload/run_local.sh "$CFG" -nr 8 -nw 0 --set device=cuda:0 $extra \
    > "logs/vggt/wire_$tag.log" 2>&1
  printf "  %-4s mIoU=%s  state=%s\n" "$tag" \
    "$(grep -aoE "'mIoU': [0-9.]+" logs/vggt/wire_$tag.log | tail -1)" \
    "$(grep -aoE 'per_tensor_scale=(True|False)' logs/vggt/wire_$tag.log | tail -1)"
done
echo "WIRE_DONE"
