#!/usr/bin/env bash
# The fp4 correction row of dinov3_correct_low_precision_status, re-measured on both sides of the
# persist fix. That row (61.191) is still a pre-fix number.
#
# Deliberately NOT `ade20k_m2f_interleaved_static_correct_fp4_topk55.json`: that config selects with
# `token_keep_ratio: 0.55` while the bf16 row selects with `token_keep_thres: 4e-5`, so the two are
# not matched on placement even though the memo's "FP4 effect at matched placement" table compares
# them. Using the base config with `--set correct_precision=fp4` keeps selection identical to the
# bf16 arms measured tonight (61.042 / 61.597), which is what makes the delta interpretable.
set -u
cd /NHNHOME/share/cjpark/AppCorr
CFG=offload/config/ade20k/ade20k_m2f_interleaved_static.json
for p in false true; do
  ps -eo pid,cmd | grep '[-]-recv-port 39990' | awk '{print $1}' | xargs -r kill -9 2>/dev/null
  sleep 5
  echo "##### ade20k fp4 persist=$p @ $(date +%H:%M) #####"
  CUDA_VISIBLE_DEVICES=0 RECV_PORT=39990 SEND_PORT=39991 \
  PYTHONUNBUFFERED=1 APPCORR_RESULT_TIMEOUT=3600 \
    ./offload/run_local.sh "$CFG" -nw 0 --set device=cuda:0 \
    --set correct_precision=fp4 \
    --set "appcorr_kwargs.persist_correction_residual=$p" \
    > "logs/vggt/refresh_ade_fp4_$p.log" 2>&1
  grep -aoE "Final Summary: \{[^}]*\}|!!! \[Worker\] Pipeline Error \(Req [0-9]+\): .{0,80}|correct_precision='fp4' is unavailable[^\"]{0,60}" \
    "logs/vggt/refresh_ade_fp4_$p.log" | tail -2
done
echo "ADE_FP4_DONE"
