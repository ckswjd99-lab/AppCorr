#!/usr/bin/env bash
# ADE20K re-measurement after the closed-loop transmission fix.
#
# Only the base+residual configs are affected. `ade20k_m2f_approx_only_l2` sends levels [2] with no
# residual and `ade20k_m2f_sequential` sends the image, so floor 56.013 and ceiling 62.236 stand and
# are not re-run. Before the fix the [2,0] round trip lost 1.85% relative L2 even with the whole
# residual transmitted, so every correction number measured on it carried that.
set -u
cd /NHNHOME/share/cjpark/AppCorr
GPU=${GPU:-0}; RECV=${RECV:-39990}
CFG=offload/config/ade20k/ade20k_m2f_interleaved_static.json
run() {  # tag  extra-set...
  local tag=$1; shift
  ps -eo pid,cmd | grep "[-]-recv-port $RECV" | awk '{print $1}' | xargs -r kill -9 2>/dev/null
  sleep 4
  echo "##### ade20k $tag @ $(date +%H:%M) #####"
  CUDA_VISIBLE_DEVICES=$GPU RECV_PORT=$RECV SEND_PORT=$((RECV+1)) \
  PYTHONUNBUFFERED=1 APPCORR_RESULT_TIMEOUT=3600 \
    ./offload/run_local.sh "$CFG" -nw 0 --set device=cuda:0 "$@" \
    > "logs/vggt/cl_ade20k_${tag}.log" 2>&1
  grep -aoE "Final Summary: \{[^}]*\}|!!! \[Worker\] Pipeline Error \(Req [0-9]+\): .{0,80}" \
    "logs/vggt/cl_ade20k_${tag}.log" | tail -2
}
run bf16
run fp4 --set correct_precision=fp4
echo "ADE20K_CLOSEDLOOP_DONE"
