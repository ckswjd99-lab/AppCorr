#!/usr/bin/env bash
# develop/fp4-no-per-tensor-scale: does the new default hold up, and where else does it apply?
#
#   ade_off / ade_on   ADE20K full-2000, correct_fp4_per_tensor_scale false/true. The off arm must
#                      still land on 61.4208 now that forward_raw is bypassed too -- the bypass is
#                      meant to cost time, not accuracy.
#   im_off  / im_on    ImageNet, correct_precision=fp4 forced on imnet_interleaved_g4 (no shipped
#                      imnet config uses correct fp4). One family is not a basis for a default.
#   l0_fp4             ade20k_m2f_sequential_fp4: raw transmission, precision=fp4, i.e. the plain L0
#                      forward run in FP4. Note this is the *approx* controller, which quantizes
#                      through torchao's config path, not FastFP4Linear -- the new switch does not
#                      reach it. Measures where FP4 lands on a full forward, nothing more.
set -u
cd /NHNHOME/share/cjpark/AppCorr
GPU=$1; RECV=$2; shift 2
run() {  # tag config extra...
  local tag=$1 cfg=$2; shift 2
  ps -eo pid,cmd | grep "[-]-recv-port $RECV" | awk '{print $1}' | xargs -r kill -9 2>/dev/null
  sleep 4
  echo "##### $tag @ $(date +%H:%M) #####"
  CUDA_VISIBLE_DEVICES=$GPU RECV_PORT=$RECV SEND_PORT=$((RECV + 1)) \
  PYTHONUNBUFFERED=1 APPCORR_RESULT_TIMEOUT=3600 \
    ./offload/run_local.sh "$cfg" -nw 0 --set device=cuda:0 "$@" \
    > "logs/vggt/pts_${tag}.log" 2>&1
  echo "  state: $(grep -aoE 'per_tensor_scale=(True|False)' logs/vggt/pts_${tag}.log | tail -1)"
  grep -aoE "Final Summary: \{[^}]*\}|!!! \[Worker\] Pipeline Error \(Req [0-9]+\): .{0,80}" \
    "logs/vggt/pts_${tag}.log" | tail -2
}
for spec in "$@"; do
  case "$spec" in
    ade_off) run ade_off offload/config/ade20k/ade20k_m2f_interleaved_static_correct_fp4_topk55.json ;;
    ade_on)  run ade_on  offload/config/ade20k/ade20k_m2f_interleaved_static_correct_fp4_topk55.json --set correct_fp4_per_tensor_scale=true ;;
    im_off)  run im_off  offload/config/imnet/imnet_interleaved_g4.json --set correct_precision=fp4 ;;
    im_on)   run im_on   offload/config/imnet/imnet_interleaved_g4.json --set correct_precision=fp4 --set correct_fp4_per_tensor_scale=true ;;
    l0_fp4)  run l0_fp4  offload/config/ade20k/ade20k_m2f_sequential_fp4.json ;;
  esac
done
echo "PTS_SUITE_DONE"
