#!/usr/bin/env bash
# Solo A/B for the `forward_raw` bias-fold bypass, on CORRECT_FORWARD latency only.
#
# The earlier 89.20 vs 103.91 ms pair was taken with three runs sharing the GPU and is not
# usable. Both arms here run alone and back to back, same config, same request count.
#
#   fold    current default -- forward_raw folds bias into _scaled_mm, no [M,N] epilogue
#   nofold  APPCORR_FP4_RAW_NO_FOLD=1 -- consumer applies scale+bias, i.e. the pre-bypass path
#
# -nr 200 because this measures a per-request stage mean, not accuracy; the accuracy verdict is
# already settled on full-2000 (61.4828 fold / 61.4208 pre-bypass, both fine).
#
# Result (2026-08-18): fold 87.24/84.18, nofold 85.28/86.86 ms. The 0.36 ms between arm means is
# inside the 1.6-3.1 ms between repeats of one arm -- no latency effect either way.
set -u
cd /NHNHOME/share/cjpark/AppCorr

CFG=offload/config/ade20k/ade20k_m2f_interleaved_static_correct_fp4_topk55.json
NR=200

run() {
  local tag=$1; shift
  echo "##### $tag #####"
  CUDA_VISIBLE_DEVICES=0 RECV_PORT=39982 SEND_PORT=39983 \
  PYTHONUNBUFFERED=1 APPCORR_RESULT_TIMEOUT=1800 "$@" timeout 5400 \
    ./offload/run_local.sh "$CFG" -nr "$NR" -nw 0 --set device=cuda:0 \
    > "logs/vggt/rawbp_${tag}.log" 2>&1
  # Wait out the dataloader workers run_local.sh leaves behind before the next arm starts.
  sleep 20
}

# Both orders, because one pair cannot separate the arms from warm-up/thermal drift -- and in the
# event it did not: fold won the reversed pair and lost the first one.
run fold
run nofold  env APPCORR_FP4_RAW_NO_FOLD=1
run nofold2 env APPCORR_FP4_RAW_NO_FOLD=1
run fold2

for tag in fold fold2 nofold nofold2; do
  echo "== $tag =="
  grep -aoE "\[FP4\] Prepared[^\"]{0,120}" "logs/vggt/rawbp_${tag}.log" | tail -1
  grep -aoE "CORRECT_FORWARD[^\\n]{0,90}" "logs/vggt/rawbp_${tag}.log" | tail -1
  grep -aoE "'mIoU': [0-9.]+" "logs/vggt/rawbp_${tag}.log" | tail -1
  echo "errors: $(grep -acE 'Traceback|Pipeline Error' "logs/vggt/rawbp_${tag}.log")"
done
echo "RAWBP_DONE"
