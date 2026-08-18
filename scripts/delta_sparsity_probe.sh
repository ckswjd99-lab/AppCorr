#!/usr/bin/env bash
# 2:4 sparsity on the correction delta, same 50-image ADE20K slice as the earlier rows.
#
# Offline this measured 0.0462 output error on the delta against 0.1036 applying the same 50% keep to
# `a+d` (2.24x). 2:4 is the structured form Blackwell tensor cores accelerate 2x; `unstructured50` is
# the freer variant, so the gap between them is what the hardware structure costs.
#
# fmt=none isolates sparsity from quantization. The fp4 and ternary rows compose the two.
#
# Reference rows on this slice: floor 45.349 | L0 bf16 54.160 | correct bf16 54.591
#   delta fp4 53.692 | delta ternary 48.519 | ternary via recompute 14.109 (below floor)
set -u
cd /NHNHOME/share/cjpark/AppCorr
APP=offload/config/ade20k/ade20k_m2f_interleaved_static.json
NR="${NR:-50}"

run() {
  local tag=$1 fmt=$2 sp=$3
  echo "##### $tag $(date +%H:%M) #####"
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 "$p" 2>/dev/null; done
  sleep 5
  CUDA_VISIBLE_DEVICES=0 RECV_PORT=39964 SEND_PORT=39965 \
  PYTHONUNBUFFERED=1 APPCORR_RESULT_TIMEOUT=3600 timeout 7200 \
    ./offload/run_local.sh "$APP" -nr "$NR" -nw 0 --set device=cuda:0 \
    --set correct_delta_split=quant_delta --set "correct_quant_format=$fmt" \
    --set "correct_delta_sparsity=$sp" \
    > "logs/vggt/sp_${tag}.log" 2>&1
  sleep 15
}

run none_24     none    2:4
run none_un50   none    unstructured50
run fp4_24      fp4     2:4
run ternary_24  ternary 2:4

echo "===== delta sparsity, nr=${NR} ====="
for tag in none_24 none_un50 fp4_24 ternary_24; do
  log="logs/vggt/sp_${tag}.log"
  printf "%-12s mIoU=%-20s errors=%s  bail=%s\n" "$tag" \
    "$(grep -aoE "'mIoU': [0-9.]+" "$log" | tail -1 | grep -oE '[0-9.]+')" \
    "$(grep -acE 'Traceback|Pipeline Error' "$log")" \
    "$(grep -ac 'BAIL\|no cached base' "$log")"
done
echo "SPARSITY_PROBE_DONE"
