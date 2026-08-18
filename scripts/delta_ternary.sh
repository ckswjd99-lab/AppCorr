#!/usr/bin/env bash
# Delta propagation: Linear(a) + Linear(quant(d)), against the same 50-image slice as
# scripts/ternary_probe.sh. The base `a` stays BF16, so unlike `quant_full` -- which overwrites the
# approximate value with a fully-requantized recompute -- a useless delta degenerates to "no
# correction" rather than to garbage. That floor-preserving property is the claim under test.
#
# Reference rows already measured on this slice:
#   floor (no correction)            45.349
#   L0 bf16                          54.160
#   correct bf16                     54.591
#   correct fp4  (quant_full, emul)  54.158
#   correct ternary (quant_full)     14.109   <- below floor: recompute in a bad format destroys
set -u
cd /NHNHOME/share/cjpark/AppCorr
APP=offload/config/ade20k/ade20k_m2f_interleaved_static.json
NR="${NR:-50}"
for fmt in fp4 ternary; do
  echo "##### delta_$fmt $(date +%H:%M) #####"
  for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 "$p" 2>/dev/null; done
  sleep 5
  CUDA_VISIBLE_DEVICES=0 RECV_PORT=39966 SEND_PORT=39967 \
  PYTHONUNBUFFERED=1 APPCORR_RESULT_TIMEOUT=3600 timeout 7200 \
    ./offload/run_local.sh "$APP" -nr "$NR" -nw 0 --set device=cuda:0 \
    --set correct_delta_split=quant_delta --set "correct_quant_format=$fmt" \
    > "logs/vggt/delta_${fmt}.log" 2>&1
  sleep 15
done
echo "===== delta propagation, nr=${NR} ====="
for fmt in fp4 ternary; do
  log="logs/vggt/delta_${fmt}.log"
  printf "delta_%-8s mIoU=%-20s errors=%s  bail=%s\n" "$fmt" \
    "$(grep -aoE "'mIoU': [0-9.]+" "$log" | tail -1 | grep -oE '[0-9.]+')" \
    "$(grep -acE 'Traceback|Pipeline Error' "$log")" \
    "$(grep -ac 'BAIL\|no cached base' "$log")"
done
echo "DELTA_TERNARY_DONE"
