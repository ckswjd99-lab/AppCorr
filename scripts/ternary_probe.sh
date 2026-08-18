#!/usr/bin/env bash
# Can ternary break the L0 baseline while AppCorr's correction survives it?
#
# Every row runs at the SAME -nr. The full-dataset numbers already on record (L0 bf16 62.236, L0 fp4
# 62.2208, correct bf16 61.846, correct fp4 61.814) are NOT comparable to a 50-image slice and must
# not be mixed into this table -- the point of re-running the reference rows here is to have
# same-slice numbers to read the ternary rows against.
#
# L0 rows use ade20k_m2f_sequential*.json  (transmit the image, no correction).
# AppCorr rows use ade20k_m2f_interleaved_static.json -- the `token_keep_thres: 4e-5` config, which
# is the better operating point (41.27% recompute, 61.846) than the topk55 one (54.97%, 61.4828).
#
# The two fp4 correction rows are deliberately both present: `fp4_kernel` is the real _scaled_mm
# path, `fp4_emul` is the same format through the fake-quant harness the ternary row uses. The gap
# between them is the emulator's own offset, and without it a ternary-vs-fp4 difference cannot be
# separated from a harness-vs-kernel difference.
#
# Accuracy only. The emulated rows are slower than what they emulate; do not time any of this.
set -u
cd /NHNHOME/share/cjpark/AppCorr

NR="${NR:-50}"
L0=offload/config/ade20k/ade20k_m2f_sequential.json
L0FP4=offload/config/ade20k/ade20k_m2f_sequential_fp4.json
APP=offload/config/ade20k/ade20k_m2f_interleaved_static.json

run() {
  local tag=$1 cfg=$2; shift 2
  echo "##### $tag $(date +%H:%M) #####"
  # A CUDA device-side assert leaves the server holding the port, and run_local.sh then aborts
  # before starting while the log still contains no traceback -- which reads as a clean run with an
  # empty result. Clear stragglers on these ports first.
  for pid in $(lsof -nP -tiTCP:39972 -tiTCP:39973 -sTCP:LISTEN 2>/dev/null); do kill -9 "$pid" 2>/dev/null; done
  CUDA_VISIBLE_DEVICES=0 RECV_PORT=39972 SEND_PORT=39973 \
  PYTHONUNBUFFERED=1 APPCORR_RESULT_TIMEOUT=1800 timeout 5400 \
    ./offload/run_local.sh "$cfg" -nr "$NR" -nw 0 --set device=cuda:0 "$@" \
    > "logs/vggt/tern_${tag}.log" 2>&1
  sleep 15
}

run l0_bf16      "$L0"
run l0_fp4       "$L0FP4"
run l0_ternary   "$L0FP4" --set approx_quant_format=ternary
run cor_bf16     "$APP"
run cor_fp4_kern "$APP" --set correct_precision=fp4
run cor_fp4_emul "$APP" --set correct_delta_split=quant_full --set correct_quant_format=fp4
run cor_ternary  "$APP" --set correct_delta_split=quant_full --set correct_quant_format=ternary

echo "===== ADE20K, nr=${NR} (same slice for every row) ====="
printf "%-14s %-22s %-8s %s\n" ARM MIOU ERRORS IMPL
for tag in l0_bf16 l0_fp4 l0_ternary cor_bf16 cor_fp4_kern cor_fp4_emul cor_ternary; do
  log="logs/vggt/tern_${tag}.log"
  printf "%-14s %-22s %-8s %s\n" "$tag" \
    "$(grep -aoE "'mIoU': [0-9.]+" "$log" | tail -1 | grep -oE '[0-9.]+')" \
    "$(grep -acE 'Traceback|Pipeline Error' "$log")" \
    "$(grep -aoE '\[(FP4|delta-split)[^]]*\][^.]{0,70}' "$log" | tail -1)"
done
# A row that produced no mIoU did not run; say so rather than leaving a blank to be read as zero.
echo "TERNARY_PROBE_DONE"
