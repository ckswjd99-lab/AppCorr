#!/usr/bin/env bash
# Full ADE20K (2000) : bf16 approximate pass + delta-propagation correction, 2:4 sparse, no
# quantization. Isolates what 2:4 sparsity on the delta costs, with nothing else moving.
#
# Reference rows already on record for this config (token_keep_thres 4e-5, 41.27% recompute):
#   floor  L2 approx only, no correction ....... 56.013
#   correct bf16, dense recompute ............... 61.846
#   ceiling  L0 full forward bf16 ............... 62.236
# 50-image preview of this exact arm: 52.182 against a 54.591 dense-bf16 reference on that slice.
#
# Emulated (fake-quant harness, masked dense GEMM -- no sparse kernel). Accuracy only, never time it.
set -u
cd /NHNHOME/share/cjpark/AppCorr
for p in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do kill -9 "$p" 2>/dev/null; done
sleep 5
CUDA_VISIBLE_DEVICES=0 RECV_PORT=39962 SEND_PORT=39963 \
PYTHONUNBUFFERED=1 APPCORR_RESULT_TIMEOUT=7200 timeout 28800 \
  ./offload/run_local.sh offload/config/ade20k/ade20k_m2f_interleaved_static.json \
  -nr 2000 -nw 0 --set device=cuda:0 \
  --set correct_delta_split=quant_delta --set correct_quant_format=none \
  --set correct_delta_sparsity=2:4 \
  > logs/vggt/full_delta24.log 2>&1
echo "result : $(grep -aoE "Final Summary: \{[^}]*\}" logs/vggt/full_delta24.log | tail -1)"
echo "errors : $(grep -acE 'Traceback|Pipeline Error' logs/vggt/full_delta24.log)"
echo "bail   : $(grep -ac 'BAIL\|no cached base' logs/vggt/full_delta24.log)"
echo "impl   : $(grep -aoE '\[delta-split\][^.]{0,90}' logs/vggt/full_delta24.log | head -1)"
echo "FULL_DELTA24_DONE"
