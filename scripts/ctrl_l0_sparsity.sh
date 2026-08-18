#!/usr/bin/env bash
# Control arm: 2:4 sparsity on the RAW L0 activation, against the same 50-image slice whose dense
# reference is 54.16012692554635. This is the comparison that makes the delta claim: 2:4 on the
# correction delta cost 0.284 mIoU on full-2000 (61.846 -> 61.562); this measures what the same
# structure costs when there is no approximate base to fall back on.
set -u
cd /NHNHOME/share/cjpark/AppCorr
CUDA_VISIBLE_DEVICES=0 RECV_PORT=39958 SEND_PORT=39959 PYTHONUNBUFFERED=1 \
APPCORR_RESULT_TIMEOUT=3600 timeout 7200 \
  ./offload/run_local.sh offload/config/ade20k/ade20k_m2f_sequential.json \
  -nr 50 -nw 0 --set device=cuda:0 \
  --set approx_quant_format=none --set approx_act_sparsity=2:4 \
  > logs/vggt/ctrl_l0_bf16_24.log 2>&1
echo "mIoU   : $(grep -aoE "'mIoU': [0-9.]+" logs/vggt/ctrl_l0_bf16_24.log | tail -1 | grep -oE '[0-9.]+')"
echo "errors : $(grep -acE 'Traceback|Pipeline Error' logs/vggt/ctrl_l0_bf16_24.log)"
echo "impl   : $(grep -aoE '\[FP4\] Prepared.{0,100}' logs/vggt/ctrl_l0_bf16_24.log | tail -1)"
echo "CTRL2_DONE"
