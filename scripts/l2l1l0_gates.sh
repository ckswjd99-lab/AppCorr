#!/bin/bash
# L2-L1-L0 re-verification gates (run before any campaign).
# Usage: PATH="$HOME/appcorr-env/bin:$PATH" bash scripts/l2l1l0_gates.sh
set -uo pipefail
cd "$(dirname "$0")/.."
LOGDIR=analysis/results/logs
mkdir -p "$LOGDIR"

echo "=== Gate 0: weights present ==="
for f in ~/cjpark/weights/dinov3/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth \
         ~/cjpark/weights/dinov3/dinov3_vit7b16_ade20k_m2f_head-bf307cb1.pth; do
  [ -f "$f" ] || { echo "MISSING $f"; exit 1; }
done
ls -la ~/cjpark/weights/dinov3/dinov3_vit7b16_pretrain_lvd1689m-a955f4ea.pth

echo "=== Gate 1: unit tests ==="
python -m pytest tests/ -q || exit 1

echo "=== Gate 2: sequential smoke (2 requests) ==="
RECV_PORT=41100 SEND_PORT=41101 CUDA_VISIBLE_DEVICES=0 \
  offload/run_local.sh offload/config/ade20k/ade20k_m2f_sequential.json \
  -nr 2 -nw 1 --set device=cuda:0 --set exp_id=gate_seq_smoke \
  2>&1 | tee "$LOGDIR/gate_seq_smoke.log" | tail -5

echo "=== Gate 3: L2-L0 interleaved smoke + persist trace ==="
APPCORR_PERSIST_TRACE=1 RECV_PORT=41200 SEND_PORT=41201 CUDA_VISIBLE_DEVICES=0 \
  offload/run_local.sh offload/config/ade20k/ade20k_m2f_interleaved_static.json \
  -nr 2 -nw 1 --set device=cuda:0 --set exp_id=gate_l2l0_smoke \
  2>&1 | tee "$LOGDIR/gate_l2l0_smoke.log" | grep -E "\[persist\]|mIoU|Error|error" | head -5

echo "=== Gate 4: L2-L1-L0 static (disjoint-capable config) smoke + persist trace ==="
APPCORR_PERSIST_TRACE=1 RECV_PORT=41300 SEND_PORT=41301 CUDA_VISIBLE_DEVICES=0 \
  offload/run_local.sh offload/config/ade20k_m2f_interleaved_l2l1l0_static.json \
  -nr 2 -nw 1 --set device=cuda:0 --set exp_id=gate_l2l1l0_smoke \
  2>&1 | tee "$LOGDIR/gate_l2l1l0_smoke.log" | grep -E "\[persist\]|mIoU|Error|error" | head -5

echo "=== Gates complete; check the [persist] line appeared in gates 3 and 4 ==="
grep -l "\[persist\] blocks_out_sum write executed" \
  "$LOGDIR/gate_l2l0_smoke.log" "$LOGDIR/gate_l2l1l0_smoke.log"
