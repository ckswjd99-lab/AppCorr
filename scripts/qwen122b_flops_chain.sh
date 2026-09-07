#!/usr/bin/env bash
set -u
cd /NHNHOME/share/cjpark/AppCorr-flops
export PATH=/home/nxclab/anaconda3/envs/appcorr/bin:$PATH
export CUDA_VISIBLE_DEVICES=0
until grep -qa "MIDNIGHT_QUEUE_V2_COMPLETE" analysis/results/midnight_queue_v2.log; do sleep 60; done
env -u HF_TOKEN python analysis/experiments/flops_report_qwen35.py \
  --model Qwen/Qwen3.5-122B-A10B-FP8 --samples 12 --groups 4 \
  --datasets chartqa realworldqa \
  --out-json analysis/results/flops/qwen122b_flops.json
echo "QWEN122B_FLOPS_COMPLETE rc=$?"
