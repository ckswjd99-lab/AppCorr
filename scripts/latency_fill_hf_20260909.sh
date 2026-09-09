#!/usr/bin/env bash
# Fill the table's Lat. / Crit. Lat. cells for the HF in-process VLMs (2026-09-09), one model at
# a time on GPU0, after the vLLM-served Qwen3.5 probes. Each report script's --latency mode times
# the SAME arms its FLOPs mode counts (appcorr.latency: identical scopes, no hooks) and merges
# into analysis/results/latency/inprocess_latency.json.
set -u
cd /NHNHOME/share/cjpark/AppCorr-vllm
export PATH=/home/nxclab/anaconda3/envs/appcorr/bin:$PATH
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTHONPATH=/NHNHOME/share/cjpark/AppCorr-vllm
unset HF_TOKEN
L=logs/latency
N=${N:-20}
step () { echo "== $1 start $(date +%H:%M:%S)"; }
done_ () { echo "== $1 rc=$2 $(date +%H:%M:%S)"; }

step ov2
python analysis/experiments/flops_report_ov2.py --latency --samples $N --keeps 0.25 0.50 --streaming \
  --datasets chartqa infovqa textvqa docvqa realworldqa pope gqa mmmu refcoco vsr vstar \
  > $L/ov2.log 2>&1; done_ ov2 $?

step gemma3
python analysis/experiments/flops_report_gemma3.py --latency --samples $N --keeps 0.25 0.50 \
  --datasets chartqa infovqa textvqa pope realworldqa docvqa gqa mmmu vsr \
  > $L/gemma3.log 2>&1; done_ gemma3 $?

step qwen25vl_32b_interleaved
python analysis/experiments/flops_report_qwen25vl_arms.py --latency --samples $N --keeps 0.25 0.50 \
  --datasets refcoco gqa realworldqa mmvp cvbench vstar \
  > $L/qwen25vl_32b_interleaved.log 2>&1; done_ qwen25vl_32b_interleaved $?
step qwen25vl_32b_streaming
python analysis/experiments/flops_report_qwen25vl_arms.py --latency --samples $N --keeps 1.0 \
  --llm-schedule streaming --datasets refcoco gqa realworldqa mmvp cvbench vstar \
  > $L/qwen25vl_32b_streaming.log 2>&1; done_ qwen25vl_32b_streaming $?

step gemma4
python analysis/experiments/flops_report_gemma4.py --latency --samples $N --keeps 0.25 0.50 \
  --arms interleaved --datasets mmvp cvbench refcoco textvqa visdrone_count visdrone_det \
  > $L/gemma4.log 2>&1; done_ gemma4 $?

step mistral24b
for ds in mmvp cvbench refcoco textvqa visdrone_count visdrone_det vstar; do
  python analysis/experiments/mistral3_oracle.py --flops --latency --dataset $ds --num-samples $N \
    --lat-keeps 0.25 0.50 1.0 > $L/mistral24b_$ds.log 2>&1; echo "   mistral $ds rc=$?"
done; done_ mistral24b 0

step museglimmer30b
python analysis/experiments/flops_report_museglimmer.py --latency --samples $N --keeps 0.25 0.50 1.0 \
  --datasets mmvp cvbench vstar refcoco textvqa visdrone_count visdrone_det \
  > $L/museglimmer30b.log 2>&1; done_ museglimmer30b $?

echo "LATENCY_FILL_HF_COMPLETE $(date)"
