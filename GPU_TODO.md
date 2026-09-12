# GPU commands owed by the driver side (worktree `AppCorr-il-driver`, CPU-only session)

Everything below was written and CPU-gated here; these are the runs that need a GPU. Order is
dependency order: 1 needs only the tower, 2-4 need agent A's engine-side `StreamingLLM.correct`
(this worktree carries the stub that raises NotImplementedError).

## 1. G5 on the real tower (no engine, tower only)

The CPU form of this gate already passes bitwise for both families (`--tiny`, see
`analysis/results/vllm_stream/interleaved_axis_gate_tiny.json` and `..._q25_tiny.json`); this is
the same gate on real weights, bf16, and a real image.

```
cd /NHNHOME/share/cjpark/AppCorr-il-driver
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTHONPATH=$PWD \
python analysis/experiments/vllm_interleaved_axis_gate.py \
    --family qwen35 --model Qwen/Qwen3.5-35B-A3B --groups 4 --keeps 1.0 0.5 \
    --dataset realworldqa --load vision
# expect: keep=1.00 bitwise=True, approx_rows=True, structure/selection/coverage True -> G5_PASS
# writes analysis/results/vllm_stream/interleaved_axis_gate.json
```

## 2. First end-to-end interleaved sample (needs the engine side)

```
# server (appcorr-vllm env)
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
PYTHONPATH=/NHNHOME/share/cjpark/AppCorr-il-driver \
<appcorr-vllm python> -m appcorr.vllm_stream.server \
    --model Qwen/Qwen3.5-35B-A3B --port 5591 --gpu-mem 0.6

# driver (appcorr env)
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTHONPATH=$PWD \
python analysis/experiments/qwen_vllm_accuracy.py --family qwen35 \
    --model Qwen/Qwen3.5-35B-A3B --port 5591 --dataset realworldqa \
    --arms streaming --llm-schedule interleaved --groups 4 --samples 8 \
    --out analysis/results/qwen_vllm_accuracy_il
# rows land in realworldqa_qwen3.5-35b-a3b_interleaved_g4.jsonl and carry `chunks` (the list of
# ("approx", 0, N-1) / ("correct", s, e, |P_r|) records) plus `image_run`, which item 4 replays.
```

## 3. Latency probe with the interleaved schedule (needs the engine side)

```
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTHONPATH=$PWD \
python analysis/experiments/latency_probe.py --family qwen35 \
    --model Qwen/Qwen3.5-35B-A3B --port 5591 --key qwen35_35b_il \
    --llm-schedule interleaved --samples 40 --warmup 4 --keeps 1.0 0.50 \
    --push-delay-ms 150 --datasets realworldqa:box vstar:pyr
# Crit. Lat. for these rows is anchored at the LAST `correct` message's t_recv_server; the
# streaming-convention anchor (last band's pixel arrival) is reported beside it as
# detail.*.ttft_last_band_pixels_ms -- see the note in the report below.
```

## 4. Interleaved FLOPs keys folded into the report (needs item 2's rows)

Full form (re-measures this run's vision half with the hooks, then adds the closed-form decoder):

```
CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTHONPATH=$PWD \
python analysis/experiments/flops_report_qwen35.py --model Qwen/Qwen3.5-35B-A3B \
    --datasets realworldqa --samples 12 --groups 4 --keeps 1.0 0.50 \
    --il-rows analysis/results/qwen_vllm_accuracy_il --il-model qwen35_35b \
    --out-json analysis/results/flops/qwen35_flops_il.json
```

CPU-only form (decoder half only, no vision added -- runs anywhere, no GPU):

```
PYTHONPATH=$PWD python analysis/experiments/flops_report_qwen35.py \
    --model Qwen/Qwen3.5-35B-A3B --datasets realworldqa --il-only \
    --il-rows analysis/results/qwen_vllm_accuracy_il \
    --out-json analysis/results/flops/qwen35_flops_il.json
```

The closed form behind both is already reconciled against the hooks on CPU:
`python analysis/experiments/flops_analytic.py --validate-qwen35` -> worst 0.004% over
refcoco/chartqa/textvqa/realworldqa.
