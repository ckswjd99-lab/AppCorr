# NORA progressive-prefill experiments

These experiments target the causal Qwen2.5-VL multimodal prefill in NORA-long.
FAST+ action token generation and decoding remain stock.

## Environment

The validated environment uses Python 3.10, PyTorch 2.11, Transformers 4.50.0,
`qwen-vl-utils`, LIBERO, and MuJoCo/EGL. The model checkpoint is
`declare-lab/nora-long-finetuned-libero-spatial`. Do not upgrade Transformers
without rerunning the parity gate because the implementation calls Qwen's
version-sensitive cache and vision internals.

`nora_libero_runtime.py` discovers the sibling `openvla_deps/LIBERO` checkout
and supplies the B200 host's required EGL defaults. Override `LIBERO_ROOT`,
`MUJOCO_EGL_DEVICE_ID`, or other environment variables explicitly on a
different host.

## Required parity gate

```bash
PYTHONPATH="$PWD" conda run -n nora \
  python analysis/experiments/nora_prefill_parity.py --device cuda:0
```

This checks stock vision against the wrapper, one-shot 100% vision correction,
and stock versus four-group causal prefill on one real frame from every
LIBERO-Spatial task. Rollouts should not be launched after a failure.

## T1 rollout

The T1 scale is ten LIBERO-Spatial tasks, 25 initial states, and three
schedules (`stock`, `pipelined`, `approx`). Results are appended after every
episode and resume by `(schedule, task_id, init_state_idx)`.

```bash
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 conda run -n nora \
  python analysis/experiments/nora_libero_eval.py \
    --task-ids 0,1,2,3,4 --num-trials 25 \
    --result-jsonl logs/nora_t1/gpu0.jsonl

PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=1 conda run -n nora \
  python analysis/experiments/nora_libero_eval.py \
    --task-ids 5,6,7,8,9 --num-trials 25 \
    --result-jsonl logs/nora_t1/gpu1.jsonl
```

The 224x224 image produces 256 raw ViT tokens and 64 merged Qwen tokens. Four
groups therefore append 16 contiguous raster tokens each. Spatial quadrants
must not be used: they are non-contiguous in the causal LLM sequence.
