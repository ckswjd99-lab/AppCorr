# OpenVLA-OFT interleaved progressive prefill

The validated OFT pipeline preserves the model's causal prefix order:

```text
BOS
-> agentview low-resolution base
-> 4 x (64-patch vision correction -> 64-token LLM append)
-> wrist low-resolution base
-> 4 x (64-patch vision correction -> 64-token LLM append)
-> proprio + language causal prefill
-> stock 56-token bidirectional action block
-> stock L1 action head
```

Each camera has independent DINOv2 and SigLIP state under the cache tags
`dino0`/`siglip0` and `dino1`/`siglip1`. A group is four complete rows of the
16x16 patch grid. Every LLM prefix position is appended exactly once.

Results produced before `pipeline_impl=vision_llm_interleaved_v2` are not valid
measurements of this pipeline: the older implementation corrected all 256
patches before chunking only the LLM. The old `full` and `approx` baselines are
still valid, but its `pipelined` rows must not be used.

## Environment and parity

The launchers set the required B200 runtime defaults before importing LIBERO:
`MUJOCO_GL=egl`, `MUJOCO_EGL_DEVICE_ID=2`,
`MUJOCO_EGL_ALLOW_ANY_DEVICE=1`, `USE_TF=0`, and `USE_TORCH=1`. Omitting the
last two can segfault inside robosuite's `eglMakeCurrent`.

```bash
PYTHONPATH="$PWD" conda run -n openvla \
  python analysis/experiments/openvla_oft_prefill_parity.py
```

The gate compares exact and grouped-full actions on one real initial frame from
all ten LIBERO-Spatial tasks. BF16 full-sequence versus chunked SDPA can differ
slightly, so the action-space bound is 0.02. It also asserts the exact runtime
operation trace and the 64-patch/64-token group sizes.

## T1 progressive rerun

The existing 50-trial `full` and `approx` baselines remain reusable. Rerun only
the corrected progressive schedule, splitting initial states across two GPUs:

```bash
PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=0 conda run -n openvla \
  python analysis/experiments/openvla_oft_libero_eval.py \
    --schedules pipelined --trial-start 0 --num-trials 25 \
    --result-jsonl /tmp/openvla_oft_interleaved_t1/gpu0.jsonl

PYTHONPATH="$PWD" CUDA_VISIBLE_DEVICES=1 conda run -n openvla \
  python analysis/experiments/openvla_oft_libero_eval.py \
    --schedules pipelined --trial-start 25 --num-trials 25 \
    --result-jsonl /tmp/openvla_oft_interleaved_t1/gpu1.jsonl
```
