# Merging the Qwen2.5-VL offload branch: what it carries and how to run it

`experiment/qwen25vl-appcorr` was 79 commits and 128 behind main when this merge was prepared. This
records what survived the merge, what was verified, and one structural thing that is easy to get
wrong afterwards.

## Two parallel Qwen lines, not one

Three Qwen branches existed, all forked from `9429f70` and sharing no commits with each other:

| branch | content | conflicts vs main |
|---|---|---|
| `develop/qwen-vl-progressive-prefill` (36) | standalone prototype | — (contained in the next) |
| `experiment/qwen-reverse-order-correct` (38) | prototype + reverse-order LLM correction | **0** |
| `experiment/qwen25vl-appcorr` (79) | **offload pipeline integration** | 3 |

`reverse-order` fully contains `prefill` (36 of its 38 commits), so `prefill` never needs merging
separately. The two remaining lines add the *same* seven files under `appcorr/models/qwen25vl/`,
byte-identical, so they do not conflict with each other.

## The three conflicts were all benign

| file | what happened | resolution |
|---|---|---|
| `offload/server/model/__init__.py` | each branch registered its own executor at the same spot | keep both — `sam3`, `openclip`, `qwen25vl` now coexist |
| `offload/common/protocol.py` | two *unrelated* fields appended to one dataclass | keep both: `num_correction_groups` (SAM 3 / crop-cover) and `text_payload` (VLM prompt) |
| `offload/policies/transmission/laplacian.py` | identical code, two independently written comments | merged the two explanations |

## What is unique to this branch

`reverse-order` measures accuracy on a standalone prototype. This branch is the serving path, and
carries things nothing else has:

- `offload/server/model/qwen25vl_executor.py` — the executor, i.e. latency with transmission overlap
- six 32B/72B RealWorldQA configs (approx_only / interleaved_g4 / sequential)
- **keep-rate sweeps over the full 765-example set** at both scales — 32B crosses its full-resolution
  baseline at **50%** keep (+0.78pp), 72B only at 100%. The crossing point depends on model size.
- `qwen25vl_{vision,llm}_fork_unittest.py`, McNemar testing, a FLOPs estimator, pscore diagnostics

## Verified on the merged tree

- Every touched module imports; both conflicting protocol fields present; the registry dispatches all
  eleven models.
- **Vision fork unit test** (real Qwen2.5-VL-32B, two aspect ratios): `approx == stock` and
  `correct(all groups) == stock` both `0.000000`, layer-chunked approx == one-shot `0.000000`.
- **LLM fork unit test** (64 layers, N=1832): `approx == stock` and `correct(all positions) == stock`
  both `0.000000`, final-position next-token argmax matches.
- **End-to-end pipeline smoke**, `realworldqa_qwen25vl_32b_approx_only`, 10 samples: completes,
  60% accuracy, per-op timings collected (FULL_INFERENCE 580ms, LOAD_INPUT 223ms). This is the check
  that matters — an executor written against a 128-commit-old pipeline importing cleanly says
  nothing about whether it still runs.

## The configs are NOT `run_local.sh` entry points

`offload/mobile/dataset.py`'s `get_dataset_loader` knows `imagenet-1k`, `coco2017`, `ade20k`, `co3d`
and `nyu_depth` — **not `realworldqa`** — and this branch never touches `mobile/`. The six configs
are templates read by a standalone driver:

```bash
python analysis/experiments/realworldqa_offload_eval.py \
    --config offload/config/realworldqa_qwen25vl_32b_approx_only.json \
    --num-samples 10 --device cuda:0
# --full runs all 765; --keep-rate / --grouping-strategy / --num-groups override the config
```

This is deliberate, not an oversight. Qwen resizes every image differently via its own
`smart_resize`, so each request needs its own CONFIG message; that does not fit the shared dataset
loader, which assumes one fixed input shape for a whole run. Reaching for `run_local.sh` here fails
with an unknown dataset name and looks like a broken merge.

Weights are already cached under `/NHNHOME/huggingface/hub` for 3B / 7B / 32B / 72B.
