# Analysis Scripts

`analysis` contains standalone experiment scripts and runtime log tools that can be rerun without the mobile/server pipeline.

## Layout

- `analysis/experiments/`: experiment entrypoints
- `analysis/log_tools/`: log post-processing and visualization tools
- `analysis/shared/`: reusable helpers shared by analysis scripts

## SR Visual Comparison

`analysis/experiments/compare_sr_coco.py` saves image grids for:

- original
- `1/4 downsample`
- `bicubic x4`
- `realesr_general_x4v3`
- `realesrgan_x4plus`

and also saves residual grids against the original image.

Example:

```bash
python analysis/experiments/compare_sr_coco.py \
  --num-images 10 \
  --image-size 256 \
  --weights-dir ~/cjpark/weights/realesrgan
```

## Token Signal Comparison

`analysis/experiments/compare_token_signals.py` compares DINOv3 token signals between:

- `L0`: original image
- `L2`: `1/4 downsample` followed by `bicubic x4`
- `L2SR`: `1/4 downsample` followed by SR
  - `realesr_general_x4v3`
  - `realesrgan_x4plus`

All images are preprocessed with `ResizeLongestSide + center pad` to the target square resolution before low-resolution generation. Padding-only patch tokens are excluded from the signal metrics.

Supported datasets:

- `imagenet-1k`
- `coco`

Outputs:

- `config.json`
- `samples.json`
- `attention_per_image.csv`
- `attention_summary.csv`
- `ffn_per_image.csv`
- `ffn_summary.csv`
- `improvement_summary.csv`

Attention outputs use a single `all_tokens` scope, treating every token equally and comparing the full token-to-token attention structure.

FFN outputs include:

- `gate_score`
- `effective_gate_score`

FFN outputs also use a single `all_tokens` scope, without separate CLS or patch-only breakdowns.

Quick smoke example:

```bash
python analysis/experiments/compare_token_signals.py \
  --dataset imagenet-1k \
  --data-root ~/data/imagenet_val \
  --batch-size 2 \
  --max-batches 1 \
  --image-size 256 \
  --layers 0,10,20,30,39 \
  --weights-dir ~/cjpark/weights/realesrgan
```

COCO example:

```bash
python analysis/experiments/compare_token_signals.py \
  --dataset coco \
  --batch-size 2 \
  --max-batches 1 \
  --image-size 256 \
  --layers 0,10,20,30,39 \
  --weights-dir ~/cjpark/weights/realesrgan
```

Interpretation:

- lower `js_divergence` / `l1_mean` is closer to `L0`
- higher `pearson` / `spearman` / `topk_overlap` is closer to `L0`
- `improvement_summary.csv` reports how often each SR variant beats the `bicubic_x4` baseline

## Log Tools

- `analysis/log_tools/log_visualizer.py`: renders request timelines from `events.jsonl`
- `analysis/log_tools/simulate_exit.py`: simulates offline early-exit thresholds from logged head outputs
- `analysis/log_tools/coco_fiftyone_viewer.py`: launches a FiftyOne web GUI for COCO detection logs

COCO detection viewer example:

```bash
python analysis/log_tools/coco_fiftyone_viewer.py \
  logs/offload/coco_appcorr_20260416_020814
```

COCO runs now also export:

- `detections/coco_results.json`: COCO evaluation-format detections
- `detections/fiftyone_predictions.json`: per-image metadata and normalized boxes for the FiftyOne viewer
