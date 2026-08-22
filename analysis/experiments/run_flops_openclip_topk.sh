#!/usr/bin/env bash
# OpenCLIP at an exact keep RATE, now that `_prune_patch_idx` supports top-k.
#
# Split out from the main campaign because that campaign was already running when top-k landed, and
# editing a bash script a running shell is still reading is a good way to corrupt it. These carry
# their own tags so the earlier threshold-driven runs stay on disk for comparison.
set -u
cd /NHNHOME/share/cjpark/AppCorr-flops
OUT=${OUT:-analysis/results/flops}; mkdir -p "$OUT"
NR=${NR:-3}
export CUDA_VISIBLE_DEVICES=${GPU:-0} APPCORR_FLOPS=1
export PATH=/home/nxclab/anaconda3/envs/appcorr/bin:$PATH   # run_local.sh calls a bare `python` that lacks transformers
for K in 0.25 0.30 0.50; do
  for TASK in imagenet cocoret; do
    CFG=offload/config/imagenet_clip_bigg_interleaved_g4.json
    [ "$TASK" = "cocoret" ] && CFG=offload/config/coco_retrieval_clip_bigg_interleaved_g4.json
    TAG="openclip_${TASK}_g4_k${K}"
    [ -s "$OUT/$TAG.json" ] && { echo "[skip ] $TAG"; continue; }
    echo "[start] $TAG $(date +%H:%M:%S)"
    APPCORR_FLOPS_OUT="$OUT/$TAG.json" timeout 2400 \
      bash offload/run_local.sh "$CFG" -nr "$NR" -nw 0 \
        --set appcorr_kwargs.token_keep_ratio=$K > "$OUT/$TAG.log" 2>&1
    echo "[done ] $TAG rc=$?  $(grep -ao 'mean_critical=[0-9.]* GF' "$OUT/$TAG.log" | tail -1)"
  done
done
echo "OPENCLIP TOPK COMPLETE $(date)"
