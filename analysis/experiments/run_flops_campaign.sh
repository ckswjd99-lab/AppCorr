#!/usr/bin/env bash
# Critical-FLOPs campaign over every offload-driven model, at the two recompute rates and the
# ceiling.
#
# Only a handful of requests per arm is needed and that is not a shortcut: FLOPs are a deterministic
# function of the input shapes and the schedule, so once the shapes repeat the number stops moving.
# ImageNet and COCO have fixed canvases, so one request already pins them; the varying-resolution
# models get a few more.
#
# The keep knob is spelled differently per executor and there is no single override that works
# everywhere -- `token_keep_ratio` on the DINOv3 family and VGGT, `keep_rate` on Qwen2.5-VL, and
# OpenCLIP exposes only a threshold. Each row below therefore carries its own override, and the
# OpenCLIP rows are run at the config's own operating point with that noted rather than pretended
# to be 30/50%.
set -u
cd /NHNHOME/share/cjpark/AppCorr-flops
OUT=${OUT:-analysis/results/flops}
NR=${NR:-3}
mkdir -p "$OUT"
export CUDA_VISIBLE_DEVICES=${GPU:-0}
export APPCORR_FLOPS=1
# `run_local.sh` invokes a bare `python`, and the system interpreter here carries torch but NOT
# transformers -- which is why every DINOv3 row passed while OpenCLIP died at
# `from transformers import CLIPModel`. Put the project env first rather than editing the runner.
export PATH=/home/nxclab/anaconda3/envs/appcorr/bin:$PATH

run () {   # run <tag> <config> <nr> [--set ...]
  local tag=$1 cfg=$2 nr=$3; shift 3
  if [ -s "$OUT/$tag.json" ]; then echo "[skip ] $tag"; return; fi
  echo "[start] $tag  $(date +%H:%M:%S)"
  APPCORR_FLOPS_OUT="$OUT/$tag.json" timeout 2400 \
    bash offload/run_local.sh "$cfg" -nr "$nr" -nw 0 "$@" > "$OUT/$tag.log" 2>&1
  local rc=$?
  echo "[done ] $tag rc=$rc  $(grep -ao 'mean_critical=[0-9.]* GF' "$OUT/$tag.log" | tail -1)"
}

# ---- DINOv3: classification, detection, segmentation, depth ---------------------------------- #
for K in 0.25 0.30 0.50; do
  run "dinov3_imagenet_g4_k${K}"  offload/config/imnet/imnet_interleaved_g4.json          "$NR" --set appcorr_kwargs.token_keep_ratio=$K
  run "dinov3_ade20k_g4_k${K}"    offload/config/ade20k/ade20k_m2f_interleaved_static.json "$NR" --set appcorr_kwargs.token_keep_ratio=$K
  run "dinov3_nyu_g4_k${K}"       offload/config/nyu/nyu_interleaved_static.json           "$NR" --set appcorr_kwargs.token_keep_ratio=$K
  run "dinov3_coco_g4_k${K}"      offload/config/coco/coco_interleaved_static.json         "$NR" --set appcorr_kwargs.token_keep_ratio=$K
done
run "dinov3_imagenet_ceiling" offload/config/imnet/imnet_sequential.json           "$NR"
run "dinov3_ade20k_ceiling"   offload/config/ade20k/ade20k_m2f_sequential.json     "$NR"
run "dinov3_nyu_ceiling"      offload/config/nyu/nyu_sequential.json               "$NR"
run "dinov3_coco_ceiling"     offload/config/coco/coco_sequential.json             "$NR"

# ---- VGGT-Omega: Co3D -------------------------------------------------------------------------- #
for K in 0.25 0.30 0.50; do
  run "vggt_co3d_g4_k${K}" offload/config/co3d/co3d_interleaved.json "$NR" \
      --set appcorr_kwargs.token_keep_ratio=$K --set transmission_kwargs.num_groups=4
done
run "vggt_co3d_ceiling" offload/config/co3d/co3d_full.json "$NR"

# ---- Qwen2.5-VL ------------------------------------------------------------------------------- #
# Measured in-process by flops_report_qwen25vl.py, not here. Its offload configs name dataset
# "realworldqa", for which `offload/mobile/dataset.get_dataset_loader` has no loader at all, so
# every run aborts at handshake with "Unknown dataset name" before a single batch moves.

# ---- OpenCLIP: ImageNet zero-shot and COCO retrieval ------------------------------------------- #
# The keep-rate arms now live in run_flops_openclip_topk.sh, since `_prune_patch_idx` learned
# `token_keep_ratio`. What stays here is the ceiling for each task, plus the config's own
# threshold operating point kept for comparison against the exact rates.
run "openclip_imagenet_g4_thres" offload/config/imagenet_clip_bigg_interleaved_g4.json      "$NR"
run "openclip_imagenet_ceiling" offload/config/imagenet_clip_bigg_sequential.json           "$NR"
run "openclip_cocoret_g4_thres" offload/config/coco_retrieval_clip_bigg_interleaved_g4.json "$NR"
run "openclip_cocoret_ceiling"  offload/config/coco_retrieval_clip_bigg_sequential.json     "$NR"

echo "FLOPS CAMPAIGN COMPLETE $(date)"
