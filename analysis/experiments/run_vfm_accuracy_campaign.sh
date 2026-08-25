#!/usr/bin/env bash
# Full-dataset "ours" (interleaved g=4, keep 25/50%) accuracy for the four VFM/VFM-ish models that
# only had FLOPs measured so far: SAM3, VGGT-Omega, OpenCLIP, DINOv3.
#
# Deliberately CO-LOCATED on GPU0 alongside the running OV2/Gemma3 table campaign rather than
# queued behind it -- GPU1 is occupied by an unrelated job we must not touch, and GPU0 has ~170GB
# free next to Gemma3's ~10GB, so sharing costs some throughput to both but no correctness risk.
# Ordered fastest-first so results land as early as possible: SAM3 (COCO val, ~35-40 min per memo)
# < OpenCLIP (COCO retrieval is small; ImageNet val is 50k) < VGGT (Co3D, sequence count TBD)
# < DINOv3 (ImageNet val 50k is the single biggest run here).
#
# No new drivers needed -- every one of these reuses the exact config + --set overrides already
# validated by the FLOPs campaign (run_flops_campaign.sh / run_flops_openclip_topk.sh), just with
# -nr omitted so run_local.sh runs the full dataset instead of a handful of debug requests.
set -u
cd /NHNHOME/share/cjpark/AppCorr-flops
PY=/home/nxclab/anaconda3/envs/appcorr/bin/python
export CUDA_VISIBLE_DEVICES=0
export PATH=/home/nxclab/anaconda3/envs/appcorr/bin:$PATH
export HF_HUB_OFFLINE=1   # SAM3 (facebook/sam3) is gated; a local cache already exists
OUT=analysis/results/vfm_accuracy
CO3D_ROOT=/NHNHOME/share/cjpark/data/co3dv2/extracted
mkdir -p "$OUT"

run () {   # run <tag> <cmd...>
  local tag=$1; shift
  if [ -s "$OUT/$tag.log" ] && grep -qa "Final Summary" "$OUT/$tag.log"; then
    echo "[skip ] $tag"; return
  fi
  echo "[start] $tag  $(date +%F' '%H:%M:%S)"
  "$@" > "$OUT/$tag.log" 2>&1
  local rc=$?
  echo "[done ] $tag rc=$rc  $(date +%F' '%H:%M:%S)"
  grep -aoE "Final Summary: .*|=== Final Summary: .*" "$OUT/$tag.log" | tail -1
}

# ---- SAM3: corrected @ k0.25/0.50, pre_global bounds (validated cheaper+at-least-as-good),
#      groups=4, full COCO val2017 tracker path ------------------------------------------------- #
for K in 0.25 0.50; do
  run "sam3_coco_k${K}" \
    "$PY" analysis/experiments/sam3_coco_oracle.py --path tracker --full --max-boxes 20 \
      --arm corrected --keep-ratio "$K" --groups 4 --bounds pre_global \
      --out-json "$OUT/sam3_coco_k${K}.json"
done

# ---- OpenCLIP: DISABLED, its correction path is broken ---------------------------------------- #
# Both arms failed on their first real run and neither failure was visible from the exit code:
#   * cocoret died instantly with `ValueError: Unknown dataset name: coco_captions` -- the loader in
#     offload/mobile/dataset.py has no entry for it -- and still exited rc=0, so this script counted
#     it as done. (Same shape as the Qwen2.5-VL "realworldqa" loader gap.)
#   * imagenet ran the full 50k and returned top1 2.13% / top5 2.79%, i.e. below the 1000-class
#     random baseline's top5. Not a degradation, a broken forward.
# Being debugged separately -- see docs/memo/openclip_correction_broken.md. Do not re-enable until
# that memo says the floor/ceiling/g=1 identity gates pass.

# ---- VGGT-Omega: Co3D interleaved @ k0.25/0.50, full sequence set ------------------------------ #
for K in 0.25 0.50; do
  run "vggt_co3d_k${K}" \
    bash offload/run_local.sh offload/config/co3d/co3d_interleaved.json -nw 0 \
      --set dataset_kwargs.data_root="$CO3D_ROOT" \
      --set appcorr_kwargs.token_keep_thres=none --set appcorr_kwargs.token_keep_ratio="$K" \
      --set transmission_kwargs.num_groups=4
done

# ---- DINOv3: ImageNet / ADE20K / NYU / COCO @ k0.25/0.50, full dataset, fresh (the 2026-08-16
#      persist-bug fix and exact top-k both postdate every prior interleaved number for this) ---- #
for K in 0.25 0.50; do
  run "dinov3_imagenet_k${K}" \
    bash offload/run_local.sh offload/config/imnet/imnet_interleaved_g4.json -nw 0 \
      --set appcorr_kwargs.token_keep_thres=none --set appcorr_kwargs.token_keep_ratio="$K"
  run "dinov3_ade20k_k${K}" \
    bash offload/run_local.sh offload/config/ade20k/ade20k_m2f_interleaved_static.json -nw 0 \
      --set appcorr_kwargs.token_keep_thres=none --set appcorr_kwargs.token_keep_ratio="$K"
  run "dinov3_nyu_k${K}" \
    bash offload/run_local.sh offload/config/nyu/nyu_interleaved_static.json -nw 0 \
      --set appcorr_kwargs.token_keep_thres=none --set appcorr_kwargs.token_keep_ratio="$K"
  run "dinov3_coco_k${K}" \
    bash offload/run_local.sh offload/config/coco/coco_interleaved_static.json -nw 0 \
      --set appcorr_kwargs.token_keep_thres=none --set appcorr_kwargs.token_keep_ratio="$K"
done

echo "VFM_ACCURACY_CAMPAIGN_COMPLETE $(date)"
