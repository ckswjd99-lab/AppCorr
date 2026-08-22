#!/usr/bin/env bash
# Re-run the FLOPs arms that failed in the first campaign pass, with the overrides they need.
#
#   NYU     every nyu config pins "device": "cuda:1", which is an invalid ordinal once
#           CUDA_VISIBLE_DEVICES masks the second GPU away -- and correction is cuda:0-only anyway.
#   ADE20K  blocked by a pre-existing signature mismatch in the crop_cover group assignment,
#           fixed alongside this script.
set -u
cd /NHNHOME/share/cjpark/AppCorr-flops
OUT=${OUT:-analysis/results/flops}; mkdir -p "$OUT"
NR=${NR:-3}
export CUDA_VISIBLE_DEVICES=${GPU:-0} APPCORR_FLOPS=1
export PATH=/home/nxclab/anaconda3/envs/appcorr/bin:$PATH   # run_local.sh calls a bare `python` that lacks transformers

run () {  # run <tag> <config> <extra...>
  local tag=$1 cfg=$2; shift 2
  [ -s "$OUT/$tag.json" ] && { echo "[skip ] $tag"; return; }
  echo "[start] $tag $(date +%H:%M:%S)"
  APPCORR_FLOPS_OUT="$OUT/$tag.json" timeout 2400 \
    bash offload/run_local.sh "$cfg" -nr "$NR" -nw 0 "$@" > "$OUT/$tag.log" 2>&1
  echo "[done ] $tag rc=$?  $(grep -ao 'mean_critical=[0-9.]* GF' "$OUT/$tag.log" | tail -1)"
}

for K in 0.25 0.30 0.50; do
  run "dinov3_nyu_g4_k${K}"    offload/config/nyu/nyu_interleaved_static.json \
      --set device=cuda:0 --set appcorr_kwargs.token_keep_thres=none --set appcorr_kwargs.token_keep_ratio=$K
  run "dinov3_ade20k_g4_k${K}" offload/config/ade20k/ade20k_m2f_interleaved_static.json \
      --set appcorr_kwargs.token_keep_thres=none --set appcorr_kwargs.token_keep_ratio=$K
done
run "dinov3_nyu_ceiling"    offload/config/nyu/nyu_sequential.json --set device=cuda:0
run "dinov3_ade20k_ceiling" offload/config/ade20k/ade20k_m2f_sequential.json
echo "FLOPS RETRY COMPLETE $(date)"
