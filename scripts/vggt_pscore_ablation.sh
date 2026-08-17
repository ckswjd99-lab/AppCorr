#!/usr/bin/env bash
# Which half of the score does the work?
#
#   attn      server attention only            (what every earlier VGGT number used)
#   residual  mobile residual energy only
#   geomean   both, fused multiplicatively
#
# `add` is deliberately not here: the two terms differ by ~7 orders of magnitude (softmax
# probability ~1e-3 vs residual energy ~1e4) and _combine_patch_scores does not normalise, so an
# additive fusion is a residual-only run wearing a combined label.
set -u
cd /NHNHOME/share/cjpark/AppCorr
R=${R:-0.20}
run () {  # name, extra --set args...
  local name=$1; shift
  ps -eo pid,cmd | grep -E "offload/(server|mobile)/main" | grep -v grep | awk '{print $1}' | xargs -r kill -9 2>/dev/null
  sleep 4
  echo "##### $name (ratio=$R) #####"
  CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 APPCORR_RESULT_TIMEOUT=1800 timeout 3600 \
    ./offload/run_local.sh offload/config/co3d/co3d_appcorr.json -nr 20 -nw 0 \
    -d /NHNHOME/share/cjpark/data/co3dv2/extracted \
    --set appcorr_kwargs.token_keep_ratio=$R "$@" > "logs/vggt/psc_${name}_r${R}.log" 2>&1
  grep -aoE "Final Summary: \{[^}]*\}|!!! \[Worker\][^:]{0,120}|ValueError: [^,]{0,100}" \
    "logs/vggt/psc_${name}_r${R}.log" | tail -2
}
run attn     --set appcorr_kwargs.mobile_pscore_weight=0.0
run residual --set appcorr_kwargs.mobile_pscore=residual_energy --set appcorr_kwargs.mobile_pscore_weight=1.0 --set appcorr_kwargs.server_pscore_weight=0.0 --set appcorr_kwargs.pscore_fusion=add
run geomean  --set appcorr_kwargs.mobile_pscore=residual_energy --set appcorr_kwargs.mobile_pscore_weight=1.0 --set appcorr_kwargs.pscore_fusion=geo_mean
echo "PSCORE_ABLATION_DONE"
