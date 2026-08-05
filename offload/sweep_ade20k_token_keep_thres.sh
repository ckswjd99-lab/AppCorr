#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DEVICE="${DEVICE:-cuda:1}"
GROUPING_STRATEGY="${GROUPING_STRATEGY:-}"
NUM_WARMUP="${NUM_WARMUP:-1}"
NUM_REQUEST="${NUM_REQUEST-}"
SERVER_STARTUP="${SERVER_STARTUP:-2}"
RECV_PORT="${RECV_PORT:-39990}"
SEND_PORT="${SEND_PORT:-39991}"
THRESHOLDS_STR="${THRESHOLDS:-0 0.00005 0.0001 0.0002 0.0005 0.001 0.002}"
GROUPING_LABEL="${GROUPING_STRATEGY:-config}"
GROUPING_LABEL="${GROUPING_LABEL//./p}"
GROUPING_LABEL="${GROUPING_LABEL//-/m}"
GROUPING_LABEL="${GROUPING_LABEL//\//_}"
SWEEP_CSV="${SWEEP_CSV:-logs/offload/ade20k_interleaved_static_${GROUPING_LABEL}_token_keep_thres_sweep_$(date +%Y%m%d_%H%M%S).csv}"

read -r -a THRESHOLDS <<< "${THRESHOLDS_STR}"

CONFIGS=(
  "offload/config/ade20k/ade20k_m2f_interleaved_static.json"
)

run_one() {
  local config_path="$1"
  local threshold="$2"
  local base_name
  local safe_threshold
  local safe_grouping
  local exp_id
  local args

  base_name="$(basename "${config_path}" .json)"
  safe_threshold="${threshold//./p}"
  safe_threshold="${safe_threshold//-/m}"
  safe_grouping="${GROUPING_STRATEGY//./p}"
  safe_grouping="${safe_grouping//-/m}"
  safe_grouping="${safe_grouping//\//_}"
  if [[ -n "${GROUPING_STRATEGY}" ]]; then
    exp_id="${base_name}_${safe_grouping}_tkt_${safe_threshold}"
  else
    exp_id="${base_name}_tkt_${safe_threshold}"
  fi

  args=(
    "${REPO_ROOT}/offload/run_local.sh"
    "${config_path}"
    --num-warmup "${NUM_WARMUP}"
    --set "device=${DEVICE}"
    --set "exp_id=${exp_id}"
    --set "appcorr_kwargs.token_keep_thres=${threshold}"
  )

  if [[ -n "${GROUPING_STRATEGY}" ]]; then
    args+=(--set "transmission_kwargs.grouping_strategy=${GROUPING_STRATEGY}")
  fi

  if [[ -n "${NUM_REQUEST}" ]]; then
    args+=(--num-request "${NUM_REQUEST}")
  fi

  echo
  echo "================================================================"
  echo "[sweep] config=${config_path}"
  echo "[sweep] token_keep_thres=${threshold}"
  echo "[sweep] exp_id=${exp_id}"
  echo "[sweep] device=${DEVICE}"
  if [[ -n "${GROUPING_STRATEGY}" ]]; then
    echo "[sweep] grouping_strategy=${GROUPING_STRATEGY}"
  else
    echo "[sweep] grouping_strategy=config"
  fi
  if [[ -n "${NUM_REQUEST}" ]]; then
    echo "[sweep] num_request=${NUM_REQUEST}"
  else
    echo "[sweep] num_request=all"
  fi
  echo "================================================================"

  RECV_PORT="${RECV_PORT}" \
  SEND_PORT="${SEND_PORT}" \
  SERVER_STARTUP="${SERVER_STARTUP}" \
  "${args[@]}"

  python - "${base_name}" "${threshold}" "${exp_id}" "${SWEEP_CSV}" "${GROUPING_STRATEGY:-config}" <<'PY'
import csv
import json
import sys
from pathlib import Path

method, threshold, exp_id, csv_path, grouping_strategy = sys.argv[1:6]
root = Path("logs") / "offload"
candidates = sorted(
    [path for path in root.glob(f"{exp_id}_*") if path.is_dir()],
    key=lambda path: path.stat().st_mtime,
    reverse=True,
)
if not candidates:
    print(f"[sweep] Warning: no log directory found for {exp_id}", file=sys.stderr)
    raise SystemExit(0)

log_dir = candidates[0]
summary_path = log_dir / "summary.json"
if not summary_path.exists():
    print(f"[sweep] Warning: no summary.json found in {log_dir}", file=sys.stderr)
    raise SystemExit(0)

with summary_path.open() as f:
    summary = json.load(f)
dataset = summary.get("dataset_summary", {})
row = {
    "method": method,
    "grouping_strategy": grouping_strategy,
    "token_keep_thres": threshold,
    "mIoU": dataset.get("mIoU", ""),
    "aAcc": dataset.get("aAcc", ""),
    "num_measured_requests": summary.get("num_measured_requests", ""),
    "avg_latency_per_batch": summary.get("avg_latency_per_batch", ""),
    "avg_partial_token_keep_pct": summary.get("avg_partial_token_keep_pct", ""),
    "avg_partial_token_kept_patch_count": summary.get("avg_partial_token_kept_patch_count", ""),
    "log_dir": str(log_dir),
}

csv_file = Path(csv_path)
csv_file.parent.mkdir(parents=True, exist_ok=True)
write_header = not csv_file.exists()
with csv_file.open("a", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(row.keys()))
    if write_header:
        writer.writeheader()
    writer.writerow(row)

print(
    "[sweep] summary "
    f"method={method} threshold={threshold} "
    f"mIoU={row['mIoU']} aAcc={row['aAcc']} log_dir={log_dir}"
)
print(f"[sweep] csv={csv_file}")
PY
}

cd "${REPO_ROOT}"

for config_path in "${CONFIGS[@]}"; do
  for threshold in "${THRESHOLDS[@]}"; do
    run_one "${config_path}" "${threshold}"
  done
done

echo
echo "[sweep] Done. Summaries are under logs/offload/*_tkt_*_<timestamp>/summary.json"
echo "[sweep] CSV: ${SWEEP_CSV}"
