#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  offload/run_local.sh CONFIG_PATH [DATA_ROOT]

Runs the AppCorr server and mobile client locally against one config.

Environment overrides:
  RECV_PORT       Server receive / mobile upload port. Default: 39998
  SEND_PORT       Server send / mobile download port. Default: 39999
  SERVER_STARTUP  Seconds to wait before launching mobile. Default: 2

Examples:
  offload/run_local.sh offload/config/coco_interleaved_dynamic.json
  offload/run_local.sh offload/config/imnet_interleaved_g4.json ~/data/imagenet_val
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -lt 1 || $# -gt 2 ]]; then
  usage >&2
  exit 1
fi

CALLER_DIR="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_PATH_INPUT="$1"
DATA_ROOT="${2:-~/data/imagenet_val}"
RECV_PORT="${RECV_PORT:-39998}"
SEND_PORT="${SEND_PORT:-39999}"
SERVER_STARTUP="${SERVER_STARTUP:-2}"

if [[ "${CONFIG_PATH_INPUT}" = /* ]]; then
  CONFIG_PATH="${CONFIG_PATH_INPUT}"
elif [[ -f "${CALLER_DIR}/${CONFIG_PATH_INPUT}" ]]; then
  CONFIG_PATH="${CALLER_DIR}/${CONFIG_PATH_INPUT}"
else
  CONFIG_PATH="${REPO_ROOT}/${CONFIG_PATH_INPUT}"
fi

cd "${REPO_ROOT}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "[run_local] Config not found: ${CONFIG_PATH_INPUT}" >&2
  exit 1
fi

SERVER_PID=""

cleanup() {
  local exit_code=$?
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "[run_local] Stopping server process ${SERVER_PID}..."
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  exit "${exit_code}"
}

trap cleanup EXIT INT TERM

echo "[run_local] Starting local AppCorr server..."
python offload/server/main.py \
  --recv-port "${RECV_PORT}" \
  --send-port "${SEND_PORT}" &
SERVER_PID=$!

echo "[run_local] Server PID: ${SERVER_PID}"
echo "[run_local] Launching mobile client in ${SERVER_STARTUP}s with config: ${CONFIG_PATH}"
sleep "${SERVER_STARTUP}"

python offload/mobile/main.py \
  --config "${CONFIG_PATH}" \
  --ip 127.0.0.1 \
  --recv-port "${RECV_PORT}" \
  --send-port "${SEND_PORT}" \
  --data "${DATA_ROOT}"

echo "[run_local] Mobile client finished. Waiting for server shutdown..."
wait "${SERVER_PID}" 2>/dev/null || true
SERVER_PID=""
