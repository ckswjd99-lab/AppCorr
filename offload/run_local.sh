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
MOBILE_PID=""
STARTED_PID=""

start_in_own_group() {
  if command -v setsid >/dev/null 2>&1; then
    setsid "$@" &
  else
    "$@" &
  fi
  STARTED_PID=$!
}

process_group_alive() {
  local pid="$1"
  kill -0 -- "-${pid}" 2>/dev/null
}

wait_for_process_group_exit() {
  local pid="$1"
  local attempts="${2:-20}"
  local i
  for ((i = 0; i < attempts; i++)); do
    if ! process_group_alive "${pid}"; then
      return 0
    fi
    sleep 0.2
  done
  return 1
}

stop_process_group() {
  local name="$1"
  local pid="$2"

  if [[ -z "${pid}" ]]; then
    return
  fi

  if process_group_alive "${pid}"; then
    echo "[run_local] Stopping ${name} process group ${pid}..."
    kill -INT -- "-${pid}" 2>/dev/null || true
    if wait_for_process_group_exit "${pid}" 20; then
      return
    fi

    echo "[run_local] ${name} did not stop after SIGINT; sending SIGTERM..."
    kill -TERM -- "-${pid}" 2>/dev/null || true
    if wait_for_process_group_exit "${pid}" 20; then
      return
    fi

    echo "[run_local] ${name} did not stop after SIGTERM; sending SIGKILL..."
    kill -KILL -- "-${pid}" 2>/dev/null || true
    wait_for_process_group_exit "${pid}" 10 || true
  elif kill -0 "${pid}" 2>/dev/null; then
    echo "[run_local] Stopping ${name} process ${pid}..."
    kill -INT "${pid}" 2>/dev/null || true
    sleep 0.5
    if kill -0 "${pid}" 2>/dev/null; then
      kill -TERM "${pid}" 2>/dev/null || true
    fi
    wait "${pid}" 2>/dev/null || true
  fi
}

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  stop_process_group "mobile" "${MOBILE_PID}"
  stop_process_group "server" "${SERVER_PID}"
  exit "${exit_code}"
}

trap cleanup EXIT INT TERM

echo "[run_local] Starting local AppCorr server..."
start_in_own_group python offload/server/main.py \
  --recv-port "${RECV_PORT}" \
  --send-port "${SEND_PORT}"
SERVER_PID="${STARTED_PID}"

echo "[run_local] Server PID: ${SERVER_PID}"
echo "[run_local] Launching mobile client in ${SERVER_STARTUP}s with config: ${CONFIG_PATH}"
sleep "${SERVER_STARTUP}"

start_in_own_group python offload/mobile/main.py \
  --config "${CONFIG_PATH}" \
  --ip 127.0.0.1 \
  --recv-port "${RECV_PORT}" \
  --send-port "${SEND_PORT}" \
  --data "${DATA_ROOT}"
MOBILE_PID="${STARTED_PID}"

set +e
wait "${MOBILE_PID}"
MOBILE_STATUS=$?
set -e
MOBILE_PID=""

if [[ "${MOBILE_STATUS}" -ne 0 ]]; then
  exit "${MOBILE_STATUS}"
fi

echo "[run_local] Mobile client finished. Waiting for server shutdown..."
wait "${SERVER_PID}" 2>/dev/null || true
SERVER_PID=""
