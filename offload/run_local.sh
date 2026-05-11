#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  offload/run_local.sh CONFIG_PATH [-nr N] [-nw N] [-d DATA_ROOT]

Runs the AppCorr server and mobile client locally against one config.

Options:
  -nr N, --num-request N  Run only N requests (batches); omit to run all
  -nw N, --num-warmup N   Run N warm-up requests before measurement. Default: 1
  -d PATH, --data PATH    Dataset root path (overrides config's dataset_kwargs.data_root)
  -ns,   --nsys           Profile the server with Nsight Systems (nsys profile)

Environment overrides:
  RECV_PORT       Server receive / mobile upload port. Default: 39998
  SEND_PORT       Server send / mobile download port. Default: 39999
  SERVER_STARTUP  Seconds to wait before launching mobile. Default: 2

Examples:
  offload/run_local.sh offload/config/coco_interleaved_dynamic.json
  offload/run_local.sh offload/config/ade20k_m2f_sequential.json -nr 10
  offload/run_local.sh offload/config/nyu_sequential.json -d ~/data/NYU -nr 10
  offload/run_local.sh offload/config/ade20k_m2f_sequential.json -nr 10 -ns
EOF
}

NUM_REQUEST=""
NUM_WARMUP="1"
USE_NSYS=false
DATA_ROOT=""
POSITIONAL_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--data)
      if [[ $# -lt 2 ]]; then
        echo "[run_local] --data requires an argument" >&2
        usage >&2
        exit 1
      fi
      DATA_ROOT="$2"
      shift 2
      ;;
    -nr|--num-request)
      if [[ $# -lt 2 ]]; then
        echo "[run_local] --num-request requires an argument" >&2
        usage >&2
        exit 1
      fi
      NUM_REQUEST="$2"
      shift 2
      ;;
    -nw|--num-warmup)
      if [[ $# -lt 2 ]]; then
        echo "[run_local] --num-warmup requires an argument" >&2
        usage >&2
        exit 1
      fi
      NUM_WARMUP="$2"
      shift 2
      ;;
    -ns|--nsys)
      USE_NSYS=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "[run_local] Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ ${#POSITIONAL_ARGS[@]} -ne 1 ]]; then
  usage >&2
  exit 1
fi

CALLER_DIR="$(pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

CONFIG_PATH_INPUT="${POSITIONAL_ARGS[0]}"
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
BASE_EXP_ID="$(python - "${CONFIG_PATH}" <<'PY'
import json
import sys

with open(sys.argv[1], "r") as f:
    data = json.load(f)
print(data.get("exp_id") or "exp")
PY
)"
RUN_START_TS="$(date +%s)"
NSYS_FINALIZED=false
NSYS_TEMP_PROFILE="${REPO_ROOT}/temp_profile.nsys-rep"

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

describe_port_usage() {
  local port="$1"

  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"${port}" -sTCP:LISTEN 2>/dev/null || true
    return
  fi

  if command -v ss >/dev/null 2>&1; then
    ss -H -ltnp "sport = :${port}" 2>/dev/null || true
    return
  fi

  if command -v netstat >/dev/null 2>&1; then
    netstat -ltnp 2>/dev/null | awk -v port=":${port}" '$4 ~ port "$"'
    return
  fi
}

port_listener_pids() {
  local port="$1"

  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -t -iTCP:"${port}" -sTCP:LISTEN 2>/dev/null || true
    return
  fi

  if command -v ss >/dev/null 2>&1; then
    ss -H -ltnp "sport = :${port}" 2>/dev/null \
      | sed -nE 's/.*pid=([0-9]+).*/\1/p' \
      | sort -u
    return
  fi

  if command -v netstat >/dev/null 2>&1; then
    netstat -ltnp 2>/dev/null \
      | awk -v port=":${port}" '$4 ~ port "$" { split($7, proc, "/"); if (proc[1] ~ /^[0-9]+$/) print proc[1] }' \
      | sort -u
    return
  fi
}

ensure_port_free() {
  local label="$1"
  local port="$2"
  local usage
  local pids

  usage="$(describe_port_usage "${port}")"
  if [[ -z "${usage}" ]]; then
    return
  fi

  echo "[run_local] ${label}=${port} is already in use; aborting before startup." >&2
  echo "[run_local] Listener using port ${port}:" >&2
  echo "${usage}" | sed 's/^/[run_local]   /' >&2

  pids="$(port_listener_pids "${port}" | sort -u | xargs || true)"
  if [[ -n "${pids}" ]]; then
    echo "[run_local] To stop the occupying process(es), run:" >&2
    echo "[run_local]   kill ${pids}" >&2
    echo "[run_local]   # if needed: kill -9 ${pids}" >&2
  else
    echo "[run_local] Could not determine PID. Try:" >&2
    echo "[run_local]   lsof -nP -iTCP:${port} -sTCP:LISTEN" >&2
  fi

  exit 1
}

find_latest_log_dir() {
  python - "${BASE_EXP_ID}" "${RUN_START_TS}" <<'PY'
import sys
from pathlib import Path

base_exp_id = sys.argv[1]
run_start_ts = float(sys.argv[2])
root = Path("logs") / "offload"
if not root.exists():
    print("")
    raise SystemExit(0)

candidates = [
    path
    for path in root.iterdir()
    if path.is_dir() and path.name.startswith(f"{base_exp_id}_")
]
if not candidates:
    print("")
    raise SystemExit(0)

def candidate_key(path: Path):
    events_path = path / "events.jsonl"
    mtime = events_path.stat().st_mtime if events_path.exists() else path.stat().st_mtime
    return (mtime >= run_start_ts - 60, mtime)

candidates.sort(key=candidate_key, reverse=True)
print(candidates[0])
PY
}

finalize_nsys_profile() {
  if [[ "${USE_NSYS}" != true || "${NSYS_FINALIZED}" == true ]]; then
    return 0
  fi
  NSYS_FINALIZED=true

  if [[ ! -f "${NSYS_TEMP_PROFILE}" ]]; then
    echo "[run_local] Warning: Nsight profile not found at ${NSYS_TEMP_PROFILE}" >&2
    return 0
  fi

  local log_dir
  log_dir="$(find_latest_log_dir || true)"
  if [[ -z "${log_dir}" ]]; then
    echo "[run_local] Warning: could not find log directory for exp_id=${BASE_EXP_ID}; leaving ${NSYS_TEMP_PROFILE} in place." >&2
    return 0
  fi

  local dest_profile="${log_dir}/server_profile.nsys-rep"
  if mv -f "${NSYS_TEMP_PROFILE}" "${dest_profile}"; then
    echo "[run_local] Moved Nsight profile to ${dest_profile}"
  else
    echo "[run_local] Warning: failed to move Nsight profile to ${dest_profile}" >&2
    return 0
  fi

  if python -m analysis.log_tools.nsys_events "${log_dir}" --profile "${dest_profile}"; then
    echo "[run_local] Generated ${log_dir}/events_nsys.jsonl"
  else
    echo "[run_local] Warning: failed to generate events_nsys.jsonl from ${dest_profile}" >&2
  fi
}

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  stop_process_group "mobile" "${MOBILE_PID}"
  stop_process_group "server" "${SERVER_PID}"
  finalize_nsys_profile || true
  exit "${exit_code}"
}

trap cleanup EXIT INT TERM

if [[ "${RECV_PORT}" == "${SEND_PORT}" ]]; then
  echo "[run_local] RECV_PORT and SEND_PORT must be different; both are ${RECV_PORT}." >&2
  exit 1
fi

ensure_port_free "RECV_PORT" "${RECV_PORT}"
ensure_port_free "SEND_PORT" "${SEND_PORT}"

echo "[run_local] Starting local AppCorr server..."
if [[ "${USE_NSYS}" == true ]]; then
  echo "[run_local] Nsight Systems profiling enabled -> temp_profile.nsys-rep"
  rm -f "${NSYS_TEMP_PROFILE}" "${REPO_ROOT}/temp_profile.sqlite"
  start_in_own_group nsys profile \
            --sample=none \
            --cpuctxsw=none \
            --trace-fork-before-exec=true \
            --trace=cuda,cuda-hw,nvtx \
            --output=temp_profile \
    --force-overwrite=true \
    -- python offload/server/main.py \
      --recv-port "${RECV_PORT}" \
      --send-port "${SEND_PORT}"
else
  start_in_own_group python offload/server/main.py \
    --recv-port "${RECV_PORT}" \
    --send-port "${SEND_PORT}"
fi
SERVER_PID="${STARTED_PID}"

echo "[run_local] Server PID: ${SERVER_PID}"
echo "[run_local] Launching mobile client in ${SERVER_STARTUP}s with config: ${CONFIG_PATH}"
sleep "${SERVER_STARTUP}"

start_in_own_group python offload/mobile/main.py \
  --config "${CONFIG_PATH}" \
  --ip 127.0.0.1 \
  --recv-port "${RECV_PORT}" \
  --send-port "${SEND_PORT}" \
  --num-warmup "${NUM_WARMUP}" \
  ${DATA_ROOT:+--data "${DATA_ROOT}"} \
  ${NUM_REQUEST:+--num-request "${NUM_REQUEST}"}
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
