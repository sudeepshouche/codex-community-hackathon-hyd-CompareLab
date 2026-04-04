#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WEB_DIR="$ROOT_DIR/web"
VENV_DIR="$ROOT_DIR/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP="$VENV_DIR/bin/pip"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NODE_VERSION="${NODE_VERSION:-22.16.0}"
BACKEND_START_TIMEOUT="${BACKEND_START_TIMEOUT:-180}"
FRONTEND_START_TIMEOUT="${FRONTEND_START_TIMEOUT:-120}"

export TRIBEV2_PORT="${TRIBEV2_PORT:-8002}"
export PORT="${PORT:-3000}"
export TRIBE_API_URL="${TRIBE_API_URL:-http://127.0.0.1:${TRIBEV2_PORT}/analyze}"
export TRIBEV2_CACHE_DIR="${TRIBEV2_CACHE_DIR:-$ROOT_DIR/cache/tribev2}"
export TRIBEV2_DEVICE="${TRIBEV2_DEVICE:-auto}"
export TRIBEV2_PROFILE="${TRIBEV2_PROFILE:-fast}"
export TRIBEV2_PREWARM="${TRIBEV2_PREWARM:-0}"
export TRIBEV2_MODEL_IDLE_TTL_SEC="${TRIBEV2_MODEL_IDLE_TTL_SEC:-300}"
export TRIBEV2_FAST_VIDEO_SAMPLING_HZ="${TRIBEV2_FAST_VIDEO_SAMPLING_HZ:-0.5}"
export TRIBEV2_FAST_VIDEO_NUM_FRAMES="${TRIBEV2_FAST_VIDEO_NUM_FRAMES:-8}"
export TRIBEV2_FAST_VIDEO_MAX_IMSIZE="${TRIBEV2_FAST_VIDEO_MAX_IMSIZE:-384}"
export TRIBEV2_FAST_VIDEO_MAX_DURATION_SEC="${TRIBEV2_FAST_VIDEO_MAX_DURATION_SEC:-30}"
export TRIBEV2_WHISPERX_CPU_COMPUTE_TYPE="${TRIBEV2_WHISPERX_CPU_COMPUTE_TYPE:-int8}"
export TRIBEV2_WHISPERX_CUDA_COMPUTE_TYPE="${TRIBEV2_WHISPERX_CUDA_COMPUTE_TYPE:-float16}"
export TRIBEV2_WHISPERX_MODEL="${TRIBEV2_WHISPERX_MODEL:-large-v3}"
export TRIBEV2_WHISPERX_BATCH_SIZE="${TRIBEV2_WHISPERX_BATCH_SIZE:-16}"
BACKEND_NICE_LEVEL="${BACKEND_NICE_LEVEL:-10}"

BACKEND_PID=""
FRONTEND_PID=""

log() {
  printf '[run.sh] %s\n' "$*"
}

fail() {
  printf '[run.sh] ERROR: %s\n' "$*" >&2
  exit 1
}

have_command() {
  command -v "$1" >/dev/null 2>&1
}

load_nvm() {
  if [[ -s "$HOME/.nvm/nvm.sh" ]]; then
    # shellcheck disable=SC1090
    source "$HOME/.nvm/nvm.sh"
  fi
}

ensure_node() {
  load_nvm
  if have_command nvm; then
    nvm install "$NODE_VERSION" >/dev/null
    nvm use "$NODE_VERSION" >/dev/null
  fi

  have_command node || fail "Node.js is required. Install Node ${NODE_VERSION} or make it available on PATH."
  have_command npm || fail "npm is required. Install Node ${NODE_VERSION} or make it available on PATH."
}

ensure_python() {
  have_command "$PYTHON_BIN" || fail "Python executable '$PYTHON_BIN' was not found."
}

ensure_venv() {
  if [[ ! -x "$VENV_PYTHON" ]]; then
    log "Creating Python virtual environment at $VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi

  # shellcheck disable=SC1090
  source "$VENV_DIR/bin/activate"
  "$VENV_PYTHON" -m pip install --upgrade pip >/dev/null
}

python_deps_ready() {
  "$VENV_PYTHON" - <<'PY' >/dev/null 2>&1
import importlib.util
import sys

required = ["numpy", "backend", "tribev2"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
sys.exit(0 if not missing else 1)
PY
}

ensure_python_deps() {
  if python_deps_ready; then
    return
  fi

  log "Installing Python dependencies into .venv"
  "$VENV_PIP" install -e "$ROOT_DIR"
}

ensure_web_deps() {
  if [[ ! -d "$WEB_DIR/node_modules" ]]; then
    log "Installing web dependencies"
    (
      cd "$WEB_DIR"
      npm install
    )
  fi
}

check_backend_health() {
  curl -fsS "http://127.0.0.1:${TRIBEV2_PORT}/health" >/dev/null 2>&1
}

check_frontend_health() {
  local body
  body="$(curl -fsS "http://127.0.0.1:${PORT}/" 2>/dev/null || true)"
  [[ "$body" == *"TRIBE Compare Lab"* ]]
}

port_is_listening() {
  if have_command nc; then
    nc -z 127.0.0.1 "$1" >/dev/null 2>&1
    return
  fi

  "$PYTHON_BIN" - "$1" <<'PY' >/dev/null 2>&1
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket()
sock.settimeout(0.5)
try:
    sock.connect(("127.0.0.1", port))
except OSError:
    sys.exit(1)
finally:
    sock.close()
sys.exit(0)
PY
}

wait_for_health() {
  local name="$1"
  local timeout="$2"
  local pid="$3"
  local check_fn="$4"

  local elapsed=0
  while (( elapsed < timeout )); do
    if "$check_fn"; then
      return 0
    fi
    if ! kill -0 "$pid" 2>/dev/null; then
      wait "$pid" || true
      fail "$name exited before becoming healthy."
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done

  fail "$name did not become healthy within ${timeout}s."
}

stop_process() {
  local pid="$1"
  if [[ -z "$pid" ]]; then
    return
  fi
  if ! kill -0 "$pid" 2>/dev/null; then
    return
  fi

  pkill -TERM -P "$pid" >/dev/null 2>&1 || true
  kill -TERM "$pid" >/dev/null 2>&1 || true

  local _=0
  while kill -0 "$pid" 2>/dev/null && (( _ < 10 )); do
    sleep 1
    _=$((_ + 1))
  done

  if kill -0 "$pid" 2>/dev/null; then
    pkill -KILL -P "$pid" >/dev/null 2>&1 || true
    kill -KILL "$pid" >/dev/null 2>&1 || true
  fi

  wait "$pid" 2>/dev/null || true
}

cleanup() {
  log "Cleaning up background processes"
  stop_process "$FRONTEND_PID"
  stop_process "$BACKEND_PID"
}

trap cleanup EXIT INT TERM

start_backend() {
  if port_is_listening "$TRIBEV2_PORT"; then
    fail "Port ${TRIBEV2_PORT} is already in use. Pick a free TRIBEV2_PORT or stop the old process."
  fi

  log "Starting backend on http://127.0.0.1:${TRIBEV2_PORT}"
  (
    cd "$ROOT_DIR"
    if have_command nice; then
      exec nice -n "$BACKEND_NICE_LEVEL" "$VENV_PYTHON" -m backend.server
    fi
    exec "$VENV_PYTHON" -m backend.server
  ) &
  BACKEND_PID=$!
  wait_for_health "Backend" "$BACKEND_START_TIMEOUT" "$BACKEND_PID" check_backend_health
}

start_frontend() {
  if port_is_listening "$PORT"; then
    fail "Port ${PORT} is already in use. Pick a free PORT or stop the old process."
  fi

  log "Starting frontend on http://127.0.0.1:${PORT}"
  (
    cd "$WEB_DIR"
    exec npm run dev
  ) &
  FRONTEND_PID=$!
  wait_for_health "Frontend" "$FRONTEND_START_TIMEOUT" "$FRONTEND_PID" check_frontend_health
}

print_status() {
  printf '\n'
  printf 'TRIBE Compare Lab is running.\n'
  printf 'Frontend: %s\n' "http://127.0.0.1:${PORT}"
  printf 'Backend:  %s\n' "http://127.0.0.1:${TRIBEV2_PORT}"
  printf 'API:      %s\n' "$TRIBE_API_URL"
  printf 'Press Ctrl-C to stop both services and clean up.\n'
  printf '\n'
}

main() {
  ensure_python
  ensure_node
  ensure_venv
  ensure_python_deps
  ensure_web_deps
  start_backend
  start_frontend
  print_status

  local status=0
  while true; do
    if ! kill -0 "$BACKEND_PID" 2>/dev/null; then
      wait "$BACKEND_PID" || status=$?
      break
    fi
    if ! kill -0 "$FRONTEND_PID" 2>/dev/null; then
      wait "$FRONTEND_PID" || status=$?
      break
    fi
    sleep 1
  done

  exit "$status"
}

main "$@"
