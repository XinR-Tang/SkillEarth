
#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 1. Configuration
# ============================================================

# Paths
PROJECT_ROOT="Path/to/SkillEarth"
MODEL_ROOT="${PROJECT_ROOT}/models/eb"

TOOL_SERVER_ROOT="${PROJECT_ROOT}/tool_server/eb"
SERVICE_ROOT="${TOOL_SERVER_ROOT}/tools"
RUNTIME_ROOT="${TOOL_SERVER_ROOT}/tool_outputs"
LOG_DIR="${TOOL_SERVER_ROOT}/logs"
LOG_FILE="${LOG_DIR}/run_tools_eb.log"

# Conda Environments
FASTAPI_ENV="fastapi"
REMOTESAM_ENV="RemoteSAM"
INSTRUCTSAM_ENV="insam"

# GPU Configuration
GPU_ID="4"

# Conda Interpreters
CONDA_BASE="/home/JJ_Group/tanghj2409/miniconda3"
FASTAPI_PY="${CONDA_BASE}/envs/${FASTAPI_ENV}/bin/python"
REMOTESAM_PY="${CONDA_BASE}/envs/${REMOTESAM_ENV}/bin/python"
INSTRUCTSAM_PY="${CONDA_BASE}/envs/${INSTRUCTSAM_ENV}/bin/python"

# InstructSAM API
INSTRUCTSAM_BASE_URL="https://ark.cn-beijing.volces.com/api/plan/v3"
API_KEY="your_api_key" # Set this to your InstructSAM API key.
INSTRUCTSAM_MODEL="doubao-seed-2.0-mini"
export INSTRUCTSAM_API_KEY="$API_KEY"

# Server
HOST="0.0.0.0"
LOCAL_HOST="127.0.0.1"
READY_ATTEMPTS=150
READY_INTERVAL=2

# Vision Backend Ports
SAM2_PORT=16000
REMOTESAM_PORT=16001
INSTRUCTSAM_PORT=16002
REMOTECLIP_PORT=16003
STRIPRCNN_PORT=16004
SM3DET_PORT=16005
CHANGEOS_PORT=16006
MSCN_PORT=16007

# MCP Service Ports
ANALYSIS_PORT=31000
INDEX_PORT=31001
INVERSION_PORT=31002
PERCEPTION_PORT=31003
STATISTICS_PORT=31004

# All Service Ports
SERVICE_PORTS=(
  "$SAM2_PORT"
  "$REMOTESAM_PORT"
  "$INSTRUCTSAM_PORT"
  "$REMOTECLIP_PORT"
  "$STRIPRCNN_PORT"
  "$SM3DET_PORT"
  "$CHANGEOS_PORT"
  "$MSCN_PORT"
  "$ANALYSIS_PORT"
  "$INDEX_PORT"
  "$INVERSION_PORT"
  "$PERCEPTION_PORT"
  "$STATISTICS_PORT"
)


# ============================================================
# 2. Environment Variables
# ============================================================

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# Geospatial Data
export PROJ_LIB="${CONDA_BASE}/envs/${FASTAPI_ENV}/share/proj"
export PROJ_DATA="$PROJ_LIB"
export GDAL_DATA="${CONDA_BASE}/envs/${FASTAPI_ENV}/share/gdal"
export PROJ_NETWORK=OFF

# Runtime
export EARTH_AGENT_OUTPUT_DIR="$RUNTIME_ROOT"

# Clean Conflicting Environment Variables
unset PYTHONPATH PYTHONHOME
unset ALL_PROXY all_proxy
unset HTTP_PROXY http_proxy
unset HTTPS_PROXY https_proxy


# ============================================================
# 3. Validation & Initialization
# ============================================================

# Validate API Key
: "${INSTRUCTSAM_API_KEY:?Set INSTRUCTSAM_API_KEY before starting the Earth-Bench tools.}"

# Validate Conda Interpreters
for interpreter in \
  "$FASTAPI_PY" \
  "$REMOTESAM_PY" \
  "$INSTRUCTSAM_PY"; do

  if [[ ! -x "$interpreter" ]]; then
    echo "[error] Missing interpreter: $interpreter" >&2
    exit 1
  fi
done

# Validate Geospatial Data
for data_dir in "$PROJ_LIB" "$GDAL_DATA"; do
  if [[ ! -d "$data_dir" ]]; then
    echo "[error] Missing geospatial data: $data_dir" >&2
    exit 1
  fi
done

# Validate Model Checkpoint
REMOTECLIP_CKPT="${MODEL_ROOT}/RemoteCLIP/checkpoint/RemoteCLIP-ViT-L-14.pt"

if [[ ! -s "$REMOTECLIP_CKPT" ]]; then
  echo "[error] Missing RemoteCLIP checkpoint: $REMOTECLIP_CKPT" >&2
  exit 1
fi

# Initialize Directories
mkdir -p "$RUNTIME_ROOT" "$LOG_DIR"
cd "$SERVICE_ROOT"

# Check Port Availability
for port in "${SERVICE_PORTS[@]}"; do
  if ss -H -ltn "sport = :${port}" | grep -q .; then
    echo "[error] Port $port is already in use." >&2
    exit 1
  fi
done

echo "[earthbench] service root : $SERVICE_ROOT"
echo "[earthbench] runtime      : $RUNTIME_ROOT"
echo "[earthbench] GPU          : $GPU_ID"
echo "[earthbench] log          : $LOG_FILE"


# ============================================================
# 4. Service Management
# ============================================================

# Track All Background Processes
child_pids=()

# Stop All Services on Exit
cleanup() {
  local status=$?
  trap - EXIT INT TERM

  echo "[earthbench] Stopping services..."

  for pid in "${child_pids[@]}"; do
    kill "$pid" 2>/dev/null || true
  done

  for pid in "${child_pids[@]}"; do
    wait "$pid" 2>/dev/null || true
  done

  exit "$status"
}

trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# Start a Background Process
run_bg() {
  "$@" >>"$LOG_FILE" 2>&1 &
  child_pids+=("$!")
}

# Wait Until a Service Opens Its Port
wait_for_listener() {
  local port="$1"
  local name="$2"
  local pid="$3"

  for ((attempt=1; attempt<=READY_ATTEMPTS; attempt++)); do

    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[error] $name exited before opening port $port" >&2
      echo "[error] See log: $LOG_FILE" >&2
      return 1
    fi

    if ss -H -ltn "sport = :${port}" | grep -q .; then
      echo "[ready] $name : $port"
      return 0
    fi

    sleep "$READY_INTERVAL"
  done

  echo "[error] $name timed out on port $port" >&2
  echo "[error] See log: $LOG_FILE" >&2
  return 1
}

# Start a Service and Wait for Readiness
run_checked() {
  local port="$1"
  local name="$2"
  shift 2

  run_bg "$@"

  local index=$((${#child_pids[@]} - 1))
  wait_for_listener "$port" "$name" "${child_pids[$index]}"
}

# Start a Vision Backend
start_vision() {
  local name="$1"
  local port="$2"
  local python_bin="$3"
  local app="$4"

  run_checked "$port" "$name" \
    env CUDA_VISIBLE_DEVICES="$GPU_ID" \
    "$python_bin" -m uvicorn "$app" \
    --host "$HOST" \
    --port "$port"
}

# Start an MCP Service
start_mcp() {
  local name="$1"
  local port="$2"
  local script="$3"
  shift 3

  run_checked "$port" "$name" \
    "$FASTAPI_PY" "$script" \
    --port "$port" \
    --temp_dir "$RUNTIME_ROOT" \
    "$@"
}


# ============================================================
# 5. Vision Backends
# ============================================================

echo "[earthbench] Starting vision backends..."

# SAM2
start_vision \
  sam2 "$SAM2_PORT" \
  "$FASTAPI_PY" app.sam2_app:app

# RemoteSAM
start_vision \
  remotesam "$REMOTESAM_PORT" \
  "$REMOTESAM_PY" app.remotesam_app:app

# InstructSAM (API-backed)
run_checked "$INSTRUCTSAM_PORT" instructsam \
  env CUDA_VISIBLE_DEVICES="$GPU_ID" \
  OPENAI_API_KEY="$INSTRUCTSAM_API_KEY" \
  OPENAI_BASE_URL="$INSTRUCTSAM_BASE_URL" \
  INSTRUCTSAM_MODEL="$INSTRUCTSAM_MODEL" \
  "$INSTRUCTSAM_PY" -m uvicorn app.instructsam_app:app \
  --host "$HOST" \
  --port "$INSTRUCTSAM_PORT"

# RemoteCLIP
start_vision \
  remoteclip "$REMOTECLIP_PORT" \
  "$INSTRUCTSAM_PY" app.remoteclip_app:app

# StripRCNN
start_vision \
  striprcnn "$STRIPRCNN_PORT" \
  "$REMOTESAM_PY" app.striprcnn_app:app

# SM3Det
start_vision \
  sm3det "$SM3DET_PORT" \
  "$REMOTESAM_PY" app.sm3det_app:app

# ChangeOS
start_vision \
  changeos "$CHANGEOS_PORT" \
  "$INSTRUCTSAM_PY" app.changeos_app:app

# MSCN
start_vision \
  mscn "$MSCN_PORT" \
  "$REMOTESAM_PY" app.mscn_app:app


# ============================================================
# 6. Earth-Bench MCP Services
# ============================================================

echo "[earthbench] Starting MCP services..."

# Analysis
start_mcp analysis "$ANALYSIS_PORT" Analysis.py

# Index
start_mcp index "$INDEX_PORT" Index.py

# Inversion
start_mcp inversion "$INVERSION_PORT" Inversion.py

# Perception (Connect All Vision Backends)
start_mcp perception "$PERCEPTION_PORT" Perception.py \
  --sam2_endpoint "http://${LOCAL_HOST}:${SAM2_PORT}" \
  --remotesam_endpoint "http://${LOCAL_HOST}:${REMOTESAM_PORT}" \
  --instructsam_endpoint "http://${LOCAL_HOST}:${INSTRUCTSAM_PORT}" \
  --remoteclip_endpoint "http://${LOCAL_HOST}:${REMOTECLIP_PORT}" \
  --striprcnn_endpoint "http://${LOCAL_HOST}:${STRIPRCNN_PORT}" \
  --sm3det_endpoint "http://${LOCAL_HOST}:${SM3DET_PORT}" \
  --changeos_endpoint "http://${LOCAL_HOST}:${CHANGEOS_PORT}" \
  --mscn_endpoint "http://${LOCAL_HOST}:${MSCN_PORT}"

# Statistics
start_mcp statistics "$STATISTICS_PORT" Statistics.py


# ============================================================
# 7. Keep Services Running
# ============================================================

echo "[earthbench] All services are ready."
echo "[earthbench] Logs: $LOG_FILE"

wait "${child_pids[@]}"
