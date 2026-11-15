
#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 1. Configuration
# ============================================================

# Paths
ROOT_DIR="Path/to/SkillEarth"
MODEL_DIR="${ROOT_DIR}/model"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Model Configuration
# MODEL_PATH: Local path to the GGUF model file.
# SERVED_MODEL_NAME: Model name exposed by the vLLM API server.
MODEL_PATH="${MODEL_DIR}/Mistral-v0.3-Instruct-Matter-Slim-A-GGUF/Mistral-v0.3-Instruct-Matter-Slim-A.Q4_K_M.gguf"
SERVED_MODEL_NAME="Mistralv0.3-Instruct"

# Chat Template
CHAT_TEMPLATE="${SCRIPT_DIR}/mistral_inst_chat_template.jinja"

# Server
HOST="0.0.0.0"
PORT=8001

# GPU Configuration
CUDA_VISIBLE_DEVICES="0"
GPU_MEMORY_UTILIZATION=0.85
MAX_MODEL_LEN=16384


# ============================================================
# 2. Environment Variables
# ============================================================

export CUDA_VISIBLE_DEVICES


# ============================================================
# 3. Initialization
# ============================================================

# Validate required files
if [[ ! -f "$MODEL_PATH" ]]; then
  echo "[vllm] model file not found: $MODEL_PATH" >&2
  exit 1
fi

if [[ ! -f "$CHAT_TEMPLATE" ]]; then
  echo "[vllm] chat template not found: $CHAT_TEMPLATE" >&2
  exit 1
fi

echo "[vllm] model                 : $MODEL_PATH"
echo "[vllm] served model name     : $SERVED_MODEL_NAME"
echo "[vllm] chat template         : $CHAT_TEMPLATE"
echo "[vllm] endpoint              : http://$HOST:$PORT/v1/chat/completions"
echo "[vllm] CUDA_VISIBLE_DEVICES  : $CUDA_VISIBLE_DEVICES"
echo "[vllm] gpu memory utilization: $GPU_MEMORY_UTILIZATION"
echo "[vllm] max model length      : $MAX_MODEL_LEN"


# ============================================================
# 4. vLLM Server
# ============================================================

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --chat-template "$CHAT_TEMPLATE" \
  --host "$HOST" \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
