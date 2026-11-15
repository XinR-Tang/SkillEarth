
#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 1. Configuration
# ============================================================

# Paths
ROOT_DIR="Path/to/SkillEarth"
MODEL_DIR="${ROOT_DIR}/model"

# Model Configuration
# Supported models:
#   1. qwen2.5-7b-instruct
#      MODEL_PATH="${MODEL_DIR}/Qwen2.5-7B-Instruct"
#      SERVED_MODEL_NAME="qwen2.5-7b-instruct"
#
#   2. qwen3-4b-instruct-2507
#      MODEL_PATH="${MODEL_DIR}/Qwen3-4B-Instruct-2507"
#      SERVED_MODEL_NAME="qwen3-4b-instruct-2507"
#
# MODEL_PATH: Local directory containing the model weights.
# SERVED_MODEL_NAME: Model name exposed by the vLLM API server.
MODEL_PATH="${MODEL_DIR}/Qwen3-4B-Instruct-2507"
SERVED_MODEL_NAME="qwen3-4b-instruct-2507"

# Server
HOST="0.0.0.0"
PORT=8001

# GPU Configuration
CUDA_VISIBLE_DEVICES="7"
GPU_MEMORY_UTILIZATION=0.90 # 0.85 for oea, 0.90 for eb
MAX_MODEL_LEN=98304 # 16384 for oea, 98304 for eb

# Tool Calling
TOOL_CALL_PARSER="hermes"


# ============================================================
# 2. Environment Variables
# ============================================================

export CUDA_VISIBLE_DEVICES


# ============================================================
# 3. Initialization
# ============================================================

echo "[vllm] model                : $MODEL_PATH"
echo "[vllm] served model name    : $SERVED_MODEL_NAME"
echo "[vllm] endpoint             : http://$HOST:$PORT/v1/chat/completions"
echo "[vllm] CUDA_VISIBLE_DEVICES : $CUDA_VISIBLE_DEVICES"
echo "[vllm] gpu memory utilization: $GPU_MEMORY_UTILIZATION"
echo "[vllm] max model length     : $MAX_MODEL_LEN"


# ============================================================
# 4. vLLM Server
# ============================================================

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --enable-auto-tool-choice \
  --tool-call-parser "$TOOL_CALL_PARSER"
