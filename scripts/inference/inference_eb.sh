#!/usr/bin/env bash
set -euo pipefail

# Run scripts/tools/run_tools_eb.sh first so the five Earth-Bench MCP services are available.
# Activate the tool-server-e1 environment before running this script.

# ============================================================
# 1. Configuration
# ============================================================

PROJECT_ROOT="Path/to/SkillEarth"
MODEL="qwen3-4b-instruct-2507"
SKILLBANK="${PROJECT_ROOT}/skillbank-eb/${MODEL}"
MCP_CONFIG="${PROJECT_ROOT}/tool_server/eb/config/config_qwen3.json"

# Example question and image path for testing.
QUESTION="Detect and segment the buildings in the image, and calculate their total built-up area in square meters (GSD = 0.20 m/px)."
IMAGE_PATH="${PROJECT_ROOT}/test/image/socal-fire_00000031_pre_disaster.png"

OUTPUT_DIR="${PROJECT_ROOT}/test/results_eb"
TRACE_PATH="${OUTPUT_DIR}/agent_trace.json"

SKILL_TOP_K=1
MAX_ROUNDS=20
TIMEOUT=900

# ============================================================
# 2. Initialization
# ============================================================

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR"

echo "[inference-eb] model      : $MODEL"
echo "[inference-eb] skill bank : $SKILLBANK"
echo "[inference-eb] image      : $IMAGE_PATH"
echo "[inference-eb] output     : $TRACE_PATH"

# ============================================================
# 3. Inference
# ============================================================

python inference/inference_eb.py \
  --question "$QUESTION" \
  --image-path "$IMAGE_PATH" \
  --skillbank "$SKILLBANK" \
  --config-path "$MCP_CONFIG" \
  --output-json "$TRACE_PATH" \
  --skill-top-k "$SKILL_TOP_K" \
  --max-rounds "$MAX_ROUNDS" \
  --timeout "$TIMEOUT"
