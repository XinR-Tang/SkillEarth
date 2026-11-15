
#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 1. Configuration
# ============================================================

# Paths
PROJECT_ROOT="Path/to/SkillEarth"
MODEL="qwen3-4b-instruct-2507"
SKILLBANK="${PROJECT_ROOT}/skillbank-oea/${MODEL}"
OUTPUT="${PROJECT_ROOT}/test/results_oea"

# Local vLLM
BASE_URL="http://127.0.0.1:8001/v1"
TEMPERATURE=0.0
MAX_TOKENS=1024
TIMEOUT=600
RETRIES=3

# Skill Retrieval
SKILL_TOP_K=1
SKILL_THRESHOLD=0.3
MAX_SKILL_CHARS=20000

# Agent Inference
MAX_ROUNDS=15

# Test Samples
# Example 1: Object Detection
QUERY="Detect the smallest plane on the runway and draw a box on it."
IMAGE_PATH="${PROJECT_ROOT}/test/image/TG_009.jpg"

# Example 2: Burn Severity Analysis
# QUERY="Generate a preview from the NBR difference layer for Istiaia-Aidipsos Municipality, Euboea, Greece between April 2018 and August 2018, highlighting severe, moderate, and low burn severity as well as enhanced regrowth and unburned areas."
# IMAGE_PATH="None"

# Output Paths
OUTPUT_JSON="${OUTPUT}/agent_trace.json"


# ============================================================
# 2. Environment Variables
# ============================================================

cd "$PROJECT_ROOT"


# ============================================================
# 3. Initialization
# ============================================================

mkdir -p "$OUTPUT"

echo "[inference] model      : $MODEL"
echo "[inference] endpoint   : $BASE_URL"
echo "[inference] skill bank : $SKILLBANK"
echo "[inference] question   : $QUERY"
echo "[inference] image      : $IMAGE_PATH"
echo "[inference] output     : $OUTPUT_JSON"

# Optional image input
IMAGE_ARGS=()
if [[ -n "$IMAGE_PATH" && "$IMAGE_PATH" != "None" ]]; then
  IMAGE_ARGS=(--image_path "$IMAGE_PATH")
fi


# ============================================================
# 4. Inference
# ============================================================

python -m inference.inference_oea \
  --base_url "$BASE_URL" \
  --model "$MODEL" \
  --skillbank "$SKILLBANK" \
  --question "$QUERY" \
  --skill_top_k "$SKILL_TOP_K" \
  --skill_threshold "$SKILL_THRESHOLD" \
  --max_skill_chars "$MAX_SKILL_CHARS" \
  --max_rounds "$MAX_ROUNDS" \
  --temperature "$TEMPERATURE" \
  --max_tokens "$MAX_TOKENS" \
  --timeout "$TIMEOUT" \
  --retries "$RETRIES" \
  --output_json "$OUTPUT_JSON" \
  "${IMAGE_ARGS[@]}"
