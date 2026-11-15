#!/usr/bin/env bash
set -euo pipefail

# Convert one Earth-Bench run into trajectory and final-answer evaluation files.
ROOT_DIR="Path/to/SkillEarth"
MODEL="qwen3-4b-instruct-2507"
MODE="AP"

# Replace these placeholders with an OpenAI-compatible answer-polishing endpoint.
POLISH_BASE_URL="https://ark.cn-beijing.volces.com/api/plan/v3"
POLISH_API_KEY="your_api_key_here"
POLISH_MODEL="doubao-seed-2.0-mini"

RUN_DIR="${ROOT_DIR}/output/eb/${MODEL}/${MODE}"
LOG_FILE="${RUN_DIR}/${MODEL}_${MODE}_langchain.log"
RESULTS_FILE="${RUN_DIR}/results_summary.json"
TOOL_CALLS_FILE="${RUN_DIR}/extracted_tool_calls.json"
EXTRACTOR="${ROOT_DIR}/tool_server/eb/process/extract_tool_Model.py"
POLISHER="${ROOT_DIR}/tool_server/eb/process/polish_answer.py"

if [[ ! -f "$LOG_FILE" || ! -f "$RESULTS_FILE" ]]; then
  echo "[error] Missing trajectory outputs in: $RUN_DIR" >&2
  exit 1
fi

if [[ "$POLISH_API_KEY" == "api_key" ]]; then
  echo "[error] Set POLISH_BASE_URL, POLISH_API_KEY, and POLISH_MODEL before processing." >&2
  exit 1
fi

echo "[eb-process] model  : $MODEL"
echo "[eb-process] mode   : $MODE"
echo "[eb-process] input  : $RUN_DIR"

python3 "$EXTRACTOR" \
  --log_file "$LOG_FILE" \
  --output_file "$TOOL_CALLS_FILE"

python3 "$POLISHER" \
  --base_dir "$RUN_DIR" \
  --log_file "$LOG_FILE" \
  --base_url "$POLISH_BASE_URL" \
  --api_key "$POLISH_API_KEY" \
  --model "$POLISH_MODEL"
