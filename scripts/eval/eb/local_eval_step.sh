#!/usr/bin/env bash
set -euo pipefail

# Evaluate Earth-Bench tool trajectories for one model and mode.
ROOT_DIR="Path/to/SkillEarth"
RESULT_MODEL="qwen3-4b-instruct-2507"
MODE="AP"
RUN_DIR="${ROOT_DIR}/output/eb/${RESULT_MODEL}/${MODE}"
GROUND_TRUTH_FILE="${ROOT_DIR}/datasets/eb/Earth-Bench/extracted_tool_calls_GT.json"

echo "[eb-step] model  : $RESULT_MODEL"
echo "[eb-step] mode   : $MODE"
echo "[eb-step] input  : $RUN_DIR"

cd "$ROOT_DIR"
python3 tool_server/eb/evaluate/step_by_step.py \
  --model_dir "$RUN_DIR" \
  --ground_truth_file "$GROUND_TRUTH_FILE"
