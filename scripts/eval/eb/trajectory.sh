#!/usr/bin/env bash
set -euo pipefail

# Generate Earth-Bench trajectories for one evaluation model and mode.
ROOT_DIR="Path/to/SkillEarth"
MODEL="qwen3-4b-instruct-2507"
METHOD="skillearth"
AUTO_PLANNING=true
NUM_QUESTIONS=249

# Derive the output mode from the evaluation setting.
if [[ "$AUTO_PLANNING" == "true" ]]; then
  MODE="AP"
else
  MODE="IF"
fi

SKILLBANK_DIR="${ROOT_DIR}/skillbank-eb/${MODEL}"
EARTHBENCH_ROOT="${ROOT_DIR}/datasets/eb/Earth-Bench"
CONFIG_PATH="${ROOT_DIR}/tool_server/eb/config/config_qwen3.json"
RUN_DIR="${ROOT_DIR}/output/eb/${MODEL}/${MODE}"

export SKILLBANK_DIR
export EARTHBENCH_ROOT

mkdir -p "$RUN_DIR"

echo "[eb-trajectory] model  : $MODEL"
echo "[eb-trajectory] mode   : $MODE"
echo "[eb-trajectory] output : $RUN_DIR"

python3 "${ROOT_DIR}/tool_server/eb/langchain_qwen.py" \
  --config_path "$CONFIG_PATH" \
  --num_questions "$NUM_QUESTIONS" \
  --method_name "$METHOD" \
  --output_dir "$RUN_DIR" \
  --autoplanning "$AUTO_PLANNING"
