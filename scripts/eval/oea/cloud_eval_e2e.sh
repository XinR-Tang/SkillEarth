
#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 1. Configuration
# ============================================================

# Paths
SKILLEARTH_ROOT="Path/to/SkillEarth"
CACHE_DIR="${SKILLEARTH_ROOT}/dataset/oea/data/gpkgs"
MODEL_NAME="gpt-5.6-terra"

# Skill Bank
SKILLEARTH_USE_SKILL="1"
TOP_K=1
MIN_SCORE=0.0
EMBEDDINGS_PROVIDER="hashing"

# API Model
API_BASE_URL="https://api.example.com/v1"
API_KEY="api_key"
SKILLEARTH_GENERIC_LLM_URL="${API_BASE_URL}"
SKILLEARTH_GENERIC_LLM_MODEL="${MODEL_NAME}"
SKILLEARTH_GENERIC_API_KEY="${API_KEY}"
SKILLEARTH_TIMEOUT_S=600
SKILLEARTH_GENERIC_MAX_TOKENS=1024
SKILLEARTH_GENERIC_MAX_INPUT_CHARS=24000

# Model Inference
PLANNER="llm"
TEMPERATURE=0.0
RETRIES=8
MAX_ROUNDS=15
BATCH_SIZE=1

# Evaluation
MODEL="skillearth"
TASK_NAME="rsagent"
TASK_TYPE="e2e"
VERBOSITY="INFO"
CONTROLLER_ADDR="http://127.0.0.1:20001"
EVAL_BASE_URL="${API_BASE_URL}"
EVAL_API_KEY="${API_KEY}"
QT_QPA_PLATFORM="offscreen"

# Output Paths
SKILLBANK_DIR="${SKILLEARTH_ROOT}/skillbank-oea/${MODEL_NAME}"
RESULT_ROOT="${SKILLEARTH_ROOT}/output/oea/${MODEL_NAME}/e2e"
OUT_DIR="${RESULT_ROOT}/eval_e2e_skillearth"
CKPT_DIR="${RESULT_ROOT}/infer_e2e_skillearth"
CKPT="${CKPT_DIR}/skillearth_rsagent_e2e_ckpt.jsonl"
TOOLS_OUT_DIR="${RESULT_ROOT}/tools_output"


# ============================================================
# 2. Environment Variables
# ============================================================

cd "$SKILLEARTH_ROOT"

export SKILLEARTH_USE_SKILL
export SKILLEARTH_GENERIC_LLM_URL
export SKILLEARTH_GENERIC_LLM_MODEL
export SKILLEARTH_GENERIC_API_KEY
export SKILLEARTH_TIMEOUT_S
export SKILLEARTH_GENERIC_MAX_TOKENS
export SKILLEARTH_GENERIC_MAX_INPUT_CHARS
export EVAL_BASE_URL
export EVAL_API_KEY
export QT_QPA_PLATFORM


# ============================================================
# 3. Initialization
# ============================================================

mkdir -p "$OUT_DIR" "$CKPT_DIR" "$TOOLS_OUT_DIR"

echo "[eval-close-e2e] skill bank : $SKILLBANK_DIR"
echo "[eval-close-e2e] API endpoint: $API_BASE_URL"
echo "[eval-close-e2e] output     : $OUT_DIR"
echo "[eval-close-e2e] checkpoint : $CKPT"


# ============================================================
# 4. Evaluation
# ============================================================

python -m tool_server.oea.tf_eval \
  --model "$MODEL" \
  --model_args "pretrained=${MODEL_NAME},skillbank_dir=${SKILLBANK_DIR},planner=${PLANNER},top_k=${TOP_K},min_score=${MIN_SCORE},embeddings_provider=${EMBEDDINGS_PROVIDER},temperature=${TEMPERATURE},retries=${RETRIES}" \
  --task_name "$TASK_NAME" \
  --task_type "$TASK_TYPE" \
  --verbosity "$VERBOSITY" \
  --output_path "$OUT_DIR" \
  --batch_size "$BATCH_SIZE" \
  --max_rounds "$MAX_ROUNDS" \
  --controller_addr "$CONTROLLER_ADDR" \
  --save_to_ckpt "rsagent=$CKPT" \
  --resume_from_ckpt "rsagent=$CKPT" \
  --cache_dir "$CACHE_DIR" \
  --save_dir "$TOOLS_OUT_DIR"
