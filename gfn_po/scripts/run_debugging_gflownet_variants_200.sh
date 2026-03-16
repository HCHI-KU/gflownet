#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"

TASK_NAME="${TASK_NAME:-object_counting}"
TRAIN_STEPS="${TRAIN_STEPS:-200}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
WANDB_MODE="${WANDB_MODE:-online}"
SAVE_DIR="${SAVE_DIR:-$REPO_DIR/debugging_gflownet_runs}"

AGENT_MODEL="${AGENT_MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
EVAL_MODEL="${EVAL_MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/huggingface}"

BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-4}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-150}"
NUM_EXAMPLE="${NUM_EXAMPLE:-5}"
SEED="${SEED:-42}"
TP_SIZE="${TP_SIZE:-1}"
EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEMORY_UTILIZATION:-0.9}"
BBH_TRAIN_SIZE="${BBH_TRAIN_SIZE:-50}"
BBH_TEST_SIZE="${BBH_TEST_SIZE:-100}"
TEST_EVAL_PERIOD="${TEST_EVAL_PERIOD:-10}"
EXPORT_EVERY="${EXPORT_EVERY:-10}"
LOG_EVERY="${LOG_EVERY:-10}"

ONLINE_RATIO="${ONLINE_RATIO:-0.5}"
OFFLINE_START_STEP="${OFFLINE_START_STEP:-100}"
TRAIN_BUFFER_MAX_SIZE="${TRAIN_BUFFER_MAX_SIZE:-1000}"
CONDITION_QUEUE_SAMPLES="${CONDITION_QUEUE_SAMPLES:-1}"
CONDITION_BUFFER_SAMPLES="${CONDITION_BUFFER_SAMPLES:-2}"
PRIOR_CHUNK_SIZE="${PRIOR_CHUNK_SIZE:-0}"
PRIOR_REDUCTION="${PRIOR_REDUCTION:-sum}"
GAMMA="${GAMMA:-1.0}"

run_variant() {
  local name="$1"
  local script_path="$2"
  shift 2

  echo "[$(date '+%F %T')] START ${name}"
  python3 "$script_path" \
    --task_names "$TASK_NAME" \
    --train_steps "$TRAIN_STEPS" \
    --run_name "${name}_${TASK_NAME}_${RUN_TAG}" \
    --save_dir "$SAVE_DIR" \
    --wandb_mode "$WANDB_MODE" \
    --agent_model "$AGENT_MODEL" \
    --eval_model "$EVAL_MODEL" \
    --cache_dir "$CACHE_DIR" \
    --batch_size "$BATCH_SIZE" \
    --grad_acc_steps "$GRAD_ACC_STEPS" \
    --max_prompt_length "$MAX_PROMPT_LENGTH" \
    --num_example "$NUM_EXAMPLE" \
    --seed "$SEED" \
    --tp_size "$TP_SIZE" \
    --eval_gpu_memory_utilization "$EVAL_GPU_MEMORY_UTILIZATION" \
    --bbh_train_size "$BBH_TRAIN_SIZE" \
    --bbh_test_size "$BBH_TEST_SIZE" \
    --test_eval_period "$TEST_EVAL_PERIOD" \
    --export_every "$EXPORT_EVERY" \
    --log_every "$LOG_EVERY" \
    --gamma "$GAMMA" \
    --prior_chunk_size "$PRIOR_CHUNK_SIZE" \
    --prior_reduction "$PRIOR_REDUCTION" \
    "$@"
  echo "[$(date '+%F %T')] END ${name}"
}

run_variant "v1_log_prior" \
  "$REPO_DIR/debugging_gflownet_v1_log_prior.py"

run_variant "v2_log_prior_buffer" \
  "$REPO_DIR/debugging_gflownet_v2_log_prior_buffer.py" \
  --use_offline_sampling \
  --online_ratio "$ONLINE_RATIO" \
  --offline_start_step "$OFFLINE_START_STEP" \
  --train_buffer_max_size "$TRAIN_BUFFER_MAX_SIZE"

run_variant "v3_log_prior_buffer_condition" \
  "$REPO_DIR/debugging_gflownet_v3_log_prior_buffer_condition.py" \
  --use_offline_sampling \
  --online_ratio "$ONLINE_RATIO" \
  --offline_start_step "$OFFLINE_START_STEP" \
  --train_buffer_max_size "$TRAIN_BUFFER_MAX_SIZE" \
  --condition_buffer \
  --condition_queue_samples "$CONDITION_QUEUE_SAMPLES" \
  --condition_buffer_samples "$CONDITION_BUFFER_SAMPLES"
