#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_DIR="$(cd "$REPO_DIR/.." && pwd)"
SHARED_ROOT="$(cd "$WORKSPACE_DIR/.." && pwd)"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"

MODEL_CANDIDATE="$SHARED_ROOT/models/Meta-Llama-3-8B-Instruct"
DEFAULT_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
if [[ -d "$MODEL_CANDIDATE" ]]; then
  DEFAULT_MODEL="$MODEL_CANDIDATE"
fi

TASKS="${TASKS:-causal_judgement movie_recommendation hyperbaton tracking_shuffled_objects_five_objects}"
SAVE_DIR="${SAVE_DIR:-$REPO_DIR/debugging_gflownet_runs}"
WANDB_PROJECT="${WANDB_PROJECT:-debugging_gflownet_object_counting_gamma_sweep}"
TRAIN_STEPS="${TRAIN_STEPS:-300}"
GAMMA="${GAMMA:-0.001}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-2}"
FEWSHOT_STRATEGY="${FEWSHOT_STRATEGY:-random}"
FEWSHOT_RESAMPLE_EACH_STEP="${FEWSHOT_RESAMPLE_EACH_STEP:-0}"
EVAL_MAX_NUM_SEQS="${EVAL_MAX_NUM_SEQS:-64}"
TEST_EVAL_PERIOD="${TEST_EVAL_PERIOD:-0}"
EXPORT_EVERY="${EXPORT_EVERY:-100}"
AGENT_MODEL="${AGENT_MODEL:-$DEFAULT_MODEL}"
EVAL_MODEL="${EVAL_MODEL:-$DEFAULT_MODEL}"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/huggingface}"

cd "$REPO_DIR"

for task in $TASKS; do
  echo "[$(date '+%F %T')] START task=$task"
  TASK_NAME="$task" \
  TASK_TAG="$task" \
  VARIANTS="v1_log_prior" \
  RUN_TAG="${RUN_TS}_${task}" \
  SAVE_DIR="$SAVE_DIR" \
  WANDB_PROJECT="$WANDB_PROJECT" \
  TRAIN_STEPS="$TRAIN_STEPS" \
  GAMMA="$GAMMA" \
  BATCH_SIZE="$BATCH_SIZE" \
  GRAD_ACC_STEPS="$GRAD_ACC_STEPS" \
  FEWSHOT_STRATEGY="$FEWSHOT_STRATEGY" \
  FEWSHOT_RESAMPLE_EACH_STEP="$FEWSHOT_RESAMPLE_EACH_STEP" \
  EVAL_MAX_NUM_SEQS="$EVAL_MAX_NUM_SEQS" \
  TEST_EVAL_PERIOD="$TEST_EVAL_PERIOD" \
  EXPORT_EVERY="$EXPORT_EVERY" \
  AGENT_MODEL="$AGENT_MODEL" \
  EVAL_MODEL="$EVAL_MODEL" \
  CACHE_DIR="$CACHE_DIR" \
  bash "$SCRIPT_DIR/run_debugging_gflownet_variants_200.sh"
  echo "[$(date '+%F %T')] DONE task=$task"
done
