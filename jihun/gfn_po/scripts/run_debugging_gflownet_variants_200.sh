#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/source_workspace_env.sh"
cd "$REPO_DIR"

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"

if [ -n "${TASK_NAMES:-}" ]; then
  read -r -a TASK_NAMES_ARR <<< "$TASK_NAMES"
else
  TASK_NAMES_ARR=("${TASK_NAME:-object_counting}")
fi

TASK_TAG="${TASK_TAG:-$(IFS=+; echo "${TASK_NAMES_ARR[*]}")}"
TRAIN_STEPS="${TRAIN_STEPS:-200}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-debugging_gflownet}"
SAVE_DIR="${SAVE_DIR:-$REPO_DIR/debugging_gflownet_runs}"

AGENT_MODEL="${AGENT_MODEL:-$GFN_PO_DEFAULT_MODEL_DIR}"
EVAL_MODEL="${EVAL_MODEL:-$GFN_PO_DEFAULT_MODEL_DIR}"
CACHE_DIR="${CACHE_DIR:-$GFN_PO_DEFAULT_HF_CACHE}"

BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-4}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-150}"
NUM_EXAMPLE="${NUM_EXAMPLE:-5}"
FEWSHOT_STRATEGY="${FEWSHOT_STRATEGY:-first}"
FEWSHOT_RESAMPLE_EACH_STEP="${FEWSHOT_RESAMPLE_EACH_STEP:-0}"
SEED="${SEED:-42}"
TP_SIZE="${TP_SIZE:-1}"
AGENT_DEVICE="${AGENT_DEVICE:-cuda:0}"
EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEMORY_UTILIZATION:-0.9}"
EVAL_MAX_NUM_SEQS="${EVAL_MAX_NUM_SEQS:-64}"
EVAL_MAX_SEQ_LEN_TO_CAPTURE="${EVAL_MAX_SEQ_LEN_TO_CAPTURE:-3072}"
EVAL_CHUNK_SIZE="${EVAL_CHUNK_SIZE:-32}"
BBH_TRAIN_SIZE="${BBH_TRAIN_SIZE:-50}"
BBH_TEST_SIZE="${BBH_TEST_SIZE:-100}"
BBH_REASONING_MAX_TOKENS="${BBH_REASONING_MAX_TOKENS:-50}"
BBH_ANSWER_MAX_TOKENS="${BBH_ANSWER_MAX_TOKENS:-1}"
TEST_EVAL_PERIOD="${TEST_EVAL_PERIOD:-0}"
EXPORT_EVERY="${EXPORT_EVERY:-100}"
LOG_EVERY="${LOG_EVERY:-10}"

ONLINE_RATIO="${ONLINE_RATIO:-0.5}"
OFFLINE_START_STEP="${OFFLINE_START_STEP:-100}"
TRAIN_BUFFER_MAX_SIZE="${TRAIN_BUFFER_MAX_SIZE:-1000}"
CONDITION_QUEUE_SAMPLES="${CONDITION_QUEUE_SAMPLES:-1}"
CONDITION_BUFFER_SAMPLES="${CONDITION_BUFFER_SAMPLES:-1}"
PRIOR_CHUNK_SIZE="${PRIOR_CHUNK_SIZE:-2}"
PRIOR_REDUCTION="${PRIOR_REDUCTION:-sum}"
GAMMA="${GAMMA:-1.0}"
VARIANTS="${VARIANTS:-v1_log_prior v2_log_prior_buffer v3_log_prior_buffer_condition}"

run_variant() {
  local name="$1"
  local script_path="$2"
  shift 2

  echo "[$(date '+%F %T')] START ${name}"
  local -a extra_args=("$@")
  if [ "$FEWSHOT_RESAMPLE_EACH_STEP" = "1" ]; then
    extra_args+=("--fewshot_resample_each_step")
  fi
  python3 "$script_path" \
    --task_names "${TASK_NAMES_ARR[@]}" \
    --train_steps "$TRAIN_STEPS" \
    --run_name "${name}_${TASK_TAG}_${RUN_TAG}" \
    --save_dir "$SAVE_DIR" \
    --wandb_mode "$WANDB_MODE" \
    --wandb_project "$WANDB_PROJECT" \
    --agent_model "$AGENT_MODEL" \
    --eval_model "$EVAL_MODEL" \
    --cache_dir "$CACHE_DIR" \
    --batch_size "$BATCH_SIZE" \
    --grad_acc_steps "$GRAD_ACC_STEPS" \
    --max_prompt_length "$MAX_PROMPT_LENGTH" \
    --num_example "$NUM_EXAMPLE" \
    --fewshot_strategy "$FEWSHOT_STRATEGY" \
    --seed "$SEED" \
    --agent_device "$AGENT_DEVICE" \
    --tp_size "$TP_SIZE" \
    --eval_gpu_memory_utilization "$EVAL_GPU_MEMORY_UTILIZATION" \
    --eval_max_num_seqs "$EVAL_MAX_NUM_SEQS" \
    --eval_max_seq_len_to_capture "$EVAL_MAX_SEQ_LEN_TO_CAPTURE" \
    --eval_chunk_size "$EVAL_CHUNK_SIZE" \
    --bbh_train_size "$BBH_TRAIN_SIZE" \
    --bbh_test_size "$BBH_TEST_SIZE" \
    --bbh_reasoning_max_tokens "$BBH_REASONING_MAX_TOKENS" \
    --bbh_answer_max_tokens "$BBH_ANSWER_MAX_TOKENS" \
    --test_eval_period "$TEST_EVAL_PERIOD" \
    --export_every "$EXPORT_EVERY" \
    --log_every "$LOG_EVERY" \
    --gamma "$GAMMA" \
    --prior_chunk_size "$PRIOR_CHUNK_SIZE" \
    --prior_reduction "$PRIOR_REDUCTION" \
    "${extra_args[@]}"
  echo "[$(date '+%F %T')] END ${name}"
}

for variant_name in $VARIANTS; do
  case "$variant_name" in
    v1_log_prior)
      run_variant "v1_log_prior" \
        "$REPO_DIR/debugging_gflownet_v1_log_prior.py"
      ;;
    v2_log_prior_buffer)
      run_variant "v2_log_prior_buffer" \
        "$REPO_DIR/debugging_gflownet_v2_log_prior_buffer.py" \
        --use_offline_sampling \
        --online_ratio "$ONLINE_RATIO" \
        --offline_start_step "$OFFLINE_START_STEP" \
        --train_buffer_max_size "$TRAIN_BUFFER_MAX_SIZE"
      ;;
    v3_log_prior_buffer_condition)
      run_variant "v3_log_prior_buffer_condition" \
        "$REPO_DIR/debugging_gflownet_v3_log_prior_buffer_condition.py" \
        --use_offline_sampling \
        --online_ratio "$ONLINE_RATIO" \
        --offline_start_step "$OFFLINE_START_STEP" \
        --train_buffer_max_size "$TRAIN_BUFFER_MAX_SIZE" \
        --condition_buffer \
        --condition_queue_samples "$CONDITION_QUEUE_SAMPLES" \
        --condition_buffer_samples "$CONDITION_BUFFER_SAMPLES"
      ;;
    *)
      echo "Unknown variant: $variant_name" >&2
      exit 1
      ;;
  esac
done
