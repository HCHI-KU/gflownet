#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
SUMMARY_FILE="$REPO_DIR/logs/debugging_gflownet_variants_bbh5_${RUN_TS}.log"
mkdir -p "$REPO_DIR/logs"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export AGENT_DEVICE="${AGENT_DEVICE:-cuda:1}"
export AGENT_MODEL="${AGENT_MODEL:-/home/work/GFlowPO/models/Meta-Llama-3-8B-Instruct}"
export EVAL_MODEL="${EVAL_MODEL:-/home/work/GFlowPO/models/Meta-Llama-3-8B-Instruct}"
export CACHE_DIR="${CACHE_DIR:-$HOME/.cache/huggingface}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-debugging_gflownet_bbh5_variants}"
export TRAIN_STEPS="${TRAIN_STEPS:-200}"
export BBH_TRAIN_SIZE="${BBH_TRAIN_SIZE:-50}"
export BBH_TEST_SIZE="${BBH_TEST_SIZE:-100}"
export TEST_EVAL_PERIOD="${TEST_EVAL_PERIOD:-10}"
export EXPORT_EVERY="${EXPORT_EVERY:-10}"
export LOG_EVERY="${LOG_EVERY:-10}"
export EVAL_GPU_MEMORY_UTILIZATION="${EVAL_GPU_MEMORY_UTILIZATION:-0.9}"
export EVAL_MAX_NUM_SEQS="${EVAL_MAX_NUM_SEQS:-8}"
export EVAL_MAX_SEQ_LEN_TO_CAPTURE="${EVAL_MAX_SEQ_LEN_TO_CAPTURE:-3072}"
export EVAL_CHUNK_SIZE="${EVAL_CHUNK_SIZE:-32}"
export BBH_REASONING_MAX_TOKENS="${BBH_REASONING_MAX_TOKENS:-50}"
export BBH_ANSWER_MAX_TOKENS="${BBH_ANSWER_MAX_TOKENS:-1}"
export PRIOR_CHUNK_SIZE="${PRIOR_CHUNK_SIZE:-2}"
export CONDITION_QUEUE_SAMPLES="${CONDITION_QUEUE_SAMPLES:-1}"
export CONDITION_BUFFER_SAMPLES="${CONDITION_BUFFER_SAMPLES:-1}"

TASK_SPECS=(
  "object_counting:4:4:150:2"
  "causal_judgment:2:8:150:0"
  "movie_recommendation:4:4:150:0"
  "hyperbaton:4:4:150:0"
  "tracking_shuffled_objects_five_objects:2:8:150:0"
)

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$SUMMARY_FILE"
}

log "debugging_gflownet_variants BBH5 run started"
log "GPU restriction: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES (do not use GPUs 2,3)"
log "Agent device: $AGENT_DEVICE"

for spec in "${TASK_SPECS[@]}"; do
  IFS=':' read -r task_name batch_size grad_acc_steps max_prompt_length seed <<< "$spec"
  task_log="$REPO_DIR/logs/debugging_gflownet_variants_${task_name}_${RUN_TS}.log"

  log "START task=$task_name bs=$batch_size ga=$grad_acc_steps max_prompt_length=$max_prompt_length seed=$seed"
  set +e
  TASK_NAME="$task_name" \
  TASK_TAG="$task_name" \
  BATCH_SIZE="$batch_size" \
  GRAD_ACC_STEPS="$grad_acc_steps" \
  MAX_PROMPT_LENGTH="$max_prompt_length" \
  SEED="$seed" \
  RUN_TAG="${RUN_TS}_${task_name}" \
  bash "$REPO_DIR/scripts/run_debugging_gflownet_variants_200.sh" 2>&1 | tee "$task_log"
  status=${PIPESTATUS[0]}
  set -e

  if [ "$status" -ne 0 ]; then
    log "FAIL task=$task_name status=$status log=$task_log"
  else
    log "DONE task=$task_name log=$task_log"
  fi
done

log "debugging_gflownet_variants BBH5 run finished"
