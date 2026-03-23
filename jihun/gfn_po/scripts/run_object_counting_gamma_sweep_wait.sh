#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

VISIBLE_GPU_IDS="${VISIBLE_GPU_IDS:-0 1}"
GPU_FREE_THRESHOLD_MIB="${GPU_FREE_THRESHOLD_MIB:-1000}"
POLL_SECONDS="${POLL_SECONDS:-30}"
GAMMA_VALUES="${GAMMA_VALUES:-0.7 0.1}"
BASE_RUN_TAG="${BASE_RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"

wait_for_gpus() {
  while true; do
    local ready=1
    while IFS=',' read -r gpu_idx mem_used _util; do
      gpu_idx="${gpu_idx// /}"
      mem_used="${mem_used// /}"
      for wanted in $VISIBLE_GPU_IDS; do
        if [[ "$gpu_idx" == "$wanted" ]] && [[ "${mem_used:-999999}" -ge "$GPU_FREE_THRESHOLD_MIB" ]]; then
          ready=0
        fi
      done
    done < <(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits)

    if [[ "$ready" -eq 1 ]]; then
      break
    fi
    echo "[$(date '+%F %T')] waiting for GPUs ${VISIBLE_GPU_IDS} to become free (< ${GPU_FREE_THRESHOLD_MIB} MiB)"
    sleep "$POLL_SECONDS"
  done
}

run_one() {
  local gamma="$1"
  local gamma_tag="${gamma//./p}"
  echo "[$(date '+%F %T')] starting gamma=${gamma}"

  export TASK_NAME="object_counting"
  export TASK_TAG="object_counting_gamma_${gamma_tag}"
  export VARIANTS="v1_log_prior"
  export RUN_TAG="${BASE_RUN_TAG}_gamma_${gamma_tag}"

  export TRAIN_STEPS="${TRAIN_STEPS:-120}"
  export BATCH_SIZE="${BATCH_SIZE:-4}"
  export GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-2}"
  export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-150}"
  export NUM_EXAMPLE="${NUM_EXAMPLE:-5}"
  export FEWSHOT_STRATEGY="${FEWSHOT_STRATEGY:-random}"
  export FEWSHOT_RESAMPLE_EACH_STEP="${FEWSHOT_RESAMPLE_EACH_STEP:-1}"
  export SEED="${SEED:-2}"

  export BBH_TRAIN_SIZE="${BBH_TRAIN_SIZE:-50}"
  export BBH_TEST_SIZE="${BBH_TEST_SIZE:-100}"
  export BBH_REASONING_MAX_TOKENS="${BBH_REASONING_MAX_TOKENS:-1024}"
  export BBH_ANSWER_MAX_TOKENS="${BBH_ANSWER_MAX_TOKENS:-1}"
  export TEST_EVAL_PERIOD="${TEST_EVAL_PERIOD:-10}"
  export EXPORT_EVERY="${EXPORT_EVERY:-10}"
  export LOG_EVERY="${LOG_EVERY:-10}"

  export PRIOR_REDUCTION="${PRIOR_REDUCTION:-mean}"
  export PRIOR_CHUNK_SIZE="${PRIOR_CHUNK_SIZE:-2}"
  export GAMMA="$gamma"
  export WANDB_PROJECT="${WANDB_PROJECT:-debugging_gflownet_object_counting_gamma_sweep}"

  bash "$SCRIPT_DIR/run_object_counting_logprior_gfnpo_style.sh"
  echo "[$(date '+%F %T')] finished gamma=${gamma}"
}

wait_for_gpus
for gamma in $GAMMA_VALUES; do
  run_one "$gamma"
done
