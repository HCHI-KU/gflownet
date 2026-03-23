#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"

QUEUE_TASKS="${QUEUE_TASKS:-movie_recommendation hyperbaton}" \
QUEUE_TAG="${QUEUE_TAG:-followup_local_${RUN_TS}}" \
SESSION_NAME="${SESSION_NAME:-gfn_followup_local_${RUN_TS}}" \
QUEUE_LOG="${QUEUE_LOG:-$REPO_DIR/logs/gfn_followup_local_${RUN_TS}.log}" \
VARIANTS="v1_log_prior" \
TRAIN_STEPS="${TRAIN_STEPS:-300}" \
GAMMA="${GAMMA:-0.001}" \
BATCH_SIZE="${BATCH_SIZE:-4}" \
GRAD_ACC_STEPS="${GRAD_ACC_STEPS:-2}" \
FEWSHOT_STRATEGY="${FEWSHOT_STRATEGY:-random}" \
FEWSHOT_RESAMPLE_EACH_STEP="${FEWSHOT_RESAMPLE_EACH_STEP:-1}" \
WANDB_PROJECT="${WANDB_PROJECT:-debugging_gflownet_object_counting_gamma_sweep}" \
bash "$SCRIPT_DIR/run_debugging_gflownet_task_queue_tmux.sh"
