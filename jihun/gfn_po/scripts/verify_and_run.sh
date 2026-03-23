#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/source_workspace_env.sh"
cd "$REPO_DIR"

if [ -f "/home/work/GFlowPO/anaconda3/etc/profile.d/conda.sh" ]; then
  source "/home/work/GFlowPO/anaconda3/etc/profile.d/conda.sh"
elif [ -f "${ORIGINAL_HOME:-$HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  source "${ORIGINAL_HOME:-$HOME}/miniconda3/etc/profile.d/conda.sh"
else
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi

export CONDA_NO_PLUGINS=true
conda activate rd_test
export HF_HOME="${HF_HOME:-$GFN_PO_DEFAULT_HF_CACHE}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_TS="$(date +%Y%m%d_%H%M%S)"
ORCH_LOG="$REPO_DIR/logs/verify_and_run_${RUN_TS}.log"
mkdir -p "$REPO_DIR/logs"

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$ORCH_LOG"
}

wait_for_train_clear() {
  log "Waiting for existing junmo.train jobs to finish before validation run"
  while ps -eo cmd | grep -F "python -m junmo.train" | grep -v grep >/dev/null; do
    ps -eo pid,etimes,cmd | grep -F "python -m junmo.train" | grep -v grep | tee -a "$ORCH_LOG"
    sleep 60
  done
}

run_smoke_task() {
  local cuda_devices="$1"
  local task_mode="$2"
  local dataset_name="$3"
  local batch_size="$4"
  local grad_acc_steps="$5"
  local seed="$6"
  local lm_sched_horizon="$7"
  local exp_name="$8"
  local task_log="$REPO_DIR/logs/${exp_name}_${RUN_TS}.log"

  log "Starting smoke task=${dataset_name} gpus=${cuda_devices} bs=${batch_size} ga=${grad_acc_steps}"
  set +e
  CUDA_VISIBLE_DEVICES="$cuda_devices" python -m junmo.train \
    --task "$task_mode" \
    --dataset "$dataset_name" \
    --agent_model "${AGENT_MODEL:-$GFN_PO_DEFAULT_MODEL_DIR}" \
    --eval_model "${EVAL_MODEL:-$GFN_PO_DEFAULT_MODEL_DIR}" \
    --cache_dir "${CACHE_DIR:-$GFN_PO_DEFAULT_HF_CACHE}" \
    --agent_device cuda:1 \
    --tp_size 1 \
    --eval_gpu_memory_utilization 0.9 \
    --train_steps 20 \
    --offline_start_step 10 \
    --eval_period 10 \
    --batch_size "$batch_size" \
    --grad_acc_steps "$grad_acc_steps" \
    --max_prompt_length 150 \
    --num_example 5 \
    --gamma 1.0 \
    --reward acc \
    --ema_decay 0.99 \
    --lr 1e-4 \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --reward_epsilon 1e-8 \
    --temp_low 0.5 \
    --temp_high 2.0 \
    --lm_sched_start 1.0 \
    --lm_sched_end 1.0 \
    --lm_sched_horizon "$lm_sched_horizon" \
    --use_offline_sampling \
    --online_ratio 0.5 \
    --condition_buffer \
    --m_step \
    --train_buffer_max_size 1000 \
    --seed "$seed" \
    --wandb_mode online \
    --wandb_project gflowpo_reasoning_bbh5_smoke \
    --exp_name "$exp_name" \
    --bbh_train_size 50 \
    --bbh_test_size 100 \
    --prior_reduction mean \
    --bbh_reasoning_max_tokens 50 \
    --bbh_answer_max_tokens 1 \
    --condition_queue_samples 1 \
    --condition_buffer_samples 1 \
    --prior_chunk_size 2 \
    --policy_eval_num_samples 4 \
    --policy_eval_start_step 10 \
    --policy_eval_period 10 \
    2>&1 | tee "$task_log"
  local status=${PIPESTATUS[0]}
  set -e
  if [ "$status" -ne 0 ]; then
    log "Smoke task failed: ${dataset_name} status=${status}"
    return "$status"
  fi

  local run_dir
  run_dir="$(ls -1dt "$REPO_DIR"/logs/${exp_name}data:${dataset_name}-seed:${seed}-* 2>/dev/null | head -n 1 || true)"
  if [ -z "$run_dir" ]; then
    log "Smoke task missing run_dir for ${dataset_name}"
    return 1
  fi

  for f in train.jsonl eval.jsonl final.jsonl; do
    if [ ! -s "$run_dir/$f" ]; then
      log "Smoke task missing ${f} in ${run_dir}"
      return 1
    fi
  done
  log "Smoke task passed: ${dataset_name} run_dir=${run_dir}"
}

wait_for_train_clear

run_smoke_task "0,1" "bbii_tg" "object_counting" 4 4 2 1000 "smoke_object_counting" &
SMOKE0=$!
run_smoke_task "2,3" "bbii_tc" "causal_judgment" 2 8 0 10 "smoke_causal_judgment" &
SMOKE1=$!
wait "$SMOKE0" "$SMOKE1"

log "Smoke validation passed. Starting canonical full run launcher"
bash "$REPO_DIR/run_bbh5_greater_compare_tmux.sh" 2>&1 | tee -a "$ORCH_LOG"
