#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
export VLLM_USE_V1="${VLLM_USE_V1:-0}"

AGENT_MODEL="${AGENT_MODEL:-/home/work/GFlowPO/models/Meta-Llama-3-8B-Instruct}"
EVAL_MODEL="${EVAL_MODEL:-/home/work/GFlowPO/models/Meta-Llama-3-8B-Instruct}"
CACHE_DIR="${CACHE_DIR:-/home/work/GFlowPO/jihun/.cache/huggingface}"
EXP_NAME="${EXP_NAME:-gflowpo_reasoning_object_counting_ref}"

python3 -m junmo.train \
  --task bbii_tg \
  --dataset object_counting \
  --agent_model "$AGENT_MODEL" \
  --eval_model "$EVAL_MODEL" \
  --cache_dir "$CACHE_DIR" \
  --agent_device "${AGENT_DEVICE:-cuda:1}" \
  --tp_size "${TP_SIZE:-1}" \
  --eval_gpu_memory_utilization "${EVAL_GPU_MEMORY_UTILIZATION:-0.9}" \
  --eval_max_num_seqs "${EVAL_MAX_NUM_SEQS:-8}" \
  --eval_max_seq_len_to_capture "${EVAL_MAX_SEQ_LEN_TO_CAPTURE:-3072}" \
  --train_steps "${TRAIN_STEPS:-200}" \
  --offline_start_step "${OFFLINE_START_STEP:-100}" \
  --eval_period "${EVAL_PERIOD:-100}" \
  --batch_size "${BATCH_SIZE:-4}" \
  --grad_acc_steps "${GRAD_ACC_STEPS:-4}" \
  --max_prompt_length "${MAX_PROMPT_LENGTH:-150}" \
  --num_example "${NUM_EXAMPLE:-5}" \
  --gamma "${GAMMA:-1.0}" \
  --reward acc \
  --ema_decay "${EMA_DECAY:-0.99}" \
  --lr "${LR:-1e-4}" \
  --lora_r "${LORA_R:-16}" \
  --lora_alpha "${LORA_ALPHA:-32}" \
  --lora_dropout "${LORA_DROPOUT:-0.05}" \
  --reward_epsilon "${REWARD_EPSILON:-1e-8}" \
  --temp_low "${TEMP_LOW:-0.5}" \
  --temp_high "${TEMP_HIGH:-2.0}" \
  --lm_sched_start "${LM_SCHED_START:-1.0}" \
  --lm_sched_end "${LM_SCHED_END:-1.0}" \
  --lm_sched_horizon "${LM_SCHED_HORIZON:-1000}" \
  --use_offline_sampling \
  --online_ratio "${ONLINE_RATIO:-0.5}" \
  --condition_buffer \
  --m_step \
  --train_buffer_max_size "${TRAIN_BUFFER_MAX_SIZE:-1000}" \
  --seed "${SEED:-2}" \
  --wandb_mode "${WANDB_MODE:-online}" \
  --wandb_project "${WANDB_PROJECT:-gflowpo_reasoning_bbh5}" \
  --exp_name "$EXP_NAME" \
  --bbh_train_size "${BBH_TRAIN_SIZE:-50}" \
  --bbh_test_size "${BBH_TEST_SIZE:-100}" \
  --prior_reduction "${PRIOR_REDUCTION:-mean}" \
  --bbh_reasoning_max_tokens "${BBH_REASONING_MAX_TOKENS:-1024}" \
  --bbh_answer_max_tokens "${BBH_ANSWER_MAX_TOKENS:-1}" \
  --condition_queue_samples "${CONDITION_QUEUE_SAMPLES:-1}" \
  --condition_buffer_samples "${CONDITION_BUFFER_SAMPLES:-1}" \
  --prior_chunk_size "${PRIOR_CHUNK_SIZE:-2}" \
  --policy_eval_num_samples "${POLICY_EVAL_NUM_SAMPLES:-4}" \
  --policy_eval_start_step "${POLICY_EVAL_START_STEP:-100}" \
  --policy_eval_period "${POLICY_EVAL_PERIOD:-10}"
