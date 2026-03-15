#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
AGENT_MODEL="${AGENT_MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
EVAL_MODEL="${EVAL_MODEL:-$AGENT_MODEL}"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/huggingface}"
SAVE_DIR="${SAVE_DIR:-$REPO_DIR/debugging_gflownet_runs}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-debugging_gflownet}"
RUN_NAME="${RUN_NAME:-object_counting_step300_wandb_$(date +%Y%m%d_%H%M%S)}"

cd "$REPO_DIR"

exec python debugging_gflownet.py \
  --task_names object_counting \
  --agent_model "$AGENT_MODEL" \
  --eval_model "$EVAL_MODEL" \
  --cache_dir "$CACHE_DIR" \
  --save_dir "$SAVE_DIR" \
  --train_steps 300 \
  --batch_size 2 \
  --grad_acc_steps 1 \
  --num_warmup_steps 1 \
  --num_example 5 \
  --fewshot_strategy first \
  --eval_gpu_memory_utilization 0.45 \
  --eval_max_num_seqs 4 \
  --bbh_reasoning_max_tokens 256 \
  --max_prompt_length 80 \
  --test_eval_period 5 \
  --log_every 1 \
  --export_every 1 \
  --wandb_mode "$WANDB_MODE" \
  --wandb_project "$WANDB_PROJECT" \
  --run_name "$RUN_NAME"
