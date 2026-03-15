#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f "/home/work/jaeyoon/tools/anaconda3/etc/profile.d/conda.sh" ]; then
  source "/home/work/jaeyoon/tools/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
fi

export CONDA_NO_PLUGINS=true
conda activate rd_test

export HF_HOME=/home/work/jaeyoon/.hf_cache
export HUGGINGFACE_HUB_CACHE=/home/work/jaeyoon/.hf_cache/hub
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"
export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUN_TS="$(date +%Y%m%d_%H%M%S)"
SUMMARY_FILE="$SCRIPT_DIR/logs/gflowpo_reasoning_bbh5_summary_${RUN_TS}.txt"
mkdir -p "$SCRIPT_DIR/logs"

echo "gflowpo_reasoning BBH5 canonical run started at $(date)" | tee -a "$SUMMARY_FILE"
echo "Summary file: $SUMMARY_FILE" | tee -a "$SUMMARY_FILE"
echo "Canonical setting: split=50/100, prior_reduction=mean, gamma=1.0" | tee -a "$SUMMARY_FILE"
echo "Sanity setting: policy_eval_num_samples=4 on test100 from step 100 every 10 steps; queue provenance + grad_norm + prompt uniqueness logging enabled." | tee -a "$SUMMARY_FILE"
echo "Worker mapping: W0=GPU0,1 (eval:0 / agent:1), W1=GPU2,3 (eval:2 / agent:3)" | tee -a "$SUMMARY_FILE"
echo "Heavy tasks use batch_size=2 grad_acc_steps=8 to avoid OOM." | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

run_worker() {
  local worker_name="$1"
  local cuda_devices="$2"
  shift 2
  local specs=("$@")

  for spec in "${specs[@]}"; do
    IFS=':' read -r task_label task_mode dataset_name batch_size grad_acc_steps max_prompt_length seed lm_sched_horizon <<< "$spec"

    local task_ts exp_name task_log status run_dir
    task_ts="$(date +%Y%m%d_%H%M%S)"
    exp_name="gflowpo_reasoning_bbh200_${task_label}"
    task_log="$SCRIPT_DIR/logs/${exp_name}_${task_ts}.log"

    echo "[$(date)] START worker=${worker_name} gpus=${cuda_devices} task=${task_label} mode=${task_mode} dataset=${dataset_name} bs=${batch_size} ga=${grad_acc_steps} max_prompt_length=${max_prompt_length}" | tee -a "$SUMMARY_FILE"
    echo "[$(date)] LOG   worker=${worker_name} $task_log" | tee -a "$SUMMARY_FILE"

    set +e
    CUDA_VISIBLE_DEVICES="${cuda_devices}" python -m junmo.train \
      --task "$task_mode" \
      --dataset "$dataset_name" \
      --agent_model /home/work/jaeyoon/models/Meta-Llama-3-8B-Instruct \
      --eval_model /home/work/jaeyoon/models/Meta-Llama-3-8B-Instruct \
      --cache_dir /home/work/jaeyoon/.hf_cache \
      --agent_device cuda:1 \
      --tp_size 1 \
      --eval_gpu_memory_utilization 0.9 \
      --train_steps 200 \
      --offline_start_step 100 \
      --eval_period 100 \
      --batch_size "$batch_size" \
      --grad_acc_steps "$grad_acc_steps" \
      --max_prompt_length "$max_prompt_length" \
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
      --wandb_project gflowpo_reasoning_bbh5 \
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
      --policy_eval_start_step 100 \
      --policy_eval_period 10 \
      2>&1 | tee "$task_log"
    status=${PIPESTATUS[0]}
    set -e

    if [ "$status" -ne 0 ]; then
      echo "[$(date)] FAIL  worker=${worker_name} task=${task_label} status=${status}" | tee -a "$SUMMARY_FILE"
      echo "" | tee -a "$SUMMARY_FILE"
      continue
    fi

    run_dir="$(ls -1dt "$SCRIPT_DIR"/logs/${exp_name}data:${dataset_name}-seed:${seed}-* 2>/dev/null | head -n 1 || true)"
    echo "[$(date)] DONE  worker=${worker_name} task=${task_label} run_dir=${run_dir:-NOT_FOUND}" | tee -a "$SUMMARY_FILE"
    echo "" | tee -a "$SUMMARY_FILE"
  done
}

WORKER0_SPECS=(
  "tracking_shuffled_objects_five_objects:bbii_tc:tracking_shuffled_objects_five_objects:2:8:150:0:10"
  "movie_recommendation:bbii_tc:movie_recommendation:4:4:150:0:10"
  "hyperbaton:bbii_tc:hyperbaton:4:4:150:0:10"
)

WORKER1_SPECS=(
  "causal_judgment:bbii_tc:causal_judgment:2:8:150:0:10"
  "object_counting:bbii_tg:object_counting:4:4:150:2:1000"
)

run_worker "W0" "0,1" "${WORKER0_SPECS[@]}" &
W0_PID=$!
run_worker "W1" "2,3" "${WORKER1_SPECS[@]}" &
W1_PID=$!

wait "$W0_PID" "$W1_PID"

echo "BBH5 greater-compare run finished at $(date)" | tee -a "$SUMMARY_FILE"
