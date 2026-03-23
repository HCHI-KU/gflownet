#!/usr/bin/env bash
set -euo pipefail

NUM_META_PROMPTS="${1:-1000}"
RUN_PREFIX="${2:-meta${NUM_META_PROMPTS}_train_test_newgen_all5_$(date +%Y%m%d_%H%M%S)}"
META_PROMPT_SEED="${3:-0}"
META_PROMPT_GENERATION_MAX_ROUNDS="${4:-6}"
META_PROMPT_GENERATION_BATCH_SIZE="${5:-128}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SAVE_ROOT="$SCRIPT_DIR/save"
RUN_DIR="$SAVE_ROOT/$RUN_PREFIX"
mkdir -p "$RUN_DIR/logs"

TASKS=(
  tracking_shuffled_objects_five_objects
  object_counting
  causal_judgement
  movie_recommendation
  hyperbaton
)
GPUS=(0 1 2 3)

declare -A PID_TO_TASK=()
declare -A PID_TO_GPU=()
declare -A TASK_TO_PID=()

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2
}

launch_task() {
  local task="$1"
  local gpu="$2"
  local log_path="$RUN_DIR/logs/${task}.log"

  log "launch task=$task gpu=$gpu"
  (
    cd "$SCRIPT_DIR"
    CUDA_VISIBLE_DEVICES="$gpu" \
      stdbuf -oL -eL \
      python3 main.py \
        --task_name "$task" \
        --tensor_parallel_size 1 \
        --gpu_memory_utilization 0.90 \
        --include_paper_opt_prompt false \
        --generate_meta_prompts true \
        --num_meta_prompts "$NUM_META_PROMPTS" \
        --meta_prompt_seed "$META_PROMPT_SEED" \
        --meta_prompt_generation_max_rounds "$META_PROMPT_GENERATION_MAX_ROUNDS" \
        --meta_prompt_generation_batch_size "$META_PROMPT_GENERATION_BATCH_SIZE" \
        --evaluate_train_split true \
        --exp_name "${RUN_PREFIX}_${task}"
  ) >"$log_path" 2>&1 &

  local pid=$!
  PID_TO_TASK["$pid"]="$task"
  PID_TO_GPU["$pid"]="$gpu"
  TASK_TO_PID["$task"]="$pid"
  log "task=$task pid=$pid log=$log_path"
}

wait_for_any_task_finish() {
  while true; do
    for pid in "${!PID_TO_TASK[@]}"; do
      if ! kill -0 "$pid" 2>/dev/null; then
        local task="${PID_TO_TASK[$pid]}"
        local gpu="${PID_TO_GPU[$pid]}"
        if wait "$pid"; then
          log "task finished successfully: task=$task gpu=$gpu pid=$pid"
        else
          log "task finished with non-zero exit: task=$task gpu=$gpu pid=$pid"
        fi
        unset "PID_TO_TASK[$pid]"
        unset "PID_TO_GPU[$pid]"
        unset "TASK_TO_PID[$task]"
        echo "$gpu"
        return 0
      fi
    done
    sleep 20
  done
}

wait_for_remaining_tasks() {
  local failed=0
  for task in "${!TASK_TO_PID[@]}"; do
    local pid="${TASK_TO_PID[$task]}"
    if wait "$pid"; then
      log "task finished successfully: task=$task pid=$pid"
    else
      log "task finished with non-zero exit: task=$task pid=$pid"
      failed=1
    fi
  done
  return "$failed"
}

write_status() {
  {
    echo "run_prefix=$RUN_PREFIX"
    echo "num_meta_prompts=$NUM_META_PROMPTS"
    echo "meta_prompt_seed=$META_PROMPT_SEED"
    echo "meta_prompt_generation_max_rounds=$META_PROMPT_GENERATION_MAX_ROUNDS"
    echo "meta_prompt_generation_batch_size=$META_PROMPT_GENERATION_BATCH_SIZE"
    echo "run_dir=$RUN_DIR"
  } > "$RUN_DIR/run_info.txt"
}

main() {
  write_status
  log "run_prefix=$RUN_PREFIX"
  log "num_meta_prompts=$NUM_META_PROMPTS"
  log "meta_prompt_seed=$META_PROMPT_SEED"
  log "meta_prompt_generation_max_rounds=$META_PROMPT_GENERATION_MAX_ROUNDS"
  log "meta_prompt_generation_batch_size=$META_PROMPT_GENERATION_BATCH_SIZE"

  launch_task "${TASKS[0]}" "${GPUS[0]}"
  launch_task "${TASKS[1]}" "${GPUS[1]}"
  launch_task "${TASKS[2]}" "${GPUS[2]}"
  launch_task "${TASKS[3]}" "${GPUS[3]}"

  local wait_output
  local freed_gpu
  wait_output="$(wait_for_any_task_finish)"
  freed_gpu="$(printf '%s\n' "$wait_output" | tail -n 1 | tr -cd '0-9')"
  if [[ -z "$freed_gpu" ]]; then
    log "failed to parse freed gpu index from wait output: $wait_output"
    return 1
  fi
  if (( freed_gpu < 0 || freed_gpu > 7 )); then
    log "parsed freed gpu index out of range: $freed_gpu (raw=$wait_output)"
    return 1
  fi
  log "launching delayed task=${TASKS[4]} gpu=$freed_gpu"
  launch_task "${TASKS[4]}" "$freed_gpu"

  if ! wait_for_remaining_tasks; then
    log "one or more tasks failed; skipping combined plotting"
    return 1
  fi

  log "all tasks completed; generating correlation artifacts"
  (
    cd "$SCRIPT_DIR"
    python3 plot_prompt_correlation.py \
      --run_root "$SAVE_ROOT" \
      --run_prefix "$RUN_PREFIX"
  ) | tee "$RUN_DIR/plot_prompt_correlation.log"

  log "done"
}

main "$@"
