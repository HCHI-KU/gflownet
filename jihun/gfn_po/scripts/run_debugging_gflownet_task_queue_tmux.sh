#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKSPACE_DIR="$(cd "$REPO_DIR/.." && pwd)"
SHARED_ROOT="$(cd "$WORKSPACE_DIR/.." && pwd)"
cd "$REPO_DIR"

TMUX_BIN="${TMUX_BIN:-/opt/kernel/tmux}"
TMUX_SOCKET="${TMUX_SOCKET:-jihun}"
TMUX_CONF="${TMUX_CONF:-$WORKSPACE_DIR/.tmux.conf}"
LOCAL_ENV_SCRIPT="${LOCAL_ENV_SCRIPT:-$WORKSPACE_DIR/bin/jihun-env.sh}"
MODEL_CANDIDATE="$SHARED_ROOT/models/Meta-Llama-3-8B-Instruct"
DEFAULT_MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
if [[ -d "$MODEL_CANDIDATE" ]]; then
  DEFAULT_MODEL="$MODEL_CANDIDATE"
fi
DEFAULT_CACHE_DIR="${HOME:-$WORKSPACE_DIR/.cache}/.cache/huggingface"
DEFAULT_DATA_ROOT="$WORKSPACE_DIR/bbh_vllm_eval/data/GreaTer_data/BBH"

log() {
  echo "[$(date '+%F %T')] $*"
}

run_worker() {
  if [[ -f "$LOCAL_ENV_SCRIPT" ]]; then
    # Optional local profile isolation for shared servers.
    source "$LOCAL_ENV_SCRIPT"
  fi
  export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
  export VLLM_USE_V1="${VLLM_USE_V1:-0}"
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

  local wait_session="${WAIT_SESSION:-}"
  local wait_pid="${WAIT_PID:-}"
  local wait_poll_seconds="${WAIT_POLL_SECONDS:-60}"
  local queue_tasks="${QUEUE_TASKS:-object_counting causal_judgement movie_recommendation hyperbaton tracking_shuffled_objects_five_objects}"
  local queue_tag="${QUEUE_TAG:-$(date +%Y%m%d_%H%M%S)}"

  local agent_model="${AGENT_MODEL:-$DEFAULT_MODEL}"
  local eval_model="${EVAL_MODEL:-$DEFAULT_MODEL}"
  local cache_dir="${CACHE_DIR:-$HOME/.cache/huggingface}"
  local save_dir="${SAVE_DIR:-$REPO_DIR/debugging_gflownet_runs}"
  local wandb_mode="${WANDB_MODE:-online}"
  local wandb_project="${WANDB_PROJECT:-debugging_gflownet_object_counting_gamma_sweep}"

  local train_steps="${TRAIN_STEPS:-300}"
  local num_warmup_steps="${NUM_WARMUP_STEPS:-20}"
  local lr="${LR:-0.0001}"
  local max_norm="${MAX_NORM:-1.0}"
  local batch_size="${BATCH_SIZE:-4}"
  local grad_acc_steps="${GRAD_ACC_STEPS:-2}"
  local max_prompt_length="${MAX_PROMPT_LENGTH:-150}"
  local sampling_top_p="${SAMPLING_TOP_P:-0.9}"
  local temp_low="${TEMP_LOW:-0.5}"
  local temp_high="${TEMP_HIGH:-2.0}"
  local beta="${BETA:-0.02}"
  local gamma="${GAMMA:-0.001}"
  local reward_epsilon="${REWARD_EPSILON:-1e-8}"
  local ema_decay="${EMA_DECAY:-0.99}"
  local prior_reduction="${PRIOR_REDUCTION:-mean}"
  local prior_chunk_size="${PRIOR_CHUNK_SIZE:-2}"

  local meta_prompt="${META_PROMPT:-I gave a friend an instruction and five inputs. The friend read the instruction and wrote an output for every one of the inputs. Here are the input-output pairs:}"
  local num_example="${NUM_EXAMPLE:-5}"
  local fewshot_strategy="${FEWSHOT_STRATEGY:-random}"
  local fewshot_seed="${FEWSHOT_SEED:-42}"
  local fewshot_resample_each_step="${FEWSHOT_RESAMPLE_EACH_STEP:-0}"
  local data_root="${DATA_ROOT:-$DEFAULT_DATA_ROOT}"
  local conversation_template="${CONVERSATION_TEMPLATE:-llama-3}"

  local bbh_train_size="${BBH_TRAIN_SIZE:-50}"
  local bbh_test_size="${BBH_TEST_SIZE:-100}"
  local bbh_reasoning_max_tokens="${BBH_REASONING_MAX_TOKENS:-1024}"
  local bbh_answer_max_tokens="${BBH_ANSWER_MAX_TOKENS:-1}"
  local eval_chunk_size="${EVAL_CHUNK_SIZE:-32}"

  local lora_r="${LORA_R:-16}"
  local lora_alpha="${LORA_ALPHA:-32}"
  local lora_dropout="${LORA_DROPOUT:-0.05}"
  local agent_device="${AGENT_DEVICE:-cuda:1}"
  local tp_size="${TP_SIZE:-1}"
  local eval_gpu_memory_utilization="${EVAL_GPU_MEMORY_UTILIZATION:-0.9}"
  local eval_max_num_seqs="${EVAL_MAX_NUM_SEQS:-64}"
  local eval_max_seq_len_to_capture="${EVAL_MAX_SEQ_LEN_TO_CAPTURE:-3072}"
  local seed="${SEED:-2}"
  local log_every="${LOG_EVERY:-10}"
  local export_every="${EXPORT_EVERY:-100}"
  local test_eval_period="${TEST_EVAL_PERIOD:-0}"
  local save_top_k="${SAVE_TOP_K:-5}"

  local gamma_tag="${gamma//./p}"

  if [[ -n "$wait_pid" ]]; then
    log "waiting for pid '$wait_pid' to finish before starting queued tasks"
    while kill -0 "$wait_pid" 2>/dev/null; do
      sleep "$wait_poll_seconds"
    done
  elif [[ -n "$wait_session" ]]; then
    log "waiting for tmux session '$wait_session' to finish before starting queued tasks"
    while "$TMUX_BIN" -L "$TMUX_SOCKET" -f "$TMUX_CONF" has-session -t "$wait_session" 2>/dev/null; do
      sleep "$wait_poll_seconds"
    done
  fi

  log "starting queued tasks: $queue_tasks"

  local task
  for task in $queue_tasks; do
    local run_name="v1_log_prior_${task}_gamma_${gamma_tag}_${queue_tag}_gamma_${gamma_tag}_${train_steps}step"
    local -a cmd=(
      python3 -c "import debugging_gflownet_variants as m; m.run_variant('v1_log_prior')"
      --task_names "$task"
      --agent_model "$agent_model"
      --eval_model "$eval_model"
      --cache_dir "$cache_dir"
      --save_dir "$save_dir"
      --run_name "$run_name"
      --wandb_mode "$wandb_mode"
      --wandb_project "$wandb_project"
      --batch_size "$batch_size"
      --grad_acc_steps "$grad_acc_steps"
      --train_steps "$train_steps"
      --num_warmup_steps "$num_warmup_steps"
      --lr "$lr"
      --max_norm "$max_norm"
      --max_prompt_length "$max_prompt_length"
      --sampling_top_p "$sampling_top_p"
      --temp_low "$temp_low"
      --temp_high "$temp_high"
      --beta "$beta"
      --gamma "$gamma"
      --reward_epsilon "$reward_epsilon"
      --ema_decay "$ema_decay"
      --prior_reduction "$prior_reduction"
      --prior_chunk_size "$prior_chunk_size"
      --meta_prompt "$meta_prompt"
      --num_example "$num_example"
      --fewshot_strategy "$fewshot_strategy"
      --fewshot_seed "$fewshot_seed"
      --data_root "$data_root"
      --conversation_template "$conversation_template"
      --bbh_train_size "$bbh_train_size"
      --bbh_test_size "$bbh_test_size"
      --bbh_reasoning_max_tokens "$bbh_reasoning_max_tokens"
      --bbh_answer_max_tokens "$bbh_answer_max_tokens"
      --eval_chunk_size "$eval_chunk_size"
      --lora_r "$lora_r"
      --lora_alpha "$lora_alpha"
      --lora_dropout "$lora_dropout"
      --agent_device "$agent_device"
      --tp_size "$tp_size"
      --eval_gpu_memory_utilization "$eval_gpu_memory_utilization"
      --eval_max_num_seqs "$eval_max_num_seqs"
      --eval_max_seq_len_to_capture "$eval_max_seq_len_to_capture"
      --seed "$seed"
      --log_every "$log_every"
      --export_every "$export_every"
      --test_eval_period "$test_eval_period"
      --save_top_k "$save_top_k"
    )

    if [[ "$fewshot_resample_each_step" == "1" ]]; then
      cmd+=(--fewshot_resample_each_step)
    fi

    log "starting task=$task run_name=$run_name"
    set +e
    "${cmd[@]}"
    local status=$?
    set -e

    if [[ "$status" -ne 0 ]]; then
      log "task=$task failed with status=$status; continuing to next queued task"
    else
      log "task=$task finished"
    fi
  done

  log "queued tasks finished"
}

launch_tmux() {
  local session_name="${SESSION_NAME:-gfn_task_queue_$(date +%Y%m%d_%H%M%S)}"
  local queue_log="${QUEUE_LOG:-$REPO_DIR/logs/${session_name}.log}"
  mkdir -p "$REPO_DIR/logs"

  "$TMUX_BIN" -L "$TMUX_SOCKET" -f "$TMUX_CONF" new-session -d -s "$session_name" \
    "cd '$REPO_DIR' && \
     LOCAL_ENV_SCRIPT='${LOCAL_ENV_SCRIPT:-$LOCAL_ENV_SCRIPT}' \
     WAIT_SESSION='${WAIT_SESSION:-}' \
     WAIT_PID='${WAIT_PID:-}' \
     WAIT_POLL_SECONDS='${WAIT_POLL_SECONDS:-60}' \
     QUEUE_TASKS='${QUEUE_TASKS:-object_counting causal_judgement movie_recommendation hyperbaton tracking_shuffled_objects_five_objects}' \
     QUEUE_TAG='${QUEUE_TAG:-$(date +%Y%m%d_%H%M%S)}' \
     AGENT_MODEL='${AGENT_MODEL:-$DEFAULT_MODEL}' \
     EVAL_MODEL='${EVAL_MODEL:-$DEFAULT_MODEL}' \
     CACHE_DIR='${CACHE_DIR:-$HOME/.cache/huggingface}' \
     SAVE_DIR='${SAVE_DIR:-$REPO_DIR/debugging_gflownet_runs}' \
     WANDB_MODE='${WANDB_MODE:-online}' \
     WANDB_PROJECT='${WANDB_PROJECT:-debugging_gflownet_object_counting_gamma_sweep}' \
     TRAIN_STEPS='${TRAIN_STEPS:-300}' \
     NUM_WARMUP_STEPS='${NUM_WARMUP_STEPS:-20}' \
     LR='${LR:-0.0001}' \
     MAX_NORM='${MAX_NORM:-1.0}' \
     BATCH_SIZE='${BATCH_SIZE:-4}' \
     GRAD_ACC_STEPS='${GRAD_ACC_STEPS:-2}' \
     MAX_PROMPT_LENGTH='${MAX_PROMPT_LENGTH:-150}' \
     SAMPLING_TOP_P='${SAMPLING_TOP_P:-0.9}' \
     TEMP_LOW='${TEMP_LOW:-0.5}' \
     TEMP_HIGH='${TEMP_HIGH:-2.0}' \
     BETA='${BETA:-0.02}' \
     GAMMA='${GAMMA:-0.001}' \
     REWARD_EPSILON='${REWARD_EPSILON:-1e-8}' \
     EMA_DECAY='${EMA_DECAY:-0.99}' \
     PRIOR_REDUCTION='${PRIOR_REDUCTION:-mean}' \
     PRIOR_CHUNK_SIZE='${PRIOR_CHUNK_SIZE:-2}' \
     META_PROMPT='${META_PROMPT:-I gave a friend an instruction and five inputs. The friend read the instruction and wrote an output for every one of the inputs. Here are the input-output pairs:}' \
     NUM_EXAMPLE='${NUM_EXAMPLE:-5}' \
     FEWSHOT_STRATEGY='${FEWSHOT_STRATEGY:-random}' \
     FEWSHOT_SEED='${FEWSHOT_SEED:-42}' \
     FEWSHOT_RESAMPLE_EACH_STEP='${FEWSHOT_RESAMPLE_EACH_STEP:-0}' \
     DATA_ROOT='${DATA_ROOT:-$DEFAULT_DATA_ROOT}' \
     CONVERSATION_TEMPLATE='${CONVERSATION_TEMPLATE:-llama-3}' \
     BBH_TRAIN_SIZE='${BBH_TRAIN_SIZE:-50}' \
     BBH_TEST_SIZE='${BBH_TEST_SIZE:-100}' \
     BBH_REASONING_MAX_TOKENS='${BBH_REASONING_MAX_TOKENS:-1024}' \
     BBH_ANSWER_MAX_TOKENS='${BBH_ANSWER_MAX_TOKENS:-1}' \
     EVAL_CHUNK_SIZE='${EVAL_CHUNK_SIZE:-32}' \
     LORA_R='${LORA_R:-16}' \
     LORA_ALPHA='${LORA_ALPHA:-32}' \
     LORA_DROPOUT='${LORA_DROPOUT:-0.05}' \
     AGENT_DEVICE='${AGENT_DEVICE:-cuda:1}' \
     TP_SIZE='${TP_SIZE:-1}' \
     EVAL_GPU_MEMORY_UTILIZATION='${EVAL_GPU_MEMORY_UTILIZATION:-0.9}' \
     EVAL_MAX_NUM_SEQS='${EVAL_MAX_NUM_SEQS:-64}' \
     EVAL_MAX_SEQ_LEN_TO_CAPTURE='${EVAL_MAX_SEQ_LEN_TO_CAPTURE:-3072}' \
     SEED='${SEED:-2}' \
     LOG_EVERY='${LOG_EVERY:-10}' \
     EXPORT_EVERY='${EXPORT_EVERY:-100}' \
     TEST_EVAL_PERIOD='${TEST_EVAL_PERIOD:-0}' \
     SAVE_TOP_K='${SAVE_TOP_K:-5}' \
     bash '$SCRIPT_DIR/run_debugging_gflownet_task_queue_tmux.sh' --worker 2>&1 | tee '$queue_log'"

  echo "SESSION=$session_name"
  echo "QUEUE_LOG=$queue_log"
}

case "${1:-}" in
  --worker)
    run_worker
    ;;
  *)
    launch_tmux
    ;;
esac
