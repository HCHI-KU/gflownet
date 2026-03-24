#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"

LOCAL_TASKS="${LOCAL_TASKS:-object_counting}"
B200_TASKS="${B200_TASKS:-causal_judgement movie_recommendation hyperbaton tracking_shuffled_objects_five_objects}"
B200_HOST="${B200_HOST:-}"
B200_REPO_DIR="${B200_REPO_DIR:-$REPO_DIR}"

echo "[$(date '+%F %T')] Launching local BBH5 tasks: $LOCAL_TASKS"
QUEUE_TASKS="$LOCAL_TASKS" \
QUEUE_TAG="${QUEUE_TAG:-bbh5_local_${RUN_TS}}" \
SESSION_NAME="${SESSION_NAME:-gfn_bbh5_local_${RUN_TS}}" \
QUEUE_LOG="${QUEUE_LOG:-$REPO_DIR/logs/gfn_bbh5_local_${RUN_TS}.log}" \
bash "$SCRIPT_DIR/run_v1_log_prior_followup_local_tmux.sh"

if [[ -z "$B200_HOST" ]]; then
  cat <<EOF
[$(date '+%F %T')] B200_HOST is not set.
Run the remaining BBH5 tasks on the b200 server with:
  cd $B200_REPO_DIR
  TASKS="$B200_TASKS" RUN_TS="$RUN_TS" bash scripts/run_v1_log_prior_b200_heavy.sh
EOF
  exit 0
fi

echo "[$(date '+%F %T')] Launching b200 BBH5 tasks on $B200_HOST: $B200_TASKS"
ssh "$B200_HOST" "cd '$B200_REPO_DIR' && TASKS='$B200_TASKS' RUN_TS='$RUN_TS' bash '$B200_REPO_DIR/scripts/run_v1_log_prior_b200_heavy.sh'"
