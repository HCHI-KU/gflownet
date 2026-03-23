#!/usr/bin/env bash

if [[ -n "${GFN_PO_LOCAL_ENV_LOADED:-}" ]]; then
  return 0 2>/dev/null || exit 0
fi

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$THIS_DIR/.." && pwd)"
WORKSPACE_DIR="$(cd "$REPO_DIR/.." && pwd)"
SHARED_ROOT="$(cd "$WORKSPACE_DIR/.." && pwd)"

export ORIGINAL_HOME="${ORIGINAL_HOME:-$HOME}"
export GFN_PO_REPO_DIR="$REPO_DIR"
export GFN_PO_WORKSPACE_ROOT="$WORKSPACE_DIR"
export GFN_PO_SHARED_ROOT="$SHARED_ROOT"

LOCAL_ENV_SCRIPT="${LOCAL_ENV_SCRIPT:-$WORKSPACE_DIR/bin/jihun-env.sh}"
if [[ -f "$LOCAL_ENV_SCRIPT" ]]; then
  # shellcheck disable=SC1090
  source "$LOCAL_ENV_SCRIPT"
fi

DEFAULT_MODEL_DIR="$SHARED_ROOT/models/Meta-Llama-3-8B-Instruct"
if [[ -d "$DEFAULT_MODEL_DIR" ]]; then
  export GFN_PO_DEFAULT_MODEL_DIR="$DEFAULT_MODEL_DIR"
else
  export GFN_PO_DEFAULT_MODEL_DIR="meta-llama/Meta-Llama-3-8B-Instruct"
fi

export GFN_PO_DEFAULT_HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
export GFN_PO_LOCAL_ENV_LOADED=1
