#!/usr/bin/env bash
# Flash-OPD on G3: Qwen3-1.7B-Base  <-  Qwen3-4B (Non-thinking)
# Rollout budget is 8192 here, not 12288, and the thinking template must stay off -- see the
# note in flash-opd.sh for what leaving it on does to the reported scores.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_ROOT="${MODEL_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)/models}"
export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-${MODEL_ROOT}/Qwen3-1.7B-Base}"
export REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-${MODEL_ROOT}/Qwen3-4B}"

export MAX_RESP_LENGTH="${MAX_RESP_LENGTH:-8192}"
export APPLY_CHAT_TEMPLATE_ENABLE_THINKING="${APPLY_CHAT_TEMPLATE_ENABLE_THINKING:-False}"
export RUN_TAG="${RUN_TAG:-flash-opd-g3}"

exec bash "${SCRIPT_DIR}/flash-opd.sh" "$@"
