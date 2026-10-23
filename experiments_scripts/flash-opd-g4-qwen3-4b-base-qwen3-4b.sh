#!/usr/bin/env bash
# Flash-OPD on G4: Qwen3-4B-Base  <-  Qwen3-4B (Non-thinking)
# Same-size teacher and student; the thinking template must stay off -- see flash-opd.sh.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_ROOT="${MODEL_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)/models}"
export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-${MODEL_ROOT}/Qwen3-4B-Base}"
export REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-${MODEL_ROOT}/Qwen3-4B}"

export MAX_RESP_LENGTH="${MAX_RESP_LENGTH:-12288}"
export APPLY_CHAT_TEMPLATE_ENABLE_THINKING="${APPLY_CHAT_TEMPLATE_ENABLE_THINKING:-False}"
export RUN_TAG="${RUN_TAG:-flash-opd-g4}"

exec bash "${SCRIPT_DIR}/flash-opd.sh" "$@"
