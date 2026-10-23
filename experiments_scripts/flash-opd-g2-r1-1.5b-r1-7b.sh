#!/usr/bin/env bash
# Flash-OPD on G2: DeepSeek-R1-Distill-Qwen-1.5B  <-  DeepSeek-R1-Distill-Qwen-7B
# Default controller (k0 = W = 128, k_min = 16); L_max is the rollout budget, so the cap is
# inert and every trajectory stops at its own verified boundary.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_ROOT="${MODEL_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)/models}"
export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-${MODEL_ROOT}/DeepSeek-R1-Distill-Qwen-1.5B}"
export REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-${MODEL_ROOT}/DeepSeek-R1-Distill-Qwen-7B}"

export MAX_RESP_LENGTH="${MAX_RESP_LENGTH:-12288}"
export RUN_TAG="${RUN_TAG:-flash-opd-g2}"

exec bash "${SCRIPT_DIR}/flash-opd.sh" "$@"
