#!/usr/bin/env bash
# OPD baseline on G2: DeepSeek-R1-Distill-Qwen-1.5B  <-  DeepSeek-R1-Distill-Qwen-7B
#
# The OPD baseline is this repository with the adaptive-truncation machinery absent:
# run_opd is called with no actor_rollout_ref.rollout.prune_opd overrides at all, so the
# prune_opd config subtree never exists and every trajectory is decoded to the full
# rollout budget. That absence IS the baseline -- there is no flag to turn off.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

export MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/models}"
export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-${MODEL_ROOT}/DeepSeek-R1-Distill-Qwen-1.5B}"
export REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-${MODEL_ROOT}/DeepSeek-R1-Distill-Qwen-7B}"

export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export MAX_RESP_LENGTH="${MAX_RESP_LENGTH:-12288}"
export MAX_VAL_RESP_LENGTH="${MAX_VAL_RESP_LENGTH:-31744}"
export DATA_SHUFFLE="${DATA_SHUFFLE:-True}"
export TEST_FREQ="${TEST_FREQ:-20}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-203}"
# Inert for the DeepSeek students, whose chat template never references enable_thinking. Set
# anyway, so both arms of this pair run on the same prompt setting.
export APPLY_CHAT_TEMPLATE_ENABLE_THINKING="${APPLY_CHAT_TEMPLATE_ENABLE_THINKING:-False}"

export RUN_TAG="${RUN_TAG:-opd-g2}"

run_opd "${RUN_TAG}" "$@"
