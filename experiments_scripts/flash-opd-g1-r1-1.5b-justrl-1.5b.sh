#!/usr/bin/env bash
# Flash-OPD on G1: DeepSeek-R1-Distill-Qwen-1.5B  <-  JustRL-DeepSeek-1.5B
#
# G1 is the pair with the longest reliable prefixes (2708 retained tokens on average, against
# 585/304/569 on G2-G4), so its controller is scaled up accordingly: verification starts at
# 1024 rather than 128, the rate is estimated over a 1024-position window, and the minimum
# advance is 512. Checking every 128 positions on a ~2700-token trajectory would spend more on
# verification than the truncation saves.
#
# L_max=4400 is a deliberate retention cap, not the rollout budget: it is the one setting in
# this directory under which a trajectory can stop before its own boundary, so Proposition 1
# does not cover the trajectories it truncates.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export MODEL_ROOT="${MODEL_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)/models}"
export ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-${MODEL_ROOT}/DeepSeek-R1-Distill-Qwen-1.5B}"
export REWARD_MODEL_PATH="${REWARD_MODEL_PATH:-${MODEL_ROOT}/JustRL-DeepSeek-1.5B}"

export MAX_RESP_LENGTH="${MAX_RESP_LENGTH:-12288}"
export ICT_K0="${ICT_K0:-1024}"
export ICT_WINDOW="${ICT_WINDOW:-1024}"
export ICT_KMIN="${ICT_KMIN:-512}"
export ICT_L_MAX="${ICT_L_MAX:-4400}"
export ICT_REUSE_VERIFY="${ICT_REUSE_VERIFY:-True}"
export RUN_TAG="${RUN_TAG:-flash-opd-g1}"

exec bash "${SCRIPT_DIR}/flash-opd.sh" "$@"
