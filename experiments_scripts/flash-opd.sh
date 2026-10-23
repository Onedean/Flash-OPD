#!/usr/bin/env bash
# Flash-OPD launcher: on-policy distillation with adaptive boundary verification.
#
# Flash-OPD replaces the batch-shared rollout horizon with a per-trajectory stopping rule.
# Each trajectory is decoded to a verification position k, the teacher is evaluated on the
# newly produced positions, and the trajectory retires at the first position where its exact
# accumulated low-compatibility count reaches the allowance B. The rate estimate only places
# the next verification position; it never enters the stopping test, so an inaccurate estimate
# costs extra decoding or extra checks, never supervision (Proposition 1).
#
# The controller lives in verl/verl/utils/ict.py and its trainer integration in
# verl/verl/utils/ict_loop.py. "ICT" (Iterative Checkpoint Truncation) is the name the code
# uses for the mechanism.
#
# This script sets the controller knobs and defers everything else to common.sh. It is not
# meant to be run directly for the reported configurations -- use the per-pair wrappers
# (flash-opd-g1-*.sh ... flash-opd-g4-*.sh), which pin the published settings.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

export MODEL_ROOT="${MODEL_ROOT:-${REPO_ROOT}/models}"
: "${ACTOR_MODEL_PATH:?ACTOR_MODEL_PATH must be set (student)}"
: "${REWARD_MODEL_PATH:?REWARD_MODEL_PATH must be set (teacher)}"

export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"
export MAX_RESP_LENGTH="${MAX_RESP_LENGTH:-12288}"
export MAX_VAL_RESP_LENGTH="${MAX_VAL_RESP_LENGTH:-31744}"
export DATA_SHUFFLE="${DATA_SHUFFLE:-True}"
export VERL_SKIP_TIMING_ALLGATHER="${VERL_SKIP_TIMING_ALLGATHER:-1}"

# Chat-template thinking mode. MUST be set explicitly for a Qwen3-Base student. common.sh only
# forwards this to Hydra when the variable is non-empty, so leaving it unset makes Qwen3 default
# to thinking ON: the base student then emits a very long chain, hits the 31744-token evaluation
# cap, and never writes \boxed{}. The score that comes back measures parse failure, not
# reasoning -- measured format_score (fraction of generations an answer could be parsed from)
# drops from 82.2% to 16.8% on Qwen3-1.7B-Base and from 73.7% to 29.4% on Qwen3-4B-Base.
# It is inert for the DeepSeek students, whose chat template never references enable_thinking,
# so all four pairs run on one setting.
export APPLY_CHAT_TEMPLATE_ENABLE_THINKING="${APPLY_CHAT_TEMPLATE_ENABLE_THINKING:-False}"

# Compatibility criterion: a position counts as a low-compatibility event when the overlap
# between the student's top-k candidates and the teacher's top-k falls below this threshold.
# This is eta in the paper, and it is shared with the overlap-criterion baselines so the
# methods are directly comparable.
export PRUNE_OPD_THRESHOLD="${PRUNE_OPD_THRESHOLD:-0.7}"
# These two configure the reward attenuation of the Prune-OPD baseline, which Flash-OPD does
# not use: with prune_opd.ict.enable=True the attenuation is derived off (prune_opd.py,
# `attenuate = not ict_on`), so neither value reaches the loss and every retained position
# carries weight 1.0. w_drop still scales the logged `E` diagnostic -- the position count a
# 1/w_drop reliability budget would have permitted -- which is why it is pinned to 1/ICT_B
# rather than left implicit.
export PRUNE_OPD_W_DROP="${PRUNE_OPD_W_DROP:-0.005}"
export PRUNE_OPD_W_BASE="${PRUNE_OPD_W_BASE:-0.0}"

# --- controller -------------------------------------------------------------------------
export ICT_B="${ICT_B:-200}"            # allowance B: low-compatibility positions permitted
export ICT_K0="${ICT_K0:-128}"          # initial verification position k0
export ICT_RHO="${ICT_RHO:-1.00}"       # step discount rho; <1 re-measures the rate sooner
export ICT_WINDOW="${ICT_WINDOW:-128}"  # trailing window W for the rate estimate
export ICT_KMIN="${ICT_KMIN:-16}"       # minimum advance k_min
# Per-trajectory retention limit. Defaults to the rollout budget, i.e. inert. Setting it lower
# makes it a real cap: a trajectory then retires at min(boundary, ICT_L_MAX), which is the one
# path that stops a trajectory BEFORE its boundary, so Proposition 1 does not cover whatever it
# truncates. G1 uses it deliberately; treat any other use as a variant, not the method.
export ICT_L_MAX="${ICT_L_MAX:-$MAX_RESP_LENGTH}"
# Tokens per verification micro-batch. This is what the fp32 log_softmax over the full
# vocabulary costs scale with, and it is what bounds peak memory during verification.
export ICT_TOKEN_BUDGET="${ICT_TOKEN_BUDGET:-4096}"
# Hand each round's verification outputs to the post-loop stages instead of recomputing them.
# Sound because the final batch is a prefix of what verification already ran on: Flash-OPD
# truncates at the boundary it verified up to. Saves a log-prob and a teacher pass per step at
# the cost of ~0.27 GB/round of extra transfer.
export ICT_REUSE_VERIFY="${ICT_REUSE_VERIFY:-False}"
# Keep the rollout engine resident across verification rounds. This is what makes the method
# affordable -- the fixed per-round overhead drops from ~20.5 s to ~4.8 s -- but the engine
# holds its memory reservation while the verification forwards run, so GPU_MEM_UTIL must leave
# them headroom. 0.6 is the reported setting; 0.8 OOMs in compute_log_prob.
export ICT_HOLD_AWAKE="${ICT_HOLD_AWAKE:-True}"
# Offload the engine's weights to CPU between rounds instead (vLLM sleep level 1), so the
# FSDP->vLLM weight derivation is paid once per step rather than once per round. Slower than
# holding the engine awake; useful when memory is too tight for ICT_HOLD_AWAKE=True.
export ICT_REUSE_W="${ICT_REUSE_W:-False}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.6}"

# The assertion in the log-prob path compares this cap against the batch's PADDED width, and
# under Flash-OPD that width grows with training (the verification probe is left-padded to the
# round's longest row). A tuned constant therefore cannot hold: pinning the cap to the model's
# own context limit makes the assertion unable to fire by construction. Peak memory is
# ICT_TOKEN_BUDGET's job, not this one's.
export LOG_PROB_MAX_TOKEN_LEN="${LOG_PROB_MAX_TOKEN_LEN:-$(( ${MAX_VAL_RESP_LENGTH:-31744} + ${MAX_PROMPT_LENGTH:-1024} ))}"

export RUN_TAG="${RUN_TAG:-flash-opd}"
export TEST_FREQ="${TEST_FREQ:-20}"
# common.sh appends TOTAL_TRAINING_STEPS after the caller's arguments, so a trailing
# trainer.total_training_steps override would be silently discarded. Set the variable.
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-203}"

SEED_OVERRIDE=()
if [[ -n "${SEED_ARG:-}" ]]; then SEED_OVERRIDE=("++data.seed=${SEED_ARG}"); fi

run_opd "${RUN_TAG}" \
    "+actor_rollout_ref.rollout.prune_opd.enable=True" \
    "+actor_rollout_ref.rollout.prune_opd.metric=overlap_ratio" \
    "+actor_rollout_ref.rollout.prune_opd.threshold=${PRUNE_OPD_THRESHOLD}" \
    "+actor_rollout_ref.rollout.prune_opd.w_drop=${PRUNE_OPD_W_DROP}" \
    "+actor_rollout_ref.rollout.prune_opd.w_base=${PRUNE_OPD_W_BASE}" \
    "+actor_rollout_ref.rollout.prune_opd.ict.enable=True" \
    "+actor_rollout_ref.rollout.prune_opd.ict.allowance=${ICT_B}" \
    "+actor_rollout_ref.rollout.prune_opd.ict.k0=${ICT_K0}" \
    "+actor_rollout_ref.rollout.prune_opd.ict.rho=${ICT_RHO}" \
    "+actor_rollout_ref.rollout.prune_opd.ict.window=${ICT_WINDOW}" \
    "+actor_rollout_ref.rollout.prune_opd.ict.k_min=${ICT_KMIN}" \
    "+actor_rollout_ref.rollout.prune_opd.ict.reuse_weights=${ICT_REUSE_W}" \
    "+actor_rollout_ref.rollout.prune_opd.ict.hold_awake=${ICT_HOLD_AWAKE}" \
    "+actor_rollout_ref.rollout.prune_opd.ict.l_max=${ICT_L_MAX}" \
    "+actor_rollout_ref.rollout.prune_opd.ict.token_budget=${ICT_TOKEN_BUDGET}" \
    "+actor_rollout_ref.rollout.prune_opd.ict.reuse_verify=${ICT_REUSE_VERIFY}" \
    "+actor_rollout_ref.rollout.prune_opd.dynamic_response_length.enable=False" \
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${LOG_PROB_MAX_TOKEN_LEN}" \
    "${SEED_OVERRIDE[@]}" "$@"
