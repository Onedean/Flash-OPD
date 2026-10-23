#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"

activate_opd_env() {
    if [[ "${AUTO_CONDA_ACTIVATE:-0}" != "1" ]]; then
        return 0
    fi

    local conda_home="${CONDA_HOME:-$HOME/miniconda3}"
    local conda_sh="${conda_home}/etc/profile.d/conda.sh"
    local target_env="${CONDA_ENV_NAME:-opd}"

    if [[ "${CONDA_DEFAULT_ENV:-}" == "${target_env}" ]]; then
        return 0
    fi

    if [[ ! -f "${conda_sh}" ]]; then
        echo "Cannot find conda activation script at ${conda_sh}" >&2
        exit 1
    fi

    # shellcheck disable=SC1090
    source "${conda_sh}"
    conda activate "${target_env}"
}

resolve_path() {
    local path="$1"
    if [[ "${path}" = /* ]]; then
        printf '%s\n' "${path}"
    else
        printf '%s\n' "${REPO_ROOT}/${path}"
    fi
}

require_path() {
    local path="$1"
    if [[ ! -e "${path}" ]]; then
        echo "Required path does not exist: ${path}" >&2
        exit 1
    fi
}

require_model_if_local_path() {
    local path="$1"
    if [[ "${path}" = /* || "${path}" == ./* || "${path}" == ../* ]]; then
        require_path "$(resolve_path "${path}")"
    fi
}

setup_logging() {
    local experiment_name="$1"
    local log_dir="${LOG_DIR:-${REPO_ROOT}/logs/opd}"
    mkdir -p "${log_dir}"

    if [[ -z "${SLURM_JOB_ID:-}" ]]; then
        local log_file="${log_dir}/${experiment_name}.log"
        exec > >(tee -a "${log_file}") 2>&1
        echo "Log file: ${log_file}"
    fi
}

cleanup_ray() {
    ray stop --force >/dev/null 2>&1 || true
}

setup_tracking() {
    export WANDB_API_KEY="${WANDB_API_KEY:-}"
    export WANDB_MODE="${WANDB_MODE:-offline}"
    export WANDB_DIR="${WANDB_DIR:-${REPO_ROOT}/logs/wandb}"
    export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${REPO_ROOT}/.cache/wandb}"
    export TRACKING_BACKENDS="${TRACKING_BACKENDS:-[console]}"

    mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}"
}

run_opd() {
    local run_name="$1"
    shift

    activate_opd_env
    setup_tracking

    : "${ACTOR_MODEL_PATH:?ACTOR_MODEL_PATH must be set}"
    export REWARD_MODEL_ENABLE="${REWARD_MODEL_ENABLE:-True}"

    require_model_if_local_path "${ACTOR_MODEL_PATH}"
    if [[ "${REWARD_MODEL_ENABLE}" == "True" ]]; then
        : "${REWARD_MODEL_PATH:?REWARD_MODEL_PATH must be set when REWARD_MODEL_ENABLE=True}"
        require_model_if_local_path "${REWARD_MODEL_PATH}"
    fi

    export PROJECT_NAME="${PROJECT_NAME:-FlashOPD}"
    export ADV_ESTIMATOR="${ADV_ESTIMATOR:-token_reward_direct}"
    export GRPO_OUTCOME_WEIGHT="${GRPO_OUTCOME_WEIGHT:-1.0}"
    export DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/datasets}"
    export TRAIN_DATASET="${TRAIN_DATASET:-${DATA_ROOT}/dapo-math-17k.parquet}"
    export TRAIN_DATASET_NAME="${TRAIN_DATASET_NAME:-DAPO-Math-17k}"
    export TEST_DATASET="${TEST_DATASET:-[\"${DATA_ROOT}/test_data/AMC23/test.parquet\",\"${DATA_ROOT}/test_data/AIME24/test.parquet\",\"${DATA_ROOT}/test_data/AIME25/test.parquet\",\"${DATA_ROOT}/test_data/HMMT24/test.parquet\",\"${DATA_ROOT}/test_data/HMMT25/test.parquet\"]}"
    export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
    export MAX_RESP_LENGTH="${MAX_RESP_LENGTH:-8192}"
    export MAX_VAL_RESP_LENGTH="${MAX_VAL_RESP_LENGTH:-31744}"
    export MINI_BATCH_SIZE="${MINI_BATCH_SIZE:-64}"
    export DATA_SHUFFLE="${DATA_SHUFFLE:-False}"
    export TEMPERATURE="${TEMPERATURE:-1.0}"
    export TEACHER_TEMPERATURE="${TEACHER_TEMPERATURE:-1.0}"
    export REPETITION_PENALTY="${REPETITION_PENALTY:-1.0}"
    export N_RESPONSES="${N_RESPONSES:-4}"
    export LOG_PROB_TOP_K="${LOG_PROB_TOP_K:-16}"
    # Samples per validation prompt. 16 is the in-domain protocol reported in the paper. The
    # transfer evaluation lowers it to 4 so the same generation budget covers 948 prompts instead
    # of 203; it is only ever safe to lower when BOTH arms of a comparison lower it together.
    export VAL_N_RESPONSES="${VAL_N_RESPONSES:-16}"
    export TOP_K_STRATEGY="${TOP_K_STRATEGY:-only_stu}"
    export REWARD_WEIGHT_MODE="${REWARD_WEIGHT_MODE:-student_p}"
    export USE_KL="${USE_KL:-False}"
    export ENABLE_FORMAT_REWARD="${ENABLE_FORMAT_REWARD:-False}"
    export MODEL_DTYPE="${MODEL_DTYPE:-fp32}"
    export IS_PLOT="${IS_PLOT:-False}"
    export LOSS_AGG_MODE="${LOSS_AGG_MODE:-token-mean}"
    export PARALLEL_SIZE="${PARALLEL_SIZE:-1}"
    export VAL_TEMPERATURE="${VAL_TEMPERATURE:-1.0}"
    export VAL_TOP_P="${VAL_TOP_P:-0.95}"
    export TEST_FREQ="${TEST_FREQ:-20}"
    export SAVE_FREQ="${SAVE_FREQ:-100}"
    export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-203}"
    # Defaulted here so a hand-written launcher cannot trip `set -u` on it; every launcher
    # shipped in this directory sets it explicitly.
    export VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-False}"

    require_path "$(resolve_path "${TRAIN_DATASET}")"
    require_path "$(resolve_path "${DATA_ROOT}/test_data/AMC23/test.parquet")"
    require_path "$(resolve_path "${DATA_ROOT}/test_data/AIME24/test.parquet")"
    require_path "$(resolve_path "${DATA_ROOT}/test_data/AIME25/test.parquet")"
    require_path "$(resolve_path "${DATA_ROOT}/test_data/HMMT24/test.parquet")"
    require_path "$(resolve_path "${DATA_ROOT}/test_data/HMMT25/test.parquet")"

    export PYTHONUNBUFFERED=1
    export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.99}"
    export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
    export NCCL_TIMEOUT="${NCCL_TIMEOUT:-7200}"
    export TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-INFO}"
    export TOKENIZERS_PARALLELISM=true
    export HYDRA_FULL_ERROR=1
    export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
    export SWANLAB_LOG_DIR="${SWANLAB_LOG_DIR:-${REPO_ROOT}/checkpoint/swanlab_log}"
    export PYTHONPATH="${REPO_ROOT}/verl${PYTHONPATH:+:${PYTHONPATH}}"

    local actor_model_name
    actor_model_name="$(basename "${ACTOR_MODEL_PATH%/}")"
    local reward_model_name
    reward_model_name="reward_disabled"
    if [[ "${REWARD_MODEL_ENABLE}" == "True" ]]; then
        reward_model_name="$(basename "${REWARD_MODEL_PATH%/}")"
    fi
    local timestamp
    timestamp="$(date +%Y-%m-%d_%H-%M-%S)"
    local max_model_len
    max_model_len=$(( MAX_RESP_LENGTH + MAX_PROMPT_LENGTH > MAX_VAL_RESP_LENGTH + MAX_PROMPT_LENGTH ? MAX_RESP_LENGTH + MAX_PROMPT_LENGTH : MAX_VAL_RESP_LENGTH + MAX_PROMPT_LENGTH ))
    local ppo_max_token_len_per_gpu
    ppo_max_token_len_per_gpu=$(( ((MAX_PROMPT_LENGTH + MAX_RESP_LENGTH) > 32768) ? (MAX_PROMPT_LENGTH + MAX_RESP_LENGTH) : 32768 ))

    local experiment_name
    experiment_name="${run_name}_${ADV_ESTIMATOR}_${actor_model_name}_${reward_model_name}_${MAX_RESP_LENGTH}-T_${TEMPERATURE}-Tch_${TEACHER_TEMPERATURE}-n_${N_RESPONSES}-mbs_${MINI_BATCH_SIZE}-topk_${LOG_PROB_TOP_K}-topk_strategy_${TOP_K_STRATEGY}-rw_${REWARD_WEIGHT_MODE}-${timestamp}"

    # Keep names under the 255-byte filename limit: log files, checkpoint dirs and
    # validation dirs are all derived from experiment_name.
    local max_name_len=200
    if (( ${#experiment_name} > max_name_len )); then
        local name_hash
        name_hash="$(printf '%s' "${experiment_name}" | md5sum | cut -c1-8)"
        # Keep the leading run_name and the trailing timestamp, hash the middle.
        experiment_name="${run_name:0:$((max_name_len - ${#timestamp} - 10))}_${name_hash}_${timestamp}"
    fi
    local ckpt_root="${CKPT_ROOT:-${REPO_ROOT}/checkpoint}"
    local ckpt_path="${ckpt_root}/${experiment_name}"

    export WANDB_PROJECT="${WANDB_PROJECT:-${PROJECT_NAME}}"
    export WANDB_NAME="${WANDB_NAME:-${experiment_name}}"
    export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-${run_name}}"

    mkdir -p "${ckpt_root}" "${SWANLAB_LOG_DIR}"
    setup_logging "${experiment_name}"

    echo "Repository root: ${REPO_ROOT}"
    echo "Run name: ${run_name}"
    echo "Student: ${ACTOR_MODEL_PATH}"
    if [[ "${REWARD_MODEL_ENABLE}" == "True" ]]; then
        echo "Teacher: ${REWARD_MODEL_PATH}"
    else
        echo "Teacher: <disabled>"
    fi
    echo "Train dataset: ${TRAIN_DATASET}"
    echo "Train shuffle: ${DATA_SHUFFLE}"
    echo "Max response length: ${MAX_RESP_LENGTH}"
    echo "Validation max response length: ${MAX_VAL_RESP_LENGTH}"
    echo "Eval frequency: ${TEST_FREQ}"
    echo "Save frequency: ${SAVE_FREQ}"
    echo "Total training steps: ${TOTAL_TRAINING_STEPS}"
    echo "Thinking override: ${APPLY_CHAT_TEMPLATE_ENABLE_THINKING:-<default>}"
    echo "Experiment name: ${experiment_name}"
    echo "Checkpoint dir: ${ckpt_path}"
    echo "Tracking backends: ${TRACKING_BACKENDS}"
    echo "W&B dir: ${WANDB_DIR}"

    local -a cmd=(
        python -m verl.trainer.main_ppo
        "algorithm.adv_estimator=${ADV_ESTIMATOR}"
        "algorithm.grpo_outcome_weight=${GRPO_OUTCOME_WEIGHT}"
        "data.shuffle=${DATA_SHUFFLE}"
        "data.train_files=${TRAIN_DATASET}"
        "data.val_files=${TEST_DATASET}"
        "data.train_batch_size=$((MINI_BATCH_SIZE * PARALLEL_SIZE))"
        "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
        "data.max_response_length=${MAX_RESP_LENGTH}"
        "data.filter_overlong_prompts=True"
        "data.truncation=error"
        "data.return_raw_chat=True"
        "actor_rollout_ref.model.path=${ACTOR_MODEL_PATH}"
        "actor_rollout_ref.model.use_remove_padding=True"
        "actor_rollout_ref.model.enable_activation_offload=True"
        "actor_rollout_ref.model.enable_gradient_checkpointing=True"
        "actor_rollout_ref.actor.optim.lr=1e-6"
        "actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH_SIZE}"
        "actor_rollout_ref.actor.use_dynamic_bsz=True"
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1"
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}"
        "actor_rollout_ref.actor.ulysses_sequence_parallel_size=${PARALLEL_SIZE}"
        "actor_rollout_ref.actor.loss_agg_mode=${LOSS_AGG_MODE}"
        "actor_rollout_ref.actor.fsdp_config.param_offload=False"
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload=False"
        "actor_rollout_ref.actor.fsdp_config.forward_prefetch=True"
        "actor_rollout_ref.actor.fsdp_config.model_dtype=${MODEL_DTYPE}"
        "actor_rollout_ref.rollout.max_num_batched_tokens=${ppo_max_token_len_per_gpu}"
        "actor_rollout_ref.ref.fsdp_config.param_offload=True"
        "actor_rollout_ref.ref.fsdp_config.model_dtype=${MODEL_DTYPE}"
        "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True"
        "actor_rollout_ref.rollout.name=vllm"
        "actor_rollout_ref.rollout.temperature=${TEMPERATURE}"
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True"
        "+actor_rollout_ref.rollout.log_prob_top_k=${LOG_PROB_TOP_K}"
        "+actor_rollout_ref.rollout.top_k_strategy=${TOP_K_STRATEGY}"
        "+actor_rollout_ref.rollout.reward_weight_mode=${REWARD_WEIGHT_MODE}"
        "+actor_rollout_ref.rollout.teacher_temperature=${TEACHER_TEMPERATURE}"
        "actor_rollout_ref.rollout.tensor_model_parallel_size=${PARALLEL_SIZE}"
        "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL:-0.8}"
        "actor_rollout_ref.rollout.max_model_len=${max_model_len}"
        "actor_rollout_ref.rollout.n=${N_RESPONSES}"
        "actor_rollout_ref.rollout.val_kwargs.do_sample=True"
        "+actor_rollout_ref.rollout.val_kwargs.max_tokens=${MAX_VAL_RESP_LENGTH}"
        "actor_rollout_ref.rollout.val_kwargs.n=${VAL_N_RESPONSES}"
        "actor_rollout_ref.rollout.val_kwargs.temperature=${VAL_TEMPERATURE}"
        "actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P}"
        "actor_rollout_ref.rollout.repetition_penalty=${REPETITION_PENALTY}"
        "actor_rollout_ref.rollout.calculate_log_probs=True"
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1"
        "reward_model.enable=${REWARD_MODEL_ENABLE}"
        "+reward_model.reward_kwargs.enable_format_reward=${ENABLE_FORMAT_REWARD}"
        "custom_reward_function.path=${REPO_ROOT}/verl/verl/utils/reward_score/ttrl_math/__init__.py"
        "custom_reward_function.name=reward_func"
        "trainer.val_before_train=${VAL_BEFORE_TRAIN}"
        "trainer.log_val_generations=2"
        "trainer.logger=${TRACKING_BACKENDS}"
        "trainer.project_name=${PROJECT_NAME}"
        "trainer.experiment_name=${experiment_name}"
        "trainer.validation_data_dir=${REPO_ROOT}/validation_log/${experiment_name}"
        "trainer.n_gpus_per_node=8"
        "trainer.nnodes=1"
        "trainer.save_freq=${SAVE_FREQ}"
        "trainer.test_freq=${TEST_FREQ}"
        "trainer.total_training_steps=${TOTAL_TRAINING_STEPS}"
        "trainer.total_epochs=1"
        "trainer.default_local_dir=${ckpt_path}"
        "trainer.is_plot=${IS_PLOT}"
    )

    if [[ "${REWARD_MODEL_ENABLE}" == "True" ]]; then
        cmd+=(
            "reward_model.model.path=${REWARD_MODEL_PATH}"
            "reward_model.model.input_tokenizer=null"
            "reward_model.model.use_remove_padding=True"
            "reward_model.model.fsdp_config.param_offload=False"
            "+reward_model.model.dtype=${MODEL_DTYPE}"
            "reward_model.micro_batch_size_per_gpu=24"
        )
    fi

    if [[ "${USE_KL}" == "True" ]]; then
        cmd+=(
            "actor_rollout_ref.actor.use_kl_loss=True"
            "actor_rollout_ref.actor.kl_loss_coef=0.005"
            "actor_rollout_ref.actor.kl_loss_type=low_var_kl"
        )
    else
        cmd+=("actor_rollout_ref.actor.use_kl_loss=False")
    fi

    if [[ -n "${APPLY_CHAT_TEMPLATE_ENABLE_THINKING:-}" ]]; then
        cmd+=("+data.apply_chat_template_kwargs.enable_thinking=${APPLY_CHAT_TEMPLATE_ENABLE_THINKING}")
    fi

    if [[ "$#" -gt 0 ]]; then
        cmd+=("$@")
    fi

    if [[ "${DRY_RUN:-0}" == "1" ]]; then
        echo "DRY_RUN=1, command preview:"
        printf '%q ' "${cmd[@]}"
        echo
        return 0
    fi

    trap cleanup_ray EXIT
    cleanup_ray
    ray start --head
    sleep 5
    "${cmd[@]}"
}
