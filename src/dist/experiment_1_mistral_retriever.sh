#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/retriever_experiment_lib.sh"

PROMPT_MODES="${PROMPT_MODES:-raw original zero-shot many-shot dynamic-shot}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_INFER="${RUN_INFER:-1}"
RUN_MATCH="${RUN_MATCH:-1}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
MISTRAL_BATCH_SIZE="${MISTRAL_BATCH_SIZE:-4}"
MISTRAL_GRAD_ACCUM_STEPS="${MISTRAL_GRAD_ACCUM_STEPS:-1}"
MISTRAL_EPOCHS="${MISTRAL_EPOCHS:-5}"
MISTRAL_NON_ORIGINAL_EPOCHS="${MISTRAL_NON_ORIGINAL_EPOCHS:-}"
MISTRAL_LR="${MISTRAL_LR:-5e-4}"
MISTRAL_MAX_TRAIN_SAMPLES="${MISTRAL_MAX_TRAIN_SAMPLES:--1}"
MISTRAL_MAX_EVAL_SAMPLES="${MISTRAL_MAX_EVAL_SAMPLES:--1}"
MISTRAL_MAX_INFER_SAMPLES="${MISTRAL_MAX_INFER_SAMPLES:--1}"
MISTRAL_INFER_BATCH_SIZE="${MISTRAL_INFER_BATCH_SIZE:-4}"
MISTRAL_INFER_LOAD_IN_4BIT="${MISTRAL_INFER_LOAD_IN_4BIT:-1}"
MISTRAL_INFER_MERGE_ADAPTER="${MISTRAL_INFER_MERGE_ADAPTER:-auto}"
MISTRAL_INFER_SORT_BY_LENGTH="${MISTRAL_INFER_SORT_BY_LENGTH:-1}"
MISTRAL_MAX_NEW_TOKENS="${MISTRAL_MAX_NEW_TOKENS:-128}"
MISTRAL_STRUCTURED_OUTPUT="${MISTRAL_STRUCTURED_OUTPUT:-canonical}"
MISTRAL_CUDA_DEVICES="${MISTRAL_CUDA_DEVICES:-${CUDA_DEVICES}}"
MISTRAL_NUM_PROCESSES="${MISTRAL_NUM_PROCESSES:-2}"
MISTRAL_DEVICE="${MISTRAL_DEVICE:-local_rank}"
MISTRAL_INFER_DEVICE="${MISTRAL_INFER_DEVICE:-0}"
MISTRAL_SAVE_STEPS="${MISTRAL_SAVE_STEPS:-625}"
MISTRAL_SAVE_TOTAL_LIMIT="${MISTRAL_SAVE_TOTAL_LIMIT:-2}"
MISTRAL_RESUME_FROM_CHECKPOINT="${MISTRAL_RESUME_FROM_CHECKPOINT:-}"
MISTRAL_FULL_NON_ORIGINAL_MAX_BATCH="${MISTRAL_FULL_NON_ORIGINAL_MAX_BATCH:-4}"
MISTRAL_ALLOW_HIGH_RISK_BATCH="${MISTRAL_ALLOW_HIGH_RISK_BATCH:-0}"
MISTRAL_EXPT_PREFIX="${MISTRAL_EXPT_PREFIX:-finqa_mistral}"
EXPT_ID_SUFFIX="${EXPT_ID_SUFFIX:-}"

expt_id_for_prompt() {
  printf '%s_%s\n' "${MISTRAL_EXPT_PREFIX}" "$(prompt_mode_suffix "$1")"
}

assert_mistral_batch_oom_guard() {
  local prompt="$1"
  local batch_size="$2"

  if is_original_prompt_mode "${prompt}"; then
    return 0
  fi
  if [[ "${RUN_TRAIN}" != "1" || "${MISTRAL_ALLOW_HIGH_RISK_BATCH}" == "1" ]]; then
    return 0
  fi
  if [[ "${MISTRAL_MAX_TRAIN_SAMPLES}" != "-1" || "${MISTRAL_MAX_EVAL_SAMPLES}" != "-1" ]]; then
    return 0
  fi
  if [[ ! "${batch_size}" =~ ^[0-9]+$ || ! "${MISTRAL_FULL_NON_ORIGINAL_MAX_BATCH}" =~ ^[0-9]+$ ]]; then
    printf "Invalid Mistral batch guard: MISTRAL_BATCH_SIZE=%s, MISTRAL_FULL_NON_ORIGINAL_MAX_BATCH=%s\n" \
      "${batch_size}" "${MISTRAL_FULL_NON_ORIGINAL_MAX_BATCH}" >&2
    return 2
  fi
  if (( batch_size > MISTRAL_FULL_NON_ORIGINAL_MAX_BATCH )); then
    cat >&2 <<EOF
Refusing high-risk Mistral non-original full run.
  prompt_mode=${prompt}
  MISTRAL_BATCH_SIZE=${batch_size}
  max_safe_without_override=${MISTRAL_FULL_NON_ORIGINAL_MAX_BATCH}

Previous full zero-shot runs OOMed at batch_size=6 and batch_size=5 on this host.
Use MISTRAL_BATCH_SIZE<=${MISTRAL_FULL_NON_ORIGINAL_MAX_BATCH}, or set
MISTRAL_ALLOW_HIGH_RISK_BATCH=1 if you intentionally want to probe above the guard.
EOF
    return 2
  fi
}

warn_mistral_checkpoint_without_resume() {
  local train_dir="$1"
  local prompt="$2"

  if [[ "${RUN_TRAIN}" != "1" || -n "${MISTRAL_RESUME_FROM_CHECKPOINT}" ]]; then
    return 0
  fi
  if compgen -G "${train_dir}/checkpoint-*" >/dev/null; then
    cat >&2 <<EOF
Warning: existing checkpoint detected for ${prompt}, but MISTRAL_RESUME_FROM_CHECKPOINT is empty.
Set MISTRAL_RESUME_FROM_CHECKPOINT=auto to resume, or leave it empty to start a fresh run intentionally.
EOF
  fi
}

if [[ "${RUN_TRAIN}" == "1" && "${RUN_PREFLIGHT}" == "1" ]]; then
  mkdir -p "${REPO_ROOT}/Experiment/preflight"
  run_gpu_preflight_log "${REPO_ROOT}/Experiment/preflight/mistral_gpu_state.log"
  run_torchrun_distributed_probe \
    "${MISTRAL_CUDA_DEVICES}" \
    "${MISTRAL_NUM_PROCESSES}" \
    "${REPO_ROOT}/Experiment/preflight/mistral_torchrun_probe.log"
fi

for prompt in ${PROMPT_MODES}; do
  train_csv="$(prompt_train_csv "${prompt}")"
  dev_csv="$(prompt_dev_csv "${prompt}")"
  test_csv="$(prompt_test_csv "${prompt}")"
  require_file "${train_csv}"
  require_file "${dev_csv}"
  require_file "${test_csv}"

  expt_id="$(expt_id_for_prompt "${prompt}")"
  expt_id="${expt_id}${EXPT_ID_SUFFIX}"
  expt_dir="${REPO_ROOT}/Experiment/${expt_id}"
  train_dir="${expt_dir}/retriever/train"
  adapter_dir="${expt_dir}/retriever/model"
  output_dir="${expt_dir}/retriever/outputs"
  prediction_txt="${output_dir}/predictions.txt"
  matched_json="${output_dir}/best_matched_with_retrieved_facts_and_questions.json"

  mkdir -p "${train_dir}" "${adapter_dir}" "${output_dir}"

  mistral_ld_library_path=""
  if [[ "${RUN_TRAIN}" == "1" || "${RUN_INFER}" == "1" ]]; then
    mistral_ld_library_path="$(mistral_retriever_ld_library_path)"
  fi

  if [[ "${RUN_TRAIN}" == "1" ]]; then
    validate_prompt_mode_contract "${prompt}" "${train_csv}" "${dev_csv}" "${test_csv}"
    train_epochs="$(epochs_for_prompt_mode "${prompt}" "${MISTRAL_EPOCHS}" "${MISTRAL_NON_ORIGINAL_EPOCHS}")"
    assert_mistral_batch_oom_guard "${prompt}" "${MISTRAL_BATCH_SIZE}"
    warn_mistral_checkpoint_without_resume "${train_dir}" "${prompt}"
    train_cmd=(
      env CUDA_VISIBLE_DEVICES="${MISTRAL_CUDA_DEVICES}" \
      LD_LIBRARY_PATH="${mistral_ld_library_path}" \
      PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
      PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF}" \
      TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING}" \
      TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG}" \
      conda run --no-capture-output -n "${CONDA_ENV}" \
      torchrun --standalone --nproc_per_node="${MISTRAL_NUM_PROCESSES}" \
      "${REPO_ROOT}/.external/FINDER/Retriever Codes/Mistral/mistral_train.py" \
      --train-csv "${train_csv}" \
      --eval-csv "${dev_csv}" \
      --output-dir "${train_dir}" \
      --adapter-dir "${adapter_dir}" \
      --max-train-samples "${MISTRAL_MAX_TRAIN_SAMPLES}" \
      --max-eval-samples "${MISTRAL_MAX_EVAL_SAMPLES}" \
      --num-train-epochs "${train_epochs}" \
      --batch-size "${MISTRAL_BATCH_SIZE}" \
      --gradient-accumulation-steps "${MISTRAL_GRAD_ACCUM_STEPS}" \
      --learning-rate "${MISTRAL_LR}" \
      --prompt-mode "${prompt}" \
      --device "${MISTRAL_DEVICE}" \
      --save-steps "${MISTRAL_SAVE_STEPS}" \
      --save-total-limit "${MISTRAL_SAVE_TOTAL_LIMIT}"
    )
    if [[ -n "${MISTRAL_RESUME_FROM_CHECKPOINT}" ]]; then
      train_cmd+=(--resume-from-checkpoint "${MISTRAL_RESUME_FROM_CHECKPOINT}")
    fi
    RETRIEVER_SPLIT_SUMMARY="$(retriever_split_row_summary "${train_csv}" "${dev_csv}" "${test_csv}" "${MISTRAL_MAX_TRAIN_SAMPLES}" "${MISTRAL_MAX_EVAL_SAMPLES}" -1)"
    run_logged "${expt_dir}/retriever/train.log" "${train_cmd[@]}"
    assert_train_log_completed "${expt_dir}/retriever/train.log" "${train_csv}" "${dev_csv}" "${test_csv}" "${MISTRAL_MAX_TRAIN_SAMPLES}" "${MISTRAL_MAX_EVAL_SAMPLES}" -1
    unset RETRIEVER_SPLIT_SUMMARY
  fi


  if [[ "${RUN_INFER}" == "1" || "${RUN_MATCH}" == "1" ]]; then
    assert_train_log_current_for_reuse \
      "${expt_dir}/retriever/train.log" \
      "${train_csv}" \
      "${dev_csv}" \
      "${test_csv}" \
      "${MISTRAL_MAX_TRAIN_SAMPLES}" \
      "${MISTRAL_MAX_EVAL_SAMPLES}" \
      -1 \
      "Mistral retriever"
  fi

  if [[ "${RUN_INFER}" == "1" ]]; then
    require_adapter_artifact "${adapter_dir}" "Mistral retriever" "${prompt}" "${expt_id}"
    mistral_infer_device="${MISTRAL_INFER_DEVICE}"
    if [[ "${INFER_PARALLEL_GPU}" == "1" && "$(visible_device_count "${MISTRAL_CUDA_DEVICES}")" -gt 1 ]]; then
      mistral_infer_device="${MISTRAL_PARALLEL_INFER_DEVICE:-0}"
    fi
    infer_cmd=(
      env LD_LIBRARY_PATH="${mistral_ld_library_path}"
      conda run --no-capture-output -n "${CONDA_ENV}"
      python -B "${REPO_ROOT}/.external/FINDER/Retriever Codes/Mistral/mistral_inference.py"
      --adapter-dir "${adapter_dir}"
      --device "${mistral_infer_device}"
      --prompt-mode "${prompt}"
      --load-in-4bit "${MISTRAL_INFER_LOAD_IN_4BIT}"
      --merge-adapter "${MISTRAL_INFER_MERGE_ADAPTER}"
      --sort-by-length "${MISTRAL_INFER_SORT_BY_LENGTH}"
      --max-new-tokens "${MISTRAL_MAX_NEW_TOKENS}"
      --structured-output "${MISTRAL_STRUCTURED_OUTPUT}"
    )
    run_parallel_inference_artifact \
      "${expt_dir}/retriever/inference.log" \
      "Mistral retriever" \
      "${MISTRAL_CUDA_DEVICES}" \
      "${test_csv}" \
      "${prediction_txt}" \
      text \
      "${MISTRAL_MAX_INFER_SAMPLES}" \
      "${MISTRAL_INFER_BATCH_SIZE}" \
      "${infer_cmd[@]}"
  fi

  if [[ "${RUN_MATCH}" == "1" ]]; then
    run_match_artifact "mistral_v0_3" "${prediction_txt}" "${matched_json}" "${expt_dir}/retriever/match.log" "${prompt}" "${expt_id}"
  fi
done
