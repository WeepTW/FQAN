#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/retriever_experiment_lib.sh"

PROMPT_MODES="${PROMPT_MODES:-raw original zero-shot many-shot dynamic-shot}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_INFER="${RUN_INFER:-1}"
RUN_MATCH="${RUN_MATCH:-1}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
# Dual RTX A4500 few10 probe default. Full FinQA still needs validation.
FLAN_BATCH_SIZE="${FLAN_BATCH_SIZE:-2}"
FLAN_GRAD_ACCUM_STEPS="${FLAN_GRAD_ACCUM_STEPS:-4}"
FLAN_EPOCHS="${FLAN_EPOCHS:-10}"
FLAN_NON_ORIGINAL_EPOCHS="${FLAN_NON_ORIGINAL_EPOCHS:-}"
FLAN_LR="${FLAN_LR:-5e-4}"
FLAN_MAX_TRAIN_SAMPLES="${FLAN_MAX_TRAIN_SAMPLES:--1}"
FLAN_MAX_EVAL_SAMPLES="${FLAN_MAX_EVAL_SAMPLES:--1}"
FLAN_MAX_INFER_SAMPLES="${FLAN_MAX_INFER_SAMPLES:--1}"
FLAN_INFER_BATCH_SIZE="${FLAN_INFER_BATCH_SIZE:-32}"
FLAN_MAX_NEW_TOKENS="${FLAN_MAX_NEW_TOKENS:-128}"
FLAN_STRUCTURED_OUTPUT="${FLAN_STRUCTURED_OUTPUT:-canonical}"
FLAN_CUDA_DEVICES="${FLAN_CUDA_DEVICES:-${CUDA_DEVICES}}"
FLAN_NUM_PROCESSES="${FLAN_NUM_PROCESSES:-2}"
FLAN_SAVE_STEPS="${FLAN_SAVE_STEPS:-500}"
FLAN_SAVE_TOTAL_LIMIT="${FLAN_SAVE_TOTAL_LIMIT:-2}"
FLAN_RESUME_FROM_CHECKPOINT="${FLAN_RESUME_FROM_CHECKPOINT:-}"
EXPT_ID_SUFFIX="${EXPT_ID_SUFFIX:-}"

expt_id_for_prompt() {
  printf 'finqa_flan_%s\n' "$(prompt_mode_suffix "$1")"
}

if [[ "${RUN_TRAIN}" == "1" && "${RUN_PREFLIGHT}" == "1" ]]; then
  mkdir -p "${REPO_ROOT}/Experiment/preflight"
  run_gpu_preflight_log "${REPO_ROOT}/Experiment/preflight/flan_gpu_state.log"
  run_torchrun_distributed_probe \
    "${FLAN_CUDA_DEVICES}" \
    "${FLAN_NUM_PROCESSES}" \
    "${REPO_ROOT}/Experiment/preflight/flan_torchrun_probe.log"
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

  if [[ "${RUN_TRAIN}" == "1" ]]; then
    require_nvidia_smi_ready "FLAN retriever training"
    validate_prompt_mode_contract "${prompt}" "${train_csv}" "${dev_csv}" "${test_csv}"
    train_epochs="$(epochs_for_prompt_mode "${prompt}" "${FLAN_EPOCHS}" "${FLAN_NON_ORIGINAL_EPOCHS}")"
    train_cmd=(
      env CUDA_VISIBLE_DEVICES="${FLAN_CUDA_DEVICES}" \
      PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
      PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF}" \
      TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING}" \
      TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG}" \
      conda run --no-capture-output -n "${CONDA_ENV}" \
      torchrun --standalone --nproc_per_node="${FLAN_NUM_PROCESSES}" \
      "${REPO_ROOT}/.external/FINDER/Retriever Codes/Flan/lora_flan_large_finqa_rel_fact.py" \
      --mode train \
      --train-csv "${train_csv}" \
      --eval-csv "${dev_csv}" \
      --output-dir "${train_dir}" \
      --adapter-dir "${adapter_dir}" \
      --max-train-samples "${FLAN_MAX_TRAIN_SAMPLES}" \
      --max-eval-samples "${FLAN_MAX_EVAL_SAMPLES}" \
      --num-train-epochs "${train_epochs}" \
      --batch-size "${FLAN_BATCH_SIZE}" \
      --gradient-accumulation-steps "${FLAN_GRAD_ACCUM_STEPS}" \
      --learning-rate "${FLAN_LR}" \
      --prompt-mode "${prompt}" \
      --save-steps "${FLAN_SAVE_STEPS}" \
      --save-total-limit "${FLAN_SAVE_TOTAL_LIMIT}"
    )
    if [[ -n "${FLAN_RESUME_FROM_CHECKPOINT}" ]]; then
      train_cmd+=(--resume-from-checkpoint "${FLAN_RESUME_FROM_CHECKPOINT}")
    fi
    RETRIEVER_SPLIT_SUMMARY="$(retriever_split_row_summary "${train_csv}" "${dev_csv}" "${test_csv}" "${FLAN_MAX_TRAIN_SAMPLES}" "${FLAN_MAX_EVAL_SAMPLES}" -1)"
    run_logged "${expt_dir}/retriever/train.log" "${train_cmd[@]}"
    assert_train_log_completed "${expt_dir}/retriever/train.log" "${train_csv}" "${dev_csv}" "${test_csv}" "${FLAN_MAX_TRAIN_SAMPLES}" "${FLAN_MAX_EVAL_SAMPLES}" -1
    unset RETRIEVER_SPLIT_SUMMARY
  fi


  if [[ "${RUN_INFER}" == "1" || "${RUN_MATCH}" == "1" ]]; then
    assert_train_log_current_for_reuse \
      "${expt_dir}/retriever/train.log" \
      "${train_csv}" \
      "${dev_csv}" \
      "${test_csv}" \
      "${FLAN_MAX_TRAIN_SAMPLES}" \
      "${FLAN_MAX_EVAL_SAMPLES}" \
      -1 \
      "FLAN retriever"
  fi

  if [[ "${RUN_INFER}" == "1" ]]; then
    require_nvidia_smi_ready "FLAN retriever inference"
    require_adapter_artifact "${adapter_dir}" "FLAN retriever" "${prompt}" "${expt_id}"
    infer_cmd=(
      conda run --no-capture-output -n "${CONDA_ENV}"
      python -B "${REPO_ROOT}/.external/FINDER/Retriever Codes/Flan/lora_flan_large_finqa_rel_fact.py"
      --mode infer
      --eval-csv "${test_csv}"
      --adapter-dir "${adapter_dir}"
      --prompt-mode "${prompt}"
      --max-new-tokens "${FLAN_MAX_NEW_TOKENS}"
      --structured-output "${FLAN_STRUCTURED_OUTPUT}"
    )
    run_parallel_inference_artifact \
      "${expt_dir}/retriever/inference.log" \
      "FLAN retriever" \
      "${FLAN_CUDA_DEVICES}" \
      "${test_csv}" \
      "${prediction_txt}" \
      text \
      "${FLAN_MAX_INFER_SAMPLES}" \
      "${FLAN_INFER_BATCH_SIZE}" \
      "${infer_cmd[@]}"
  fi

  if [[ "${RUN_MATCH}" == "1" ]]; then
    run_match_artifact "flan_t5_large" "${prediction_txt}" "${matched_json}" "${expt_dir}/retriever/match.log" "${prompt}" "${expt_id}"
  fi
done
