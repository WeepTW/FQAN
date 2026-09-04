#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/retriever_experiment_lib.sh"

PROMPT_MODES="${PROMPT_MODES:-raw original zero-shot many-shot dynamic-shot}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_INFER="${RUN_INFER:-1}"
RUN_MATCH="${RUN_MATCH:-1}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
T5GEMMA_BASE_MODEL="${T5GEMMA_BASE_MODEL:-google/t5gemma-2-1b-1b}"
T5GEMMA_BATCH_SIZE="${T5GEMMA_BATCH_SIZE:-3}"
T5GEMMA_GRAD_ACCUM_STEPS="${T5GEMMA_GRAD_ACCUM_STEPS:-1}"
T5GEMMA_EPOCHS="${T5GEMMA_EPOCHS:-10}"
T5GEMMA_NON_ORIGINAL_EPOCHS="${T5GEMMA_NON_ORIGINAL_EPOCHS:-}"
T5GEMMA_LR="${T5GEMMA_LR:-5e-4}"
T5GEMMA_MAX_TRAIN_SAMPLES="${T5GEMMA_MAX_TRAIN_SAMPLES:--1}"
T5GEMMA_MAX_EVAL_SAMPLES="${T5GEMMA_MAX_EVAL_SAMPLES:--1}"
T5GEMMA_MAX_INFER_SAMPLES="${T5GEMMA_MAX_INFER_SAMPLES:--1}"
T5GEMMA_INFER_BATCH_SIZE="${T5GEMMA_INFER_BATCH_SIZE:-16}"
T5GEMMA_MAX_NEW_TOKENS="${T5GEMMA_MAX_NEW_TOKENS:-128}"
T5GEMMA_STRUCTURED_OUTPUT="${T5GEMMA_STRUCTURED_OUTPUT:-canonical}"
T5GEMMA_CUDA_DEVICES="${T5GEMMA_CUDA_DEVICES:-${CUDA_DEVICES}}"
T5GEMMA_NUM_PROCESSES="${T5GEMMA_NUM_PROCESSES:-2}"
T5GEMMA_SAVE_STEPS="${T5GEMMA_SAVE_STEPS:-500}"
T5GEMMA_SAVE_TOTAL_LIMIT="${T5GEMMA_SAVE_TOTAL_LIMIT:-2}"
T5GEMMA_RESUME_FROM_CHECKPOINT="${T5GEMMA_RESUME_FROM_CHECKPOINT:-}"
EXPT_ID_SUFFIX="${EXPT_ID_SUFFIX:-}"

expt_id_for_prompt() {
  printf 'finqa_t5gemma2_%s\n' "$(prompt_mode_suffix "$1")"
}

if [[ "${RUN_TRAIN}" == "1" && "${RUN_PREFLIGHT}" == "1" ]]; then
  mkdir -p "${REPO_ROOT}/Experiment/preflight"
  run_gpu_preflight_log "${REPO_ROOT}/Experiment/preflight/t5gemma_gpu_state.log"
  require_nvidia_smi_ready "T5Gemma retriever preflight"
  run_hf_model_access_probe "${T5GEMMA_BASE_MODEL}" "${REPO_ROOT}/Experiment/preflight/t5gemma_hf_access.log" 1
  run_torchrun_distributed_probe \
    "${T5GEMMA_CUDA_DEVICES}" \
    "${T5GEMMA_NUM_PROCESSES}" \
    "${REPO_ROOT}/Experiment/preflight/t5gemma_torchrun_probe.log"
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
    require_nvidia_smi_ready "T5Gemma retriever training"
    validate_prompt_mode_contract "${prompt}" "${train_csv}" "${dev_csv}" "${test_csv}"
    train_epochs="$(epochs_for_prompt_mode "${prompt}" "${T5GEMMA_EPOCHS}" "${T5GEMMA_NON_ORIGINAL_EPOCHS}")"
    train_cmd=(
      env CUDA_VISIBLE_DEVICES="${T5GEMMA_CUDA_DEVICES}" \
      PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
      PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF}" \
      TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING}" \
      TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG}" \
      conda run --no-capture-output -n "${CONDA_ENV}" \
      torchrun --standalone --nproc_per_node="${T5GEMMA_NUM_PROCESSES}" \
      "${REPO_ROOT}/.external/FINDER/Retriever Codes/t5gemma-2/t5gemma-2_train.py" \
      --mode train \
      --base-model "${T5GEMMA_BASE_MODEL}" \
      --train-csv "${train_csv}" \
      --eval-csv "${dev_csv}" \
      --output-dir "${train_dir}" \
      --adapter-dir "${adapter_dir}" \
      --output-txt "${prediction_txt}" \
      --max-train-samples "${T5GEMMA_MAX_TRAIN_SAMPLES}" \
      --max-eval-samples "${T5GEMMA_MAX_EVAL_SAMPLES}" \
      --num-train-epochs "${train_epochs}" \
      --batch-size "${T5GEMMA_BATCH_SIZE}" \
      --gradient-accumulation-steps "${T5GEMMA_GRAD_ACCUM_STEPS}" \
      --learning-rate "${T5GEMMA_LR}" \
      --prompt-mode "${prompt}" \
      --save-steps "${T5GEMMA_SAVE_STEPS}" \
      --save-total-limit "${T5GEMMA_SAVE_TOTAL_LIMIT}"
    )
    if [[ -n "${T5GEMMA_RESUME_FROM_CHECKPOINT}" ]]; then
      train_cmd+=(--resume-from-checkpoint "${T5GEMMA_RESUME_FROM_CHECKPOINT}")
    fi
    RETRIEVER_SPLIT_SUMMARY="$(retriever_split_row_summary "${train_csv}" "${dev_csv}" "${test_csv}" "${T5GEMMA_MAX_TRAIN_SAMPLES}" "${T5GEMMA_MAX_EVAL_SAMPLES}" -1)"
    run_logged "${expt_dir}/retriever/train.log" "${train_cmd[@]}"
    assert_train_log_completed "${expt_dir}/retriever/train.log" "${train_csv}" "${dev_csv}" "${test_csv}" "${T5GEMMA_MAX_TRAIN_SAMPLES}" "${T5GEMMA_MAX_EVAL_SAMPLES}" -1
    unset RETRIEVER_SPLIT_SUMMARY
  fi


  if [[ "${RUN_INFER}" == "1" || "${RUN_MATCH}" == "1" ]]; then
    assert_train_log_current_for_reuse \
      "${expt_dir}/retriever/train.log" \
      "${train_csv}" \
      "${dev_csv}" \
      "${test_csv}" \
      "${T5GEMMA_MAX_TRAIN_SAMPLES}" \
      "${T5GEMMA_MAX_EVAL_SAMPLES}" \
      -1 \
      "T5Gemma retriever"
  fi

  if [[ "${RUN_INFER}" == "1" ]]; then
    require_nvidia_smi_ready "T5Gemma retriever inference"
    require_adapter_artifact "${adapter_dir}" "T5Gemma retriever" "${prompt}" "${expt_id}"
    infer_cmd=(
      conda run --no-capture-output -n "${CONDA_ENV}"
      python -B "${REPO_ROOT}/.external/FINDER/Retriever Codes/t5gemma-2/t5gemma-2_train.py"
      --mode infer
      --base-model "${T5GEMMA_BASE_MODEL}"
      --train-csv "${train_csv}"
      --eval-csv "${test_csv}"
      --output-dir "${train_dir}"
      --adapter-dir "${adapter_dir}"
      --prompt-mode "${prompt}"
      --max-new-tokens "${T5GEMMA_MAX_NEW_TOKENS}"
      --structured-output "${T5GEMMA_STRUCTURED_OUTPUT}"
    )
    run_parallel_inference_artifact \
      "${expt_dir}/retriever/inference.log" \
      "T5Gemma retriever" \
      "${T5GEMMA_CUDA_DEVICES}" \
      "${test_csv}" \
      "${prediction_txt}" \
      text \
      "${T5GEMMA_MAX_INFER_SAMPLES}" \
      "${T5GEMMA_INFER_BATCH_SIZE}" \
      "${infer_cmd[@]}"
  fi

  if [[ "${RUN_MATCH}" == "1" ]]; then
    run_match_artifact "t5gemma_2_1b_1b" "${prediction_txt}" "${matched_json}" "${expt_dir}/retriever/match.log" "${prompt}" "${expt_id}"
  fi
done
