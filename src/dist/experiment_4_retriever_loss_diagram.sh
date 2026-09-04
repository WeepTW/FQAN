#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/retriever_experiment_lib.sh"

# Exploratory route only. Formal retriever experiments should use
# experiment_1/2/3 so train, inference, schema assembly, and match stay on one
# canonical flow.
BEST_MODEL="${BEST_MODEL:-mistral_v0_3}"
PROMPT_MODE="${PROMPT_MODE:-original}"
LOSS_EPOCHS="${LOSS_EPOCHS:-10}"
LOSS_EXPT_ID="${LOSS_EXPT_ID:-loss_${BEST_MODEL}_${PROMPT_MODE}}"
RUN_TRAIN="${RUN_TRAIN:-1}"
RUN_INFER="${RUN_INFER:-1}"
RUN_MATCH="${RUN_MATCH:-1}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
MISTRAL_CUDA_DEVICES="${MISTRAL_CUDA_DEVICES:-${CUDA_DEVICES}}"
MISTRAL_INFER_CUDA_DEVICES="${MISTRAL_INFER_CUDA_DEVICES:-${INFER_CUDA_DEVICES:-${CUDA_DEVICE}}}"
MISTRAL_NUM_PROCESSES="${MISTRAL_NUM_PROCESSES:-2}"
MISTRAL_DEVICE="${MISTRAL_DEVICE:-local_rank}"
MISTRAL_BATCH_SIZE="${MISTRAL_BATCH_SIZE:-2}"
MISTRAL_GRAD_ACCUM_STEPS="${MISTRAL_GRAD_ACCUM_STEPS:-1}"
MISTRAL_MAX_TRAIN_SAMPLES="${MISTRAL_MAX_TRAIN_SAMPLES:--1}"
MISTRAL_MAX_EVAL_SAMPLES="${MISTRAL_MAX_EVAL_SAMPLES:--1}"
MISTRAL_MAX_INFER_SAMPLES="${MISTRAL_MAX_INFER_SAMPLES:--1}"
MISTRAL_INFER_BATCH_SIZE="${MISTRAL_INFER_BATCH_SIZE:-8}"
MISTRAL_MAX_NEW_TOKENS="${MISTRAL_MAX_NEW_TOKENS:-128}"
MISTRAL_STRUCTURED_OUTPUT="${MISTRAL_STRUCTURED_OUTPUT:-assembler}"
MISTRAL_SAVE_STEPS="${MISTRAL_SAVE_STEPS:-625}"
MISTRAL_SAVE_TOTAL_LIMIT="${MISTRAL_SAVE_TOTAL_LIMIT:-2}"
MISTRAL_RESUME_FROM_CHECKPOINT="${MISTRAL_RESUME_FROM_CHECKPOINT:-}"
FLAN_CUDA_DEVICES="${FLAN_CUDA_DEVICES:-${CUDA_DEVICES}}"
FLAN_INFER_CUDA_DEVICES="${FLAN_INFER_CUDA_DEVICES:-${INFER_CUDA_DEVICES:-${CUDA_DEVICE}}}"
FLAN_NUM_PROCESSES="${FLAN_NUM_PROCESSES:-2}"
# Keep the FLAN loss-diagram route aligned with the formal FLAN retriever
# script: effective batch size stays 8 on two GPUs while avoiding A4500 OOM.
FLAN_BATCH_SIZE="${FLAN_BATCH_SIZE:-1}"
FLAN_GRAD_ACCUM_STEPS="${FLAN_GRAD_ACCUM_STEPS:-4}"
FLAN_MAX_TRAIN_SAMPLES="${FLAN_MAX_TRAIN_SAMPLES:--1}"
FLAN_MAX_EVAL_SAMPLES="${FLAN_MAX_EVAL_SAMPLES:--1}"
FLAN_MAX_INFER_SAMPLES="${FLAN_MAX_INFER_SAMPLES:--1}"
FLAN_INFER_BATCH_SIZE="${FLAN_INFER_BATCH_SIZE:-64}"
FLAN_MAX_NEW_TOKENS="${FLAN_MAX_NEW_TOKENS:-128}"
FLAN_STRUCTURED_OUTPUT="${FLAN_STRUCTURED_OUTPUT:-assembler}"
FLAN_SAVE_STEPS="${FLAN_SAVE_STEPS:-500}"
FLAN_SAVE_TOTAL_LIMIT="${FLAN_SAVE_TOTAL_LIMIT:-2}"
FLAN_RESUME_FROM_CHECKPOINT="${FLAN_RESUME_FROM_CHECKPOINT:-}"
T5GEMMA_BASE_MODEL="${T5GEMMA_BASE_MODEL:-google/t5gemma-2-1b-1b}"
T5GEMMA_CUDA_DEVICES="${T5GEMMA_CUDA_DEVICES:-${CUDA_DEVICES}}"
T5GEMMA_INFER_CUDA_DEVICES="${T5GEMMA_INFER_CUDA_DEVICES:-${INFER_CUDA_DEVICES:-${CUDA_DEVICE}}}"
T5GEMMA_NUM_PROCESSES="${T5GEMMA_NUM_PROCESSES:-2}"
T5GEMMA_BATCH_SIZE="${T5GEMMA_BATCH_SIZE:-1}"
T5GEMMA_GRAD_ACCUM_STEPS="${T5GEMMA_GRAD_ACCUM_STEPS:-1}"
T5GEMMA_LR="${T5GEMMA_LR:-5e-4}"
T5GEMMA_MAX_TRAIN_SAMPLES="${T5GEMMA_MAX_TRAIN_SAMPLES:--1}"
T5GEMMA_MAX_EVAL_SAMPLES="${T5GEMMA_MAX_EVAL_SAMPLES:--1}"
T5GEMMA_MAX_INFER_SAMPLES="${T5GEMMA_MAX_INFER_SAMPLES:--1}"
T5GEMMA_INFER_BATCH_SIZE="${T5GEMMA_INFER_BATCH_SIZE:-64}"
T5GEMMA_MAX_NEW_TOKENS="${T5GEMMA_MAX_NEW_TOKENS:-128}"
T5GEMMA_STRUCTURED_OUTPUT="${T5GEMMA_STRUCTURED_OUTPUT:-assembler}"
T5GEMMA_SAVE_STEPS="${T5GEMMA_SAVE_STEPS:-500}"
T5GEMMA_SAVE_TOTAL_LIMIT="${T5GEMMA_SAVE_TOTAL_LIMIT:-2}"
T5GEMMA_RESUME_FROM_CHECKPOINT="${T5GEMMA_RESUME_FROM_CHECKPOINT:-}"

if [[ "${LOSS_EXPLORATORY_ACK:-0}" != "1" ]]; then
  printf "Loss-diagram runs are exploratory and may override formal epoch settings.\n" >&2
  printf "Set LOSS_EXPLORATORY_ACK=1 after confirming the run is logged separately from formal experiments.\n" >&2
  exit 2
fi

train_csv="$(prompt_train_csv "${PROMPT_MODE}")"
dev_csv="$(prompt_dev_csv "${PROMPT_MODE}")"
test_csv="$(prompt_test_csv "${PROMPT_MODE}")"
require_file "${train_csv}"
require_file "${dev_csv}"
require_file "${test_csv}"

expt_dir="${REPO_ROOT}/Experiment/${LOSS_EXPT_ID}"
train_dir="${expt_dir}/retriever/train"
adapter_dir="${expt_dir}/retriever/model"
output_dir="${expt_dir}/retriever/outputs"
prediction_txt="${output_dir}/predictions.txt"
matched_json="${output_dir}/best_matched_with_retrieved_facts_and_questions.json"
mkdir -p "${train_dir}" "${adapter_dir}" "${output_dir}"

if [[ "${RUN_TRAIN}" == "1" ]]; then
  validate_prompt_mode_contract "${PROMPT_MODE}" "${train_csv}" "${dev_csv}" "${test_csv}"
fi

case "${BEST_MODEL}" in
  mistral|mistral_v0_3)
    mistral_ld_library_path="$(mistral_retriever_ld_library_path)"
    if [[ "${RUN_TRAIN}" == "1" ]]; then
      if [[ "${RUN_PREFLIGHT}" == "1" ]]; then
        run_gpu_preflight_log "${expt_dir}/retriever/preflight_gpu_state.log"
        run_torchrun_distributed_probe "${MISTRAL_CUDA_DEVICES}" "${MISTRAL_NUM_PROCESSES}" "${expt_dir}/retriever/preflight_torchrun_probe.log"
      fi
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
        --num-train-epochs "${LOSS_EPOCHS}" \
        --batch-size "${MISTRAL_BATCH_SIZE}" \
        --gradient-accumulation-steps "${MISTRAL_GRAD_ACCUM_STEPS}" \
        --learning-rate "${MISTRAL_LR:-5e-4}" \
        --prompt-mode "${PROMPT_MODE}" \
        --device "${MISTRAL_DEVICE}" \
        --save-steps "${MISTRAL_SAVE_STEPS}" \
        --save-total-limit "${MISTRAL_SAVE_TOTAL_LIMIT}"
      )
      if [[ -n "${MISTRAL_RESUME_FROM_CHECKPOINT}" ]]; then
        train_cmd+=(--resume-from-checkpoint "${MISTRAL_RESUME_FROM_CHECKPOINT}")
      fi
      export RETRIEVER_SPLIT_SUMMARY
      RETRIEVER_SPLIT_SUMMARY="$(retriever_split_row_summary "${train_csv}" "${dev_csv}" "${test_csv}" "${MISTRAL_MAX_TRAIN_SAMPLES}" "${MISTRAL_MAX_EVAL_SAMPLES}" "${MISTRAL_MAX_INFER_SAMPLES}")"
      run_logged "${expt_dir}/retriever/train.log" "${train_cmd[@]}"
      assert_train_log_completed "${expt_dir}/retriever/train.log" "${train_csv}" "${dev_csv}" "${test_csv}" "${MISTRAL_MAX_TRAIN_SAMPLES}" "${MISTRAL_MAX_EVAL_SAMPLES}" "${MISTRAL_MAX_INFER_SAMPLES}"
    fi
    if [[ "${RUN_INFER}" == "1" || "${RUN_MATCH}" == "1" ]]; then
      assert_train_log_current_for_reuse \
        "${expt_dir}/retriever/train.log" \
        "${train_csv}" \
        "${dev_csv}" \
        "${test_csv}" \
        "${MISTRAL_MAX_TRAIN_SAMPLES}" \
        "${MISTRAL_MAX_EVAL_SAMPLES}" \
        "${MISTRAL_MAX_INFER_SAMPLES}" \
        "Mistral retriever loss-diagram"
    fi
    if [[ "${RUN_INFER}" == "1" ]]; then
      require_adapter_artifact "${adapter_dir}" "Mistral retriever loss-diagram" "${PROMPT_MODE}" "${LOSS_EXPT_ID}"
      run_logged "${expt_dir}/retriever/inference.log" \
        env CUDA_VISIBLE_DEVICES="${MISTRAL_INFER_CUDA_DEVICES}" \
        LD_LIBRARY_PATH="${mistral_ld_library_path}" \
        PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
        PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF}" \
        conda run --no-capture-output -n "${CONDA_ENV}" \
        python -B "${REPO_ROOT}/.external/FINDER/Retriever Codes/Mistral/mistral_inference.py" \
        --input-csv "${test_csv}" \
        --adapter-dir "${adapter_dir}" \
        --output-txt "${prediction_txt}" \
        --device "${MISTRAL_INFER_DEVICE:-0}" \
        --prompt-mode "${PROMPT_MODE}" \
        --max-infer-samples "${MISTRAL_MAX_INFER_SAMPLES}" \
        --batch-size "${MISTRAL_INFER_BATCH_SIZE}" \
        --max-new-tokens "${MISTRAL_MAX_NEW_TOKENS}" \
        --structured-output "${MISTRAL_STRUCTURED_OUTPUT}"
    fi
    retriever_model="mistral_v0_3"
    ;;
  flan|flan_t5_large)
    if [[ "${RUN_TRAIN}" == "1" ]]; then
      if [[ "${RUN_PREFLIGHT}" == "1" ]]; then
        run_gpu_preflight_log "${expt_dir}/retriever/preflight_gpu_state.log"
        run_torchrun_distributed_probe "${FLAN_CUDA_DEVICES}" "${FLAN_NUM_PROCESSES}" "${expt_dir}/retriever/preflight_torchrun_probe.log"
      fi
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
        --num-train-epochs "${LOSS_EPOCHS}" \
        --batch-size "${FLAN_BATCH_SIZE}" \
        --gradient-accumulation-steps "${FLAN_GRAD_ACCUM_STEPS}" \
        --learning-rate "${FLAN_LR:-5e-4}" \
        --prompt-mode "${PROMPT_MODE}" \
        --save-steps "${FLAN_SAVE_STEPS}" \
        --save-total-limit "${FLAN_SAVE_TOTAL_LIMIT}"
      )
      if [[ -n "${FLAN_RESUME_FROM_CHECKPOINT}" ]]; then
        train_cmd+=(--resume-from-checkpoint "${FLAN_RESUME_FROM_CHECKPOINT}")
      fi
      export RETRIEVER_SPLIT_SUMMARY
      RETRIEVER_SPLIT_SUMMARY="$(retriever_split_row_summary "${train_csv}" "${dev_csv}" "${test_csv}" "${FLAN_MAX_TRAIN_SAMPLES}" "${FLAN_MAX_EVAL_SAMPLES}" "${FLAN_MAX_INFER_SAMPLES}")"
      run_logged "${expt_dir}/retriever/train.log" "${train_cmd[@]}"
      assert_train_log_completed "${expt_dir}/retriever/train.log" "${train_csv}" "${dev_csv}" "${test_csv}" "${FLAN_MAX_TRAIN_SAMPLES}" "${FLAN_MAX_EVAL_SAMPLES}" "${FLAN_MAX_INFER_SAMPLES}"
    fi
    if [[ "${RUN_INFER}" == "1" || "${RUN_MATCH}" == "1" ]]; then
      assert_train_log_current_for_reuse \
        "${expt_dir}/retriever/train.log" \
        "${train_csv}" \
        "${dev_csv}" \
        "${test_csv}" \
        "${FLAN_MAX_TRAIN_SAMPLES}" \
        "${FLAN_MAX_EVAL_SAMPLES}" \
        "${FLAN_MAX_INFER_SAMPLES}" \
        "FLAN retriever loss-diagram"
    fi
    if [[ "${RUN_INFER}" == "1" ]]; then
      require_adapter_artifact "${adapter_dir}" "FLAN retriever loss-diagram" "${PROMPT_MODE}" "${LOSS_EXPT_ID}"
      run_logged "${expt_dir}/retriever/inference.log" \
        env CUDA_VISIBLE_DEVICES="${FLAN_INFER_CUDA_DEVICES}" \
        PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
        PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF}" \
        conda run --no-capture-output -n "${CONDA_ENV}" \
        python -B "${REPO_ROOT}/.external/FINDER/Retriever Codes/Flan/lora_flan_large_finqa_rel_fact.py" \
        --mode infer \
        --input-csv "${test_csv}" \
        --eval-csv "${test_csv}" \
        --adapter-dir "${adapter_dir}" \
        --output-txt "${prediction_txt}" \
        --prompt-mode "${PROMPT_MODE}" \
        --max-infer-samples "${FLAN_MAX_INFER_SAMPLES}" \
        --batch-size "${FLAN_INFER_BATCH_SIZE}" \
        --max-new-tokens "${FLAN_MAX_NEW_TOKENS}" \
        --structured-output "${FLAN_STRUCTURED_OUTPUT}"
    fi
    retriever_model="flan_t5_large"
    ;;
  t5gemma|t5gemma_2_1b_1b)
    if [[ "${RUN_TRAIN}" == "1" ]]; then
      if [[ "${RUN_PREFLIGHT}" == "1" ]]; then
        run_gpu_preflight_log "${expt_dir}/retriever/preflight_gpu_state.log"
        run_hf_model_access_probe "${T5GEMMA_BASE_MODEL}" "${expt_dir}/retriever/preflight_hf_access.log" 1
        run_torchrun_distributed_probe "${T5GEMMA_CUDA_DEVICES}" "${T5GEMMA_NUM_PROCESSES}" "${expt_dir}/retriever/preflight_torchrun_probe.log"
      fi
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
        --num-train-epochs "${LOSS_EPOCHS}" \
        --batch-size "${T5GEMMA_BATCH_SIZE}" \
        --gradient-accumulation-steps "${T5GEMMA_GRAD_ACCUM_STEPS}" \
        --learning-rate "${T5GEMMA_LR}" \
        --prompt-mode "${PROMPT_MODE}" \
        --save-steps "${T5GEMMA_SAVE_STEPS}" \
        --save-total-limit "${T5GEMMA_SAVE_TOTAL_LIMIT}"
      )
      if [[ -n "${T5GEMMA_RESUME_FROM_CHECKPOINT}" ]]; then
        train_cmd+=(--resume-from-checkpoint "${T5GEMMA_RESUME_FROM_CHECKPOINT}")
      fi
      export RETRIEVER_SPLIT_SUMMARY
      RETRIEVER_SPLIT_SUMMARY="$(retriever_split_row_summary "${train_csv}" "${dev_csv}" "${test_csv}" "${T5GEMMA_MAX_TRAIN_SAMPLES}" "${T5GEMMA_MAX_EVAL_SAMPLES}" "${T5GEMMA_MAX_INFER_SAMPLES}")"
      run_logged "${expt_dir}/retriever/train.log" "${train_cmd[@]}"
      assert_train_log_completed "${expt_dir}/retriever/train.log" "${train_csv}" "${dev_csv}" "${test_csv}" "${T5GEMMA_MAX_TRAIN_SAMPLES}" "${T5GEMMA_MAX_EVAL_SAMPLES}" "${T5GEMMA_MAX_INFER_SAMPLES}"
    fi
    if [[ "${RUN_INFER}" == "1" || "${RUN_MATCH}" == "1" ]]; then
      assert_train_log_current_for_reuse \
        "${expt_dir}/retriever/train.log" \
        "${train_csv}" \
        "${dev_csv}" \
        "${test_csv}" \
        "${T5GEMMA_MAX_TRAIN_SAMPLES}" \
        "${T5GEMMA_MAX_EVAL_SAMPLES}" \
        "${T5GEMMA_MAX_INFER_SAMPLES}" \
        "T5Gemma retriever loss-diagram"
    fi
    if [[ "${RUN_INFER}" == "1" ]]; then
      require_adapter_artifact "${adapter_dir}" "T5Gemma retriever loss-diagram" "${PROMPT_MODE}" "${LOSS_EXPT_ID}"
      run_logged "${expt_dir}/retriever/inference.log" \
        env CUDA_VISIBLE_DEVICES="${T5GEMMA_INFER_CUDA_DEVICES}" \
        PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
        PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF}" \
        conda run --no-capture-output -n "${CONDA_ENV}" \
        python -B "${REPO_ROOT}/.external/FINDER/Retriever Codes/t5gemma-2/t5gemma-2_train.py" \
        --mode infer \
        --base-model "${T5GEMMA_BASE_MODEL}" \
        --train-csv "${train_csv}" \
        --eval-csv "${test_csv}" \
        --input-csv "${test_csv}" \
        --output-dir "${train_dir}" \
        --adapter-dir "${adapter_dir}" \
        --output-txt "${prediction_txt}" \
        --prompt-mode "${PROMPT_MODE}" \
        --max-infer-samples "${T5GEMMA_MAX_INFER_SAMPLES}" \
        --batch-size "${T5GEMMA_INFER_BATCH_SIZE}" \
        --max-new-tokens "${T5GEMMA_MAX_NEW_TOKENS}" \
        --structured-output "${T5GEMMA_STRUCTURED_OUTPUT}"
    fi
    retriever_model="t5gemma_2_1b_1b"
    ;;
  *)
    printf "Unsupported BEST_MODEL=%s\n" "${BEST_MODEL}" >&2
    exit 2
    ;;
esac

if [[ "${RUN_MATCH}" == "1" ]]; then
  run_match_artifact "${retriever_model}" "${prediction_txt}" "${matched_json}" "${expt_dir}/retriever/match.log" "${PROMPT_MODE}" "${LOSS_EXPT_ID}" test
fi

if [[ "${RUN_TRAIN}" == "1" ]]; then
  printf "Loss-diagram run completed. Inspect %s for per-step loss records.\n" "${expt_dir}/retriever/train.log"
else
  printf "Loss-diagram path check completed without training.\n"
fi
