#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/retriever_experiment_lib.sh"

EXPT_ID="${EXPT_ID:-experiment_6_narratives_$(date -u +%Y%m%dT%H%M%SZ)}"
EXPT_DIR="${REPO_ROOT}/Experiment/${EXPT_ID}"
DEFAULT_MATRIX="6_FinFlier_flan_z:finqa_flan_z:narrative_original 6_FinFlier_flan_m:finqa_flan_m:narrative_original 6_FinFlier_flan_d:finqa_flan_d:narrative_original 6_FinFlier_mistral_z:finqa_mistral_z:narrative_original 6_FinFlier_mistral_m:finqa_mistral_m:narrative_original 6_FinFlier_mistral_d:finqa_mistral_d:narrative_original 6_FinFlier_t5gemma2_z:finqa_t5gemma2_z:narrative_original 6_FinFlier_t5gemma2_m:finqa_t5gemma2_m:narrative_original 6_FinFlier_t5gemma2_d:finqa_t5gemma2_d:narrative_original 6_FinFlier_qwen:qwen3_6:narrative_original 6_FinFlier_llama:llama3_3:narrative_original 6_FinFlier_mistral4:mistral4:narrative_original 6_FinFlier_mistral_base:mistral_v0_3:narrative_original 6_FinFlier_flan_base:flan_t5_large:narrative_original 6_FinFlier_t5gemma2_base:t5gemma_2_1b_1b:narrative_original 6_FinFlier_gpt5.3-CodexS:gpt5_3_codexS:narrative_original 6_FinFlier_gpt5.5:gpt5_5:narrative_original"

export EXPT_ID
export EXPERIMENT6_BINDING_MATRIX="${EXPERIMENT6_BINDING_MATRIX:-${DEFAULT_MATRIX}}"
export EXPERIMENT6_PREPARE_GOLD_DATA="${EXPERIMENT6_PREPARE_GOLD_DATA:-0}"
export EXPERIMENT6_PREPARE_PROMPT_DATA="${EXPERIMENT6_PREPARE_PROMPT_DATA:-1}"
export EXPERIMENT6_GENERATE_BINDING_PREDICTIONS="${EXPERIMENT6_GENERATE_BINDING_PREDICTIONS:-1}"
export EXPERIMENT6_GENERATION_MODE="${EXPERIMENT6_GENERATION_MODE:-mixed}"
export EXPERIMENT6_PREPARE_CONTROLLED_DATA="${EXPERIMENT6_PREPARE_CONTROLLED_DATA:-0}"
export EXPERIMENT6_PREPARE_REAL_PREDICTIONS="${EXPERIMENT6_PREPARE_REAL_PREDICTIONS:-0}"
export EXPERIMENT6_BUILD_LIMIT="${EXPERIMENT6_BUILD_LIMIT:-0}"
export STRICT_INPUTS="${STRICT_INPUTS:-1}"
export NARRATIVE_PRED_DIR="${NARRATIVE_PRED_DIR:-${EXPT_DIR}/binding_eval_predictions}"

exec "${SCRIPT_DIR}/experiment_6_data_binding_evaluation.sh" "$@"
