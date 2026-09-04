#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/retriever_experiment_lib.sh"

EXPT_ID="${EXPT_ID:-experiment_6_retriever_$(date -u +%Y%m%dT%H%M%SZ)}"
EXPT_DIR="${REPO_ROOT}/Experiment/${EXPT_ID}"
DEFAULT_MATRIX="6_flan_z:finqa_flan_z:narrative_zero_shot 6_flan_m:finqa_flan_m:narrative_many_shot 6_flan_d:finqa_flan_d:narrative_dynamic_shot 6_mistral_z:finqa_mistral_z:narrative_zero_shot 6_mistral_m:finqa_mistral_m:narrative_many_shot 6_mistral_d:finqa_mistral_d:narrative_dynamic_shot 6_t5gemma2_z:finqa_t5gemma2_z:narrative_zero_shot 6_t5gemma2_m:finqa_t5gemma2_m:narrative_many_shot 6_t5gemma2_d:finqa_t5gemma2_d:narrative_dynamic_shot"

export EXPT_ID
export EXPERIMENT6_BINDING_MATRIX="${EXPERIMENT6_BINDING_MATRIX:-${DEFAULT_MATRIX}}"
export EXPERIMENT6_PREPARE_GOLD_DATA="${EXPERIMENT6_PREPARE_GOLD_DATA:-0}"
export EXPERIMENT6_PREPARE_PROMPT_DATA="${EXPERIMENT6_PREPARE_PROMPT_DATA:-1}"
export EXPERIMENT6_GENERATE_BINDING_PREDICTIONS="${EXPERIMENT6_GENERATE_BINDING_PREDICTIONS:-1}"
export EXPERIMENT6_GENERATION_MODE="${EXPERIMENT6_GENERATION_MODE:-retriever}"
export EXPERIMENT6_PREPARE_CONTROLLED_DATA="${EXPERIMENT6_PREPARE_CONTROLLED_DATA:-0}"
export EXPERIMENT6_PREPARE_REAL_PREDICTIONS="${EXPERIMENT6_PREPARE_REAL_PREDICTIONS:-0}"
export EXPERIMENT6_BUILD_LIMIT="${EXPERIMENT6_BUILD_LIMIT:-0}"
export STRICT_INPUTS="${STRICT_INPUTS:-1}"
export NARRATIVE_PRED_DIR="${NARRATIVE_PRED_DIR:-${EXPT_DIR}/binding_eval_predictions}"

exec "${SCRIPT_DIR}/experiment_6_data_binding_evaluation.sh" "$@"
