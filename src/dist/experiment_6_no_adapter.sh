#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/retriever_experiment_lib.sh"

EXPT_ID="${EXPT_ID:-experiment_6_no_adapter_$(date -u +%Y%m%dT%H%M%SZ)}"
EXPT_DIR="${REPO_ROOT}/Experiment/${EXPT_ID}"
DEFAULT_MATRIX="6_mistral_base_z:mistral_v0_3:narrative_zero_shot 6_mistral_base_m:mistral_v0_3:narrative_many_shot 6_mistral_base_d:mistral_v0_3:narrative_dynamic_shot 6_flan_base_z:flan_t5_large:narrative_zero_shot 6_flan_base_m:flan_t5_large:narrative_many_shot 6_flan_base_d:flan_t5_large:narrative_dynamic_shot 6_t5gemma2_base_z:t5gemma_2_1b_1b:narrative_zero_shot 6_t5gemma2_base_m:t5gemma_2_1b_1b:narrative_many_shot 6_t5gemma2_base_d:t5gemma_2_1b_1b:narrative_dynamic_shot 6_gpt5.3-CodexS_z:gpt5_3_codexS:narrative_zero_shot 6_gpt5.3-CodexS_m:gpt5_3_codexS:narrative_many_shot 6_gpt5.3-CodexS_d:gpt5_3_codexS:narrative_dynamic_shot 6_gpt5.5_z:gpt5_5:narrative_zero_shot 6_gpt5.5_m:gpt5_5:narrative_many_shot 6_gpt5.5_d:gpt5_5:narrative_dynamic_shot 6_qwen_z:qwen3_6:narrative_zero_shot 6_qwen_m:qwen3_6:narrative_many_shot 6_qwen_d:qwen3_6:narrative_dynamic_shot 6_mistral4_z:mistral4:narrative_zero_shot 6_mistral4_m:mistral4:narrative_many_shot 6_mistral4_d:mistral4:narrative_dynamic_shot 6_llama_z:llama3_3:narrative_zero_shot 6_llama_m:llama3_3:narrative_many_shot 6_llama_d:llama3_3:narrative_dynamic_shot"

export EXPT_ID
export EXPERIMENT6_BINDING_MATRIX="${EXPERIMENT6_BINDING_MATRIX:-${DEFAULT_MATRIX}}"
export EXPERIMENT6_PREPARE_GOLD_DATA="${EXPERIMENT6_PREPARE_GOLD_DATA:-0}"
export EXPERIMENT6_PREPARE_PROMPT_DATA="${EXPERIMENT6_PREPARE_PROMPT_DATA:-1}"
export EXPERIMENT6_GENERATE_BINDING_PREDICTIONS="${EXPERIMENT6_GENERATE_BINDING_PREDICTIONS:-1}"
export EXPERIMENT6_GENERATION_MODE="${EXPERIMENT6_GENERATION_MODE:-no-adapter}"
export EXPERIMENT6_PREPARE_CONTROLLED_DATA="${EXPERIMENT6_PREPARE_CONTROLLED_DATA:-0}"
export EXPERIMENT6_PREPARE_REAL_PREDICTIONS="${EXPERIMENT6_PREPARE_REAL_PREDICTIONS:-0}"
export EXPERIMENT6_BUILD_LIMIT="${EXPERIMENT6_BUILD_LIMIT:-0}"
export STRICT_INPUTS="${STRICT_INPUTS:-1}"
export NARRATIVE_PRED_DIR="${NARRATIVE_PRED_DIR:-${EXPT_DIR}/binding_eval_predictions}"

exec "${SCRIPT_DIR}/experiment_6_data_binding_evaluation.sh" "$@"
