#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/retriever_experiment_lib.sh"
source "${SCRIPT_DIR}/generator_resource_guard.sh"

EXPT_ID="${EXPT_ID:-experiment_7_generator_answer}"
FLOW_SCOPE="formal_final_answer_computation_cache_only"
DEFAULT_EXPERIMENT7_ENGINES="qwen3_6 llama4 gpt4_1 gpt5_3_codexS gpt5_5"
DEFAULT_EXPERIMENT7_MATRIX="finqa_flan_o:finqa_test finqa_flan_o:finqa_dev finqa_flan_z:finqa_test finqa_flan_z:finqa_dev finqa_flan_m:finqa_test finqa_flan_m:finqa_dev finqa_flan_d:finqa_test finqa_flan_d:finqa_dev finqa_mistral_o:finqa_test finqa_mistral_o:finqa_dev finqa_mistral_z:finqa_test finqa_mistral_z:finqa_dev finqa_mistral_m:finqa_test finqa_mistral_m:finqa_dev finqa_mistral_d:finqa_test finqa_mistral_d:finqa_dev finqa_t5gemma2_o:finqa_test finqa_t5gemma2_o:finqa_dev finqa_t5gemma2_z:finqa_test finqa_t5gemma2_z:finqa_dev finqa_t5gemma2_m:finqa_test finqa_t5gemma2_m:finqa_dev finqa_t5gemma2_d:finqa_test finqa_t5gemma2_d:finqa_dev"
ENGINES="${ENGINES:-${DEFAULT_EXPERIMENT7_ENGINES}}"
PROFILE="${PROFILE:-greedy}"
LIMIT="${LIMIT:--1}"
EXPERIMENT7_SAMPLE_STRATEGY="${EXPERIMENT7_SAMPLE_STRATEGY:-first}"
MAX_TOKENS="${MAX_TOKENS:-128}"
QWEN3_6_ENABLE_THINKING="${QWEN3_6_ENABLE_THINKING:-true}"
QWEN3_6_DEV_ENABLE_THINKING="${QWEN3_6_DEV_ENABLE_THINKING:-${QWEN3_6_ENABLE_THINKING}}"
export QWEN3_6_ENABLE_THINKING
export QWEN3_6_DEV_ENABLE_THINKING
MISTRAL4_REASONING_EFFORT="${MISTRAL4_REASONING_EFFORT:-high}"
MISTRAL4_REASONING_TEMPERATURE="${MISTRAL4_REASONING_TEMPERATURE:-0.7}"
MISTRAL4_REASONING_TOP_P="${MISTRAL4_REASONING_TOP_P:-1.0}"
MISTRAL4_REASONING_EFFORT_NORMALIZED="$(printf "%s" "${MISTRAL4_REASONING_EFFORT}" | tr '[:upper:]' '[:lower:]')"
case "${MISTRAL4_REASONING_EFFORT_NORMALIZED}" in
  ""|none|off|0|false) ;;
  *) LLAMA_CPP_ALLOW_REASONING_EFFORT="${LLAMA_CPP_ALLOW_REASONING_EFFORT:-1}" ;;
esac
export MISTRAL4_REASONING_EFFORT MISTRAL4_REASONING_TEMPERATURE MISTRAL4_REASONING_TOP_P LLAMA_CPP_ALLOW_REASONING_EFFORT
RUN_EXECUTE="${RUN_EXECUTE:-0}"
REGENERATE_EA_ONLY="${REGENERATE_EA_ONLY:-0}"
EXPERIMENT7_EA_ID_SUFFIX="${EXPERIMENT7_EA_ID_SUFFIX:-}"
UPDATE_EA_LATEST="${UPDATE_EA_LATEST:-1}"
case "${UPDATE_EA_LATEST}" in
  0|1) ;;
  *) printf 'UPDATE_EA_LATEST must be 0 or 1.\n' >&2; exit 2 ;;
esac
ALLOW_FALLBACK_SMOKE_EXECUTE="${ALLOW_FALLBACK_SMOKE_EXECUTE:-0}"
FAIL_FAST_ON_EXECUTE_ERROR="${FAIL_FAST_ON_EXECUTE_ERROR:-1}"
STRICT_INPUTS="${STRICT_INPUTS:-0}"
SLEEP_SECONDS="${SLEEP_SECONDS:-0}"
SHOW_PROMPT="${SHOW_PROMPT:-1}"
RESUME_OUTPUT="${RESUME_OUTPUT:-0}"
GPT_RETRY_AFTER_SECONDS="${GPT_RETRY_AFTER_SECONDS:-14400}"
GPT_MAX_QUOTA_RETRIES="${GPT_MAX_QUOTA_RETRIES:-1}"
RUN_RETRIEVER_INFER="${RUN_RETRIEVER_INFER:-1}"
FORCE_REBUILD_RETRIEVER="${FORCE_REBUILD_RETRIEVER:-0}"
RETRIEVER_MAX_INFER_SAMPLES="${RETRIEVER_MAX_INFER_SAMPLES:-${LIMIT}}"
RETRIEVER_INFER_CUDA_DEVICES="${RETRIEVER_INFER_CUDA_DEVICES:-${INFER_CUDA_DEVICES:-1}}"
EXPERIMENT7_RETRIEVER_INFER_BATCH_SIZE="${EXPERIMENT7_RETRIEVER_INFER_BATCH_SIZE:-${EXPERIMENT6_RETRIEVER_INFER_BATCH_SIZE:-${FLAN_INFER_BATCH_SIZE:-16}}}"
EXPERIMENT7_T5GEMMA_INFER_BATCH_SIZE="${EXPERIMENT7_T5GEMMA_INFER_BATCH_SIZE:-${T5GEMMA_BATCH_SIZE:-8}}"
EXPERIMENT7_MATCH_EMBED_BATCH_SIZE="${EXPERIMENT7_MATCH_EMBED_BATCH_SIZE:-${EXPERIMENT6_MATCH_EMBED_BATCH_SIZE:-${MATCH_EMBED_BATCH_SIZE}}}"
EXPERIMENT7_MATRIX="${EXPERIMENT7_MATRIX:-${EXPERIMENT6_MATRIX:-${DEFAULT_EXPERIMENT7_MATRIX}}}"
GPT5_3_CODEX_ROUTE="${GPT5_3_CODEX_ROUTE:-api_key}"
export GPT5_3_CODEX_ROUTE
EXAMPLE_SELECTION_MODE="${EXAMPLE_SELECTION_MODE:-cache}"
EXAMPLE_SELECTION_SHOT_NUMBER="${EXAMPLE_SELECTION_SHOT_NUMBER:-4}"
EXAMPLE_SELECTION_REQUIRE_POLICY="${EXAMPLE_SELECTION_REQUIRE_POLICY:-0}"
EXAMPLE_SELECTION_REQUIRE_CACHE="${EXAMPLE_SELECTION_REQUIRE_CACHE:-1}"
FORMAL_FINDER_READY="${FORMAL_FINDER_READY:-1}"
ALLOW_LEGACY_SELECTION_BINDING="${ALLOW_LEGACY_SELECTION_BINDING:-0}"
EXPERIMENT7_REQUIRE_TARGET_SELECTION_CACHE="${EXPERIMENT7_REQUIRE_TARGET_SELECTION_CACHE:-${FORMAL_FINDER_READY}}"
EXPERIMENT7_REQUIRE_BINDING_AUDIT="${EXPERIMENT7_REQUIRE_BINDING_AUDIT:-${FORMAL_FINDER_READY}}"
EXPERIMENT7_ALLOW_MATERIALIZED_SELECTION_CACHE="${EXPERIMENT7_ALLOW_MATERIALIZED_SELECTION_CACHE:-0}"
EXPERIMENT7_SELECTION_EXPT_ID="${EXPERIMENT7_SELECTION_EXPT_ID:-${IN_CONTEXT_SELECTION_EXPT_ID:-}}"
EXPERIMENT7_SELECTION_ENGINE="${EXPERIMENT7_SELECTION_ENGINE:-gpt5_5}"
EXAMPLE_SELECTION_CACHE_ROOT="${EXAMPLE_SELECTION_CACHE_ROOT:-}"
EXPERIMENT7_SELECTION_CACHE_BINDING_ROOT="${EXPERIMENT7_SELECTION_CACHE_BINDING_ROOT:-}"
EXAMPLE_SELECTION_CANDIDATE_JSON="${EXAMPLE_SELECTION_CANDIDATE_JSON:-}"
EXAMPLE_SELECTION_POLICY_OUTPUT="${EXAMPLE_SELECTION_POLICY_OUTPUT:-}"
PROMPT_TYPE_TRAIN_CSV="${PROMPT_TYPE_TRAIN_CSV:-}"
TARGET_PROMPT_TYPE="${TARGET_PROMPT_TYPE:-}"
export EXAMPLE_SELECTION_MODE EXAMPLE_SELECTION_SHOT_NUMBER EXAMPLE_SELECTION_REQUIRE_POLICY EXAMPLE_SELECTION_REQUIRE_CACHE
export FORMAL_FINDER_READY ALLOW_LEGACY_SELECTION_BINDING EXPERIMENT7_REQUIRE_TARGET_SELECTION_CACHE EXPERIMENT7_REQUIRE_BINDING_AUDIT
export EXPERIMENT7_ALLOW_MATERIALIZED_SELECTION_CACHE EXPERIMENT7_SELECTION_EXPT_ID EXPERIMENT7_SELECTION_ENGINE EXAMPLE_SELECTION_CACHE_ROOT
export EXPERIMENT7_SELECTION_CACHE_BINDING_ROOT
export EXAMPLE_SELECTION_CANDIDATE_JSON EXAMPLE_SELECTION_POLICY_OUTPUT PROMPT_TYPE_TRAIN_CSV TARGET_PROMPT_TYPE
if [[ "${FORMAL_FINDER_READY}" == "1" ]]; then
  if [[ "${EXAMPLE_SELECTION_MODE}" != "cache" ]]; then
    printf "Formal Experiment 7 requires EXAMPLE_SELECTION_MODE=cache when FORMAL_FINDER_READY=1.\n" >&2
    exit 2
  fi
  if [[ "${EXAMPLE_SELECTION_REQUIRE_CACHE}" != "1" ]]; then
    printf "Formal Experiment 7 requires EXAMPLE_SELECTION_REQUIRE_CACHE=1 when FORMAL_FINDER_READY=1.\n" >&2
    exit 2
  fi
fi
EXPT_DIR="${REPO_ROOT}/Experiment/${EXPT_ID}"
STATUS_JSON="${EXPT_DIR}/generator/execution_status.json"
TIMELINE_JSONL="${EXPT_DIR}/generator/timeline.jsonl"
RUN_SHELL_ID="$(generator_shell_id)"

mkdir -p "${EXPT_DIR}/generator"

write_timeline_event() {
  local phase="$1"
  local status="$2"
  local engine="${3:-}"
  local retriever_id="${4:-}"
  local dataset_id="${5:-}"
  local detail="${6:-}"
  local case_timeline="${7:-}"
  local now
  now="$(utc_now)"
  TIMELINE_TIME="${now}" \
  TIMELINE_EXPT_ID="${EXPT_ID}" \
  TIMELINE_SHELL_ID="${RUN_SHELL_ID}" \
  TIMELINE_PHASE="${phase}" \
  TIMELINE_STATUS="${status}" \
  TIMELINE_ENGINE="${engine}" \
  TIMELINE_RETRIEVER_ID="${retriever_id}" \
  TIMELINE_DATASET_ID="${dataset_id}" \
  TIMELINE_DETAIL="${detail}" \
  python3 - <<'PYTIMELINE' >> "${TIMELINE_JSONL}"
import json, os
payload = {
    "time": os.environ["TIMELINE_TIME"],
    "expt_id": os.environ["TIMELINE_EXPT_ID"],
    "shell_id": os.environ["TIMELINE_SHELL_ID"],
    "phase": os.environ["TIMELINE_PHASE"],
    "status": os.environ["TIMELINE_STATUS"],
    "engine": os.environ.get("TIMELINE_ENGINE") or None,
    "retriever_id": os.environ.get("TIMELINE_RETRIEVER_ID") or None,
    "dataset": os.environ.get("TIMELINE_DATASET_ID") or None,
    "detail": os.environ.get("TIMELINE_DETAIL") or None,
}
print(json.dumps(payload, ensure_ascii=False))
PYTIMELINE
  if [[ -n "${case_timeline}" ]]; then
    tail -n 1 "${TIMELINE_JSONL}" >> "${case_timeline}"
  fi
}

set_overall_rc() {
  local rc="$1"
  if [[ "${overall_rc}" -eq 0 && "${rc}" -ne 0 ]]; then
    overall_rc="${rc}"
  fi
}

sanitize_id() {
  printf "%s" "$1" | tr '[:lower:]-' '[:upper:]_' | tr -cd '[:alnum:]_'
}

case_dir_name() {
  printf "%s_%s\n" "$1" "$2" | tr '-' '_'
}

prediction_alias_for_dataset() {
  case "$1" in
    finqa_test|finqa_test_*) printf "predictions_test.txt\n" ;;
    finqa_dev|finqa_dev_*) printf "predictions_dev.txt\n" ;;
    *) return 1 ;;
  esac
}

ensure_prediction_alias() {
  local prediction_txt="$1"
  local dataset_id="$2"
  local source_dir="$3"
  local alias_name=""
  alias_name="$(prediction_alias_for_dataset "${dataset_id}" || true)"
  if [[ -z "${alias_name}" || ! -s "${prediction_txt}" ]]; then
    return 0
  fi
  cp -p "${prediction_txt}" "${source_dir}/${alias_name}"
}

first_existing_file() {
  local candidate
  for candidate in "$@"; do
    if [[ -n "${candidate}" && -f "${candidate}" ]]; then
      printf "%s\n" "${candidate}"
      return 0
    fi
  done
  return 1
}

normalize_engine_alias() {
  case "$1" in
    qwen|qwen3_6|qwen3_6_35b_a3b_fp8|qwen3.6-35b-a3b-fp8) printf "qwen3_6\n" ;;
    mistral|mistral4|mistral_small_4|mistral-small-4) printf "mistral4\n" ;;
    llama|llama4|llama4_scout|llama-4|llama-4-scout-17b-16e-instruct) printf "llama4\n" ;;
    llama3_3|llama3.3|llama-3.3) printf "llama3_3\n" ;;
    gpt55|gpt-5.5) printf "gpt5_5\n" ;;
    gpt5_3_codex|gpt5_3_codexS|gpt-5.3-codex-spark) printf "gpt5_3_codexS\n" ;;
    gpt41|gpt4.1|gpt-4.1) printf "gpt4_1\n" ;;
    *) printf "%s\n" "$1" ;;
  esac
}

normalize_selection_engine_alias() {
  normalize_engine_alias "$1"
}

canonical_retriever_id() {
  case "$1" in
    finqa_flan_r|flan_r) printf "finqa_flan_r\n" ;;
    finqa_flan_o|flan_o) printf "finqa_flan_o\n" ;;
    finqa_flan_z|flan_z) printf "finqa_flan_z\n" ;;
    finqa_flan_m|flan_m) printf "finqa_flan_m\n" ;;
    finqa_flan_d|flan_d) printf "finqa_flan_d\n" ;;
    finqa_Mistral_r|finqa_mistral_r|mistral_r) printf "finqa_mistral_r\n" ;;
    finqa_Mistral_o|finqa_mistral_o|mistral_o) printf "finqa_mistral_o\n" ;;
    finqa_Mistral_z|finqa_mistral_z|mistral_z) printf "finqa_mistral_z\n" ;;
    finqa_Mistral_m|finqa_mistral_m|mistral_m) printf "finqa_mistral_m\n" ;;
    finqa_Mistral_d|finqa_mistral_d|mistral_d) printf "finqa_mistral_d\n" ;;
    finqa_t5gemma2_o|finqa_t5gemma_o|t5gemma2_o|t5gemma_o) printf "finqa_t5gemma2_o\n" ;;
    finqa_t5gemma2_z|finqa_t5gemma_z|t5gemma2_z|t5gemma_z) printf "finqa_t5gemma2_z\n" ;;
    finqa_t5gemma2_m|finqa_t5gemma_m|t5gemma2_m|t5gemma_m) printf "finqa_t5gemma2_m\n" ;;
    finqa_t5gemma2_d|finqa_t5gemma_d|t5gemma2_d|t5gemma_d) printf "finqa_t5gemma2_d\n" ;;
    finqa10_formal_smoke|finqa10_smoke|finqa_10_formal_smoke) printf "finqa10_formal_smoke\n" ;;
    finqa_train_formal|finqa_train_shared_selection) printf "finqa_train_formal\n" ;;
    apollo) printf "apollo\n" ;;
    *) printf "%s\n" "$1" ;;
  esac
}

is_test_dataset_id() {
  case "$1" in
    finqa_test|finqa_test_original|finqa_test_zero_shot|finqa_test_many_shot|finqa_test_dynamic_shot|apollo) return 0 ;;
    *) return 1 ;;
  esac
}

is_dev_dataset_id() {
  case "$1" in
    finqa_dev|finqa_dev_original|finqa_dev_zero_shot|finqa_dev_many_shot|finqa_dev_dynamic_shot) return 0 ;;
    *) return 1 ;;
  esac
}

is_formal_csv_source() {
  case "$(canonical_retriever_id "$1")" in
    finqa10_formal_smoke|finqa_train_formal) return 0 ;;
    *) return 1 ;;
  esac
}

formal_csv_source_mode_for() {
  case "$(canonical_retriever_id "$1")" in
    finqa10_formal_smoke) printf "finqa10_formal_smoke\n" ;;
    finqa_train_formal) printf "finqa_train_formal\n" ;;
    *) return 2 ;;
  esac
}

source_experiment_candidates() {
  case "$(canonical_retriever_id "$1")" in
    finqa_flan_r) printf "%s\n" finqa_flan_r old_finqa_flan_r ;;
    finqa_flan_o) printf "%s\n" finqa_flan_o old_finqa_flan_o ;;
    finqa_flan_z) printf "%s\n" finqa_flan_z finqa_flan_z_new old_finqa_flan_z finqa_flan_z_assembler_few10_current ;;
    finqa_flan_m) printf "%s\n" finqa_flan_m finqa_flan_m_new old_finqa_flan_m ;;
    finqa_flan_d) printf "%s\n" finqa_flan_d finqa_flan_d_new old_finqa_flan_d finqa_flan_d_preflight_all_prompt_smoke ;;
    finqa_mistral_r) printf "%s\n" finqa_mistral_r finqa_Mistral_r old_finqa_Mistral_r ;;
    finqa_mistral_o) printf "%s\n" finqa_mistral_o finqa_Mistral_o old_finqa_Mistral_o ;;
    finqa_mistral_z) printf "%s\n" finqa_mistral_z finqa_mistral_z_new finqa_Mistral_z old_finqa_Mistral_z finqa_Mistral_z_assembler_few10_current ;;
    finqa_mistral_m) printf "%s\n" finqa_mistral_m finqa_mistral_m_new finqa_Mistral_m old_finqa_Mistral_m ;;
    finqa_mistral_d) printf "%s\n" finqa_mistral_d finqa_mistral_d_new finqa_Mistral_d old_finqa_Mistral_d ;;
    finqa_t5gemma2_o) printf "%s\n" finqa_t5gemma2_o old_finqa_t5gemma2_o ;;
    finqa_t5gemma2_z) printf "%s\n" finqa_t5gemma2_z finqa_t5gemma2_z_assembler_few10_current old_finqa_t5gemma2_z ;;
    finqa_t5gemma2_m) printf "%s\n" finqa_t5gemma2_m old_finqa_t5gemma2_m ;;
    finqa_t5gemma2_d) printf "%s\n" finqa_t5gemma2_d old_finqa_t5gemma2_d ;;
    *) printf "%s\n" "$1" ;;
  esac
}

retriever_model_for_source() {
  case "$(canonical_retriever_id "$1")" in
    finqa_flan_*) printf "flan_t5_large\n" ;;
    finqa_mistral_*) printf "mistral_v0_3\n" ;;
    finqa_t5gemma2_*) printf "t5gemma_2_1b_1b\n" ;;
    *) return 2 ;;
  esac
}

matched_artifact_for() {
  local retriever_id="$1"
  local dataset_id="$2"
  local cache_path="${3:-}"
  local candidate
  local override_var
  override_var="MATCHED_JSON_$(sanitize_id "${retriever_id}_${dataset_id}")"
  if [[ -n "${!override_var:-}" ]]; then
    printf "%s\n" "${!override_var}"
    return 0
  fi

  if [[ -n "${cache_path}" && -f "${cache_path}" ]]; then
    printf "%s\n" "${cache_path}"
    return 0
  fi

  if [[ "$(canonical_retriever_id "${retriever_id}")" == "apollo" ]]; then
    case "${dataset_id}" in
      apollo|finqa_test)
        first_existing_file \
          "${REPO_ROOT}/Data_Target_Module/Apollo/output/best_matched_with_retrieved_facts_and_questions_apollo.json" \
          || printf "\n"
        ;;
      *) printf "\n" ;;
    esac
    return 0
  fi

  if [[ "${dataset_id}" == finqa_dev* ]]; then
    local canonical_retriever
    canonical_retriever="$(canonical_retriever_id "${retriever_id}")"
    first_existing_file \
      "${REPO_ROOT}/Experiment/experiment_7_target_selection_gpt55_all_cases_20260612T012548Z/retriever_sources/${canonical_retriever}_finqa_dev/best_matched_with_retrieved_facts_and_questions.json" \
      && return 0
    if [[ "${ALLOW_EXPERIMENT7_DEV_RETFACT_FALLBACK:-0}" != "1" ]]; then
      printf "\n"
      return 0
    fi
    first_existing_file \
      "${REPO_ROOT}/Experiment/experiment_7_matrix_smoke_gpt41_limit1_20260612T0216Z/retriever_sources/${canonical_retriever}_finqa_dev/best_matched_with_retrieved_facts_and_questions.json" \
      "${REPO_ROOT}/Experiment/experiment_7_smoke_llama_gpt41_flan_o_20260612T0224Z/retriever_sources/${canonical_retriever}_finqa_dev/best_matched_with_retrieved_facts_and_questions.json" \
      && return 0
    shopt -s nullglob
    for candidate in "${REPO_ROOT}"/Experiment/*/retriever_sources/"${canonical_retriever}_finqa_dev"/best_matched_with_retrieved_facts_and_questions.json; do
      if [[ -f "${candidate}" ]]; then
        shopt -u nullglob
        printf "%s\n" "${candidate}"
        return 0
      fi
    done
    shopt -u nullglob
    printf "\n"
    return 0
  fi

  if ! is_test_dataset_id "${dataset_id}"; then
    printf "\n"
    return 0
  fi

  for candidate in $(source_experiment_candidates "${retriever_id}"); do
    first_existing_file \
      "${REPO_ROOT}/Experiment/${candidate}/retriever/outputs/best_matched_with_retrieved_facts_and_questions.json" \
      "${REPO_ROOT}/Experiment/${candidate}/retriever_0.3/outputs/best_matched_with_retrieved_facts_and_questions.json" \
      "${REPO_ROOT}/Experiment/${candidate}/retriever_/outputs/best_matched_with_retrieved_facts_and_questions.json" \
      && return 0
    if [[ "$(canonical_retriever_id "${retriever_id}")" != finqa_mistral_* ]]; then
      first_existing_file \
        "${REPO_ROOT}/Experiment/${candidate}/retriever0.2/outputs/best_matched_with_retrieved_facts_and_questions.json" \
        && return 0
    fi
  done
  printf "\n"
}

prompt_mode_for_source() {
  case "$(canonical_retriever_id "$1")" in
    *_r) printf "raw\n" ;;
    *_o) printf "original\n" ;;
    *_z) printf "zero-shot\n" ;;
    *_m) printf "many-shot\n" ;;
    *_d) printf "dynamic-shot\n" ;;
    *) return 2 ;;
  esac
}

target_prompt_type_for_case() {
  local retriever_id="$1"
  local dataset_id="$2"
  if [[ -n "${TARGET_PROMPT_TYPE}" ]]; then
    printf "%s\n" "${TARGET_PROMPT_TYPE}"
    return 0
  fi
  if prompt_mode_for_source "${retriever_id}" >/dev/null 2>&1; then
    prompt_mode_for_source "${retriever_id}"
    return 0
  fi
  case "${dataset_id}" in
    *zero_shot*|*zero-shot*) printf "zero-shot\n" ;;
    *many_shot*|*many-shot*) printf "many-shot\n" ;;
    *dynamic_shot*|*dynamic-shot*) printf "dynamic-shot\n" ;;
    *original*|finqa_train|finqa_test|finqa_dev|finqa_10|finqa10) printf "original\n" ;;
    *) return 2 ;;
  esac
}

materialized_selection_jsonl_for_case() {
  local dataset_id="$1"
  local target_prompt_type="$2"
  if [[ -z "${EXPERIMENT7_SELECTION_CACHE_BINDING_ROOT}" ]]; then
    printf "\n"
    return 0
  fi
  local dataset_dir=""
  case "${dataset_id}" in
    finqa_test|finqa_test_original|finqa_test_zero_shot|finqa_test_many_shot|finqa_test_dynamic_shot) dataset_dir="finqa_test" ;;
    finqa_dev|finqa_dev_original|finqa_dev_zero_shot|finqa_dev_many_shot|finqa_dev_dynamic_shot) dataset_dir="finqa_dev" ;;
    *) printf "\n"; return 0 ;;
  esac
  local candidate="${EXPERIMENT7_SELECTION_CACHE_BINDING_ROOT%/}/${dataset_dir}/${target_prompt_type}/materialized_selected_examples.jsonl"
  if [[ -f "${candidate}" ]]; then
    printf "%s\n" "${candidate}"
    return 0
  fi
  printf "\n"
}

prompt_type_train_csv_for() {
  local prompt_type="$1"
  if [[ -n "${PROMPT_TYPE_TRAIN_CSV}" ]]; then
    printf "%s\n" "${PROMPT_TYPE_TRAIN_CSV}"
    return 0
  fi
  case "${prompt_type}" in
    raw) printf "%s/data/src/FINDER/finqa_train_rel_fact_instruction.csv\n" "${WORKSPACE_ROOT}" ;;
    original) printf "%s/data/finqa_original/finqa_train_rel_fact_instruction.csv\n" "${WORKSPACE_ROOT}" ;;
    zero-shot) printf "%s/data/finqa_zero_shot/finqa_train_rel_fact_instruction.csv\n" "${WORKSPACE_ROOT}" ;;
    many-shot) printf "%s/data/finqa_many_shot/finqa_train_rel_fact_instruction.csv\n" "${WORKSPACE_ROOT}" ;;
    dynamic-shot) printf "%s/data/finqa_dynamic_shot/finqa_train_rel_fact_instruction.csv\n" "${WORKSPACE_ROOT}" ;;
    *) return 2 ;;
  esac
}

selection_cache_matches_source_mode() {
  local candidate="$1"
  local expected_source_mode="$2"
  if [[ ! -f "${candidate}" ]]; then
    return 1
  fi
  if [[ -z "${expected_source_mode}" ]]; then
    return 0
  fi
  CACHE_JSON="${candidate}" EXPECTED_SOURCE_MODE="${expected_source_mode}" python3 - <<'PYCACHE'
import json
import os
import sys
from pathlib import Path
path = Path(os.environ["CACHE_JSON"])
expected = os.environ["EXPECTED_SOURCE_MODE"]
try:
    payload = json.loads(path.read_text(encoding="utf-8"))
except Exception:
    sys.exit(1)
source_mode = payload.get("source_mode") or payload.get("metadata", {}).get("source_mode")
source = str(payload.get("source") or payload.get("metadata", {}).get("source") or "")

def infer_source_mode_from_path(value: str) -> str | None:
    value = value.replace("\\", "/")
    if value.endswith("finqa_10_rel_fact_instruction.csv") and "/data/testing/" in value:
        return "finqa10_formal_smoke"
    if value.endswith("finqa_train_rel_fact_instruction.csv") and any(
        marker in value
        for marker in (
            "/data/src/FINDER/",
            "/data/finqa/",
            "/data/finqa_original/",
            "/data/finqa_zero_shot/",
            "/data/finqa_many_shot/",
            "/data/finqa_dynamic_shot/",
        )
    ):
        return "finqa_train_formal"
    return None

inferred_source_mode = infer_source_mode_from_path(source)
sys.exit(0 if source_mode == expected or inferred_source_mode == expected else 1)
PYCACHE
}

selection_cache_matches_dataset_scope() {
  local candidate="$1"
  local dataset_id="$2"
  if is_dev_dataset_id "${dataset_id}"; then
    return 0
  fi
  case "${candidate}" in
    *_finqa_dev/*|*/finqa_dev/*) return 1 ;;
  esac
  if [[ ! -f "${candidate}" ]]; then
    return 1
  fi
  CACHE_JSON="${candidate}" python3 - <<'PYSCOPE'
import json
import os
import sys
from pathlib import Path
try:
    payload = json.loads(Path(os.environ["CACHE_JSON"]).read_text(encoding="utf-8"))
except Exception:
    sys.exit(0)
source = str(payload.get("source") or payload.get("metadata", {}).get("source") or "").replace("\\", "/")
sys.exit(1 if "_finqa_dev/" in source or "/finqa_dev/" in source else 0)
PYSCOPE
}

discover_latest_selection_cache() {
  local selection_engine="$1"
  local expected_source_mode="${2:-}"
  local candidate
  local latest=""
  shopt -s nullglob
  for candidate in \
    "${REPO_ROOT}"/Experiment/*/in_context_selection/"${selection_engine}"/selection_cache.json \
    "${REPO_ROOT}"/Experiment/*/in_context_selection/shared/selection_cache.json \
    "${REPO_ROOT}"/Experiment/*/in_context_selection/selection_cache.json; do
    if selection_cache_matches_source_mode "${candidate}" "${expected_source_mode}"; then
      if [[ -z "${latest}" || "${candidate}" -nt "${latest}" ]]; then
        latest="${candidate}"
      fi
    fi
  done
  shopt -u nullglob
  if [[ -n "${latest}" ]]; then
    printf "%s\n" "${latest}"
    return 0
  fi
  return 1
}

selection_cache_for_case() {
  local engine="$1"
  local retriever_id="$2"
  local dataset_id="$3"
  local selection_engine
  selection_engine="$(normalize_selection_engine_alias "${EXPERIMENT7_SELECTION_ENGINE:-gpt5_5}")"
  local expected_source_mode=""
  expected_source_mode="$(formal_csv_source_mode_for "${retriever_id}" 2>/dev/null || true)"
  if [[ -z "${expected_source_mode}" ]] && is_dev_dataset_id "${dataset_id}"; then
    expected_source_mode="matched_retriever_artifact"
  fi
  local scoped
  scoped="$(scoped_env_value EXAMPLE_SELECTION_CACHE_JSON "${selection_engine}" "${retriever_id}" "${dataset_id}")"
  if [[ -z "${scoped}" && "${selection_engine}" != "${engine}" ]]; then
    scoped="$(scoped_env_value EXAMPLE_SELECTION_CACHE_JSON "${engine}" "${retriever_id}" "${dataset_id}")"
  fi
  if [[ -n "${scoped}" ]]; then
    if [[ ! -f "${scoped}" ]] || { selection_cache_matches_source_mode "${scoped}" "${expected_source_mode}" && selection_cache_matches_dataset_scope "${scoped}" "${dataset_id}"; }; then
      printf "%s\n" "${scoped}"
      return 0
    fi
  fi
  if [[ -n "${EXAMPLE_SELECTION_CACHE_ROOT}" ]]; then
    local cache_root
    local case_name
    local candidate
    cache_root="${EXAMPLE_SELECTION_CACHE_ROOT%/}"
    case_name="$(case_dir_name "${retriever_id}" "${dataset_id}")"
    for candidate in \
      "${cache_root}/${selection_engine}/${case_name}/selection_cache.json" \
      "${cache_root}/${selection_engine}/${retriever_id}/${dataset_id}/selection_cache.json" \
      "${cache_root}/${selection_engine}/selection_cache.json" \
      "${cache_root}/shared/selection_cache.json" \
      "${cache_root}/selection_cache.json"; do
      if selection_cache_matches_source_mode "${candidate}" "${expected_source_mode}" && selection_cache_matches_dataset_scope "${candidate}" "${dataset_id}"; then
        printf "%s\n" "${candidate}"
        return 0
      fi
    done
    printf "%s/%s/selection_cache.json\n" "${cache_root}" "${selection_engine}"
    return 0
  fi
  if [[ -n "${EXPERIMENT7_SELECTION_EXPT_ID}" ]]; then
    local selection_root="${REPO_ROOT}/Experiment/${EXPERIMENT7_SELECTION_EXPT_ID}/in_context_selection"
    local candidate
    for candidate in \
      "${selection_root}/${selection_engine}/selection_cache.json" \
      "${selection_root}/shared/selection_cache.json" \
      "${selection_root}/selection_cache.json"; do
      if selection_cache_matches_source_mode "${candidate}" "${expected_source_mode}" && selection_cache_matches_dataset_scope "${candidate}" "${dataset_id}"; then
        printf "%s\n" "${candidate}"
        return 0
      fi
    done
    printf "%s/%s/selection_cache.json\n" "${selection_root}" "${selection_engine}"
    return 0
  fi
  local discovered
  if discovered="$(discover_latest_selection_cache "${selection_engine}" "${expected_source_mode}")"; then
    if selection_cache_matches_dataset_scope "${discovered}" "${dataset_id}"; then
      printf "%s\n" "${discovered}"
      return 0
    fi
  fi
  printf "%s/in_context_selection/%s/selection_cache.json\n" "${EXPT_DIR}" "${selection_engine}"
}

dataset_csv_for() {
  local retriever_id="$1"
  local dataset_id="$2"
  local override_var="INPUT_CSV_$(sanitize_id "${retriever_id}_${dataset_id}")"
  local prompt_mode
  if [[ -n "${!override_var:-}" ]]; then
    printf "%s\n" "${!override_var}"
    return 0
  fi
  case "$(canonical_retriever_id "${retriever_id}")" in
    finqa10_formal_smoke)
      case "${dataset_id}" in
        finqa_10|finqa10|finqa_10_rel_fact)
          first_existing_path "${WORKSPACE_ROOT}/data/testing/finqa_10_rel_fact_instruction.csv"
          return 0
          ;;
        *)
          printf "\n"
          return 1
          ;;
      esac
      ;;
    finqa_train_formal)
      case "${dataset_id}" in
        finqa_train|finqa_train_original) printf "%s/data/finqa_original/finqa_train_rel_fact_instruction.csv\n" "${WORKSPACE_ROOT}" ;;
        finqa_train_zero_shot|finqa_train_zero-shot) printf "%s/data/finqa_zero_shot/finqa_train_rel_fact_instruction.csv\n" "${WORKSPACE_ROOT}" ;;
        finqa_train_many_shot|finqa_train_many-shot) printf "%s/data/finqa_many_shot/finqa_train_rel_fact_instruction.csv\n" "${WORKSPACE_ROOT}" ;;
        finqa_train_dynamic_shot|finqa_train_dynamic-shot) printf "%s/data/finqa_dynamic_shot/finqa_train_rel_fact_instruction.csv\n" "${WORKSPACE_ROOT}" ;;
        *)
          printf "\n"
          return 1
          ;;
      esac
      return 0
      ;;
  esac
  prompt_mode="$(prompt_mode_for_source "${retriever_id}")" || return 2
  case "${dataset_id}" in
    finqa_test|finqa_test_original|finqa_test_zero_shot|finqa_test_many_shot|finqa_test_dynamic_shot)
      printf "%s/finqa_test_rel_fact_instruction.csv\n" "$(prompt_data_dir "${prompt_mode}")"
      ;;
    finqa_dev|finqa_dev_original|finqa_dev_zero_shot|finqa_dev_many_shot|finqa_dev_dynamic_shot)
      printf "%s/finqa_dev_rel_fact_instruction.csv\n" "$(prompt_data_dir "${prompt_mode}")"
      ;;
    *)
      printf "\n"
      return 1
      ;;
  esac
}

retriever_adapter_for_source() {
  local retriever_id="$1"
  local override_var="RETRIEVER_ADAPTER_$(sanitize_id "${retriever_id}")"
  local candidate
  if [[ -n "${!override_var:-}" ]]; then
    printf "%s\n" "${!override_var}"
    return 0
  fi
  for candidate in $(source_experiment_candidates "${retriever_id}"); do
    if [[ -f "${REPO_ROOT}/Experiment/${candidate}/retriever/model/adapter_config.json" ]]; then
      printf "%s\n" "${REPO_ROOT}/Experiment/${candidate}/retriever/model"
      return 0
    fi
    if [[ -f "${REPO_ROOT}/Experiment/${candidate}/retriever_0.3/model/adapter_config.json" ]]; then
      printf "%s\n" "${REPO_ROOT}/Experiment/${candidate}/retriever_0.3/model"
      return 0
    fi
    if [[ -f "${REPO_ROOT}/Experiment/${candidate}/retriever_/model/adapter_config.json" ]]; then
      printf "%s\n" "${REPO_ROOT}/Experiment/${candidate}/retriever_/model"
      return 0
    fi
    if [[ "$(canonical_retriever_id "${retriever_id}")" != finqa_mistral_* && -f "${REPO_ROOT}/Experiment/${candidate}/retriever0.2/model/adapter_config.json" ]]; then
      printf "%s\n" "${REPO_ROOT}/Experiment/${candidate}/retriever0.2/model"
      return 0
    fi
  done
  return 1
}

write_source_status() {
  local path="$1"
  local retriever_id="$2"
  local dataset_id="$3"
  local status="$4"
  local blocked_reason="$5"
  local input_csv="$6"
  local prompt_mode="$7"
  local adapter_dir="$8"
  local matched_json="$9"
  SOURCE_STATUS_PATH="${path}" \
  SOURCE_TIME="$(utc_now)" \
  SOURCE_RETRIEVER_ID="${retriever_id}" \
  SOURCE_DATASET_ID="${dataset_id}" \
  SOURCE_STATUS="${status}" \
  SOURCE_BLOCKED_REASON="${blocked_reason}" \
  SOURCE_INPUT_CSV="${input_csv}" \
  SOURCE_PROMPT_MODE="${prompt_mode}" \
  SOURCE_ADAPTER_DIR="${adapter_dir}" \
  SOURCE_MATCHED_JSON="${matched_json}" \
  SOURCE_RUN_RETRIEVER_INFER="${RUN_RETRIEVER_INFER}" \
  SOURCE_FORCE_REBUILD_RETRIEVER="${FORCE_REBUILD_RETRIEVER}" \
  SOURCE_MAX_INFER_SAMPLES="${RETRIEVER_MAX_INFER_SAMPLES}" \
  SOURCE_INFER_CUDA_DEVICES="${RETRIEVER_INFER_CUDA_DEVICES}" \
  SOURCE_INFER_BATCH_SIZE="${EXPERIMENT7_RETRIEVER_INFER_BATCH_SIZE}" \
  SOURCE_MATCH_EMBED_BATCH_SIZE="${EXPERIMENT7_MATCH_EMBED_BATCH_SIZE}" \
  conda run --no-capture-output -n "${CONDA_ENV}" python -B - <<'PYSOURCE'
import json
import os
from pathlib import Path

payload = {
    "time": os.environ["SOURCE_TIME"],
    "experiment": "7",
    "stage": "retriever_conditioning",
    "retriever_id": os.environ["SOURCE_RETRIEVER_ID"],
    "dataset": os.environ["SOURCE_DATASET_ID"],
    "status": os.environ["SOURCE_STATUS"],
    "blocked_reason": os.environ["SOURCE_BLOCKED_REASON"],
    "input_csv": os.environ["SOURCE_INPUT_CSV"],
    "prompt_mode": os.environ["SOURCE_PROMPT_MODE"],
    "adapter_dir": os.environ["SOURCE_ADAPTER_DIR"],
    "matched_json": os.environ["SOURCE_MATCHED_JSON"],
    "run_retriever_infer": os.environ["SOURCE_RUN_RETRIEVER_INFER"],
    "force_rebuild_retriever": os.environ["SOURCE_FORCE_REBUILD_RETRIEVER"],
    "max_infer_samples": os.environ["SOURCE_MAX_INFER_SAMPLES"],
    "infer_cuda_devices": os.environ["SOURCE_INFER_CUDA_DEVICES"],
    "infer_batch_size": int(os.environ["SOURCE_INFER_BATCH_SIZE"]),
    "match_embed_batch_size": int(os.environ["SOURCE_MATCH_EMBED_BATCH_SIZE"]),
    "shell_id": os.environ.get("EXPERIMENT_SHELL_ID") or os.environ.get("TMUX_PANE") or "pid_unknown",
}
path = Path(os.environ["SOURCE_STATUS_PATH"])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PYSOURCE
}

ensure_retriever_matched() {
  local retriever_id="$1"
  local dataset_id="$2"
  local source_dir="$3"
  local source_status_json="$4"
  local prompt_mode=""
  local input_csv=""
  local adapter_dir=""
  local retriever_model=""
  local prediction_txt="${source_dir}/predictions.txt"
  local data_json="${source_dir}/input_data.json"
  local matched_json="${source_dir}/best_matched_with_retrieved_facts_and_questions.json"
  local legacy_json=""

  ENSURED_MATCHED_JSON=""
  ENSURED_INPUT_CSV=""
  ENSURED_FORMAL_CSV_SOURCE=""
  mkdir -p "${source_dir}"
  if is_formal_csv_source "${retriever_id}"; then
    prompt_mode="formal-csv"
    input_csv="$(dataset_csv_for "${retriever_id}" "${dataset_id}")" || input_csv=""
    if [[ -z "${input_csv}" || ! -f "${input_csv}" ]]; then
      write_source_status "${source_status_json}" "${retriever_id}" "${dataset_id}" "blocked_missing_formal_csv" "missing formal CSV source" "${input_csv}" "${prompt_mode}" "" ""
      return 1
    fi
    ENSURED_INPUT_CSV="${input_csv}"
    ENSURED_FORMAL_CSV_SOURCE="$(formal_csv_source_mode_for "${retriever_id}")"
    write_source_status "${source_status_json}" "${retriever_id}" "${dataset_id}" "formal_csv_ready" "" "${input_csv}" "${prompt_mode}" "" ""
    return 0
  fi
  legacy_json="$(matched_artifact_for "${retriever_id}" "${dataset_id}" "${matched_json}")"
  if [[ "${FORCE_REBUILD_RETRIEVER}" != "1" && -n "${legacy_json}" && -f "${legacy_json}" ]]; then
    ENSURED_MATCHED_JSON="${legacy_json}"
    if [[ "${legacy_json}" != "${matched_json}" || ! -f "${source_status_json}" ]]; then
      write_source_status "${source_status_json}" "${retriever_id}" "${dataset_id}" "cache_reused" "" "" "" "" "${legacy_json}"
    fi
    return 0
  fi
  if [[ "${retriever_id}" == "apollo" ]]; then
    write_source_status "${source_status_json}" "${retriever_id}" "${dataset_id}" "blocked_missing_apollo_artifact" "APOLLO is artifact-only in this workflow" "" "" "" "${matched_json}"
    return 1
  fi
  if [[ "${RUN_RETRIEVER_INFER}" != "1" ]]; then
    write_source_status "${source_status_json}" "${retriever_id}" "${dataset_id}" "blocked_retriever_inference_required" "matched cache is missing and RUN_RETRIEVER_INFER=${RUN_RETRIEVER_INFER}" "" "" "" "${matched_json}"
    return 1
  fi
  prompt_mode="$(prompt_mode_for_source "${retriever_id}")" || prompt_mode=""
  input_csv="$(dataset_csv_for "${retriever_id}" "${dataset_id}")" || input_csv=""
  adapter_dir="$(retriever_adapter_for_source "${retriever_id}")" || adapter_dir=""
  retriever_model="$(retriever_model_for_source "${retriever_id}")" || retriever_model=""
  if [[ -z "${prompt_mode}" || -z "${input_csv}" || ! -f "${input_csv}" || -z "${adapter_dir}" || ! -f "${adapter_dir}/adapter_config.json" || -z "${retriever_model}" ]]; then
    write_source_status "${source_status_json}" "${retriever_id}" "${dataset_id}" "blocked_missing_retriever_input_or_adapter" "missing input CSV or source retriever adapter" "${input_csv}" "${prompt_mode}" "${adapter_dir}" "${matched_json}"
    return 1
  fi
  case "$(canonical_retriever_id "${retriever_id}")" in
    finqa_flan_*)
      if ! run_logged "${source_dir}/retriever_inference.log" env CUDA_VISIBLE_DEVICES="${RETRIEVER_INFER_CUDA_DEVICES}" PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF}" conda run --no-capture-output -n "${CONDA_ENV}" python -B "${REPO_ROOT}/.external/FINDER/Retriever Codes/Flan/lora_flan_large_finqa_rel_fact.py" --mode infer --input-csv "${input_csv}" --eval-csv "${input_csv}" --adapter-dir "${adapter_dir}" --output-txt "${prediction_txt}" --max-infer-samples "${RETRIEVER_MAX_INFER_SAMPLES}" --prompt-mode "${prompt_mode}" --batch-size "${EXPERIMENT7_RETRIEVER_INFER_BATCH_SIZE}" --max-new-tokens "${FLAN_MAX_NEW_TOKENS:-128}" --structured-output "${FLAN_STRUCTURED_OUTPUT:-assembler}"; then
        write_source_status "${source_status_json}" "${retriever_id}" "${dataset_id}" "blocked_retriever_inference_failed" "oom_or_infer_failed" "${input_csv}" "${prompt_mode}" "${adapter_dir}" "${matched_json}"
        return 1
      fi
      ;;
    finqa_mistral_*)
      local mistral_retriever_ld_library_path
      mistral_retriever_ld_library_path="$(prepend_library_path "$(conda_cuda13_library_dirs)" "${LD_LIBRARY_PATH:-}")"
      if ! run_logged "${source_dir}/retriever_inference.log" env CUDA_VISIBLE_DEVICES="${RETRIEVER_INFER_CUDA_DEVICES}" PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF}" LD_LIBRARY_PATH="${mistral_retriever_ld_library_path}" conda run --no-capture-output -n "${CONDA_ENV}" python -B "${REPO_ROOT}/.external/FINDER/Retriever Codes/Mistral/mistral_inference.py" --input-csv "${input_csv}" --adapter-dir "${adapter_dir}" --output-txt "${prediction_txt}" --max-infer-samples "${RETRIEVER_MAX_INFER_SAMPLES}" --prompt-mode "${prompt_mode}" --batch-size "${EXPERIMENT7_MISTRAL_INFER_BATCH_SIZE:-${MISTRAL_INFER_BATCH_SIZE:-2}}" --max-new-tokens "${MISTRAL_MAX_NEW_TOKENS:-256}" --structured-output "${MISTRAL_STRUCTURED_OUTPUT:-assembler}"; then
        write_source_status "${source_status_json}" "${retriever_id}" "${dataset_id}" "blocked_retriever_inference_failed" "oom_or_infer_failed" "${input_csv}" "${prompt_mode}" "${adapter_dir}" "${matched_json}"
        return 1
      fi
      ;;
    finqa_t5gemma2_*)
      if ! run_logged "${source_dir}/retriever_inference.log" env CUDA_VISIBLE_DEVICES="${RETRIEVER_INFER_CUDA_DEVICES}" PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF}" conda run --no-capture-output -n "${CONDA_ENV}" python -B "${REPO_ROOT}/.external/FINDER/Retriever Codes/t5gemma-2/t5gemma-2_train.py" --mode infer --train-csv "${input_csv}" --eval-csv "${input_csv}" --input-csv "${input_csv}" --output-dir "${source_dir}/t5gemma_infer" --adapter-dir "${adapter_dir}" --output-txt "${prediction_txt}" --max-infer-samples "${RETRIEVER_MAX_INFER_SAMPLES}" --prompt-mode "${prompt_mode}" --batch-size "${EXPERIMENT7_T5GEMMA_INFER_BATCH_SIZE}" --max-new-tokens "${T5GEMMA_MAX_NEW_TOKENS:-128}" --structured-output "${T5GEMMA_STRUCTURED_OUTPUT:-assembler}"; then
        write_source_status "${source_status_json}" "${retriever_id}" "${dataset_id}" "blocked_retriever_inference_failed" "oom_or_infer_failed" "${input_csv}" "${prompt_mode}" "${adapter_dir}" "${matched_json}"
        return 1
      fi
      ;;
    *)
      write_source_status "${source_status_json}" "${retriever_id}" "${dataset_id}" "blocked_unsupported_retriever" "unsupported retriever source" "${input_csv}" "${prompt_mode}" "${adapter_dir}" "${matched_json}"
      return 1
      ;;
  esac
  if [[ ! -s "${prediction_txt}" ]]; then
    write_source_status "${source_status_json}" "${retriever_id}" "${dataset_id}" "blocked_retriever_inference_failed" "oom_or_infer_failed" "${input_csv}" "${prompt_mode}" "${adapter_dir}" "${matched_json}"
    return 1
  fi
  ensure_prediction_alias "${prediction_txt}" "${dataset_id}" "${source_dir}" || true
  if ! run_logged "${source_dir}/build_input_data_json.log" conda run --no-capture-output -n "${CONDA_ENV}" python -B "${SCRIPT_DIR}/build_retriever_few_data_json.py" --input-csv "${input_csv}" --output-json "${data_json}"; then
    write_source_status "${source_status_json}" "${retriever_id}" "${dataset_id}" "blocked_retriever_match_failed" "match_failed_or_missing_artifact" "${input_csv}" "${prompt_mode}" "${adapter_dir}" "${matched_json}"
    return 1
  fi
  local -a match_args=(
    conda run --no-capture-output -n "${CONDA_ENV}"
    python -B "${REPO_ROOT}/result_organization.py" match
    --dataset finqa
    --retriever-model "${retriever_model}"
    --prompt-mode "${prompt_mode}"
    --input-txt "${prediction_txt}"
    --data-json "${data_json}"
    --relfact-csv "${input_csv}"
    --embedding-batch-size "${EXPERIMENT7_MATCH_EMBED_BATCH_SIZE}"
    --output-json "${matched_json}"
    --execute
    --require-valid-schema
  )
  if [[ "${RETRIEVER_MAX_INFER_SAMPLES}" != "-1" ]] && is_dev_dataset_id "${dataset_id}"; then
    match_args+=(--allow-partial)
  fi
  if ! run_logged "${source_dir}/match.log" "${match_args[@]}"; then
    write_source_status "${source_status_json}" "${retriever_id}" "${dataset_id}" "blocked_retriever_match_failed" "match_failed_or_missing_artifact" "${input_csv}" "${prompt_mode}" "${adapter_dir}" "${matched_json}"
    return 1
  fi
  if [[ ! -s "${matched_json}" ]]; then
    write_source_status "${source_status_json}" "${retriever_id}" "${dataset_id}" "blocked_retriever_match_failed" "match_failed_or_missing_artifact" "${input_csv}" "${prompt_mode}" "${adapter_dir}" "${matched_json}"
    return 1
  fi
  ENSURED_MATCHED_JSON="${matched_json}"
  write_source_status "${source_status_json}" "${retriever_id}" "${dataset_id}" "retriever_matched_generated" "" "${input_csv}" "${prompt_mode}" "${adapter_dir}" "${matched_json}"
  return 0
}

engine_available_from_status() {
  conda run --no-capture-output -n "${CONDA_ENV}" \
    python -B - "$1" <<'PY'
import json
import sys

payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
print("1" if payload.get("engine", {}).get("available") else "0")
PY
}

engine_field_from_status() {
  conda run --no-capture-output -n "${CONDA_ENV}" \
    python -B - "$1" "$2" <<'PY'
import json
import sys

payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
engine = payload.get("engine", {})
field = sys.argv[2]
value = engine.get(field)
if value is None and field == "actual_model":
    value = engine.get("model")
print("" if value is None else str(value))
PY
}

llm_service_label_from_backend() {
  case "$1" in
    vllm) printf "vllm\n" ;;
    chatmock) printf "chatmock\n" ;;
    *) printf "\n" ;;
  esac
}

missing_credentials_json_from_status() {
  conda run --no-capture-output -n "${CONDA_ENV}" \
    python -B - "$1" <<'PY'
import json
import sys

payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
missing = payload.get("engine", {}).get("missing_credentials") or []
print(json.dumps(missing, ensure_ascii=False))
PY
}

missing_credentials_text_from_status() {
  conda run --no-capture-output -n "${CONDA_ENV}" \
    python -B - "$1" <<'PY'
import json
import sys

payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
missing = payload.get("engine", {}).get("missing_credentials") or []
print(", ".join(str(item) for item in missing))
PY
}

write_skipped_execute_log() {
  local path="$1"
  local reason="$2"
  mkdir -p "$(dirname "${path}")"
  {
    printf "started_at=%s\n" "$(utc_now)"
    printf "command=skipped_generator_execute\n"
    printf "run_execute=%q\n" "${RUN_EXECUTE}"
    printf "blocked_reason=%q\n" "${reason}"
    printf "finished_at=%s\n" "$(utc_now)"
    printf "exit_code=0\n"
  } | tee "${path}"
}

shell_quote() {
  printf "%q" "$1"
}

qwen3_6_enable_thinking_for_case() {
  local engine="$1"
  local dataset_id="$2"
  if [[ "${engine}" == "qwen3_6" ]] && is_dev_dataset_id "${dataset_id}"; then
    printf "%s\n" "${QWEN3_6_DEV_ENABLE_THINKING}"
    return 0
  fi
  printf "%s\n" "${QWEN3_6_ENABLE_THINKING}"
}

scoped_env_value() {
  local prefix="$1"
  local engine="$2"
  local retriever_id="$3"
  local dataset_id="$4"
  local var
  for var in \
    "${prefix}_$(sanitize_id "${engine}_${retriever_id}_${dataset_id}")" \
    "${prefix}_$(sanitize_id "${engine}")" \
    "${prefix}"; do
    if [[ -n "${!var:-}" ]]; then
      printf "%s\n" "${!var}"
      return 0
    fi
  done
  printf "\n"
}

build_experiment7_resume_command() {
  local engine="$1"
  local matrix_item="$2"
  local retriever_id="${matrix_item%%:*}"
  local dataset_id="${matrix_item#*:}"
  local override_var="MATCHED_JSON_$(sanitize_id "${retriever_id}_${dataset_id}")"
  local resume_qwen3_6_enable_thinking
  resume_qwen3_6_enable_thinking="$(qwen3_6_enable_thinking_for_case "${engine}" "${dataset_id}")"
  printf "cd %s && EXPT_ID=%s ENGINES=%s EXPERIMENT7_MATRIX=%s RUN_EXECUTE=auto REGENERATE_EA_ONLY=0 UPDATE_EA_LATEST=%s LIMIT=%s MAX_TOKENS=%s SLEEP_SECONDS=%s SHOW_PROMPT=%s RESUME_OUTPUT=1 RUN_RETRIEVER_INFER=%s FORCE_REBUILD_RETRIEVER=%s ALLOW_FALLBACK_SMOKE_EXECUTE=%s STRICT_INPUTS=%s QWEN3_6_ENABLE_THINKING=%s MISTRAL4_REASONING_EFFORT=%s MISTRAL4_REASONING_TEMPERATURE=%s MISTRAL4_REASONING_TOP_P=%s LLAMA_CPP_ALLOW_REASONING_EFFORT=%s bash dist/experiment_7_generator_answer.sh" \
    "$(shell_quote "${REPO_ROOT}")" \
    "$(shell_quote "${EXPT_ID}")" \
    "$(shell_quote "${engine}")" \
    "$(shell_quote "${matrix_item}")" \
    "$(shell_quote "${UPDATE_EA_LATEST}")" \
    "$(shell_quote "${LIMIT}")" \
    "$(shell_quote "${MAX_TOKENS}")" \
    "$(shell_quote "${SLEEP_SECONDS}")" \
    "$(shell_quote "${SHOW_PROMPT}")" \
    "$(shell_quote "${RUN_RETRIEVER_INFER}")" \
    "$(shell_quote "${FORCE_REBUILD_RETRIEVER}")" \
    "$(shell_quote "${ALLOW_FALLBACK_SMOKE_EXECUTE}")" \
    "$(shell_quote "${STRICT_INPUTS}")" \
    "$(shell_quote "${resume_qwen3_6_enable_thinking}")" \
    "$(shell_quote "${MISTRAL4_REASONING_EFFORT}")" \
    "$(shell_quote "${MISTRAL4_REASONING_TEMPERATURE}")" \
    "$(shell_quote "${MISTRAL4_REASONING_TOP_P}")" \
    "$(shell_quote "${LLAMA_CPP_ALLOW_REASONING_EFFORT:-}")"
  printf " EXAMPLE_SELECTION_MODE=%s EXAMPLE_SELECTION_SHOT_NUMBER=%s EXAMPLE_SELECTION_REQUIRE_POLICY=%s EXAMPLE_SELECTION_REQUIRE_CACHE=%s FORMAL_FINDER_READY=%s ALLOW_LEGACY_SELECTION_BINDING=%s EXPERIMENT7_REQUIRE_TARGET_SELECTION_CACHE=%s EXPERIMENT7_REQUIRE_BINDING_AUDIT=%s EXPERIMENT7_ALLOW_MATERIALIZED_SELECTION_CACHE=%s EXPERIMENT7_SELECTION_EXPT_ID=%s EXPERIMENT7_SELECTION_ENGINE=%s EXAMPLE_SELECTION_CACHE_ROOT=%s EXAMPLE_SELECTION_CANDIDATE_JSON=%s EXAMPLE_SELECTION_POLICY_OUTPUT=%s" \
    "$(shell_quote "${EXAMPLE_SELECTION_MODE}")" \
    "$(shell_quote "${EXAMPLE_SELECTION_SHOT_NUMBER}")" \
    "$(shell_quote "${EXAMPLE_SELECTION_REQUIRE_POLICY}")" \
    "$(shell_quote "${EXAMPLE_SELECTION_REQUIRE_CACHE}")" \
    "$(shell_quote "${FORMAL_FINDER_READY}")" \
    "$(shell_quote "${ALLOW_LEGACY_SELECTION_BINDING}")" \
    "$(shell_quote "${EXPERIMENT7_REQUIRE_TARGET_SELECTION_CACHE}")" \
    "$(shell_quote "${EXPERIMENT7_REQUIRE_BINDING_AUDIT}")" \
    "$(shell_quote "${EXPERIMENT7_ALLOW_MATERIALIZED_SELECTION_CACHE}")" \
    "$(shell_quote "${EXPERIMENT7_SELECTION_EXPT_ID}")" \
    "$(shell_quote "${EXPERIMENT7_SELECTION_ENGINE}")" \
    "$(shell_quote "${EXAMPLE_SELECTION_CACHE_ROOT}")" \
    "$(shell_quote "${EXAMPLE_SELECTION_CANDIDATE_JSON}")" \
    "$(shell_quote "${EXAMPLE_SELECTION_POLICY_OUTPUT}")"
  if [[ -n "${!override_var:-}" ]]; then
    printf " %s=%s" "${override_var}" "$(shell_quote "${!override_var}")"
  fi
}

read_execute_failure_metadata() {
  local path="$1"
  conda run --no-capture-output -n "${CONDA_ENV}" python -B - "${path}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
error = payload.get("error") or {}
resume = error.get("resume") or {}
notes = resume.get("notes") or []
print(error.get("category") or "")
print(resume.get("command") or "")
print(" | ".join(str(note) for note in notes))
print("1" if resume.get("command") else "0")
PY
}

jsonl_row_count() {
  local path="$1"
  conda run --no-capture-output -n "${CONDA_ENV}" python -B - "${path}" <<'PYROWS'
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    print(0)
else:
    with path.open("r", encoding="utf-8") as handle:
        print(sum(1 for line in handle if line.strip()))
PYROWS
}

route_name_from_status() {
  local path="$1"
  conda run --no-capture-output -n "${CONDA_ENV}" python -B - "${path}" <<'PYROUTE'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    print("")
    raise SystemExit
payload = json.loads(path.read_text(encoding="utf-8"))
engine = payload.get("engine") or {}
print(engine.get("route") or engine.get("backend") or payload.get("backend") or "")
PYROUTE
}

completed_rows_from_status_or_output() {
  local status_json="$1"
  local output_jsonl="$2"
  conda run --no-capture-output -n "${CONDA_ENV}" python -B - "${status_json}" "${output_jsonl}" <<'PYDONE'
import json
import sys
from pathlib import Path

status_path = Path(sys.argv[1])
output_path = Path(sys.argv[2])
completed = None
if status_path.is_file():
    try:
        payload = json.loads(status_path.read_text(encoding="utf-8"))
        error = payload.get("error") or {}
        completed = error.get("completed_rows_before_failure")
        if completed is None:
            result = payload.get("result") or {}
            completed = result.get("total_output_rows") or result.get("rows")
    except Exception:
        completed = None
if completed is None:
    if output_path.is_file():
        with output_path.open("r", encoding="utf-8") as handle:
            completed = sum(1 for line in handle if line.strip())
    else:
        completed = 0
print(int(completed))
PYDONE
}

write_quota_wait_checkpoint() {
  local checkpoint_path="$1"
  local engine="$2"
  local route="$3"
  local output_jsonl="$4"
  local completed_rows="$5"
  local retry_count="$6"
  local retry_after_seconds="$7"
  local status="${8:-quota_waiting}"
  local retry_exit_code="${9:-}"
  QUOTA_CHECKPOINT_PATH="${checkpoint_path}" \
  QUOTA_ENGINE="${engine}" \
  QUOTA_ROUTE="${route}" \
  QUOTA_OUTPUT_JSONL="${output_jsonl}" \
  QUOTA_COMPLETED_ROWS="${completed_rows}" \
  QUOTA_RETRY_COUNT="${retry_count}" \
  QUOTA_RETRY_AFTER_SECONDS="${retry_after_seconds}" \
  QUOTA_STATUS="${status}" \
  QUOTA_RETRY_EXIT_CODE="${retry_exit_code}" \
  conda run --no-capture-output -n "${CONDA_ENV}" python -B - <<'PYQUOTA'
import json
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

failed_at = datetime.now(timezone.utc)
retry_after = failed_at + timedelta(seconds=int(os.environ["QUOTA_RETRY_AFTER_SECONDS"]))
payload = {
    "engine": os.environ["QUOTA_ENGINE"],
    "route": os.environ["QUOTA_ROUTE"],
    "last_output_jsonl": os.environ["QUOTA_OUTPUT_JSONL"],
    "completed_rows": int(os.environ["QUOTA_COMPLETED_ROWS"]),
    "failed_at_utc": failed_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
    "retry_after_utc": retry_after.strftime("%Y-%m-%dT%H:%M:%SZ"),
    "retry_count": int(os.environ["QUOTA_RETRY_COUNT"]),
    "status": os.environ["QUOTA_STATUS"],
}
if os.environ.get("QUOTA_RETRY_EXIT_CODE"):
    payload["retry_exit_code"] = int(os.environ["QUOTA_RETRY_EXIT_CODE"])
path = Path(os.environ["QUOTA_CHECKPOINT_PATH"])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PYQUOTA
}

print_resume_hint() {
  local category="$1"
  local command="$2"
  local direct_command="$3"
  local headline="Generator execution interruption detected."
  local action="Fix the generator runtime, then resume this matrix case with:"
  case "${category}" in
    api_authentication_error)
      headline="Generator authentication interruption detected."
      action="Refresh ChatMock login, replace the API key, or switch GPT route, then resume this matrix case with:"
      ;;
    api_quota_or_rate_limit)
      headline="Generator quota/rate-limit interruption detected."
      action="Refresh quota, lower concurrency, replace the API key, or wait for reset, then resume this matrix case with:"
      ;;
  esac
  cat >&2 <<EOF
${headline}
${action}
  ${command}

Direct single-run resume command:
  ${direct_command}
EOF
}

write_route_status() {
  local path="$1"
  local engine="$2"
  local retriever_id="$3"
  local dataset_id="$4"
  local input_json="$5"
  local output_jsonl="$6"
  local matched_json="$7"
  local validate_status_json="$8"
  local execute_status_json="$9"
  local engine_available="${10}"
  local missing_credentials_json="${11}"
  local execute_attempted="${12}"
  local blocked_reason="${13}"
  local exit_code="${14}"
  local status="${15}"
  local failure_category="${16:-}"
  local resume_command="${17:-}"
  local resume_notes="${18:-}"
  local direct_resume_command="${19:-}"

  ROUTE_STATUS_PATH="${path}" \
  ROUTE_TIME="$(utc_now)" \
  ROUTE_FLOW_SCOPE="${FLOW_SCOPE}" \
  ROUTE_ENGINE="${engine}" \
  ROUTE_RETRIEVER_ID="${retriever_id}" \
  ROUTE_DATASET_ID="${dataset_id}" \
  ROUTE_PROFILE="${PROFILE}" \
  ROUTE_LIMIT="${LIMIT}" \
  ROUTE_MAX_TOKENS="${MAX_TOKENS}" \
  ROUTE_RUN_EXECUTE="${RUN_EXECUTE}" \
  ROUTE_INPUT_JSON="${input_json}" \
  ROUTE_OUTPUT_JSONL="${output_jsonl}" \
  ROUTE_MATCHED_JSON="${matched_json}" \
  ROUTE_VALIDATE_STATUS_JSON="${validate_status_json}" \
  ROUTE_EXECUTE_STATUS_JSON="${execute_status_json}" \
  ROUTE_ENGINE_AVAILABLE="${engine_available}" \
  ROUTE_MISSING_CREDENTIALS_JSON="${missing_credentials_json:-[]}" \
  ROUTE_EXECUTE_ATTEMPTED="${execute_attempted}" \
  ROUTE_BLOCKED_REASON="${blocked_reason}" \
  ROUTE_EXIT_CODE="${exit_code}" \
  ROUTE_STATUS="${status}" \
  ROUTE_FAILURE_CATEGORY="${failure_category}" \
  ROUTE_RESUME_COMMAND="${resume_command}" \
  ROUTE_RESUME_NOTES="${resume_notes}" \
  ROUTE_DIRECT_RESUME_COMMAND="${direct_resume_command}" \
  ROUTE_SHELL_ID="${RUN_SHELL_ID}" \
  ROUTE_COMPLETED_AT="$(utc_now)" \
  ROUTE_VLLM_RUNTIME_PROFILE="${VLLM_RUNTIME_PROFILE:-}" \
  ROUTE_VLLM_TP="${VLLM_TENSOR_PARALLEL_SIZE:-}" \
  ROUTE_VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-}" \
  ROUTE_VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-}" \
  ROUTE_VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-}" \
  ROUTE_VLLM_DTYPE="${VLLM_DTYPE:-}" \
  ROUTE_VLLM_KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-}" \
  ROUTE_VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-}" \
  ROUTE_VLLM_CPU_OFFLOAD_GB="${VLLM_CPU_OFFLOAD_GB:-}" \
  ROUTE_QWEN3_6_ENABLE_THINKING="${CURRENT_QWEN3_6_ENABLE_THINKING:-${QWEN3_6_ENABLE_THINKING:-}}" \
  ROUTE_MISTRAL4_REASONING_EFFORT="${MISTRAL4_REASONING_EFFORT:-}" \
  ROUTE_MISTRAL4_REASONING_TEMPERATURE="${MISTRAL4_REASONING_TEMPERATURE:-}" \
  ROUTE_MISTRAL4_REASONING_TOP_P="${MISTRAL4_REASONING_TOP_P:-}" \
  ROUTE_LLAMA_CPP_ALLOW_REASONING_EFFORT="${LLAMA_CPP_ALLOW_REASONING_EFFORT:-}" \
  conda run --no-capture-output -n "${CONDA_ENV}" python -B - <<'PY'
import json
import os
from pathlib import Path


def nullable_bool(value: str):
    if value in {"1", "true", "True"}:
        return True
    if value in {"0", "false", "False"}:
        return False
    return None


missing_raw = os.environ.get("ROUTE_MISSING_CREDENTIALS_JSON") or "[]"
try:
    missing_credentials = json.loads(missing_raw)
except json.JSONDecodeError:
    missing_credentials = [missing_raw]

engine_contract = {}
local_llama_cpp_runtime = {}
validate_status = os.environ.get("ROUTE_VALIDATE_STATUS_JSON") or ""
if validate_status:
    validate_path = Path(validate_status)
    if validate_path.is_file():
        try:
            validation_payload = json.loads(validate_path.read_text(encoding="utf-8"))
            engine_contract = validation_payload.get("engine") or {}
            local_llama_cpp_runtime = validation_payload.get("local_llama_cpp_runtime") or {}
        except Exception as exc:
            engine_contract = {"status_parse_error": str(exc)}

payload = {
    "time": os.environ["ROUTE_TIME"],
    "experiment": "7",
    "flow_scope": os.environ["ROUTE_FLOW_SCOPE"],
    "engine": os.environ["ROUTE_ENGINE"],
    "retriever_id": os.environ["ROUTE_RETRIEVER_ID"],
    "dataset": os.environ["ROUTE_DATASET_ID"],
    "profile": os.environ["ROUTE_PROFILE"],
    "limit": int(os.environ["ROUTE_LIMIT"]),
    "max_tokens": int(os.environ["ROUTE_MAX_TOKENS"]),
    "run_execute": os.environ["ROUTE_RUN_EXECUTE"],
    "engine_available": nullable_bool(os.environ.get("ROUTE_ENGINE_AVAILABLE", "")),
    "missing_credentials": missing_credentials,
    "execute_attempted": bool(int(os.environ["ROUTE_EXECUTE_ATTEMPTED"])),
    "blocked_reason": os.environ["ROUTE_BLOCKED_REASON"],
    "matched_json": os.environ["ROUTE_MATCHED_JSON"],
    "input_json": os.environ["ROUTE_INPUT_JSON"],
    "output_jsonl": os.environ["ROUTE_OUTPUT_JSONL"],
    "validate_status_json": os.environ["ROUTE_VALIDATE_STATUS_JSON"],
    "execute_status_json": os.environ["ROUTE_EXECUTE_STATUS_JSON"],
    "failure_category": os.environ["ROUTE_FAILURE_CATEGORY"] or None,
    "resume_command": os.environ["ROUTE_RESUME_COMMAND"] or None,
    "direct_resume_command": os.environ["ROUTE_DIRECT_RESUME_COMMAND"] or None,
    "resume_notes": os.environ["ROUTE_RESUME_NOTES"] or None,
    "exit_code": int(os.environ["ROUTE_EXIT_CODE"]),
    "status": os.environ["ROUTE_STATUS"],
    "shell_id": os.environ["ROUTE_SHELL_ID"],
    "completed_at": os.environ["ROUTE_COMPLETED_AT"],
    "generator_runtime": {
        "requested_engine": engine_contract.get("requested_engine"),
        "normalized_engine": engine_contract.get("engine"),
        "backend": engine_contract.get("backend"),
        "formal_model": engine_contract.get("formal_model"),
        "actual_model": engine_contract.get("actual_model") or engine_contract.get("model"),
        "runtime_profile": engine_contract.get("runtime_profile"),
        "available": engine_contract.get("available"),
        "missing_credentials": engine_contract.get("missing_credentials"),
        "status_parse_error": engine_contract.get("status_parse_error"),
    },
    "local_vllm_runtime": {
        "profile": os.environ["ROUTE_VLLM_RUNTIME_PROFILE"] or None,
        "tensor_parallel_size": os.environ["ROUTE_VLLM_TP"] or None,
        "max_model_len": os.environ["ROUTE_VLLM_MAX_MODEL_LEN"] or None,
        "max_num_seqs": os.environ["ROUTE_VLLM_MAX_NUM_SEQS"] or None,
        "gpu_memory_utilization": os.environ["ROUTE_VLLM_GPU_MEMORY_UTILIZATION"] or None,
        "dtype": os.environ["ROUTE_VLLM_DTYPE"] or None,
        "kv_cache_dtype": os.environ["ROUTE_VLLM_KV_CACHE_DTYPE"] or None,
        "quantization": os.environ["ROUTE_VLLM_QUANTIZATION"] or None,
        "cpu_offload_gb": os.environ["ROUTE_VLLM_CPU_OFFLOAD_GB"] or None,
    },
    "local_llama_cpp_runtime": local_llama_cpp_runtime or None,
    "qwen3_6_thinking_policy": {
        "enable_thinking": nullable_bool(os.environ.get("ROUTE_QWEN3_6_ENABLE_THINKING", "")),
    } if os.environ["ROUTE_ENGINE"] == "qwen3_6" else None,
    "mistral4_reasoning_policy": {
        "reasoning_effort": os.environ["ROUTE_MISTRAL4_REASONING_EFFORT"] or None,
        "temperature": os.environ["ROUTE_MISTRAL4_REASONING_TEMPERATURE"] or None,
        "top_p": os.environ["ROUTE_MISTRAL4_REASONING_TOP_P"] or None,
        "llama_cpp_allow_reasoning_effort": nullable_bool(os.environ.get("ROUTE_LLAMA_CPP_ALLOW_REASONING_EFFORT", "")),
    } if os.environ["ROUTE_ENGINE"] == "mistral4" else None,
}
path = Path(os.environ["ROUTE_STATUS_PATH"])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY
}

write_aggregate_status() {
  AGG_STATUS_PATH="${STATUS_JSON}" \
  AGG_TIME="$(utc_now)" \
  AGG_FLOW_SCOPE="${FLOW_SCOPE}" \
  AGG_ENGINES="${ENGINES}" \
  AGG_MATRIX="${EXPERIMENT7_MATRIX}" \
  AGG_RUN_EXECUTE="${RUN_EXECUTE}" \
  AGG_ALLOW_FALLBACK_SMOKE_EXECUTE="${ALLOW_FALLBACK_SMOKE_EXECUTE}" \
  AGG_RUN_RETRIEVER_INFER="${RUN_RETRIEVER_INFER}" \
  AGG_FORCE_REBUILD_RETRIEVER="${FORCE_REBUILD_RETRIEVER}" \
  AGG_RETRIEVER_MAX_INFER_SAMPLES="${RETRIEVER_MAX_INFER_SAMPLES}" \
  AGG_RETRIEVER_INFER_CUDA_DEVICES="${RETRIEVER_INFER_CUDA_DEVICES}" \
  AGG_RETRIEVER_INFER_BATCH_SIZE="${EXPERIMENT7_RETRIEVER_INFER_BATCH_SIZE}" \
  AGG_MATCH_EMBED_BATCH_SIZE="${EXPERIMENT7_MATCH_EMBED_BATCH_SIZE}" \
  AGG_EXIT_CODE="${overall_rc}" \
  conda run --no-capture-output -n "${CONDA_ENV}" \
    python -B - "${route_status_files[@]}" <<'PY'
import json
import os
import sys
from pathlib import Path

exit_code = int(os.environ["AGG_EXIT_CODE"])
resume_commands = []
status_counts = {}
failure_categories = {}
for status_path in sys.argv[1:]:
    path = Path(status_path)
    if not path.is_file():
        continue
    try:
        status_payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    status_value = status_payload.get("status") or "unknown"
    status_counts[status_value] = status_counts.get(status_value, 0) + 1
    category = status_payload.get("failure_category")
    if category:
        failure_categories[category] = failure_categories.get(category, 0) + 1
    command = status_payload.get("resume_command")
    if command:
        resume_commands.append(
            {
                "engine": status_payload.get("engine"),
                "retriever_id": status_payload.get("retriever_id"),
                "dataset": status_payload.get("dataset"),
                "failure_category": status_payload.get("failure_category"),
                "resume_command": command,
                "direct_resume_command": status_payload.get("direct_resume_command"),
                "resume_notes": status_payload.get("resume_notes"),
                "status_json": status_path,
            }
        )
payload = {
    "time": os.environ["AGG_TIME"],
    "experiment": "7",
    "flow_scope": os.environ["AGG_FLOW_SCOPE"],
    "engines": os.environ["AGG_ENGINES"].split(),
    "matrix": os.environ["AGG_MATRIX"].split(),
    "run_execute": os.environ["AGG_RUN_EXECUTE"],
    "allow_fallback_smoke_execute": os.environ["AGG_ALLOW_FALLBACK_SMOKE_EXECUTE"],
    "run_retriever_infer": os.environ["AGG_RUN_RETRIEVER_INFER"],
    "force_rebuild_retriever": os.environ["AGG_FORCE_REBUILD_RETRIEVER"],
    "retriever_max_infer_samples": os.environ["AGG_RETRIEVER_MAX_INFER_SAMPLES"],
    "retriever_infer_cuda_devices": os.environ["AGG_RETRIEVER_INFER_CUDA_DEVICES"],
    "retriever_infer_batch_size": int(os.environ["AGG_RETRIEVER_INFER_BATCH_SIZE"]),
    "match_embed_batch_size": int(os.environ["AGG_MATCH_EMBED_BATCH_SIZE"]),
    "status_files": sys.argv[1:],
    "status_counts": status_counts,
    "failure_categories": failure_categories,
    "resume_commands": resume_commands,
    "exit_code": exit_code,
    "status": "completed_or_validation_blocked" if exit_code == 0 else "blocked_or_failed",
}
path = Path(os.environ["AGG_STATUS_PATH"])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PY
}

ea_id_for_engine() {
  case "$(normalize_engine_alias "$1")" in
    qwen3_6) printf "qwen\n" ;;
    mistral4) printf "mistral4\n" ;;
    llama3_3) printf "llama\n" ;;
    llama4) printf "llama4\n" ;;
    gpt5_5) printf "gpt5.5\n" ;;
    gpt5_3_codexS) printf "gptCodexS\n" ;;
    gpt4_1) printf "gpt4.1\n" ;;
    *) printf "%s\n" "$(normalize_engine_alias "$1")" ;;
  esac
}

write_ea_summaries() {
  EA_REPO_ROOT="${REPO_ROOT}" \
  EA_EXPT_ID="${EXPT_ID}" \
  EA_TIME="$(utc_now)" \
  EA_ID_SUFFIX="${EXPERIMENT7_EA_ID_SUFFIX}" \
  EA_UPDATE_LATEST="${UPDATE_EA_LATEST}" \
  conda run --no-capture-output -n "${CONDA_ENV}" python -B - "${route_status_files[@]}" <<'PYEA'
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

repo_root = Path(os.environ["EA_REPO_ROOT"])
expt_id = os.environ["EA_EXPT_ID"]
now = os.environ["EA_TIME"]

def ea_id_for_engine(engine: str) -> str:
    base = {
        "qwen3_6": "qwen",
        "mistral4": "mistral4",
        "llama3_3": "llama",
        "llama4": "llama4",
        "gpt5_5": "gpt5.5",
        "gpt5_3_codexS": "gptCodexS",
        "gpt4_1": "gpt4.1",
    }.get(engine, engine)
    suffix = os.environ.get("EA_ID_SUFFIX", "").strip().strip("_")
    return f"{base}_{suffix}" if suffix else base


def input_row_count(raw_path: object) -> int | None:
    if not raw_path:
        return None
    path = Path(str(raw_path))
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        for key in ("items", "examples", "rows", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return len(value)
    return None


groups: dict[str, list[dict[str, object]]] = defaultdict(list)
for raw_status_path in sys.argv[1:]:
    status_path = Path(raw_status_path)
    if not status_path.is_file():
        continue
    route = json.loads(status_path.read_text(encoding="utf-8"))
    engine = str(route.get("engine") or "unknown")
    execute_status_path = Path(str(route.get("execute_status_json") or ""))
    execute_payload = {}
    if execute_status_path.is_file():
        try:
            execute_payload = json.loads(execute_status_path.read_text(encoding="utf-8"))
        except Exception as exc:
            execute_payload = {"status_parse_error": str(exc)}
    result = execute_payload.get("result") or {}
    generator_runtime = route.get("generator_runtime") or {}
    groups[ea_id_for_engine(engine)].append({
        "time": now,
        "experiment": "7",
        "source_expt_id": expt_id,
        "engine": engine,
        "ea_id": ea_id_for_engine(engine),
        "retriever_id": route.get("retriever_id"),
        "dataset": route.get("dataset"),
        "route_status": route.get("status"),
        "generator_backend": generator_runtime.get("backend"),
        "generator_actual_model": generator_runtime.get("actual_model"),
        "generator_formal_model": generator_runtime.get("formal_model"),
        "execute_attempted": route.get("execute_attempted"),
        "execution_accuracy": result.get("execution_accuracy"),
        "raw_execution_accuracy": result.get("raw_execution_accuracy"),
        "rows": result.get("rows"),
        "input_rows": input_row_count(route.get("input_json")),
        "total_output_rows": result.get("total_output_rows"),
        "generated_nonempty_rows": result.get("generated_nonempty_rows"),
        "generated_nonempty_rate": result.get("generated_nonempty_rate"),
        "executed_non_null_rows": result.get("executed_non_null_rows"),
        "executed_non_null_rate": result.get("executed_non_null_rate"),
        "correct": result.get("correct"),
        "wrong": result.get("wrong"),
        "failure_category": result.get("failure_category"),
        "output_jsonl": result.get("output_jsonl") or route.get("output_jsonl"),
        "route_status_json": str(status_path),
        "execute_status_json": str(execute_status_path) if str(execute_status_path) else None,
    })

for ea_id, items in groups.items():
    out_dir = repo_root / "Experiment" / "EA" / ea_id
    out_dir.mkdir(parents=True, exist_ok=True)
    completed = [item for item in items if item.get("execution_accuracy") is not None]
    formal_ready_items = [
        item for item in completed
        if item.get("input_rows") is not None
        and item.get("rows") == item.get("input_rows")
    ]
    formal_finder_ready = os.environ.get("FORMAL_FINDER_READY") in {"1", "true", "True"}
    formal_full_row_count = (
        bool(items)
        and len(formal_ready_items) == len(items)
        and all(item.get("input_rows") is not None for item in items)
    )
    diagnostic_backends = {"chatmock", "chatmock_openai_compatible"}
    diagnostic_route_backends = sorted({
        str(item.get("generator_backend"))
        for item in items
        if item.get("generator_backend") in diagnostic_backends
    })
    formal_route_ready = not diagnostic_route_backends
    formal_claim_ready = formal_finder_ready and formal_full_row_count and formal_route_ready
    if diagnostic_route_backends:
        formal_claim_blocker = (
            "ChatMock output is diagnostic-only and must not be claimed as formal GPT-series EA."
        )
    elif not formal_claim_ready:
        formal_claim_blocker = (
            "formal EA requires FORMAL_FINDER_READY=1 and completed rows equal input rows for every case."
        )
    else:
        formal_claim_blocker = None
    payload = {
        "time": now,
        "experiment": "7",
        "source_expt_id": expt_id,
        "ea_id": ea_id,
        "items": items,
        "completed_cases": len(completed),
        "total_cases": len(items),
        "formal_finder_ready": formal_finder_ready,
        "formal_full_row_count": formal_full_row_count,
        "formal_route_ready": formal_route_ready,
        "diagnostic_route_backends": diagnostic_route_backends,
        "formal_claim_ready": formal_claim_ready,
        "formal_claim_blocker": formal_claim_blocker,
        "formal_metric_policy": {
            "execution_accuracy": "strict FINDER EA; finqa_equal(..., include_percentage=False)",
            "diagnostic_percentage_equivalent_accuracy": "excluded from formal EA claim",
        },
        "mean_execution_accuracy_unweighted": (
            sum(float(item["execution_accuracy"]) for item in completed) / len(completed)
            if completed else None
        ),
    }
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    (out_dir / f"{expt_id}_execution_accuracy.json").write_text(text, encoding="utf-8")
    if os.environ.get("EA_UPDATE_LATEST") == "1":
        (out_dir / "latest_execution_accuracy.json").write_text(text, encoding="utf-8")
    with (out_dir / f"{expt_id}_items.jsonl").open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
PYEA
}


write_score_report() {
  write_timeline_event "score_report" "start" "" "" "" "cases=${#route_status_files[@]}"
  set +e
  SCORE_REPORT_JSON="${EXPT_DIR}/generator/score_report.json" \
  SCORE_REPORT_MD="${EXPT_DIR}/generator/score_report.md" \
  SCORE_TIME="$(utc_now)" \
  SCORE_EXPT_ID="${EXPT_ID}" \
  conda run --no-capture-output -n "${CONDA_ENV}" python -B - "${route_status_files[@]}" <<'PYSCORE'
import json
import os
import sys
from collections import Counter
from pathlib import Path


def input_row_count(raw_path: object) -> int | None:
    if not raw_path:
        return None
    path = Path(str(raw_path))
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict):
        for key in ("items", "examples", "rows", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                return len(value)
    return None


def normalized_failure_category(route: dict, execute_payload: dict) -> str | None:
    result = execute_payload.get("result") or {}
    if result.get("execution_accuracy") is not None:
        return None
    route_status = str(route.get("status") or "")
    route_failure = str(route.get("failure_category") or "")
    execute_failure = str(execute_payload.get("failure_category") or "")
    result_failure = str(result.get("failure_category") or "")
    missing_credentials = route.get("missing_credentials") or []
    if route_status == "blocked_example_selection_cache" or route_failure == "selection_cache_missing":
        return "selection_cache_blocked"
    if route_status == "blocked_retriever_conditioning":
        return "retriever_conditioning_blocked"
    if route_status == "validation_completed_execute_disabled":
        return None
    if missing_credentials or route_status == "blocked_runtime_credentials" or route_failure == "credential_blocked":
        return "credential_blocked"
    if result_failure == "execute_extraction_failed":
        return "execute_extraction_failed"
    if route_failure == "api_quota_or_rate_limit" or execute_failure == "api_quota_or_rate_limit":
        return "api_quota_or_rate_limit"
    if route_status in {"validation_completed_runtime_blocked", "blocked_resource_guard", "validation_completed_fallback_smoke_blocked"}:
        return "runtime_blocked"
    if execute_failure in {"endpoint_unreachable", "model_not_ready", "loading_timeout", "runtime_blocked"}:
        return "runtime_blocked"
    return "execute_failed"


items = []
for raw_status_path in sys.argv[1:]:
    status_path = Path(raw_status_path)
    if not status_path.is_file():
        continue
    route = json.loads(status_path.read_text(encoding="utf-8"))
    execute_path = Path(str(route.get("execute_status_json") or ""))
    execute_payload = {}
    if execute_path.is_file():
        try:
            execute_payload = json.loads(execute_path.read_text(encoding="utf-8"))
        except Exception as exc:
            execute_payload = {"status_parse_error": str(exc)}
    result = execute_payload.get("result") or {}
    diagnostic_pct_ea = result.get("percentage_equivalent_accuracy")
    item = {
        "engine": route.get("engine"),
        "retriever_id": route.get("retriever_id"),
        "dataset": route.get("dataset"),
        "execution_accuracy": result.get("execution_accuracy"),
        "raw_execution_accuracy": result.get("raw_execution_accuracy"),
        "percentage_equivalent_accuracy": diagnostic_pct_ea,
        "percentage_equivalent_correct": result.get("percentage_equivalent_correct"),
        "diagnostic_percentage_equivalent_accuracy": diagnostic_pct_ea,
        "diagnostic_percentage_equivalent_correct": result.get("percentage_equivalent_correct"),
        "percentage_equivalent_metric_scope": result.get("percentage_equivalent_metric_scope"),
        "route_status": route.get("status"),
        "rows": result.get("rows"),
        "input_rows": input_row_count(route.get("input_json")),
        "generated_nonempty_rows": result.get("generated_nonempty_rows"),
        "generated_nonempty_rate": result.get("generated_nonempty_rate"),
        "executed_non_null_rows": result.get("executed_non_null_rows"),
        "executed_non_null_rate": result.get("executed_non_null_rate"),
        "output_jsonl": result.get("output_jsonl") or route.get("output_jsonl"),
        "score_status": "scored" if result.get("execution_accuracy") is not None else ("validation_only" if str(route.get("status") or "") == "validation_completed_execute_disabled" else "blocked"),
        "failure_category": normalized_failure_category(route, execute_payload),
        "route_status_json": str(status_path),
        "execute_status_json": str(execute_path) if str(execute_path) else None,
    }
    items.append(item)

score_counts = Counter(item["score_status"] for item in items)
failure_counts = Counter(item["failure_category"] for item in items if item.get("failure_category"))
completed = [item for item in items if item.get("execution_accuracy") is not None]
diagnostic_completed = [
    item for item in completed
    if item.get("diagnostic_percentage_equivalent_accuracy") is not None
]
mean_diagnostic_percentage_equivalent_accuracy = (
    sum(float(item["diagnostic_percentage_equivalent_accuracy"]) for item in diagnostic_completed) / len(diagnostic_completed)
    if diagnostic_completed else None
)
payload = {
    "time": os.environ["SCORE_TIME"],
    "experiment": "7",
    "stage": "generator_score_report",
    "expt_id": os.environ["SCORE_EXPT_ID"],
    "items": items,
    "total_cases": len(items),
    "completed_cases": len(completed),
    "status_counts": dict(score_counts),
    "failure_category_counts": dict(failure_counts),
    "mean_execution_accuracy_unweighted": (
        sum(float(item["execution_accuracy"]) for item in completed) / len(completed)
        if completed else None
    ),
    "mean_percentage_equivalent_accuracy_unweighted": mean_diagnostic_percentage_equivalent_accuracy,
    "mean_diagnostic_percentage_equivalent_accuracy_unweighted": mean_diagnostic_percentage_equivalent_accuracy,
    "diagnostic_metric_policy": {
        "diagnostic_percentage_equivalent_accuracy": "finqa_equal(..., include_percentage=True); not formal FINDER EA",
        "execution_accuracy": "strict FINDER EA; finqa_equal(..., include_percentage=False)",
    },
}
json_path = Path(os.environ["SCORE_REPORT_JSON"])
json_path.parent.mkdir(parents=True, exist_ok=True)
json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
lines = [
    f"# Experiment 7 Score Report\n",
    f"- expt_id: {os.environ['SCORE_EXPT_ID']}",
    f"- total_cases: {payload['total_cases']}",
    f"- completed_cases: {payload['completed_cases']}",
    f"- mean_execution_accuracy_unweighted: {payload['mean_execution_accuracy_unweighted']}",
    f"- mean_diagnostic_percentage_equivalent_accuracy_unweighted: {payload['mean_diagnostic_percentage_equivalent_accuracy_unweighted']}\n",
    "| engine | retriever_id | dataset | execution_accuracy | diagnostic_percentage_equivalent_accuracy | raw_execution_accuracy | executed_non_null | generated_nonempty | route_status | rows | input_rows | failure_category | output_jsonl |",
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
]
for item in items:
    lines.append(
        "| {engine} | {retriever_id} | {dataset} | {execution_accuracy} | {diagnostic_percentage_equivalent_accuracy} | {raw_execution_accuracy} | {executed_non_null_rows}/{rows} ({executed_non_null_rate}) | {generated_nonempty_rows}/{rows} ({generated_nonempty_rate}) | {route_status} | {rows} | {input_rows} | {failure_category} | {output_jsonl} |".format(**item)
    )
Path(os.environ["SCORE_REPORT_MD"]).write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PYSCORE
  score_rc=$?
  set -e
  write_timeline_event "score_report" "finish" "" "" "" "rc=${score_rc}"
  return "${score_rc}"
}

route_status_files=()
overall_rc=0

discover_existing_route_status_files() {
  route_status_files=()
  local status_path engine item retriever_id dataset_id case_name
  if [[ -n "${ENGINES//[[:space:]]/}" ]] && [[ -n "${EXPERIMENT7_MATRIX//[[:space:]]/}" ]]; then
    for engine in ${ENGINES}; do
      engine="$(normalize_engine_alias "${engine}")"
      for item in ${EXPERIMENT7_MATRIX}; do
        retriever_id="${item%%:*}"
        dataset_id="${item#*:}"
        case_name="$(case_dir_name "${retriever_id}" "${dataset_id}")"
        status_path="${EXPT_DIR}/generator/${engine}/${case_name}/execution_status.json"
        if [[ -f "${status_path}" ]]; then
          route_status_files+=("${status_path}")
        fi
      done
    done
    return
  fi
  shopt -s nullglob
  for status_path in "${EXPT_DIR}"/generator/*/*/execution_status.json; do
    route_status_files+=("${status_path}")
  done
  shopt -u nullglob
}

regenerate_existing_ea_statuses() {
  discover_existing_route_status_files
  if [[ "${#route_status_files[@]}" -eq 0 ]]; then
    printf "No existing route status files found under %s/generator.\n" "${EXPT_DIR}" >&2
    return 2
  fi

  write_timeline_event "score_existing" "start" "" "" "" "cases=${#route_status_files[@]}"
  local route_status_json case_dir engine input_json output_jsonl execute_status_json case_profile case_limit case_max_tokens
  local score_rc
  for route_status_json in "${route_status_files[@]}"; do
    mapfile -t route_meta < <(
      ROUTE_STATUS_JSON="${route_status_json}" \
        conda run --no-capture-output -n "${CONDA_ENV}" python -B - <<'PYROUTE'
import json
import os
from pathlib import Path

path = Path(os.environ["ROUTE_STATUS_JSON"])
payload = json.loads(path.read_text(encoding="utf-8"))
fields = [
    payload.get("engine") or "",
    payload.get("input_json") or "",
    payload.get("output_jsonl") or "",
    payload.get("execute_status_json") or str(path.with_name("execute_status.json")),
    str(payload.get("profile") or os.environ.get("PROFILE") or "greedy"),
    str(payload.get("limit") if payload.get("limit") is not None else os.environ.get("LIMIT", "-1")),
    str(payload.get("max_tokens") if payload.get("max_tokens") is not None else os.environ.get("MAX_TOKENS", "128")),
]
for field in fields:
    print(field)
PYROUTE
)
    engine="${route_meta[0]:-}"
    input_json="${route_meta[1]:-}"
    output_jsonl="${route_meta[2]:-}"
    execute_status_json="${route_meta[3]:-}"
    case_profile="${route_meta[4]:-${PROFILE}}"
    case_limit="${route_meta[5]:-${LIMIT}}"
    case_max_tokens="${route_meta[6]:-${MAX_TOKENS}}"
    case_dir="${route_status_json%/*}"

    if [[ -z "${engine}" || -z "${input_json}" || -z "${output_jsonl}" || -z "${execute_status_json}" ]]; then
      printf "Skipping malformed route status: %s\n" "${route_status_json}" >&2
      set_overall_rc 2
      continue
    fi

    score_cmd=(
      conda run --no-capture-output -n "${CONDA_ENV}"
      python -B "${REPO_ROOT}/new_full_finqa_run.py"
      --engine "${engine}"
      --input-json "${input_json}"
      --output-jsonl "${output_jsonl}"
      --status-json "${execute_status_json}"
      --profile "${case_profile}"
      --limit "${case_limit}"
      --max-tokens "${case_max_tokens}"
      --credential-purpose test
      --score-existing-output
    )
    set +e
    run_logged "${case_dir}/score_existing.log" "${score_cmd[@]}"
    score_rc=$?
    set -e
    if [[ "${score_rc}" -ne 0 ]]; then
      set_overall_rc "${score_rc}"
    fi
  done
  write_timeline_event "score_existing" "finish" "" "" "" "rc=${overall_rc}"
}

if [[ "${REGENERATE_EA_ONLY}" == "1" ]]; then
  regenerate_existing_ea_statuses || set_overall_rc "$?"
  write_aggregate_status
  write_ea_summaries || true
  write_score_report || true
  exit "${overall_rc}"
fi

for engine in ${ENGINES}; do
  engine="$(normalize_engine_alias "${engine}")"
  for item in ${EXPERIMENT7_MATRIX}; do
    retriever_id="${item%%:*}"
    dataset_id="${item#*:}"
    case_qwen3_6_enable_thinking="$(qwen3_6_enable_thinking_for_case "${engine}" "${dataset_id}")"
    CURRENT_QWEN3_6_ENABLE_THINKING="${case_qwen3_6_enable_thinking}"
    case_name="$(case_dir_name "${retriever_id}" "${dataset_id}")"
    case_dir="${EXPT_DIR}/generator/${engine}/${case_name}"
    input_json="${case_dir}/generator_input.json"
    output_jsonl="${case_dir}/${engine}_${dataset_id}_generated.jsonl"
    validate_status_json="${case_dir}/validate_status.json"
    execute_status_json="${case_dir}/execute_status.json"
    route_status_json="${case_dir}/execution_status.json"
    case_timeline_jsonl="${case_dir}/timeline.jsonl"
    source_dir="${EXPT_DIR}/retriever_sources/${case_name}"
    source_status_json="${source_dir}/source_status.json"
    matched_json=""

    mkdir -p "${case_dir}" "${source_dir}"
    route_status_files+=("${route_status_json}")

    ENSURED_MATCHED_JSON=""
    if ! ensure_retriever_matched "${retriever_id}" "${dataset_id}" "${source_dir}" "${source_status_json}"; then
      matched_json="${ENSURED_MATCHED_JSON}"
      reason="retriever-conditioned RetFact unavailable; see ${source_status_json}"
      printf "Skipping %s:%s for %s because %s. Override with MATCHED_JSON_%s or INPUT_CSV_%s.
"         "${retriever_id}" "${dataset_id}" "${engine}" "${reason}"         "$(sanitize_id "${retriever_id}_${dataset_id}")" "$(sanitize_id "${retriever_id}_${dataset_id}")" >&2
      write_route_status "${route_status_json}" "${engine}" "${retriever_id}" "${dataset_id}"         "${input_json}" "${output_jsonl}" "${matched_json}" "" "" "" "[]"         0 "${reason}" 0 "blocked_retriever_conditioning"
      if [[ "${STRICT_INPUTS}" == "1" ]]; then
        set_overall_rc 2
      fi
      continue
    fi
    matched_json="${ENSURED_MATCHED_JSON}"
    input_csv="${ENSURED_INPUT_CSV:-}"
    formal_csv_source="${ENSURED_FORMAL_CSV_SOURCE:-}"

    target_prompt_type=""
    prompt_type_train_csv=""
    case_selection_cache=""
    materialized_selection_jsonl=""
    case_allow_legacy_selection_binding="${ALLOW_LEGACY_SELECTION_BINDING}"
    case_require_target_selection_cache="${EXPERIMENT7_REQUIRE_TARGET_SELECTION_CACHE}"
    case_allow_materialized_selection_cache="${EXPERIMENT7_ALLOW_MATERIALIZED_SELECTION_CACHE}"
    if ! is_dev_dataset_id "${dataset_id}"; then
      case_allow_legacy_selection_binding="1"
      case_require_target_selection_cache="0"
      case_allow_materialized_selection_cache="1"
    fi
    example_selection_cmd=(
      env "EXAMPLE_SELECTION_ENGINE=${engine}" "FORMAL_FINDER_READY=${FORMAL_FINDER_READY}" "ALLOW_LEGACY_SELECTION_BINDING=${case_allow_legacy_selection_binding}"
      conda run --no-capture-output -n "${CONDA_ENV}"
      python -B "${REPO_ROOT}/dist/example_selection.py"
    )
    if [[ -n "${formal_csv_source}" ]]; then
      example_selection_cmd+=(--input-csv "${input_csv}" --formal-csv-source "${formal_csv_source}")
    else
      example_selection_cmd+=(--input-json "${matched_json}")
    fi
    example_selection_cmd+=(
      --output-json "${input_json}"
      --limit "${LIMIT}"
      --shot-number "${EXAMPLE_SELECTION_SHOT_NUMBER}"
      --selection-mode "${EXAMPLE_SELECTION_MODE}"
    )

    if [[ "${EXAMPLE_SELECTION_MODE}" == "cache" ]]; then
      target_prompt_type="$(target_prompt_type_for_case "${retriever_id}" "${dataset_id}")" || target_prompt_type=""
      prompt_type_train_csv="$(prompt_type_train_csv_for "${target_prompt_type}")" || prompt_type_train_csv=""
      case_selection_engine="$(normalize_selection_engine_alias "${EXPERIMENT7_SELECTION_ENGINE:-gpt5_5}")"
      case_policy_output="$(scoped_env_value EXAMPLE_SELECTION_POLICY_OUTPUT "${case_selection_engine}" "${retriever_id}" "${dataset_id}")"
      case_promptpg_ckpt="$(scoped_env_value PROMPTPG_CKPT "${case_selection_engine}" "${retriever_id}" "${dataset_id}")"
      if [[ "${case_require_target_selection_cache}" == "1" && -z "${formal_csv_source}" ]]; then
        case_selection_cache="${case_dir}/selection_cache.json"
        if [[ ! -f "${case_selection_cache}" ]]; then
          if [[ -z "${case_policy_output}" && -z "${case_promptpg_ckpt}" ]]; then
            reason="target-specific selection cache missing for ${retriever_id}:${dataset_id}, and no EXAMPLE_SELECTION_POLICY_OUTPUT/PROMPTPG_CKPT was provided for ${case_selection_engine}; refusing legacy shared cache binding"
            selection_command_hint="EXAMPLE_SELECTION_MODE=policy EXAMPLE_SELECTION_REQUIRE_POLICY=1 EXAMPLE_SELECTION_POLICY_OUTPUT=/path/to/policy.json or PROMPTPG_CKPT=/path/to/ckpt ENGINES=${engine} EXPERIMENT7_MATRIX=${retriever_id}:${dataset_id} bash dist/experiment_7_generator_answer.sh"
            write_route_status "${route_status_json}" "${engine}" "${retriever_id}" "${dataset_id}"               "${input_json}" "${output_jsonl}" "${matched_json:-${input_csv}}" "" "" "" "[]"               0 "${reason}" 2 "blocked_example_selection_cache" "selection_cache_missing" "" "${reason}" "${selection_command_hint}"
            set_overall_rc 2
            continue
          fi
          target_selection_cmd=(
            env "EXAMPLE_SELECTION_ENGINE=${case_selection_engine}" "FORMAL_FINDER_READY=1" "ALLOW_LEGACY_SELECTION_BINDING=0"
            conda run --no-capture-output -n "${CONDA_ENV}"
            python -B "${REPO_ROOT}/dist/example_selection.py"
            --input-json "${matched_json}"
            --output-json "${case_dir}/target_selected_examples.json"
            --limit "${LIMIT}"
            --shot-number "${EXAMPLE_SELECTION_SHOT_NUMBER}"
            --selection-mode policy
            --selection-cache-json "${case_selection_cache}"
            --promptpg-generated-policy-output "${case_dir}/promptpg_selected_examples.json"
            --require-policy
            --formal-finder-ready
          )
          if [[ -n "${case_policy_output}" ]]; then
            target_selection_cmd+=(--policy-output "${case_policy_output}")
          fi
          if [[ -n "${case_promptpg_ckpt}" ]]; then
            target_selection_cmd+=(--promptpg-ckpt "${case_promptpg_ckpt}")
          fi
          if [[ -n "${PROMPTPG_TRAIN_FILE:-}" ]]; then
            target_selection_cmd+=(--promptpg-train-file "${PROMPTPG_TRAIN_FILE}")
          fi
          write_timeline_event "target_selection" "start" "${engine}" "${retriever_id}" "${dataset_id}" "target-specific PromptPG selection" "${case_timeline_jsonl}"
          target_selection_rc=0
          set +e
          run_logged "${case_dir}/target_selection_cache.log" "${target_selection_cmd[@]}"
          target_selection_rc=$?
          set -e
          write_timeline_event "target_selection" "finish" "${engine}" "${retriever_id}" "${dataset_id}" "rc=${target_selection_rc}" "${case_timeline_jsonl}"
          if [[ "${target_selection_rc}" -ne 0 || ! -f "${case_selection_cache}" ]]; then
            if [[ "${target_selection_rc}" -eq 0 ]]; then
              target_selection_rc=2
            fi
            reason="target-specific in-context selection failed; see ${case_dir}/target_selection_cache.log"
            write_route_status "${route_status_json}" "${engine}" "${retriever_id}" "${dataset_id}"               "${input_json}" "${output_jsonl}" "${matched_json:-${input_csv}}" "" "" "" "[]"               0 "${reason}" "${target_selection_rc}" "blocked_example_selection_cache" "selection_cache_missing" "" "${reason}" ""
            set_overall_rc "${target_selection_rc}"
            continue
          fi
        fi
      else
        case_selection_cache="$(selection_cache_for_case "${engine}" "${retriever_id}" "${dataset_id}")"
      fi
      if [[ "${EXAMPLE_SELECTION_REQUIRE_CACHE}" == "1" && ! -f "${case_selection_cache}" ]]; then
        selection_dataset_hint="${formal_csv_source:-matched_retriever_artifact}"
        selection_input_hint="${matched_json:-${input_csv:-}}"
        selection_command_hint="Generate target-specific selection cache from ${selection_input_hint}; set EXAMPLE_SELECTION_POLICY_OUTPUT or PROMPTPG_CKPT, then rerun ${retriever_id}:${dataset_id}."
        reason="selection cache missing: ${case_selection_cache}; run target-specific selection stage first: ${selection_command_hint}"
        write_route_status "${route_status_json}" "${engine}" "${retriever_id}" "${dataset_id}"           "${input_json}" "${output_jsonl}" "${matched_json:-${input_csv}}" "" "" "" "[]"           0 "${reason}" 2 "blocked_example_selection_cache" "selection_cache_missing" "" "${reason}" "${selection_command_hint}"
        set_overall_rc 2
        continue
      fi
      example_selection_cmd+=(
        --selection-cache-json "${case_selection_cache}"
        --target-prompt-type "${target_prompt_type}"
        --prompt-type-train-csv "${prompt_type_train_csv}"
      )
      if [[ "${case_allow_materialized_selection_cache}" == "1" ]]; then
        materialized_selection_jsonl="$(materialized_selection_jsonl_for_case "${dataset_id}" "${target_prompt_type}")"
        if [[ -n "${materialized_selection_jsonl}" ]]; then
          example_selection_cmd+=(--materialized-selection-jsonl "${materialized_selection_jsonl}")
        fi
      fi
      if [[ "${EXAMPLE_SELECTION_REQUIRE_CACHE}" == "1" ]]; then
        example_selection_cmd+=(--require-cache)
      fi
      if [[ "${FORMAL_FINDER_READY}" == "1" ]]; then
        example_selection_cmd+=(--formal-finder-ready)
      fi
    else
      example_selection_cmd+=(--selection-cache-json "${case_dir}/selection_cache.json")
      if [[ -n "${EXAMPLE_SELECTION_CANDIDATE_JSON}" ]]; then
        example_selection_cmd+=(--candidate-json "${EXAMPLE_SELECTION_CANDIDATE_JSON}")
      fi
      case_policy_output="$(scoped_env_value EXAMPLE_SELECTION_POLICY_OUTPUT "${engine}" "${retriever_id}" "${dataset_id}")"
      case_promptpg_ckpt="$(scoped_env_value PROMPTPG_CKPT "${engine}" "${retriever_id}" "${dataset_id}")"
      if [[ -n "${case_policy_output}" ]]; then
        example_selection_cmd+=(--policy-output "${case_policy_output}")
      fi
      if [[ -n "${case_promptpg_ckpt}" ]]; then
        example_selection_cmd+=(--promptpg-ckpt "${case_promptpg_ckpt}")
      fi
      if [[ -n "${PROMPTPG_TEST_FILE:-}" ]]; then
        example_selection_cmd+=(--promptpg-test-file "${PROMPTPG_TEST_FILE}")
      fi
      if [[ -n "${PROMPTPG_TRAIN_FILE:-}" ]]; then
        example_selection_cmd+=(--promptpg-train-file "${PROMPTPG_TRAIN_FILE}")
      fi
      if [[ "${EXAMPLE_SELECTION_REQUIRE_POLICY}" == "1" ]]; then
        example_selection_cmd+=(--require-policy)
      fi
    fi
    case_timeline_jsonl="${case_dir}/timeline.jsonl"
    write_timeline_event "prepare_input" "start" "${engine}" "${retriever_id}" "${dataset_id}" "example_selection" "${case_timeline_jsonl}"
    prepare_rc=0
    set +e
    run_logged "${case_dir}/prepare_input.log" "${example_selection_cmd[@]}"
    prepare_rc=$?
    set -e
    write_timeline_event "prepare_input" "finish" "${engine}" "${retriever_id}" "${dataset_id}" "rc=${prepare_rc}" "${case_timeline_jsonl}"
    if [[ "${prepare_rc}" -ne 0 ]]; then
      route_block_status="blocked_example_selection"
      if [[ "${EXAMPLE_SELECTION_MODE}" == "cache" ]]; then
        route_block_status="blocked_example_selection_cache"
        reason="selection cache missing or selected ids could not be backfilled; see ${case_dir}/prepare_input.log"
      else
        reason="example selection failed; see ${case_dir}/prepare_input.log"
      fi
      write_route_status "${route_status_json}" "${engine}" "${retriever_id}" "${dataset_id}" \
        "${input_json}" "${output_jsonl}" "${matched_json:-${input_csv}}" "" "" "" "[]" \
        0 "${reason}" "${prepare_rc}" "${route_block_status}"
      set_overall_rc "${prepare_rc}"
      continue
    fi

    case "${EXPERIMENT7_SAMPLE_STRATEGY}" in
      first|none)
        ;;
      stress_first25)
        stress_limit="${LIMIT}"
        if [[ "${stress_limit}" -lt 0 ]]; then
          stress_limit=25
        fi
        write_timeline_event "stress_sample" "start" "${engine}" "${retriever_id}" "${dataset_id}" "strategy=stress_first25;limit=${stress_limit}" "${case_timeline_jsonl}"
        stress_rc=0
        set +e
        run_logged "${case_dir}/stress_first25.log" conda run --no-capture-output -n "${CONDA_ENV}" python -B "${REPO_ROOT}/dist/experiment_7_stress_first25.py" \
          --input-json "${input_json}" \
          --output-json "${input_json}" \
          --report-json "${case_dir}/stress_first25_report.json" \
          --limit "${stress_limit}"
        stress_rc=$?
        set -e
        write_timeline_event "stress_sample" "finish" "${engine}" "${retriever_id}" "${dataset_id}" "rc=${stress_rc}" "${case_timeline_jsonl}"
        if [[ "${stress_rc}" -ne 0 ]]; then
          reason="stress first-25 sample selection failed; see ${case_dir}/stress_first25.log"
          write_route_status "${route_status_json}" "${engine}" "${retriever_id}" "${dataset_id}" \
            "${input_json}" "${output_jsonl}" "${matched_json:-${input_csv}}" "" "" "" "[]" \
            0 "${reason}" "${stress_rc}" "blocked_stress_sample"
          set_overall_rc "${stress_rc}"
          continue
        fi
        ;;
      *)
        printf "Unsupported EXPERIMENT7_SAMPLE_STRATEGY=%s; use first or stress_first25.
" "${EXPERIMENT7_SAMPLE_STRATEGY}" >&2
        exit 2
        ;;
    esac

    credential_purpose="test"
    case "${RUN_EXECUTE}" in
      auto|1|true|True|yes|Yes)
        credential_purpose="execute"
        ;;
    esac
    validate_cmd=(
      env "QWEN3_6_ENABLE_THINKING=${case_qwen3_6_enable_thinking}"
      conda run --no-capture-output -n "${CONDA_ENV}"
      python -B "${REPO_ROOT}/new_full_finqa_run.py"
      --engine "${engine}"
      --input-json "${input_json}"
      --output-jsonl "${output_jsonl}"
      --status-json "${validate_status_json}"
      --profile "${PROFILE}"
      --limit "${LIMIT}"
      --max-tokens "${MAX_TOKENS}"
      --credential-purpose "${credential_purpose}"
    )
    if [[ "${SHOW_PROMPT}" == "1" ]]; then
      validate_cmd+=(--show-prompt)
    fi
    write_timeline_event "validate" "start" "${engine}" "${retriever_id}" "${dataset_id}" "new_full_finqa_run --credential-purpose ${credential_purpose}" "${case_timeline_jsonl}"
    validate_rc=0
    set +e
    run_logged "${case_dir}/validate.log" "${validate_cmd[@]}"
    validate_rc=$?
    set -e
    write_timeline_event "validate" "finish" "${engine}" "${retriever_id}" "${dataset_id}" "rc=${validate_rc}" "${case_timeline_jsonl}"
    if [[ "${validate_rc}" -ne 0 ]]; then
      reason="generator validation failed; see ${case_dir}/validate.log"
      write_route_status "${route_status_json}" "${engine}" "${retriever_id}" "${dataset_id}" \
        "${input_json}" "${output_jsonl}" "${matched_json:-${input_csv}}" "${validate_status_json}" "" "" "[]" \
        0 "${reason}" "${validate_rc}" "blocked_validation"
      set_overall_rc "${validate_rc}"
      continue
    fi

    engine_available="$(engine_available_from_status "${validate_status_json}")"
    missing_credentials_json="$(missing_credentials_json_from_status "${validate_status_json}")"
    missing_credentials_text="$(missing_credentials_text_from_status "${validate_status_json}")"
    runtime_profile="$(engine_field_from_status "${validate_status_json}" runtime_profile)"
    backend="$(engine_field_from_status "${validate_status_json}" backend)"
    expected_service="$(llm_service_label_from_backend "${backend}")"
    execute_attempted=0
    execute_rc=0
    blocked_reason=""
    route_status=""
    failure_category=""
    resume_command=""
    resume_notes=""
    direct_resume_command=""

    case "${RUN_EXECUTE}" in
      0|false|False|no|No)
        blocked_reason="RUN_EXECUTE=${RUN_EXECUTE}"
        route_status="validation_completed_execute_disabled"
        ;;
      auto)
        if [[ "${engine_available}" == "1" ]]; then
          if [[ "${runtime_profile}" == "fallback_smoke" && "${ALLOW_FALLBACK_SMOKE_EXECUTE}" != "1" ]]; then
            blocked_reason="fallback_smoke runtime profile; set ALLOW_FALLBACK_SMOKE_EXECUTE=1 for smoke execution"
            route_status="validation_completed_fallback_smoke_blocked"
          else
            execute_attempted=1
          fi
        else
          blocked_reason="missing credentials: ${missing_credentials_text:-unknown}"
          route_status="validation_completed_runtime_blocked"
        fi
        ;;
      1|true|True|yes|Yes)
        if [[ "${engine_available}" == "1" ]]; then
          if [[ "${runtime_profile}" == "fallback_smoke" && "${ALLOW_FALLBACK_SMOKE_EXECUTE}" != "1" ]]; then
            blocked_reason="fallback_smoke runtime profile; set ALLOW_FALLBACK_SMOKE_EXECUTE=1 for smoke execution"
            route_status="blocked_fallback_smoke_execution"
            set_overall_rc 2
          else
            execute_attempted=1
          fi
        else
          blocked_reason="missing credentials: ${missing_credentials_text:-unknown}"
          route_status="blocked_runtime_credentials"
          set_overall_rc 2
        fi
        ;;
      *)
        printf "Unsupported RUN_EXECUTE=%s; use auto, 0, or 1.\n" "${RUN_EXECUTE}" >&2
        exit 2
        ;;
    esac

    if [[ "${execute_attempted}" == "1" && -n "${expected_service}" ]]; then
      guard_rc=0
      set +e
      GENERATOR_RESOURCE_GUARD_ALLOWED_SERVICES="${expected_service}" \
        generator_resource_guard_before_llm "experiment7_execute" "${engine}" "${EXPT_DIR}/resource_guard"
      guard_rc=$?
      set -e
      if [[ "${guard_rc}" -ne 0 ]]; then
        execute_attempted=0
        execute_rc="${guard_rc}"
        blocked_reason="resource guard paused or failed before ${engine}; see ${EXPT_DIR}/resource_guard"
        route_status="blocked_resource_guard"
        set_overall_rc "${guard_rc}"
      fi
    fi

    if [[ "${execute_attempted}" == "1" ]]; then
      execute_cmd=(
        env "QWEN3_6_ENABLE_THINKING=${case_qwen3_6_enable_thinking}"
        conda run --no-capture-output -n "${CONDA_ENV}"
        python -B "${REPO_ROOT}/new_full_finqa_run.py"
        --engine "${engine}"
        --input-json "${input_json}"
        --output-jsonl "${output_jsonl}"
        --status-json "${execute_status_json}"
        --profile "${PROFILE}"
        --limit "${LIMIT}"
        --max-tokens "${MAX_TOKENS}"
        --sleep-seconds "${SLEEP_SECONDS}"
        --credential-purpose execute
        --execute
      )
      if [[ "${RESUME_OUTPUT}" == "1" ]]; then
        execute_cmd+=(--resume-output)
      fi
      write_timeline_event "inference" "start" "${engine}" "${retriever_id}" "${dataset_id}" "limit=${LIMIT};max_tokens=${MAX_TOKENS}" "${case_timeline_jsonl}"
      retry_rc=0
      set +e
      run_logged "${case_dir}/execute.log" "${execute_cmd[@]}"
      execute_rc=$?
      set -e
      write_timeline_event "inference" "finish" "${engine}" "${retriever_id}" "${dataset_id}" "rc=${execute_rc}" "${case_timeline_jsonl}"
      if [[ "${execute_rc}" -eq 0 ]]; then
        route_status="completed"
      else
        route_status="blocked_or_failed"
        mapfile -t failure_meta < <(read_execute_failure_metadata "${execute_status_json}")
        failure_category="${failure_meta[0]:-}"
        direct_resume_command="${failure_meta[1]:-}"
        resume_notes="${failure_meta[2]:-}"
        if [[ "${failure_meta[3]:-0}" == "1" ]]; then
          resume_command="$(build_experiment7_resume_command "${engine}" "${item}")"
          print_resume_hint "${failure_category}" "${resume_command}" "${direct_resume_command}"
        fi
        if [[ "${failure_category}" == "api_quota_or_rate_limit" && "${GPT_MAX_QUOTA_RETRIES}" -gt 0 ]]; then
          quota_checkpoint_json="${EXPT_DIR}/checkpoints/${engine}.quota_wait.json"
          quota_route="$(route_name_from_status "${validate_status_json}")"
          quota_completed_rows="$(completed_rows_from_status_or_output "${execute_status_json}" "${output_jsonl}")"
          write_quota_wait_checkpoint "${quota_checkpoint_json}" "${engine}" "${quota_route}" "${output_jsonl}" "${quota_completed_rows}" 0 "${GPT_RETRY_AFTER_SECONDS}" "quota_waiting"
          printf "[%s] experiment7_quota_wait engine=%s retriever=%s dataset=%s seconds=%s checkpoint=%s\n" \
            "$(utc_now)" "${engine}" "${retriever_id}" "${dataset_id}" "${GPT_RETRY_AFTER_SECONDS}" "${quota_checkpoint_json}" >&2
          sleep "${GPT_RETRY_AFTER_SECONDS}"
          retry_cmd=("${execute_cmd[@]}")
          if [[ " ${retry_cmd[*]} " != *" --resume-output "* ]]; then
            retry_cmd+=(--resume-output)
          fi
          write_timeline_event "inference_retry" "start" "${engine}" "${retriever_id}" "${dataset_id}" "quota_retry_after=${GPT_RETRY_AFTER_SECONDS}" "${case_timeline_jsonl}"
          set +e
          run_logged "${case_dir}/execute.retry1.log" "${retry_cmd[@]}"
          retry_rc=$?
          set -e
          write_timeline_event "inference_retry" "finish" "${engine}" "${retriever_id}" "${dataset_id}" "rc=${retry_rc}" "${case_timeline_jsonl}"
          quota_completed_rows="$(completed_rows_from_status_or_output "${execute_status_json}" "${output_jsonl}")"
          if [[ "${retry_rc}" -eq 0 ]]; then
            execute_rc=0
            failure_category=""
            blocked_reason=""
            route_status="completed"
            write_quota_wait_checkpoint "${quota_checkpoint_json}" "${engine}" "${quota_route}" "${output_jsonl}" "${quota_completed_rows}" 1 "0" "retried_success" "${retry_rc}"
          else
            execute_rc="${retry_rc}"
            mapfile -t failure_meta < <(read_execute_failure_metadata "${execute_status_json}")
            failure_category="${failure_meta[0]:-api_quota_or_rate_limit}"
            direct_resume_command="${failure_meta[1]:-${direct_resume_command}}"
            resume_notes="${failure_meta[2]:-${resume_notes}}"
            blocked_reason="quota/rate limit retry exhausted after ${GPT_MAX_QUOTA_RETRIES} retry"
            route_status="blocked_or_failed"
            write_quota_wait_checkpoint "${quota_checkpoint_json}" "${engine}" "${quota_route}" "${output_jsonl}" "${quota_completed_rows}" 1 "0" "blocked_retry_exhausted" "${retry_rc}"
          fi
        fi
        set_overall_rc "${execute_rc}"
      fi
      if [[ -n "${expected_service}" ]]; then
        GENERATOR_RESOURCE_GUARD_SERVICE_FILTER="${expected_service}" \
          GENERATOR_RESOURCE_GUARD_ALLOWED_SERVICES="" \
          generator_cleanup_services_after_llm "experiment7_execute" "${engine}" "${EXPT_DIR}/resource_guard" || true
      fi
    else
      write_skipped_execute_log "${case_dir}/execute.log" "${blocked_reason}"
    fi

    write_route_status "${route_status_json}" "${engine}" "${retriever_id}" "${dataset_id}" \
      "${input_json}" "${output_jsonl}" "${matched_json}" "${validate_status_json}" \
      "${execute_status_json}" "${engine_available}" "${missing_credentials_json}" \
      "${execute_attempted}" "${blocked_reason}" "${execute_rc}" "${route_status}" \
      "${failure_category}" "${resume_command}" "${resume_notes}" "${direct_resume_command}"
    printf "[%s] experiment7_route_complete engine=%s retriever=%s dataset=%s shell_id=%s status=%s execute_attempted=%s exit_code=%s\n" \
      "$(utc_now)" "${engine}" "${retriever_id}" "${dataset_id}" "${RUN_SHELL_ID}" "${route_status}" "${execute_attempted}" "${execute_rc}"
    if [[ "${execute_attempted}" == "1" && "${execute_rc}" -ne 0 && "${FAIL_FAST_ON_EXECUTE_ERROR}" == "1" ]]; then
      write_aggregate_status
      write_ea_summaries || true
      write_score_report || true
      exit "${overall_rc}"
    fi
  done
done

write_aggregate_status
write_ea_summaries || true
write_score_report || true
exit "${overall_rc}"
