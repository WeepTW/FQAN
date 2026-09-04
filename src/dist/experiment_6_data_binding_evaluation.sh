#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/retriever_experiment_lib.sh"

EXPT_ID="${EXPT_ID:-experiment_6_data_binding_evaluation}"
FLOW_SCOPE="finflier_narrative_data_binding_evaluation"
EXPERIMENT6_BINDING_MATRIX="${EXPERIMENT6_BINDING_MATRIX:-6_flan_z:finqa_flan_z:narrative_zero_shot 6_flan_m:finqa_flan_m:narrative_many_shot 6_flan_d:finqa_flan_d:narrative_dynamic_shot 6_mistral_z:finqa_mistral_z:narrative_zero_shot 6_mistral_m:finqa_mistral_m:narrative_many_shot 6_mistral_d:finqa_mistral_d:narrative_dynamic_shot 6_t5gemma2_z:finqa_t5gemma2_z:narrative_zero_shot 6_t5gemma2_m:finqa_t5gemma2_m:narrative_many_shot 6_t5gemma2_d:finqa_t5gemma2_d:narrative_dynamic_shot 6_untrain_z:untrained_models:narrative_zero_shot 6_untrain_m:untrained_models:narrative_many_shot 6_untrain_d:untrained_models:narrative_dynamic_shot original_no_gpt41:retriever_models_no_gpt41:narrative_original original_with_gpt41:retriever_models_with_gpt41:narrative_original}"
EXPT_DIR="${REPO_ROOT}/Experiment/${EXPT_ID}"
NARRATIVE_GOLD_DIR="${NARRATIVE_GOLD_DIR:-${WORKSPACE_ROOT}/data/financial_narratives/gold}"
NARRATIVE_TESTING_GOLD_JSONL="${NARRATIVE_TESTING_GOLD_JSONL:-${WORKSPACE_ROOT}/data/testing/narratives_gold.jsonl}"
NARRATIVE_PRED_DIR="${NARRATIVE_PRED_DIR:-${EXPT_DIR}/binding_eval_predictions}"
EXPERIMENT6_PREPARE_GOLD_DATA="${EXPERIMENT6_PREPARE_GOLD_DATA:-0}"
EXPERIMENT6_PREPARE_CONTROLLED_DATA="${EXPERIMENT6_PREPARE_CONTROLLED_DATA:-0}"
EXPERIMENT6_PREPARE_REAL_PREDICTIONS="${EXPERIMENT6_PREPARE_REAL_PREDICTIONS:-0}"
EXPERIMENT6_PREPARE_PROMPT_DATA="${EXPERIMENT6_PREPARE_PROMPT_DATA:-0}"
EXPERIMENT6_GENERATE_BINDING_PREDICTIONS="${EXPERIMENT6_GENERATE_BINDING_PREDICTIONS:-0}"
EXPERIMENT6_GENERATION_MODE="${EXPERIMENT6_GENERATION_MODE:-no-adapter}"
EXPERIMENT6_GENERATION_BATCH_SIZE="${EXPERIMENT6_GENERATION_BATCH_SIZE:-1}"
EXPERIMENT6_GENERATION_MAX_TOKENS="${EXPERIMENT6_GENERATION_MAX_TOKENS:-1024}"
EXPERIMENT6_CASE_TIMEOUT_SECONDS="${EXPERIMENT6_CASE_TIMEOUT_SECONDS:-0}"
EXPERIMENT6_ROW_TIMEOUT_SECONDS="${EXPERIMENT6_ROW_TIMEOUT_SECONDS:-120}"
EXPERIMENT6_NUM_RUNS="${EXPERIMENT6_NUM_RUNS:-1}"
EXPERIMENT6_TOP_K="${EXPERIMENT6_TOP_K:-3}"
EXPERIMENT6_RETRY_MAX="${EXPERIMENT6_RETRY_MAX:-0}"
EXPERIMENT6_RETRY_WAIT_SECONDS="${EXPERIMENT6_RETRY_WAIT_SECONDS:-600}"
EXPERIMENT6_BINDING_CONVERSION="${EXPERIMENT6_BINDING_CONVERSION:-1}"
EXPERIMENT6_BINDING_CONVERTER_ENGINE="${EXPERIMENT6_BINDING_CONVERTER_ENGINE:-gpt5_5}"
EXPERIMENT6_BINDING_CONVERTER_MAX_TOKENS="${EXPERIMENT6_BINDING_CONVERTER_MAX_TOKENS:-1024}"
EXPERIMENT6_BINDING_CONVERTER_ROW_TIMEOUT_SECONDS="${EXPERIMENT6_BINDING_CONVERTER_ROW_TIMEOUT_SECONDS:-${EXPERIMENT6_ROW_TIMEOUT_SECONDS}}"
EXPERIMENT6_BINDING_GENERATOR_PARALLELISM="${EXPERIMENT6_BINDING_GENERATOR_PARALLELISM:-1}"
EXPERIMENT6_BINDING_GENERATOR_TOTAL_TIMEOUT_SECONDS="${EXPERIMENT6_BINDING_GENERATOR_TOTAL_TIMEOUT_SECONDS:-0}"
EXPERIMENT6_LOAD_VARIABLES_MD="${EXPERIMENT6_LOAD_VARIABLES_MD:-1}"
EXPERIMENT6_VARIABLES_MD="${EXPERIMENT6_VARIABLES_MD:-${FQAN_ASSET_ROOT}/workspace/variables.md}"
EXPERIMENT6_RESUME_RUNS="${EXPERIMENT6_RESUME_RUNS:-1}"
EXPERIMENT6_DEBUG="${EXPERIMENT6_DEBUG:-0}"
EXPERIMENT6_CUDA_VISIBLE_DEVICES="${EXPERIMENT6_CUDA_VISIBLE_DEVICES:-1}"
EXPERIMENT6_BUILD_LIMIT="${EXPERIMENT6_BUILD_LIMIT:-0}"
EXPERIMENT6_MIN_ROWS="${EXPERIMENT6_MIN_ROWS:-2}"
EXPERIMENT6_REAL_PREDICTION_EXPECTED_ROWS="${EXPERIMENT6_REAL_PREDICTION_EXPECTED_ROWS:-85}"
EXPERIMENT6_SOURCE_XLSX="${EXPERIMENT6_SOURCE_XLSX:-${WORKSPACE_ROOT}/data/src/narratives/narratives1.xlsx}"
EXPERIMENT6_OUTPUT_ROOT="${EXPERIMENT6_OUTPUT_ROOT:-${EXPT_DIR}/binding_eval_source}"
EFFECTIVE_NARRATIVE_GOLD_DIR="${NARRATIVE_GOLD_DIR}"
EFFECTIVE_NARRATIVE_TESTING_GOLD_JSONL="${NARRATIVE_TESTING_GOLD_JSONL}"
if [[ "${EXPERIMENT6_PREPARE_CONTROLLED_DATA}" == "1" ]]; then
  EFFECTIVE_NARRATIVE_GOLD_DIR="${EXPERIMENT6_OUTPUT_ROOT}/case_gold"
  EFFECTIVE_NARRATIVE_TESTING_GOLD_JSONL="${EXPERIMENT6_OUTPUT_ROOT}/testing_gold.jsonl"
fi
STRICT_INPUTS="${STRICT_INPUTS:-1}"
STATUS_JSON="${EXPT_DIR}/binding_eval/execution_status.json"
VOCABULARY_TYPES="${VOCABULARY_TYPES:-subject trend numerical}"

mkdir -p "${EXPT_DIR}/binding_eval"

if [[ "${EXPERIMENT6_PREPARE_REAL_PREDICTIONS}" == "1" ]]; then
  printf "EXPERIMENT6_PREPARE_REAL_PREDICTIONS is disabled for formal Experiment 6. Binding_Result is gold/reference only, not prediction.
" >&2
  printf "Use EXPERIMENT6_GENERATE_BINDING_PREDICTIONS=1 instead.
" >&2
  exit 2
fi

if [[ "${EXPERIMENT6_PREPARE_PROMPT_DATA}" == "1" ]]; then
  conda run --no-capture-output -n "${CONDA_ENV}" \
    python -B "${SCRIPT_DIR}/build_experiment6_prompt_data.py" \
    --source-xlsx "${EXPERIMENT6_SOURCE_XLSX}" \
    --limit "${EXPERIMENT6_BUILD_LIMIT}" \
    --min-rows "${EXPERIMENT6_MIN_ROWS}" \
    --report-json "${EXPT_DIR}/binding_eval_source/prompt_data_report.json"
fi

if [[ "${EXPERIMENT6_PREPARE_GOLD_DATA}" == "1" || "${EXPERIMENT6_PREPARE_CONTROLLED_DATA}" == "1" ]]; then
  build_args=(
    --source-xlsx "${EXPERIMENT6_SOURCE_XLSX}"
    --limit "${EXPERIMENT6_BUILD_LIMIT}"
    --min-rows "${EXPERIMENT6_MIN_ROWS}"
    --output-root "${EXPERIMENT6_OUTPUT_ROOT}"
    --gold-dir "${EFFECTIVE_NARRATIVE_GOLD_DIR}"
    --testing-gold-jsonl "${EFFECTIVE_NARRATIVE_TESTING_GOLD_JSONL}"
    --expt-id "${EXPT_ID}"
    --matrix "${EXPERIMENT6_BINDING_MATRIX}"
  )
  if [[ "${EXPERIMENT6_PREPARE_CONTROLLED_DATA}" == "1" ]]; then
    build_args+=(--write-controlled-predictions --skip-shared-prompt-gold)
  fi
  conda run --no-capture-output -n "${CONDA_ENV}" \
    python -B "${SCRIPT_DIR}/build_narrative_few1_data.py" \
    "${build_args[@]}"
fi

if [[ "${EXPERIMENT6_GENERATE_BINDING_PREDICTIONS}" == "1" ]]; then
  conda run --no-capture-output -n "${CONDA_ENV}" \
    python -B "${SCRIPT_DIR}/run_experiment6_binding_generation.py" \
    --mode "${EXPERIMENT6_GENERATION_MODE}" \
    --matrix "${EXPERIMENT6_BINDING_MATRIX}" \
    --pred-dir "${NARRATIVE_PRED_DIR}" \
    --limit "${EXPERIMENT6_BUILD_LIMIT}" \
    --batch-size "${EXPERIMENT6_GENERATION_BATCH_SIZE}" \
    --max-tokens "${EXPERIMENT6_GENERATION_MAX_TOKENS}" \
    --case-timeout-seconds "${EXPERIMENT6_CASE_TIMEOUT_SECONDS}" \
    --row-timeout-seconds "${EXPERIMENT6_ROW_TIMEOUT_SECONDS}" \
    --num-runs "${EXPERIMENT6_NUM_RUNS}" \
    --top-k "${EXPERIMENT6_TOP_K}" \
    --retry-max "${EXPERIMENT6_RETRY_MAX}" \
    --retry-wait-seconds "${EXPERIMENT6_RETRY_WAIT_SECONDS}" \
    --binding-conversion "${EXPERIMENT6_BINDING_CONVERSION}" \
    --binding-converter-engine "${EXPERIMENT6_BINDING_CONVERTER_ENGINE}" \
    --binding-converter-max-tokens "${EXPERIMENT6_BINDING_CONVERTER_MAX_TOKENS}" \
    --binding-converter-row-timeout-seconds "${EXPERIMENT6_BINDING_CONVERTER_ROW_TIMEOUT_SECONDS}" \
    --binding-generator-parallelism "${EXPERIMENT6_BINDING_GENERATOR_PARALLELISM}" \
    --binding-generator-total-timeout-seconds "${EXPERIMENT6_BINDING_GENERATOR_TOTAL_TIMEOUT_SECONDS}" \
    --load-variables-md "${EXPERIMENT6_LOAD_VARIABLES_MD}" \
    --variables-md "${EXPERIMENT6_VARIABLES_MD}" \
    --resume-runs "${EXPERIMENT6_RESUME_RUNS}" \
    --debug "${EXPERIMENT6_DEBUG}" \
    --cuda-visible-devices "${EXPERIMENT6_CUDA_VISIBLE_DEVICES}" \
    --report-json "${NARRATIVE_PRED_DIR}/generation_report.json"
fi

sanitize_id() {
  printf "%s" "$1" | tr '[:lower:]-.' '[:upper:]__' | tr -cd '[:alnum:]_'
}

case_dir_name() {
  printf "%s\n" "$1" | tr '-' '_'
}

default_gold_jsonl() {
  if [[ "${EXPERIMENT6_PREPARE_CONTROLLED_DATA}" == "1" ]]; then
    case "$1" in
      narrative_original|narrative_zero_shot|narrative_many_shot|narrative_dynamic_shot)
        printf "%s\n" "${EXPERIMENT6_OUTPUT_ROOT}/prompt_type_gold/$1.jsonl"
        return
        ;;
      *)
        printf "%s\n" "${EFFECTIVE_NARRATIVE_TESTING_GOLD_JSONL}"
        return
        ;;
    esac
  fi
  case "$1" in
    narrative_original) printf "%s\n" "${WORKSPACE_ROOT}/data/finqa_original/narratives_gold.jsonl" ;;
    narrative_zero_shot) printf "%s\n" "${WORKSPACE_ROOT}/data/finqa_zero_shot/narratives_gold.jsonl" ;;
    narrative_many_shot) printf "%s\n" "${WORKSPACE_ROOT}/data/finqa_many_shot/narratives_gold.jsonl" ;;
    narrative_dynamic_shot) printf "%s\n" "${WORKSPACE_ROOT}/data/finqa_dynamic_shot/narratives_gold.jsonl" ;;
    *) printf "%s\n" "${EFFECTIVE_NARRATIVE_TESTING_GOLD_JSONL}" ;;
  esac
}

status_files=()
overall_rc=0

set_overall_rc() {
  local rc="$1"
  if [[ "${overall_rc}" -eq 0 && "${rc}" -ne 0 ]]; then
    overall_rc="${rc}"
  fi
}

for item in ${EXPERIMENT6_BINDING_MATRIX}; do
  IFS=":" read -r experiment_id source_id narrative_route <<<"${item}"
  if [[ -z "${experiment_id}" || -z "${source_id}" || -z "${narrative_route}" ]]; then
    printf "Invalid EXPERIMENT6_BINDING_MATRIX item: %s\n" "${item}" >&2
    exit 2
  fi

  case_name="$(case_dir_name "${experiment_id}")"
  case_dir="${EXPT_DIR}/binding_eval/${case_name}"
  status_json="${case_dir}/status.json"
  metrics_json="${case_dir}/metrics.json"
  gold_var="BINDING_GOLD_JSONL_$(sanitize_id "${experiment_id}")"
  pred_var="BINDING_PRED_JSONL_$(sanitize_id "${experiment_id}")"
  explicit_gold_jsonl="${!gold_var:-}"
  if [[ -n "${explicit_gold_jsonl}" ]]; then
    gold_jsonl="${explicit_gold_jsonl}"
  else
    gold_jsonl="$(default_gold_jsonl "${narrative_route}")"
  fi
  pred_jsonl="${!pred_var:-${NARRATIVE_PRED_DIR}/${experiment_id}.jsonl}"
  require_data_args=()
  controlled_prediction_args=()

  mkdir -p "${case_dir}"
  status_files+=("${status_json}")
  if [[ "${STRICT_INPUTS}" == "1" ]]; then
    require_data_args+=(--require-data)
  fi
  if [[ "${EXPERIMENT6_PREPARE_CONTROLLED_DATA}" == "1" ]]; then
    controlled_prediction_args+=(--allow-controlled-predictions)
  fi

  set +e
  conda run --no-capture-output -n "${CONDA_ENV}" \
    python -B "${SCRIPT_DIR}/evaluate_data_binding.py" \
    --experiment-id "${experiment_id}" \
    --source-id "${source_id}" \
    --narrative-route "${narrative_route}" \
    --gold-jsonl "${gold_jsonl}" \
    --pred-jsonl "${pred_jsonl}" \
    --metrics-json "${metrics_json}" \
    --status-json "${status_json}" \
    --vocabulary-types ${VOCABULARY_TYPES} \
    "${require_data_args[@]}" \
    "${controlled_prediction_args[@]}"
  case_rc=$?
  set -e
  set_overall_rc "${case_rc}"
done

AGG_STATUS_PATH="${STATUS_JSON}" \
AGG_TIME="$(utc_now)" \
AGG_FLOW_SCOPE="${FLOW_SCOPE}" \
AGG_MATRIX="${EXPERIMENT6_BINDING_MATRIX}" \
AGG_EXIT_CODE="${overall_rc}" \
AGG_NARRATIVE_TESTING_GOLD_JSONL="${EFFECTIVE_NARRATIVE_TESTING_GOLD_JSONL}" \
conda run --no-capture-output -n "${CONDA_ENV}" python -B - "${status_files[@]}" <<'PYAGG'
import json
import os
import sys
from pathlib import Path

status_counts = {}
failure_categories = {}
items = []
for raw in sys.argv[1:]:
    path = Path(raw)
    if not path.is_file():
        continue
    payload = json.loads(path.read_text(encoding="utf-8"))
    items.append(payload)
    status = payload.get("status") or "unknown"
    status_counts[status] = status_counts.get(status, 0) + 1
    category = payload.get("failure_category")
    if category:
        failure_categories[category] = failure_categories.get(category, 0) + 1

exit_code = int(os.environ["AGG_EXIT_CODE"])
if exit_code != 0:
    aggregate_status = "blocked_or_failed"
elif items and all((item.get("status") == "completed") for item in items):
    aggregate_status = "completed"
elif status_counts.get("runtime_blocked"):
    aggregate_status = "completed_or_runtime_blocked"
else:
    aggregate_status = "completed_or_runtime_blocked"

aggregate = {
    "time": os.environ["AGG_TIME"],
    "experiment": "6",
    "stage": "data_binding_evaluation",
    "flow_scope": os.environ["AGG_FLOW_SCOPE"],
    "matrix": os.environ["AGG_MATRIX"].split(),
    "gold_data_prepared": os.environ.get("EXPERIMENT6_PREPARE_GOLD_DATA") == "1",
    "controlled_smoke": os.environ.get("EXPERIMENT6_PREPARE_CONTROLLED_DATA") == "1",
    "real_predictions_prepared": False,
    "prompt_data_prepared": os.environ.get("EXPERIMENT6_PREPARE_PROMPT_DATA") == "1",
    "model_predictions_generated": os.environ.get("EXPERIMENT6_GENERATE_BINDING_PREDICTIONS") == "1",
    "generation_mode": os.environ.get("EXPERIMENT6_GENERATION_MODE"),
    "generation_report_json": str(Path(os.environ.get("NARRATIVE_PRED_DIR", "")) / "generation_report.json") if os.environ.get("EXPERIMENT6_GENERATE_BINDING_PREDICTIONS") == "1" else None,
    "real_prediction_report_json": None,
    "api_key_execution": False,
    "narrative_gold_assignment": "route_based_shared_gold_per_data_binding_md",
    "narrative_testing_gold_jsonl": os.environ.get("AGG_NARRATIVE_TESTING_GOLD_JSONL"),
    "legacy_case_gold_dir": os.environ.get("NARRATIVE_GOLD_DIR"),
    "narrative_pred_dir": os.environ.get("NARRATIVE_PRED_DIR"),
    "status_files": sys.argv[1:],
    "status_counts": status_counts,
    "failure_categories": failure_categories,
    "items": items,
    "exit_code": exit_code,
    "status": aggregate_status,
}
path = Path(os.environ["AGG_STATUS_PATH"])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(aggregate, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(aggregate, ensure_ascii=False, indent=2))
PYAGG

exit "${overall_rc}"
