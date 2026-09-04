#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/retriever_experiment_lib.sh"

EXPT_ID="${EXPT_ID:-experiment_7_selection_cache_binding_$(date -u +%Y%m%dT%H%M%SZ)}"
SELECTION_EXPT_ID="${SELECTION_EXPT_ID:-${EXPERIMENT7_SELECTION_EXPT_ID:-}}"
SELECTION_ENGINE="${SELECTION_ENGINE:-${EXPERIMENT7_SELECTION_ENGINE:-gpt5_5}}"
SELECTION_CACHE_JSON="${SELECTION_CACHE_JSON:-${EXAMPLE_SELECTION_CACHE_JSON:-}}"
EXPECTED_SELECTION_SOURCE_MODE="${EXPECTED_SELECTION_SOURCE_MODE:-finqa_train_formal}"
DATASETS="${DATASETS:-finqa_test finqa_dev}"
PROMPT_TYPES="${PROMPT_TYPES:-original zero-shot many-shot dynamic-shot}"
LIMIT="${LIMIT:--1}"
SHOT_NUMBER="${EXAMPLE_SELECTION_SHOT_NUMBER:-4}"
CONDA_ENV="${CONDA_ENV:-fnqa}"
EXPT_DIR="${REPO_ROOT}/Experiment/${EXPT_ID}"
OUTPUT_ROOT="${EXPT_DIR}/selection_cache_binding"
STATUS_JSON="${OUTPUT_ROOT}/execution_status.json"
ITEMS_JSONL="${OUTPUT_ROOT}/items.jsonl"
mkdir -p "${OUTPUT_ROOT}"
: >"${ITEMS_JSONL}"

overall_rc=0

set_overall_rc() {
  local rc="$1"
  if [[ "${overall_rc}" -eq 0 && "${rc}" -ne 0 ]]; then
    overall_rc="${rc}"
  fi
}

write_item() {
  local dataset="$1"
  local prompt_type="$2"
  local status="$3"
  local detail="$4"
  local exit_code="${5:-0}"
  local output_jsonl="${6:-}"
  local report_json="${7:-}"
  ITEM_PATH="${ITEMS_JSONL}" ITEM_TIME="$(utc_now)" ITEM_DATASET="${dataset}" ITEM_PROMPT_TYPE="${prompt_type}" ITEM_STATUS="${status}" ITEM_DETAIL="${detail}" ITEM_EXIT_CODE="${exit_code}" ITEM_OUTPUT_JSONL="${output_jsonl}" ITEM_REPORT_JSON="${report_json}"     python3 - <<'PYITEM'
import json
import os
payload = {
    "time": os.environ["ITEM_TIME"],
    "dataset": os.environ["ITEM_DATASET"],
    "prompt_type": os.environ["ITEM_PROMPT_TYPE"],
    "status": os.environ["ITEM_STATUS"],
    "detail": os.environ["ITEM_DETAIL"],
    "exit_code": int(os.environ["ITEM_EXIT_CODE"]),
    "output_jsonl": os.environ.get("ITEM_OUTPUT_JSONL") or None,
    "report_json": os.environ.get("ITEM_REPORT_JSON") or None,
}
with open(os.environ["ITEM_PATH"], "a", encoding="utf-8") as handle:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
print(json.dumps(payload, ensure_ascii=False), flush=True)
PYITEM
}

prompt_type_dir() {
  case "$1" in
    raw) printf "src/FINDER\n" ;;
    original) printf "finqa_original\n" ;;
    zero-shot) printf "finqa_zero_shot\n" ;;
    many-shot) printf "finqa_many_shot\n" ;;
    dynamic-shot) printf "finqa_dynamic_shot\n" ;;
    *) return 2 ;;
  esac
}

prompt_type_train_csv() {
  case "$1" in
    raw) printf "%s/data/src/FINDER/finqa_train_rel_fact_instruction.csv\n" "${WORKSPACE_ROOT}" ;;
    original) printf "%s/data/finqa_original/finqa_train_rel_fact_instruction.csv\n" "${WORKSPACE_ROOT}" ;;
    zero-shot) printf "%s/data/finqa_zero_shot/finqa_train_rel_fact_instruction.csv\n" "${WORKSPACE_ROOT}" ;;
    many-shot) printf "%s/data/finqa_many_shot/finqa_train_rel_fact_instruction.csv\n" "${WORKSPACE_ROOT}" ;;
    dynamic-shot) printf "%s/data/finqa_dynamic_shot/finqa_train_rel_fact_instruction.csv\n" "${WORKSPACE_ROOT}" ;;
    *) return 2 ;;
  esac
}

normalize_selection_engine_alias() {
  case "$1" in
    gpt55|gpt-5.5)
      printf "gpt5_5\n"
      ;;
    *)
      printf "%s\n" "$1"
      ;;
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

sys.exit(0 if source_mode == expected or infer_source_mode_from_path(source) == expected else 1)
PYCACHE
}

dataset_input_csv() {
  local prompt_dir
  prompt_dir="$(prompt_type_dir "$1")"
  printf "%s/data/%s/%s_rel_fact_instruction.csv\n" "${WORKSPACE_ROOT}" "${prompt_dir}" "$2"
}

selection_cache_path() {
  local normalized_engine
  normalized_engine="$(normalize_selection_engine_alias "${SELECTION_ENGINE}")"
  if [[ -n "${SELECTION_CACHE_JSON}" ]]; then
    if selection_cache_matches_source_mode "${SELECTION_CACHE_JSON}" "${EXPECTED_SELECTION_SOURCE_MODE}"; then
      printf "%s\n" "${SELECTION_CACHE_JSON}"
      return 0
    fi
    return 1
  fi
  local candidate
  local latest=""
  if [[ -n "${SELECTION_EXPT_ID}" ]]; then
    for candidate in       "${REPO_ROOT}/Experiment/${SELECTION_EXPT_ID}/in_context_selection/${normalized_engine}/selection_cache.json"       "${REPO_ROOT}/Experiment/${SELECTION_EXPT_ID}/in_context_selection/shared/selection_cache.json"       "${REPO_ROOT}/Experiment/${SELECTION_EXPT_ID}/in_context_selection/selection_cache.json"; do
      if selection_cache_matches_source_mode "${candidate}" "${EXPECTED_SELECTION_SOURCE_MODE}"; then
        printf "%s\n" "${candidate}"
        return 0
      fi
    done
    return 1
  fi
  shopt -s nullglob
  for candidate in     "${REPO_ROOT}"/Experiment/*/in_context_selection/"${normalized_engine}"/selection_cache.json     "${REPO_ROOT}"/Experiment/*/in_context_selection/shared/selection_cache.json     "${REPO_ROOT}"/Experiment/*/in_context_selection/selection_cache.json; do
    if selection_cache_matches_source_mode "${candidate}" "${EXPECTED_SELECTION_SOURCE_MODE}"; then
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

cache_json="$(selection_cache_path || true)"
if [[ -z "${cache_json}" || ! -f "${cache_json}" ]]; then
  write_item "all" "all" "blocked_selection_cache_missing" "Shared selection cache is unavailable; set SELECTION_EXPT_ID or SELECTION_CACHE_JSON first." 2
  set_overall_rc 2
else
  for dataset in ${DATASETS}; do
    for prompt_type in ${PROMPT_TYPES}; do
      case_dir="${OUTPUT_ROOT}/${dataset}/${prompt_type}"
      output_jsonl="${case_dir}/materialized_selected_examples.jsonl"
      report_json="${case_dir}/materialization_report.json"
      extracted_jsonl="${case_dir}/extracted.jsonl"
      input_csv="$(dataset_input_csv "${prompt_type}" "${dataset}")"
      train_csv="$(prompt_type_train_csv "${prompt_type}")"
      mkdir -p "${case_dir}"
      if [[ ! -f "${input_csv}" ]]; then
        write_item "${dataset}" "${prompt_type}" "blocked_missing_dataset_csv" "Missing target input CSV: ${input_csv}" 2 "${output_jsonl}" "${report_json}"
        set_overall_rc 2
        continue
      fi
      if [[ ! -f "${train_csv}" ]]; then
        write_item "${dataset}" "${prompt_type}" "blocked_missing_train_csv" "Missing prompt-type train CSV: ${train_csv}" 2 "${output_jsonl}" "${report_json}"
        set_overall_rc 2
        continue
      fi
      set +e
      run_logged "${case_dir}/materialize.log"         conda run --no-capture-output -n "${CONDA_ENV}"         python -B "${REPO_ROOT}/dist/binding_extraction.py" materialize-selection-cache         --selection-cache-json "${cache_json}"         --target-input-csv "${input_csv}"         --target-prompt-type "${prompt_type}"         --prompt-type-train-csv "${train_csv}"         --output-jsonl "${output_jsonl}"         --report-json "${report_json}"         --extracted-jsonl "${extracted_jsonl}"         --shot-number "${SHOT_NUMBER}"         --limit "${LIMIT}"
      rc=$?
      set -e
      if [[ "${rc}" -eq 0 ]]; then
        write_item "${dataset}" "${prompt_type}" "completed" "Materialized selected examples and derived extraction artifacts." 0 "${output_jsonl}" "${report_json}"
      else
        write_item "${dataset}" "${prompt_type}" "blocked_or_failed" "materialize-selection-cache failed; see ${case_dir}/materialize.log" "${rc}" "${output_jsonl}" "${report_json}"
        set_overall_rc "${rc}"
      fi
    done
  done
fi

BINDING_STATUS_JSON="${STATUS_JSON}" BINDING_ITEMS_JSONL="${ITEMS_JSONL}" BINDING_TIME="$(utc_now)" BINDING_EXPT_ID="${EXPT_ID}" BINDING_SELECTION_CACHE_JSON="${cache_json:-}" BINDING_EXIT_CODE="${overall_rc}"   python3 - <<'PYAGG'
import json
import os
from collections import Counter
from pathlib import Path

items_path = Path(os.environ["BINDING_ITEMS_JSONL"])
items = []
if items_path.is_file():
    for line in items_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            items.append(json.loads(line))
status_counts = Counter(item.get("status", "unknown") for item in items)
payload = {
    "time": os.environ["BINDING_TIME"],
    "experiment": "7",
    "stage": "selection_cache_binding",
    "expt_id": os.environ["BINDING_EXPT_ID"],
    "selection_cache_json": os.environ.get("BINDING_SELECTION_CACHE_JSON") or None,
    "items": items,
    "status_counts": dict(status_counts),
    "status": "completed" if int(os.environ["BINDING_EXIT_CODE"]) == 0 else "completed_with_blockers",
    "exit_code": int(os.environ["BINDING_EXIT_CODE"]),
}
path = Path(os.environ["BINDING_STATUS_JSON"])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PYAGG

exit "${overall_rc}"
