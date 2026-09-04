#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/retriever_experiment_lib.sh"

EXPT_ID="${EXPT_ID:-experiment_7_in_context_selection_$(date -u +%Y%m%dT%H%M%SZ)}"
ENGINES="${ENGINES:-gpt5_5}"
SELECTION_DATASET_WAS_SET=0
if [[ -n "${SELECTION_DATASET+x}" ]]; then
  SELECTION_DATASET_WAS_SET=1
fi
SELECTION_INPUT_CSV_WAS_SET=0
if [[ -n "${SELECTION_INPUT_CSV+x}" ]]; then
  SELECTION_INPUT_CSV_WAS_SET=1
fi
FORMAL_CSV_SOURCE_WAS_SET=0
if [[ -n "${EXAMPLE_SELECTION_FORMAL_CSV_SOURCE+x}" ]]; then
  FORMAL_CSV_SOURCE_WAS_SET=1
fi
SELECTION_DATASET="${SELECTION_DATASET:-finqa_train_formal}"
case "${SELECTION_DATASET}" in
  finqa_train_formal|finqa_train)
    DEFAULT_SELECTION_INPUT_CSV="$(first_existing_path "${WORKSPACE_ROOT}/data/src/FINDER/finqa_train_rel_fact_instruction.csv" "${WORKSPACE_ROOT}/data/finqa_original/finqa_train_rel_fact_instruction.csv" "${WORKSPACE_ROOT}/data/finqa/finqa_train_rel_fact_instruction.csv")"
    DEFAULT_FORMAL_CSV_SOURCE="finqa_train_formal"
    ;;
  finqa10_formal_smoke|finqa10|finqa_10)
    DEFAULT_SELECTION_INPUT_CSV="$(first_existing_path "${WORKSPACE_ROOT}/data/testing/finqa_10_rel_fact_instruction.csv")"
    DEFAULT_FORMAL_CSV_SOURCE="finqa10_formal_smoke"
    ;;
  narratives5_formal_smoke|narratives5|narratives1_5_for_all)
    DEFAULT_SELECTION_INPUT_CSV="$(first_existing_path "${WORKSPACE_ROOT}/data/testing/narratives1_5_for_all.csv")"
    DEFAULT_FORMAL_CSV_SOURCE="narratives5_formal_smoke"
    ;;
  *)
    printf "Unsupported SELECTION_DATASET: %s\n" "${SELECTION_DATASET}" >&2
    exit 2
    ;;
esac
SELECTION_INPUT_CSV="${SELECTION_INPUT_CSV:-${DEFAULT_SELECTION_INPUT_CSV}}"
EXAMPLE_SELECTION_FORMAL_CSV_SOURCE="${EXAMPLE_SELECTION_FORMAL_CSV_SOURCE:-${DEFAULT_FORMAL_CSV_SOURCE}}"

infer_formal_csv_source_from_path() {
  local csv_path="$1"
  case "${csv_path}" in
    */data/testing/finqa_10_rel_fact_instruction.csv)
      printf "finqa10_formal_smoke\n"
      ;;
    */data/src/FINDER/finqa_train_rel_fact_instruction.csv|*/data/finqa/finqa_train_rel_fact_instruction.csv|*/data/finqa_original/finqa_train_rel_fact_instruction.csv|*/data/finqa_zero_shot/finqa_train_rel_fact_instruction.csv|*/data/finqa_many_shot/finqa_train_rel_fact_instruction.csv|*/data/finqa_dynamic_shot/finqa_train_rel_fact_instruction.csv)
      printf "finqa_train_formal\n"
      ;;
    *)
      return 1
      ;;
  esac
}

if [[ "${FORMAL_CSV_SOURCE_WAS_SET}" == "0" ]]; then
  inferred_source_mode="$(infer_formal_csv_source_from_path "${SELECTION_INPUT_CSV}" 2>/dev/null || true)"
  if [[ -n "${inferred_source_mode}" && ( "${SELECTION_INPUT_CSV_WAS_SET}" == "1" || "${SELECTION_DATASET_WAS_SET}" == "0" ) ]]; then
    EXAMPLE_SELECTION_FORMAL_CSV_SOURCE="${inferred_source_mode}"
  fi
fi
LIMIT="${LIMIT:--1}"
EXAMPLE_SELECTION_SHOT_NUMBER="${EXAMPLE_SELECTION_SHOT_NUMBER:-4}"
RUN_PROMPTPG="${RUN_PROMPTPG:-0}"
ALLOW_SELECTION_SMOKE_FALLBACK="${ALLOW_SELECTION_SMOKE_FALLBACK:-0}"
PROMPTPG_TRAIN_NUMBER="${PROMPTPG_TRAIN_NUMBER:-1}"
PROMPTPG_CAND_NUMBER="${PROMPTPG_CAND_NUMBER:-4}"
PROMPTPG_LEARN_POLICY_SHOT_NUMBER="${PROMPTPG_LEARN_POLICY_SHOT_NUMBER:-2}"
PROMPTPG_LR="${PROMPTPG_LR:-0.001}"
PROMPTPG_EPOCHS="${PROMPTPG_EPOCHS:-1}"
PROMPTPG_EPOCH_SEEDS="${PROMPTPG_EPOCH_SEEDS:-1}"
PROMPTPG_EMBEDDING_SIZE="${PROMPTPG_EMBEDDING_SIZE:-128}"
PROMPTPG_BATCH_SIZE="${PROMPTPG_BATCH_SIZE:-1}"
PROMPTPG_MAX_TOKENS="${PROMPTPG_MAX_TOKENS:-128}"
PROMPTPG_MODEL_CONFIG="${PROMPTPG_MODEL_CONFIG:-bert-base-uncased}"
PROMPTPG_GPU="${PROMPTPG_GPU:-cpu}"
PROMPTPG_TRAIN_FILE="${PROMPTPG_TRAIN_FILE:-finqa_train promptpg_samples.json}"
EXPT_DIR="${REPO_ROOT}/Experiment/${EXPT_ID}"
SELECTION_ROOT="${EXPT_DIR}/in_context_selection"
STATUS_JSON="${EXPT_DIR}/in_context_selection_status.json"
ITEMS_JSONL="${SELECTION_ROOT}/items.jsonl"
mkdir -p "${SELECTION_ROOT}"
: >"${ITEMS_JSONL}"

overall_rc=0

set_overall_rc() {
  local rc="$1"
  if [[ "${overall_rc}" -eq 0 && "${rc}" -ne 0 ]]; then
    overall_rc="${rc}"
  fi
}

sanitize_id() {
  printf "%s" "$1" | tr '[:lower:]-' '[:upper:]_' | tr -cd '[:alnum:]_'
}

normalize_engine_alias() {
  case "$1" in
    qwen|qwen3_6|qwen3_6_35b_a3b_fp8|qwen3.6-35b-a3b-fp8) printf "qwen3_6\n" ;;
    mistral|mistral4|mistral_small_4|mistral-small-4) printf "mistral4\n" ;;
    llama|llama3_3|llama3.3|llama-3.3) printf "llama3_3\n" ;;
    gpt55|gpt-5.5) printf "gpt5_5\n" ;;
    gpt5_3_codex|gpt5_3_codexS|gpt-5.3-codex-spark) printf "gpt5_3_codexS\n" ;;
    *) printf "%s\n" "$1" ;;
  esac
}

scoped_env_value() {
  local prefix="$1"
  local engine="$2"
  local var
  for var in "${prefix}_$(sanitize_id "${engine}")" "${prefix}"; do
    if [[ -n "${!var:-}" ]]; then
      printf "%s\n" "${!var}"
      return 0
    fi
  done
  printf "\n"
}

write_item() {
  local engine="$1"
  local status="$2"
  local detail="$3"
  local exit_code="${4:-0}"
  local cache_json="${5:-}"
  local policy_json="${6:-}"
  local ckpt_path="${7:-}"
  ITEM_PATH="${ITEMS_JSONL}" ITEM_TIME="$(utc_now)" ITEM_ENGINE="${engine}" \
  ITEM_STATUS="${status}" ITEM_DETAIL="${detail}" ITEM_EXIT_CODE="${exit_code}" \
  ITEM_CACHE_JSON="${cache_json}" ITEM_POLICY_JSON="${policy_json}" ITEM_CKPT_PATH="${ckpt_path}" \
  python3 - <<'PYITEM'
import json
import os
payload = {
    "time": os.environ["ITEM_TIME"],
    "engine": os.environ["ITEM_ENGINE"],
    "status": os.environ["ITEM_STATUS"],
    "detail": os.environ["ITEM_DETAIL"],
    "exit_code": int(os.environ["ITEM_EXIT_CODE"]),
}
for key, env_name in {
    "selection_cache_json": "ITEM_CACHE_JSON",
    "policy_json": "ITEM_POLICY_JSON",
    "ckpt_path": "ITEM_CKPT_PATH",
}.items():
    value = os.environ.get(env_name) or ""
    if value:
        payload[key] = value
with open(os.environ["ITEM_PATH"], "a", encoding="utf-8") as handle:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
print(json.dumps(payload, ensure_ascii=False), flush=True)
PYITEM
}

copy_policy_output() {
  local source_json="$1"
  local target_json="$2"
  if [[ -z "${source_json}" || "${source_json}" == "${target_json}" ]]; then
    return 0
  fi
  SOURCE_JSON="${source_json}" TARGET_JSON="${target_json}" python3 - <<'PYCOPY'
import shutil
import os
from pathlib import Path
source = Path(os.environ["SOURCE_JSON"])
target = Path(os.environ["TARGET_JSON"])
if source.is_file():
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
PYCOPY
}


publish_shared_selection_cache() {
  local engine="$1"
  local cache_json="$2"
  if [[ "${engine}" != "gpt5_5" ]]; then
    return 0
  fi
  if [[ ! -f "${cache_json}" ]]; then
    return 0
  fi
  mkdir -p "${SELECTION_ROOT}/shared"
  cp -f "${cache_json}" "${SELECTION_ROOT}/shared/selection_cache.json"
  cp -f "${cache_json}" "${SELECTION_ROOT}/selection_cache.json"
}

run_promptpg_train() {
  local engine="$1"
  local label="$2"
  local engine_dir="$3"
  conda run --no-capture-output -n "${CONDA_ENV}" python -B "${REPO_ROOT}/.external/FINDER/In_Context_Selection/learn_policy.py" \
    --label "${label}" \
    --ckpt_root "${REPO_ROOT}/checkpoints" \
    --engine "${engine}" \
    --train-file "${PROMPTPG_TRAIN_FILE}" \
    --train_number "${PROMPTPG_TRAIN_NUMBER}" \
    --cand_number "${PROMPTPG_CAND_NUMBER}" \
    --epochs "${PROMPTPG_EPOCHS}" \
    --epoch-seeds "${PROMPTPG_EPOCH_SEEDS}" \
    --batch_size "${PROMPTPG_BATCH_SIZE}" \
    --shot_number "${PROMPTPG_LEARN_POLICY_SHOT_NUMBER}" \
    --lr "${PROMPTPG_LR}" \
    --max_tokens "${PROMPTPG_MAX_TOKENS}" \
    --embedding_size "${PROMPTPG_EMBEDDING_SIZE}" \
    --model_config "${PROMPTPG_MODEL_CONFIG}" \
    --gpu "${PROMPTPG_GPU}" >"${engine_dir}/learn_policy.log" 2>&1
}

run_engine() {
  local requested_engine="$1"
  local engine
  engine="$(normalize_engine_alias "${requested_engine}")"
  local engine_dir="${SELECTION_ROOT}/${engine}"
  local cache_json="${engine_dir}/selection_cache.json"
  local selection_rows_json="${engine_dir}/selection_rows.json"
  local policy_json="${engine_dir}/promptpg_selected_examples.json"
  local label="experiment7_selection_${engine}_${EXPT_ID}"
  local mode="policy"
  local require_policy="1"
  mkdir -p "${engine_dir}"

  local case_policy_output
  local case_promptpg_ckpt
  case_policy_output="$(scoped_env_value EXAMPLE_SELECTION_POLICY_OUTPUT "${engine}")"
  case_promptpg_ckpt="$(scoped_env_value PROMPTPG_CKPT "${engine}")"

  if [[ "${RUN_PROMPTPG}" == "1" && -z "${case_policy_output}" && -z "${case_promptpg_ckpt}" ]]; then
    if ! run_logged "${engine_dir}/learn_policy_wrapper.log" run_promptpg_train "${engine}" "${label}" "${engine_dir}"; then
      write_item "${engine}" "blocked_promptpg_training" "PromptPG learn_policy failed; see ${engine_dir}/learn_policy.log" 1 "" "" "${REPO_ROOT}/checkpoints/${label}/ckpt_final_0.pt"
      set_overall_rc 2
      return 0
    fi
    case_promptpg_ckpt="${REPO_ROOT}/checkpoints/${label}/ckpt_final_0.pt"
  fi

  if [[ -z "${case_policy_output}" && -z "${case_promptpg_ckpt}" ]]; then
    if [[ "${ALLOW_SELECTION_SMOKE_FALLBACK}" == "1" ]]; then
      mode="similarity"
      require_policy="0"
    else
      write_item "${engine}" "blocked_policy_missing" "No policy output or PromptPG checkpoint was supplied; formal selection refuses similarity fallback." 2 "${cache_json}" "" ""
      set_overall_rc 2
      return 0
    fi
  fi

  cmd=(
    env "EXAMPLE_SELECTION_ENGINE=${engine}" "FORMAL_FINDER_READY=1"
    conda run --no-capture-output -n "${CONDA_ENV}"
    python -B "${REPO_ROOT}/dist/example_selection.py"
    --input-csv "${SELECTION_INPUT_CSV}"
    --formal-csv-source "${EXAMPLE_SELECTION_FORMAL_CSV_SOURCE}"
    --output-json "${selection_rows_json}"
    --selection-cache-json "${cache_json}"
    --limit "${LIMIT}"
    --shot-number "${EXAMPLE_SELECTION_SHOT_NUMBER}"
    --selection-mode "${mode}"
  )
  if [[ "${require_policy}" == "1" ]]; then
    cmd+=(--require-policy)
  fi
  if [[ -n "${case_policy_output}" ]]; then
    cmd+=(--policy-output "${case_policy_output}")
  fi
  if [[ -n "${case_promptpg_ckpt}" ]]; then
    cmd+=(--promptpg-ckpt "${case_promptpg_ckpt}" --promptpg-generated-policy-output "${policy_json}")
  fi

  local rc=0
  set +e
  run_logged "${engine_dir}/example_selection.log" "${cmd[@]}"
  rc=$?
  set -e
  if [[ "${rc}" -ne 0 ]]; then
    write_item "${engine}" "blocked_example_selection" "example_selection.py failed; see ${engine_dir}/example_selection.log" "${rc}" "${cache_json}" "${policy_json}" "${case_promptpg_ckpt}"
    set_overall_rc "${rc}"
    return 0
  fi
  copy_policy_output "${case_policy_output}" "${policy_json}"
  publish_shared_selection_cache "${engine}" "${cache_json}"
  write_item "${engine}" "completed" "Selection cache completed without final answer computation." 0 "${cache_json}" "${policy_json}" "${case_promptpg_ckpt}"
}

if [[ ! -f "${SELECTION_INPUT_CSV}" ]]; then
  printf 'experiment_7_in_context_selection.sh: missing shared selection source: %s\n' "${SELECTION_INPUT_CSV}" >&2
  exit 2
fi

for engine in ${ENGINES}; do
  run_engine "${engine}"
done

STATUS_PATH="${STATUS_JSON}" STATUS_TIME="$(utc_now)" STATUS_EXPT_ID="${EXPT_ID}" \
STATUS_INPUT_CSV="${SELECTION_INPUT_CSV}" STATUS_SOURCE_MODE="${EXAMPLE_SELECTION_FORMAL_CSV_SOURCE}" STATUS_SELECTION_DATASET="${SELECTION_DATASET}" STATUS_ENGINES="${ENGINES}" STATUS_EXIT_CODE="${overall_rc}" \
python3 - "${ITEMS_JSONL}" <<'PYSTATUS'
import json
import os
import sys
from pathlib import Path
items = []
items_path = Path(sys.argv[1])
if items_path.is_file():
    with items_path.open(encoding="utf-8") as handle:
        items = [json.loads(line) for line in handle if line.strip()]
status_counts = {}
for item in items:
    status = item.get("status") or "unknown"
    status_counts[status] = status_counts.get(status, 0) + 1
payload = {
    "time": os.environ["STATUS_TIME"],
    "experiment": "7",
    "stage": "in_context_selection",
    "expt_id": os.environ["STATUS_EXPT_ID"],
    "input_csv": os.environ["STATUS_INPUT_CSV"],
    "engines": os.environ["STATUS_ENGINES"].split(),
    "items": items,
    "status_counts": status_counts,
    "exit_code": int(os.environ["STATUS_EXIT_CODE"]),
    "status": "completed_or_blocked" if int(os.environ["STATUS_EXIT_CODE"]) == 0 else "blocked_or_failed",
    "artifact_contract": "Experiment/<EXPT_ID>/in_context_selection/<engine>/selection_cache.json is the formal final-computation handoff. For gpt5_5, shared aliases are also published at in_context_selection/shared/selection_cache.json and in_context_selection/selection_cache.json; no final answer is computed here.",
}
path = Path(os.environ["STATUS_PATH"])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PYSTATUS

exit "${overall_rc}"
