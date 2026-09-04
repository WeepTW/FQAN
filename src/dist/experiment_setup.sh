#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/retriever_experiment_lib.sh"

SETUP_ID="${SETUP_ID:-experiment_setup_$(date -u +%Y%m%dT%H%M%SZ)}"
EXPT_DIR="${REPO_ROOT}/Experiment/${SETUP_ID}"
SETUP_ITEMS_TSV="${EXPT_DIR}/setup_items.tsv"
SETUP_REPORT_JSON="${EXPT_DIR}/setup_report.json"
SETUP_WORKSPACE_LOG_JSON="${FQAN_LOG_ROOT}/${SETUP_ID}.json"
SETUP_PROMPT_MODES="${SETUP_PROMPT_MODES:-raw original zero-shot many-shot dynamic-shot}"
SETUP_DOWNLOAD_MODE="${SETUP_DOWNLOAD_MODE:-metadata}"
SETUP_LOCAL_FILES_ONLY="${SETUP_LOCAL_FILES_ONLY:-0}"
SETUP_RUN_PROMPT_CONTRACT="${SETUP_RUN_PROMPT_CONTRACT:-1}"
SETUP_CHECK_ARTIFACTS="${SETUP_CHECK_ARTIFACTS:-1}"
SETUP_ARTIFACTS_REQUIRED="${SETUP_ARTIFACTS_REQUIRED:-0}"
SETUP_STRICT="${SETUP_STRICT:-0}"
SETUP_EXPECT_TRAIN_ROWS="${SETUP_EXPECT_TRAIN_ROWS:-6251}"
SETUP_EXPECT_DEV_ROWS="${SETUP_EXPECT_DEV_ROWS:-883}"
SETUP_EXPECT_TEST_ROWS="${SETUP_EXPECT_TEST_ROWS:-1147}"
SETUP_EXPECT_NARRATIVE_ROWS="${SETUP_EXPECT_NARRATIVE_ROWS:-85}"

MODELS_ROOT="${MODELS_ROOT:-${FQAN_MODELS_ROOT}}"
HF_HOME="${HF_HOME:-${MODELS_ROOT}/.cache/huggingface}"
HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"

resolve_cached_snapshot() {
  local model_id="$1"
  local cache_id="models--${model_id//\//--}"
  local cache_root snapshot
  for cache_root in "${HF_HOME}/hub/${cache_id}" "${HF_HOME}/${cache_id}"; do
    [[ -d "${cache_root}/snapshots" ]] || continue
    snapshot="$(find "${cache_root}/snapshots" -mindepth 1 -maxdepth 1 -type d -print -quit)"
    if [[ -n "${snapshot}" ]]; then
      printf '%s\n' "${snapshot}"
      return 0
    fi
  done
  printf '%s\n' "${model_id}"
}
FLAN_BASE_MODEL="${FLAN_BASE_MODEL:-google/flan-t5-large}"
MISTRAL_BASE_MODEL="${MISTRAL_BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
T5GEMMA_BASE_MODEL="${T5GEMMA_BASE_MODEL:-google/t5gemma-2-1b-1b}"
QWEN3_6_MODEL_PATH="${QWEN3_6_MODEL_PATH:-$(resolve_cached_snapshot Qwen/Qwen3.6-35B-A3B-FP8)}"
MISTRAL_SMALL_MODEL_PATH="${MISTRAL_SMALL_MODEL_PATH:-${MODELS_ROOT}/mistral_small_4_119b_2603_gguf/UD-Q4_K_M/Mistral-Small-4-119B-2603-UD-Q4_K_M-00001-of-00003.gguf}"
DEEPSEEK_MODEL_PATH="${DEEPSEEK_MODEL_PATH:-$(resolve_cached_snapshot deepseek-ai/DeepSeek-R1-Distill-Qwen-32B)}"
QWYTHOS_MODEL_PATH="${QWYTHOS_MODEL_PATH:-$(resolve_cached_snapshot empero-ai/Qwythos-9B-Claude-Mythos-5-1M)}"
LLAMA4_MODEL_PATH="${LLAMA4_MODEL_PATH:-$(resolve_cached_snapshot meta-llama/Llama-4-Scout-17B-16E-Instruct)}"

export MODELS_ROOT HF_HOME HF_HUB_CACHE TRANSFORMERS_CACHE
export FLAN_BASE_MODEL MISTRAL_BASE_MODEL T5GEMMA_BASE_MODEL
export QWEN3_6_MODEL_PATH MISTRAL_SMALL_MODEL_PATH
export DEEPSEEK_MODEL_PATH QWYTHOS_MODEL_PATH LLAMA4_MODEL_PATH

mkdir -p "${EXPT_DIR}" "${HF_HOME}" "${FQAN_LOG_ROOT}"
: >"${SETUP_ITEMS_TSV}"

ok_count=0
warn_count=0
blocker_count=0

clean_field() {
  printf "%s" "$1" | tr '\t\r\n' '   '
}

record_item() {
  local status="$1"
  local category="$2"
  local name="$3"
  local path="${4:-}"
  local detail="${5:-}"
  local action="${6:-}"
  case "${status}" in
    ok) ok_count=$((ok_count + 1)) ;;
    warn|skipped) warn_count=$((warn_count + 1)) ;;
    missing|blocked) blocker_count=$((blocker_count + 1)) ;;
    *) warn_count=$((warn_count + 1)) ;;
  esac
  printf "%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(clean_field "${status}")" \
    "$(clean_field "${category}")" \
    "$(clean_field "${name}")" \
    "$(clean_field "${path}")" \
    "$(clean_field "${detail}")" \
    "$(clean_field "${action}")" >>"${SETUP_ITEMS_TSV}"
  printf "[%s] %s/%s %s\n" "${status}" "${category}" "${name}" "${detail}"
}

is_local_ref() {
  case "$1" in
    /*|./*|../*) return 0 ;;
    *) return 1 ;;
  esac
}

check_command_available() {
  local name="$1"
  if command -v "${name}" >/dev/null 2>&1; then
    record_item ok "runtime" "${name}" "$(command -v "${name}")" "command available" ""
  else
    record_item missing "runtime" "${name}" "" "command missing" "Install or activate the expected runtime before experiments."
  fi
}

check_conda_python() {
  if conda run --no-capture-output -n "${CONDA_ENV}" python -B -c 'import sys; print(sys.executable)' >/dev/null 2>&1; then
    record_item ok "runtime" "conda:${CONDA_ENV}:python" "${CONDA_ENV}" "conda Python available" ""
  else
    record_item missing "runtime" "conda:${CONDA_ENV}:python" "${CONDA_ENV}" "conda Python unavailable" "Activate or repair the fnqa environment."
  fi
}

check_file_required() {
  local category="$1"
  local name="$2"
  local path="$3"
  local action="${4:-}"
  if [[ -s "${path}" ]]; then
    record_item ok "${category}" "${name}" "${path}" "file present" ""
  else
    record_item missing "${category}" "${name}" "${path}" "file missing or empty" "${action}"
  fi
}

check_path_required() {
  local category="$1"
  local name="$2"
  local path="$3"
  local action="${4:-}"
  if [[ -e "${path}" ]]; then
    record_item ok "${category}" "${name}" "${path}" "path present" ""
  else
    record_item missing "${category}" "${name}" "${path}" "path missing" "${action}"
  fi
}

check_json_file() {
  local category="$1"
  local name="$2"
  local path="$3"
  if [[ ! -s "${path}" ]]; then
    record_item missing "${category}" "${name}" "${path}" "JSON file missing or empty" ""
    return
  fi
  if conda run --no-capture-output -n "${CONDA_ENV}" python -B -m json.tool "${path}" >/dev/null 2>&1; then
    record_item ok "${category}" "${name}" "${path}" "valid JSON" ""
  else
    record_item blocked "${category}" "${name}" "${path}" "invalid JSON" "Repair JSON before running experiments."
  fi
}

check_csv_rows() {
  local category="$1"
  local name="$2"
  local path="$3"
  local expected_rows="$4"
  local action="${5:-}"
  if [[ ! -s "${path}" ]]; then
    record_item missing "${category}" "${name}" "${path}" "CSV missing or empty" "${action}"
    return
  fi
  local rows
  if ! rows="$(csv_data_row_count "${path}" -1 2>/dev/null)"; then
    record_item blocked "${category}" "${name}" "${path}" "cannot read CSV row count" "Check CSV encoding/header."
    return
  fi
  if [[ "${expected_rows}" != "-1" && "${rows}" != "${expected_rows}" ]]; then
    record_item blocked "${category}" "${name}" "${path}" "row_count=${rows}, expected=${expected_rows}" "Regenerate or verify this dataset split."
  else
    record_item ok "${category}" "${name}" "${path}" "row_count=${rows}" ""
  fi
}

check_prompt_contract() {
  local prompt="$1"
  local train_csv="$2"
  local dev_csv="$3"
  local test_csv="$4"
  local log_file="${EXPT_DIR}/prompt_contract_${prompt//-/_}.log"
  if [[ "${SETUP_RUN_PROMPT_CONTRACT}" != "1" ]]; then
    record_item skipped "prompt_contract" "${prompt}" "" "SETUP_RUN_PROMPT_CONTRACT=${SETUP_RUN_PROMPT_CONTRACT}" ""
    return
  fi
  if conda run --no-capture-output -n "${CONDA_ENV}" \
      python -B "${SCRIPT_DIR}/prompt_mode_contract_smoke.py" \
      --prompt-mode "${prompt}" \
      --train-csv "${train_csv}" \
      --eval-csv "${dev_csv}" \
      --test-csv "${test_csv}" \
      >"${log_file}" 2>&1; then
    record_item ok "prompt_contract" "${prompt}" "${log_file}" "contract smoke passed" ""
  else
    record_item blocked "prompt_contract" "${prompt}" "${log_file}" "contract smoke failed" "See log, then fix prompt CSV/schema mismatch."
  fi
}

probe_hf_model() {
  local label="$1"
  local model_ref="$2"
  local require_weights="$3"
  local log_file="${EXPT_DIR}/hf_${label}.log"
  if [[ "${SETUP_DOWNLOAD_MODE}" == "none" || "${SETUP_DOWNLOAD_MODE}" == "0" ]]; then
    record_item skipped "model" "${label}" "${model_ref}" "SETUP_DOWNLOAD_MODE=${SETUP_DOWNLOAD_MODE}" ""
    return
  fi
  if is_local_ref "${model_ref}"; then
    if [[ -e "${model_ref}" ]]; then
      record_item ok "model" "${label}" "${model_ref}" "local model path exists" ""
    else
      record_item missing "model" "${label}" "${model_ref}" "local model path missing" "Download or set the ${label} base-model env var to an existing path under utils/models/."
    fi
    return
  fi
  local -a probe_cmd=(
    conda run --no-capture-output -n "${CONDA_ENV}"
    python -B "${SCRIPT_DIR}/hf_model_access_probe.py"
    --model-id "${model_ref}"
    --cache-dir "${HF_HOME}/hub"
  )
  if [[ "${SETUP_LOCAL_FILES_ONLY}" == "1" ]]; then
    probe_cmd+=(--local-files-only)
  fi
  if [[ "${require_weights}" == "1" ]]; then
    probe_cmd+=(--require-weights)
  fi
  if "${probe_cmd[@]}" >"${log_file}" 2>&1; then
    record_item ok "model" "${label}" "${model_ref}" "HF access/cache probe passed; cache_dir=${HF_HOME}" ""
  else
    record_item blocked "model" "${label}" "${model_ref}" "HF access/cache probe failed; log=${log_file}" "Check HF_TOKEN, network, gated access, or predownload under HF_HOME."
  fi
}

check_local_model_path() {
  local label="$1"
  local path="$2"
  local env_name="$3"
  if [[ "${path}" == "${MODELS_ROOT}"* ]]; then
    if [[ -e "${path}" ]]; then
      record_item ok "generator_model" "${label}" "${path}" "local path under utils/models/ exists" ""
    else
      record_item missing "generator_model" "${label}" "${path}" "local path under utils/models/ is missing" "Download model or correct ${env_name}."
    fi
  else
    record_item blocked "generator_model" "${label}" "${path}" "path is not under utils/models/" "Set ${env_name} under ${MODELS_ROOT}."
  fi
}

check_retriever_artifacts() {
  local family="$1"
  local prefix="$2"
  local prompt="$3"
  local suffix
  suffix="$(prompt_mode_suffix "${prompt}")"
  local expt_id="${prefix}_${suffix}"
  local adapter="${REPO_ROOT}/Experiment/${expt_id}/retriever/model/adapter_config.json"
  local train_log="${REPO_ROOT}/Experiment/${expt_id}/retriever/train.log"
  local predictions="${REPO_ROOT}/Experiment/${expt_id}/retriever/outputs/predictions.txt"
  local matched="${REPO_ROOT}/Experiment/${expt_id}/retriever/outputs/best_matched_with_retrieved_facts_and_questions.json"
  local status="warn"
  local action="Run Experiment ${family} for prompt_mode=${prompt} if this artifact is needed."
  if [[ "${SETUP_ARTIFACTS_REQUIRED}" == "1" ]]; then
    status="missing"
  fi
  if [[ -s "${adapter}" ]]; then
    record_item ok "retriever_artifact" "${expt_id}:adapter" "${adapter}" "adapter present" ""
  else
    record_item "${status}" "retriever_artifact" "${expt_id}:adapter" "${adapter}" "adapter missing" "${action}"
  fi
  if [[ -s "${train_log}" ]]; then
    record_item ok "retriever_artifact" "${expt_id}:train_log" "${train_log}" "train log present" ""
  else
    record_item "${status}" "retriever_artifact" "${expt_id}:train_log" "${train_log}" "train log missing" "${action}"
  fi
  if [[ -s "${predictions}" ]]; then
    record_item ok "retriever_artifact" "${expt_id}:predictions" "${predictions}" "test predictions present" ""
  else
    record_item "${status}" "retriever_artifact" "${expt_id}:predictions" "${predictions}" "test predictions missing" "${action}"
  fi
  if [[ -s "${matched}" ]]; then
    record_item ok "retriever_artifact" "${expt_id}:matched" "${matched}" "test matched artifact present" ""
  else
    record_item "${status}" "retriever_artifact" "${expt_id}:matched" "${matched}" "test matched artifact missing" "${action}"
  fi
}

write_report() {
  SETUP_ITEMS_TSV="${SETUP_ITEMS_TSV}" \
  SETUP_REPORT_JSON="${SETUP_REPORT_JSON}" \
  SETUP_WORKSPACE_LOG_JSON="${SETUP_WORKSPACE_LOG_JSON}" \
  SETUP_ID="${SETUP_ID}" \
  SETUP_REPO_ROOT="${REPO_ROOT}" \
  SETUP_WORKSPACE_ROOT="${WORKSPACE_ROOT}" \
  SETUP_LOG_ROOT="${FQAN_LOG_ROOT}" \
  SETUP_MODELS_ROOT="${MODELS_ROOT}" \
  SETUP_HF_HOME="${HF_HOME}" \
  SETUP_DOWNLOAD_MODE="${SETUP_DOWNLOAD_MODE}" \
  SETUP_LOCAL_FILES_ONLY="${SETUP_LOCAL_FILES_ONLY}" \
  SETUP_OK_COUNT="${ok_count}" \
  SETUP_WARN_COUNT="${warn_count}" \
  SETUP_BLOCKER_COUNT="${blocker_count}" \
  conda run --no-capture-output -n "${CONDA_ENV}" python -B - <<'PY'
import json
import os
from pathlib import Path

items = []
items_path = Path(os.environ["SETUP_ITEMS_TSV"])
for line in items_path.read_text(encoding="utf-8").splitlines():
    status, category, name, path, detail, action = (line.split("\t") + [""] * 6)[:6]
    items.append(
        {
            "status": status,
            "category": category,
            "name": name,
            "path": path or None,
            "detail": detail,
            "action": action or None,
        }
    )

payload = {
    "expt_id": os.environ["SETUP_ID"],
    "status": "blocked" if int(os.environ["SETUP_BLOCKER_COUNT"]) else "ok",
    "repo_root": os.environ["SETUP_REPO_ROOT"],
    "workspace_root": os.environ["SETUP_WORKSPACE_ROOT"],
    "models_root": os.environ["SETUP_MODELS_ROOT"],
    "hf_home": os.environ["SETUP_HF_HOME"],
    "download_mode": os.environ["SETUP_DOWNLOAD_MODE"],
    "local_files_only": os.environ["SETUP_LOCAL_FILES_ONLY"] == "1",
    "counts": {
        "ok": int(os.environ["SETUP_OK_COUNT"]),
        "warn": int(os.environ["SETUP_WARN_COUNT"]),
        "blockers": int(os.environ["SETUP_BLOCKER_COUNT"]),
    },
    "items": items,
}

for target in (Path(os.environ["SETUP_REPORT_JSON"]), Path(os.environ["SETUP_WORKSPACE_LOG_JSON"])):
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

index_path = Path(os.environ["SETUP_LOG_ROOT"]) / "index.json"
entry = {
    "time": os.environ["SETUP_ID"].replace("experiment_setup_", ""),
    "path": str(Path(os.environ["SETUP_WORKSPACE_LOG_JSON"]).relative_to(Path(os.environ["SETUP_WORKSPACE_ROOT"]))),
    "repo": os.environ["SETUP_REPO_ROOT"],
    "kind": "experiment_setup",
    "status": payload["status"],
    "summary": f"Experiment setup blockers={payload['counts']['blockers']} warnings={payload['counts']['warn']}",
    "tags": ["experiment_setup", "preflight", "models", "datasets"],
}
try:
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {"entries": []}
except json.JSONDecodeError:
    index = {"entries": []}
entries = index.setdefault("entries", [])
entries.append(entry)
index_path.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
PY
}

case "${SETUP_DOWNLOAD_MODE}" in
  none|0|metadata|weights) ;;
  *)
    printf "Unsupported SETUP_DOWNLOAD_MODE=%s; use none, metadata, or weights.\n" "${SETUP_DOWNLOAD_MODE}" >&2
    exit 2
    ;;
esac

printf "Experiment setup: %s\n" "${SETUP_ID}"
printf "repo_root=%s\nworkspace_root=%s\nmodels_root=%s\nhf_home=%s\n" \
  "${REPO_ROOT}" "${WORKSPACE_ROOT}" "${MODELS_ROOT}" "${HF_HOME}"

check_command_available bash
check_command_available conda
check_conda_python
check_command_available nvidia-smi
check_command_available tmux

check_json_file "config" "docs/args.json" "${FQAN_DOCS_ROOT}/args.json"
check_json_file "config" "docs/log/index.json" "${FQAN_LOG_ROOT}/index.json"

if bash -n \
    "${SCRIPT_DIR}/experiment_1_mistral_retriever.sh" \
    "${SCRIPT_DIR}/experiment_2_flan_retriever.sh" \
    "${SCRIPT_DIR}/experiment_3_t5gemma_retriever.sh" \
    "${SCRIPT_DIR}/experiment_4_retriever_loss_diagram.sh" \
    "${SCRIPT_DIR}/experiment_5_qwen_few10_smoke.sh" \
    "${SCRIPT_DIR}/experiment_6_data_binding_evaluation.sh" \
    "${SCRIPT_DIR}/experiment_7_in_context_selection.sh" \
    "${SCRIPT_DIR}/experiment_7_generator_answer.sh" \
    "${SCRIPT_DIR}/experiment_7_formal_tmux_run.sh" \
    "${SCRIPT_DIR}/retriever_experiment_lib.sh" >/dev/null 2>&1; then
  record_item ok "script_syntax" "experiments_1_7" "${SCRIPT_DIR}" "bash -n passed" ""
else
  record_item blocked "script_syntax" "experiments_1_7" "${SCRIPT_DIR}" "bash -n failed" "Run bash -n manually for the listed scripts."
fi

require_weights=0
if [[ "${SETUP_DOWNLOAD_MODE}" == "weights" ]]; then
  require_weights=1
fi
probe_hf_model "flan_t5_large" "${FLAN_BASE_MODEL}" "${require_weights}"
probe_hf_model "mistral_v0_3" "${MISTRAL_BASE_MODEL}" "${require_weights}"
probe_hf_model "t5gemma_2_1b_1b" "${T5GEMMA_BASE_MODEL}" "${require_weights}"

check_local_model_path "qwen3_6" "${QWEN3_6_MODEL_PATH}" "QWEN3_6_MODEL_PATH"
check_local_model_path "mistral4" "${MISTRAL_SMALL_MODEL_PATH}" "MISTRAL_SMALL_MODEL_PATH"
check_local_model_path "deepseek" "${DEEPSEEK_MODEL_PATH}" "DEEPSEEK_MODEL_PATH"
check_local_model_path "qwythos" "${QWYTHOS_MODEL_PATH}" "QWYTHOS_MODEL_PATH"
check_local_model_path "llama4" "${LLAMA4_MODEL_PATH}" "LLAMA4_MODEL_PATH"

for prompt in ${SETUP_PROMPT_MODES}; do
  train_csv="$(prompt_train_csv "${prompt}")"
  dev_csv="$(prompt_dev_csv "${prompt}")"
  test_csv="$(prompt_test_csv "${prompt}")"
  check_csv_rows "retriever_dataset" "${prompt}:train" "${train_csv}" "${SETUP_EXPECT_TRAIN_ROWS}" "Regenerate prompt data before Experiment 1-4."
  check_csv_rows "retriever_dataset" "${prompt}:dev" "${dev_csv}" "${SETUP_EXPECT_DEV_ROWS}" "Regenerate prompt data before Experiment 1-4."
  check_csv_rows "retriever_dataset" "${prompt}:test" "${test_csv}" "${SETUP_EXPECT_TEST_ROWS}" "Regenerate prompt data before Experiment 1-4."
  if [[ -s "${train_csv}" && -s "${dev_csv}" && -s "${test_csv}" ]]; then
    check_prompt_contract "${prompt}" "${train_csv}" "${dev_csv}" "${test_csv}"
  fi
done

check_csv_rows "experiment_5_dataset" "few10" "${WORKSPACE_ROOT}/data/testing/finqa_10_rel_fact_instruction.csv" 10 "Create or restore data/testing/finqa_10_rel_fact_instruction.csv."

check_file_required "experiment_6_source" "narrative2.xlsx" "${FQAN_DATA_ROOT}/src/narratives/narrative2.xlsx" "Restore source narrative workbook."
check_file_required "experiment_6_source" "full_example.csv" "${WORKSPACE_ROOT}/data/src/full_example.csv" "Restore many/dynamic-shot example source."
for route in original zero_shot many_shot dynamic_shot; do
  case "${route}" in
    original) dir="finqa_original" ;;
    zero_shot) dir="finqa_zero_shot" ;;
    many_shot) dir="finqa_many_shot" ;;
    dynamic_shot) dir="finqa_dynamic_shot" ;;
  esac
  check_csv_rows "experiment_6_dataset" "${route}:narratives_csv" "${WORKSPACE_ROOT}/data/${dir}/narratives1_rel_fact_instruction.csv" "${SETUP_EXPECT_NARRATIVE_ROWS}" "Run EXPERIMENT6_PREPARE_PROMPT_DATA=1 bash dist/experiment_6_data_binding_evaluation.sh."
  check_file_required "experiment_6_dataset" "${route}:gold_jsonl" "${WORKSPACE_ROOT}/data/${dir}/narratives_gold.jsonl" "Run EXPERIMENT6_PREPARE_GOLD_DATA=1 bash dist/experiment_6_data_binding_evaluation.sh."
done
check_file_required "experiment_6_dataset" "testing_gold_jsonl" "${WORKSPACE_ROOT}/data/testing/narratives_gold.jsonl" "Run EXPERIMENT6_PREPARE_GOLD_DATA=1 bash dist/experiment_6_data_binding_evaluation.sh."

check_csv_rows "experiment_7_dataset" "selection_train_source" "${WORKSPACE_ROOT}/data/src/FINDER/finqa_train_rel_fact_instruction.csv" "${SETUP_EXPECT_TRAIN_ROWS}" "Restore raw FinQA train source."
for prompt in original zero-shot many-shot dynamic-shot; do
  check_csv_rows "experiment_7_dataset" "${prompt}:dev" "$(prompt_dev_csv "${prompt}")" "${SETUP_EXPECT_DEV_ROWS}" "Regenerate prompt data before Experiment 7 dev runs."
  check_csv_rows "experiment_7_dataset" "${prompt}:test" "$(prompt_test_csv "${prompt}")" "${SETUP_EXPECT_TEST_ROWS}" "Regenerate prompt data before Experiment 7 test runs."
done

if [[ "${SETUP_CHECK_ARTIFACTS}" == "1" ]]; then
  for prompt in ${SETUP_PROMPT_MODES}; do
    check_retriever_artifacts "1" "finqa_mistral" "${prompt}"
    check_retriever_artifacts "2" "finqa_flan" "${prompt}"
    check_retriever_artifacts "3" "finqa_t5gemma2" "${prompt}"
  done
  selection_cache="${EXPERIMENT7_SELECTION_CACHE_JSON:-${REPO_ROOT}/Experiment/${EXPERIMENT7_SELECTION_EXPT_ID:-experiment_7_selection_gpt55_finqa_train_formal_20260611T080113Z}/in_context_selection/${EXPERIMENT7_SELECTION_ENGINE:-gpt5_5}/selection_cache.json}"
  if [[ -s "${selection_cache}" ]]; then
    record_item ok "experiment_7_artifact" "selection_cache" "${selection_cache}" "selection cache present" ""
  else
    record_item warn "experiment_7_artifact" "selection_cache" "${selection_cache}" "selection cache missing" "Run dist/experiment_7_in_context_selection.sh before formal Experiment 7 answer runs."
  fi
else
  record_item skipped "artifact_check" "retriever_and_selection_artifacts" "" "SETUP_CHECK_ARTIFACTS=${SETUP_CHECK_ARTIFACTS}" ""
fi

if [[ -n "${OPENAI_API_KEY:-}" || -n "${AZURE_OPENAI_API_KEY:-}" ]]; then
  record_item ok "api_credentials" "openai_or_azure" "" "API credential env present; value not logged" ""
else
  record_item warn "api_credentials" "openai_or_azure" "" "no OpenAI/Azure credential env present" "Expected blocker for gpt4_1/gpt5 API routes until credentials are exported."
fi
if [[ -n "${CODEX_CLI_PATH:-}" || -n "$(command -v codex 2>/dev/null || true)" ]]; then
  record_item ok "api_credentials" "codex_cli" "${CODEX_CLI_PATH:-$(command -v codex 2>/dev/null || true)}" "Codex CLI route present" ""
else
  record_item warn "api_credentials" "codex_cli" "" "Codex CLI not found" "Required for default gpt5_5/gpt5_3_codexS CLI routes unless using API/ChatMock route."
fi

write_report

printf "\nSetup report: %s\n" "${SETUP_REPORT_JSON}"
printf "Workspace log: %s\n" "${SETUP_WORKSPACE_LOG_JSON}"
printf "Summary: ok=%s warn=%s blockers=%s\n" "${ok_count}" "${warn_count}" "${blocker_count}"

if [[ "${SETUP_STRICT}" == "1" && "${blocker_count}" -gt 0 ]]; then
  exit 2
fi
