#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/retriever_experiment_lib.sh"

EXPT_ID="${EXPT_ID:-experiment_7_dev_retfact_refresh_$(date -u +%Y%m%dT%H%M%SZ)}"
EXPT_DIR="${REPO_ROOT}/Experiment/${EXPT_ID}"
TARGET_ROOT="${TARGET_ROOT:-${REPO_ROOT}/Experiment/experiment_7_target_selection_gpt55_all_cases_20260612T012548Z/retriever_sources}"
DEFAULT_REFRESH_MATRIX="finqa_flan_o finqa_flan_z finqa_flan_m finqa_flan_d finqa_mistral_o finqa_mistral_z finqa_mistral_m finqa_mistral_d finqa_t5gemma2_o finqa_t5gemma2_z finqa_t5gemma2_m finqa_t5gemma2_d"
REFRESH_MATRIX="${REFRESH_MATRIX:-${DEFAULT_REFRESH_MATRIX}}"
PRECHECK_ONLY="${PRECHECK_ONLY:-0}"
SKIP_VALIDATED_EXISTING="${SKIP_VALIDATED_EXISTING:-1}"
REPLACE_CANONICAL="${REPLACE_CANONICAL:-1}"
WRITE_WORKSPACE_LOG="${WRITE_WORKSPACE_LOG:-1}"
CHECK_GPU_AVAILABLE="${CHECK_GPU_AVAILABLE:-1}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-0}"
GPU_WAIT_POLL_SECONDS="${GPU_WAIT_POLL_SECONDS:-60}"
RETRIEVER_INFER_CUDA_DEVICES="${RETRIEVER_INFER_CUDA_DEVICES:-${INFER_CUDA_DEVICES:-1}}"
RETRIEVER_MAX_INFER_SAMPLES="${RETRIEVER_MAX_INFER_SAMPLES:--1}"
EXPERIMENT7_RETRIEVER_INFER_BATCH_SIZE="${EXPERIMENT7_RETRIEVER_INFER_BATCH_SIZE:-${EXPERIMENT6_RETRIEVER_INFER_BATCH_SIZE:-${FLAN_INFER_BATCH_SIZE:-16}}}"
EXPERIMENT7_MISTRAL_INFER_BATCH_SIZE="${EXPERIMENT7_MISTRAL_INFER_BATCH_SIZE:-${MISTRAL_INFER_BATCH_SIZE:-2}}"
EXPERIMENT7_T5GEMMA_INFER_BATCH_SIZE="${EXPERIMENT7_T5GEMMA_INFER_BATCH_SIZE:-${T5GEMMA_BATCH_SIZE:-8}}"
EXPERIMENT7_MATCH_EMBED_BATCH_SIZE="${EXPERIMENT7_MATCH_EMBED_BATCH_SIZE:-${EXPERIMENT6_MATCH_EMBED_BATCH_SIZE:-${MATCH_EMBED_BATCH_SIZE:-1024}}}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF}}"
FLAN_MAX_NEW_TOKENS="${FLAN_MAX_NEW_TOKENS:-128}"
MISTRAL_MAX_NEW_TOKENS="${MISTRAL_MAX_NEW_TOKENS:-256}"
T5GEMMA_MAX_NEW_TOKENS="${T5GEMMA_MAX_NEW_TOKENS:-128}"
FLAN_STRUCTURED_OUTPUT="${FLAN_STRUCTURED_OUTPUT:-assembler}"
MISTRAL_STRUCTURED_OUTPUT="${MISTRAL_STRUCTURED_OUTPUT:-assembler}"
T5GEMMA_STRUCTURED_OUTPUT="${T5GEMMA_STRUCTURED_OUTPUT:-assembler}"

MANIFEST_JSON="${EXPT_DIR}/dev_retfact_refresh_manifest.json"
CASE_JSONL="${EXPT_DIR}/dev_retfact_refresh_cases.jsonl"
GPU_BUSY_JSON="${EXPT_DIR}/gpu_busy.json"
TIMELINE_JSONL="${EXPT_DIR}/timeline.jsonl"
BACKUP_ROOT="${EXPT_DIR}/backup_target_selection_retriever_sources"
mkdir -p "${EXPT_DIR}" "${EXPT_DIR}/retriever_sources"
: > "${CASE_JSONL}"

if [[ "${REPLACE_CANONICAL}" == "1" && "${RETRIEVER_MAX_INFER_SAMPLES}" != "-1" ]]; then
  printf "REPLACE_CANONICAL=1 requires RETRIEVER_MAX_INFER_SAMPLES=-1; got %s\n" "${RETRIEVER_MAX_INFER_SAMPLES}" >&2
  exit 2
fi

case_dir_name() {
  printf "%s_finqa_dev\n" "$1"
}

case_family() {
  case "$1" in
    finqa_flan_*) printf "flan\n" ;;
    finqa_mistral_*) printf "mistral\n" ;;
    finqa_t5gemma2_*) printf "t5gemma2\n" ;;
    *) printf "unsupported retriever: %s\n" "$1" >&2; return 2 ;;
  esac
}

case_prompt_mode() {
  case "$1" in
    *_r) printf "raw\n" ;;
    *_o) printf "original\n" ;;
    *_z) printf "zero-shot\n" ;;
    *_m) printf "many-shot\n" ;;
    *_d) printf "dynamic-shot\n" ;;
    *) printf "unsupported retriever prompt suffix: %s\n" "$1" >&2; return 2 ;;
  esac
}

retriever_model_for_family() {
  case "$1" in
    flan) printf "flan_t5_large\n" ;;
    mistral) printf "mistral_v0_3\n" ;;
    t5gemma2) printf "t5gemma_2_1b_1b\n" ;;
    *) return 2 ;;
  esac
}

canonical_adapter_dir() {
  printf "%s/Experiment/%s/retriever/model\n" "${REPO_ROOT}" "$1"
}

input_csv_for_case() {
  local prompt_mode="$1"
  printf "%s/finqa_dev_rel_fact_instruction.csv\n" "$(prompt_data_dir "${prompt_mode}")"
}

adapter_base_model() {
  local adapter_dir="$1"
  ADAPTER_CONFIG="${adapter_dir}/adapter_config.json" python3 - <<'PYBASE'
import json
import os
from pathlib import Path
path = Path(os.environ["ADAPTER_CONFIG"])
payload = json.loads(path.read_text(encoding="utf-8"))
print(payload.get("base_model_name_or_path") or "")
PYBASE
}

write_timeline() {
  local phase="$1"
  local status="$2"
  local detail="${3:-}"
  TIMELINE_PATH="${TIMELINE_JSONL}" \
  TIMELINE_TIME="$(utc_now)" \
  TIMELINE_EXPT_ID="${EXPT_ID}" \
  TIMELINE_PHASE="${phase}" \
  TIMELINE_STATUS="${status}" \
  TIMELINE_DETAIL="${detail}" \
  python3 - <<'PYTIMELINE'
import json
import os
from pathlib import Path
payload = {
    "time": os.environ["TIMELINE_TIME"],
    "expt_id": os.environ["TIMELINE_EXPT_ID"],
    "phase": os.environ["TIMELINE_PHASE"],
    "status": os.environ["TIMELINE_STATUS"],
    "detail": os.environ.get("TIMELINE_DETAIL") or None,
}
path = Path(os.environ["TIMELINE_PATH"])
with path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
PYTIMELINE
}

record_case() {
  local retriever="$1"
  local status="$2"
  local reason="${3:-}"
  local prompt_mode="${4:-}"
  local input_csv="${5:-}"
  local adapter_dir="${6:-}"
  local base_model="${7:-}"
  local source_dir="${8:-}"
  local target_dir="${9:-}"
  CASE_JSONL="${CASE_JSONL}" \
  CASE_TIME="$(utc_now)" \
  CASE_RETRIEVER="${retriever}" \
  CASE_STATUS="${status}" \
  CASE_REASON="${reason}" \
  CASE_PROMPT_MODE="${prompt_mode}" \
  CASE_INPUT_CSV="${input_csv}" \
  CASE_ADAPTER_DIR="${adapter_dir}" \
  CASE_BASE_MODEL="${base_model}" \
  CASE_SOURCE_DIR="${source_dir}" \
  CASE_TARGET_DIR="${target_dir}" \
  python3 - <<'PYCASE'
import json
import os
from pathlib import Path
payload = {
    "time": os.environ["CASE_TIME"],
    "retriever": os.environ["CASE_RETRIEVER"],
    "dataset": "finqa_dev",
    "status": os.environ["CASE_STATUS"],
    "reason": os.environ.get("CASE_REASON") or None,
    "prompt_mode": os.environ.get("CASE_PROMPT_MODE") or None,
    "input_csv": os.environ.get("CASE_INPUT_CSV") or None,
    "adapter_dir": os.environ.get("CASE_ADAPTER_DIR") or None,
    "base_model": os.environ.get("CASE_BASE_MODEL") or None,
    "source_dir": os.environ.get("CASE_SOURCE_DIR") or None,
    "target_dir": os.environ.get("CASE_TARGET_DIR") or None,
}
with Path(os.environ["CASE_JSONL"]).open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
PYCASE
}

write_manifest() {
  local status="$1"
  local rc="$2"
  local reason="${3:-}"
  MANIFEST_JSON="${MANIFEST_JSON}" \
  CASE_JSONL="${CASE_JSONL}" \
  GPU_BUSY_JSON="${GPU_BUSY_JSON}" \
  REFRESH_TIME="$(utc_now)" \
  REFRESH_EXPT_ID="${EXPT_ID}" \
  REFRESH_EXPT_DIR="${EXPT_DIR}" \
  REFRESH_TARGET_ROOT="${TARGET_ROOT}" \
  REFRESH_BACKUP_ROOT="${BACKUP_ROOT}" \
  REFRESH_STATUS="${status}" \
  REFRESH_RC="${rc}" \
  REFRESH_REASON="${reason}" \
  REFRESH_MATRIX="${REFRESH_MATRIX}" \
  REFRESH_REPLACE_CANONICAL="${REPLACE_CANONICAL}" \
  REFRESH_PRECHECK_ONLY="${PRECHECK_ONLY}" \
  REFRESH_MAX_INFER_SAMPLES="${RETRIEVER_MAX_INFER_SAMPLES}" \
  python3 - <<'PYMANIFEST'
import json
import os
from pathlib import Path
cases = []
case_path = Path(os.environ["CASE_JSONL"])
if case_path.is_file():
    for line in case_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            cases.append(json.loads(line))
gpu_busy = None
gpu_path = Path(os.environ["GPU_BUSY_JSON"])
if gpu_path.is_file():
    gpu_busy = json.loads(gpu_path.read_text(encoding="utf-8"))
payload = {
    "time": os.environ["REFRESH_TIME"],
    "experiment": "7",
    "stage": "dev_retfact_refresh",
    "expt_id": os.environ["REFRESH_EXPT_ID"],
    "expt_dir": os.environ["REFRESH_EXPT_DIR"],
    "target_root": os.environ["REFRESH_TARGET_ROOT"],
    "backup_root": os.environ["REFRESH_BACKUP_ROOT"],
    "status": os.environ["REFRESH_STATUS"],
    "rc": int(os.environ["REFRESH_RC"]),
    "reason": os.environ.get("REFRESH_REASON") or None,
    "matrix": os.environ["REFRESH_MATRIX"].split(),
    "replace_canonical": os.environ["REFRESH_REPLACE_CANONICAL"] == "1",
    "precheck_only": os.environ["REFRESH_PRECHECK_ONLY"] == "1",
    "max_infer_samples": os.environ["REFRESH_MAX_INFER_SAMPLES"],
    "cases": cases,
    "gpu_busy": gpu_busy,
}
Path(os.environ["MANIFEST_JSON"]).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PYMANIFEST
}

write_workspace_log() {
  local status="$1"
  local rc="$2"
  local reason="${3:-}"
  if [[ "${WRITE_WORKSPACE_LOG}" != "1" ]]; then
    return 0
  fi
  WORKSPACE_LOG_DIR="${FQAN_LOG_ROOT}" \
  MANIFEST_JSON="${MANIFEST_JSON}" \
  LOG_TIME="$(utc_now)" \
  LOG_STAMP="$(date -u +%Y%m%dT%H%M%SZ)" \
  LOG_STATUS="${status}" \
  LOG_RC="${rc}" \
  LOG_REASON="${reason}" \
  REPO_ROOT="${REPO_ROOT}" \
  python3 - <<'PYLOG'
import json
import os
from pathlib import Path
log_dir = Path(os.environ["WORKSPACE_LOG_DIR"])
log_dir.mkdir(parents=True, exist_ok=True)
manifest_path = Path(os.environ["MANIFEST_JSON"])
manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
log_path = log_dir / f"{os.environ['LOG_STAMP']}_experiment7_dev_retfact_refresh.json"
payload = {
    "time": os.environ["LOG_TIME"],
    "kind": "experiment7_dev_retfact_refresh",
    "repo": os.environ["REPO_ROOT"],
    "status": os.environ["LOG_STATUS"],
    "rc": int(os.environ["LOG_RC"]),
    "reason": os.environ.get("LOG_REASON") or None,
    "manifest_json": str(manifest_path),
    "expt_id": manifest.get("expt_id"),
    "target_root": manifest.get("target_root"),
    "case_count": len(manifest.get("cases", [])),
}
log_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
index_path = log_dir / "index.json"
if index_path.is_file():
    index = json.loads(index_path.read_text(encoding="utf-8"))
else:
    index = {"entries": []}
index.setdefault("entries", []).append({
    "time": payload["time"],
    "path": f"indexed docs/log/{log_path.name}",
    "repo": payload["repo"],
    "kind": payload["kind"],
    "status": payload["status"],
    "summary": f"Experiment 7 finqa_dev RetFact refresh status={payload['status']} cases={payload['case_count']}",
    "tags": ["experiment_7", "finqa", "retfact", "retriever", "dev_refresh"],
})
index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PYLOG
}

finish() {
  local status="$1"
  local rc="$2"
  local reason="${3:-}"
  write_manifest "${status}" "${rc}" "${reason}" || true
  write_workspace_log "${status}" "${rc}" "${reason}" || true
  write_timeline "refresh" "${status}" "rc=${rc};${reason}" || true
  exit "${rc}"
}

gpu_busy_processes() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null \
    | awk 'NF && $1 !~ /No running/ {print}'
}

write_gpu_busy_json() {
  local processes="$1"
  GPU_BUSY_JSON="${GPU_BUSY_JSON}" GPU_BUSY_PROCESSES="${processes}" GPU_BUSY_TIME="$(utc_now)" python3 - <<'PYGPU'
import json
import os
from pathlib import Path
lines = [line.strip() for line in os.environ.get("GPU_BUSY_PROCESSES", "").splitlines() if line.strip()]
items = []
for line in lines:
    parts = [part.strip() for part in line.split(",")]
    items.append({
        "pid": parts[0] if len(parts) > 0 else None,
        "process_name": parts[1] if len(parts) > 1 else None,
        "used_memory_mib": parts[2] if len(parts) > 2 else None,
        "raw": line,
    })
payload = {"time": os.environ["GPU_BUSY_TIME"], "status": "gpu_busy", "processes": items}
Path(os.environ["GPU_BUSY_JSON"]).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PYGPU
}

wait_for_gpu_available() {
  if [[ "${CHECK_GPU_AVAILABLE}" != "1" ]]; then
    return 0
  fi
  local start now elapsed busy
  start="$(date +%s)"
  while true; do
    busy="$(gpu_busy_processes || true)"
    if [[ -z "${busy}" ]]; then
      rm -f "${GPU_BUSY_JSON}"
      return 0
    fi
    write_gpu_busy_json "${busy}"
    now="$(date +%s)"
    elapsed=$((now - start))
    if [[ "${GPU_WAIT_SECONDS}" -le 0 || "${elapsed}" -ge "${GPU_WAIT_SECONDS}" ]]; then
      printf "GPU busy; not killing existing processes. Set GPU_WAIT_SECONDS>0 to wait longer, or free GPU and rerun.\n" >&2
      printf "%s\n" "${busy}" >&2
      return 2
    fi
    sleep "${GPU_WAIT_POLL_SECONDS}"
  done
}

validate_case_config() {
  local retriever="$1"
  local family prompt_mode input_csv adapter_dir base_model target_dir source_dir
  family="$(case_family "${retriever}")"
  prompt_mode="$(case_prompt_mode "${retriever}")"
  input_csv="$(input_csv_for_case "${prompt_mode}")"
  adapter_dir="$(canonical_adapter_dir "${retriever}")"
  target_dir="${TARGET_ROOT}/$(case_dir_name "${retriever}")"
  source_dir="${EXPT_DIR}/retriever_sources/$(case_dir_name "${retriever}")"
  if [[ ! -f "${input_csv}" ]]; then
    record_case "${retriever}" "blocked_missing_input_csv" "${input_csv}" "${prompt_mode}" "${input_csv}" "${adapter_dir}" "" "${source_dir}" "${target_dir}"
    return 2
  fi
  if [[ ! -f "${adapter_dir}/adapter_config.json" ]]; then
    record_case "${retriever}" "blocked_missing_canonical_adapter" "${adapter_dir}/adapter_config.json" "${prompt_mode}" "${input_csv}" "${adapter_dir}" "" "${source_dir}" "${target_dir}"
    return 2
  fi
  base_model="$(adapter_base_model "${adapter_dir}")"
  if [[ "${family}" == "mistral" && "${base_model}" != "mistralai/Mistral-7B-Instruct-v0.3" ]]; then
    record_case "${retriever}" "blocked_wrong_mistral_base_model" "base_model=${base_model}" "${prompt_mode}" "${input_csv}" "${adapter_dir}" "${base_model}" "${source_dir}" "${target_dir}"
    return 2
  fi
  record_case "${retriever}" "precheck_ok" "" "${prompt_mode}" "${input_csv}" "${adapter_dir}" "${base_model}" "${source_dir}" "${target_dir}"
}

write_source_status() {
  local retriever="$1"
  local status="$2"
  local reason="$3"
  local prompt_mode="$4"
  local input_csv="$5"
  local adapter_dir="$6"
  local base_model="$7"
  local matched_json="$8"
  local source_dir="$9"
  SOURCE_STATUS_PATH="${source_dir}/source_status.json" \
  SOURCE_TIME="$(utc_now)" \
  SOURCE_RETRIEVER="${retriever}" \
  SOURCE_STATUS="${status}" \
  SOURCE_REASON="${reason}" \
  SOURCE_PROMPT_MODE="${prompt_mode}" \
  SOURCE_INPUT_CSV="${input_csv}" \
  SOURCE_ADAPTER_DIR="${adapter_dir}" \
  SOURCE_BASE_MODEL="${base_model}" \
  SOURCE_MATCHED_JSON="${matched_json}" \
  SOURCE_MAX_INFER_SAMPLES="${RETRIEVER_MAX_INFER_SAMPLES}" \
  SOURCE_INFER_DEVICES="${RETRIEVER_INFER_CUDA_DEVICES}" \
  SOURCE_FLAN_BATCH="${EXPERIMENT7_RETRIEVER_INFER_BATCH_SIZE}" \
  SOURCE_MISTRAL_BATCH="${EXPERIMENT7_MISTRAL_INFER_BATCH_SIZE}" \
  SOURCE_T5GEMMA_BATCH="${EXPERIMENT7_T5GEMMA_INFER_BATCH_SIZE}" \
  SOURCE_MATCH_BATCH="${EXPERIMENT7_MATCH_EMBED_BATCH_SIZE}" \
  python3 - <<'PYSOURCE'
import json
import os
from pathlib import Path
payload = {
    "time": os.environ["SOURCE_TIME"],
    "experiment": "7",
    "stage": "dev_retfact_refresh",
    "retriever_id": os.environ["SOURCE_RETRIEVER"],
    "dataset": "finqa_dev",
    "status": os.environ["SOURCE_STATUS"],
    "blocked_reason": os.environ.get("SOURCE_REASON") or "",
    "input_csv": os.environ["SOURCE_INPUT_CSV"],
    "prompt_mode": os.environ["SOURCE_PROMPT_MODE"],
    "adapter_dir": os.environ["SOURCE_ADAPTER_DIR"],
    "base_model": os.environ["SOURCE_BASE_MODEL"],
    "matched_json": os.environ["SOURCE_MATCHED_JSON"],
    "max_infer_samples": os.environ["SOURCE_MAX_INFER_SAMPLES"],
    "infer_cuda_devices": os.environ["SOURCE_INFER_DEVICES"],
    "flan_batch_size": int(os.environ["SOURCE_FLAN_BATCH"]),
    "mistral_batch_size": int(os.environ["SOURCE_MISTRAL_BATCH"]),
    "t5gemma_batch_size": int(os.environ["SOURCE_T5GEMMA_BATCH"]),
    "match_embed_batch_size": int(os.environ["SOURCE_MATCH_BATCH"]),
}
Path(os.environ["SOURCE_STATUS_PATH"]).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PYSOURCE
}


validate_matched_json() {
  local retriever="$1"
  local input_csv="$2"
  local matched_json="$3"
  local report_json="$4"
  VALIDATE_RETRIEVER="${retriever}" \
  VALIDATE_INPUT_CSV="${input_csv}" \
  VALIDATE_MATCHED_JSON="${matched_json}" \
  VALIDATE_REPORT_JSON="${report_json}" \
  python3 - <<'PYVALIDATE'
import csv
import json
import os
import re
import sys
from pathlib import Path
csv.field_size_limit(sys.maxsize)
retriever = os.environ["VALIDATE_RETRIEVER"]
input_csv = Path(os.environ["VALIDATE_INPUT_CSV"])
matched_json = Path(os.environ["VALIDATE_MATCHED_JSON"])
report_json = Path(os.environ["VALIDATE_REPORT_JSON"])
errors = []
allowed_matched_by = {"prediction_fragment", "question", "retfact_vs_rel_fact_label"}
def norm(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())
with input_csv.open(newline="", encoding="utf-8") as handle:
    csv_questions = [norm(row.get("Question")) for row in csv.DictReader(handle)]
try:
    payload = json.loads(matched_json.read_text(encoding="utf-8"))
except Exception as exc:
    errors.append(f"matched-json parse failed: {exc}")
    payload = None
if not isinstance(payload, list):
    errors.append("matched-json is not a JSON list")
    rows = []
else:
    rows = payload
if len(rows) != 883:
    errors.append(f"row_count={len(rows)}, expected 883")
json_questions = [norm(row.get("question")) for row in rows if isinstance(row, dict)]
if csv_questions != json_questions:
    errors.append("question order mismatch between CSV and matched-json")
required = {"question", "retrieved", "retrieved_with_scores", "predicted_retfact_for_match"}
for index, row in enumerate(rows):
    if not isinstance(row, dict):
        errors.append(f"row {index} is not an object")
        continue
    missing = sorted(required - set(row))
    if missing:
        errors.append(f"row {index} missing keys: {missing}")
    if "experiment7_backfill_provenance" in row:
        errors.append(f"row {index} contains experiment7_backfill_provenance")
    retrieved = row.get("retrieved")
    scored = row.get("retrieved_with_scores")
    if not isinstance(retrieved, list):
        errors.append(f"row {index} retrieved is not list")
        retrieved = []
    if not isinstance(scored, list):
        errors.append(f"row {index} retrieved_with_scores is not list")
        scored = []
    diagnostic_texts = set()
    formal_match_texts = set()
    for match_index, match in enumerate(scored):
        if not isinstance(match, dict):
            errors.append(f"row {index} match {match_index} is not object")
            continue
        matched_by = str(match.get("matched_by", ""))
        text_value = str(match.get("text", "")).strip()
        if matched_by not in allowed_matched_by:
            errors.append(f"row {index} match {match_index} has unexpected matched_by={matched_by!r}")
        if matched_by == "retfact_vs_rel_fact_label" and text_value:
            diagnostic_texts.add(text_value)
        elif matched_by in {"prediction_fragment", "question"} and text_value:
            formal_match_texts.add(text_value)
    retrieved_texts = {str(item).strip() for item in retrieved if str(item).strip()}
    diagnostic_only = diagnostic_texts - formal_match_texts
    leaked = diagnostic_only & retrieved_texts
    if leaked:
        errors.append(f"row {index} diagnostic-only retfact_vs_rel_fact_label leaked into retrieved")
report = {
    "retriever": retriever,
    "input_csv": str(input_csv),
    "matched_json": str(matched_json),
    "rows": len(rows),
    "expected_rows": 883,
    "status": "ok" if not errors else "failed",
    "errors": errors[:100],
    "error_count": len(errors),
}
report_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
if errors:
    print(json.dumps(report, ensure_ascii=False, indent=2), file=sys.stderr)
    raise SystemExit(2)
print(json.dumps(report, ensure_ascii=False, indent=2))
PYVALIDATE
}

run_refresh_case() {
  local retriever="$1"
  local family prompt_mode input_csv adapter_dir base_model retriever_model case_name source_dir target_dir prediction_txt data_json matched_json report_json
  family="$(case_family "${retriever}")"
  prompt_mode="$(case_prompt_mode "${retriever}")"
  input_csv="$(input_csv_for_case "${prompt_mode}")"
  adapter_dir="$(canonical_adapter_dir "${retriever}")"
  base_model="$(adapter_base_model "${adapter_dir}")"
  retriever_model="$(retriever_model_for_family "${family}")"
  case_name="$(case_dir_name "${retriever}")"
  source_dir="${EXPT_DIR}/retriever_sources/${case_name}"
  target_dir="${TARGET_ROOT}/${case_name}"
  prediction_txt="${source_dir}/predictions.txt"
  data_json="${source_dir}/input_data.json"
  matched_json="${source_dir}/best_matched_with_retrieved_facts_and_questions.json"
  report_json="${source_dir}/validation_report.json"
  mkdir -p "${source_dir}"
  write_timeline "${retriever}" "start" "prompt_mode=${prompt_mode}"

  if [[ "${SKIP_VALIDATED_EXISTING}" == "1" && -s "${matched_json}" && -s "${report_json}" ]]; then
    if REPORT_JSON="${report_json}" python3 - <<'PYSKIP'
import json, os, sys
from pathlib import Path
payload = json.loads(Path(os.environ["REPORT_JSON"]).read_text(encoding="utf-8"))
if payload.get("status") == "ok" and int(payload.get("rows", 0)) == 883 and int(payload.get("error_count", 1)) == 0:
    raise SystemExit(0)
raise SystemExit(1)
PYSKIP
    then
      if [[ -s "${prediction_txt}" ]]; then
        cp -p "${prediction_txt}" "${source_dir}/predictions_dev.txt"
      fi
      write_source_status "${retriever}" "retriever_matched_reused" "validated artifact already exists; SKIP_VALIDATED_EXISTING=1" "${prompt_mode}" "${input_csv}" "${adapter_dir}" "${base_model}" "${matched_json}" "${source_dir}"
      record_case "${retriever}" "reused_validated" "" "${prompt_mode}" "${input_csv}" "${adapter_dir}" "${base_model}" "${source_dir}" "${target_dir}"
      write_timeline "${retriever}" "finish" "reused_validated"
      return 0
    fi
  fi

  case_failed() {
    local status="$1"
    local reason="$2"
    write_source_status "${retriever}" "${status}" "${reason}" "${prompt_mode}" "${input_csv}" "${adapter_dir}" "${base_model}" "${matched_json}" "${source_dir}" || true
    record_case "${retriever}" "${status}" "${reason}" "${prompt_mode}" "${input_csv}" "${adapter_dir}" "${base_model}" "${source_dir}" "${target_dir}" || true
    write_timeline "${retriever}" "${status}" "${reason}" || true
    return 2
  }

  case "${family}" in
    flan)
      if ! run_logged "${source_dir}/retriever_inference.log" \
        env CUDA_VISIBLE_DEVICES="${RETRIEVER_INFER_CUDA_DEVICES}" \
          PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
          PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF}" \
        conda run --no-capture-output -n "${CONDA_ENV}" \
        python -B "${REPO_ROOT}/.external/FINDER/Retriever Codes/Flan/lora_flan_large_finqa_rel_fact.py" \
          --mode infer \
          --input-csv "${input_csv}" \
          --eval-csv "${input_csv}" \
          --adapter-dir "${adapter_dir}" \
          --output-txt "${prediction_txt}" \
          --max-infer-samples "${RETRIEVER_MAX_INFER_SAMPLES}" \
          --prompt-mode "${prompt_mode}" \
          --batch-size "${EXPERIMENT7_RETRIEVER_INFER_BATCH_SIZE}" \
          --max-new-tokens "${FLAN_MAX_NEW_TOKENS}" \
          --structured-output "${FLAN_STRUCTURED_OUTPUT}"; then
        case_failed "blocked_retriever_inference_failed" "retriever inference failed"
        return 2
      fi
      ;;
    mistral)
      local mistral_retriever_ld_library_path
      mistral_retriever_ld_library_path="$(prepend_library_path "$(conda_cuda13_library_dirs)" "${LD_LIBRARY_PATH:-}")"
      if ! run_logged "${source_dir}/retriever_inference.log" \
        env CUDA_VISIBLE_DEVICES="${RETRIEVER_INFER_CUDA_DEVICES}" \
          PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
          PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF}" \
          LD_LIBRARY_PATH="${mistral_retriever_ld_library_path}" \
        conda run --no-capture-output -n "${CONDA_ENV}" \
        python -B "${REPO_ROOT}/.external/FINDER/Retriever Codes/Mistral/mistral_inference.py" \
          --input-csv "${input_csv}" \
          --adapter-dir "${adapter_dir}" \
          --output-txt "${prediction_txt}" \
          --max-infer-samples "${RETRIEVER_MAX_INFER_SAMPLES}" \
          --prompt-mode "${prompt_mode}" \
          --batch-size "${EXPERIMENT7_MISTRAL_INFER_BATCH_SIZE}" \
          --max-new-tokens "${MISTRAL_MAX_NEW_TOKENS}" \
          --structured-output "${MISTRAL_STRUCTURED_OUTPUT}"; then
        case_failed "blocked_retriever_inference_failed" "retriever inference failed"
        return 2
      fi
      ;;
    t5gemma2)
      if ! run_logged "${source_dir}/retriever_inference.log" \
        env CUDA_VISIBLE_DEVICES="${RETRIEVER_INFER_CUDA_DEVICES}" \
          PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
          PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF}" \
        conda run --no-capture-output -n "${CONDA_ENV}" \
        python -B "${REPO_ROOT}/.external/FINDER/Retriever Codes/t5gemma-2/t5gemma-2_train.py" \
          --mode infer \
          --train-csv "${input_csv}" \
          --eval-csv "${input_csv}" \
          --input-csv "${input_csv}" \
          --output-dir "${source_dir}/t5gemma_infer" \
          --adapter-dir "${adapter_dir}" \
          --output-txt "${prediction_txt}" \
          --max-infer-samples "${RETRIEVER_MAX_INFER_SAMPLES}" \
          --prompt-mode "${prompt_mode}" \
          --batch-size "${EXPERIMENT7_T5GEMMA_INFER_BATCH_SIZE}" \
          --max-new-tokens "${T5GEMMA_MAX_NEW_TOKENS}" \
          --structured-output "${T5GEMMA_STRUCTURED_OUTPUT}"; then
        case_failed "blocked_retriever_inference_failed" "retriever inference failed"
        return 2
      fi
      ;;
  esac

  if [[ ! -s "${prediction_txt}" ]]; then
    case_failed "blocked_retriever_inference_failed" "missing predictions after retriever inference"
    return 2
  fi
  cp -p "${prediction_txt}" "${source_dir}/predictions_dev.txt"

  if ! run_logged "${source_dir}/build_input_data_json.log" \
    conda run --no-capture-output -n "${CONDA_ENV}" \
    python -B "${SCRIPT_DIR}/build_retriever_few_data_json.py" \
      --input-csv "${input_csv}" \
      --output-json "${data_json}"; then
    case_failed "blocked_build_input_data_failed" "build_retriever_few_data_json failed"
    return 2
  fi

  if ! run_logged "${source_dir}/match.log" \
    conda run --no-capture-output -n "${CONDA_ENV}" \
    python -B "${REPO_ROOT}/result_organization.py" match \
      --dataset finqa \
      --retriever-model "${retriever_model}" \
      --prompt-mode "${prompt_mode}" \
      --input-txt "${prediction_txt}" \
      --data-json "${data_json}" \
      --relfact-csv "${input_csv}" \
      --embedding-batch-size "${EXPERIMENT7_MATCH_EMBED_BATCH_SIZE}" \
      --output-json "${matched_json}" \
      --execute \
      --require-valid-schema; then
    case_failed "blocked_match_failed" "result_organization match failed"
    return 2
  fi

  if ! validate_matched_json "${retriever}" "${input_csv}" "${matched_json}" "${report_json}"; then
    case_failed "blocked_validation_failed" "matched-json validation failed"
    return 2
  fi
  write_source_status "${retriever}" "retriever_matched_generated" "" "${prompt_mode}" "${input_csv}" "${adapter_dir}" "${base_model}" "${matched_json}" "${source_dir}"
  record_case "${retriever}" "generated_validated" "" "${prompt_mode}" "${input_csv}" "${adapter_dir}" "${base_model}" "${source_dir}" "${target_dir}"
  write_timeline "${retriever}" "finish" "validated"
}

replace_case() {
  local retriever="$1"
  local case_name source_dir target_dir tmp_dir backup_dir
  case_name="$(case_dir_name "${retriever}")"
  source_dir="${EXPT_DIR}/retriever_sources/${case_name}"
  target_dir="${TARGET_ROOT}/${case_name}"
  tmp_dir="${TARGET_ROOT}/.${case_name}.tmp.${EXPT_ID}"
  backup_dir="${BACKUP_ROOT}/${case_name}"
  [[ -f "${source_dir}/best_matched_with_retrieved_facts_and_questions.json" ]] || return 2
  mkdir -p "${TARGET_ROOT}" "${BACKUP_ROOT}"
  rm -rf "${tmp_dir}"
  cp -a "${source_dir}" "${tmp_dir}"
  if [[ -e "${target_dir}" ]]; then
    rm -rf "${backup_dir}"
    mv "${target_dir}" "${backup_dir}"
  fi
  mv "${tmp_dir}" "${target_dir}"
  write_timeline "${retriever}" "replaced" "target=${target_dir};backup=${backup_dir}"
}

write_timeline "refresh" "start" "matrix=${REFRESH_MATRIX}"

precheck_rc=0
for retriever in ${REFRESH_MATRIX}; do
  if ! validate_case_config "${retriever}"; then
    precheck_rc=2
  fi
done
if [[ "${precheck_rc}" -ne 0 ]]; then
  finish "blocked_precheck" "${precheck_rc}" "missing input/adapters or invalid Mistral base model"
fi

if [[ "${PRECHECK_ONLY}" == "1" ]]; then
  finish "precheck_ok" 0 "PRECHECK_ONLY=1"
fi

if ! wait_for_gpu_available; then
  finish "blocked_gpu_busy" 2 "GPU has existing compute processes; no process was killed"
fi

for retriever in ${REFRESH_MATRIX}; do
  if ! run_refresh_case "${retriever}"; then
    finish "blocked_case_failed" 2 "case failed: ${retriever}"
  fi
done

if [[ "${REPLACE_CANONICAL}" == "1" ]]; then
  for retriever in ${REFRESH_MATRIX}; do
    if ! replace_case "${retriever}"; then
      finish "blocked_replace_failed" 2 "replacement failed: ${retriever}"
    fi
  done
  finish "completed_replaced" 0 "refreshed and replaced canonical finqa_dev RetFact artifacts"
else
  finish "completed_staged" 0 "refreshed artifacts staged only; REPLACE_CANONICAL=0"
fi
