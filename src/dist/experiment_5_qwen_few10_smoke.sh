#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/retriever_experiment_lib.sh"

EXPT_ID="${EXPT_ID:-experiment_5_qwen3_6_few10_smoke}"
FLOW_SCOPE="isolated_generator_smoke"
LIMIT="${LIMIT:-10}"
ENGINE="${ENGINE:-qwen3_6}"
MAX_TOKENS="${MAX_TOKENS:-512}"
SLEEP_SECONDS="${SLEEP_SECONDS:-0}"
RUN_EXECUTE="${RUN_EXECUTE:-auto}"
CSV_PATH="${CSV_PATH:-$(first_existing_path "${WORKSPACE_ROOT}/data/testing/finqa_10_rel_fact_instruction.csv")}"
EXPT_DIR="${REPO_ROOT}/Experiment/${EXPT_ID}"
INPUT_JSON="${EXPT_DIR}/generator/few10_generator_input.json"
OUTPUT_JSONL="${EXPT_DIR}/generator/${ENGINE}_few10_generated.jsonl"
EXECUTE_STATUS_JSON="${EXPT_DIR}/generator/execute_status.json"
STATUS_JSON="${EXPT_DIR}/generator/execution_status.json"

mkdir -p "${EXPT_DIR}/generator"
require_file "${CSV_PATH}"

shell_quote() {
  printf "%q" "$1"
}

build_experiment5_resume_command() {
  printf "cd %s && EXPT_ID=%s ENGINE=%s RUN_EXECUTE=auto LIMIT=%s MAX_TOKENS=%s SLEEP_SECONDS=%s CSV_PATH=%s bash dist/experiment_5_qwen_few10_smoke.sh" \
    "$(shell_quote "${REPO_ROOT}")" \
    "$(shell_quote "${EXPT_ID}")" \
    "$(shell_quote "${ENGINE}")" \
    "$(shell_quote "${LIMIT}")" \
    "$(shell_quote "${MAX_TOKENS}")" \
    "$(shell_quote "${SLEEP_SECONDS}")" \
    "$(shell_quote "${CSV_PATH}")"
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

print_resume_hint() {
  local command="$1"
  local direct_command="$2"
  cat >&2 <<EOF
Generator API quota/rate-limit interruption detected.
Refresh quota, replace the API key, or restart ChatMock, then resume with:
  ${command}

Direct single-run resume command:
  ${direct_command}
EOF
}

run_logged "${EXPT_DIR}/generator/prepare_input.log" \
  conda run --no-capture-output -n "${CONDA_ENV}" \
  python -B "${REPO_ROOT}/dist/build_few10_generator_input.py" \
  --input-csv "${CSV_PATH}" \
  --output-json "${INPUT_JSON}" \
  --limit "${LIMIT}" \
  --allow-relfact-smoke

VALIDATE_CREDENTIAL_PURPOSE="test"
case "${RUN_EXECUTE}" in
  auto|1|true|True|yes|Yes)
    VALIDATE_CREDENTIAL_PURPOSE="execute"
    ;;
esac

run_logged "${EXPT_DIR}/generator/validate.log" \
  conda run --no-capture-output -n "${CONDA_ENV}" \
  python -B "${REPO_ROOT}/new_full_finqa_run.py" \
  --engine "${ENGINE}" \
  --input-json "${INPUT_JSON}" \
  --output-jsonl "${OUTPUT_JSONL}" \
  --limit "${LIMIT}" \
  --max-tokens "${MAX_TOKENS}" \
  --credential-purpose "${VALIDATE_CREDENTIAL_PURPOSE}" \
  --show-prompt

execute_attempted=0
execute_rc=0
blocked_reason=""
failure_category=""
resume_command=""
resume_notes=""
direct_resume_command=""

case "${RUN_EXECUTE}" in
  0|false|False|no|No)
    blocked_reason="RUN_EXECUTE=${RUN_EXECUTE}"
    ;;
  1|true|True|yes|Yes)
    execute_attempted=1
    ;;
  auto)
    case "${ENGINE}" in
      qwen3_6|mistral4|llama3_3)
        if [[ -n "${VLLM_BASE_URL:-}" ]]; then
          execute_attempted=1
        else
          blocked_reason="missing VLLM_BASE_URL"
        fi
        ;;
      gpt4|gpt4_1)
        if [[ -n "${OPENAI_BASE_URL:-}" && -n "${OPENAI_API_KEY:-}" && "${ALLOW_OPENAI_COMPATIBLE_EXECUTE:-0}" == "1" ]]; then
          execute_attempted=1
        elif [[ -n "${OPENAI_BASE_URL:-}" && -n "${AZURE_OPENAI_ENDPOINT:-}" && -n "${AZURE_OPENAI_API_KEY:-}" && ( -n "${AZURE_OPENAI_GPT4_1_DEPLOYMENT:-}" || -n "${AZURE_OPENAI_GPT4_DEPLOYMENT:-}" || -n "${AZURE_OPENAI_DEPLOYMENT:-}" ) ]]; then
          execute_attempted=1
        elif [[ -z "${OPENAI_BASE_URL:-}" && -n "${OPENAI_API_KEY:-}" ]]; then
          execute_attempted=1
        elif [[ -n "${AZURE_OPENAI_ENDPOINT:-}" && -n "${AZURE_OPENAI_API_KEY:-}" && ( -n "${AZURE_OPENAI_GPT4_1_DEPLOYMENT:-}" || -n "${AZURE_OPENAI_GPT4_DEPLOYMENT:-}" || -n "${AZURE_OPENAI_DEPLOYMENT:-}" ) ]]; then
          execute_attempted=1
        else
          blocked_reason="missing formal GPT-4.1 credentials: unset OPENAI_BASE_URL and set official OPENAI_API_KEY, or set Azure OpenAI endpoint/key/deployment"
        fi
        ;;
      gpt5_3_codexS|gpt5_5)
        execute_attempted=1
        ;;
      *)
        execute_attempted=1
        ;;
    esac
    ;;
  *)
    printf "Unsupported RUN_EXECUTE=%s; use auto, 0, or 1.\n" "${RUN_EXECUTE}" >&2
    exit 2
    ;;
esac

if [[ "${execute_attempted}" == "1" ]]; then
  set +e
  run_logged "${EXPT_DIR}/generator/execute.log" \
    conda run --no-capture-output -n "${CONDA_ENV}" \
    python -B "${REPO_ROOT}/new_full_finqa_run.py" \
    --engine "${ENGINE}" \
    --input-json "${INPUT_JSON}" \
    --output-jsonl "${OUTPUT_JSONL}" \
    --status-json "${EXECUTE_STATUS_JSON}" \
    --limit "${LIMIT}" \
    --max-tokens "${MAX_TOKENS}" \
    --sleep-seconds "${SLEEP_SECONDS}" \
    --credential-purpose execute \
    --execute
  execute_rc=$?
  set -e
  if [[ "${execute_rc}" -ne 0 ]]; then
    mapfile -t failure_meta < <(read_execute_failure_metadata "${EXECUTE_STATUS_JSON}")
    failure_category="${failure_meta[0]:-}"
    direct_resume_command="${failure_meta[1]:-}"
    resume_notes="${failure_meta[2]:-}"
    if [[ "${failure_meta[3]:-0}" == "1" ]]; then
      resume_command="$(build_experiment5_resume_command)"
      print_resume_hint "${resume_command}" "${direct_resume_command}"
    fi
  fi
else
  {
    printf "started_at=%s\n" "$(utc_now)"
    printf "command=skipped_generator_execute\n"
    printf "run_execute=%q\n" "${RUN_EXECUTE}"
    printf "blocked_reason=%q\n" "${blocked_reason}"
    printf "finished_at=%s\n" "$(utc_now)"
    printf "exit_code=0\n"
  } | tee "${EXPT_DIR}/generator/execute.log"
fi

STATUS_TIME="$(utc_now)" \
STATUS_JSON_PATH="${STATUS_JSON}" \
STATUS_FLOW_SCOPE="${FLOW_SCOPE}" \
STATUS_ENGINE="${ENGINE}" \
STATUS_MAX_TOKENS="${MAX_TOKENS}" \
STATUS_RUN_EXECUTE="${RUN_EXECUTE}" \
STATUS_EXECUTE_ATTEMPTED="${execute_attempted}" \
STATUS_BLOCKED_REASON="${blocked_reason}" \
STATUS_INPUT_CSV="${CSV_PATH}" \
STATUS_INPUT_JSON="${INPUT_JSON}" \
STATUS_OUTPUT_JSONL="${OUTPUT_JSONL}" \
STATUS_EXECUTE_LOG="${EXPT_DIR}/generator/execute.log" \
STATUS_EXECUTE_STATUS_JSON="${EXECUTE_STATUS_JSON}" \
STATUS_EXIT_CODE="${execute_rc}" \
STATUS_FAILURE_CATEGORY="${failure_category}" \
STATUS_RESUME_COMMAND="${resume_command}" \
STATUS_DIRECT_RESUME_COMMAND="${direct_resume_command}" \
STATUS_RESUME_NOTES="${resume_notes}" \
STATUS_VLLM_RUNTIME_PROFILE="${VLLM_RUNTIME_PROFILE:-}" \
STATUS_VLLM_TP="${VLLM_TENSOR_PARALLEL_SIZE:-}" \
STATUS_VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-}" \
STATUS_VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-}" \
STATUS_VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-}" \
STATUS_VLLM_DTYPE="${VLLM_DTYPE:-}" \
STATUS_VLLM_KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-}" \
STATUS_VLLM_QUANTIZATION="${VLLM_QUANTIZATION:-}" \
STATUS_VLLM_CPU_OFFLOAD_GB="${VLLM_CPU_OFFLOAD_GB:-}" \
conda run --no-capture-output -n "${CONDA_ENV}" python -B - <<'PY'
import json
import os
from pathlib import Path

execute_attempted = os.environ["STATUS_EXECUTE_ATTEMPTED"] == "1"
exit_code = int(os.environ["STATUS_EXIT_CODE"])
if not execute_attempted:
    status = "validation_completed_runtime_blocked"
elif exit_code == 0:
    status = "completed"
else:
    status = "blocked_or_failed"

payload = {
    "time": os.environ["STATUS_TIME"],
    "experiment": "5",
    "flow_scope": os.environ["STATUS_FLOW_SCOPE"],
    "retrieved_source": "csv_rel_fact_smoke_only",
    "engine": os.environ["STATUS_ENGINE"],
    "max_tokens": int(os.environ["STATUS_MAX_TOKENS"]),
    "run_execute": os.environ["STATUS_RUN_EXECUTE"],
    "execute_attempted": execute_attempted,
    "blocked_reason": os.environ["STATUS_BLOCKED_REASON"],
    "input_csv": os.environ["STATUS_INPUT_CSV"],
    "input_json": os.environ["STATUS_INPUT_JSON"],
    "output_jsonl": os.environ["STATUS_OUTPUT_JSONL"],
    "execute_log": os.environ["STATUS_EXECUTE_LOG"],
    "execute_status_json": os.environ["STATUS_EXECUTE_STATUS_JSON"],
    "failure_category": os.environ["STATUS_FAILURE_CATEGORY"] or None,
    "resume_command": os.environ["STATUS_RESUME_COMMAND"] or None,
    "direct_resume_command": os.environ["STATUS_DIRECT_RESUME_COMMAND"] or None,
    "resume_notes": os.environ["STATUS_RESUME_NOTES"] or None,
    "local_vllm_runtime": {
        "profile": os.environ["STATUS_VLLM_RUNTIME_PROFILE"] or None,
        "tensor_parallel_size": os.environ["STATUS_VLLM_TP"] or None,
        "max_model_len": os.environ["STATUS_VLLM_MAX_MODEL_LEN"] or None,
        "max_num_seqs": os.environ["STATUS_VLLM_MAX_NUM_SEQS"] or None,
        "gpu_memory_utilization": os.environ["STATUS_VLLM_GPU_MEMORY_UTILIZATION"] or None,
        "dtype": os.environ["STATUS_VLLM_DTYPE"] or None,
        "kv_cache_dtype": os.environ["STATUS_VLLM_KV_CACHE_DTYPE"] or None,
        "quantization": os.environ["STATUS_VLLM_QUANTIZATION"] or None,
        "cpu_offload_gb": os.environ["STATUS_VLLM_CPU_OFFLOAD_GB"] or None,
    },
    "exit_code": exit_code,
    "status": status,
}
path = Path(os.environ["STATUS_JSON_PATH"])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY

cat "${STATUS_JSON}"
if [[ "${execute_attempted}" == "1" ]]; then
  exit "${execute_rc}"
fi
