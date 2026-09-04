#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/retriever_experiment_lib.sh"
source "${SCRIPT_DIR}/generator_runtime_profiles.sh"

ENGINE="${ENGINE:-qwen3_6}"
ENGINES="${ENGINES:-qwen3_6 mistral4 llama3_3}"
ALL_ENGINES="${ALL_ENGINES:-0}"
MATRIX_STATUS_JSON="${MATRIX_STATUS_JSON:-${REPO_ROOT}/Experiment/generator_validation/generator_route_matrix_status.json}"
INPUT_JSON="${INPUT_JSON:-${REPO_ROOT}/Experiment/finqa_flan_o/retriever/outputs/best_matched_with_retrieved_facts_and_questions.json}"
OUTPUT_JSONL_EXPLICIT=0
STATUS_JSON_EXPLICIT=0
if [[ -n "${OUTPUT_JSONL:-}" ]]; then
  OUTPUT_JSONL_EXPLICIT=1
else
  OUTPUT_JSONL="${REPO_ROOT}/Experiment/generator_validation/${ENGINE}_preflight.jsonl"
fi
if [[ -n "${STATUS_JSON:-}" ]]; then
  STATUS_JSON_EXPLICIT=1
else
  STATUS_JSON="${REPO_ROOT}/Experiment/generator_validation/${ENGINE}_preflight_status.json"
fi
LIMIT="${LIMIT:-1}"
MAX_TOKENS="${MAX_TOKENS:-128}"
SHOW_PROMPT="${SHOW_PROMPT:-0}"
CHECK_ENDPOINT="${CHECK_ENDPOINT:-1}"
ensure_cuda_compat_library_path "${CONDA_ENV}"

usage() {
  cat <<'EOF'
Usage:
  bash dist/generator_runtime_preflight.sh [options]

Options:
  --engine NAME          Generator engine alias.
  --engines "A B"        Engine aliases for --all-engines.
  --all-engines          Validate every engine in --engines and write a route matrix.
  --input-json PATH      Matched retriever JSON input.
  --output-jsonl PATH    Planned generator JSONL output.
  --status-json PATH     Validation status JSON path.
  --limit N              Number of rows to validate. Default: 1.
  --max-tokens N         Max generation tokens. Default: 512.
  --show-prompt          Print prompt preview in validation output.
  --no-endpoint-check    Skip curl check for OpenAI-compatible endpoints.

This script is validation-only. It does not pass --execute to new_full_finqa_run.py.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --engine)
      ENGINE="$2"
      shift 2
      ;;
    --engines)
      ENGINES="$2"
      shift 2
      ;;
    --all-engines)
      ALL_ENGINES="1"
      shift
      ;;
    --input-json)
      INPUT_JSON="$2"
      shift 2
      ;;
    --output-jsonl)
      OUTPUT_JSONL="$2"
      OUTPUT_JSONL_EXPLICIT=1
      shift 2
      ;;
    --status-json)
      STATUS_JSON="$2"
      STATUS_JSON_EXPLICIT=1
      shift 2
      ;;
    --limit)
      LIMIT="$2"
      shift 2
      ;;
    --max-tokens)
      MAX_TOKENS="$2"
      shift 2
      ;;
    --show-prompt)
      SHOW_PROMPT="1"
      shift
      ;;
    --no-endpoint-check)
      CHECK_ENDPOINT="0"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf "Unknown option: %s\n" "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

require_file "${INPUT_JSON}"

if [[ "${OUTPUT_JSONL_EXPLICIT}" != "1" ]]; then
  OUTPUT_JSONL="${REPO_ROOT}/Experiment/generator_validation/${ENGINE}_preflight.jsonl"
fi
if [[ "${STATUS_JSON_EXPLICIT}" != "1" ]]; then
  STATUS_JSON="${REPO_ROOT}/Experiment/generator_validation/${ENGINE}_preflight_status.json"
fi

if [[ "${ALL_ENGINES}" == "1" ]]; then
  overall_rc=0
  for engine_item in ${ENGINES}; do
    engine_output_jsonl="${REPO_ROOT}/Experiment/generator_validation/${engine_item}_preflight.jsonl"
    engine_status_json="${REPO_ROOT}/Experiment/generator_validation/${engine_item}_preflight_status.json"
    show_prompt_arg=()
    if [[ "${SHOW_PROMPT}" == "1" ]]; then
      show_prompt_arg=(--show-prompt)
    fi
    set +e
    ENGINE="${engine_item}" INPUT_JSON="${INPUT_JSON}" OUTPUT_JSONL="${engine_output_jsonl}" STATUS_JSON="${engine_status_json}" LIMIT="${LIMIT}" MAX_TOKENS="${MAX_TOKENS}" CHECK_ENDPOINT="${CHECK_ENDPOINT}" ALL_ENGINES=0       bash "${BASH_SOURCE[0]}" --engine "${engine_item}" --input-json "${INPUT_JSON}" --output-jsonl "${engine_output_jsonl}" --status-json "${engine_status_json}" --limit "${LIMIT}" --max-tokens "${MAX_TOKENS}" "${show_prompt_arg[@]}"
    rc=$?
    set -e
    if [[ "${overall_rc}" -eq 0 && "${rc}" -ne 0 ]]; then
      overall_rc="${rc}"
    fi
  done
  conda run --no-capture-output -n "${CONDA_ENV}"     python -B "${SCRIPT_DIR}/generator_agents_smoke.py"     --input-json "${INPUT_JSON}"     --output-json "${MATRIX_STATUS_JSON}"     --max-tokens "${MAX_TOKENS}"     --engines ${ENGINES}
  cat "${MATRIX_STATUS_JSON}"
  exit "${overall_rc}"
fi

mkdir -p "$(dirname "${OUTPUT_JSONL}")" "$(dirname "${STATUS_JSON}")"

case "${ENGINE}" in
  qwen3_6|mistral4|llama3_3)
    apply_generator_runtime_profile "${ENGINE}" "${VLLM_RUNTIME_PROFILE:-}"
    ;;
esac

run_logged "${REPO_ROOT}/Experiment/generator_validation/${ENGINE}_environment.log"   conda run --no-capture-output -n "${CONDA_ENV}" \
  python -B -c 'import importlib.metadata as md; import torch, torchvision, torchaudio, transformers, vllm, openai, httpx, numpy, llguidance, xgrammar, mistral_common, bitsandbytes; print("python_ok=1"); print("torch=" + torch.__version__); print("torch_cuda=" + str(torch.version.cuda)); print("cuda_available=" + str(torch.cuda.is_available())); print("cuda_device_count=" + str(torch.cuda.device_count())); print("torchvision=" + torchvision.__version__); print("torchaudio=" + torchaudio.__version__); print("transformers=" + transformers.__version__); print("vllm=" + vllm.__version__); print("llguidance=" + md.version("llguidance")); print("xgrammar=" + md.version("xgrammar")); print("mistral_common=" + md.version("mistral_common")); print("bitsandbytes=" + md.version("bitsandbytes")); print("openai=" + md.version("openai")); print("httpx=" + md.version("httpx")); print("numpy=" + md.version("numpy"))'

if [[ "${CHECK_ENDPOINT}" == "1" && -n "${VLLM_BASE_URL:-}" ]]; then
  if command -v curl >/dev/null 2>&1; then
    run_logged "${REPO_ROOT}/Experiment/generator_validation/${ENGINE}_vllm_models.log" \
      curl -s "${VLLM_BASE_URL%/}/models"
  else
    printf "curl is not available; skipping VLLM /models check.\n" >&2
  fi
fi

validate_cmd=(
  conda run --no-capture-output -n "${CONDA_ENV}"
  python -B "${REPO_ROOT}/new_full_finqa_run.py"
  --engine "${ENGINE}"
  --input-json "${INPUT_JSON}"
  --output-jsonl "${OUTPUT_JSONL}"
  --status-json "${STATUS_JSON}"
  --limit "${LIMIT}"
  --max-tokens "${MAX_TOKENS}"
)

if [[ "${SHOW_PROMPT}" == "1" ]]; then
  validate_cmd+=(--show-prompt)
fi

run_logged "${REPO_ROOT}/Experiment/generator_validation/${ENGINE}_validate.log" "${validate_cmd[@]}"

cat "${STATUS_JSON}"
