#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/retriever_experiment_lib.sh"
source "${SCRIPT_DIR}/generator_runtime_profiles.sh"

ENGINE="${ENGINE:-qwen3_6}"
HOST="${VLLM_HOST:-localhost}"
PORT="${VLLM_PORT:-8010}"
API_KEY="${VLLM_API_KEY:-EMPTY}"
apply_generator_runtime_profile "${ENGINE}" "${VLLM_RUNTIME_PROFILE:-}"
TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-2}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-2048}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1}"
GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.82}"
CPU_OFFLOAD_GB="${VLLM_CPU_OFFLOAD_GB:-0}"
DTYPE="${VLLM_DTYPE:-auto}"
KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-}"
QUANTIZATION="${VLLM_QUANTIZATION:-}"
LOAD_FORMAT="${VLLM_LOAD_FORMAT:-}"
TOKENIZER="${VLLM_TOKENIZER:-}"
SAFETENSORS_LOAD_STRATEGY="${VLLM_SAFETENSORS_LOAD_STRATEGY:-}"
MAX_PARALLEL_LOADING_WORKERS="${VLLM_MAX_PARALLEL_LOADING_WORKERS:-}"
SWAP_SPACE="${VLLM_SWAP_SPACE:-}"
MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-}"
KV_CACHE_MEMORY_BYTES="${VLLM_KV_CACHE_MEMORY_BYTES:-}"
DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-0}"
SKIP_MM_PROFILING="${VLLM_SKIP_MM_PROFILING:-0}"
ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-}"
OFFLOAD_BACKEND="${VLLM_OFFLOAD_BACKEND:-}"
CPU_OFFLOAD_PARAMS="${VLLM_CPU_OFFLOAD_PARAMS:-}"
OFFLOAD_PARAMS="${VLLM_OFFLOAD_PARAMS:-}"
OFFLOAD_GROUP_SIZE="${VLLM_OFFLOAD_GROUP_SIZE:-}"
OFFLOAD_NUM_IN_GROUP="${VLLM_OFFLOAD_NUM_IN_GROUP:-}"
OFFLOAD_PREFETCH_STEP="${VLLM_OFFLOAD_PREFETCH_STEP:-}"
ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-}"
MOE_BACKEND="${VLLM_MOE_BACKEND:-}"
LINEAR_BACKEND="${VLLM_LINEAR_BACKEND:-}"
ENABLE_EXPERT_PARALLEL="${VLLM_ENABLE_EXPERT_PARALLEL:-0}"
LANGUAGE_MODEL_ONLY="${VLLM_LANGUAGE_MODEL_ONLY:-}"
REASONING_PARSER="${VLLM_REASONING_PARSER:-}"
TOOL_CALL_PARSER="${VLLM_TOOL_CALL_PARSER:-}"
ENABLE_AUTO_TOOL_CHOICE="${VLLM_ENABLE_AUTO_TOOL_CHOICE:-}"
VLLM_USE_V1_VALUE="${VLLM_USE_V1:-}"
NCCL_P2P_DISABLE_VALUE="${VLLM_NCCL_P2P_DISABLE:-${NCCL_P2P_DISABLE:-1}}"
NCCL_IB_DISABLE_VALUE="${VLLM_NCCL_IB_DISABLE:-${NCCL_IB_DISABLE:-1}}"
SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-${ENGINE}}"
DRY_RUN="${DRY_RUN:-0}"
VLLM_TIMELINE_JSONL="${VLLM_TIMELINE_JSONL:-}"
FLASH_ATTN_SHIM_FILE="${REPO_ROOT}/runtime_shims/flash_attn/ops/triton/rotary.py"
FORCE_CUDA_PLATFORM_PLUGIN="${VLLM_FORCE_CUDA_PLATFORM_PLUGIN:-0}"
FORCE_CUDA_PLATFORM_PLUGIN_NAME="vllm_force_cuda_platform"

vllm_timeline_event() {
  local phase="$1"
  local status="$2"
  local detail="${3:-}"
  local now
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '[%s] vllm_%s %s engine=%s served_model=%s profile=%s port=%s detail=%s\n' "${now}" "${phase}" "${status}" "${ENGINE}" "${SERVED_MODEL_NAME}" "${VLLM_RUNTIME_PROFILE:-}" "${PORT}" "${detail}"
  if [[ -n "${VLLM_TIMELINE_JSONL}" ]]; then
    mkdir -p "$(dirname "${VLLM_TIMELINE_JSONL}")"
    TIMELINE_TIME="${now}" \
    TIMELINE_PHASE="vllm_${phase}" \
    TIMELINE_STATUS="${status}" \
    TIMELINE_ENGINE="${ENGINE}" \
    TIMELINE_MODEL="${MODEL:-}" \
    TIMELINE_SERVED_MODEL="${SERVED_MODEL_NAME}" \
    TIMELINE_PROFILE="${VLLM_RUNTIME_PROFILE:-}" \
    TIMELINE_PORT="${PORT}" \
    TIMELINE_MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
    TIMELINE_DETAIL="${detail}" \
    python3 - <<'PYTIMELINE' >> "${VLLM_TIMELINE_JSONL}"
import json, os
payload = {
    "time": os.environ["TIMELINE_TIME"],
    "phase": os.environ["TIMELINE_PHASE"],
    "status": os.environ["TIMELINE_STATUS"],
    "engine": os.environ["TIMELINE_ENGINE"],
    "model": os.environ.get("TIMELINE_MODEL") or None,
    "served_model": os.environ["TIMELINE_SERVED_MODEL"],
    "runtime_profile": os.environ.get("TIMELINE_PROFILE") or None,
    "port": os.environ["TIMELINE_PORT"],
    "max_model_len": os.environ["TIMELINE_MAX_MODEL_LEN"],
    "detail": os.environ.get("TIMELINE_DETAIL") or None,
}
print(json.dumps(payload, ensure_ascii=False))
PYTIMELINE
  fi
}
FLASH_ATTN_OPS_SHIM="${VLLM_FLASH_ATTN_OPS_SHIM:-}"
if [[ -z "${FLASH_ATTN_OPS_SHIM}" && ( "${ENGINE}" == "llama3_3" || "${ENGINE}" == "llama4" || "${ENGINE}" == "qwen3_6" ) && -f "${FLASH_ATTN_SHIM_FILE}" ]]; then
  FLASH_ATTN_OPS_SHIM="1"
fi
if [[ -z "${FLASH_ATTN_OPS_SHIM}" ]]; then
  FLASH_ATTN_OPS_SHIM="0"
fi

if [[ "${MAX_MODEL_LEN}" -lt 512 ]]; then
  printf "VLLM_MAX_MODEL_LEN must stay >=512; got %s\n" "${MAX_MODEL_LEN}" >&2
  exit 2
fi

MODEL="$(generator_model_for_engine "${ENGINE}")"
command=(
  conda run --no-capture-output -n "${CONDA_ENV}"
  python -m vllm.entrypoints.openai.api_server
  --host "${HOST}"
  --port "${PORT}"
  --api-key "${API_KEY}"
  --model "${MODEL}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --cpu-offload-gb "${CPU_OFFLOAD_GB}"
  --dtype "${DTYPE}"
  --trust-remote-code
)

if [[ -n "${TOKENIZER}" ]]; then
  command+=(--tokenizer "${TOKENIZER}")
fi
if [[ -n "${QUANTIZATION}" ]]; then
  command+=(--quantization "${QUANTIZATION}")
fi
if [[ -n "${LOAD_FORMAT}" ]]; then
  command+=(--load-format "${LOAD_FORMAT}")
fi
if [[ -n "${KV_CACHE_DTYPE}" ]]; then
  command+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF}}"
if [[ "${FLASH_ATTN_OPS_SHIM}" == "1" && ! -f "${FLASH_ATTN_SHIM_FILE}" ]]; then
  printf "warning: disabling VLLM_FLASH_ATTN_OPS_SHIM because shim file is missing: %s\n" "${FLASH_ATTN_SHIM_FILE}" >&2
  FLASH_ATTN_OPS_SHIM="0"
fi
if [[ "${FLASH_ATTN_OPS_SHIM}" == "1" ]]; then
  SHIM_PYTHONPATH="${REPO_ROOT}/runtime_shims"
  export PYTHONPATH="${SHIM_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}"
fi
if [[ "${FORCE_CUDA_PLATFORM_PLUGIN}" == "1" ]]; then
  SHIM_PYTHONPATH="${REPO_ROOT}/runtime_shims"
  case ":${PYTHONPATH:-}:" in
    *":${SHIM_PYTHONPATH}:"*) ;;
    *) export PYTHONPATH="${SHIM_PYTHONPATH}${PYTHONPATH:+:${PYTHONPATH}}" ;;
  esac
  case ",${VLLM_PLUGINS:-}," in
    *",${FORCE_CUDA_PLATFORM_PLUGIN_NAME},"*) ;;
    ",,") export VLLM_PLUGINS="${FORCE_CUDA_PLATFORM_PLUGIN_NAME}" ;;
    *) export VLLM_PLUGINS="${FORCE_CUDA_PLATFORM_PLUGIN_NAME},${VLLM_PLUGINS}" ;;
  esac
fi
vllm_timeline_event "environment" "ready" "flash_attn_ops_shim=${FLASH_ATTN_OPS_SHIM}"
ensure_cuda_compat_library_path "${CONDA_ENV}"
if [[ "${FLASH_ATTN_OPS_SHIM}" == "1" ]]; then
  if ! conda run --no-capture-output -n "${CONDA_ENV}" python - <<'PYFLASHATTN' >/dev/null
from flash_attn.ops.triton.rotary import apply_rotary
PYFLASHATTN
  then
    printf "flash_attn.ops.triton.rotary import failed even with VLLM_FLASH_ATTN_OPS_SHIM=1\n" >&2
    exit 2
  fi
fi
vllm_timeline_event "help_probe" "start" "api_server --help"
VLLM_API_SERVER_HELP="$(conda run --no-capture-output -n "${CONDA_ENV}" python -m vllm.entrypoints.openai.api_server --help 2>&1 || true)"
vllm_timeline_event "help_probe" "finish" "api_server --help"

supports_api_server_flag() {
  local flag="$1"
  grep -Fq -- "${flag}" <<<"${VLLM_API_SERVER_HELP}"
}

append_space_separated_flag_values() {
  local flag="$1"
  local values="$2"
  if [[ -z "${values}" ]]; then
    return 0
  fi
  if supports_api_server_flag "${flag}"; then
    # shellcheck disable=SC2206
    local parts=(${values})
    command+=("${flag}" "${parts[@]}")
  else
    printf "warning: vLLM api_server does not support %s; skipping values=%s\n" "${flag}" "${values}" >&2
  fi
}

if [[ -n "${SAFETENSORS_LOAD_STRATEGY}" ]]; then
  if supports_api_server_flag "--safetensors-load-strategy"; then
    command+=(--safetensors-load-strategy "${SAFETENSORS_LOAD_STRATEGY}")
  else
    printf "warning: vLLM api_server does not support --safetensors-load-strategy; skipping SAFETENSORS_LOAD_STRATEGY=%s\n" "${SAFETENSORS_LOAD_STRATEGY}" >&2
  fi
fi
if [[ -n "${MAX_PARALLEL_LOADING_WORKERS}" ]]; then
  if supports_api_server_flag "--max-parallel-loading-workers"; then
    command+=(--max-parallel-loading-workers "${MAX_PARALLEL_LOADING_WORKERS}")
  else
    printf "warning: vLLM api_server does not support --max-parallel-loading-workers; skipping MAX_PARALLEL_LOADING_WORKERS=%s\n" "${MAX_PARALLEL_LOADING_WORKERS}" >&2
  fi
fi
if [[ -n "${KV_CACHE_MEMORY_BYTES}" ]]; then
  if supports_api_server_flag "--kv-cache-memory-bytes"; then
    command+=(--kv-cache-memory-bytes "${KV_CACHE_MEMORY_BYTES}")
  else
    printf "warning: vLLM api_server does not support --kv-cache-memory-bytes; skipping KV_CACHE_MEMORY_BYTES=%s\n" "${KV_CACHE_MEMORY_BYTES}" >&2
  fi
fi
if [[ -n "${SWAP_SPACE}" ]]; then
  if supports_api_server_flag "--swap-space"; then
    command+=(--swap-space "${SWAP_SPACE}")
  else
    printf "warning: vLLM api_server does not support --swap-space on this host; skipping SWAP_SPACE=%s
" "${SWAP_SPACE}" >&2
  fi
fi
if [[ -n "${MAX_NUM_BATCHED_TOKENS}" ]]; then
  if supports_api_server_flag "--max-num-batched-tokens"; then
    command+=(--max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}")
  else
    printf "warning: vLLM api_server does not support --max-num-batched-tokens; skipping MAX_NUM_BATCHED_TOKENS=%s\n" "${MAX_NUM_BATCHED_TOKENS}" >&2
  fi
fi
if [[ -n "${OFFLOAD_BACKEND}" ]]; then
  if supports_api_server_flag "--offload-backend"; then
    command+=(--offload-backend "${OFFLOAD_BACKEND}")
  else
    printf "warning: vLLM api_server does not support --offload-backend; skipping OFFLOAD_BACKEND=%s\n" "${OFFLOAD_BACKEND}" >&2
  fi
fi
if [[ -n "${ATTENTION_BACKEND}" ]]; then
  if supports_api_server_flag "--attention-backend"; then
    command+=(--attention-backend "${ATTENTION_BACKEND}")
  else
    printf "warning: vLLM api_server does not support --attention-backend; skipping ATTENTION_BACKEND=%s\n" "${ATTENTION_BACKEND}" >&2
  fi
fi
if [[ -n "${OFFLOAD_GROUP_SIZE}" ]]; then
  if supports_api_server_flag "--offload-group-size"; then
    command+=(--offload-group-size "${OFFLOAD_GROUP_SIZE}")
  else
    printf "warning: vLLM api_server does not support --offload-group-size; skipping OFFLOAD_GROUP_SIZE=%s\n" "${OFFLOAD_GROUP_SIZE}" >&2
  fi
fi
if [[ -n "${OFFLOAD_NUM_IN_GROUP}" ]]; then
  if supports_api_server_flag "--offload-num-in-group"; then
    command+=(--offload-num-in-group "${OFFLOAD_NUM_IN_GROUP}")
  else
    printf "warning: vLLM api_server does not support --offload-num-in-group; skipping OFFLOAD_NUM_IN_GROUP=%s\n" "${OFFLOAD_NUM_IN_GROUP}" >&2
  fi
fi
if [[ -n "${OFFLOAD_PREFETCH_STEP}" ]]; then
  if supports_api_server_flag "--offload-prefetch-step"; then
    command+=(--offload-prefetch-step "${OFFLOAD_PREFETCH_STEP}")
  else
    printf "warning: vLLM api_server does not support --offload-prefetch-step; skipping OFFLOAD_PREFETCH_STEP=%s\n" "${OFFLOAD_PREFETCH_STEP}" >&2
  fi
fi
if [[ -n "${REASONING_PARSER}" ]]; then
  if supports_api_server_flag "--reasoning-parser"; then
    command+=(--reasoning-parser "${REASONING_PARSER}")
  else
    printf "warning: vLLM api_server does not support --reasoning-parser; skipping REASONING_PARSER=%s\n" "${REASONING_PARSER}" >&2
  fi
fi
if [[ -n "${TOOL_CALL_PARSER}" ]]; then
  if supports_api_server_flag "--tool-call-parser"; then
    command+=(--tool-call-parser "${TOOL_CALL_PARSER}")
  else
    printf "warning: vLLM api_server does not support --tool-call-parser; skipping TOOL_CALL_PARSER=%s\n" "${TOOL_CALL_PARSER}" >&2
  fi
fi
if [[ -n "${CPU_OFFLOAD_PARAMS}" ]]; then
  append_space_separated_flag_values "--cpu-offload-params" "${CPU_OFFLOAD_PARAMS}"
fi
if [[ -n "${OFFLOAD_PARAMS}" ]]; then
  append_space_separated_flag_values "--offload-params" "${OFFLOAD_PARAMS}"
fi
if [[ -n "${MOE_BACKEND}" ]]; then
  if supports_api_server_flag "--moe-backend"; then
    command+=(--moe-backend "${MOE_BACKEND}")
  else
    printf "warning: vLLM api_server does not support --moe-backend; skipping MOE_BACKEND=%s\n" "${MOE_BACKEND}" >&2
  fi
fi
if [[ -n "${LINEAR_BACKEND}" ]]; then
  if supports_api_server_flag "--linear-backend"; then
    command+=(--linear-backend "${LINEAR_BACKEND}")
  else
    printf "warning: vLLM api_server does not support --linear-backend; skipping LINEAR_BACKEND=%s\n" "${LINEAR_BACKEND}" >&2
  fi
fi
if [[ "${ENABLE_EXPERT_PARALLEL}" == "1" ]]; then
  if supports_api_server_flag "--enable-expert-parallel"; then
    command+=(--enable-expert-parallel)
  else
    printf "warning: vLLM api_server does not support --enable-expert-parallel; skipping VLLM_ENABLE_EXPERT_PARALLEL=1\n" >&2
  fi
fi
if [[ "${ENABLE_AUTO_TOOL_CHOICE}" == "1" ]]; then
  if supports_api_server_flag "--enable-auto-tool-choice"; then
    command+=(--enable-auto-tool-choice)
  else
    printf "warning: vLLM api_server does not support --enable-auto-tool-choice; skipping VLLM_ENABLE_AUTO_TOOL_CHOICE=1\n" >&2
  fi
fi
if [[ "${LANGUAGE_MODEL_ONLY}" == "1" ]]; then
  if supports_api_server_flag "--language-model-only"; then
    command+=(--language-model-only)
  else
    printf "warning: vLLM api_server does not support --language-model-only; skipping VLLM_LANGUAGE_MODEL_ONLY=1\n" >&2
  fi
elif [[ "${LANGUAGE_MODEL_ONLY}" == "0" ]]; then
  if supports_api_server_flag "--no-language-model-only"; then
    command+=(--no-language-model-only)
  fi
fi
if [[ "${DISABLE_CUSTOM_ALL_REDUCE}" == "1" ]]; then
  command+=(--disable-custom-all-reduce)
fi
if [[ "${ENABLE_CHUNKED_PREFILL}" == "0" ]]; then
  if supports_api_server_flag "--no-enable-chunked-prefill"; then
    command+=(--no-enable-chunked-prefill)
  else
    printf "warning: vLLM api_server does not support --no-enable-chunked-prefill; skipping VLLM_ENABLE_CHUNKED_PREFILL=0\n" >&2
  fi
fi
if [[ "${SKIP_MM_PROFILING}" == "1" ]]; then
  if supports_api_server_flag "--skip-mm-profiling"; then
    command+=(--skip-mm-profiling)
  else
    printf "warning: vLLM api_server does not support --skip-mm-profiling; skipping SKIP_MM_PROFILING=1\n" >&2
  fi
fi
if [[ "${VLLM_ENFORCE_EAGER:-1}" == "1" ]]; then
  command+=(--enforce-eager)
fi

cat <<EOF
export VLLM_BASE_URL="http://${HOST}:${PORT}/v1"
export VLLM_API_KEY="${API_KEY}"
export VLLM_SERVED_MODEL_NAME="${SERVED_MODEL_NAME}"
export VLLM_RUNTIME_PROFILE="${VLLM_RUNTIME_PROFILE}"
export VLLM_FLASH_ATTN_OPS_SHIM="${FLASH_ATTN_OPS_SHIM}"
export VLLM_FORCE_CUDA_PLATFORM_PLUGIN="${FORCE_CUDA_PLATFORM_PLUGIN}"
export VLLM_PLUGINS="${VLLM_PLUGINS:-}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-}"
export PYTHONPATH="${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF}"
export VLLM_USE_V1="${VLLM_USE_V1_VALUE}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE_VALUE}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE_VALUE}"
EOF

export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE_VALUE}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE_VALUE}"
if [[ -n "${VLLM_USE_V1_VALUE}" ]]; then
  export VLLM_USE_V1="${VLLM_USE_V1_VALUE}"
fi

printf "command="
printf "%q " "${command[@]}"
printf "\n"

if [[ "${DRY_RUN}" == "1" ]]; then
  vllm_timeline_event "process" "dry_run" "command_not_executed"
  exit 0
fi

vllm_timeline_event "process" "start" "exec_api_server"
exec "${command[@]}"
