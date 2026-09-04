#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

SGLANG_HOST="${SGLANG_HOST:-localhost}"
SGLANG_PORT="${SGLANG_PORT:-8013}"
SGLANG_MODEL_PATH="${SGLANG_MODEL_PATH:?SGLANG_MODEL_PATH is required}"
SGLANG_SERVED_MODEL_NAME="${SGLANG_SERVED_MODEL_NAME:-mistral4}"
SGLANG_FORMAL_MODEL="${SGLANG_FORMAL_MODEL:-${SGLANG_MODEL_PATH}}"
SGLANG_RUNTIME_PROFILE="${SGLANG_RUNTIME_PROFILE:-sglang}"
SGLANG_QUANTIZATION="${SGLANG_QUANTIZATION:-}"
SGLANG_LOAD_FORMAT="${SGLANG_LOAD_FORMAT:-auto}"
SGLANG_TP="${SGLANG_TP:-2}"
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-4096}"
SGLANG_MEM_FRACTION_STATIC="${SGLANG_MEM_FRACTION_STATIC:-0.82}"
SGLANG_CHUNKED_PREFILL_SIZE="${SGLANG_CHUNKED_PREFILL_SIZE:-2048}"
SGLANG_MAX_RUNNING_REQUESTS="${SGLANG_MAX_RUNNING_REQUESTS:-1}"
SGLANG_DTYPE="${SGLANG_DTYPE:-auto}"
SGLANG_KV_CACHE_DTYPE="${SGLANG_KV_CACHE_DTYPE:-auto}"
SGLANG_ATTENTION_BACKEND="${SGLANG_ATTENTION_BACKEND:-}"
SGLANG_SAMPLING_BACKEND="${SGLANG_SAMPLING_BACKEND:-}"
SGLANG_CUDA_VISIBLE_DEVICES="${SGLANG_CUDA_VISIBLE_DEVICES:-}"

if [[ -n "${SGLANG_CUDA_VISIBLE_DEVICES}" ]]; then
  export CUDA_VISIBLE_DEVICES="${SGLANG_CUDA_VISIBLE_DEVICES}"
fi
SGLANG_CUDA_LIB_DIR="${SGLANG_CUDA_LIB_DIR:-${CONDA_PREFIX:-}/lib/python3.10/site-packages/nvidia/cu13/lib}"
if [[ -d "${SGLANG_CUDA_LIB_DIR}" ]]; then
  export LD_LIBRARY_PATH="${SGLANG_CUDA_LIB_DIR}:${LD_LIBRARY_PATH:-}"
fi

args=(
  -m sglang.launch_server
  --model-path "${SGLANG_MODEL_PATH}"
  --host "${SGLANG_HOST}"
  --port "${SGLANG_PORT}"
  --served-model-name "${SGLANG_SERVED_MODEL_NAME}"
  --tensor-parallel-size "${SGLANG_TP}"
  --context-length "${SGLANG_CONTEXT_LENGTH}"
  --mem-fraction-static "${SGLANG_MEM_FRACTION_STATIC}"
  --chunked-prefill-size "${SGLANG_CHUNKED_PREFILL_SIZE}"
  --max-running-requests "${SGLANG_MAX_RUNNING_REQUESTS}"
  --dtype "${SGLANG_DTYPE}"
  --kv-cache-dtype "${SGLANG_KV_CACHE_DTYPE}"
)

if [[ -n "${SGLANG_QUANTIZATION}" ]]; then
  args+=(--quantization "${SGLANG_QUANTIZATION}")
fi
if [[ -n "${SGLANG_LOAD_FORMAT}" && "${SGLANG_LOAD_FORMAT}" != "auto" ]]; then
  args+=(--load-format "${SGLANG_LOAD_FORMAT}")
fi
if [[ -n "${SGLANG_ATTENTION_BACKEND}" ]]; then
  args+=(--attention-backend "${SGLANG_ATTENTION_BACKEND}")
fi
if [[ -n "${SGLANG_SAMPLING_BACKEND}" ]]; then
  args+=(--sampling-backend "${SGLANG_SAMPLING_BACKEND}")
fi

printf "backend=local_sglang_openai_compatible
"
printf "formal_model=%s
" "${SGLANG_FORMAL_MODEL}"
printf "actual_model=%s
" "${SGLANG_MODEL_PATH}"
printf "served_model_name=%s
" "${SGLANG_SERVED_MODEL_NAME}"
printf "runtime_profile=%s
" "${SGLANG_RUNTIME_PROFILE}"
printf "quantization=%s
" "${SGLANG_QUANTIZATION}"
printf "load_format=%s
" "${SGLANG_LOAD_FORMAT}"
printf "tp=%s
" "${SGLANG_TP}"
printf "context_length=%s
" "${SGLANG_CONTEXT_LENGTH}"
printf "mem_fraction_static=%s
" "${SGLANG_MEM_FRACTION_STATIC}"
printf "chunked_prefill_size=%s
" "${SGLANG_CHUNKED_PREFILL_SIZE}"
printf "max_running_requests=%s
" "${SGLANG_MAX_RUNNING_REQUESTS}"
printf "cuda_visible_devices=%s
" "${CUDA_VISIBLE_DEVICES:-}"
printf "command=python %q" "${args[0]}"
printf " %q" "${args[@]:1}"
printf "
"

exec python "${args[@]}"
