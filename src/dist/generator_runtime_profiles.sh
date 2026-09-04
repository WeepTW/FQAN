#!/usr/bin/env bash

# Shared vLLM runtime profiles for generator smoke/probe scripts.
# These are local execution profiles, not thesis-quality claims.

if [[ -z "${WORKSPACE_ROOT:-}" ]]; then
  GENERATOR_WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
else
  GENERATOR_WORKSPACE_ROOT="${WORKSPACE_ROOT}"
fi
MODELS_ROOT="${FQAN_MODELS_ROOT:-${MODELS_ROOT:-${GENERATOR_WORKSPACE_ROOT}/utils/models}}"
export MODELS_ROOT
export HF_HOME="${HF_HOME:-${MODELS_ROOT}/.cache/huggingface}"

generator_model_for_engine() {
  case "$1" in
    deepseek|deepseek_r1_qwen32b|deepseek-r1-qwen32b|deepseek-r1-distill-qwen-32b)
      printf "%s
" "${DEEPSEEK_MODEL_PATH:-${DEEPSEEK_R1_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Qwen-32B}}"
      ;;
    qwen|qwen3_6|qwen3_6_35b_a3b_fp8|qwen3.6-35b-a3b-fp8)
      printf "%s
" "${QWEN3_6_MODEL_PATH:-${QWEN3_6_MODEL:-${QWEN3_6_35B_MODEL:-Qwen/Qwen3.6-35B-A3B-FP8}}}"
      ;;
    mistral|mistral4|mistral_small_4|mistral-small-4)
      printf "%s
" "${MISTRAL_SMALL_MODEL_PATH:-${MISTRAL_SMALL_MODEL:-mistralai/Mistral-Small-4-119B-2603-NVFP4}}"
      ;;
    llama3_3|llama3.3|llama3_3_70b)
      printf "%s
" "${LLAMA3_3_MODEL_PATH:-${LLAMA3_3_MODEL:-meta-llama/Llama-3.3-70B-Instruct}}"
      ;;
    llama4|llama4_scout|llama-4|llama-4-scout-17b-16e-instruct)
      printf "%s
" "${LLAMA4_MODEL_PATH:-${LLAMA4_MODEL:-meta-llama/Llama-4-Scout-17B-16E-Instruct}}"
      ;;
    qwythos|qwythos9b|qwythos-9b)
      printf "%s
" "${QWYTHOS_MODEL_PATH:-${QWYTHOS_MODEL:-empero-ai/Qwythos-9B-Claude-Mythos-5-1M}}"
      ;;
    *)
      printf "Unsupported local Experiment 7 ENGINE=%s; allowed: deepseek, qwen3_6/qwen3_6_35b_a3b_fp8, mistral4/mistral_small_4, llama3_3, llama4, qwythos
" "$1" >&2
      return 2
      ;;
  esac
}

default_generator_runtime_profile() {
  case "$1" in
    deepseek|deepseek_r1_qwen32b|deepseek-r1-qwen32b|deepseek-r1-distill-qwen-32b) printf "deepseek_r1_qwen32b_tp2_32k
" ;;
    qwen|qwen3_6|qwen3_6_35b_a3b_fp8|qwen3.6-35b-a3b-fp8) printf "qwen_fp8_tp2_precise_kv
" ;;
    mistral|mistral4|mistral_small_4|mistral-small-4) printf "mistral_nvfp4_lmo_ep_offload4
" ;;
    llama3_3) printf "llama3_3_bitsandbytes_offload
" ;;
    llama4|llama4_scout|llama-4|llama-4-scout-17b-16e-instruct) printf "llama4_scout_tp2_short_context
" ;;
    qwythos|qwythos9b|qwythos-9b) printf "qwythos9b_tp1_32k
" ;;
    *) printf "auto
" ;;
  esac
}

probe_profiles_for_engine() {
  case "$1" in
    deepseek|deepseek_r1_qwen32b|deepseek-r1-qwen32b|deepseek-r1-distill-qwen-32b)
      printf "%s
" "${DEEPSEEK_VLLM_PROFILES:-deepseek_r1_qwen32b_tp2_32k}"
      ;;
    qwen|qwen3_6|qwen3_6_35b_a3b_fp8|qwen3.6-35b-a3b-fp8)
      printf "%s
" "${QWEN_VLLM_PROFILES:-qwen_fp8_tp2_precise_kv qwen_fp8_low_memory qwen_fp8_tp1_cpu_offload}"
      ;;
    mistral|mistral4|mistral_small_4|mistral-small-4)
      printf "%s
" "${MISTRAL_VLLM_PROFILES:-mistral_nvfp4_lmo_ep_offload4 mistral_nvfp4_lmo_offload8 mistral_nvfp4_lmo_moe_emulation mistral_experts_int8_prefetch mistral_moe_wna16_prefetch}"
      ;;
    llama3_3)
      printf "%s
" "${LLAMA_VLLM_PROFILES:-llama3_3_bitsandbytes_offload llama3_3_fp8_no_cpu_offload llama3_3_prefetch_fp8}"
      ;;
    llama4|llama4_scout|llama-4|llama-4-scout-17b-16e-instruct)
      printf "%s
" "${LLAMA4_VLLM_PROFILES:-llama4_scout_tp2_short_context}"
      ;;
    qwythos|qwythos9b|qwythos-9b)
      printf "%s
" "${QWYTHOS_VLLM_PROFILES:-qwythos9b_tp1_32k}"
      ;;
    *)
      printf "%s
" "$(default_generator_runtime_profile "$1")"
      ;;
  esac
}

reset_generator_vllm_profile_vars() {
  unset VLLM_TENSOR_PARALLEL_SIZE
  unset VLLM_MAX_MODEL_LEN
  unset VLLM_MAX_NUM_SEQS
  unset VLLM_GPU_MEMORY_UTILIZATION
  unset VLLM_KV_CACHE_DTYPE
  unset VLLM_QUANTIZATION
  unset VLLM_LOAD_FORMAT
  unset VLLM_CPU_OFFLOAD_GB
  unset VLLM_SWAP_SPACE
  unset VLLM_MAX_NUM_BATCHED_TOKENS
  unset VLLM_DISABLE_CUSTOM_ALL_REDUCE
  unset VLLM_DTYPE
  unset VLLM_ENFORCE_EAGER
  unset VLLM_SKIP_MM_PROFILING
  unset VLLM_NCCL_P2P_DISABLE
  unset VLLM_NCCL_IB_DISABLE
  unset VLLM_ENABLE_CHUNKED_PREFILL
  unset VLLM_OFFLOAD_BACKEND
  unset VLLM_OFFLOAD_GROUP_SIZE
  unset VLLM_OFFLOAD_NUM_IN_GROUP
  unset VLLM_OFFLOAD_PREFETCH_STEP
  unset VLLM_ATTENTION_BACKEND
  unset VLLM_MOE_BACKEND
  unset VLLM_LINEAR_BACKEND
  unset VLLM_ENABLE_EXPERT_PARALLEL
  unset VLLM_CPU_OFFLOAD_PARAMS
  unset VLLM_OFFLOAD_PARAMS
  unset VLLM_KV_CACHE_MEMORY_BYTES
  unset VLLM_SAFETENSORS_LOAD_STRATEGY
  unset VLLM_MAX_PARALLEL_LOADING_WORKERS
  unset VLLM_LANGUAGE_MODEL_ONLY
  unset VLLM_REASONING_PARSER
  unset VLLM_TOOL_CALL_PARSER
  unset VLLM_ENABLE_AUTO_TOOL_CHOICE
  unset VLLM_USE_FLASHINFER_SAMPLER
  unset VLLM_FLASH_ATTN_OPS_SHIM
  unset VLLM_USE_V1
}

set_profile_var() {
  local name="$1"
  local value="$2"
  export "${name}=${value}"
}

ensure_cuda_compat_library_path() {
  local conda_env="${1:-${CONDA_ENV:-fnqa}}"
  local cuda_lib_dir="${VLLM_CUDA_COMPAT_LIB_DIR:-}"
  if [[ -z "${cuda_lib_dir}" ]]; then
    cuda_lib_dir="$(conda run --no-capture-output -n "${conda_env}" python - <<'PYCUDALIB' 2>/dev/null || true
import sys
from pathlib import Path

root = Path(sys.prefix) / "lib"
for path in sorted(root.glob("python*/site-packages/nvidia/cu13/lib")):
    if (path / "libnvJitLink.so.13").exists():
        print(path)
        break
PYCUDALIB
)"
  fi
  if [[ -d "${cuda_lib_dir}" ]]; then
    case ":${LD_LIBRARY_PATH:-}:" in
      *":${cuda_lib_dir}:"*) ;;
      *) export LD_LIBRARY_PATH="${cuda_lib_dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
    esac
  fi
}

apply_generator_runtime_profile() {
  local engine="$1"
  local profile="${2:-${VLLM_RUNTIME_PROFILE:-}}"
  if [[ -z "${profile}" ]]; then
    profile="$(default_generator_runtime_profile "${engine}")"
  fi

  case "${profile}" in
    deepseek_r1_qwen32b_tp2_32k)
      set_profile_var VLLM_TENSOR_PARALLEL_SIZE "${DEEPSEEK_VLLM_TENSOR_PARALLEL_SIZE:-2}"
      set_profile_var VLLM_MAX_MODEL_LEN "${DEEPSEEK_VLLM_MAX_MODEL_LEN:-32768}"
      set_profile_var VLLM_MAX_NUM_SEQS "${DEEPSEEK_VLLM_MAX_NUM_SEQS:-1}"
      set_profile_var VLLM_GPU_MEMORY_UTILIZATION "${DEEPSEEK_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
      set_profile_var VLLM_MAX_NUM_BATCHED_TOKENS "${DEEPSEEK_VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
      set_profile_var VLLM_CPU_OFFLOAD_GB "${DEEPSEEK_VLLM_CPU_OFFLOAD_GB:-0}"
      set_profile_var VLLM_DTYPE "${DEEPSEEK_VLLM_DTYPE:-auto}"
      set_profile_var VLLM_ENFORCE_EAGER "${DEEPSEEK_VLLM_ENFORCE_EAGER:-1}"
      set_profile_var VLLM_SKIP_MM_PROFILING "${DEEPSEEK_VLLM_SKIP_MM_PROFILING:-1}"
      set_profile_var VLLM_LANGUAGE_MODEL_ONLY "${DEEPSEEK_VLLM_LANGUAGE_MODEL_ONLY:-1}"
      ;;
    qwythos9b_tp1_32k)
      set_profile_var VLLM_TENSOR_PARALLEL_SIZE "${QWYTHOS_VLLM_TENSOR_PARALLEL_SIZE:-1}"
      set_profile_var VLLM_MAX_MODEL_LEN "${QWYTHOS_VLLM_MAX_MODEL_LEN:-32768}"
      set_profile_var VLLM_MAX_NUM_BATCHED_TOKENS "${QWYTHOS_VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
      set_profile_var VLLM_MAX_NUM_SEQS "${QWYTHOS_VLLM_MAX_NUM_SEQS:-8}"
      set_profile_var VLLM_GPU_MEMORY_UTILIZATION "${QWYTHOS_VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
      set_profile_var VLLM_CPU_OFFLOAD_GB "${QWYTHOS_VLLM_CPU_OFFLOAD_GB:-0}"
      set_profile_var VLLM_DTYPE "${QWYTHOS_VLLM_DTYPE:-auto}"
      set_profile_var VLLM_ENFORCE_EAGER "${QWYTHOS_VLLM_ENFORCE_EAGER:-1}"
      set_profile_var VLLM_SKIP_MM_PROFILING "${QWYTHOS_VLLM_SKIP_MM_PROFILING:-1}"
      set_profile_var VLLM_LANGUAGE_MODEL_ONLY "${QWYTHOS_VLLM_LANGUAGE_MODEL_ONLY:-1}"
      ;;
    llama4_scout_tp2_short_context)
      set_profile_var VLLM_TENSOR_PARALLEL_SIZE "${LLAMA4_VLLM_TENSOR_PARALLEL_SIZE:-2}"
      set_profile_var VLLM_MAX_MODEL_LEN "${LLAMA4_VLLM_MAX_MODEL_LEN:-8192}"
      set_profile_var VLLM_MAX_NUM_SEQS "${LLAMA4_VLLM_MAX_NUM_SEQS:-1}"
      set_profile_var VLLM_GPU_MEMORY_UTILIZATION "${LLAMA4_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
      set_profile_var VLLM_MAX_NUM_BATCHED_TOKENS "${LLAMA4_VLLM_MAX_NUM_BATCHED_TOKENS:-4096}"
      set_profile_var VLLM_CPU_OFFLOAD_GB "${LLAMA4_VLLM_CPU_OFFLOAD_GB:-0}"
      set_profile_var VLLM_DTYPE "${LLAMA4_VLLM_DTYPE:-auto}"
      set_profile_var VLLM_ENFORCE_EAGER "${LLAMA4_VLLM_ENFORCE_EAGER:-1}"
      set_profile_var VLLM_SKIP_MM_PROFILING "${LLAMA4_VLLM_SKIP_MM_PROFILING:-1}"
      set_profile_var VLLM_LANGUAGE_MODEL_ONLY "${LLAMA4_VLLM_LANGUAGE_MODEL_ONLY:-1}"
      ;;
    llama4_scout_w4a16_tp2_offload16)
      set_profile_var VLLM_TENSOR_PARALLEL_SIZE "${LLAMA4_W4A16_VLLM_TENSOR_PARALLEL_SIZE:-2}"
      set_profile_var VLLM_MAX_MODEL_LEN "${LLAMA4_W4A16_VLLM_MAX_MODEL_LEN:-4096}"
      set_profile_var VLLM_MAX_NUM_BATCHED_TOKENS "${LLAMA4_W4A16_VLLM_MAX_NUM_BATCHED_TOKENS:-1024}"
      set_profile_var VLLM_MAX_NUM_SEQS "${LLAMA4_W4A16_VLLM_MAX_NUM_SEQS:-1}"
      set_profile_var VLLM_GPU_MEMORY_UTILIZATION "${LLAMA4_W4A16_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
      set_profile_var VLLM_CPU_OFFLOAD_GB "${LLAMA4_W4A16_VLLM_CPU_OFFLOAD_GB:-16}"
      set_profile_var VLLM_DTYPE "${LLAMA4_W4A16_VLLM_DTYPE:-auto}"
      set_profile_var VLLM_ENABLE_CHUNKED_PREFILL "${LLAMA4_W4A16_VLLM_ENABLE_CHUNKED_PREFILL:-1}"
      set_profile_var VLLM_DISABLE_CUSTOM_ALL_REDUCE "${LLAMA4_W4A16_VLLM_DISABLE_CUSTOM_ALL_REDUCE:-1}"
      set_profile_var VLLM_ENFORCE_EAGER "${LLAMA4_W4A16_VLLM_ENFORCE_EAGER:-1}"
      set_profile_var VLLM_SKIP_MM_PROFILING "${LLAMA4_W4A16_VLLM_SKIP_MM_PROFILING:-1}"
      set_profile_var VLLM_LANGUAGE_MODEL_ONLY "${LLAMA4_W4A16_VLLM_LANGUAGE_MODEL_ONLY:-1}"
      ;;
    qwen_fp8_tp2_precise_kv)
      set_profile_var VLLM_TENSOR_PARALLEL_SIZE "${QWEN_VLLM_TENSOR_PARALLEL_SIZE:-2}"
      set_profile_var VLLM_MAX_MODEL_LEN "${QWEN_VLLM_MAX_MODEL_LEN:-8192}"
      set_profile_var VLLM_MAX_NUM_SEQS "${QWEN_VLLM_MAX_NUM_SEQS:-1}"
      set_profile_var VLLM_GPU_MEMORY_UTILIZATION "${QWEN_VLLM_GPU_MEMORY_UTILIZATION:-0.95}"
      set_profile_var VLLM_QUANTIZATION "${QWEN_VLLM_QUANTIZATION:-fp8}"
      set_profile_var VLLM_KV_CACHE_DTYPE "${QWEN_VLLM_KV_CACHE_DTYPE:-}"
      set_profile_var VLLM_CPU_OFFLOAD_GB "${QWEN_VLLM_CPU_OFFLOAD_GB:-0}"
      set_profile_var VLLM_USE_V1 "${QWEN_VLLM_USE_V1:-0}"
      set_profile_var VLLM_SKIP_MM_PROFILING "${QWEN_VLLM_SKIP_MM_PROFILING:-1}"
      set_profile_var VLLM_LANGUAGE_MODEL_ONLY "${QWEN_VLLM_LANGUAGE_MODEL_ONLY:-1}"
      set_profile_var VLLM_REASONING_PARSER "${QWEN_VLLM_REASONING_PARSER:-qwen3}"
      ;;
    qwen_fp8_low_memory)
      set_profile_var VLLM_TENSOR_PARALLEL_SIZE "${QWEN_VLLM_TENSOR_PARALLEL_SIZE:-2}"
      set_profile_var VLLM_MAX_MODEL_LEN "${QWEN_VLLM_MAX_MODEL_LEN:-8192}"
      set_profile_var VLLM_MAX_NUM_SEQS "${QWEN_VLLM_MAX_NUM_SEQS:-1}"
      set_profile_var VLLM_GPU_MEMORY_UTILIZATION "${QWEN_VLLM_GPU_MEMORY_UTILIZATION:-0.95}"
      set_profile_var VLLM_QUANTIZATION "${QWEN_VLLM_QUANTIZATION:-fp8}"
      set_profile_var VLLM_KV_CACHE_DTYPE "${QWEN_VLLM_KV_CACHE_DTYPE:-fp8}"
      set_profile_var VLLM_CPU_OFFLOAD_GB "${QWEN_VLLM_CPU_OFFLOAD_GB:-0}"
      set_profile_var VLLM_USE_V1 "${QWEN_VLLM_USE_V1:-0}"
      set_profile_var VLLM_SKIP_MM_PROFILING "${QWEN_VLLM_SKIP_MM_PROFILING:-1}"
      set_profile_var VLLM_LANGUAGE_MODEL_ONLY "${QWEN_VLLM_LANGUAGE_MODEL_ONLY:-1}"
      set_profile_var VLLM_REASONING_PARSER "${QWEN_VLLM_REASONING_PARSER:-qwen3}"
      ;;
    qwen_fp8_tp1_cpu_offload)
      set_profile_var VLLM_TENSOR_PARALLEL_SIZE "${QWEN_TP1_VLLM_TENSOR_PARALLEL_SIZE:-1}"
      set_profile_var VLLM_MAX_MODEL_LEN "${QWEN_TP1_VLLM_MAX_MODEL_LEN:-2048}"
      set_profile_var VLLM_MAX_NUM_SEQS "${QWEN_TP1_VLLM_MAX_NUM_SEQS:-1}"
      set_profile_var VLLM_GPU_MEMORY_UTILIZATION "${QWEN_TP1_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
      set_profile_var VLLM_QUANTIZATION "${QWEN_TP1_VLLM_QUANTIZATION:-fp8}"
      set_profile_var VLLM_KV_CACHE_DTYPE "${QWEN_TP1_VLLM_KV_CACHE_DTYPE:-fp8}"
      set_profile_var VLLM_CPU_OFFLOAD_GB "${QWEN_TP1_VLLM_CPU_OFFLOAD_GB:-32}"
      set_profile_var VLLM_USE_V1 "${QWEN_TP1_VLLM_USE_V1:-0}"
      set_profile_var VLLM_SKIP_MM_PROFILING "${QWEN_TP1_VLLM_SKIP_MM_PROFILING:-1}"
      set_profile_var VLLM_ENABLE_CHUNKED_PREFILL "${QWEN_TP1_VLLM_ENABLE_CHUNKED_PREFILL:-0}"
      set_profile_var VLLM_MAX_NUM_BATCHED_TOKENS "${QWEN_TP1_VLLM_MAX_NUM_BATCHED_TOKENS:-2048}"
      set_profile_var VLLM_OFFLOAD_BACKEND "${QWEN_TP1_VLLM_OFFLOAD_BACKEND:-}"
      set_profile_var VLLM_OFFLOAD_GROUP_SIZE "${QWEN_TP1_VLLM_OFFLOAD_GROUP_SIZE:-}"
      set_profile_var VLLM_OFFLOAD_NUM_IN_GROUP "${QWEN_TP1_VLLM_OFFLOAD_NUM_IN_GROUP:-}"
      set_profile_var VLLM_OFFLOAD_PREFETCH_STEP "${QWEN_TP1_VLLM_OFFLOAD_PREFETCH_STEP:-}"
      set_profile_var VLLM_LANGUAGE_MODEL_ONLY "${QWEN_TP1_VLLM_LANGUAGE_MODEL_ONLY:-1}"
      set_profile_var VLLM_REASONING_PARSER "${QWEN_TP1_VLLM_REASONING_PARSER:-qwen3}"
      ;;
    mistral_nvfp4_prefetch)
      set_profile_var VLLM_TENSOR_PARALLEL_SIZE "${MISTRAL_VLLM_TENSOR_PARALLEL_SIZE:-2}"
      set_profile_var VLLM_MAX_MODEL_LEN "${MISTRAL_VLLM_MAX_MODEL_LEN:-1024}"
      set_profile_var VLLM_MAX_NUM_SEQS "${MISTRAL_VLLM_MAX_NUM_SEQS:-1}"
      set_profile_var VLLM_GPU_MEMORY_UTILIZATION "${MISTRAL_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
      set_profile_var VLLM_DTYPE "${MISTRAL_VLLM_DTYPE:-auto}"
      set_profile_var VLLM_QUANTIZATION "${MISTRAL_VLLM_QUANTIZATION:-}"
      set_profile_var VLLM_KV_CACHE_DTYPE "${MISTRAL_VLLM_KV_CACHE_DTYPE:-fp8}"
      set_profile_var VLLM_CPU_OFFLOAD_GB "${MISTRAL_VLLM_CPU_OFFLOAD_GB:-0}"
      set_profile_var VLLM_OFFLOAD_BACKEND "${MISTRAL_VLLM_OFFLOAD_BACKEND:-prefetch}"
      set_profile_var VLLM_OFFLOAD_GROUP_SIZE "${MISTRAL_VLLM_OFFLOAD_GROUP_SIZE:-2}"
      set_profile_var VLLM_OFFLOAD_NUM_IN_GROUP "${MISTRAL_VLLM_OFFLOAD_NUM_IN_GROUP:-1}"
      set_profile_var VLLM_OFFLOAD_PREFETCH_STEP "${MISTRAL_VLLM_OFFLOAD_PREFETCH_STEP:-1}"
      set_profile_var VLLM_ATTENTION_BACKEND "${MISTRAL_VLLM_ATTENTION_BACKEND:-TRITON_MLA}"
      ;;
    mistral_nvfp4_lmo_ep_offload4)
      set_profile_var VLLM_TENSOR_PARALLEL_SIZE "${MISTRAL_EP_VLLM_TENSOR_PARALLEL_SIZE:-2}"
      set_profile_var VLLM_MAX_MODEL_LEN "${MISTRAL_EP_VLLM_MAX_MODEL_LEN:-512}"
      set_profile_var VLLM_MAX_NUM_SEQS "${MISTRAL_EP_VLLM_MAX_NUM_SEQS:-1}"
      set_profile_var VLLM_GPU_MEMORY_UTILIZATION "${MISTRAL_EP_VLLM_GPU_MEMORY_UTILIZATION:-0.82}"
      set_profile_var VLLM_DTYPE "${MISTRAL_EP_VLLM_DTYPE:-auto}"
      set_profile_var VLLM_QUANTIZATION "${MISTRAL_EP_VLLM_QUANTIZATION:-}"
      set_profile_var VLLM_KV_CACHE_DTYPE "${MISTRAL_EP_VLLM_KV_CACHE_DTYPE:-fp8}"
      set_profile_var VLLM_CPU_OFFLOAD_GB "${MISTRAL_EP_VLLM_CPU_OFFLOAD_GB:-4}"
      set_profile_var VLLM_OFFLOAD_BACKEND "${MISTRAL_EP_VLLM_OFFLOAD_BACKEND:-prefetch}"
      set_profile_var VLLM_OFFLOAD_GROUP_SIZE "${MISTRAL_EP_VLLM_OFFLOAD_GROUP_SIZE:-2}"
      set_profile_var VLLM_OFFLOAD_NUM_IN_GROUP "${MISTRAL_EP_VLLM_OFFLOAD_NUM_IN_GROUP:-1}"
      set_profile_var VLLM_OFFLOAD_PREFETCH_STEP "${MISTRAL_EP_VLLM_OFFLOAD_PREFETCH_STEP:-1}"
      set_profile_var VLLM_ATTENTION_BACKEND "${MISTRAL_EP_VLLM_ATTENTION_BACKEND:-TRITON_MLA}"
      set_profile_var VLLM_ENABLE_EXPERT_PARALLEL "${MISTRAL_EP_VLLM_ENABLE_EXPERT_PARALLEL:-1}"
      set_profile_var VLLM_MAX_PARALLEL_LOADING_WORKERS "${MISTRAL_EP_VLLM_MAX_PARALLEL_LOADING_WORKERS:-1}"
      set_profile_var VLLM_MAX_NUM_BATCHED_TOKENS "${MISTRAL_EP_VLLM_MAX_NUM_BATCHED_TOKENS:-512}"
      set_profile_var VLLM_LANGUAGE_MODEL_ONLY "${MISTRAL_EP_VLLM_LANGUAGE_MODEL_ONLY:-1}"
      ;;
    mistral_nvfp4_lmo_offload8)
      set_profile_var VLLM_TENSOR_PARALLEL_SIZE "${MISTRAL_OFFLOAD8_VLLM_TENSOR_PARALLEL_SIZE:-2}"
      set_profile_var VLLM_MAX_MODEL_LEN "${MISTRAL_OFFLOAD8_VLLM_MAX_MODEL_LEN:-512}"
      set_profile_var VLLM_MAX_NUM_SEQS "${MISTRAL_OFFLOAD8_VLLM_MAX_NUM_SEQS:-1}"
      set_profile_var VLLM_GPU_MEMORY_UTILIZATION "${MISTRAL_OFFLOAD8_VLLM_GPU_MEMORY_UTILIZATION:-0.78}"
      set_profile_var VLLM_DTYPE "${MISTRAL_OFFLOAD8_VLLM_DTYPE:-auto}"
      set_profile_var VLLM_QUANTIZATION "${MISTRAL_OFFLOAD8_VLLM_QUANTIZATION:-}"
      set_profile_var VLLM_KV_CACHE_DTYPE "${MISTRAL_OFFLOAD8_VLLM_KV_CACHE_DTYPE:-fp8}"
      set_profile_var VLLM_CPU_OFFLOAD_GB "${MISTRAL_OFFLOAD8_VLLM_CPU_OFFLOAD_GB:-8}"
      set_profile_var VLLM_OFFLOAD_BACKEND "${MISTRAL_OFFLOAD8_VLLM_OFFLOAD_BACKEND:-prefetch}"
      set_profile_var VLLM_OFFLOAD_GROUP_SIZE "${MISTRAL_OFFLOAD8_VLLM_OFFLOAD_GROUP_SIZE:-2}"
      set_profile_var VLLM_OFFLOAD_NUM_IN_GROUP "${MISTRAL_OFFLOAD8_VLLM_OFFLOAD_NUM_IN_GROUP:-1}"
      set_profile_var VLLM_OFFLOAD_PREFETCH_STEP "${MISTRAL_OFFLOAD8_VLLM_OFFLOAD_PREFETCH_STEP:-1}"
      set_profile_var VLLM_ATTENTION_BACKEND "${MISTRAL_OFFLOAD8_VLLM_ATTENTION_BACKEND:-TRITON_MLA}"
      set_profile_var VLLM_MAX_PARALLEL_LOADING_WORKERS "${MISTRAL_OFFLOAD8_VLLM_MAX_PARALLEL_LOADING_WORKERS:-1}"
      set_profile_var VLLM_MAX_NUM_BATCHED_TOKENS "${MISTRAL_OFFLOAD8_VLLM_MAX_NUM_BATCHED_TOKENS:-512}"
      set_profile_var VLLM_LANGUAGE_MODEL_ONLY "${MISTRAL_OFFLOAD8_VLLM_LANGUAGE_MODEL_ONLY:-1}"
      ;;
    mistral_nvfp4_lmo_moe_emulation)
      set_profile_var VLLM_TENSOR_PARALLEL_SIZE "${MISTRAL_EMULATION_VLLM_TENSOR_PARALLEL_SIZE:-2}"
      set_profile_var VLLM_MAX_MODEL_LEN "${MISTRAL_EMULATION_VLLM_MAX_MODEL_LEN:-512}"
      set_profile_var VLLM_MAX_NUM_SEQS "${MISTRAL_EMULATION_VLLM_MAX_NUM_SEQS:-1}"
      set_profile_var VLLM_GPU_MEMORY_UTILIZATION "${MISTRAL_EMULATION_VLLM_GPU_MEMORY_UTILIZATION:-0.76}"
      set_profile_var VLLM_DTYPE "${MISTRAL_EMULATION_VLLM_DTYPE:-auto}"
      set_profile_var VLLM_QUANTIZATION "${MISTRAL_EMULATION_VLLM_QUANTIZATION:-}"
      set_profile_var VLLM_KV_CACHE_DTYPE "${MISTRAL_EMULATION_VLLM_KV_CACHE_DTYPE:-fp8}"
      set_profile_var VLLM_CPU_OFFLOAD_GB "${MISTRAL_EMULATION_VLLM_CPU_OFFLOAD_GB:-8}"
      set_profile_var VLLM_OFFLOAD_BACKEND "${MISTRAL_EMULATION_VLLM_OFFLOAD_BACKEND:-prefetch}"
      set_profile_var VLLM_OFFLOAD_GROUP_SIZE "${MISTRAL_EMULATION_VLLM_OFFLOAD_GROUP_SIZE:-2}"
      set_profile_var VLLM_OFFLOAD_NUM_IN_GROUP "${MISTRAL_EMULATION_VLLM_OFFLOAD_NUM_IN_GROUP:-1}"
      set_profile_var VLLM_OFFLOAD_PREFETCH_STEP "${MISTRAL_EMULATION_VLLM_OFFLOAD_PREFETCH_STEP:-1}"
      set_profile_var VLLM_ATTENTION_BACKEND "${MISTRAL_EMULATION_VLLM_ATTENTION_BACKEND:-TRITON_MLA}"
      set_profile_var VLLM_MOE_BACKEND "${MISTRAL_EMULATION_VLLM_MOE_BACKEND:-emulation}"
      set_profile_var VLLM_MAX_PARALLEL_LOADING_WORKERS "${MISTRAL_EMULATION_VLLM_MAX_PARALLEL_LOADING_WORKERS:-1}"
      set_profile_var VLLM_MAX_NUM_BATCHED_TOKENS "${MISTRAL_EMULATION_VLLM_MAX_NUM_BATCHED_TOKENS:-512}"
      set_profile_var VLLM_LANGUAGE_MODEL_ONLY "${MISTRAL_EMULATION_VLLM_LANGUAGE_MODEL_ONLY:-1}"
      ;;
    mistral_nvfp4_memreserve)
      set_profile_var VLLM_TENSOR_PARALLEL_SIZE "${MISTRAL_MEMRESERVE_VLLM_TENSOR_PARALLEL_SIZE:-2}"
      set_profile_var VLLM_MAX_MODEL_LEN "${MISTRAL_MEMRESERVE_VLLM_MAX_MODEL_LEN:-512}"
      set_profile_var VLLM_MAX_NUM_SEQS "${MISTRAL_MEMRESERVE_VLLM_MAX_NUM_SEQS:-1}"
      set_profile_var VLLM_GPU_MEMORY_UTILIZATION "${MISTRAL_MEMRESERVE_VLLM_GPU_MEMORY_UTILIZATION:-0.84}"
      set_profile_var VLLM_DTYPE "${MISTRAL_MEMRESERVE_VLLM_DTYPE:-auto}"
      set_profile_var VLLM_QUANTIZATION "${MISTRAL_MEMRESERVE_VLLM_QUANTIZATION:-}"
      set_profile_var VLLM_KV_CACHE_DTYPE "${MISTRAL_MEMRESERVE_VLLM_KV_CACHE_DTYPE:-fp8}"
      set_profile_var VLLM_CPU_OFFLOAD_GB "${MISTRAL_MEMRESERVE_VLLM_CPU_OFFLOAD_GB:-0}"
      set_profile_var VLLM_OFFLOAD_BACKEND "${MISTRAL_MEMRESERVE_VLLM_OFFLOAD_BACKEND:-prefetch}"
      set_profile_var VLLM_OFFLOAD_GROUP_SIZE "${MISTRAL_MEMRESERVE_VLLM_OFFLOAD_GROUP_SIZE:-2}"
      set_profile_var VLLM_OFFLOAD_NUM_IN_GROUP "${MISTRAL_MEMRESERVE_VLLM_OFFLOAD_NUM_IN_GROUP:-1}"
      set_profile_var VLLM_OFFLOAD_PREFETCH_STEP "${MISTRAL_MEMRESERVE_VLLM_OFFLOAD_PREFETCH_STEP:-1}"
      set_profile_var VLLM_ATTENTION_BACKEND "${MISTRAL_MEMRESERVE_VLLM_ATTENTION_BACKEND:-TRITON_MLA}"
      set_profile_var VLLM_LANGUAGE_MODEL_ONLY "${MISTRAL_MEMRESERVE_VLLM_LANGUAGE_MODEL_ONLY:-1}"
      set_profile_var VLLM_REASONING_PARSER "${MISTRAL_MEMRESERVE_VLLM_REASONING_PARSER:-}"
      set_profile_var VLLM_TOOL_CALL_PARSER "${MISTRAL_MEMRESERVE_VLLM_TOOL_CALL_PARSER:-}"
      set_profile_var VLLM_ENABLE_AUTO_TOOL_CHOICE "${MISTRAL_MEMRESERVE_VLLM_ENABLE_AUTO_TOOL_CHOICE:-0}"
      ;;
    mistral_experts_int8_prefetch)
      set_profile_var VLLM_TENSOR_PARALLEL_SIZE "${MISTRAL_EXPERTS_INT8_VLLM_TENSOR_PARALLEL_SIZE:-2}"
      set_profile_var VLLM_MAX_MODEL_LEN "${MISTRAL_EXPERTS_INT8_VLLM_MAX_MODEL_LEN:-512}"
      set_profile_var VLLM_MAX_NUM_SEQS "${MISTRAL_EXPERTS_INT8_VLLM_MAX_NUM_SEQS:-1}"
      set_profile_var VLLM_GPU_MEMORY_UTILIZATION "${MISTRAL_EXPERTS_INT8_VLLM_GPU_MEMORY_UTILIZATION:-0.84}"
      set_profile_var VLLM_DTYPE "${MISTRAL_EXPERTS_INT8_VLLM_DTYPE:-auto}"
      set_profile_var VLLM_QUANTIZATION "${MISTRAL_EXPERTS_INT8_VLLM_QUANTIZATION:-experts_int8}"
      set_profile_var VLLM_KV_CACHE_DTYPE "${MISTRAL_EXPERTS_INT8_VLLM_KV_CACHE_DTYPE:-fp8}"
      set_profile_var VLLM_CPU_OFFLOAD_GB "${MISTRAL_EXPERTS_INT8_VLLM_CPU_OFFLOAD_GB:-0}"
      set_profile_var VLLM_OFFLOAD_BACKEND "${MISTRAL_EXPERTS_INT8_VLLM_OFFLOAD_BACKEND:-prefetch}"
      set_profile_var VLLM_OFFLOAD_GROUP_SIZE "${MISTRAL_EXPERTS_INT8_VLLM_OFFLOAD_GROUP_SIZE:-2}"
      set_profile_var VLLM_OFFLOAD_NUM_IN_GROUP "${MISTRAL_EXPERTS_INT8_VLLM_OFFLOAD_NUM_IN_GROUP:-1}"
      set_profile_var VLLM_OFFLOAD_PREFETCH_STEP "${MISTRAL_EXPERTS_INT8_VLLM_OFFLOAD_PREFETCH_STEP:-1}"
      set_profile_var VLLM_ATTENTION_BACKEND "${MISTRAL_EXPERTS_INT8_VLLM_ATTENTION_BACKEND:-TRITON_MLA}"
      ;;
    mistral_moe_wna16_prefetch)
      set_profile_var VLLM_TENSOR_PARALLEL_SIZE "${MISTRAL_MOE_WNA16_VLLM_TENSOR_PARALLEL_SIZE:-2}"
      set_profile_var VLLM_MAX_MODEL_LEN "${MISTRAL_MOE_WNA16_VLLM_MAX_MODEL_LEN:-512}"
      set_profile_var VLLM_MAX_NUM_SEQS "${MISTRAL_MOE_WNA16_VLLM_MAX_NUM_SEQS:-1}"
      set_profile_var VLLM_GPU_MEMORY_UTILIZATION "${MISTRAL_MOE_WNA16_VLLM_GPU_MEMORY_UTILIZATION:-0.84}"
      set_profile_var VLLM_DTYPE "${MISTRAL_MOE_WNA16_VLLM_DTYPE:-auto}"
      set_profile_var VLLM_QUANTIZATION "${MISTRAL_MOE_WNA16_VLLM_QUANTIZATION:-moe_wna16}"
      set_profile_var VLLM_KV_CACHE_DTYPE "${MISTRAL_MOE_WNA16_VLLM_KV_CACHE_DTYPE:-fp8}"
      set_profile_var VLLM_CPU_OFFLOAD_GB "${MISTRAL_MOE_WNA16_VLLM_CPU_OFFLOAD_GB:-0}"
      set_profile_var VLLM_OFFLOAD_BACKEND "${MISTRAL_MOE_WNA16_VLLM_OFFLOAD_BACKEND:-prefetch}"
      set_profile_var VLLM_OFFLOAD_GROUP_SIZE "${MISTRAL_MOE_WNA16_VLLM_OFFLOAD_GROUP_SIZE:-2}"
      set_profile_var VLLM_OFFLOAD_NUM_IN_GROUP "${MISTRAL_MOE_WNA16_VLLM_OFFLOAD_NUM_IN_GROUP:-1}"
      set_profile_var VLLM_OFFLOAD_PREFETCH_STEP "${MISTRAL_MOE_WNA16_VLLM_OFFLOAD_PREFETCH_STEP:-1}"
      set_profile_var VLLM_ATTENTION_BACKEND "${MISTRAL_MOE_WNA16_VLLM_ATTENTION_BACKEND:-TRITON_MLA}"
      ;;
    mistral_auto_tp2_short_context)
      set_profile_var VLLM_TENSOR_PARALLEL_SIZE "${MISTRAL_VLLM_TENSOR_PARALLEL_SIZE:-2}"
      set_profile_var VLLM_MAX_MODEL_LEN "${MISTRAL_VLLM_MAX_MODEL_LEN:-1024}"
      set_profile_var VLLM_MAX_NUM_SEQS "${MISTRAL_VLLM_MAX_NUM_SEQS:-1}"
      set_profile_var VLLM_GPU_MEMORY_UTILIZATION "${MISTRAL_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
      set_profile_var VLLM_DTYPE "${MISTRAL_VLLM_DTYPE:-auto}"
      set_profile_var VLLM_CPU_OFFLOAD_GB "${MISTRAL_VLLM_CPU_OFFLOAD_GB:-24}"
      set_profile_var VLLM_USE_V1 "${MISTRAL_VLLM_USE_V1:-0}"
      set_profile_var VLLM_QUANTIZATION "${MISTRAL_VLLM_QUANTIZATION:-}"
      set_profile_var VLLM_KV_CACHE_DTYPE "${MISTRAL_VLLM_KV_CACHE_DTYPE:-}"
      ;;
    mistral_fp8_offload)
      set_profile_var VLLM_TENSOR_PARALLEL_SIZE "${MISTRAL_VLLM_TENSOR_PARALLEL_SIZE:-2}"
      set_profile_var VLLM_MAX_MODEL_LEN "${MISTRAL_VLLM_MAX_MODEL_LEN:-1024}"
      set_profile_var VLLM_MAX_NUM_SEQS "${MISTRAL_VLLM_MAX_NUM_SEQS:-1}"
      set_profile_var VLLM_GPU_MEMORY_UTILIZATION "${MISTRAL_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
      set_profile_var VLLM_DTYPE "${MISTRAL_VLLM_DTYPE:-auto}"
      set_profile_var VLLM_QUANTIZATION "${MISTRAL_VLLM_QUANTIZATION:-fp8}"
      set_profile_var VLLM_KV_CACHE_DTYPE "${MISTRAL_VLLM_KV_CACHE_DTYPE:-fp8}"
      set_profile_var VLLM_CPU_OFFLOAD_GB "${MISTRAL_VLLM_CPU_OFFLOAD_GB:-24}"
      set_profile_var VLLM_USE_V1 "${MISTRAL_VLLM_USE_V1:-0}"
      ;;
    llama3_3_bitsandbytes_offload)
      set_profile_var VLLM_TENSOR_PARALLEL_SIZE "${LLAMA_BNB_VLLM_TENSOR_PARALLEL_SIZE:-2}"
      set_profile_var VLLM_MAX_MODEL_LEN "${LLAMA_BNB_VLLM_MAX_MODEL_LEN:-4096}"
      set_profile_var VLLM_MAX_NUM_SEQS "${LLAMA_BNB_VLLM_MAX_NUM_SEQS:-1}"
      set_profile_var VLLM_GPU_MEMORY_UTILIZATION "${LLAMA_BNB_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
      set_profile_var VLLM_DTYPE "${LLAMA_BNB_VLLM_DTYPE:-auto}"
      set_profile_var VLLM_QUANTIZATION "${LLAMA_BNB_VLLM_QUANTIZATION:-bitsandbytes}"
      set_profile_var VLLM_LOAD_FORMAT "${LLAMA_BNB_VLLM_LOAD_FORMAT:-bitsandbytes}"
      set_profile_var VLLM_KV_CACHE_DTYPE "${LLAMA_BNB_VLLM_KV_CACHE_DTYPE:-fp8}"
      set_profile_var VLLM_CPU_OFFLOAD_GB "${LLAMA_BNB_VLLM_CPU_OFFLOAD_GB:-16}"
      set_profile_var VLLM_SAFETENSORS_LOAD_STRATEGY "${LLAMA_BNB_VLLM_SAFETENSORS_LOAD_STRATEGY:-eager}"
      set_profile_var VLLM_MAX_PARALLEL_LOADING_WORKERS "${LLAMA_BNB_VLLM_MAX_PARALLEL_LOADING_WORKERS:-1}"
      set_profile_var VLLM_MAX_NUM_BATCHED_TOKENS "${LLAMA_BNB_VLLM_MAX_NUM_BATCHED_TOKENS:-4096}"
      set_profile_var VLLM_ENABLE_CHUNKED_PREFILL "${LLAMA_BNB_VLLM_ENABLE_CHUNKED_PREFILL:-1}"
      set_profile_var VLLM_FLASH_ATTN_OPS_SHIM "${LLAMA_BNB_VLLM_FLASH_ATTN_OPS_SHIM:-1}"
      ;;
    llama3_3_awq_prefetch)
      set_profile_var VLLM_TENSOR_PARALLEL_SIZE "${LLAMA_VLLM_TENSOR_PARALLEL_SIZE:-2}"
      set_profile_var VLLM_MAX_MODEL_LEN "${LLAMA_VLLM_MAX_MODEL_LEN:-1024}"
      set_profile_var VLLM_MAX_NUM_SEQS "${LLAMA_VLLM_MAX_NUM_SEQS:-1}"
      set_profile_var VLLM_GPU_MEMORY_UTILIZATION "${LLAMA_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
      set_profile_var VLLM_DTYPE "${LLAMA_VLLM_DTYPE:-float16}"
      set_profile_var VLLM_QUANTIZATION "${LLAMA_VLLM_QUANTIZATION:-}"
      set_profile_var VLLM_KV_CACHE_DTYPE "${LLAMA_VLLM_KV_CACHE_DTYPE:-fp8}"
      set_profile_var VLLM_CPU_OFFLOAD_GB "${LLAMA_VLLM_CPU_OFFLOAD_GB:-0}"
      set_profile_var VLLM_OFFLOAD_BACKEND "${LLAMA_VLLM_OFFLOAD_BACKEND:-prefetch}"
      set_profile_var VLLM_OFFLOAD_GROUP_SIZE "${LLAMA_VLLM_OFFLOAD_GROUP_SIZE:-8}"
      set_profile_var VLLM_OFFLOAD_NUM_IN_GROUP "${LLAMA_VLLM_OFFLOAD_NUM_IN_GROUP:-1}"
      set_profile_var VLLM_OFFLOAD_PREFETCH_STEP "${LLAMA_VLLM_OFFLOAD_PREFETCH_STEP:-1}"
      set_profile_var VLLM_FLASH_ATTN_OPS_SHIM "${LLAMA_VLLM_FLASH_ATTN_OPS_SHIM:-1}"
      ;;
    llama3_3_fp8_no_cpu_offload)
      set_profile_var VLLM_TENSOR_PARALLEL_SIZE "${LLAMA_NO_OFFLOAD_VLLM_TENSOR_PARALLEL_SIZE:-2}"
      set_profile_var VLLM_MAX_MODEL_LEN "${LLAMA_NO_OFFLOAD_VLLM_MAX_MODEL_LEN:-512}"
      set_profile_var VLLM_MAX_NUM_SEQS "${LLAMA_NO_OFFLOAD_VLLM_MAX_NUM_SEQS:-1}"
      set_profile_var VLLM_GPU_MEMORY_UTILIZATION "${LLAMA_NO_OFFLOAD_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
      set_profile_var VLLM_DTYPE "${LLAMA_NO_OFFLOAD_VLLM_DTYPE:-auto}"
      set_profile_var VLLM_QUANTIZATION "${LLAMA_NO_OFFLOAD_VLLM_QUANTIZATION:-fp8}"
      set_profile_var VLLM_KV_CACHE_DTYPE "${LLAMA_NO_OFFLOAD_VLLM_KV_CACHE_DTYPE:-fp8}"
      set_profile_var VLLM_CPU_OFFLOAD_GB "${LLAMA_NO_OFFLOAD_VLLM_CPU_OFFLOAD_GB:-0}"
      set_profile_var VLLM_ENABLE_CHUNKED_PREFILL "${LLAMA_NO_OFFLOAD_VLLM_ENABLE_CHUNKED_PREFILL:-1}"
      set_profile_var VLLM_FLASH_ATTN_OPS_SHIM "${LLAMA_NO_OFFLOAD_VLLM_FLASH_ATTN_OPS_SHIM:-1}"
      ;;
    llama3_3_prefetch_fp8)
      set_profile_var VLLM_TENSOR_PARALLEL_SIZE "${LLAMA_PREFETCH_FP8_VLLM_TENSOR_PARALLEL_SIZE:-2}"
      set_profile_var VLLM_MAX_MODEL_LEN "${LLAMA_PREFETCH_FP8_VLLM_MAX_MODEL_LEN:-512}"
      set_profile_var VLLM_MAX_NUM_SEQS "${LLAMA_PREFETCH_FP8_VLLM_MAX_NUM_SEQS:-1}"
      set_profile_var VLLM_GPU_MEMORY_UTILIZATION "${LLAMA_PREFETCH_FP8_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
      set_profile_var VLLM_DTYPE "${LLAMA_PREFETCH_FP8_VLLM_DTYPE:-auto}"
      set_profile_var VLLM_QUANTIZATION "${LLAMA_PREFETCH_FP8_VLLM_QUANTIZATION:-fp8}"
      set_profile_var VLLM_KV_CACHE_DTYPE "${LLAMA_PREFETCH_FP8_VLLM_KV_CACHE_DTYPE:-fp8}"
      set_profile_var VLLM_CPU_OFFLOAD_GB "${LLAMA_PREFETCH_FP8_VLLM_CPU_OFFLOAD_GB:-0}"
      set_profile_var VLLM_OFFLOAD_BACKEND "${LLAMA_PREFETCH_FP8_VLLM_OFFLOAD_BACKEND:-prefetch}"
      set_profile_var VLLM_OFFLOAD_GROUP_SIZE "${LLAMA_PREFETCH_FP8_VLLM_OFFLOAD_GROUP_SIZE:-8}"
      set_profile_var VLLM_OFFLOAD_NUM_IN_GROUP "${LLAMA_PREFETCH_FP8_VLLM_OFFLOAD_NUM_IN_GROUP:-1}"
      set_profile_var VLLM_OFFLOAD_PREFETCH_STEP "${LLAMA_PREFETCH_FP8_VLLM_OFFLOAD_PREFETCH_STEP:-1}"
      set_profile_var VLLM_ENABLE_CHUNKED_PREFILL "${LLAMA_PREFETCH_FP8_VLLM_ENABLE_CHUNKED_PREFILL:-1}"
      set_profile_var VLLM_FLASH_ATTN_OPS_SHIM "${LLAMA_PREFETCH_FP8_VLLM_FLASH_ATTN_OPS_SHIM:-1}"
      ;;
    llama3_3_fp8_offload)
      set_profile_var VLLM_TENSOR_PARALLEL_SIZE "${LLAMA_VLLM_TENSOR_PARALLEL_SIZE:-2}"
      set_profile_var VLLM_MAX_MODEL_LEN "${LLAMA_VLLM_MAX_MODEL_LEN:-2048}"
      set_profile_var VLLM_MAX_NUM_SEQS "${LLAMA_VLLM_MAX_NUM_SEQS:-1}"
      set_profile_var VLLM_GPU_MEMORY_UTILIZATION "${LLAMA_VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
      set_profile_var VLLM_DTYPE "${LLAMA_VLLM_DTYPE:-auto}"
      set_profile_var VLLM_QUANTIZATION "${LLAMA_VLLM_QUANTIZATION:-fp8}"
      set_profile_var VLLM_KV_CACHE_DTYPE "${LLAMA_VLLM_KV_CACHE_DTYPE:-fp8}"
      set_profile_var VLLM_CPU_OFFLOAD_GB "${LLAMA_VLLM_CPU_OFFLOAD_GB:-8}"
      set_profile_var VLLM_USE_V1 "${LLAMA_VLLM_USE_V1:-0}"
      set_profile_var VLLM_FLASH_ATTN_OPS_SHIM "${LLAMA_VLLM_FLASH_ATTN_OPS_SHIM:-1}"
      ;;
    auto)
      ;;
    *)
      printf "Unsupported VLLM_RUNTIME_PROFILE=%s for ENGINE=%s\n" "${profile}" "${engine}" >&2
      return 2
      ;;
  esac

  set_profile_var VLLM_USE_FLASHINFER_SAMPLER "${GENERATOR_VLLM_USE_FLASHINFER_SAMPLER:-0}"
  set_profile_var VLLM_SWAP_SPACE "${VLLM_SWAP_SPACE:-16}"
  set_profile_var VLLM_MAX_NUM_BATCHED_TOKENS "${VLLM_MAX_NUM_BATCHED_TOKENS:-512}"
  set_profile_var VLLM_DISABLE_CUSTOM_ALL_REDUCE "${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-1}"
  set_profile_var VLLM_DTYPE "${VLLM_DTYPE:-auto}"
  set_profile_var VLLM_ENFORCE_EAGER "${VLLM_ENFORCE_EAGER:-1}"
  set_profile_var VLLM_SKIP_MM_PROFILING "${VLLM_SKIP_MM_PROFILING:-0}"
  set_profile_var VLLM_NCCL_P2P_DISABLE "${VLLM_NCCL_P2P_DISABLE:-1}"
  set_profile_var VLLM_NCCL_IB_DISABLE "${VLLM_NCCL_IB_DISABLE:-1}"
  set_profile_var VLLM_RUNTIME_PROFILE "${profile}"
  export GENERATOR_RUNTIME_PROFILE_RESOLVED="${profile}"
}

generator_runtime_precision_policy() {
  case "$1" in
    deepseek_r1_qwen32b_tp2_32k)
      printf "DeepSeek R1 Distill Qwen 32B vLLM profile with TP2, 32k context, eager execution, and Models-root HF cache.
"
      ;;
    qwythos9b_tp1_32k)
      printf "Qwythos 9B vLLM profile with TP1, 32k context, 8192 batched tokens, 8 sequences, and 0.85 GPU memory utilization.
"
      ;;
    llama4_scout_tp2_short_context)
      printf "Llama 4 Scout vLLM feasibility profile with TP2 and short context; do not use the advertised 10M context for FinQA smoke.
"
      ;;
    llama4_scout_w4a16_tp2_offload16)
      printf "Llama 4 Scout RedHatAI W4A16 compressed-tensors feasibility profile with TP2, 4k context, and 16 GiB CPU offload per GPU; bounded smoke only until validated.
"
      ;;
    qwen_fp8_tp2_precise_kv)
      printf "Qwen FP8 checkpoint for vLLM 0.22.1 with TP2, short context, language-model-only, and qwen3 reasoning parser.\n"
      ;;
    qwen_fp8_low_memory)
      printf "Qwen FP8 checkpoint with FP8 KV cache for local memory pressure.\n"
      ;;
    qwen_fp8_tp1_cpu_offload)
      printf "Qwen FP8 checkpoint single-GPU CPU-offload fallback to avoid local NCCL tensor-parallel startup failures; local smoke only.\n"
      ;;
    mistral_nvfp4_lmo_ep_offload4)
      printf "Mistral Small 4 NVFP4 exact probe with text-only mode, TP2, expert parallel, 4GiB CPU offload, FP8 KV cache, and short context.\n"
      ;;
    mistral_nvfp4_lmo_offload8)
      printf "Mistral Small 4 NVFP4 exact probe with text-only mode, TP2, 8GiB CPU offload, lower GPU utilization, FP8 KV cache, and short context.\n"
      ;;
    mistral_nvfp4_lmo_moe_emulation)
      printf "Mistral Small 4 NVFP4 exact probe with text-only mode, TP2, 8GiB CPU offload, and emulated MoE backend to avoid Marlin scale-conversion OOM.\n"
      ;;
    mistral_nvfp4_memreserve)
      printf "Mistral Small 4 NVFP4 mem-reserve probe: TP2, short context, FP8 KV cache, prefetch offload, no UVA CPU offload.\n"
      ;;
    mistral_experts_int8_prefetch)
      printf "Mistral Small 4 experts_int8 quantization probe with TP2, short context, FP8 KV cache, and prefetch offload.\n"
      ;;
    mistral_moe_wna16_prefetch)
      printf "Mistral Small 4 moe_wna16 quantization probe with TP2, short context, FP8 KV cache, and prefetch offload.\n"
      ;;
    mistral_auto_tp2_short_context)
      printf "Mistral Small 4 legacy auto/model-default short-context probe; not in the default Experiment 7 local profile sweep.\n"
      ;;
    mistral_fp8_offload)
      printf "Mistral Small 4 legacy FP8/UVA offload local fallback; not in the default Experiment 7 local profile sweep.\n"
      ;;
    llama3_3_bitsandbytes_offload)
      printf "Llama 3.3 official checkpoint bitsandbytes feasibility profile with TP2, short context, FP8 KV cache, and CPU offload.\n"
      ;;
    llama3_3_awq_prefetch)
      printf "Llama 3.3 official checkpoint with TP2 and prefetch offload.\n"
      ;;
    llama3_3_fp8_no_cpu_offload)
      printf "Llama 3.3 official checkpoint online FP8 probe with TP2, FP8 KV cache, and CPU offload disabled to avoid UVA meta-tensor failures.\n"
      ;;
    llama3_3_prefetch_fp8)
      printf "Llama 3.3 official checkpoint online FP8 prefetch probe with TP2 and CPU offload disabled; local feasibility profile only.\n"
      ;;
    llama3_3_fp8_offload)
      printf "Llama 3.3 legacy online FP8/UVA offload local feasibility profile; not in the default Experiment 7 local profile sweep.\n"
      ;;
    *)
      printf "auto vLLM precision profile.\n"
      ;;
  esac
}
