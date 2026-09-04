#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Conservative local feasibility default: Qwen35 only.
# This no-API route is not a formal FINDER reproduction unless PromptPG policy
# output is supplied and the run metadata marks formal_finder_ready=true.
# Callers may override ENGINES / EXPERIMENT7_MATRIX / LIMIT for wider runs.
export EXPT_ID="${EXPT_ID:-experiment_7_generator_no_api_key_$(date -u +%Y%m%dT%H%M%SZ)}"
export ENGINES="${ENGINES:-qwen3_6}"
export EXPERIMENT7_MATRIX="${EXPERIMENT7_MATRIX:-finqa_flan_o:finqa_test}"
export RUN_EXECUTE="${RUN_EXECUTE:-auto}"
export RUN_RETRIEVER_INFER="${RUN_RETRIEVER_INFER:-0}"
export STRICT_INPUTS="${STRICT_INPUTS:-1}"
export LIMIT="${LIMIT:-1}"
export MAX_TOKENS="${NO_API_MAX_TOKENS:-128}"
export SHOW_PROMPT="${SHOW_PROMPT:-0}"
export RESTART_LLM="${RESTART_LLM:-1}"
export CLEANUP_LLM_AFTER_ENGINE="${CLEANUP_LLM_AFTER_ENGINE:-1}"
export FAIL_FAST_ON_CHILD_ERROR="${FAIL_FAST_ON_CHILD_ERROR:-1}"
export LLM_SERVICE_MODE="${LLM_SERVICE_MODE:-auto}"
export VLLM_READY_TIMEOUT_SECONDS="${VLLM_READY_TIMEOUT_SECONDS:-900}"
export EXAMPLE_SELECTION_MODE="${EXAMPLE_SELECTION_MODE:-cache}"
export EXAMPLE_SELECTION_REQUIRE_CACHE="${EXAMPLE_SELECTION_REQUIRE_CACHE:-1}"
export FORMAL_FINDER_READY="${FORMAL_FINDER_READY:-1}"
export EXPERIMENT7_SELECTION_EXPT_ID="${EXPERIMENT7_SELECTION_EXPT_ID:-${IN_CONTEXT_SELECTION_EXPT_ID:-}}"
export EXPERIMENT7_SELECTION_ENGINE="${EXPERIMENT7_SELECTION_ENGINE:-gpt5_5}"
case "${EXPERIMENT7_SELECTION_ENGINE}" in
  gpt55|gpt-5.5)
    export EXPERIMENT7_SELECTION_ENGINE="gpt5_5"
    ;;
esac

for engine in ${ENGINES}; do
  case "${engine}" in
    qwen|qwen3_6|qwen3_6_35b_a3b_fp8|mistral|mistral4|mistral_small_4|llama|llama3_3|gpt55|gpt5_5|gpt-5.5|gpt5_3_codex|gpt5_3_codexS) ;;
    *)
      printf '%s\n' "generator_no_api_key.sh: unsupported no-API engine '${engine}'. Allowed aliases include qwen3_6/qwen3_6_35b_a3b_fp8, mistral4/mistral_small_4, llama3_3, gpt5_5, or gpt5_3_codexS/gpt5_3_codex via ChatMock." >&2
      exit 2
      ;;
  esac
done

exec "${SCRIPT_DIR}/experiment_7_generator_answer.sh" "$@"
