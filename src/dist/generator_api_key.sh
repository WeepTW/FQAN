#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Formal API-key route.  GPT-4.1 execute must use official OpenAI or Azure
# credentials unless ALLOW_OPENAI_COMPATIBLE_EXECUTE=1 is explicitly set.
export EXPT_ID="${EXPT_ID:-experiment_7_generator_api_key_$(date -u +%Y%m%dT%H%M%SZ)}"
export ENGINES="${ENGINES:-gpt4_1}"
export EXPERIMENT7_MATRIX="${EXPERIMENT7_MATRIX:-finqa_flan_o:finqa_test}"
export RUN_EXECUTE="${RUN_EXECUTE:-auto}"
export RUN_RETRIEVER_INFER="${RUN_RETRIEVER_INFER:-0}"
export STRICT_INPUTS="${STRICT_INPUTS:-1}"
export LIMIT="${LIMIT:-1}"
export SHOW_PROMPT="${SHOW_PROMPT:-1}"
export MAX_TOKENS="${MAX_TOKENS:-128}"
export GPT5_3_CODEX_ROUTE="${GPT5_3_CODEX_ROUTE:-api_key}"
export EXAMPLE_SELECTION_MODE="${EXAMPLE_SELECTION_MODE:-cache}"
export EXAMPLE_SELECTION_SHOT_NUMBER="${EXAMPLE_SELECTION_SHOT_NUMBER:-4}"
export EXAMPLE_SELECTION_REQUIRE_POLICY="${EXAMPLE_SELECTION_REQUIRE_POLICY:-0}"
export EXAMPLE_SELECTION_REQUIRE_CACHE="${EXAMPLE_SELECTION_REQUIRE_CACHE:-1}"
export FORMAL_FINDER_READY="${FORMAL_FINDER_READY:-1}"
export EXPERIMENT7_SELECTION_EXPT_ID="${EXPERIMENT7_SELECTION_EXPT_ID:-${IN_CONTEXT_SELECTION_EXPT_ID:-}}"
export EXPERIMENT7_SELECTION_ENGINE="${EXPERIMENT7_SELECTION_ENGINE:-gpt5_5}"
case "${EXPERIMENT7_SELECTION_ENGINE}" in
  gpt55|gpt-5.5)
    export EXPERIMENT7_SELECTION_ENGINE="gpt5_5"
    ;;
esac

formal_blockers=()
if [[ "${EXAMPLE_SELECTION_MODE}" == "cache" && -z "${EXPERIMENT7_SELECTION_EXPT_ID:-}" && -z "${EXAMPLE_SELECTION_CACHE_JSON:-}" && -z "${EXAMPLE_SELECTION_CACHE_ROOT:-}" ]]; then
  formal_blockers+=("selection_cache_missing")
fi

case "${RUN_EXECUTE}" in
  auto|1|true|True|yes|Yes)
    if [[ -n "${OPENAI_BASE_URL:-}" && "${ALLOW_OPENAI_COMPATIBLE_EXECUTE:-0}" != "1" ]]; then
      printf '%s\n' "generator_api_key.sh: OPENAI_BASE_URL is ignored for formal execute unless ALLOW_OPENAI_COMPATIBLE_EXECUTE=1." >&2
    fi
    if [[ -z "${OPENAI_API_KEY:-}" && -z "${AZURE_OPENAI_API_KEY:-}" ]]; then
      printf '%s\n' "generator_api_key.sh: no official OpenAI/Azure API key is exported; run should report credential_blocked." >&2
    fi
    ;;
esac

if (( ${#formal_blockers[@]} )); then
  printf 'generator_api_key.sh: formal preflight blockers: %s
' "${formal_blockers[*]}" >&2
fi

exec "${SCRIPT_DIR}/experiment_7_generator_answer.sh" "$@"
