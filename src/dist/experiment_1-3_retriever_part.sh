#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROMPT_MODES="${PROMPT_MODES:-original}"

exec "${SCRIPT_DIR}/experiment_1_mistral_retriever.sh" "$@"
