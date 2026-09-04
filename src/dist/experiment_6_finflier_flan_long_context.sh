#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONDA_ENV="${CONDA_ENV:-fnqa}"

if ! command -v conda >/dev/null 2>&1; then
  printf 'Required command is unavailable: conda\n' >&2
  exit 2
fi

exec conda run --no-capture-output -n "${CONDA_ENV}" python -B \
  "${SCRIPT_DIR}/experiment6_finflier_flan_long_context.py" "$@"
