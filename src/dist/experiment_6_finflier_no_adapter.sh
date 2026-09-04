#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
EXPECTED_ENV="fnqa"

PYTHON_BIN="${EXPERIMENT6_PYTHON:-}"
if [[ -z "${PYTHON_BIN}" && "${CONDA_DEFAULT_ENV:-}" == "${EXPECTED_ENV}" ]]; then
  PYTHON_BIN="$(command -v python)"
fi
if [[ -z "${PYTHON_BIN}" ]] && command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base)"
  PYTHON_BIN="${CONDA_BASE}/envs/${EXPECTED_ENV}/bin/python"
fi
if [[ -z "${PYTHON_BIN}" || ! -x "${PYTHON_BIN}" ]]; then
  echo "required conda environment is unavailable: ${EXPECTED_ENV}" >&2
  exit 2
fi

cd "${REPO_ROOT}"
exec "${PYTHON_BIN}" -B "${SCRIPT_DIR}/experiment6_finflier.py" "$@"
