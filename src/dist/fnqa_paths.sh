#!/usr/bin/env bash
# shellcheck shell=bash
_FQAN_PATH_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
_FQAN_PATH_PYTHON="${FQAN_PATH_PYTHON:-python3}"
if [[ ! -x "$(command -v "${_FQAN_PATH_PYTHON}" 2>/dev/null || true)" ]]; then
  printf 'FQAN path resolver requires Python: %s\n' "${_FQAN_PATH_PYTHON}" >&2
  return 2 2>/dev/null || exit 2
fi
eval "$("${_FQAN_PATH_PYTHON}" -B "${_FQAN_PATH_SCRIPT_DIR}/experiment6_paths.py" --format shell)"
unset _FQAN_PATH_SCRIPT_DIR _FQAN_PATH_PYTHON
