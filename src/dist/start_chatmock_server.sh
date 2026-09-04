#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/retriever_experiment_lib.sh"

CHATMOCK_ROOT="${CHATMOCK_ROOT:-${REPO_ROOT}/.external/ChatMock}"
HOST="${CHATMOCK_HOST:-localhost}"
PORT="${CHATMOCK_PORT:-8000}"
REASONING_EFFORT="${CHATMOCK_REASONING_EFFORT:-medium}"
REASONING_SUMMARY="${CHATMOCK_REASONING_SUMMARY:-none}"
DRY_RUN="${DRY_RUN:-0}"

if [[ -f "${CHATMOCK_ROOT}/chatmock.py" ]]; then
  command=(
    conda run --no-capture-output -n "${CONDA_ENV}"
    python "${CHATMOCK_ROOT}/chatmock.py"
    serve
    --host "${HOST}"
    --port "${PORT}"
    --reasoning-effort "${REASONING_EFFORT}"
    --reasoning-summary "${REASONING_SUMMARY}"
  )
elif command -v chatmock >/dev/null 2>&1; then
  command=(
    chatmock
    serve
    --host "${HOST}"
    --port "${PORT}"
    --reasoning-effort "${REASONING_EFFORT}"
    --reasoning-summary "${REASONING_SUMMARY}"
  )
else
  cat >&2 <<EOF
ChatMock is not installed at ${CHATMOCK_ROOT}, and no chatmock CLI was found on PATH.
Install or clone RayBytes/ChatMock at CHATMOCK_ROOT, or install the chatmock CLI in the active PATH.
After it is available, rerun this script or point CHATMOCK_BASE_URL at an existing ChatMock OpenAI-compatible server.
EOF
  exit 2
fi

cat <<EOF
export CHATMOCK_BASE_URL="http://${HOST}:${PORT}/v1"
export CHATMOCK_API_KEY="${CHATMOCK_API_KEY:-key}"
EOF

printf "command="
printf "%q " "${command[@]}"
printf "\n"

if [[ "${DRY_RUN}" == "1" ]]; then
  exit 0
fi

exec "${command[@]}"
