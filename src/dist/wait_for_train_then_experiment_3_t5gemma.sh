#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

TMUX_SESSION="${TMUX_SESSION:-train}"
# 2026-05-18 estimate:
# - Active train session is running FLAN zero-shot at about 77% with ~36 min left.
# - experiment_2_flan_retriever.sh defaults to original/zero-shot/many-shot/dynamic-shot.
# - Original is complete; many-shot and dynamic-shot likely remain.
# Use a conservative 7-hour first sleep, then poll every 10 minutes.
INITIAL_SLEEP_SECONDS="${INITIAL_SLEEP_SECONDS:-25200}"
POLL_SECONDS="${POLL_SECONDS:-600}"
LOG_FILE="${LOG_FILE:-${REPO_ROOT}/Experiment/watchers/experiment_3_t5gemma_after_train.log}"
RUN_COMMAND="${RUN_COMMAND:-bash dist/experiment_3_t5gemma_retriever.sh}"

mkdir -p "$(dirname "${LOG_FILE}")"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
  printf "[%s] %s\n" "$(timestamp)" "$*" | tee -a "${LOG_FILE}"
}

active_train_processes() {
  ps -eo pid=,ppid=,etime=,cmd= \
    | awk '
      /awk / { next }
      /wait_for_train_then_experiment_3_t5gemma/ { next }
      /\/src\// && (
        /torchrun/ ||
        /lora_flan_large_finqa_rel_fact.py/ ||
        /mistral_train.py/ ||
        /t5gemma-2_train.py/ ||
        /experiment_1_mistral_retriever.sh/ ||
        /experiment_2_flan_retriever.sh/ ||
        /experiment_3_t5gemma_retriever.sh/ ||
        /experiment_1-3_retriever_part.sh/
      ) { print }
    '
}

if ! tmux has-session -t "${TMUX_SESSION}" 2>/dev/null; then
  log "tmux session not found: ${TMUX_SESSION}"
  exit 2
fi

log "watcher_start tmux_session=${TMUX_SESSION} initial_sleep=${INITIAL_SLEEP_SECONDS}s poll=${POLL_SECONDS}s"
log "run_command=${RUN_COMMAND}"
log "sleeping before first check"
sleep "${INITIAL_SLEEP_SECONDS}"

while true; do
  running="$(active_train_processes || true)"
  if [[ -z "${running}" ]]; then
    log "no active FINDER-Mistral retriever training process detected"
    break
  fi
  log "training still active; next check in ${POLL_SECONDS}s"
  printf "%s\n" "${running}" | tee -a "${LOG_FILE}"
  sleep "${POLL_SECONDS}"
done

log "starting Experiment 3 T5Gemma retriever"
cd "${REPO_ROOT}"
exec bash -lc "${RUN_COMMAND}"
