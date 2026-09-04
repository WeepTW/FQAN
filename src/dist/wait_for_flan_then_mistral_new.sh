#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

WAIT_FOR_SESSION="${WAIT_FOR_SESSION:-flan_no_original_20260606_1053}"
POLL_SECONDS="${POLL_SECONDS:-600}"
PROMPT_MODES="${PROMPT_MODES:-zero-shot many-shot dynamic-shot}"
MISTRAL_EXPT_PREFIX="${MISTRAL_EXPT_PREFIX:-finqa_mistral}"
EXPT_ID_SUFFIX="${EXPT_ID_SUFFIX:-_new}"
LOG_FILE="${LOG_FILE:-${REPO_ROOT}/Experiment/watchers/wait_flan_then_mistral_new.log}"

mkdir -p "$(dirname "${LOG_FILE}")"

timestamp() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

log() {
  printf "[%s] %s\n" "$(timestamp)" "$*" | tee -a "${LOG_FILE}"
}

target_dir_for_prompt() {
  case "$1" in
    zero-shot|zero_shot) printf "%s/Experiment/%s_z%s\n" "${REPO_ROOT}" "${MISTRAL_EXPT_PREFIX}" "${EXPT_ID_SUFFIX}" ;;
    many-shot|many_shot) printf "%s/Experiment/%s_m%s\n" "${REPO_ROOT}" "${MISTRAL_EXPT_PREFIX}" "${EXPT_ID_SUFFIX}" ;;
    dynamic-shot|dynamic_shot) printf "%s/Experiment/%s_d%s\n" "${REPO_ROOT}" "${MISTRAL_EXPT_PREFIX}" "${EXPT_ID_SUFFIX}" ;;
    *) return 2 ;;
  esac
}

log "watcher_start wait_for_session=${WAIT_FOR_SESSION} poll=${POLL_SECONDS}s"
while tmux has-session -t "${WAIT_FOR_SESSION}" 2>/dev/null; do
  log "tmux session still active: ${WAIT_FOR_SESSION}; next check in ${POLL_SECONDS}s"
  sleep "${POLL_SECONDS}"
done

log "tmux session finished or absent: ${WAIT_FOR_SESSION}"
for prompt in ${PROMPT_MODES}; do
  target_dir="$(target_dir_for_prompt "${prompt}")"
  if [[ -e "${target_dir}" ]]; then
    log "refusing to overwrite existing target: ${target_dir}"
    exit 2
  fi
done

log "starting Mistral retriever prompt_modes=${PROMPT_MODES} prefix=${MISTRAL_EXPT_PREFIX} suffix=${EXPT_ID_SUFFIX}"
cd "${REPO_ROOT}"
exec env \
  PROMPT_MODES="${PROMPT_MODES}" \
  MISTRAL_EXPT_PREFIX="${MISTRAL_EXPT_PREFIX}" \
  EXPT_ID_SUFFIX="${EXPT_ID_SUFFIX}" \
  bash dist/experiment_1_mistral_retriever.sh
