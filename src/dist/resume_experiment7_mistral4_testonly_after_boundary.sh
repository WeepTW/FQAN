#!/usr/bin/env bash
set -Eeuo pipefail

CONDA_ENV=fnqa
export CONDA_ENV

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
EXPT_ID="experiment_7_mistral4_remaining_queue_20260823T014334Z"
EXPT_DIR="$REPO_ROOT/Experiment/$EXPT_ID"
QUEUE_DIR="$EXPT_DIR/remaining_queue"
STOP_FILE="$QUEUE_DIR/STOP_AFTER_CURRENT_CASE"
STOP_STATUS="$QUEUE_DIR/stop_status.json"
HANDOFF_LOG="${EXPERIMENT7_HANDOFF_LOG:?EXPERIMENT7_HANDOFF_LOG is required}"
OLD_SESSION="${EXPERIMENT7_OLD_SESSION:-fnqa_exp7_20260824T005539Z}"
SELECTION_CACHE_JSON="$EXPT_DIR/in_context_selection/gpt5_5/selection_cache.json"

log() {
  printf '%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$HANDOFF_LOG"
}

log "handoff=waiting old_session=$OLD_SESSION"
while tmux has-session -t "$OLD_SESSION" 2>/dev/null; do
  sleep 60
done

if [[ ! -f "$STOP_STATUS" ]]; then
  log "handoff=blocked reason=missing_stop_status path=$STOP_STATUS"
  exit 2
fi
jq -e '.status == "stopped_after_current_case" and .completedItem == "finqa_mistral_o:finqa_test"' "$STOP_STATUS" >/dev/null

if pgrep -af 'run_experiment7_mistral4_remaining_queue|new_full_finqa_run.py.*mistral4|llama-server.*8012' | grep -v "resume_experiment7_mistral4_testonly_after_boundary" >/dev/null; then
  log "handoff=blocked reason=writer_or_server_still_running"
  exit 2
fi

if [[ ! -f "$STOP_FILE" ]]; then
  log "handoff=blocked reason=stop_request_missing path=$STOP_FILE"
  exit 2
fi
rm -- "$STOP_FILE"
log "handoff=resuming scope=finqa_test_only cases=8"

cd "$REPO_ROOT"
exec env \
  CONDA_ENV=fnqa \
  EXPERIMENT7_EXPT_ID="$EXPT_ID" \
  EXPERIMENT7_SELECTION_EXPT_ID="$EXPT_ID" \
  EXPERIMENT7_SELECTION_ENGINE=gpt5_5 \
  EXPERIMENT7_SELECTION_CACHE_JSON="$SELECTION_CACHE_JSON" \
  bash dist/run_experiment7_mistral4_remaining_queue.sh
