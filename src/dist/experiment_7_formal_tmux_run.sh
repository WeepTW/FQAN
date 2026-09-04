#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Canonical formal entrypoint kept for docs/asset/workflow/experiments.md.
# The FQAN implementation owns the actual tmux orchestration so the
# formal launcher and implementation cannot drift into two different queues. The
# queue topology is API_key/chatmock/vllm/llama_cpp/run/monitor.
export EXPT_ID="${EXPT_ID:-experiment_7_fqan_formal_$(date -u +%Y%m%dT%H%M%SZ)}"
export TMUX_RUN_SESSION="${TMUX_RUN_SESSION:-monitor}"
export WINDOW_PREFIX="${WINDOW_PREFIX:-exp7_$(date -u +%Y%m%d_%H%M%S)}"
export WAIT_BEFORE_START_SECONDS="${WAIT_BEFORE_START_SECONDS:-0}"

if [[ -n "${FORMAL_ENGINES:-}" ]]; then
  printf '[%s] FORMAL_ENGINES is deprecated for Experiment 7 formal queue; experiment_7_runner.sh runs the documented full engine matrix.\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >&2
fi

exec "${SCRIPT_DIR}/experiment_7_runner.sh"
