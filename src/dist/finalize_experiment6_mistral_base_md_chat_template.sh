#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 --generation-root ROOT --session-m NAME --session-d NAME"
}

GEN_ROOT=""
SESSION_M=""
SESSION_D=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --generation-root) GEN_ROOT="$2"; shift 2 ;;
    --session-m) SESSION_M="$2"; shift 2 ;;
    --session-d) SESSION_D="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
[[ -n "$GEN_ROOT" && -n "$SESSION_M" && -n "$SESSION_D" ]] || { usage >&2; exit 2; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_ENV=fnqa
# Post-processing is CPU-only. The generation queue pins CUDA device 0/1;
# do not leak that topology-only setting into evaluators that deliberately
# reject a visible CUDA device.
PY=(env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMEXPR_NUM_THREADS=4 conda run --no-capture-output -n "$CONDA_ENV" python -B)
EVENTS="$GEN_ROOT/scheduler/finalizer_events.jsonl"
M_LOG="$GEN_ROOT/scheduler/base_m.log"
D_LOG="$GEN_ROOT/scheduler/base_d.log"
mkdir -p "$GEN_ROOT/scheduler"

event() {
  local kind="$1"
  local detail="$2"
  printf '{"time":"%s","event":"%s","detail":"%s"}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$kind" "$detail" >> "$EVENTS"
}

event "finalizer_started" "waiting_for_20_case_runs"
while tmux has-session -t "$SESSION_M" 2>/dev/null || tmux has-session -t "$SESSION_D" 2>/dev/null; do
  if grep -Eq 'case_run_failed|runtime_blocked|Traceback' "$M_LOG" "$D_LOG"; then
    event "finalizer_blocked" "generation_failure_marker"
    exit 2
  fi
  sleep 60
done

if grep -Eq 'case_run_failed|runtime_blocked|Traceback' "$M_LOG" "$D_LOG"; then
  event "finalizer_blocked" "generation_failure_marker"
  exit 2
fi
FINISHED="$(grep -h '"event": "case_run_finished"' "$M_LOG" "$D_LOG" | wc -l | tr -d ' ')"
MANIFESTS="$(find "$GEN_ROOT/manifests" -maxdepth 1 -type f -name '*.json' | wc -l | tr -d ' ')"
if [[ "$FINISHED" != "20" || "$MANIFESTS" != "20" ]]; then
  event "finalizer_blocked" "coverage_finished_${FINISHED}_manifests_${MANIFESTS}"
  exit 2
fi
event "generation_complete" "20_case_runs_1700_rows"

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
BIND_ROOT="${GEN_ROOT}_binding_repaired_v1_${STAMP}"
COMPONENT_ROOT="${GEN_ROOT}_evaluation_components_${STAMP}"
FINAL_ROOT="${GEN_ROOT%_rerun_*}_evaluation_v6_1_0_${STAMP}"

cd "$REPO_ROOT"
"${PY[@]}" dist/materialize_experiment6_mistral_chat_repaired_projection.py \
  --generation-root "$GEN_ROOT" \
  --output-root "$BIND_ROOT"
event "binding_materialization_complete" "$BIND_ROOT"

"${PY[@]}" dist/evaluate_experiment6_binding_candidates_v1.py \
  --version v6.1.0 \
  --scope mistral-base-md \
  --candidate-root "$BIND_ROOT" \
  --evaluation-root "$COMPONENT_ROOT/v610" \
  --config config/experiment6_mistral_base_md_evaluation_v6_1.json
event "v610_five_field_complete" "$COMPONENT_ROOT/v610"

"${PY[@]}" dist/build_experiment6_judge_examples_v4.py \
  --config config/experiment6_narrative2_hybrid_v4_no_gpt41.json \
  --output-dir "$BIND_ROOT/judge_examples"
event "judge_examples_complete" "$BIND_ROOT/judge_examples"

MODELS="$(curl -fsS --max-time 5 http://localhost:8000/v1/models)"
if ! jq -e '.data[] | select(.id == "gpt-5.5")' <<<"$MODELS" >/dev/null; then
  event "finalizer_blocked" "chatmock_gpt_5_5_unavailable"
  exit 2
fi
"${PY[@]}" dist/evaluate_narrative2_reference_aligned_v5.py \
  --config config/experiment6_narrative2_hybrid_v4_no_gpt41.json \
  --mistral-chat-projection \
  --output-root "$BIND_ROOT" \
  --evaluation-root "$COMPONENT_ROOT/text_semantic" \
  --only-case 6_mistral_base_m \
  --only-case 6_mistral_base_d
event "semantic_text_complete" "$COMPONENT_ROOT/text_semantic"

"${PY[@]}" dist/combine_experiment6_v610_with_text_semantic.py \
  --v610-report "$COMPONENT_ROOT/v610/evaluation_report.json" \
  --semantic-text-report "$COMPONENT_ROOT/text_semantic/evaluation_report.json" \
  --output-root "$FINAL_ROOT"
event "finalizer_complete" "$FINAL_ROOT"
