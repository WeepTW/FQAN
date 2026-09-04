#!/usr/bin/env bash
set -Eeuo pipefail

CONDA_ENV=fnqa
CONFIG="config/experiment6_narrative2_generation_mistral_base_md_chat_template_v4.json"
WAIT_SENTINEL="${MISTRAL_WAIT_SENTINEL:?set MISTRAL_WAIT_SENTINEL to the FLAN terminal sentinel}"
STAMP="${MISTRAL_QUEUE_STAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
GEN_ROOT="${MISTRAL_GENERATION_ROOT:-Experiment/experiment_6_mistral_base_md_chat_template_v4_rerun_${STAMP}}"
SMOKE_ROOT="${GEN_ROOT}_smoke"
SCHEDULER="$GEN_ROOT/scheduler"

mkdir -p "$SCHEDULER"
event() {
  printf '%s\t%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "$SCHEDULER/events.log"
}

event "queue=waiting sentinel=$WAIT_SENTINEL"
until [[ -e "$WAIT_SENTINEL" ]]; do
  sleep 60
done
event "queue=starting generation_root=$GEN_ROOT"

conda run --no-capture-output -n "$CONDA_ENV" python -B \
  dist/test_experiment6_mistral_base_md_chat_template_v4.py \
  >"$SCHEDULER/test_v4.log" 2>&1
conda run --no-capture-output -n "$CONDA_ENV" python -B \
  dist/run_experiment6_mistral_base_md_chat_template_v4.py \
  --config "$CONFIG" --output-root "$GEN_ROOT/preflight" --preflight-only \
  --base-route-mode formal --no-resume \
  >"$SCHEDULER/preflight.log" 2>&1
event "validation=passed"

run_smoke() {
  local subdir="$1" case_id="$2" source="$3"
  conda run --no-capture-output -n "$CONDA_ENV" python -B \
    dist/run_experiment6_mistral_base_md_chat_template_v4.py \
    --config "$CONFIG" --output-root "$SMOKE_ROOT/$subdir" \
    --case "$case_id" --run 1 --row-source "$source" --smoke-only \
    --base-route-mode formal --cuda-visible-devices 0 --no-resume \
    >"$SCHEDULER/smoke_${subdir}.log" 2>&1
}

run_smoke dynamic_echo_econ020 6_mistral_base_d Econ_020
run_smoke many_echo_econ066 6_mistral_base_m Econ_066
run_smoke dynamic_success_econ026 6_mistral_base_d Econ_026
conda run --no-capture-output -n "$CONDA_ENV" python -B \
  dist/validate_experiment6_mistral_chat_v4_smoke.py \
  --root "$SMOKE_ROOT" --output "$SCHEDULER/smoke_gate.json" \
  >"$SCHEDULER/smoke_gate.log" 2>&1
event "smoke=passed"

formal_resume_args=(--no-resume)
resume_mode="fresh"
if find "$GEN_ROOT/manifests" -maxdepth 1 -type f -name '*.json' -print -quit 2>/dev/null | grep -q .; then
  formal_resume_args=()
  resume_mode="checkpoint-resume"
fi
event "generation=starting mode=$resume_mode"

run_case() {
  local case_id="$1" device="$2" log="$3"
  conda run --no-capture-output -n "$CONDA_ENV" python -B \
    dist/run_experiment6_mistral_base_md_chat_template_v4.py \
    --config "$CONFIG" --output-root "$GEN_ROOT" --case "$case_id" \
    --base-route-mode formal --cuda-visible-devices "$device" \
    "${formal_resume_args[@]}" \
    >"$log" 2>&1
}

run_case 6_mistral_base_m 0 "$SCHEDULER/base_m.log" &
pid_m=$!
run_case 6_mistral_base_d 1 "$SCHEDULER/base_d.log" &
pid_d=$!
status=0
wait "$pid_m" || status=1
wait "$pid_d" || status=1
if [[ "$status" -ne 0 ]]; then
  event "queue=blocked stage=generation"
  exit 2
fi
event "generation=finished"

bash dist/finalize_experiment6_mistral_base_md_chat_template.sh \
  --generation-root "$GEN_ROOT" \
  --session-m "mistral_v4_finished_m_${STAMP}" \
  --session-d "mistral_v4_finished_d_${STAMP}" \
  >"$SCHEDULER/finalizer.log" 2>&1
conda run --no-capture-output -n "$CONDA_ENV" python -B \
  dist/audit_experiment6_mistral_base_md_v4_completion.py \
  --generation-root "$GEN_ROOT" \
  --finalizer-events "$SCHEDULER/finalizer_events.jsonl" \
  --output "$SCHEDULER/completion_audit.json" \
  >"$SCHEDULER/completion_audit.log" 2>&1
conda run --no-capture-output -n "$CONDA_ENV" python -B \
  dist/record_experiment6_mistral_base_md_v4_completion.py \
  --completion-audit "$SCHEDULER/completion_audit.json" \
  --log-root "../docs/log" --index "../docs/log/index.json" --workspace ".." \
  >"$SCHEDULER/completion_record.log" 2>&1
event "queue=complete"
