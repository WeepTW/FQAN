#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
  printf 'usage: %s OUTPUT_ROOT\n' "$0" >&2
  exit 64
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_root="$1"
if [[ "$output_root" != /* ]]; then
  output_root="$repo_root/$output_root"
fi
runtime_dir="$output_root/runtime"
smoke_root="$output_root/smoke_fixed_v2"
config_path="$repo_root/config/experiment6_narrative2_generation.json"
runner="$repo_root/dist/run_experiment6_narrative2_generation.py"
evaluator="$repo_root/dist/evaluate_narrative2_fixed_v2.py"
phase_recorder="$repo_root/dist/record_experiment6_fixed_v2_phase.py"
full_coordinator="$repo_root/dist/experiment_6_narrative2_full.sh"
state_path="$runtime_dir/fixed_v2_coordinator.status"
mkdir -p "$runtime_dir"
cd "$repo_root"

write_state() {
  printf '%s phase=%s status=%s detail=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" "${3:-}" >> "$state_path"
}

wait_for_endpoint() {
  local base_url="$1"
  local attempts="${2:-180}"
  for ((attempt=1; attempt<=attempts; attempt++)); do
    if curl -fsS "${base_url%/}/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 10
  done
  return 1
}

verify_smoke() {
  local output_id="$1"
  local status_path="$smoke_root/cases/$output_id/run_01/status.json"
  jq -e '
    .protocol == "experiment6-narrative2-full-v2"
    and .expectedRows == 1
    and .acceptedRows == 1
    and .rejectedRows == 0
    and .runtimeBlockedRows == 0
    and .status == "completed"
    and .converterModel == "gpt-5.5"
    and .reasoningEffort == "medium"
  ' "$status_path" >/dev/null
}

run_case_worker() {
  local family="$1"
  local output_id="$2"
  local log_path="$runtime_dir/part1_${output_id}.log"
  local rc_path="$runtime_dir/part1_${output_id}.rc"
  write_state "$family" case_started "$output_id"
  set +e
  conda run --no-capture-output -n fnqa python -B "$runner" \
    --config "$config_path" --output-root "$output_root" --case "$output_id" \
    > "$log_path" 2>&1
  local worker_rc=$?
  set -e
  printf '%s\n' "$worker_rc" > "$rc_path"
  if [[ "$worker_rc" == "0" ]]; then
    write_state "$family" case_completed "$output_id"
  else
    write_state "$family" case_blocked "$output_id,rc=$worker_rc"
  fi
  return "$worker_rc"
}

wait_workers() {
  local family="$1"
  shift
  local family_rc=0
  local worker_pid
  for worker_pid in "$@"; do
    if ! wait "$worker_pid"; then
      family_rc=2
    fi
  done
  printf '%s\n' "$family_rc" > "$runtime_dir/part1_${family}.rc"
  return "$family_rc"
}

exec 9> "$runtime_dir/fixed_v2_coordinator.lock"
if ! flock -n 9; then
  printf 'another fixed-v2 coordinator owns %s\n' \
    "$runtime_dir/fixed_v2_coordinator.lock" >&2
  exit 73
fi

: > "$state_path"
write_state coordinator started "$output_root"

write_state preflight running
conda run --no-capture-output -n fnqa python -B "$runner" \
  --config "$config_path" --output-root "$output_root" --preflight-only \
  > "$runtime_dir/preflight_fixed_v2.log" 2>&1
write_state preflight completed "54_cases,10_runs,85_rows,prompt_bundles_frozen"

chatmock_url="$(jq -er '.runtimeRoutes.chatmock.baseUrl' "$config_path")"
chatmock_port="$(jq -er '.runtimeRoutes.chatmock.port' "$config_path")"
chatmock_session="$(jq -er '.runtimeRoutes.chatmock.session' "$config_path")"
if ! wait_for_endpoint "$chatmock_url" 1; then
  if ! tmux has-session -t "$chatmock_session" 2>/dev/null; then
    tmux new-session -d -s "$chatmock_session" -c "$repo_root" \
      "CHATMOCK_PORT=$chatmock_port CHATMOCK_REASONING_EFFORT=medium CHATMOCK_REASONING_SUMMARY=none bash dist/start_chatmock_server.sh > $runtime_dir/chatmock.log 2>&1"
  fi
  wait_for_endpoint "$chatmock_url" 180
fi
write_state chatmock ready "model=gpt-5.5,reasoning_effort=medium"

write_state smoke running "one row per retriever family"
conda run --no-capture-output -n fnqa python -B "$runner" \
  --config "$config_path" --output-root "$smoke_root" \
  --smoke-only --limit 1 --no-resume \
  --case 6_flan_z --case 6_mistral_z --case 6_t5gemma2_z \
  > "$runtime_dir/part1_smoke.log" 2>&1
verify_smoke 6_flan_z
verify_smoke 6_mistral_z
verify_smoke 6_t5gemma2_z
write_state smoke completed "flan,mistral,t5gemma2"

write_state part1 running "9_cases,90_case_runs,7650_predictions"
run_case_worker flan 6_flan_z &
flan_z_pid=$!
run_case_worker flan 6_flan_m &
flan_m_pid=$!
run_case_worker flan 6_flan_d &
flan_d_pid=$!
run_case_worker t5gemma2 6_t5gemma2_z &
t5_z_pid=$!
run_case_worker t5gemma2 6_t5gemma2_m &
t5_m_pid=$!
run_case_worker t5gemma2 6_t5gemma2_d &
t5_d_pid=$!

set +e
wait_workers flan "$flan_z_pid" "$flan_m_pid" "$flan_d_pid"
flan_rc=$?
if [[ "$flan_rc" == "0" ]]; then
  run_case_worker mistral 6_mistral_z &
  mistral_z_pid=$!
  run_case_worker mistral 6_mistral_m &
  mistral_m_pid=$!
  run_case_worker mistral 6_mistral_d &
  mistral_d_pid=$!
fi
wait_workers t5gemma2 "$t5_z_pid" "$t5_m_pid" "$t5_d_pid"
t5_rc=$?
mistral_rc=2
if [[ -n "${mistral_z_pid:-}" ]]; then
  wait_workers mistral "$mistral_z_pid" "$mistral_m_pid" "$mistral_d_pid"
  mistral_rc=$?
fi
set -e
if [[ "$flan_rc" != "0" || "$t5_rc" != "0" || "$mistral_rc" != "0" ]]; then
  write_state part1 blocked "flan=$flan_rc,t5gemma2=$t5_rc,mistral=$mistral_rc"
  exit 2
fi
write_state part1 completed "9_cases,90_case_runs"

fixed_args=(
  --output-root "$output_root"
  --only-case 6_flan_z
  --only-case 6_flan_m
  --only-case 6_flan_d
  --only-case 6_mistral_z
  --only-case 6_mistral_m
  --only-case 6_mistral_d
  --only-case 6_t5gemma2_z
  --only-case 6_t5gemma2_m
  --only-case 6_t5gemma2_d
)
write_state part1_evaluation running "protocol=narrative2-fixed-python-v2"
conda run --no-capture-output -n fnqa python -B "$evaluator" \
  "${fixed_args[@]}" > "$runtime_dir/part1_fixed_v2_evaluation.log" 2>&1
if ! jq -e '
  .status == "development_partial_no_ranking"
  and .rankingPublished == false
  and .completedOfficialCases == 9
  and .evaluatedCaseRuns == 90
  and .formalPredictions == 7650
' "$output_root/evaluation_fixed_v2/evaluation_report.json" >/dev/null; then
  write_state part1_evaluation blocked "report_contract_failed"
  exit 2
fi
write_state part1_evaluation completed "9_cases,no_global_ranking"
conda run --no-capture-output -n fnqa python -B "$phase_recorder" \
  --output-root "$output_root" --phase part1 \
  > "$runtime_dir/part1_src_log_record.json"

write_state remaining_matrix scheduling "45_official_cases_plus_controls"
EXPERIMENT6_SKIP_ROUTE_SMOKE=1 bash "$full_coordinator" "$output_root" \
  > "$runtime_dir/post_part1_full_coordinator.log" 2>&1
write_state remaining_matrix scheduled

while [[ ! -f "$runtime_dir/formal_local_queue.rc" ]]; do
  sleep 60
done
final_rc="$(tr -d '[:space:]' < "$runtime_dir/formal_local_queue.rc")"
if [[ "$final_rc" != "0" ]]; then
  write_state coordinator blocked_no_ranking "formal_local_queue_rc=$final_rc"
  exit "$final_rc"
fi
conda run --no-capture-output -n fnqa python -B "$phase_recorder" \
  --output-root "$output_root" --phase full \
  > "$runtime_dir/full_src_log_record.json"
write_state coordinator completed "54_cases,fixed_v2_evaluated"
