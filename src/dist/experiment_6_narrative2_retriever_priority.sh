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
config_path="$repo_root/config/experiment6_narrative2_generation.json"
runner="$repo_root/dist/run_experiment6_narrative2_generation.py"
verifier="$repo_root/dist/verify_experiment6_retrievers.py"
evaluator="$repo_root/dist/evaluate_narrative2_hybrid.py"
migrator="$repo_root/dist/migrate_experiment6_retriever_runtime_profiles.py"
mkdir -p "$runtime_dir"
cd "$repo_root"

state_path="$runtime_dir/retriever_priority.status"
write_state() {
  printf '%s phase=%s status=%s detail=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" "${3:-}" >> "$state_path"
}
run_family_worker() {
  local family="$1"
  local shard_number="$2"
  shift 2
  local log_path="$runtime_dir/formal_retriever_${family}_shard_${shard_number}.log"
  local rc_path="$runtime_dir/formal_retriever_${family}_shard_${shard_number}.rc"
  local case_args=()
  local output_id
  for output_id in "$@"; do
    case_args+=(--case "$output_id")
  done
  write_state "$family" shard_start "shard=$shard_number,cases=$*"
  set +e
  conda run --no-capture-output -n fnqa python -B "$runner" \
    --config "$config_path" --output-root "$output_root" "${case_args[@]}" \
    >> "$log_path" 2>&1
  local worker_rc=$?
  set -e
  printf '%s\n' "$worker_rc" > "$rc_path"
  if [[ "$worker_rc" == "0" ]]; then
    write_state "$family" shard_completed "shard=$shard_number"
  else
    write_state "$family" shard_blocked "shard=$shard_number,rc=$worker_rc"
  fi
  return "$worker_rc"
}

run_family() {
  local family="$1"
  local expected_workers
  expected_workers="$(jq -er --arg family "$family"     '.retriever.scheduler.familyWorkerOverrides[$family] // .retriever.scheduler.familyWorkers'     "$config_path")"
  local shard_jsons=()
  mapfile -t shard_jsons < <(
    jq -cer --arg family "$family" '.retriever.scheduler.familyShards[$family][]' "$config_path"
  )
  if [[ "${#shard_jsons[@]}" -ne "$expected_workers" ]]; then
    write_state "$family" formal_blocked "configured_shards=${#shard_jsons[@]},expected=$expected_workers"
    return 2
  fi

  write_state "$family" formal_start "workers=$expected_workers"
  local pids=()
  local shard_index
  for shard_index in "${!shard_jsons[@]}"; do
    local shard_cases=()
    mapfile -t shard_cases < <(jq -r '.[]' <<< "${shard_jsons[$shard_index]}")
    run_family_worker "$family" "$((shard_index + 1))" "${shard_cases[@]}" &
    pids+=("$!")
  done

  local family_rc=0
  local worker_pid
  for worker_pid in "${pids[@]}"; do
    if ! wait "$worker_pid"; then
      family_rc=2
    fi
  done
  printf '%s\n' "$family_rc" > "$runtime_dir/formal_retriever_${family}.rc"
  if [[ "$family_rc" == "0" ]]; then
    write_state "$family" formal_completed "workers=$expected_workers"
  else
    write_state "$family" formal_blocked "rc=$family_rc"
  fi
  return "$family_rc"
}

exec 9> "$runtime_dir/retriever_priority.lock"
if ! flock -n 9; then
  printf 'another retriever-priority coordinator owns %s\n' "$runtime_dir/retriever_priority.lock" >&2
  exit 73
fi

: > "$state_path"
write_state coordinator started "$output_root"

chatmock_url="$(jq -er '.runtimeRoutes.chatmock.baseUrl' "$config_path")"
if ! curl -fsS "${chatmock_url%/}/models" >/dev/null; then
  write_state chatmock blocked "endpoint unavailable"
  exit 2
fi
if ! jq -e '
  .converter.requestedModel == "gpt-5.5"
  and .converter.actualModelRequired == "gpt-5.5"
  and .converter.reasoningEffort == "medium"
' "$config_path" >/dev/null; then
  write_state chatmock blocked "converter identity contract failed"
  exit 2
fi
write_state chatmock ready "requested=required=gpt-5.5,effort=medium;per-row identity enforced"

write_state preflight running
conda run --no-capture-output -n fnqa python -B "$runner" \
  --config "$config_path" --output-root "$output_root" --preflight-only \
  > "$runtime_dir/retriever_priority_preflight.log" 2>&1
write_state preflight completed

set +e
run_family flan &
flan_pid=$!
run_family t5gemma2 &
t5gemma_pid=$!

wait "$flan_pid"
flan_rc=$?
mistral_rc=2
if [[ "$flan_rc" == "0" ]]; then
  write_state gpu0_transition running "flan_completed,mistral_starting_while_t5gemma2_continues"
  run_family mistral &
  mistral_pid=$!
fi

wait "$t5gemma_pid"
t5gemma_rc=$?
if [[ -n "${mistral_pid:-}" ]]; then
  wait "$mistral_pid"
  mistral_rc=$?
fi
set -e
if [[ "$flan_rc" != "0" || "$t5gemma_rc" != "0" || "$mistral_rc" != "0" ]]; then
  write_state retriever_families blocked "flan_rc=$flan_rc,t5gemma2_rc=$t5gemma_rc,mistral_rc=$mistral_rc"
  exit 2
fi
write_state retriever_families completed "flan=100,t5gemma2=100,mistral=100_case_runs"

write_state provenance_migration running "metadata_only,require_complete"
migration_marker="$runtime_dir/formal_retriever_provenance_migration.started"
: > "$migration_marker"
set +e
conda run --no-capture-output -n fnqa python -B "$migrator" \
  --config "$config_path" --output-root "$output_root" \
  --require-complete --apply \
  > "$runtime_dir/formal_retriever_provenance_migration.log" 2>&1
migration_rc=$?
set -e
printf '%s\n' "$migration_rc" > "$runtime_dir/formal_retriever_provenance_migration.rc"
if [[ "$migration_rc" != "0" ]]; then
  write_state provenance_migration blocked "rc=$migration_rc"
  exit "$migration_rc"
fi
mapfile -t fresh_migration_reports < <(
  find "$output_root/diagnostics/provenance" \
    -path '*/retriever_runtime_profile_*/audit_report.json' \
    -type f -newer "$migration_marker" -print | sort
)
if [[ "${#fresh_migration_reports[@]}" -ne 1 ]]; then
  write_state provenance_migration blocked \
    "fresh_audit_reports=${#fresh_migration_reports[@]}"
  exit 2
fi
migration_report="${fresh_migration_reports[0]}"
if ! jq -e '
  .status == "completed_metadata_only"
  and .plannedRuns == 300
  and .expectedRuns == 300
  and (.missingRuns | length) == 0
  and .predictionOrCandidateTextChanged == false
  and .converterRawResponseTextChanged == false
  and .originalsPreserved == true
' "$migration_report" >/dev/null; then
  write_state provenance_migration blocked "audit_contract_failed"
  exit 2
fi
write_state provenance_migration completed \
  "runs=300,originals_preserved,audit=$migration_report"

write_state verification running
set +e
conda run --no-capture-output -n fnqa python -B "$verifier" \
  --config "$config_path" --output-root "$output_root" \
  > "$runtime_dir/formal_retriever_verification.log" 2>&1
verify_rc=$?
set -e
printf '%s\n' "$verify_rc" > "$runtime_dir/formal_retriever_verification.rc"
if [[ "$verify_rc" != "0" ]]; then
  write_state verification blocked "rc=$verify_rc"
  exit "$verify_rc"
fi
write_state verification completed "cases=30,case_runs=300,predictions=25500"

mapfile -t retriever_cases < <(
  conda run --no-capture-output -n fnqa python -B "$verifier" \
    --config "$config_path" --output-root "$output_root" --print-output-ids
)
if [[ "${#retriever_cases[@]}" -ne 30 ]]; then
  write_state evaluation blocked "retriever_case_count=${#retriever_cases[@]}"
  exit 2
fi
evaluation_args=()
for output_id in "${retriever_cases[@]}"; do
  evaluation_args+=(--only-case "$output_id")
done

write_state evaluation running "partial_no_ranking"
evaluation_marker="$runtime_dir/formal_retriever_evaluation.started"
: > "$evaluation_marker"
set +e
conda run --no-capture-output -n fnqa python -B "$evaluator" \
  --output-root "$output_root" "${evaluation_args[@]}" \
  > "$runtime_dir/formal_retriever_evaluation.log" 2>&1
evaluation_raw_rc=$?
set -e
printf '%s\n' "$evaluation_raw_rc" > "$runtime_dir/formal_retriever_evaluation.raw.rc"
evaluation_report="$output_root/evaluation/evaluation_report.json"
if [[ ! -f "$evaluation_report" || ! "$evaluation_report" -nt "$evaluation_marker" ]]; then
  write_state evaluation blocked "fresh_report_missing,raw_rc=$evaluation_raw_rc"
  exit 2
fi
if ! jq -e '
  .status == "development_partial_no_ranking"
  and .completedOfficialCases == 30
  and .completedControlCases == 0
  and .formalPredictions == 25500
  and .diagnostics.rankingInterpretationStatus == "development_partial_no_ranking"
' "$evaluation_report" >/dev/null; then
  write_state evaluation blocked "report_contract_failed,raw_rc=$evaluation_raw_rc"
  exit 2
fi
printf '0\n' > "$runtime_dir/formal_retriever_evaluation.rc"
write_state evaluation completed "30_cases,no_ranking,raw_rc=$evaluation_raw_rc"

write_state direct_resume scheduling
set +e
EXPERIMENT6_SKIP_ROUTE_SMOKE=1 bash "$repo_root/dist/experiment_6_narrative2_full.sh" "$output_root" \
  > "$runtime_dir/post_retriever_resume.log" 2>&1
resume_rc=$?
set -e
printf '%s\n' "$resume_rc" > "$runtime_dir/post_retriever_resume.rc"
if [[ "$resume_rc" != "0" ]]; then
  write_state direct_resume blocked "rc=$resume_rc"
  exit "$resume_rc"
fi
write_state direct_resume scheduled
write_state coordinator completed "local_retrievers_verified_and_evaluated"
