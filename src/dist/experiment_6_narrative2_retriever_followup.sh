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
state_path="$runtime_dir/retriever_followup.status"
mkdir -p "$runtime_dir"
cd "$repo_root"

write_state() {
  printf '%s phase=%s status=%s detail=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" "${3:-}" >> "$state_path"
}
wait_for_file() {
  local path="$1"
  while [[ ! -f "$path" ]]; do
    sleep 60
  done
}

: > "$state_path"
write_state followup waiting "flan_and_seeded_t5gemma2"
wait_for_file "$runtime_dir/formal_retriever_flan.rc"
wait_for_file "$runtime_dir/formal_retriever_t5gemma2_seeded.rc"
flan_rc="$(tr -d '[:space:]' < "$runtime_dir/formal_retriever_flan.rc")"
t5gemma2_rc="$(tr -d '[:space:]' < "$runtime_dir/formal_retriever_t5gemma2_seeded.rc")"
if [[ "$flan_rc" != "0" || "$t5gemma2_rc" != "0" ]]; then
  write_state followup blocked "flan_rc=$flan_rc,t5gemma2_seeded_rc=$t5gemma2_rc"
  exit 2
fi
write_state parallel_families completed "flan=100_case_runs,t5gemma2=100_case_runs"

while tmux has-session -t exp6v2_retriever_priority 2>/dev/null; do
  sleep 30
done
if [[ -f "$runtime_dir/retriever_priority.status" ]]; then
  cp -a "$runtime_dir/retriever_priority.status" "$runtime_dir/retriever_priority_initial.status"
fi

write_state priority_resume running
set +e
bash "$repo_root/dist/experiment_6_narrative2_retriever_priority.sh" "$output_root" \
  > "$runtime_dir/retriever_priority_followup.log" 2>&1
followup_rc=$?
set -e
printf '%s\n' "$followup_rc" > "$runtime_dir/retriever_priority_followup.rc"
if [[ "$followup_rc" != "0" ]]; then
  write_state priority_resume blocked "rc=$followup_rc"
  exit "$followup_rc"
fi
write_state priority_resume completed "mistral_verified_evaluated_direct_resumed"
