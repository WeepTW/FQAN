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
report="$output_root/evaluation_fixed_v2/evaluation_report.json"
coordinator_rc="$runtime_dir/fixed_v2_coordinator.rc"
recorder="$repo_root/dist/record_experiment6_fixed_v2_phase.py"
status="$runtime_dir/fixed_v2_log_watch.status"
mkdir -p "$runtime_dir"

write_state() {
  printf '%s phase=%s status=%s detail=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" "${3:-}" >> "$status"
}

: > "$status"
write_state watcher started
while ! [[ -f "$report" ]] || ! jq -e '
  .status == "development_partial_no_ranking"
  and .rankingPublished == false
  and .completedOfficialCases == 9
  and .evaluatedCaseRuns == 90
  and .formalPredictions == 7650
' "$report" >/dev/null 2>&1; do
  if [[ -f "$coordinator_rc" ]] && \
     [[ "$(tr -d '[:space:]' < "$coordinator_rc")" != "0" ]]; then
    write_state part1 blocked coordinator_failed_before_part1_report
    exit 2
  fi
  sleep 60
done
conda run --no-capture-output -n fnqa python -B "$recorder" \
  --output-root "$output_root" --phase part1 \
  > "$runtime_dir/part1_src_log_record.json"
write_state part1 recorded

while [[ ! -f "$coordinator_rc" ]]; do
  sleep 60
done
final_rc="$(tr -d '[:space:]' < "$coordinator_rc")"
if [[ "$final_rc" != "0" ]]; then
  write_state full blocked "coordinator_rc=$final_rc"
  exit "$final_rc"
fi
conda run --no-capture-output -n fnqa python -B "$recorder" \
  --output-root "$output_root" --phase full \
  > "$runtime_dir/full_src_log_record.json"
write_state full recorded
