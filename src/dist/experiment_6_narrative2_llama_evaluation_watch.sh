#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  printf 'usage: %s OUTPUT_ROOT [llama4|mistral4]\n' "$0" >&2
  exit 64
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_root="$1"
[[ "$output_root" == /* ]] || output_root="$repo_root/$output_root"
family="${2:-llama4}"
case "$family" in
  llama4)
    output_ids=(6_FinFlier_llama 6_llama_z 6_llama_m 6_llama_d)
    ;;
  mistral4)
    output_ids=(6_FinFlier_mistral4 6_mistral4_z 6_mistral4_m 6_mistral4_d)
    ;;
  *)
    printf 'unsupported family: %s\n' "$family" >&2
    exit 64
    ;;
esac
runtime_dir="$output_root/runtime"
generation_config="$repo_root/config/experiment6_narrative2_generation.json"
evaluation_config="$repo_root/config/experiment6_narrative2_evaluation.json"
evaluator="$repo_root/dist/evaluate_narrative2_hybrid.py"
poll_seconds="${EXPERIMENT6_EVALUATION_POLL_SECONDS:-60}"
formal_rc="$runtime_dir/formal_${family}.rc"
evaluation_rc="$runtime_dir/formal_${family}_partial_evaluation.rc"
evaluation_log="$runtime_dir/formal_${family}_partial_evaluation.log"
state="$runtime_dir/${family}_partial_evaluation.status"
probe="$runtime_dir/${family}_partial_judge_probe.json"
chatmock_url="$(jq -er '.runtimeRoutes.chatmock.baseUrl' "$generation_config")"
judge_model="$(jq -er '.judge.model' "$evaluation_config")"
judge_effort="$(jq -er '.judge.reasoningEffort' "$evaluation_config")"

write_state() {
  printf '%s phase=%s status=%s detail=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" "${3:-}" >> "$state"
}

mkdir -p "$runtime_dir"
: > "$state"
: > "$evaluation_rc"
write_state "$family" wait formal_rc
while [[ ! -s "$formal_rc" ]]; do
  sleep "$poll_seconds"
done
llama_rc="$(tr -d '[:space:]' < "$formal_rc")"
if [[ "$llama_rc" != "0" ]]; then
  printf '%s\n' "$llama_rc" > "$evaluation_rc"
  write_state "$family" blocked "formal_rc=$llama_rc,no_ranking"
  exit "$llama_rc"
fi

write_state judge wait_endpoint "model=$judge_model,effort=$judge_effort"
for _ in $(seq 1 "${EXPERIMENT6_CHATMOCK_WAIT_ATTEMPTS:-180}"); do
  curl -fsS "${chatmock_url%/}/models" >/dev/null 2>&1 && break
  sleep 10
done
if ! curl -fsS "${chatmock_url%/}/models" >/dev/null 2>&1; then
  printf '2\n' > "$evaluation_rc"
  write_state judge blocked endpoint_unavailable_no_ranking
  exit 2
fi

curl -fsS \
  -H 'Authorization: Bearer key' \
  -H 'Content-Type: application/json' \
  "${chatmock_url%/}/chat/completions" \
  --data "{\"model\":\"$judge_model\",\"messages\":[{\"role\":\"user\",\"content\":\"Return exactly {\\\"ok\\\":true}.\"}],\"reasoning_effort\":\"$judge_effort\",\"max_completion_tokens\":64}" \
  > "$probe"
if ! jq -e --arg model "$judge_model" '
  .model == $model
  and .choices[0].finish_reason == "stop"
  and .choices[0].message.content == "{\"ok\":true}"
' "$probe" >/dev/null; then
  printf '2\n' > "$evaluation_rc"
  write_state judge blocked identity_probe_failed_no_ranking
  exit 2
fi
write_state judge ready "actual=$judge_model,effort=$judge_effort"

write_state evaluation start partial_no_ranking
case_args=()
for output_id in "${output_ids[@]}"; do
  case_args+=(--only-case "$output_id")
done
set +e
env CHATMOCK_API_KEY=key conda run --no-capture-output -n fnqa python -B "$evaluator" \
  --config "$evaluation_config" \
  --output-root "$output_root" \
  "${case_args[@]}" \
  > "$evaluation_log" 2>&1
rc=$?
set -e
printf '%s\n' "$rc" > "$evaluation_rc"
if [[ "$rc" != "0" ]]; then
  write_state evaluation blocked "rc=$rc,no_ranking"
  exit "$rc"
fi

report="$output_root/evaluation/evaluation_report.json"
progress="$output_root/evaluation/evaluation_progress.json"
if ! jq -e '
  .status == "development_partial_no_ranking"
  and .completedOfficialCases == 4
  and .completedControlCases == 0
  and .formalPredictions == 3400
  and .diagnostics.rankingInterpretationStatus == "development_partial_no_ranking"
' "$report" >/dev/null || ! jq -e '
  .status == "development_partial_no_ranking"
  and .completedCaseRuns == 40
  and (.blockedCaseRuns | length) == 0
  and .rankingInterpretationStatus == "development_partial_no_ranking"
' "$progress" >/dev/null; then
  printf '2\n' > "$evaluation_rc"
  write_state evaluation blocked report_gate_failed_no_ranking
  exit 2
fi
write_state evaluation completed "$family,40_runs,3400_rows,no_ranking"
