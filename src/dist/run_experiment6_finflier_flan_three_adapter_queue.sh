#!/usr/bin/env bash
set -Eeuo pipefail

CONDA_ENV=fnqa
export CONDA_ENV
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/retriever_experiment_lib.sh"

CONFIG="config/experiment6_narrative2_generation_finflier_flan_three_adapter_long_context.json"
EVAL_CONFIG="config/experiment6_finflier_flan_three_adapter_long_context_evaluation_v6_1.json"
OUTPUT_ROOT="${EXPERIMENT6_OUTPUT_ROOT:-Experiment/experiment_6_finflier_flan_three_adapter_long_context_generation_$(date -u +%Y%m%dT%H%M%SZ)}"
CANDIDATE_ROOT="${EXPERIMENT6_CANDIDATE_ROOT:-${OUTPUT_ROOT}_binding_candidates_v1}"
EVALUATION_ROOT="${EXPERIMENT6_EVALUATION_ROOT:-${OUTPUT_ROOT/_generation_/_evaluation_v6_1_0_}}"
MAX_WORKERS="${EXPERIMENT6_MAX_WORKERS:-4}"
MAX_ATTEMPTS="${EXPERIMENT6_TASK_MAX_ATTEMPTS:-3}"
EXPECTED_ROWS=85
CHATMOCK_BASE_URL="${CHATMOCK_BASE_URL:-http://localhost:8000/v1}"
CHATMOCK_API_KEY="${CHATMOCK_API_KEY:-key}"
CASE_IDS="6_finflier_prompt_flan_z_adapter_long_context 6_finflier_prompt_flan_m_adapter_long_context 6_finflier_prompt_flan_d_adapter_long_context"
STOP_REQUEST_FILE="${WAITING_QUEUE_STOP_FILE:-$OUTPUT_ROOT/STOP_AFTER_CURRENT_CASE}"

mkdir -p "$OUTPUT_ROOT/logs" "$OUTPUT_ROOT/queue"
QUEUE_LOG="$OUTPUT_ROOT/queue/timeline.log"
FAILURE_LOG="$OUTPUT_ROOT/queue/failures.log"

event() {
  printf '%s\t%s\n' "$(utc_now)" "$*" | tee -a "$QUEUE_LOG"
}

chatmock_ready() {
  curl -fsS --max-time 10 \
    -H "Authorization: Bearer $CHATMOCK_API_KEY" \
    "${CHATMOCK_BASE_URL%/}/models" >/dev/null
}

wait_for_chatmock() {
  local attempts=0
  until chatmock_ready; do
    attempts=$((attempts + 1))
    if [[ "$attempts" -ge 60 ]]; then
      event "chatmock=blocked attempts=$attempts"
      return 1
    fi
    sleep 10
  done
}

manifest_complete() {
  local manifest="$1"
  [[ -f "$manifest" ]] || return 1
  jq -e --argjson expected "$EXPECTED_ROWS" '
    (.status | startswith("completed"))
    and (.runtimeBlockedRows == 0)
    and ((.acceptedRows + .rejectedRows) == $expected)
  ' "$manifest" >/dev/null
}

run_task() {
  local case_id="$1"
  local run="$2"
  local device="$3"
  local run_padded
  run_padded="$(printf '%02d' "$run")"
  local manifest="$OUTPUT_ROOT/manifests/${case_id}__run_${run_padded}.json"
  local log="$OUTPUT_ROOT/logs/${case_id}__run_${run_padded}__gpu_${device}.log"
  if manifest_complete "$manifest"; then
    event "task=skip case=$case_id run=$run gpu=$device reason=validated_manifest"
    return 0
  fi
  local attempt
  for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    wait_for_chatmock || return 1
    event "task=start case=$case_id run=$run gpu=$device attempt=$attempt"
    set +e
    conda run --no-capture-output -n "$CONDA_ENV" \
      python -B dist/run_experiment6_narrative2_generation.py \
      --config "$CONFIG" \
      --output-root "$OUTPUT_ROOT" \
      --case "$case_id" \
      --cuda-visible-devices "$device" \
      --run "$run" \
      >>"$log" 2>&1
    local rc=$?
    set -e
    if [[ "$rc" -eq 0 ]] && manifest_complete "$manifest"; then
      event "task=completed case=$case_id run=$run gpu=$device attempt=$attempt"
      return 0
    fi
    event "task=retry case=$case_id run=$run gpu=$device attempt=$attempt rc=$rc"
    sleep 15
  done
  printf '%s\tcase=%s run=%s gpu=%s attempts=%s\n' \
    "$(utc_now)" "$case_id" "$run" "$device" "$MAX_ATTEMPTS" \
    >>"$FAILURE_LOG"
  return 1
}

if [[ "$MAX_WORKERS" -lt 2 ]]; then
  printf 'EXPERIMENT6_MAX_WORKERS must be >= 2.\n' >&2
  exit 2
fi
wait_for_chatmock
event "queue=start output_root=$OUTPUT_ROOT workers=$MAX_WORKERS gpu_policy=one_retriever_per_gpu attention_query_chunk_size=512 no_cpu_fallback=1 stop_file=$STOP_REQUEST_FILE"

failures=0
completed_case_ids=""
stopped=0
for case_id in $CASE_IDS; do
  active=0
  case_failures=0
  for run in {1..10}; do
    device="$(( (run - 1) % 2 ))"
    run_task "$case_id" "$run" "$device" &
    active=$((active + 1))
    if [[ "$active" -ge "$MAX_WORKERS" ]]; then
      if ! wait -n; then
        case_failures=$((case_failures + 1))
      fi
      active=$((active - 1))
    fi
  done
  while [[ "$active" -gt 0 ]]; do
    if ! wait -n; then
      case_failures=$((case_failures + 1))
    fi
    active=$((active - 1))
  done
  for run in {1..10}; do
    manifest="$OUTPUT_ROOT/manifests/${case_id}__run_$(printf '%02d' "$run").json"
    if ! manifest_complete "$manifest"; then
      case_failures=$((case_failures + 1))
      printf '%s\tcase=%s run=%s reason=manifest_not_complete\n' \
        "$(utc_now)" "$case_id" "$run" >>"$FAILURE_LOG"
    fi
  done
  if [[ "$case_failures" -ne 0 ]]; then
    failures=$((failures + case_failures))
    event "queue=blocked case=$case_id failures=$case_failures failure_log=$FAILURE_LOG"
    break
  fi
  completed_case_ids="${completed_case_ids}${completed_case_ids:+ }${case_id}"
  event "case=finalized case=$case_id completed_cases=$(wc -w <<<"$completed_case_ids")"
  if [[ -f "$STOP_REQUEST_FILE" ]]; then
    stopped=1
    event "stop=requested boundary=case completed_case=$case_id"
    break
  fi
done
if [[ "$failures" -ne 0 ]]; then
  event "queue=blocked failures=$failures failure_log=$FAILURE_LOG"
  exit 1
fi

if [[ -z "$completed_case_ids" ]]; then
  event "queue=blocked reason=no_completed_case"
  exit 1
fi

MATERIALIZATION_CONFIG="$OUTPUT_ROOT/finalization/materialization_config.json"
RUNTIME_EVAL_CONFIG="$OUTPUT_ROOT/finalization/evaluation_config.json"
mkdir -p "$OUTPUT_ROOT/finalization/logs"
GENERATION_ROOT="$OUTPUT_ROOT" MATERIALIZATION_CONFIG="$MATERIALIZATION_CONFIG" \
COMPLETED_CASE_IDS="$completed_case_ids" BASE_EVAL_CONFIG="$EVAL_CONFIG" \
RUNTIME_EVAL_CONFIG="$RUNTIME_EVAL_CONFIG" \
  conda run --no-capture-output -n "$CONDA_ENV" python -B - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["GENERATION_ROOT"])
output = Path(os.environ["MATERIALIZATION_CONFIG"])
case_ids = os.environ["COMPLETED_CASE_IDS"].split()
manifests = [
    json.loads(
        (root / "manifests" / f"{case_id}__run_{run:02d}.json").read_text(
            encoding="utf-8"
        )
    )
    for case_id in case_ids
    for run in range(1, 11)
]
protocols = {item["protocol"] for item in manifests}
fingerprints = {item["compatibilityFingerprint"] for item in manifests}
if len(protocols) != 1 or len(fingerprints) != 1:
    raise SystemExit(
        f"incompatible manifests: protocols={protocols}, fingerprints={fingerprints}"
    )
payload = {
    "schemaVersion": 1,
    "protocol": "experiment6-binding-candidate-materialization-v1",
    "sourceProtocol": next(iter(protocols)),
    "sourceCompatibilityFingerprint": next(iter(fingerprints)),
    "expectedCases": len(case_ids),
    "expectedRuns": 10,
    "expectedRows": 85,
    "caseIds": case_ids,
    "requiredBindingKeys": [
        "ObjectName",
        "DataName",
        "Position",
        "Trend",
        "Num",
        "Text",
    ],
    "requireRepairCoverage": True,
}
output.parent.mkdir(parents=True, exist_ok=True)
output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
eval_payload = json.loads(Path(os.environ["BASE_EVAL_CONFIG"]).read_text(encoding="utf-8"))
eval_payload["expectedCandidateCases"] = len(case_ids)
eval_payload["expectedMergedCases"] = len(case_ids)
eval_payload["completedCaseIds"] = case_ids
eval_output = Path(os.environ["RUNTIME_EVAL_CONFIG"])
eval_output.write_text(
    json.dumps(eval_payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(output)
PY

event "finalization=materialize"
conda run --no-capture-output -n "$CONDA_ENV" \
  python -B dist/materialize_experiment6_binding_candidates.py \
  --generation-root "$OUTPUT_ROOT" \
  --config "$MATERIALIZATION_CONFIG" \
  --output-root "$CANDIDATE_ROOT" \
  >"$OUTPUT_ROOT/finalization/logs/materialize.log" 2>&1

event "finalization=validate"
conda run --no-capture-output -n "$CONDA_ENV" \
  python -B dist/validate_experiment6_binding_candidates.py \
  --root "$CANDIDATE_ROOT" \
  >"$OUTPUT_ROOT/finalization/logs/validate.log" 2>&1

event "finalization=evaluate"
mkdir -p "$EVALUATION_ROOT"
conda run --no-capture-output -n "$CONDA_ENV" \
  python -B dist/evaluate_experiment6_binding_candidates_v1.py \
  --version v6.1.0 \
  --scope flan-long-context \
  --candidate-root "$CANDIDATE_ROOT" \
  --evaluation-root "$EVALUATION_ROOT" \
  --config "$RUNTIME_EVAL_CONFIG" \
  >"$OUTPUT_ROOT/finalization/logs/evaluate.log" 2>&1

conda run --no-capture-output -n "$CONDA_ENV" \
  python -B dist/build_experiment6_binding_candidate_score_tables.py \
  --evaluation-report "$EVALUATION_ROOT/evaluation_report.json" \
  --evaluation-root "$EVALUATION_ROOT" \
  --source-registry config/experiment6_source_registry.json \
  --output-dir "$EVALUATION_ROOT" \
  >"$OUTPUT_ROOT/finalization/logs/build_tables.log" 2>&1

if [[ "$stopped" -eq 1 ]]; then
  STOP_TIME="$(utc_now)" STOP_OUTPUT_ROOT="$OUTPUT_ROOT" \
  STOP_EVALUATION_ROOT="$EVALUATION_ROOT" STOP_CASE_IDS="$completed_case_ids" \
  STOP_REQUEST_FILE="$STOP_REQUEST_FILE" \
    conda run --no-capture-output -n "$CONDA_ENV" python -B - <<'PY_STOP'
import json
import os
from pathlib import Path

root = Path(os.environ["STOP_OUTPUT_ROOT"])
payload = {
    "time": os.environ["STOP_TIME"],
    "status": "stopped_after_current_case",
    "metric": "Experiment 6 binding evaluation (not execution accuracy)",
    "completedCaseIds": os.environ["STOP_CASE_IDS"].split(),
    "evaluationReport": str(Path(os.environ["STOP_EVALUATION_ROOT"]) / "evaluation_report.json"),
    "stopRequestFile": os.environ["STOP_REQUEST_FILE"],
}
(root / "queue" / "stop_status.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
PY_STOP
  printf '%s\n' "$(utc_now)" >"$OUTPUT_ROOT/queue.stopped"
  event "queue=stopped evaluation_root=$EVALUATION_ROOT completed_cases=$(wc -w <<<"$completed_case_ids")"
else
  printf '%s\n' "$(utc_now)" >"$OUTPUT_ROOT/queue.completed"
  event "queue=completed evaluation_root=$EVALUATION_ROOT"
fi
