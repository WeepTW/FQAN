#!/usr/bin/env bash
set -Eeuo pipefail

CONDA_ENV=fnqa
export CONDA_ENV
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/retriever_experiment_lib.sh"

EXPT_ID="${EXPERIMENT7_EXPT_ID:-experiment_7_mistral4_remaining_queue_$(date -u +%Y%m%dT%H%M%SZ)}"
EXPT_DIR="$REPO_ROOT/Experiment/$EXPT_ID"
QUEUE_DIR="$EXPT_DIR/remaining_queue"
QUEUE_LOG="$QUEUE_DIR/timeline.log"
SELECTION_EXPT_ID="${EXPERIMENT7_SELECTION_EXPT_ID:-experiment_7_selection_gpt55_finqa_train_formal_20260611T080113Z}"
SELECTION_ENGINE="${EXPERIMENT7_SELECTION_ENGINE:-gpt5_5}"
SELECTION_CACHE_JSON="${EXPERIMENT7_SELECTION_CACHE_JSON:-$REPO_ROOT/Experiment/$SELECTION_EXPT_ID/in_context_selection/$SELECTION_ENGINE/selection_cache.json}"
BINDING_ROOT="$EXPT_DIR/selection_cache_binding"
OLD_CASE_ROOT="$REPO_ROOT/Experiment/experiment_7_rest_flan_d_mistral4_v5/generator/mistral4/finqa_flan_d_finqa_test"
MATRIX="finqa_flan_d:finqa_test finqa_mistral_o:finqa_test finqa_t5gemma2_o:finqa_test finqa_t5gemma2_z:finqa_test finqa_mistral_z:finqa_test finqa_mistral_m:finqa_test finqa_t5gemma2_m:finqa_test finqa_t5gemma2_d:finqa_test"
MAX_ATTEMPTS="${EXPERIMENT7_CASE_MAX_ATTEMPTS:-5}"
REQUEST_TIMEOUT_SECONDS="${EXPERIMENT7_REQUEST_TIMEOUT_SECONDS:-1800}"
FLAN_D_PREFIX_MAX_ROWS="${EXPERIMENT7_FLAN_D_PREFIX_MAX_ROWS:-17}"
SERVER_BIN="$WORKSPACE_ROOT/utils/llama.cpp/build/bin/llama-server"
SERVER_LIB_DIR="$(dirname "$SERVER_BIN")"
MODEL_PATH="$WORKSPACE_ROOT/utils/models/mistral_small_4_119b_2603_gguf/UD-Q4_K_M/Mistral-Small-4-119B-2603-UD-Q4_K_M-00001-of-00003.gguf"
BASE_URL=http://localhost:8012/v1
STOP_REQUEST_FILE="${WAITING_QUEUE_STOP_FILE:-$QUEUE_DIR/STOP_AFTER_CURRENT_CASE}"
mkdir -p "$QUEUE_DIR"

event() {
  printf '%s\t%s\n' "$(utc_now)" "$*" | tee -a "$QUEUE_LOG"
}

if [[ -n "${EXPERIMENT6_SENTINEL:-}" ]]; then
  event "wait=experiment6 sentinel=$EXPERIMENT6_SENTINEL"
  while [[ ! -f "$EXPERIMENT6_SENTINEL" ]]; do
    sleep 30
  done
fi

for required in "$SELECTION_CACHE_JSON" "$SERVER_BIN" "$MODEL_PATH"; do
  if [[ ! -f "$required" ]]; then
    event "preflight=blocked missing=$required"
    exit 2
  fi
done

event "selection_binding=start datasets=test prompt_types=original,zero-shot,many-shot,dynamic-shot"
EXPT_ID="$EXPT_ID" \
SELECTION_EXPT_ID="$SELECTION_EXPT_ID" \
SELECTION_ENGINE="$SELECTION_ENGINE" \
SELECTION_CACHE_JSON="$SELECTION_CACHE_JSON" \
EXPECTED_SELECTION_SOURCE_MODE=matched_retriever_artifact \
DATASETS="finqa_test" \
PROMPT_TYPES="original zero-shot many-shot dynamic-shot" \
LIMIT=-1 \
CONDA_ENV=fnqa \
  bash dist/experiment_7_selection_cache_binding.sh \
  >"$QUEUE_DIR/selection_binding.log" 2>&1
jq -e '.status == "completed"' "$BINDING_ROOT/execution_status.json" >/dev/null
event "selection_binding=completed root=$BINDING_ROOT"

if curl -fsS --max-time 5 "${BASE_URL%/}/models" >/dev/null 2>&1; then
  event "mistral4_server=blocked reason=port_8012_already_in_use"
  exit 2
fi

server_args=(
  --host localhost
  --port 8012
  --model "$MODEL_PATH"
  --alias mistral4
  --ctx-size 8192
  --parallel 1
  --n-gpu-layers 20
  --split-mode row
  --tensor-split 1,1
  --main-gpu 0
  --batch-size 192
  --ubatch-size 48
  --cache-type-k f16
  --cache-type-v f16
  --no-op-offload
  --flash-attn off
  --cache-ram 0
)
{
  printf 'CUDA_VISIBLE_DEVICES=0,1 LD_LIBRARY_PATH=%q ' "$SERVER_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  printf '%q ' "$SERVER_BIN" "${server_args[@]}"
  printf '\n'
} >"$QUEUE_DIR/mistral4_server.command"
CUDA_VISIBLE_DEVICES=0,1 LD_LIBRARY_PATH="$SERVER_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$SERVER_BIN" "${server_args[@]}" \
  >"$QUEUE_DIR/mistral4_server.log" 2>&1 &
server_pid=$!

cleanup() {
  if kill -0 "$server_pid" 2>/dev/null; then
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

for attempt in {1..240}; do
  if curl -fsS --max-time 5 "${BASE_URL%/}/models" >/dev/null 2>&1; then
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    event "mistral4_server=failed pid=$server_pid log=$QUEUE_DIR/mistral4_server.log"
    exit 2
  fi
  sleep 10
done
curl -fsS --max-time 5 "${BASE_URL%/}/models" >/dev/null
event "mistral4_server=ready pid=$server_pid profile=tp2_ngl20_row"

generator_case() {
  local item="$1"
  local execute_mode="$2"
  local regenerate_ea_only="${3:-0}"
  local log_name="${4:-}"
  local case_name
  if [[ -n "$log_name" ]]; then
    case_name="$log_name"
  else
    case_name="$(printf '%s' "$item" | tr ':' '_')"
  fi
  env \
    CONDA_ENV=fnqa \
    EXPT_ID="$EXPT_ID" \
    ENGINES=mistral4 \
    EXPERIMENT7_MATRIX="$item" \
    EXPERIMENT7_SELECTION_EXPT_ID="$SELECTION_EXPT_ID" \
    EXPERIMENT7_SELECTION_ENGINE="$SELECTION_ENGINE" \
    EXAMPLE_SELECTION_CACHE_JSON="$SELECTION_CACHE_JSON" \
    EXPERIMENT7_SELECTION_CACHE_BINDING_ROOT="$BINDING_ROOT" \
    EXPERIMENT7_REQUIRE_TARGET_SELECTION_CACHE=0 \
    EXPERIMENT7_REQUIRE_BINDING_AUDIT=1 \
    EXPERIMENT7_ALLOW_MATERIALIZED_SELECTION_CACHE=1 \
    EXAMPLE_SELECTION_MODE=cache \
    EXAMPLE_SELECTION_REQUIRE_CACHE=1 \
    FORMAL_FINDER_READY=1 \
    RUN_RETRIEVER_INFER=0 \
    STRICT_INPUTS=1 \
    MAX_TOKENS=128 \
    RESUME_OUTPUT=1 \
    SHOW_PROMPT=0 \
    RUN_EXECUTE="$execute_mode" \
    LIMIT=-1 \
    FAIL_FAST_ON_EXECUTE_ERROR=1 \
    GENERATOR_BATCH_SIZE=1 \
    UPDATE_EA_LATEST=1 \
    REGENERATE_EA_ONLY="$regenerate_ea_only" \
    MISTRAL_SMALL_RUNTIME_BACKEND=llama_cpp \
    LLAMA_CPP_BASE_URL="$BASE_URL" \
    LLAMA_CPP_MODEL_PATH="$MODEL_PATH" \
    LLAMA_CPP_MODEL_ALIAS=mistral4 \
    LLAMA_CPP_QUANT=UD-Q4_K_M \
    LLAMA_CPP_CTX_SIZE=8192 \
    LLAMA_CPP_N_GPU_LAYERS=20 \
    LLAMA_CPP_TENSOR_SPLIT=1,1 \
    LLAMA_CPP_SPLIT_MODE=row \
    LLAMA_CPP_PARALLEL=1 \
    LLAMA_CPP_BATCH_SIZE=192 \
    LLAMA_CPP_UBATCH_SIZE=48 \
    LLAMA_CPP_CACHE_TYPE_K=f16 \
    LLAMA_CPP_CACHE_TYPE_V=f16 \
    LLAMA_CPP_OP_OFFLOAD=off \
    LLAMA_CPP_FLASH_ATTN=off \
    LLAMA_CPP_CACHE_RAM=0 \
    LLAMA_CPP_MAIN_GPU=0 \
    MISTRAL4_REASONING_EFFORT=high \
    MISTRAL4_REASONING_TEMPERATURE=0.7 \
    MISTRAL4_REASONING_TOP_P=1.0 \
    LLAMA_CPP_ALLOW_REASONING_EFFORT=1 \
    LOCAL_OPENAI_REQUEST_TIMEOUT_SECONDS="$REQUEST_TIMEOUT_SECONDS" \
      bash dist/experiment_7_generator_answer.sh \
      >>"$QUEUE_DIR/${case_name}.log" 2>&1
}

event "flan_d_test=prepare_for_prefix_recovery"
generator_case finqa_flan_d:finqa_test 0
NEW_CASE_ROOT="$EXPT_DIR/generator/mistral4/finqa_flan_d_finqa_test"
NEW_INPUT="$NEW_CASE_ROOT/generator_input.json"
NEW_OUTPUT="$NEW_CASE_ROOT/mistral4_finqa_test_generated.jsonl"
if [[ ! -f "$NEW_OUTPUT" ]]; then
  NEW_INPUT_PREFIX="$QUEUE_DIR/flan_d_test_input_prefix_${FLAN_D_PREFIX_MAX_ROWS}.json"
  if [[ ! -f "$NEW_INPUT_PREFIX" ]]; then
    jq -c ".[0:${FLAN_D_PREFIX_MAX_ROWS}]" "$NEW_INPUT" >"$NEW_INPUT_PREFIX"
  fi
  conda run --no-capture-output -n fnqa \
    python -B dist/repair_experiment7_output_prefix.py \
    --input-json "$NEW_INPUT_PREFIX" \
    --source-jsonl "$OLD_CASE_ROOT/mistral4_finqa_test_generated.jsonl" \
    --output-jsonl "$NEW_OUTPUT" \
    --report-json "$QUEUE_DIR/flan_d_test_prefix_repair.json" \
    >"$QUEUE_DIR/flan_d_test_prefix_repair.log" 2>&1
fi
recovered_rows="$(jq -r '.recoveredPrefixRows' "$QUEUE_DIR/flan_d_test_prefix_repair.json" 2>/dev/null || printf 'existing')"
source_sha256="$(jq -r '.sourceSha256 // empty' "$QUEUE_DIR/flan_d_test_prefix_repair.json")"
current_source_sha256="$(sha256sum "$OLD_CASE_ROOT/mistral4_finqa_test_generated.jsonl" | awk '{print $1}')"
if [[ "$recovered_rows" != "$FLAN_D_PREFIX_MAX_ROWS" ]] || \
   [[ "$(jq -r '.sourceModified' "$QUEUE_DIR/flan_d_test_prefix_repair.json")" != "false" ]] || \
   [[ -z "$source_sha256" ]] || [[ "$source_sha256" != "$current_source_sha256" ]]; then
  event "flan_d_test=blocked reason=prefix_recovery_guard_failed recovered_rows=$recovered_rows source_sha256=$current_source_sha256"
  exit 2
fi
event "flan_d_test=prefix_ready recovered_rows=$recovered_rows source_unchanged=1"
event "queue=scope dataset=finqa_test matrix=$MATRIX"

completed_matrix=""
for item in $MATRIX; do
  completed=0
  for attempt in $(seq 1 "$MAX_ATTEMPTS"); do
    event "case=start item=$item attempt=$attempt timeout_seconds=$REQUEST_TIMEOUT_SECONDS"
    set +e
    generator_case "$item" 1
    rc=$?
    set -e
    if [[ "$rc" -eq 0 ]]; then
      event "case=completed item=$item attempt=$attempt"
      completed=1
      break
    fi
    event "case=retry item=$item attempt=$attempt rc=$rc"
    sleep 30
  done
  if [[ "$completed" -ne 1 ]]; then
    event "queue=blocked item=$item attempts=$MAX_ATTEMPTS"
    exit 1
  fi

  completed_matrix="${completed_matrix:+$completed_matrix }$item"
  event "ea=refresh item=$item matrix=$completed_matrix"
  set +e
  generator_case "$completed_matrix" 0 1 ea_refresh
  ea_rc=$?
  set -e
  SCORE_REPORT="$EXPT_DIR/generator/score_report.json"
  expected_completed_cases="$(wc -w <<<"$completed_matrix")"
  actual_completed_cases="$(jq -r '.completed_cases // 0' "$SCORE_REPORT")"
  if [[ "$actual_completed_cases" -lt "$expected_completed_cases" ]]; then
    event "queue=blocked item=$item reason=ea_refresh_incomplete rc=$ea_rc expected_completed=$expected_completed_cases actual_completed=$actual_completed_cases"
    exit 1
  fi
  if [[ "$ea_rc" -ne 0 ]]; then
    event "ea=warning item=$item rc=$ea_rc reason=noncompleted_diagnostic_statuses_ignored"
  fi
  current_ea="$(jq -r '.mean_execution_accuracy_unweighted // "null"' "$SCORE_REPORT")"
  completed_cases="$(jq -r '.completed_cases // 0' "$SCORE_REPORT")"
  event "ea=reported item=$item completed_cases=$completed_cases mean_execution_accuracy=$current_ea report=$SCORE_REPORT"

  if [[ -f "$STOP_REQUEST_FILE" ]]; then
    STOP_TIME="$(utc_now)" STOP_ITEM="$item" STOP_REPORT="$SCORE_REPORT" \
    STOP_QUEUE_DIR="$QUEUE_DIR" STOP_REQUEST_FILE="$STOP_REQUEST_FILE" \
      conda run --no-capture-output -n "$CONDA_ENV" python -B - <<'PY_STOP'
import json
import os
from pathlib import Path

report_path = Path(os.environ["STOP_REPORT"])
report = json.loads(report_path.read_text(encoding="utf-8"))
payload = {
    "time": os.environ["STOP_TIME"],
    "status": "stopped_after_current_case",
    "completedItem": os.environ["STOP_ITEM"],
    "completedCases": report.get("completed_cases"),
    "meanExecutionAccuracyUnweighted": report.get("mean_execution_accuracy_unweighted"),
    "scoreReport": str(report_path),
    "stopRequestFile": os.environ["STOP_REQUEST_FILE"],
    "items": report.get("items", []),
}
(Path(os.environ["STOP_QUEUE_DIR"]) / "stop_status.json").write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
PY_STOP
    printf '%s\n' "$(utc_now)" >"$QUEUE_DIR/queue.stopped"
    event "queue=stopped boundary=case completed_item=$item ea=$current_ea report=$SCORE_REPORT"
    exit 0
  fi
done

event "final_audit=start scope=finqa_test_only"
set +e
conda run --no-capture-output -n "$CONDA_ENV" \
  python -B dist/finalize_experiment7_mistral4_testonly.py \
  --experiment-root "$EXPT_DIR" \
  --log-root "$WORKSPACE_ROOT/docs/log" \
  >"$QUEUE_DIR/final_audit.log" 2>&1
final_audit_rc=$?
set -e
if [[ "$final_audit_rc" -ne 0 ]]; then
  event "queue=blocked reason=final_audit_failed rc=$final_audit_rc log=$QUEUE_DIR/final_audit.log"
  exit 1
fi
completion_report="$(jq -r '.report' "$QUEUE_DIR/completion_artifacts.json")"
printf '%s\n' "$(utc_now)" >"$QUEUE_DIR/queue.completed"
event "queue=completed experiment=7 matrix_cases=$(wc -w <<<"$MATRIX") report=$completion_report"
