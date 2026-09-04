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
smoke_root="$output_root/smoke"
config_path="$repo_root/config/experiment6_narrative2_generation.json"
runner="$repo_root/dist/run_experiment6_narrative2_generation.py"
inspector="$repo_root/dist/inspect_experiment6_narrative2_results.py"
reuse_llama_server="${EXPERIMENT6_REUSE_LLAMA_SERVER:-0}"
mkdir -p "$runtime_dir"
cd "$repo_root"

route_value() {
  jq -er "$1" "$config_path"
}

qwen_session="$(route_value '.runtimeRoutes.qwen3_6.session')"
llama_url="$(route_value '.runtimeRoutes.llama4.baseUrl')"
llama_port="$(route_value '.runtimeRoutes.llama4.port')"
llama_session="$(route_value '.runtimeRoutes.llama4.session')"
llama_served_model="$(route_value '.runtimeRoutes.llama4.servedModel')"
llama_snapshot="$(route_value '.runtimeRoutes.llama4.modelPath')"
llama_profile="$(route_value '.runtimeRoutes.llama4.runtimeProfile')"
llama_quant="$(route_value '.runtimeRoutes.llama4.quantization')"
llama_context="$(route_value '.runtimeRoutes.llama4.contextTokens')"
llama_max_num_seqs="$(route_value '.runtimeRoutes.llama4.maxNumSeqs')"
llama_response_format="$(route_value '.runtimeRoutes.llama4.responseFormat')"
llama_response_schema="$(route_value '.runtimeRoutes.llama4.responseSchemaPath')"
mistral_url="$(route_value '.runtimeRoutes.mistral4.baseUrl')"
mistral_port="$(route_value '.runtimeRoutes.mistral4.port')"
mistral_session="$(route_value '.runtimeRoutes.mistral4.session')"
mistral_served_model="$(route_value '.runtimeRoutes.mistral4.servedModel')"
mistral_bin="$(route_value '.runtimeRoutes.mistral4.serverBinary')"
mistral_model="$(route_value '.runtimeRoutes.mistral4.modelPath')"
mistral_profile="$(route_value '.runtimeRoutes.mistral4.runtimeProfile')"
mistral_quant="$(route_value '.runtimeRoutes.mistral4.quantization')"
mistral_context="$(route_value '.runtimeRoutes.mistral4.contextTokens')"
mistral_gpu_layers="$(route_value '.runtimeRoutes.mistral4.nGpuLayers')"
mistral_tensor_split="$(route_value '.runtimeRoutes.mistral4.tensorSplit')"
mistral_split_mode="$(route_value '.runtimeRoutes.mistral4.splitMode')"
mistral_batch="$(route_value '.runtimeRoutes.mistral4.batchSize')"
mistral_ubatch="$(route_value '.runtimeRoutes.mistral4.ubatchSize')"
mistral_response_format="$(route_value '.runtimeRoutes.mistral4.responseFormat')"
mistral_response_schema="$(route_value '.runtimeRoutes.mistral4.responseSchemaPath')"

state_path="$runtime_dir/local_queue.status"
write_state() {
  printf '%s phase=%s status=%s detail=%s\n'     "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" "${3:-}" >> "$state_path"
}
wait_for_rc() {
  local path="$1"
  while [[ ! -f "$path" ]]; do sleep 60; done
}
wait_for_endpoint() {
  local base_url="$1"
  local api_key="${2:-}"
  local attempts="${3:-180}"
  local header=()
  [[ -n "$api_key" ]] && header=(-H "Authorization: Bearer $api_key")
  for ((attempt=1; attempt<=attempts; attempt++)); do
    if curl -fsS "${header[@]}" "${base_url%/}/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 10
  done
  return 1
}
wait_for_gpu_release() {
  for _ in {1..60}; do
    if ! nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | rg -q '[0-9]'; then
      return 0
    fi
    sleep 10
  done
  return 1
}
stop_tmux_session() {
  local session="$1"
  tmux has-session -t "$session" 2>/dev/null && tmux kill-session -t "$session" || true
}
verify_smoke_case() {
  local output_id="$1"
  local source_id="$2"
  local expected_rows="${3:-1}"
  local status_path="$smoke_root/cases/$output_id/run_01/status.json"
  local prediction_path="$smoke_root/cases/$output_id/run_01/predictions.jsonl"
  local raw_path="$smoke_root/cases/$output_id/run_01/raw/$output_id.jsonl"
  jq -e --arg source_id "$source_id" --argjson expected_rows "$expected_rows" '
    .protocol == "experiment6-narrative2-full-v2"
    and .sourceId == $source_id
    and .expectedRows == $expected_rows
    and .acceptedRows == $expected_rows
    and .rejectedRows == 0
    and .runtimeBlockedRows == 0
    and .formatComplianceRate == 1
    and .status == "completed"
    and .responseFormat == "json_schema"
  ' "$status_path" >/dev/null \
    && jq -se --argjson expected_rows "$expected_rows" '
      length == $expected_rows
      and all(.[]; .formatValid == true and .responseFormat == "json_schema")
    ' "$prediction_path" >/dev/null \
    && jq -se --argjson expected_rows "$expected_rows" '
      length == $expected_rows
      and all(.[];
        .status == "completed"
        and .response.requestResponseFormat == "json_schema"
        and .response.finishReasons == ["stop"]
      )
    ' "$raw_path" >/dev/null
}

: > "$state_path"
write_state qwen wait formal_qwen_rc
wait_for_rc "$runtime_dir/formal_qwen.rc"
qwen_rc="$(cat "$runtime_dir/formal_qwen.rc")"
if [[ "$qwen_rc" != "0" ]]; then
  write_state qwen blocked "rc=$qwen_rc"
  exit "$qwen_rc"
fi
write_state qwen completed
stop_tmux_session "$qwen_session"
wait_for_gpu_release || { write_state gpu blocked qwen_release_timeout; exit 2; }

if [[ "$reuse_llama_server" == "1" ]]; then
  write_state llama4 server_reuse_requested
else
  write_state llama4 server_start
  tmux new-session -d -s "$llama_session" -c "$repo_root"   "env ENGINE=llama4 VLLM_PORT=$llama_port VLLM_SERVED_MODEL_NAME=$llama_served_model VLLM_RUNTIME_PROFILE=$llama_profile LLAMA4_W4A16_VLLM_MAX_NUM_SEQS=$llama_max_num_seqs LLAMA4_W4A16_VLLM_MAX_MODEL_LEN=$llama_context LLAMA4_MODEL_PATH=$llama_snapshot CUDA_VISIBLE_DEVICES=0,1 VLLM_TIMELINE_JSONL=$runtime_dir/llama4_vllm_timeline.jsonl bash dist/start_vllm_openai_server.sh > $runtime_dir/llama4_vllm.log 2>&1"
fi
if ! wait_for_endpoint "$llama_url" EMPTY 240; then
  write_state llama4 blocked endpoint_not_ready
  exit 2
fi
[[ "$reuse_llama_server" == "1" ]] && write_state llama4 server_reused
write_state llama4 smoke_start
set +e
env VLLM_BASE_URL="$llama_url" VLLM_API_KEY=EMPTY VLLM_SERVED_MODEL_NAME="$llama_served_model"   VLLM_RUNTIME_PROFILE="$llama_profile" VLLM_QUANTIZATION="$llama_quant" VLLM_MAX_NUM_SEQS="$llama_max_num_seqs"   GENERATOR_RESPONSE_FORMAT="$llama_response_format" GENERATOR_RESPONSE_SCHEMA_PATH="$llama_response_schema"   LLAMA4_MODEL_PATH="$llama_snapshot"   conda run --no-capture-output -n fnqa python -B "$runner"   --config "$config_path" --output-root "$smoke_root" --smoke-only --limit 2 --no-resume --source-id llama4   > "$runtime_dir/smoke_llama4.log" 2>&1
llama_smoke_rc=$?
set -e
printf '%s\n' "$llama_smoke_rc" > "$runtime_dir/smoke_llama4.rc"
if [[ "$llama_smoke_rc" != "0" ]] || ! verify_smoke_case 6_llama_z llama4 2; then
  write_state llama4 blocked "smoke_rc=$llama_smoke_rc"
  exit "$llama_smoke_rc"
fi
write_state llama4 formal_start
set +e
env VLLM_BASE_URL="$llama_url" VLLM_API_KEY=EMPTY VLLM_SERVED_MODEL_NAME="$llama_served_model"   VLLM_RUNTIME_PROFILE="$llama_profile" VLLM_QUANTIZATION="$llama_quant" VLLM_MAX_NUM_SEQS="$llama_max_num_seqs"   GENERATOR_RESPONSE_FORMAT="$llama_response_format" GENERATOR_RESPONSE_SCHEMA_PATH="$llama_response_schema"   LLAMA4_MODEL_PATH="$llama_snapshot"   conda run --no-capture-output -n fnqa python -B "$runner"   --config "$config_path" --output-root "$output_root" --source-id llama4   > "$runtime_dir/formal_llama4.log" 2>&1
llama_rc=$?
set -e
printf '%s\n' "$llama_rc" > "$runtime_dir/formal_llama4.rc"
if [[ "$llama_rc" != "0" ]]; then
  write_state llama4 blocked "formal_rc=$llama_rc"
  exit "$llama_rc"
fi
write_state llama4 completed
stop_tmux_session "$llama_session"
wait_for_gpu_release || { write_state gpu blocked llama_release_timeout; exit 2; }

write_state mistral4 server_start
tmux new-session -d -s "$mistral_session" -c "$repo_root"   "$mistral_bin --host localhost --port $mistral_port --model $mistral_model --alias $mistral_served_model --ctx-size $mistral_context --parallel 1 --n-gpu-layers $mistral_gpu_layers --split-mode $mistral_split_mode --tensor-split $mistral_tensor_split --main-gpu 0 --batch-size $mistral_batch --ubatch-size $mistral_ubatch --cache-type-k f16 --cache-type-v f16 --no-op-offload --flash-attn off --cache-ram 0 > $runtime_dir/mistral4_llama_cpp.log 2>&1"
if ! wait_for_endpoint "$mistral_url" "" 240; then
  write_state mistral4 blocked endpoint_not_ready
  exit 2
fi
write_state mistral4 token_preflight
conda run --no-capture-output -n fnqa python -B dist/preflight_experiment6_service_tokens.py   --config "$config_path" --output "$runtime_dir/mistral4_token_preflight.json"   --base-url "$mistral_url" --source-id mistral4   > "$runtime_dir/mistral4_token_preflight.log" 2>&1
mistral_env=(
  MISTRAL_SMALL_RUNTIME_BACKEND=llama_cpp
  LLAMA_CPP_BASE_URL="$mistral_url"
  LLAMA_CPP_MODEL_PATH="$mistral_model"
  LLAMA_CPP_MODEL_ALIAS="$mistral_served_model"
  LLAMA_CPP_QUANT="$mistral_quant"
  LLAMA_CPP_CTX_SIZE="$mistral_context"
  LLAMA_CPP_N_GPU_LAYERS="$mistral_gpu_layers"
  LLAMA_CPP_TENSOR_SPLIT="$mistral_tensor_split"
  LLAMA_CPP_SPLIT_MODE="$mistral_split_mode"
  LLAMA_CPP_PARALLEL=1
  LLAMA_CPP_BATCH_SIZE="$mistral_batch"
  LLAMA_CPP_UBATCH_SIZE="$mistral_ubatch"
  LLAMA_CPP_CACHE_TYPE_K=f16
  LLAMA_CPP_CACHE_TYPE_V=f16
  LLAMA_CPP_OP_OFFLOAD=off
  LLAMA_CPP_FLASH_ATTN=off
  LLAMA_CPP_CACHE_RAM=0
  GENERATOR_RESPONSE_FORMAT="$mistral_response_format"
  GENERATOR_RESPONSE_SCHEMA_PATH="$mistral_response_schema"
)
write_state mistral4 smoke_start
set +e
env "${mistral_env[@]}" conda run --no-capture-output -n fnqa python -B "$runner"   --config "$config_path" --output-root "$smoke_root" --smoke-only --no-resume --source-id mistral4   > "$runtime_dir/smoke_mistral4.log" 2>&1
mistral_smoke_rc=$?
set -e
printf '%s\n' "$mistral_smoke_rc" > "$runtime_dir/smoke_mistral4.rc"
if [[ "$mistral_smoke_rc" != "0" ]] || ! verify_smoke_case 6_mistral4_z mistral4; then
  write_state mistral4 blocked "smoke_rc=$mistral_smoke_rc"
  exit "$mistral_smoke_rc"
fi
write_state mistral4 formal_start
set +e
env "${mistral_env[@]}" conda run --no-capture-output -n fnqa python -B "$runner"   --config "$config_path" --output-root "$output_root" --source-id mistral4   > "$runtime_dir/formal_mistral4.log" 2>&1
mistral_rc=$?
set -e
printf '%s\n' "$mistral_rc" > "$runtime_dir/formal_mistral4.rc"
if [[ "$mistral_rc" != "0" ]]; then
  write_state mistral4 blocked "formal_rc=$mistral_rc"
  exit "$mistral_rc"
fi
write_state mistral4 completed
stop_tmux_session "$mistral_session"
wait_for_gpu_release || { write_state gpu blocked mistral_release_timeout; exit 2; }

write_state retrievers wait_gpt_and_controls
wait_for_rc "$runtime_dir/formal_controls.rc"
wait_for_rc "$runtime_dir/formal_gpt5.5.rc"
wait_for_rc "$runtime_dir/formal_gpt5.3.rc"
wait_for_rc "$runtime_dir/formal_gpt4.rc"
controls_rc="$(tr -d '[:space:]' < "$runtime_dir/formal_controls.rc")"
gpt55_rc="$(tr -d '[:space:]' < "$runtime_dir/formal_gpt5.5.rc")"
gpt53_rc="$(tr -d '[:space:]' < "$runtime_dir/formal_gpt5.3.rc")"
gpt4_rc="$(tr -d '[:space:]' < "$runtime_dir/formal_gpt4.rc")"
if [[ "$controls_rc" != "0" || "$gpt55_rc" != "0" || \
      "$gpt53_rc" != "0" || "$gpt4_rc" != "0" ]]; then
  conda run --no-capture-output -n fnqa python -B "$inspector" \
    --config "$config_path" --output-root "$output_root" \
    > "$runtime_dir/formal_completion_gate.log" 2>&1 || true
  write_state evaluation blocked_no_ranking \
    "controls_rc=$controls_rc,gpt55_rc=$gpt55_rc,gpt53_rc=$gpt53_rc,gpt4_rc=$gpt4_rc"
  exit 2
fi
write_state retrievers formal_start
set +e
conda run --no-capture-output -n fnqa python -B "$runner"   --config "$config_path" --output-root "$output_root"   --source-id finqa_flan_z --source-id finqa_flan_m --source-id finqa_flan_d   --source-id finqa_mistral_z --source-id finqa_mistral_m --source-id finqa_mistral_d   --source-id finqa_t5gemma2_z --source-id finqa_t5gemma2_m --source-id finqa_t5gemma2_d   --source-id flan_t5_large --source-id mistral_v0_3 --source-id t5gemma_2_1b_1b   > "$runtime_dir/formal_retrievers.log" 2>&1
retriever_rc=$?
set -e
printf '%s\n' "$retriever_rc" > "$runtime_dir/formal_retrievers.rc"
if [[ "$retriever_rc" != "0" ]]; then
  write_state retrievers blocked "formal_rc=$retriever_rc"
  exit "$retriever_rc"
fi
write_state retrievers completed

write_state evaluation completion_gate
set +e
conda run --no-capture-output -n fnqa python -B "$inspector" \
  --config "$config_path" --output-root "$output_root" \
  > "$runtime_dir/formal_completion_gate.log" 2>&1
completion_gate_rc=$?
set -e
printf '%s\n' "$completion_gate_rc" > "$runtime_dir/formal_completion_gate.rc"
progress_report="$output_root/diagnostics/progress_report.json"
if [[ "$completion_gate_rc" != "0" || ! -f "$progress_report" ]] || \
   ! jq -e '
     .status == "completed_ready_for_ranking"
     and .rankingPublished == false
     and .coverage.officialCasesComplete == 54
     and .coverage.officialCaseRunsComplete == 540
     and .coverage.formalPredictionsComplete == 45900
     and .coverage.controlCasesComplete == 4
     and .coverage.controlCaseRunsComplete == 40
     and .coverage.controlPredictionsComplete == 3400
     and .coverage.manifestCount == 580
     and .coverage.expectedManifestCount == 580
     and (.integrityErrors | length) == 0
   ' "$progress_report" >/dev/null; then
  write_state evaluation blocked_no_ranking \
    "completion_gate_rc=$completion_gate_rc,report=$progress_report"
  exit 2
fi
set +e
conda run --no-capture-output -n fnqa python -B dist/evaluate_narrative2_fixed_v2.py   --output-root "$output_root" > "$runtime_dir/formal_evaluation_fixed_v2.log" 2>&1
evaluation_rc=$?
set -e
printf '%s\n' "$evaluation_rc" > "$runtime_dir/formal_evaluation_fixed_v2.rc"
if [[ "$evaluation_rc" != "0" ]]; then
  write_state evaluation blocked_no_ranking "rc=$evaluation_rc"
  exit "$evaluation_rc"
fi
write_state evaluation completed "protocol=narrative2-fixed-python-v2"
