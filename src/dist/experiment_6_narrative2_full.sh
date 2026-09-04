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
local_queue="$repo_root/dist/experiment_6_narrative2_local_queue.sh"
mkdir -p "$runtime_dir"
cd "$repo_root"

route_value() {
  jq -r "$1 | if . == null then error(\"missing route value: $1\") else . end" \
    "$config_path"
}
write_state() {
  printf '%s phase=%s status=%s detail=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$1" "$2" "${3:-}" >> "$runtime_dir/coordinator.status"
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
session_exists() {
  tmux has-session -t "$1" 2>/dev/null
}
rc_completed() {
  [[ -f "$1" ]] && [[ "$(tr -d '[:space:]' < "$1")" == "0" ]]
}
launch_job() {
  local session="$1"
  local rc_path="$2"
  local command="$3"
  if session_exists "$session"; then
    write_state "$session" active
    return 0
  fi
  if rc_completed "$rc_path"; then
    write_state "$session" completed_resume
    return 0
  fi
  tmux new-session -d -s "$session" -c "$repo_root" \
    "set +e; $command; job_rc=\$?; printf '%s\\n' \"\$job_rc\" > $rc_path; exit \"\$job_rc\""
  write_state "$session" launched
}

chatmock_url="$(route_value '.runtimeRoutes.chatmock.baseUrl')"
chatmock_port="$(route_value '.runtimeRoutes.chatmock.port')"
chatmock_session="$(route_value '.runtimeRoutes.chatmock.session')"
qwen_url="$(route_value '.runtimeRoutes.qwen3_6.baseUrl')"
qwen_port="$(route_value '.runtimeRoutes.qwen3_6.port')"
qwen_session="$(route_value '.runtimeRoutes.qwen3_6.session')"
qwen_served_model="$(route_value '.runtimeRoutes.qwen3_6.servedModel')"
qwen_model_path="$(route_value '.runtimeRoutes.qwen3_6.modelPath')"
qwen_profile="$(route_value '.runtimeRoutes.qwen3_6.runtimeProfile')"
qwen_quant="$(route_value '.runtimeRoutes.qwen3_6.quantization')"
qwen_context="$(route_value '.runtimeRoutes.qwen3_6.contextTokens')"
qwen_enable_thinking="$(route_value '.runtimeRoutes.qwen3_6.enableThinking')"
qwen_response_format="$(route_value '.runtimeRoutes.qwen3_6.responseFormat')"
qwen_response_schema="$(route_value '.runtimeRoutes.qwen3_6.responseSchemaPath')"

: > "$runtime_dir/coordinator.status"
write_state coordinator started "$output_root"

write_state preflight running
conda run --no-capture-output -n fnqa python -B "$runner" \
  --config "$config_path" --output-root "$output_root" --preflight-only \
  > "$runtime_dir/preflight.log" 2>&1
write_state preflight completed

if ! wait_for_endpoint "$chatmock_url" "" 1; then
  if ! session_exists "$chatmock_session"; then
    tmux new-session -d -s "$chatmock_session" -c "$repo_root" \
      "CHATMOCK_PORT=$chatmock_port CHATMOCK_REASONING_EFFORT=medium CHATMOCK_REASONING_SUMMARY=none bash dist/start_chatmock_server.sh > $runtime_dir/chatmock.log 2>&1"
  fi
  wait_for_endpoint "$chatmock_url" "" 180
fi
write_state chatmock ready "gpt-5.5 medium"

if [[ "${EXPERIMENT6_SKIP_ROUTE_SMOKE:-0}" != "1" ]]; then
  write_state smoke_family running
  set +e
  conda run --no-capture-output -n fnqa python -B "$runner" \
    --config "$config_path" --output-root "$smoke_root" --smoke-only \
    --case 6_flan_z --case 6_flan_m --case 6_flan_d \
    --case 6_mistral_z --case 6_mistral_m --case 6_mistral_d \
    --case 6_t5gemma2_z --case 6_t5gemma2_m --case 6_t5gemma2_d \
    --case 6_flan_base_z --case 6_mistral_base_z --case 6_t5gemma2_base_z \
    > "$runtime_dir/smoke_family.log" 2>&1
  family_smoke_rc=$?
  set -e
  printf '%s\n' "$family_smoke_rc" > "$runtime_dir/smoke_family.rc"
  [[ "$family_smoke_rc" == "0" ]] || { write_state smoke_family blocked "rc=$family_smoke_rc"; exit "$family_smoke_rc"; }
  write_state smoke_family completed

  write_state smoke_gpt5_5 running
  env CHATMOCK_BASE_URL="$chatmock_url" CHATMOCK_GPT5_5_MODEL=gpt-5.5 \
    GPT5_5_CODEX_ROUTE=chatmock ALLOW_DIAGNOSTIC_CHATMOCK_FORMAL=1 \
    conda run --no-capture-output -n fnqa python -B "$runner" \
    --config "$config_path" --output-root "$smoke_root" --smoke-only --case 6_gpt5.5_z \
    > "$runtime_dir/smoke_gpt5.5.log" 2>&1
  write_state smoke_gpt5_5 completed

  write_state smoke_gpt5_3 running
  env CODEX_CLI_ASSUME_AUTH=1 CODEX_CLI_SERVICE_TIER=fast CODEX_CLI_DISABLED_FEATURES=image_generation GPT5_3_CODEX_ROUTE=codex_cli \
    conda run --no-capture-output -n fnqa python -B "$runner" \
    --config "$config_path" --output-root "$smoke_root" --smoke-only --case 6_gpt5.3-CodexS_z \
    > "$runtime_dir/smoke_gpt5.3.log" 2>&1
  write_state smoke_gpt5_3 completed

  write_state smoke_gpt4_1 running
  set +e
  conda run --no-capture-output -n fnqa python -B "$runner" \
    --config "$config_path" --output-root "$smoke_root" --smoke-only --case 6_gpt4.1_z \
    > "$runtime_dir/smoke_gpt4.1.log" 2>&1
  gpt4_smoke_rc=$?
  set -e
  printf '%s\n' "$gpt4_smoke_rc" > "$runtime_dir/smoke_gpt4.1.rc"
  if [[ "$gpt4_smoke_rc" == "0" ]]; then
    write_state smoke_gpt4_1 completed
  else
    write_state smoke_gpt4_1 runtime_blocked "rc=$gpt4_smoke_rc"
  fi
else
  write_state smoke skipped explicit_resume_override
fi

if ! wait_for_endpoint "$qwen_url" EMPTY 1; then
  wait_for_gpu_release || { write_state qwen blocked gpu_busy; exit 2; }
  if ! session_exists "$qwen_session"; then
    tmux new-session -d -s "$qwen_session" -c "$repo_root" \
      "env ENGINE=qwen3_6 VLLM_PORT=$qwen_port VLLM_SERVED_MODEL_NAME=$qwen_served_model VLLM_RUNTIME_PROFILE=$qwen_profile QWEN_VLLM_MAX_MODEL_LEN=$qwen_context QWEN3_6_MODEL_PATH=$qwen_model_path CUDA_VISIBLE_DEVICES=0,1 VLLM_TIMELINE_JSONL=$runtime_dir/qwen_vllm_timeline.jsonl bash dist/start_vllm_openai_server.sh > $runtime_dir/qwen_vllm.log 2>&1"
  fi
  wait_for_endpoint "$qwen_url" EMPTY 240
fi
write_state qwen ready "$qwen_profile"

if [[ "${EXPERIMENT6_SKIP_ROUTE_SMOKE:-0}" != "1" ]]; then
  env VLLM_BASE_URL="$qwen_url" VLLM_API_KEY=EMPTY VLLM_SERVED_MODEL_NAME="$qwen_served_model" \
    VLLM_RUNTIME_PROFILE="$qwen_profile" VLLM_QUANTIZATION="$qwen_quant" QWEN3_6_ENABLE_THINKING="$qwen_enable_thinking" GENERATOR_RESPONSE_FORMAT="$qwen_response_format" GENERATOR_RESPONSE_SCHEMA_PATH="$qwen_response_schema" QWEN3_6_MODEL_PATH="$qwen_model_path" \
    conda run --no-capture-output -n fnqa python -B "$runner" \
    --config "$config_path" --output-root "$smoke_root" --smoke-only --case 6_qwen_z \
    > "$runtime_dir/smoke_qwen.log" 2>&1
  write_state smoke_qwen completed
fi

launch_job exp6v2_formal_qwen "$runtime_dir/formal_qwen.rc" \
  "env VLLM_BASE_URL=$qwen_url VLLM_API_KEY=EMPTY VLLM_SERVED_MODEL_NAME=$qwen_served_model VLLM_RUNTIME_PROFILE=$qwen_profile VLLM_QUANTIZATION=$qwen_quant QWEN3_6_ENABLE_THINKING=$qwen_enable_thinking GENERATOR_RESPONSE_FORMAT=$qwen_response_format GENERATOR_RESPONSE_SCHEMA_PATH=$qwen_response_schema QWEN3_6_MODEL_PATH=$qwen_model_path conda run --no-capture-output -n fnqa python -B $runner --config $config_path --output-root $output_root --case 6_qwen_z --case 6_qwen_m --case 6_qwen_d --case 6_FinFlier_qwen > $runtime_dir/formal_qwen.log 2>&1"

launch_job exp6v2_formal_controls "$runtime_dir/formal_controls.rc" \
  "conda run --no-capture-output -n fnqa python -B $runner --config $config_path --output-root $output_root --case control_converter_original --case control_converter_zero --case control_converter_many --case control_converter_dynamic > $runtime_dir/formal_controls.log 2>&1"

if ! session_exists exp6v2_formal_gpt && \
   { ! rc_completed "$runtime_dir/formal_gpt5.5.rc" || ! rc_completed "$runtime_dir/formal_gpt5.3.rc"; }; then
  tmux new-session -d -s exp6v2_formal_gpt -c "$repo_root" \
    "while [[ ! -f $runtime_dir/formal_controls.rc ]]; do sleep 30; done; control_rc=\$(tr -d '[:space:]' < $runtime_dir/formal_controls.rc); [[ \"\$control_rc\" == 0 ]] || exit \"\$control_rc\"; set +e; env CHATMOCK_BASE_URL=$chatmock_url CHATMOCK_GPT5_5_MODEL=gpt-5.5 GPT5_5_CODEX_ROUTE=chatmock ALLOW_DIAGNOSTIC_CHATMOCK_FORMAL=1 conda run --no-capture-output -n fnqa python -B $runner --config $config_path --output-root $output_root --case 6_gpt5.5_z --case 6_gpt5.5_m --case 6_gpt5.5_d --case 6_FinFlier_gpt5.5 > $runtime_dir/formal_gpt5.5.log 2>&1; gpt55_rc=\$?; printf '%s\\n' \"\$gpt55_rc\" > $runtime_dir/formal_gpt5.5.rc; env CODEX_CLI_ASSUME_AUTH=1 CODEX_CLI_SERVICE_TIER=fast CODEX_CLI_DISABLED_FEATURES=image_generation GPT5_3_CODEX_ROUTE=codex_cli conda run --no-capture-output -n fnqa python -B $runner --config $config_path --output-root $output_root --case 6_gpt5.3-CodexS_z --case 6_gpt5.3-CodexS_m --case 6_gpt5.3-CodexS_d --case 6_FinFlier_gpt5.3-CodexS > $runtime_dir/formal_gpt5.3.log 2>&1; gpt53_rc=\$?; printf '%s\\n' \"\$gpt53_rc\" > $runtime_dir/formal_gpt5.3.rc; [[ \"\$gpt55_rc\" == 0 && \"\$gpt53_rc\" == 0 ]]"
  write_state exp6v2_formal_gpt launched
fi

launch_job exp6v2_formal_gpt4 "$runtime_dir/formal_gpt4.rc" \
  "conda run --no-capture-output -n fnqa python -B $runner --config $config_path --output-root $output_root --source-id gpt4_1 > $runtime_dir/formal_gpt4.log 2>&1"

launch_job exp6v2_local_queue "$runtime_dir/formal_local_queue.rc" \
  "bash $local_queue $output_root > $runtime_dir/formal_local_queue.log 2>&1"

write_state coordinator scheduled "ranking withheld until evaluator completion gate passes"
printf 'Experiment 6 v2 phases scheduled under %s\n' "$output_root"
