#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/retriever_experiment_lib.sh"
source "${SCRIPT_DIR}/generator_runtime_profiles.sh"
source "${SCRIPT_DIR}/generator_resource_guard.sh"

ENGINES="${ENGINES:-qwen3_6 llama3_3 mistral4}"
LIMIT="${LIMIT:-1}"
MAX_TOKENS="${MAX_TOKENS:-256}"
PROFILE="${PROFILE:-greedy}"
BASE_PORT="${VLLM_BASE_PORT:-8010}"
ENGINE_PORT_STEP="${VLLM_ENGINE_PORT_STEP:-10}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-${VLLM_READY_TIMEOUT_SECONDS:-300}}"
POLL_SECONDS="${VLLM_POLL_SECONDS:-10}"
START_TIMEOUT_SECONDS="${VLLM_START_TIMEOUT_SECONDS:-1800}"
RUN_EXPERIMENT5="${RUN_EXPERIMENT5:-1}"
RUN_EXPERIMENT7="${RUN_EXPERIMENT7:-${RUN_EXPERIMENT6:-1}}"
EXPERIMENT7_MATRIX="${EXPERIMENT7_MATRIX:-${EXPERIMENT6_MATRIX:-finqa_flan_o:finqa_test}}"
PROBE_ID="${PROBE_ID:-$(date -u +%Y%m%dT%H%M%SZ)_vllm_generator_probe}"
PROBE_ROOT="${PROBE_ROOT:-${REPO_ROOT}/Experiment/vllm_servers}"
SUMMARY_JSON="${SUMMARY_JSON:-${PROBE_ROOT}/generator_model_probe_summary.json}"
STOP_EXISTING_VLLM="${STOP_EXISTING_VLLM:-1}"
REQUIRE_WEIGHTS_ACCESS="${REQUIRE_WEIGHTS_ACCESS:-0}"
EXPERIMENT7_MATCHED_FALLBACK="${EXPERIMENT7_MATCHED_FALLBACK:-${EXPERIMENT6_MATCHED_FALLBACK:-${REPO_ROOT}/Experiment/old_finqa_flan_o/retriever/outputs/best_matched_with_retrieved_facts_and_questions.json}}"

GPU_RELEASE_TIMEOUT_SECONDS="${GPU_RELEASE_TIMEOUT_SECONDS:-180}"
GPU_RELEASE_POLL_SECONDS="${GPU_RELEASE_POLL_SECONDS:-5}"
GPU_IDLE_MEMORY_MB="${GPU_IDLE_MEMORY_MB:-256}"
ENGINE_COOLDOWN_SECONDS="${ENGINE_COOLDOWN_SECONDS:-40}"
STOP_ON_CUDA_PREFLIGHT_FAILURE="${STOP_ON_CUDA_PREFLIGHT_FAILURE:-0}"

AUTO_TP_FALLBACK="${VLLM_AUTO_TP_FALLBACK:-1}"
TP_FALLBACK_TARGET="${VLLM_TP_FALLBACK_TARGET:-1}"

VLLM_CONSERVATIVE_GPU_MEMORY_UTILIZATION="${VLLM_CONSERVATIVE_GPU_MEMORY_UTILIZATION:-0.82}"
VLLM_CONSERVATIVE_MAX_NUM_BATCHED_TOKENS="${VLLM_CONSERVATIVE_MAX_NUM_BATCHED_TOKENS:-128}"
VLLM_CONSERVATIVE_MAX_MODEL_LEN="${VLLM_CONSERVATIVE_MAX_MODEL_LEN:-1024}"
VLLM_CONSERVATIVE_MAX_NUM_SEQS="${VLLM_CONSERVATIVE_MAX_NUM_SEQS:-1}"
VLLM_CONSERVATIVE_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_CONSERVATIVE_DISABLE_CUSTOM_ALL_REDUCE:-1}"

mkdir -p "${PROBE_ROOT}"

RESOURCE_REPORT="${PROBE_ROOT}/resource_services_before_probe.tsv"
generator_guard_pid_rows >"${RESOURCE_REPORT}" || true
if [[ -s "${RESOURCE_REPORT}" ]]; then
  printf "Generator service inventory before vLLM probe (report-only; no ChatMock/other service is stopped): %s\n" "${RESOURCE_REPORT}" >&2
  cat "${RESOURCE_REPORT}" >&2
fi

model_for_engine() {
  generator_model_for_engine "$1"
}

profiles_for_engine() {
  probe_profiles_for_engine "$1"
}

apply_runtime_profile() {
  local engine="$1"
  local profile="$2"
  reset_generator_vllm_profile_vars
  apply_generator_runtime_profile "${engine}" "${profile}"
}

engine_port_for() {
  local engine="$1"
  case "${engine}" in
    qwen3_6) printf "%d\n" "${BASE_PORT}" ;;
    llama3_3) printf "%d\n" "$((BASE_PORT + ENGINE_PORT_STEP))" ;;
    mistral4) printf "%d\n" "$((BASE_PORT + ENGINE_PORT_STEP * 2))" ;;
    deepseek|deepseek_r1_qwen32b) printf "%d\n" "$((BASE_PORT + ENGINE_PORT_STEP * 3))" ;;
    qwythos|qwythos9b) printf "%d\n" "$((BASE_PORT + ENGINE_PORT_STEP * 4))" ;;
    llama4|llama4_scout) printf "%d\n" "$((BASE_PORT + ENGINE_PORT_STEP * 5))" ;;
    *) printf "%d\n" "${BASE_PORT}" ;;
  esac
}

capture_nvidia_smi_snapshot() {
  local output_path="$1"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi
  {
    echo "time=$(utc_now)"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
    echo "---"
    nvidia-smi || true
  } >"${output_path}" 2>&1 || true
}

gpu_has_active_compute_processes() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 1
  fi
  local rows
  rows="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | awk 'NF {count++} END {print count+0}')"
  [[ "${rows}" -gt 0 ]]
}

gpu_has_err_state() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 1
  fi
  nvidia-smi 2>/dev/null | rg -q 'ERR!'
}

gpu_memory_above_idle_threshold() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 1
  fi
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | \
    awk -v th="${GPU_IDLE_MEMORY_MB}" '{gsub(/[^0-9]/, "", $1); if (($1+0) > th) above=1} END {exit(above?0:1)}'
}

wait_for_gpu_release() {
  local timeout_seconds="${1:-${GPU_RELEASE_TIMEOUT_SECONDS}}"
  local snapshot_path="${2:-}"
  local deadline=$((SECONDS + timeout_seconds))

  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi

  while (( SECONDS < deadline )); do
    if ! gpu_has_active_compute_processes && ! gpu_memory_above_idle_threshold && ! gpu_has_err_state; then
      [[ -n "${snapshot_path}" ]] && capture_nvidia_smi_snapshot "${snapshot_path}"
      return 0
    fi
    if ! gpu_has_active_compute_processes && ! gpu_memory_above_idle_threshold && gpu_has_err_state; then
      [[ -n "${snapshot_path}" ]] && capture_nvidia_smi_snapshot "${snapshot_path}"
      return 1
    fi
    sleep "${GPU_RELEASE_POLL_SECONDS}"
  done

  [[ -n "${snapshot_path}" ]] && capture_nvidia_smi_snapshot "${snapshot_path}"
  return 1
}

preflight_cuda_health() {
  local output_json="$1"
  local snapshot_path="$2"
  local required_devices="${3:-0}"
  capture_nvidia_smi_snapshot "${snapshot_path}"

  if gpu_has_active_compute_processes; then
    python3 - "${output_json}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
path = Path(sys.argv[1])
payload = {
    "time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "ok": False,
    "reason": "active_compute_processes",
}
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
PY
    return 1
  fi

  set +e
  PREFLIGHT_REQUIRED_DEVICES="${required_devices}" \
  conda run --no-capture-output -n "${CONDA_ENV}" python - "${output_json}" <<'PY'
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

out = Path(sys.argv[1])
payload = {
    "time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "ok": False,
    "reason": "unknown",
    "cuda_available": None,
    "device_count": None,
    "device_checks": [],
    "required_devices": None,
    "tested_device_count": 0,
}

try:
    import torch
except Exception as exc:
    payload["reason"] = "torch_import_failed"
    payload["error"] = str(exc)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    raise

payload["cuda_available"] = bool(torch.cuda.is_available())
payload["device_count"] = int(torch.cuda.device_count()) if payload["cuda_available"] else 0
required = os.environ.get("PREFLIGHT_REQUIRED_DEVICES", "0")
try:
    required = int(required)
except Exception:
    required = 0
if required <= 0:
    required = payload["device_count"]
payload["required_devices"] = required

if not payload["cuda_available"]:
    payload["reason"] = "cuda_not_available"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    raise SystemExit(1)

if payload["device_count"] <= 0:
    payload["reason"] = "no_cuda_devices"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    raise SystemExit(1)

if payload["device_count"] < required:
    payload["reason"] = "insufficient_cuda_devices"
    payload["required_devices"] = required
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    raise SystemExit(1)

ok = True
for i in range(required):
    check = {"device": i, "ok": True, "error": None}
    try:
        torch.cuda.set_device(i)
        _ = torch.cuda.current_device()
        torch.cuda.synchronize(i)
    except Exception as exc:
        check["ok"] = False
        check["error"] = str(exc)
        ok = False
    payload["device_checks"].append(check)

payload["tested_device_count"] = len(payload["device_checks"])
payload["ok"] = ok
payload["reason"] = "ok" if ok else "cuda_set_device_failed"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
if not ok:
    raise SystemExit(1)
PY
  local rc=$?
  set -e
  return "${rc}"
}

cleanup_vllm() {
  if [[ "${STOP_EXISTING_VLLM}" != "1" ]]; then
    return 0
  fi
  local patterns=(
    'vllm.entrypoints.openai.api_server'
    'VLLM::EngineCore'
    'VLLM::Worker'
  )
  local pids pattern
  for pattern in "${patterns[@]}"; do
    pids="$(pgrep -f "${pattern}" || true)"
    if [[ -n "${pids}" ]]; then
      printf "Stopping vLLM process (%s) pid(s): %s\n" "${pattern}" "${pids}" >&2
      kill ${pids} 2>/dev/null || true
    fi
  done
  sleep 10
  for pattern in "${patterns[@]}"; do
    pids="$(pgrep -f "${pattern}" || true)"
    if [[ -n "${pids}" ]]; then
      printf "Force-stopping vLLM process (%s) pid(s): %s\n" "${pattern}" "${pids}" >&2
      kill -9 ${pids} 2>/dev/null || true
    fi
  done
  sleep 5
}

cleanup_vllm_on_exit() {
  if [[ "${PROBE_CLEANUP_ON_EXIT:-1}" == "1" ]]; then
    cleanup_vllm
  fi
}
trap cleanup_vllm_on_exit EXIT

classify_log() {
  local log_file="$1"
  if [[ ! -s "${log_file}" ]]; then
    printf "runtime_blocked\n"
  elif grep -Eqi 'CUDA-capable device\(s\) is/are busy or unavailable|cudaErrorDevicesUnavailable|device\(s\) is/are busy or unavailable' "${log_file}"; then
    printf "cuda_device_unavailable\n"
  elif grep -Eqi 'GatedRepoError|403 Forbidden|gated repo|not in the authorized list' "${log_file}"; then
    printf "hf_access_blocked\n"
  elif grep -Eqi 'out of memory|not enough GPU memory|OutOfMemoryError|CUDA out of memory|OOM' "${log_file}"; then
    printf "oom_blocked\n"
  elif grep -Eqi 'Engine core initialization failed|WorkerProc initialization failed|Traceback|failed' "${log_file}"; then
    printf "runtime_blocked\n"
  else
    printf "runtime_blocked\n"
  fi
}

failure_signature_from_log() {
  local log_file="$1"
  if [[ ! -s "${log_file}" ]]; then
    printf ""
    return 0
  fi
  local sig
  sig="$(rg -n "CUDA-capable device\(s\) is/are busy or unavailable|cudaErrorDevicesUnavailable|device\(s\) is/are busy or unavailable" "${log_file}" -m 1 -i | sed 's/^[0-9]\+://' || true)"
  if [[ -n "${sig}" ]]; then
    printf "%s" "${sig}"
    return 0
  fi
  sig="$(rg -n "out of memory|not enough GPU memory|OutOfMemoryError|CUDA out of memory|OOM" "${log_file}" -m 1 -i | sed 's/^[0-9]\+://' || true)"
  if [[ -n "${sig}" ]]; then
    printf "%s" "${sig}"
    return 0
  fi
  rg -n "Engine core initialization failed|WorkerProc failed to start|RuntimeError|Traceback|Killed" "${log_file}" -m 1 -i | sed 's/^[0-9]\+://' || true
}

write_status() {
  local path="$1"
  local engine="$2"
  local model="$3"
  local status="$4"
  local detail="$5"
  local hf_rc="$6"
  local ready="$7"
  local exp5_rc="$8"
  local exp7_rc="$9"
  local failure_signature="${10:-}"
  local preflight_json="${11:-}"
  local gpu_snapshot="${12:-}"
  local tp_attempt="${13:-}"

  STATUS_PATH="${path}" \
  STATUS_TIME="$(utc_now)" \
  STATUS_ENGINE="${engine}" \
  STATUS_MODEL="${model}" \
  STATUS_STATUS="${status}" \
  STATUS_DETAIL="${detail}" \
  STATUS_HF_RC="${hf_rc}" \
  STATUS_READY="${ready}" \
  STATUS_EXP5_RC="${exp5_rc}" \
  STATUS_EXP7_RC="${exp7_rc}" \
  STATUS_BASE_URL="${VLLM_BASE_URL:-}" \
  STATUS_SERVED_MODEL="${VLLM_SERVED_MODEL_NAME:-}" \
  STATUS_TP="${VLLM_TENSOR_PARALLEL_SIZE:-}" \
  STATUS_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-}" \
  STATUS_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-}" \
  STATUS_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-}" \
  STATUS_KV_CACHE_DTYPE="${VLLM_KV_CACHE_DTYPE:-}" \
  STATUS_QUANTIZATION="${VLLM_QUANTIZATION:-}" \
  STATUS_CPU_OFFLOAD_GB="${VLLM_CPU_OFFLOAD_GB:-}" \
  STATUS_SWAP_SPACE="${VLLM_SWAP_SPACE:-}" \
  STATUS_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-}" \
  STATUS_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-}" \
  STATUS_RUNTIME_PROFILE="${VLLM_RUNTIME_PROFILE:-}" \
  STATUS_DTYPE="${VLLM_DTYPE:-}" \
  STATUS_LOAD_FORMAT="${VLLM_LOAD_FORMAT:-}" \
  STATUS_FAILURE_SIGNATURE="${failure_signature}" \
  STATUS_PREFLIGHT_JSON="${preflight_json}" \
  STATUS_GPU_SNAPSHOT="${gpu_snapshot}" \
  STATUS_TP_ATTEMPT="${tp_attempt}" \
  STATUS_PRECISION_POLICY="$(generator_runtime_precision_policy "${VLLM_RUNTIME_PROFILE:-auto}")" \
  python3 - <<'PYSTATUS'
import json
import os
from pathlib import Path

def int_or_none(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

payload = {
    "time": os.environ["STATUS_TIME"],
    "engine": os.environ["STATUS_ENGINE"],
    "model": os.environ["STATUS_MODEL"],
    "status": os.environ["STATUS_STATUS"],
    "detail": os.environ["STATUS_DETAIL"],
    "hf_access_rc": int_or_none(os.environ.get("STATUS_HF_RC")),
    "endpoint_ready": os.environ.get("STATUS_READY") == "1",
    "experiment_5_rc": int_or_none(os.environ.get("STATUS_EXP5_RC")),
    "experiment_7_rc": int_or_none(os.environ.get("STATUS_EXP7_RC")),
    "failure_signature": os.environ.get("STATUS_FAILURE_SIGNATURE") or None,
    "preflight_json": os.environ.get("STATUS_PREFLIGHT_JSON") or None,
    "gpu_health_snapshot": os.environ.get("STATUS_GPU_SNAPSHOT") or None,
    "tp_attempt": os.environ.get("STATUS_TP_ATTEMPT") or None,
    "client": {
        "vllm_base_url": os.environ.get("STATUS_BASE_URL"),
        "served_model_name": os.environ.get("STATUS_SERVED_MODEL"),
    },
    "runtime_profile": {
        "tensor_parallel_size": os.environ.get("STATUS_TP"),
        "max_model_len": os.environ.get("STATUS_MAX_MODEL_LEN"),
        "max_num_seqs": os.environ.get("STATUS_MAX_NUM_SEQS"),
        "gpu_memory_utilization": os.environ.get("STATUS_GPU_MEMORY_UTILIZATION"),
        "kv_cache_dtype": os.environ.get("STATUS_KV_CACHE_DTYPE"),
        "quantization": os.environ.get("STATUS_QUANTIZATION"),
        "cpu_offload_gb": os.environ.get("STATUS_CPU_OFFLOAD_GB"),
        "swap_space": os.environ.get("STATUS_SWAP_SPACE"),
        "max_num_batched_tokens": os.environ.get("STATUS_MAX_NUM_BATCHED_TOKENS"),
        "disable_custom_all_reduce": os.environ.get("STATUS_DISABLE_CUSTOM_ALL_REDUCE"),
        "name": os.environ.get("STATUS_RUNTIME_PROFILE"),
        "dtype": os.environ.get("STATUS_DTYPE"),
        "load_format": os.environ.get("STATUS_LOAD_FORMAT"),
        "precision_policy": os.environ.get("STATUS_PRECISION_POLICY"),
    },
}
path = Path(os.environ["STATUS_PATH"])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PYSTATUS
}

wait_for_endpoint() {
  local base_url="$1"
  local service_pid="${2:-}"
  local deadline=$((SECONDS + READY_TIMEOUT_SECONDS))
  while (( SECONDS < deadline )); do
    if [[ -n "${service_pid}" ]] && ! kill -0 "${service_pid}" 2>/dev/null; then
      return 1
    fi
    if curl -fsS --connect-timeout 2 --max-time 4 \
      -H "Authorization: Bearer ${VLLM_API_KEY:-EMPTY}" \
      "${base_url%/}/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep "${POLL_SECONDS}"
  done
  return 1
}

apply_conservative_runtime_overrides() {
  export VLLM_GPU_MEMORY_UTILIZATION="${VLLM_CONSERVATIVE_GPU_MEMORY_UTILIZATION}"
  export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_CONSERVATIVE_MAX_NUM_BATCHED_TOKENS}"
  export VLLM_MAX_NUM_SEQS="${VLLM_CONSERVATIVE_MAX_NUM_SEQS}"
  export VLLM_DISABLE_CUSTOM_ALL_REDUCE="${VLLM_CONSERVATIVE_DISABLE_CUSTOM_ALL_REDUCE}"

  local current_max_model_len
  current_max_model_len="${VLLM_MAX_MODEL_LEN:-2048}"
  if [[ "${current_max_model_len}" =~ ^[0-9]+$ ]] && [[ "${VLLM_CONSERVATIVE_MAX_MODEL_LEN}" =~ ^[0-9]+$ ]]; then
    if (( current_max_model_len > VLLM_CONSERVATIVE_MAX_MODEL_LEN )); then
      export VLLM_MAX_MODEL_LEN="${VLLM_CONSERVATIVE_MAX_MODEL_LEN}"
    fi
  else
    export VLLM_MAX_MODEL_LEN="${VLLM_CONSERVATIVE_MAX_MODEL_LEN}"
  fi
}

run_experiments_for_engine() {
  local engine="$1"
  RUN_EXP5_RC=0
  RUN_EXP7_RC=0
  if [[ "${RUN_EXPERIMENT5}" == "1" ]]; then
    set +e
    EXPT_ID="${PROBE_ID}_experiment_5_${engine}" \
    RUN_EXECUTE=1 \
    ENGINE="${engine}" \
    LIMIT="${LIMIT}" \
    MAX_TOKENS="${MAX_TOKENS}" \
    CONDA_ENV="${CONDA_ENV}" \
      bash "${SCRIPT_DIR}/experiment_5_qwen_few10_smoke.sh"
    RUN_EXP5_RC=$?
    set -e
  fi
  if [[ "${RUN_EXPERIMENT7}" == "1" ]]; then
    local override_env=()
    if [[ -z "${MATCHED_JSON_FINQA_FLAN_O_FINQA_TEST:-}" \
      && ! -f "${REPO_ROOT}/Experiment/finqa_flan_o/retriever/outputs/best_matched_with_retrieved_facts_and_questions.json" \
      && -f "${EXPERIMENT7_MATCHED_FALLBACK}" ]]; then
      override_env+=("MATCHED_JSON_FINQA_FLAN_O_FINQA_TEST=${EXPERIMENT7_MATCHED_FALLBACK}")
    fi
    set +e
    env "${override_env[@]}" \
      EXPT_ID="${PROBE_ID}_experiment_7_${engine}" \
      RUN_EXECUTE=1 \
      STRICT_INPUTS=1 \
      LIMIT="${LIMIT}" \
      MAX_TOKENS="${MAX_TOKENS}" \
      SHOW_PROMPT=0 \
      ENGINES="${engine}" \
      PROFILE="${PROFILE}" \
      EXPERIMENT7_MATRIX="${EXPERIMENT7_MATRIX}" \
      CONDA_ENV="${CONDA_ENV}" \
        bash "${SCRIPT_DIR}/experiment_7_generator_answer.sh"
    RUN_EXP7_RC=$?
    set -e
  fi
}

cleanup_vllm
wait_for_gpu_release "${GPU_RELEASE_TIMEOUT_SECONDS}" "${PROBE_ROOT}/startup_gpu_release_snapshot.log" || true
status_files=()

for engine in ${ENGINES}; do
  model="$(model_for_engine "${engine}")"
  engine_dir="${PROBE_ROOT}/${engine}"
  mkdir -p "${engine_dir}"
  hf_log="${engine_dir}/hf_access.log"
  status_json="${engine_dir}/probe_status.json"
  status_files+=("${status_json}")

  printf "\n=== Probing %s (%s) ===\n" "${engine}" "${model}"
  cleanup_vllm
  if ! wait_for_gpu_release "${GPU_RELEASE_TIMEOUT_SECONDS}" "${engine_dir}/gpu_release_before_engine.log"; then
    write_status "${status_json}" "${engine}" "${model}" \
      "runtime_feasibility_blocked" \
      "GPU did not return to idle state before engine start" \
      0 0 0 0 "gpu_release_timeout" "" "${engine_dir}/gpu_release_before_engine.log" ""
    continue
  fi

  hf_args=(--model-id "${model}")
  if [[ "${REQUIRE_WEIGHTS_ACCESS}" == "1" ]]; then
    hf_args+=(--require-weights)
  fi
  set +e
  conda run --no-capture-output -n "${CONDA_ENV}" python -B "${SCRIPT_DIR}/hf_model_access_probe.py" "${hf_args[@]}" >"${hf_log}" 2>&1
  hf_rc=$?
  set -e
  if [[ "${hf_rc}" -ne 0 ]]; then
    status="$(classify_log "${hf_log}")"
    write_status "${status_json}" "${engine}" "${model}" "${status}" "hf access probe failed; see ${hf_log}" "${hf_rc}" 0 0 0 "$(failure_signature_from_log "${hf_log}")" "" "${hf_log}" ""
    continue
  fi

  profile_succeeded=0
  engine_attempt_count=0
  engine_cuda_unavailable_count=0
  last_preflight_json=""
  last_gpu_snapshot=""
  last_tp_attempt=""

  for runtime_profile in $(profiles_for_engine "${engine}"); do
    profile_dir="${engine_dir}/${runtime_profile}"
    mkdir -p "${profile_dir}"

    apply_runtime_profile "${engine}" "${runtime_profile}"
    apply_conservative_runtime_overrides

    export ENGINE="${engine}"
    export VLLM_HOST="${VLLM_HOST:-localhost}"
    export VLLM_PORT="$(engine_port_for "${engine}")"
    export VLLM_API_KEY="${VLLM_API_KEY:-EMPTY}"
    export VLLM_SERVED_MODEL_NAME="${model}"
    export VLLM_BASE_URL="http://${VLLM_HOST}:${VLLM_PORT}/v1"
    case "${engine}" in
      qwen3_6) export QWEN3_6_MODEL="${model}" ;;
      llama3_3) export LLAMA3_3_MODEL="${model}" ;;
      mistral4) export MISTRAL_SMALL_MODEL="${model}" ;;
      deepseek|deepseek_r1_qwen32b) export DEEPSEEK_R1_MODEL="${model}" ;;
      qwythos|qwythos9b) export QWYTHOS_MODEL="${model}" ;;
      llama4|llama4_scout) export LLAMA4_MODEL="${model}" ;;
    esac

    configured_tp="${VLLM_TENSOR_PARALLEL_SIZE:-2}"
    tp_attempts=("${configured_tp}")
    if [[ "${AUTO_TP_FALLBACK}" == "1" && "${configured_tp}" == "2" && "${TP_FALLBACK_TARGET}" != "2" ]]; then
      tp_attempts+=("${TP_FALLBACK_TARGET}")
    fi

    for tp in "${tp_attempts[@]}"; do
      attempt_tag="${runtime_profile}_tp${tp}"
      start_log="${profile_dir}/start_tp${tp}.log"
      attempt_status_json="${profile_dir}/probe_status_tp${tp}.json"
      preflight_json="${profile_dir}/preflight_tp${tp}.json"
      preflight_gpu_snapshot="${profile_dir}/preflight_tp${tp}_nvidia_smi.log"
      failure_gpu_snapshot="${profile_dir}/failure_tp${tp}_nvidia_smi.log"

      export VLLM_TENSOR_PARALLEL_SIZE="${tp}"
      printf "%s\n" "--- Runtime profile: ${runtime_profile} (tp=${tp}) ---"
      printf "precision_policy=%s\n" "$(generator_runtime_precision_policy "${runtime_profile}")"
      printf "base_url=%s\n" "${VLLM_BASE_URL}"

      cleanup_vllm
      wait_for_gpu_release "${GPU_RELEASE_TIMEOUT_SECONDS}" "${profile_dir}/gpu_release_before_${attempt_tag}.log" || true

      if ! preflight_cuda_health "${preflight_json}" "${preflight_gpu_snapshot}" "${tp}"; then
        engine_attempt_count=$((engine_attempt_count + 1))
        engine_cuda_unavailable_count=$((engine_cuda_unavailable_count + 1))
        last_preflight_json="${preflight_json}"
        last_gpu_snapshot="${preflight_gpu_snapshot}"
        last_tp_attempt="${tp}"
        write_status "${attempt_status_json}" "${engine}" "${model}" "cuda_device_unavailable" \
          "preflight_cuda_health failed; see ${preflight_json}" "${hf_rc}" 0 0 0 "preflight_cuda_health_failed" "${preflight_json}" "${preflight_gpu_snapshot}" "${tp}"
        cp "${attempt_status_json}" "${status_json}"
        if [[ "${STOP_ON_CUDA_PREFLIGHT_FAILURE}" == "1" && "${tp}" == "${tp_attempts[${#tp_attempts[@]}-1]}" ]]; then
          break 2
        fi
        continue
      fi

      set +e
      timeout "${START_TIMEOUT_SECONDS}s" bash "${SCRIPT_DIR}/start_vllm_openai_server.sh" >"${start_log}" 2>&1 &
      vllm_pid=$!
      set -e

      if wait_for_endpoint "${VLLM_BASE_URL}" "${vllm_pid}"; then
        run_experiments_for_engine "${engine}"
        if [[ "${RUN_EXPERIMENT5}" == "1" || "${RUN_EXPERIMENT7}" == "1" ]]; then
          if [[ "${RUN_EXP5_RC}" -eq 0 && "${RUN_EXP7_RC}" -eq 0 ]]; then
            final_status="passed_smoke"
            detail="vLLM endpoint ready and requested generator smoke routes completed"
          else
            final_status="runtime_blocked"
            detail="vLLM endpoint ready but one or more generator smoke routes failed"
          fi
        else
          final_status="ready"
          detail="vLLM endpoint ready; smoke execution disabled by RUN_EXPERIMENT5/RUN_EXPERIMENT7"
        fi
        write_status "${attempt_status_json}" "${engine}" "${model}" "${final_status}" "${detail}; profile=${runtime_profile}; tp=${tp}" "${hf_rc}" 1 "${RUN_EXP5_RC}" "${RUN_EXP7_RC}" "" "${preflight_json}" "${preflight_gpu_snapshot}" "${tp}"
        cp "${attempt_status_json}" "${status_json}"
        profile_succeeded=1
        cleanup_vllm
        wait "${vllm_pid}" >/dev/null 2>&1 || true
        break
      fi

      status="$(classify_log "${start_log}")"
      signature="$(failure_signature_from_log "${start_log}")"
      capture_nvidia_smi_snapshot "${failure_gpu_snapshot}"
      engine_attempt_count=$((engine_attempt_count + 1))
      if [[ "${status}" == "cuda_device_unavailable" ]]; then
        engine_cuda_unavailable_count=$((engine_cuda_unavailable_count + 1))
      fi
      last_preflight_json="${preflight_json}"
      last_gpu_snapshot="${failure_gpu_snapshot}"
      last_tp_attempt="${tp}"
      write_status "${attempt_status_json}" "${engine}" "${model}" "${status}" \
        "vLLM endpoint did not become ready; see ${start_log}; profile=${runtime_profile}; tp=${tp}" \
        "${hf_rc}" 0 0 0 "${signature}" "${preflight_json}" "${failure_gpu_snapshot}" "${tp}"
      cp "${attempt_status_json}" "${status_json}"
      cleanup_vllm
      wait "${vllm_pid}" >/dev/null 2>&1 || true

      if [[ "${status}" != "cuda_device_unavailable" ]]; then
        break
      fi
    done

    if [[ "${profile_succeeded}" == "1" ]]; then
      break
    fi
  done

  if [[ "${profile_succeeded}" != "1" && ! -f "${status_json}" ]]; then
    write_status "${status_json}" "${engine}" "${model}" "runtime_blocked" "no runtime profile completed" "${hf_rc:-0}" 0 0 0 "" "" "" ""
  fi

  if [[ "${profile_succeeded}" != "1" && "${engine_attempt_count}" -gt 0 && "${engine_attempt_count}" -eq "${engine_cuda_unavailable_count}" ]]; then
    write_status "${status_json}" "${engine}" "${model}" "runtime_feasibility_blocked" \
      "all runtime attempts failed with cuda_device_unavailable; classify as host feasibility blocker" \
      "${hf_rc:-0}" 0 0 0 "cudaErrorDevicesUnavailable_repeated" "${last_preflight_json}" "${last_gpu_snapshot:-${engine_dir}/nvidia_smi_after.log}" "${last_tp_attempt}"
  fi

  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi >"${engine_dir}/nvidia_smi_after.log" 2>&1 || true
  fi

  if [[ "${profile_succeeded}" == "1" || "${engine_attempt_count}" -ne "${engine_cuda_unavailable_count}" ]]; then
    sleep "${ENGINE_COOLDOWN_SECONDS}"
  fi
done

SUMMARY_PATH="${SUMMARY_JSON}" python3 - "${status_files[@]}" <<'PYSUMMARY'
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

items = []
for raw in sys.argv[1:]:
    path = Path(raw)
    if path.is_file():
        items.append(json.loads(path.read_text(encoding="utf-8")))
counts = Counter(item.get("status", "unknown") for item in items)
payload = {
    "time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "kind": "vllm_generator_model_probe_summary",
    "status_counts": dict(counts),
    "items": items,
}
path = Path(os.environ["SUMMARY_PATH"])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PYSUMMARY

cleanup_vllm
wait_for_gpu_release "${GPU_RELEASE_TIMEOUT_SECONDS}" "${PROBE_ROOT}/shutdown_gpu_release_snapshot.log" || true
