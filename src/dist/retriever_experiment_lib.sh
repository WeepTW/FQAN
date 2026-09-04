#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/fnqa_paths.sh"
REPO_ROOT="${FQAN_SRC_ROOT}"
WORKSPACE_ROOT="${FQAN_ROOT}"

CONDA_ENV="${CONDA_ENV:-fnqa}"
CUDA_DEVICES="${CUDA_DEVICES:-${CUDA_DEVICE:-0,1}}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
DATA_JSON="${DATA_JSON:-${FQAN_CODE_ROOT}/Data/Data_Target_Module/Finqa/finqa_test_with_table_text.json}"
RETRIEVER_FEW_CSV="${RETRIEVER_FEW_CSV:-}"
DISTRIBUTED_PROBE_TIMEOUT_SECONDS="${DISTRIBUTED_PROBE_TIMEOUT_SECONDS:-60}"
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-${PYTORCH_CUDA_ALLOC_CONF}}"
# Prefer the current PyTorch NCCL variable. If callers explicitly export the
# legacy NCCL_ASYNC_ERROR_HANDLING it will still be inherited by child commands.
NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-}"
TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG:-DETAIL}"
ACCELERATE_NUM_MACHINES="${ACCELERATE_NUM_MACHINES:-1}"
ACCELERATE_MACHINE_RANK="${ACCELERATE_MACHINE_RANK:-0}"
ACCELERATE_MAIN_PROCESS_IP="${ACCELERATE_MAIN_PROCESS_IP:-localhost}"
ACCELERATE_MAIN_PROCESS_PORT="${ACCELERATE_MAIN_PROCESS_PORT:-29501}"
ACCELERATE_RDZV_BACKEND="${ACCELERATE_RDZV_BACKEND:-c10d}"
INFER_PARALLEL_GPU="${INFER_PARALLEL_GPU:-1}"
MATCH_PARALLEL_GPU="${MATCH_PARALLEL_GPU:-1}"
MATCH_CUDA_DEVICES="${MATCH_CUDA_DEVICES:-${CUDA_DEVICES}}"
MATCH_EMBED_BATCH_SIZE="${MATCH_EMBED_BATCH_SIZE:-1024}"

utc_now() {
  date -u +"%Y-%m-%dT%H:%M:%SZ"
}

prompt_data_dir() {
  case "$1" in
    raw|raw-finqa|raw_finqa|finqa-raw|finqa_raw) printf "%s\n" "${WORKSPACE_ROOT}/data/src/FINDER" ;;
    original) printf "%s\n" "${WORKSPACE_ROOT}/data/finqa_original" ;;
    zero-shot|zero_shot) printf "%s\n" "${WORKSPACE_ROOT}/data/finqa_zero_shot" ;;
    many-shot|many_shot) printf "%s\n" "${WORKSPACE_ROOT}/data/finqa_many_shot" ;;
    dynamic-shot|dynamic_shot) printf "%s\n" "${WORKSPACE_ROOT}/data/finqa_dynamic_shot" ;;
    *)
      printf "Unsupported prompt mode: %s\n" "$1" >&2
      return 2
      ;;
  esac
}

prompt_train_csv() {
  local prompt_mode="$1"
  if [[ -n "${RETRIEVER_FEW_CSV}" ]]; then
    printf "%s\n" "${RETRIEVER_FEW_CSV}"
    return 0
  fi
  printf "%s/finqa_train_rel_fact_instruction.csv\n" "$(prompt_data_dir "${prompt_mode}")"
}

prompt_dev_csv() {
  local prompt_mode="$1"
  if [[ -n "${RETRIEVER_FEW_CSV}" ]]; then
    printf "%s\n" "${RETRIEVER_FEW_CSV}"
    return 0
  fi
  printf "%s/finqa_dev_rel_fact_instruction.csv\n" "$(prompt_data_dir "${prompt_mode}")"
}

prompt_test_csv() {
  local prompt_mode="$1"
  if [[ -n "${RETRIEVER_FEW_CSV}" ]]; then
    printf "%s\n" "${RETRIEVER_FEW_CSV}"
    return 0
  fi
  printf "%s/finqa_test_rel_fact_instruction.csv\n" "$(prompt_data_dir "${prompt_mode}")"
}

prompt_csv_for_split() {
  local prompt_mode="$1"
  local split="$2"
  case "${split}" in
    train) prompt_train_csv "${prompt_mode}" ;;
    dev|eval|validation) prompt_dev_csv "${prompt_mode}" ;;
    test) prompt_test_csv "${prompt_mode}" ;;
    *)
      printf "Unsupported prompt CSV split: %s\n" "${split}" >&2
      return 2
      ;;
  esac
}

first_existing_path() {
  local candidate
  local fallback=""
  for candidate in "$@"; do
    if [[ -z "${fallback}" ]]; then
      fallback="${candidate}"
    fi
    if [[ -f "${candidate}" ]]; then
      printf "%s\n" "${candidate}"
      return 0
    fi
  done
  printf "%s\n" "${fallback}"
}

require_file() {
  if [[ ! -f "$1" ]]; then
    printf "Missing required file: %s\n" "$1" >&2
    return 2
  fi
}

build_match_data_json_from_csv() {
  local input_csv="$1"
  local output_json="$2"
  conda run --no-capture-output -n "${CONDA_ENV}" \
    python -B "${SCRIPT_DIR}/build_retriever_few_data_json.py" \
    --input-csv "${input_csv}" \
    --output-json "${output_json}"
}

validate_prompt_mode_contract() {
  local prompt_mode="$1"
  local train_csv="$2"
  local eval_csv="$3"
  local test_csv="${4:-}"
  local -a cmd=(
    conda run --no-capture-output -n "${CONDA_ENV}"
    python -B "${SCRIPT_DIR}/prompt_mode_contract_smoke.py"
    --prompt-mode "${prompt_mode}"
    --train-csv "${train_csv}"
    --eval-csv "${eval_csv}"
  )
  if [[ -n "${test_csv}" ]]; then
    cmd+=(--test-csv "${test_csv}")
  fi
  "${cmd[@]}"
}

is_original_prompt_mode() {
  case "$1" in
    original|orig|raw|raw-finqa|raw_finqa|finqa-raw|finqa_raw) return 0 ;;
    *) return 1 ;;
  esac
}

prompt_mode_suffix() {
  case "$1" in
    raw|raw-finqa|raw_finqa|finqa-raw|finqa_raw) printf "r\n" ;;
    original|orig) printf "o\n" ;;
    zero-shot|zero_shot) printf "z\n" ;;
    many-shot|many_shot) printf "m\n" ;;
    dynamic-shot|dynamic_shot) printf "d\n" ;;
    *)
      printf "Unsupported prompt mode for artifact suffix: %s\n" "$1" >&2
      return 2
      ;;
  esac
}

epochs_for_prompt_mode() {
  local prompt_mode="$1"
  local original_epochs="$2"
  local non_original_epochs="${3:-}"
  if is_original_prompt_mode "${prompt_mode}"; then
    printf "%s\n" "${original_epochs}"
    return 0
  fi
  if [[ -n "${non_original_epochs}" ]]; then
    printf "%s\n" "${non_original_epochs}"
    return 0
  fi
  printf "%s\n" "${original_epochs}"
}

require_adapter_artifact() {
  local adapter_dir="$1"
  local route_name="$2"
  local prompt_mode="${3:-unknown}"
  local expt_id="${4:-unknown}"
  if [[ ! -d "${adapter_dir}" ]]; then
    printf "%s inference cannot start: missing adapter directory.\n" "${route_name}" >&2
    printf "prompt_mode=%s expt_id=%s adapter_dir=%s\n" "${prompt_mode}" "${expt_id}" "${adapter_dir}" >&2
    printf "Run training first, or point the script at an existing adapter artifact.\n" >&2
    return 2
  fi
  if [[ ! -f "${adapter_dir}/adapter_config.json" ]]; then
    printf "%s inference cannot start: adapter_config.json is missing.\n" "${route_name}" >&2
    printf "prompt_mode=%s expt_id=%s adapter_dir=%s\n" "${prompt_mode}" "${expt_id}" "${adapter_dir}" >&2
    printf "The adapter artifact is incomplete; rerun training or verify the adapter path.\n" >&2
    return 2
  fi
}

backup_existing_log() {
  local log_file="$1"
  if [[ -s "${log_file}" ]]; then
    local backup_file="${log_file}.$(date -u +"%Y%m%dT%H%M%SZ").bak"
    cp -p "${log_file}" "${backup_file}"
    printf "Backed up existing log: %s\n" "${backup_file}" >&2
  fi
}

conda_cuda13_library_dirs() {
  conda run --no-capture-output -n "${CONDA_ENV}" python -B - <<'PYCUDA'
from pathlib import Path
import site

candidates = []
for base in site.getsitepackages():
    root = Path(base)
    candidates.extend([
        root / "nvidia" / "cu13" / "lib",
        root / "nvidia" / "nvjitlink" / "lib",
    ])
print(":".join(str(path) for path in candidates if path.exists()))
PYCUDA
}

prepend_library_path() {
  local extra="$1"
  local current="${2:-}"
  if [[ -z "${extra}" ]]; then
    printf "%s\n" "${current}"
  elif [[ -z "${current}" ]]; then
    printf "%s\n" "${extra}"
  else
    printf "%s:%s\n" "${extra}" "${current}"
  fi
}

mistral_retriever_ld_library_path() {
  prepend_library_path "$(conda_cuda13_library_dirs)" "${LD_LIBRARY_PATH:-}"
}

run_logged() {
  local log_file="$1"
  shift
  mkdir -p "$(dirname "${log_file}")"
  backup_existing_log "${log_file}"
  {
    printf "started_at=%s\n" "$(utc_now)"
    printf "cwd=%q\n" "$(pwd)"
    printf "cuda_visible_devices=%q\n" "${CUDA_VISIBLE_DEVICES:-${CUDA_DEVICES:-}}"
    printf "pytorch_cuda_alloc_conf=%q\n" "${PYTORCH_CUDA_ALLOC_CONF:-}"
    printf "pytorch_alloc_conf=%q\n" "${PYTORCH_ALLOC_CONF:-}"
    printf "nccl_async_error_handling=%q\n" "${NCCL_ASYNC_ERROR_HANDLING:-}"
    printf "torch_nccl_async_error_handling=%q\n" "${TORCH_NCCL_ASYNC_ERROR_HANDLING:-}"
    printf "torch_distributed_debug=%q\n" "${TORCH_DISTRIBUTED_DEBUG:-}"
    printf "accelerate_main_process_ip=%q\n" "${ACCELERATE_MAIN_PROCESS_IP:-}"
    printf "accelerate_main_process_port=%q\n" "${ACCELERATE_MAIN_PROCESS_PORT:-}"
    printf "accelerate_rdzv_backend=%q\n" "${ACCELERATE_RDZV_BACKEND:-}"
    if [[ -n "${HF_TOKEN:-}" ]]; then
      printf "hf_token_env_present=1\n"
    else
      printf "hf_token_env_present=0\n"
    fi
    if conda run --no-capture-output -n "${CONDA_ENV}" \
      python -B -c 'import os; from huggingface_hub import get_token; raise SystemExit(0 if os.environ.get("HF_TOKEN") or get_token() else 1)' \
      >/dev/null 2>&1; then
      printf "hf_token_available=1\n"
    else
      printf "hf_token_available=0\n"
    fi
    if [[ -n "${RETRIEVER_SPLIT_SUMMARY:-}" ]]; then
      printf "retriever_split_summary=%s\n" "${RETRIEVER_SPLIT_SUMMARY}"
    fi
    printf "command="
    printf "%q " "$@"
    printf "\n"
    set +e
    "$@"
    local rc=$?
    set -e
    printf "finished_at=%s\n" "$(utc_now)"
    printf "exit_code=%s\n" "${rc}"
    exit "${rc}"
  } 2>&1 | tee "${log_file}"
  return "${PIPESTATUS[0]}"
}

visible_device_count() {
  local devices="${1//[[:space:]]/}"
  if [[ -z "${devices}" ]]; then
    printf "0\n"
    return
  fi
  if [[ "${devices}" == "all" ]]; then
    nvidia-smi -L 2>/dev/null | wc -l
    return
  fi
  local count=1
  local rest="${devices}"
  while [[ "${rest}" == *,* ]]; do
    count=$((count + 1))
    rest="${rest#*,}"
  done
  printf "%s\n" "${count}"
}

require_distributed_device_count() {
  local devices="$1"
  local processes="$2"
  local route_name="$3"
  local count
  count="$(visible_device_count "${devices}")"
  if [[ "${processes}" -lt 1 ]]; then
    printf "%s requires a positive process count; got %s\n" "${route_name}" "${processes}" >&2
    return 2
  fi
  if [[ "${count}" -ne "${processes}" ]]; then
    printf "%s requires CUDA_VISIBLE_DEVICES count to match process count; devices=%s count=%s processes=%s\n" \
      "${route_name}" "${devices}" "${count}" "${processes}" >&2
    return 2
  fi
}

loaded_nvidia_module_version() {
  if [[ -r /proc/driver/nvidia/version ]]; then
    sed -n 's/^NVRM version:.*Kernel Module  \([^[:space:]]*\).*/\1/p' /proc/driver/nvidia/version | head -n 1
  fi
}

installed_nvidia_module_version() {
  if command -v modinfo >/dev/null 2>&1; then
    modinfo -F version nvidia 2>/dev/null | head -n 1
  fi
}

require_nvidia_smi_ready() {
  local route_name="$1"
  local output
  if ! output="$(nvidia-smi 2>&1)"; then
    printf "%s cannot start because NVIDIA NVML is not healthy.\n" "${route_name}" >&2
    printf "%s\n" "${output}" >&2
    printf "loaded_nvidia_kernel_module_version=%s\n" "$(loaded_nvidia_module_version || true)" >&2
    printf "installed_nvidia_module_version=%s\n" "$(installed_nvidia_module_version || true)" >&2
    printf "Fix the host driver/library mismatch first, then retry. Usually this requires restarting the NVIDIA driver stack or rebooting the machine after a driver update.\n" >&2
    return 2
  fi
}

write_gpu_state() {
  require_nvidia_smi_ready "GPU preflight" || return $?
  nvidia-smi
  nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv
  nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv
}

run_gpu_preflight_log() {
  local log_file="$1"
  run_logged "${log_file}" write_gpu_state
}

run_torchrun_distributed_probe() {
  local devices="$1"
  local processes="$2"
  local log_file="$3"
  require_distributed_device_count "${devices}" "${processes}" "torchrun distributed probe"
  require_nvidia_smi_ready "torchrun distributed probe"
  run_logged "${log_file}" \
    env CUDA_VISIBLE_DEVICES="${devices}" \
      PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
      PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF}" \
      TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING}" \
      TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG}" \
      DISTRIBUTED_PROBE_TIMEOUT_SECONDS="${DISTRIBUTED_PROBE_TIMEOUT_SECONDS}" \
    timeout "${DISTRIBUTED_PROBE_TIMEOUT_SECONDS}" \
    conda run --no-capture-output -n "${CONDA_ENV}" \
    torchrun --standalone --nproc_per_node="${processes}" \
    "${SCRIPT_DIR}/distributed_probe.py"
}

run_accelerate_distributed_probe() {
  local devices="$1"
  local processes="$2"
  local log_file="$3"
  require_distributed_device_count "${devices}" "${processes}" "accelerate distributed probe"
  require_nvidia_smi_ready "accelerate distributed probe"
  run_logged "${log_file}" \
    env CUDA_VISIBLE_DEVICES="${devices}" \
      PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
      PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF}" \
      TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING}" \
      TORCH_DISTRIBUTED_DEBUG="${TORCH_DISTRIBUTED_DEBUG}" \
      DISTRIBUTED_PROBE_TIMEOUT_SECONDS="${DISTRIBUTED_PROBE_TIMEOUT_SECONDS}" \
    timeout "${DISTRIBUTED_PROBE_TIMEOUT_SECONDS}" \
    conda run --no-capture-output -n "${CONDA_ENV}" \
    accelerate launch --multi_gpu --num_processes "${processes}" \
    --num_machines "${ACCELERATE_NUM_MACHINES}" \
    --machine_rank "${ACCELERATE_MACHINE_RANK}" \
    --main_process_ip "${ACCELERATE_MAIN_PROCESS_IP}" \
    --main_process_port "${ACCELERATE_MAIN_PROCESS_PORT}" \
    --rdzv_backend "${ACCELERATE_RDZV_BACKEND}" \
    "${SCRIPT_DIR}/distributed_probe.py"
}

hf_token_available() {
  if [[ -n "${HF_TOKEN:-}" ]]; then
    return 0
  fi
  conda run --no-capture-output -n "${CONDA_ENV}" \
    python -B -c 'import os; from huggingface_hub import get_token; raise SystemExit(0 if os.environ.get("HF_TOKEN") or get_token() else 1)' \
    >/dev/null 2>&1
}

require_hf_token() {
  local route_name="$1"
  if ! hf_token_available; then
    printf "%s requires HF_TOKEN in the environment; token value is never logged.\n" "${route_name}" >&2
    return 2
  fi
}

run_hf_model_access_probe() {
  local model_id="$1"
  local log_file="$2"
  local require_weights="${3:-0}"
  require_hf_token "HF model access probe for ${model_id}"
  local probe_cmd=(
    conda run --no-capture-output -n "${CONDA_ENV}"
    python -B "${SCRIPT_DIR}/hf_model_access_probe.py"
    --model-id "${model_id}"
  )
  if [[ "${require_weights}" == "1" ]]; then
    probe_cmd+=(--require-weights)
  fi
  run_logged "${log_file}" "${probe_cmd[@]}"
}

assert_train_log_completed() {
  local log_file="$1"
  local train_csv="${2:-}"
  local eval_csv="${3:-}"
  local test_csv="${4:-}"
  local max_train_samples="${5:--1}"
  local max_eval_samples="${6:--1}"
  local max_test_samples="${7:--1}"
  if [[ ! -s "${log_file}" ]]; then
    printf "Training log was not written: %s\n" "${log_file}" >&2
    return 3
  fi
  if ! grep -q "training completed; adapter saved" "${log_file}"; then
    printf "Training log exists but has no completion marker: %s\n" "${log_file}" >&2
    return 4
  fi
  if [[ -n "${train_csv}" ]]; then
    local expected_train_rows
    expected_train_rows="$(csv_data_row_count "${train_csv}" "${max_train_samples}")"
    if ! grep -q "train_rows=${expected_train_rows}" "${log_file}"; then
      printf "Training log does not record expected train_rows=%s: %s\n" "${expected_train_rows}" "${log_file}" >&2
      return 5
    fi
  fi
  if [[ -n "${eval_csv}" ]]; then
    local expected_eval_rows
    expected_eval_rows="$(csv_data_row_count "${eval_csv}" "${max_eval_samples}")"
    if ! grep -q "eval_rows=${expected_eval_rows}" "${log_file}"; then
      printf "Training log does not record expected eval_rows=%s: %s\n" "${expected_eval_rows}" "${log_file}" >&2
      return 6
    fi
    if [[ "${RETRIEVER_ALLOW_MISSING_EVAL_LOSS:-0}" != "1" ]] && ! grep -q "eval_loss" "${log_file}"; then
      printf "Training log has no eval_loss; set RETRIEVER_ALLOW_MISSING_EVAL_LOSS=1 only for explicit few/smoke runs: %s\n" "${log_file}" >&2
      return 7
    fi
  fi
  if [[ -n "${test_csv}" ]]; then
    local expected_test_rows
    expected_test_rows="$(csv_data_row_count "${test_csv}" "${max_test_samples}")"
    if ! grep -q "test_rows=${expected_test_rows}" "${log_file}"; then
      printf "Training log does not record expected test_rows=%s: %s\n" "${expected_test_rows}" "${log_file}" >&2
      return 8
    fi
  fi
}

assert_train_log_current_for_reuse() {
  local log_file="$1"
  local train_csv="${2:-}"
  local eval_csv="${3:-}"
  local test_csv="${4:-}"
  local max_train_samples="${5:--1}"
  local max_eval_samples="${6:--1}"
  local max_test_samples="${7:--1}"
  local route_name="${8:-retriever}"
  if [[ "${RUN_TRAIN:-0}" == "1" ]]; then
    return 0
  fi
  if [[ "${RETRIEVER_ALLOW_STALE_TRAIN_LOG:-0}" == "1" ]]; then
    printf "Warning: %s is reusing existing train artifact without formal train-log validation.\n" "${route_name}" >&2
    return 0
  fi
  assert_train_log_completed \
    "${log_file}" \
    "${train_csv}" \
    "${eval_csv}" \
    "${test_csv}" \
    "${max_train_samples}" \
    "${max_eval_samples}" \
    "${max_test_samples}"
}


device_at_index() {
  local devices="${1//[[:space:]]/}"
  local index="$2"
  if [[ "${devices}" == "all" ]]; then
    nvidia-smi --query-gpu=index --format=csv,noheader,nounits 2>/dev/null | sed -n "$((index + 1))p"
    return
  fi
  IFS=',' read -r -a parts <<< "${devices}"
  printf "%s\n" "${parts[${index}]}"
}

csv_data_row_count() {
  conda run --no-capture-output -n "${CONDA_ENV}" python -B - "$1" "${2:--1}" <<'PY'
import csv
import sys

csv.field_size_limit(sys.maxsize)
path = sys.argv[1]
max_samples = int(sys.argv[2])
with open(path, newline='', encoding='utf-8') as handle:
    reader = csv.reader(handle)
    rows = list(reader)
count = max(0, len(rows) - 1)
if max_samples >= 0:
    count = min(count, max_samples)
print(count)
PY
}

retriever_split_row_summary() {
  local train_csv="$1"
  local eval_csv="$2"
  local test_csv="$3"
  local max_train_samples="${4:--1}"
  local max_eval_samples="${5:--1}"
  local max_test_samples="${6:--1}"
  printf "train_rows=%s eval_rows=%s test_rows=%s train_csv=%s eval_csv=%s test_csv=%s" \
    "$(csv_data_row_count "${train_csv}" "${max_train_samples}")" \
    "$(csv_data_row_count "${eval_csv}" "${max_eval_samples}")" \
    "$(csv_data_row_count "${test_csv}" "${max_test_samples}")" \
    "${train_csv}" "${eval_csv}" "${test_csv}"
}

prediction_record_count() {
  conda run --no-capture-output -n "${CONDA_ENV}" python -B - "${REPO_ROOT}" "$1" <<'PY'
import sys
from pathlib import Path

repo = Path(sys.argv[1])
sys.path.insert(0, str(repo))
from result_organization import read_prediction_records

print(len(read_prediction_records(Path(sys.argv[2]))))
PY
}

split_csv_for_parallel() {
  conda run --no-capture-output -n "${CONDA_ENV}" python -B - "$1" "$2" "$3" "${4:--1}" <<'PY'
import csv
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)
input_csv = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
chunks = max(1, int(sys.argv[3]))
max_samples = int(sys.argv[4])
out_dir.mkdir(parents=True, exist_ok=True)
with input_csv.open(newline='', encoding='utf-8') as handle:
    reader = csv.reader(handle)
    header = next(reader)
    rows = list(reader)
if max_samples >= 0:
    rows = rows[:max_samples]
chunks = min(chunks, max(1, len(rows)))
for index in range(chunks):
    start = index * len(rows) // chunks
    end = (index + 1) * len(rows) // chunks
    path = out_dir / f'input_{index}.csv'
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows[start:end])
    print(path)
PY
}

prediction_extension_for_format() {
  case "${1,,}" in
    json) printf ".json\n" ;;
    jsonl) printf ".jsonl\n" ;;
    text|txt) printf ".txt\n" ;;
    *) printf "Unsupported prediction output format for merge: %s\n" "$1" >&2; return 2 ;;
  esac
}

merge_prediction_shards() {
  conda run --no-capture-output -n "${CONDA_ENV}" python -B - "$1" "$2" "${@:3}" <<'PY'
import json
import sys
from pathlib import Path

fmt = sys.argv[1].strip().lower()
out = Path(sys.argv[2])
inputs = [Path(value) for value in sys.argv[3:]]
out.parent.mkdir(parents=True, exist_ok=True)
if fmt == 'json':
    merged = []
    for path in inputs:
        payload = json.loads(path.read_text(encoding='utf-8'))
        if isinstance(payload, dict):
            for key in ('records', 'predictions', 'data'):
                if isinstance(payload.get(key), list):
                    payload = payload[key]
                    break
        if not isinstance(payload, list):
            raise ValueError(f'{path} is not a JSON list prediction shard')
        merged.extend(payload)
    out.write_text(json.dumps(merged, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
else:
    with out.open('w', encoding='utf-8') as handle:
        for path in inputs:
            text = path.read_text(encoding='utf-8')
            if text and not text.endswith('\n'):
                text += '\n'
            handle.write(text)
PY
}

split_match_inputs_for_parallel() {
  conda run --no-capture-output -n "${CONDA_ENV}" python -B - "${REPO_ROOT}" "$1" "$2" "$3" "$4" "$5" <<'PY'
import csv
import json
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)

repo = Path(sys.argv[1])
sys.path.insert(0, str(repo))
from result_organization import read_prediction_records

prediction_txt = Path(sys.argv[2])
data_json = Path(sys.argv[3])
relfact_csv = Path(sys.argv[4])
out_dir = Path(sys.argv[5])
chunks = max(1, int(sys.argv[6]))
out_dir.mkdir(parents=True, exist_ok=True)
records = read_prediction_records(prediction_txt)
data = json.loads(data_json.read_text(encoding='utf-8'))
with relfact_csv.open(newline='', encoding='utf-8') as handle:
    reader = csv.DictReader(handle)
    fieldnames = reader.fieldnames
    rows = list(reader)
if fieldnames is None:
    raise ValueError(f'{relfact_csv} has no CSV header')
if len(records) > len(data):
    raise ValueError(f'Prediction records ({len(records)}) exceed data rows ({len(data)})')
if len(records) > len(rows):
    raise ValueError(f'Prediction records ({len(records)}) exceed Rel_Fact rows ({len(rows)})')
count = len(records)
chunks = min(chunks, max(1, count))
for index in range(chunks):
    start = index * count // chunks
    end = (index + 1) * count // chunks
    pred_path = out_dir / f'predictions_{index}.txt'
    data_path = out_dir / f'data_{index}.json'
    csv_path = out_dir / f'relfact_{index}.csv'
    pred_path.write_text('\n'.join(records[start:end]) + ('\n' if end > start else ''), encoding='utf-8')
    data_path.write_text(json.dumps(data[start:end], ensure_ascii=False) + '\n', encoding='utf-8')
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows[start:end])
    print(f'{pred_path}\t{data_path}\t{csv_path}')
PY
}

merge_json_shards() {
  conda run --no-capture-output -n "${CONDA_ENV}" python -B - "$1" "${@:2}" <<'PY'
import json
import sys
from pathlib import Path

out = Path(sys.argv[1])
inputs = [Path(value) for value in sys.argv[2:]]
merged = []
for path in inputs:
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, list):
        raise ValueError(f'{path} is not a matched JSON list')
    merged.extend(payload)
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(merged, ensure_ascii=False) + '\n', encoding='utf-8')
print(f'merged_rows={len(merged)}')
PY
}

similarity_jsonl_output_for_match() {
  printf "%s/predictions_with_similarity.jsonl\n" "$(dirname "$1")"
}

summarize_matched_artifact() {
  local output_json="$1"
  local similarity_jsonl_output="${2:-$(similarity_jsonl_output_for_match "${output_json}")}"
  conda run --no-capture-output -n "${CONDA_ENV}" \
    python -B "${REPO_ROOT}/result_organization.py" summarize \
    --matched-json "${output_json}" \
    --similarity-jsonl-output "${similarity_jsonl_output}"
}


run_parallel_inference_artifact() {
  local log_file="$1"
  local route_name="$2"
  local devices="$3"
  local input_csv="$4"
  local output_txt="$5"
  local output_format="$6"
  local max_infer_samples="$7"
  local batch_size="$8"
  shift 8
  local -a cmd=("$@")
  local rows device_count chunks
  rows="$(csv_data_row_count "${input_csv}" "${max_infer_samples}")"
  device_count="$(visible_device_count "${devices}")"
  if [[ "${INFER_PARALLEL_GPU}" != "1" || "${device_count}" -lt 2 || "${rows}" -lt 2 ]]; then
    run_logged "${log_file}" \
      env CUDA_VISIBLE_DEVICES="${devices}" \
      PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
      PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF}" \
      "${cmd[@]}" \
      --input-csv "${input_csv}" \
      --output-txt "${output_txt}" \
      --max-infer-samples "${max_infer_samples}" \
      --batch-size "${batch_size}"
    return $?
  fi
  chunks="${device_count}"
  if [[ "${rows}" -lt "${chunks}" ]]; then
    chunks="${rows}"
  fi
  backup_existing_log "${log_file}"
  {
    printf "started_at=%s\n" "$(utc_now)"
    printf "cwd=%q\n" "$(pwd)"
    printf "parallel_inference=1\n"
    printf "route_name=%q\n" "${route_name}"
    printf "cuda_visible_devices=%q\n" "${devices}"
    printf "rows=%q\n" "${rows}"
    printf "chunks=%q\n" "${chunks}"
    printf "batch_size_per_process=%q\n" "${batch_size}"
    printf "output_format=%q\n" "${output_format}"
    local shard_dir ext rc
    shard_dir="$(dirname "${output_txt}")/.parallel_infer_$(date -u +"%Y%m%dT%H%M%SZ")"
    mkdir -p "${shard_dir}"
    ext="$(prediction_extension_for_format "${output_format}")"
    mapfile -t shard_csvs < <(split_csv_for_parallel "${input_csv}" "${shard_dir}" "${chunks}" "${max_infer_samples}")
    if [[ "${#shard_csvs[@]}" -ne "${chunks}" ]]; then
      printf "Expected %s CSV shards, got %s\n" "${chunks}" "${#shard_csvs[@]}" >&2
      printf "finished_at=%s\n" "$(utc_now)"
      printf "exit_code=2\n"
      exit 2
    fi
    local -a shard_outputs shard_logs pids
    rc=0
    for index in "${!shard_csvs[@]}"; do
      local device
      device="$(device_at_index "${devices}" "${index}")"
      shard_outputs[${index}]="${shard_dir}/predictions_${index}${ext}"
      shard_logs[${index}]="${shard_dir}/infer_${index}.log"
      printf "launch_shard=%s device=%s input=%s output=%s log=%s\n" \
        "${index}" "${device}" "${shard_csvs[${index}]}" "${shard_outputs[${index}]}" "${shard_logs[${index}]}"
      (
        CUDA_VISIBLE_DEVICES="${device}" \
        PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF}" \
        PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF}" \
        "${cmd[@]}" \
        --input-csv "${shard_csvs[${index}]}" \
        --output-txt "${shard_outputs[${index}]}" \
        --max-infer-samples -1 \
        --batch-size "${batch_size}"
      ) >"${shard_logs[${index}]}" 2>&1 &
      pids[${index}]=$!
    done
    for index in "${!pids[@]}"; do
      if ! wait "${pids[${index}]}"; then
        rc=1
        printf "failed_shard=%s log=%s\n" "${index}" "${shard_logs[${index}]}" >&2
        tail -n 120 "${shard_logs[${index}]}" >&2 || true
      fi
    done
    if [[ "${rc}" -eq 0 ]]; then
      merge_prediction_shards "${output_format}" "${output_txt}" "${shard_outputs[@]}"
      printf "merged_output=%s\n" "${output_txt}"
    fi
    printf "finished_at=%s\n" "$(utc_now)"
    printf "exit_code=%s\n" "${rc}"
    exit "${rc}"
  } 2>&1 | tee "${log_file}"
  return "${PIPESTATUS[0]}"
}

run_parallel_match_artifact() {
  local retriever_model="$1"
  local prediction_txt="$2"
  local output_json="$3"
  local log_file="$4"
  local prompt_mode="$5"
  local expt_id="$6"
  local match_data_json="$7"
  local match_relfact_csv="$8"
  local rows expected_rows allow_partial device_count chunks similarity_jsonl_output
  rows="$(prediction_record_count "${prediction_txt}")"
  expected_rows="${9:-}"
  allow_partial="${10:-0}"
  if [[ -z "${expected_rows}" ]]; then
    expected_rows="$(csv_data_row_count "${match_relfact_csv}" -1)"
  fi
  if [[ "${allow_partial}" != "1" && "${rows}" != "${expected_rows}" ]]; then
    printf "Match row-count check failed: predictions=%s expected_rows=%s prediction_txt=%s relfact_csv=%s\n" \
      "${rows}" "${expected_rows}" "${prediction_txt}" "${match_relfact_csv}" >&2
    printf "Set MATCH_ALLOW_PARTIAL=1 only for explicit partial smoke runs.\n" >&2
    return 2
  fi
  local -a match_row_args=(--expected-rows "${expected_rows}")
  if [[ "${allow_partial}" == "1" ]]; then
    match_row_args+=(--allow-partial)
  fi
  device_count="$(visible_device_count "${MATCH_CUDA_DEVICES}")"
  similarity_jsonl_output="$(similarity_jsonl_output_for_match "${output_json}")"
  if [[ "${MATCH_PARALLEL_GPU}" != "1" || "${device_count}" -lt 2 || "${rows}" -lt 2 ]]; then
    run_logged "${log_file}" \
      env CUDA_VISIBLE_DEVICES="${MATCH_CUDA_DEVICES}" \
      conda run --no-capture-output -n "${CONDA_ENV}" \
      python -B "${REPO_ROOT}/result_organization.py" match \
      --dataset finqa \
      --retriever-model "${retriever_model}" \
      --prompt-mode "${prompt_mode}" \
      --input-txt "${prediction_txt}" \
      --data-json "${match_data_json}" \
      --relfact-csv "${match_relfact_csv}" \
      --embedding-batch-size "${MATCH_EMBED_BATCH_SIZE}" \
      --output-json "${output_json}" \
      --similarity-jsonl-output "${similarity_jsonl_output}" \
      --execute \
      --require-valid-schema \
      "${match_row_args[@]}"
    return $?
  fi
  chunks="${device_count}"
  if [[ "${rows}" -lt "${chunks}" ]]; then
    chunks="${rows}"
  fi
  backup_existing_log "${log_file}"
  {
    printf "started_at=%s\n" "$(utc_now)"
    printf "cwd=%q\n" "$(pwd)"
    printf "parallel_match=1\n"
    printf "retriever_model=%q\n" "${retriever_model}"
    printf "prompt_mode=%q\n" "${prompt_mode}"
    printf "expt_id=%q\n" "${expt_id}"
    printf "cuda_visible_devices=%q\n" "${MATCH_CUDA_DEVICES}"
    printf "rows=%q\n" "${rows}"
    printf "chunks=%q\n" "${chunks}"
    printf "embedding_batch_size=%q\n" "${MATCH_EMBED_BATCH_SIZE}"
    local shard_dir rc
    shard_dir="$(dirname "${output_json}")/.parallel_match_$(date -u +"%Y%m%dT%H%M%SZ")"
    mkdir -p "${shard_dir}"
    mapfile -t shard_specs < <(split_match_inputs_for_parallel "${prediction_txt}" "${match_data_json}" "${match_relfact_csv}" "${shard_dir}" "${chunks}")
    if [[ "${#shard_specs[@]}" -ne "${chunks}" ]]; then
      printf "Expected %s match shards, got %s\n" "${chunks}" "${#shard_specs[@]}" >&2
      printf "finished_at=%s\n" "$(utc_now)"
      printf "exit_code=2\n"
      exit 2
    fi
    local -a shard_outputs shard_logs pids
    rc=0
    for index in "${!shard_specs[@]}"; do
      local shard_pred shard_data shard_csv device shard_expected_rows
      IFS=$'\t' read -r shard_pred shard_data shard_csv <<< "${shard_specs[${index}]}"
      device="$(device_at_index "${MATCH_CUDA_DEVICES}" "${index}")"
      shard_expected_rows="$(prediction_record_count "${shard_pred}")"
      shard_outputs[${index}]="${shard_dir}/matched_${index}.json"
      shard_logs[${index}]="${shard_dir}/match_${index}.log"
      printf "launch_match_shard=%s device=%s input=%s output=%s log=%s\n" \
        "${index}" "${device}" "${shard_pred}" "${shard_outputs[${index}]}" "${shard_logs[${index}]}"
      (
        CUDA_VISIBLE_DEVICES="${device}" \
        conda run --no-capture-output -n "${CONDA_ENV}" \
        python -B "${REPO_ROOT}/result_organization.py" match \
        --dataset finqa \
        --retriever-model "${retriever_model}" \
        --prompt-mode "${prompt_mode}" \
        --input-txt "${shard_pred}" \
        --data-json "${shard_data}" \
        --relfact-csv "${shard_csv}" \
        --embedding-batch-size "${MATCH_EMBED_BATCH_SIZE}" \
        --output-json "${shard_outputs[${index}]}" \
        --execute \
        --require-valid-schema \
        --expected-rows "${shard_expected_rows}"
      ) >"${shard_logs[${index}]}" 2>&1 &
      pids[${index}]=$!
    done
    for index in "${!pids[@]}"; do
      if ! wait "${pids[${index}]}"; then
        rc=1
        printf "failed_match_shard=%s log=%s\n" "${index}" "${shard_logs[${index}]}" >&2
        tail -n 120 "${shard_logs[${index}]}" >&2 || true
      fi
    done
    if [[ "${rc}" -eq 0 ]]; then
      merge_json_shards "${output_json}" "${shard_outputs[@]}"
      printf "merged_output=%s\n" "${output_json}"
      if summarize_matched_artifact "${output_json}" "${similarity_jsonl_output}"; then
        printf "similarity_jsonl_output=%s\n" "${similarity_jsonl_output}"
      else
        rc=1
        printf "failed_similarity_export=%s\n" "${similarity_jsonl_output}" >&2
      fi
    fi
    printf "finished_at=%s\n" "$(utc_now)"
    printf "exit_code=%s\n" "${rc}"
    exit "${rc}"
  } 2>&1 | tee "${log_file}"
  return "${PIPESTATUS[0]}"
}

run_match_artifact() {
  local retriever_model="$1"
  local prediction_txt="$2"
  local output_json="$3"
  local log_file="$4"
  local prompt_mode="${5:-original}"
  local expt_id="${6:-unknown}"
  local match_split="${7:-test}"
  local match_data_json="${DATA_JSON}"
  local match_relfact_csv expected_rows
  match_relfact_csv="$(prompt_csv_for_split "${prompt_mode}" "${match_split}")"
  if [[ ! -f "${prediction_txt}" ]]; then
    printf "Match cannot start: missing predictions artifact.\n" >&2
    printf "retriever_model=%s prompt_mode=%s expt_id=%s prediction_txt=%s\n" \
      "${retriever_model}" "${prompt_mode}" "${expt_id}" "${prediction_txt}" >&2
    printf "Run inference first, or point RUN_MATCH at an existing predictions.txt.\n" >&2
    return 2
  fi
  if [[ -n "${RETRIEVER_FEW_CSV}" ]]; then
    match_data_json="${RETRIEVER_FEW_DATA_JSON:-$(dirname "${output_json}")/few_retriever_data.json}"
    build_match_data_json_from_csv "${match_relfact_csv}" "${match_data_json}"
  elif [[ "${match_split}" != "test" || ! -f "${match_data_json}" ]]; then
    match_data_json="${MATCH_DATA_JSON:-$(dirname "${output_json}")/finqa_${match_split}_retriever_data.json}"
    build_match_data_json_from_csv "${match_relfact_csv}" "${match_data_json}"
  fi
  if [[ ! -f "${match_data_json}" ]]; then
    printf "Match cannot start: missing FinQA data JSON.\n" >&2
    printf "retriever_model=%s prompt_mode=%s match_split=%s expt_id=%s data_json=%s\n" \
      "${retriever_model}" "${prompt_mode}" "${match_split}" "${expt_id}" "${match_data_json}" >&2
    return 2
  fi
  if [[ ! -f "${match_relfact_csv}" ]]; then
    printf "Match cannot start: missing Rel_Fact CSV.\n" >&2
    printf "retriever_model=%s prompt_mode=%s match_split=%s expt_id=%s relfact_csv=%s\n" \
      "${retriever_model}" "${prompt_mode}" "${match_split}" "${expt_id}" "${match_relfact_csv}" >&2
    return 2
  fi
  expected_rows="$(csv_data_row_count "${match_relfact_csv}" -1)"
  run_parallel_match_artifact \
    "${retriever_model}" \
    "${prediction_txt}" \
    "${output_json}" \
    "${log_file}" \
    "${prompt_mode}" \
    "${expt_id}" \
    "${match_data_json}" \
    "${match_relfact_csv}" \
    "${expected_rows}" \
    "${MATCH_ALLOW_PARTIAL:-0}"
}
