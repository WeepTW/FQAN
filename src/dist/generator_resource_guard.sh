#!/usr/bin/env bash

# Guard long-running generator services before starting or executing LLM routes.
# Source this file from experiment scripts after retriever_experiment_lib.sh.

GENERATOR_RESOURCE_GUARD_MIN_RSS_MB="${GENERATOR_RESOURCE_GUARD_MIN_RSS_MB:-8192}"
GENERATOR_RESOURCE_GUARD_ACTION="${GENERATOR_RESOURCE_GUARD_ACTION:-prompt}"
GENERATOR_RESOURCE_GUARD_NONINTERACTIVE_ACTION="${GENERATOR_RESOURCE_GUARD_NONINTERACTIVE_ACTION:-pause}"
GENERATOR_RESOURCE_GUARD_CLEANUP_AFTER_ENGINE="${GENERATOR_RESOURCE_GUARD_CLEANUP_AFTER_ENGINE:-1}"
GENERATOR_RESOURCE_GUARD_ALLOWED_SERVICES="${GENERATOR_RESOURCE_GUARD_ALLOWED_SERVICES:-}"
GENERATOR_RESOURCE_GUARD_SERVICE_FILTER="${GENERATOR_RESOURCE_GUARD_SERVICE_FILTER:-}"

generator_shell_id() {
  if [[ -n "${EXPERIMENT_SHELL_ID:-}" ]]; then
    printf "%s\n" "${EXPERIMENT_SHELL_ID}"
  elif [[ -n "${TMUX_PANE:-}" ]]; then
    printf "%s\n" "${TMUX_PANE}"
  else
    printf "pid_%s\n" "$$"
  fi
}

generator_resource_patterns() {
  cat <<'EOF'
vllm:vllm.entrypoints.openai.api_server
vllm:python -m vllm.entrypoints.openai.api_server
chatmock:chatmock.py serve
chatmock:ChatMock.*serve
chatmock:(^|[/:])chatmock[[:space:]]+serve
tgi:text-generation-launcher
tgi:text-generation-server
ollama:ollama serve
triton:tritonserver
ray:raylet
EOF
}

generator_resource_service_list_contains() {
  local label="$1"
  local service
  for service in $2; do
    [[ "${service}" == "${label}" ]] && return 0
  done
  return 1
}

generator_resource_label_selected() {
  local label="$1"
  if [[ -n "${GENERATOR_RESOURCE_GUARD_SERVICE_FILTER}" ]]     && ! generator_resource_service_list_contains "${label}" "${GENERATOR_RESOURCE_GUARD_SERVICE_FILTER}"; then
    return 1
  fi
  if [[ -n "${GENERATOR_RESOURCE_GUARD_ALLOWED_SERVICES}" ]]     && generator_resource_service_list_contains "${label}" "${GENERATOR_RESOURCE_GUARD_ALLOWED_SERVICES}"; then
    return 1
  fi
  return 0
}

generator_guard_pid_rows() {
  local seen=" "
  local label pattern pids pid rss_kb rss_mb cmd gpu_mem
  while IFS=":" read -r label pattern; do
    [[ -z "${label}" || -z "${pattern}" ]] && continue
    generator_resource_label_selected "${label}" || continue
    pids="$(pgrep -f "${pattern}" 2>/dev/null || true)"
    for pid in ${pids}; do
      [[ "${pid}" == "$$" ]] && continue
      [[ "${seen}" == *" ${pid} "* ]] && continue
      seen="${seen}${pid} "
      rss_kb="$(ps -o rss= -p "${pid}" 2>/dev/null | tr -d '[:space:]' || true)"
      [[ -z "${rss_kb}" ]] && rss_kb=0
      rss_mb=$((rss_kb / 1024))
      cmd="$(ps -o args= -p "${pid}" 2>/dev/null || true)"
      gpu_mem="$(generator_guard_gpu_memory_for_pid "${pid}")"
      if [[ "${gpu_mem}" == "0" && "${rss_mb}" -lt "${GENERATOR_RESOURCE_GUARD_MIN_RSS_MB}" ]]; then
        continue
      fi
      printf "%s\t%s\t%s\t%s\t%s\n" "${label}" "${pid}" "${rss_mb}" "${gpu_mem}" "${cmd}"
    done
  done < <(generator_resource_patterns)
}

generator_guard_gpu_memory_for_pid() {
  local pid="$1"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    printf "unknown"
    return
  fi
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null \
    | awk -F, -v target="${pid}" '
      {gsub(/^[ \t]+|[ \t]+$/, "", $1); gsub(/^[ \t]+|[ \t]+$/, "", $2)}
      $1 == target {sum += $2}
      END {if (sum == "") print "0"; else print sum}
    '
}

generator_guard_print_gpu_summary() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv 2>/dev/null || true
    nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_memory --format=csv 2>/dev/null || true
  else
    printf "nvidia-smi=unavailable\n"
  fi
}

generator_guard_conflict_pids() {
  awk -F '\t' '{print $2}' "$1" | tr '\n' ' '
}

generator_guard_has_conflicts() {
  [[ -s "$1" ]]
}

generator_resource_guard_before_llm() {
  local context="$1"
  local engine="$2"
  local guard_dir="${3:-${REPO_ROOT:-.}/Experiment/resource_guard}"
  local ts shell_id report_file action pids
  ts="$(utc_now 2>/dev/null || date -u +"%Y-%m-%dT%H:%M:%SZ")"
  shell_id="$(generator_shell_id)"
  mkdir -p "${guard_dir}"
  report_file="${guard_dir}/resource_guard_${context}_${engine}_$(date -u +%Y%m%dT%H%M%SZ).tsv"

  generator_guard_pid_rows >"${report_file}"
  printf "[%s] resource_guard context=%s engine=%s shell_id=%s report=%s\n" \
    "${ts}" "${context}" "${engine}" "${shell_id}" "${report_file}"

  if ! generator_guard_has_conflicts "${report_file}"; then
    printf "[%s] resource_guard clear: no known vLLM/ChatMock/high-memory generator service found.\n" "${ts}"
    return 0
  fi

  printf "\nPotential generator service conflict before %s/%s:\n" "${context}" "${engine}" >&2
  printf "service\tpid\trss_mb\tgpu_mem_mib\tcommand\n" >&2
  cat "${report_file}" >&2
  printf "\nGPU summary:\n" >&2
  generator_guard_print_gpu_summary >&2

  action="${GENERATOR_RESOURCE_GUARD_ACTION}"
  if [[ "${action}" == "prompt" ]]; then
    if [[ -t 0 ]]; then
      printf "\nChoose action for this LLM case: [s]top services / [w]ait / [p]ause this LLM experiment: " >&2
      read -r reply
      case "${reply}" in
        s|S|stop|STOP) action="stop" ;;
        w|W|wait|WAIT) action="wait" ;;
        *) action="pause" ;;
      esac
    else
      action="${GENERATOR_RESOURCE_GUARD_NONINTERACTIVE_ACTION}"
      printf "Non-interactive shell; using GENERATOR_RESOURCE_GUARD_NONINTERACTIVE_ACTION=%s\n" "${action}" >&2
    fi
  fi

  case "${action}" in
    stop|kill)
      pids="$(generator_guard_conflict_pids "${report_file}")"
      printf "[%s] resource_guard stopping pids: %s\n" "$(utc_now)" "${pids}" >&2
      kill ${pids} 2>/dev/null || true
      sleep 10
      pids="$(generator_guard_pid_rows | tee "${report_file}.after_stop" | awk -F '\t' '{print $2}' | tr '\n' ' ')"
      if [[ -n "${pids//[[:space:]]/}" ]]; then
        printf "[%s] resource_guard force-stopping pids: %s\n" "$(utc_now)" "${pids}" >&2
        kill -9 ${pids} 2>/dev/null || true
        sleep 5
      fi
      return 0
      ;;
    pause|skip)
      printf "[%s] resource_guard paused engine=%s context=%s shell_id=%s\n" "$(utc_now)" "${engine}" "${context}" "${shell_id}" >&2
      return 75
      ;;
    wait)
      local wait_seconds max_wait_seconds elapsed_seconds
      wait_seconds="${GENERATOR_RESOURCE_GUARD_WAIT_SECONDS:-60}"
      max_wait_seconds="${GENERATOR_RESOURCE_GUARD_MAX_WAIT_SECONDS:-0}"
      elapsed_seconds=0
      while generator_guard_has_conflicts "${report_file}"; do
        printf "[%s] resource_guard waiting engine=%s context=%s elapsed_seconds=%s report=%s\n" \
          "$(utc_now)" "${engine}" "${context}" "${elapsed_seconds}" "${report_file}" >&2
        if [[ "${max_wait_seconds}" != "0" && "${elapsed_seconds}" -ge "${max_wait_seconds}" ]]; then
          printf "[%s] resource_guard wait timed out engine=%s context=%s max_wait_seconds=%s\n" \
            "$(utc_now)" "${engine}" "${context}" "${max_wait_seconds}" >&2
          return 75
        fi
        sleep "${wait_seconds}"
        elapsed_seconds=$((elapsed_seconds + wait_seconds))
        generator_guard_pid_rows >"${report_file}"
      done
      printf "[%s] resource_guard clear after wait engine=%s context=%s\n" "$(utc_now)" "${engine}" "${context}" >&2
      return 0
      ;;
    fail)
      printf "[%s] resource_guard failed by policy for engine=%s context=%s\n" "$(utc_now)" "${engine}" "${context}" >&2
      return 2
      ;;
    *)
      printf "Unsupported GENERATOR_RESOURCE_GUARD_ACTION=%s; use prompt, stop, wait, pause, or fail.\n" "${action}" >&2
      return 2
      ;;
  esac
}

generator_cleanup_services_after_llm() {
  local context="$1"
  local engine="$2"
  local guard_dir="${3:-${REPO_ROOT:-.}/Experiment/resource_guard}"
  local report_file pids
  if [[ "${GENERATOR_RESOURCE_GUARD_CLEANUP_AFTER_ENGINE}" != "1" ]]; then
    return 0
  fi
  mkdir -p "${guard_dir}"
  report_file="${guard_dir}/resource_cleanup_${context}_${engine}_$(date -u +%Y%m%dT%H%M%SZ).tsv"
  generator_guard_pid_rows >"${report_file}"
  if [[ ! -s "${report_file}" ]]; then
    printf "[%s] resource_cleanup clear context=%s engine=%s shell_id=%s\n" \
      "$(utc_now 2>/dev/null || date -u +"%Y-%m-%dT%H:%M:%SZ")" "${context}" "${engine}" "$(generator_shell_id)"
    return 0
  fi
  printf "[%s] resource_cleanup stopping services after engine=%s context=%s report=%s\n" \
    "$(utc_now)" "${engine}" "${context}" "${report_file}" >&2
  cat "${report_file}" >&2
  pids="$(generator_guard_conflict_pids "${report_file}")"
  kill ${pids} 2>/dev/null || true
  sleep 10
  pids="$(generator_guard_pid_rows | tee "${report_file}.after_stop" | awk -F '\t' '{print $2}' | tr '\n' ' ')"
  if [[ -n "${pids//[[:space:]]/}" ]]; then
    printf "[%s] resource_cleanup force-stopping pids: %s\n" "$(utc_now)" "${pids}" >&2
    kill -9 ${pids} 2>/dev/null || true
    sleep 5
  fi
}
