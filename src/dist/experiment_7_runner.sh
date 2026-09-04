#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/retriever_experiment_lib.sh"

EXPT_ID="${EXPT_ID:-experiment_7_fqan_formal_$(date -u +%Y%m%dT%H%M%SZ)}"
TMUX_RUN_SESSION="${TMUX_RUN_SESSION:-monitor}"
WINDOW_PREFIX="${WINDOW_PREFIX:-exp7}"
WAIT_BEFORE_START_SECONDS="${WAIT_BEFORE_START_SECONDS:-300}"
SELECTION_EXPT_ID="${SELECTION_EXPT_ID:-experiment_7_selection_gpt55_finqa_train_formal_20260611T080113Z}"
SELECTION_ENGINE="${SELECTION_ENGINE:-gpt5_5}"
SELECTION_CACHE_JSON="${SELECTION_CACHE_JSON:-${REPO_ROOT}/Experiment/${SELECTION_EXPT_ID}/in_context_selection/${SELECTION_ENGINE}/selection_cache.json}"
DEFAULT_EXPERIMENT7_FULL_MATRIX="finqa_flan_o:finqa_test finqa_flan_o:finqa_dev finqa_flan_z:finqa_test finqa_flan_z:finqa_dev finqa_flan_m:finqa_test finqa_flan_m:finqa_dev finqa_flan_d:finqa_test finqa_flan_d:finqa_dev finqa_mistral_o:finqa_test finqa_mistral_o:finqa_dev finqa_mistral_z:finqa_test finqa_mistral_z:finqa_dev finqa_mistral_m:finqa_test finqa_mistral_m:finqa_dev finqa_mistral_d:finqa_test finqa_mistral_d:finqa_dev finqa_t5gemma2_o:finqa_test finqa_t5gemma2_o:finqa_dev finqa_t5gemma2_z:finqa_test finqa_t5gemma2_z:finqa_dev finqa_t5gemma2_m:finqa_test finqa_t5gemma2_m:finqa_dev finqa_t5gemma2_d:finqa_test finqa_t5gemma2_d:finqa_dev"
ACTIVE_RETFACT_MATRIX="$(WORKSPACE_ARGS_JSON="${FQAN_DOCS_ROOT}/args.json" python3 - <<'PYMATRIX'
import json, os
from pathlib import Path
path = Path(os.environ['WORKSPACE_ARGS_JSON'])
try:
    args = json.loads(path.read_text(encoding='utf-8'))
    matrix = args.get('pipeline_contracts', {}).get('experiment_7', {}).get('active_retfact_scope', {}).get('matrix_env_string', '')
except Exception:
    matrix = ''
print(matrix.strip())
PYMATRIX
)"
EXPERIMENT7_USE_ACTIVE_RETFACT_SCOPE="${EXPERIMENT7_USE_ACTIVE_RETFACT_SCOPE:-0}"
if [[ "${EXPERIMENT7_USE_ACTIVE_RETFACT_SCOPE}" == "1" ]]; then
  DEFAULT_EXPERIMENT7_MATRIX="${ACTIVE_RETFACT_MATRIX:-${DEFAULT_EXPERIMENT7_FULL_MATRIX}}"
else
  DEFAULT_EXPERIMENT7_MATRIX="${DEFAULT_EXPERIMENT7_FULL_MATRIX}"
fi
EXPERIMENT7_MATRIX="${EXPERIMENT7_MATRIX:-${DEFAULT_EXPERIMENT7_MATRIX}}"
MAX_TOKENS="${MAX_TOKENS:-128}"
FIRST_GATE_MATRIX="${FIRST_GATE_MATRIX:-${FIRST50_GATE_MATRIX:-finqa_flan_o:finqa_test}}"
FIRST_GATE_LIMIT="${FIRST_GATE_LIMIT:-${FIRST50_GATE_LIMIT:-10}}"
CONDA_ENV="${CONDA_ENV:-fnqa}"
CHATMOCK_PORT="${CHATMOCK_PORT:-8000}"
EXPERIMENT7_USE_CHATMOCK_SERVICE="${EXPERIMENT7_USE_CHATMOCK_SERVICE:-0}"
EXPERIMENT7_ALLOW_DIAGNOSTIC_RETFACT_BACKFILL="${EXPERIMENT7_ALLOW_DIAGNOSTIC_RETFACT_BACKFILL:-0}"
QWEN_VLLM_PORT="${QWEN_VLLM_PORT:-8121}"
MISTRAL_LLAMA_CPP_PORT="${MISTRAL_LLAMA_CPP_PORT:-8012}"
LLAMA4_VLLM_PORT="${LLAMA4_VLLM_PORT:-8010}"
REPORT_INTERVAL_SECONDS="${REPORT_INTERVAL_SECONDS:-300}"
EXPT_DIR="${REPO_ROOT}/Experiment/${EXPT_ID}"
ORCH_DIR="${EXPT_DIR}/fqan_tmux_run"
QUEUE_TS="$(date -u +%Y%m%dT%H%M%SZ)"
QUEUE_LOG_JSON="${FQAN_LOG_ROOT}/${QUEUE_TS}_experiment7_fqan_formal_queue.json"
STATUS_JSON="${ORCH_DIR}/status.json"
MATCHED_ENV="${ORCH_DIR}/matched_overrides.env"
mkdir -p "${ORCH_DIR}" "${FQAN_LOG_ROOT}"

sanitize_name() {
  printf "%s" "$1" | tr '[:upper:]' '[:lower:]' | tr '.-' '__' | tr -cd '[:alnum:]_'
}

write_status() {
  local status="$1"
  local detail="$2"
  local exit_code="${3:-0}"
  STATUS_PATH="${STATUS_JSON}" QUEUE_LOG_PATH="${QUEUE_LOG_JSON}" INDEX_PATH="${FQAN_LOG_ROOT}/index.json" \
  STATUS_TIME="$(utc_now)" STATUS_EXPT_ID="${EXPT_ID}" STATUS_TMUX="${TMUX_RUN_SESSION}" \
  STATUS_SELECTION_EXPT_ID="${SELECTION_EXPT_ID}" STATUS_SELECTION_CACHE_JSON="${SELECTION_CACHE_JSON}" \
  STATUS_STATUS="${status}" STATUS_DETAIL="${detail}" STATUS_EXIT_CODE="${exit_code}" \
  STATUS_REPO="${REPO_ROOT}" STATUS_WORKSPACE="${WORKSPACE_ROOT}" STATUS_ORCH_DIR="${ORCH_DIR}" \
  STATUS_QUEUE_LOG="${QUEUE_LOG_JSON}" STATUS_LLAMA4_SENTINEL="${ORCH_DIR}/answer_llama4.smoke_started" STATUS_WINDOW_PREFIX="${WINDOW_PREFIX}" \
  python3 - <<'PYSTATUS'
import json
import os
from pathlib import Path

payload = {
    "time": os.environ["STATUS_TIME"],
    "experiment": "7",
    "stage": "fqan_formal_queue",
    "expt_id": os.environ["STATUS_EXPT_ID"],
    "selection_expt_id": os.environ["STATUS_SELECTION_EXPT_ID"],
    "selection_cache_json": os.environ["STATUS_SELECTION_CACHE_JSON"],
    "tmux_session": os.environ["STATUS_TMUX"],
    "queue_windows": [
        f"{os.environ.get('STATUS_WINDOW_PREFIX', 'exp7')}_preflight",
        f"{os.environ.get('STATUS_WINDOW_PREFIX', 'exp7')}_qwen_vllm",
        f"{os.environ.get('STATUS_WINDOW_PREFIX', 'exp7')}_gpt55",
        f"{os.environ.get('STATUS_WINDOW_PREFIX', 'exp7')}_gptCodexS",
        f"{os.environ.get('STATUS_WINDOW_PREFIX', 'exp7')}_qwen",
        f"{os.environ.get('STATUS_WINDOW_PREFIX', 'exp7')}_gpt41_gate",
        f"{os.environ.get('STATUS_WINDOW_PREFIX', 'exp7')}_llama4",
        f"{os.environ.get('STATUS_WINDOW_PREFIX', 'exp7')}_report",
    ],
    "status": os.environ["STATUS_STATUS"],
    "detail": os.environ["STATUS_DETAIL"],
    "exit_code": int(os.environ["STATUS_EXIT_CODE"]),
    "orchestrator_dir": os.environ["STATUS_ORCH_DIR"],
    "score_report_json": str(Path(os.environ["STATUS_ORCH_DIR"]).parent / "score_report.json"),
    "llama4_smoke_started_sentinel": os.environ["STATUS_LLAMA4_SENTINEL"],
    "llama4_smoke_started": Path(os.environ["STATUS_LLAMA4_SENTINEL"]).is_file(),
}
for raw in (os.environ["STATUS_PATH"], os.environ["QUEUE_LOG_PATH"]):
    path = Path(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
index_path = Path(os.environ["INDEX_PATH"])
try:
    index = json.loads(index_path.read_text(encoding="utf-8")) if index_path.is_file() else {"entries": []}
except Exception:
    index = {"entries": []}
rel_log = str(Path(os.environ["QUEUE_LOG_PATH"]).relative_to(Path(os.environ["STATUS_WORKSPACE"])))
entry = {
    "time": payload["time"],
    "path": rel_log,
    "repo": os.environ["STATUS_REPO"],
    "kind": "experiment7_fqan_formal_queue",
    "status": payload["status"],
    "summary": payload["detail"],
    "tags": ["experiment_7", "finqa", "ea", "tmux", "fqan", "formal_queue"],
}
entries = [item for item in index.setdefault("entries", []) if item.get("path") != rel_log]
entries.append(entry)
index["entries"] = entries
index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False, indent=2))
PYSTATUS
}

window_exists() {
  tmux list-windows -t "${TMUX_RUN_SESSION}" -F '#W' 2>/dev/null | grep -Fxq "$1"
}

ensure_window_name_free() {
  local window_name="$1"
  if window_exists "${window_name}"; then
    write_status "blocked_window_exists" "tmux ${TMUX_RUN_SESSION}:${window_name} already exists; refusing to overwrite." 2
    exit 2
  fi
}

wait_for_rc_success() {
  local label="$1"
  local rc_path="$2"
  while [[ ! -f "${rc_path}" ]]; do sleep 10; done
  local value
  value="$(cat "${rc_path}")"
  if [[ "${value}" != "0" ]]; then
    printf '[%s] dependency %s failed rc=%s path=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${label}" "${value}" "${rc_path}" >&2
    return "${value}"
  fi
}

endpoint_ready() {
  local base_url="$1"
  BASE_URL="${base_url%/}" python3 - <<'PYREADY' >/dev/null 2>&1
import os
import urllib.request
url = os.environ["BASE_URL"].removesuffix("/v1") + "/v1/models"
request = urllib.request.Request(
    url,
    headers={"Authorization": f"Bearer {os.environ.get('VLLM_API_KEY', 'EMPTY')}"},
)
with urllib.request.urlopen(request, timeout=5) as response:
    if response.status >= 400:
        raise SystemExit(1)
PYREADY
}

write_preflight_runner() {
  local runner="${ORCH_DIR}/preflight.sh"
  local rc="${ORCH_DIR}/preflight.rc"
  cat >"${runner}" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
cd ${REPO_ROOT@Q}
rm -f ${rc@Q}
timeline_event() {
  TIMELINE_PATH=${ORCH_DIR@Q}/orchestration_timeline.jsonl \
  TIMELINE_TIME="\$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  TIMELINE_PHASE="\$1" TIMELINE_STATUS="\$2" TIMELINE_DETAIL="\${3:-}" python3 - <<'PYTIMELINE' >> ${ORCH_DIR@Q}/orchestration_timeline.jsonl
import json, os
payload = {
    "time": os.environ["TIMELINE_TIME"],
    "phase": os.environ["TIMELINE_PHASE"],
    "status": os.environ["TIMELINE_STATUS"],
    "detail": os.environ.get("TIMELINE_DETAIL") or None,
}
print(json.dumps(payload, ensure_ascii=False))
PYTIMELINE
}
timeline_event preflight start "wait_seconds=${WAIT_BEFORE_START_SECONDS}"
printf '[%s] waiting %s seconds before Experiment 7 preflight\n' "\$(date -u +%Y-%m-%dT%H:%M:%SZ)" ${WAIT_BEFORE_START_SECONDS@Q}
sleep ${WAIT_BEFORE_START_SECONDS@Q}
set +e
(
  set -e
  bash -n dist/experiment_7_generator_answer.sh dist/experiment_7_in_context_selection.sh dist/experiment_7_selection_cache_binding.sh dist/experiment_7_runner.sh dist/generator_no_api_key.sh dist/generator_api_key.sh dist/generator_resource_guard.sh dist/start_vllm_openai_server.sh dist/start_chatmock_server.sh
  conda run --no-capture-output -n ${CONDA_ENV@Q} python -m py_compile new_full_finqa_run.py dist/*.py
  jq empty ${FQAN_DOCS_ROOT@Q}/args.json ${FQAN_LOG_ROOT@Q}/index.json
  if [[ ${EXPERIMENT7_ALLOW_DIAGNOSTIC_RETFACT_BACKFILL@Q} == "1" ]]; then
    conda run --no-capture-output -n ${CONDA_ENV@Q} python -B dist/experiment_7_backfill_mistral_dev_matched_json.py \
      --repo-root ${REPO_ROOT@Q} \
      --workspace-root ${WORKSPACE_ROOT@Q} \
      --output-root ${EXPT_DIR@Q}/backfilled_retriever_sources \
      --manifest-json ${EXPT_DIR@Q}/backfilled_retriever_sources/manifest.json
  else
    timeline_event retriever_backfill skipped "formal run requires retriever-specific matched-json; set EXPERIMENT7_ALLOW_DIAGNOSTIC_RETFACT_BACKFILL=1 only for diagnostic backfill"
  fi
  timeline_event selection_materialize start "datasets=finqa_test,finqa_dev;prompt_types=original,zero-shot,many-shot,dynamic-shot"
  EXPT_ID=${EXPT_ID@Q} SELECTION_EXPT_ID=${SELECTION_EXPT_ID@Q} SELECTION_ENGINE=${SELECTION_ENGINE@Q} SELECTION_CACHE_JSON=${SELECTION_CACHE_JSON@Q} EXPECTED_SELECTION_SOURCE_MODE=finqa_train_formal DATASETS='finqa_test finqa_dev' PROMPT_TYPES='original zero-shot many-shot dynamic-shot' LIMIT=-1 CONDA_ENV=${CONDA_ENV@Q} bash dist/experiment_7_selection_cache_binding.sh
  timeline_event selection_materialize finish "rc=0"
  CACHE_JSON=${SELECTION_CACHE_JSON@Q} MATRIX=${EXPERIMENT7_MATRIX@Q} MATCHED_ENV=${MATCHED_ENV@Q} PREFLIGHT_JSON=${ORCH_DIR@Q}/preflight_status.json REPO_ROOT=${REPO_ROOT@Q} BACKFILL_ROOT=${EXPT_DIR@Q}/backfilled_retriever_sources BACKFILL_MANIFEST=${EXPT_DIR@Q}/backfilled_retriever_sources/manifest.json ALLOW_DIAGNOSTIC_RETFACT_BACKFILL=${EXPERIMENT7_ALLOW_DIAGNOSTIC_RETFACT_BACKFILL@Q} python3 - <<'PYPREFLIGHT'
import json
import os
import shlex
import sys
from pathlib import Path

repo = Path(os.environ['REPO_ROOT'])
cache_path = Path(os.environ['CACHE_JSON'])
matrix = os.environ['MATRIX'].split()
matched_env = Path(os.environ['MATCHED_ENV'])
preflight_json = Path(os.environ['PREFLIGHT_JSON'])
backfill_root = Path(os.environ['BACKFILL_ROOT'])
backfill_manifest = Path(os.environ['BACKFILL_MANIFEST'])
allow_diagnostic_backfill = os.environ.get('ALLOW_DIAGNOSTIC_RETFACT_BACKFILL') == '1'
errors = []
warnings = []
processed = {}
backfilled = {}

if not cache_path.is_file():
    errors.append(f"selection cache missing: {cache_path}")
else:
    payload = json.loads(cache_path.read_text(encoding='utf-8'))
    source_mode = payload.get('source_mode') or payload.get('metadata', {}).get('source_mode')
    items = payload.get('items') or payload.get('rows') or payload.get('examples') or payload.get('data') or []
    if source_mode != 'finqa_train_formal':
        errors.append(f"selection cache source_mode={source_mode!r}, expected finqa_train_formal")
    if len(items) != 6251:
        errors.append(f"selection cache count={len(items)}, expected 6251")

def sanitize(value: str) -> str:
    out = ''.join(ch for ch in value.upper().replace('-', '_') if ch.isalnum() or ch == '_')
    return out

source_candidates = {
    'finqa_flan_o': ['finqa_flan_o', 'old_finqa_flan_o'],
    'finqa_flan_z': ['finqa_flan_z', 'finqa_flan_z_new', 'old_finqa_flan_z', 'finqa_flan_z_assembler_few10_current'],
    'finqa_flan_m': ['finqa_flan_m', 'finqa_flan_m_new', 'old_finqa_flan_m'],
    'finqa_flan_d': ['finqa_flan_d', 'finqa_flan_d_new', 'old_finqa_flan_d', 'finqa_flan_d_preflight_all_prompt_smoke'],
    'finqa_mistral_o': ['finqa_mistral_o', 'finqa_Mistral_o', 'old_finqa_Mistral_o'],
    'finqa_mistral_z': ['finqa_mistral_z', 'finqa_mistral_z_new', 'finqa_Mistral_z', 'old_finqa_Mistral_z', 'finqa_Mistral_z_assembler_few10_current'],
    'finqa_mistral_m': ['finqa_mistral_m', 'finqa_mistral_m_new', 'finqa_Mistral_m', 'old_finqa_Mistral_m'],
    'finqa_mistral_d': ['finqa_mistral_d', 'finqa_mistral_d_new', 'finqa_Mistral_d', 'old_finqa_Mistral_d'],
    'finqa_t5gemma2_o': ['finqa_t5gemma2_o', 'old_finqa_t5gemma2_o'],
    'finqa_t5gemma2_z': ['finqa_t5gemma2_z', 'finqa_t5gemma2_z_assembler_few10_current', 'old_finqa_t5gemma2_z'],
    'finqa_t5gemma2_m': ['finqa_t5gemma2_m', 'old_finqa_t5gemma2_m'],
    'finqa_t5gemma2_d': ['finqa_t5gemma2_d', 'old_finqa_t5gemma2_d'],
}

def find_test(retriever: str) -> Path | None:
    base_rels = [
        'retriever/outputs/best_matched_with_retrieved_facts_and_questions.json',
        'retriever_0.3/outputs/best_matched_with_retrieved_facts_and_questions.json',
        'retriever_/outputs/best_matched_with_retrieved_facts_and_questions.json',
    ]
    if not retriever.startswith('finqa_mistral_'):
        base_rels.append('retriever0.2/outputs/best_matched_with_retrieved_facts_and_questions.json')
    for expt in source_candidates.get(retriever, [retriever]):
        for rel in base_rels:
            path = repo / 'Experiment' / expt / rel
            if path.is_file():
                return path
    return None

def find_dev(retriever: str) -> Path | None:
    preferred = repo / 'Experiment' / 'experiment_7_target_selection_gpt55_all_cases_20260612T012548Z' / 'retriever_sources' / f'{retriever}_finqa_dev' / 'best_matched_with_retrieved_facts_and_questions.json'
    if preferred.is_file():
        return preferred
    if not allow_diagnostic_backfill:
        return None
    backfilled_path = backfill_root / f'{retriever}_finqa_dev' / 'best_matched_with_retrieved_facts_and_questions.json'
    if backfilled_path.is_file():
        return backfilled_path
    matches = sorted((repo / 'Experiment').glob(f'*/retriever_sources/{retriever}_finqa_dev/best_matched_with_retrieved_facts_and_questions.json'))
    if matches:
        return matches[-1]
    legacy_backfilled = list((repo / 'Experiment').glob(f'*/backfilled_retriever_sources/{retriever}_finqa_dev/best_matched_with_retrieved_facts_and_questions.json'))
    if legacy_backfilled:
        return max(legacy_backfilled, key=lambda item: item.stat().st_mtime)
    return None


exports = [f"# Generated by experiment_7_runner.sh for {os.environ.get('EXPT_ID', '')}"]
for item in matrix:
    if ':' not in item:
        errors.append(f"bad matrix item: {item}")
        continue
    retriever, dataset = item.split(':', 1)
    path = find_dev(retriever) if dataset == 'finqa_dev' else find_test(retriever)
    key = f'{retriever}:{dataset}'
    if path is None:
        errors.append(f"matched-json missing: {key}")
        continue
    processed[key] = str(path)
    path_is_backfilled = False
    try:
        path.relative_to(backfill_root)
        path_is_backfilled = True
    except ValueError:
        path_is_backfilled = 'backfilled_retriever_sources' in path.parts
    if path_is_backfilled:
        backfilled[key] = str(path)
        if allow_diagnostic_backfill:
            warnings.append(f"matched-json backfilled: {key} -> {path}")
        else:
            errors.append(f"formal matched-json cannot use backfill: {key} -> {path}")
    var = 'MATCHED_JSON_' + sanitize(f'{retriever}_{dataset}')
    exports.append(f'export {var}={shlex.quote(str(path))}')

matched_env.parent.mkdir(parents=True, exist_ok=True)
matched_env.write_text('\n'.join(exports) + '\n', encoding='utf-8')
report = {
    'selection_cache_json': str(cache_path),
    'matched_env': str(matched_env),
    'processed_input_paths': processed,
    'backfilled_input_paths': backfilled,
    'backfill_manifest_json': str(backfill_manifest) if backfill_manifest.is_file() else None,
    'matched_count': len(processed),
    'expected_matched_count': len(matrix),
    'errors': errors,
    'warnings': warnings,
    'status': 'ok' if not errors and len(processed) == len(matrix) else 'blocked',
}
preflight_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(report, ensure_ascii=False, indent=2))
if report['status'] != 'ok':
    sys.exit(2)
PYPREFLIGHT
) 2>&1 | tee ${ORCH_DIR@Q}/preflight.log
rc=\${PIPESTATUS[0]}
set -e
timeline_event preflight finish "rc=\${rc}"
printf '%s\n' "\${rc}" > ${rc@Q}
exit "\${rc}"
EOF
  chmod +x "${runner}"
}

write_generator_worker() {
  local name="$1"
  local engine="$2"
  local rc="${ORCH_DIR}/${name}.rc"
  local runner="${ORCH_DIR}/${name}.sh"
  local extra_env="$3"
  local deps="$4"
  local full_limit="${5:--1}"
  cat >"${runner}" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
cd ${REPO_ROOT@Q}
rm -f ${rc@Q}
wait_for_rc_success() { local label="\$1"; local rc_path="\$2"; while [[ ! -f "\${rc_path}" ]]; do sleep 10; done; local value; value="\$(cat "\${rc_path}")"; if [[ "\${value}" != "0" ]]; then printf '[%s] dependency %s failed rc=%s path=%s\n' "\$(date -u +%Y-%m-%dT%H:%M:%SZ)" "\${label}" "\${value}" "\${rc_path}" >&2; return "\${value}"; fi; }
wait_for_terminal_state() { local label="\$1"; local state_path="\$2"; while [[ ! -f "\${state_path}" ]]; do sleep 10; done; STATE_PATH="\${state_path}" LABEL="\${label}" python3 - <<'PYSTATEWAIT'
import json, os, sys
from pathlib import Path
path = Path(os.environ['STATE_PATH'])
try:
    payload = json.loads(path.read_text(encoding='utf-8'))
except Exception as exc:
    print(f"[{os.environ['LABEL']}] malformed terminal state: {exc}", file=sys.stderr)
    raise SystemExit(2)
status = payload.get('status')
if status in {'completed', 'blocked'}:
    print(json.dumps({'dependency': os.environ['LABEL'], 'terminal_status': status, 'rc': payload.get('rc'), 'reason': payload.get('reason')}, ensure_ascii=False))
    raise SystemExit(0)
print(json.dumps({'dependency': os.environ['LABEL'], 'terminal_status': status, 'rc': payload.get('rc'), 'reason': payload.get('reason')}, ensure_ascii=False), file=sys.stderr)
raise SystemExit(int(payload.get('rc') or 2))
PYSTATEWAIT
}
write_terminal_state() { local status="\$1"; local state_rc="\$2"; local reason="\${3:-}"; local resume="\${4:-}"; STATE_PATH=${ORCH_DIR@Q}/${name}.state.json STATE_TIME="\$(date -u +%Y-%m-%dT%H:%M:%SZ)" STATE_NAME=${name@Q} STATE_ENGINE=${engine@Q} STATE_STATUS="\${status}" STATE_RC="\${state_rc}" STATE_REASON="\${reason}" STATE_RESUME="\${resume}" python3 - <<'PYSTATE'
import json, os
from pathlib import Path
payload = {
    "time": os.environ["STATE_TIME"],
    "name": os.environ["STATE_NAME"],
    "engine": os.environ["STATE_ENGINE"],
    "status": os.environ["STATE_STATUS"],
    "rc": int(os.environ["STATE_RC"]),
    "reason": os.environ.get("STATE_REASON") or None,
    "resume_command": os.environ.get("STATE_RESUME") or None,
}
Path(os.environ["STATE_PATH"]).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PYSTATE
}
finish_worker() { local worker_rc="\$1"; local reason="\${2:-}"; local resume="\${3:-}"; local status="completed"; if [[ "\${worker_rc}" != "0" ]]; then status="blocked"; fi; write_terminal_state "\${status}" "\${worker_rc}" "\${reason}" "\${resume}"; printf '%s\n' "\${worker_rc}" > ${rc@Q}; exit "\${worker_rc}"; }
trap 'exit_rc=$?; trap - EXIT; if [[ ! -f ${rc@Q} ]]; then write_terminal_state blocked "\${exit_rc}" worker_exit_without_finish; printf "%s\n" "\${exit_rc}" > ${rc@Q}; fi' EXIT
wait_for_rc_success preflight ${ORCH_DIR@Q}/preflight.rc || { rc_value=\$?; finish_worker "\${rc_value}" preflight_dependency_failed; }
${deps} || { rc_value=\$?; finish_worker "\${rc_value}" dependency_blocked; }
source ${MATCHED_ENV@Q}
common_env=(
  EXPT_ID=${EXPT_ID@Q}_${name@Q}
  ENGINES=${engine@Q}
  EXPERIMENT7_MATRIX=${EXPERIMENT7_MATRIX@Q}
  EXPERIMENT7_SELECTION_EXPT_ID=${SELECTION_EXPT_ID@Q}
  EXPERIMENT7_SELECTION_ENGINE=${SELECTION_ENGINE@Q}
  EXPERIMENT7_SELECTION_CACHE_BINDING_ROOT=${EXPT_DIR@Q}/selection_cache_binding
  EXAMPLE_SELECTION_MODE=cache
  EXAMPLE_SELECTION_REQUIRE_CACHE=1
  FORMAL_FINDER_READY=1
  RUN_RETRIEVER_INFER=0
  STRICT_INPUTS=1
  MAX_TOKENS=${MAX_TOKENS@Q}
  RESUME_OUTPUT=1
  SHOW_PROMPT=0
  FAIL_FAST_ON_EXECUTE_ERROR=1
  GPT_RETRY_AFTER_SECONDS=14400
  GPT_MAX_QUOTA_RETRIES=1
  ${extra_env}
)
run_phase() {
  local phase="\$1"
  local run_execute="\$2"
  local phase_limit="\$3"
  set +e
  env "\${common_env[@]}" RUN_EXECUTE="\${run_execute}" LIMIT="\${phase_limit}" bash dist/experiment_7_generator_answer.sh 2>&1 | tee ${ORCH_DIR@Q}/${name}_"\${phase}".log
  phase_rc=\${PIPESTATUS[0]}
  set -e
  return "\${phase_rc}"
}
run_phase validation 0 1 || { rc_value=\$?; finish_worker "\${rc_value}" validation_failed; }
run_phase smoke auto 1 || { rc_value=\$?; finish_worker "\${rc_value}" smoke_failed; }
first_gate_expt_id=${EXPT_ID@Q}_${name@Q}_first10
set +e
env "\${common_env[@]}" EXPT_ID="\${first_gate_expt_id}" EXPERIMENT7_MATRIX=${FIRST_GATE_MATRIX@Q} RUN_EXECUTE=auto LIMIT=${FIRST_GATE_LIMIT@Q} bash dist/experiment_7_generator_answer.sh 2>&1 | tee ${ORCH_DIR@Q}/${name}_first10.log
first_gate_rc=\${PIPESTATUS[0]}
set -e
if [[ "\${first_gate_rc}" -ne 0 ]]; then finish_worker "\${first_gate_rc}" first_gate_failed "cd ${REPO_ROOT@Q} && EXPT_ID=\${first_gate_expt_id} EXPERIMENT7_MATRIX=${FIRST_GATE_MATRIX@Q} RUN_EXECUTE=auto LIMIT=${FIRST_GATE_LIMIT@Q} bash dist/experiment_7_generator_answer.sh"; fi
set +e
FIRST_GATE_SCORE_REPORT="Experiment/\${first_gate_expt_id}/generator/score_report.json" python3 - <<'PYFIRSTGATE'
import json, math, os, sys
from pathlib import Path
report = Path(os.environ['FIRST_GATE_SCORE_REPORT'])
if not report.is_file():
    raise SystemExit(f'first gate score report missing: {report}')
payload = json.loads(report.read_text(encoding='utf-8'))
items = payload.get('items') or []
if not items:
    raise SystemExit('first gate has no score items')
item = items[0]
if item.get('failure_category'):
    raise SystemExit(f"first gate failure_category={item.get('failure_category')}")
if item.get('generated_nonempty_rows') != 10:
    raise SystemExit(f"first gate generated_nonempty_rows={item.get('generated_nonempty_rows')}, expected 10")
executed_rate = item.get('executed_non_null_rate')
if executed_rate is None or float(executed_rate) < 0.8:
    raise SystemExit(f'first gate executed_non_null_rate={executed_rate}, expected >=0.8')
output = Path(str(item.get('output_jsonl') or ''))
if not output.is_file():
    raise SystemExit(f'first gate output jsonl missing: {output}')
first = json.loads(output.read_text(encoding='utf-8').splitlines()[0])
if first.get('id') != 'ETR/2016/page_23.pdf-2':
    raise SystemExit(f"first gate first id={first.get('id')}, expected ETR/2016/page_23.pdf-2")
executed = first.get('executed')
if executed is None or not math.isclose(float(executed), 94.0, rel_tol=1e-9, abs_tol=1e-9):
    raise SystemExit(f'first gate first executed={executed}, expected 94')
print(json.dumps({'status': 'first10_gate_passed', 'score_report': str(report), 'output_jsonl': str(output)}, ensure_ascii=False))
PYFIRSTGATE
first_gate_check_rc=\${PIPESTATUS[0]}
set -e
if [[ "\${first_gate_check_rc}" -ne 0 ]]; then
  first_gate_reason=first_gate_contract_failed
  first_gate_report="Experiment/\${first_gate_expt_id}/generator/score_report.json"
  if [[ -f "\${first_gate_report}" ]]; then
    first_gate_category="\$(python3 - "\${first_gate_report}" <<'PYFIRSTREASON' || true
import json, sys
from pathlib import Path
try:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
except Exception:
    raise SystemExit(0)
for item in payload.get('items') or []:
    category = item.get('failure_category')
    if category:
        print(str(category))
        break
PYFIRSTREASON
)"
    if [[ -n "\${first_gate_category}" ]]; then
      first_gate_reason="first_gate_\${first_gate_category}"
    fi
  fi
  finish_worker "\${first_gate_check_rc}" "\${first_gate_reason}"
fi
set +e
run_phase full auto ${full_limit@Q}
rc_value=\$?
set -e
full_reason=full_finished
if [[ "\${rc_value}" -ne 0 ]]; then
  full_reason=full_failed
  full_report="Experiment/${EXPT_ID@Q}_${name}/generator/score_report.json"
  if [[ -f "\${full_report}" ]]; then
    full_category="\$(python3 - "\${full_report}" <<'PYFULLREASON' || true
import json, sys
from pathlib import Path
try:
    payload = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
except Exception:
    raise SystemExit(0)
for item in payload.get('items') or []:
    category = item.get('failure_category')
    if category:
        print(str(category))
        break
PYFULLREASON
)"
    if [[ -n "\${full_category}" ]]; then
      full_reason="full_\${full_category}"
    fi
  fi
fi
finish_worker "\${rc_value}" "\${full_reason}"
EOF
  chmod +x "${runner}"
}

write_chatmock_runner() {
  local runner="${ORCH_DIR}/chatmock_service.sh"
  local rc="${ORCH_DIR}/chatmock_service.rc"
  cat >"${runner}" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
cd ${REPO_ROOT@Q}
rm -f ${rc@Q}
while [[ ! -f ${ORCH_DIR@Q}/preflight.rc ]]; do sleep 10; done
if [[ "\$(cat ${ORCH_DIR@Q}/preflight.rc)" != "0" ]]; then printf '2\n' > ${rc@Q}; exit 2; fi
base_url="http://localhost:${CHATMOCK_PORT}/v1"
endpoint_ready() { BASE_URL="\${base_url}" python3 - <<'PYREADY' >/dev/null 2>&1
import os, urllib.request
with urllib.request.urlopen(os.environ['BASE_URL'].removesuffix('/v1') + '/v1/models', timeout=5) as r:
    raise SystemExit(0 if r.status < 400 else 1)
PYREADY
}
if ! endpoint_ready; then
  CHATMOCK_PORT=${CHATMOCK_PORT@Q} bash dist/start_chatmock_server.sh 2>&1 | tee ${ORCH_DIR@Q}/chatmock_server.log &
  server_pid=\$!
else
  server_pid=""
fi
for _ in {1..120}; do endpoint_ready && break; sleep 5; done
if ! endpoint_ready; then write_terminal_state blocked 2 endpoint_not_ready; printf '2\n' > ${rc@Q}; exit 2; fi
CHATMOCK_BASE_URL="\${base_url}" CHATMOCK_API_KEY="\${CHATMOCK_API_KEY:-key}" python3 - <<'PYCHAT'
import json, os, urllib.request
base = os.environ['CHATMOCK_BASE_URL'].rstrip('/')
headers = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + os.environ.get('CHATMOCK_API_KEY', 'key')}
for model in ('gpt-5.5', 'gpt-5.3-codex-spark'):
    body = json.dumps({'model': model, 'messages': [{'role': 'user', 'content': 'Return 1.'}], 'max_tokens': 8}).encode()
    req = urllib.request.Request(base + '/chat/completions', data=body, headers=headers, method='POST')
    with urllib.request.urlopen(req, timeout=120) as response:
        payload = json.loads(response.read().decode())
        if not payload.get('choices'):
            raise SystemExit(f'no choices for {model}')
print(json.dumps({'status': 'ok', 'models': ['gpt-5.5', 'gpt-5.3-codex-spark']}))
PYCHAT
printf '0\n' > ${rc@Q}
if [[ -n "\${server_pid}" ]]; then wait "\${server_pid}"; else while true; do sleep 3600; done; fi
EOF
  chmod +x "${runner}"
}

write_qwen_service_runner() {
  local runner="${ORCH_DIR}/qwen_vllm.sh"
  local rc="${ORCH_DIR}/qwen_vllm.rc"
  cat >"${runner}" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
cd ${REPO_ROOT@Q}
rm -f ${rc@Q}
while [[ ! -f ${ORCH_DIR@Q}/preflight.rc ]]; do sleep 10; done
if [[ "\$(cat ${ORCH_DIR@Q}/preflight.rc)" != "0" ]]; then printf '2\n' > ${rc@Q}; exit 2; fi
wait_for_terminal_state() { local label="\$1"; local state_path="\$2"; while [[ ! -f "\${state_path}" ]]; do sleep 10; done; STATE_PATH="\${state_path}" LABEL="\${label}" python3 - <<'PYSTATEWAIT'
import json, os, sys
from pathlib import Path
path = Path(os.environ['STATE_PATH'])
try:
    payload = json.loads(path.read_text(encoding='utf-8'))
except Exception as exc:
    print(f"[{os.environ['LABEL']}] malformed terminal state: {exc}", file=sys.stderr)
    raise SystemExit(2)
status = payload.get('status')
if status in {'completed', 'blocked'}:
    print(json.dumps({'dependency': os.environ['LABEL'], 'terminal_status': status, 'rc': payload.get('rc'), 'reason': payload.get('reason')}, ensure_ascii=False))
    raise SystemExit(0)
print(json.dumps({'dependency': os.environ['LABEL'], 'terminal_status': status, 'rc': payload.get('rc'), 'reason': payload.get('reason')}, ensure_ascii=False), file=sys.stderr)
raise SystemExit(int(payload.get('rc') or 2))
PYSTATEWAIT
}
MODELS_ROOT=${FQAN_MODELS_ROOT@Q}
HF_HOME="\${MODELS_ROOT}/.cache/huggingface"
TRANSFORMERS_CACHE="\${HF_HOME}"
QWEN3_6_MODEL_PATH="\${QWEN3_6_MODEL_PATH:-\${HF_HOME}/hub/models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/95a723d08a9490559dae23d0cff1d9466213d989}"
export MODELS_ROOT HF_HOME TRANSFORMERS_CACHE QWEN3_6_MODEL_PATH
base_url="http://localhost:${QWEN_VLLM_PORT}/v1"
endpoint_ready() { BASE_URL="\${base_url}" python3 - <<'PYREADY' >/dev/null 2>&1
import os, urllib.request
url = os.environ['BASE_URL'].removesuffix('/v1') + '/v1/models'
request = urllib.request.Request(url, headers={'Authorization': f"Bearer {os.environ.get('VLLM_API_KEY', 'EMPTY')}"})
with urllib.request.urlopen(request, timeout=5) as r:
    raise SystemExit(0 if r.status < 400 else 1)
PYREADY
}
if ! endpoint_ready; then
  ENGINE=qwen3_6 VLLM_PORT=${QWEN_VLLM_PORT@Q} VLLM_RUNTIME_PROFILE=qwen_fp8_tp2_precise_kv VLLM_TIMELINE_JSONL=${ORCH_DIR@Q}/qwen_vllm_timeline.jsonl MODELS_ROOT="\${MODELS_ROOT}" HF_HOME="\${HF_HOME}" TRANSFORMERS_CACHE="\${TRANSFORMERS_CACHE}" QWEN3_6_MODEL_PATH="\${QWEN3_6_MODEL_PATH}" bash dist/start_vllm_openai_server.sh 2>&1 | tee ${ORCH_DIR@Q}/qwen_vllm_server.log &
  server_pid=\$!
else
  server_pid=""
fi
for _ in {1..240}; do
  endpoint_ready && break
  if [[ -n "\${server_pid}" ]] && ! kill -0 "\${server_pid}" 2>/dev/null; then
    break
  fi
  sleep 10
done
if endpoint_ready; then
  TIMELINE_PATH=${ORCH_DIR@Q}/qwen_vllm_timeline.jsonl TIMELINE_TIME="$(date -u +%Y-%m-%dT%H:%M:%SZ)" TIMELINE_PORT=${QWEN_VLLM_PORT@Q} python3 - <<'PYQWENTIMELINE'
import json, os
payload = {"time": os.environ["TIMELINE_TIME"], "phase": "vllm_endpoint", "status": "ready", "engine": "qwen3_6", "port": os.environ["TIMELINE_PORT"], "detail": "/v1/models ready"}
with open(os.environ["TIMELINE_PATH"], "a", encoding="utf-8") as fh:
    fh.write(json.dumps(payload, ensure_ascii=False) + "\\n")
PYQWENTIMELINE
fi
write_qwen_state() { local status="\$1"; local state_rc="\$2"; local reason="\${3:-}"; STATE_PATH=${ORCH_DIR@Q}/qwen_vllm.state.json STATE_TIME="\$(date -u +%Y-%m-%dT%H:%M:%SZ)" STATE_STATUS="\${status}" STATE_RC="\${state_rc}" STATE_REASON="\${reason}" python3 - <<'PYSTATE'
import json, os
from pathlib import Path
payload = {"time": os.environ["STATE_TIME"], "name": "qwen_vllm", "engine": "qwen3_6", "status": os.environ["STATE_STATUS"], "rc": int(os.environ["STATE_RC"]), "reason": os.environ.get("STATE_REASON") or None}
Path(os.environ["STATE_PATH"]).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PYSTATE
}
if ! endpoint_ready; then
  mkdir -p ${EXPT_DIR@Q}/blockers
  BLOCKER_PATH=${EXPT_DIR@Q}/blockers/qwen_vllm_start.json SERVER_LOG=${ORCH_DIR@Q}/qwen_vllm_server.log python3 - <<'PYQWENBLOCK'
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

payload = {
    "time": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "experiment": "7",
    "stage": "qwen_vllm_start",
    "status": "blocked",
    "engine": "qwen3_6",
    "route": "vllm_openai_compatible",
    "reason": "qwen_vllm_endpoint_not_ready_after_server_start",
    "server_log": os.environ["SERVER_LOG"],
}
try:
    payload["gpu_snapshot"] = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip().splitlines()
except Exception as exc:
    payload["gpu_snapshot_error"] = str(exc)
Path(os.environ["BLOCKER_PATH"]).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False))
PYQWENBLOCK
  write_qwen_state blocked 2 endpoint_not_ready
  printf '2\\n' > ${rc@Q}
  exit 2
fi
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv | tee ${ORCH_DIR@Q}/qwen_gpu_ready.csv || true
write_qwen_state completed 0 ready
printf '0\n' > ${rc@Q}
if [[ -n "\${server_pid}" ]]; then wait "\${server_pid}"; else while true; do sleep 3600; done; fi
EOF
  chmod +x "${runner}"
}

write_gpt41_gate_runner() {
  local runner="${ORCH_DIR}/gpt41_gate.sh"
  local rc="${ORCH_DIR}/gpt41_gate.rc"
  cat >"${runner}" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
cd ${REPO_ROOT@Q}
rm -f ${rc@Q}
wait_for_rc_success() { local label="\$1"; local rc_path="\$2"; while [[ ! -f "\${rc_path}" ]]; do sleep 10; done; local value; value="\$(cat "\${rc_path}")"; if [[ "\${value}" != "0" ]]; then printf '[%s] dependency %s failed rc=%s path=%s\n' "\$(date -u +%Y-%m-%dT%H:%M:%SZ)" "\${label}" "\${value}" "\${rc_path}" >&2; return "\${value}"; fi; }
write_terminal_state() { local status="\$1"; local state_rc="\$2"; local reason="\${3:-}"; STATE_PATH=${ORCH_DIR@Q}/gpt41_gate.state.json STATE_TIME="\$(date -u +%Y-%m-%dT%H:%M:%SZ)" STATE_STATUS="\${status}" STATE_RC="\${state_rc}" STATE_REASON="\${reason}" python3 - <<'PYSTATE'
import json, os
from pathlib import Path
payload = {"time": os.environ["STATE_TIME"], "name": "gpt41_gate", "engine": "gpt4_1", "status": os.environ["STATE_STATUS"], "rc": int(os.environ["STATE_RC"]), "reason": os.environ.get("STATE_REASON") or None}
Path(os.environ["STATE_PATH"]).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PYSTATE
}
finish_gate() { local gate_rc="\$1"; local reason="\${2:-}"; local status="completed"; if [[ "\${gate_rc}" != "0" ]]; then status="blocked"; fi; write_terminal_state "\${status}" "\${gate_rc}" "\${reason}"; printf '%s\n' "\${gate_rc}" > ${rc@Q}; exit "\${gate_rc}"; }
wait_for_terminal_state() { local label="\$1"; local state_path="\$2"; while [[ ! -f "\${state_path}" ]]; do sleep 10; done; STATE_PATH="\${state_path}" LABEL="\${label}" python3 - <<'PYSTATEWAIT'
import json, os, sys
from pathlib import Path
path = Path(os.environ['STATE_PATH'])
try:
    payload = json.loads(path.read_text(encoding='utf-8'))
except Exception as exc:
    print(f"[{os.environ['LABEL']}] malformed terminal state: {exc}", file=sys.stderr)
    raise SystemExit(2)
status = payload.get('status')
if status in {'completed', 'blocked'}:
    print(json.dumps({'dependency': os.environ['LABEL'], 'terminal_status': status, 'rc': payload.get('rc'), 'reason': payload.get('reason')}, ensure_ascii=False))
    raise SystemExit(0)
print(json.dumps({'dependency': os.environ['LABEL'], 'terminal_status': status, 'rc': payload.get('rc'), 'reason': payload.get('reason')}, ensure_ascii=False), file=sys.stderr)
raise SystemExit(int(payload.get('rc') or 2))
PYSTATEWAIT
}
wait_for_rc_success preflight ${ORCH_DIR@Q}/preflight.rc || { rc_value=\$?; finish_gate "\${rc_value}" preflight_dependency_failed; }
source ${MATCHED_ENV@Q}
eval "\$(VARIABLES_MD=${FQAN_ASSET_ROOT@Q}/workspace/variables.md python3 - <<'PYAZURE'
import os
from pathlib import Path

allowed = {
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_DEPLOYMENT",
    "AZURE_OPENAI_API_VERSION",
}
path = Path(os.environ["VARIABLES_MD"])
if path.is_file():
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line.startswith("export ") or "=" not in line:
            continue
        key = line[len("export "):].split("=", 1)[0].strip()
        if key in allowed:
            print(line)
PYAZURE
)"
unset OPENAI_BASE_URL
common_env=(EXPT_ID=${EXPT_ID@Q}_gpt41_gate ENGINES=gpt4_1 EXPERIMENT7_MATRIX=${EXPERIMENT7_MATRIX@Q} EXPERIMENT7_SELECTION_EXPT_ID=${SELECTION_EXPT_ID@Q} EXPERIMENT7_SELECTION_ENGINE=${SELECTION_ENGINE@Q} EXPERIMENT7_SELECTION_CACHE_BINDING_ROOT=${EXPT_DIR@Q}/selection_cache_binding EXAMPLE_SELECTION_MODE=cache EXAMPLE_SELECTION_REQUIRE_CACHE=1 FORMAL_FINDER_READY=1 RUN_RETRIEVER_INFER=0 STRICT_INPUTS=1 MAX_TOKENS=${MAX_TOKENS@Q} RESUME_OUTPUT=1 SHOW_PROMPT=0 GPT_RETRY_AFTER_SECONDS=14400 GPT_MAX_QUOTA_RETRIES=1)
set +e
env "\${common_env[@]}" RUN_EXECUTE=0 LIMIT=1 bash dist/experiment_7_generator_answer.sh 2>&1 | tee ${ORCH_DIR@Q}/gpt41_validation.log
validation_rc=\${PIPESTATUS[0]}
set -e
if [[ "\${validation_rc}" -ne 0 ]]; then finish_gate "\${validation_rc}" validation_failed; fi
set +e
env "\${common_env[@]}" RUN_EXECUTE=auto LIMIT=1 bash dist/experiment_7_generator_answer.sh 2>&1 | tee ${ORCH_DIR@Q}/gpt41_smoke.log
smoke_rc=\${PIPESTATUS[0]}
set -e
if [[ "\${smoke_rc}" -ne 0 ]]; then finish_gate "\${smoke_rc}" smoke_failed; fi
gpt41_scored_count="\$(jq -r '.status_counts.scored // 0' Experiment/${EXPT_ID@Q}_gpt41_gate/generator/score_report.json 2>/dev/null || printf '0')"
if [[ "\${gpt41_scored_count}" == "0" ]]; then
  printf '[%s] gpt4_1 smoke produced no scored cases; treating route as blocked without model downgrade.\n' "\$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  finish_gate 0 credential_or_route_blocked_no_scored_cases
fi
set +e
env "\${common_env[@]}" RUN_EXECUTE=auto LIMIT=-1 bash dist/experiment_7_generator_answer.sh 2>&1 | tee ${ORCH_DIR@Q}/gpt41_full.log
full_rc=\${PIPESTATUS[0]}
set -e
finish_gate "\${full_rc}" full_finished
EOF
  chmod +x "${runner}"
}

write_mistral_runner() {
  local runner="${ORCH_DIR}/mistral4.sh"
  local rc="${ORCH_DIR}/mistral4.rc"
  cat >"${runner}" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
cd ${REPO_ROOT@Q}
rm -f ${rc@Q}
wait_for_terminal_state() { local label="\$1"; local state_path="\$2"; while [[ ! -f "\${state_path}" ]]; do sleep 10; done; STATE_PATH="\${state_path}" LABEL="\${label}" python3 - <<'PYSTATEWAIT'
import json, os, sys
from pathlib import Path
path = Path(os.environ['STATE_PATH'])
try:
    payload = json.loads(path.read_text(encoding='utf-8'))
except Exception as exc:
    print(f"[{os.environ['LABEL']}] malformed terminal state: {exc}", file=sys.stderr)
    raise SystemExit(2)
status = payload.get('status')
if status in {'completed', 'blocked'}:
    print(json.dumps({'dependency': os.environ['LABEL'], 'terminal_status': status, 'rc': payload.get('rc'), 'reason': payload.get('reason')}, ensure_ascii=False))
    raise SystemExit(0)
print(json.dumps({'dependency': os.environ['LABEL'], 'terminal_status': status, 'rc': payload.get('rc'), 'reason': payload.get('reason')}, ensure_ascii=False), file=sys.stderr)
raise SystemExit(int(payload.get('rc') or 2))
PYSTATEWAIT
}
write_terminal_state() { local status="\$1"; local state_rc="\$2"; local reason="\${3:-}"; STATE_PATH=${ORCH_DIR@Q}/mistral4.state.json STATE_TIME="\$(date -u +%Y-%m-%dT%H:%M:%SZ)" STATE_STATUS="\${status}" STATE_RC="\${state_rc}" STATE_REASON="\${reason}" python3 - <<'PYSTATE'
import json, os
from pathlib import Path
payload = {"time": os.environ["STATE_TIME"], "name": "mistral4", "engine": "mistral4", "status": os.environ["STATE_STATUS"], "rc": int(os.environ["STATE_RC"]), "reason": os.environ.get("STATE_REASON") or None}
Path(os.environ["STATE_PATH"]).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PYSTATE
}
while [[ ! -f ${ORCH_DIR@Q}/preflight.rc ]]; do sleep 10; done
preflight_value="\$(cat ${ORCH_DIR@Q}/preflight.rc)"
if [[ "\${preflight_value}" != "0" ]]; then
  write_terminal_state failed "\${preflight_value}" preflight_dependency_failed
  printf '%s\n' "\${preflight_value}" > ${rc@Q}
  exit "\${preflight_value}"
fi
source ${MATCHED_ENV@Q}
base_url="http://localhost:${MISTRAL_LLAMA_CPP_PORT}/v1"
: "\${LLAMA_CPP_SERVER_BIN:=${FQAN_UTILS_ROOT}/llama.cpp/build/bin/llama-server}"
: "\${LLAMA_CPP_MODEL_PATH:=${FQAN_MODELS_ROOT}/mistral_small_4_119b_2603_gguf/UD-Q4_K_M/Mistral-Small-4-119B-2603-UD-Q4_K_M-00001-of-00003.gguf}"
: "\${LLAMA_CPP_MODEL_ALIAS:=mistral4}"
: "\${LLAMA_CPP_QUANT:=UD-Q4_K_M}"
: "\${LLAMA_CPP_CTX_SIZE:=8192}"
: "\${LLAMA_CPP_PARALLEL:=1}"
: "\${LLAMA_CPP_SPLIT_MODE:=row}"
: "\${LLAMA_CPP_TENSOR_SPLIT:=1,1}"
: "\${LLAMA_CPP_MAIN_GPU:=0}"
: "\${LLAMA_CPP_N_GPU_LAYERS:=20}"
: "\${LLAMA_CPP_BATCH_SIZE:=192}"
: "\${LLAMA_CPP_UBATCH_SIZE:=48}"
: "\${LLAMA_CPP_CACHE_TYPE_K:=f16}"
: "\${LLAMA_CPP_CACHE_TYPE_V:=f16}"
: "\${LLAMA_CPP_CPU_MOE:=}"
: "\${LLAMA_CPP_OP_OFFLOAD:=off}"
: "\${LLAMA_CPP_FLASH_ATTN:=off}"
: "\${LLAMA_CPP_CACHE_RAM:=0}"
endpoint_ready() { BASE_URL="\${base_url}" python3 - <<'PYREADY' >/dev/null 2>&1
import os, urllib.request
with urllib.request.urlopen(os.environ['BASE_URL'].removesuffix('/v1') + '/v1/models', timeout=5) as r:
    raise SystemExit(0 if r.status < 400 else 1)
PYREADY
}
if ! endpoint_ready; then
  server_bin="\${LLAMA_CPP_SERVER_BIN:-\$(command -v llama-server || true)}"
  if [[ -z "\${server_bin}" || -z "\${LLAMA_CPP_MODEL_PATH:-}" ]]; then
    printf '[%s] missing llama.cpp server or LLAMA_CPP_MODEL_PATH; cannot start Mistral4 GGUF route.\n' "\$(date -u +%Y-%m-%dT%H:%M:%SZ)" >&2
    write_terminal_state blocked 2 missing_llama_cpp_or_model
    printf '2\n' > ${rc@Q}
    exit 2
  fi
  server_args=(--host localhost --port ${MISTRAL_LLAMA_CPP_PORT@Q} --model "\${LLAMA_CPP_MODEL_PATH}" --alias "\${LLAMA_CPP_MODEL_ALIAS}" --ctx-size "\${LLAMA_CPP_CTX_SIZE}" --parallel "\${LLAMA_CPP_PARALLEL}")
  [[ -n "\${LLAMA_CPP_N_GPU_LAYERS:-}" ]] && server_args+=(--n-gpu-layers "\${LLAMA_CPP_N_GPU_LAYERS}")
  [[ -n "\${LLAMA_CPP_SPLIT_MODE:-}" ]] && server_args+=(--split-mode "\${LLAMA_CPP_SPLIT_MODE}")
  [[ -n "\${LLAMA_CPP_TENSOR_SPLIT:-}" ]] && server_args+=(--tensor-split "\${LLAMA_CPP_TENSOR_SPLIT}")
  [[ -n "\${LLAMA_CPP_MAIN_GPU:-}" ]] && server_args+=(--main-gpu "\${LLAMA_CPP_MAIN_GPU}")
  [[ -n "\${LLAMA_CPP_DEVICE:-}" ]] && server_args+=(--device "\${LLAMA_CPP_DEVICE}")
  [[ -n "\${LLAMA_CPP_BATCH_SIZE:-}" ]] && server_args+=(--batch-size "\${LLAMA_CPP_BATCH_SIZE}")
  [[ -n "\${LLAMA_CPP_UBATCH_SIZE:-}" ]] && server_args+=(--ubatch-size "\${LLAMA_CPP_UBATCH_SIZE}")
  [[ -n "\${LLAMA_CPP_CACHE_TYPE_K:-}" ]] && server_args+=(--cache-type-k "\${LLAMA_CPP_CACHE_TYPE_K}")
  [[ -n "\${LLAMA_CPP_CACHE_TYPE_V:-}" ]] && server_args+=(--cache-type-v "\${LLAMA_CPP_CACHE_TYPE_V}")
  [[ -n "\${LLAMA_CPP_FIT_TARGET:-}" ]] && server_args+=(--fit-target "\${LLAMA_CPP_FIT_TARGET}")
  [[ -n "\${LLAMA_CPP_FIT_CTX:-}" ]] && server_args+=(--fit-ctx "\${LLAMA_CPP_FIT_CTX}")
  [[ -n "\${LLAMA_CPP_FLASH_ATTN:-}" ]] && server_args+=(--flash-attn "\${LLAMA_CPP_FLASH_ATTN}")
  [[ -n "\${LLAMA_CPP_CACHE_RAM:-}" ]] && server_args+=(--cache-ram "\${LLAMA_CPP_CACHE_RAM}")
  [[ -n "\${LLAMA_CPP_N_CPU_MOE:-}" ]] && server_args+=(--n-cpu-moe "\${LLAMA_CPP_N_CPU_MOE}")
  case "\${LLAMA_CPP_CPU_MOE:-}" in 1|true|TRUE|on|ON|yes|YES) server_args+=(--cpu-moe) ;; esac
  case "\${LLAMA_CPP_KV_OFFLOAD:-}" in 0|false|FALSE|off|OFF|no|NO) server_args+=(--no-kv-offload) ;; 1|true|TRUE|on|ON|yes|YES) server_args+=(--kv-offload) ;; esac
  case "\${LLAMA_CPP_OP_OFFLOAD:-}" in 0|false|FALSE|off|OFF|no|NO) server_args+=(--no-op-offload) ;; 1|true|TRUE|on|ON|yes|YES) server_args+=(--op-offload) ;; esac
  case "\${LLAMA_CPP_NO_MMAP:-}" in 1|true|TRUE|on|ON|yes|YES) server_args+=(--no-mmap) ;; esac
  "\${server_bin}" "\${server_args[@]}" 2>&1 | tee ${ORCH_DIR@Q}/mistral4_llama_cpp_server.log &
  server_pid=\$!
else
  server_pid=""
fi
for _ in {1..240}; do endpoint_ready && break; sleep 10; done
if ! endpoint_ready; then write_terminal_state blocked 2 endpoint_not_ready; printf '2\n' > ${rc@Q}; exit 2; fi
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv | tee ${ORCH_DIR@Q}/mistral_gpu_ready.csv || true
set +e
env EXPT_ID=${EXPT_ID@Q}_mistral4 ENGINES=mistral4 EXPERIMENT7_MATRIX=${EXPERIMENT7_MATRIX@Q} EXPERIMENT7_SELECTION_EXPT_ID=${SELECTION_EXPT_ID@Q} EXPERIMENT7_SELECTION_ENGINE=${SELECTION_ENGINE@Q} EXPERIMENT7_SELECTION_CACHE_BINDING_ROOT=${EXPT_DIR@Q}/selection_cache_binding EXAMPLE_SELECTION_MODE=cache EXAMPLE_SELECTION_REQUIRE_CACHE=1 FORMAL_FINDER_READY=1 RUN_RETRIEVER_INFER=0 STRICT_INPUTS=1 MAX_TOKENS=${MAX_TOKENS@Q} RESUME_OUTPUT=1 SHOW_PROMPT=0 RUN_EXECUTE=auto LIMIT=-1 MISTRAL_SMALL_RUNTIME_BACKEND=llama_cpp LLAMA_CPP_BASE_URL="\${base_url}" LLAMA_CPP_MODEL_PATH="\${LLAMA_CPP_MODEL_PATH}" LLAMA_CPP_MODEL_ALIAS="\${LLAMA_CPP_MODEL_ALIAS}" LLAMA_CPP_QUANT="\${LLAMA_CPP_QUANT}" LLAMA_CPP_CTX_SIZE="\${LLAMA_CPP_CTX_SIZE}" LLAMA_CPP_N_GPU_LAYERS="\${LLAMA_CPP_N_GPU_LAYERS:-}" LLAMA_CPP_TENSOR_SPLIT="\${LLAMA_CPP_TENSOR_SPLIT:-}" LLAMA_CPP_SPLIT_MODE="\${LLAMA_CPP_SPLIT_MODE:-}" LLAMA_CPP_PARALLEL="\${LLAMA_CPP_PARALLEL}" LLAMA_CPP_BATCH_SIZE="\${LLAMA_CPP_BATCH_SIZE:-}" LLAMA_CPP_UBATCH_SIZE="\${LLAMA_CPP_UBATCH_SIZE:-}" LLAMA_CPP_CACHE_TYPE_K="\${LLAMA_CPP_CACHE_TYPE_K:-}" LLAMA_CPP_CACHE_TYPE_V="\${LLAMA_CPP_CACHE_TYPE_V:-}" LLAMA_CPP_KV_OFFLOAD="\${LLAMA_CPP_KV_OFFLOAD:-}" LLAMA_CPP_CPU_MOE="\${LLAMA_CPP_CPU_MOE:-}" LLAMA_CPP_N_CPU_MOE="\${LLAMA_CPP_N_CPU_MOE:-}" LLAMA_CPP_FIT_TARGET="\${LLAMA_CPP_FIT_TARGET:-}" LLAMA_CPP_FIT_CTX="\${LLAMA_CPP_FIT_CTX:-}" LLAMA_CPP_OP_OFFLOAD="\${LLAMA_CPP_OP_OFFLOAD:-}" LLAMA_CPP_FLASH_ATTN="\${LLAMA_CPP_FLASH_ATTN:-}" LLAMA_CPP_CACHE_RAM="\${LLAMA_CPP_CACHE_RAM:-}" LLAMA_CPP_DEVICE="\${LLAMA_CPP_DEVICE:-}" LLAMA_CPP_MAIN_GPU="\${LLAMA_CPP_MAIN_GPU:-}" LLAMA_CPP_NO_MMAP="\${LLAMA_CPP_NO_MMAP:-}" bash dist/experiment_7_generator_answer.sh 2>&1 | tee ${ORCH_DIR@Q}/mistral4_full.log
run_rc=\${PIPESTATUS[0]}
set -e
if [[ "\${run_rc}" == "0" ]]; then write_terminal_state completed 0 full_finished; else write_terminal_state blocked "\${run_rc}" full_failed; fi
printf '%s\n' "\${run_rc}" > ${rc@Q}
if [[ -n "\${server_pid}" ]]; then wait "\${server_pid}"; fi
exit "\${run_rc}"
EOF
  chmod +x "${runner}"
}

write_llama4_smoke_runner() {
  local runner="${ORCH_DIR}/llama4.sh"
  local rc="${ORCH_DIR}/llama4.rc"
  cat >"${runner}" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
cd ${REPO_ROOT@Q}
rm -f ${rc@Q}
server_pid=""
cleanup() {
  if [[ -n "\${server_pid:-}" ]] && kill -0 "\${server_pid}" 2>/dev/null; then
    kill "\${server_pid}" 2>/dev/null || true
    sleep 5
    kill -9 "\${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
wait_for_terminal_state() { local label="\$1"; local state_path="\$2"; while [[ ! -f "\${state_path}" ]]; do sleep 10; done; STATE_PATH="\${state_path}" LABEL="\${label}" python3 - <<'PYSTATEWAIT'
import json, os, sys
from pathlib import Path
path = Path(os.environ['STATE_PATH'])
try:
    payload = json.loads(path.read_text(encoding='utf-8'))
except Exception as exc:
    print(f"[{os.environ['LABEL']}] malformed terminal state: {exc}", file=sys.stderr)
    raise SystemExit(2)
status = payload.get('status')
if status in {'completed', 'blocked'}:
    print(json.dumps({'dependency': os.environ['LABEL'], 'terminal_status': status, 'rc': payload.get('rc'), 'reason': payload.get('reason')}, ensure_ascii=False))
    raise SystemExit(0)
print(json.dumps({'dependency': os.environ['LABEL'], 'terminal_status': status, 'rc': payload.get('rc'), 'reason': payload.get('reason')}, ensure_ascii=False), file=sys.stderr)
raise SystemExit(int(payload.get('rc') or 2))
PYSTATEWAIT
}
write_terminal_state() { local status="\$1"; local state_rc="\$2"; local reason="\${3:-}"; STATE_PATH=${ORCH_DIR@Q}/llama4.state.json STATE_TIME="\$(date -u +%Y-%m-%dT%H:%M:%SZ)" STATE_STATUS="\${status}" STATE_RC="\${state_rc}" STATE_REASON="\${reason}" python3 - <<'PYSTATE'
import json, os
from pathlib import Path
payload = {"time": os.environ["STATE_TIME"], "name": "llama4", "engine": "llama4", "status": os.environ["STATE_STATUS"], "rc": int(os.environ["STATE_RC"]), "reason": os.environ.get("STATE_REASON") or None}
Path(os.environ["STATE_PATH"]).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PYSTATE
}
wait_for_terminal_state qwen ${ORCH_DIR@Q}/qwen.state.json || { rc_value=\$?; write_terminal_state failed "\${rc_value}" dependency_failed; printf '%s\n' "\${rc_value}" > ${rc@Q}; exit "\${rc_value}"; }
source ${MATCHED_ENV@Q}
MODELS_ROOT=${FQAN_MODELS_ROOT@Q}
HF_HOME="\${MODELS_ROOT}/.cache/huggingface"
TRANSFORMERS_CACHE="\${HF_HOME}"
LLAMA4_MODEL_PATH="\${LLAMA4_MODEL_PATH:-\${MODELS_ROOT}/meta-llama/Llama-4-Scout-17B-16E-Instruct}"
export MODELS_ROOT HF_HOME TRANSFORMERS_CACHE LLAMA4_MODEL_PATH
if ! LLAMA4_MODEL_PATH="\${LLAMA4_MODEL_PATH}" HF_HOME="\${HF_HOME}" python3 - <<'PYWEIGHTS'
import os
from pathlib import Path
roots = [Path(os.environ['LLAMA4_MODEL_PATH'])]
hf_home = Path(os.environ['HF_HOME'])
roots.extend([
    hf_home / 'hub' / 'models--meta-llama--Llama-4-Scout-17B-16E-Instruct' / 'snapshots',
    hf_home / 'models--meta-llama--Llama-4-Scout-17B-16E-Instruct' / 'snapshots',
])
for root in roots:
    if not root.exists():
        continue
    for path in root.rglob('*'):
        if not path.is_file() or '.no_exist' in path.parts:
            continue
        if path.suffix in {'.safetensors', '.bin'} and path.stat().st_size > 1_000_000:
            raise SystemExit(0)
raise SystemExit(1)
PYWEIGHTS
then
  write_terminal_state blocked 2 weights_missing_or_gated
  printf '2\n' > ${rc@Q}
  exit 2
fi
printf '[%s] releasing qwen vLLM before llama4 smoke\n' "\$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee ${ORCH_DIR@Q}/llama4_gpu_release.log
pkill -f 'vllm.entrypoints.openai.api_server.*Qwen3.6-35B-A3B-FP8' 2>/dev/null || true
pkill -f 'VLLM::.*' 2>/dev/null || true
for _ in {1..60}; do pgrep -f 'vllm.entrypoints.openai.api_server.*Qwen3.6-35B-A3B-FP8|VLLM::' >/dev/null || break; sleep 5; done
base_url="http://localhost:${LLAMA4_VLLM_PORT}/v1"
endpoint_ready() { BASE_URL="\${base_url}" python3 - <<'PYREADY' >/dev/null 2>&1
import os, urllib.request
url = os.environ['BASE_URL'].removesuffix('/v1') + '/v1/models'
request = urllib.request.Request(url, headers={'Authorization': f"Bearer {os.environ.get('VLLM_API_KEY', 'EMPTY')}"})
with urllib.request.urlopen(request, timeout=5) as r:
    raise SystemExit(0 if r.status < 400 else 1)
PYREADY
}
if ! endpoint_ready; then
  ENGINE=llama4 VLLM_PORT=${LLAMA4_VLLM_PORT@Q} VLLM_RUNTIME_PROFILE=llama4_scout_tp2_short_context VLLM_TIMELINE_JSONL=${ORCH_DIR@Q}/llama4_vllm_timeline.jsonl bash dist/start_vllm_openai_server.sh 2>&1 | tee ${ORCH_DIR@Q}/llama4_vllm_server.log &
  server_pid=\$!
fi
for _ in {1..360}; do endpoint_ready && break; sleep 10; done
if endpoint_ready; then
  TIMELINE_PATH=${ORCH_DIR@Q}/llama4_vllm_timeline.jsonl TIMELINE_TIME="\$(date -u +%Y-%m-%dT%H:%M:%SZ)" TIMELINE_PORT=${LLAMA4_VLLM_PORT@Q} python3 - <<'PYLLAMA4TIMELINE'
import json, os
payload = {"time": os.environ["TIMELINE_TIME"], "phase": "vllm_endpoint", "status": "ready", "engine": "llama4", "port": os.environ["TIMELINE_PORT"], "detail": "/v1/models ready"}
with open(os.environ["TIMELINE_PATH"], "a", encoding="utf-8") as fh:
    fh.write(json.dumps(payload, ensure_ascii=False) + "\\n")
PYLLAMA4TIMELINE
fi
if ! endpoint_ready; then write_terminal_state blocked 2 endpoint_not_ready; printf '2\n' > ${rc@Q}; exit 2; fi
SENTINEL_PATH=${ORCH_DIR@Q}/answer_llama4.smoke_started SENTINEL_TIME="\$(date -u +%Y-%m-%dT%H:%M:%SZ)" python3 - <<'PYSENTINEL'
import json, os
from pathlib import Path
payload = {'time': os.environ['SENTINEL_TIME'], 'status': 'smoke_started', 'engine': 'llama4'}
Path(os.environ['SENTINEL_PATH']).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(payload, ensure_ascii=False))
PYSENTINEL
nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv | tee ${ORCH_DIR@Q}/llama4_gpu_ready.csv || true
set +e
env EXPT_ID=${EXPT_ID@Q}_llama4_smoke ENGINES=llama4 EXPERIMENT7_MATRIX=${FIRST_GATE_MATRIX@Q} EXPERIMENT7_SELECTION_EXPT_ID=${SELECTION_EXPT_ID@Q} EXPERIMENT7_SELECTION_ENGINE=${SELECTION_ENGINE@Q} EXPERIMENT7_SELECTION_CACHE_BINDING_ROOT=${EXPT_DIR@Q}/selection_cache_binding EXAMPLE_SELECTION_MODE=cache EXAMPLE_SELECTION_REQUIRE_CACHE=1 FORMAL_FINDER_READY=1 RUN_RETRIEVER_INFER=0 STRICT_INPUTS=1 MAX_TOKENS=${MAX_TOKENS@Q} RESUME_OUTPUT=1 SHOW_PROMPT=0 RUN_EXECUTE=auto LIMIT=${FIRST_GATE_LIMIT@Q} VLLM_BASE_URL="\${base_url}" VLLM_SERVED_MODEL_NAME=llama4 bash dist/experiment_7_generator_answer.sh 2>&1 | tee ${ORCH_DIR@Q}/llama4_smoke.log
run_rc=\${PIPESTATUS[0]}
set -e
if [[ "\${run_rc}" == "0" ]]; then write_terminal_state completed 0 smoke_finished; else write_terminal_state blocked "\${run_rc}" smoke_failed; fi
printf '%s\n' "\${run_rc}" > ${rc@Q}
exit "\${run_rc}"
EOF
  chmod +x "${runner}"
}

write_report_runner() {
  local runner="${ORCH_DIR}/report.sh"
  cat >"${runner}" <<EOF
#!/usr/bin/env bash
set -Eeuo pipefail
cd ${REPO_ROOT@Q}
while true; do
  REPORT_JSON=${EXPT_DIR@Q}/score_report.json REPORT_MD=${EXPT_DIR@Q}/score_report.md STATUS_JSON=${STATUS_JSON@Q} QUEUE_LOG_JSON=${QUEUE_LOG_JSON@Q} INDEX_PATH=${FQAN_LOG_ROOT@Q}/index.json ORCH_DIR=${ORCH_DIR@Q} TOP_EXPT_ID=${EXPT_ID@Q} WORKSPACE_ROOT=${WORKSPACE_ROOT@Q} REPO_ROOT=${REPO_ROOT@Q} EXPERIMENT7_USE_CHATMOCK_SERVICE=${EXPERIMENT7_USE_CHATMOCK_SERVICE@Q} python3 - <<'PYREPORT'
import json, os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

orch = Path(os.environ['ORCH_DIR'])
now = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
children = [
    ('gpt55', ['gpt55']),
    ('gptCodexS', ['gptCodexS']),
    ('qwen', ['qwen']),
    ('gpt41_gate', ['gpt41_gate']),
    ('llama4', ['llama4_smoke']),
]
items = []
for child, suffixes in children:
    for suffix in suffixes:
        child_expt = f"{os.environ['TOP_EXPT_ID']}_{suffix}"
        path = Path('Experiment') / child_expt / 'generator/score_report.json'
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding='utf-8'))
        except Exception as exc:
            items.append({'child_expt_id': child_expt, 'engine_group': child, 'score_status': 'invalid', 'error': str(exc), 'score_report_json': str(path)})
            continue
        for item in payload.get('items', []):
            item = dict(item)
            item['child_expt_id'] = child_expt
            item['engine_group'] = child
            item['score_report_json'] = str(path)
            items.append(item)
        break
quota_checkpoints = []
for path in sorted((Path('Experiment') / os.environ['TOP_EXPT_ID']).glob('**/*.quota_wait.json')):
    try:
        quota_checkpoints.append({'path': str(path), 'payload': json.loads(path.read_text(encoding='utf-8'))})
    except Exception as exc:
        quota_checkpoints.append({'path': str(path), 'error': str(exc)})
blocker_audits = []
for blocker_path in sorted((Path('Experiment') / os.environ['TOP_EXPT_ID'] / 'blockers').glob('*.json')):
    try:
        blocker_audits.append({'path': str(blocker_path), 'payload': json.loads(blocker_path.read_text(encoding='utf-8'))})
    except Exception as exc:
        blocker_audits.append({'path': str(blocker_path), 'error': str(exc)})
selection_cache_binding = None
selection_example_artifacts = []
binding_path = Path('Experiment') / os.environ['TOP_EXPT_ID'] / 'selection_cache_binding' / 'execution_status.json'
if binding_path.is_file():
    try:
        selection_cache_binding = json.loads(binding_path.read_text(encoding='utf-8'))
        for bind_item in selection_cache_binding.get('items', []):
            enriched = dict(bind_item)
            report_json = enriched.get('report_json')
            if report_json and Path(report_json).is_file():
                try:
                    materialization = json.loads(Path(report_json).read_text(encoding='utf-8'))
                    for key in (
                        'rows',
                        'shot_number',
                        'cache_hit_rows',
                        'cache_missing_rows',
                        'extracted_materialized_rows',
                        'skipped_no_binding_payload_rows',
                        'output_jsonl',
                        'extracted_jsonl',
                    ):
                        enriched[key] = materialization.get(key)
                except Exception as exc:
                    enriched['materialization_report_error'] = str(exc)
            selection_example_artifacts.append(enriched)
        selection_cache_binding['example_artifacts'] = selection_example_artifacts
        selection_cache_binding['execution_status_json'] = str(binding_path)
    except Exception as exc:
        selection_cache_binding = {'status': 'invalid', 'execution_status_json': str(binding_path), 'error': str(exc)}
processed = {}
preflight_errors = []
preflight_warnings = []
preflight_matched_count = None
preflight_expected_matched_count = None
preflight_selection_cache_json = None
preflight_backfilled_input_paths = {}
preflight_backfill_manifest_json = None
preflight = orch / 'preflight_status.json'
if preflight.is_file():
    try:
        preflight_payload = json.loads(preflight.read_text(encoding='utf-8'))
        processed = preflight_payload.get('processed_input_paths', {})
        preflight_errors = preflight_payload.get('errors', [])
        preflight_warnings = preflight_payload.get('warnings', [])
        preflight_matched_count = preflight_payload.get('matched_count')
        preflight_expected_matched_count = preflight_payload.get('expected_matched_count')
        preflight_selection_cache_json = preflight_payload.get('selection_cache_json')
        preflight_backfilled_input_paths = preflight_payload.get('backfilled_input_paths', {})
        preflight_backfill_manifest_json = preflight_payload.get('backfill_manifest_json')
    except Exception as exc:
        preflight_errors = [f'preflight_status_parse_error: {exc}']
        processed = {}
terminal_states = {}
for state_path in sorted(orch.glob('*.state.json')):
    try:
        terminal_states[state_path.stem.replace('.state', '')] = json.loads(state_path.read_text(encoding='utf-8'))
    except Exception as exc:
        terminal_states[state_path.name] = {'status': 'failed', 'error': str(exc)}
rc_status = {}
rc_names = ['preflight','qwen_vllm','gpt55','gptCodexS','qwen','gpt41_gate','llama4']
if os.environ.get('EXPERIMENT7_USE_CHATMOCK_SERVICE') == '1':
    rc_names.append('chatmock_service')
for name in rc_names:
    rc_path = orch / f'{name}.rc'
    rc_status[name] = int(rc_path.read_text().strip()) if rc_path.is_file() and rc_path.read_text().strip().isdigit() else None
llama4_sentinel = orch / 'answer_llama4.smoke_started'
status_counts = Counter(str(item.get('score_status') or item.get('route_status') or 'unknown') for item in items)

def parse_ts(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    except ValueError:
        return None

timeline_events = []
for path in sorted((Path('Experiment') / os.environ['TOP_EXPT_ID']).glob('**/*timeline*.jsonl')):
    try:
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            item['timeline_path'] = str(path)
            timeline_events.append(item)
    except Exception as exc:
        timeline_events.append({'timeline_path': str(path), 'phase': 'timeline_parse', 'status': 'error', 'detail': str(exc), 'time': now})

pending_starts = {}
stage_durations = []
for event in sorted(timeline_events, key=lambda item: item.get('time') or ''):
    key = (event.get('phase'), event.get('engine'), event.get('retriever_id'), event.get('dataset'))
    status = event.get('status')
    ts = parse_ts(event.get('time'))
    if status == 'start':
        pending_starts[key] = event
    elif status in {'finish', 'ready', 'blocked', 'error'} and key in pending_starts:
        start = pending_starts.pop(key)
        start_ts = parse_ts(start.get('time'))
        duration = (ts - start_ts).total_seconds() if ts and start_ts else None
        stage_durations.append({
            'phase': event.get('phase'),
            'status': status,
            'engine': event.get('engine'),
            'retriever_id': event.get('retriever_id'),
            'dataset': event.get('dataset'),
            'started_at': start.get('time'),
            'finished_at': event.get('time'),
            'duration_seconds': duration,
            'timeline_path': event.get('timeline_path'),
        })

rows_per_minute = []
inference_by_key = {
    (item.get('engine'), item.get('retriever_id'), item.get('dataset')): item
    for item in stage_durations
    if item.get('phase') in {'inference', 'inference_retry'} and item.get('duration_seconds')
}
for item in items:
    duration_item = inference_by_key.get((item.get('engine'), item.get('retriever_id'), item.get('dataset')))
    rows = item.get('rows') or 0
    input_rows = item.get('input_rows') or rows
    duration = duration_item.get('duration_seconds') if duration_item else None
    if duration and rows:
        rpm = rows / (duration / 60.0)
        remaining = max((input_rows or 0) - rows, 0)
        rows_per_minute.append({
            'engine': item.get('engine'),
            'retriever_id': item.get('retriever_id'),
            'dataset': item.get('dataset'),
            'rows': rows,
            'input_rows': input_rows,
            'duration_seconds': duration,
            'rows_per_minute': rpm,
            'eta_seconds': (remaining / rpm * 60.0) if rpm and remaining else 0,
        })
payload = {
    'time': now,
    'experiment': '7',
    'stage': 'fqan_formal_score_report',
    'top_expt_id': os.environ['TOP_EXPT_ID'],
    'items': items,
    'status_counts': dict(status_counts),
    'processed_input_paths': processed,
    'preflight_errors': preflight_errors,
    'preflight_warnings': preflight_warnings,
    'preflight_matched_count': preflight_matched_count,
    'preflight_expected_matched_count': preflight_expected_matched_count,
    'preflight_selection_cache_json': preflight_selection_cache_json,
    'preflight_backfilled_input_paths': preflight_backfilled_input_paths,
    'preflight_backfill_manifest_json': preflight_backfill_manifest_json,
    'blocker_audits': blocker_audits,
    'selection_cache_binding': selection_cache_binding,
    'selection_example_artifacts': selection_example_artifacts,
    'gpt_quota_checkpoints': quota_checkpoints,
    'rc_status': rc_status,
    'terminal_states': terminal_states,
    'timeline_event_count': len(timeline_events),
    'stage_durations': stage_durations,
    'rows_per_minute': rows_per_minute,
    'llama4_smoke_started': llama4_sentinel.is_file(),
    'llama4_smoke_started_sentinel': str(llama4_sentinel),
    'status': 'running',
}
if rc_status.get('preflight') not in (None, 0):
    payload['status'] = 'blocked_preflight'
elif any(value not in (None, 0) for value in rc_status.values()):
    payload['status'] = 'blocked_runtime'
elif all(value == 0 for value in rc_status.values() if value is not None) and all(value is not None for value in rc_status.values()):
    payload['status'] = 'completed'
Path(os.environ['REPORT_JSON']).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
lines = [
    '# Experiment 7 FQAN Formal Score Report',
    f"- top_expt_id: {payload['top_expt_id']}",
    f"- status: {payload['status']}",
    f"- llama_full_started: {payload['llama_full_started']}",
    f"- observed_cases: {len(items)}",
    f"- preflight_matched_count: {preflight_matched_count}/{preflight_expected_matched_count}",
    f"- blocker_audits: {len(blocker_audits)}",
    f"- preflight_backfilled_inputs: {len(preflight_backfilled_input_paths)}",
    f"- selection_cache_binding: {(selection_cache_binding or {}).get('status', 'missing')} ({len(selection_example_artifacts)} cases)",
    f"- timeline_events: {len(timeline_events)}",
    '',
]
if stage_durations:
    lines.extend(['## Stage Durations', '| phase | engine | retriever_id | dataset | duration_seconds | status |', '| --- | --- | --- | --- | ---: | --- |'])
    for duration in stage_durations[-40:]:
        lines.append('| {phase} | {engine} | {retriever_id} | {dataset} | {duration_seconds} | {status} |'.format(**duration))
    lines.append('')
if rows_per_minute:
    lines.extend(['## Throughput / ETA', '| engine | retriever_id | dataset | rows | input_rows | rows_per_minute | eta_seconds |', '| --- | --- | --- | ---: | ---: | ---: | ---: |'])
    for row in rows_per_minute:
        lines.append('| {engine} | {retriever_id} | {dataset} | {rows} | {input_rows} | {rows_per_minute:.4f} | {eta_seconds:.1f} |'.format(**row))
    lines.append('')
if preflight_errors:
    lines.extend(['## Preflight Blockers'])
    lines.extend(f"- {error}" for error in preflight_errors)
    lines.append('')
if preflight_backfilled_input_paths:
    lines.extend(['## Preflight Backfilled Matched Inputs'])
    for key, value in sorted(preflight_backfilled_input_paths.items()):
        lines.append(f"- {key}: {value}")
    if preflight_backfill_manifest_json:
        lines.append(f"- manifest: {preflight_backfill_manifest_json}")
    lines.append('')
if blocker_audits:
    lines.extend(['## Blocker Audits'])
    for audit in blocker_audits:
        payload_audit = audit.get('payload') or {}
        lines.append(f"- {audit.get('path')}: {payload_audit.get('conclusion', audit.get('error', 'unknown'))}")
        for missing_case in payload_audit.get('missing_cases', []):
            lines.append(f"  - {missing_case}")
    lines.append('')
if selection_example_artifacts:
    lines.extend([
        '## Selection Cache Binding',
        '| dataset | prompt_type | rows | cache_hit_rows | cache_missing_rows | output_jsonl |',
        '| --- | --- | --- | --- | --- | --- |',
    ])
    for artifact in selection_example_artifacts:
        lines.append('| {dataset} | {prompt_type} | {rows} | {cache_hit_rows} | {cache_missing_rows} | {output_jsonl} |'.format(
            dataset=artifact.get('dataset'),
            prompt_type=artifact.get('prompt_type'),
            rows=artifact.get('rows'),
            cache_hit_rows=artifact.get('cache_hit_rows'),
            cache_missing_rows=artifact.get('cache_missing_rows'),
            output_jsonl=artifact.get('output_jsonl'),
        ))
    lines.append('')
lines.extend([
    '| engine | retriever_id | dataset | execution_accuracy | score_status | route_status | failure_category | output_jsonl |',
    '| --- | --- | --- | --- | --- | --- | --- | --- |',
])
for item in items:
    lines.append('| {engine} | {retriever_id} | {dataset} | {execution_accuracy} | {score_status} | {route_status} | {failure_category} | {output_jsonl} |'.format(
        engine=item.get('engine'), retriever_id=item.get('retriever_id'), dataset=item.get('dataset'), execution_accuracy=item.get('execution_accuracy'), score_status=item.get('score_status'), route_status=item.get('route_status'), failure_category=item.get('failure_category'), output_jsonl=item.get('output_jsonl')))
Path(os.environ['REPORT_MD']).write_text('\n'.join(lines) + '\n', encoding='utf-8')
for raw in (os.environ['STATUS_JSON'], os.environ['QUEUE_LOG_JSON']):
    Path(raw).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
index_path = Path(os.environ['INDEX_PATH'])
try:
    index = json.loads(index_path.read_text(encoding='utf-8')) if index_path.is_file() else {'entries': []}
except Exception:
    index = {'entries': []}
rel_log = str(Path(os.environ['QUEUE_LOG_JSON']).relative_to(Path(os.environ['WORKSPACE_ROOT'])))
entry = {'time': now, 'path': rel_log, 'repo': os.environ['REPO_ROOT'], 'kind': 'experiment7_fqan_formal_queue', 'status': payload['status'], 'summary': f"Experiment 7 fqan queue; llama4_smoke_started={payload['llama4_smoke_started']}; selection_cache_binding={(selection_cache_binding or {}).get('status', 'missing')}; blocker_audits={len(blocker_audits)}", 'tags': ['experiment_7','finqa','ea','tmux','fqan','formal_queue']}
entries = [item for item in index.setdefault('entries', []) if item.get('path') != rel_log]
entries.append(entry)
index['entries'] = entries
index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps({'status': payload['status'], 'observed_cases': len(items), 'llama4_smoke_started': payload['llama4_smoke_started'], 'selection_cache_binding': (selection_cache_binding or {}).get('status', 'missing'), 'blocker_audits': len(blocker_audits)}, ensure_ascii=False))
PYREPORT
  sleep ${REPORT_INTERVAL_SECONDS@Q}
done
EOF
  chmod +x "${runner}"
}

if [[ "${PUBLIC_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  (
    cd "${REPO_ROOT}"
    scripts=(
      dist/experiment_7_generator_answer.sh
      dist/experiment_7_in_context_selection.sh
      dist/experiment_7_selection_cache_binding.sh
      dist/experiment_7_runner.sh
      dist/generator_no_api_key.sh
      dist/generator_api_key.sh
    )
    python_files=(
      new_full_finqa_run.py
      dist/experiment_7_backfill_mistral_dev_matched_json.py
      dist/experiment_7_corrected_status.py
      dist/experiment_7_stress_first25.py
    )
    bash -n "${scripts[@]}"
    conda run --no-capture-output -n "${CONDA_ENV}" python -B -m py_compile "${python_files[@]}"
    jq empty "${FQAN_DOCS_ROOT}/args.json"
    EXPERIMENT7_MATRIX_VALUE="${EXPERIMENT7_MATRIX}" python3 -c 'import os; items=os.environ["EXPERIMENT7_MATRIX_VALUE"].split(); assert len(items)==24 and len(set(items))==24 and all(item.count(":")==1 for item in items)'
  )
  printf '[%s] Experiment 7 public preflight passed; runtime artifacts were not required.\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  exit 0
fi

if [[ "${PREFLIGHT_ONLY:-0}" == "1" ]]; then
  write_preflight_runner
  bash "${ORCH_DIR}/preflight.sh"
  printf '[%s] Experiment 7 preflight-only completed without creating tmux windows.\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  exit 0
fi

for required_session in API_key chatmock vllm/llama_cpp run monitor; do
  if tmux has-session -t "${required_session}" 2>/dev/null; then
    :
  else
    tmux new-session -d -s "${required_session}" -n base -c "${REPO_ROOT}" "exec bash"
  fi
done
TMUX_RUN_SESSION="monitor"

ensure_window_name_free_in_session() {
  local session_name="$1"
  local window_name="$2"
  if tmux list-windows -t "${session_name}" -F '#W' 2>/dev/null | grep -Fxq "${window_name}"; then
    write_status "blocked_window_exists" "tmux ${session_name}:${window_name} already exists; refusing to overwrite." 2
    exit 2
  fi
}

window_pairs=(
  "monitor:${WINDOW_PREFIX}_preflight"
  "API_key:${WINDOW_PREFIX}_gpt41_gate"
  "chatmock:${WINDOW_PREFIX}_gpt55"
  "chatmock:${WINDOW_PREFIX}_gptCodexS"
  "vllm/llama_cpp:${WINDOW_PREFIX}_qwen_vllm"
  "run:${WINDOW_PREFIX}_qwen"
  "vllm/llama_cpp:${WINDOW_PREFIX}_llama4"
  "monitor:${WINDOW_PREFIX}_report"
)
if [[ "${EXPERIMENT7_USE_CHATMOCK_SERVICE}" == "1" ]]; then
  window_pairs+=("chatmock:${WINDOW_PREFIX}_chatmock_service")
fi
for pair in "${window_pairs[@]}"; do
  ensure_window_name_free_in_session "${pair%%:*}" "${pair#*:}"
done

write_preflight_runner
if [[ "${EXPERIMENT7_USE_CHATMOCK_SERVICE}" == "1" ]]; then
  write_chatmock_runner
fi
write_qwen_service_runner
EXPERIMENT7_USE_CHATMOCK_ROUTE="${EXPERIMENT7_USE_CHATMOCK_ROUTE:-1}"
if [[ "${EXPERIMENT7_USE_CHATMOCK_ROUTE}" == "1" ]]; then
  write_generator_worker gpt55 gpt5_5 "CHATMOCK_BASE_URL=http://localhost:${CHATMOCK_PORT}/v1 CHATMOCK_API_KEY=${CHATMOCK_API_KEY:-key} ALLOW_DIAGNOSTIC_CHATMOCK_FORMAL=1 GPT5_5_CODEX_ROUTE=${GPT5_5_CODEX_ROUTE:-chatmock}" ":"
  write_generator_worker gptCodexS gpt5_3_codexS "CHATMOCK_BASE_URL=http://localhost:${CHATMOCK_PORT}/v1 CHATMOCK_API_KEY=${CHATMOCK_API_KEY:-key} ALLOW_DIAGNOSTIC_CHATMOCK_FORMAL=1 GPT5_3_CODEX_ROUTE=${GPT5_3_CODEX_ROUTE:-chatmock}" ":"
else
  write_generator_worker gpt55 gpt5_5 "GPT5_5_CODEX_ROUTE=${GPT5_5_CODEX_ROUTE:-openai}" ":"
  write_generator_worker gptCodexS gpt5_3_codexS "GPT5_3_CODEX_ROUTE=${GPT5_3_CODEX_ROUTE:-api_key}" ":"
fi
write_generator_worker qwen qwen3_6 "MODELS_ROOT=${FQAN_MODELS_ROOT} HF_HOME=${HF_HOME} HF_HUB_CACHE=${HF_HUB_CACHE} TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE} QWEN3_6_MODEL_PATH=${FQAN_MODELS_ROOT}/.cache/huggingface/hub/models--Qwen--Qwen3.6-35B-A3B-FP8/snapshots/95a723d08a9490559dae23d0cff1d9466213d989 VLLM_BASE_URL=http://localhost:${QWEN_VLLM_PORT}/v1 VLLM_API_KEY=EMPTY VLLM_SERVED_MODEL_NAME=qwen3_6 VLLM_RUNTIME_PROFILE=qwen_fp8_tp2_precise_kv" "wait_for_rc_success qwen_vllm ${ORCH_DIR@Q}/qwen_vllm.rc"
write_gpt41_gate_runner
write_llama4_smoke_runner
write_report_runner

write_status "queue_created" "Experiment 7 queue created; qwen full and GPT routes start after preflight, llama4 smoke waits for qwen terminal state; preflight waits ${WAIT_BEFORE_START_SECONDS}s before checks." 0
tmux new-window -t "monitor" -n "${WINDOW_PREFIX}_preflight" -c "${REPO_ROOT}" "bash ${ORCH_DIR@Q}/preflight.sh; exec bash"
if [[ "${EXPERIMENT7_USE_CHATMOCK_SERVICE}" == "1" ]]; then
  tmux new-window -t "chatmock" -n "${WINDOW_PREFIX}_chatmock_service" -c "${REPO_ROOT}" "bash ${ORCH_DIR@Q}/chatmock_service.sh; exec bash"
fi
tmux new-window -t "API_key" -n "${WINDOW_PREFIX}_gpt41_gate" -c "${REPO_ROOT}" "bash ${ORCH_DIR@Q}/gpt41_gate.sh; exec bash"
tmux new-window -t "chatmock" -n "${WINDOW_PREFIX}_gpt55" -c "${REPO_ROOT}" "bash ${ORCH_DIR@Q}/gpt55.sh; exec bash"
tmux new-window -t "chatmock" -n "${WINDOW_PREFIX}_gptCodexS" -c "${REPO_ROOT}" "bash ${ORCH_DIR@Q}/gptCodexS.sh; exec bash"
tmux new-window -t "vllm/llama_cpp" -n "${WINDOW_PREFIX}_qwen_vllm" -c "${REPO_ROOT}" "bash ${ORCH_DIR@Q}/qwen_vllm.sh; exec bash"
tmux new-window -t "run" -n "${WINDOW_PREFIX}_qwen" -c "${REPO_ROOT}" "bash ${ORCH_DIR@Q}/qwen.sh; exec bash"
tmux new-window -t "vllm/llama_cpp" -n "${WINDOW_PREFIX}_llama4" -c "${REPO_ROOT}" "bash ${ORCH_DIR@Q}/llama4.sh; exec bash"
tmux new-window -t "monitor" -n "${WINDOW_PREFIX}_report" -c "${REPO_ROOT}" "bash ${ORCH_DIR@Q}/report.sh; exec bash"
write_status "queued" "Experiment 7 fqan queue is running in tmux session ${TMUX_RUN_SESSION}." 0
