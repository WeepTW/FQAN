#!/usr/bin/env bash
set -Eeuo pipefail

CONDA_ENV=fnqa
export CONDA_ENV
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/retriever_experiment_lib.sh"

QUEUE_TS="${WAITING_QUEUE_TIMESTAMP:-$(date -u +%Y%m%dT%H%M%SZ)}"
QUEUE_ID="waiting_experiments_$QUEUE_TS"
QUEUE_ROOT="$REPO_ROOT/Experiment/$QUEUE_ID"
EXP6_ROOT="$REPO_ROOT/Experiment/experiment_6_finflier_flan_three_adapter_long_context_generation_$QUEUE_TS"
EXP7_ID="experiment_7_mistral4_remaining_queue_$QUEUE_TS"
STOP_REQUEST_FILE="$QUEUE_ROOT/STOP_AFTER_CURRENT_CASE"
mkdir -p "$QUEUE_ROOT"

QUEUE_ROOT="$QUEUE_ROOT" EXP6_ROOT="$EXP6_ROOT" EXP7_ID="$EXP7_ID" \
  conda run --no-capture-output -n fnqa python -B - <<'PY'
import json
import os
from pathlib import Path
payload = {
    "schemaVersion": 1,
    "protocol": "fnqa-waiting-experiments-queue-v1",
    "status": "running",
    "sequence": [
        "experiment6_flan_finflier_z_m_d",
        "experiment7_mistral4_flan_d_mistral_o_z_m_t5gemma2_o_z_m_d",
    ],
    "experiment6Root": os.environ["EXP6_ROOT"],
    "experiment7Id": os.environ["EXP7_ID"],
    "condaEnvironment": "fnqa",
    "stopRequestFile": str(Path(os.environ["QUEUE_ROOT"]) / "STOP_AFTER_CURRENT_CASE"),
    "progressCommand": (
        "conda run --no-capture-output -n fnqa python -B "
        "dist/show_waiting_experiments_progress.py "
        + os.environ["QUEUE_ROOT"]
    ),
}
path = Path(os.environ["QUEUE_ROOT"]) / "queue_paths.json"
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(path)
PY

update_queue_status() {
  local status="$1"
  local stage="$2"
  QUEUE_PATHS="$QUEUE_ROOT/queue_paths.json" QUEUE_STATUS="$status" QUEUE_STAGE="$stage" \
    conda run --no-capture-output -n fnqa python -B - <<'PY_STATUS'
import json
import os
from pathlib import Path

path = Path(os.environ["QUEUE_PATHS"])
payload = json.loads(path.read_text(encoding="utf-8"))
payload["status"] = os.environ["QUEUE_STATUS"]
payload["terminalStage"] = os.environ["QUEUE_STAGE"]
path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
PY_STATUS
}

EXPERIMENT6_OUTPUT_ROOT="$EXP6_ROOT" \
WAITING_QUEUE_STOP_FILE="$STOP_REQUEST_FILE" \
  bash dist/run_experiment6_finflier_flan_three_adapter_queue.sh \
  2>&1 | tee "$QUEUE_ROOT/experiment6_queue.log"

if [[ -f "$EXP6_ROOT/queue.stopped" ]]; then
  cp "$EXP6_ROOT/queue/stop_status.json" "$QUEUE_ROOT/stop_status.json"
  update_queue_status stopped experiment6
  printf '%s\n' "$(utc_now)" >"$QUEUE_ROOT/queue.stopped"
  exit 0
fi

EXPERIMENT6_SENTINEL="$EXP6_ROOT/queue.completed" \
EXPERIMENT7_EXPT_ID="$EXP7_ID" \
WAITING_QUEUE_STOP_FILE="$STOP_REQUEST_FILE" \
  bash dist/run_experiment7_mistral4_remaining_queue.sh \
  2>&1 | tee "$QUEUE_ROOT/experiment7_queue.log"

EXP7_QUEUE_DIR="$REPO_ROOT/Experiment/$EXP7_ID/remaining_queue"
if [[ -f "$EXP7_QUEUE_DIR/queue.stopped" ]]; then
  cp "$EXP7_QUEUE_DIR/stop_status.json" "$QUEUE_ROOT/stop_status.json"
  update_queue_status stopped experiment7
  printf '%s\n' "$(utc_now)" >"$QUEUE_ROOT/queue.stopped"
  exit 0
fi

update_queue_status completed experiment7
printf '%s\n' "$(utc_now)" >"$QUEUE_ROOT/queue.completed"
