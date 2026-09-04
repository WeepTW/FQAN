#!/usr/bin/env bash
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_XLSX="${SOURCE_XLSX:-${ROOT_DIR}/data/src/narratives/narratives1.xlsx}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT_DIR}/data}"
PROMPT_DIRS="${PROMPT_DIRS:-finqa_original finqa_zero_shot finqa_many_shot finqa_dynamic_shot testing}"
NARRATIVE1_GENERATOR="${NARRATIVE1_GENERATOR:-}"
CONDA_ENV="${CONDA_ENV:-fnqa}"

if [[ ! -f "${SOURCE_XLSX}" ]]; then
  echo "missing narrative source: ${SOURCE_XLSX}" >&2
  exit 2
fi

for prompt_dir in ${PROMPT_DIRS}; do
  prompt_root="${OUTPUT_ROOT}/${prompt_dir}"
  target_dir="${prompt_root}/finqa_narrative1"
  if [[ ! -d "${prompt_root}" ]]; then
    echo "skip missing prompt dir: ${prompt_root}" >&2
    continue
  fi
  mkdir -p "${target_dir}"

  if [[ -n "${NARRATIVE1_GENERATOR}" ]]; then
    conda run --no-capture-output -n "${CONDA_ENV}" python -B "${NARRATIVE1_GENERATOR}" \
      --source-xlsx "${SOURCE_XLSX}" \
      --prompt-dir "${prompt_root}" \
      --output-dir "${target_dir}"
  else
    conda run --no-capture-output -n "${CONDA_ENV}" python -B - "${SOURCE_XLSX}" "${prompt_root}" "${target_dir}" <<'PY_STATUS'
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

source_xlsx = Path(sys.argv[1])
prompt_root = Path(sys.argv[2])
target_dir = Path(sys.argv[3])
payload = {
    "time": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    "status": "scaffold_ready",
    "failure_category": "blocked_schema_pending",
    "source_xlsx": str(source_xlsx),
    "prompt_dir": str(prompt_root),
    "output_dir": str(target_dir),
    "created_files": ["build_status.json"],
    "next_step": "Define narrative1 schema and set NARRATIVE1_GENERATOR to the concrete generator script.",
}
(target_dir / "build_status.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(payload, ensure_ascii=False))
PY_STATUS
  fi
done
