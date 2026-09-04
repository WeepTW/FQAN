#!/usr/bin/env bash
# Runs the existing No-adaptor x FinFlier 3-case generation orchestrator
# (Mistral + T5Gemma2; FLAN case in this matrix stays permanently blocked by
# its own 8192-token preflight gate, covered separately by the dedicated
# FLAN long-context diagnostic) to completion, then materializes/validates/
# evaluates/builds tables with the same generic tools already used for the
# FLAN long-context diagnostic. GPU-bound (Mistral 4-bit, T5Gemma2), does not
# compete with the CPU-bound FLAN jobs.
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_SRC="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_SRC"
eval "$(conda shell.bash hook)"
conda activate fnqa

ROOT="Experiment/experiment_6_narrative2_finflier_no_adapter_$(date -u +%Y%m%dT%H%M%SZ)"
CANDIDATE_ROOT="${ROOT}_binding_candidates_v1"
EVAL_ROOT="${ROOT/_no_adapter_/_no_adapter_evaluation_v6_1_0_}"
EVAL_CONFIG="config/experiment6_finflier_no_adapter_evaluation_v6_1.json"
CASE_IDS=("6_finflier_prompt_mistral_base" "6_finflier_prompt_t5gemma2_base")

echo "root=$ROOT"

attempt=0
max_attempts=30
while true; do
  attempt=$((attempt + 1))
  cmd="resume"
  [[ "$attempt" -eq 1 ]] && cmd="start"
  out="$(python -B dist/experiment6_finflier.py "$cmd" --output-root "$ROOT" 2>&1)"
  rc=$?
  printf '%s\n' "$out"
  if [[ "$rc" -ne 0 ]] && printf '%s' "$out" | grep -q "compatibility fingerprint differs"; then
    echo "fatal: fresh root still hit a fingerprint mismatch; not retrying" >&2
    exit 1
  fi
  status_json="$(python -B dist/experiment6_finflier.py status --output-root "$ROOT")"
  st="$(printf '%s' "$status_json" | python3 -c 'import json,sys; print(json.load(sys.stdin)["status"])')"
  echo "attempt=$attempt rc=$rc status=$st"
  if [[ "$st" == "completed" || "$st" == "completed_with_runtime_blockers" ]]; then
    break
  fi
  if [[ "$attempt" -ge "$max_attempts" ]]; then
    echo "giving up after $max_attempts resume attempts; last status: $status_json" >&2
    exit 1
  fi
  sleep 20
done

echo "== generation complete; materializing/evaluating =="
python3 - "$ROOT" "$CANDIDATE_ROOT" <<PYEOF
import json, hashlib
from pathlib import Path
root, candidate_root = Path("$ROOT"), Path("$CANDIDATE_ROOT")
case_ids = ["6_finflier_prompt_mistral_base", "6_finflier_prompt_t5gemma2_base"]
def sha256_file(p):
    d = hashlib.sha256()
    with p.open("rb") as h:
        for chunk in iter(lambda: h.read(1024*1024), b""):
            d.update(chunk)
    return d.hexdigest()
manifests = []
for case_id in case_ids:
    for r in range(1, 11):
        manifests.append(json.loads((root / "manifests" / f"{case_id}__run_{r:02d}.json").read_text()))
protocols = {m["protocol"] for m in manifests}
fingerprints = {m["compatibilityFingerprint"] for m in manifests}
assert len(protocols) == 1 and len(fingerprints) == 1, (protocols, fingerprints)
out = root / "finalization" / "materialization_config.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({
    "schemaVersion": 1,
    "protocol": "experiment6-binding-candidate-materialization-v1",
    "sourceProtocol": next(iter(protocols)),
    "sourceCompatibilityFingerprint": next(iter(fingerprints)),
    "expectedCases": 2,
    "expectedRuns": 10,
    "expectedRows": 85,
    "caseIds": case_ids,
    "requiredBindingKeys": ["ObjectName", "DataName", "Position", "Trend", "Num", "Text"],
    "requireRepairCoverage": True,
}, indent=2) + "\n")
print(out)
PYEOF

MAT_CONFIG="$ROOT/finalization/materialization_config.json"
mkdir -p "$ROOT/finalization/logs"

python -B dist/materialize_experiment6_binding_candidates.py \
  --generation-root "$ROOT" --config "$MAT_CONFIG" --output-root "$CANDIDATE_ROOT" \
  > "$ROOT/finalization/logs/materialize.log" 2>&1 || { echo "materialize failed, see log" >&2; exit 1; }

python -B dist/validate_experiment6_binding_candidates.py --root "$CANDIDATE_ROOT" \
  > "$ROOT/finalization/logs/validate.log" 2>&1 || { echo "validate failed, see log" >&2; exit 1; }

mkdir -p "$EVAL_ROOT"
python -B dist/evaluate_experiment6_binding_candidates_v1.py \
  --version v6.1.0 --scope flan-long-context \
  --candidate-root "$CANDIDATE_ROOT" --evaluation-root "$EVAL_ROOT" \
  --config "$EVAL_CONFIG" \
  > "$ROOT/finalization/logs/evaluate.log" 2>&1 || { echo "evaluate failed, see log" >&2; exit 1; }

python -B dist/build_experiment6_binding_candidate_score_tables.py \
  --evaluation-report "$EVAL_ROOT/evaluation_report.json" \
  --evaluation-root "$EVAL_ROOT" \
  --source-registry config/experiment6_source_registry.json \
  --output-dir "$EVAL_ROOT" \
  > "$ROOT/finalization/logs/build_tables.log" 2>&1 || { echo "build_tables failed, see log" >&2; exit 1; }

echo "DONE evaluationRoot=$EVAL_ROOT"
python3 -c "
import json
r = json.load(open('$EVAL_ROOT/evaluation_report.json'))
print('caseMeanMacroF1:', r['overall']['caseMeanMacroF1'])
"
