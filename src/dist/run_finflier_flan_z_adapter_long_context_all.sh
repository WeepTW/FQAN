#!/usr/bin/env bash
# Concise driver for the Fine-tuned FLAN (finqa_flan_z adapter) + full FinFlier
# long-context diagnostic: 10 runs x 85 rows, then materialize/validate/
# evaluate/build-tables using the SAME generic, already-existing tools the
# No-adaptor FLAN long-context diagnostic uses (evaluator --scope
# flan-long-context is generic; it does not hardcode case IDs or route).
#
# This is a non-canonical, thesis-copilot-authored extension: no officially
# run matrix in this repo currently exercises Fine-tuned + FinFlier. The
# checkpoint (finqa_flan_z, zero-shot-trained) is a user-approved stand-in --
# there is no "correct" checkpoint choice, since no adapter was trained
# against the FinFlier prompt itself. Treat results as exploratory only.
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_SRC="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_SRC"
eval "$(conda shell.bash hook)"
conda activate fnqa

# This box has 24 cores and runs this job concurrently with the No-adaptor
# FLAN diagnostic; both are unbounded-thread CPU-bound PyTorch inference and
# were observed oversubscribing (load average > 30 on 24 cores) when both
# defaulted to all-core threading. Cap this one to leave the other its half.
export OMP_NUM_THREADS=11
export MKL_NUM_THREADS=11
export OPENBLAS_NUM_THREADS=11
export NUMEXPR_NUM_THREADS=11

CONFIG="config/experiment6_narrative2_generation_finflier_flan_z_adapter_long_context.json"
EVAL_CONFIG="config/experiment6_finflier_flan_z_adapter_long_context_evaluation_v6_1.json"
CASE_ID="6_finflier_prompt_flan_z_adapter_long_context"
ROOT="Experiment/experiment_6_finflier_flan_z_adapter_long_context_generation_$(date -u +%Y%m%dT%H%M%SZ)"
CANDIDATE_ROOT="${ROOT}_binding_candidates_v1"
EVAL_ROOT="${ROOT/_generation_/_evaluation_v6_1_0_}"

echo "root=$ROOT"
mkdir -p "$ROOT"

run_one() {
  local run="$1" device="$2"
  local log="$ROOT/logs/run_${run}_device_${device}.log"
  mkdir -p "$ROOT/logs"
  python -B dist/run_experiment6_narrative2_generation.py \
    --config "$CONFIG" \
    --output-root "$ROOT" \
    --case "$CASE_ID" \
    --cuda-visible-devices "$device" \
    --run "$run" \
    > "$log" 2>&1
  return $?
}

for run in 1 2 3 4 5 6 7 8 9 10; do
  manifest="$ROOT/manifests/${CASE_ID}__run_$(printf '%02d' "$run").json"
  if [[ -f "$manifest" ]]; then
    echo "run=$run already complete; skipping"
    continue
  fi
  echo "== run=$run attempting GPU 0 =="
  if run_one "$run" 0; then
    echo "run=$run: GPU 0 succeeded"
    continue
  fi
  log="$ROOT/logs/run_${run}_device_0.log"
  run_dir="$ROOT/cases/$CASE_ID/run_$(printf '%02d' "$run")"
  # The subprocess retriever writes its own traceback into raw/*.log and
  # retriever_attempts/*/*.log, not into this wrapper's captured stdout --
  # check all of them for the OOM signature.
  if grep -qiE "cuda out of memory|torch.outofmemoryerror|cublas_status_alloc_failed" \
       "$log" "$run_dir"/raw/*.log "$run_dir"/retriever_attempts/*/*.log 2>/dev/null; then
    echo "run=$run: GPU 0 OOM, falling back to CPU (this is slow: ~5 min/row x 85 rows)"
    if run_one "$run" cpu; then
      echo "run=$run: CPU succeeded"
      continue
    else
      echo "run=$run: CPU attempt also failed; see $ROOT/logs/run_${run}_device_cpu.log" >&2
      exit 1
    fi
  else
    echo "run=$run: non-OOM failure; see $log" >&2
    exit 1
  fi
done

echo "== all 10 runs present; materializing/evaluating =="
python3 - "$ROOT" "$CANDIDATE_ROOT" <<'PYEOF'
import json, sys, hashlib
from pathlib import Path
root, candidate_root = Path(sys.argv[1]), Path(sys.argv[2])
case_id = "6_finflier_prompt_flan_z_adapter_long_context"
def sha256_file(p):
    d = hashlib.sha256()
    with p.open("rb") as h:
        for chunk in iter(lambda: h.read(1024*1024), b""):
            d.update(chunk)
    return d.hexdigest()
manifests = [json.loads((root / "manifests" / f"{case_id}__run_{r:02d}.json").read_text()) for r in range(1, 11)]
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
    "expectedCases": 1,
    "expectedRuns": 10,
    "expectedRows": 85,
    "caseIds": [case_id],
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
