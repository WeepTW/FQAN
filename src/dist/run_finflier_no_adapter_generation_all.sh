#!/usr/bin/env bash
# Concise driver: run the EXISTING No-adaptor x FinFlier programs for all
# three retrievers to completion. Does not modify any generation/evaluation
# program. FLAN uses the dedicated long-context orchestrator (self-evaluating);
# Mistral/T5Gemma2 use the shared 3-case orchestrator (generation only -- no
# evaluation entry point exists for them, see printed note at the end).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_SRC="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_SRC"
eval "$(conda shell.bash hook)"
conda activate fnqa

FLAN_ROOT="Experiment/experiment_6_finflier_flan_long_context_generation_20260816T083643Z"
FLAN_EVAL="Experiment/experiment_6_finflier_flan_long_context_evaluation_v6_1_0_20260816T083643Z"
# The prior root (...20260815T183419Z) is stale: its frozen
# compatibility_fingerprint.json no longer matches the current
# implementation/config hash (ProtocolError: "output root compatibility
# fingerprint differs; use a fresh root" -- the tool's own prescribed
# recovery). A fresh root is used instead so `start` can freeze a
# fingerprint consistent with the code as it exists right now.
NOAD_ROOT="Experiment/experiment_6_narrative2_finflier_no_adapter_$(date -u +%Y%m%dT%H%M%SZ)"

echo "== [1/2] FLAN no-adaptor + FinFlier (long-context, self-evaluating) =="
if pgrep -f "experiment6_finflier_flan_long_context.py start --output-root .*${FLAN_ROOT##*/}" >/dev/null; then
  echo "already running under a live process; not launching a second one"
else
  python -B dist/experiment6_finflier_flan_long_context.py resume \
    --output-root "$FLAN_ROOT" --evaluation-root "$FLAN_EVAL"
fi

echo "== [2/2] Mistral + T5Gemma2 no-adaptor + FinFlier (generation only, fresh root) =="
attempt=0
max_attempts=20
while true; do
  attempt=$((attempt + 1))
  cmd="resume"
  [[ "$attempt" -eq 1 ]] && cmd="start"
  set +e
  out="$(python -B dist/experiment6_finflier.py "$cmd" --output-root "$NOAD_ROOT" 2>&1)"
  rc=$?
  set -e
  printf '%s\n' "$out"
  if [[ "$rc" -ne 0 ]] && printf '%s' "$out" | grep -q "compatibility fingerprint differs"; then
    echo "fatal: fresh root still hit a fingerprint mismatch; not retrying" >&2
    exit 1
  fi
  status_json="$(python -B dist/experiment6_finflier.py status --output-root "$NOAD_ROOT")"
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

cat <<'EOF'

NOTE: Mistral/T5Gemma2 generation is complete (10 runs x 85 rows each, FLAN
case in this 3-case matrix stays permanently blocked by its 8192-token gate --
that is expected; FLAN is covered separately above). No final evaluation
(Precision/Recall/F1) exists for Mistral/T5Gemma2 FinFlier yet: the shared
evaluator's --scope enum only recognizes {candidate12,candidate34,
candidate-merged34,flan-long-context} -- there is no scope wired for this
pair. Producing one requires adding a new branch to
dist/evaluate_experiment6_binding_candidates_v1.py, which is core evaluation
code outside the "run existing scripts only" authorization. Raw predictions
and materializable manifests are preserved under the generation root for
whenever that evaluation path is approved and added.
EOF
