#!/usr/bin/env bash
set -Eeuo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/fnqa_paths.sh"
cd "${FQAN_SRC_ROOT}"
EXPT_ID=experiment_6_api_key_gpt41_formal_20260614T1800Z
PRED_DIR=Experiment/${EXPT_ID}/binding_eval_predictions
mkdir -p Experiment/run_logs "${PRED_DIR}/raw"
export AZURE_OPENAI_GPT4_1_DEPLOYMENT="${AZURE_OPENAI_GPT4_1_DEPLOYMENT:-gpt-4.1}"
export AZURE_OPENAI_API_VERSION="${AZURE_OPENAI_API_VERSION:-2024-12-01-preview}"
export OPENAI_MAX_RETRIES="${OPENAI_MAX_RETRIES:-0}"
export OPENAI_REQUEST_TIMEOUT_SECONDS="${OPENAI_REQUEST_TIMEOUT_SECONDS:-300}"
PRE_FLIGHT_JSON="Experiment/${EXPT_ID}/binding_eval_predictions/gpt41_route_preflight.json"
conda run --no-capture-output -n fnqa python -B new_full_finqa_run.py --engine gpt4_1 --input-json /dev/null --output-jsonl /tmp/exp6_gpt41_preflight_unused.jsonl --status-json "${PRE_FLIGHT_JSON}" --profile greedy --limit 0 --max-tokens 1 --credential-purpose execute || true
CASES=(
  "6_FinFlier_gpt4.1:gpt4_1:narrative_original"
  "6_gpt4.1_z:gpt4_1:narrative_zero_shot"
  "6_gpt4.1_m:gpt4_1:narrative_many_shot"
  "6_gpt4.1_d:gpt4_1:narrative_dynamic_shot"
)
for spec in "${CASES[@]}"; do
  IFS=":" read -r case_id source_id route <<<"${spec}"
  for run in $(seq -w 1 10); do
    raw="${PRED_DIR}/raw/${case_id}.run_${run}.jsonl"
    pred="${PRED_DIR}/runs/run_${run}/${case_id}.jsonl"
    if [[ -s "${pred}" ]]; then
      printf "%s\n" "SKIP completed ${case_id} run_${run}"
      continue
    fi
    printf "%s\n" "REPAIR ${case_id} run_${run} $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    conda run --no-capture-output -n fnqa \
      python dist/repair_experiment6_run_from_raw.py \
        --experiment-id "${case_id}" \
        --source-id "${source_id}" \
        --narrative-route "${route}" \
        --raw-jsonl "${raw}" \
        --run-pred-jsonl "${pred}" \
        --engine gpt4_1 \
        --row-timeout-seconds 300 \
        --retry-max 1000 \
        --retry-wait-seconds 600 \
        --request-spacing-seconds 20
  done
done
EXPT_ID="${EXPT_ID}" \
EXPERIMENT6_NUM_RUNS=10 \
EXPERIMENT6_TOP_K=3 \
EXPERIMENT6_BUILD_LIMIT=0 \
EXPERIMENT6_PREPARE_PROMPT_DATA=0 \
EXPERIMENT6_GENERATE_BINDING_PREDICTIONS=1 \
EXPERIMENT6_BINDING_GENERATOR_PARALLELISM=1 \
EXPERIMENT6_ROW_TIMEOUT_SECONDS=300 \
OPENAI_REQUEST_TIMEOUT_SECONDS=300 \
OPENAI_MAX_RETRIES=0 \
EXPERIMENT6_RETRY_MAX=0 \
EXPERIMENT6_RESUME_RUNS=1 \
EXPERIMENT6_DEBUG=1 \
STRICT_INPUTS=1 \
bash dist/experiment_6_api_key.sh
