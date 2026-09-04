#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SRC_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
FQAN_ROOT="$(cd -- "${SRC_ROOT}/.." && pwd)"
COMMAND="${1:-}"
PROFILE="${2:-}"

usage() {
  cat <<'EOF'
Usage:
  bash src/dist/reproduce.sh requirements
  bash src/dist/reproduce.sh environment
  bash src/dist/reproduce.sh upstreams
  bash src/dist/reproduce.sh data [public|formal]
  bash src/dist/reproduce.sh models [smoke|retrievers|formal_generators]
  bash src/dist/reproduce.sh experiments [smoke|dry-run|0|1|2|3|4|5|6|7] [experiment-6-action]
EOF
}

run_experiment_dry_run() {
  local temporary_root
  temporary_root="$(mktemp -d -t fqan-experiment-dry-run-XXXXXX)"
  trap 'rm -rf -- "${temporary_root}"' RETURN

  SETUP_ID="release_dry_run" \
    SETUP_DOWNLOAD_MODE=none \
    SETUP_ARTIFACTS_REQUIRED=0 \
    SETUP_STRICT=0 \
    bash "${SCRIPT_DIR}/experiment_setup.sh"

  PROMPT_MODES=original RUN_PREFLIGHT=0 RUN_TRAIN=0 RUN_INFER=0 RUN_MATCH=0 \
    EXPT_ID_SUFFIX=_release_dry_run \
    bash "${SCRIPT_DIR}/experiment_1_mistral_retriever.sh"
  PROMPT_MODES=original RUN_PREFLIGHT=0 RUN_TRAIN=0 RUN_INFER=0 RUN_MATCH=0 \
    EXPT_ID_SUFFIX=_release_dry_run \
    bash "${SCRIPT_DIR}/experiment_2_flan_retriever.sh"
  PROMPT_MODES=original RUN_PREFLIGHT=0 RUN_TRAIN=0 RUN_INFER=0 RUN_MATCH=0 \
    EXPT_ID_SUFFIX=_release_dry_run \
    bash "${SCRIPT_DIR}/experiment_3_t5gemma_retriever.sh"
  LOSS_EXPLORATORY_ACK=1 PROMPT_MODE=original RUN_PREFLIGHT=0 \
    RUN_TRAIN=0 RUN_INFER=0 RUN_MATCH=0 LOSS_EXPT_ID=loss_release_dry_run \
    bash "${SCRIPT_DIR}/experiment_4_retriever_loss_diagram.sh"
  EXPT_ID=experiment_5_release_dry_run RUN_EXECUTE=0 LIMIT=1 \
    bash "${SCRIPT_DIR}/experiment_5_qwen_few10_smoke.sh"
  if ! bash "${SCRIPT_DIR}/experiment_6.sh" public-preflight "${temporary_root}/experiment_6"; then
    printf 'Experiment 6 requires locally installed upstream research files; see the blocker above\n'
  fi
  if ! PUBLIC_PREFLIGHT_ONLY=1 PREFLIGHT_ONLY=1 WAIT_BEFORE_START_SECONDS=0 \
    EXPT_ID=experiment_7_release_dry_run \
    bash "${SCRIPT_DIR}/experiment_7_formal_tmux_run.sh"; then
    printf 'Experiment 7 requires locally installed upstream research files; see the blocker above\n'
  fi

  printf 'Experiments 0-5 dry-run passed; Experiments 6-7 reported any external-source blockers above\n'
}

run_formal_experiment() {
  local experiment_id="$1"
  local experiment_6_action="${2:-auto}"

  case "${experiment_id}" in
    0)
      bash "${SCRIPT_DIR}/experiment_setup.sh"
      ;;
    1)
      bash "${SCRIPT_DIR}/experiment_1_mistral_retriever.sh"
      ;;
    2)
      bash "${SCRIPT_DIR}/experiment_2_flan_retriever.sh"
      ;;
    3)
      bash "${SCRIPT_DIR}/experiment_3_t5gemma_retriever.sh"
      ;;
    4)
      LOSS_EXPLORATORY_ACK=1 bash "${SCRIPT_DIR}/experiment_4_retriever_loss_diagram.sh"
      ;;
    5)
      bash "${SCRIPT_DIR}/experiment_5_qwen_few10_smoke.sh"
      ;;
    6)
      bash "${SCRIPT_DIR}/experiment_6.sh" "${experiment_6_action}"
      ;;
    7)
      bash "${SCRIPT_DIR}/experiment_7_formal_tmux_run.sh"
      ;;
    *)
      printf 'Unsupported experiment: %s\n' "${experiment_id}" >&2
      usage >&2
      return 2
      ;;
  esac
}

case "${COMMAND}" in
  requirements)
    for required_command in git git-lfs conda; do
      command -v "${required_command}" >/dev/null 2>&1 || {
        printf 'Missing required command: %s\n' "${required_command}" >&2
        exit 2
      }
    done
    git lfs version >/dev/null
    conda --version
    printf 'Requirements preflight passed\n'
    ;;
  environment)
    if ! conda run -n fnqa python -V >/dev/null 2>&1; then
      conda create -n fnqa python=3.10 -y
    fi
    conda run -n fnqa python -m pip install --upgrade pip
    conda run -n fnqa python -m pip install -r "${SRC_ROOT}/requirements.txt"
    printf 'Environment fnqa is ready\n'
    ;;
  upstreams)
    conda run -n fnqa python -B "${SCRIPT_DIR}/install_upstreams.py"
    ;;
  data)
    PROFILE="${PROFILE:-public}"
    case "${PROFILE}" in public|formal) ;; *) usage >&2; exit 2 ;; esac
    git -C "${FQAN_ROOT}" lfs pull
    conda run -n fnqa python -B "${SCRIPT_DIR}/check_data.py" --profile "${PROFILE}"
    ;;
  models)
    PROFILE="${PROFILE:-smoke}"
    case "${PROFILE}" in smoke|retrievers|formal_generators) ;; *) usage >&2; exit 2 ;; esac
    conda run -n fnqa python -B "${SCRIPT_DIR}/install_models.py" --profile "${PROFILE}"
    ;;
  experiments)
    PROFILE="${PROFILE:-smoke}"
    case "${PROFILE}" in
      smoke)
        conda run -n fnqa python -B "${SCRIPT_DIR}/test_experiment6_paths.py"
        conda run -n fnqa python -B "${SCRIPT_DIR}/test_finqa_target_execution.py"
        SETUP_ID="release_smoke" \
          SETUP_DOWNLOAD_MODE=none \
          SETUP_ARTIFACTS_REQUIRED=0 \
          SETUP_STRICT=0 \
          bash "${SCRIPT_DIR}/experiment_setup.sh"
        ;;
      dry-run)
        run_experiment_dry_run
        ;;
      0|1|2|3|4|5|6|7)
        run_formal_experiment "${PROFILE}" "${3:-auto}"
        ;;
      *)
        usage >&2
        exit 2
        ;;
    esac
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
