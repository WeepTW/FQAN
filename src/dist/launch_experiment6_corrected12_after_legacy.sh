#!/usr/bin/env bash
set -euo pipefail

cat >&2 <<'EOF'
This launcher is retired: it called the formal Experiment 6 dispatcher with the
corrected-12 command set. Use dist/experiment_6_corrected12_dual_gpu.sh with
start, resume, or status and an explicit --output-root.
EOF
exit 2
