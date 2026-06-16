#!/usr/bin/env bash
# Short LOGAN / UCMT demo: run the frozen capsule, then point at BYOP.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHONPATH=src python -m logical_gans.modelbuilder.cli prove-countermodel \
  examples/problems/cycle3_countermodel.json

cat <<'EOF'

To try your own finite signature, copy examples/problems/TEMPLATE_problem.json
and run:
  PYTHONPATH=src python -m logical_gans.modelbuilder.cli prove-countermodel your_problem.json --auto-train
EOF
