#!/usr/bin/env bash
# Signature-parametric Builder demo: MCTS over generic semantic edits on an
# arbitrary mixed signature (relation E/2, function s/1, constant a).
set -euo pipefail

OUT="results/signature_parametric_demo.json"
mkdir -p results

PYTHONPATH=src python -m logical_gans.modelbuilder.cli mcts-semantic \
  --theory examples/theories/toy_mixed_signature.json \
  --n 2 \
  --rollouts 200 | tee "$OUT"

echo "saved: $OUT"
