#!/usr/bin/env bash
# LOGAN / UCMT proof capsule showcase.
# Runs the countermodel capsule from the self-contained problem file and
# verifies the machine-readable certificate it emits.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CERT="results/cycle3_countermodel_certificate.json"

PYTHONPATH=src python -m logical_gans.modelbuilder.cli prove-countermodel \
  examples/problems/cycle3_countermodel.json

PYTHONIOENCODING=utf-8 python - "$CERT" <<'PY'
import json, sys

cert = json.load(open(sys.argv[1], encoding="utf-8"))
c, g, s, abl = cert["certificate"], cert["generator"], cert["structure"], cert["ablation"]

def check(cond, msg):
    if not cond:
        print("FAIL:", msg)
        sys.exit(1)

check(cert.get("accepted") is True, "accepted = true")
check(c["model_relation"] == "A ⊩_{3,800} T", "certificate.model_relation = A ⊩_{3,800} T")
check(c["counterclaim_relation"] == "A ⊭_{3,800} C", "certificate.counterclaim_relation = A ⊭_{3,800} C")
check(g["kind"] == "neural_semantic_mcts", "generator.kind = neural_semantic_mcts")
check(g["uses_neural_policy"] is True, "generator.uses_neural_policy = true")
check(g["random_rollouts"] == "disabled", "generator.random_rollouts = disabled")
check(g["success_via"] == "guided_tree_policy", "generator.success_via = guided_tree_policy")
check(c["theory_status"] == "satisfied", "certificate.theory_status = satisfied")
check(c["claim_status"] == "failed", "certificate.claim_status = failed")
check(bool(cert.get("witness")), "witness exists")
check(any(v in ("true", "false") for v in s["relations"].values()), "structure contains a relation")
check(any(v is not None for v in s["functions"].values()), "structure contains a function")
check(any(v is not None for v in s["constants"].values()), "structure contains a constant")
check(abl["neural"]["success"] is True, "ablation.neural.success = true")
check(abl["uniform"]["success"] is False, "ablation.uniform.success = false")
check(abl["obligation_first"]["success"] is False, "ablation.obligation_first.success = false")

print("certificate checks passed")
PY

echo "LOGAN UCMT proof capsule passed"
