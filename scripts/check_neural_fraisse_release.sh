#!/usr/bin/env bash
# Neural Fraïssé release smoke: the fight command runs and the benchmark verdicts.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PYTHONPATH=src

FIGHT_OUT="$(python -m logical_gans.modelbuilder.neural_fraisse.fight \
  examples/problems/cycle3_fight.json)"
echo "$FIGHT_OUT"

BENCH_OUT="$(python -m logical_gans.modelbuilder.neural_fraisse.benchmark \
  --train-sizes 3,4 --test-sizes 5 --tasks 20 --budget 800 --seed 0)"
echo "$BENCH_OUT"

PYTHONIOENCODING=utf-8 python - "$FIGHT_OUT" "$BENCH_OUT" <<'PY'
import sys
fight, bench = sys.argv[1], sys.argv[2]

def need(text, token, where):
    if token not in text:
        print(f"FAIL: missing {token!r} in {where}")
        sys.exit(1)

for tok in ("Devil", "God", "Judge"):
    need(fight, tok, "fight output")
if ("GOD_WINS" not in fight) and ("DRAW" not in fight):
    print("FAIL: fight output has no GOD_WINS/DRAW outcome"); sys.exit(1)
if ("NEURAL_FRAISSE_POC = PASS" not in bench) and ("NEURAL_FRAISSE_POC = PARTIAL" not in bench):
    print("FAIL: benchmark verdict not PASS/PARTIAL"); sys.exit(1)
print("checks passed")
PY
[ $? -eq 0 ] || exit 1

echo "LOGAN Neural Fraisse release smoke passed"
