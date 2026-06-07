#!/usr/bin/env bash
set -euo pipefail
python -m pytest -q
python experiments/e1_groups.py --out results_groups.jsonl
