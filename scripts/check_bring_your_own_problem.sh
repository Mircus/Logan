#!/usr/bin/env bash
# Bring-your-own-problem smoke check.
#  - copies the editable template, renames it, validates it, runs the capsule
#    on it, and accepts either a real countermodel OR a clean not_found/unknown
#    (no crash);
#  - then runs a known editable sample that MUST produce a verified countermodel.
set -uo pipefail   # not -e: the template run may legitimately exit non-zero

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TMP="${TMPDIR:-/tmp}/logan_user_problem.json"
CLI="python -m logical_gans.modelbuilder.cli"
export PYTHONPATH=src

# 1-2. copy template -> tmp, rename to "tmp_user_problem"
PYTHONIOENCODING=utf-8 python - "$TMP" <<'PY'
import json, sys
p = json.load(open("examples/problems/TEMPLATE_problem.json", encoding="utf-8"))
p["name"] = "tmp_user_problem"
json.dump(p, open(sys.argv[1], "w", encoding="utf-8"), indent=2)
print("wrote", sys.argv[1])
PY

# 3. validate the user problem
$CLI validate-problem "$TMP" || { echo "FAIL: template validation"; exit 1; }

# 4. run the capsule (template may not refute within (k,b); tolerate non-zero exit)
$CLI prove-countermodel "$TMP" --auto-train || true

# 5. template result must be clean: accepted OR not_found/unknown, never a crash
PYTHONIOENCODING=utf-8 python - <<'PY'
import json, sys
c = json.load(open("results/tmp_user_problem_certificate.json", encoding="utf-8"))
status = c.get("status")
ok = (c.get("accepted") is True) or (status in ("not_found", "unknown", "refuted_unverified"))
if not ok:
    print("FAIL: template run not clean: status =", status); sys.exit(1)
print("template run clean: status =", status)
PY
[ $? -eq 0 ] || exit 1

# 6. known editable sample that MUST succeed (verified countermodel)
$CLI validate-problem examples/problems/byop_sample_countermodel.json \
  || { echo "FAIL: byop_sample validation"; exit 1; }
$CLI prove-countermodel examples/problems/byop_sample_countermodel.json --auto-train || true

PYTHONIOENCODING=utf-8 python - <<'PY'
import json, sys
c = json.load(open("results/byop_sample_countermodel_certificate.json", encoding="utf-8"))
def check(cond, msg):
    if not cond:
        print("FAIL:", msg); sys.exit(1)
check(c["accepted"] is True, "byop_sample accepted")
check(c["status"] == "refuted", "byop_sample status refuted")
check(c["generator"]["uses_neural_policy"] is True, "byop_sample uses_neural_policy")
check(c["generator"]["success_via"] == "guided_tree_policy", "byop_sample success_via")
check(c["certificate"]["theory_status"] == "satisfied", "theory satisfied")
check(c["certificate"]["claim_status"] == "failed", "claim failed")
check(bool(c["witness"]), "witness exists")
s = c["structure"]
check(any(v in ("true", "false") for v in s["relations"].values()), "relation present")
check(any(v is not None for v in s["functions"].values()), "function present")
check(any(v is not None for v in s["constants"].values()), "constant present")
check(c["ablation"]["uniform"]["success"] is False, "uniform fails")
check(c["ablation"]["obligation_first"]["success"] is False, "obligation_first fails")
check(c["ablation"]["neural"]["success"] is True, "neural succeeds")
print("byop_sample certificate checks passed")
PY
[ $? -eq 0 ] || exit 1

echo "LOGAN bring-your-own-problem check passed"
