import json
from pathlib import Path

from logical_gans.modelbuilder.cli import main
from logical_gans.modelbuilder.core.atoms import EqAtom, RelAtom
from logical_gans.modelbuilder.core.backtracking import backtracking_generate
from logical_gans.modelbuilder.core.depth import atom_depth, clause_depth, term_depth
from logical_gans.modelbuilder.core.devil import run_devil_bounded
from logical_gans.modelbuilder.core.loader import load_theory
from logical_gans.modelbuilder.core.partial_structure import PartialStructure
from logical_gans.modelbuilder.core.terms import Func, Var
from logical_gans.modelbuilder.core.types import Truth

THEORIES = Path(__file__).resolve().parents[1] / "examples" / "theories"


def _total_preorder_n3():
    theory = load_theory(THEORIES / "preorder.json")
    s = PartialStructure.empty(theory.signature, 3)
    for a in range(3):
        for b in range(3):
            s.set_relation("R", (a, b), Truth.TRUE)
    return theory, s


def test_term_and_clause_depth_compute_correctly():
    x, y, z = Var("x"), Var("y"), Var("z")
    assert term_depth(x) == 0
    assert term_depth(Func("mul", (x, y))) == 1
    nested = Func("mul", (Func("mul", (x, y)), z))
    assert term_depth(nested) == 2
    assert atom_depth(RelAtom("R", (x, y))) == 0
    assert atom_depth(EqAtom(nested, x)) == 2

    preorder = load_theory(THEORIES / "preorder.json")
    assert max(clause_depth(c) for c in preorder.clauses) == 0
    semigroup = load_theory(THEORIES / "semigroup.json")
    assert clause_depth(semigroup.clauses[0]) == 2  # associativity


def test_bounded_devil_skips_clauses_above_k():
    theory = load_theory(THEORIES / "semigroup.json")  # associativity depth 2
    s = PartialStructure.empty(theory.signature, 1)
    res = run_devil_bounded(s, theory.clauses, k=1)
    assert res.skipped_by_depth == 1
    assert res.status == "ok"
    assert res.checked_instances == 0


def test_bounded_devil_respects_budget():
    theory, s = _total_preorder_n3()  # all instances OK -> budget forces stop
    res = run_devil_bounded(s, theory.clauses, budget=5)
    assert res.status == "ok"
    assert res.checked_instances == 5
    assert res.budget_exhausted is True


def test_search_records_k_and_budget_in_trace():
    theory = load_theory(THEORIES / "preorder.json")
    res = backtracking_generate(theory, 3, k=1, budget=10)
    challenges = [e for e in res.trace if e.get("event") == "challenge"]
    assert challenges
    assert all(e["k"] == 1 and e["budget"] == 10 for e in challenges)
    assert all("checked_instances" in e and "budget_exhausted" in e for e in challenges)


def test_search_tiny_budget_survives_bounded_without_exhaustive():
    res = backtracking_generate(load_theory(THEORIES / "preorder.json"), 3, budget=1)
    assert res.status == "satisfied"
    # survived the bounded attack but did NOT exhaustively decide every cell
    assert res.structure.unknown_relation_cells() != []
    assert any(
        e.get("event") == "challenge" and e.get("budget_exhausted") is True
        for e in res.trace
    )


def test_unbounded_behavior_unchanged():
    res = backtracking_generate(load_theory(THEORIES / "preorder.json"), 3)
    assert res.status == "satisfied"
    # exhaustive: every cell decided, identity preorder
    assert res.structure.unknown_relation_cells() == []
    # unbounded trace carries no bounded metadata
    assert all("budget" not in e for e in res.trace if e.get("event") == "challenge")


def test_cli_accepts_k_and_budget(capsys):
    rc = main(["search", "--theory", str(THEORIES / "preorder.json"),
               "--n", "3", "--k", "1", "--budget", "10", "--max-nodes", "10000"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] in {"satisfied", "unsat", "unknown"}
    challenges = [e for e in out["trace"] if e.get("event") == "challenge"]
    assert challenges and challenges[0]["budget"] == 10

    rc = main(["synthesize", "--theory", str(THEORIES / "preorder.json"),
               "--n", "3", "--k", "1", "--budget", "50"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] in {"satisfied", "unsat", "unknown"}
