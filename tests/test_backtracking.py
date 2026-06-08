import json
from pathlib import Path

from logical_gans.modelbuilder.cli import main
from logical_gans.modelbuilder.core.atoms import EqAtom
from logical_gans.modelbuilder.core.backtracking import backtracking_generate
from logical_gans.modelbuilder.core.clauses import HornClause
from logical_gans.modelbuilder.core.devil import run_devil
from logical_gans.modelbuilder.core.loader import load_theory
from logical_gans.modelbuilder.core.signature import Signature
from logical_gans.modelbuilder.core.terms import Var
from logical_gans.modelbuilder.core.theory import Theory
from logical_gans.modelbuilder.core.types import Truth

THEORIES = Path(__file__).resolve().parents[1] / "examples" / "theories"


def _contradictory_theory() -> Theory:
    # forall x, y: x = y  -- unsatisfiable for any domain with >= 2 elements.
    clause = HornClause(
        name="all_equal",
        variables=("x", "y"),
        premises=(),
        conclusion=EqAtom(Var("x"), Var("y")),
    )
    return Theory(name="all_equal", signature=Signature.build(), clauses=[clause])


def test_backtracking_preorder_n3_succeeds():
    theory = load_theory(THEORIES / "preorder.json")
    res = backtracking_generate(theory, 3, max_nodes=10000)
    assert res.status == "satisfied"
    # the returned structure really satisfies the theory
    assert run_devil(res.structure, theory.clauses).status == "ok"
    for i in range(3):
        assert res.structure.get_relation("R", (i, i)) is Truth.TRUE


def test_backtracking_semigroup_n2_succeeds():
    theory = load_theory(THEORIES / "semigroup.json")
    res = backtracking_generate(theory, 2, max_nodes=10000)
    assert res.status == "satisfied"
    assert res.structure.unknown_function_cells() == []
    assert run_devil(res.structure, theory.clauses).status == "ok"


def test_backtracking_returns_unsat_on_contradiction():
    res = backtracking_generate(_contradictory_theory(), 2, max_nodes=10000)
    assert res.status == "unsat"


def test_backtracking_respects_max_nodes():
    theory = load_theory(THEORIES / "preorder.json")
    res = backtracking_generate(theory, 3, max_nodes=1)
    assert res.status == "unknown"
    assert res.nodes <= 2  # tripped the budget almost immediately


def test_cli_search_preorder(capsys):
    rc = main(["search", "--theory", str(THEORIES / "preorder.json"),
               "--n", "3", "--max-nodes", "10000"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "satisfied"
    assert out["nodes"] >= 1
    assert "trace" in out


def test_cli_search_semigroup(capsys):
    rc = main(["search", "--theory", str(THEORIES / "semigroup.json"),
               "--n", "2", "--max-nodes", "10000"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "satisfied"
