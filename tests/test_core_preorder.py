import os
import subprocess
import sys
from pathlib import Path

from logical_gans.modelbuilder.core.atoms import EqAtom, RelAtom
from logical_gans.modelbuilder.core.eval import eval_atom, eval_term
from logical_gans.modelbuilder.core.generator import generate
from logical_gans.modelbuilder.core.partial_structure import PartialStructure
from logical_gans.modelbuilder.core.signature import Signature
from logical_gans.modelbuilder.core.terms import Func, Var
from logical_gans.modelbuilder.core.types import Truth
from logical_gans.modelbuilder.examples.preorder import (
    antisymmetry_refutation,
    empty_preorder_structure,
    preorder_clauses,
)


def test_partial_structure_initializes_unknowns():
    sig = Signature.build(relations=[("R", 2)], functions=[("f", 1)], constants=["c"])
    A = PartialStructure.empty(sig, 3)
    assert A.domain == (0, 1, 2)
    # every relation tuple is UNKNOWN
    assert len(A.relations) == 9
    assert all(v is Truth.UNKNOWN for v in A.relations.values())
    # every function tuple and constant is None
    assert len(A.functions) == 3
    assert all(v is None for v in A.functions.values())
    assert A.get_constant("c") is None
    assert len(A.unknown_relation_cells()) == 9
    assert len(A.unknown_function_cells()) == 3


def test_eval_relation_atom():
    A = empty_preorder_structure(2)
    atom = RelAtom("R", (Var("x"), Var("y")))
    assign = {"x": 0, "y": 1}
    assert eval_atom(atom, assign, A) is Truth.UNKNOWN
    A.set_relation("R", (0, 1), Truth.TRUE)
    assert eval_atom(atom, assign, A) is Truth.TRUE
    A.set_relation("R", (0, 1), Truth.FALSE)
    assert eval_atom(atom, assign, A) is Truth.FALSE


def test_eval_function_term_unknown():
    sig = Signature.build(functions=[("mul", 2)])
    A = PartialStructure.empty(sig, 2)
    term = Func("mul", (Var("x"), Var("y")))
    assign = {"x": 0, "y": 1}
    assert eval_term(term, assign, A) is None
    # equality of two unknown terms is UNKNOWN
    assert eval_atom(EqAtom(term, term), assign, A) is Truth.UNKNOWN
    A.set_function("mul", (0, 1), 1)
    assert eval_term(term, assign, A) == 1


def test_preorder_generation_n3_succeeds():
    result = generate(empty_preorder_structure(3), preorder_clauses())
    assert result.status == "satisfied"
    # at least all reflexive facts hold
    for i in range(3):
        assert result.structure.get_relation("R", (i, i)) is Truth.TRUE
    # no UNKNOWN cells remain in a satisfied structure
    assert result.structure.unknown_relation_cells() == []


def test_preorder_antisymmetry_counterexample_witness():
    out = antisymmetry_refutation()
    assert out["preorder_status"] == "ok"
    assert out["status"] == "refuted"
    w = out["witness"]
    assert w is not None
    assert w["assignment"] == {"x": 0, "y": 1}
    assert w["conclusion_value"] == "false"
    assert w["status"] == "failed"


def test_import_modelbuilder_does_not_import_torch():
    repo_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(repo_root / "src") + os.pathsep + env.get("PYTHONPATH", "")
    code = (
        "import logical_gans.modelbuilder, sys; "
        "assert 'torch' not in sys.modules, 'torch was imported'; "
        "assert 'torch_geometric' not in sys.modules, 'torch_geometric was imported'"
    )
    proc = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
