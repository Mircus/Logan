from pathlib import Path

from logical_gans.modelbuilder.core.atoms import EqAtom, RelAtom
from logical_gans.modelbuilder.core.clauses import HornClause
from logical_gans.modelbuilder.core.devil import run_devil
from logical_gans.modelbuilder.core.loader import load_theory
from logical_gans.modelbuilder.core.obligations import extract_obligation
from logical_gans.modelbuilder.core.partial_structure import PartialStructure
from logical_gans.modelbuilder.core.signature import Signature
from logical_gans.modelbuilder.core.terms import Const, Func, Var
from logical_gans.modelbuilder.core.theory import Theory
from logical_gans.modelbuilder.core.types import Truth

THEORIES = Path(__file__).resolve().parents[1] / "examples" / "theories"


def test_relation_obligation_from_preorder():
    theory = load_theory(THEORIES / "preorder.json")
    s = PartialStructure.empty(theory.signature, 3)
    for (i, j) in [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2)]:
        s.set_relation("R", (i, j), Truth.TRUE)
    for (i, j) in [(1, 0), (2, 0), (2, 1), (0, 2)]:
        pass  # leave R(0,2) UNKNOWN; others below FALSE
    for (i, j) in [(1, 0), (2, 0), (2, 1)]:
        s.set_relation("R", (i, j), Truth.FALSE)
    result = run_devil(s, theory.clauses)
    obl = extract_obligation(s, result)
    assert obl is not None
    assert obl.kind == "relation" and obl.symbol == "R"
    assert obl.args == (0, 2)
    assert set(obl.suggested_values) == {Truth.TRUE, Truth.FALSE}


def test_function_obligation_from_unary_involution():
    sig = Signature.build(functions=[("f", 1)])
    x = Var("x")
    clause = HornClause("involution", ("x",), (),
                        EqAtom(Func("f", (Func("f", (x,)),)), x))
    theory = Theory("inv", sig, [clause])
    s = PartialStructure.empty(sig, 2)
    obl = extract_obligation(s, run_devil(s, theory.clauses))
    assert obl is not None
    assert obl.kind == "function" and obl.symbol == "f"
    assert obl.suggested_values == (0, 1)


def test_constant_obligation_from_constant_fact():
    sig = Signature.build(relations=[("P", 1)], constants=["c"])
    clause = HornClause("Pc", (), (), RelAtom("P", (Const("c"),)))
    theory = Theory("pc", sig, [clause])
    s = PartialStructure.empty(sig, 2)
    obl = extract_obligation(s, run_devil(s, theory.clauses))
    assert obl is not None
    assert obl.kind == "constant" and obl.symbol == "c"
    assert obl.suggested_values == (0, 1)


def test_no_obligation_when_ok():
    sig = Signature.build(relations=[("P", 1)], constants=["c"])
    clause = HornClause("Pc", (), (), RelAtom("P", (Const("c"),)))
    theory = Theory("pc", sig, [clause])
    s = PartialStructure.empty(sig, 2)
    s.set_constant("c", 0)
    s.set_relation("P", (0,), Truth.TRUE)
    s.set_relation("P", (1,), Truth.FALSE)
    assert extract_obligation(s, run_devil(s, theory.clauses)) is None
