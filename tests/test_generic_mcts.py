from pathlib import Path

from logical_gans.modelbuilder.core.atoms import EqAtom, RelAtom
from logical_gans.modelbuilder.core.clauses import HornClause
from logical_gans.modelbuilder.core.devil import run_devil
from logical_gans.modelbuilder.core.loader import load_seed_open_world, load_theory
from logical_gans.modelbuilder.core.partial_structure import PartialStructure
from logical_gans.modelbuilder.core.signature import Signature
from logical_gans.modelbuilder.core.terms import Const, Func, Var
from logical_gans.modelbuilder.core.theory import Theory
from logical_gans.modelbuilder.core.types import Truth
from logical_gans.modelbuilder.learned.generic_mcts import mcts_semantic_build

ROOT = Path(__file__).resolve().parents[1]
THEORIES = ROOT / "examples" / "theories"
SEED_FILE = ROOT / "examples" / "seeds" / "preorder_chain_3.json"


def _verify(theory, struct_json, n):
    s = PartialStructure.empty(theory.signature, n)
    for key, val in struct_json["relations"].items():
        inside = key[key.index("(") + 1: key.index(")")]
        args = tuple(int(x) for x in inside.split(","))
        s.set_relation(key[: key.index("(")], args, Truth(val))
    for key, val in struct_json["functions"].items():
        if val is not None:
            inside = key[key.index("(") + 1: key.index(")")]
            args = tuple(int(x) for x in inside.split(","))
            s.set_function(key[: key.index("(")], args, val)
    for name, val in struct_json["constants"].items():
        if val is not None:
            s.set_constant(name, val)
    return run_devil(s, theory.clauses).status


def _kinds(out):
    return {e["edit"]["kind"] for e in out["trace"]}


def test_generic_mcts_on_preorder_relation():
    theory = load_theory(THEORIES / "preorder.json")
    seed = load_seed_open_world(SEED_FILE, theory.signature)
    out = mcts_semantic_build(theory, 3, seed_structure=seed, rollouts=200)
    assert out["status"] == "satisfied"
    assert out["builder"] == "mcts_semantic"
    assert out["structure"]["relations"]["R(0,2)"] == "true"
    assert _kinds(out) <= {"set_relation"}
    # seed cells never revised
    for e in out["trace"]:
        assert tuple(e["edit"]["args"]) not in {(0, 1), (1, 2)}


def test_generic_mcts_on_unary_function():
    sig = Signature.build(functions=[("f", 1)])
    x = Var("x")
    clause = HornClause("involution", ("x",), (), EqAtom(Func("f", (Func("f", (x,)),)), x))
    theory = Theory("inv", sig, [clause])
    out = mcts_semantic_build(theory, 2, rollouts=200)
    assert out["status"] == "satisfied"
    assert "set_function" in _kinds(out)
    assert _verify(theory, out["structure"], 2) == "ok"


def test_generic_mcts_on_constant_and_relation():
    sig = Signature.build(relations=[("P", 1)], constants=["c"])
    clause = HornClause("Pc", (), (), RelAtom("P", (Const("c"),)))
    theory = Theory("pc", sig, [clause])
    out = mcts_semantic_build(theory, 2, rollouts=200)
    assert out["status"] == "satisfied"
    kinds = _kinds(out)
    assert "set_constant" in kinds and "set_relation" in kinds
    assert _verify(theory, out["structure"], 2) == "ok"
