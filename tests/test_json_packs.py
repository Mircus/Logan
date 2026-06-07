from pathlib import Path

import pytest

from logical_gans.modelbuilder.core.loader import (
    TheoryLoadError,
    load_claim,
    load_structure,
    load_theory,
)
from logical_gans.modelbuilder.core.policy import MaximalHornPolicy, SparseHornPolicy
from logical_gans.modelbuilder.core.runner import check, refute, synthesize
from logical_gans.modelbuilder.core.types import Truth

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
THEORIES = EXAMPLES / "theories"
CLAIMS = EXAMPLES / "claims"
STRUCTURES = EXAMPLES / "structures"


def test_load_preorder_theory_from_json():
    theory = load_theory(THEORIES / "preorder.json")
    assert theory.name == "preorder"
    assert "R" in theory.signature.relations
    assert theory.signature.relations["R"].arity == 2
    assert [c.name for c in theory.clauses] == ["reflexive", "transitive"]


def test_synthesize_preorder_json_sparse():
    theory = load_theory(THEORIES / "preorder.json")
    res = synthesize(theory, 3, SparseHornPolicy())
    assert res.status == "satisfied"
    assert res.policy == "sparse_horn"
    for i in range(3):
        for j in range(3):
            expected = Truth.TRUE if i == j else Truth.FALSE
            assert res.structure.get_relation("R", (i, j)) is expected


def test_synthesize_preorder_json_maximal():
    theory = load_theory(THEORIES / "preorder.json")
    res = synthesize(theory, 3, MaximalHornPolicy())
    assert res.status == "satisfied"
    assert res.policy == "maximal_horn"
    for i in range(3):
        for j in range(3):
            assert res.structure.get_relation("R", (i, j)) is Truth.TRUE


def test_load_total_preorder_2_structure_from_json():
    theory = load_theory(THEORIES / "preorder.json")
    structure = load_structure(STRUCTURES / "total_preorder_2.json", theory.signature)
    assert structure.domain == (0, 1)
    for i in range(2):
        for j in range(2):
            assert structure.get_relation("R", (i, j)) is Truth.TRUE
    # and it actually satisfies the preorder theory
    assert check(theory, structure)["status"] == "satisfied"


def test_generic_refute_antisymmetry_witness():
    theory = load_theory(THEORIES / "preorder.json")
    claim = load_claim(CLAIMS / "antisymmetry.json")
    out = refute(theory, claim, n=2, policy=MaximalHornPolicy())
    assert out["status"] == "refuted"
    w = out["witness"]
    assert w["assignment"] == {"x": 0, "y": 1}
    assert w["conclusion_value"] == "false"
    assert w["status"] == "failed"


def test_load_semigroup_theory_from_json():
    theory = load_theory(THEORIES / "semigroup.json")
    assert theory.name == "semigroup"
    assert "mul" in theory.signature.functions
    assert theory.signature.functions["mul"].arity == 2
    assert [c.name for c in theory.clauses] == ["associativity"]


def test_synthesize_semigroup_n1_from_json():
    theory = load_theory(THEORIES / "semigroup.json")
    res = synthesize(theory, 1, SparseHornPolicy())
    assert res.status == "satisfied"
    assert res.structure.get_function("mul", (0, 0)) == 0


def test_synthesize_equivalence_relation_json_sparse():
    theory = load_theory(THEORIES / "equivalence_relation.json")
    res = synthesize(theory, 3, SparseHornPolicy())
    assert res.status == "satisfied"
    for i in range(3):
        for j in range(3):
            expected = Truth.TRUE if i == j else Truth.FALSE
            assert res.structure.get_relation("E", (i, j)) is expected


def test_bad_json_gives_readable_error(tmp_path):
    bad = tmp_path / "broken.json"
    bad.write_text("{ this is not valid json", encoding="utf-8")
    with pytest.raises(TheoryLoadError) as exc:
        load_theory(bad)
    msg = str(exc.value)
    assert "broken.json" in msg
    assert "invalid JSON" in msg
