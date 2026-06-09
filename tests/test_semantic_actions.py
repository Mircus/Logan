from logical_gans.modelbuilder.core.partial_structure import PartialStructure
from logical_gans.modelbuilder.core.signature import Signature
from logical_gans.modelbuilder.core.types import Truth
from logical_gans.modelbuilder.learned.semantic_actions import (
    SetConstant,
    SetFunction,
    SetRelation,
    apply_semantic_edit,
    legal_semantic_edits,
    semantic_edit_from_json,
    semantic_edit_to_json,
)

SIG = Signature.build(
    relations=[("R", 2), ("P", 1)],
    functions=[("f", 2), ("h", 1)],
    constants=["c"],
)


def _empty(n=2):
    return PartialStructure.empty(SIG, n)


def test_legal_action_count_mixed_signature():
    edits = legal_semantic_edits(_empty(2))
    # R/2: 4*2=8 ; P/1: 2*2=4 ; f/2: 4*2=8 ; h/1: 2*2=4 ; c: 2  => 26
    assert len(edits) == 26
    kinds = {type(e).__name__ for e in edits}
    assert kinds == {"SetRelation", "SetFunction", "SetConstant"}


def test_known_relation_cell_not_revisable():
    s = _empty(2)
    s.set_relation("R", (0, 1), Truth.TRUE)
    edits = legal_semantic_edits(s)
    assert not any(isinstance(e, SetRelation) and e.symbol == "R" and e.args == (0, 1) for e in edits)
    assert len(edits) == 24  # lost 2 R(0,1) edits


def test_known_function_cell_not_revisable():
    s = _empty(2)
    s.set_function("f", (0, 1), 1)
    edits = legal_semantic_edits(s)
    assert not any(isinstance(e, SetFunction) and e.symbol == "f" and e.args == (0, 1) for e in edits)
    assert len(edits) == 24  # lost 2 f(0,1) edits


def test_known_constant_not_revisable():
    s = _empty(2)
    s.set_constant("c", 0)
    edits = legal_semantic_edits(s)
    assert not any(isinstance(e, SetConstant) and e.symbol == "c" for e in edits)
    assert len(edits) == 24  # lost 2 c edits


def test_apply_returns_copy():
    s = _empty(2)
    s2 = apply_semantic_edit(s, SetRelation("R", (0, 0), Truth.TRUE))
    assert s.get_relation("R", (0, 0)) is Truth.UNKNOWN
    assert s2.get_relation("R", (0, 0)) is Truth.TRUE
    s3 = apply_semantic_edit(s, SetFunction("f", (0, 0), 1))
    assert s.get_function("f", (0, 0)) is None and s3.get_function("f", (0, 0)) == 1
    s4 = apply_semantic_edit(s, SetConstant("c", 1))
    assert s.get_constant("c") is None and s4.get_constant("c") == 1


def test_json_roundtrip():
    for edit in (SetRelation("R", (0, 1), Truth.TRUE),
                 SetFunction("f", (1, 0), 1),
                 SetConstant("c", 0)):
        assert semantic_edit_from_json(semantic_edit_to_json(edit)) == edit
