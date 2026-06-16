from logical_gans.modelbuilder.core.partial_structure import PartialStructure
from logical_gans.modelbuilder.core.signature import Signature
from logical_gans.modelbuilder.core.types import Truth
from logical_gans.modelbuilder.learned.semantic_actions import (
    SetConstant,
    SetFunction,
    SetRelation,
)
from logical_gans.modelbuilder.learned.semantic_features import FEATURE_DIM, feature_vector

SIG = Signature.build(relations=[("E", 2)], functions=[("s", 1)], constants=["a"])


def _struct():
    return PartialStructure.empty(SIG, 2)


def test_features_for_relation_function_constant():
    s = _struct()
    fr = feature_vector(SetRelation("E", (0, 1), Truth.TRUE), s)
    ff = feature_vector(SetFunction("s", (0,), 1), s)
    fc = feature_vector(SetConstant("a", 0), s)
    for f in (fr, ff, fc):
        assert len(f) == FEATURE_DIM
    # is_relation / is_function / is_constant flags in slots [0:3]
    assert fr[0] == 1.0 and ff[1] == 1.0 and fc[2] == 1.0
    assert fr[0:3] == [1.0, 0.0, 0.0]
    assert ff[0:3] == [0.0, 1.0, 0.0]
    assert fc[0:3] == [0.0, 0.0, 1.0]


def test_matches_obligation_flag():
    from logical_gans.modelbuilder.core.obligations import DevilObligation
    s = _struct()
    obl = DevilObligation("relation", "E", (0, 1), (Truth.TRUE, Truth.FALSE), "conclusion", witness=None)
    matching = feature_vector(SetRelation("E", (0, 1), Truth.TRUE), s, obl)
    other = feature_vector(SetRelation("E", (1, 0), Truth.TRUE), s, obl)
    assert matching[8] == 1.0
    assert other[8] == 0.0
    # source one-hot = relation
    assert matching[9:13] == [1.0, 0.0, 0.0, 0.0]
