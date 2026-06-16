import json

import pytest

from logical_gans.modelbuilder.core.partial_structure import PartialStructure
from logical_gans.modelbuilder.core.signature import Signature
from logical_gans.modelbuilder.core.types import Truth
from logical_gans.modelbuilder.learned.generic_llm_protocol import (
    GenericLLMProtocolError,
    parse_generic_llm_output,
    validate_generic_llm_actions,
)
from logical_gans.modelbuilder.learned.semantic_actions import (
    SetConstant,
    SetFunction,
    SetRelation,
    legal_semantic_edits,
)

SIG = Signature.build(relations=[("R", 2)], functions=[("f", 2)], constants=["c"])


def _allowed(n=2):
    return legal_semantic_edits(PartialStructure.empty(SIG, n))


def _out(actions, explanation="ignored"):
    return parse_generic_llm_output(json.dumps({"proposed_actions": actions, "explanation": explanation}))


def test_valid_set_relation_accepted():
    plan = validate_generic_llm_actions(
        _out([{"kind": "set_relation", "symbol": "R", "args": [0, 1], "value": "true"}]), _allowed())
    assert plan.validated == [SetRelation("R", (0, 1), Truth.TRUE)]


def test_valid_set_function_accepted():
    plan = validate_generic_llm_actions(
        _out([{"kind": "set_function", "symbol": "f", "args": [0, 1], "value": 1}]), _allowed())
    assert plan.validated == [SetFunction("f", (0, 1), 1)]


def test_valid_set_constant_accepted():
    plan = validate_generic_llm_actions(
        _out([{"kind": "set_constant", "symbol": "c", "value": 0}]), _allowed())
    assert plan.validated == [SetConstant("c", 0)]


def test_malformed_json_rejected():
    with pytest.raises(GenericLLMProtocolError):
        parse_generic_llm_output("{not json")


def test_unknown_kind_rejected():
    plan = validate_generic_llm_actions(
        _out([{"kind": "delete_relation", "symbol": "R", "args": [0, 1], "value": "true"}]), _allowed())
    assert plan.validated == [] and plan.rejected[0]["reason"] == "unknown_action_kind"


def test_known_cell_rejected():
    s = PartialStructure.empty(SIG, 2)
    s.set_relation("R", (0, 1), Truth.TRUE)  # now known
    plan = validate_generic_llm_actions(
        _out([{"kind": "set_relation", "symbol": "R", "args": [0, 1], "value": "false"}]),
        legal_semantic_edits(s))
    assert plan.validated == [] and plan.rejected[0]["reason"] == "not_in_allowed_actions"


def test_out_of_range_function_value_rejected():
    plan = validate_generic_llm_actions(
        _out([{"kind": "set_function", "symbol": "f", "args": [0, 1], "value": 5}]), _allowed(2))
    assert plan.validated == [] and plan.rejected[0]["reason"] == "not_in_allowed_actions"


def test_truth_value_on_function_rejected():
    plan = validate_generic_llm_actions(
        _out([{"kind": "set_function", "symbol": "f", "args": [0, 1], "value": "true"}]), _allowed())
    assert plan.validated == [] and plan.rejected[0]["reason"] == "invalid_value_for_function"


def test_wrong_arity_rejected():
    plan = validate_generic_llm_actions(
        _out([{"kind": "set_relation", "symbol": "R", "args": [0], "value": "true"}]), _allowed())
    assert plan.validated == [] and plan.rejected[0]["reason"] == "not_in_allowed_actions"


def test_explanation_ignored():
    a = validate_generic_llm_actions(
        _out([{"kind": "set_constant", "symbol": "c", "value": 0}], explanation="A"), _allowed())
    b = validate_generic_llm_actions(
        _out([{"kind": "set_constant", "symbol": "c", "value": 0}], explanation="totally different"), _allowed())
    assert a.validated == b.validated and len(a.validated) == 1
