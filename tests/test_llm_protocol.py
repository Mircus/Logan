import json
from pathlib import Path

import pytest

from logical_gans.modelbuilder.core.devil import run_devil
from logical_gans.modelbuilder.core.loader import load_seed_open_world, load_theory
from logical_gans.modelbuilder.core.partial_structure import PartialStructure
from logical_gans.modelbuilder.core.types import Truth
from logical_gans.modelbuilder.learned.actions import RelationEdit, legal_relation_edits
from logical_gans.modelbuilder.learned.llm_protocol import (
    LLMProtocolError,
    apply_validated_llm_action,
    parse_llm_output,
    validate_llm_actions,
)
from logical_gans.modelbuilder.learned.mcts import mcts_relation_build
from logical_gans.modelbuilder.learned.mock_llm import mock_llm_json, mock_llm_prior

ROOT = Path(__file__).resolve().parents[1]
THEORIES = ROOT / "examples" / "theories"
SEED_FILE = ROOT / "examples" / "seeds" / "preorder_chain_3.json"


def _theory():
    return load_theory(THEORIES / "preorder.json")


def _action_json(relation, args, value, kind="set_relation", explanation="hi"):
    return json.dumps({
        "proposed_actions": [{"kind": kind, "relation": relation, "args": list(args), "value": value}],
        "explanation": explanation,
    })


def test_valid_llm_json_parses():
    out = parse_llm_output(_action_json("R", (0, 2), "true"))
    assert len(out.proposed_actions) == 1
    a = out.proposed_actions[0]
    assert a.kind == "set_relation" and a.relation == "R" and a.args == (0, 2)


def test_malformed_json_rejected():
    with pytest.raises(LLMProtocolError):
        parse_llm_output("{ not valid json")
    with pytest.raises(LLMProtocolError):
        parse_llm_output(json.dumps({"proposed_actions": "nope"}))


def test_illegal_cell_rejected():
    empty = PartialStructure.empty(_theory().signature, 3)
    allowed = legal_relation_edits(empty, "R")
    out = parse_llm_output(_action_json("R", (5, 5), "true"))  # out of range
    plan = validate_llm_actions(out, allowed)
    assert plan.validated == []
    assert plan.rejected and plan.rejected[0]["reason"] == "not_in_allowed_actions"


def test_known_seed_cell_rejected():
    seed = load_seed_open_world(SEED_FILE, _theory().signature)
    allowed = legal_relation_edits(seed, "R")  # (0,1),(1,2) are known -> not allowed
    out = parse_llm_output(_action_json("R", (0, 1), "false"))
    plan = validate_llm_actions(out, allowed)
    assert plan.validated == []
    assert plan.rejected[0]["reason"] == "not_in_allowed_actions"


def test_unknown_kind_and_bad_value_rejected():
    empty = PartialStructure.empty(_theory().signature, 3)
    allowed = legal_relation_edits(empty, "R")
    bad_kind = parse_llm_output(_action_json("R", (0, 1), "true", kind="delete_relation"))
    assert validate_llm_actions(bad_kind, allowed).rejected[0]["reason"] == "unknown_action_kind"
    bad_val = parse_llm_output(_action_json("R", (0, 1), "maybe"))
    assert validate_llm_actions(bad_val, allowed).rejected[0]["reason"] == "invalid_truth_value"


def test_valid_action_applies_to_copy_not_original():
    empty = PartialStructure.empty(_theory().signature, 3)
    edit = RelationEdit("R", (0, 1), Truth.TRUE)
    new = apply_validated_llm_action(empty, edit)
    assert empty.get_relation("R", (0, 1)) is Truth.UNKNOWN
    assert new.get_relation("R", (0, 1)) is Truth.TRUE


def test_explanation_is_ignored_by_validation():
    empty = PartialStructure.empty(_theory().signature, 3)
    allowed = legal_relation_edits(empty, "R")
    a = validate_llm_actions(parse_llm_output(_action_json("R", (0, 1), "true", explanation="A")), allowed)
    b = validate_llm_actions(parse_llm_output(_action_json("R", (0, 1), "true", explanation="totally different")), allowed)
    assert a.validated == b.validated and len(a.validated) == 1


def test_mock_llm_proposal_validates():
    empty = PartialStructure.empty(_theory().signature, 3)
    allowed = legal_relation_edits(empty, "R")
    plan = validate_llm_actions(parse_llm_output(mock_llm_json("first_true", allowed)), allowed)
    assert len(plan.validated) == 1
    assert plan.validated[0].value is Truth.TRUE
    assert plan.validated[0] in set(allowed)


def test_mcts_with_mock_llm_prior_preserves_seed_and_verifies():
    theory = _theory()
    seed = load_seed_open_world(SEED_FILE, theory.signature)
    out = mcts_relation_build(theory, "R", 3, seed_structure=seed, rollouts=150,
                              llm_prior=mock_llm_prior("witness_match"))
    assert out["uses_llm_prior"] is True
    assert out["status"] == "satisfied"
    rels = out["structure"]["relations"]
    assert rels["R(0,1)"] == "true" and rels["R(1,2)"] == "true"   # seed preserved
    assert rels["R(0,2)"] == "true"                                # forced fact
    # never revised a seed cell
    for ev in out["trace"]:
        assert tuple(ev["edit"]["args"]) not in {(0, 1), (1, 2)}
    # independent exhaustive verification
    final = PartialStructure.empty(theory.signature, 3)
    for key, val in rels.items():
        inside = key[key.index("(") + 1: key.index(")")]
        i, j = (int(x) for x in inside.split(","))
        final.set_relation("R", (i, j), Truth(val))
    assert run_devil(final, theory.clauses).status == "ok"
