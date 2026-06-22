"""Gate R3: legal Devil-move enumeration.

The symbolic Devil exposes its bounded challenges as a list of structurally valid
candidate moves, and chooses from that list. No neural Devil yet.
"""
from pathlib import Path

import pytest

from logical_gans.modelbuilder.core.partial_structure import PartialStructure
from logical_gans.modelbuilder.neural_fraisse.fight import build_task, load_problem
from logical_gans.modelbuilder.neural_fraisse.game import DevilMove, GameState
from logical_gans.modelbuilder.neural_fraisse.players import (
    SymbolicActiveDevil, enumerate_legal_devil_moves)

ROOT = Path(__file__).resolve().parents[1]
CYCLE3 = ROOT / "examples" / "problems" / "cycle3_fight.json"
CYCLE2 = ROOT / "examples" / "problems" / "cycle2_impossible_fight.json"


def _initial_state(path):
    task = build_task(load_problem(path))
    structure = PartialStructure.empty(task.signature, task.domain_size)
    return GameState(structure, task)


@pytest.mark.parametrize("path", [CYCLE3, CYCLE2])
def test_legal_move_list_nonempty(path):
    moves = enumerate_legal_devil_moves(_initial_state(path))
    assert len(moves) > 0


@pytest.mark.parametrize("path", [CYCLE3, CYCLE2])
def test_every_move_is_structurally_valid(path):
    moves = enumerate_legal_devil_moves(_initial_state(path))
    for m in moves:
        assert isinstance(m, DevilMove)
        assert m.kind in {"ChallengeClauseInstance", "ChallengeGoalCell"}
        assert isinstance(m.clause_name, str) and m.clause_name
        assert isinstance(m.assignment, dict)
        cell = m.target_cell
        assert cell is not None
        assert cell["kind"] in {"relation", "function", "constant"}
        assert isinstance(cell["symbol"], str) and cell["symbol"]
        if cell["kind"] == "constant":
            assert cell["args"] is None
        else:
            assert isinstance(cell["args"], list) and len(cell["args"]) > 0
        assert m.obligation is not None


@pytest.mark.parametrize("path", [CYCLE3, CYCLE2])
def test_symbolic_devil_chooses_from_the_list(path):
    state = _initial_state(path)
    moves = enumerate_legal_devil_moves(state)
    chosen = SymbolicActiveDevil().choose_move(state.structure, state.task)
    assert chosen is not None
    assert chosen in moves          # DevilMove.__eq__ ignores the obligation field
